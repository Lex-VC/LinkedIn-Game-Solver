import mss
import numpy as np
import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

TESSERACT_DIGIT_CONFIG = "--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789"


class GridInfo:
    def __init__(self, x: int, y: int, width: int, height: int, rows: int, cols: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.rows = rows
        self.cols = cols

    @property
    def cell_w(self) -> float:
        return self.width / self.cols

    @property
    def cell_h(self) -> float:
        return self.height / self.rows


class NumberCell:
    def __init__(self, row: int, col: int, number: int):
        self.row = row
        self.col = col
        self.number = number


def capture_screen() -> np.ndarray:
    with mss.mss() as sct:
        shot = sct.grab(sct.monitors[1])
        return cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)


def _grid_line_mask(img: np.ndarray) -> np.ndarray:
    """Pixels that are medium gray and unsaturated — the grid line colour."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(img)
    max_diff = cv2.max(cv2.max(cv2.absdiff(b, g), cv2.absdiff(g, r)), cv2.absdiff(b, r))
    low_sat = (max_diff < 40).astype(np.uint8) * 255
    in_range = cv2.inRange(gray, 100, 235)
    return cv2.bitwise_and(low_sat, in_range)


def _line_positions(line_img: np.ndarray, axis: int) -> list[int]:
    """Project a binary line image and return the centre of each bright band."""
    projection = line_img.sum(axis=axis).astype(np.float32)
    if projection.max() > 0:
        projection = projection / projection.max() * 255
    mask = (projection > 76).astype(np.uint8)

    positions: list[int] = []
    in_run = False
    start = 0
    for i, v in enumerate(mask):
        if v and not in_run:
            in_run, start = True, i
        elif not v and in_run:
            in_run = False
            positions.append((start + i) // 2)
    if in_run:
        positions.append((start + len(mask)) // 2)
    return positions


def _merge_close(positions: list[int], gap: int = 8) -> list[int]:
    """Merge positions within `gap` pixels into their midpoint."""
    if not positions:
        return []
    merged = [positions[0]]
    for p in positions[1:]:
        if p - merged[-1] <= gap:
            merged[-1] = (merged[-1] + p) // 2
        else:
            merged.append(p)
    return merged


def _uniform_spacing(positions: list[int], tolerance: float = 0.25) -> int | None:
    """Return the median gap if all gaps are within tolerance, else None."""
    if len(positions) < 2:
        return None
    gaps = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
    median_gap = float(np.median(gaps))
    if median_gap < 4:
        return None
    if any(abs(g - median_gap) / median_gap > tolerance for g in gaps):
        return None
    return int(round(median_gap))


def _best_uniform_cluster(positions: list[int], min_count: int = 3,
                           tolerance: float = 0.20) -> list[int] | None:
    """Longest consecutive sub-sequence of positions with uniform spacing."""
    best: list[int] | None = None
    n = len(positions)
    for start in range(n):
        for end in range(start + min_count, n + 1):
            subset = positions[start:end]
            if _uniform_spacing(subset, tolerance) is not None:
                if best is None or len(subset) > len(best):
                    best = subset
    return best


def find_grid(img: np.ndarray) -> GridInfo | None:
    mask = _grid_line_mask(img)

    # Directional dilation bridges sub-pixel gaps in thin lines
    h_mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1)))
    v_mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9)))

    # 50px open filters out short corner arcs, keeping only full grid lines
    h_lines = cv2.morphologyEx(h_mask, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1)))
    v_lines = cv2.morphologyEx(v_mask, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50)))


    expand = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    intersections = cv2.bitwise_and(
        cv2.dilate(h_lines, expand),
        cv2.dilate(v_lines, expand),
    )

    h_positions = _merge_close(_line_positions(intersections, axis=1))
    v_positions = _merge_close(_line_positions(intersections, axis=0))

    if not h_positions or not v_positions:
        return None

    h_cluster = _best_uniform_cluster(h_positions)
    v_cluster = _best_uniform_cluster(v_positions)

    if h_cluster is None or v_cluster is None:
        return None

    rows = len(h_cluster) - 1
    cols = len(v_cluster) - 1
    if rows < 2 or cols < 2:
        return None

    return GridInfo(
        x      = v_cluster[0],
        y      = h_cluster[0],
        width  = v_cluster[-1] - v_cluster[0],
        height = h_cluster[-1] - h_cluster[0],
        rows   = rows,
        cols   = cols,
    )


def _preprocess_circle_roi(roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2,
    )
    return cv2.resize(thresh, (64, 64), interpolation=cv2.INTER_CUBIC)


def find_numbers(img: np.ndarray, grid: GridInfo) -> list[NumberCell]:
    grid_crop = img[grid.y : grid.y + grid.height, grid.x : grid.x + grid.width]
    gray = cv2.cvtColor(grid_crop, cv2.COLOR_BGR2GRAY)

    _, dark_mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, k)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cell_area = grid.cell_w * grid.cell_h
    results: list[NumberCell] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (cell_area * 0.08 < area < cell_area * 0.75):
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0 or (4 * np.pi * area / perimeter ** 2) < 0.5:
            continue

        bx, by, bw, bh = cv2.boundingRect(cnt)
        col = max(0, min(int((bx + bw / 2) / grid.cell_w), grid.cols - 1))
        row = max(0, min(int((by + bh / 2) / grid.cell_h), grid.rows - 1))

        roi = grid_crop[max(0, by - 2): by + bh + 2, max(0, bx - 2): bx + bw + 2]
        if roi.size == 0:
            continue

        text = pytesseract.image_to_string(
            _preprocess_circle_roi(roi), config=TESSERACT_DIGIT_CONFIG
        ).strip()
        if text.isdigit():
            results.append(NumberCell(row=row, col=col, number=int(text)))

    results.sort(key=lambda c: c.number)
    return results


def draw_debug(img: np.ndarray, grid: GridInfo, cells: list[NumberCell]) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (grid.x, grid.y),
                  (grid.x + grid.width, grid.y + grid.height), (0, 255, 0), 2)
    for r in range(grid.rows + 1):
        y = int(grid.y + r * grid.cell_h)
        cv2.line(out, (grid.x, y), (grid.x + grid.width, y), (0, 200, 0), 1)
    for c in range(grid.cols + 1):
        x = int(grid.x + c * grid.cell_w)
        cv2.line(out, (x, grid.y), (x, grid.y + grid.height), (0, 200, 0), 1)
    for cell in cells:
        cx = int(grid.x + (cell.col + 0.5) * grid.cell_w)
        cy = int(grid.y + (cell.row + 0.5) * grid.cell_h)
        cv2.putText(out, str(cell.number), (cx - 8, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return out


def detect_board(debug: bool = False) -> tuple[GridInfo | None, list[NumberCell]]:
    print("Capturing screen ...")
    img = capture_screen()

    print("Detecting grid ...")
    grid = find_grid(img)
    if grid is None:
        print("ERROR: Could not find the game grid on screen.")
        return None, []

    print(f"Grid found: {grid.cols}x{grid.rows} at ({grid.x}, {grid.y}), "
          f"{grid.width}x{grid.height}px, cell ~{grid.cell_w:.1f}x{grid.cell_h:.1f}px")

    print("Reading numbers ...")
    cells = find_numbers(img, grid)

    if cells:
        print(f"Found {len(cells)} number(s):")
        for c in cells:
            print(f"  {c.number:2d}  ->  row {c.row}, col {c.col}")
    else:
        print("WARNING: No numbered cells detected.")

    if debug:
        cv2.imshow("Zip -- board detection", draw_debug(img, grid, cells))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return grid, cells


if __name__ == "__main__":
    detect_board(debug=True)
