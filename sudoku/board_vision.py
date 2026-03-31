import mss
import numpy as np
import cv2
from pathlib import Path

TEMPLATE_DIR = Path(__file__).parent / "templates"
TEMPLATE_SIZE = (64, 64)
MATCH_THRESHOLD = 0.55

_templates: dict[int, list[np.ndarray]] = {}


def _augment_template(img: np.ndarray, shift: int = 4, step: int = 2) -> list[np.ndarray]:
    """Generate shifted variants of a template to handle crop misalignment."""
    h, w = img.shape
    variants = [img]
    for dx in range(-shift, shift + 1, step):
        for dy in range(-shift, shift + 1, step):
            if dx == 0 and dy == 0:
                continue
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            variants.append(cv2.warpAffine(img, M, (w, h), borderValue=0))
    return variants


def _load_templates() -> dict[int, list[np.ndarray]]:
    """Load and preprocess digit templates from the templates directory.
    Files must be named 1.png through 9.png.
    """
    if _templates:
        return _templates
    for path in TEMPLATE_DIR.glob("*.png"):
        if path.stem.isdigit():
            n = int(path.stem)
            if 1 <= n <= 9:
                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    resized = cv2.resize(binary, TEMPLATE_SIZE, interpolation=cv2.INTER_CUBIC)
                    _templates[n] = _augment_template(resized)
    return _templates


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


def _extend_cluster(cluster: list[int], image_size: int) -> list[list[int]]:
    """Return candidates with one extra line prepended and/or appended."""
    gaps = [cluster[i + 1] - cluster[i] for i in range(len(cluster) - 1)]
    gap = int(np.median(gaps))
    candidates = []
    if cluster[0] - gap >= 0:
        candidates.append([cluster[0] - gap] + cluster)
    if cluster[-1] + gap < image_size:
        candidates.append(cluster + [cluster[-1] + gap])
    return candidates


def find_grid(img: np.ndarray) -> GridInfo | None:
    mask = _grid_line_mask(img)

    # 50px open filters out short corner arcs, keeping only full grid lines
    h_lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1)))
    v_lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
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

    # If not square, try extending the shorter axis one line at a time
    if rows != cols:
        target = max(rows, cols)
        for _ in range(2):
            rows = len(h_cluster) - 1
            cols = len(v_cluster) - 1
            if rows == cols:
                break
            if rows < cols:
                for candidate in _extend_cluster(h_cluster, img.shape[0]):
                    if len(candidate) - 1 <= target:
                        h_cluster = candidate
                        break
            else:
                for candidate in _extend_cluster(v_cluster, img.shape[1]):
                    if len(candidate) - 1 <= target:
                        v_cluster = candidate
                        break

    rows = len(h_cluster) - 1
    cols = len(v_cluster) - 1
    if rows < 2 or cols < 2:
        return None

    return GridInfo(
        x=v_cluster[0],
        y=h_cluster[0],
        width=v_cluster[-1] - v_cluster[0],
        height=h_cluster[-1] - h_cluster[0],
        rows=rows,
        cols=cols,
    )


def _preprocess_roi(roi: np.ndarray) -> np.ndarray:
    """Convert a cell ROI to a binary image matching the template format."""
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.resize(thresh, TEMPLATE_SIZE, interpolation=cv2.INTER_CUBIC)


def _score_templates(roi_processed: np.ndarray) -> dict[int, float]:
    """Return the best match score for every loaded template digit."""
    templates = _load_templates()
    scores: dict[int, float] = {}
    for num, variants in templates.items():
        scores[num] = max(
            float(cv2.matchTemplate(roi_processed, tmpl, cv2.TM_CCOEFF_NORMED)[0][0])
            for tmpl in variants
        )
    return scores


def find_numbers(img: np.ndarray, grid: GridInfo) -> list[list[int]]:
    """Scan all cells and return a 9x9 board with 0 for empty cells."""
    grid_crop = img[grid.y: grid.y + grid.height, grid.x: grid.x + grid.width]

    # Use center 65% of each cell to avoid grid lines and sub-box borders
    crop_frac = 0.65
    board = [[0] * grid.cols for _ in range(grid.rows)]

    for r in range(grid.rows):
        for c in range(grid.cols):
            cx = (c + 0.5) * grid.cell_w
            cy = (r + 0.5) * grid.cell_h
            half_w = grid.cell_w * crop_frac / 2
            half_h = grid.cell_h * crop_frac / 2
            x1 = max(0, int(cx - half_w))
            y1 = max(0, int(cy - half_h))
            x2 = min(grid_crop.shape[1], int(cx + half_w))
            y2 = min(grid_crop.shape[0], int(cy + half_h))
            roi = grid_crop[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            processed = _preprocess_roi(roi)
            scores = _score_templates(processed)
            if not scores:
                continue
            best_num = max(scores, key=scores.__getitem__)
            if scores[best_num] >= MATCH_THRESHOLD:
                board[r][c] = best_num

    return board


def draw_debug(img: np.ndarray, grid: GridInfo, board: list[list[int]]) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (grid.x, grid.y),
                  (grid.x + grid.width, grid.y + grid.height), (0, 255, 0), 2)
    for r in range(grid.rows + 1):
        y = int(grid.y + r * grid.cell_h)
        cv2.line(out, (grid.x, y), (grid.x + grid.width, y), (0, 200, 0), 1)
    for c in range(grid.cols + 1):
        x = int(grid.x + c * grid.cell_w)
        cv2.line(out, (x, grid.y), (x, grid.y + grid.height), (0, 200, 0), 1)
    for r in range(grid.rows):
        for c in range(grid.cols):
            if board[r][c] != 0:
                px = int(grid.x + (c + 0.5) * grid.cell_w)
                py = int(grid.y + (r + 0.5) * grid.cell_h)
                cv2.putText(out, str(board[r][c]), (px - 8, py + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return out


def detect_board(debug: bool = False) -> tuple[GridInfo | None, list[list[int]]]:
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
    board = find_numbers(img, grid)

    filled = sum(1 for r in board for v in r if v != 0)
    print(f"Found {filled} pre-filled cell(s).")
    for r, row in enumerate(board):
        print("  " + " ".join(str(v) if v else "." for v in row))

    if debug:
        cv2.imshow("Sudoku -- board detection", draw_debug(img, grid, board))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return grid, board


if __name__ == "__main__":
    detect_board(debug=True)
