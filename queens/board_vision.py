import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2
import screen

_MIN_LINE_LEN = 25
_CROP_FRAC = 0.50


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


class QueensBoard:
    def __init__(self, grid: GridInfo, regions: list[list[int]]):
        self.grid = grid
        self.regions = regions  # regions[r][c] = region_id (0-based)


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
    gaps = [cluster[i + 1] - cluster[i] for i in range(len(cluster) - 1)]
    gap = int(np.median(gaps))
    candidates = []
    if cluster[0] - gap >= 0:
        candidates.append([cluster[0] - gap] + cluster)
    if cluster[-1] + gap < image_size:
        candidates.append(cluster + [cluster[-1] + gap])
    return candidates


def _extract_grid_lines(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (h_lines, v_lines) binary masks using Canny edge detection."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 90)

    h_lines = cv2.morphologyEx(
        edges, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (_MIN_LINE_LEN, 1)))
    v_lines = cv2.morphologyEx(
        edges, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, _MIN_LINE_LEN)))
    return h_lines, v_lines


def find_grid(img: np.ndarray) -> GridInfo | None:
    h_lines, v_lines = _extract_grid_lines(img)

    # Estimate cell spacing from raw projections to size the dilation kernel
    h_raw = _merge_close(_line_positions(h_lines, axis=1))
    v_raw = _merge_close(_line_positions(v_lines, axis=0))
    if len(h_raw) < 2 or len(v_raw) < 2:
        return None
    all_gaps = ([h_raw[i + 1] - h_raw[i] for i in range(len(h_raw) - 1)]
              + [v_raw[i + 1] - v_raw[i] for i in range(len(v_raw) - 1)])
    cell_est = float(np.median(all_gaps))
    expand_sz = max(9, int(cell_est * 0.35))

    expand = cv2.getStructuringElement(cv2.MORPH_RECT, (expand_sz, expand_sz))
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


def _cell_color(crop: np.ndarray, grid: GridInfo, r: int, c: int) -> np.ndarray:
    """Return the mean LAB colour of the centre of cell (r, c)."""
    half_w = grid.cell_w * _CROP_FRAC / 2
    half_h = grid.cell_h * _CROP_FRAC / 2
    cx = (c + 0.5) * grid.cell_w
    cy = (r + 0.5) * grid.cell_h
    x1 = max(0, int(cx - half_w))
    y1 = max(0, int(cy - half_h))
    x2 = min(crop.shape[1], int(cx + half_w))
    y2 = min(crop.shape[0], int(cy + half_h))

    roi = crop[y1:y2, x1:x2]
    if roi.size == 0:
        return np.zeros(3, dtype=np.float32)

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    return lab.mean(axis=(0, 1)).astype(np.float32)


def find_regions(img: np.ndarray, grid: GridInfo) -> list[list[int]]:
    """Cluster cell colours in LAB space with K-means (K = grid size)."""
    crop = img[grid.y: grid.y + grid.height, grid.x: grid.x + grid.width]
    n = grid.rows

    flat = np.array([_cell_color(crop, grid, r, c)
                     for r in range(n) for c in range(n)], dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    _, labels, _ = cv2.kmeans(flat, n, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    labels = labels.flatten()
    return [[int(labels[r * n + c]) for c in range(n)] for r in range(n)]


def draw_debug(img: np.ndarray, board: QueensBoard) -> np.ndarray:
    out = img.copy()
    g = board.grid

    cv2.rectangle(out, (g.x, g.y), (g.x + g.width, g.y + g.height), (0, 255, 0), 2)
    for r in range(g.rows + 1):
        y = int(g.y + r * g.cell_h)
        cv2.line(out, (g.x, y), (g.x + g.width, y), (0, 200, 0), 1)
    for c in range(g.cols + 1):
        x = int(g.x + c * g.cell_w)
        cv2.line(out, (x, g.y), (x, g.y + g.height), (0, 200, 0), 1)

    for r in range(g.rows):
        for c in range(g.cols):
            rid = board.regions[r][c]
            px = int(g.x + (c + 0.5) * g.cell_w)
            py = int(g.y + (r + 0.5) * g.cell_h)
            cv2.putText(out, str(rid), (px - 6, py + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(out, str(rid), (px - 6, py + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return out


def detect_board(debug: bool = False) -> QueensBoard | None:
    print("Capturing screen...")
    img = screen.capture()

    print("Detecting grid...")
    grid = find_grid(img)
    if grid is None:
        print("ERROR: Could not find the Queens game grid on screen.")
        return None

    print(f"Grid found: {grid.cols}x{grid.rows} at ({grid.x}, {grid.y}), "
          f"{grid.width}x{grid.height}px, cell ~{grid.cell_w:.1f}x{grid.cell_h:.1f}px")

    print("Detecting colour regions...")
    regions = find_regions(img, grid)

    n_regions = len(set(regions[r][c] for r in range(grid.rows) for c in range(grid.cols)))
    print(f"  {n_regions} regions detected")
    for row in regions:
        print("  " + " ".join(str(r) for r in row))

    board = QueensBoard(grid, regions)

    if debug:
        cv2.imshow("Queens — board detection", draw_debug(img, board))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return board


if __name__ == "__main__":
    screen.init_game_region()
    detect_board(debug=True)
