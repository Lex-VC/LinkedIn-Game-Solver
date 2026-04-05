from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2
import screen


TEMPLATE_DIR = Path(__file__).parent / "templates"
TEMPLATE_SIZE = (64, 64)
MATCH_THRESHOLD = 0.55

_MIN_LINE_LEN = 40
_DASH_BRIDGE = 15
_CROP_FRAC = 0.90
_SAT_THRESHOLD = 20
_COLOUR_FRAC = 0.06

_digit_templates: dict[int, list[np.ndarray]] = {}
_shape_templates: dict[str, list[np.ndarray]] = {}
_SHAPE_TYPES = ['wide', 'tall', 'square', 'any']


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


class PatchSeed:
    """One seed on the board."""
    def __init__(self, row: int, col: int, size: int, shape_type: str):
        self.row = row
        self.col = col
        self.size = size                # 0 means "unknown"
        self.shape_type = shape_type    # 'square' | 'wide' | 'tall' | 'any'

    def __repr__(self) -> str:
        return (f"PatchSeed(r={self.row}, c={self.col}, "
                f"size={self.size}, type={self.shape_type!r})")


class PatchesBoard:
    def __init__(self, grid: GridInfo, seeds: list[PatchSeed]):
        self.grid = grid
        self.seeds = seeds


def _line_positions(line_img: np.ndarray, axis: int) -> list[int]:
    """Project a binary image along axis and return centres of bright bands."""
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
    """Return (h_lines, v_lines) binary masks, bridging dashed gaps."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 30, 90)

    h_bridge = cv2.dilate(edges,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (_DASH_BRIDGE, 1)))
    h_lines = cv2.morphologyEx(
        h_bridge, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (_MIN_LINE_LEN, 1)))

    v_bridge = cv2.dilate(edges,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (1, _DASH_BRIDGE)))
    v_lines = cv2.morphologyEx(
        v_bridge, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, _MIN_LINE_LEN)))

    return h_lines, v_lines


def find_grid(img: np.ndarray) -> GridInfo | None:
    h_lines, v_lines = _extract_grid_lines(img)

    h_raw = _merge_close(_line_positions(h_lines, axis=1))
    v_raw = _merge_close(_line_positions(v_lines, axis=0))

    if len(h_raw) < 2 or len(v_raw) < 2:
        return None

    h_cluster = _best_uniform_cluster(h_raw, min_count=4, tolerance=0.15)
    v_cluster = _best_uniform_cluster(v_raw, min_count=4, tolerance=0.15)

    if h_cluster is None or v_cluster is None:
        return None

    rows = len(h_cluster) - 1
    cols = len(v_cluster) - 1

    if rows != cols:
        for _ in range(3):
            rows = len(h_cluster) - 1
            cols = len(v_cluster) - 1
            if rows == cols:
                break
            trimmed = False
            if rows > cols:
                for candidate in [h_cluster[1:], h_cluster[:-1]]:
                    if len(candidate) - 1 == cols and _uniform_spacing(candidate, 0.15):
                        h_cluster = candidate
                        trimmed = True
                        break
            else:
                for candidate in [v_cluster[1:], v_cluster[:-1]]:
                    if len(candidate) - 1 == rows and _uniform_spacing(candidate, 0.15):
                        v_cluster = candidate
                        trimmed = True
                        break
            if not trimmed:
                break

        if len(h_cluster) - 1 != len(v_cluster) - 1:
            target = max(len(h_cluster) - 1, len(v_cluster) - 1)
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


def _cell_roi(crop: np.ndarray, grid: GridInfo, r: int, c: int,
              frac: float = _CROP_FRAC) -> np.ndarray:
    """Return the centre-cropped ROI of cell (r, c)."""
    half_w = grid.cell_w * frac / 2
    half_h = grid.cell_h * frac / 2
    cx = (c + 0.5) * grid.cell_w
    cy = (r + 0.5) * grid.cell_h
    x1 = max(0, int(cx - half_w))
    y1 = max(0, int(cy - half_h))
    x2 = min(crop.shape[1], int(cx + half_w))
    y2 = min(crop.shape[0], int(cy + half_h))
    return crop[y1:y2, x1:x2]


def _is_seed_cell(roi: np.ndarray) -> bool:
    if roi.size == 0:
        return False
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sat_mask = hsv[:, :, 1] > _SAT_THRESHOLD
    area = roi.shape[0] * roi.shape[1]
    return int(sat_mask.sum()) > area * _COLOUR_FRAC


def _detect_shape_type(roi: np.ndarray) -> str:
    """Classify shape type via template matching against shape templates."""
    if roi.size == 0:
        return 'any'

    templates = _load_shape_templates()
    if not templates:
        return 'any'

    processed = _preprocess_roi_for_shape(roi)
    scores = _score_shape_templates(processed)

    best_shape = 'any'
    best_score = -1.0
    for shape, score in scores.items():
        if score > best_score:
            best_score = score
            best_shape = shape

    return best_shape if best_score >= MATCH_THRESHOLD else 'any'


def find_seeds(img: np.ndarray, grid: GridInfo) -> list[PatchSeed]:
    """Detect seed cells and classify their shape type."""
    crop = img[grid.y: grid.y + grid.height, grid.x: grid.x + grid.width]
    seeds: list[PatchSeed] = []

    for r in range(grid.rows):
        for c in range(grid.cols):
            roi = _cell_roi(crop, grid, r, c)
            if _is_seed_cell(roi):
                shape = _detect_shape_type(roi)
                seeds.append(PatchSeed(row=r, col=c, size=0, shape_type=shape))
    return seeds


def _load_template_binary(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    resized = cv2.resize(img, TEMPLATE_SIZE, interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
    return binary


def _load_digit_templates() -> dict[int, list[np.ndarray]]:
    if _digit_templates:
        return _digit_templates
    for path in sorted(TEMPLATE_DIR.glob("*.png")):
        digit_str = path.stem.split("_", 1)[0]
        if not digit_str.isdigit():
            continue
        n = int(digit_str)
        if not 1 <= n <= 9:
            continue
        tmpl = _load_template_binary(path)
        if tmpl is not None:
            _digit_templates.setdefault(n, []).append(tmpl)
    return _digit_templates


def _load_shape_templates() -> dict[str, list[np.ndarray]]:
    if _shape_templates:
        return _shape_templates
    for path in sorted(TEMPLATE_DIR.glob("*.png")):
        shape_name = path.stem.split("_", 1)[0]
        if shape_name not in _SHAPE_TYPES:
            continue
        tmpl = _load_template_binary(path)
        if tmpl is not None:
            _shape_templates.setdefault(shape_name, []).append(tmpl)
    return _shape_templates


def _sat_mask_and_holes(roi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (sat_mask, holes) for a seed ROI."""
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV) if len(roi.shape) == 3 else \
          cv2.cvtColor(cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)

    sat_mask = (hsv[:, :, 1] > _SAT_THRESHOLD).astype(np.uint8) * 255

    h, w = sat_mask.shape
    inv = cv2.bitwise_not(sat_mask)
    ff = inv.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, ff_mask, (0, 0), 0)
    holes = ff

    return sat_mask, holes


def _to_template_binary(mask: np.ndarray) -> np.ndarray:
    resized = cv2.resize(mask, TEMPLATE_SIZE, interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
    return binary


def _preprocess_roi_for_shape(roi: np.ndarray) -> np.ndarray:
    """Solid shape silhouette: sat_mask with number holes filled in."""
    if roi.size == 0:
        return np.zeros(TEMPLATE_SIZE, dtype=np.uint8)
    sat_mask, holes = _sat_mask_and_holes(roi)
    filled = cv2.bitwise_or(sat_mask, holes)
    return _to_template_binary(filled)


def _preprocess_roi_for_digit(roi: np.ndarray) -> np.ndarray | None:
    """Number silhouette only: interior holes enclosed by the seed blob."""
    if roi.size == 0:
        return None
    _, holes = _sat_mask_and_holes(roi)
    if holes.sum() == 0:
        return None
    return _to_template_binary(holes)


def _score_shape_templates(roi_processed: np.ndarray) -> dict[str, float]:
    templates = _load_shape_templates()
    scores: dict[str, float] = {}
    for shape_name in _SHAPE_TYPES:
        if shape_name not in templates or not templates[shape_name]:
            scores[shape_name] = -1.0
            continue
        scores[shape_name] = max(
            float(cv2.matchTemplate(roi_processed, tmpl, cv2.TM_CCOEFF_NORMED)[0][0])
            for tmpl in templates[shape_name]
        )
    return scores


def _detect_number(roi: np.ndarray) -> int:
    """Try to read a digit from the ROI. Returns 0 if no number found."""
    templates = _load_digit_templates()
    if not templates:
        return 0

    processed = _preprocess_roi_for_digit(roi)
    if processed is None:
        return 0

    best_num = 0
    best_score = -1.0

    for num, samples in templates.items():
        score = max(
            float(cv2.matchTemplate(processed, tmpl, cv2.TM_CCOEFF_NORMED)[0][0])
            for tmpl in samples
        )
        if score > best_score:
            best_score = score
            best_num = num

    return best_num if best_score >= MATCH_THRESHOLD else 0


def read_seed_numbers(img: np.ndarray, grid: GridInfo,
                      seeds: list[PatchSeed]) -> None:
    """Update each seed's .size by reading the digit in its cell."""
    crop = img[grid.y: grid.y + grid.height, grid.x: grid.x + grid.width]
    for seed in seeds:
        roi = _cell_roi(crop, grid, seed.row, seed.col, frac=_CROP_FRAC)
        num = _detect_number(roi)
        if num > 0:
            seed.size = num


def draw_debug(img: np.ndarray, board: PatchesBoard,
               solution: list[list[int]] | None = None) -> np.ndarray:
    out = img.copy()
    g = board.grid

    cv2.rectangle(out, (g.x, g.y), (g.x + g.width, g.y + g.height), (0, 255, 0), 2)
    for r in range(g.rows + 1):
        y = int(g.y + r * g.cell_h)
        cv2.line(out, (g.x, y), (g.x + g.width, y), (0, 200, 0), 1)
    for c in range(g.cols + 1):
        x = int(g.x + c * g.cell_w)
        cv2.line(out, (x, g.y), (x, g.y + g.height), (0, 200, 0), 1)

    for i, s in enumerate(board.seeds):
        px = int(g.x + (s.col + 0.5) * g.cell_w)
        py = int(g.y + (s.row + 0.5) * g.cell_h)
        label = chr(ord('A') + i % 26)
        info = f"{label}:{s.size}" if s.size else label
        info += f" {s.shape_type[0].upper()}"
        cv2.putText(out, info, (px - 20, py + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    if solution is not None:
        for r in range(g.rows):
            for c in range(g.cols):
                si = solution[r][c]
                if si < 0:
                    continue
                px = int(g.x + (c + 0.5) * g.cell_w)
                py = int(g.y + (r + 0.5) * g.cell_h)
                lbl = chr(ord('a') + si % 26)
                cv2.putText(out, lbl, (px - 5, py + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 200), 1)

    return out


def draw_grid_detection(img: np.ndarray) -> np.ndarray:
    h_lines, v_lines = _extract_grid_lines(img)

    h_raw = _merge_close(_line_positions(h_lines, axis=1))
    v_raw = _merge_close(_line_positions(v_lines, axis=0))
    all_gaps = []
    if len(h_raw) >= 2:
        all_gaps += [h_raw[i + 1] - h_raw[i] for i in range(len(h_raw) - 1)]
    if len(v_raw) >= 2:
        all_gaps += [v_raw[i + 1] - v_raw[i] for i in range(len(v_raw) - 1)]
    if all_gaps:
        cell_est = float(np.median(all_gaps))
        expand_sz = max(9, int(cell_est * 0.35))
    else:
        expand_sz = 25

    expand = cv2.getStructuringElement(cv2.MORPH_RECT, (expand_sz, expand_sz))
    intersections = cv2.bitwise_and(
        cv2.dilate(h_lines, expand),
        cv2.dilate(v_lines, expand),
    )

    def _to_bgr(mask: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    overlay = img.copy()
    grid = find_grid(img)
    if grid is not None:
        g = grid
        cv2.rectangle(overlay, (g.x, g.y), (g.x + g.width, g.y + g.height),
                      (0, 255, 0), 2)
        for r in range(g.rows + 1):
            y = int(g.y + r * g.cell_h)
            cv2.line(overlay, (g.x, y), (g.x + g.width, y), (0, 200, 0), 1)
        for c in range(g.cols + 1):
            x = int(g.x + c * g.cell_w)
            cv2.line(overlay, (x, g.y), (x, g.y + g.height), (0, 200, 0), 1)
        label = f"{g.cols}x{g.rows} @ ({g.x},{g.y})"
    else:
        label = "NO GRID FOUND"

    h_bgr = _to_bgr(h_lines)
    v_bgr = _to_bgr(v_lines)
    inter_bgr = _to_bgr(intersections)

    for panel, text in [
        (img, "original"),
        (h_bgr, "H lines (bridged)"),
        (v_bgr, "V lines (bridged)"),
        (inter_bgr, "intersections"),
        (overlay, label),
    ]:
        cv2.putText(panel, text, (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

    return np.hstack([img, h_bgr, v_bgr, inter_bgr, overlay])


def detect_board(debug: bool = False) -> PatchesBoard | None:
    print("Capturing screen...")
    img = screen.capture()

    print("Detecting grid...")
    grid = find_grid(img)
    if grid is None:
        print("ERROR: Could not find the Patches game grid on screen.")
        return None

    print(f"Grid found: {grid.cols}x{grid.rows} at ({grid.x}, {grid.y}), "
          f"{grid.width}x{grid.height}px, cell ~{grid.cell_w:.1f}x{grid.cell_h:.1f}px")

    print("Detecting seed cells and shape types...")
    seeds = find_seeds(img, grid)
    print(f"  Found {len(seeds)} seed(s)")

    print("Reading seed numbers...")
    read_seed_numbers(img, grid, seeds)

    for s in seeds:
        print(f"  [{s.row},{s.col}] size={s.size} type={s.shape_type}")

    board = PatchesBoard(grid, seeds)

    if debug:
        cv2.imshow("Patches — board detection", draw_debug(img, board))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return board


if __name__ == "__main__":
    screen.init_game_region()
    detect_board(debug=True)
