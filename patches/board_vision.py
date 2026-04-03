"""Patches board vision — detect the grid, seeds, numbers, and shape types.

The LinkedIn Patches game presents:
  - A square grid (typically 6×6) with light-grey **dashed** grid lines.
  - Coloured "seed" shapes in some cells, each with:
      • a colour
      • optionally a white number indicating the patch size
      • a dashed outline indicating the shape type (square / wide / tall)
  - Seed shapes without a number have unknown size.

Detection pipeline:
  1. Find the grid via Canny edge detection + morphological closing (to
     bridge dashed gaps) + intersection clustering.
  2. Find seed cells by detecting saturated coloured blobs.
  3. Read numbers on seeds via digit-template matching.
  4. Determine shape type from the dashed outline aspect ratio.
"""
from __future__ import annotations

import mss
import numpy as np
import cv2
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEMPLATE_DIR = Path(__file__).parent / "templates"
TEMPLATE_SIZE = (64, 64)
MATCH_THRESHOLD = 0.55

_MIN_LINE_LEN = 40          # morphological open length for grid lines
_DASH_BRIDGE = 15            # dilation to bridge dashed-line gaps
_CROP_FRAC = 0.60            # centre-crop fraction to avoid grid lines
_SAT_THRESHOLD = 50          # minimum saturation to count as "coloured"
_COLOUR_FRAC = 0.06          # min fraction of cell area for a seed

_digit_templates: dict[int, list[np.ndarray]] = {}
_shape_templates: dict[str, list[np.ndarray]] = {}
_SHAPE_TYPES = ['wide', 'tall', 'square', 'any']


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Screen capture
# ---------------------------------------------------------------------------

def capture_screen() -> np.ndarray:
    with mss.mss() as sct:
        shot = sct.grab(sct.monitors[1])
        return cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)


# ---------------------------------------------------------------------------
# Game-region localisation
# ---------------------------------------------------------------------------

def _find_game_region(img: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return (x, y, w, h) of the white game card containing the Patches grid.

    Strategy:
      1. Threshold to find near-white pixels (the game card background).
      2. Keep only large contiguous white blobs.
      3. Among those, find the one that also contains the most saturated
         (coloured) pixels — that's the game card.
    Returns None if no suitable region is found.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Near-white: value > 235
    _, white = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)

    # Flood-fill small holes so the card interior is solid white
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    white_closed = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(white_closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    img_area = img.shape[0] * img.shape[1]
    # Must be at least 1% of screen area to be a game card, not a tooltip
    min_area = img_area * 0.01

    # Build saturation map once
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]

    best_region = None
    best_score = -1.0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < min_area:
            continue
        # Aspect ratio: game card is roughly square-ish (0.3 – 3.0)
        ratio = w / h if h > 0 else 0
        if not (0.3 <= ratio <= 3.0):
            continue

        # Count saturated pixels inside this bounding box
        roi_sat = sat[y:y + h, x:x + w]
        sat_count = int((roi_sat > _SAT_THRESHOLD).sum())
        # Score = saturated fraction (want some colour but not a photo/video)
        sat_frac = sat_count / area
        if sat_frac < 0.002:
            continue  # no coloured seeds visible

        score = sat_frac * np.sqrt(area)  # prefer larger regions with colour
        if score > best_score:
            best_score = score
            best_region = (x, y, w, h)

    return best_region


# ---------------------------------------------------------------------------
# Grid detection (dashed lines → Canny + bridging)
# ---------------------------------------------------------------------------

def _line_positions(line_img: np.ndarray, axis: int) -> list[int]:
    """Project a binary image along *axis* and return centres of bright bands."""
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
    """Return (h_lines, v_lines) binary masks.

    Patches uses dashed grid lines, so after Canny we dilate along each axis
    to bridge the gaps between dashes, then erode back to thin the result.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 30, 90)

    # Bridge dashes: dilate horizontally then erode to restore thickness
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


def find_grid(img: np.ndarray,
              region_offset: tuple[int, int] = (0, 0)) -> GridInfo | None:
    """Detect the grid inside *img* (which may be a crop of the full screen).

    *region_offset* = (ox, oy) is added to all returned coordinates so that
    GridInfo always stores screen-space positions.

    Strategy: cluster the raw H/V line positions directly (no intersection
    filter) so that UI noise from other parts of the panel does not destroy
    the grid projection.  The best uniform-spacing cluster selects the grid.
    """
    ox, oy = region_offset
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

    # Try to make the grid square.  Prefer trimming an extra line from the
    # longer axis (spurious border) over extending the shorter one (missing
    # grid line), then fall back to extending if trimming cannot help.
    if rows != cols:
        # Pass 1: trim the longer cluster (front or back)
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

        # Pass 2: if still not square, extend the shorter axis
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
        x=v_cluster[0] + ox,
        y=h_cluster[0] + oy,
        width=v_cluster[-1] - v_cluster[0],
        height=h_cluster[-1] - h_cluster[0],
        rows=rows,
        cols=cols,
    )


# ---------------------------------------------------------------------------
# Seed detection — find coloured blobs in cells
# ---------------------------------------------------------------------------

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
    """Return True if the ROI has enough saturated (coloured) pixels."""
    if roi.size == 0:
        return False
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sat_mask = hsv[:, :, 1] > _SAT_THRESHOLD
    area = roi.shape[0] * roi.shape[1]
    return int(sat_mask.sum()) > area * _COLOUR_FRAC


def _detect_shape_type(roi: np.ndarray) -> str:
    """Classify the shape type using pattern matching against shape templates.

    The game uses four shape types: wide, tall, square, any.
    This function matches the seed ROI against learned binary templates for
    each shape type and returns the best match.

    If templates are not yet loaded, falls back to 'any' (graceful degradation).
    """
    if roi.size == 0:
        return 'any'

    templates = _load_shape_templates()

    # If no templates loaded yet, default to 'any'
    if not templates:
        return 'any'

    # Preprocess ROI: fill white numbers with seed color, convert to binary
    processed = _preprocess_roi_for_shape(roi)

    # Score against all shape templates
    scores = _score_shape_templates(processed)

    # Find best match
    best_shape = 'any'
    best_score = -1.0
    for shape, score in scores.items():
        if score > best_score:
            best_score = score
            best_shape = shape

    # Only accept if above threshold, otherwise default to 'any'
    if best_score >= MATCH_THRESHOLD:
        return best_shape
    return 'any'


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


# ---------------------------------------------------------------------------
# Number detection on seeds
# ---------------------------------------------------------------------------

def _load_digit_templates() -> dict[int, list[np.ndarray]]:
    """Load digit templates from templates directory.

    Naming: <digit>.png or <digit>_<NNN>.png
    """
    if _digit_templates:
        return _digit_templates
    for path in sorted(TEMPLATE_DIR.glob("*.png")):
        stem = path.stem
        digit_str = stem.split("_", 1)[0]
        if not digit_str.isdigit():
            continue
        n = int(digit_str)
        if not 1 <= n <= 9:
            continue
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        resized = cv2.resize(binary, TEMPLATE_SIZE, interpolation=cv2.INTER_CUBIC)
        _digit_templates.setdefault(n, []).append(resized)
    return _digit_templates


def _load_shape_templates() -> dict[str, list[np.ndarray]]:
    """Load shape templates from templates directory.

    Naming: <shape>.png or <shape>_<NNN>.png where shape in {wide, tall, square, any}
    """
    if _shape_templates:
        return _shape_templates
    for path in sorted(TEMPLATE_DIR.glob("*.png")):
        stem = path.stem
        shape_name = stem.split("_", 1)[0]
        if shape_name not in _SHAPE_TYPES:
            continue
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        resized = cv2.resize(binary, TEMPLATE_SIZE, interpolation=cv2.INTER_CUBIC)
        _shape_templates.setdefault(shape_name, []).append(resized)
    return _shape_templates


def _preprocess_roi_for_digit(roi: np.ndarray) -> np.ndarray:
    """Convert a cell ROI to binary for digit template matching.

    Seeds with numbers show white text on a coloured background.
    We isolate the white channel: high value, low saturation.
    """
    if len(roi.shape) == 3:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # White text: low saturation, high value
        white_mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 80, 255]))
        # Also try simple grayscale threshold
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, simple = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Use whichever has more contrast
        if white_mask.sum() > 0:
            binary = white_mask
        else:
            binary = simple
    else:
        _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return cv2.resize(binary, TEMPLATE_SIZE, interpolation=cv2.INTER_CUBIC)


def _preprocess_roi_for_shape(roi: np.ndarray) -> np.ndarray:
    """Convert a seed ROI to binary for shape template matching.

    Key step: Fill white numbers (text) with the seed color to prevent them
    from creating false edges that confuse shape matching.
    """
    if roi.size == 0:
        return np.zeros(TEMPLATE_SIZE, dtype=np.uint8)

    if len(roi.shape) == 3:
        roi_copy = roi.copy()
        hsv = cv2.cvtColor(roi_copy, cv2.COLOR_BGR2HSV)

        # Detect white text: low saturation, high value
        white_mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 80, 255]))

        if white_mask.sum() > 0:
            # Dilate white regions slightly to fill gaps in letters
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            white_dilated = cv2.dilate(white_mask, kernel, iterations=1)

            # Get dominant seed color from non-white regions
            # Sample the center of the ROI (likely to be pure seed color)
            cy, cx = roi_copy.shape[0] // 2, roi_copy.shape[1] // 2
            h_start, h_end = max(0, cy - 5), min(roi_copy.shape[0], cy + 5)
            w_start, w_end = max(0, cx - 5), min(roi_copy.shape[1], cx + 5)
            center_sample = roi_copy[h_start:h_end, w_start:w_end]

            if center_sample.size > 0:
                # Get median color from center (most likely pure seed)
                seed_color = np.median(center_sample, axis=(0, 1)).astype(np.uint8)
                # Fill white regions with seed color
                for i in range(3):
                    roi_copy[white_dilated > 0, i] = seed_color[i]

        # Convert to grayscale and apply Otsu threshold
        gray = cv2.cvtColor(roi_copy, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return cv2.resize(binary, TEMPLATE_SIZE, interpolation=cv2.INTER_CUBIC)


def _score_shape_templates(roi_processed: np.ndarray) -> dict[str, float]:
    """Return the best match score for every loaded shape template."""
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
        roi = _cell_roi(crop, grid, seed.row, seed.col, frac=0.55)
        num = _detect_number(roi)
        if num > 0:
            seed.size = num




# ---------------------------------------------------------------------------
# Debug visualisation
# ---------------------------------------------------------------------------

def draw_debug(img: np.ndarray, board: PatchesBoard,
               solution: list[list[int]] | None = None) -> np.ndarray:
    out = img.copy()
    g = board.grid

    # Draw grid boundary and lines
    cv2.rectangle(out, (g.x, g.y), (g.x + g.width, g.y + g.height), (0, 255, 0), 2)
    for r in range(g.rows + 1):
        y = int(g.y + r * g.cell_h)
        cv2.line(out, (g.x, y), (g.x + g.width, y), (0, 200, 0), 1)
    for c in range(g.cols + 1):
        x = int(g.x + c * g.cell_w)
        cv2.line(out, (x, g.y), (x, g.y + g.height), (0, 200, 0), 1)

    # Mark seed cells
    for i, s in enumerate(board.seeds):
        px = int(g.x + (s.col + 0.5) * g.cell_w)
        py = int(g.y + (s.row + 0.5) * g.cell_h)
        label = chr(ord('A') + i % 26)
        info = f"{label}:{s.size}" if s.size else label
        info += f" {s.shape_type[0].upper()}"
        cv2.putText(out, info, (px - 20, py + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Draw solution overlay if provided
    if solution is not None:
        # Label each cell with its seed index
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
    """Side-by-side debug panel for grid line detection stages."""
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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def detect_board(debug: bool = False) -> PatchesBoard | None:
    print("Capturing screen...")
    img = capture_screen()

    print("Localising game panel...")
    region = _find_game_region(img)
    if region is not None:
        rx, ry, rw, rh = region
        print(f"  Game region: ({rx},{ry}) {rw}x{rh}px")
        panel = img[ry:ry + rh, rx:rx + rw]
        offset = (rx, ry)
    else:
        print("  WARNING: Could not localise game panel — using full screen")
        panel = img
        offset = (0, 0)

    print("Detecting grid...")
    grid = find_grid(panel, region_offset=offset)
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
    detect_board(debug=True)
