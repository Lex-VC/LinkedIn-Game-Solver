import mss
import numpy as np
import cv2
from pathlib import Path

SUN = 'sun'
MOON = 'moon'
EMPTY = 'empty'

EQUAL = 'equal'       # '=' — adjacent cells must have the same symbol
OPPOSITE = 'opposite' # '×' — adjacent cells must have different symbols

# After dilating grid-line masks with a 25×25 kernel, a true 1-px line creates
# an intersection band ~50px wide in the projection.  Grey pre-filled cells
# survive the morphological open and produce bands of ≈ (cell_size + 50)px —
# typically >80px.  Keeping only bands ≤ _MAX_LINE_BAND removes the grey-cell
# artefacts while accepting every real grid-line peak.
_MAX_LINE_BAND = 155

TEMPLATE_DIR            = Path(__file__).parent / "templates"
CONSTRAINT_TEMPLATE_SIZE = (32, 32)
CONSTRAINT_THRESHOLD     = 0.55

_constraint_templates: dict[str, list[np.ndarray]] = {}

# HSV colour ranges (OpenCV: H in [0, 180], S/V in [0, 255])
_SUN_LO  = np.array([10,  150, 150])
_SUN_HI  = np.array([30,  255, 255])
_MOON_LO = np.array([95,   80,  50])
_MOON_HI = np.array([130, 255, 220])

_SYMBOL_FRAC = 0.02   # min fraction of cell area to count as a symbol
_CROP_FRAC   = 0.70   # centre-crop fraction — avoids grid-line borders


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


class TangoBoard:
    def __init__(
        self,
        grid: GridInfo,
        cells: list[list[str]],
        h_constraints: dict[tuple[int, int], str],
        v_constraints: dict[tuple[int, int], str],
    ):
        self.grid = grid
        self.cells = cells              # cells[r][c] ∈ {SUN, MOON, EMPTY}
        self.h_constraints = h_constraints  # (r, c): constraint on edge below row r at col c
        self.v_constraints = v_constraints  # (r, c): constraint on edge right of col c at row r


# ---------------------------------------------------------------------------
# Screen capture
# ---------------------------------------------------------------------------

def capture_screen() -> np.ndarray:
    with mss.mss() as sct:
        shot = sct.grab(sct.monitors[1])
        return cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)


# ---------------------------------------------------------------------------
# Grid detection (shared logic + grey-cell robustness)
# ---------------------------------------------------------------------------

def _grid_line_mask(img: np.ndarray) -> np.ndarray:
    """Pixels that are medium-gray and unsaturated — the grid-line colour."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(img)
    max_diff = cv2.max(cv2.max(cv2.absdiff(b, g), cv2.absdiff(g, r)), cv2.absdiff(b, r))
    low_sat = (max_diff < 40).astype(np.uint8) * 255
    in_range = cv2.inRange(gray, 100, 235)
    return cv2.bitwise_and(low_sat, in_range)


def _line_positions(line_img: np.ndarray, axis: int,
                    max_band_width: int | None = None) -> list[int]:
    """Project a binary line image and return the centre of each bright band.

    max_band_width: when set, only bands no wider than this are kept —
    this filters out the wide blobs that grey pre-filled Tango cells produce
    in the projection after morphological opening and dilation.
    """
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
            if max_band_width is None or (i - start) <= max_band_width:
                positions.append((start + i) // 2)
    if in_run:
        if max_band_width is None or (len(mask) - start) <= max_band_width:
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


def find_grid(img: np.ndarray) -> GridInfo | None:
    mask = _grid_line_mask(img)

    h_lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1)))
    v_lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50)))

    expand = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    intersections = cv2.bitwise_and(
        cv2.dilate(h_lines, expand),
        cv2.dilate(v_lines, expand),
    )

    # max_band_width rejects wide blobs from grey pre-filled cells.
    # True grid-line intersections produce bands ≈50px wide; grey cells ≈80+px.
    h_positions = _merge_close(_line_positions(intersections, axis=1,
                                               max_band_width=_MAX_LINE_BAND))
    v_positions = _merge_close(_line_positions(intersections, axis=0,
                                               max_band_width=_MAX_LINE_BAND))

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


# ---------------------------------------------------------------------------
# Cell-symbol detection
# ---------------------------------------------------------------------------

def _classify_cell(roi: np.ndarray) -> str:
    """Return SUN, MOON, or EMPTY based on dominant colour in the ROI."""
    if roi.size == 0:
        return EMPTY
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    area = roi.shape[0] * roi.shape[1]
    threshold = area * _SYMBOL_FRAC

    sun_px  = int(cv2.inRange(hsv, _SUN_LO,  _SUN_HI ).sum()) // 255
    moon_px = int(cv2.inRange(hsv, _MOON_LO, _MOON_HI).sum()) // 255

    if sun_px >= threshold and sun_px >= moon_px:
        return SUN
    if moon_px >= threshold:
        return MOON
    return EMPTY


def find_cells(img: np.ndarray, grid: GridInfo) -> list[list[str]]:
    """Return a 2-D grid of SUN | MOON | EMPTY for every cell."""
    crop = img[grid.y: grid.y + grid.height, grid.x: grid.x + grid.width]
    half_w = grid.cell_w * _CROP_FRAC / 2
    half_h = grid.cell_h * _CROP_FRAC / 2
    cells = [[EMPTY] * grid.cols for _ in range(grid.rows)]
    for r in range(grid.rows):
        for c in range(grid.cols):
            cx = (c + 0.5) * grid.cell_w
            cy = (r + 0.5) * grid.cell_h
            x1 = max(0, int(cx - half_w))
            y1 = max(0, int(cy - half_h))
            x2 = min(crop.shape[1], int(cx + half_w))
            y2 = min(crop.shape[0], int(cy + half_h))
            cells[r][c] = _classify_cell(crop[y1:y2, x1:x2])
    return cells


# ---------------------------------------------------------------------------
# Constraint detection  ('=' vs '×' on cell edges)
# ---------------------------------------------------------------------------

def _augment_template(img: np.ndarray, shift: int = 3, step: int = 1) -> list[np.ndarray]:
    """Generate shifted variants of a template to handle crop misalignment."""
    h, w = img.shape
    variants = [img]
    for dx in range(-shift, shift + 1, step):
        for dy in range(-shift, shift + 1, step):
            if dx == 0 and dy == 0:
                continue
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            variants.append(cv2.warpAffine(img, M, (w, h), borderValue=255))
    return variants


def _load_constraint_templates() -> dict[str, list[np.ndarray]]:
    """Load equal.png and opposite.png from the templates directory."""
    if _constraint_templates:
        return _constraint_templates
    for label, filename in [(EQUAL, "equal.png"), (OPPOSITE, "opposite.png")]:
        path = TEMPLATE_DIR / filename
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            resized = cv2.resize(binary, CONSTRAINT_TEMPLATE_SIZE, interpolation=cv2.INTER_CUBIC)
            _constraint_templates[label] = _augment_template(resized)
    return _constraint_templates


def _preprocess_constraint(window: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY) if len(window.shape) == 3 else window
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.resize(binary, CONSTRAINT_TEMPLATE_SIZE, interpolation=cv2.INTER_CUBIC)


def _classify_constraint(window: np.ndarray) -> str | None:
    """Classify a small edge-centred crop as EQUAL, OPPOSITE, or None
    using template matching against equal.png / opposite.png.
    """
    if window.size == 0:
        return None
    templates = _load_constraint_templates()
    if not templates:
        return None

    processed = _preprocess_constraint(window)
    best_label: str | None = None
    best_score = -1.0

    for label, variants in templates.items():
        score = max(
            float(cv2.matchTemplate(processed, tmpl, cv2.TM_CCOEFF_NORMED)[0][0])
            for tmpl in variants
        )
        if score > best_score:
            best_score = score
            best_label = label

    return best_label if best_score >= CONSTRAINT_THRESHOLD else None


def find_constraints(
    img: np.ndarray, grid: GridInfo
) -> tuple[dict[tuple[int, int], str], dict[tuple[int, int], str]]:
    """Detect = / × constraints on every internal cell edge.

    Returns:
        h_constraints — {(r, c): EQUAL|OPPOSITE}  edge below row r at col c
        v_constraints — {(r, c): EQUAL|OPPOSITE}  edge right of col c at row r
    """
    crop = img[grid.y: grid.y + grid.height, grid.x: grid.x + grid.width]
    win = max(8, int(min(grid.cell_w, grid.cell_h) * 0.18))

    h_constraints: dict[tuple[int, int], str] = {}
    v_constraints: dict[tuple[int, int], str] = {}

    for r in range(grid.rows - 1):
        ey = int((r + 1) * grid.cell_h)
        for c in range(grid.cols):
            cx = int((c + 0.5) * grid.cell_w)
            x1, x2 = max(0, cx - win), min(crop.shape[1], cx + win)
            y1, y2 = max(0, ey - win), min(crop.shape[0], ey + win)
            ct = _classify_constraint(crop[y1:y2, x1:x2])
            if ct:
                h_constraints[(r, c)] = ct

    for c in range(grid.cols - 1):
        ex = int((c + 1) * grid.cell_w)
        for r in range(grid.rows):
            cy = int((r + 0.5) * grid.cell_h)
            x1, x2 = max(0, ex - win), min(crop.shape[1], ex + win)
            y1, y2 = max(0, cy - win), min(crop.shape[0], cy + win)
            ct = _classify_constraint(crop[y1:y2, x1:x2])
            if ct:
                v_constraints[(r, c)] = ct

    return h_constraints, v_constraints


# ---------------------------------------------------------------------------
# Debug visualisation
# ---------------------------------------------------------------------------

_CELL_COLOUR = {SUN: (0, 165, 255), MOON: (200, 80, 0)}
_CELL_LABEL  = {SUN: 'S', MOON: 'M'}
_CON_LABEL   = {EQUAL: '=', OPPOSITE: 'x'}


def draw_debug(img: np.ndarray, board: TangoBoard) -> np.ndarray:
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
            cell = board.cells[r][c]
            if cell in _CELL_LABEL:
                px = int(g.x + (c + 0.5) * g.cell_w)
                py = int(g.y + (r + 0.5) * g.cell_h)
                cv2.putText(out, _CELL_LABEL[cell], (px - 8, py + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, _CELL_COLOUR[cell], 2)

    for (r, c), ct in board.h_constraints.items():
        px = int(g.x + (c + 0.5) * g.cell_w)
        py = int(g.y + (r + 1) * g.cell_h)
        cv2.putText(out, _CON_LABEL[ct], (px - 5, py + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1)

    for (r, c), ct in board.v_constraints.items():
        px = int(g.x + (c + 1) * g.cell_w)
        py = int(g.y + (r + 0.5) * g.cell_h)
        cv2.putText(out, _CON_LABEL[ct], (px - 5, py + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1)

    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def detect_board(debug: bool = False) -> TangoBoard | None:
    print("Capturing screen...")
    img = capture_screen()

    print("Detecting grid...")
    grid = find_grid(img)
    if grid is None:
        print("ERROR: Could not find the Tango game grid on screen.")
        return None

    print(f"Grid found: {grid.cols}×{grid.rows} at ({grid.x}, {grid.y}), "
          f"{grid.width}×{grid.height}px, cell ~{grid.cell_w:.1f}×{grid.cell_h:.1f}px")

    print("Reading cell symbols...")
    cells = find_cells(img, grid)
    suns  = sum(1 for row in cells for v in row if v == SUN)
    moons = sum(1 for row in cells for v in row if v == MOON)
    print(f"  {suns} sun(s), {moons} moon(s)")
    for row in cells:
        print("  " + " ".join(v[0].upper() if v != EMPTY else '.' for v in row))

    print("Detecting constraints...")
    h_con, v_con = find_constraints(img, grid)
    print(f"  Horizontal constraints (below row r, at col c): {sorted(h_con.items())}")
    print(f"  Vertical constraints   (right of col c, at row r): {sorted(v_con.items())}")

    board = TangoBoard(grid, cells, h_con, v_con)

    if debug:
        cv2.imshow("Tango — board detection", draw_debug(img, board))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return board


if __name__ == "__main__":
    detect_board(debug=True)
