"""Crossclimb board vision — detect rows, word length, and read clue text.

The LinkedIn Crossclimb game presents:
  - A vertical stack of rows (rounded rectangles), each holding a word.
  - Two locked rows (orange/peach) at the top and bottom.
  - Middle rows that are either selected (teal) or empty (light gray).
  - Drag handles (=) on both sides of each row.
  - A clue dropdown at the bottom showing the clue for the selected row.

Detection pipeline:
  1. Find the orange locked bars (top & bottom anchors).
  2. Scan between them with brightness projection to find all rows.
  3. Classify rows by colour (locked / selected / empty).
  4. Count dashes in the selected row to determine word length.
  5. OCR the clue region below the board.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2
import pytesseract
import screen

# HSV colour ranges
_PEACH_LO = np.array([3, 30, 180])
_PEACH_HI = np.array([25, 200, 255])

_TEAL_LO = np.array([75, 15, 160])
_TEAL_HI = np.array([115, 130, 255])

_SAT_THRESHOLD = 20  # minimum saturation to count as "coloured"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class RowInfo:
    """One row on the Crossclimb board."""

    def __init__(self, x: int, y: int, w: int, h: int,
                 row_type: str, index: int):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.row_type = row_type   # 'locked' | 'selected' | 'empty'
        self.index = index         # position from top, 0-based

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)

    @property
    def left_handle(self) -> tuple[int, int]:
        """Approximate position of the left drag handle (=)."""
        return (self.x - 20, self.y + self.h // 2)

    @property
    def right_handle(self) -> tuple[int, int]:
        return (self.x + self.w + 20, self.y + self.h // 2)

    def __repr__(self) -> str:
        return (f"Row({self.index}, type={self.row_type!r}, "
                f"y={self.y}, h={self.h})")


class BoardInfo:
    """Complete Crossclimb board state."""

    def __init__(self, rows: list[RowInfo], word_length: int,
                 clue_region: tuple[int, int, int, int] | None = None):
        self.rows = rows
        self.word_length = word_length
        self.clue_region = clue_region  # (x, y, w, h) for OCR

    @property
    def locked_rows(self) -> list[RowInfo]:
        return [r for r in self.rows if r.row_type == 'locked']

    @property
    def middle_rows(self) -> list[RowInfo]:
        return [r for r in self.rows if r.row_type != 'locked']




# ---------------------------------------------------------------------------
# Colour bar detection
# ---------------------------------------------------------------------------

def _find_bars(screen: np.ndarray, hsv_lo: np.ndarray, hsv_hi: np.ndarray,
               min_w: int = 200, min_h: int = 20
               ) -> list[tuple[int, int, int, int]]:
    """Find horizontal bars matching an HSV colour range.

    Returns list of (x, y, w, h) sorted by vertical position.
    """
    hsv = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lo, hsv_hi)

    # Close small gaps (rounded corners, anti-aliasing)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    bars = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w >= min_w and h >= min_h:
            bars.append((x, y, w, h))

    bars.sort(key=lambda b: b[1])
    return bars


# ---------------------------------------------------------------------------
# Word-length detection
# ---------------------------------------------------------------------------

def _detect_word_length(screen: np.ndarray, row: RowInfo | None) -> int:
    """Count the dashes / underscores in the selected (teal) row."""
    if row is None:
        return 4  # safe default

    # Crop the inner 70 % of the row (skip drag handles & rounded edges)
    margin_x = int(row.w * 0.15)
    margin_y = int(row.h * 0.20)
    x1 = row.x + margin_x
    x2 = row.x + row.w - margin_x
    y1 = row.y + margin_y
    y2 = row.y + row.h - margin_y
    roi = screen[y1:y2, x1:x2]

    if roi.size == 0:
        return 4

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Dashes are darker than the teal background (~180-210 gray)
    _, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)

    # Remove tiny noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Keep only dash-shaped blobs: wider than tall, minimum width
    dashes = []
    for cnt in contours:
        dx, dy, dw, dh = cv2.boundingRect(cnt)
        if dw > 8 and dw > dh * 1.5:
            dashes.append((dx, dy, dw, dh))

    return len(dashes) if dashes else 4


# ---------------------------------------------------------------------------
# Board detection (main entry point)
# ---------------------------------------------------------------------------

def detect_board(img: np.ndarray | None = None,
                 debug: bool = False) -> BoardInfo | None:
    """Detect the Crossclimb board.

    Returns a BoardInfo with all rows, word length, and clue region,
    or None if the board cannot be found.

    Coordinates are relative to the game region (see screen module).
    """
    if img is None:
        img = screen.capture()

    # --- 1. Find locked (orange / peach) bars ---
    peach_bars = _find_bars(img, _PEACH_LO, _PEACH_HI)
    if len(peach_bars) < 2:
        print(f"ERROR: Found {len(peach_bars)} locked row(s), need at least 2.")
        return None

    top_locked = peach_bars[0]
    bottom_locked = peach_bars[-1]
    print(f"Locked rows: top y={top_locked[1]}  bottom y={bottom_locked[1]}")

    # Reference dimensions
    row_x = min(top_locked[0], bottom_locked[0])
    row_w = max(top_locked[2], bottom_locked[2])

    # --- 2. Scan between locked bars with brightness projection ---
    scan_y1 = top_locked[1]
    scan_y2 = bottom_locked[1] + bottom_locked[3]
    scan = img[scan_y1:scan_y2, row_x:row_x + row_w]

    gray = cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY)
    projection = gray.mean(axis=1)

    # Rows are darker than the pure-white (255) background
    is_row = projection < 250

    # Extract contiguous bands
    bands: list[tuple[int, int]] = []
    in_band = False
    start = 0
    for i in range(len(is_row)):
        if is_row[i] and not in_band:
            in_band = True
            start = i
        elif not is_row[i] and in_band:
            in_band = False
            bands.append((start, i))
    if in_band:
        bands.append((start, len(is_row)))

    # Filter by plausible row height
    ref_h = top_locked[3]
    bands = [(s, e) for s, e in bands
             if ref_h * 0.35 <= (e - s) <= ref_h * 2.5]

    if len(bands) < 3:
        print(f"ERROR: Only found {len(bands)} row band(s), need at least 3.")
        return None

    # --- 3. Classify each band ---
    teal_bars = _find_bars(img, _TEAL_LO, _TEAL_HI)

    rows: list[RowInfo] = []
    for i, (bs, be) in enumerate(bands):
        y = scan_y1 + bs
        h = be - bs

        # Check overlap with locked bars
        is_top = abs(y - top_locked[1]) < ref_h * 0.6
        is_bot = abs(y - bottom_locked[1]) < ref_h * 0.6
        is_teal = any(abs(y - tb[1]) < ref_h * 0.6 for tb in teal_bars)

        if is_top or is_bot:
            rtype = 'locked'
        elif is_teal:
            rtype = 'selected'
        else:
            rtype = 'empty'

        rows.append(RowInfo(row_x, y, row_w, h, rtype, i))

    # --- 4. Word length ---
    selected = next((r for r in rows if r.row_type == 'selected'), None)
    word_length = _detect_word_length(img, selected)

    # --- 5. Clue region (below the bottom locked bar) ---
    # Skip the "Reveal row | Hint" buttons (~80 px) and only capture
    # the narrow clue-text dropdown.  Inset horizontally to avoid the
    # side arrows / decorations.
    clue_y1 = scan_y2 + 80
    clue_y2 = min(img.shape[0], clue_y1 + 80)
    clue_region = (row_x, clue_y1,
                   row_w, clue_y2 - clue_y1)

    board = BoardInfo(rows, word_length, clue_region)

    print(f"Detected {len(rows)} rows  (word length = {word_length})")
    for r in rows:
        print(f"  {r}")

    if debug:
        _show_debug(img, board)

    return board


# ---------------------------------------------------------------------------
# Clue reading (OCR)
# ---------------------------------------------------------------------------

def read_clue(img: np.ndarray, board: BoardInfo) -> str | None:
    """OCR the clue text shown at the bottom of the game.

    Returns the clue string, or None if nothing could be read.
    """
    if board.clue_region is None:
        return None

    cx, cy, cw, ch = board.clue_region
    cy = max(0, cy)
    ch = min(ch, img.shape[0] - cy)
    cx = max(0, cx)
    cw = min(cw, img.shape[1] - cx)
    roi = img[cy:cy + ch, cx:cx + cw]

    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Upscale for better OCR accuracy
    scale = 2
    gray = cv2.resize(gray, None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_CUBIC)

    text = pytesseract.image_to_string(gray, config="--psm 6").strip()

    # Filter out known UI element text
    skip_lower = {'reveal row', 'hint', 'v', '>', '<', ''}
    lines = []
    for line in text.split('\n'):
        cleaned = line.strip()
        if cleaned.lower() in skip_lower:
            continue
        # Skip very short fragments (likely OCR noise)
        if len(cleaned) < 3:
            continue
        lines.append(cleaned)

    return ' '.join(lines) if lines else None


# ---------------------------------------------------------------------------
# Debug visualisation
# ---------------------------------------------------------------------------

def _show_debug(img: np.ndarray, board: BoardInfo) -> None:
    dbg = img.copy()
    colours = {
        'locked':   (0, 0, 255),
        'selected': (255, 128, 0),
        'empty':    (0, 255, 0),
    }

    for row in board.rows:
        c = colours.get(row.row_type, (128, 128, 128))
        cv2.rectangle(dbg, (row.x, row.y),
                      (row.x + row.w, row.y + row.h), c, 2)
        label = f"{row.index} {row.row_type}"
        cv2.putText(dbg, label, (row.x + 5, row.y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)

        # Mark drag handles
        lx, ly = row.left_handle
        cv2.drawMarker(dbg, (lx, ly), c, cv2.MARKER_CROSS, 12, 1)

    if board.clue_region:
        cx, cy, cw, ch = board.clue_region
        cv2.rectangle(dbg, (cx, cy), (cx + cw, cy + ch), (255, 255, 0), 2)
        cv2.putText(dbg, "clue region", (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    cv2.putText(dbg, f"word_length={board.word_length}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)
    sh, sw = img.shape[:2]
    cv2.putText(dbg, f"{sw}x{sh}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 1)

    cv2.imshow("Crossclimb — board detection",
               cv2.resize(dbg, None, fx=0.5, fy=0.5))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    screen.init_game_region()
    detect_board(debug=True)
