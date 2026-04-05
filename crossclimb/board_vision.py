from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2
import pytesseract
import screen

_PEACH_LO = np.array([3, 30, 180])
_PEACH_HI = np.array([25, 200, 255])

_TEAL_LO = np.array([75, 15, 160])
_TEAL_HI = np.array([115, 130, 255])


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
        self.clue_region = clue_region

    @property
    def locked_rows(self) -> list[RowInfo]:
        return [r for r in self.rows if r.row_type == 'locked']

    @property
    def middle_rows(self) -> list[RowInfo]:
        return [r for r in self.rows if r.row_type != 'locked']


def _find_bars(img: np.ndarray, hsv_lo: np.ndarray, hsv_hi: np.ndarray,
               min_w: int = 200, min_h: int = 20
               ) -> list[tuple[int, int, int, int]]:
    """Find horizontal bars matching an HSV colour range."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lo, hsv_hi)

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


def _detect_word_length(img: np.ndarray, row: RowInfo | None) -> int:
    """Count the dashes in the selected (teal) row."""
    if row is None:
        return 4

    margin_x = int(row.w * 0.05)
    margin_y = int(row.h * 0.20)
    x1 = row.x + margin_x
    x2 = row.x + row.w - margin_x
    y1 = row.y + margin_y
    y2 = row.y + row.h - margin_y
    roi = img[y1:y2, x1:x2]

    if roi.size == 0:
        return 4

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    dashes = []
    for cnt in contours:
        dx, dy, dw, dh = cv2.boundingRect(cnt)
        if dw > 8 and dw > dh * 1.5:
            dashes.append((dx, dy, dw, dh))

    return len(dashes) if dashes else 4


def detect_board(img: np.ndarray | None = None,
                 debug: bool = False) -> BoardInfo | None:
    """Detect the Crossclimb board. Returns BoardInfo or None."""
    if img is None:
        img = screen.capture()

    peach_bars = _find_bars(img, _PEACH_LO, _PEACH_HI)
    if len(peach_bars) < 2:
        print(f"ERROR: Found {len(peach_bars)} locked row(s), need at least 2.")
        return None

    top_locked = peach_bars[0]
    bottom_locked = peach_bars[-1]
    print(f"Locked rows: top y={top_locked[1]}  bottom y={bottom_locked[1]}")

    row_x = min(top_locked[0], bottom_locked[0])
    row_w = max(top_locked[2], bottom_locked[2])

    scan_y1 = top_locked[1]
    scan_y2 = bottom_locked[1] + bottom_locked[3]
    scan = img[scan_y1:scan_y2, row_x:row_x + row_w]

    gray = cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY)
    projection = gray.mean(axis=1)

    is_row = projection < 250

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

    ref_h = top_locked[3]
    bands = [(s, e) for s, e in bands
             if ref_h * 0.35 <= (e - s) <= ref_h * 2.5]

    if len(bands) < 3:
        print(f"ERROR: Only found {len(bands)} row band(s), need at least 3.")
        return None

    teal_bars = _find_bars(img, _TEAL_LO, _TEAL_HI)

    rows: list[RowInfo] = []
    for i, (bs, be) in enumerate(bands):
        y = scan_y1 + bs
        h = be - bs

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

    selected = next((r for r in rows if r.row_type == 'selected'), None)
    word_length = _detect_word_length(img, selected)

    # Skip the "Reveal row | Hint" buttons (~80 px) above the clue-text dropdown
    clue_y1 = scan_y2 + 80
    clue_y2 = min(img.shape[0], clue_y1 + 150)
    clue_region = (row_x, clue_y1, row_w, clue_y2 - clue_y1)

    board = BoardInfo(rows, word_length, clue_region)

    print(f"Detected {len(rows)} rows  (word length = {word_length})")
    for r in rows:
        print(f"  {r}")

    if debug:
        _show_debug(img, board)

    return board


def read_clue(img: np.ndarray, board: BoardInfo) -> str | None:
    """OCR the clue text shown at the bottom of the game."""
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
    scale = 2
    gray = cv2.resize(gray, None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_CUBIC)

    text = pytesseract.image_to_string(gray, config="--psm 6").strip()

    skip_lower = {'reveal row', 'hint', 'v', '>', '<', ''}
    lines = []
    for line in text.split('\n'):
        cleaned = line.strip()
        if cleaned.lower() in skip_lower:
            continue
        if len(cleaned) < 3:
            continue
        lines.append(cleaned)

    return ' '.join(lines) if lines else None


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


if __name__ == "__main__":
    screen.init_game_region()
    detect_board(debug=True)
