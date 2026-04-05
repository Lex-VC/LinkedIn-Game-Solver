"""Pinpoint board vision — captures screen and extracts revealed clue words via OCR."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import re

import numpy as np
import cv2
import pytesseract
import screen

_BLUE_LO = np.array([100, 30, 120])
_BLUE_HI = np.array([125, 255, 255])

_MIN_BLOCK_AREA = 30000
_MIN_BLOCK_WIDTH = 150
_MIN_BLOCK_HEIGHT = 100


def _find_gradient_block(img: np.ndarray) -> tuple[int, int, int, int] | None:
    """Find the Pinpoint gradient block (the whole blue region).

    Returns (x, y, w, h) or None.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _BLUE_LO, _BLUE_HI)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 20))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area > _MIN_BLOCK_AREA and w > _MIN_BLOCK_WIDTH and h > _MIN_BLOCK_HEIGHT:
            mid = y + h // 2
            top_v = float(np.mean(hsv[y:mid, x:x + w, 2]))
            bot_v = float(np.mean(hsv[mid:y + h, x:x + w, 2]))
            if top_v > bot_v:
                candidates.append((x, y, w, h, area))

    if not candidates:
        return None

    best = max(candidates, key=lambda c: c[4])
    return best[:4]


def extract_clue_words(img: np.ndarray) -> list[str]:
    """Extract visible clue words from the Pinpoint gradient block via OCR."""
    block = _find_gradient_block(img)
    if not block:
        return []

    x, y, w, h = block
    roi = img[y:y + h, x:x + w]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)

    if w < 400:
        scale = 2
        thresh = cv2.resize(thresh, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    text = pytesseract.image_to_string(thresh, config="--psm 6").strip()

    clues = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if re.match(r'\b\w*clue\w*\b', line, flags=re.IGNORECASE):
            continue
        clues.append(line)

    return clues


def find_input_box(img: np.ndarray) -> tuple[int, int] | None:
    """Find the 'Guess the category...' input box below the gradient block."""
    block = _find_gradient_block(img)
    if not block:
        return None

    bx, by, bw, bh = block

    search_x1 = max(0, bx - 50)
    search_x2 = min(img.shape[1], bx + bw + 50)
    search_y1 = by + bh + 20
    search_y2 = min(img.shape[0], by + bh + 350)

    roi = img[search_y1:search_y2, search_x1:search_x2]
    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    text_data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

    n = len(text_data["text"])
    lines: dict[tuple[int, int], list[int]] = {}
    for i in range(n):
        key = (text_data["block_num"][i], text_data["line_num"][i])
        lines.setdefault(key, []).append(i)

    best_match = None
    best_y = -1
    for key, indices in lines.items():
        line_text = " ".join(text_data["text"][i] for i in indices).strip()
        if re.search(r"(?i)guess.*category", line_text):
            tops = [text_data["top"][i] for i in indices]
            heights = [text_data["height"][i] for i in indices]
            ly = min(tops)
            lh = max(t + h for t, h in zip(tops, heights)) - ly
            if ly > best_y:
                best_y = ly
                best_match = (ly, lh)

    if best_match:
        ly, lh = best_match
        cx = bx + bw // 2
        cy = search_y1 + ly + lh // 2
        return (cx, cy)

    return None


def check_result(img: np.ndarray) -> str:
    """Check game state after submitting a guess. Returns 'correct' or 'unknown'."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 50, 100])
    upper_green = np.array([80, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_pixels = cv2.countNonZero(green_mask)

    if green_pixels > 1000:
        return "correct"
    return "unknown"


def draw_debug(img: np.ndarray, block: tuple[int, int, int, int] | None,
               clues: list[str], input_pos: tuple[int, int] | None) -> np.ndarray:
    out = img.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if block:
        x, y, w, h = block
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)

        mid = y + h // 2
        top_v = float(np.mean(hsv[y:mid, x:x + w, 2]))
        bot_v = float(np.mean(hsv[mid:y + h, x:x + w, 2]))
        cv2.putText(out, f"top V={top_v:.0f}", (x + 5, y - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(out, f"bot V={bot_v:.0f}", (x + 5, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        for i, clue in enumerate(clues):
            cv2.putText(out, f"clue {i + 1}: {clue}", (x + w + 10, y + 25 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        search_y1 = y + h + 20
        search_y2 = min(img.shape[0], y + h + 350)
        cv2.rectangle(out, (max(0, x - 50), search_y1),
                      (min(img.shape[1], x + w + 50), search_y2),
                      (255, 255, 0), 1)
        cv2.putText(out, "input search region", (x - 50, search_y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    if input_pos:
        cx, cy = input_pos
        cv2.drawMarker(out, (cx, cy), (255, 0, 255), cv2.MARKER_CROSS, 30, 2)
        cv2.putText(out, "input box", (cx + 20, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    sh, sw = img.shape[:2]
    cv2.putText(out, f"Screenshot: {sw}x{sh}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)

    return out


def detect_board(debug: bool = False) -> list[str] | None:
    """Capture screen, detect gradient block, OCR clues.

    Returns the list of revealed clue words, or None on failure.
    """
    print("Capturing screen...")
    img = screen.capture()
    print(f"Screenshot size: {img.shape[1]}x{img.shape[0]}")

    print("Detecting gradient block...")
    block = _find_gradient_block(img)
    if not block:
        print("ERROR: Could not find Pinpoint gradient block on screen.")
        if debug:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, _BLUE_LO, _BLUE_HI)
            cv2.imshow("Blue mask (what was detected)", cv2.resize(mask, None, fx=0.5, fy=0.5))
            cv2.imshow("Screenshot", cv2.resize(img, None, fx=0.5, fy=0.5))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return None

    x, y, w, h = block
    print(f"Gradient block found: {w}x{h} at ({x}, {y})")

    print("Extracting clue text (OCR)...")
    clues = extract_clue_words(img)
    for i, clue in enumerate(clues):
        print(f"  clue {i + 1}: {clue!r}")
    if not clues:
        print("  (no revealed clues yet)")

    print("Locating input box...")
    input_pos = find_input_box(img)
    if input_pos:
        print(f"Input box at: {input_pos}")
    else:
        print("WARNING: Could not locate input box.")

    if debug:
        dbg = draw_debug(img, block, clues, input_pos)
        cv2.imshow("Pinpoint — board detection", cv2.resize(dbg, None, fx=0.5, fy=0.5))

        bx, by, bw, bh = block
        roi = img[by:by + bh, bx:bx + bw]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        cv2.imshow("OCR threshold", thresh)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return clues if clues else None


if __name__ == "__main__":
    screen.init_game_region()
    detect_board(debug=True)
