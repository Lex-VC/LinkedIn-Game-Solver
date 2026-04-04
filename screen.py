"""Shared screen-capture and game-region utilities.

Every LinkedIn game lives inside a white "card" on the page.  This module
detects that card once (via init_game_region) and then restricts all
subsequent captures to that region, giving each game's board_vision a
clean, tightly-cropped image to work with.

Usage — dispatcher / standalone controller:
    import screen
    screen.init_game_region()   # one-time, after the page has loaded
    img = screen.capture()      # returns only the game-region crop

Coordinates returned by board_vision are *region-relative*.  Controllers
must add screen.game_offset() before sending mouse events.
"""
from __future__ import annotations

import ctypes
import numpy as np
import cv2
import mss

# ---------------------------------------------------------------------------
# DPI awareness (Windows) — must run before any mss capture
# ---------------------------------------------------------------------------
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor DPI aware
except Exception:
    pass

# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------
_game_region: tuple[int, int, int, int] | None = None   # (x, y, w, h)

_SAT_THRESHOLD = 20  # minimum saturation to count as "coloured"


# ---------------------------------------------------------------------------
# Game-region detection
# ---------------------------------------------------------------------------

def find_game_region(img: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return (x, y, w, h) of the white game card on screen.

    Strategy:
      1. Threshold to find near-white pixels (the card background).
      2. Morphological close to fill small holes.
      3. Among large contours, pick the one with the highest saturated-pixel
         score (coloured game content inside the card).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, white = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    white_closed = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(white_closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    img_area = img.shape[0] * img.shape[1]
    min_area = img_area * 0.01

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]

    best_region = None
    best_score = -1.0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < min_area:
            continue
        ratio = w / h if h > 0 else 0
        if not (0.3 <= ratio <= 3.0):
            continue

        roi_sat = sat[y:y + h, x:x + w]
        sat_count = int((roi_sat > _SAT_THRESHOLD).sum())
        sat_frac = sat_count / area
        if sat_frac < 0.002:
            continue

        score = sat_frac * np.sqrt(area)
        if score > best_score:
            best_score = score
            best_region = (x, y, w, h)

    return best_region


# ---------------------------------------------------------------------------
# Capture helpers
# ---------------------------------------------------------------------------

def _capture_full() -> np.ndarray:
    """Grab the entire primary monitor as a BGR numpy array."""
    with mss.mss() as sct:
        shot = sct.grab(sct.monitors[1])
        return cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)


def capture() -> np.ndarray:
    """Capture the game region (or full screen if no region is set)."""
    with mss.mss() as sct:
        if _game_region:
            x, y, w, h = _game_region
            monitor = {"left": x, "top": y, "width": w, "height": h}
        else:
            monitor = sct.monitors[1]
        shot = sct.grab(monitor)
        return cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)


# ---------------------------------------------------------------------------
# Region management
# ---------------------------------------------------------------------------

def init_game_region() -> bool:
    """Capture full screen, detect the game card, and store its bounds.

    Call once after the game page has loaded.  Returns True on success.
    """
    global _game_region
    img = _capture_full()
    region = find_game_region(img)
    if region is not None:
        _game_region = region
        x, y, w, h = region
        print(f"Game region: ({x},{y}) {w}x{h}px")
        return True
    print("WARNING: Could not find game region — using full screen")
    _game_region = None
    return False


def reset_game_region() -> None:
    """Clear the stored game region (e.g. before navigating to a new game)."""
    global _game_region
    _game_region = None


def game_offset() -> tuple[int, int]:
    """Return (ox, oy) to convert region-relative coords to screen coords."""
    if _game_region:
        return (_game_region[0], _game_region[1])
    return (0, 0)
