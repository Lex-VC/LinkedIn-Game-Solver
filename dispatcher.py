"""Dispatcher — plays all LinkedIn games in sequence.

Usage:
  python dispatcher.py [--countdown N] [--pause N]

Flow for each game:
  1. Ctrl+L -> type the game URL -> Enter  (navigate browser)
  2. Wait for the page to load
  3. Detect the game region (white card)
  4. Run the game's controller directly
  5. Wait --pause seconds before the next game (default: 5)

If any game fails (exception, detection failure) it is skipped.

Safety:
  Move the mouse to the top-left corner of the screen (<=5 px) to abort.
"""

import argparse
import ctypes
import ctypes.wintypes
import sys
import time
import os
from pathlib import Path

# Ensure project root is on sys.path for package imports
_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import screen
from patches.controller import run_patches
from zip.controller import run_zip
from sudoku.controller import run_sudoku
from tango.controller import run_tango
from queens.controller import run_queens
from pinpoint.controller import run_pinpoint
from crossclimb.controller import run_crossclimb

_user32 = ctypes.windll.user32
_KEYEVENTF_KEYUP = 0x0002
_VK_CONTROL = 0x11
_VK_RETURN  = 0x0D
_MOUSEEVENTF_MOVE     = 0x0001
_MOUSEEVENTF_ABSOLUTE = 0x8000

# Game definitions: (url, display name, runner function)
GAMES = [
    ("https://www.linkedin.com/games/patches/",      "patches",    run_patches),
    ("https://www.linkedin.com/games/zip/",           "zip",        run_zip),
    ("https://www.linkedin.com/games/mini-sudoku/",   "sudoku",     run_sudoku),
    ("https://www.linkedin.com/games/tango/",         "tango",      run_tango),
    ("https://www.linkedin.com/games/queens/",        "queens",     run_queens),
    ("https://www.linkedin.com/games/pinpoint/",      "pinpoint",   run_pinpoint),
    ("https://www.linkedin.com/games/crossclimb/",    "crossclimb", run_crossclimb),
]

# ---------- keyboard helpers ----------

def _key_down(vk: int) -> None:
    _user32.keybd_event(vk, 0, 0, 0)

def _key_up(vk: int) -> None:
    _user32.keybd_event(vk, 0, _KEYEVENTF_KEYUP, 0)

def _press(vk: int, duration: float = 0.03) -> None:
    _key_down(vk)
    time.sleep(duration)
    _key_up(vk)
    time.sleep(0.02)

def _type_text(text: str) -> None:
    """Type a string character by character using VkKeyScanW + keybd_event."""
    VK_SHIFT = 0x10
    for char in text:
        vk_result = _user32.VkKeyScanW(ord(char))
        vk = vk_result & 0xFF
        shift = (vk_result >> 8) & 0x01

        if shift:
            _user32.keybd_event(VK_SHIFT, 0, 0, 0)
            time.sleep(0.01)

        _user32.keybd_event(vk, 0, 0, 0)
        time.sleep(0.01)
        _user32.keybd_event(vk, 0, _KEYEVENTF_KEYUP, 0)
        time.sleep(0.01)

        if shift:
            _user32.keybd_event(VK_SHIFT, 0, _KEYEVENTF_KEYUP, 0)
            time.sleep(0.01)

def _park_mouse() -> None:
    """Move the mouse to the bottom-right corner so it doesn't interfere with CV detection."""
    screen_w = _user32.GetSystemMetrics(0)
    screen_h = _user32.GetSystemMetrics(1)
    nx = int((screen_w - 10) * 65535 / screen_w)
    ny = int((screen_h - 10) * 65535 / screen_h)
    _user32.mouse_event(_MOUSEEVENTF_MOVE | _MOUSEEVENTF_ABSOLUTE, nx, ny, 0, 0)


def _check_abort() -> bool:
    """Return True if mouse is in the top-left corner (abort signal)."""
    pt = ctypes.wintypes.POINT()
    _user32.GetCursorPos(ctypes.byref(pt))
    return pt.x <= 5 and pt.y <= 5


def navigate_to(url: str, load_wait: float = 6.0) -> None:
    """Focus the browser address bar (Ctrl+L), type the URL, and press Enter."""
    # Ctrl+L to focus address bar
    _key_down(_VK_CONTROL)
    time.sleep(0.05)
    _press(0x4C)  # 'L'
    _key_up(_VK_CONTROL)
    time.sleep(0.3)

    # Type the URL
    _type_text(url)
    time.sleep(0.1)

    # Press Enter
    _press(_VK_RETURN)

    # Wait for page to load
    print(f"  Waiting {load_wait:.0f}s for page to load...")
    time.sleep(load_wait)


def run_game(name: str, runner, countdown: int = 3) -> bool:
    """Run a game's controller function. Returns True on success."""
    try:
        return runner(countdown=countdown)
    except Exception as e:
        print(f"  Error running {name}: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Play all LinkedIn games in sequence.")
    parser.add_argument("--countdown", type=int, default=5,
                        help="Initial countdown before starting (default: 5)")
    parser.add_argument("--pause", type=float, default=5.0,
                        help="Seconds to wait between games (default: 5)")
    parser.add_argument("--load-wait", type=float, default=6.0,
                        help="Seconds to wait for page load after navigating (default: 6)")
    parser.add_argument("--game-countdown", type=int, default=3,
                        help="Countdown passed to each game controller (default: 3)")
    args = parser.parse_args()

    print(f"=== LinkedIn Game Dispatcher ===")
    print(f"Will play {len(GAMES)} games. Move mouse to top-left corner to abort.\n")

    # Initial countdown
    for i in range(args.countdown, 0, -1):
        if _check_abort():
            print("Aborted (mouse in corner).")
            return
        print(f"Starting in {i}...")
        time.sleep(1)

    results = {}

    for idx, (url, name, runner) in enumerate(GAMES, 1):
        if _check_abort():
            print("\nAborted (mouse in corner).")
            break

        print(f"\n[{idx}/{len(GAMES)}] {name.upper()}")
        print(f"  Navigating to {url}")
        navigate_to(url, load_wait=args.load_wait)

        if _check_abort():
            print("\nAborted (mouse in corner).")
            break

        _park_mouse()

        # Detect the game region for this game
        print(f"  Detecting game region...")
        screen.reset_game_region()
        screen.init_game_region()

        print(f"  Running solver...")
        success = run_game(name, runner, countdown=args.game_countdown)

        if success:
            print(f"  {name.upper()} completed successfully!")
            results[name] = "OK"
        else:
            print(f"  {name.upper()} failed — skipping.")
            results[name] = "FAILED"

        # Wait between games (except after the last one)
        if idx < len(GAMES):
            print(f"  Waiting {args.pause:.0f}s before next game...")
            time.sleep(args.pause)

    # Summary
    print("\n=== Results ===")
    for _, name, _ in GAMES:
        status = results.get(name, "SKIPPED")
        marker = "+" if status == "OK" else "-"
        print(f"  [{marker}] {name:12s} {status}")

    ok = sum(1 for v in results.values() if v == "OK")
    print(f"\n{ok}/{len(GAMES)} games completed successfully.")


if __name__ == "__main__":
    main()
