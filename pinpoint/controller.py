"""Pinpoint controller — detects clues, guesses category via LLM, and types the answer."""

import time
import ctypes
import ctypes.wintypes
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pinpoint.board_vision import extract_clue_words, find_input_box
from pinpoint.solver import guess_category
import screen

_user32 = ctypes.windll.user32
_MOUSEEVENTF_MOVE     = 0x0001
_MOUSEEVENTF_LEFTDOWN = 0x0002
_MOUSEEVENTF_LEFTUP   = 0x0004
_MOUSEEVENTF_ABSOLUTE = 0x8000
_KEYEVENTF_KEYUP      = 0x0002
_VK_RETURN = 0x0D

_MAX_ROUNDS = 5


def _move(x: int, y: int) -> None:
    ox, oy = screen.game_offset()
    screen_w = _user32.GetSystemMetrics(0)
    screen_h = _user32.GetSystemMetrics(1)
    nx = int((x + ox) * 65535 / screen_w)
    ny = int((y + oy) * 65535 / screen_h)
    _user32.mouse_event(_MOUSEEVENTF_MOVE | _MOUSEEVENTF_ABSOLUTE, nx, ny, 0, 0)


def _click(x: int, y: int) -> None:
    _move(x, y)
    time.sleep(0.05)
    _user32.mouse_event(_MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    time.sleep(0.03)
    _user32.mouse_event(_MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


def _type_text(text: str) -> None:
    """Type a string character by character using SendInput."""
    for char in text:
        # Use VkKeyScanW to get the virtual key for each character
        vk_result = _user32.VkKeyScanW(ord(char))
        vk = vk_result & 0xFF
        shift = (vk_result >> 8) & 0x01

        if shift:
            _user32.keybd_event(0x10, 0, 0, 0)  # VK_SHIFT down
            time.sleep(0.01)

        _user32.keybd_event(vk, 0, 0, 0)
        time.sleep(0.01)
        _user32.keybd_event(vk, 0, _KEYEVENTF_KEYUP, 0)
        time.sleep(0.01)

        if shift:
            _user32.keybd_event(0x10, 0, _KEYEVENTF_KEYUP, 0)  # VK_SHIFT up
            time.sleep(0.01)


def _press_enter() -> None:
    _user32.keybd_event(_VK_RETURN, 0, 0, 0)
    time.sleep(0.02)
    _user32.keybd_event(_VK_RETURN, 0, _KEYEVENTF_KEYUP, 0)


def _aborted() -> bool:
    pos = ctypes.wintypes.POINT()
    _user32.GetCursorPos(ctypes.byref(pos))
    return pos.x <= 5 and pos.y <= 5


def run_pinpoint(countdown: int = 3, debug: bool = False) -> bool:
    """Main loop: detect clues, guess, type answer, repeat until correct or out of clues."""

    print(f"Starting in {countdown}s — switch to the browser now ...")
    for i in range(countdown, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    previous_guesses: list[str] = []
    prev_clue_count = 0

    for round_num in range(1, _MAX_ROUNDS + 1):
        if _aborted():
            print("Aborted: mouse moved to top-left corner.")
            return False

        print(f"\n--- Round {round_num} ---")

        # Capture and extract clues
        img = screen.capture()
        clues = extract_clue_words(img)

        if not clues:
            print("No clue words detected. Waiting for first clue reveal...")
            time.sleep(2)
            continue

        # Check if new clue appeared (meaning previous guess was wrong)
        if len(clues) == prev_clue_count and round_num > 1:
            print("No new clue appeared — might have won or game ended.")
            return True

        prev_clue_count = len(clues)
        print(f"Clues so far: {clues}")

        # Get AI guess
        guess = guess_category(clues, previous_guesses)
        print(f"AI guesses: {guess}")

        # Find and click the input box
        input_pos = find_input_box(img)
        if input_pos is None:
            print("Could not find input box.")
            return False

        _click(input_pos[0], input_pos[1])
        time.sleep(0.3)

        # Type the guess
        _type_text(guess)
        time.sleep(0.3)

        # Press Enter to submit
        _press_enter()
        time.sleep(2)  # Wait for game to process and potentially reveal next clue

        previous_guesses.append(guess)

        # Check if we need to continue — the next iteration will detect if a new clue appeared
        if round_num == _MAX_ROUNDS:
            print("Used all 5 rounds.")

    print("\nFinished all rounds.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve LinkedIn Pinpoint game.")
    parser.add_argument("--countdown", type=int, default=3,
                        help="Seconds before execution starts (default: 3)")
    parser.add_argument("--debug", action="store_true",
                        help="Show debug windows")
    args = parser.parse_args()

    screen.init_game_region()
    run_pinpoint(countdown=args.countdown, debug=args.debug)
