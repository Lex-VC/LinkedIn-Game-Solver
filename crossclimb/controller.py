"""Crossclimb controller — automates the full Crossclimb puzzle.

Game flow:
  Phase 1 — Detect the board (rows, word length, clue region).
  Phase 2 — Click each middle row to collect its clue via OCR.
  Phase 3 — Send all clues to the AI; get answers + word-ladder order.
  Phase 4 — Click each row and type the answer.
  Phase 5 — Drag rows into the correct word-ladder order.
  Phase 6 — Solve the two locked endpoint rows.

Safety:
  Move the mouse to the top-left corner (<=5 px) to abort at any time.
"""

import time
import ctypes
import ctypes.wintypes
import sys
import argparse

sys.path.insert(0, ".")
from board_vision import capture_screen, detect_board, read_clue, BoardInfo
from solver import solve_crossclimb, solve_endpoint

_user32 = ctypes.windll.user32
_MOUSEEVENTF_MOVE      = 0x0001
_MOUSEEVENTF_LEFTDOWN  = 0x0002
_MOUSEEVENTF_LEFTUP    = 0x0004
_MOUSEEVENTF_ABSOLUTE  = 0x8000
_KEYEVENTF_KEYUP       = 0x0002
_VK_RETURN = 0x0D
_VK_BACK   = 0x08

_SCREEN_W = _user32.GetSystemMetrics(0)
_SCREEN_H = _user32.GetSystemMetrics(1)


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def _move(x: int, y: int) -> None:
    nx = int(x * 65535 / _SCREEN_W)
    ny = int(y * 65535 / _SCREEN_H)
    _user32.mouse_event(_MOUSEEVENTF_MOVE | _MOUSEEVENTF_ABSOLUTE,
                        nx, ny, 0, 0)


def _click(x: int, y: int) -> None:
    _move(x, y)
    time.sleep(0.05)
    _user32.mouse_event(_MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    time.sleep(0.03)
    _user32.mouse_event(_MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


def _type_text(text: str) -> None:
    """Type a string character by character."""
    for char in text:
        vk_result = _user32.VkKeyScanW(ord(char))
        vk = vk_result & 0xFF
        shift = (vk_result >> 8) & 0x01
        if shift:
            _user32.keybd_event(0x10, 0, 0, 0)
            time.sleep(0.01)
        _user32.keybd_event(vk, 0, 0, 0)
        time.sleep(0.01)
        _user32.keybd_event(vk, 0, _KEYEVENTF_KEYUP, 0)
        time.sleep(0.01)
        if shift:
            _user32.keybd_event(0x10, 0, _KEYEVENTF_KEYUP, 0)
            time.sleep(0.01)


def _press_enter() -> None:
    _user32.keybd_event(_VK_RETURN, 0, 0, 0)
    time.sleep(0.02)
    _user32.keybd_event(_VK_RETURN, 0, _KEYEVENTF_KEYUP, 0)


def _drag(from_x: int, from_y: int,
          to_x: int, to_y: int,
          duration: float = 0.5) -> None:
    """Smooth drag from (from_x, from_y) to (to_x, to_y)."""
    _move(from_x, from_y)
    time.sleep(0.15)
    _user32.mouse_event(_MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    time.sleep(0.15)

    steps = max(15, int(abs(to_y - from_y) / 3))
    for i in range(1, steps + 1):
        t = i / steps
        ix = int(from_x + (to_x - from_x) * t)
        iy = int(from_y + (to_y - from_y) * t)
        _move(ix, iy)
        time.sleep(duration / steps)

    time.sleep(0.15)
    _user32.mouse_event(_MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


def _aborted() -> bool:
    """Return True when the mouse is in the top-left escape corner."""
    pos = ctypes.wintypes.POINT()
    _user32.GetCursorPos(ctypes.byref(pos))
    return pos.x <= 5 and pos.y <= 5


# ---------------------------------------------------------------------------
# Game phases
# ---------------------------------------------------------------------------

def _collect_clues(board: BoardInfo) -> dict[int, str]:
    """Click each middle row and OCR the clue shown at the bottom.

    Returns {row_index: clue_text}.
    """
    clues: dict[int, str] = {}
    middle = board.middle_rows

    for row in middle:
        if _aborted():
            break

        cx, cy = row.center
        _click(cx, cy)
        time.sleep(1.2)   # wait for clue dropdown to update

        screen = capture_screen()
        clue = read_clue(screen, board)

        if clue:
            clues[row.index] = clue
            print(f"  Row {row.index}: \"{clue}\"")
        else:
            print(f"  Row {row.index}: [OCR failed]")

    return clues


def _type_answers(board: BoardInfo,
                  answers: dict[int, str]) -> None:
    """Click each middle row and type its answer word."""
    middle = sorted(board.middle_rows, key=lambda r: r.y)

    for row in middle:
        if _aborted():
            return
        if row.index not in answers:
            continue

        word = answers[row.index]
        cx, cy = row.center
        _click(cx, cy)
        time.sleep(0.6)

        _type_text(word.lower())
        time.sleep(0.5)

        print(f"  Typed '{word}' in row {row.index}")


def _reorder_rows(board: BoardInfo,
                  target_order: list[int]) -> None:
    """Drag middle rows into the word-ladder order.

    *target_order* is a list of row indices from top to bottom.
    Uses selection sort: for each target position, find the row that
    belongs there and drag it into place.
    """
    middle = sorted(board.middle_rows, key=lambda r: r.y)

    if len(middle) != len(target_order):
        print(f"WARNING: {len(middle)} rows but {len(target_order)} in target")
        return

    # Build a mutable list tracking current positions
    current = list(middle)

    for target_pos in range(len(target_order)):
        target_idx = target_order[target_pos]

        # Find where this row currently is
        src_pos = None
        for i, row in enumerate(current):
            if row.index == target_idx:
                src_pos = i
                break

        if src_pos is None:
            print(f"  WARNING: row {target_idx} not found in current list")
            continue
        if src_pos == target_pos:
            continue  # already in place

        if _aborted():
            return

        src_row = current[src_pos]
        dst_row = current[target_pos]

        # Drag from the source row's left handle to the destination row's
        # left handle position.  Use the handle (≡) so we grab the drag
        # affordance, not a text-input cell.
        sx, sy = src_row.left_handle
        tx, ty = dst_row.left_handle

        print(f"  Drag row {target_idx}: pos {src_pos} -> pos {target_pos}")
        _drag(sx, sy, tx, ty)
        time.sleep(0.6)

        # Update tracking list
        moved = current.pop(src_pos)
        current.insert(target_pos, moved)


def _solve_locked_rows(board: BoardInfo,
                       ladder: list[tuple[int, str]]) -> None:
    """After middle rows are correctly ordered, solve the two locked rows.

    Each locked row's answer must differ from its adjacent ladder word
    by exactly one letter.
    """
    locked = board.locked_rows
    if len(locked) < 2:
        print("  Could not find both locked rows.")
        return

    top_locked = min(locked, key=lambda r: r.y)
    bottom_locked = max(locked, key=lambda r: r.y)

    first_word = ladder[0][1]   # word adjacent to top locked
    last_word = ladder[-1][1]   # word adjacent to bottom locked

    for label, row, adjacent in [
        ("Top", top_locked, first_word),
        ("Bottom", bottom_locked, last_word),
    ]:
        if _aborted():
            return

        _click(*row.center)
        time.sleep(1.2)

        screen = capture_screen()
        clue = read_clue(screen, board)

        if not clue:
            print(f"  {label} locked row: [OCR failed]")
            continue

        print(f"  {label} clue: \"{clue}\"  (adjacent: {adjacent})")
        answer = solve_endpoint(clue, board.word_length, adjacent)
        print(f"  {label} answer: {answer}")

        _click(*row.center)
        time.sleep(0.5)
        _type_text(answer.lower())
        time.sleep(0.5)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_crossclimb(countdown: int = 3, debug: bool = False) -> None:
    """Run the full Crossclimb solver."""

    print(f"Starting in {countdown}s — switch to the browser now ...")
    for i in range(countdown, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    # ---- Phase 1: Detect board ----
    print("\n=== Phase 1: Board Detection ===")
    screen = capture_screen()
    board = detect_board(screen, debug=debug)
    if board is None:
        print("FATAL: Could not detect the Crossclimb board.")
        return

    middle = board.middle_rows
    print(f"{len(middle)} middle rows, word length = {board.word_length}")

    # ---- Phase 2: Collect clues ----
    print("\n=== Phase 2: Collecting Clues ===")
    clues = _collect_clues(board)

    if len(clues) < len(middle):
        print(f"WARNING: Read {len(clues)}/{len(middle)} clues.")
    if not clues:
        print("FATAL: No clues could be read.")
        return

    # ---- Phase 3: AI solving ----
    print("\n=== Phase 3: AI Solving ===")
    ladder = solve_crossclimb(clues, board.word_length)
    print("Word-ladder solution:")
    for idx, word in ladder:
        print(f"  Row {idx}: {word}")

    # ---- Phase 4: Type answers ----
    print("\n=== Phase 4: Typing Answers ===")
    answers = {idx: word for idx, word in ladder}
    _type_answers(board, answers)

    # ---- Phase 5: Reorder rows ----
    print("\n=== Phase 5: Reordering Rows ===")
    target_order = [idx for idx, _ in ladder]
    current_order = [r.index for r in sorted(middle, key=lambda r: r.y)]
    print(f"  Current: {current_order}")
    print(f"  Target:  {target_order}")

    if current_order != target_order:
        # Re-detect board to get updated positions after typing
        time.sleep(1.0)
        screen = capture_screen()
        board2 = detect_board(screen)
        if board2 is not None:
            _reorder_rows(board2, target_order)
        else:
            print("  WARNING: Re-detection failed, attempting with old positions")
            _reorder_rows(board, target_order)
    else:
        print("  Already in correct order!")

    # ---- Phase 6: Solve locked rows ----
    print("\n=== Phase 6: Solving Locked Rows ===")
    time.sleep(2.0)  # wait for unlock animation

    screen = capture_screen()
    board3 = detect_board(screen)
    if board3 is None:
        print("  WARNING: Re-detection failed, using original board")
        board3 = board

    _solve_locked_rows(board3, ladder)

    print("\n=== Done! ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solve LinkedIn Crossclimb game.")
    parser.add_argument("--countdown", type=int, default=3,
                        help="Seconds before execution starts (default: 3)")
    parser.add_argument("--debug", action="store_true",
                        help="Show debug window with detected board")
    args = parser.parse_args()

    run_crossclimb(countdown=args.countdown, debug=args.debug)
