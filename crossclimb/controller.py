import time
import ctypes
import ctypes.wintypes
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crossclimb.board_vision import detect_board, read_clue, BoardInfo
from crossclimb.solver import solve_crossclimb, solve_endpoints
import screen

_user32 = ctypes.windll.user32
_MOUSEEVENTF_MOVE      = 0x0001
_MOUSEEVENTF_LEFTDOWN  = 0x0002
_MOUSEEVENTF_LEFTUP    = 0x0004
_MOUSEEVENTF_ABSOLUTE  = 0x8000
_KEYEVENTF_KEYUP       = 0x0002
_VK_RETURN = 0x0D


def _move(x: int, y: int) -> None:
    ox, oy = screen.game_offset()
    screen_w = _user32.GetSystemMetrics(0)
    screen_h = _user32.GetSystemMetrics(1)
    nx = int((x + ox) * 65535 / screen_w)
    ny = int((y + oy) * 65535 / screen_h)
    _user32.mouse_event(_MOUSEEVENTF_MOVE | _MOUSEEVENTF_ABSOLUTE,
                        nx, ny, 0, 0)


def _click(x: int, y: int) -> None:
    _move(x, y)
    time.sleep(0.05)
    _user32.mouse_event(_MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    time.sleep(0.03)
    _user32.mouse_event(_MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


def _type_text(text: str) -> None:
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
          duration: float = 0.5,
          overshoot: int = 30) -> None:
    """Smooth drag with overshoot to ensure the drop registers."""
    _move(from_x, from_y)
    time.sleep(0.15)
    _user32.mouse_event(_MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    time.sleep(0.15)

    dx = to_x - from_x
    dy = to_y - from_y
    dist = max(1, (dx**2 + dy**2) ** 0.5)
    os_x = int(to_x + overshoot * dx / dist)
    os_y = int(to_y + overshoot * dy / dist)

    steps = max(15, int(abs(to_y - from_y) / 3))
    for i in range(1, steps + 1):
        t = i / steps
        ix = int(from_x + (os_x - from_x) * t)
        iy = int(from_y + (os_y - from_y) * t)
        _move(ix, iy)
        time.sleep(duration / steps)

    time.sleep(0.15)
    _user32.mouse_event(_MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


def _aborted() -> bool:
    pos = ctypes.wintypes.POINT()
    _user32.GetCursorPos(ctypes.byref(pos))
    return pos.x <= 5 and pos.y <= 5


def _collect_clues(board: BoardInfo) -> dict[int, str]:
    """Click each middle row and OCR the clue shown at the bottom."""
    clues: dict[int, str] = {}
    for row in board.middle_rows:
        if _aborted():
            break

        cx, cy = row.center
        _click(cx, cy)
        time.sleep(1.2)

        img = screen.capture()
        clue = read_clue(img, board)

        if clue:
            clues[row.index] = clue
            print(f"  Row {row.index}: \"{clue}\"")
        else:
            print(f"  Row {row.index}: [OCR failed]")

    return clues


def _type_answers(board: BoardInfo, answers: dict[int, str]) -> None:
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


def _reorder_rows(board: BoardInfo, target_order: list[int]) -> None:
    """Drag middle rows into word-ladder order using selection sort."""
    middle = sorted(board.middle_rows, key=lambda r: r.y)

    if len(middle) != len(target_order):
        print(f"WARNING: {len(middle)} rows but {len(target_order)} in target")
        return

    slot_coords = [(r.left_handle[0], r.left_handle[1]) for r in middle]
    current = [r.index for r in middle]

    for target_pos in range(len(target_order)):
        target_idx = target_order[target_pos]

        src_pos = None
        for i, idx in enumerate(current):
            if idx == target_idx:
                src_pos = i
                break

        if src_pos is None:
            print(f"  WARNING: row {target_idx} not found in current list")
            continue
        if src_pos == target_pos:
            continue

        if _aborted():
            return

        sx, sy = slot_coords[src_pos]
        tx, ty = slot_coords[target_pos]

        print(f"  Drag row {target_idx}: slot {src_pos} -> slot {target_pos}")
        _drag(sx, sy, tx, ty)
        time.sleep(0.6)

        moved = current.pop(src_pos)
        current.insert(target_pos, moved)


def _solve_locked_rows(board: BoardInfo, ladder: list[tuple[int, str]]) -> None:
    """Solve the two locked endpoint rows from their shared clue."""
    locked = board.locked_rows
    if len(locked) < 2:
        print("  Could not find both locked rows.")
        return

    top_locked = min(locked, key=lambda r: r.y)
    bottom_locked = max(locked, key=lambda r: r.y)

    top_adjacent = ladder[0][1]
    bottom_adjacent = ladder[-1][1]

    if _aborted():
        return
    _click(*top_locked.center)
    time.sleep(1.2)

    img = screen.capture()
    clue = read_clue(img, board)

    if not clue:
        print("  Locked rows: [OCR failed]")
        return

    print(f"  Clue: \"{clue}\"")
    print(f"  Top adjacent: {top_adjacent}, Bottom adjacent: {bottom_adjacent}")

    top_word, bottom_word = solve_endpoints(
        clue, board.word_length, top_adjacent, bottom_adjacent
    )
    print(f"  Top answer: {top_word}, Bottom answer: {bottom_word}")

    if _aborted():
        return
    _click(*top_locked.center)
    time.sleep(0.5)
    _type_text(top_word.lower())
    time.sleep(0.5)

    if _aborted():
        return
    _click(*bottom_locked.center)
    time.sleep(0.5)
    _type_text(bottom_word.lower())
    time.sleep(0.5)


def run_crossclimb(countdown: int = 3, debug: bool = False) -> bool:
    """Run the full Crossclimb solver. Returns True on success."""

    print(f"Starting in {countdown}s — switch to the browser now ...")
    for i in range(countdown, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    img = screen.capture()
    board = detect_board(img, debug=debug)
    if board is None:
        print("FATAL: Could not detect the Crossclimb board.")
        return False

    middle = board.middle_rows
    print(f"{len(middle)} middle rows, word length = {board.word_length}")

    clues = _collect_clues(board)

    if len(clues) < len(middle):
        print(f"WARNING: Read {len(clues)}/{len(middle)} clues.")
    if not clues:
        print("FATAL: No clues could be read.")
        return False

    ladder = solve_crossclimb(clues, board.word_length)
    print("Word-ladder solution:")
    for idx, word in ladder:
        print(f"  Row {idx}: {word}")

    answers = {idx: word for idx, word in ladder}
    _type_answers(board, answers)

    target_order = [idx for idx, _ in ladder]
    current_order = [r.index for r in sorted(middle, key=lambda r: r.y)]
    print(f"  Current: {current_order}")
    print(f"  Target:  {target_order}")

    if current_order != target_order:
        time.sleep(1.0)
        img = screen.capture()
        board2 = detect_board(img)
        if board2 is not None:
            _reorder_rows(board2, target_order)
        else:
            print("  WARNING: Re-detection failed, attempting with old positions")
            _reorder_rows(board, target_order)
    else:
        print("  Already in correct order!")

    time.sleep(4.0)

    img = screen.capture()
    board3 = detect_board(img)
    if board3 is None:
        print("  WARNING: Re-detection failed, using original board")
        board3 = board

    _solve_locked_rows(board3, ladder)

    print("\nDone!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solve LinkedIn Crossclimb game.")
    parser.add_argument("--countdown", type=int, default=3,
                        help="Seconds before execution starts (default: 3)")
    parser.add_argument("--debug", action="store_true",
                        help="Show debug window with detected board")
    args = parser.parse_args()

    screen.init_game_region()
    run_crossclimb(countdown=args.countdown, debug=args.debug)
