"""Queens controller — detects, solves, and executes the Queens puzzle.

Click mechanics: 2 clicks per queen cell (1st = X mark, 2nd = queen).
"""
import time
import ctypes
import ctypes.wintypes
import sys
import argparse

sys.path.insert(0, ".")
from board_vision import GridInfo, detect_board
from solver import solve, print_board

_user32 = ctypes.windll.user32
_MOUSEEVENTF_MOVE     = 0x0001
_MOUSEEVENTF_LEFTDOWN = 0x0002
_MOUSEEVENTF_LEFTUP   = 0x0004
_MOUSEEVENTF_ABSOLUTE = 0x8000

_QUEEN_CLICKS = 2  # 1st click = X mark, 2nd click = queen


def _move(x: int, y: int) -> None:
    screen_w = _user32.GetSystemMetrics(0)
    screen_h = _user32.GetSystemMetrics(1)
    nx = int(x * 65535 / screen_w)
    ny = int(y * 65535 / screen_h)
    _user32.mouse_event(_MOUSEEVENTF_MOVE | _MOUSEEVENTF_ABSOLUTE, nx, ny, 0, 0)


def _click(x: int, y: int) -> None:
    _move(x, y)
    _user32.mouse_event(_MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    _user32.mouse_event(_MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


def _aborted() -> bool:
    pos = ctypes.wintypes.POINT()
    _user32.GetCursorPos(ctypes.byref(pos))
    return pos.x <= 5 and pos.y <= 5


def _cell_center(grid: GridInfo, r: int, c: int) -> tuple[int, int]:
    x = int(grid.x + (c + 0.5) * grid.cell_w)
    y = int(grid.y + (r + 0.5) * grid.cell_h)
    return x, y


def execute_solution(
    grid: GridInfo,
    queens: list[tuple[int, int]],
    cell_delay: float = 0.10,
    countdown: int = 3,
) -> None:
    if not queens:
        print("No queens to place.")
        return

    print(f"Starting in {countdown}s — switch to the browser now ...")
    for i in range(countdown, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    for r, c in queens:
        if _aborted():
            print("Aborted: mouse moved to top-left corner.")
            return

        x, y = _cell_center(grid, r, c)
        for _ in range(_QUEEN_CLICKS):
            _click(x, y)
            time.sleep(0.05)

        if cell_delay > 0:
            time.sleep(cell_delay)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve and execute a LinkedIn Queens puzzle.")
    parser.add_argument("--delay",     type=float, default=0.10,
                        help="Seconds between cells (default: 0.10)")
    parser.add_argument("--countdown", type=int,   default=3,
                        help="Seconds before execution starts (default: 3)")
    parser.add_argument("--debug",     action="store_true",
                        help="Show debug windows with detected grid and regions")
    args = parser.parse_args()

    board_obj = detect_board(debug=args.debug)
    if board_obj is None:
        print("Could not detect board.")
        sys.exit(1)

    print("\nSolving...")
    result = solve(board_obj.regions)

    if result is None:
        print("No solution found.")
        sys.exit(1)

    print("Solution:")
    print_board(board_obj.regions, result)
    print(f"\nQueens: {result}")

    execute_solution(board_obj.grid, result,
                     cell_delay=args.delay, countdown=args.countdown)
