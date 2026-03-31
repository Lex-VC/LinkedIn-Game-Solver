import copy
import time
import ctypes

from board_vision import GridInfo, detect_board
from solver import solve, print_board

_user32 = ctypes.windll.user32
_MOUSEEVENTF_MOVE     = 0x0001
_MOUSEEVENTF_LEFTDOWN = 0x0002
_MOUSEEVENTF_LEFTUP   = 0x0004
_MOUSEEVENTF_ABSOLUTE = 0x8000

# Virtual-key codes for digit keys 1-9
_VK_DIGIT = {i: 0x30 + i for i in range(1, 10)}


def _move(x: int, y: int) -> None:
    screen_w = _user32.GetSystemMetrics(0)
    screen_h = _user32.GetSystemMetrics(1)
    nx = int(x * 65535 / screen_w)
    ny = int(y * 65535 / screen_h)
    _user32.mouse_event(_MOUSEEVENTF_MOVE | _MOUSEEVENTF_ABSOLUTE, nx, ny, 0, 0)


def _click(x: int, y: int) -> None:
    _move(x, y)
    time.sleep(0.05)
    _user32.mouse_event(_MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    time.sleep(0.03)
    _user32.mouse_event(_MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


def _press_digit(digit: int) -> None:
    vk = _VK_DIGIT[digit]
    _user32.keybd_event(vk, 0, 0, 0)
    time.sleep(0.02)
    _user32.keybd_event(vk, 0, 2, 0)  # KEYEVENTF_KEYUP = 2


def _cell_center(grid: GridInfo, row: int, col: int) -> tuple[int, int]:
    x = int(grid.x + (col + 0.5) * grid.cell_w)
    y = int(grid.y + (row + 0.5) * grid.cell_h)
    return x, y


def execute_solution(grid: GridInfo, original: list[list[int]], solution: list[list[int]],
                     cell_delay: float = 0.08, countdown: int = 3) -> None:
    """Click each empty cell and type the solved digit.

    Args:
        grid:        Detected grid info (screen coordinates).
        original:    9x9 board as detected (0 = empty).
        solution:    9x9 fully-filled board from the solver.
        cell_delay:  Seconds to pause after typing each digit.
        countdown:   Seconds to wait before starting.
    """
    n = len(original)
    empty_cells = [
        (r, c)
        for r in range(n)
        for c in range(n)
        if original[r][c] == 0
    ]

    if not empty_cells:
        print("No empty cells to fill.")
        return

    print(f"Starting in {countdown}s — switch to the browser now ...")
    for i in range(countdown, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    for r, c in empty_cells:
        # Abort if mouse is at top-left corner
        pos = ctypes.wintypes.POINT()
        _user32.GetCursorPos(ctypes.byref(pos))
        if pos.x <= 5 and pos.y <= 5:
            print("Aborted: mouse moved to top-left corner.")
            return

        x, y = _cell_center(grid, r, c)
        _click(x, y)
        time.sleep(0.03)
        _press_digit(solution[r][c])
        if cell_delay > 0:
            time.sleep(cell_delay)

    print("Done.")


if __name__ == "__main__":
    import sys
    import argparse
    sys.path.insert(0, ".")

    parser = argparse.ArgumentParser(description="Solve and execute a LinkedIn Sudoku puzzle.")
    parser.add_argument("--delay",     type=float, default=0.03,
                        help="Seconds between each cell input (default: 0.08)")
    parser.add_argument("--countdown", type=int,   default=3,
                        help="Seconds to wait before executing (default: 3)")
    parser.add_argument("--debug",     action="store_true",
                        help="Show debug window with detected grid and numbers")
    args = parser.parse_args()

    grid, board = detect_board(debug=args.debug)

    if grid is None:
        print("Could not detect board.")
        sys.exit(1)

    original = copy.deepcopy(board)

    print("\nDetected board:")
    print_board(board)

    print("\nSolving ...")
    result = solve(board)

    if result is None:
        print("No solution found.")
        sys.exit(1)

    print("Solution:")
    print_board(result, original)

    execute_solution(grid, original, result, cell_delay=args.delay, countdown=args.countdown)
