import time
import ctypes

from board_vision import GridInfo, detect_board
from solver import solve, print_solution

_user32 = ctypes.windll.user32
_MOUSEEVENTF_MOVE       = 0x0001
_MOUSEEVENTF_LEFTDOWN   = 0x0002
_MOUSEEVENTF_LEFTUP     = 0x0004
_MOUSEEVENTF_ABSOLUTE   = 0x8000


def _move(x: int, y: int) -> None:
    # Normalise to 0-65535 range required by SendInput absolute mode
    screen_w = _user32.GetSystemMetrics(0)
    screen_h = _user32.GetSystemMetrics(1)
    nx = int(x * 65535 / screen_w)
    ny = int(y * 65535 / screen_h)
    _user32.mouse_event(_MOUSEEVENTF_MOVE | _MOUSEEVENTF_ABSOLUTE, nx, ny, 0, 0)


def _mouse_down() -> None:
    _user32.mouse_event(_MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)


def _mouse_up() -> None:
    _user32.mouse_event(_MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


def _cell_center(grid: GridInfo, row: int, col: int) -> tuple[int, int]:
    x = int(grid.x + (col + 0.5) * grid.cell_w)
    y = int(grid.y + (row + 0.5) * grid.cell_h)
    return x, y


def execute_solution(grid: GridInfo, path: list[tuple[int, int]],
                     move_delay: float = 0.02, countdown: int = 3) -> None:
    """Click and drag through the solution path on screen.

    Args:
        grid:        Detected grid info (screen coordinates).
        path:        Ordered list of (row, col) from the solver.
        move_delay:  Seconds to pause between each cell move.
        countdown:   Seconds to wait before starting (time to focus the browser).
    """
    if not path:
        print("Empty path — nothing to execute.")
        return

    print(f"Starting in {countdown}s — switch to the browser now ...")
    for i in range(countdown, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    start_x, start_y = _cell_center(grid, *path[0])

    _move(start_x, start_y)
    time.sleep(0.1)
    _mouse_down()
    time.sleep(move_delay)

    for row, col in path[1:]:
        x, y = _cell_center(grid, row, col)
        _move(x, y)
        if move_delay > 0:
            time.sleep(move_delay)

    _mouse_up()
    print("Done.")


if __name__ == "__main__":
    import sys
    import argparse
    sys.path.insert(0, ".")

    parser = argparse.ArgumentParser(description="Solve and execute a LinkedIn Zip puzzle.")
    parser.add_argument("--delay",     type=float, default=0.02,
                        help="Seconds between each cell move (default: 0.02)")
    parser.add_argument("--countdown", type=int,   default=3,
                        help="Seconds to wait before executing so you can focus the browser (default: 3)")
    parser.add_argument("--debug",     action="store_true",
                        help="Show debug window with detected grid and walls")
    args = parser.parse_args()

    grid, cells, h_walls, v_walls = detect_board(debug=args.debug)

    if grid is None:
        print("Could not detect board.")
        sys.exit(1)

    print("Solving ...")
    path = solve(grid, cells, h_walls, v_walls)

    if path is None:
        print("No solution found.")
        sys.exit(1)

    print_solution(grid, path, cells)
    execute_solution(grid, path, move_delay=args.delay, countdown=args.countdown)
