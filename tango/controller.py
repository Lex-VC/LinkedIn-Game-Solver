"""Tango controller — detects, solves, and executes the Tango puzzle.

Click mechanics:
  1 click  → sun
  2 clicks → moon
Pre-filled cells are skipped automatically.
"""
import copy
import time
import ctypes
import ctypes.wintypes
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tango.board_vision import (
    SUN   as V_SUN,
    MOON  as V_MOON,
    EMPTY as V_EMPTY,
    GridInfo,
    detect_board,
)
from tango.solver import SUN, MOON, EMPTY, solve, print_board
import screen

_user32 = ctypes.windll.user32
_MOUSEEVENTF_MOVE     = 0x0001
_MOUSEEVENTF_LEFTDOWN = 0x0002
_MOUSEEVENTF_LEFTUP   = 0x0004
_MOUSEEVENTF_ABSOLUTE = 0x8000

# Clicks required to reach each symbol from an empty cell
_CLICKS = {SUN: 1, MOON: 2}


# ---------------------------------------------------------------------------
# Mouse helpers
# ---------------------------------------------------------------------------

def _move(x: int, y: int) -> None:
    ox, oy = screen.game_offset()
    screen_w = _user32.GetSystemMetrics(0)
    screen_h = _user32.GetSystemMetrics(1)
    nx = int((x + ox) * 65535 / screen_w)
    ny = int((y + oy) * 65535 / screen_h)
    _user32.mouse_event(_MOUSEEVENTF_MOVE | _MOUSEEVENTF_ABSOLUTE, nx, ny, 0, 0)


def _click(x: int, y: int) -> None:
    _move(x, y)
    _user32.mouse_event(_MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    _user32.mouse_event(_MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


def _aborted() -> bool:
    """Return True if the mouse has been moved to the top-left escape corner."""
    pos = ctypes.wintypes.POINT()
    _user32.GetCursorPos(ctypes.byref(pos))
    return pos.x <= 5 and pos.y <= 5


def _cell_center(grid: GridInfo, r: int, c: int) -> tuple[int, int]:
    x = int(grid.x + (c + 0.5) * grid.cell_w)
    y = int(grid.y + (r + 0.5) * grid.cell_h)
    return x, y


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def execute_solution(
    grid: GridInfo,
    original_cells: list[list[str]],
    solution: list[list[int]],
    cell_delay: float = 0.08,
    countdown: int = 3,
) -> None:
    """Click empty cells to realise the solution.

    Args:
        grid:           Detected grid info (screen coordinates).
        original_cells: Vision board (strings) — pre-filled cells are skipped.
        solution:       Fully-filled solver board (ints).
        cell_delay:     Pause between cells (seconds).
        countdown:      Seconds to wait before starting.
    """
    n = len(original_cells)
    actions = [
        (r, c, _CLICKS[solution[r][c]])
        for r in range(n)
        for c in range(n)
        if original_cells[r][c] == V_EMPTY
    ]

    if not actions:
        print("No empty cells to fill.")
        return

    print(f"Starting in {countdown}s — switch to the browser now ...")
    for i in range(countdown, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    for r, c, clicks in actions:
        if _aborted():
            print("Aborted: mouse moved to top-left corner.")
            return

        x, y = _cell_center(grid, r, c)
        for _ in range(clicks):
            _click(x, y)
            time.sleep(0.05)

        if cell_delay > 0:
            time.sleep(cell_delay)

    print("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_tango(countdown: int = 3, delay: float = 0.08,
              debug: bool = False) -> bool:
    """Detect, solve, and execute the Tango puzzle. Returns True on success."""
    board_obj = detect_board(debug=debug)
    if board_obj is None:
        print("Could not detect board.")
        return False

    grid     = board_obj.grid
    cells    = board_obj.cells
    h_con    = board_obj.h_constraints
    v_con    = board_obj.v_constraints
    n        = grid.rows

    # Convert vision strings -> solver ints
    def _vi(v: str) -> int:
        return SUN if v == V_SUN else (MOON if v == V_MOON else EMPTY)

    board = [[_vi(cells[r][c]) for c in range(n)] for r in range(n)]

    print("\nDetected board:")
    print_board(board)
    print(f"H-constraints: {sorted(h_con.items())}")
    print(f"V-constraints: {sorted(v_con.items())}")

    original_cells = copy.deepcopy(cells)

    print("\nSolving...")
    result = solve(board, h_con, v_con)

    if result is None:
        print("No solution found.")
        return False

    print("Solution:")
    print_board(result)

    execute_solution(grid, original_cells, result,
                     cell_delay=delay, countdown=countdown)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve and execute a LinkedIn Tango puzzle.")
    parser.add_argument("--delay",     type=float, default=0.08,
                        help="Seconds between cells (default: 0.08)")
    parser.add_argument("--countdown", type=int,   default=3,
                        help="Seconds before execution starts (default: 3)")
    parser.add_argument("--debug",     action="store_true",
                        help="Show debug windows with detected grid")
    args = parser.parse_args()

    screen.init_game_region()
    run_tango(countdown=args.countdown, delay=args.delay, debug=args.debug)
