"""Patches controller — detects, solves, and executes the Patches puzzle.

Interaction model (LinkedIn Patches):
  Each empty cell must be clicked once to assign it to the shape it belongs to.
  The game infers the shape from which seed the cell is adjacent/connected to,
  so clicking every non-seed cell in the solved rectangle (top-left → bottom-right)
  completes the shape.

  If the game requires a different mechanic (e.g. click-drag or multi-click),
  adjust execute_solution() accordingly.

Safety:
  Move the mouse to the top-left corner of the screen (≤ 5 px from corner) to
  abort execution at any time.
"""
import copy
import sys
import time
import ctypes
import ctypes.wintypes
import argparse

sys.path.insert(0, ".")
from board_vision import PatchesBoard, GridInfo, PatchSeed, detect_board, draw_debug
from solver import solve, print_board, EMPTY

import cv2

_user32 = ctypes.windll.user32
_MOUSEEVENTF_MOVE     = 0x0001
_MOUSEEVENTF_LEFTDOWN = 0x0002
_MOUSEEVENTF_LEFTUP   = 0x0004
_MOUSEEVENTF_ABSOLUTE = 0x8000


# ---------------------------------------------------------------------------
# Mouse helpers
# ---------------------------------------------------------------------------

def _move(x: int, y: int) -> None:
    screen_w = _user32.GetSystemMetrics(0)
    screen_h = _user32.GetSystemMetrics(1)
    nx = int(x * 65535 / screen_w)
    ny = int(y * 65535 / screen_h)
    _user32.mouse_event(_MOUSEEVENTF_MOVE | _MOUSEEVENTF_ABSOLUTE, nx, ny, 0, 0)


def _click(x: int, y: int) -> None:
    _move(x, y)
    _user32.mouse_event(_MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    _user32.mouse_event(_MOUSEEVENTF_LEFTUP,   0, 0, 0, 0)


def _aborted() -> bool:
    """Return True when the mouse is in the top-left escape corner."""
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
    grid:       GridInfo,
    seeds:      list[PatchSeed],
    solution:   list[list[int]],
    cell_delay: float = 0.08,
    countdown:  int   = 3,
) -> None:
    """Click every non-seed cell to realise the solved layout on screen.

    Cells are visited shape-by-shape (in seed order), top-left to bottom-right
    within each shape, so the browser has an unambiguous assignment sequence.
    Seed cells are skipped — they are already present on the board.

    Args:
        grid:        Detected grid (screen coordinates).
        seeds:       Seed list from board_vision.
        solution:    2-D grid[r][c] = seed index from solver.
        cell_delay:  Pause between individual cell clicks (seconds).
        countdown:   Seconds before execution begins.
    """
    rows = len(solution)
    cols = len(solution[0]) if rows else 0
    seed_positions = {(s.row, s.col) for s in seeds}

    # Build per-shape click lists (exclude the seed cell itself)
    shape_actions: dict[int, list[tuple[int, int]]] = {}
    for r in range(rows):
        for c in range(cols):
            si = solution[r][c]
            if si == EMPTY:
                continue
            if (r, c) in seed_positions:
                continue  # already placed
            shape_actions.setdefault(si, []).append((r, c))

    total_clicks = sum(len(v) for v in shape_actions.values())
    if total_clicks == 0:
        print("No empty cells to fill.")
        return

    print(f"Starting in {countdown}s — switch to the browser now ...")
    for i in range(countdown, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    for si in sorted(shape_actions):
        seed = seeds[si]
        print(f"  Shape {si} (row={seed.row}, col={seed.col}, "
              f"size={seed.size}, type={seed.shape_type})")
        for r, c in shape_actions[si]:
            if _aborted():
                print("Aborted: mouse moved to top-left corner.")
                return
            x, y = _cell_center(grid, r, c)
            _click(x, y)
            if cell_delay > 0:
                time.sleep(cell_delay)

    print("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solve and execute a LinkedIn Patches puzzle.")
    parser.add_argument("--delay",     type=float, default=0.08,
                        help="Seconds between cell clicks (default: 0.08)")
    parser.add_argument("--countdown", type=int,   default=3,
                        help="Seconds before execution starts (default: 3)")
    parser.add_argument("--debug",     action="store_true",
                        help="Show debug window with detected grid and solution")
    parser.add_argument("--solve-only", action="store_true",
                        help="Detect and solve but do not click")
    args = parser.parse_args()

    board_obj = detect_board(debug=False)
    if board_obj is None:
        print("Could not detect board.")
        sys.exit(1)

    grid  = board_obj.grid
    seeds = board_obj.seeds

    print(f"\nDetected {len(seeds)} seed(s):")
    for s in seeds:
        print(f"  [{s.row},{s.col}] size={s.size} type={s.shape_type}")

    total_seed_area = sum(s.size for s in seeds)
    grid_area       = grid.rows * grid.cols
    if total_seed_area != grid_area:
        print(f"WARNING: seed areas sum to {total_seed_area}, "
              f"but grid has {grid_area} cells — puzzle may be incomplete.")

    print("\nSolving ...")
    result = solve(grid.rows, grid.cols, seeds)

    if result is None:
        print("No solution found.")
        sys.exit(1)

    print("Solution:")
    print_board(result, seeds)

    if args.debug:
        from board_vision import capture_screen
        img = capture_screen()
        cv2.imshow("Patches — solution", draw_debug(img, board_obj, result))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if not args.solve_only:
        execute_solution(grid, seeds, result,
                         cell_delay=args.countdown,
                         countdown=args.countdown)
