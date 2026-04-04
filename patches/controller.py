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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from patches.board_vision import PatchesBoard, GridInfo, PatchSeed, detect_board, draw_debug
from patches.solver import solve, print_board, EMPTY
import screen

import cv2

_user32 = ctypes.windll.user32
_MOUSEEVENTF_MOVE     = 0x0001
_MOUSEEVENTF_LEFTDOWN = 0x0002
_MOUSEEVENTF_LEFTUP   = 0x0004
_MOUSEEVENTF_ABSOLUTE = 0x8000


# ---------------------------------------------------------------------------
# Mouse helpers
# ---------------------------------------------------------------------------

_SCREEN_W = _user32.GetSystemMetrics(0)
_SCREEN_H = _user32.GetSystemMetrics(1)


def _move(x: int, y: int) -> None:
    ox, oy = screen.game_offset()
    nx = int((x + ox) * 65535 / _SCREEN_W)
    ny = int((y + oy) * 65535 / _SCREEN_H)
    _user32.mouse_event(_MOUSEEVENTF_MOVE | _MOUSEEVENTF_ABSOLUTE, nx, ny, 0, 0)


def _drag_through(points: list[tuple[int, int]], delay: float = 0.0) -> None:
    """Mousedown at first point, move through each subsequent point, mouseup."""
    _move(*points[0])
    _user32.mouse_event(_MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    for x, y in points[1:]:
        if delay > 0:
            time.sleep(delay)
        _move(x, y)
    _user32.mouse_event(_MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


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
    move_delay: float = 0.08,
    countdown:  int   = 3,
) -> None:
    """Execute the solution by dragging between opposing corners of each shape.

    For each seed's patch, find the bounding rectangle of all cells assigned
    to it, then drag from the top-left pixel corner to the bottom-right pixel
    corner.  The game associates the dragged rectangle with the seed inside it.

    Args:
        grid:         Detected grid (screen coordinates).
        seeds:        Seed list from board_vision.
        solution:     2-D grid[r][c] = seed index from solver.
        shape_delay:  Pause between shapes (seconds).
        countdown:    Seconds before execution begins.
    """
    rows = len(solution)
    cols = len(solution[0]) if rows else 0

    # Collect all cells per shape index
    shape_cells: dict[int, list[tuple[int, int]]] = {}
    for r in range(rows):
        for c in range(cols):
            si = solution[r][c]
            if si == EMPTY:
                continue
            shape_cells.setdefault(si, []).append((r, c))

    if not shape_cells:
        print("No shapes to draw.")
        return

    print(f"Starting in {countdown}s — switch to the browser now ...")
    for i in range(countdown, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    for si in sorted(shape_cells):
        if _aborted():
            print("Aborted: mouse moved to top-left corner.")
            return

        seed = seeds[si]
        cells = shape_cells[si]

        min_r = min(r for r, _ in cells)
        min_c = min(c for _, c in cells)
        max_r = max(r for r, _ in cells)
        max_c = max(c for _, c in cells)

        print(f"  Shape {si} (row={seed.row}, col={seed.col}, "
              f"size={seed.size}, type={seed.shape_type}) "
              f"bbox rows {min_r}-{max_r} cols {min_c}-{max_c}")

        points = [_cell_center(grid, r, c) for r, c in sorted(cells)]
        _drag_through(points, delay=move_delay)

    print("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_patches(countdown: int = 3, delay: float = 0.08,
                debug: bool = False, solve_only: bool = False) -> bool:
    """Detect, solve, and execute the Patches puzzle. Returns True on success."""
    board_obj = detect_board(debug=False)
    if board_obj is None:
        print("Could not detect board.")
        return False

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
        return False

    print("Solution:")
    print_board(result, seeds)

    if debug:
        img = screen.capture()
        cv2.imshow("Patches — solution", draw_debug(img, board_obj, result))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if not solve_only:
        execute_solution(grid, seeds, result,
                         move_delay=delay,
                         countdown=countdown)
    return True


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

    screen.init_game_region()
    run_patches(countdown=args.countdown, delay=args.delay,
                debug=args.debug, solve_only=args.solve_only)
