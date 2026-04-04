"""Patches puzzle solver — rectangle-packing constraint-satisfaction.

Rules:
  1. The grid must be *completely* tiled by N non-overlapping rectangles.
  2. Each rectangle i must contain the seed cell for shape i.
  3. Rectangle i must have exactly seed_i.size cells  (width × height = size).
  4. Shape-type constraint:
       'square' → width == height
       'wide'   → width  > height
       'tall'   → height > width
       'any'    → no restriction (any valid factoring)

Algorithm — "first unassigned cell" backtracking:
  At each recursive step we find the topmost-leftmost cell that has not yet
  been assigned to a rectangle.  That cell *must* be covered by some not-yet-
  placed seed's rectangle.  We try every (seed, placement) pair that covers
  that cell, place the rectangle, and recurse.  This ordering guarantees we
  never leave a gap and prunes the search tree aggressively.
"""
from __future__ import annotations

EMPTY = -1

Board = list[list[int]]  # grid[r][c] = seed index (0-based) or EMPTY


# ---------------------------------------------------------------------------
# Rectangle enumeration helpers
# ---------------------------------------------------------------------------

def _factorings(size: int, shape_type: str) -> list[tuple[int, int]]:
    """Return all (height, width) factor pairs for *size* that satisfy shape_type."""
    result: list[tuple[int, int]] = []
    for h in range(1, size + 1):
        if size % h != 0:
            continue
        w = size // h
        if shape_type == 'square' and h != w:
            continue
        if shape_type == 'wide'   and w <= h:
            continue
        if shape_type == 'tall'   and h <= w:
            continue
        result.append((h, w))
    return result


def _placements_covering(
    seed_r: int, seed_c: int, size: int, shape_type: str,
    target_r: int, target_c: int,
    rows: int, cols: int,
    occupied: list[list[bool]],
) -> list[list[tuple[int, int]]]:
    """Return every valid rectangle placement for a seed that covers *target*.

    A placement is valid when it:
      • contains the seed position (seed_r, seed_c)
      • contains the target cell (target_r, target_c)
      • has area == size with the correct shape type
      • fits inside the grid
      • does not overlap any already-occupied cell
    """
    placements: list[list[tuple[int, int]]] = []

    for h, w in _factorings(size, shape_type):
        # Derive legal top-left row range so the rect contains both required cells.
        r0_min = max(0,        seed_r - h + 1, target_r - h + 1)
        r0_max = min(rows - h, seed_r,          target_r)

        c0_min = max(0,        seed_c - w + 1, target_c - w + 1)
        c0_max = min(cols - w, seed_c,          target_c)

        for r0 in range(r0_min, r0_max + 1):
            for c0 in range(c0_min, c0_max + 1):
                cells = [
                    (r0 + dr, c0 + dc)
                    for dr in range(h)
                    for dc in range(w)
                ]
                if all(not occupied[r][c] for r, c in cells):
                    placements.append(cells)

    return placements


# ---------------------------------------------------------------------------
# Backtracking solver
# ---------------------------------------------------------------------------

def solve(rows: int, cols: int, seeds: list) -> Board | None:
    """Tile the grid with rectangles satisfying all seed constraints.

    Args:
        rows, cols: grid dimensions.
        seeds:      list of PatchSeed-like objects with attributes
                    .row, .col, .size, .shape_type

    Returns:
        A 2-D list grid[r][c] = seed index, or None if no solution exists.
    """
    n      = len(seeds)
    placed = [False] * n
    occupied: list[list[bool]] = [[False] * cols for _ in range(rows)]
    grid: Board = [[EMPTY] * cols for _ in range(rows)]

    def backtrack() -> bool:
        # Find the first unassigned cell in reading order.
        first_r = first_c = -1
        for r in range(rows):
            for c in range(cols):
                if not occupied[r][c]:
                    first_r, first_c = r, c
                    break
            if first_r >= 0:
                break

        if first_r < 0:
            return True  # every cell is covered — success

        # Try each unplaced seed; only those whose rectangle can reach (first_r, first_c).
        for si in range(n):
            if placed[si]:
                continue
            seed = seeds[si]

            candidates = _placements_covering(
                seed.row, seed.col, seed.size, seed.shape_type,
                first_r, first_c,
                rows, cols, occupied,
            )

            for cells in candidates:
                # Place
                for r, c in cells:
                    occupied[r][c] = True
                    grid[r][c]     = si
                placed[si] = True

                if backtrack():
                    return True

                # Undo
                for r, c in cells:
                    occupied[r][c] = False
                    grid[r][c]     = EMPTY
                placed[si] = False

        return False  # no seed could cover (first_r, first_c)

    return grid if backtrack() else None


# ---------------------------------------------------------------------------
# Debug output
# ---------------------------------------------------------------------------

def print_board(grid: Board, seeds: list) -> None:
    """Print a text representation of the solved grid.

    Seed cells are shown as uppercase letters (A, B, …); non-seed cells of the
    same shape as lowercase.  EMPTY cells are shown as '.'.
    """
    if not grid:
        print("  (no solution)")
        return
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    seed_positions = {(s.row, s.col): si for si, s in enumerate(seeds)}
    for r in range(rows):
        row_str = ""
        for c in range(cols):
            idx = grid[r][c]
            if idx == EMPTY:
                row_str += ".  "
            else:
                ch = chr(ord('A') + idx % 26)
                if (r, c) in seed_positions and seed_positions[(r, c)] == idx:
                    row_str += ch + "  "   # seed cell — uppercase
                else:
                    row_str += ch.lower() + "  "
        print(" ", row_str)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import screen
    screen.init_game_region()
    from patches.board_vision import detect_board

    board_obj = detect_board(debug=False)
    if board_obj is None:
        print("Could not detect board.")
        sys.exit(1)

    grid   = board_obj.grid
    seeds  = board_obj.seeds

    print(f"\nGrid: {grid.cols}×{grid.rows}")
    print(f"Seeds: {seeds}\n")

    result = solve(grid.rows, grid.cols, seeds)
    if result is None:
        print("No solution found.")
    else:
        print("Solution:")
        print_board(result, seeds)
