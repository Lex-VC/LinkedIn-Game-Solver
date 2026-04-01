"""LinkedIn Queens puzzle solver.

Rules:
  1. Exactly one queen per row.
  2. Exactly one queen per column.
  3. Exactly one queen per colour region.
  4. No two queens may touch, not even diagonally (8-directional adjacency).
"""
from __future__ import annotations

Regions = list[list[int]]  # regions[r][c] = region_id


def solve(regions: Regions) -> list[tuple[int, int]] | None:
    """Return (row, col) positions for each queen, or None if unsolvable.

    Places queens row by row using backtracking.  Because each row gets
    exactly one queen and rows are filled in order, adjacency only needs
    to be checked against the queen in the immediately preceding row.
    """
    n = len(regions)
    used_cols: set[int] = set()
    used_regions: set[int] = set()
    queens: list[tuple[int, int]] = []

    def _backtrack(row: int) -> bool:
        if row == n:
            return True
        for col in range(n):
            rid = regions[row][col]
            if col in used_cols or rid in used_regions:
                continue
            # Adjacency: only the queen in (row-1) can be within 1 row
            if queens and abs(queens[-1][1] - col) <= 1:
                continue

            queens.append((row, col))
            used_cols.add(col)
            used_regions.add(rid)
            if _backtrack(row + 1):
                return True
            queens.pop()
            used_cols.remove(col)
            used_regions.remove(rid)
        return False

    return queens if _backtrack(0) else None


def print_board(regions: Regions, queens: list[tuple[int, int]]) -> None:
    n = len(regions)
    queen_set = set(queens)
    for r in range(n):
        row_str = ""
        for c in range(n):
            if (r, c) in queen_set:
                row_str += " Q"
            else:
                row_str += f" {regions[r][c]}"
        print(row_str)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from board_vision import detect_board

    board_obj = detect_board(debug=False)
    if board_obj is None:
        print("Could not detect board.")
        sys.exit(1)

    print("\nSolving...")
    result = solve(board_obj.regions)
    if result is None:
        print("No solution found.")
    else:
        print("Solution:")
        print_board(board_obj.regions, result)
        print(f"\nQueens: {result}")
