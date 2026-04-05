"""Tango puzzle solver.

Rules:
  1. Every cell is SUN (1) or MOON (2).
  2. No three consecutive same values in any row or column.
  3. Each row and each column has exactly n//2 suns and n//2 moons.
  4. Cells separated by '=' must share the same value.
  5. Cells separated by 'x' must have opposite values.

Constraint keys follow board_vision.py conventions:
  h_constraints[(r, c)] — edge between cell(r, c) and cell(r+1, c)
  v_constraints[(r, c)] — edge between cell(r, c) and cell(r, c+1)
"""
from __future__ import annotations

SUN   = 1
MOON  = 2
EMPTY = 0

EQUAL    = 'equal'
OPPOSITE = 'opposite'

Board        = list[list[int]]
HConstraints = dict[tuple[int, int], str]
VConstraints = dict[tuple[int, int], str]


def _edge_ok(a: int, b: int, constraint: str | None) -> bool:
    if constraint is None or a == EMPTY or b == EMPTY:
        return True
    if constraint == EQUAL    and a != b: return False
    if constraint == OPPOSITE and a == b: return False
    return True


def _can_place(board: Board, n: int, r: int, c: int, val: int,
                h_con: HConstraints, v_con: VConstraints) -> bool:
    half = n // 2

    if r > 0:
        if not _edge_ok(val, board[r-1][c], h_con.get((r-1, c))): return False
    if r < n-1:
        if not _edge_ok(val, board[r+1][c], h_con.get((r, c))):   return False
    if c > 0:
        if not _edge_ok(val, board[r][c-1], v_con.get((r, c-1))): return False
    if c < n-1:
        if not _edge_ok(val, board[r][c+1], v_con.get((r, c))):   return False

    row = board[r]
    if sum(1 for v in row if v == val) + 1 > half:
        return False
    for start in (c - 2, c - 1, c):
        if 0 <= start and start + 2 < n:
            w = [row[start + i] if start + i != c else val for i in range(3)]
            if EMPTY not in w and w[0] == w[1] == w[2]:
                return False

    col = [board[rr][c] for rr in range(n)]
    if sum(1 for v in col if v == val) + 1 > half:
        return False
    for start in (r - 2, r - 1, r):
        if 0 <= start and start + 2 < n:
            w = [col[start + i] if start + i != r else val for i in range(3)]
            if EMPTY not in w and w[0] == w[1] == w[2]:
                return False

    return True


def solve(board: Board, h_con: HConstraints, v_con: VConstraints) -> Board | None:
    """Fill all EMPTY cells. Modifies board in-place; returns it on success or None."""
    n = len(board)
    for r in range(n):
        for c in range(n):
            if board[r][c] == EMPTY:
                for val in (SUN, MOON):
                    if _can_place(board, n, r, c, val, h_con, v_con):
                        board[r][c] = val
                        if solve(board, h_con, v_con) is not None:
                            return board
                        board[r][c] = EMPTY
                return None
    return board


def print_board(board: Board) -> None:
    sym = {EMPTY: '.', SUN: 'S', MOON: 'M'}
    for row in board:
        print(' '.join(sym[v] for v in row))


if __name__ == "__main__":
    import sys
    import copy
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import screen
    screen.init_game_region()
    from tango.board_vision import (
        SUN  as V_SUN,
        MOON as V_MOON,
        detect_board,
    )

    board_obj = detect_board(debug=False)
    if board_obj is None:
        print("Could not detect board.")
        sys.exit(1)

    def _to_int(v: str) -> int:
        return SUN if v == V_SUN else (MOON if v == V_MOON else EMPTY)

    n = board_obj.grid.rows
    board = [[_to_int(board_obj.cells[r][c]) for c in range(n)] for r in range(n)]

    print("Detected board:")
    print_board(board)

    result = solve(board, board_obj.h_constraints, board_obj.v_constraints)
    if result is None:
        print("No solution found.")
    else:
        print("Solution:")
        print_board(result)
