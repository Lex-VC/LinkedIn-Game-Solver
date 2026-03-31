import math

Board = list[list[int]]


def _box_dims(n: int) -> tuple[int, int]:
    """Return (box_rows, box_cols) for an n×n Sudoku grid."""
    box_rows = math.isqrt(n)
    while n % box_rows != 0:
        box_rows -= 1
    box_cols = n // box_rows
    return box_rows, box_cols


def _possible(board: Board, row: int, col: int, num: int,
              box_rows: int, box_cols: int) -> bool:
    n = len(board)
    if num in board[row]:
        return False
    if any(board[r][col] == num for r in range(n)):
        return False
    br = (row // box_rows) * box_rows
    bc = (col // box_cols) * box_cols
    for dr in range(box_rows):
        for dc in range(box_cols):
            if board[br + dr][bc + dc] == num:
                return False
    return True


def solve(board: Board) -> Board | None:
    """Solve the Sudoku in-place using backtracking. Returns the solved board or None."""
    n = len(board)
    box_rows, box_cols = _box_dims(n)

    for row in range(n):
        for col in range(n):
            if board[row][col] == 0:
                for num in range(1, n + 1):
                    if _possible(board, row, col, num, box_rows, box_cols):
                        board[row][col] = num
                        if solve(board) is not None:
                            return board
                        board[row][col] = 0
                return None
    return board


def print_board(board: Board, original: Board | None = None) -> None:
    n = len(board)
    box_rows, box_cols = _box_dims(n)
    for r in range(n):
        if r > 0 and r % box_rows == 0:
            print("  " + "-" * (n * 4 - 1))
        row_str = ""
        for c in range(n):
            if c > 0 and c % box_cols == 0:
                row_str += " |"
            v = board[r][c]
            if original is not None and original[r][c] == 0:
                row_str += f" ({v})"
            else:
                row_str += f"  {v if v else '.'} "
        print(row_str)


if __name__ == "__main__":
    import sys
    import copy
    sys.path.insert(0, ".")
    from board_vision import detect_board

    grid, board = detect_board(debug=False)
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
    else:
        print("Solution:")
        print_board(result, original)
