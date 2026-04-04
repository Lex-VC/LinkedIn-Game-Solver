from collections import deque
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zip.board_vision import GridInfo, NumberCell, WallSet


def _neighbors(r: int, c: int, rows: int, cols: int,
               h_walls: WallSet, v_walls: WallSet):
    """Yield grid neighbours reachable from (r, c) without crossing a wall."""
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if not (0 <= nr < rows and 0 <= nc < cols):
            continue
        if dr == 1  and (r,  c)  in h_walls: continue   # wall on bottom of (r,c)
        if dr == -1 and (nr, nc) in h_walls: continue   # wall on bottom of (r-1,c)
        if dc == 1  and (r,  c)  in v_walls: continue   # wall on right of (r,c)
        if dc == -1 and (nr, nc) in v_walls: continue   # wall on right of (r,c-1)
        yield nr, nc


def _all_unvisited_reachable(r: int, c: int, visited: set, total: int,
                              rows: int, cols: int,
                              h_walls: WallSet, v_walls: WallSet) -> bool:
    """Return True if every unvisited cell can be reached from (r, c)."""
    unvisited_count = total - len(visited)
    if unvisited_count == 0:
        return True
    reachable: set = set()
    queue: deque = deque([(r, c)])
    while queue:
        cr, cc = queue.popleft()
        for nr, nc in _neighbors(cr, cc, rows, cols, h_walls, v_walls):
            if (nr, nc) not in visited and (nr, nc) not in reachable:
                reachable.add((nr, nc))
                queue.append((nr, nc))
    return len(reachable) == unvisited_count


def solve(grid: GridInfo, cells: list[NumberCell],
          h_walls: WallSet, v_walls: WallSet) -> list[tuple[int, int]] | None:
    """Find a path that visits every cell and passes through numbered cells in order.

    Returns a list of (row, col) tuples representing the solution path,
    or None if no solution exists.
    """
    rows, cols = grid.rows, grid.cols
    total = rows * cols

    number_at: dict[tuple[int, int], int] = {(c.row, c.col): c.number for c in cells}
    max_num = max(c.number for c in cells)

    start = next(c for c in cells if c.number == 1)

    path: list[tuple[int, int]] = [(start.row, start.col)]
    visited: set[tuple[int, int]] = {(start.row, start.col)}
    next_required = 2

    def dfs() -> bool:
        nonlocal next_required

        if len(visited) == total:
            return next_required > max_num

        r, c = path[-1]

        # Prune: if any unvisited cell is now unreachable, this branch is dead
        if not _all_unvisited_reachable(r, c, visited, total, rows, cols, h_walls, v_walls):
            return False

        for nr, nc in _neighbors(r, c, rows, cols, h_walls, v_walls):
            if (nr, nc) in visited:
                continue

            num = number_at.get((nr, nc))
            # Numbered cells must be visited in order
            if num is not None and num != next_required:
                continue

            path.append((nr, nc))
            visited.add((nr, nc))
            prev_required = next_required
            if num is not None:
                next_required = num + 1

            if dfs():
                return True

            path.pop()
            visited.remove((nr, nc))
            next_required = prev_required

        return False

    return path if dfs() else None


def print_solution(grid: GridInfo, path: list[tuple[int, int]],
                   cells: list[NumberCell]) -> None:
    """Print a human-readable grid showing the solution path."""
    step_at = {pos: i + 1 for i, pos in enumerate(path)}
    number_at = {(c.row, c.col): c.number for c in cells}

    col_w = len(str(grid.rows * grid.cols)) + 1
    for r in range(grid.rows):
        row_str = ""
        for c in range(grid.cols):
            pos = (r, c)
            step = step_at.get(pos, 0)
            marker = f"[{number_at[pos]}]" if pos in number_at else f" {step} "
            row_str += marker.center(col_w)
        print(row_str)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import screen
    screen.init_game_region()
    from zip.board_vision import detect_board

    grid, cells, h_walls, v_walls = detect_board(debug=False)

    if grid is None:
        print("Could not detect board.")
        sys.exit(1)

    print("Solving ...")
    path = solve(grid, cells, h_walls, v_walls)

    if path is None:
        print("No solution found.")
    else:
        print(f"Solution found in {len(path)} steps:")
        print_solution(grid, path, cells)
