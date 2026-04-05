"""video_demo.py — Game-region video with full CV pipeline visualization.

Records only the detected game area.  For each game:
  1. Runs the actual CV pipeline step-by-step, writing each intermediate
     image directly into the video file (no pop-up windows).
  2. Starts a live capture of the game area, then executes the solution
     in the browser — recording the actual play to video.

Detailed CV steps shown per game:
  Patches   — grayscale, Canny edges, H/V line detection, grid, seed
              saturation mask, shape template match, digit match, solution
  Zip       — gray saturation mask, H/V lines, grid, HoughCircles,
              Otsu + template match panel, Hamiltonian path solution
  Sudoku    — gray mask, grid, Otsu + digit template match, solution
  Tango     — HSV sun/moon masks, grid, constraint template match, solution
  Queens    — Canny edges, grid, LAB k-means color regions, solution
  Pinpoint  — HSV blue mask, thresholded OCR block, detected clues
  Crossclimb — peach/teal HSV masks, row bands, clue region

Usage:
    python video_demo.py [--output demo.mp4] [--load-wait 6] [--countdown 5]
                         [--step-time 2.0] [--game NAME] [--no-record]

GAME_KEYS: pinpoint, patches, zip, sudoku, tango, queens, crossclimb
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import ctypes.wintypes
import sys
import threading
import time
from pathlib import Path

import cv2
import mss
import numpy as np

_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import screen

# ─── Public vision/solver imports ────────────────────────────────────────────

from patches.board_vision import (
    find_grid          as _p_find_grid,
    find_seeds         as _p_find_seeds,
    read_seed_numbers  as _p_read_numbers,
    draw_debug         as _p_draw,
    PatchesBoard,
)
from patches.solver     import solve as _p_solve
from patches.controller import execute_solution as _p_exec

from zip.board_vision import (
    find_grid    as _z_find_grid,
    find_numbers as _z_find_numbers,
    find_walls   as _z_find_walls,
    draw_debug   as _z_draw,
)
from zip.solver     import solve as _z_solve
from zip.controller import execute_solution as _z_exec

from sudoku.board_vision import (
    find_grid    as _s_find_grid,
    find_numbers as _s_find_numbers,
    draw_debug   as _s_draw,
)
from sudoku.solver     import solve as _s_solve
from sudoku.controller import execute_solution as _s_exec

from tango.board_vision import (
    find_grid        as _t_find_grid,
    find_cells       as _t_find_cells,
    find_constraints as _t_find_constraints,
    draw_debug       as _t_draw,
    TangoBoard,
    SUN as T_SUN, MOON as T_MOON, EMPTY as T_EMPTY,
)
from tango.solver     import solve as _t_solve, SUN as SOL_SUN, MOON as SOL_MOON
from tango.controller import execute_solution as _t_exec

from queens.board_vision import (
    find_grid    as _q_find_grid,
    find_regions as _q_find_regions,
    draw_debug   as _q_draw,
    QueensBoard,
)
from queens.solver     import solve as _q_solve
from queens.controller import execute_solution as _q_exec

from pinpoint.board_vision import (
    _find_gradient_block,
    extract_clue_words,
    find_input_box,
    draw_debug as _pin_draw,
)
from pinpoint.board_vision import _BLUE_LO as _PIN_LO, _BLUE_HI as _PIN_HI
from pinpoint.controller import run_pinpoint

from crossclimb.board_vision import (
    detect_board as _cc_detect,
    read_clue,
    _PEACH_LO, _PEACH_HI, _TEAL_LO, _TEAL_HI,
)
from crossclimb.controller import run_crossclimb

# ─── Private vision helpers (for intermediate CV steps) ──────────────────────

from patches.board_vision import (
    _extract_grid_lines    as _p_extract_lines,
    _cell_roi              as _p_cell_roi,
    _preprocess_roi_for_shape as _p_prep_shape,
    _load_shape_templates  as _p_load_shape_tmpls,
    _score_shape_templates as _p_score_shape,
    _DASH_BRIDGE           as _P_BRIDGE,
    _MIN_LINE_LEN          as _P_MIN_LINE,
)
from zip.board_vision    import _grid_line_mask as _z_line_mask, _preprocess_roi as _z_prep_roi, _load_templates as _z_load_tmpls
from sudoku.board_vision import _grid_line_mask as _s_line_mask, _preprocess_roi as _s_prep_roi, _load_templates as _s_load_tmpls
from tango.board_vision  import _extract_grid_lines as _t_extract_lines, _SUN_LO, _SUN_HI, _MOON_LO, _MOON_HI
from queens.board_vision import _extract_grid_lines as _q_extract_lines

# ─── Windows input ────────────────────────────────────────────────────────────

_user32           = ctypes.windll.user32
_KEYEVENTF_KEYUP  = 0x0002
_VK_CONTROL       = 0x11
_VK_RETURN        = 0x0D
_MOUSEEVENTF_MOVE     = 0x0001
_MOUSEEVENTF_ABSOLUTE = 0x8000

# ─── GameRecorder ─────────────────────────────────────────────────────────────

class GameRecorder:
    """Writes frames to an MP4 cropped to the game region.

    Two modes:
      write_static(frame, seconds) — inject a pre-rendered frame for N seconds.
      start_live() / stop_live()   — capture the game area from the screen live.
    """

    def __init__(self, path: str, fps: int = 30):
        self._path  = path
        self._fps   = fps
        self._w = self._h = 0
        self._x = self._y = 0
        self._writer: cv2.VideoWriter | None = None
        self._live_thread: threading.Thread | None = None
        self._live_running = False
        self._lock = threading.Lock()

    def init(self) -> None:
        """Call after screen.init_game_region() to obtain dimensions."""
        region = screen._game_region
        if region:
            self._x, self._y, self._w, self._h = region
        else:
            with mss.mss() as sct:
                m = sct.monitors[1]
                self._x, self._y, self._w, self._h = 0, 0, m["width"], m["height"]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(self._path, fourcc, self._fps,
                                       (self._w, self._h))
        print(f"  Recording {self._w}×{self._h} → {self._path}")

    def _write(self, frame: np.ndarray) -> None:
        f = frame if frame.shape[:2] == (self._h, self._w) else \
            cv2.resize(frame, (self._w, self._h))
        with self._lock:
            if self._writer and self._writer.isOpened():
                self._writer.write(f)

    def write_static(self, frame: np.ndarray, seconds: float) -> None:
        n = max(1, int(seconds * self._fps))
        f = frame if frame.shape[:2] == (self._h, self._w) else \
            cv2.resize(frame, (self._w, self._h))
        for _ in range(n):
            with self._lock:
                if self._writer and self._writer.isOpened():
                    self._writer.write(f)

    def start_live(self) -> None:
        self._live_running = True
        self._live_thread = threading.Thread(target=self._live_loop, daemon=True)
        self._live_thread.start()

    def stop_live(self) -> None:
        self._live_running = False
        if self._live_thread:
            self._live_thread.join(timeout=4)
        self._live_thread = None

    def _live_loop(self) -> None:
        interval = 1.0 / self._fps
        mon = {"left": self._x, "top": self._y, "width": self._w, "height": self._h}
        with mss.mss() as sct:
            while self._live_running:
                t0 = time.perf_counter()
                shot = sct.grab(mon)
                frame = cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)
                with self._lock:
                    if self._writer and self._writer.isOpened():
                        self._writer.write(frame)
                gap = interval - (time.perf_counter() - t0)
                if gap > 0:
                    time.sleep(gap)

    def finish(self) -> None:
        self.stop_live()
        with self._lock:
            if self._writer:
                self._writer.release()
                self._writer = None
        print(f"  Saved → {self._path}")


class _NullRecorder:
    """Drop-in for GameRecorder when --no-record is set."""
    def write_static(self, _frame, _secs): pass
    def start_live(self): pass
    def stop_live(self): pass
    def finish(self): pass


# ─── Frame helpers ────────────────────────────────────────────────────────────

def _label(img: np.ndarray, title: str, hint: str = "") -> np.ndarray:
    """Add a semi-transparent label bar at the top of the image."""
    out = img.copy()
    h, w = out.shape[:2]
    bar = 36
    # Darken top strip
    out[:bar] = (out[:bar].astype(np.float32) * 0).astype(np.uint8)
    cv2.putText(out, title, (7, bar - 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    if hint:
        tw, _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0]
        cv2.putText(out, hint, (w - tw - 7, bar - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (70, 210, 90), 1, cv2.LINE_AA)
    return out


def _g2b(gray: np.ndarray) -> np.ndarray:
    """Grayscale → BGR."""
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _mask_overlay(base: np.ndarray, mask: np.ndarray,
                  color: tuple[int, int, int], alpha: float = 0.92,
                  dilate: int = 3) -> np.ndarray:
    """Colorize binary mask pixels on top of base image.

    Only masked pixels are modified — background is left untouched.
    `dilate` thickens thin lines so they're visible in small game regions.
    """
    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate, dilate))
        mask = cv2.dilate(mask, k)
    out  = base.copy()
    m    = mask > 0
    col  = np.array(color, dtype=np.float32)
    orig = out[m].astype(np.float32)
    out[m] = np.clip(orig * (1.0 - alpha) + col * alpha, 0, 255).astype(np.uint8)
    return out


def _draw_grid_on(img: np.ndarray, grid) -> np.ndarray:
    out = img.copy()
    gx, gy, gw, gh = grid.x, grid.y, grid.width, grid.height
    cv2.rectangle(out, (gx, gy), (gx + gw, gy + gh), (0, 255, 0), 2)
    for r in range(grid.rows + 1):
        y = int(gy + r * grid.cell_h)
        cv2.line(out, (gx, y), (gx + gw, y), (0, 200, 0), 1)
    for c in range(grid.cols + 1):
        x = int(gx + c * grid.cell_w)
        cv2.line(out, (x, gy), (x, gy + gh), (0, 200, 0), 1)
    cv2.putText(out, f"{grid.cols}x{grid.rows}", (gx + 4, gy - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def _inset(main: np.ndarray, panel: np.ndarray, margin: int = 8) -> np.ndarray:
    """Composite panel into bottom-right corner of main."""
    out = main.copy()
    ph, pw = panel.shape[:2]
    mh, mw = out.shape[:2]
    y1 = mh - ph - margin
    x1 = mw - pw - margin
    if y1 >= 0 and x1 >= 0:
        cv2.rectangle(out, (x1 - 2, y1 - 2), (x1 + pw + 2, y1 + ph + 2),
                      (70, 210, 90), 1)
        out[y1:y1 + ph, x1:x1 + pw] = panel
    return out


def _match_panel(roi_bgr: np.ndarray, proc_gray: np.ndarray,
                 tmpl_gray: np.ndarray | None, label: str) -> np.ndarray:
    """Three-column panel: cell ROI | processed binary | best template."""
    S = 64
    roi_s   = cv2.resize(roi_bgr,  (S, S))
    proc_s  = _g2b(cv2.resize(proc_gray, (S, S)))
    tmpl_s  = _g2b(cv2.resize(tmpl_gray, (S, S))) if tmpl_gray is not None \
              else np.zeros((S, S, 3), dtype=np.uint8)
    div = np.full((S, 2, 3), 55, dtype=np.uint8)
    strip = np.hstack([roi_s, div, proc_s, div, tmpl_s])
    bar = np.zeros((20, strip.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, label, (3, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.37, (70, 210, 90), 1, cv2.LINE_AA)
    return np.vstack([strip, bar])


# ─── Patches CV steps ─────────────────────────────────────────────────────────

def _patches_steps(img: np.ndarray, step_time: float, rec: GameRecorder) -> tuple | None:
    """Write all detection visualization frames.  Returns (board, solution) or None."""

    # 1 Raw
    rec.write_static(_label(img, "Patches  —  raw capture"), step_time)

    # 2 Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rec.write_static(_label(_g2b(gray), "Grayscale conversion"), step_time)

    # 3 Canny edges (raw, before bridging)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges   = cv2.Canny(blurred, 30, 90)
    rec.write_static(_label(_mask_overlay(img, edges, (0, 255, 255)),
                             "Canny edge detection  [low=30, high=90]"), step_time)

    # 4 H-lines + V-lines (bridged & opened)
    h_lines, v_lines = _p_extract_lines(img)
    hv = _mask_overlay(_mask_overlay(img, h_lines, (80, 80, 255)),
                       v_lines, (255, 165, 0))
    rec.write_static(_label(hv,
                             "H-lines (red, bridge+open)  +  V-lines (orange, bridge+open)"),
                     step_time)

    # 5 Grid detected
    grid = _p_find_grid(img)
    if grid is None:
        print("  ERROR: grid not found"); return None
    rec.write_static(_label(_draw_grid_on(img, grid),
                             f"Uniform-spacing cluster → grid {grid.cols}×{grid.rows}",
                             f"({grid.x},{grid.y}) {grid.width}×{grid.height}px"), step_time)

    # 6 Seeds — saturation mask highlight + shape template match inset
    seeds = _p_find_seeds(img, grid)
    _p_read_numbers(img, grid, seeds)
    board = PatchesBoard(grid, seeds)

    seeds_frame = _p_draw(img, board)
    if seeds:
        s = seeds[0]
        crop = img[grid.y:grid.y + grid.height, grid.x:grid.x + grid.width]
        roi  = _p_cell_roi(crop, grid, s.row, s.col)
        proc = _p_prep_shape(roi)
        scores = _p_score_shape(proc)
        best   = max(scores, key=scores.get) if scores else "any"
        tmpls  = _p_load_shape_tmpls()
        tmpl   = tmpls.get(best, [None])[0]
        panel  = _match_panel(roi, proc, tmpl,
                               f"ROI | sat-mask binary | best template: {best} ({scores.get(best,0):.2f})")
        seeds_frame = _inset(seeds_frame, panel)

    rec.write_static(_label(seeds_frame,
                             f"Seed detection: {len(seeds)} seed(s)  "
                             f"(saturation threshold + shape/digit templates)"), step_time)

    # 7 Solution
    result = _p_solve(grid.rows, grid.cols, seeds)
    if result is None:
        print("  ERROR: no solution"); return None
    rec.write_static(_label(_p_draw(img, board, result),
                             "Solution  —  constraint-propagation rectangle packing"), step_time)

    return board, result


# ─── Zip CV steps ─────────────────────────────────────────────────────────────

def _zip_steps(img: np.ndarray, step_time: float, rec: GameRecorder) -> tuple | None:
    rec.write_static(_label(img, "Zip  —  raw capture"), step_time)

    # Gray saturation mask (grid lines)
    mask = _z_line_mask(img)
    rec.write_static(_label(_mask_overlay(img, mask, (0, 220, 255)),
                             "Gray saturation mask  (unsaturated mid-gray = grid lines)"), step_time)

    # H / V lines
    h_lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1)))
    v_lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50)))
    hv = _mask_overlay(_mask_overlay(img, h_lines, (80, 80, 255)), v_lines, (255, 165, 0))
    rec.write_static(_label(hv, "H-lines (red)  +  V-lines (orange)  via morphological open (len=50)"),
                     step_time)

    # Grid
    grid = _z_find_grid(img)
    if grid is None:
        print("  ERROR: grid not found"); return None
    rec.write_static(_label(_draw_grid_on(img, grid),
                             f"Grid {grid.cols}×{grid.rows}  from intersection cluster"), step_time)

    # Circles + template match
    cells   = _z_find_numbers(img, grid)
    h_walls, v_walls = _z_find_walls(img, grid)
    detected = _z_draw(img, grid, cells, h_walls, v_walls)

    if cells:
        c = cells[0]
        grid_crop = img[grid.y:grid.y + grid.height, grid.x:grid.x + grid.width]
        cx = int((c.col + 0.5) * grid.cell_w)
        cy = int((c.row + 0.5) * grid.cell_h)
        r  = int(grid.cell_w * 0.35)
        inner = int(r * 0.6)
        x1, y1 = max(0, cx - inner), max(0, cy - inner)
        x2, y2 = min(grid_crop.shape[1], cx + inner), min(grid_crop.shape[0], cy + inner)
        roi = grid_crop[y1:y2, x1:x2]
        if roi.size > 0:
            proc  = _z_prep_roi(roi)
            tmpls = _z_load_tmpls()
            tmpl  = tmpls.get(c.number, [None])[0]
            panel = _match_panel(roi, proc, tmpl,
                                  f"ROI | Otsu binary | template: {c.number}")
            detected = _inset(detected, panel)

    rec.write_static(_label(detected,
                             f"HoughCircles → {len(cells)} circle(s) + walls (dark strip scan) "
                             f"+ Otsu + template match"), step_time)

    # Solution path
    path = _z_solve(grid, cells, h_walls, v_walls)
    if path is None:
        print("  ERROR: no solution"); return None

    sol = _z_draw(img, grid, cells, h_walls, v_walls)
    n   = len(path) - 1
    for i in range(n):
        r1, c1 = path[i];  r2, c2 = path[i + 1]
        x1 = int(grid.x + (c1 + 0.5) * grid.cell_w)
        y1 = int(grid.y + (r1 + 0.5) * grid.cell_h)
        x2 = int(grid.x + (c2 + 0.5) * grid.cell_w)
        y2 = int(grid.y + (r2 + 0.5) * grid.cell_h)
        t  = i / max(n, 1)
        cv2.line(sol, (x1, y1), (x2, y2),
                 (int(255 * (1 - t)), int(165 * t), int(255 * t)), 4, cv2.LINE_AA)
    rec.write_static(_label(sol, "Hamiltonian path solution  (BFS/DFS)"), step_time)

    return grid, cells, h_walls, v_walls, path


# ─── Sudoku CV steps ──────────────────────────────────────────────────────────

def _sudoku_steps(img: np.ndarray, step_time: float, rec: GameRecorder) -> tuple | None:
    rec.write_static(_label(img, "Mini Sudoku  —  raw capture"), step_time)

    mask = _s_line_mask(img)
    rec.write_static(_label(_mask_overlay(img, mask, (0, 220, 255)),
                             "Gray saturation mask  (unsaturated mid-gray = grid lines)"), step_time)

    grid = _s_find_grid(img)
    if grid is None:
        print("  ERROR: grid not found"); return None
    rec.write_static(_label(_draw_grid_on(img, grid),
                             f"Grid {grid.cols}×{grid.rows}  detected"), step_time)

    board    = _s_find_numbers(img, grid)
    original = copy.deepcopy(board)
    detected = _s_draw(img, grid, board)

    # Digit template match inset: find first non-zero cell
    for r in range(grid.rows):
        for c in range(grid.cols):
            if board[r][c] != 0:
                gc = img[grid.y:grid.y + grid.height, grid.x:grid.x + grid.width]
                cx  = (c + 0.5) * grid.cell_w
                cy2 = (r + 0.5) * grid.cell_h
                hw  = grid.cell_w * 0.65 / 2
                hh  = grid.cell_h * 0.65 / 2
                x1, y1 = max(0, int(cx - hw)), max(0, int(cy2 - hh))
                x2, y2 = min(gc.shape[1], int(cx + hw)), min(gc.shape[0], int(cy2 + hh))
                roi  = gc[y1:y2, x1:x2]
                if roi.size > 0:
                    proc  = _s_prep_roi(roi)
                    tmpls = _s_load_tmpls()
                    tmpl  = tmpls.get(board[r][c], [None])[0]
                    panel = _match_panel(roi, proc, tmpl,
                                         f"ROI | Otsu binary | digit template: {board[r][c]}")
                    detected = _inset(detected, panel)
                break
        else:
            continue
        break

    rec.write_static(_label(detected,
                             f"Digit detection: {sum(v!=0 for row in board for v in row)} "
                             f"pre-filled  (Otsu + template match)"), step_time)

    result = _s_solve(board)
    if result is None:
        print("  ERROR: no solution"); return None

    sol = _s_draw(img, grid, original)
    for r in range(grid.rows):
        for c in range(grid.cols):
            if original[r][c] == 0 and result[r][c] != 0:
                px = int(grid.x + (c + 0.5) * grid.cell_w)
                py = int(grid.y + (r + 0.5) * grid.cell_h)
                cv2.putText(sol, str(result[r][c]), (px - 8, py + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 120, 255), 2, cv2.LINE_AA)
    rec.write_static(_label(sol, "Solution  —  backtracking constraint solver"), step_time)

    return grid, original, result


# ─── Tango CV steps ───────────────────────────────────────────────────────────

def _tango_steps(img: np.ndarray, step_time: float, rec: GameRecorder) -> tuple | None:
    rec.write_static(_label(img, "Tango  —  raw capture"), step_time)

    # HSV sun + moon masks
    hsv      = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sun_mask = cv2.inRange(hsv, _SUN_LO,  _SUN_HI)
    moon_mask= cv2.inRange(hsv, _MOON_LO, _MOON_HI)
    hsv_frame = _mask_overlay(
        _mask_overlay(img, sun_mask, (0, 220, 255)),
        moon_mask, (255, 60, 200))
    rec.write_static(_label(hsv_frame,
                             "HSV sun mask (yellow, H 10-30)  +  moon mask (magenta, H 95-130)"),
                     step_time)

    # Grid lines (Canny-based)
    h_lines, v_lines = _t_extract_lines(img)
    hv = _mask_overlay(_mask_overlay(img, h_lines, (80, 80, 255)), v_lines, (255, 165, 0))
    rec.write_static(_label(hv, "H-lines  +  V-lines  (Canny → morphological open)"), step_time)

    # Grid + symbols + constraints
    grid = _t_find_grid(img)
    if grid is None:
        print("  ERROR: grid not found"); return None
    cells    = _t_find_cells(img, grid)
    h_con, v_con = _t_find_constraints(img, grid)
    board    = TangoBoard(grid, cells, h_con, v_con)
    detected = _t_draw(img, board)
    rec.write_static(_label(detected,
                             f"Grid {grid.cols}×{grid.rows}  +  symbols  +  constraints  "
                             f"({len(h_con)+len(v_con)} edge(s) via template match)"), step_time)

    # Solution
    def _vi(v: str) -> int:
        return SOL_SUN if v == T_SUN else (SOL_MOON if v == T_MOON else 0)
    solver_board = [[_vi(cells[r][c]) for c in range(grid.cols)]
                    for r in range(grid.rows)]
    result = _t_solve(solver_board, h_con, v_con)
    if result is None:
        print("  ERROR: no solution"); return None

    sol = _t_draw(img, board)
    _lbl = {SOL_SUN: "S", SOL_MOON: "M"}
    _col = {SOL_SUN: (0, 165, 255), SOL_MOON: (180, 60, 0)}
    for r in range(grid.rows):
        for c in range(grid.cols):
            if cells[r][c] == T_EMPTY:
                v = result[r][c]
                if v in _lbl:
                    px = int(grid.x + (c + 0.5) * grid.cell_w)
                    py = int(grid.y + (r + 0.5) * grid.cell_h)
                    cv2.putText(sol, _lbl[v], (px - 10, py + 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, _col[v], 2, cv2.LINE_AA)
    rec.write_static(_label(sol, "Solution  —  backtracking CSP"), step_time)

    return grid, cells, h_con, v_con, result


# ─── Queens CV steps ──────────────────────────────────────────────────────────

def _queens_steps(img: np.ndarray, step_time: float, rec: GameRecorder) -> tuple | None:
    rec.write_static(_label(img, "Queens  —  raw capture"), step_time)

    # Canny edges
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 90)
    rec.write_static(_label(_mask_overlay(img, edges, (0, 255, 255)),
                             "Canny edge detection  [low=30, high=90]"), step_time)

    # H / V lines
    h_lines, v_lines = _q_extract_lines(img)
    hv = _mask_overlay(_mask_overlay(img, h_lines, (80, 80, 255)), v_lines, (255, 165, 0))
    rec.write_static(_label(hv, "H-lines  +  V-lines  (morphological open, len=25)"), step_time)

    # Grid
    grid = _q_find_grid(img)
    if grid is None:
        print("  ERROR: grid not found"); return None
    rec.write_static(_label(_draw_grid_on(img, grid),
                             f"Grid {grid.cols}×{grid.rows}  detected"), step_time)

    # Color regions (LAB k-means)
    regions = _q_find_regions(img, grid)
    board   = QueensBoard(grid, regions)
    rec.write_static(_label(_q_draw(img, board),
                             f"LAB k-means (k={grid.rows})  →  {grid.rows} colour regions"), step_time)

    # Solution
    result = _q_solve(regions)
    if result is None:
        print("  ERROR: no solution"); return None

    sol = _q_draw(img, board)
    for r, c in result:
        px = int(grid.x + (c + 0.5) * grid.cell_w)
        py = int(grid.y + (r + 0.5) * grid.cell_h)
        cv2.circle(sol, (px, py), int(grid.cell_w * 0.28), (220, 30, 200), -1, cv2.LINE_AA)
        cv2.putText(sol, "Q", (px - 9, py + 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    rec.write_static(_label(sol, "Solution  —  N-Queens backtracking"), step_time)

    return grid, result


# ─── Pinpoint CV steps (called every round) ──────────────────────────────────

def _pinpoint_detect_frame(img: np.ndarray, round_num: int, max_rounds: int,
                            step_time: float, rec, first_round: bool) -> tuple:
    """Write CV visualization frames for one Pinpoint round.

    On the first round also shows the raw capture and HSV mask.
    Returns (clues, input_pos).
    """
    if first_round:
        rec.write_static(_label(img, "Pinpoint  —  raw capture"), step_time)
        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        blue = cv2.inRange(hsv, _PIN_LO, _PIN_HI)
        rec.write_static(_label(_mask_overlay(img, blue, (255, 80, 0)),
                                 "HSV blue mask  [H 100-125, S 30-255, V 120-255]"),
                         step_time)

    block     = _find_gradient_block(img)
    clues     = extract_clue_words(img)
    input_pos = find_input_box(img)
    detected  = _pin_draw(img, block, clues, input_pos)

    # Tesseract threshold inset
    if block:
        bx, by, bw, bh = block
        roi  = img[by:by + bh, bx:bx + bw]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)
        panel_bgr = _g2b(cv2.resize(thresh, (150, 150)))
        bar = np.zeros((20, 150, 3), dtype=np.uint8)
        cv2.putText(bar, "Tesseract input", (3, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, (70, 210, 90), 1, cv2.LINE_AA)
        detected = _inset(detected, np.vstack([panel_bgr, bar]))

    clue_str = ", ".join(f'"{c}"' for c in clues) if clues else "(none yet)"
    rec.write_static(_label(detected,
                             f"Round {round_num}/{max_rounds}  —  OCR clues: {clue_str}",
                             "Tesseract + HSV block"), step_time)
    return clues, input_pos


# ─── Crossclimb CV steps ──────────────────────────────────────────────────────

def _crossclimb_steps(img: np.ndarray, step_time: float,
                       rec: GameRecorder) -> object | None:
    rec.write_static(_label(img, "Crossclimb  —  raw capture"), step_time)

    # Peach + teal HSV masks
    hsv   = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    peach = cv2.inRange(hsv, _PEACH_LO, _PEACH_HI)
    teal  = cv2.inRange(hsv, _TEAL_LO,  _TEAL_HI)
    masks = _mask_overlay(_mask_overlay(img, peach, (0, 100, 255)), teal, (255, 210, 0))
    rec.write_static(_label(masks,
                             "Peach (locked rows, orange) + Teal (selected row, yellow) HSV masks"),
                     step_time)

    # Board detection
    board = _cc_detect(img)
    if board is None:
        print("  ERROR: board not found"); return None

    out = img.copy()
    colors = {"locked": (0, 80, 255), "selected": (255, 200, 0), "empty": (0, 255, 120)}
    for row in board.rows:
        col = colors.get(row.row_type, (180, 180, 180))
        cv2.rectangle(out, (row.x, row.y), (row.x + row.w, row.y + row.h), col, 2)
        cv2.putText(out, row.row_type, (row.x + 5, row.y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1, cv2.LINE_AA)
    if board.clue_region:
        cx2, cy2, cw, ch = board.clue_region
        cv2.rectangle(out, (cx2, cy2), (cx2 + cw, cy2 + ch), (255, 230, 0), 2)
        cv2.putText(out, "clue  (OCR → Groq LLM)", (cx2 + 5, cy2 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 230, 0), 1, cv2.LINE_AA)
    cv2.putText(out, f"word_length={board.word_length}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 220), 2, cv2.LINE_AA)
    clue = read_clue(img, board) or ""
    if clue:
        cv2.putText(out, f"OCR: {clue[:65]}", (10, out.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 230, 0), 1, cv2.LINE_AA)
    rec.write_static(_label(out,
                             f"{len(board.rows)} rows detected  +  word-ladder LLM (Groq)"), step_time)
    return board


# ─── Per-game demo functions ──────────────────────────────────────────────────

def _demo_patches(rec: GameRecorder, step_time: float) -> bool:
    img = screen.capture()
    out = _patches_steps(img, step_time, rec)
    if out is None:
        return False
    board, result = out
    rec.start_live()
    _p_exec(board.grid, board.seeds, result, move_delay=0.08, countdown=1)
    time.sleep(3)
    rec.stop_live()
    return True


def _demo_zip(rec: GameRecorder, step_time: float) -> bool:
    img = screen.capture()
    out = _zip_steps(img, step_time, rec)
    if out is None:
        return False
    grid, cells, h_walls, v_walls, path = out
    rec.start_live()
    _z_exec(grid, path, move_delay=0.06, countdown=1)
    time.sleep(3)
    rec.stop_live()
    return True


def _demo_sudoku(rec: GameRecorder, step_time: float) -> bool:
    img = screen.capture()
    out = _sudoku_steps(img, step_time, rec)
    if out is None:
        return False
    grid, original, result = out
    rec.start_live()
    _s_exec(grid, original, result, cell_delay=0.03, countdown=1)
    time.sleep(3)
    rec.stop_live()
    return True


def _demo_tango(rec: GameRecorder, step_time: float) -> bool:
    img = screen.capture()
    out = _tango_steps(img, step_time, rec)
    if out is None:
        return False
    grid, cells, h_con, v_con, result = out
    rec.start_live()
    _t_exec(grid, copy.deepcopy(cells), result, cell_delay=0.08, countdown=1)
    time.sleep(3)
    rec.stop_live()
    return True


def _demo_queens(rec: GameRecorder, step_time: float) -> bool:
    img = screen.capture()
    out = _queens_steps(img, step_time, rec)
    if out is None:
        return False
    grid, result = out
    rec.start_live()
    _q_exec(grid, result, cell_delay=0.10, countdown=1)
    time.sleep(3)
    rec.stop_live()
    return True


def _demo_pinpoint(rec: GameRecorder, step_time: float) -> bool:
    from pinpoint.controller import (
        _click      as _pin_click,
        _type_text  as _pin_type,
        _press_enter as _pin_enter,
        _MAX_ROUNDS,
    )
    from pinpoint.solver import guess_category

    previous_guesses: list[str] = []

    for round_num in range(1, _MAX_ROUNDS + 1):
        if _aborted():
            return False

        print(f"  [Pinpoint] Round {round_num}/{_MAX_ROUNDS}")
        img = screen.capture()

        # CV visualization for this round
        clues, input_pos = _pinpoint_detect_frame(
            img, round_num, _MAX_ROUNDS, step_time, rec,
            first_round=(round_num == 1))

        if not clues:
            print("  [Pinpoint] No clues yet, waiting...")
            time.sleep(2)
            continue

        guess = guess_category(clues, previous_guesses)
        print(f"  [Pinpoint] Guess: {guess!r}")

        if input_pos is None:
            print("  [Pinpoint] Input box not found")
            return False

        # Live-record the typing + response
        rec.start_live()
        _pin_click(input_pos[0], input_pos[1])
        time.sleep(0.3)
        _pin_type(guess)
        time.sleep(0.3)
        _pin_enter()
        time.sleep(2.5)
        previous_guesses.append(guess)

        post_img = screen.capture()
        solved = find_input_box(post_img) is None
        if solved:
            time.sleep(2)   # record the win screen
        rec.stop_live()

        if solved:
            print("  [Pinpoint] Solved!")
            return True

    return True


def _crossclimb_clue_frame(img: np.ndarray, board, active_row,
                            clue: str | None, hint: str) -> np.ndarray:
    """Draw board overlay with the active row and OCR'd clue highlighted."""
    out = img.copy()
    colors = {"locked": (0, 80, 255), "selected": (255, 200, 0), "empty": (0, 255, 120)}
    for row in board.rows:
        is_active = (row.index == active_row.index)
        col = (0, 255, 255) if is_active else colors.get(row.row_type, (180, 180, 180))
        thick = 3 if is_active else 1
        cv2.rectangle(out, (row.x, row.y), (row.x + row.w, row.y + row.h), col, thick)
        cv2.putText(out, row.row_type, (row.x + 5, row.y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1, cv2.LINE_AA)
    if board.clue_region:
        cx, cy, cw, ch = board.clue_region
        cv2.rectangle(out, (cx, cy), (cx + cw, cy + ch), (255, 230, 0), 2)
        if clue:
            # Wrap long clue text over two lines
            words = clue.split()
            mid   = len(words) // 2
            line1 = " ".join(words[:mid]) if mid else clue
            line2 = " ".join(words[mid:]) if mid else ""
            cv2.putText(out, line1, (cx + 5, cy + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 230, 0), 1, cv2.LINE_AA)
            if line2:
                cv2.putText(out, line2, (cx + 5, cy + 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 230, 0), 1, cv2.LINE_AA)
    clue_str = f'"{clue}"' if clue else "(OCR failed)"
    return _label(out, f"Row {active_row.index}  OCR →  {clue_str}", hint)


def _demo_crossclimb(rec: GameRecorder, step_time: float) -> bool:
    from crossclimb.controller import (
        _click         as _cc_click,
        _type_text     as _cc_type,
        _type_answers  as _cc_type_answers,
        _reorder_rows  as _cc_reorder,
    )
    from crossclimb.solver import solve_crossclimb, solve_endpoints

    # ── 1. Initial board detection visualization ──────────────────────────────
    img = screen.capture()
    board = _crossclimb_steps(img, step_time, rec)
    if board is None:
        return False

    middle = board.middle_rows

    # ── 2. Collect clues — show OCR visualization after each row click ────────
    clues: dict[int, str] = {}
    rec.start_live()

    for row in middle:
        if _aborted():
            rec.stop_live(); return False

        _cc_click(*row.center)
        time.sleep(1.2)

        img = screen.capture()
        clue = read_clue(img, board)

        rec.stop_live()
        rec.write_static(
            _crossclimb_clue_frame(img, board, row, clue, "Tesseract OCR"),
            step_time)
        rec.start_live()

        if clue:
            clues[row.index] = clue
            print(f"  Row {row.index}: \"{clue}\"")
        else:
            print(f"  Row {row.index}: [OCR failed]")

    if not clues:
        rec.stop_live()
        print("  ERROR: no clues read")
        return False

    # ── 3. LLM solve → type answers → reorder (all live) ──────────────────��───
    ladder       = solve_crossclimb(clues, board.word_length)
    answers      = {idx: word for idx, word in ladder}
    target_order = [idx for idx, _ in ladder]
    current_order = [r.index for r in sorted(middle, key=lambda r: r.y)]

    print(f"  Ladder: {ladder}")
    _cc_type_answers(board, answers)

    if current_order != target_order:
        time.sleep(1.0)
        img2   = screen.capture()
        board2 = _cc_detect(img2) or board
        _cc_reorder(board2, target_order)
    else:
        print("  Already in correct order!")

    time.sleep(4.0)

    # ── 4. Locked rows — OCR visualization then type ───────────────────────────
    img3   = screen.capture()
    board3 = _cc_detect(img3) or board
    locked = board3.locked_rows

    if len(locked) >= 2:
        top_locked    = min(locked, key=lambda r: r.y)
        bottom_locked = max(locked, key=lambda r: r.y)

        _cc_click(*top_locked.center)
        time.sleep(1.2)

        img4 = screen.capture()
        clue = read_clue(img4, board3)

        rec.stop_live()
        rec.write_static(
            _crossclimb_clue_frame(img4, board3, top_locked, clue,
                                   "Locked-row clue  (OCR → Groq)"),
            step_time)
        rec.start_live()

        if clue:
            top_adj = ladder[0][1]
            bot_adj = ladder[-1][1]
            top_word, bot_word = solve_endpoints(
                clue, board3.word_length, top_adj, bot_adj)
            print(f"  Top: {top_word}  Bottom: {bot_word}")

            _cc_click(*top_locked.center)
            time.sleep(0.5)
            _cc_type(top_word.lower())
            time.sleep(0.5)
            _cc_click(*bottom_locked.center)
            time.sleep(0.5)
            _cc_type(bot_word.lower())
            time.sleep(0.5)

    time.sleep(3)
    rec.stop_live()
    return True


# ─── Navigation helpers ───────────────────────────────────────────────────────

def _press(vk: int, dur: float = 0.03) -> None:
    _user32.keybd_event(vk, 0, 0, 0);        time.sleep(dur)
    _user32.keybd_event(vk, 0, _KEYEVENTF_KEYUP, 0); time.sleep(0.02)


def _type_url(text: str) -> None:
    VK_SHIFT = 0x10
    for ch in text:
        vk = _user32.VkKeyScanW(ord(ch))
        k, shift = vk & 0xFF, (vk >> 8) & 1
        if shift:
            _user32.keybd_event(VK_SHIFT, 0, 0, 0); time.sleep(0.01)
        _user32.keybd_event(k, 0, 0, 0); time.sleep(0.01)
        _user32.keybd_event(k, 0, _KEYEVENTF_KEYUP, 0); time.sleep(0.01)
        if shift:
            _user32.keybd_event(VK_SHIFT, 0, _KEYEVENTF_KEYUP, 0); time.sleep(0.01)


def _navigate(url: str, load_wait: float) -> None:
    _user32.keybd_event(_VK_CONTROL, 0, 0, 0); time.sleep(0.05)
    _press(0x4C)
    _user32.keybd_event(_VK_CONTROL, 0, _KEYEVENTF_KEYUP, 0); time.sleep(0.3)
    _type_url(url); time.sleep(0.1)
    _press(_VK_RETURN)
    print(f"  Waiting {load_wait:.0f}s for page load...")
    time.sleep(load_wait)


def _park_mouse() -> None:
    sw = _user32.GetSystemMetrics(0)
    sh = _user32.GetSystemMetrics(1)
    _user32.mouse_event(
        _MOUSEEVENTF_MOVE | _MOUSEEVENTF_ABSOLUTE,
        int((sw - 10) * 65535 / sw), int((sh - 10) * 65535 / sh), 0, 0)


def _aborted() -> bool:
    pt = ctypes.wintypes.POINT()
    _user32.GetCursorPos(ctypes.byref(pt))
    return pt.x <= 5 and pt.y <= 5


# ─── Game table ──────────────────────────────────────────────────────────────

_GAMES: dict[str, tuple[str, str, callable]] = {
    "pinpoint":   ("https://www.linkedin.com/games/pinpoint/",    "Pinpoint",    _demo_pinpoint),
    "patches":    ("https://www.linkedin.com/games/patches/",     "Patches",     _demo_patches),
    "zip":        ("https://www.linkedin.com/games/zip/",         "Zip",         _demo_zip),
    "sudoku":     ("https://www.linkedin.com/games/mini-sudoku/", "Mini Sudoku", _demo_sudoku),
    "tango":      ("https://www.linkedin.com/games/tango/",       "Tango",       _demo_tango),
    "queens":     ("https://www.linkedin.com/games/queens/",      "Queens",      _demo_queens),
    "crossclimb": ("https://www.linkedin.com/games/crossclimb/",  "Crossclimb",  _demo_crossclimb),
}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record a game-area demo video of all LinkedIn games with full CV overlays.")
    parser.add_argument("--output",    default="demo.mp4")
    parser.add_argument("--load-wait", type=float, default=4.0)
    parser.add_argument("--countdown", type=int,   default=3)
    parser.add_argument("--step-time", type=float, default=1.0,
                        help="Seconds per CV analysis frame (default: 2.0)")
    parser.add_argument("--game",      default=None, choices=list(_GAMES.keys()))
    parser.add_argument("--no-record", action="store_true",
                        help="Detect & solve but skip video writing")
    args = parser.parse_args()

    queue = [(k, *v) for k, v in _GAMES.items()] if not args.game \
            else [(args.game, *_GAMES[args.game])]

    print("=== LinkedIn Game Solver — Video Demo ===")
    print(f"Games: {[name for _, _, name, _ in queue]}")
    if not args.no_record:
        print(f"Output: {args.output}")
    print("Move mouse to top-left corner (≤5 px) to abort.\n")

    for i in range(args.countdown, 0, -1):
        if _aborted():
            print("Aborted."); return
        print(f"Starting in {i}...")
        time.sleep(1)

    results: dict[str, str] = {}

    for idx, (key, url, name, demo_fn) in enumerate(queue, 1):
        if _aborted():
            print("\nAborted."); break

        print(f"\n[{idx}/{len(queue)}] {name.upper()}")
        _navigate(url, args.load_wait)

        if _aborted():
            print("\nAborted."); break

        _park_mouse()
        screen.reset_game_region()
        screen.init_game_region()

        if args.no_record:
            rec = _NullRecorder()
        else:
            # Each game gets its own file (game region dimensions can differ)
            out_path = args.output if len(queue) == 1 else \
                       args.output.replace(".mp4", f"_{key}.mp4")
            rec = GameRecorder(out_path)
            rec.init()

        try:
            ok = demo_fn(rec, args.step_time)
            results[name] = "OK" if ok else "FAILED"
        except Exception as exc:
            print(f"  ERROR: {exc}")
            results[name] = "ERROR"
        finally:
            rec.finish()
            rec = None

        time.sleep(2)

    print("\n=== Results ===")
    for _, _, name, _ in queue:
        status = results.get(name, "SKIPPED")
        print(f"  [{'OK' if status=='OK' else '!!'}] {name:14s} {status}")


if __name__ == "__main__":
    main()
