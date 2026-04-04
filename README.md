# LinkedIn Game Solver

A Windows bot that uses computer vision and automated mouse/keyboard input to play all seven daily LinkedIn games. Open a game in your browser, run the solver, and watch it play.

## Games

| Game | Approach | LLM needed? |
|------|----------|-------------|
| **Patches** | Detect grid & shape seeds via CV, solve with constraint propagation, click each cell | No |
| **Zip** | Detect numbered cells & walls via CV, solve Hamiltonian path with BFS, drag through cells | No |
| **Mini Sudoku** | Template-match digits via CV, solve with backtracking, click & type digits | No |
| **Tango** | Detect sun/moon symbols & constraints via CV, solve with backtracking, click cells | No |
| **Queens** | Detect colour regions via CV, solve N-queens variant with backtracking, click cells | No |
| **Pinpoint** | OCR clue words with Tesseract, guess category via Groq LLM, type answer | Yes |
| **Crossclimb** | OCR clues with Tesseract, solve crossword clues & word ladder via Groq LLM, type & drag | Yes |

Each game follows the same architecture:

- `board_vision.py` — screen capture (mss) + OpenCV detection/OCR
- `solver.py` — pure logic solver (or LLM call for Pinpoint/Crossclimb)
- `controller.py` — orchestrates detection, solving, and mouse/keyboard execution

## Requirements

- **Windows** (uses `ctypes.windll` for mouse/keyboard input)
- **Python 3.11+**
- A browser with LinkedIn Games open and visible on screen

### Python dependencies

```
pip install opencv-python numpy mss pytesseract groq
```

### External tools

- **Tesseract OCR** — required by Pinpoint and Crossclimb for reading clue text. Install from [github.com/tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract) and make sure `tesseract` is on your PATH.

### Environment variables

- **`GROQ_API_KEY`** — required by Pinpoint and Crossclimb. Get a free key at [console.groq.com/keys](https://console.groq.com/keys).

## Usage

### Play all games (dispatcher)

The dispatcher navigates to each game in your browser, runs its solver, and moves on:

```
python dispatcher.py
```

Make sure the browser is focused and visible before the countdown finishes. The dispatcher uses Ctrl+L to open the address bar and type each game URL.

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--countdown` | 5 | Seconds before starting |
| `--pause` | 5 | Seconds to wait between games |
| `--load-wait` | 6 | Seconds to wait for page load |
| `--timeout` | 120 | Max seconds per game before skipping |

If any game fails it is skipped and the dispatcher moves on.

### Play a single game

Navigate to the game in your browser, then run its controller from inside the game folder:

```
cd patches && python controller.py
cd zip && python controller.py
cd sudoku && python controller.py
cd tango && python controller.py
cd queens && python controller.py
cd pinpoint && python controller.py
cd crossclimb && python controller.py
```

Most controllers accept `--countdown`, `--delay`, and `--debug` flags. Use `--debug` to show the CV detection overlay.

### Safety

Move your mouse to the **top-left corner** of the screen (within 5 px) at any time to abort execution.
