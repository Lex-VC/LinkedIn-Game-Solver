# LinkedIn Game Solver

A Windows bot that uses computer vision and automated mouse/keyboard input to play all seven daily LinkedIn games. Open a game in your browser, run the solver, and watch it play.

## Demo

<!-- Upload your MP4 to GitHub by editing this file on github.com and dragging the video in, then paste the resulting URL below -->
https://github.com/user-attachments/assets/YOUR_VIDEO_ID_HERE

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

### Quick start

1. Open LinkedIn Games in your browser and make sure the window is visible (not minimised).
2. Set your `GROQ_API_KEY` environment variable if you want Pinpoint or Crossclimb solved.
3. Run the dispatcher — it will handle the rest:

```
python dispatcher.py
```

Switch focus to your browser before the countdown ends. The dispatcher uses Ctrl+L to navigate to each game URL automatically.

### Play all games (dispatcher)

```
python dispatcher.py [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--countdown` | 5 | Seconds before starting — time to switch to the browser |
| `--pause` | 5 | Seconds to wait between games |
| `--load-wait` | 6 | Seconds to wait for a page to finish loading |
| `--game-countdown` | 3 | Countdown passed to each individual game controller |

If a game fails or times out it is skipped and the dispatcher moves on to the next one.

### Play a single game

Navigate to the game in your browser first, then run its controller:

```
cd patches     && python controller.py
cd zip         && python controller.py
cd sudoku      && python controller.py
cd tango       && python controller.py
cd queens      && python controller.py
cd pinpoint    && python controller.py
cd crossclimb  && python controller.py
```

Common flags accepted by most controllers:

| Flag | Description |
|------|-------------|
| `--countdown N` | Seconds before the solver starts (default 5) |
| `--delay N` | Seconds between individual clicks/actions |
| `--debug` | Show the CV detection overlay — useful for diagnosing misdetections |

### Debugging detection

If the solver misreads the board, run with `--debug` to see what OpenCV is detecting:

```
cd queens && python controller.py --debug
```

A window will pop up overlaying the detected grid, colours, or OCR regions on the captured screenshot. Press any key to proceed past each debug frame.

### Safety

Move your mouse to the **top-left corner** of the screen (within 5 px) at any time to immediately abort execution.
