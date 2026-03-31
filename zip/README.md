# Zip Solver

Automatically detects, solves, and executes the [LinkedIn Zip](https://www.linkedin.com/games/zip) puzzle using screen capture, computer vision, and mouse automation.

## How it works

1. **Captures** the screen and locates the game grid
2. **Detects** numbered cells via template matching and walls via pixel analysis
3. **Solves** the puzzle using backtracking with connectivity pruning
4. **Executes** the solution by clicking and dragging through the path

## Dependencies

Python 3.11+ required.

```bash
pip install mss numpy opencv-python
```

> Mouse control uses the Windows `ctypes` API directly — no additional packages needed.

## Usage

```bash
cd zip
python controller.py
```

You have 3 seconds after the script starts to click on the browser window before it executes.

### Options

| Flag | Default | Description |
|---|---|---|
| `--delay` | `0.02` | Seconds between each cell move. Increase if the game misses steps. |
| `--countdown` | `3` | Seconds before executing, giving you time to focus the browser. |
| `--debug` | off | Shows a window with the detected grid, numbers, and walls before solving. |

### Examples

```bash
python controller.py --delay 0.05
python controller.py --countdown 5
python controller.py --debug
```

## Notes

- Move the mouse to the **top-left corner** at any time to abort execution
