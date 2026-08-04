# Air Mouse — Rust controller

Fast native controller for hand gesture laptop control.

## Architecture

- **Python `tracker.py`** — only runs the webcam and MediaPipe hand tracking. It streams one JSON line per frame with the 21 hand landmarks. No control actions happen in Python.
- **Rust `air-mouse`** — reads the landmark stream, recognizes gestures, and drives the mouse/keyboard/volume with very low latency using `enigo`.

The heavy AI inference is still MediaPipe’s C++ engine; Python is just a thin wrapper. Moving the control loop to Rust removes Python GIL/input lag and makes the system feel much snappier.

## Run it

```bash
cd ~/hand-control-rust
./run.sh
```

Press **Ctrl+C** to stop.

## Options

```bash
./run.sh --dry-run          # recognize gestures, print them, but don't act
./run.sh --camera 1         # use a different webcam
```

## Gestures

| Gesture | Action |
|--------|--------|
| Point index finger | Move cursor |
| Pinch thumb + index | Left click |
| Pinch thumb + middle | Right click |
| Open palm, flip up / down | Increase / decrease volume |
| Open palm, swipe left/right | Switch browser/app tabs |
| Closed fist | Close current tab (`Ctrl+W`) |
| Flip hand left / right | Switch workspace / virtual desktop |
| Clap twice, then lower right hand | Shut down the system |

> **Warning:** The shutdown gesture is active in non-dry-run mode and calls `systemctl poweroff` with a 2-second delay. Make sure you can cancel it (`Ctrl+C` in the terminal) while testing.

## Build yourself

```bash
cargo build --release
```

The release binary is at `./target/release/air-mouse`.

## Notes

- You need the Python venv from `~/hand-control/.venv` (created in the previous step) because it contains MediaPipe and OpenCV.
- Volume changes use `pactl`. If you use PipeWire, make sure the PulseAudio compatibility socket is running.
- On Wayland, Enigo/X11 mouse and keyboard injection may need accessibility/remote-desktop permissions.
- If the cursor feels jumpy from hand shake, lower `CURSOR_ALPHA` in `src/main.rs`. If it feels too slow or too fast, change `CURSOR_GAIN`.
- If volume flips are hard to trigger (or fire too easily), adjust `FLIP_VEL` in `src/main.rs` (lower = more sensitive). To change how much each flip changes the volume, edit `VOLUME_STEP` (percent per flip); to allow faster repeated flips, lower `VOLUME_FLIP_COOLDOWN`.
- A fully native Rust inference pipeline is possible but would require re-implementing MediaPipe’s model decoder (anchors, NMS, rotation, landmarks). This hybrid is the practical fast path.

# hand-control-rust
