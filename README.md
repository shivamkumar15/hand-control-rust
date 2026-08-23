# Air Mouse — Rust controller

Fast native controller for hand gesture laptop control.

## Architecture

- **Python `tracker.py`** — only runs the webcam and MediaPipe hand tracking. It streams one JSON line per frame with the 21 hand landmarks. No control actions happen in Python.
- **Rust `air-mouse`** — reads the landmark stream, recognizes gestures, and drives the mouse/keyboard/volume with very low latency using a native uinput virtual device.

The heavy AI inference is still MediaPipe’s C++ engine; Python is just a thin wrapper. Moving the control loop to Rust removes Python GIL/input lag and makes the system feel much snappier.

## Run it

```bash
cd ~/hand-control-rust
./run.sh
```

Press **Ctrl+C** to stop.

## Options

```bash
./run.sh --dry-run            # recognize gestures, print them, but don't act
./run.sh --debug              # live finger/state telemetry on stderr
./run.sh --preview            # show the camera window so you can see tracking
./run.sh --camera 1           # use a different webcam (auto-scans 0-4 if it fails)
```

## Gestures

| Gesture | Action |
|--------|--------|
| Point index finger | Move cursor — **whole camera view maps to the whole screen** |
| Pinch thumb + index | Left click |
| Pinch thumb + middle | Right click |
| Point index + middle, move up/down | Scroll (fingers down = page down) |
| Open palm, wave/tilt left-right (like saying "no") | Switch workspace — lean right = next, left = previous |
| Open palm, quick flip up / down | Volume up / down |
| **Raised** closed fist, hold ~0.4 s | Close current tab (`Ctrl+W`) — fires once per fist |
| Clap twice, then lower right hand | Shut down the system |

### Built-in intelligence

- **No dead zones** — the full camera frame (small margin) controls the cursor, so resting your hand low in view still reaches the bottom of the screen.
- **Adaptive smoothing** — slow hand = stable cursor, fast hand = instant tracking. The cursor snaps (no glide) each time you start pointing.
- **Safe fist** — only a *raised* fist held ~0.4 s closes a tab, once per gesture. A resting or transitioning hand can never fire it.
- **No phantom clicks** — pinch needs the finger actually extended; a curled resting hand is inert.
- **Velocity-gated palm** — merely holding or rotating an open palm does nothing; you must genuinely wave (workspace) or flip (volume) along the matching axis.

## Tuning

All knobs are constants at the top of `src/main.rs`:

| Symptom | Knob |
|---------|------|
| Cursor too twitchy | raise `SMOOTH_RANGE_PX` / lower `SMOOTH_MIN` |
| Cursor laggy | lower `SMOOTH_RANGE_PX` |
| Clicks fire too easily | raise `PINCH_ON` (and `PINCH_OFF`) |
| Clicks hard to trigger | lower `PINCH_ON` |
| Scroll too fine/coarse | `SCROLL_TICK` |
| Workspace wave too eager | raise `WAVE_VEL` or `TILT_FIRE_DEG` |
| Volume flips too eager | raise `FLIP_VEL` |
| Volume step size | `VOLUME_STEP` |
| Fist too strict/loose | `FIST_HOLD`, `FIST_MAX_WRIST_Y` |

> **Warning:** The shutdown gesture is active in non-dry-run mode and calls `systemctl poweroff` with a 2-second delay. Make sure you can cancel it (`Ctrl+C` in the terminal) while testing.

## Build yourself

```bash
cargo build --release
cargo test --release   # gesture-engine unit tests
```

The release binary is at `./target/release/air-mouse`.

## Notes

- You need the Python venv from `~/hand-control/.venv` because it contains MediaPipe and OpenCV.
- Volume changes use `pactl`. If you use PipeWire, make sure the PulseAudio compatibility socket is running.
- The uinput virtual device needs write access to `/dev/uinput` (usually via the `input` group).
- On Hyprland, workspace switching talks to `hyprctl` directly; sway/xdotool fallbacks are included.
- A fully native Rust inference pipeline is possible but would require re-implementing MediaPipe’s model decoder (anchors, NMS, rotation, landmarks). This hybrid is the practical fast path.
