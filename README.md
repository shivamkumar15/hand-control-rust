# Air Mouse — Rust controller

Fast native controller for hand gesture laptop control.

## Architecture

- **Python `tracker.py`** — only runs the webcam and MediaPipe hand tracking. It streams one JSON line per frame with the 21 hand landmarks. No control actions happen in Python.
- **Python `voice.py`** — optional offline voice command listener (Vosk). Streams recognized utterances as JSON lines; the Rust side decides what they mean.
- **Rust `air-mouse`** — reads the landmark stream, recognizes gestures, and drives the mouse/keyboard/volume with very low latency using a native uinput virtual device. It also spawns `voice.py` and reacts to voice commands.


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
./run.sh --no-voice           # disable the voice command listener
```

## Gestures

| Gesture | Action |
|--------|--------|
| Point index finger | Move cursor — **whole camera view maps to the whole screen** |
| Pinch thumb + index | Left click |
| Pinch thumb + middle | Right click |
| Point index + middle, move up/down | Scroll (fingers down = page down) |
| Open palm, swipe left/right | Switch workspace — swipe right = next, left = previous |
| Open palm, quick flip up / down | Volume up / down |
| **Raised** closed fist, hold ~0.4 s | Close current tab (`Ctrl+W`) — fires once per fist |
| Clap twice, then lower right hand | Shut down the system |
| Say **"shutdown now"** | Shut down the system (offline voice recognition, fires on the partial so it triggers fast; 10 s cooldown) |

### Built-in intelligence

- **No dead zones** — the full camera frame (small margin) controls the cursor, so resting your hand low in view still reaches the bottom of the screen.
- **Adaptive smoothing** — slow hand = stable cursor, fast hand = instant tracking. The cursor snaps (no glide) each time you start pointing.
- **Safe fist** — only a *raised* fist held ~0.4 s closes a tab, once per gesture. A resting or transitioning hand can never fire it.
- **No phantom clicks** — pinch needs the finger actually extended; a curled resting hand is inert.
- **Hysteresis finger reading** — finger angles are measured in 3D and use two thresholds, so slightly bent fingers still register and borderline fingers never flicker between gestures.
- **Debounced modes** — a pose must hold for 3 consecutive frames before the mode changes, and palm gestures wait ~200 ms after engaging, so stray poses mid-motion can't fire clicks, volume or workspace switches.
- **Displacement-based swipe** — workspace switching tracks how far your palm actually traveled, in the direction you swiped. Slow drift never fires, and the anchor decays when you stop, so stale distance can't fire later.
- **Velocity-gated palm** — merely holding or rotating an open palm does nothing; you must genuinely swipe (workspace) or flip (volume) along the matching axis.

## Tuning

All knobs are constants at the top of `src/main.rs`:

| Symptom | Knob |
|---------|------|
| Cursor too twitchy | raise `SMOOTH_RANGE_PX` / lower `SMOOTH_MIN` |
| Cursor laggy | lower `SMOOTH_RANGE_PX` |
| Clicks fire too easily | raise `PINCH_ON` / `PINCH_PALM_REL_ON` (and `PINCH_OFF` / `PINCH_PALM_REL_OFF`) |
| Clicks hard to trigger | lower `PINCH_ON` / `PINCH_PALM_REL_ON`, or lower `PINCH_MIN_ANGLE_DEG` |
| Pinch needs finger unnaturally straight | lower `PINCH_MIN_ANGLE_DEG` (default 90 = "not folded") |
| Scroll too fine/coarse | `SCROLL_TICK` |
| Workspace swipe too eager | raise `SWIPE_DIST` or `SWIPE_VEL` |
| Workspace swipe needs too much travel | lower `SWIPE_DIST` |
| Gentle swipes ignored | lower `SWIPE_VEL` |
| Volume flips too eager | raise `FLIP_VEL` |
| Volume step size | `VOLUME_STEP` |
| Fist too strict/loose | `FIST_HOLD`, `FIST_MAX_WRIST_Y` |
| Gestures need perfectly straight fingers | lower `FINGER_UP_DEG` (and `FINGER_DOWN_DEG`) |
| Finger states flicker between gestures | widen the `FINGER_UP_DEG` / `FINGER_DOWN_DEG` gap |
| Modes react too slowly | lower `MODE_CONFIRM_FRAMES` |
| Stray actions right after opening the palm | raise `PALM_SETTLE` |

> **Warning:** The shutdown gesture and the "shutdown now" voice command are active in non-dry-run mode and call `systemctl poweroff` with a 2-second delay. Make sure you can cancel it (`Ctrl+C` in the terminal) while testing.

## Voice commands

Offline speech recognition via [Vosk](https://alphacephei.com/vosk/) — no cloud, no API key. Setup (already done in this repo):

```bash
/home/dranzer/hand-control/.venv/bin/pip install vosk
curl -L -o /tmp/vosk.zip https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip /tmp/vosk.zip -d models/ && mv models/vosk-model-small-en-us-0.15 models/vosk-model-small-en-us
```

- The listener starts automatically with `./run.sh` (unless `--no-voice`). If no mic is available it just warns and continues with gestures only.
- To pick a specific microphone: `python voice.py --list-devices`, then set the device in `voice.py` or pass `--device N` when testing it standalone.
- Matching is done in Rust (`match_voice_command` in `src/main.rs`): lowercase, punctuation-insensitive, both partial and final hypotheses trigger, with a 10 s cooldown against double-fires.

## Build yourself

```bash
cargo build --release
cargo test --release   # gesture-engine unit tests
```

The release binary is at `./target/release/air-mouse`.

## Notes

- You need the Python venv from `~/hand-control/.venv` because it contains MediaPipe and OpenCV (plus Vosk for voice commands).
- Volume changes use `pactl`. If you use PipeWire, make sure the PulseAudio compatibility socket is running.
- The uinput virtual device needs write access to `/dev/uinput` (usually via the `input` group).
- On Hyprland, workspace switching talks to `hyprctl` directly; sway/xdotool fallbacks are included.
- A fully native Rust inference pipeline is possible but would require re-implementing MediaPipe’s model decoder (anchors, NMS, rotation, landmarks). This hybrid is the practical fast path.
