use std::collections::VecDeque;
use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use enigo::{Axis, Button, Coordinate, Direction, Enigo, Key, Keyboard, Mouse, Settings};
use evdev::Key as EvKey;
use serde::Deserialize;

mod input;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
const MIN_HAND_SPAN: f32 = 0.06; // reject spurious tiny detections

// Pinch clicks (thumb + index / thumb + middle). Fire when tips come close,
// re-arm only after they separate again -> no machine-gun clicking.
const PINCH_ON: f32 = 0.045;
const PINCH_OFF: f32 = 0.08;
const PINCH_PALM_REL: f32 = 0.30; // also relative to palm size
const CLICK_COOLDOWN: Duration = Duration::from_millis(180);

// Cursor: the full camera view (minus a small margin) maps to the whole
// screen. No saturating gain — every hand position is reachable.
const MARGIN: f32 = 0.08;
const CURSOR_JITTER_PX: f32 = 1.5; // ignore sub-pixel shake
const SMOOTH_MIN: f32 = 0.10; // smoothing for slow movement
const SMOOTH_MAX: f32 = 1.0; // fast movement snaps instantly
const SMOOTH_RANGE_PX: f32 = 120.0; // distance over which smoothing ramps up

// Two-finger scroll: displacement-based (frame-rate independent), needs the
// gesture held briefly so transitions don't emit stray ticks.
const SCROLL_TICK: f32 = 0.035; // fingertip travel per wheel tick (normalized)
const SCROLL_SETTLE: Duration = Duration::from_millis(120);
const SCROLL_MAX_TICKS: i32 = 3;

// Open-palm gestures are velocity-gated so merely holding/rotating the hand
// does nothing until you actually wave or flip it.
const WAVE_VEL: f32 = 0.70; // normalized units / second
const FLIP_VEL: f32 = 0.85;
const DOMINANCE: f32 = 1.3; // axis must beat the other by this factor
const TILT_FIRE_DEG: f32 = 24.0;
const TILT_ARM_DEG: f32 = 14.0;
const TILT_ALPHA: f32 = 0.45;
const WORKSPACE_COOLDOWN: Duration = Duration::from_millis(400);
const VOLUME_FLIP_COOLDOWN: Duration = Duration::from_millis(500);
const VOLUME_STEP: u32 = 5;

// Fist: must be HELD to fire, fires once, and re-arms only after the hand
// opens again. Transitions through a half-closed hand never close tabs.
const FIST_HOLD: Duration = Duration::from_millis(400);
const FIST_REARM: Duration = Duration::from_millis(400);
const FIST_MAX_WRIST_Y: f32 = 0.60; // fist must be raised; low/resting hands never fire

// Finger extension uses hysteresis: the angle must rise above FINGER_UP_DEG to
// count as extended, then fall below FINGER_DOWN_DEG to count as folded. In
// between, the previous state is kept, so borderline fingers stop flickering.
const FINGER_UP_DEG: f32 = 140.0;
const FINGER_DOWN_DEG: f32 = 122.0;

// A candidate mode must be seen this many consecutive frames before it
// replaces the current one. Kills single-frame misreads that used to fire
// stray volume flips / workspace switches.
const MODE_CONFIRM_FRAMES: u32 = 3;

// After the palm gesture engages, wait this long before waves/flips can fire,
// so leftover velocity from pointing doesn't trigger them instantly.
const PALM_SETTLE: Duration = Duration::from_millis(200);

const CLAP_THRESHOLD: f32 = 0.12;
const CLAP_TIMEOUT: Duration = Duration::from_millis(1500);
const CLAP_COOLDOWN: Duration = Duration::from_millis(300);
const AFTER_CLAP_WINDOW: Duration = Duration::from_millis(2500);
const SHUTDOWN_DOWN_THRESHOLD: f32 = 0.22;
const SHUTDOWN_COOLDOWN: Duration = Duration::from_secs(5);

// ---------------------------------------------------------------------------
// Landmark data from Python tracker
// ---------------------------------------------------------------------------
#[derive(Deserialize, Debug)]
struct Hand {
    landmarks: Vec<[f32; 3]>,
    handedness: Option<String>,
}

#[derive(Deserialize, Debug)]
struct Frame {
    hands: Vec<Hand>,
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------
fn dist(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

fn dist2d(a: &[f32; 2], b: &[f32; 2]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

fn hand_span(lm: &[[f32; 3]]) -> f32 {
    let mut min_x = f32::MAX;
    let mut max_x = f32::MIN;
    let mut min_y = f32::MAX;
    let mut max_y = f32::MIN;
    for p in lm {
        min_x = min_x.min(p[0]);
        max_x = max_x.max(p[0]);
        min_y = min_y.min(p[1]);
        max_y = max_y.max(p[1]);
    }
    (max_x - min_x).max(max_y - min_y)
}

/// Angle at joint `b` of the triangle a-b-c, in degrees. Straight = ~180.
/// Uses all three coordinates, so a finger pointing toward the camera
/// (foreshortened in 2D) is still measured correctly.
fn joint_angle(a: &[f32; 3], b: &[f32; 3], c: &[f32; 3]) -> f32 {
    let v1 = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let v2 = [c[0] - b[0], c[1] - b[1], c[2] - b[2]];
    let m1 = (v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]).sqrt();
    let m2 = (v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]).sqrt();
    if m1 < 1e-6 || m2 < 1e-6 {
        return 180.0;
    }
    ((v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]) / (m1 * m2))
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees()
}

/// PIP joint angle of each finger (thumb first), in degrees.
fn finger_angles(lm: &[[f32; 3]]) -> [f32; 5] {
    [
        joint_angle(&lm[2], &lm[3], &lm[4]),   // thumb
        joint_angle(&lm[5], &lm[6], &lm[8]),   // index
        joint_angle(&lm[9], &lm[10], &lm[12]), // middle
        joint_angle(&lm[13], &lm[14], &lm[16]), // ring
        joint_angle(&lm[17], &lm[18], &lm[20]), // pinky
    ]
}

/// Hysteresis step for one finger's extended state.
fn hysteresis_step(prev: bool, angle: f32) -> bool {
    if angle >= FINGER_UP_DEG {
        true
    } else if angle <= FINGER_DOWN_DEG {
        false
    } else {
        prev
    }
}

fn palm_centroid(lm: &[[f32; 3]]) -> [f32; 2] {
    let mut sum = [0.0; 2];
    for i in [0, 5, 17] {
        sum[0] += lm[i][0];
        sum[1] += lm[i][1];
    }
    [sum[0] / 3.0, sum[1] / 3.0]
}

/// Tilt of the fingers about the wrist in degrees. Positive = leaning right.
fn hand_tilt_deg(lm: &[[f32; 3]]) -> f32 {
    let dx = lm[9][0] - lm[0][0];
    let dy = lm[9][1] - lm[0][1]; // y grows downward
    dx.atan2(-dy).to_degrees()
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Mode {
    Idle,
    Cursor,
    Scroll,
    Palm,
}

impl Mode {
    fn label(self) -> &'static str {
        match self {
            Mode::Idle => "idle",
            Mode::Cursor => "cursor",
            Mode::Scroll => "scroll",
            Mode::Palm => "palm",
        }
    }
}

struct Sample {
    t: Instant,
    p: [f32; 2],
}

// ---------------------------------------------------------------------------
// Volume control via pactl (relative steps)
// ---------------------------------------------------------------------------
fn adjust_volume(up: bool, step: u32) -> Result<()> {
    let sign = if up { "+" } else { "-" };
    Command::new("pactl")
        .args(["set-sink-volume", "@DEFAULT_SINK@", &format!("{}{}%", sign, step)])
        .status()
        .context("failed to run pactl")?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Main controller
// ---------------------------------------------------------------------------
struct Controller {
    enigo: Enigo,
    screen_size: (i32, i32),
    virtual_input: Option<input::VirtualInput>,
    dry_run: bool,
    debug: bool,

    mode: Mode,
    mode_candidate: Option<Mode>,
    cand_frames: u32,
    finger_up: [bool; 5],
    last_status: String,

    cursor_ema: Option<[f32; 2]>,

    pinch_left_armed: bool,
    pinch_right_armed: bool,
    last_click: Instant,

    hist: VecDeque<Sample>,

    scroll_entered: Option<Instant>,
    scroll_last_y: Option<f32>,
    scroll_acc: f32,

    fist_since: Option<Instant>,
    fist_fired: bool,
    last_fist_fire: Instant,

    tilt_ema: Option<f32>,
    tilt_armed: bool,
    vol_armed: bool,
    palm_since: Option<Instant>,
    last_workspace_switch: Instant,
    last_volume_flip: Instant,

    clap_count: u32,
    last_clap_time: Instant,
    clap_together: bool,
    shutdown_window_until: Option<Instant>,
    right_hand_start_y: Option<f32>,
    last_shutdown: Option<Instant>,
}

impl Controller {
    fn new(dry_run: bool, debug: bool) -> Result<Self> {
        let enigo = Enigo::new(&Settings::default()).context("Enigo init failed")?;
        let screen_size = enigo.main_display().context("screen size failed")?;

        let virtual_input = if dry_run {
            None
        } else {
            match input::VirtualInput::new("air-mouse virtual input", screen_size.0, screen_size.1) {
                Ok(vi) => {
                    println!("Using native uinput virtual device (works on Wayland and X11).");
                    Some(vi)
                }
                Err(e) => {
                    eprintln!("Warning: could not create uinput device, falling back to Enigo: {e:?}");
                    None
                }
            }
        };

        Ok(Self {
            enigo,
            screen_size,
            virtual_input,
            dry_run,
            debug,
            mode: Mode::Idle,
            mode_candidate: None,
            cand_frames: 0,
            finger_up: [false; 5],
            last_status: String::new(),
            cursor_ema: None,
            pinch_left_armed: true,
            pinch_right_armed: true,
            last_click: Instant::now(),
            hist: VecDeque::with_capacity(64),
            scroll_entered: None,
            scroll_last_y: None,
            scroll_acc: 0.0,
            fist_since: None,
            fist_fired: false,
            last_fist_fire: Instant::now(),
            tilt_ema: None,
            tilt_armed: true,
            vol_armed: true,
            palm_since: None,
            last_workspace_switch: Instant::now(),
            last_volume_flip: Instant::now(),
            clap_count: 0,
            last_clap_time: Instant::now(),
            clap_together: false,
            shutdown_window_until: None,
            right_hand_start_y: None,
            last_shutdown: None,
        })
    }

    fn status(&mut self, s: &str) {
        if self.last_status != s {
            self.last_status = s.to_string();
            println!("{s}");
        }
    }

    /// Called on frames where no usable hand is present.
    fn on_hand_lost(&mut self) {
        self.cursor_ema = None;
        self.hist.clear();
        self.tilt_ema = None;
        self.tilt_armed = true;
        self.vol_armed = true;
        self.scroll_last_y = None;
        self.scroll_acc = 0.0;
        self.fist_since = None;
        self.finger_up = [false; 5];
        self.mode_candidate = None;
        self.cand_frames = 0;
        self.palm_since = None;
        self.set_mode(Mode::Idle);
    }

    fn set_mode(&mut self, m: Mode) {
        if self.mode != m {
            self.mode = m;
            match m {
                Mode::Cursor => self.cursor_ema = None, // snap on entry, no glide
                Mode::Scroll => self.scroll_entered = Some(Instant::now()),
                // Restart motion history so pointing speed doesn't leak into
                // palm gesture detection.
                Mode::Palm => {
                    self.palm_since = Some(Instant::now());
                    self.hist.clear();
                    self.tilt_ema = None;
                }
                _ => {}
            }
            self.status(&format!("[MODE] {}", m.label()));
        }
    }

    /// Update per-finger extended states with hysteresis.
    fn update_fingers(&mut self, lm: &[[f32; 3]]) -> [bool; 5] {
        let angles = finger_angles(lm);
        for (i, &a) in angles.iter().enumerate() {
            self.finger_up[i] = hysteresis_step(self.finger_up[i], a);
        }
        self.finger_up
    }

    fn process(&mut self, lm: &[[f32; 3]]) -> Result<()> {
        let span = hand_span(lm);
        if span < MIN_HAND_SPAN {
            return Ok(());
        }

        let now = Instant::now();
        let fingers = self.update_fingers(lm);
        let four = [fingers[1], fingers[2], fingers[3], fingers[4]];
        let ext_count = four.iter().filter(|&&x| x).count();
        let fist_now = ext_count == 0 && lm[0][1] < FIST_MAX_WRIST_Y;
        let palm_size = dist(&lm[0], &lm[9]);

        if self.debug {
            eprintln!(
                "[STATE] span={span:.3} palm={palm_size:.3} fingers={fingers:?} mode={:?}",
                self.mode
            );
        }

        // Motion history for velocity estimates (rolling ~350ms window).
        self.hist.push_back(Sample { t: now, p: palm_centroid(lm) });
        while let Some(s) = self.hist.front() {
            if now.duration_since(s.t) > Duration::from_millis(350) {
                self.hist.pop_front();
            } else {
                break;
            }
        }
        let (vx, vy) = self.velocity();

        // ---- Fist: hold-to-confirm, single shot --------------------------
        if fist_now {
            match self.fist_since {
                None => self.fist_since = Some(now),
                Some(t0) => {
                    if !self.fist_fired && now.duration_since(t0) >= FIST_HOLD {
                        self.close_tab()?;
                        self.fist_fired = true;
                        self.last_fist_fire = now;
                    }
                }
            }
        } else {
            self.fist_since = None;
            if self.fist_fired && now.duration_since(self.last_fist_fire) > FIST_REARM {
                self.fist_fired = false;
            }
        }

        // ---- Pinch clicks (suppressed while the hand is closing) ---------
        if !fist_now {
            let d_left = dist(&lm[4], &lm[8]);
            let d_right = dist(&lm[4], &lm[12]);
            let rel = palm_size * PINCH_PALM_REL;
            let can_click = now.duration_since(self.last_click) > CLICK_COOLDOWN;

            // The triggering finger must be extended, otherwise a curled
            // resting hand brings tips close together and fires phantom
            // clicks.
            if self.pinch_left_armed && can_click && fingers[1] && d_left < PINCH_ON.min(rel) {
                self.do_click(Button::Left, "left click", now)?;
                self.pinch_left_armed = false;
            } else if d_left > PINCH_OFF {
                self.pinch_left_armed = true;
            }

            if self.pinch_right_armed && can_click && fingers[2] && d_right < PINCH_ON.min(rel) {
                self.do_click(Button::Right, "right click", now)?;
                self.pinch_right_armed = false;
            } else if d_right > PINCH_OFF {
                self.pinch_right_armed = true;
            }
        }

        // ---- Mode selection (debounced) ----------------------------------
        let new_mode = if fist_now || self.fist_fired {
            Mode::Idle
        } else if fingers[1] && fingers[2] && !fingers[3] && !fingers[4] {
            Mode::Scroll
        } else if fingers[1] && ext_count == 1 {
            Mode::Cursor
        } else if ext_count >= 3 {
            Mode::Palm
        } else {
            Mode::Idle
        };
        // The candidate must repeat for MODE_CONFIRM_FRAMES consecutive
        // frames before it replaces the current mode.
        if new_mode == self.mode {
            self.mode_candidate = None;
            self.cand_frames = 0;
        } else if self.mode_candidate == Some(new_mode) {
            self.cand_frames += 1;
            if self.cand_frames >= MODE_CONFIRM_FRAMES {
                self.set_mode(new_mode);
                self.mode_candidate = None;
                self.cand_frames = 0;
            }
        } else {
            self.mode_candidate = Some(new_mode);
            self.cand_frames = 1;
        }

        match self.mode {
            Mode::Scroll => self.handle_scroll(lm, now)?,
            Mode::Cursor => self.move_cursor(lm)?,
            Mode::Palm => self.handle_palm(lm, vx, vy, now)?,
            Mode::Idle => {
                self.scroll_acc = 0.0;
                self.scroll_last_y = None;
                self.tilt_ema = None;
            }
        }

        Ok(())
    }

    /// Mean velocity over the history window (normalized units / second).
    fn velocity(&self) -> (f32, f32) {
        if self.hist.len() < 2 {
            return (0.0, 0.0);
        }
        let first = self.hist.front().unwrap();
        let last = self.hist.back().unwrap();
        let dt = last.t.duration_since(first.t).as_secs_f32();
        if dt < 0.05 {
            return (0.0, 0.0);
        }
        (
            (last.p[0] - first.p[0]) / dt,
            (last.p[1] - first.p[1]) / dt,
        )
    }

    fn do_click(&mut self, button: Button, name: &str, now: Instant) -> Result<()> {
        if self.dry_run {
            println!("[GESTURE] {name}");
        } else {
            self.button_click(button)
                .with_context(|| format!("{name} failed"))?;
        }
        self.last_click = now;
        Ok(())
    }

    fn handle_scroll(&mut self, lm: &[[f32; 3]], now: Instant) -> Result<()> {
        // Wait for the gesture to settle so pointing->scroll transitions
        // don't emit phantom ticks.
        if now.duration_since(self.scroll_entered.unwrap_or(now)) < SCROLL_SETTLE {
            self.scroll_last_y = Some((lm[8][1] + lm[12][1]) / 2.0);
            return Ok(());
        }

        let current_y = (lm[8][1] + lm[12][1]) / 2.0;
        if let Some(last_y) = self.scroll_last_y {
            self.scroll_acc += current_y - last_y;
        }
        self.scroll_last_y = Some(current_y);

        let mut ticks = 0;
        while self.scroll_acc.abs() >= SCROLL_TICK && ticks < SCROLL_MAX_TICKS {
            // Fingers move down -> page scrolls down (standard mouse feel).
            let dir = if self.scroll_acc > 0.0 { -1 } else { 1 };
            if !self.dry_run {
                self.scroll_mouse(dir)?;
            } else {
                println!("[GESTURE] scroll {}", if dir < 0 { "down" } else { "up" });
            }
            self.scroll_acc -= SCROLL_TICK * self.scroll_acc.signum();
            ticks += 1;
        }
        Ok(())
    }

    fn move_cursor(&mut self, lm: &[[f32; 3]]) -> Result<()> {
        let tip = lm[8];
        let raw_x = ((tip[0] - MARGIN) / (1.0 - 2.0 * MARGIN)).clamp(0.0, 1.0);
        let raw_y = ((tip[1] - MARGIN) / (1.0 - 2.0 * MARGIN)).clamp(0.0, 1.0);
        let raw = [raw_x * self.screen_size.0 as f32, raw_y * self.screen_size.1 as f32];

        let smoothed = match self.cursor_ema {
            None => raw, // first frame: snap directly to the hand
            Some(e) => {
                let dpx = dist2d(&e, &raw);
                if dpx < CURSOR_JITTER_PX {
                    e // hold still against micro-shake
                } else {
                    // Adaptive smoothing: slow hand = heavy smoothing (stable),
                    // fast hand = near-instant tracking (responsive).
                    let alpha = (SMOOTH_MIN + (SMOOTH_MAX - SMOOTH_MIN) * (dpx / SMOOTH_RANGE_PX).min(1.0)).min(SMOOTH_MAX);
                    [e[0] + alpha * (raw[0] - e[0]), e[1] + alpha * (raw[1] - e[1])]
                }
            }
        };

        self.cursor_ema = Some(smoothed);
        let x = smoothed[0] as i32;
        let y = smoothed[1] as i32;

        if self.debug {
            eprintln!("[CURSOR] target=({:.3},{:.3}) screen=({}, {})", tip[0], tip[1], x, y);
        }
        if !self.dry_run {
            self.move_mouse(x, y)?;
        }
        Ok(())
    }

    /// Open palm: horizontal tilt-wave switches workspace, vertical flip
    /// steps volume. Both require real velocity along their axis.
    fn handle_palm(
        &mut self,
        lm: &[[f32; 3]],
        vx: f32,
        vy: f32,
        now: Instant,
    ) -> Result<()> {
        // Ignore the first moments of palm mode so residual motion from
        // pointing can't fire workspace/volume gestures.
        if now.duration_since(self.palm_since.unwrap_or(now)) < PALM_SETTLE {
            return Ok(());
        }

        // --- Workspace via tilt-wave ---
        let waving = vx.abs() > WAVE_VEL && vx.abs() > DOMINANCE * vy.abs();
        if waving {
            let raw = hand_tilt_deg(lm);
            let smooth = match self.tilt_ema {
                Some(t) => t + TILT_ALPHA * (raw - t),
                None => raw,
            };
            self.tilt_ema = Some(smooth);

            if self.debug {
                eprintln!(
                    "[TILT] raw={raw:+.1} smooth={smooth:+.1} vx={vx:+.2} armed={}",
                    self.tilt_armed
                );
            }

            if self.tilt_armed
                && smooth.abs() >= TILT_FIRE_DEG
                && now.duration_since(self.last_workspace_switch) > WORKSPACE_COOLDOWN
            {
                // Lean the hand toward your right -> workspace to the right.
                let dir = if smooth > 0.0 { "right" } else { "left" };
                self.switch_workspace(dir)?;
                self.last_workspace_switch = now;
                self.tilt_armed = false;
            }
        } else {
            // Not waving: re-arm once the hand settles back near vertical.
            if self.tilt_ema.map(|t| t.abs()).unwrap_or(0.0) < TILT_ARM_DEG {
                self.tilt_armed = true;
            }
            if self.tilt_armed {
                self.tilt_ema = None;
            }
        }

        // --- Volume via vertical flip ---
        let flipping = vy.abs() > FLIP_VEL && vy.abs() > DOMINANCE * vx.abs();
        if flipping && self.vol_armed
            && now.duration_since(self.last_volume_flip) > VOLUME_FLIP_COOLDOWN
        {
            let up = vy < 0.0; // flip up -> volume up
            if self.dry_run {
                println!("[GESTURE] volume {}", if up { "up" } else { "down" });
            } else {
                adjust_volume(up, VOLUME_STEP)?;
            }
            self.last_volume_flip = now;
            self.vol_armed = false;
            self.hist.clear(); // restart velocity window cleanly
        } else if !flipping && vy.abs() < FLIP_VEL * 0.5 {
            self.vol_armed = true;
        }

        Ok(())
    }

    fn close_tab(&mut self) -> Result<()> {
        if self.dry_run {
            println!("[GESTURE] close tab");
            return Ok(());
        }
        self.key_combo(&[EvKey::KEY_LEFTCTRL], EvKey::KEY_W)
    }

    fn switch_workspace(&mut self, direction: &str) -> Result<()> {
        if self.dry_run {
            println!("[GESTURE] switch workspace {direction}");
            return Ok(());
        }

        if std::env::var_os("HYPRLAND_INSTANCE_SIGNATURE").is_some() && command_exists("hyprctl") {
            let delta = if direction == "left" { "e-1" } else { "e+1" };
            return Command::new("hyprctl")
                .args(["dispatch", "workspace", delta])
                .status()
                .context("hyprctl failed")
                .map(|_| ());
        }

        if let Some(vi) = &mut self.virtual_input {
            let key = match direction {
                "left" => EvKey::KEY_LEFT,
                _ => EvKey::KEY_RIGHT,
            };
            return vi.key_combo(&[EvKey::KEY_LEFTCTRL, EvKey::KEY_LEFTALT], key);
        }

        if command_exists("hyprctl") {
            let delta = if direction == "left" { "-1" } else { "+1" };
            return Command::new("hyprctl")
                .args(["dispatch", "workspace", delta])
                .status()
                .context("hyprctl failed")
                .map(|_| ());
        }
        if command_exists("xdotool") {
            let delta = if direction == "left" { "-1" } else { "+1" };
            return Command::new("xdotool")
                .args(["set_desktop", "--relative", delta])
                .status()
                .context("xdotool failed")
                .map(|_| ());
        }
        anyhow::bail!("no workspace-switching backend found (tried uinput, hyprctl and xdotool)");
    }

    fn check_shutdown_gesture(&mut self, hands: &[Hand]) -> Result<()> {
        let now = Instant::now();
        if let Some(t) = self.last_shutdown && now.duration_since(t) < SHUTDOWN_COOLDOWN {
            return Ok(());
        }

        if self.shutdown_window_until.is_none()
            && self.clap_count > 0
            && now.duration_since(self.last_clap_time) > CLAP_TIMEOUT
        {
            self.clap_count = 0;
        }

        if let Some(until) = self.shutdown_window_until {
            if now > until {
                self.shutdown_window_until = None;
                self.right_hand_start_y = None;
                self.clap_count = 0;
            } else if self.right_hand_start_y.is_none() {
                if let Some(h) = hands.iter().find(|h| h.handedness.as_deref() == Some("Right")) {
                    self.right_hand_start_y = h.landmarks.first().map(|p| p[1]);
                }
            } else if let Some(h) = hands.iter().find(|h| h.handedness.as_deref() == Some("Right"))
                && let Some(wrist) = h.landmarks.first()
            {
                let current_y = wrist[1];
                if current_y - self.right_hand_start_y.unwrap() > SHUTDOWN_DOWN_THRESHOLD {
                    self.trigger_shutdown()?;
                }
            }
        }

        let clapping = self.hands_are_clapping(hands);
        if clapping && !self.clap_together && now.duration_since(self.last_clap_time) > CLAP_COOLDOWN
        {
            self.clap_count += 1;
            self.last_clap_time = now;
            if self.clap_count == 2 {
                self.shutdown_window_until = Some(now + AFTER_CLAP_WINDOW);
                self.right_hand_start_y = None;
            }
        }
        self.clap_together = clapping;

        Ok(())
    }

    fn hands_are_clapping(&self, hands: &[Hand]) -> bool {
        let valid: Vec<_> = hands.iter().filter(|h| h.landmarks.len() >= 21).collect();
        if valid.len() < 2 {
            return false;
        }
        let w1 = &valid[0].landmarks[0];
        let w2 = &valid[1].landmarks[0];
        dist(w1, w2) < CLAP_THRESHOLD
    }

    fn trigger_shutdown(&mut self) -> Result<()> {
        if self.dry_run {
            println!("[GESTURE] shutdown (dry-run, not executing)");
            self.reset_shutdown_state(Instant::now());
            return Ok(());
        }

        eprintln!("!!! SHUTDOWN GESTURE DETECTED !!!");
        eprintln!("Powering off in 2 seconds... (Ctrl+C to cancel)");
        std::thread::sleep(Duration::from_secs(2));

        self.reset_shutdown_state(Instant::now());

        if Command::new("systemctl").arg("poweroff").status().is_ok() {
            return Ok(());
        }
        if Command::new("shutdown").args(["-h", "now"]).status().is_ok() {
            return Ok(());
        }
        if Command::new("poweroff").status().is_ok() {
            return Ok(());
        }
        anyhow::bail!("failed to shut down the system");
    }

    fn reset_shutdown_state(&mut self, now: Instant) {
        self.last_shutdown = Some(now);
        self.shutdown_window_until = None;
        self.right_hand_start_y = None;
        self.clap_count = 0;
    }

    fn move_mouse(&mut self, x: i32, y: i32) -> Result<()> {
        if let Some(vi) = &mut self.virtual_input {
            vi.move_mouse(x, y)
        } else {
            self.enigo.move_mouse(x, y, Coordinate::Abs).context("mouse move failed")
        }
    }

    fn scroll_mouse(&mut self, y: i32) -> Result<()> {
        if let Some(vi) = &mut self.virtual_input {
            vi.scroll(y)
        } else {
            self.enigo.scroll(y, Axis::Vertical).context("mouse scroll failed")
        }
    }

    fn button_click(&mut self, button: Button) -> Result<()> {
        if let Some(vi) = &mut self.virtual_input {
            vi.button_click(button)
        } else {
            self.enigo.button(button, Direction::Click).context("button click failed")
        }
    }

    fn key_combo(&mut self, modifiers: &[EvKey], key: EvKey) -> Result<()> {
        if let Some(vi) = &mut self.virtual_input {
            vi.key_combo(modifiers, key)
        } else {
            for m in modifiers {
                let enigo_key = evdev_to_enigo_key(*m)
                    .with_context(|| format!("unsupported modifier for Enigo fallback: {m:?}"))?;
                self.enigo.key(enigo_key, Direction::Press)?;
            }
            let enigo_key = evdev_to_enigo_key(key)
                .with_context(|| format!("unsupported key for Enigo fallback: {key:?}"))?;
            self.enigo.key(enigo_key, Direction::Click)?;
            for m in modifiers.iter().rev() {
                let enigo_key = evdev_to_enigo_key(*m)
                    .with_context(|| format!("unsupported modifier for Enigo fallback: {m:?}"))?;
                self.enigo.key(enigo_key, Direction::Release)?;
            }
            Ok(())
        }
    }
}

fn command_exists(cmd: &str) -> bool {
    Command::new("which")
        .arg(cmd)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn evdev_to_enigo_key(code: EvKey) -> Option<Key> {
    match code {
        EvKey::KEY_LEFTCTRL => Some(Key::Control),
        EvKey::KEY_LEFTSHIFT => Some(Key::Shift),
        EvKey::KEY_LEFTALT => Some(Key::Alt),
        EvKey::KEY_LEFTMETA => Some(Key::Meta),
        EvKey::KEY_TAB => Some(Key::Tab),
        EvKey::KEY_W => Some(Key::Unicode('w')),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dry_run = args.iter().any(|a| a == "--dry-run");
    let debug = args.iter().any(|a| a == "--debug");
    let preview = args.iter().any(|a| a == "--preview");
    let camera = args
        .iter()
        .position(|a| a == "--camera")
        .and_then(|i| args.get(i + 1))
        .unwrap_or(&"0".to_string())
        .clone();

    let python = std::env::var("AIR_MOUSE_PYTHON")
        .unwrap_or_else(|_| "/home/dranzer/hand-control/.venv/bin/python".into());

    let mut cmd = Command::new(python);
    cmd.arg("tracker.py")
        .arg("--camera")
        .arg(&camera)
        .stdout(Stdio::piped());
    if preview {
        cmd.arg("--preview");
    }
    let mut child = cmd.spawn().context("failed to start tracker.py")?;

    let stdout = child.stdout.take().context("no stdout")?;
    let reader = BufReader::new(stdout);

    let mut ctrl = Controller::new(dry_run, debug)?;

    println!("Air Mouse (Rust controller) started.");
    println!("Screen: {}x{}", ctrl.screen_size.0, ctrl.screen_size.1);
    if dry_run {
        println!("Dry-run mode: no real mouse/key/volume actions.");
    }
    println!("Press Ctrl+C to quit.");

    for line in reader.lines() {
        let line = line.context("read failed")?;
        let frame: Frame = serde_json::from_str(&line).context("bad JSON")?;

        if let Err(e) = ctrl.check_shutdown_gesture(&frame.hands) {
            eprintln!("shutdown gesture error: {e:?}");
        }

        if ctrl.shutdown_window_until.is_some() {
            continue;
        }

        // Prefer the right hand; fall back to any detected hand.
        let primary = frame
            .hands
            .iter()
            .find(|h| h.handedness.as_deref() == Some("Right"))
            .or_else(|| frame.hands.first());

        match primary {
            Some(hand) if hand.landmarks.len() >= 21 => {
                if let Err(e) = ctrl.process(&hand.landmarks) {
                    eprintln!("process error: {e:?}");
                }
            }
            _ => ctrl.on_hand_lost(),
        }
    }

    let _ = child.kill();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Open palm: wrist at bottom, fingers straight up.
    fn open_palm() -> [[f32; 3]; 21] {
        let mut lm = [[0.0; 3]; 21];
        lm[0] = [0.50, 0.90, 0.0];
        // thumb (extended to the left)
        lm[1] = [0.44, 0.84, 0.0];
        lm[2] = [0.40, 0.78, 0.0];
        lm[3] = [0.37, 0.72, 0.0];
        lm[4] = [0.34, 0.66, 0.0];
        // index / middle / ring / pinky: mcp, pip, dip, tip straight up
        for (col, base) in [(0usize, 5usize), (1, 9), (2, 13), (3, 17)] {
            let x = 0.44 + col as f32 * 0.055;
            let y_top = 0.70 - col as f32 * 0.02;
            lm[base] = [x, y_top + 0.00, 0.0];
            lm[base + 1] = [x, y_top - 0.08, 0.0];
            lm[base + 2] = [x, y_top - 0.14, 0.0];
            lm[base + 3] = [x, y_top - 0.20, 0.0];
        }
        lm
    }

    fn curl(lm: &mut [[f32; 3]; 21], base: usize) {
        // Fold pip/dip/tip back down toward the palm.
        let x = lm[base][0];
        let y = lm[base][1];
        lm[base + 1] = [x, y - 0.02, 0.0];
        lm[base + 2] = [x + 0.02, y + 0.02, 0.0];
        lm[base + 3] = [x + 0.04, y + 0.04, 0.0];
    }

    fn fingers_up(lm: &[[f32; 3]]) -> [bool; 5] {
        finger_angles(lm).map(|a| a > FINGER_UP_DEG)
    }

    #[test]
    fn open_hand_all_extended() {
        assert!(fingers_up(&open_palm()).iter().all(|&x| x));
    }

    #[test]
    fn fist_all_curled() {
        let mut lm = open_palm();
        // Thumb hooked back across the palm (tip curls toward the wrist).
        lm[3] = [0.38, 0.74, 0.0];
        lm[4] = [0.42, 0.79, 0.0];
        for b in [5, 9, 13, 17] {
            curl(&mut lm, b);
        }
        let f = fingers_up(&lm);
        assert!(f.iter().all(|&x| !x), "fist misread as {f:?}");
    }

    #[test]
    fn pointing_only_index() {
        let mut lm = open_palm();
        curl(&mut lm, 1);
        for b in [9, 13, 17] {
            curl(&mut lm, b);
        }
        let f = fingers_up(&lm);
        assert_eq!((f[1], f[2], f[3], f[4]), (true, false, false, false));
    }

    #[test]
    fn finger_toward_camera_still_reads_straight() {
        // Index finger aimed at the camera: barely moves in x/y but gains
        // depth. The old 2D-only angle read this as folded.
        let mut lm = open_palm();
        lm[5][2] = 0.00;
        lm[6][2] = 0.05;
        lm[7][2] = 0.10;
        lm[8][2] = 0.15;
        assert!(finger_angles(&lm)[1] > FINGER_UP_DEG);
    }

    #[test]
    fn finger_hysteresis_holds_between_thresholds() {
        let angles = [180.0, FINGER_UP_DEG + 1.0, 130.0, FINGER_DOWN_DEG - 1.0, 60.0];
        let expected = [
            true,  // straight -> up
            true,  // above ON threshold -> up
            true,  // inside band -> stays up (was up)
            false, // below OFF threshold -> down
            false, // curled -> down
        ];
        for (a, e) in angles.iter().zip(expected) {
            assert_eq!(hysteresis_step(true, *a), e);
        }
        // Same band value keeps a previously-folded finger folded.
        assert!(!hysteresis_step(false, 130.0));
    }

    #[test]
    fn cursor_mapping_full_frame() {
        let norm = |v: f32| ((v - MARGIN) / (1.0 - 2.0 * MARGIN)).clamp(0.0, 1.0);
        assert_eq!(norm(MARGIN), 0.0);
        assert_eq!(norm(1.0 - MARGIN), 1.0);
        assert!((norm(0.5) - 0.5).abs() < 1e-6);
        // A hand low in frame (resting position) is still reachable.
        assert!(norm(0.85) > 0.85);
    }

    #[test]
    fn tilt_sign_uses_knuckle_direction() {
        let mut lm = open_palm();
        // Shift knuckles right of the wrist -> fingers lean right -> positive.
        for i in 1..21 {
            lm[i][0] += 0.15;
        }
        assert!(hand_tilt_deg(&lm) > 10.0);
        let mut lm2 = open_palm();
        for i in 1..21 {
            lm2[i][0] -= 0.15;
        }
        assert!(hand_tilt_deg(&lm2) < -10.0);
    }
}
