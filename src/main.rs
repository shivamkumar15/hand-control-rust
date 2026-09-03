use std::collections::VecDeque;
use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;
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
//
// Thresholds are palm-relative (so distance from the camera doesn't matter)
// with an absolute floor/ceiling so tiny far-away hands still work.
// Fire if EITHER the absolute or the relative distance is beaten; re-arm
// only once BOTH are cleared, giving a wide hysteresis band.
const PINCH_ON: f32 = 0.055;
const PINCH_OFF: f32 = 0.095;
const PINCH_PALM_REL_ON: f32 = 0.35; // fire when d < palm * this ...
const PINCH_PALM_REL_OFF: f32 = 0.45; // ... re-arm when d > palm * this
// The pinching finger must not be fully folded (touching the thumb
// necessarily curls it, so demanding "fully extended" made real pinches
// almost impossible). 90 deg = clearly not tucked into the palm.
const PINCH_MIN_ANGLE_DEG: f32 = 90.0;
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

// Open-palm gestures are motion-gated so merely holding/rotating the hand
// does nothing until you actually swipe or flip it.
//
// Workspace: displacement-based swipe. The palm position is anchored when
// palm mode settles; horizontal travel past SWIPE_DIST fires a switch in the
// travel direction. Travel only accumulates while genuinely swiping, so slow
// drift never fires, and the anchor eases back toward the hand when
// stationary so stale distance can't fire later.
const SWIPE_DIST: f32 = 0.18; // normalized units of travel per switch
const SWIPE_VEL: f32 = 0.35; // min horizontal speed for travel to count
const FLIP_VEL: f32 = 0.85;
const DOMINANCE: f32 = 1.3; // axis must beat the other by this factor
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
// A tip-to-knuckle distance fallback (relative to palm size) rescues cases
// where the angle alone is unreliable: finger aimed at the camera, hand
// tilted, or noisy depth (z) from MediaPipe.
const FINGER_UP_DEG: f32 = 135.0;
const FINGER_DOWN_DEG: f32 = 115.0;
// Extension ratio = dist(tip, mcp) / palm_size. Clearly spread > 0.6,
// clearly curled < 0.4, dead-band in between (keeps previous state).
const EXT_RATIO_UP: f32 = 0.60;
const EXT_RATIO_DOWN: f32 = 0.42;

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

// Voice command: saying "shutdown now" powers off. Final hypotheses must be
// matched, but partials fire too so the command works even while you keep
// talking. Matched words are consumed from the buffer so leftovers can't
// re-trigger.
const VOICE_CMD_WORDS: [&str; 2] = ["shutdown", "now"];
const VOICE_PARTIAL_MIN_LEN: usize = 7; // "shutdown now".len()
const VOICE_COOLDOWN: Duration = Duration::from_secs(10);

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

/// One line from voice.py
#[derive(Deserialize, Debug)]
struct VoiceEvent {
    text: String,
    #[serde(default)]
    partial: bool,
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------
fn dist(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

/// Full 3D tip distance (includes depth). Pinch tips coincide in x/y but
/// MediaPipe still reports a small z gap; including z with a mild weight
/// keeps far/near hands comparable without letting noisy z dominate.
fn dist_pinch(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = (a[2] - b[2]) * 0.5;
    (dx * dx + dy * dy + dz * dz).sqrt()
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

/// Hysteresis step for one finger's extended state, combining two cues:
/// the PIP joint angle and the tip-to-knuckle spread (relative to palm).
/// Either cue can force a decision; when both are in the dead-band the
/// previous state is kept, so borderline fingers stop flickering.
fn hysteresis_step(prev: bool, angle: f32, ratio: f32) -> bool {
    let angle_up = angle >= FINGER_UP_DEG;
    let angle_down = angle <= FINGER_DOWN_DEG;
    let ratio_up = ratio >= EXT_RATIO_UP;
    let ratio_down = ratio <= EXT_RATIO_DOWN;
    if angle_up || ratio_up {
        true
    } else if angle_down && ratio_down {
        false
    } else if angle_down || ratio_down {
        // One cue says folded, the other is undecided -> fold, unless we
        // were clearly up and neither cue is strongly folded.
        // Keep hysteresis tight: fold only if the angle is well below UP.
        if angle <= FINGER_DOWN_DEG && ratio < EXT_RATIO_UP {
            false
        } else {
            prev
        }
    } else {
        prev
    }
}

/// Tip-to-knuckle spread per finger (thumb first), normalized by palm size.
/// Extended ≈ 0.7–1.0, curled ≈ 0.3–0.5. Scale-free, so it works near/far.
fn finger_ratios(lm: &[[f32; 3]], palm_size: f32) -> [f32; 5] {
    const PAIRS: [(usize, usize); 5] = [(2, 4), (5, 8), (9, 12), (13, 16), (17, 20)];
    let denom = palm_size.max(1e-4);
    let mut out = [0.0; 5];
    for (i, (mcp, tip)) in PAIRS.iter().enumerate() {
        out[i] = dist(&lm[*mcp], &lm[*tip]) / denom;
    }
    out
}

/// Pinch fire/re-arm test with palm-relative hysteresis.
/// Fire when tips are close by EITHER measure; re-arm only once they are
/// far by BOTH measures, so the trigger doesn't chatter at the boundary.
fn pinch_closed(d: f32, palm_size: f32) -> bool {
    d < PINCH_ON || d < palm_size * PINCH_PALM_REL_ON
}

fn pinch_open(d: f32, palm_size: f32) -> bool {
    d > PINCH_OFF && d > palm_size * PINCH_PALM_REL_OFF
}

fn palm_centroid(lm: &[[f32; 3]]) -> [f32; 2] {
    let mut sum = [0.0; 2];
    for i in [0, 5, 17] {
        sum[0] += lm[i][0];
        sum[1] += lm[i][1];
    }
    [sum[0] / 3.0, sum[1] / 3.0]
}

/// Outcome of one swipe-tracking step.
enum SwipeOutcome {
    /// Keep accumulating; the hand is mid-swipe.
    Hold,
    /// Fire a workspace switch toward this direction, re-anchoring here.
    Fire(&'static str),
    /// Not swiping: ease the anchor toward the hand so stale travel decays.
    Reanchor([f32; 2]),
}

const SWIPE_REANCHOR: f32 = 0.05; // anchor easing per non-swiping frame

/// Pure swipe logic: given the current anchor and hand position/velocity,
/// decide whether to fire a switch, keep holding, or re-anchor.
fn swipe_step(anchor: [f32; 2], pos: [f32; 2], vx: f32, vy: f32) -> SwipeOutcome {
    let swiping = vx.abs() > SWIPE_VEL && vx.abs() > DOMINANCE * vy.abs();
    if !swiping {
        return SwipeOutcome::Reanchor([
            anchor[0] + SWIPE_REANCHOR * (pos[0] - anchor[0]),
            anchor[1] + SWIPE_REANCHOR * (pos[1] - anchor[1]),
        ]);
    }
    let dx = pos[0] - anchor[0];
    if dx.abs() >= SWIPE_DIST {
        SwipeOutcome::Fire(if dx > 0.0 { "right" } else { "left" })
    } else {
        SwipeOutcome::Hold
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Mode {
    Idle,
    Cursor,
    Scroll,
    Palm,
}

/// Match a voice command against an utterance. The utterance is normalized
/// (lowercase, alphanumeric only) and the command words must appear in order,
/// adjacent ("shut down now" also matches via the word list). Returns the
/// normalized utterance with the matched words consumed, so a final result
/// following a fired partial can't re-trigger.
fn match_voice_command(utterance: &str) -> bool {
    let norm: String = utterance
        .to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect();
    let words: Vec<&str> = norm.split_whitespace().collect();
    if words.len() < VOICE_CMD_WORDS.len() {
        return false;
    }
    'outer: for start in 0..=(words.len() - VOICE_CMD_WORDS.len()) {
        for (i, cmd) in VOICE_CMD_WORDS.iter().enumerate() {
            if words[start + i] != *cmd {
                continue 'outer;
            }
        }
        return true;
    }
    false
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

/// Forgiving pose classifier (pure, unit-testable).
///
/// - Cursor: index up, middle down. Ring/pinky wobble is ignored, so a
///   slightly lifted ring finger no longer kills the cursor.
/// - Scroll: index AND middle up, but NOT a full open palm (at least one
///   of ring/pinky folded). Survives one borderline finger.
/// - Palm: 3+ of the four fingers up.
/// - Else Idle. Fist is handled by the caller (forces Idle).
fn classify_mode(fingers: [bool; 5], fist_now: bool, fist_fired: bool) -> Mode {
    if fist_now || fist_fired {
        return Mode::Idle;
    }
    let (_, index, middle, ring, pinky) = (fingers[0], fingers[1], fingers[2], fingers[3], fingers[4]);
    let ext_count = [index, middle, ring, pinky].iter().filter(|&&x| x).count();
    if index && middle && !(ring && pinky) {
        Mode::Scroll
    } else if index && !middle {
        Mode::Cursor
    } else if ext_count >= 3 {
        Mode::Palm
    } else {
        Mode::Idle
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

    swipe_anchor: Option<[f32; 2]>,
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
    last_voice_shutdown: Option<Instant>,
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
            swipe_anchor: None,
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
            last_voice_shutdown: None,
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
        self.swipe_anchor = None;
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
            // Leaving palm mode invalidates swipe tracking.
            if self.mode == Mode::Palm && m != Mode::Palm {
                self.swipe_anchor = None;
            }
            self.mode = m;
            match m {
                Mode::Cursor => self.cursor_ema = None, // snap on entry, no glide
                Mode::Scroll => self.scroll_entered = Some(Instant::now()),
                // Restart motion history so pointing speed doesn't leak into
                // palm gesture detection.
                Mode::Palm => {
                    self.palm_since = Some(Instant::now());
                    self.hist.clear();
                }
                _ => {}
            }
            self.status(&format!("[MODE] {}", m.label()));
        }
    }

    /// Update per-finger extended states with hysteresis (angle + spread).
    fn update_fingers(&mut self, lm: &[[f32; 3]], palm_size: f32) -> ([bool; 5], [f32; 5], [f32; 5]) {
        let angles = finger_angles(lm);
        let ratios = finger_ratios(lm, palm_size);
        for i in 0..5 {
            self.finger_up[i] = hysteresis_step(self.finger_up[i], angles[i], ratios[i]);
        }
        (self.finger_up, angles, ratios)
    }

    fn process(&mut self, lm: &[[f32; 3]]) -> Result<()> {
        let span = hand_span(lm);
        if span < MIN_HAND_SPAN {
            return Ok(());
        }

        let now = Instant::now();
        let palm_size = dist(&lm[0], &lm[9]).max(1e-4);
        let (fingers, angles, ratios) = self.update_fingers(lm, palm_size);
        let four = [fingers[1], fingers[2], fingers[3], fingers[4]];
        let ext_count = four.iter().filter(|&&x| x).count();
        let fist_now = ext_count == 0 && lm[0][1] < FIST_MAX_WRIST_Y;

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
        // NOTE: the pinching finger only needs to be NOT-folded (angle >
        // PINCH_MIN_ANGLE_DEG). Touching the thumb always curls the finger,
        // so requiring "fully extended" rejected almost all real pinches.
        if !fist_now {
            let d_left = dist_pinch(&lm[4], &lm[8]);
            let d_right = dist_pinch(&lm[4], &lm[12]);
            let can_click = now.duration_since(self.last_click) > CLICK_COOLDOWN;
            let index_open = angles[1] > PINCH_MIN_ANGLE_DEG;
            let middle_open = angles[2] > PINCH_MIN_ANGLE_DEG;

            if self.pinch_left_armed && can_click && index_open && pinch_closed(d_left, palm_size) {
                self.do_click(Button::Left, "left click", now)?;
                self.pinch_left_armed = false;
            } else if pinch_open(d_left, palm_size) {
                self.pinch_left_armed = true;
            }

            if self.pinch_right_armed && can_click && middle_open && pinch_closed(d_right, palm_size) {
                self.do_click(Button::Right, "right click", now)?;
                self.pinch_right_armed = false;
            } else if pinch_open(d_right, palm_size) {
                self.pinch_right_armed = true;
            }

            if self.debug {
                eprintln!(
                    "[PINCH] L={d_left:.3} (armed={}) R={d_right:.3} (armed={}) idx_ang={:.0} mid_ang={:.0} palm={palm_size:.3}",
                    self.pinch_left_armed, self.pinch_right_armed, angles[1], angles[2]
                );
            }
        }

        // ---- Mode selection (debounced, forgiving) -------------------------
        let new_mode = classify_mode(fingers, fist_now, self.fist_fired);
        if self.debug {
            eprintln!(
                "[STATE] span={span:.3} palm={palm_size:.3} fingers={fingers:?} ang=[{:.0},{:.0},{:.0},{:.0},{:.0}] rat=[{:.2},{:.2},{:.2},{:.2},{:.2}] mode={:?} cand={:?}({}) ext={ext_count}",
                angles[0],
                angles[1],
                angles[2],
                angles[3],
                angles[4],
                ratios[0],
                ratios[1],
                ratios[2],
                ratios[3],
                ratios[4],
                self.mode,
                self.mode_candidate,
                self.cand_frames
            );
        }
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
                self.swipe_anchor = None;
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

    /// Open palm: a horizontal swipe switches workspace, a vertical flip
    /// steps volume. Both need genuine motion along their axis.
    fn handle_palm(
        &mut self,
        lm: &[[f32; 3]],
        vx: f32,
        vy: f32,
        now: Instant,
    ) -> Result<()> {
        let pos = palm_centroid(lm);

        // Ignore the first moments of palm mode so residual motion from
        // pointing can't fire workspace/volume gestures. Anchor the swipe
        // only after settling.
        if now.duration_since(self.palm_since.unwrap_or(now)) < PALM_SETTLE {
            self.swipe_anchor = Some(pos);
            return Ok(());
        }

        // --- Workspace via horizontal swipe ---
        match self.swipe_anchor {
            None => self.swipe_anchor = Some(pos),
            Some(anchor) => match swipe_step(anchor, pos, vx, vy) {
                SwipeOutcome::Hold => {}
                SwipeOutcome::Reanchor(a) => self.swipe_anchor = Some(a),
                SwipeOutcome::Fire(dir) => {
                    if now.duration_since(self.last_workspace_switch) > WORKSPACE_COOLDOWN {
                        self.switch_workspace(dir)?;
                        self.last_workspace_switch = now;
                    }
                    // Re-anchor where the swipe ended, fired or not.
                    self.swipe_anchor = Some(pos);
                }
            },
        }
        if self.debug {
            let dx = pos[0] - self.swipe_anchor.unwrap_or(pos)[0];
            eprintln!("[SWIPE] dx={dx:+.3} vx={vx:+.2} vy={vy:+.2}");
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

    /// Handle one utterance from voice.py. Partials fire early (so the
    /// command triggers even if the final result lags or never comes),
    /// finals are also checked; the cooldown blocks double-fires.
    fn handle_voice_event(&mut self, ev: &VoiceEvent) -> Result<()> {
        let now = Instant::now();
        if let Some(t) = self.last_voice_shutdown
            && now.duration_since(t) < VOICE_COOLDOWN
        {
            return Ok(());
        }

        let matched = match_voice_command(&ev.text)
            || (ev.partial
                && ev.text.to_lowercase().trim().len() >= VOICE_PARTIAL_MIN_LEN
                && match_voice_command(&ev.text));
        if !matched {
            if self.debug {
                eprintln!("[VOICE] heard {:?} (no match)", ev.text);
            }
            return Ok(());
        }

        self.last_voice_shutdown = Some(now);
        self.trigger_shutdown_voice(&ev.text)
    }

    fn check_shutdown_gesture(&mut self, hands: &[Hand]) -> Result<()> {        let now = Instant::now();
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

        eprintln!("!!! SHUTDOWN DETECTED !!!");
        eprintln!("Powering off in 2 seconds... (Ctrl+C to cancel)");
        std::thread::sleep(Duration::from_secs(2));

        self.reset_shutdown_state(Instant::now());

        poweroff()
    }

    /// Voice-triggered shutdown: same action, different wording.
    fn trigger_shutdown_voice(&mut self, heard: &str) -> Result<()> {
        if self.dry_run {
            println!("[GESTURE] voice shutdown: {heard:?} (dry-run, not executing)");
            self.reset_shutdown_state(Instant::now());
            return Ok(());
        }

        eprintln!("!!! VOICE COMMAND: {heard:?} !!!");
        eprintln!("Powering off in 2 seconds... (Ctrl+C to cancel)");
        std::thread::sleep(Duration::from_secs(2));

        self.reset_shutdown_state(Instant::now());

        poweroff()
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

/// Power the machine off, trying the usual suspects.
fn poweroff() -> Result<()> {
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

/// Spawn voice.py and stream its utterances over a channel. The listener is
/// best-effort: if the mic or model is missing, the controller still runs
/// with hand gestures only.
fn spawn_voice_listener(python: &str, debug: bool) -> Option<mpsc::Receiver<VoiceEvent>> {
    let child = Command::new(python)
        .arg("voice.py")
        .stdout(Stdio::piped())
        .stderr(if debug { Stdio::inherit() } else { Stdio::null() })
        .spawn();
    let mut child = match child {
        Ok(c) => c,
        Err(e) => {
            eprintln!("voice listener not started: {e}");
            return None;
        }
    };

    let stdout = child.stdout.take()?;
    let (tx, rx) = mpsc::channel::<VoiceEvent>();
    thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            let Ok(line) = line else { break };
            match serde_json::from_str::<VoiceEvent>(&line) {
                Ok(ev) => {
                    if tx.send(ev).is_err() {
                        break;
                    }
                }
                Err(_) => continue,
            }
        }
        let _ = child.kill();
        let _ = child.wait();
    });
    Some(rx)
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dry_run = args.iter().any(|a| a == "--dry-run");
    let debug = args.iter().any(|a| a == "--debug");
    let preview = args.iter().any(|a| a == "--preview");
    let no_voice = args.iter().any(|a| a == "--no-voice");
    let camera = args
        .iter()
        .position(|a| a == "--camera")
        .and_then(|i| args.get(i + 1))
        .unwrap_or(&"0".to_string())
        .clone();

    let python = std::env::var("AIR_MOUSE_PYTHON")
        .unwrap_or_else(|_| "/home/dranzer/hand-control/.venv/bin/python".into());

    let mut cmd = Command::new(&python);
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

    // Voice command listener: optional because a mic may not exist.
    let voice_rx = if no_voice {
        None
    } else {
        spawn_voice_listener(&python, debug)
    };

    println!("Air Mouse (Rust controller) started.");
    println!("Screen: {}x{}", ctrl.screen_size.0, ctrl.screen_size.1);
    if dry_run {
        println!("Dry-run mode: no real mouse/key/volume actions.");
    }
    if voice_rx.is_some() {
        println!("Voice commands enabled: say \"shutdown now\" to power off.");
    }
    println!("Press Ctrl+C to quit.");

    for line in reader.lines() {
        // Drain any voice events that arrived since the last camera frame.
        if let Some(rx) = &voice_rx {
            while let Ok(ev) = rx.try_recv() {
                if let Err(e) = ctrl.handle_voice_event(&ev) {
                    eprintln!("voice error: {e:?}");
                }
            }
        }

        let line = line.context("read failed")?;
        let frame: Frame = serde_json::from_str(&line).context("bad JSON")?;

        if let Err(e) = ctrl.check_shutdown_gesture(&frame.hands) {
            eprintln!("shutdown gesture error: {e:?}");
        }

        if ctrl.shutdown_window_until.is_some() {
            continue;
        }

        // Track the largest (closest) hand. Handedness labels flip when the
        // frame is mirrored and MediaPipe mislabels at angles, so preferring
        // "Right" made the cursor jump between hands. Biggest span = the
        // hand the user is actively gesturing with.
        let primary = frame
            .hands
            .iter()
            .filter(|h| h.landmarks.len() >= 21)
            .max_by(|a, b| {
                hand_span(&a.landmarks)
                    .partial_cmp(&hand_span(&b.landmarks))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

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
        let palm = dist(&lm[0], &lm[9]).max(1e-4);
        let angles = finger_angles(lm);
        let ratios = finger_ratios(lm, palm);
        let mut out = [false; 5];
        for i in 0..5 {
            // Fresh read without history: extended if either cue is strong.
            out[i] = angles[i] >= FINGER_UP_DEG || ratios[i] >= EXT_RATIO_UP;
        }
        out
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
        // Strong cues decide; dead-band keeps history.
        assert!(hysteresis_step(true, 180.0, 0.8)); // straight -> up
        assert!(hysteresis_step(true, FINGER_UP_DEG + 1.0, 0.5)); // angle ON -> up
        assert!(hysteresis_step(true, 125.0, 0.8)); // ratio ON -> up
        assert!(hysteresis_step(true, 125.0, 0.5)); // both mid -> stays up
        assert!(!hysteresis_step(true, FINGER_DOWN_DEG - 1.0, 0.3)); // both OFF -> down
        assert!(!hysteresis_step(false, 60.0, 0.3)); // curled -> down
        // Same mid-band value keeps a previously-folded finger folded.
        assert!(!hysteresis_step(false, 125.0, 0.5));
    }

    #[test]
    fn curled_finger_has_small_spread_ratio() {
        let mut lm = open_palm();
        for b in [5, 9, 13, 17] {
            curl(&mut lm, b);
        }
        let palm = dist(&lm[0], &lm[9]).max(1e-4);
        let ratios = finger_ratios(&lm, palm);
        for r in [ratios[1], ratios[2], ratios[3], ratios[4]] {
            assert!(r < EXT_RATIO_UP, "curled ratio should be small, got {r}");
        }
        let open = open_palm();
        let palm_o = dist(&open[0], &open[9]).max(1e-4);
        let ro = finger_ratios(&open, palm_o);
        for r in [ro[1], ro[2], ro[3], ro[4]] {
            assert!(r >= EXT_RATIO_UP, "open ratio should be large, got {r}");
        }
    }

    #[test]
    fn pinch_uses_palm_relative_hysteresis() {
        let palm = 0.20;
        // Touching tips fire even though absolute distance alone is borderline.
        assert!(pinch_closed(0.03, palm));
        assert!(pinch_closed(0.060, palm)); // 0.06 < 0.20*0.35=0.07 -> fire
        assert!(!pinch_closed(0.10, palm));
        // Re-arm needs BOTH measures cleared.
        assert!(pinch_open(0.10, palm)); // > 0.095 and > 0.09
        assert!(!pinch_open(0.06, palm));
    }

    #[test]
    fn pinch_allows_partially_curled_finger() {
        // A real pinch curls the index to ~100-120 deg. The old gate
        // (finger fully extended) rejected this; the new gate allows it.
        assert!(110.0 > PINCH_MIN_ANGLE_DEG);
        assert!(95.0 > PINCH_MIN_ANGLE_DEG);
        assert!(70.0 < PINCH_MIN_ANGLE_DEG); // fully tucked still blocked
    }

    #[test]
    fn mode_classifier_is_forgiving() {
        // Pointing survives ring/pinky wobble.
        assert_eq!(classify_mode([false, true, false, false, false], false, false), Mode::Cursor);
        assert_eq!(classify_mode([false, true, false, true, false], false, false), Mode::Cursor);
        assert_eq!(classify_mode([true, true, false, true, true], false, false), Mode::Cursor);
        // Scroll survives one borderline finger.
        assert_eq!(classify_mode([false, true, true, false, false], false, false), Mode::Scroll);
        assert_eq!(classify_mode([false, true, true, true, false], false, false), Mode::Scroll);
        assert_eq!(classify_mode([false, true, true, false, true], false, false), Mode::Scroll);
        // Full palm still palm.
        assert_eq!(classify_mode([true, true, true, true, true], false, false), Mode::Palm);
        assert_eq!(classify_mode([false, true, true, true, true], false, false), Mode::Palm);
        // Fist forces idle.
        assert_eq!(classify_mode([false, false, false, false, false], true, false), Mode::Idle);
        assert_eq!(classify_mode([false, true, false, false, false], false, true), Mode::Idle);
    }

    #[test]
    fn pointing_pose_classifies_to_cursor() {
        let mut lm = open_palm();
        curl(&mut lm, 1);
        for b in [9, 13, 17] {
            curl(&mut lm, b);
        }
        let f = fingers_up(&lm);
        assert_eq!((f[1], f[2], f[3], f[4]), (true, false, false, false));
        assert_eq!(classify_mode(f, false, false), Mode::Cursor);
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
    fn fast_horizontal_swipe_fires() {
        let anchor = [0.5, 0.5];
        // Swipe right fast and clean: crosses the distance threshold.
        match swipe_step(anchor, [anchor[0] + SWIPE_DIST + 0.01, 0.5], 1.2, 0.1) {
            SwipeOutcome::Fire(dir) => assert_eq!(dir, "right"),
            _ => panic!("expected fire"),
        }
        // Same travel but below the speed gate -> no fire, just re-anchor.
        assert!(matches!(
            swipe_step(anchor, [anchor[0] + SWIPE_DIST + 0.01, 0.5], 0.1, 0.05),
            SwipeOutcome::Reanchor(_)
        ));
    }

    #[test]
    fn slow_drift_never_fires_workspace() {
        let anchor = [0.5, 0.5];
        let mut a = anchor;
        // Simulate many frames of slow rightward drift.
        for _ in 0..600 {
            match swipe_step(a, [a[0] + 0.001, a[1]], 0.15, 0.02) {
                SwipeOutcome::Fire(_) => panic!("slow drift fired"),
                SwipeOutcome::Reanchor(next) => a = next,
                SwipeOutcome::Hold => {}
            }
        }
    }

    #[test]
    fn vertical_motion_does_not_fire_workspace() {
        let anchor = [0.5, 0.5];
        // Big horizontal displacement but vertical motion dominates.
        assert!(matches!(
            swipe_step(anchor, [anchor[0] + SWIPE_DIST, 0.7], 0.3, 1.5),
            SwipeOutcome::Reanchor(_)
        ));
    }

    #[test]
    fn voice_command_matches() {
        assert!(match_voice_command("shutdown now"));
        assert!(match_voice_command("SHUTDOWN NOW"));
        assert!(match_voice_command("Shutdown, now!"));
        assert!(match_voice_command("please shutdown now"));
        assert!(match_voice_command("shutdown now please"));
        // Vosk may split it as two tokens or hear adjacent words.
        assert!(match_voice_command("the shutdown now thing"));
    }

    #[test]
    fn voice_command_rejects_near_misses() {
        assert!(!match_voice_command("shutdown"));
        assert!(!match_voice_command("now"));
        assert!(!match_voice_command("shut down"));
        assert!(!match_voice_command("shutdown the laptop now is off"));
        assert!(!match_voice_command("how about shutting down"));
        assert!(!match_voice_command(""));
    }
}
