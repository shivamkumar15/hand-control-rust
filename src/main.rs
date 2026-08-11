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
const CLICK_THRESHOLD: f32 = 0.035;
const CLICK_THRESHOLD_REL: f32 = 0.35; // relative to palm size
const CLICK_COOLDOWN: Duration = Duration::from_millis(350);

const MIN_HAND_SPAN: f32 = 0.12;

const FLIP_VEL: f32 = 1.5; // normalized Y units per second for a vertical flip
const VOLUME_FLIP_COOLDOWN: Duration = Duration::from_millis(600);
const VOLUME_STEP: u32 = 5; // percent change per flip
const SWIPE_VEL: f32 = 1.8; // normalized X units per second
const SWIPE_COOLDOWN: Duration = Duration::from_millis(1000);
const FIST_COOLDOWN: Duration = Duration::from_millis(1000);
const WORKSPACE_COOLDOWN: Duration = Duration::from_millis(900);

const CLAP_THRESHOLD: f32 = 0.12;
const CLAP_TIMEOUT: Duration = Duration::from_millis(1500);
const CLAP_COOLDOWN: Duration = Duration::from_millis(300);
const AFTER_CLAP_WINDOW: Duration = Duration::from_millis(2500);
const SHUTDOWN_DOWN_THRESHOLD: f32 = 0.22;
const SHUTDOWN_COOLDOWN: Duration = Duration::from_secs(5);

const MOUSE_REGION: (f32, f32, f32, f32) = (0.1, 0.1, 0.9, 0.7); // x_min, y_min, x_max, y_max
const CURSOR_GAIN: f32 = 3.5; // >1 increases sensitivity; movement is amplified around the center
const CURSOR_ALPHA: f32 = 0.22; // exponential moving average factor (lower = smoother)
const CURSOR_MIN_MOVE_PX: f32 = 1.5; // ignore sub-pixel jitter

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
// Helpers
// ---------------------------------------------------------------------------
fn dist(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

fn hand_span(lm: &[[f32; 3]]) -> f32 {
    let mut min_x = 1.0f32;
    let mut max_x = 0.0f32;
    let mut min_y = 1.0f32;
    let mut max_y = 0.0f32;
    for p in lm {
        min_x = min_x.min(p[0]);
        max_x = max_x.max(p[0]);
        min_y = min_y.min(p[1]);
        max_y = max_y.max(p[1]);
    }
    (max_x - min_x).max(max_y - min_y)
}

fn fingers_up(lm: &[[f32; 3]]) -> [bool; 5] {
    let tips = [8, 12, 16, 20];
    let pips = [6, 10, 14, 18];
    let mut up = [false; 5];

    // Thumb heuristic: compare distance from wrist (0) to tip vs wrist to IP (3)
    let wrist = lm[0];
    let thumb_tip = lm[4];
    let thumb_ip = lm[3];
    up[0] = dist(&wrist, &thumb_tip) > dist(&wrist, &thumb_ip);

    for i in 0..4 {
        up[i + 1] = lm[tips[i]][1] < lm[pips[i]][1]; // y grows downward
    }
    up
}

fn is_open_hand(fingers: &[bool; 5]) -> bool {
    fingers.iter().all(|&x| x)
}

fn is_fist(fingers: &[bool; 5]) -> bool {
    fingers.iter().all(|&x| !x)
}

fn is_index_pointing(fingers: &[bool; 5]) -> bool {
    // Index finger extended, but hand is not a full open palm.
    fingers[1] && !is_open_hand(fingers)
}

fn is_two_fingers(fingers: &[bool; 5]) -> bool {
    // Index and middle fingers extended, ring and pinky closed.
    fingers[1] && fingers[2] && !fingers[3] && !fingers[4]
}

fn palm_centroid(lm: &[[f32; 3]]) -> [f32; 2] {
    let mut sum = [0.0; 2];
    for i in [0, 5, 17] {
        sum[0] += lm[i][0];
        sum[1] += lm[i][1];
    }
    sum[0] /= 3.0;
    sum[1] /= 3.0;
    sum
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
    cursor_ema: Option<[f32; 2]>,
    virtual_input: Option<input::VirtualInput>,

    click_ready: bool,
    click_timer: Instant,

    last_volume_flip: Instant,

    palm_positions: Vec<[f32; 2]>,
    palm_times: Vec<Instant>,
    last_swipe: Instant,
    last_fist: Instant,

    last_handedness: Option<String>,
    last_workspace_switch: Instant,
    last_two_finger_y: Option<f32>,

    clap_count: u32,
    last_clap_time: Instant,
    clap_together: bool,
    shutdown_window_until: Option<Instant>,
    right_hand_start_y: Option<f32>,
    last_shutdown: Option<Instant>,

    dry_run: bool,
    debug: bool,
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
            cursor_ema: None,
            virtual_input,
            click_ready: true,
            click_timer: Instant::now(),
            last_volume_flip: Instant::now(),
            palm_positions: Vec::with_capacity(10),
            palm_times: Vec::with_capacity(10),
            last_swipe: Instant::now(),
            last_fist: Instant::now(),

            last_handedness: None,
            last_workspace_switch: Instant::now(),
            last_two_finger_y: None,

            clap_count: 0,
            last_clap_time: Instant::now(),
            clap_together: false,
            shutdown_window_until: None,
            right_hand_start_y: None,
            last_shutdown: None,

            dry_run,
            debug,
        })
    }

    fn process(
        &mut self,
        lm: &[[f32; 3]],
        handedness: Option<&str>,
    ) -> Result<()> {
        let span = hand_span(lm);
        if span < MIN_HAND_SPAN {
            // Ignore spurious/tiny detections that are not a real hand.
            return Ok(());
        }

        let fingers = fingers_up(lm);
        let now = Instant::now();
        let palm = dist(&lm[0], &lm[9]);
        if self.debug {
            eprintln!(
                "[STATE] span={:.3} palm={:.3} fingers={:?} open={} fist={} pointing={}",
                span,
                palm,
                fingers,
                is_open_hand(&fingers),
                is_fist(&fingers),
                is_index_pointing(&fingers)
            );
        }

        // Pinch clicks (thumb + index / thumb + middle) work regardless of
        // whether the hand is classified as open, so they can be used while
        // pointing or with an open palm.
        let d_left = dist(&lm[4], &lm[8]);
        let d_right = dist(&lm[4], &lm[12]);
        if self.click_ready {
            if d_left < CLICK_THRESHOLD && d_left < palm * CLICK_THRESHOLD_REL {
                self.do_click(Button::Left, "left click", now)?;
            } else if d_right < CLICK_THRESHOLD && d_right < palm * CLICK_THRESHOLD_REL {
                self.do_click(Button::Right, "right click", now)?;
            }
        }

        if !self.click_ready && now.duration_since(self.click_timer) > CLICK_COOLDOWN {
            self.click_ready = true;
        }

        // Volume is handled by vertical flips in detect_swipe_or_fist().

        // Cursor: point index finger (an open palm is reserved for swipes/volume flips).
        if is_two_fingers(&fingers) {
            self.handle_scroll(lm)?;
        } else {
            self.last_two_finger_y = None;
        }

        if is_index_pointing(&fingers) && !is_two_fingers(&fingers) {
            self.move_cursor(lm)?;
        }

        // Swipes / fist
        self.detect_swipe_or_fist(lm, now)?;

        // Workspace switch on hand flip (detected by handedness change).
        if let Some(h) = handedness
            && now.duration_since(self.last_workspace_switch) > WORKSPACE_COOLDOWN
        {
            match (self.last_handedness.as_deref(), h) {
                (Some("Right"), "Left") => {
                    self.switch_workspace("left")?;
                    self.last_workspace_switch = now;
                }
                (Some("Left"), "Right") => {
                    self.switch_workspace("right")?;
                    self.last_workspace_switch = now;
                }
                _ => {}
            }
            self.last_handedness = Some(h.to_string());
        }

        Ok(())
    }

    fn do_click(&mut self, button: Button, name: &str, now: Instant) -> Result<()> {
        if self.dry_run {
            println!("[GESTURE] {}", name);
        } else {
            self.button_click(button)
                .with_context(|| format!("{} failed", name))?;
        }
        self.click_ready = false;
        self.click_timer = now;
        Ok(())
    }

    fn handle_scroll(&mut self, lm: &[[f32; 3]]) -> Result<()> {
        // Average Y position of index and middle finger tips
        let current_y = (lm[8][1] + lm[12][1]) / 2.0;

        if let Some(last_y) = self.last_two_finger_y {
            let dy = current_y - last_y;
            // Scroll threshold and multiplier
            let scroll_threshold = 0.02; 
            if dy.abs() > scroll_threshold {
                let scroll_amount = if dy > 0.0 { -1 } else { 1 }; // Invert for natural scrolling
                if !self.dry_run {
                    self.scroll_mouse(scroll_amount)?;
                } else {
                    println!("[GESTURE] scroll {}", scroll_amount);
                }
                // Only update last_y when a scroll is triggered to accumulate small movements
                self.last_two_finger_y = Some(current_y);
            }
        } else {
            self.last_two_finger_y = Some(current_y);
        }

        Ok(())
    }

    fn move_cursor(&mut self, lm: &[[f32; 3]]) -> Result<()> {
        let tip = lm[8];
        let (rx_min, ry_min, rx_max, ry_max) = MOUSE_REGION;

        let dx = ((tip[0] - rx_min) / (rx_max - rx_min)).clamp(0.0, 1.0);
        let dy = ((tip[1] - ry_min) / (ry_max - ry_min)).clamp(0.0, 1.0);

        // Amplify movement around the center to make the cursor more sensitive.
        let dx = ((dx - 0.5) * CURSOR_GAIN + 0.5).clamp(0.0, 1.0);
        let dy = ((dy - 0.5) * CURSOR_GAIN + 0.5).clamp(0.0, 1.0);

        let raw_x = dx * self.screen_size.0 as f32;
        let raw_y = dy * self.screen_size.1 as f32;

        let (smooth_x, smooth_y) = match self.cursor_ema {
            Some([ex, ey]) => {
                let dx = (raw_x - ex).abs();
                let dy = (raw_y - ey).abs();
                if dx < CURSOR_MIN_MOVE_PX && dy < CURSOR_MIN_MOVE_PX {
                    // Ignore tiny movements from hand shake.
                    (ex, ey)
                } else {
                    // Low-pass exponential moving average.
                    (
                        CURSOR_ALPHA * raw_x + (1.0 - CURSOR_ALPHA) * ex,
                        CURSOR_ALPHA * raw_y + (1.0 - CURSOR_ALPHA) * ey,
                    )
                }
            }
            None => (raw_x, raw_y),
        };

        self.cursor_ema = Some([smooth_x, smooth_y]);
        let avg_x = smooth_x as i32;
        let avg_y = smooth_y as i32;

        if self.debug {
            eprintln!("[CURSOR] target={:.3},{:.3} screen={}x{}", tip[0], tip[1], avg_x, avg_y);
        }
        if !self.dry_run {
            self.move_mouse(avg_x, avg_y)?;
        }
        Ok(())
    }

    fn detect_swipe_or_fist(&mut self, lm: &[[f32; 3]], now: Instant) -> Result<()> {
        let fingers = fingers_up(lm);
        let c = palm_centroid(lm);
        self.palm_positions.push(c);
        self.palm_times.push(now);
        if self.palm_positions.len() > 10 {
            self.palm_positions.remove(0);
            self.palm_times.remove(0);
        }

        if is_fist(&fingers) && now.duration_since(self.last_fist) > FIST_COOLDOWN {
            if self.dry_run {
                println!("[GESTURE] close tab");
            } else {
                // Ctrl+W
                self.key_combo(&[EvKey::KEY_LEFTCTRL], EvKey::KEY_W)?;
            }
            self.last_fist = now;
            self.palm_positions.clear();
            self.palm_times.clear();
            return Ok(());
        }

        if !is_open_hand(&fingers) || self.palm_positions.len() < 3 {
            return Ok(());
        }

        let xs: Vec<f32> = self.palm_positions.iter().map(|p| p[0]).collect();
        let ys: Vec<f32> = self.palm_positions.iter().map(|p| p[1]).collect();
        let dt = self.palm_times.last().unwrap().duration_since(self.palm_times[0]).as_secs_f32();
        if dt == 0.0 {
            return Ok(());
        }
        let vx = (xs.last().unwrap() - xs.first().unwrap()) / dt;
        let vy = (ys.last().unwrap() - ys.first().unwrap()) / dt;

        // Vertical flip -> volume step. Require the motion to be predominantly
        // vertical so it does not steal horizontal swipes.
        if now.duration_since(self.last_volume_flip) > VOLUME_FLIP_COOLDOWN && vy.abs() > vx.abs() {
            if vy < -FLIP_VEL {
                self.volume_flip(true)?; // flip up -> volume up
                self.last_volume_flip = now;
                self.palm_positions.clear();
                self.palm_times.clear();
                return Ok(());
            } else if vy > FLIP_VEL {
                self.volume_flip(false)?; // flip down -> volume down
                self.last_volume_flip = now;
                self.palm_positions.clear();
                self.palm_times.clear();
                return Ok(());
            }
        }

        if now.duration_since(self.last_swipe) > SWIPE_COOLDOWN {
            if vx > SWIPE_VEL {
                self.switch_tab("right")?;
                self.last_swipe = now;
            } else if vx < -SWIPE_VEL {
                self.switch_tab("left")?;
                self.last_swipe = now;
            }
        }

        Ok(())
    }

    fn volume_flip(&mut self, up: bool) -> Result<()> {
        if self.dry_run {
            println!("[GESTURE] volume {}", if up { "up" } else { "down" });
            return Ok(());
        }
        adjust_volume(up, VOLUME_STEP)
    }

    fn switch_tab(&mut self, direction: &str) -> Result<()> {
        if self.dry_run {
            println!("[GESTURE] switch tab {}", direction);
            return Ok(());
        }
        if direction == "right" {
            self.key_combo(&[EvKey::KEY_LEFTCTRL], EvKey::KEY_TAB)?;
        } else {
            self.key_combo(
                &[EvKey::KEY_LEFTCTRL, EvKey::KEY_LEFTSHIFT],
                EvKey::KEY_TAB,
            )?;
        }
        Ok(())
    }

    fn switch_workspace(&mut self, direction: &str) -> Result<()> {
        if self.dry_run {
            println!("[GESTURE] switch workspace {}", direction);
            return Ok(());
        }
        if let Some(vi) = &mut self.virtual_input {
            let key = match direction {
                "left" => EvKey::KEY_LEFT,
                _ => EvKey::KEY_RIGHT,
            };
            return vi.key_combo(&[EvKey::KEY_LEFTCTRL, EvKey::KEY_LEFTALT], key);
        }

        // Fallback to compositor/desktop-specific commands.
        if command_exists("hyprctl") {
            let delta = if direction == "left" { "-1" } else { "+1" };
            return Command::new("hyprctl")
                .args(["dispatch", "workspace", delta])
                .status()
                .context("hyprctl failed")
                .map(|_| ());
        }
        if command_exists("swaymsg") {
            let cmd = if direction == "left" {
                "workspace prev_on_output"
            } else {
                "workspace next_on_output"
            };
            return Command::new("swaymsg")
                .args([cmd])
                .status()
                .context("swaymsg failed")
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
        anyhow::bail!("no workspace-switching backend found (tried uinput, hyprctl, swaymsg and xdotool)");
    }

    fn check_shutdown_gesture(&mut self, hands: &[Hand]) -> Result<()> {
        let now = Instant::now();
        if let Some(t) = self.last_shutdown && now.duration_since(t) < SHUTDOWN_COOLDOWN {
            return Ok(());
        }

        // Reset clap count if the window is closed and too much time passed.
        if self.shutdown_window_until.is_none()
            && self.clap_count > 0
            && now.duration_since(self.last_clap_time) > CLAP_TIMEOUT
        {
            self.clap_count = 0;
        }

        // Expire the shutdown window.
        if let Some(until) = self.shutdown_window_until {
            if now > until {
                self.shutdown_window_until = None;
                self.right_hand_start_y = None;
                self.clap_count = 0;
            } else if self.right_hand_start_y.is_none() {
                // Capture baseline right-hand wrist Y when the window opens.
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

        // Count a fresh clap event.
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
        let valid: Vec<_> = hands
            .iter()
            .filter(|h| h.landmarks.len() >= 21)
            .collect();
        if valid.len() < 2 {
            return false;
        }
        let w1 = &valid[0].landmarks[0];
        let w2 = &valid[1].landmarks[0];
        let d = dist(w1, w2);
        d < CLAP_THRESHOLD
    }

    fn trigger_shutdown(&mut self) -> Result<()> {
        if self.dry_run {
            println!("[GESTURE] shutdown (dry-run, not executing)");
            self.last_shutdown = Some(Instant::now());
            self.shutdown_window_until = None;
            self.right_hand_start_y = None;
            self.clap_count = 0;
            return Ok(());
        }

        eprintln!("!!! SHUTDOWN GESTURE DETECTED !!!");
        eprintln!("Powering off in 2 seconds... (Ctrl+C to cancel)");
        std::thread::sleep(Duration::from_secs(2));

        self.last_shutdown = Some(Instant::now());
        self.shutdown_window_until = None;
        self.right_hand_start_y = None;
        self.clap_count = 0;

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

    fn move_mouse(&mut self, x: i32, y: i32) -> Result<()> {
        if let Some(vi) = &mut self.virtual_input {
            vi.move_mouse(x, y)
        } else {
            self.enigo
                .move_mouse(x, y, Coordinate::Abs)
                .context("mouse move failed")
        }
    }

    fn scroll_mouse(&mut self, y: i32) -> Result<()> {
        if let Some(vi) = &mut self.virtual_input {
            vi.scroll(y)
        } else {
            self.enigo
                .scroll(y, Axis::Vertical)
                .context("mouse scroll failed")
        }
    }

    fn button_click(&mut self, button: Button) -> Result<()> {
        if let Some(vi) = &mut self.virtual_input {
            vi.button_click(button)
        } else {
            self.enigo
                .button(button, Direction::Click)
                .context("button click failed")
        }
    }

    fn key_combo(&mut self, modifiers: &[EvKey], key: EvKey) -> Result<()> {
        if let Some(vi) = &mut self.virtual_input {
            vi.key_combo(modifiers, key)
        } else {
            for m in modifiers {
                let enigo_key = evdev_to_enigo_key(*m)
                    .with_context(|| format!("unsupported modifier for Enigo fallback: {:?}", m))?;
                self.enigo.key(enigo_key, Direction::Press)?;
            }
            let enigo_key = evdev_to_enigo_key(key)
                .with_context(|| format!("unsupported key for Enigo fallback: {:?}", key))?;
            self.enigo.key(enigo_key, Direction::Click)?;
            for m in modifiers.iter().rev() {
                let enigo_key = evdev_to_enigo_key(*m)
                    .with_context(|| format!("unsupported modifier for Enigo fallback: {:?}", m))?;
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
    let camera = args
        .iter()
        .position(|a| a == "--camera")
        .and_then(|i| args.get(i + 1))
        .unwrap_or(&"0".to_string())
        .clone();

    let python = std::env::var("AIR_MOUSE_PYTHON")
        .unwrap_or_else(|_| "/home/dranzer/hand-control/.venv/bin/python".into());
    let mut child = Command::new(python)
        .arg("tracker.py")
        .arg("--camera")
        .arg(&camera)
        .stdout(Stdio::piped())
        .spawn()
        .context("failed to start tracker.py")?;

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

        // Use the right hand for normal control if available, otherwise any detected hand.
        // Pause normal gestures while the shutdown sequence is waiting for the hand drop.
        if ctrl.shutdown_window_until.is_none() {
            let primary = frame
                .hands
                .iter()
                .find(|h| h.handedness.as_deref() == Some("Right"))
                .or_else(|| frame.hands.first());

            if let Some(hand) = primary {
                let lm = &hand.landmarks;
                if lm.len() >= 21 && let Err(e) = ctrl.process(lm, hand.handedness.as_deref()) {
                    eprintln!("process error: {e:?}");
                }
            }
        }
    }

    let _ = child.kill();
    Ok(())
}
