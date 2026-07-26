use anyhow::{Context, Result};
use enigo::Button;
use evdev::{
    uinput::VirtualDeviceBuilder, AbsInfo, AbsoluteAxisType, AttributeSet, EventType, InputEvent,
    Key as EvKey, Synchronization, UinputAbsSetup,
};

/// Native Linux virtual input device (uinput). Works on both X11 and Wayland,
/// as long as the user has write access to `/dev/uinput` (usually via the
/// `input` group).
pub struct VirtualInput {
    device: evdev::uinput::VirtualDevice,
}

impl VirtualInput {
    pub fn new(name: &str, width: i32, height: i32) -> Result<Self> {
        let mut keys = AttributeSet::<EvKey>::new();
        keys.insert(EvKey::BTN_LEFT);
        keys.insert(EvKey::BTN_RIGHT);
        keys.insert(EvKey::KEY_LEFTCTRL);
        keys.insert(EvKey::KEY_LEFTSHIFT);
        keys.insert(EvKey::KEY_LEFTALT);
        keys.insert(EvKey::KEY_TAB);
        keys.insert(EvKey::KEY_W);
        keys.insert(EvKey::KEY_LEFT);
        keys.insert(EvKey::KEY_RIGHT);

        let abs_x = UinputAbsSetup::new(
            AbsoluteAxisType::ABS_X,
            AbsInfo::new(0, 0, width, 0, 0, 1),
        );
        let abs_y = UinputAbsSetup::new(
            AbsoluteAxisType::ABS_Y,
            AbsInfo::new(0, 0, height, 0, 0, 1),
        );

        let device = VirtualDeviceBuilder::new()
            .context("/dev/uinput not available; are you in the 'input' group?")?
            .name(name)
            .with_absolute_axis(&abs_x)
            .context("failed to register ABS_X axis")?
            .with_absolute_axis(&abs_y)
            .context("failed to register ABS_Y axis")?
            .with_keys(&keys)
            .context("failed to register keys")?
            .build()
            .context("failed to create uinput device")?;

        Ok(Self { device })
    }

    pub fn move_mouse(&mut self, x: i32, y: i32) -> Result<()> {
        self.emit(&[
            InputEvent::new(EventType::ABSOLUTE, AbsoluteAxisType::ABS_X.0, x),
            InputEvent::new(EventType::ABSOLUTE, AbsoluteAxisType::ABS_Y.0, y),
            InputEvent::new(EventType::SYNCHRONIZATION, Synchronization::SYN_REPORT.0, 0),
        ])
    }

    pub fn button_click(&mut self, button: Button) -> Result<()> {
        let code = match button {
            Button::Left => EvKey::BTN_LEFT,
            Button::Right => EvKey::BTN_RIGHT,
            Button::Middle => EvKey::BTN_MIDDLE,
            Button::Back => EvKey::BTN_BACK,
            Button::Forward => EvKey::BTN_FORWARD,
            _ => EvKey::BTN_LEFT,
        };
        self.key_event(code, true)?;
        self.key_event(code, false)
    }

    /// Press modifier(s) + key and release them in reverse order.
    pub fn key_combo(&mut self, modifiers: &[EvKey], key: EvKey) -> Result<()> {
        for m in modifiers {
            self.key_event(*m, true)?;
        }
        self.key_event(key, true)?;
        self.key_event(key, false)?;
        for m in modifiers.iter().rev() {
            self.key_event(*m, false)?;
        }
        Ok(())
    }

    fn key_event(&mut self, code: EvKey, pressed: bool) -> Result<()> {
        self.emit(&[
            InputEvent::new(EventType::KEY, code.code(), i32::from(pressed)),
            InputEvent::new(EventType::SYNCHRONIZATION, Synchronization::SYN_REPORT.0, 0),
        ])
    }

    fn emit(&mut self, events: &[InputEvent]) -> Result<()> {
        self.device.emit(events).context("failed to emit uinput event")
    }
}
