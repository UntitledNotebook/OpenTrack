import struct
import threading
import os
import time

try:
    from pynput import keyboard
    _PYNPUT_IMPORT_ERROR = None
except ImportError as exc:
    keyboard = None
    _PYNPUT_IMPORT_ERROR = exc


# Digital buttons whose pulse should be latched for at least
# SIM_BUTTON_LATCH_MS milliseconds, so that a 50Hz consumer
# (e.g. G1_deploy ControlStep) cannot miss a short key tap.
LATCHED_BUTTONS = {
    "L1", "L2", "R1", "R2",
    "A", "B", "X", "Y",
    "Up", "Down", "Left", "Right",
    "Select", "F1", "F3", "Start",
    "Reset", "SimStart",
}


def _now_ms():
    return int(time.monotonic() * 1000)


class UnitreeRemoteController:
    def __init__(self):
        # Keep controller state synchronization independent from the simulation lock.
        # Otherwise, the 500Hz simulation thread can starve keyboard callbacks.
        self.buttons_lock = threading.Lock()
        # key
        self.Lx = 0
        self.Rx = 0
        self.Ry = 0
        self.Ly = 0

        # button
        self.buttons = {
            "L1": 0,
            "L2": 0,
            "R1": 0,
            "R2": 0,
            "A": 0,
            "B": 0,
            "X": 0,
            "Y": 0,
            "Up": 0,
            "Down": 0,
            "Left": 0,
            "Right": 0,
            "Select": 0,
            "F1": 0,
            "F3": 0,
            "Start": 0,
            "Reset": 0,
            "Lx+": 0,
            "Lx-": 0,
            "Ly+": 0,
            "Ly-": 0,
            "Rx+": 0,
            "Rx-": 0,
            "Ry+": 0,
            "Ry-": 0,
            "SimStart": 0,
        }

        # For each latched button, store the monotonic-time (ms) at which
        # we are allowed to drop it back to 0. While the key is physically
        # held, release_after is repeatedly updated to "now+latch_ms" so the
        # state stays at 1 continuously.
        self.release_after_ms = {b: 0 for b in LATCHED_BUTTONS}

        try:
            self.latch_ms = max(0, int(os.getenv("SIM_BUTTON_LATCH_MS", "140")))
        except ValueError:
            self.latch_ms = 140

        # Map normalized key tokens to button names.
        # Supported token types:
        #   - character: "4", "a", "x", ...
        #   - special names: "up", "down", "f5", "end", ...
        #   - vk code: "vk:65460" (useful for numpad on X11)
        #
        # NOTE on MuJoCo viewer hotkey overlap (per-user choice to keep '9' and 'r'):
        #   1-9, 0  : toggle vis flags
        #   r       : reset to keyframe 0  (also resets MuJoCo qpos!)
        #   Space   : pause physics; Tab: ui panel; Bksp: reset viewport
        # User explicitly wants '9' as SimStart and 'r' as Reset.
        # If viewer steals the key, focus the terminal window before pressing,
        # or use the F5 / F8 / End fallbacks.
        self.key_token_mapping = {
            "1": ["L1"],
            "2": ["L2"],
            "3": ["R1"],
            "4": ["R2"],
            "5": ["Select"],
            "6": ["F1"],
            "7": ["F3"],
            "8": ["Start"],
            "a": ["A"],
            "b": ["B"],
            "x": ["X"],
            "y": ["Y"],
            "j": ["Lx-"],
            "l": ["Lx+"],
            "k": ["Ly-"],
            "i": ["Ly+"],
            "u": ["Rx-"],
            "o": ["Rx+"],
            "up": ["Up"],
            "down": ["Down"],
            "left": ["Left"],
            "right": ["Right"],
            # SimStart: primary binding is '9' (per user preference); extras for safety.
            "9": ["SimStart"],
            "f5": ["SimStart"],
            "`": ["SimStart"],
            # Reset: primary binding is 'r' (per user preference); extras retained.
            "r": ["Reset"],
            "f8": ["Reset"],
            "end": ["Reset"],
            # X11 keypad keysyms.
            "vk:65457": ["L1"],   # KP_1
            "vk:65458": ["L2"],   # KP_2
            "vk:65459": ["R1"],   # KP_3
            "vk:65460": ["R2"],   # KP_4
            "vk:65461": ["Select"],  # KP_5
            "vk:65462": ["F1"],   # KP_6
            "vk:65463": ["F3"],   # KP_7
            "vk:65464": ["Start"], # KP_8
            "vk:65465": ["SimStart"], # KP_9
            "vk:65456": ["Reset"],    # KP_0
        }
        self.key_debug = os.getenv("SIM_KEY_DEBUG", "0") == "1"
        self._sim_verbose = os.getenv("SIM_VERBOSE", "").strip().lower() in ("1", "true", "on", "yes")

        # Auto-press scheduler for headless reproduction:
        #   SIM_AUTO_PRESS="1000:R2,3000:A,5000:X,7000:SimStart"
        # The number is the offset in ms after start, the name is a button
        # name from `self.buttons`. Each entry triggers a synthetic press
        # (latched for SIM_BUTTON_LATCH_MS, like a real tap).
        self._auto_press_spec = os.getenv("SIM_AUTO_PRESS", "").strip()
        self._auto_thread = None

    # ------------------------------------------------------------------ keys
    def _key_to_tokens(self, key):
        tokens = set()

        key_name = getattr(key, "name", None)
        if isinstance(key_name, str):
            tokens.add(key_name.lower())

        key_char = getattr(key, "char", None)
        if isinstance(key_char, str) and key_char != "":
            tokens.add(key_char.lower())

        key_vk = getattr(key, "vk", None)
        if key_vk is not None:
            tokens.add(f"vk:{key_vk}")

        return tokens

    def _set_button(self, button_name, pressed, source):
        """Set button value with latching applied for digital buttons."""
        now = _now_ms()
        verbose = self._sim_verbose or self.key_debug
        if pressed:
            self.buttons[button_name] = 1
            if button_name in LATCHED_BUTTONS:
                self.release_after_ms[button_name] = now + self.latch_ms
            if verbose:
                print(f"[SIM_KEY] {button_name} pressed (src={source})")
        else:
            if button_name in LATCHED_BUTTONS:
                # Defer the falling edge until the latch window expires;
                # the periodic latch sweep will clear it.
                self.release_after_ms[button_name] = max(
                    self.release_after_ms[button_name], now + self.latch_ms
                )
                if verbose:
                    print(
                        f"[SIM_KEY] {button_name} release_deferred "
                        f"(src={source}, release_in_ms="
                        f"{self.release_after_ms[button_name] - now})"
                    )
            else:
                self.buttons[button_name] = 0
                if verbose:
                    print(f"[SIM_KEY] {button_name} released (src={source})")

    def _update_buttons_by_key(self, key, pressed, source="kbd"):
        tokens = self._key_to_tokens(key)
        hit = False

        for token in tokens:
            button_names = self.key_token_mapping.get(token, [])
            for button_name in button_names:
                self._set_button(button_name, pressed, source)
                hit = True

        if (self._sim_verbose or self.key_debug) and not hit:
            print(f"[SIM_KEY_DEBUG] Unmapped key={key!r}, tokens={sorted(tokens)}")

    def _sweep_latched(self):
        """Drop latched buttons whose release window has expired."""
        now = _now_ms()
        for b in LATCHED_BUTTONS:
            if self.buttons.get(b, 0) and self.release_after_ms[b] <= now:
                self.buttons[b] = 0
                if self._sim_verbose or self.key_debug:
                    print(f"[SIM_KEY] {b} latch_expired (released)")

    # ----------------------------------------------------------------- input
    def listen_keyboard(self):
        if keyboard is not None:
            self.listener = keyboard.Listener(
                on_press=self.on_press, on_release=self.on_release
            )
            self.listener.start()
        elif not self._auto_press_spec:
            raise RuntimeError(
                "keyboard input is unavailable and SIM_AUTO_PRESS is empty"
            ) from _PYNPUT_IMPORT_ERROR
        else:
            self.listener = None
            print(
                "[SIM_KEY] pynput/X11 unavailable; using headless auto-press only"
            )
        if self._auto_press_spec:
            self._start_auto_press()

    def on_press(self, key):
        with self.buttons_lock:
            self._update_buttons_by_key(key, True)

    def on_release(self, key):
        with self.buttons_lock:
            self._update_buttons_by_key(key, False)

    # --------------------------------------------------------------- queries
    def is_button_pressed(self, button_name):
        with self.buttons_lock:
            self._sweep_latched()
            return self.buttons.get(button_name, 0) == 1

    def encode_botton(self):
        with self.buttons_lock:
            self._sweep_latched()
            buttons = self.buttons.copy()

        wireless_remote = [0 for _ in range(40)]
        lx_offset = 4
        rx_offset = 8
        ry_offset = 12
        ly_offset = 20

        if buttons["Lx+"] == 1:
            cmd = struct.pack("<f", 0.5)
            wireless_remote[lx_offset:lx_offset + 4] = cmd
        if buttons["Lx-"] == 1:
            cmd = struct.pack("<f", -0.5)
            wireless_remote[lx_offset:lx_offset + 4] = cmd
        if buttons["Ly+"] == 1:
            cmd = struct.pack("<f", 0.5)
            wireless_remote[ly_offset:ly_offset + 4] = cmd
        if buttons["Ly-"] == 1:
            cmd = struct.pack("<f", -0.5)
            wireless_remote[ly_offset:ly_offset + 4] = cmd
        if buttons["Rx+"] == 1:
            cmd = struct.pack("<f", 0.5)
            wireless_remote[rx_offset:rx_offset + 4] = cmd
        if buttons["Rx-"] == 1:
            cmd = struct.pack("<f", -0.5)
            wireless_remote[rx_offset:rx_offset + 4] = cmd
        if buttons["Ry+"] == 1:
            cmd = struct.pack("<f", 0.5)
            wireless_remote[ry_offset:ry_offset + 4] = cmd
        if buttons["Ry-"] == 1:
            cmd = struct.pack("<f", -0.5)
            wireless_remote[ry_offset:ry_offset + 4] = cmd

        data1 = 0
        data2 = 0
        data1 |= buttons["R1"] << 0
        data1 |= buttons["L1"] << 1
        data1 |= buttons["Start"] << 2
        data1 |= buttons["Select"] << 3
        data1 |= buttons["R2"] << 4
        data1 |= buttons["L2"] << 5
        data1 |= buttons["F1"] << 6
        data1 |= buttons["F3"] << 7
        data2 |= buttons["A"] << 0
        data2 |= buttons["B"] << 1
        data2 |= buttons["X"] << 2
        data2 |= buttons["Y"] << 3
        data2 |= buttons["Up"] << 4
        data2 |= buttons["Right"] << 5
        data2 |= buttons["Down"] << 6
        data2 |= buttons["Left"] << 7

        wireless_remote[2], wireless_remote[3] = data1, data2
        return wireless_remote

    # --------------------------------------------------------- auto press
    def _start_auto_press(self):
        # Parse "delay_ms:NAME,delay_ms:NAME,..." spec.
        entries = []
        for raw in self._auto_press_spec.split(","):
            raw = raw.strip()
            if not raw:
                continue
            try:
                delay_str, name = raw.split(":", 1)
                delay = int(delay_str)
                name = name.strip()
                if name not in self.buttons:
                    print(f"[SIM_KEY] auto-press: unknown button '{name}', skip")
                    continue
                entries.append((delay, name))
            except ValueError:
                print(f"[SIM_KEY] auto-press: bad entry '{raw}', skip")
                continue
        entries.sort()
        if not entries:
            return

        def _run():
            start = _now_ms()
            print(f"[SIM_KEY] auto-press scheduled: {entries}")
            for delay, name in entries:
                target = start + delay
                while True:
                    remaining = target - _now_ms()
                    if remaining <= 0:
                        break
                    time.sleep(min(0.05, remaining / 1000.0))
                with self.buttons_lock:
                    # Simulate a tap: press then immediate release; latch
                    # logic keeps it visible to the 50Hz consumer.
                    self._update_buttons_by_key(
                        type("Key", (), {"char": None, "name": None, "vk": None})(),
                        True,
                        source=f"auto:{name}",
                    ) if False else self._set_button(name, True, f"auto:{name}")
                time.sleep(0.02)
                with self.buttons_lock:
                    self._set_button(name, False, f"auto:{name}")
            print("[SIM_KEY] auto-press done")

        self._auto_thread = threading.Thread(target=_run, daemon=True)
        self._auto_thread.start()
