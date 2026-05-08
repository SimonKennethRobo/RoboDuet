"""
JoyController: manages a ZMQ joystick subscriber in a background thread.

This is a pure-Python helper with no Isaac Gym dependencies.  The gym
environment wrapper (JoyWrapper) lives in __init__.py and uses this class.

Standalone usage::

    ctrl = JoyController("/path/to/joy_mapping.yaml")
    # ... in your loop:
    cmds = ctrl.resolve_commands(current_dog_cmds, current_arm_cmds)

YAML format — see config/joy_mapping.yaml for the full example.
"""

import json
import os
import threading
import yaml

# ---------------------------------------------------------------------------
# Default channel table: logical name -> (target, index)
# target "dog" -> commands_dog, "arm" -> commands_arm
# ---------------------------------------------------------------------------
DEFAULT_CHANNELS: dict = {
    "x_vel":        {"target": "dog", "index": 0},
    "y_vel":        {"target": "dog", "index": 1},
    "yaw_vel":      {"target": "dog", "index": 2},
    "arm_x":   {"target": "arm", "index": 0},
    "arm_z":    {"target": "arm", "index": 1},
    "arm_y":      {"target": "arm", "index": 2},
    "arm_roll":     {"target": "arm", "index": 3},
    "arm_pitch": {"target": "arm", "index": 4},
    "arm_ee_yaw":   {"target": "arm", "index": 5},
}

_DEFAULT_CONFIG_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__),
                 "..", "..", "..", "config", "joy_mapping.yaml")
)


class JoyController:
    """Subscribes to a ZMQ joystick stream and provides parsed command deltas.

    Parameters
    ----------
    config_path:
        Path to the YAML mapping file.  Defaults to config/joy_mapping.yaml
        at the repository root.
    """

    def __init__(self, config_path: str = None):
        path = config_path or _DEFAULT_CONFIG_PATH
        self._load_config(path)

        self._axes: list = []
        self._buttons: list = []
        self._lock = threading.Lock()
        self._zmq_thread: threading.Thread = None

        self._start_zmq_subscriber()

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def _load_config(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self._cfg = cfg
        self._axes_cfg = {int(k): v for k, v in cfg.get("axes", {}).items()}
        self._buttons_cfg = {int(k): v for k, v in cfg.get("buttons", {}).items()}
        self._channels: dict = {**DEFAULT_CHANNELS, **cfg.get("channels", {})}
        self._dpad_step_cfg = cfg.get("dpad_step", {})

    # ------------------------------------------------------------------
    # ZMQ
    # ------------------------------------------------------------------

    def _start_zmq_subscriber(self):
        try:
            import zmq
        except ImportError:
            print("[JoyController] pyzmq not available — joystick input disabled.")
            return

        zmq_cfg = self._cfg.get("zmq", {})
        endpoint = zmq_cfg.get("endpoint", "ipc:///tmp/roboduet_joy.sock")
        topic = zmq_cfg.get("topic", "joy")

        def _listen():
            ctx = zmq.Context.instance()
            sock = ctx.socket(zmq.SUB)
            sock.setsockopt_string(zmq.SUBSCRIBE, topic)
            sock.connect(endpoint)

            poller = zmq.Poller()
            poller.register(sock, zmq.POLLIN)

            while True:
                events = dict(poller.poll(1000))
                if sock not in events:
                    continue
                try:
                    parts = sock.recv_multipart(flags=zmq.NOBLOCK)
                except zmq.Again:
                    continue

                if not parts:
                    continue
                payload = parts[-1]
                try:
                    data = json.loads(payload.decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    continue

                axes = data.get("axes", [])
                buttons = data.get("buttons", [])

                with self._lock:
                    self._axes = list(axes)
                    self._buttons = list(buttons)

        self._zmq_thread = threading.Thread(
            target=_listen, name="joy_zmq", daemon=True
        )
        self._zmq_thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def connected(self) -> bool:
        """True once at least one /joy message has been received."""
        with self._lock:
            return bool(self._axes or self._buttons)

    def get_raw(self):
        """Return (axes, buttons) as plain lists (thread-safe snapshot)."""
        with self._lock:
            return list(self._axes), list(self._buttons)

    def resolve_commands(
        self,
        dog_cmds: list,
        arm_cmds: list,
    ):
        """Apply joystick state to mutable command lists.

        Parameters
        ----------
        dog_cmds:
            Mutable list/array of dog command values (at least 3 elements).
        arm_cmds:
            Mutable list/array of arm command values (at least 6 elements).

        Returns
        -------
        action : str or None
            "reset" if the reset button was pressed, else None.
        """
        axes, buttons = self.get_raw()
        if not axes and not buttons:
            return None

        action = None
        combo_delta, combo_step = self._combo_delta(axes)

        # ---- Axes ----
        for idx, cfg in self._axes_cfg.items():
            if idx >= len(axes):
                continue
            raw = self._deadzone(axes[idx], cfg.get("deadzone", 0.0))
            value = raw * cfg.get("scale", 1.0) + cfg.get("offset", 0.0)
            clamp = cfg.get("clamp")
            if clamp:
                value = max(clamp[0], min(clamp[1], value))
            channel = cfg.get("channel")
            self._write(channel, value, dog_cmds, arm_cmds)

        # ---- Buttons ----
        for idx, cfg in self._buttons_cfg.items():
            if idx >= len(buttons) or not buttons[idx]:
                continue
            btn_action = cfg.get("action", "")
            step = float(cfg.get("step", 0.05))
            sign = float(cfg.get("sign", 1.0))
            clamp = cfg.get("clamp", [-1.5, 1.5])

            if btn_action == "reset":
                action = "reset"
            elif btn_action == "arm_roll_left":
                arm_cmds[3] = max(-1.5, min(1.5, arm_cmds[3] + step))
            elif btn_action == "arm_roll_right":
                arm_cmds[3] = max(-1.5, min(1.5, arm_cmds[3] - step))
            elif btn_action == "arm_pitch_up":
                arm_cmds[4] = max(-1.5, min(1.5, arm_cmds[4] + step))
            elif btn_action == "arm_pitch_down":
                arm_cmds[4] = max(-1.5, min(1.5, arm_cmds[4] - step))
            elif btn_action == "arm_yaw_left":
                arm_cmds[5] = max(-1.5, min(1.5, arm_cmds[5] + step))
            elif btn_action == "arm_yaw_right":
                arm_cmds[5] = max(-1.5, min(1.5, arm_cmds[5] - step))
            elif btn_action == "arm_roll":
                delta = self._scale_combo_delta(combo_delta, combo_step, step)
                if delta != 0.0:
                    arm_cmds[3] = max(clamp[0], min(clamp[1], arm_cmds[3] + delta * sign))
            elif btn_action == "arm_pitch":
                delta = self._scale_combo_delta(combo_delta, combo_step, step)
                if delta != 0.0:
                    arm_cmds[4] = max(clamp[0], min(clamp[1], arm_cmds[4] + delta * sign))
            elif btn_action == "arm_yaw":
                delta = self._scale_combo_delta(combo_delta, combo_step, step)
                if delta != 0.0:
                    arm_cmds[5] = max(clamp[0], min(clamp[1], arm_cmds[5] + delta * sign))

        self._apply_dpad_steps(axes, dog_cmds, arm_cmds, combo_delta)

        return action

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _deadzone(value: float, dz: float) -> float:
        if abs(value) < dz:
            return 0.0
        sign = 1.0 if value > 0 else -1.0
        return sign * (abs(value) - dz) / (1.0 - dz + 1e-9)

    def _write(self, channel: str, value: float, dog: list, arm: list):
        if not channel or channel not in self._channels:
            return
        ch = self._channels[channel]
        target, idx = ch["target"], ch["index"]
        if target == "dog" and idx < len(dog):
            dog[idx] = value
        elif target == "arm" and idx < len(arm):
            arm[idx] = value

    def _read(self, channel: str, dog: list, arm: list) -> float:
        if not channel or channel not in self._channels:
            return 0.0
        ch = self._channels[channel]
        target, idx = ch["target"], ch["index"]
        if target == "dog" and idx < len(dog):
            return float(dog[idx])
        if target == "arm" and idx < len(arm):
            return float(arm[idx])
        return 0.0

    def _apply_dpad_steps(
        self,
        axes: list,
        dog_cmds: list,
        arm_cmds: list,
        combo_delta: float,
    ):
        cfg = self._dpad_step_cfg
        if not cfg:
            return
        if abs(combo_delta) <= 1e-6:
            return
        h_idx = int(cfg.get("horizontal_axis", 6))
        v_idx = int(cfg.get("vertical_axis", 7))
        threshold = float(cfg.get("threshold", 0.5))
        x_range = cfg.get("arm_x_range", [-1.0, 1.0])
        y_range = cfg.get("arm_y_range", [-1.0, 1.0])
        z_range = cfg.get("arm_z_range", [-1.0, 1.0])

        def _axis(idx):
            return axes[idx] if idx < len(axes) else 0.0

        dpad_h = _axis(h_idx)
        dpad_v = _axis(v_idx)

        if dpad_v < -threshold:
            value = self._read("arm_x", dog_cmds, arm_cmds) + combo_delta
            value = max(x_range[0], min(x_range[1], value))
            self._write("arm_x", value, dog_cmds, arm_cmds)

        if dpad_h < -threshold:
            value = self._read("arm_y", dog_cmds, arm_cmds) + combo_delta
            value = max(y_range[0], min(y_range[1], value))
            self._write("arm_y", value, dog_cmds, arm_cmds)

        if dpad_v > threshold:
            value = self._read("arm_z", dog_cmds, arm_cmds) + combo_delta
            value = max(z_range[0], min(z_range[1], value))
            self._write("arm_z", value, dog_cmds, arm_cmds)

    def _combo_delta(self, axes: list) -> tuple:
        cfg = self._dpad_step_cfg
        if not cfg:
            return 0.0, 0.0
        lt_idx = int(cfg.get("lt_axis", 6))
        rt_idx = int(cfg.get("rt_axis", 5))
        step = float(cfg.get("step", 0.05))
        mode = cfg.get("trigger_mode", "zero_to_one")
        threshold = float(cfg.get("trigger_threshold", 0.1))

        def _axis(idx):
            return axes[idx] if idx < len(axes) else 0.0

        def _trigger_mag(value: float) -> float:
            if mode == "one_to_minus_one":
                # rest=+1, pressed=-1
                return max(0.0, min(1.0, (1.0 - value) / 2.0))
            # default: rest=0, pressed=+1
            return max(0.0, min(1.0, value))

        lt_mag = _trigger_mag(_axis(lt_idx))
        rt_mag = _trigger_mag(_axis(rt_idx))
        if max(rt_mag, lt_mag) < threshold:
            return 0.0, step
        delta = step * (rt_mag - lt_mag)
        return delta, step

    @staticmethod
    def _scale_combo_delta(combo_delta: float, combo_step: float, step: float) -> float:
        if abs(combo_delta) <= 1e-6:
            return 0.0
        if combo_step > 0.0:
            return combo_delta * (step / combo_step)
        return combo_delta
