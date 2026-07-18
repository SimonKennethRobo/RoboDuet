import argparse
import os
import time
from dataclasses import dataclass

import isaacgym  # noqa: F401 – must be imported before torch
import joylink_client
import torch

from go1_gym.envs import *  # noqa: F403
from go1_gym.envs.config import configure_privileged_obs_dims
from go1_gym.envs.roboduet.wbc_env import WBCEnv
from go1_gym.utils.viz import add_rerun_args, make_rerun_logger
from go1_gym.envs.roboduet.utils import get_play_command_limits
from scripts.load_policy import load_arm_policy, load_dog_policy, load_env

COMMAND_KEYS = {
    "dog": {
        0: "x_vel",
        1: "y_vel",
        2: "yaw_vel",
        # body pose (indices 3-5 always present when n_cmd >= 6)
        3: "body_pitch",
        4: "body_roll",
        5: "body_height_delta",
        # gait params (indices 6-10, only when use_dynamic_gait=True)
        6: "gait_freq",
        7: "footswing_height",
        8: "stance_width",
        9: "stance_length",
        10: "gait_duration",
    },
    "arm": {
        0: "arm_x",
        1: "arm_z",
        2: "arm_y",
        3: "arm_roll",
        4: "arm_pitch",
        5: "arm_yaw",
    },
}

# Reverse lookup: target -> key_name -> index
_COMMAND_INDEX: dict = {
    target: {name: idx for idx, name in mapping.items()} for target, mapping in COMMAND_KEYS.items()
}

JOYSTICK_COMMAND_MAP = {
    "left_stick_x": {
        "source": "axis",
        "mode": "absolute",
        "command": {"target": "dog", "cmd_key": "x_vel"},
        "deadzone": 0.08,
        "clamp": (-1.5, 1.5),
    },
    "left_stick_y": {
        "source": "axis",
        "mode": "absolute",
        "command": {"target": "dog", "cmd_key": "y_vel"},
        "deadzone": 0.08,
        "clamp": (-1, 1),
    },
    "right_stick_y": {
        "source": "axis",
        "mode": "absolute",
        "command": {"target": "dog", "cmd_key": "yaw_vel"},
        "deadzone": 0.08,
        "clamp": (-1, 1),
    },
    "right_stick_x": {
        "source": "axis",
        "mode": "absolute",
        "command": {"target": "dog", "cmd_key": "body_pitch"},
        "deadzone": 0.08,
        "scale": -1,
        "clamp": (-0.4, 0.4),
    },
    "rt": {
        "source": "axis",
        "mode": "absolute",
        "command": {"target": "dog", "cmd_key": "body_roll"},
        "deadzone": 0.05,
        "scale": -0.3,
        "clamp": (-0.4, 0.4),
    },
    "lt": {
        "source": "axis",
        "mode": "absolute",
        "command": {"target": "dog", "cmd_key": "body_roll"},
        "deadzone": 0.05,
        "scale": 0.3,
        "clamp": (-0.4, 0.4),
    },
    "f2": {
        "source": "button",
        "mode": "reset",
        "command": {"target": "env", "cmd_key": "reset"},
    },
    "a": {
        "source": "button",
        "mode": "step_button",
        "command": {"target": "dog", "cmd_key": "stance_length"},
        "direction": -1,
        "step": 0.05,
        "clamp": (0.2, 0.5),
    },
    "b": {
        "source": "button",
        "mode": "step_button",
        "command": {"target": "dog", "cmd_key": "stance_length"},
        "direction": 1,
        "step": 0.05,
        "clamp": (0.2, 0.5),
    },
    "x": {
        "source": "button",
        "mode": "step_button",
        "command": {"target": "dog", "cmd_key": "stance_width"},
        "direction": -1,
        "step": 0.05,
        "clamp": (0.25, 0.45),
    },
    "y": {
        "source": "button",
        "mode": "step_button",
        "command": {"target": "dog", "cmd_key": "stance_width"},
        "direction": 1,
        "step": 0.05,
        "clamp": (0.25, 0.45),
    },
    "dpad_x:up": {
        "source": "axis_combo",
        "mode": "step_once",
        "axis": "dpad_x",
        "axis_direction": "positive",
        "command": {"target": "dog", "cmd_key": "body_height_delta"},
        "delta": 0.05,
        "clamp": (-0.3, 0.3),
    },
    "dpad_x:down": {
        "source": "axis_combo",
        "mode": "step_once",
        "axis": "dpad_x",
        "axis_direction": "negative",
        "command": {"target": "dog", "cmd_key": "body_height_delta"},
        "delta": -0.05,
        "clamp": (-0.3, 0.3),
    },
    "dpad_y:left": {
        "source": "axis_combo",
        "mode": "step_once",
        "axis": "dpad_y",
        "axis_direction": "positive",
        "command": {"target": "dog", "cmd_key": "gait_freq"},
        "delta": -0.5,
        "clamp": (1.0, 8.0),
    },
    "dpad_y:right": {
        "source": "axis_combo",
        "mode": "step_once",
        "axis": "dpad_y",
        "axis_direction": "negative",
        "command": {"target": "dog", "cmd_key": "gait_freq"},
        "delta": 0.5,
        "clamp": (1.0, 8.0),
    },
}

TRIGGER_DELTA_CONFIG = {
    "negative": "lt",
    "positive": "rt",
    "threshold": 0.1,
}
DPAD_THRESHOLD = 0.5


def apply_checkpoint_command_limits(cfg):
    limits = get_play_command_limits(cfg)
    for mapping in JOYSTICK_COMMAND_MAP.values():
        command = mapping.get("command", {})
        target = command.get("target")
        cmd_key = command.get("cmd_key")
        limit = limits.get(target, {}).get(cmd_key)
        if limit is not None:
            mapping["clamp"] = limit


@dataclass
class DogInitCmd:
    x_vel: float = 0.0
    y_vel: float = 0.0
    yaw_vel: float = 0.0
    body_pitch: float = 0.0
    body_roll: float = 0.0
    body_height_delta: float = 0.0
    gait_freq: float = 4.0
    footswing_height: float = 0.06
    stance_width: float = 0.30
    stance_length: float = 0.4
    gait_duration: float = 0.49


@dataclass
class ArmInitCmd:
    l: float = 0.5
    p: float = 0.2
    y: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0


# ---------------------------------------------------------------------------
# Pure math helpers
# ---------------------------------------------------------------------------


def deadzone(value, threshold):
    value = float(value)
    if abs(value) < threshold:
        return 0.0
    sign = 1.0 if value > 0.0 else -1.0
    return sign * (abs(value) - threshold) / (1.0 - threshold + 1e-9)


def clamp(value, limits):
    return max(limits[0], min(limits[1], value))


def trigger_magnitude(value):
    value = float(value or 0.0)
    if value < 0.0:
        return clamp((1.0 - value) / 2.0, (0.0, 1.0))
    return clamp(value, (0.0, 1.0))


# ---------------------------------------------------------------------------
# Env command helpers
# ---------------------------------------------------------------------------


def set_command(env, target, index, value):
    if target == "dog" and index < env.commands_dog.shape[1]:
        env.commands_dog[:, index] = value
    elif target == "arm" and index < env.commands_arm.shape[1]:
        env.commands_arm[:, index] = value


def add_arm_command(env, index, delta, limits):
    if index >= env.commands_arm.shape[1]:
        return
    value = float(env.commands_arm[0, index]) + delta
    env.commands_arm[:, index] = clamp(value, limits)


def add_dog_command(env, index, delta, limits):
    if index >= env.commands_dog.shape[1]:
        return
    value = float(env.commands_dog[0, index]) + delta
    env.commands_dog[:, index] = clamp(value, limits)


def set_mapped_command(env, mapping, value):
    command = mapping["command"]
    idx = _COMMAND_INDEX[command["target"]][command["cmd_key"]]
    set_command(env, command["target"], idx, value)


def add_mapped_command(env, mapping, delta):
    command = mapping["command"]
    limits = mapping.get("clamp", (-1e9, 1e9))
    idx = _COMMAND_INDEX[command["target"]][command["cmd_key"]]
    if command["target"] == "dog":
        add_dog_command(env, idx, delta, limits)
    elif command["target"] == "arm":
        add_arm_command(env, idx, delta, limits)


def apply_all_dog_commands(env, cfg, cmd: DogInitCmd):
    """Write DogInitCmd values to env.commands_dog, guarded by the runtime command width."""
    n_cmd = env.commands_dog.shape[1]
    values = [
        cmd.x_vel,
        cmd.y_vel,
        cmd.yaw_vel,
        cmd.body_pitch,
        cmd.body_roll,
        cmd.body_height_delta,
        cmd.gait_freq,
        cmd.footswing_height,
        cmd.stance_width,
        cmd.stance_length,
        cmd.gait_duration,
    ]
    for index, value in enumerate(values[:n_cmd]):
        env.commands_dog[:, index] = value


def apply_all_arm_commands(env, cmd: ArmInitCmd):
    values = [cmd.l, cmd.p, cmd.y, cmd.roll, cmd.pitch, cmd.yaw]
    for index, value in enumerate(values[: env.commands_arm.shape[1]]):
        env.commands_arm[:, index] = value
    env.env.sync_arm_commands_to_obs(env_ids=slice(0, 1))


def format_robot_state(env):
    """Return a one-line string of the actual robot state (body frame)."""
    base = env.env
    vx = float(base.base_lin_vel[0, 0])
    vy = float(base.base_lin_vel[0, 1])
    wz = float(base.base_ang_vel[0, 2])
    z = float(base.root_states[0, 2])
    pitch = float(base.pitch[0])
    roll = float(base.roll[0])
    return f"vx={vx:+.2f}  vy={vy:+.2f}  wz={wz:+.2f}  |  h={z:.3f}  pitch={pitch:+.2f}  roll={roll:+.2f}"


# ---------------------------------------------------------------------------
# Joystick controller
# ---------------------------------------------------------------------------


class JoystickController:
    """Owns the JoyLink client and translates joystick input into env commands."""

    def __init__(self, config_path: str, dog_init_cmd: DogInitCmd, arm_init_cmd: ArmInitCmd):
        self._client = joylink_client.JoylinkClient(config_path)
        self._client.connect()
        self._config_path = config_path
        self._dog_init_cmd = dog_init_cmd
        self._arm_init_cmd = arm_init_cmd
        self._prev_buttons: dict = {}
        self._prev_axes: dict = {}

    def print_command_mapping(self):
        client = self._client
        axis_names = ", ".join(client.axis_names) if hasattr(client, "axis_names") else "unknown"
        button_names = ", ".join(client.button_names) if hasattr(client, "button_names") else "unknown"

        print("\n[RoboDuet] JoyLink command mapping")
        print(f"  JoyLink config: {self._config_path}")
        print(f"  JoyLink axes: {axis_names}")
        print(f"  JoyLink buttons: {button_names}")
        print("  Joystick key -> command key:")
        for joystick_key, mapping in JOYSTICK_COMMAND_MAP.items():
            cmd = mapping["command"]
            if cmd["target"] == "env":
                print(f"    {joystick_key:<16} -> env.{cmd['cmd_key']} mode={mapping['mode']}")
                continue
            command_desc = command_key(cmd["target"], cmd["cmd_key"])
            if mapping["mode"] == "absolute":
                print(
                    "    "
                    f"{joystick_key:<16} -> {command_desc} "
                    f"mode=absolute deadzone={mapping['deadzone']} clamp={mapping['clamp']}"
                )
                continue
            if mapping["mode"] in ("step_once", "step_button"):
                if mapping["source"] == "axis_combo":
                    trigger = mapping.get("axis_direction", mapping.get("direction", "?"))
                    delta = mapping.get("delta", mapping.get("step", "?"))
                    print(
                        "    "
                        f"{joystick_key:<16} -> {command_desc} "
                        f"mode=step_once trigger={mapping.get('axis', '?')}:{trigger} "
                        f"delta={delta} clamp={mapping.get('clamp', 'none')}"
                    )
                else:
                    direction = mapping.get("direction", 1)
                    sign = "+" if direction in (1, "positive") else "-"
                    print(
                        "    "
                        f"{joystick_key:<16} -> {command_desc} "
                        f"mode=step_once {sign}{mapping.get('step', '?')} clamp={mapping.get('clamp', 'none')}"
                    )
                continue
            print(
                "    "
                f"{joystick_key:<16} -> {command_desc} "
                f"mode={mapping['mode']} step={mapping.get('step', '?')} clamp={mapping.get('clamp', 'none')}"
            )
        print(
            "  "
            f"trigger_delta = {TRIGGER_DELTA_CONFIG['positive']} - {TRIGGER_DELTA_CONFIG['negative']} "
            f"(threshold={TRIGGER_DELTA_CONFIG['threshold']})",
            flush=True,
        )

    def step(self, env):
        """Receive one frame from JoyLink and apply it to env commands."""
        joy_data = self._client.receive(timeout_ms=0)
        self._apply(env, joy_data)

    def _apply(self, env, joy_data):
        if not joy_data:
            return
        axes = joy_data.get("axes", {})
        buttons = joy_data.get("buttons", {})

        # Accumulate all "axis absolute" contributions per command slot, then write once.
        # Allows multiple axes (e.g. lt and rt) to contribute additively to one index.
        _accum = {}  # (target, index) -> [total, clamp]
        for joystick_key, mapping in JOYSTICK_COMMAND_MAP.items():
            if mapping["source"] != "axis" or mapping["mode"] != "absolute":
                continue
            if joystick_key not in axes:
                continue
            raw = deadzone(axes[joystick_key], mapping["deadzone"]) * mapping.get("scale", 1.0)
            cmd = mapping["command"]
            accum_key = (cmd["target"], _COMMAND_INDEX[cmd["target"]][cmd["cmd_key"]])
            if accum_key not in _accum:
                _accum[accum_key] = [0.0, mapping["clamp"]]
            _accum[accum_key][0] += raw
        for (target, index), (total, clamp_range) in _accum.items():
            set_command(env, target, index, clamp(total, clamp_range))

        lt = trigger_magnitude(axes.get(TRIGGER_DELTA_CONFIG["negative"], 0.0))
        rt = trigger_magnitude(axes.get(TRIGGER_DELTA_CONFIG["positive"], 0.0))
        trigger_delta = rt - lt
        if max(lt, rt) < TRIGGER_DELTA_CONFIG["threshold"]:
            trigger_delta = 0.0

        for joystick_key, mapping in JOYSTICK_COMMAND_MAP.items():
            if mapping["source"] != "button":
                continue
            pressed = bool(buttons.get(joystick_key, 0))
            was_pressed = bool(self._prev_buttons.get(joystick_key, 0))
            rising_edge = pressed and not was_pressed

            if not pressed and not rising_edge:
                continue

            if mapping["mode"] == "reset" and rising_edge:
                env.reset()
                apply_all_dog_commands(env, env.env.cfg, self._dog_init_cmd)
                apply_all_arm_commands(env, self._arm_init_cmd)
                continue

            if mapping["mode"] in ("step_once", "step_button") and rising_edge:
                direction = mapping.get("direction", 1)
                add_mapped_command(env, mapping, direction * mapping.get("step", 0.05))
                continue

            if mapping["mode"] == "step_with_trigger" and pressed and abs(trigger_delta) > 1e-6:
                add_mapped_command(env, mapping, trigger_delta * mapping["step"])
                continue

        for joystick_key, mapping in JOYSTICK_COMMAND_MAP.items():
            if mapping["source"] != "axis_combo":
                continue
            axis_name = mapping["axis"]
            axis_value = float(axes.get(axis_name, 0.0) or 0.0)
            prev_axis_value = float(self._prev_axes.get(axis_name, 0.0))

            axis_direction = mapping.get("axis_direction", mapping.get("direction", "positive"))
            if mapping["mode"] == "step_once":
                delta = mapping.get("delta")
                if delta is None:
                    delta = mapping.get("step", 0.05)
                    if axis_direction == "negative":
                        delta = -delta

                if axis_direction == "positive":
                    if axis_value > DPAD_THRESHOLD and prev_axis_value <= DPAD_THRESHOLD:
                        add_mapped_command(env, mapping, delta)
                elif axis_direction == "negative":
                    if axis_value < -DPAD_THRESHOLD and prev_axis_value >= -DPAD_THRESHOLD:
                        add_mapped_command(env, mapping, delta)
            elif mapping["mode"] == "step_with_trigger" and abs(trigger_delta) > 1e-6:
                if axis_direction == "positive" and axis_value <= DPAD_THRESHOLD:
                    continue
                if axis_direction == "negative" and axis_value >= -DPAD_THRESHOLD:
                    continue
                add_mapped_command(env, mapping, trigger_delta * mapping["step"])

        self._prev_buttons = dict(buttons)
        self._prev_axes = {
            m["axis"]: float(axes.get(m["axis"], 0.0) or 0.0)
            for m in JOYSTICK_COMMAND_MAP.values()
            if m["source"] == "axis_combo"
        }
        env.env.sync_arm_commands_to_obs(env_ids=slice(0, 1))


def command_key(target, key):
    idx = _COMMAND_INDEX.get(target, {}).get(key, "?")
    return f"commands_{target}[{idx}] ({key})"


def main(args):
    logdir = args.logdir
    lock_arm = bool(getattr(args, "lock_arm", False))
    ckpt_id_arg = str(args.ckptid)
    ckpt_id = "last" if ckpt_id_arg == "last" else ckpt_id_arg.zfill(6)

    from go1_gym.utils.global_switch import global_switch

    stage1_only = bool(getattr(args, "stage1_only", False))
    stage1_arm_intensity = max(0.0, min(1.0, float(getattr(args, "stage1_arm_intensity", 1.0))))
    if stage1_only:
        global_switch.switch_flag = False
        global_switch.count = 0
        global_switch.stage1_count = 0
        global_switch.pretrained_to_wbc_start = 10**12
        global_switch.pretrained_to_wbc_end = global_switch.pretrained_to_wbc_start + 1
    else:
        global_switch.open_switch()

    dog_cmd = DogInitCmd()
    arm_cmd = ArmInitCmd()

    env, cfg = load_env(
        logdir, wrapper=WBCEnv, headless=False, device=args.sim_device, robot=getattr(args, "robot", None)
    )
    apply_checkpoint_command_limits(cfg)

    config_path = os.path.join(os.path.dirname(joylink_client.__file__), "../../config/loco_ctrl.yaml")
    joy_ctrl = JoystickController(config_path, dog_cmd, arm_cmd)
    joy_ctrl.print_command_mapping()
    dog_policy = load_dog_policy(logdir, ckpt_id, cfg)
    arm_policy = None if stage1_only else load_arm_policy(logdir, ckpt_id, cfg)
    if stage1_only:
        env.env.cfg.env.stage1_arm_curriculum = True
        env.env.stage1_arm_play_intensity = stage1_arm_intensity
    if stage1_only and getattr(args, "disable_stage1_arm_curriculum", False):
        env.env.cfg.env.stage1_arm_curriculum = False
    if lock_arm:
        env.env.cfg.env.stage1_arm_curriculum = False
    configure_privileged_obs_dims(cfg)

    env.env.enable_viewer_sync = True

    rerun_logger = make_rerun_logger(
        args,
        dog_command_names=COMMAND_KEYS["dog"],
        app_id="roboduet_play_by_joy",
    )

    env.reset()
    apply_all_dog_commands(env, cfg, dog_cmd)
    apply_all_arm_commands(env, arm_cmd)

    if env.commands_dog.shape[1] >= 11 and not getattr(env.env.cfg.commands, "use_dynamic_gait", False):
        print(
            "[warn] commands_dog has gait fields, but cfg.commands.use_dynamic_gait=False; "
            "freq/swing/width/length/duration will display but the gait clock uses fixed defaults.",
            flush=True,
        )

    if lock_arm:
        print("[arm] LOCKED — zero actions sent every step", flush=True)
    # Reserve two lines that the periodic printer will overwrite in place
    print(f"[cmd]   {env.env.format_dog_commands()}")
    print(f"[state] {format_robot_state(env)}", flush=True)

    _CMD_PRINT_INTERVAL = 0.1  # seconds
    _last_print = time.monotonic()

    while True:
        joy_ctrl.step(env)

        now = time.monotonic()
        if now - _last_print >= _CMD_PRINT_INTERVAL:
            cmd_line = f"[cmd]   {env.env.format_dog_commands()}"
            state_line = f"[state] {format_robot_state(env)}"
            # \033[2A moves cursor up 2 lines to overwrite both lines in place
            print(f"\033[2A{cmd_line}\n{state_line}\033[K", flush=True)
            _last_print = now

        with torch.no_grad():
            if lock_arm or arm_policy is None:
                actions_arm = env.arm_fake_actions
            else:
                obs = env.get_arm_observations()
                actions_arm = arm_policy(obs).to(env.env.device)
                env.plan(actions_arm[..., -2:])

            dog_obs = env.get_dog_observations()
            actions_dog = dog_policy(dog_obs).to(env.env.device)

        if lock_arm or arm_policy is None:
            env.step(actions_dog, actions_arm)
        else:
            env.step(actions_dog, actions_arm[..., :-2])

        rerun_logger.log(env)


def parse_args():
    parser = argparse.ArgumentParser(description="RoboDuet — joystick inference")
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--ckptid", type=str, default="last")
    parser.add_argument("--robot", type=str, default="go2", choices=["go1", "go2"])
    parser.add_argument(
        "--stage1_only",
        action="store_true",
        default=False,
        help="Run pure stage1 play: keep global switch closed and do not load/call the arm policy.",
    )
    parser.add_argument(
        "--stage1_arm_intensity",
        type=float,
        default=1,
        help="Stage1 arm disturbance intensity for stage1-only play, applied directly in [0, 1].",
    )
    parser.add_argument(
        "--lock_arm",
        action="store_true",
        default=False,
        help="Send zero arm actions every step (hold arm at default position), ignoring any loaded arm policy.",
    )
    add_rerun_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
