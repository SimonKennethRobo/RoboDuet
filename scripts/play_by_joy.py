"""play_by_joy.py — policy inference with JoyLink joystick control.

Usage::

    python scripts/play_by_joy.py \\
        --logdir runs/test_roboduet/2024-10-13/auto_train/003436.678552_seed9145 \\
        --ckptid 40000

Joystick transport is handled by JoyLink's Python client. Pass
--joylink_config to select the JoyLink backend config.
"""

import argparse
import os
import sys
import time
import types

import isaacgym  # noqa: F401 – must be imported before torch
import pytorch3d.transforms as pt3d
import torch
from isaacgym.torch_utils import quat_from_euler_xyz, quat_mul

from go1_gym.envs import *  # noqa: F403
from go1_gym.envs.roboduet import WBCEnv
from go1_gym.envs.roboduet.legged_robot import quaternion_to_rpy
from go1_gym.envs.roboduet.wbc_env_config import configure_privileged_obs_dims
from scripts.load_policy import load_arm_policy, load_dog_policy, load_env

# Default run/checkpoint (overridden by CLI args)
logdir = "runs/test_roboduet/2024-10-13/auto_train/003436.678552_seed9145"
ckpt_id = "040000"

# Initial command values — velocity
x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
# Initial command values — arm
l_cmd, p_cmd, y_cmd = 0.5, 0.2, 0.0
roll_cmd, pitch_cmd, yaw_cmd = 0.0, 0.0, 0.0
# Set to True to send zero arm actions every step (hold arm at default position)
lock_arm = True
# Initial command values — dog body pose (commands_dog[:,3:6])
body_pitch_cmd = 0.0  # rad   commands_dog[:, 3]
body_roll_cmd = 0.0  # rad   commands_dog[:, 4]
body_height_delta_cmd = 0  # m     commands_dog[:, 5] (added to base_height_target)
# Initial command values — gait params (commands_dog[:,6:11], only if use_dynamic_gait)
gait_freq_cmd = 4  # Hz    commands_dog[:, 6]
footswing_height_cmd = 0.08  # m     commands_dog[:, 7]
stance_width_cmd = 0.30  # m     commands_dog[:, 8]
stance_length_cmd = 0.45  # m     commands_dog[:, 9]
gait_duration_cmd = 0.5  # frac  commands_dog[:, 10]

DEFAULT_JOYLINK_CLIENT_DIR = "/home/simon/Projects/Simon/JoyLink/client/python"
DEFAULT_JOYLINK_CONFIG = "/home/simon/Projects/Simon/JoyLink/config/loco_ctrl.yaml"

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

JOYSTICK_COMMAND_MAP = {
    "left_stick_x": {
        "source": "axis",
        "mode": "absolute",
        "command": {"target": "dog", "index": 0, "key": "x_vel"},
        "deadzone": 0.08,
        "clamp": (-1.5, 1.5),
    },
    "left_stick_y": {
        "source": "axis",
        "mode": "absolute",
        "command": {"target": "dog", "index": 1, "key": "y_vel"},
        "deadzone": 0.08,
        "clamp": (-0.5, 0.5),
    },
    "right_stick_y": {
        "source": "axis",
        "mode": "absolute",
        "command": {"target": "dog", "index": 2, "key": "yaw_vel"},
        "deadzone": 0.08,
        "clamp": (-1.5, 1.5),
    },
    "right_stick_x": {
        "source": "axis",
        "mode": "absolute",
        "command": {"target": "dog", "index": 3, "key": "body_pitch"},
        "deadzone": 0.08,
        "scale": -1,
        "clamp": (-0.4, 0.4),
    },
    "rt": {
        "source": "axis",
        "mode": "absolute",
        "command": {"target": "dog", "index": 4, "key": "body_roll"},
        "deadzone": 0.05,
        "scale": -0.3,
        "clamp": (-0.4, 0.4),
    },
    "lt": {
        "source": "axis",
        "mode": "absolute",
        "command": {"target": "dog", "index": 4, "key": "body_roll"},
        "deadzone": 0.05,
        "scale": 0.3,
        "clamp": (-0.4, 0.4),
    },
    "a": {
        "source": "button",
        "mode": "reset",
        "command": {"target": "env", "key": "reset"},
    },
    "x": {
        "source": "button",
        "mode": "step_button",
        "command": {"target": "dog", "index": 8, "key": "stance_width"},
        "direction": -1,
        "step": 0.05,
        "clamp": (-1.5, 1.5),
    },
    "y": {
        "source": "button",
        "mode": "step_button",
        "command": {"target": "dog", "index": 8, "key": "stance_width"},
        "direction": 1,
        "step": 0.05,
        "clamp": (-1.5, 1.5),
    },
    "b": {
        "source": "button",
        "mode": "step_with_trigger",
        "command": {"target": "arm", "index": 5, "key": "arm_yaw"},
        "step": 0.05,
        "clamp": (-1.5, 1.5),
    },
    "dpad_x:up": {
        "source": "axis_combo",
        "mode": "step_once",
        "axis": "dpad_x",
        "direction": "positive",
        "command": {"target": "dog", "index": 5, "key": "body_height_delta"},
        "step": 0.05,
        "clamp": (-0.3, 0.3),
    },
    "dpad_x:down": {
        "source": "axis_combo",
        "mode": "step_once",
        "axis": "dpad_x",
        "direction": "negative",
        "command": {"target": "dog", "index": 5, "key": "body_height_delta"},
        "step": 0.05,
        "clamp": (-0.3, 0.3),
    },
    "dpad_y:left": {
        "source": "axis_combo",
        "mode": "step_once",
        "axis": "dpad_y",
        "direction": "positive",
        "command": {"target": "dog", "index": 6, "key": "gait_freq"},
        "step": 0.5,
        "clamp": (1.0, 4.0),
    },
    "dpad_y:right": {
        "source": "axis_combo",
        "mode": "step_once",
        "axis": "dpad_y",
        "direction": "negative",
        "command": {"target": "dog", "index": 6, "key": "gait_freq"},
        "step": 0.5,
        "clamp": (1.0, 4.0),
    },
}

TRIGGER_DELTA_CONFIG = {
    "negative": "lt",
    "positive": "rt",
    "threshold": 0.1,
}
DPAD_THRESHOLD = 0.5

# Tracks previous button/axis states for rising-edge detection
_prev_buttons: dict = {}
_prev_axes: dict = {}


def command_key(target, index):
    return f"commands_{target}[{index}] ({COMMAND_KEYS.get(target, {}).get(index, 'unknown')})"


def print_joy_command_mapping(client, joylink_config):
    axis_names = ", ".join(client.axis_names) if hasattr(client, "axis_names") else "unknown"
    button_names = ", ".join(client.button_names) if hasattr(client, "button_names") else "unknown"

    print("\n[RoboDuet] JoyLink command mapping")
    print(f"  JoyLink config: {joylink_config}")
    print(f"  JoyLink axes: {axis_names}")
    print(f"  JoyLink buttons: {button_names}")
    print("  Joystick key -> command key:")
    for joystick_key, mapping in JOYSTICK_COMMAND_MAP.items():
        command = mapping["command"]
        if command["target"] == "env":
            print(f"    {joystick_key:<16} -> env.{command['key']} mode={mapping['mode']}")
            continue
        command_desc = command_key(command["target"], command["index"])
        if mapping["mode"] == "absolute":
            print(
                "    "
                f"{joystick_key:<16} -> {command_desc} "
                f"mode=absolute deadzone={mapping['deadzone']} clamp={mapping['clamp']}"
            )
            continue
        if mapping["mode"] in ("step_once", "step_button"):
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


def _load_joylink_client_class(client_dir):
    if client_dir not in sys.path:
        sys.path.insert(0, client_dir)
    try:
        from joystick_client import JoystickClient

        return JoystickClient
    except TypeError as exc:
        if "unsupported operand type(s) for |" not in str(exc):
            raise

    client_path = os.path.join(client_dir, "joystick_client.py")
    with open(client_path, "r", encoding="utf-8") as f:
        source = f.read()
    module = types.ModuleType("_roboduet_play_joylink_client")
    module.__file__ = client_path
    code = compile("from __future__ import annotations\n" + source, client_path, "exec")
    exec(code, module.__dict__)
    return module.JoystickClient


def build_joylink_client(args):
    client_dir = args.joylink_client_dir or DEFAULT_JOYLINK_CLIENT_DIR
    config_path = args.joylink_config or DEFAULT_JOYLINK_CONFIG
    JoystickClient = _load_joylink_client_class(client_dir)
    client = JoystickClient(config_path)
    client.connect()
    return client, config_path


def latest_joylink_data(client):
    latest = None
    while True:
        data = client.receive(timeout_ms=0)
        if data is None:
            break
        latest = data
    return latest


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
    set_command(env, command["target"], command["index"], value)


def add_mapped_command(env, mapping, delta):
    command = mapping["command"]
    limits = mapping.get("clamp", (-1e9, 1e9))
    if command["target"] == "dog":
        add_dog_command(env, command["index"], delta, limits)
    elif command["target"] == "arm":
        add_arm_command(env, command["index"], delta, limits)


def sync_arm_command_obs(env):
    base_env = env.env
    base_env.commands_arm_obs[0:1, 0] = base_env.commands_arm[0:1, 0]
    base_env.commands_arm_obs[0:1, 1] = base_env.commands_arm[0:1, 1]
    base_env.commands_arm_obs[0:1, 2] = base_env.commands_arm[0:1, 2]

    roll = base_env.commands_arm[0:1, 3]
    pitch = base_env.commands_arm[0:1, 4]
    yaw = base_env.commands_arm[0:1, 5]

    zero_vec = torch.zeros_like(roll)
    q1 = quat_from_euler_xyz(zero_vec, zero_vec, yaw)
    q2 = quat_from_euler_xyz(zero_vec, pitch, zero_vec)
    q3 = quat_from_euler_xyz(roll, zero_vec, zero_vec)
    quats = quat_mul(q1, quat_mul(q2, q3))

    base_env.obj_quats[0:1] = quats.reshape(-1, 4)

    if base_env.cfg.hybrid.use_vision:
        base_env._get_object_pose_in_ee()
        base_env._get_object_abg_in_ee()

    base_env.visual_rpy[0:1] = quaternion_to_rpy(base_env.obj_quats[0:1]).to(base_env.device)
    if base_env.cfg.use_rot6d:
        r6d = pt3d.matrix_to_rotation_6d(pt3d.quaternion_to_matrix(quats[:, [3, 0, 1, 2]]))
        base_env.commands_arm_obs[0:1, 3:9] = r6d.to(base_env.device)
    else:
        rpy = base_env.quat_to_angle(base_env.obj_quats[0:1]).to(base_env.device)
        base_env.commands_arm_obs[0:1, 3] = rpy[:, 0]
        base_env.commands_arm_obs[0:1, 4] = rpy[:, 1]
        base_env.commands_arm_obs[0:1, 5] = rpy[:, 2]


def apply_all_dog_commands(env, cfg):
    """Write all dog command module variables to env.commands_dog.

    Respects n_cmd (body pose only written if column exists) and
    use_dynamic_gait (gait params only written when the flag is set).
    Safe to call before and during the step loop.
    """
    n_cmd = env.commands_dog.shape[1]
    # velocity (always present)
    if n_cmd > 0:
        env.commands_dog[:, 0] = x_vel_cmd
    if n_cmd > 1:
        env.commands_dog[:, 1] = y_vel_cmd
    if n_cmd > 2:
        env.commands_dog[:, 2] = yaw_vel_cmd
    # body pose
    if n_cmd > 3:
        env.commands_dog[:, 3] = body_pitch_cmd
    if n_cmd > 4:
        env.commands_dog[:, 4] = body_roll_cmd
    if n_cmd > 5:
        env.commands_dog[:, 5] = body_height_delta_cmd
    # gait params — only when the env uses dynamic gait
    use_dg = getattr(getattr(cfg, "commands", None), "use_dynamic_gait", False)
    if use_dg:
        if n_cmd > 6:
            env.commands_dog[:, 6] = gait_freq_cmd
        if n_cmd > 7:
            env.commands_dog[:, 7] = footswing_height_cmd
        if n_cmd > 8:
            env.commands_dog[:, 8] = stance_width_cmd
        if n_cmd > 9:
            env.commands_dog[:, 9] = stance_length_cmd
        if n_cmd > 10:
            env.commands_dog[:, 10] = gait_duration_cmd


def format_dog_commands(env, cfg):
    """Return a one-line human-readable string of the active dog commands."""
    c = env.commands_dog[0]  # read from env, not module vars
    n_cmd = env.commands_dog.shape[1]
    parts = [
        f"vx={float(c[0]):+.2f}",
        f"vy={float(c[1]):+.2f}",
        f"wz={float(c[2]):+.2f}",
    ]
    if n_cmd > 5:
        parts += [
            f"pitch={float(c[3]):+.2f}",
            f"roll={float(c[4]):+.2f}",
            f"dh={float(c[5]):+.3f}",
        ]
    if n_cmd > 6:
        parts.append(f"freq={float(c[6]):.2f}")
    if n_cmd > 10:
        parts += [
            f"swing={float(c[7]):.3f}",
            f"sw={float(c[8]):.3f}",
            f"sl={float(c[9]):.3f}",
            f"dur={float(c[10]):.2f}",
        ]
    return "  ".join(parts)


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


def apply_joylink_commands(env, joy_data):
    global _prev_buttons, _prev_axes
    if not joy_data:
        return
    axes = joy_data.get("axes", {})
    buttons = joy_data.get("buttons", {})

    # Accumulate all "axis absolute" contributions per command slot, then write once.
    # This lets multiple axes (e.g. lt and rt) contribute additively to one index.
    _accum = {}  # (target, index) -> [total, clamp]
    for joystick_key, mapping in JOYSTICK_COMMAND_MAP.items():
        if mapping["source"] != "axis" or mapping["mode"] != "absolute":
            continue
        if joystick_key not in axes:
            continue
        raw = deadzone(axes[joystick_key], mapping["deadzone"]) * mapping.get("scale", 1.0)
        cmd = mapping["command"]
        key = (cmd["target"], cmd["index"])
        if key not in _accum:
            _accum[key] = [0.0, mapping["clamp"]]
        _accum[key][0] += raw
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
        was_pressed = bool(_prev_buttons.get(joystick_key, 0))
        rising_edge = pressed and not was_pressed

        if not pressed and not rising_edge:
            continue

        if mapping["mode"] == "reset" and rising_edge:
            env.reset()
            env.commands_dog[:, :3] = 0.0
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
        prev_axis_value = float(_prev_axes.get(axis_name, 0.0))

        if mapping["mode"] == "step_once":
            if mapping["direction"] == "positive":
                if axis_value > DPAD_THRESHOLD and prev_axis_value <= DPAD_THRESHOLD:
                    add_mapped_command(env, mapping, mapping.get("step", 0.05))
            elif mapping["direction"] == "negative":
                if axis_value < -DPAD_THRESHOLD and prev_axis_value >= -DPAD_THRESHOLD:
                    add_mapped_command(env, mapping, -mapping.get("step", 0.05))
        elif mapping["mode"] == "step_with_trigger" and abs(trigger_delta) > 1e-6:
            if mapping["direction"] == "positive" and axis_value <= DPAD_THRESHOLD:
                continue
            if mapping["direction"] == "negative" and axis_value >= -DPAD_THRESHOLD:
                continue
            add_mapped_command(env, mapping, trigger_delta * mapping["step"])

    _prev_buttons = dict(buttons)
    _prev_axes = {
        m["axis"]: float(axes.get(m["axis"], 0.0) or 0.0)
        for m in JOYSTICK_COMMAND_MAP.values()
        if m["source"] == "axis_combo"
    }
    sync_arm_command_obs(env)


def main(args):
    global logdir, ckpt_id, lock_arm

    logdir = args.logdir
    lock_arm = bool(getattr(args, "lock_arm", False))
    ckpt_id_arg = str(args.ckptid)
    ckpt_id = "last" if ckpt_id_arg == "last" else ckpt_id_arg.zfill(6)

    from go1_gym.utils.global_switch import global_switch

    stage1_only = bool(getattr(args, "stage1_only", False))
    if stage1_only:
        global_switch.switch_flag = False
        global_switch.count = 0
        ramp_iters = max(1, int(getattr(args, "stage1_arm_ramp_iterations", 1)))
        global_switch.stage1_arm_ramp_iterations = ramp_iters
        stage1_arm_intensity = float(getattr(args, "stage1_arm_intensity", 1.0))
        global_switch.stage1_count = int(max(0.0, min(1.0, stage1_arm_intensity)) * ramp_iters)
        global_switch.pretrained_to_hybrid_start = getattr(args, "num_eval_steps", 30000) + 1
        global_switch.pretrained_to_hybrid_end = global_switch.pretrained_to_hybrid_start + 1
    else:
        global_switch.open_switch()

    joy_client, joylink_config = build_joylink_client(args)
    print_joy_command_mapping(joy_client, joylink_config)

    env, cfg = load_env(
        logdir, wrapper=WBCEnv, headless=args.headless, device=args.sim_device, robot=getattr(args, "robot", None)
    )
    dog_policy = load_dog_policy(logdir, ckpt_id, cfg)
    arm_policy = None if stage1_only else load_arm_policy(logdir, ckpt_id, cfg)
    if stage1_only and getattr(args, "disable_stage1_arm_curriculum", False):
        env.env.cfg.env.stage1_arm_curriculum = False
    if lock_arm:
        env.env.cfg.env.stage1_arm_curriculum = False
    configure_privileged_obs_dims(cfg)

    env.env.enable_viewer_sync = True

    num_eval_steps = getattr(args, "num_eval_steps", 30000)

    obs = env.reset()

    # Set initial commands
    apply_all_dog_commands(env, cfg)
    env.commands_arm[:, 0] = l_cmd
    env.commands_arm[:, 1] = p_cmd
    env.commands_arm[:, 2] = y_cmd
    env.commands_arm[:, 3] = roll_cmd
    env.commands_arm[:, 4] = pitch_cmd
    env.commands_arm[:, 5] = yaw_cmd
    sync_arm_command_obs(env)

    if lock_arm:
        print("[arm] LOCKED — zero actions sent every step", flush=True)
    # Reserve two lines that the periodic printer will overwrite in place
    print(f"[cmd]   {format_dog_commands(env, cfg)}")
    print(f"[state] {format_robot_state(env)}", flush=True)

    _CMD_PRINT_INTERVAL = 0.1  # seconds
    _last_print = time.monotonic()

    for step_i in range(num_eval_steps):
        apply_joylink_commands(env, latest_joylink_data(joy_client))

        now = time.monotonic()
        if now - _last_print >= _CMD_PRINT_INTERVAL:
            cmd_line = f"[cmd]   {format_dog_commands(env, cfg)}"
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


def parse_args():
    parser = argparse.ArgumentParser(description="RoboDuet — joystick inference")
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--ckptid", type=str, default="last")
    parser.add_argument("--robot", type=str, default="go2", choices=["go1", "go2"])
    parser.add_argument("--num_eval_steps", type=int, default=30000)
    parser.add_argument(
        "--stage1_only",
        action="store_true",
        default=False,
        help="Run pure stage1 play: keep global switch closed and do not load/call the arm policy.",
    )
    parser.add_argument(
        "--stage1_arm_intensity",
        type=float,
        default=0.3,
        help="Stage1 arm disturbance curriculum intensity for play, in [0, 1].",
    )
    parser.add_argument(
        "--stage1_arm_ramp_iterations",
        type=int,
        default=1,
        help="Synthetic ramp length used to realize --stage1_arm_intensity during play.",
    )
    parser.add_argument(
        "--disable_stage1_arm_curriculum",
        action="store_true",
        default=False,
        help="Keep the arm fixed instead of applying stage1 arm disturbance during stage1-only play.",
    )
    parser.add_argument(
        "--lock_arm",
        action="store_true",
        default=False,
        help="Send zero arm actions every step (hold arm at default position), ignoring any loaded arm policy.",
    )
    parser.add_argument(
        "--joylink_config",
        type=str,
        default=None,
        help="Path to JoyLink config (default: /home/simon/Projects/Simon/JoyLink/config/loco_ctrl.yaml)",
    )
    parser.add_argument(
        "--joylink_client_dir",
        type=str,
        default=None,
        help="Path to JoyLink client/python directory (default: /home/simon/Projects/Simon/JoyLink/client/python)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
