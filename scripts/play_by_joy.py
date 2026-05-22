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

# Initial command values
x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
l_cmd, p_cmd, y_cmd = 0.5, 0.2, 0.0
roll_cmd, pitch_cmd, yaw_cmd = 0.1, 0.5, 0.0

DEFAULT_JOYLINK_CLIENT_DIR = "/home/simon/Projects/Simon/JoyLink/client/python"
DEFAULT_JOYLINK_CONFIG = "/home/simon/Projects/Simon/JoyLink/config/loco_ctrl.yaml"

COMMAND_KEYS = {
    "dog": {
        0: "x_vel",
        1: "y_vel",
        2: "yaw_vel",
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
        "scale": 1.5,
        "deadzone": 0.08,
        "clamp": (-1.5, 1.5),
    },
    "left_stick_y": {
        "source": "axis",
        "mode": "absolute",
        "command": {"target": "dog", "index": 1, "key": "y_vel"},
        "scale": 0.5,
        "deadzone": 0.08,
        "clamp": (-0.5, 0.5),
    },
    "right_stick_y": {
        "source": "axis",
        "mode": "absolute",
        "command": {"target": "dog", "index": 2, "key": "yaw_vel"},
        "scale": 1.5,
        "deadzone": 0.08,
        "clamp": (-1.5, 1.5),
    },
    "right_stick_x": {
        "source": "axis",
        "mode": "absolute",
        "command": {"target": "arm", "index": 1, "key": "arm_z"},
        "scale": 0.2,
        "deadzone": 0.08,
        "clamp": (-1.0, 1.0),
    },
    "a": {
        "source": "button",
        "mode": "reset",
        "command": {"target": "env", "key": "reset"},
    },
    "x": {
        "source": "button",
        "mode": "step_with_trigger",
        "command": {"target": "arm", "index": 3, "key": "arm_roll"},
        "step": 0.05,
        "clamp": (-1.5, 1.5),
    },
    "y": {
        "source": "button",
        "mode": "step_with_trigger",
        "command": {"target": "arm", "index": 4, "key": "arm_pitch"},
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
    "dpad_y:up": {
        "source": "axis_combo",
        "mode": "step_with_trigger",
        "axis": "dpad_y",
        "direction": "positive",
        "command": {"target": "arm", "index": 1, "key": "arm_z"},
        "step": 0.05,
        "clamp": (-1.0, 1.0),
    },
    "dpad_y:down": {
        "source": "axis_combo",
        "mode": "step_with_trigger",
        "axis": "dpad_y",
        "direction": "negative",
        "command": {"target": "arm", "index": 0, "key": "arm_x"},
        "step": 0.05,
        "clamp": (-1.0, 1.0),
    },
    "dpad_x:left": {
        "source": "axis_combo",
        "mode": "step_with_trigger",
        "axis": "dpad_x",
        "direction": "negative",
        "command": {"target": "arm", "index": 2, "key": "arm_y"},
        "step": 0.05,
        "clamp": (-1.0, 1.0),
    },
}

TRIGGER_DELTA_CONFIG = {
    "negative": "left_trigger",
    "positive": "right_trigger",
    "threshold": 0.1,
}
DPAD_THRESHOLD = 0.5


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
                f"mode=absolute scale={mapping['scale']} deadzone={mapping['deadzone']} clamp={mapping['clamp']}"
            )
            continue
        print(
            "    "
            f"{joystick_key:<16} -> {command_desc} "
            f"mode={mapping['mode']} step={mapping['step']} clamp={mapping['clamp']}"
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


def set_mapped_command(env, mapping, value):
    command = mapping["command"]
    set_command(env, command["target"], command["index"], value)


def add_mapped_command(env, mapping, delta):
    command = mapping["command"]
    if command["target"] != "arm":
        return
    add_arm_command(env, command["index"], delta, mapping["clamp"])


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


def apply_joylink_commands(env, joy_data):
    if not joy_data:
        return
    axes = joy_data.get("axes", {})
    buttons = joy_data.get("buttons", {})

    for joystick_key, mapping in JOYSTICK_COMMAND_MAP.items():
        if mapping["source"] != "axis" or mapping["mode"] != "absolute":
            continue
        if joystick_key not in axes:
            continue
        value = deadzone(axes[joystick_key], mapping["deadzone"]) * mapping["scale"]
        value = clamp(value, mapping["clamp"])
        set_mapped_command(env, mapping, value)

    lt = trigger_magnitude(axes.get(TRIGGER_DELTA_CONFIG["negative"], 0.0))
    rt = trigger_magnitude(axes.get(TRIGGER_DELTA_CONFIG["positive"], 0.0))
    trigger_delta = rt - lt
    if max(lt, rt) < TRIGGER_DELTA_CONFIG["threshold"]:
        trigger_delta = 0.0

    for joystick_key, mapping in JOYSTICK_COMMAND_MAP.items():
        if mapping["source"] != "button" or not buttons.get(joystick_key, 0):
            continue
        if mapping["mode"] == "reset":
            env.reset()
            env.commands_dog[:, :3] = 0.0
            continue
        if mapping["mode"] != "step_with_trigger" or abs(trigger_delta) <= 1e-6:
            continue
        add_mapped_command(env, mapping, trigger_delta * mapping["step"])

    for joystick_key, mapping in JOYSTICK_COMMAND_MAP.items():
        if mapping["source"] != "axis_combo" or abs(trigger_delta) <= 1e-6:
            continue
        axis_value = float(axes.get(mapping["axis"], 0.0) or 0.0)
        if mapping["direction"] == "positive" and axis_value <= DPAD_THRESHOLD:
            continue
        if mapping["direction"] == "negative" and axis_value >= -DPAD_THRESHOLD:
            continue
        add_mapped_command(env, mapping, trigger_delta * mapping["step"])

    sync_arm_command_obs(env)


def main(args):
    global logdir, ckpt_id

    logdir = args.logdir
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
    configure_privileged_obs_dims(cfg)

    env.env.enable_viewer_sync = True

    num_eval_steps = getattr(args, "num_eval_steps", 30000)

    obs = env.reset()

    # Set initial commands
    env.commands_dog[:, 0] = x_vel_cmd
    env.commands_dog[:, 1] = y_vel_cmd
    env.commands_dog[:, 2] = yaw_vel_cmd
    env.commands_arm[:, 0] = l_cmd
    env.commands_arm[:, 1] = p_cmd
    env.commands_arm[:, 2] = y_cmd
    env.commands_arm[:, 3] = roll_cmd
    env.commands_arm[:, 4] = pitch_cmd
    env.commands_arm[:, 5] = yaw_cmd
    sync_arm_command_obs(env)

    for _ in range(num_eval_steps):
        apply_joylink_commands(env, latest_joylink_data(joy_client))

        with torch.no_grad():
            if arm_policy is None:
                actions_arm = env.arm_fake_actions
            else:
                obs = env.get_arm_observations()
                actions_arm = arm_policy(obs).to(env.env.device)
                env.plan(actions_arm[..., -2:])

            dog_obs = env.get_dog_observations()
            actions_dog = dog_policy(dog_obs).to(env.env.device)

        if arm_policy is None:
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
        default=1.0,
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
