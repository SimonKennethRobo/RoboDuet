"""play_by_joy.py — policy inference with ROS2 joystick control.

Usage::

    python scripts/play_by_joy.py \\
        --logdir runs/test_roboduet/2024-10-13/auto_train/003436.678552_seed9145 \\
        --ckptid 40000

Controller mapping is read from config/joy_mapping.yaml by default.
Pass --joy_config to override with a custom YAML file.

Make sure `ros2 run joy joy_node` is running before starting this script.
"""

import time
import argparse

import isaacgym  # noqa: F401 – must be imported before torch
import torch
from isaacgym.torch_utils import *  # noqa: F403

from go1_gym.envs import *  # noqa: F403
from go1_gym.envs.automatic import JoyWrapper
from scripts.load_policy import load_dog_policy, load_arm_policy, load_env

# Default run/checkpoint (overridden by CLI args)
logdir = "runs/test_roboduet/2024-10-13/auto_train/003436.678552_seed9145"
ckpt_id = "040000"

# Initial command values
x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
l_cmd, p_cmd, y_cmd = 0.5, 0.2, 0.0
roll_cmd, pitch_cmd, yaw_cmd = 0.1, 0.5, 0.0


def play_go1(args):
    global logdir, ckpt_id

    logdir = args.logdir
    ckpt_id = str(args.ckptid).zfill(6)

    from go1_gym.utils.global_switch import global_switch
    global_switch.open_switch()

    # Build a factory that passes the joy_config_path through to JoyWrapper
    def make_wrapper(sim_device, headless, cfg):
        return JoyWrapper(
            sim_device=sim_device,
            headless=headless,
            cfg=cfg,
            joy_config_path=args.joy_config or None,
        )

    env, cfg = load_env(logdir, wrapper=make_wrapper,
                        headless=args.headless, device=args.sim_device)
    dog_policy = load_dog_policy(logdir, ckpt_id, cfg)
    arm_policy = load_arm_policy(logdir, ckpt_id, cfg)

    env.env.enable_viewer_sync = True

    num_eval_steps = 30000

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

    obs = env.get_arm_observations()
    for _ in range(num_eval_steps):
        with torch.no_grad():
            obs = env.get_arm_observations()
            actions_arm = arm_policy(obs)
            env.plan(actions_arm[..., -2:])

            dog_obs = env.get_dog_observations()
            actions_dog = dog_policy(dog_obs)

        env.step(actions_dog, actions_arm[..., :-2])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RoboDuet — joystick inference")
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--ckptid", type=int, default=40000)
    parser.add_argument(
        "--joy_config",
        type=str,
        default=None,
        help="Path to a custom joy_mapping.yaml (default: config/joy_mapping.yaml)",
    )
    args = parser.parse_args()
    play_go1(args)
