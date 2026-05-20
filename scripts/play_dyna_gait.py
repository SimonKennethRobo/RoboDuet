import argparse
import time

import isaacgym

assert isaacgym
import torch
from isaacgym.torch_utils import *

from go1_gym.envs import *
from go1_gym.envs.roboduet import KeyboardWrapper
from scripts.load_policy import load_arm_policy, load_dog_policy, load_env

x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
l_cmd, p_cmd, y_cmd = 0.5, 0.2, 0.0
roll_cmd, pitch_cmd, yaw_cmd = 0.1, 0.5, 0.0


def play_go1(args):
    global x_vel_cmd, y_vel_cmd, yaw_vel_cmd, l_cmd, p_cmd, y_cmd, roll_cmd, pitch_cmd, yaw_cmd

    logdir = args.logdir
    ckpt_id = str(args.ckptid).zfill(6)


    from go1_gym.utils.global_switch import global_switch

    global_switch.open_switch()

    env, cfg = load_env(logdir, wrapper=KeyboardWrapper, headless=args.headless, device=args.sim_device)
    dog_policy = load_dog_policy(logdir, ckpt_id, cfg)
    arm_policy = load_arm_policy(logdir, ckpt_id, cfg)

    env.env.enable_viewer_sync = True
    n_plan = env.num_plan_actions

    num_eval_steps = 30000

    obs = env.reset()

    env.commands_dog[:, 0] = x_vel_cmd
    env.commands_dog[:, 1] = y_vel_cmd
    env.commands_dog[:, 2] = yaw_vel_cmd
    env.commands_arm[:, 0] = l_cmd
    env.commands_arm[:, 1] = p_cmd
    env.commands_arm[:, 2] = y_cmd
    env.commands_arm[:, 3] = roll_cmd
    env.commands_arm[:, 4] = pitch_cmd
    env.commands_arm[:, 5] = yaw_cmd

    count = 0

    obs = env.get_arm_observations()
    for i in range(num_eval_steps):
        with torch.no_grad():
            t1 = time.time()

            obs = env.get_arm_observations()
            actions_arm = arm_policy(obs)
            env.plan(actions_arm[..., -n_plan:])

            dog_obs = env.get_dog_observations()
            actions_dog = dog_policy(dog_obs)
        ret = env.step(actions_dog, actions_arm[..., :-n_plan])

        if count % 500 == 0 and cfg.commands.use_dynamic_gait:
            print(
                f"step={count:6d}  "
                f"freq={env.commands_dog[0, 5]:.2f}  "
                f"swing_h={env.commands_dog[0, 6]:.3f}  "
                f"stance_w={env.commands_dog[0, 7]:.3f}  "
                f"stance_l={env.commands_dog[0, 8]:.3f}  "
                f"dur={env.commands_dog[0, 9]:.3f}  "
                f"pitch={env.commands_dog[0, 3]:.3f}  "
                f"roll={env.commands_dog[0, 4]:.3f}"
            )
        count += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Go1 Dyna Gait Play")
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--ckptid", type=int, default=40000)

    args = parser.parse_args()
    play_go1(args)
