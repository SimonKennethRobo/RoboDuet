import argparse
import time

import isaacgym
import torch
from isaacgym.torch_utils import *

from go1_gym.envs import *
from go1_gym.envs.config import configure_privileged_obs_dims
from go1_gym.envs.roboduet.wbc_env_wrapper import KeyboardStage1Wrapper
from go1_gym.utils.viz import add_rerun_args, make_rerun_logger
from scripts.load_policy import load_arm_policy, load_dog_policy, load_env

x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
l_cmd, p_cmd, y_cmd = 0.5, 0.2, 0.0
roll_cmd, pitch_cmd, yaw_cmd = 0.0, 0.0, 0.0
lock_arm = False
# dog body pose
body_pitch_cmd = 0.0
body_roll_cmd = 0.0
body_height_delta_cmd = 0.0
# gait params (only used when use_dynamic_gait=True)
gait_freq_cmd = 4.0
footswing_height_cmd = 0.06
stance_width_cmd = 0.3
stance_length_cmd = 0.4
gait_duration_cmd = 0.5


def main(args):
    global \
        x_vel_cmd, \
        y_vel_cmd, \
        yaw_vel_cmd, \
        l_cmd, \
        p_cmd, \
        y_cmd, \
        roll_cmd, \
        pitch_cmd, \
        yaw_cmd, \
        logdir, \
        ckpt_id, \
        lock_arm

    logdir = args.logdir
    lock_arm = bool(getattr(args, "lock_arm", False))
    ckpt_id_arg = str(args.ckptid)
    ckpt_id = "last" if ckpt_id_arg == "last" else ckpt_id_arg.zfill(6)

    from go1_gym.utils.global_switch import global_switch

    stage1_only = True
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

    env, cfg = load_env(
        logdir,
        wrapper=KeyboardStage1Wrapper,
        headless=args.headless,
        device=args.sim_device,
        robot=getattr(args, "robot", None),
    )
    dog_policy = load_dog_policy(logdir, ckpt_id, cfg)
    arm_policy = None if stage1_only else load_arm_policy(logdir, ckpt_id, cfg)
    if stage1_only and getattr(args, "disable_stage1_arm_curriculum", False):
        env.env.cfg.env.stage1_arm_curriculum = False
    if lock_arm:
        env.env.cfg.env.stage1_arm_curriculum = False
    configure_privileged_obs_dims(cfg)

    env.env.enable_viewer_sync = True

    rerun_logger = make_rerun_logger(args, app_id="roboduet_play_by_key_stage1")

    num_eval_steps = getattr(args, "num_eval_steps", 30000)

    obs = env.reset()

    n_cmd = env.commands_dog.shape[1]
    if n_cmd > 0:
        env.commands_dog[:, 0] = x_vel_cmd
    if n_cmd > 1:
        env.commands_dog[:, 1] = y_vel_cmd
    if n_cmd > 2:
        env.commands_dog[:, 2] = yaw_vel_cmd
    if n_cmd > 3:
        env.commands_dog[:, 3] = body_pitch_cmd
    if n_cmd > 4:
        env.commands_dog[:, 4] = body_roll_cmd
    if n_cmd > 5:
        env.commands_dog[:, 5] = body_height_delta_cmd
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

    env.commands_arm[:, 0] = l_cmd
    env.commands_arm[:, 1] = p_cmd
    env.commands_arm[:, 2] = y_cmd
    env.commands_arm[:, 3] = roll_cmd
    env.commands_arm[:, 4] = pitch_cmd
    env.commands_arm[:, 5] = yaw_cmd

    if lock_arm:
        print("[arm] LOCKED — zero actions sent every step", flush=True)

    for i in range(num_eval_steps):
        with torch.no_grad():
            if lock_arm or arm_policy is None:
                actions_arm = env.arm_fake_actions
            else:
                obs = env.get_arm_observations()
                actions_arm = arm_policy(obs)
                env.plan(actions_arm[..., -2:])

            dog_obs = env.get_dog_observations()
            actions_dog = dog_policy(dog_obs).to(env.env.device)

        if lock_arm or arm_policy is None:
            env.step(actions_dog, actions_arm)
        else:
            env.step(actions_dog, actions_arm[..., :-2].to(env.env.device))

        rerun_logger.log(env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RoboDuet — keyboard inference")
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--ckptid", type=str, default="last")
    parser.add_argument("--robot", type=str, default="go2", choices=["go1", "go2"])
    parser.add_argument("--num_eval_steps", type=int, default=30000)
    parser.add_argument(
        "--stage1_arm_intensity",
        type=float,
        default=1,
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
    add_rerun_args(parser)

    args = parser.parse_args()
    main(args)
