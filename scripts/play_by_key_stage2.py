import argparse

import isaacgym  # noqa: F401 - must be imported before torch
import torch

from go1_gym.envs.config import configure_privileged_obs_dims
from go1_gym.envs.roboduet import KeyboardStage2Wrapper
from go1_gym.utils.viz import add_rerun_args, make_rerun_logger


def main(args):
    from go1_gym.utils.global_switch import global_switch
    from scripts.load_policy import load_arm_policy, load_dog_policy, load_env

    global_switch.open_switch()
    ckpt_id_arg = str(args.ckptid)
    ckpt_id = "last" if ckpt_id_arg == "last" else ckpt_id_arg.zfill(6)

    env, cfg = load_env(
        args.logdir,
        wrapper=KeyboardStage2Wrapper,
        headless=args.headless,
        device=args.sim_device,
        robot=args.robot,
    )
    configure_privileged_obs_dims(cfg)
    dog_policy = load_dog_policy(args.logdir, ckpt_id, cfg)
    arm_policy = load_arm_policy(args.logdir, ckpt_id, cfg)

    env.env.enable_viewer_sync = True
    env.reset()
    env.initialize_stage2_marker()

    mode = "whole-body upper policy" if env.num_plan_actions == 6 else "legacy stage-2 arm policy"
    print(
        "[stage2 marker] "
        f"{mode}; W/S=x, A/D=y, Q/E=z, I/K=roll, J/L=pitch, U/O=yaw, C=confirm",
        flush=True,
    )
    if args.headless:
        print("[stage2 marker] headless mode has no keyboard input or marker rendering", flush=True)

    rerun_logger = make_rerun_logger(args, app_id="roboduet_play_by_key_stage2")

    for _ in range(args.num_eval_steps):
        with torch.no_grad():
            arm_obs = env.get_arm_observations()
            actions_arm = arm_policy(arm_obs).to(env.env.device)
            if env.num_plan_actions > 0:
                env.plan(actions_arm)

            dog_obs = env.get_dog_observations()
            actions_dog = dog_policy(dog_obs).to(env.env.device)

        env.step(
            actions_dog,
            actions_arm[..., : env.env.num_actions_arm],
        )
        rerun_logger.log(env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RoboDuet stage-2 keyboard EE-goal marker")
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--ckptid", type=str, default="last")
    parser.add_argument("--robot", type=str, default="go2", choices=["go1", "go2"])
    parser.add_argument("--num_eval_steps", type=int, default=30000)
    add_rerun_args(parser)
    main(parser.parse_args())
