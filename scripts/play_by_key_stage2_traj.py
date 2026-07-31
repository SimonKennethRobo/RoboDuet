import argparse

import isaacgym  # noqa: F401 - must be imported before torch
import torch

from go1_gym.envs.config import configure_privileged_obs_dims
from go1_gym.envs.roboduet import KeyboardStage2TrajWrapper
from go1_gym.utils.viz import add_rerun_args, make_rerun_logger


def main(args):
    from go1_gym.utils.global_switch import global_switch
    from scripts.load_policy import load_arm_policy, load_dog_policy, load_env

    global_switch.open_switch()
    ckpt_id_arg = str(args.ckptid)
    ckpt_id = "last" if ckpt_id_arg == "last" else ckpt_id_arg.zfill(6)

    env, cfg = load_env(
        args.logdir,
        wrapper=KeyboardStage2TrajWrapper,
        headless=args.headless,
        device=args.sim_device,
        robot=args.robot,
    )
    configure_privileged_obs_dims(cfg)
    dog_policy = load_dog_policy(args.logdir, ckpt_id, cfg)
    arm_policy = load_arm_policy(args.logdir, ckpt_id, cfg)

    env.env.enable_viewer_sync = True
    env.reset()

    print(
        "[traj eval] moving SE(3) trajectory tracking; "
        "N=next traj, ]=harder, [=easier, R=restart, P=pause/resume",
        flush=True,
    )
    if args.headless:
        print("[traj eval] headless: no keyboard input or viewer overlay", flush=True)

    rerun_logger = make_rerun_logger(args, app_id="roboduet_play_by_key_stage2_traj")

    for step in range(args.num_eval_steps):
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

        if args.print_every > 0 and step % args.print_every == 0:
            _print_tracking_stats(env.env)


def _print_tracking_stats(env):
    """One-line live tracking summary for env 0."""
    L = float(env.traj_batch.L[0])
    s = float(env.traj_s[0])
    progress = s / max(L, 1e-6)
    sdot_ref = float(env.traj_batch.sdot_ref(env.traj_sim_time[0:1])[0])
    print(
        f"[traj] s/L={progress:5.1%}  d_lat={float(env.traj_d_lat[0]):.3f}m  "
        f"timing_err={float(env.traj_timing_err[0]):+.3f}m  "
        f"sdot={float(env.traj_sdot_meas[0]):.2f}/{sdot_ref:.2f} m/s  "
        f"rho={float(env.goal_rho[0]):.2f}  t={float(env.traj_sim_time[0]):.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RoboDuet stage-2 trajectory-tracking keyboard eval")
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--ckptid", type=str, default="last")
    parser.add_argument("--robot", type=str, default="go2", choices=["go1", "go2", "go2_x5"])
    parser.add_argument("--num_eval_steps", type=int, default=30000)
    parser.add_argument("--print_every", type=int, default=50, help="steps between tracking-stat prints (0=off)")
    add_rerun_args(parser)
    main(parser.parse_args())
