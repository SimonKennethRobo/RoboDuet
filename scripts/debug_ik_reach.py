"""Standalone goal-reaching test bed for the stage-2 DLS-IK controller
(WBCEnv._solve_arm_dls_ik_step / _apply_stage2_arm_ik_action).

Spawns a handful of envs, opens the stage-2 switch, and for each episode
samples a fresh random SE(3) target the same way training does
(WBCEnv._resample_arm_target), then holds the arm action at zero (or a
constant small tanh residual via --residual) so the observed convergence is
purely a function of arm.ik.damping / arm.ik.step_gain -- this is the tool
for tuning those two numbers before spending GPU-hours on real RL training.

Run with a viewer to *see* the target (cyan sphere+axes) and EE frame
(yellow) converge:
    python scripts/debug_ik_reach.py --num_envs 4 --num_episodes 20

Headless, for a quick pass/fail sweep over many episodes:
    python scripts/debug_ik_reach.py --headless --num_envs 64 --num_episodes 50

Sweep IK gains without touching wbc.py:
    python scripts/debug_ik_reach.py --headless --damping 0.1 --step_gain 0.3

By default the base is pinned (root state rewritten back to its post-reset
pose every step) so results measure the IK/arm loop in isolation, without
confounding it with the dog policy's (in)ability to hold a stance under zero
action -- see --no_pin_base to test the combined system instead.
"""

import argparse

import isaacgym  # noqa: F401  (must import before torch)
from isaacgym import gymtorch
import torch

from go1_gym.envs.config import build_roboduet_config
from go1_gym.envs.config.core import RoboDuetRuntimeOptions
from go1_gym.envs.roboduet.wbc_env import WBCEnv
from go1_gym.envs.roboduet.wbc_env_wrapper import HistoryWrapper
from go1_gym.utils.global_switch import global_switch


def main(args):
    cfg = build_roboduet_config(options=RoboDuetRuntimeOptions(num_envs=args.num_envs, robot=args.robot))
    cfg.env.num_envs = args.num_envs
    cfg.terrain.num_rows = 1
    cfg.terrain.num_cols = 1
    cfg.terrain.curriculum = False
    # Mount-transform randomization varies the arm's effective reach per env,
    # which would make cross-env error stats noisier for what's meant to be
    # a controlled IK-tuning test.
    cfg.domain_rand.randomize_mount_position = False
    cfg.domain_rand.randomize_mount_rotation = False
    if args.damping is not None:
        cfg.arm.ik.damping = args.damping
    if args.step_gain is not None:
        cfg.arm.ik.step_gain = args.step_gain
    if args.residual_scale is not None:
        cfg.arm.ik.residual_scale = args.residual_scale
    if args.no_arm_dr:
        s2 = cfg.domain_rand.stage2_arm
        s2.randomize_Kp_factor = False
        s2.randomize_Kd_factor = False
        s2.randomize_motor_strength = False
        s2.randomize_motor_offset = False
    if args.fix_base:
        # Weld the trunk to the world so arm reaction forces can't wobble the
        # base mid-decimation -- isolates the IK integration from base
        # compliance (which pin_base only removes between control steps).
        cfg.asset.fix_base_link = True
    if args.no_arm_init_noise:
        cfg.env.stage1_arm_init_dof_pos_noise = 0.0
    if args.freeze_target:
        # The per-episode target is sampled on the first post-physics step
        # (deferred so the nominal reads the actual post-reset EE pose); make
        # the resample window effectively infinite so that first target holds
        # for the whole episode -- a clean per-target convergence measurement.
        cfg.arm.target.resample_time_s = [1.0e6, 1.0e6]
    if args.rp_deg is not None:
        r = __import__("math").radians(args.rp_deg)
        cfg.arm.target.roll_ee = [-r, r]
        cfg.arm.target.pitch_ee = [-r, r]
    if args.yaw_deg is not None:
        y = __import__("math").radians(args.yaw_deg)
        cfg.arm.target.yaw_ee = [-y, y]
    if args.pos_box is not None:
        b = args.pos_box
        cfg.arm.target.pos_range = [[-b, b], [-b, b], [-b, b]]
    if args.arm_stiff_scale != 1.0:
        for k in cfg.arm.control.stiffness_arm:
            cfg.arm.control.stiffness_arm[k] *= args.arm_stiff_scale
            cfg.arm.control.damping_arm[k] *= args.arm_stiff_scale**0.5

    raw_env = WBCEnv(sim_device=args.sim_device, headless=args.headless, cfg=cfg, physics_engine="SIM_PHYSX")
    env = HistoryWrapper(raw_env)
    env.reset()
    global_switch.switch_flag = True  # stage 2: arm target + IK path active
    env.reset()

    print(
        f"arm.ik: damping={cfg.arm.ik.damping} step_gain={cfg.arm.ik.step_gain} "
        f"residual_scale={cfg.arm.ik.residual_scale}  pin_base={args.pin_base}  "
        f"residual_action={args.residual}"
    )

    nominal_root = raw_env.root_states[: raw_env.num_envs].clone()

    def pin_base():
        if not args.pin_base:
            return
        raw_env.root_states[: raw_env.num_envs] = nominal_root
        raw_env.gym.set_actor_root_state_tensor(raw_env.sim, gymtorch.unwrap_tensor(raw_env.root_states))
        raw_env.gym.refresh_actor_root_state_tensor(raw_env.sim)

    all_final_pos_err, all_final_rot_err, all_converged = [], [], []
    for ep in range(args.num_episodes):
        env.reset()
        pin_base()
        if args.freeze_target:
            # Training periodically re-samples the target mid-episode (every
            # 2-3s, see arm.target.resample_time_s) so a long-running policy
            # sees more than one target per episode. For a clean per-target
            # convergence measurement we don't want that mid-episode -- push
            # the next resample far out so the target sampled at reset holds
            # for the whole episode.
            raw_env.arm_target_resample_steps[:] = 10**9
        # _resample_arm_target already ran inside reset(); force one fresh
        # error computation now so step 0's IK correction is not computed
        # against a stale error left over from whatever target was active
        # a moment ago (see the "stale error" bug this script exists to
        # avoid re-discovering the hard way).
        raw_env._update_ee_task_space_error()

        actions_dog = torch.zeros(env.num_envs, env.num_actions_loco, device=env.device)
        actions_arm = torch.full((env.num_envs, env.num_actions_arm), args.residual, device=env.device)

        pos_err_hist, rot_err_hist = [], []
        for step in range(args.episode_steps):
            env.step(actions_dog, actions_arm)
            pin_base()
            pos_err_hist.append(raw_env.ee_pos_err.norm(dim=-1).clone())
            rot_err_hist.append(raw_env.ee_rot_err_axis_angle.norm(dim=-1).clone())
            if args.verbose and step % args.print_every == 0:
                print(
                    f"  ep {ep} step {step}: pos_err(env0)={pos_err_hist[-1][0].item():.4f} "
                    f"rot_err(env0)={rot_err_hist[-1][0].item():.4f}"
                )

        final_pos, final_rot = pos_err_hist[-1], rot_err_hist[-1]
        converged = (final_pos < args.pos_threshold) & (final_rot < args.rot_threshold)
        all_final_pos_err.append(final_pos)
        all_final_rot_err.append(final_rot)
        all_converged.append(converged)
        print(
            f"episode {ep:3d}: final pos_err mean={final_pos.mean().item():.4f} max={final_pos.max().item():.4f}  "
            f"rot_err mean={final_rot.mean().item():.4f} max={final_rot.max().item():.4f}  "
            f"converged={100 * converged.float().mean().item():.0f}%"
        )

    all_pos = torch.cat(all_final_pos_err)
    all_rot = torch.cat(all_final_rot_err)
    all_conv = torch.cat(all_converged)
    print(f"\n=== SUMMARY over {args.num_episodes} episodes x {args.num_envs} envs ({all_pos.numel()} trials) ===")
    print(
        f"final pos_err: mean={all_pos.mean().item():.4f} median={all_pos.median().item():.4f} max={all_pos.max().item():.4f}"
    )
    print(
        f"final rot_err: mean={all_rot.mean().item():.4f} median={all_rot.median().item():.4f} max={all_rot.max().item():.4f}"
    )
    print(
        f"convergence rate (pos<{args.pos_threshold}m, rot<{args.rot_threshold}rad): "
        f"{100 * all_conv.float().mean().item():.1f}%"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", type=str, default="go2_x5", choices=["go1", "go2", "go2_x5"])
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--episode_steps", type=int, default=150, help="steps per episode before scoring")
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument(
        "--pin_base",
        action="store_true",
        default=True,
        help="rewrite root state back to nominal every step, isolating the arm/IK loop (default on)",
    )
    parser.add_argument("--no_pin_base", dest="pin_base", action="store_false")
    parser.add_argument(
        "--freeze_target",
        action="store_true",
        default=True,
        help="hold one target for the whole episode, disabling the periodic ~2-3s mid-episode "
        "resample training uses (default on -- needed for a clean per-target convergence read)",
    )
    parser.add_argument("--no_freeze_target", dest="freeze_target", action="store_false")
    parser.add_argument("--damping", type=float, default=None, help="override arm.ik.damping")
    parser.add_argument("--step_gain", type=float, default=None, help="override arm.ik.step_gain")
    parser.add_argument("--residual_scale", type=float, default=None, help="override arm.ik.residual_scale")
    parser.add_argument(
        "--fix_base",
        action="store_true",
        default=False,
        help="weld trunk to world (isolate arm/IK from base compliance)",
    )
    parser.add_argument(
        "--no_arm_dr", action="store_true", default=False, help="disable stage2 arm Kp/Kd/strength/offset randomization"
    )
    parser.add_argument("--rp_deg", type=float, default=None, help="override roll/pitch_ee target half-range (deg)")
    parser.add_argument("--yaw_deg", type=float, default=None, help="override yaw_ee target half-range (deg)")
    parser.add_argument("--pos_box", type=float, default=None, help="override symmetric pos_delta box half-size (m)")
    parser.add_argument(
        "--no_arm_init_noise",
        action="store_true",
        default=False,
        help="zero stage1_arm_init_dof_pos_noise so each episode starts at the nominal arm pose",
    )
    parser.add_argument(
        "--arm_stiff_scale",
        type=float,
        default=1.0,
        help="multiply all arm stiffness_arm by this (damping by sqrt) to probe gravity-droop limits",
    )
    parser.add_argument(
        "--residual",
        type=float,
        default=0.0,
        help="constant raw arm action in [-1,1] applied every step (0 = pure IK, no residual)",
    )
    parser.add_argument("--pos_threshold", type=float, default=0.03, help="success threshold, meters")
    parser.add_argument("--rot_threshold", type=float, default=0.15, help="success threshold, radians (bounded proxy)")
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="print per-step error, not just per-episode"
    )
    parser.add_argument("--print_every", type=int, default=25)
    args = parser.parse_args()
    main(args)
