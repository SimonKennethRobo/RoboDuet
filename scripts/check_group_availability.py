"""How often is R5's consistency term actually available?

R5's penalty is masked off for any group whose phase and command timing have
drifted apart, which happens as soon as one member falls and resets.  With a
group of four that is a much stronger condition than it looks: if each
environment has probability ``p`` of resetting somewhere inside the group's
shared resample window, the group is usable only ``(1 - p)^4`` of the time.

This script measures that directly by *forcing* falls at a chosen rate, rather
than waiting for a policy to produce them -- so the availability curve can be
read off without a trained policy and without the answer depending on how bad
that policy happens to be.

    python scripts/check_group_availability.py --fall_rate 0.002
    python scripts/check_group_availability.py --fall_rate 0.09 --steps 600

Why it exists: ``perf_group_desync_fraction`` in a training run is NOT monotone
and is easy to misread.  It sits near zero early on -- not because groups are
synchronised, but because most environments are still inside their first
episode and have never reset -- then jumps to 1.0 at the first timeout wave,
then decays as the policy stops falling.  Reading the early part as "the term is
available" is wrong, and it was: an earlier version of the config comment
claimed 0.36 early in training when the true value there is ~1.0.
"""

import argparse
import sys

import isaacgym  # noqa: F401  must precede torch
from isaacgym import gymtorch
import torch

from go1_gym.envs.config import build_roboduet_config
from go1_gym.envs.roboduet.wbc_env import WBCEnv
from go1_gym.envs.roboduet.wbc_env_wrapper import HistoryWrapper
from go1_gym.utils import global_switch


def build_env(num_envs, sim_device, robot):
    args = argparse.Namespace(
        robot=robot, num_envs=num_envs, dyna_gait=True, goal_reaching=False,
        traj_tracking=False, arm_action_mode=None, no_reach_table=False,
        dyna_gait_min_frequency=0.0, video=False,
    )
    cfg = build_roboduet_config(args)
    cfg.env.arm_policy_enabled = False
    cfg.env.record_video = False
    global_switch.pretrained_to_wbc_start = 10 ** 9
    global_switch.pretrained_to_wbc_end = 10 ** 9 + 1
    global_switch.init_sigmoid_lr()
    return HistoryWrapper(WBCEnv(sim_device=sim_device, headless=True, cfg=cfg)), cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fall_rate", type=float, default=0.002,
                        help="per-env probability of a forced fall each step")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--robot", type=str, default="go2_x5")
    parser.add_argument("--report_every", type=int, default=50)
    args = parser.parse_args()

    env, cfg = build_env(args.num_envs, args.sim_device, args.robot)
    base = env.env
    grouping = base.grouping
    if grouping.num_groups == 0:
        raise SystemExit("grouping is disabled, or too few envs to form a group")

    dog_actions = torch.zeros(args.num_envs, cfg.dog.num_actions_loco, device=base.device)
    arm_actions = torch.zeros(args.num_envs, base.num_actions_arm, device=base.device)
    env.reset()

    scoreable = grouping.is_grouped & ~grouping.is_twin
    marks = resyncs = resets = 0
    original_mark, original_resync = grouping.mark_desync, grouping.resync

    def spy_mark(env_ids):
        nonlocal marks
        marks += int(env_ids.numel())
        return original_mark(env_ids)

    def spy_resync(group_ids):
        nonlocal resyncs
        resyncs += int(group_ids.numel())
        return original_resync(group_ids)

    grouping.mark_desync, grouping.resync = spy_mark, spy_resync

    interval = int(cfg.commands.resampling_time / base.dt)
    window_p = 1.0 - (1.0 - args.fall_rate) ** interval
    print(f"  {args.num_envs} envs -> {grouping.num_groups} groups of "
          f"{grouping.group_size}; shared resample interval {interval} steps")
    print(f"  forcing {args.fall_rate:.2%} of envs to fall per step "
          f"(mean episode ~{1 / max(args.fall_rate, 1e-9):.0f} steps)")
    print(f"  predicted: p(reset inside a window) = {window_p:.3f} -> "
          f"group usable {(1 - window_p) ** grouping.group_size:.1%} of the time")
    print(f"\n  {'step':>6}{'resets':>9}{'marks':>8}{'resyncs':>9}{'desync':>9}")

    for step in range(1, args.steps + 1):
        doomed = (
            torch.rand(args.num_envs, device=base.device) < args.fall_rate
        ).nonzero(as_tuple=False).flatten()
        if doomed.numel():
            base.root_states[doomed, 2] = -5.0
            base.gym.set_actor_root_state_tensor(
                base.sim, gymtorch.unwrap_tensor(base.root_states)
            )
        env.step(dog_actions, arm_actions)
        resets += int(base.reset_buf.sum())
        if step % args.report_every == 0:
            desync = 1.0 - float(grouping.valid[scoreable].mean())
            print(f"  {step:>6}{resets:>9}{marks:>8}{resyncs:>9}{desync:>9.3f}")
            marks = resyncs = resets = 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
