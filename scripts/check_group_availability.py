"""How often is R5's consistency term actually available?

R5's penalty is masked off for any group inside the settling window that any
member's reset starts.  With a group of four that is a stronger condition than
it looks: if each environment has probability ``p`` of resetting somewhere
inside a window of ``settle_steps``, the group is usable only ``(1 - p)^4`` of
the time.

This script measures that directly by *forcing* falls at a chosen rate, rather
than waiting for a policy to produce them -- so the availability curve can be
read off without a trained policy and without the answer depending on how bad
that policy happens to be.

    python scripts/check_group_availability.py --fall_rate 0.002
    python scripts/check_group_availability.py --fall_rate 0.09 --steps 600

Why it exists: ``perf_group_desync_fraction`` in a training run is a sawtooth
over the settling window and is easy to misread from a single sample.  It also
sits near zero for the first few tens of iterations -- not because groups are
synchronised, but because most environments are still inside their first
episode and have never reset.  Reading either as "the term is available" is
wrong, and it was, twice: an earlier config comment claimed 0.36 early in
training when the true value there is ~1.0, and the 20k run was read as 0.94
at the end when its stage-3 average was 0.82.  ``perf_group_availability_ema``
is the number to plot; this script is how you predict it before running.

What this script is measuring is also what motivated the settling window.  The
mask used to latch until the group's next shared resample, which made
availability a function of the resample period and the episode length rather
than of the reset rate:

    masked fraction = 1 - (L / ((G+1) W)) * (1 - (1 - W/L)^(G+1))

At W = 500, L = 1000, G = 4 that is 0.61 **for a policy that never falls** --
the mask was driven by episode timeouts, not by falls.  With the settling window
the same expression holds with W replaced by ``settle_steps``, which is 50, so
the fall-free floor is ~0.09.
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
    settle = grouping.settle_steps
    episode = int(cfg.env.episode_length_s / base.dt)
    # Timeouts count too, and used to dominate: every env resets once per
    # episode whether or not it ever falls.
    hazard = args.fall_rate + 1.0 / max(episode, 1)
    window_p = 1.0 - (1.0 - hazard) ** settle
    print(f"  {args.num_envs} envs -> {grouping.num_groups} groups of "
          f"{grouping.group_size}; settling window {settle} steps, "
          f"shared resample interval {interval} steps")
    print(f"  forcing {args.fall_rate:.2%} of envs to fall per step "
          f"(mean episode ~{1 / max(hazard, 1e-9):.0f} steps, cap {episode})")
    print(f"  predicted: p(reset inside a window) = {window_p:.3f} -> "
          f"group usable {(1 - window_p) ** grouping.group_size:.1%} of the time")
    print(f"\n  {'step':>6}{'resets':>9}{'marks':>8}{'resyncs':>9}{'desync':>9}"
          f"{'avail_ema':>10}{'gain':>7}")

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
            ema = float(base.response_consistency_availability)
            gain = float(base.response_consistency_gain)
            print(f"  {step:>6}{resets:>9}{marks:>8}{resyncs:>9}{desync:>9.3f}"
                  f"{ema:>10.3f}{gain:>7.2f}")
            marks = resyncs = resets = 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
