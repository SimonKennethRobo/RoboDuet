"""R8.2 -- calibrate the reference model from a stage-1 policy.

R2's ``omega_n`` and ``rate_limit`` are starting values taken from the
requirements document.  R8.2 replaces them with values measured from what the
robot can actually do, and states two rules that are not optional:

**Take the 20th percentile, not the mean and not the maximum.**  A reference the
hard domains cannot realise leaves the policy choosing between failing to track
it and destabilising itself trying.  With margin, the consistency reward stays
reachable in every domain and nothing has to be traded away -- which is also the
answer to "how do you know this does not cost robustness".

**Bin the posture channels jointly by (domain, gait phase at the step).**  Body
pitch and height are changed by pushing against the stance legs, so the
authority available varies through the gait cycle -- in trot it drops sharply at
the double-support swap.  Binning by domain alone yields a model that is
realisable at the average phase and not at the unfavourable one.  The velocity
channels do not need this and are binned by domain only.

Usage::

    python scripts/calibrate_reference_model.py --policy <stage1 ckpt> \
        --out configs/reference_model_go2_x5.json

The policy must be a **stage-1** checkpoint: one trained with the curriculum in
stage 1, so reference tracking was never in its reward.  A policy already pulled
towards a reference model is not a neutral measurement of the plant, and using
one closes a loop that has no business being closed.

Diagnostic to watch afterwards: if the training reward develops a ripple at the
gait frequency, the posture channels were binned wrongly -- re-run this.  See
``go1_gym.response.curriculum.gait_frequency_ripple``.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import isaacgym  # noqa: F401  must precede torch
import torch

from go1_gym.envs.config import build_roboduet_config
from go1_gym.envs.roboduet.wbc_env import WBCEnv
from go1_gym.envs.roboduet.wbc_env_wrapper import HistoryWrapper
from go1_gym.response.calibration import (
    omega_from_rise_time,
    percentile,
    rate_limit_from_peak_rate,
    saturation_ratio,
)
from go1_gym.utils import global_switch

#: Channels whose authority varies through the gait cycle, so their samples are
#: binned jointly by (domain, phase).  R8.2 names exactly these two.
PHASE_BINNED_CHANNELS = ("height", "pitch")


def build_env(num_envs, sim_device, robot):
    args = argparse.Namespace(
        robot=robot, num_envs=num_envs, dyna_gait=True, goal_reaching=False,
        traj_tracking=False, arm_action_mode=None, no_reach_table=False,
        dyna_gait_min_frequency=0.0, video=False,
    )
    cfg = build_roboduet_config(args)
    cfg.env.arm_policy_enabled = False
    cfg.env.record_video = False
    # Excitation would fight the step commands this script issues.
    cfg.response.excitation.enabled = False
    # R8.2 measures over the FULL randomisation range and takes the 20th
    # percentile across domains.  Without this the env would be built at
    # iteration 0, i.e. at the stage-1 randomisation floor (0.30), and the
    # percentile would be taken over an easy domain set -- yielding an omega_n
    # that is unachievable in the domains the policy will actually be trained
    # in, which is precisely the failure the percentile rule exists to prevent.
    # Set before the env is constructed: friction, restitution and payload are
    # sampled in _create_envs and never resampled here.  The disturbance
    # schedule is left alone -- pushes during a step response would corrupt the
    # measurement, and stage 1 is where it is off.
    cfg.response.curriculum.randomization_floor = 1.0
    global_switch.pretrained_to_wbc_start = 10 ** 9
    global_switch.pretrained_to_wbc_end = 10 ** 9 + 1
    global_switch.init_sigmoid_lr()
    return HistoryWrapper(WBCEnv(sim_device=sim_device, headless=True, cfg=cfg)), cfg


def load_policy(path, cfg, device):
    from go1_gym_learn.ppo_cse_automatic.dog_ac import DogActorCritic

    model = DogActorCritic(
        num_obs=cfg.dog.dog_num_observations,
        num_privileged_obs=cfg.dog.dog_num_privileged_obs,
        num_obs_history=cfg.dog.dog_num_obs_history,
        num_actions=cfg.dog.dog_actions,
        use_adaptation_module=cfg.dog.use_adaptation_module,
    ).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def measure_channel(env, cfg, policy, channel, amplitude, settle_steps, hold_steps):
    """One step command on one channel; return per-env (rise time, peak rate).

    Every env is stepped at the same instant but sits in a different domain, so
    one rollout yields one sample per domain -- which is what the percentile is
    taken over.  The gait phase at the instant of the step is recorded alongside,
    for the posture channels' joint binning.
    """
    base = env.env
    device = base.device
    arm_actions = torch.zeros(base.num_envs, base.num_actions_arm, device=device)
    column = channel.cmd_index

    def act():
        observations = env.get_dog_observations()
        with torch.no_grad():
            return policy.act_inference({"obs_history": observations["obs_history"]})

    # Settle at zero on this channel so the step starts from rest.
    base.commands_dog[:, column] = 0.0
    for _ in range(settle_steps):
        env.step(act(), arm_actions)
        base.commands_dog[:, column] = 0.0

    phase_at_step = base.gait_indices.clone()
    baseline = base._response_measured()[:, channel.index].clone()
    base.commands_dog[:, column] = amplitude

    target = baseline + 0.5 * (amplitude - baseline)
    rise_step = torch.zeros(base.num_envs, device=device)
    risen = torch.zeros(base.num_envs, dtype=torch.bool, device=device)
    peak_rate = torch.zeros(base.num_envs, device=device)
    previous = baseline.clone()
    alive = torch.ones(base.num_envs, dtype=torch.bool, device=device)

    for step in range(1, hold_steps + 1):
        env.step(act(), arm_actions)
        base.commands_dog[:, column] = amplitude
        measured = base._response_measured()[:, channel.index]
        rate = (measured - previous).abs() / base.dt
        previous = measured.clone()
        peak_rate = torch.maximum(peak_rate, rate * alive.float())
        crossed = (~risen) & (measured >= target) & alive
        rise_step = torch.where(crossed, torch.full_like(rise_step, float(step)), rise_step)
        risen |= crossed
        # An env that terminated mid-step contributes nothing: its response is a
        # fall, not a step response, and a fall is arbitrarily fast.
        alive &= base.reset_buf == 0

    return {
        "rise_time": rise_step * base.dt,
        "peak_rate": peak_rate,
        "phase": phase_at_step,
        "valid": risen & alive,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", required=True, help="stage-1 dog checkpoint")
    parser.add_argument("--num_envs", type=int, default=512)
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--robot", type=str, default="go2_x5")
    parser.add_argument("--percentile", type=float, default=0.2,
                        help="R8.2 mandates the 20th percentile")
    parser.add_argument("--phase_bins", type=int, default=4)
    parser.add_argument("--settle_s", type=float, default=2.0)
    parser.add_argument("--hold_s", type=float, default=2.0)
    parser.add_argument("--repeats", type=int, default=4,
                        help="rollouts per channel; more phases at the step instant")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if not os.path.exists(args.policy):
        raise SystemExit(f"checkpoint not found: {args.policy}")
    if not 0.0 < args.percentile < 1.0:
        raise SystemExit(f"--percentile must be in (0, 1), got {args.percentile}")
    if abs(args.percentile - 0.2) > 1e-9:
        print(f"[warn] --percentile {args.percentile} deviates from the 0.2 R8.2 mandates")

    env, cfg = build_env(args.num_envs, args.sim_device, args.robot)
    base = env.env
    policy = load_policy(args.policy, cfg, base.device)
    amplitudes = dict(cfg.response.reward.calibration_amplitudes)
    settle_steps = int(args.settle_s / base.dt)
    hold_steps = int(args.hold_s / base.dt)

    channels = []
    for index, spec in enumerate(base.response_ref.channels):
        spec.index = index          # measure_channel indexes the response vector
        channels.append(spec)

    results = {}
    print(f"  {args.num_envs} envs x {args.repeats} rollouts, "
          f"p{args.percentile:.0%}, {args.phase_bins} phase bins\n")
    print(f"  {'channel':<9}{'samples':>9}{'omega_n':>10}{'rate lim':>10}"
          f"{'sat':>7}{'was wn':>9}{'was rl':>9}")
    for channel in channels:
        amplitude = float(amplitudes[channel.name])
        buckets = defaultdict(list)
        for _ in range(args.repeats):
            sample = measure_channel(
                env, cfg, policy, channel, amplitude, settle_steps, hold_steps
            )
            valid = sample["valid"]
            if not bool(valid.any()):
                continue
            phase_bin = torch.floor(sample["phase"] * args.phase_bins).long() % args.phase_bins
            rise = sample["rise_time"][valid].tolist()
            rate = sample["peak_rate"][valid].tolist()
            bins = phase_bin[valid].tolist()
            for k in range(len(rise)):
                # Velocity channels: one bucket, i.e. binned by domain only.
                # Posture channels: one bucket per phase, so the percentile is
                # taken within a phase and the unfavourable phase cannot be
                # averaged away by the favourable one.
                key = bins[k] if channel.name in PHASE_BINNED_CHANNELS else 0
                buckets[key].append((rise[k], rate[k]))

        if not buckets:
            print(f"  {channel.name:<9}{0:>9}   no valid step responses -- policy fell?")
            continue

        # Percentile within each bucket, then the worst bucket wins: a reference
        # realisable at the average phase but not the unfavourable one is
        # exactly what R8.2 forbids.
        per_bucket = {}
        for key, samples in buckets.items():
            rise = percentile([s[0] for s in samples], 1.0 - args.percentile)
            rate = percentile([s[1] for s in samples], args.percentile)
            per_bucket[key] = (omega_from_rise_time(rise), rate_limit_from_peak_rate(rate))
        omega = min(v[0] for v in per_bucket.values())
        rate_limit = min(v[1] for v in per_bucket.values())
        count = sum(len(v) for v in buckets.values())
        ratio = saturation_ratio(rate_limit, omega, amplitude)
        print(f"  {channel.name:<9}{count:>9}{omega:>10.3f}{rate_limit:>10.4f}"
              f"{ratio:>7.2f}{channel.omega_n:>9.2f}{channel.rate_limit:>9.3f}")
        results[channel.name] = {
            "omega_n": omega,
            "rate_limit": rate_limit,
            "amplitude": amplitude,
            "samples": count,
            "phase_binned": channel.name in PHASE_BINNED_CHANNELS,
            "per_bucket": {str(k): {"omega_n": v[0], "rate_limit": v[1]}
                           for k, v in per_bucket.items()},
            "saturation_ratio": ratio,
            "previous": {"omega_n": channel.omega_n, "rate_limit": channel.rate_limit},
        }

    print("\n  saturation ratio near 1 = bandwidth-limited; well below = slew-limited")
    print("\nconfig snippet for go1_gym/envs/config/wbc.py:\n")
    print('    "response.omega_n": {'
          + ", ".join(f'"{k}": {v["omega_n"]:.2f}' for k, v in results.items()) + "},")
    print('    "response.rate_limit": {'
          + ", ".join(f'"{k}": {v["rate_limit"]:.3f}' for k, v in results.items()) + "},")
    print("\nRe-run scripts/calibrate_reward_sigma.py afterwards: sigma is "
          "calibrated against omega_n and rate_limit and is now stale.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "policy": args.policy,
                    "robot": args.robot,
                    "percentile": args.percentile,
                    "phase_bins": args.phase_bins,
                    "num_envs": args.num_envs,
                    "repeats": args.repeats,
                    "channels": results,
                },
                handle,
                indent=2,
            )
            handle.write("\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
