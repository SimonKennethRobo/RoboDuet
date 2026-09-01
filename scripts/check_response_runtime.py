"""Runtime acceptance checks for the response-consistent locomotion policy.

The unit tests under ``go1_gym/response/`` pin the maths on synthetic signals.
These checks are the other half: they run the real IsaacGym env and verify the
acceptance criteria in ``docs/project-design-rlmpc-v3-coding.md`` that only make
sense on a moving robot.

    python scripts/check_response_runtime.py --check all
    python scripts/check_response_runtime.py --check r3 --policy <ckpt.pt>

R1  frozen command channels stay inside their bands and only change at a
    resample; body roll is identically zero; gait frequency stays in its band.
R2  the reference model's position never advances faster than its rate limit,
    peaks exactly at the limit, and holds unit steady-state gain.
R3  the phase-conditioned residual is a clean low-order periodic waveform.
    Needs a walking policy, so pass --policy; without one this check is skipped.
R5  a group shares one command vector and one gait phase, its twin really is
    nominal, and a fall really does switch the group's consistency term off
    until the next shared resample.
R6  the identification environments really are excited (jump count, chirp band),
    really are excluded from the curriculum, and really are the only ones
    touched.
"""

import argparse
import math
import os
import sys

import isaacgym  # noqa: F401  must precede torch
from isaacgym import gymtorch
import torch

from go1_gym.envs.config import build_roboduet_config
from go1_gym.envs.roboduet.wbc_env import WBCEnv
from go1_gym.envs.roboduet.wbc_env_wrapper import HistoryWrapper
from go1_gym.response import (
    DECISION_CMD_INDEX,
    DOG_COMMAND_NAMES,
    FROZEN_CMD_INDEX,
    SEMI_FREE_CMD_INDEX,
)
from go1_gym.response.excitation import CHIRP, SIGNAL_NAMES
from go1_gym.utils import global_switch


def build_env(num_envs, sim_device, robot="go2_x5"):
    args = argparse.Namespace(
        robot=robot, num_envs=num_envs, dyna_gait=True, goal_reaching=False,
        traj_tracking=False, arm_action_mode=None, no_reach_table=False,
        dyna_gait_min_frequency=0.0, video=False,
    )
    cfg = build_roboduet_config(args)
    cfg.env.arm_policy_enabled = False
    cfg.env.record_video = False
    # Keep the stage-2 switch permanently shut: these checks are stage-1 only.
    global_switch.pretrained_to_wbc_start = 10 ** 9
    global_switch.pretrained_to_wbc_end = 10 ** 9 + 1
    global_switch.init_sigmoid_lr()
    env = HistoryWrapper(WBCEnv(sim_device=sim_device, headless=True, cfg=cfg))
    return env, cfg


def zero_actions(env, cfg):
    base = env.env
    return (
        torch.zeros(cfg.env.num_envs, cfg.dog.num_actions_loco, device=base.device),
        torch.zeros(cfg.env.num_envs, base.num_actions_arm, device=base.device),
    )


def load_dog_policy(path, cfg, device):
    from go1_gym_learn.ppo_cse_automatic.dog_ac import DogActorCritic

    ac = DogActorCritic(
        cfg.dog.dog_num_observations,
        cfg.dog.dog_num_privileged_obs,
        cfg.dog.dog_num_obs_history,
        cfg.dog.dog_actions,
        use_adaptation_module=False,
    )
    state = torch.load(path, map_location="cpu")
    missing, unexpected = ac.load_state_dict(state, strict=False)
    if missing:
        raise SystemExit(
            f"checkpoint {path} is missing {len(missing)} tensors, e.g. {missing[:3]}.\n"
            "The observation layout has probably changed since it was trained."
        )
    return ac.eval().to(device)


# ---------------------------------------------------------------------------
# R1
# ---------------------------------------------------------------------------


def check_r1(env, cfg, steps=1200):
    """Frozen channels constant, semi-free channel inside its band.

    "Frozen" per R1 means pinned, or jittered inside a narrow band for
    robustness -- so the test is (a) every value inside the configured band and
    (b) the value only ever changes at a resample, never during one.  Asserting
    "constant over the whole rollout" instead is a false alarm waiting to
    happen, because the narrow-band jitter is resampled once per episode.
    """
    base = env.env
    env.reset()
    base.episode_length_buf = torch.randint_like(
        base.episode_length_buf, high=int(base.max_episode_length)
    )
    dog_a, arm_a = zero_actions(env, cfg)

    history = []
    for _ in range(steps):
        env.step(dog_a, arm_a)
        history.append(base.commands_dog.clone())
    commands = torch.stack(history)

    bands = {
        "body_roll": cfg.commands.limit_body_roll,
        "footswing_height": cfg.commands.limit_footswing_height,
        "stance_width": cfg.commands.limit_stance_width,
        "stance_length": cfg.commands.limit_stance_length,
        "gait_duration": cfg.commands.limit_gait_duration,
    }
    # Resamples every 500 steps plus a reset per episode, over a randomised
    # start phase: a generous upper bound on legitimate changes.
    max_changes = steps // 500 + 4

    failures = []
    group = {i: "frozen" for i in FROZEN_CMD_INDEX}
    group.update({i: "semi-free" for i in SEMI_FREE_CMD_INDEX})
    group.update({i: "decision" for i in DECISION_CMD_INDEX})

    print(f"  {steps} steps x {cfg.env.num_envs} envs")
    print(f"  {'idx':<4}{'name':<20}{'group':<11}{'min':>9}{'max':>9}{'changes/env':>13}")
    for index, name in enumerate(DOG_COMMAND_NAMES):
        column = commands[:, :, index]
        changes = (column[1:] != column[:-1]).sum(dim=0).max().item()
        print(
            f"  {index:<4}{name:<20}{group[index]:<11}"
            f"{column.min():>9.4f}{column.max():>9.4f}{changes:>13.0f}"
        )
        if group[index] != "frozen":
            continue
        low, high = bands[name]
        if column.min().item() < low - 1e-6 or column.max().item() > high + 1e-6:
            failures.append(
                f"{name} left its band [{low}, {high}] -> "
                f"[{column.min():.4f}, {column.max():.4f}]"
            )
        if changes > max_changes:
            failures.append(
                f"{name} changed {changes:.0f}x per env, more than the {max_changes} "
                "resample/reset events -- it is varying mid-episode"
            )

    roll = commands[:, :, DOG_COMMAND_NAMES.index("body_roll")]
    if not torch.all(roll == 0.0):
        failures.append(f"body roll is not identically 0: [{roll.min():.5f}, {roll.max():.5f}]")

    frequency = commands[:, :, DOG_COMMAND_NAMES.index("gait_frequency")]
    moving = frequency > 0.01
    if moving.any():
        low, high = cfg.commands.limit_gait_frequency
        span = (frequency[moving].min().item(), frequency[moving].max().item())
        standing = (~moving).float().mean().item()
        print(
            f"  gait frequency while moving: [{span[0]:.3f}, {span[1]:.3f}]; "
            f"{standing:.1%} of samples are standing (clock pinned to 0)"
        )
        if span[0] < low - 1e-4 or span[1] > high + 1e-4:
            failures.append(f"gait frequency left [{low}, {high}] -> {span}")
    return failures


# ---------------------------------------------------------------------------
# R2
# ---------------------------------------------------------------------------


def check_r2(env, cfg, steps=900):
    """Rate saturation really bounds the reference, and the DC gain is 1."""
    base = env.env
    reference = base.response_ref
    env.reset()
    dog_a, arm_a = zero_actions(env, cfg)

    xi_hist, rate_hist, cmd_hist, reset_hist = [], [], [], []
    for _ in range(steps):
        env.step(dog_a, arm_a)
        xi_hist.append(reference.xi.clone())
        rate_hist.append(reference.xi_dot.clone())
        cmd_hist.append(reference.gather_commands(base.commands_dog).clone())
        reset_hist.append(base.reset_buf.clone())
    xi = torch.stack(xi_hist)
    rate = torch.stack(rate_hist)
    commands = torch.stack(cmd_hist)
    resets = torch.stack(reset_hist).bool()

    failures = []
    if not torch.isfinite(xi).all():
        failures.append("reference state contains non-finite values")

    print(f"  {'channel':<10}{'peak rate':>11}{'limit':>9}{'max |dxi|':>12}{'limit*dt':>11}")
    for index, name in enumerate(reference.channel_names):
        limit = reference.rate_limit[0, index].item()
        # Skip steps where the env reset: the reference realigns to the
        # measurement there, which is a legitimate jump.
        keep = ~(resets[1:] | resets[:-1])
        increments = (xi[1:, :, index] - xi[:-1, :, index]).abs()[keep]
        peak_rate = rate[:, :, index].abs().max().item()
        largest = increments.max().item()
        bound = limit * base.dt
        print(f"  {name:<10}{peak_rate:>11.4f}{limit:>9.3f}{largest:>12.5f}{bound:>11.5f}")
        if peak_rate > limit + 1e-5:
            failures.append(f"{name}: peak rate {peak_rate:.4f} exceeds the limit {limit}")
        # 2% covers the single boundary-crossing step, which takes the linear
        # branch and is the one place the piecewise update is O(dt) approximate.
        if largest > bound * 1.02:
            failures.append(
                f"{name}: reference advanced {largest:.5f} in one step, over rate_limit*dt "
                f"= {bound:.5f}"
            )

    # Steady-state gain, over windows where the command was genuinely constant
    # for 200 consecutive steps AND no reset realigned the reference.  Comparing
    # only the two endpoints of the window -- which is what this did before R6 --
    # is not the same statement: a chirp command returns to its old value twice
    # per cycle having swept the whole band in between, and the reference is
    # then correctly nowhere near it.
    window = 200
    quiet = (commands[1:] != commands[:-1]).any(dim=-1) | resets[1:] | resets[:-1]
    cumulative = torch.cat(
        [torch.zeros(1, quiet.shape[1], device=quiet.device), quiet.float().cumsum(dim=0)]
    )
    held = (cumulative[window:] - cumulative[:-window]) == 0
    if held.any():
        error = (xi[window:] - commands[window:]).abs()[held].max().item()
        print(f"  steady-state |xi - u| after {window} held steps: {error:.5f} "
              f"({int(held.sum())} qualifying samples)")
        if error > 0.02:
            failures.append(f"steady-state gain is not 1 (|xi - u| = {error:.4f})")
    else:
        failures.append(
            f"no command stayed constant for {window} steps -- the steady-state "
            "gain was not tested"
        )
    return failures


# ---------------------------------------------------------------------------
# R3
# ---------------------------------------------------------------------------

# R3's wording is "dominated by the first one or two harmonics".  Taken
# literally that is wrong for half the channels, and for a physical reason: in
# trot both diagonal pairs push once per gait cycle, so forward velocity and
# body height oscillate at TWICE the gait frequency and their energy sits in the
# even harmonics (h2, h4).  Pitch and yaw rate, which alternate once per cycle,
# really are h1-dominated.  The property the criterion is reaching for is "a
# clean low-order periodic waveform, not noise", so the gate is concentration in
# the first four harmonics; the h1+h2 share is still reported for reference.
HARMONIC_GATE = 0.85
HARMONIC_COUNT = 4


def check_r3(env, cfg, policy_path, seconds=60.0, held_speed=0.5):
    base = env.env
    policy = load_dog_policy(policy_path, cfg, base.device)

    held = {0: held_speed, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0,
            6: 3.0, 7: 0.06, 8: 0.30, 9: 0.44, 10: 0.5}

    def hold():
        for index, value in held.items():
            base.commands_dog[:, index] = value

    env.reset()
    hold()
    _, arm_a = zero_actions(env, cfg)

    channels = len(base.response_ref.channel_names)
    raw_error = torch.zeros(channels)
    detrended_error = torch.zeros(channels)
    samples = 0
    with torch.no_grad():
        for _ in range(int(seconds / base.dt)):
            observations = env.get_dog_observations()
            actions = policy.act_inference({"obs_history": observations["obs_history"]})
            env.step(actions, arm_a)
            hold()
            active = base._residual_sample_active()
            if active.any():
                raw_error += (base.response_measured - base.response_ref.xi).abs()[active].mean(0).cpu()
                detrended_error += (base.response_detrended - base.response_ref.xi).abs()[active].mean(0).cpu()
                samples += 1
    raw_error /= max(samples, 1)
    detrended_error /= max(samples, 1)

    estimator = base.response_residual
    speed_bin = estimator.speed_bin(torch.norm(base.commands_dog[:, :2], dim=-1))[0].item()
    visits = estimator.sample_count[:, speed_bin].median().item()
    print(f"  held vx={held_speed} for {seconds:.0f}s | speed bin {speed_bin} | "
          f"median visits per phase bin {visits:.0f}")

    failures = []
    print(f"  {'channel':<10}{'pk-pk':>9}{'h1+h2':>9}{f'h1..h{HARMONIC_COUNT}':>9}"
          f"{'raw MAE':>10}{'detrended':>11}{'change':>9}")
    for index, name in enumerate(base.response_ref.channel_names):
        curve = estimator.delta_hat[:, speed_bin, :, index].mean(0).double()
        power = (torch.fft.rfft(curve).abs() ** 2)[1:]
        total = power.sum()
        low_two = (power[:2].sum() / total).item() if total > 0 else 0.0
        low_n = (power[:HARMONIC_COUNT].sum() / total).item() if total > 0 else 0.0
        peak_to_peak = (curve.max() - curve.min()).item()
        change = (detrended_error[index] - raw_error[index]) / raw_error[index] * 100
        print(f"  {name:<10}{peak_to_peak:>9.4f}{low_two:>8.0%}{low_n:>9.0%}"
              f"{raw_error[index]:>10.4f}{detrended_error[index]:>11.4f}{change:>8.1f}%")
        harmonics = "  ".join(f"h{k + 1}={v:.0%}" for k, v in enumerate((power / total).tolist()[:6]))
        print(f"  {'':<10}{harmonics}")
        if peak_to_peak > 1e-4 and low_n < HARMONIC_GATE:
            failures.append(
                f"{name}: harmonics 1-{HARMONIC_COUNT} carry only {low_n:.1%} of the AC "
                "power -- the residual looks like noise, not a gait waveform"
            )
        if visits < estimator.min_samples:
            failures.append(
                f"phase bins saw {visits:.0f} visits, below the convergence gate "
                f"({estimator.min_samples:.0f})"
            )
            break
    return failures


# ---------------------------------------------------------------------------
# R4
# ---------------------------------------------------------------------------


def check_r4(env, cfg, policy_path, seconds=40.0):
    """Report the raw magnitude of each R4 term and how the gate behaves.

    Weights for the two penalties cannot be guessed: the total reward is
    ``positive * exp(negative / sigma_rew_neg)``, and because the reward scales
    are multiplied by dt (0.02) while sigma_rew_neg is also 0.02, a penalty's
    configured weight IS its coefficient in the exponent.  So the weight that
    produces a given attenuation is ``-ln(attenuation) / mean_term_value``, and
    that needs the measured term value.
    """
    base = env.env
    policy = load_dog_policy(policy_path, cfg, base.device)
    container = base.reward_container

    env.reset()
    _, arm_a = zero_actions(env, cfg)
    totals = {
        "ref_tracking": 0.0,
        "phase_variance": 0.0,
        "steady_gain": 0.0,
        "domain_consistency": 0.0,
    }
    gate_shut = 0.0
    desync = 0.0
    slip_samples, accel_samples = [], []
    steps = int(seconds / base.dt)
    reward_cfg = cfg.response.reward
    with torch.no_grad():
        for _ in range(steps):
            observations = env.get_dog_observations()
            actions = policy.act_inference({"obs_history": observations["obs_history"]})
            previous = base.prev_base_lin_vel.clone()
            env.step(actions, arm_a)
            totals["ref_tracking"] += container._reward_ref_tracking().mean().item()
            totals["phase_variance"] += container._reward_phase_variance().mean().item()
            totals["steady_gain"] += container._reward_steady_gain().mean().item()
            # R5's term is averaged over the envs it is ACTIVE for, not over all
            # of them: it is masked off for twins, for ungrouped envs and for
            # desynchronised groups, and a plain mean over everything would be
            # diluted by roughly a factor of three by construction -- which would
            # then be baked into the derived weight.
            consistency = container._reward_domain_consistency()
            active = base.grouping.valid > 0
            if bool(active.any()):
                totals["domain_consistency"] += consistency[active].mean().item()
            scoreable = base.grouping.is_grouped & ~base.grouping.is_twin
            if bool(scoreable.any()):
                desync += float(1.0 - base.grouping.valid[scoreable].mean())
            gate_shut += (base.response_soft_gate == 0).float().mean().item()
            # Recompute the two trigger quantities so their distributions can be
            # inspected; the gate itself only exposes the combined result.
            contact = (base.contact_forces[:, base.feet_indices, 2] > 1.0).float()
            slip = torch.norm(base.foot_velocities[:, :, :2], dim=-1)
            slip_samples.append(
                ((slip * contact).sum(-1) / torch.clamp(contact.sum(-1), min=1.0)).cpu()
            )
            accel_samples.append(
                torch.norm((base.base_lin_vel[:, :2] - previous[:, :2]) / base.dt, dim=-1).cpu()
            )
    for key in totals:
        totals[key] /= steps

    slip = torch.cat(slip_samples)
    accel = torch.cat(accel_samples)
    print(f"  {seconds:.0f}s rollout, {cfg.env.num_envs} envs")
    print(f"  soft gate shut for {gate_shut / steps:.1%} of samples "
          f"(hold {base.soft_gate_hold_steps} steps)")
    print(f"  R5 group desynchronised for {desync / steps:.1%} of samples")
    print(f"\n  {'trigger':<22}{'p50':>9}{'p90':>9}{'p99':>9}{'p99.9':>9}"
          f"{'threshold':>11}{'fires':>8}")
    for name, values, threshold in (
        ("mean contact slip m/s", slip, float(reward_cfg.soft_gate_slip_speed)),
        ("horiz accel m/s^2", accel, float(reward_cfg.soft_gate_accel)),
    ):
        quantiles = torch.quantile(values, torch.tensor([0.5, 0.9, 0.99, 0.999]))
        rate = (values > threshold).float().mean().item()
        print(f"  {name:<22}{quantiles[0]:>9.3f}{quantiles[1]:>9.3f}{quantiles[2]:>9.3f}"
              f"{quantiles[3]:>9.3f}{threshold:>11.2f}{rate:>7.2%}")
    print(f"\n  {'term':<18}{'mean raw':>12}{'sign':>7}   weight for 5% / 10% / 20% attenuation")
    for name, value in totals.items():
        if name == "ref_tracking":
            print(f"  {name:<18}{value:>12.4f}{'+':>7}   (task term, positive)")
            continue
        weights = [f"{-math.log(1 - a) / max(value, 1e-12):7.1f}" for a in (0.05, 0.10, 0.20)]
        print(f"  {name:<18}{value:>12.6f}{'-':>7}   " + " ".join(weights))

    failures = []
    if totals["ref_tracking"] <= 0.0:
        failures.append("reference-tracking reward is identically zero")
    gate_rate = gate_shut / steps
    if gate_rate > 0.15:
        failures.append(
            f"soft gate is shut {gate_rate:.1%} of the time -- thresholds are too "
            "sensitive; remember each trigger holds it for 25 steps, so a trigger "
            "firing on x% of steps shuts the gate for ~25x% of the time"
        )
    return failures


# ---------------------------------------------------------------------------
# R5
# ---------------------------------------------------------------------------


def check_r5(env, cfg, steps=1200):
    """R5's three acceptance criteria, plus the twin's nominal domain.

    The termination-asymmetry criterion is checked by *causing* a fall rather
    than waiting for one: an env is teleported below the termination height, and
    the group's validity mask has to drop to zero and stay there until the next
    shared resample.  Waiting for a natural fall makes the check depend on how
    bad the policy is, which is exactly the kind of flakiness an acceptance gate
    should not have.
    """
    base = env.env
    grouping = base.grouping
    if grouping.num_groups == 0:
        return ["grouping is disabled or there are too few envs to form a group"]

    env.reset()
    dog_a, arm_a = zero_actions(env, cfg)
    failures = []
    size = grouping.group_size
    print(f"  {cfg.env.num_envs} envs -> {grouping.num_groups} groups of {size}; "
          f"{int(grouping.is_grouped.sum())} grouped, twins at "
          f"{grouping.is_twin.nonzero().flatten()[:4].tolist()}...")

    # -- 1. the twin's domain parameters are nominal ------------------------
    twins = grouping.is_twin
    others = grouping.is_grouped & ~twins
    nominal = [
        ("friction", base.friction_coeffs, base.default_friction),
        ("restitution", base.restitutions, base.default_restitution),
        ("payload", base.payloads, 0.0),
        ("com_displacement", base.com_displacements, 0.0),
        ("motor_strength", base.motor_strengths, 1.0),
        ("motor_offset", base.motor_offsets, 0.0),
        ("Kp_factor", base.Kp_factors, 1.0),
        ("Kd_factor", base.Kd_factors, 1.0),
    ]
    if hasattr(base, "stage1_ee_payload_mass"):
        nominal.append(("ee_payload", base.stage1_ee_payload_mass, 0.0))
    if hasattr(base, "arm_link_mass_scales"):
        nominal.append(("arm_link_mass_scale", base.arm_link_mass_scales, 1.0))
        nominal.append(("arm_link_com_offset", base.arm_link_com_offsets, 0.0))

    print(f"  {'parameter':<22}{'twin span':>26}{'others span':>26}")
    for name, buffer, want in nominal:
        twin_values = buffer[twins]
        other_values = buffer[others]
        twin_span = (float(twin_values.min()), float(twin_values.max()))
        other_span = (float(other_values.min()), float(other_values.max()))
        print(f"  {name:<22}[{twin_span[0]:>10.5f},{twin_span[1]:>10.5f}]  "
              f"[{other_span[0]:>10.5f},{other_span[1]:>10.5f}]")
        if abs(twin_span[0] - want) > 1e-5 or abs(twin_span[1] - want) > 1e-5:
            failures.append(
                f"twin {name} is {twin_span}, expected exactly {want} -- a "
                "randomisation site is not covered by _nominalize_twins()"
            )
        # A parameter that is not randomised at all makes its row vacuous; say
        # so rather than passing silently on a comparison that proves nothing.
        if abs(other_span[1] - other_span[0]) < 1e-9 and abs(other_span[0] - want) < 1e-5:
            print(f"      note: {name} is not randomised, so this row proves nothing")

    twin_bucket = base.arm_mount_bucket_of_env[twins] if hasattr(
        base, "arm_mount_bucket_of_env") else None
    if twin_bucket is not None and int(twin_bucket.max()) != 0:
        failures.append(f"twin mount TF bucket is not 0: max {int(twin_bucket.max())}")

    # -- 2. shared command vector and gait phase ----------------------------
    command_mismatch = 0
    phase_mismatch = 0
    synced_samples = 0
    for _ in range(steps):
        env.step(dog_a, arm_a)
        healthy = grouping.valid > 0
        if not healthy.any():
            continue
        synced_samples += 1
        twin_row = base.commands_dog[grouping.twin_of]
        command_mismatch += int(
            ((base.commands_dog != twin_row).any(dim=-1) & healthy).sum()
        )
        twin_phase = base.gait_indices[grouping.twin_of]
        phase_mismatch += int(
            (((base.gait_indices - twin_phase).abs() > 1e-6) & healthy).sum()
        )
    print(f"  over {synced_samples} steps with a synchronised group: "
          f"{command_mismatch} command mismatches, {phase_mismatch} phase mismatches")
    if synced_samples == 0:
        failures.append("no group was ever synchronised -- the check is vacuous")
    if command_mismatch:
        failures.append(
            f"{command_mismatch} env-steps where a synchronised group's command "
            "differed from its twin's"
        )
    if phase_mismatch:
        failures.append(
            f"{phase_mismatch} env-steps where a synchronised group's gait phase "
            "differed from its twin's"
        )

    # -- 3. a fall switches the group off until the next shared resample ----
    victim = int((grouping.is_grouped & ~grouping.is_twin).nonzero().flatten()[0])
    group = int(grouping.group_of[victim])
    # Wait for the group to be freshly synchronised so the observation is clean.
    interval = int(cfg.commands.resampling_time / base.dt)
    while bool(grouping.group_desync[group]):
        env.step(dog_a, arm_a)
    before = float(grouping.valid[victim])

    base.root_states[victim, 2] = -5.0
    base.gym.set_actor_root_state_tensor(base.sim, gymtorch.unwrap_tensor(base.root_states))
    env.step(dog_a, arm_a)
    after = float(grouping.valid[victim])
    print(f"  forced a fall in env {victim} (group {group}): "
          f"valid {before:.0f} -> {after:.0f}")
    if not (before == 1.0 and after == 0.0):
        failures.append(
            f"forcing env {victim} to fall did not invalidate group {group} "
            f"(valid {before} -> {after})"
        )

    recovered_at = None
    for step in range(1, interval + 2):
        env.step(dog_a, arm_a)
        if grouping.valid[victim] > 0:
            recovered_at = step
            break
    print(f"  group {group} recovered after {recovered_at} steps "
          f"(shared resample interval is {interval})")
    if recovered_at is None:
        failures.append(f"group {group} never recovered within one resample interval")
    return failures


# ---------------------------------------------------------------------------
# R6
# ---------------------------------------------------------------------------


def check_r6(env, cfg, steps=1100):
    """Excitation reaches only the identification envs, and the curriculum never
    sees them.

    Everything here is measured **within one excitation plan**.  A plan is
    redrawn on every reset, so a rollout longer than an episode contains several
    of them, and comparing a whole trace against the plan that happens to be
    current at the end mixes channels: the previous plan's excited channel then
    reads as a passive channel that moves 1000 times.  (That is not a
    hypothetical -- it is what the first version of this check reported.)

    The curriculum exclusion is the invariant worth spending a spy on: it fails
    *silently*, and the symptom -- the command range slowly ratcheting shut for
    every environment, identification or not -- looks nothing like its cause.
    """
    base = env.env
    sampler = base.response_excitation
    if not sampler.active:
        return ["no identification environments -- run with more --num_envs"]

    curriculum = base.curricula[0]
    original_update = curriculum.update
    original_resample = base._resample_commands
    seen = {"resampled": [], "updated": []}

    def spy_resample(env_ids):
        seen["resampled"].append(env_ids.clone())
        return original_resample(env_ids)

    def spy_update(old_bins, *rest, **kwargs):
        seen["updated"].append(len(old_bins))
        return original_update(old_bins, *rest, **kwargs)

    base._resample_commands = spy_resample
    curriculum.update = spy_update
    try:
        env.reset()
        seen["resampled"].clear()
        seen["updated"].clear()
        dog_a, arm_a = zero_actions(env, cfg)
        history, signal_history, channel_history, episode_history = [], [], [], []
        generation_history = []
        for _ in range(steps):
            env.step(dog_a, arm_a)
            # Read after the step: this is the plan that produced these commands.
            history.append(base.commands_dog.clone())
            signal_history.append(sampler.signal.clone())
            channel_history.append(sampler.channel.clone())
            generation_history.append(sampler.plan_generation.clone())
            episode_history.append(base.episode_length_buf.clone())
    finally:
        base._resample_commands = original_resample
        curriculum.update = original_update

    commands = torch.stack(history)                  # (T, E, C)
    signals = torch.stack(signal_history)            # (T, E)
    channels = torch.stack(channel_history)          # (T, E)
    episodes = torch.stack(episode_history)          # (T, E)
    generations = torch.stack(generation_history)    # (T, E)
    is_identification = sampler.is_identification
    failures = []

    n_id = int(is_identification.sum())
    print(f"  {steps} steps x {cfg.env.num_envs} envs; "
          f"{n_id} identification ({n_id / cfg.env.num_envs:.1%}), "
          f"train pool {base.num_train_envs}")
    print("  signals: " + ", ".join(f"{k}={v}" for k, v in sampler.signal_counts().items())
          + "  |  channels: "
          + ", ".join(f"{k}={v}" for k, v in sampler.channel_counts().items()))

    # Steps k -> k+1 that stayed inside a single plan and a single episode.
    # Plan identity is the generation counter, not (signal, channel): a redraw
    # hits the same pair about one time in fifteen, and the baseline command
    # step that accompanies it then reads as a passive channel moving inside a
    # plan.  That false positive is what this counter exists to remove.
    steady = generations[1:] == generations[:-1]
    steady &= episodes[1:] > episodes[:-1]
    changed = commands[1:] != commands[:-1]                    # (T-1, E, C)
    decision = torch.tensor(list(DECISION_CMD_INDEX), device=commands.device)
    excited = sampler.cmd_index[channels]                      # (T, E)

    # -- 1. consistency envs are untouched ----------------------------------
    resample_bound = steps // 500 + 4
    passive_envs = ~is_identification
    passive_changes = changed[:, passive_envs][:, :, decision].sum(dim=0).max().item()
    print(f"  max decision-channel changes over the whole rollout: "
          f"consistency env {passive_changes:.0f} (bound {resample_bound})")
    if passive_changes > resample_bound:
        failures.append(
            f"a consistency env changed a decision channel {passive_changes:.0f}x, "
            f"more than the {resample_bound} resample/reset events"
        )

    # -- 2. only the excited channel moves, inside a plan --------------------
    is_excited = torch.zeros_like(changed, dtype=torch.bool)
    is_excited.scatter_(2, excited[:-1].unsqueeze(-1), True)
    passive_moved = (
        changed & ~is_excited & steady.unsqueeze(-1) & is_identification.view(1, -1, 1)
    )[:, :, decision]
    worst = int(passive_moved.sum(dim=0).max())
    print(f"  passive decision-channel moves inside a plan: {worst} (must be 0)")
    if worst > 0:
        env_index = int(passive_moved.sum(dim=(0, 2)).argmax())
        failures.append(
            f"identification env {env_index}: a passive decision channel moved "
            f"{worst}x inside a single plan -- the mid-episode resample is not "
            "suppressed, or excitation is writing more than one channel"
        )

    # -- 3. jump count, per 20 s episode, measured inside a plan -------------
    excited_moved = changed & is_excited & steady.unsqueeze(-1)
    excited_moved = excited_moved.any(dim=-1)                  # (T-1, E)
    episode_steps = float(cfg.env.episode_length_s) / base.dt
    rates = {}
    for index, name in enumerate(SIGNAL_NAMES):
        member = (signals[:-1] == index) & steady & is_identification.view(1, -1)
        samples = int(member.sum())
        if samples == 0:
            continue
        rate = float(excited_moved[member].float().mean()) * episode_steps
        rates[name] = rate
        print(f"  {name:<6} command jumps per {cfg.env.episode_length_s:.0f} s episode: "
              f"{rate:6.1f}   ({samples} in-plan samples)")
        if name != "prbs" and rate < 20.0:
            failures.append(f"{name} envs jump only {rate:.1f}x per episode, need >= 20")
    group_member = steady & is_identification.view(1, -1)
    group_rate = float(excited_moved[group_member].float().mean()) * episode_steps
    print(f"  identification group mean: {group_rate:.1f} jumps per episode")
    if group_rate < 20.0:
        failures.append(f"identification group jumps {group_rate:.1f}x per episode, need >= 20")
    if "prbs" in rates and rates["prbs"] >= 20.0:
        print("  note: PRBS now clears 20 jumps/episode -- the documented "
              "U(0.5, 3.0) s shortfall no longer applies, check the config")

    # -- 4. chirp band coverage, over one uninterrupted sweep ----------------
    segment = _longest_plan_segment(generations, signals, episodes, is_identification, CHIRP)
    if segment is None:
        failures.append("no uninterrupted chirp segment was produced")
    else:
        env_index, lo, hi = segment
        column = int(excited[lo, env_index])
        signal = commands[lo:hi, env_index, column].double()
        signal = signal - signal.mean()
        spectrum = torch.fft.rfft(signal).abs()
        freqs = torch.fft.rfftfreq(signal.numel(), d=base.dt)
        in_band = (freqs >= 0.1) & (freqs <= 2.0)
        share = float(spectrum[in_band].sum() / spectrum.sum())
        duration = (hi - lo) * base.dt
        gaps = [
            f"{lo_hz}-{2 * lo_hz}Hz"
            for lo_hz in (0.1, 0.25, 0.5, 1.0)
            if float(spectrum[(freqs >= lo_hz) & (freqs < 2 * lo_hz)].max())
            <= 0.02 * float(spectrum.max())
        ]
        print(f"  chirp env {env_index} ({DOG_COMMAND_NAMES[column]}), "
              f"{duration:.1f}s uninterrupted: "
              f"{share:.1%} of spectral energy inside 0.1-2.0 Hz")
        # A sweep shorter than the configured duration only reaches part of the
        # band, so scale what is demanded of it rather than failing on it.
        expected = min(1.0, duration / float(cfg.response.excitation.chirp_duration_s))
        if share < 0.85 * expected:
            failures.append(
                f"chirp energy inside 0.1-2.0 Hz is only {share:.1%} over a "
                f"{duration:.1f}s sweep"
            )
        if gaps and expected > 0.9:
            failures.append(f"chirp spectrum has empty octaves: {', '.join(gaps)}")

    # -- 5. curriculum never scored an identification env --------------------
    total_resampled = sum(int(ids.numel()) for ids in seen["resampled"])
    identification_resampled = sum(
        int(is_identification[ids].sum()) for ids in seen["resampled"]
    )
    scored = sum(seen["updated"])
    expected_scored = total_resampled - identification_resampled
    print(f"  curriculum: {scored} envs scored out of {total_resampled} resampled "
          f"({identification_resampled} of them identification)")
    if identification_resampled == 0:
        failures.append("no identification env was ever resampled -- the check is vacuous")
    if scored != expected_scored:
        failures.append(
            f"curriculum.update() scored {scored} envs, expected {expected_scored} "
            "(identification envs must be filtered out)"
        )
    return failures


def _longest_plan_segment(generations, signals, episodes, is_identification, wanted_signal):
    """Longest ``(env, start, stop)`` run inside one plan and one episode."""
    best = None
    for env_index in is_identification.nonzero().flatten().tolist():
        generation = generations[:, env_index]
        signal = signals[:, env_index]
        episode = episodes[:, env_index]
        start = 0
        for k in range(1, signal.numel() + 1):
            broken = k == signal.numel() or not (
                generation[k] == generation[start]
                and episode[k] > episode[k - 1]
            )
            if not broken:
                continue
            if signal[start] == wanted_signal and (best is None or k - start > best[2] - best[1]):
                best = (env_index, start, k)
            start = k
    return best


# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", default="all",
                        choices=["r1", "r2", "r3", "r4", "r5", "r6", "all"])
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--robot", type=str, default="go2_x5")
    parser.add_argument("--policy", type=str, default=None,
                        help="stage-1 dog checkpoint; required for the R3 check")
    parser.add_argument("--seconds", type=float, default=60.0, help="R3 rollout length")
    args = parser.parse_args()

    env, cfg = build_env(args.num_envs, args.sim_device, args.robot)
    wanted = ["r1", "r2", "r3", "r4", "r5", "r6"] if args.check == "all" else [args.check]

    results = {}
    for name in wanted:
        print(f"\n=== {name.upper()} ===")
        if name == "r1":
            results[name] = check_r1(env, cfg)
        elif name == "r2":
            results[name] = check_r2(env, cfg)
        elif name == "r3":
            if not args.policy:
                print("  skipped: needs a walking policy, pass --policy <ckpt.pt>")
                continue
            if not os.path.exists(args.policy):
                raise SystemExit(f"checkpoint not found: {args.policy}")
            results[name] = check_r3(env, cfg, args.policy, seconds=args.seconds)
        elif name == "r4":
            if not args.policy:
                print("  skipped: needs a walking policy, pass --policy <ckpt.pt>")
                continue
            results[name] = check_r4(env, cfg, args.policy)
        elif name == "r5":
            results[name] = check_r5(env, cfg)
        elif name == "r6":
            results[name] = check_r6(env, cfg)

    print("\n=== summary ===")
    failed = False
    for name, failures in results.items():
        if failures:
            failed = True
            print(f"  {name.upper()} FAILED:")
            for failure in failures:
                print(f"    - {failure}")
        else:
            print(f"  {name.upper()} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
