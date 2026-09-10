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
CONV the pitch/roll sign convention agrees across the observation, the
    reference model and the attitude reward.
R8  the stage curriculum really reaches the reward: terms are registered from
    iteration 0, contribute exactly nothing in stage 1, and ramp to their
    configured target; the randomisation and disturbance intensities reach the
    samplers; and R8.2's gait-frequency ripple alarm is both fed and able to see.
R6  the identification environments really are excited (jump count, chirp band),
    really are excluded from the curriculum, and really are the only ones
    touched.
DR  base mass / CoM randomisation and the EE payload's own CoM offset reach the
    simulator rather than only the tensors, and the sensing latency delays the
    measured half of the observation while leaving commands untouched.
TERRAIN  the mild-rough tiles carry the amplitudes they were configured with,
    the nominal twins stand on the flat tier at every curriculum intensity, and
    body height is measured against the local ground.
"""

import argparse
import math
import os
import sys

import isaacgym  # noqa: F401  must precede torch
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_from_angle_axis, quat_mul, quat_rotate_inverse
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
    #
    # The twin-nominality table above is also re-checked here, on every step,
    # for a reason worth stating: it used to be sampled once, before anything
    # had been stepped.  _post_physics_step_callback re-draws motor strength,
    # motor offset and the Kp/Kd factors MID-EPISODE every rand_interval steps,
    # and that site had no _nominalize_twins() call -- so the twins started at
    # exactly nominal, passed the table, and drifted off it a few seconds in.
    # A gate that only visits the parameters at t=0 is not a gate.
    drifted = set()
    command_mismatch = 0
    phase_mismatch = 0
    synced_samples = 0
    for _ in range(steps):
        env.step(dog_a, arm_a)
        for name, buffer, want in nominal:
            values = buffer[twins]
            if values.numel() and float((values - want).abs().max()) > 1e-5:
                drifted.add(name)
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
    print(f"  twin parameters stayed nominal over {steps} stepped steps: "
          f"{'no -- ' + ', '.join(sorted(drifted)) if drifted else 'yes'}")
    if drifted:
        failures.append(
            "twin " + ", ".join(sorted(drifted)) + " left nominal DURING the "
            "rollout -- a mid-episode randomisation site is not followed by "
            "_nominalize_twins()"
        )
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

    # -- 3. a fall switches the group off for exactly the settling window ---
    victim = int((grouping.is_grouped & ~grouping.is_twin).nonzero().flatten()[0])
    group = int(grouping.group_of[victim])
    # Wait for the group to be freshly synchronised so the observation is clean.
    settle = grouping.settle_steps
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

    # The window is a fixed countdown, so this is an equality check, not a
    # bound.  Recovering early would mean a resample cut the window short (the
    # bug the countdown replaced); recovering late would mean advance() is not
    # running every step.  A member other than the victim resetting during the
    # wait would restart the countdown and make the observation meaningless, so
    # allow one extra step of slack and no more.
    recovered_at = None
    for step in range(1, settle + 3):
        env.step(dog_a, arm_a)
        if grouping.valid[victim] > 0:
            recovered_at = step
            break
    print(f"  group {group} recovered after {recovered_at} steps "
          f"(settling window is {settle})")
    if recovered_at is None:
        failures.append(
            f"group {group} never recovered within its {settle}-step settling window"
        )
    elif recovered_at < settle:
        failures.append(
            f"group {group} recovered after {recovered_at} steps, before its "
            f"{settle}-step settling window elapsed -- something is clearing "
            "the countdown early"
        )

    failures += _check_twin_arm_is_held_still(env, cfg)
    return failures


def _check_twin_arm_is_held_still(env, cfg):
    """R5: "arm fixed" is part of the twin's definition, and it has to hold
    while the stage-1 arm curriculum is actually moving everyone else's arm.

    This needs forcing.  The curriculum's intensity is zero until 10% of its
    ramp has elapsed, so every check and every smoke run so far exercised only
    the intensity == 0 branch -- and the twin branch below it, which is the one
    R5 added, had never executed anywhere.  It crashed the first time it ran, at
    iteration 600 of a real training run.  Forcing the intensity is the whole
    point of this check: an acceptance gate that only visits the code paths a
    short rollout happens to reach is not a gate.
    """
    base = env.env
    if not base._stage1_arm_curriculum_active():
        print("  stage-1 arm curriculum is off; skipped the twin-arm check")
        return []

    failures = []
    dog_a, arm_a = zero_actions(env, cfg)
    previous = getattr(base, "stage1_arm_play_intensity", None)
    arm_slice = slice(base.num_actions_loco, base.num_actions_loco + base.num_actions_arm)
    twins = base.is_nominal_twin
    others = base.grouping.is_grouped & ~twins
    try:
        base.stage1_arm_play_intensity = 1.0
        for _ in range(20):
            env.step(dog_a, arm_a)
        twin_motion = float(base.stage1_arm_target_offset[twins].abs().max())
        other_motion = float(base.stage1_arm_target_offset[others].abs().max())
    finally:
        if previous is None:
            if hasattr(base, "stage1_arm_play_intensity"):
                del base.stage1_arm_play_intensity
        else:
            base.stage1_arm_play_intensity = previous

    print(f"  arm curriculum forced to full: twin |offset| max {twin_motion:.6f}, "
          f"non-twin {other_motion:.4f}")
    if twin_motion != 0.0:
        failures.append(
            f"the twin's arm moved ({twin_motion:.6f} rad) while the curriculum "
            "was running -- the twin is supposed to be the no-disturbance reference"
        )
    if other_motion == 0.0:
        failures.append(
            "no non-twin arm moved at full curriculum intensity, so the check "
            "proved nothing about the twin"
        )
    return failures


# ---------------------------------------------------------------------------
# Sign conventions
# ---------------------------------------------------------------------------


def check_conv(env, cfg, amplitude=0.3):
    """One sign convention for body attitude, checked in all three places.

    This exists because they silently disagreed.  ``_reward_orientation_control``
    negated the command, so it drove pitch to **-command**, while the
    observation handed the policy ``pose_target - pose_measured`` with
    ``pose_target = +command`` and R2's reference model read ``+command`` too.
    A trained stage-1 policy measured a DC gain of -0.28: inverted and weak.

    It would have got worse rather than better -- once R4.1 ramps in at stage 2
    it drives pitch to +command at weight 2.0 against this term's -command, so
    the two would have fought each other with the policy in between.

    The convention, decided 2026-09-02: **a command of +x means the body reaches
    +x radians in the standard rpy sense**, everywhere.
    """
    base = env.env
    failures = []
    env.reset()
    dog_a, arm_a = zero_actions(env, cfg)
    env.step(dog_a, arm_a)

    print("  convention: a command of +x means the body reaches +x rad\n")
    print(f"  {'channel':<8}{'command':>9}{'reward wants':>14}{'reference xi':>14}"
          f"{'observed target':>17}")
    # projected_gravity[0] = +sin(pitch); projected_gravity[1] = -sin(roll)
    for name, column, axis in (("pitch", 3, 0), ("roll", 4, 1)):
        for sign in (-1.0, 1.0):
            command = sign * amplitude
            base.commands_dog[:, 3] = 0.0
            base.commands_dog[:, 4] = 0.0
            base.commands_dog[:, column] = command

            # 1. what the attitude reward is minimised at
            pitch_command = base.commands_dog[:, 3]
            roll_command = base.commands_dog[:, 4]
            quat_roll = quat_from_angle_axis(
                roll_command, torch.tensor([1.0, 0.0, 0.0], device=base.device))
            quat_pitch = quat_from_angle_axis(
                pitch_command, torch.tensor([0.0, 1.0, 0.0], device=base.device))
            desired = quat_rotate_inverse(
                quat_mul(quat_roll, quat_pitch), base.gravity_vec)
            wanted = float(torch.asin(desired[:, axis].clamp(-1, 1).mean()))
            if name == "roll":
                wanted = -wanted     # projected_gravity[1] = -sin(roll)

            # 2. what the reference model integrates towards (pitch only)
            xi_target = (float(base.response_ref.gather_commands(base.commands_dog)[:, 4].mean())
                         if name == "pitch" else float("nan"))

            # 3. what the observation presents as the target
            observed = float(
                base.commands_dog[:, column].mean() * (
                    base.obs_scales.body_pitch_cmd if name == "pitch"
                    else base.obs_scales.body_roll_cmd
                )
            )
            print(f"  {name:<8}{command:>9.3f}{wanted:>14.4f}{xi_target:>14.4f}"
                  f"{observed:>17.4f}")

            if abs(wanted - command) > 0.02:
                failures.append(
                    f"the attitude reward drives {name} to {wanted:+.3f} for a "
                    f"command of {command:+.3f} -- signs disagree"
                )
            if name == "pitch" and abs(xi_target - command) > 1e-6:
                failures.append(
                    f"the reference model integrates pitch towards {xi_target:+.3f} "
                    f"for a command of {command:+.3f}"
                )
            if observed * command < 0:
                failures.append(
                    f"the observation presents {name} target {observed:+.3f} for a "
                    f"command of {command:+.3f} -- opposite sign"
                )

    registered = set(base.reward_names)
    if "pitch_control" in registered and "orientation_control" in registered:
        print("\n  attitude reward is split: orientation_control = roll, "
              "pitch_control = pitch")
    else:
        failures.append(
            "expected both orientation_control (roll) and pitch_control (pitch) "
            f"to be registered, found {sorted(registered & {'orientation_control', 'pitch_control'})}"
        )
    return failures


# ---------------------------------------------------------------------------
# R8
# ---------------------------------------------------------------------------


def check_r8(env, cfg, steps=120):
    """The curriculum has to reach the reward, not merely exist.

    The specific failure this guards against is structural rather than
    numerical: ``_prepare_reward_function`` drops zero-scale terms *before*
    registering them, so a consistency term shipped at weight 0 -- which is how
    they shipped before R8 -- would have no function to call and could never be
    ramped up.  Expressing "off" as a multiplier is what fixes it, and this
    check confirms the fix end to end by driving the iteration counter through
    every stage and reading the realised reward.
    """
    base = env.env
    curriculum = base.response_curriculum
    failures = []
    dog_a, arm_a = zero_actions(env, cfg)
    original_count = int(getattr(global_switch, "count", 0))

    terms = sorted(curriculum.term_stage)
    registered = set(base.reward_names)
    missing = [name for name in terms if name not in registered]
    print(f"  stages {curriculum.stage_boundaries}, ramp "
          f"{curriculum.ramp_iterations} iterations")
    print(f"  registered reward terms: "
          + ", ".join(f"{n}={'yes' if n in registered else 'NO'}" for n in terms))
    if missing:
        failures.append(
            f"{missing} are not registered reward functions -- a term that is "
            "not registered can never be ramped up, whatever its multiplier"
        )
        return failures

    # Sample the schedule at the start of each stage, mid-ramp, and after it.
    probes = [0]
    for boundary in curriculum.stage_boundaries:
        probes += [boundary, boundary + curriculum.ramp_iterations // 2,
                   boundary + curriculum.ramp_iterations]
    probes = sorted(set(probes))

    print(f"\n  {'iter':>8}{'stage':>7}" + "".join(f"{n[:12]:>14}" for n in terms))
    realised = {}
    try:
        for iteration in probes:
            global_switch.count = iteration
            env.reset()
            totals = {name: 0.0 for name in terms}
            for _ in range(steps):
                env.step(dog_a, arm_a)
                scales = base.response_curriculum.apply(
                    global_switch.get_reward_scales(), iteration
                )
                for name in terms:
                    index = base.reward_names.index(name)
                    value = base.reward_functions[index]() * scales[name]
                    totals[name] += float(value.abs().mean())
            realised[iteration] = {k: v / steps for k, v in totals.items()}
            row = "".join(f"{realised[iteration][n]:>14.6f}" for n in terms)
            print(f"  {iteration:>8}{curriculum.stage(iteration):>7}{row}")
    finally:
        global_switch.count = original_count

    # Stage 1: every scheduled term contributes exactly zero.
    for name, value in realised[0].items():
        if value != 0.0:
            failures.append(
                f"{name} contributed {value:.6g} in stage 1; stage 1 is supposed "
                "to run the original rewards only"
            )

    # Each term must be zero at the very start of its ramp and non-zero once the
    # ramp has finished -- but only if the term is non-zero AT ALL under this
    # rollout.  These checks run on zero actions, so the robot stands still and
    # every domain behaves nearly alike; R5's cross-domain term is then
    # genuinely ~0 for reasons that have nothing to do with the schedule.
    # Failing on that would be testing the robot, not the curriculum, so the
    # inactive case is reported rather than failed.
    for name, stage in curriculum.term_stage.items():
        start = curriculum.stage_start(stage)
        done = start + curriculum.ramp_iterations
        if start not in realised:
            continue
        if realised[start][name] != 0.0:
            failures.append(f"{name} is non-zero at the very start of its ramp ({start})")
        # Reachability, not a per-iteration value.  Whether this term is
        # non-zero at one particular iteration is a property of what the robot
        # did in that rollout -- R5's cross-domain term is intermittently
        # exactly zero because it is masked whenever its group is
        # desynchronised.  What the schedule is responsible for is that the term
        # can reach the reward at all once its ramp is done.
        reachable = max(
            (realised[i][name] for i in realised if i >= done), default=0.0
        )
        if reachable == 0.0:
            failures.append(
                f"{name} never became non-zero at or after the end of its ramp "
                f"({done}) -- its multiplier is not reaching the reward"
            )

    # The multiplier itself, independent of what the robot happened to do.
    weights = curriculum.multiplier(curriculum.stage_start(3) + curriculum.ramp_iterations)
    for name, stage in curriculum.term_stage.items():
        if stage <= 3 and weights[name] != 1.0:
            failures.append(f"{name} multiplier is {weights[name]}, expected 1.0")

    failures += _check_domain_schedule(env, cfg, curriculum, original_count)
    failures += _check_ripple_alarm(env, cfg)
    return failures


def _check_ripple_alarm(env, cfg):
    """R8.2's alarm: does the ripple diagnostic reach the logger, and can it see?

    Two halves, because they fail independently.  The *recorder* has to run on
    every step -- if it does not, the eligibility counter never fills, the
    metric is silently never emitted, and the alarm is off for the whole run
    with no error anywhere.  The *detector* has to separate a gait-frequency
    ripple from a reward that merely moves, which is checked by feeding the env
    a synthetic window rather than by hoping the rollout produces one: with zero
    actions the robot does not walk, and a check that depends on it walking
    would be testing the policy, not the wiring.
    """
    base = env.env
    failures = []
    dog_a, arm_a = zero_actions(env, cfg)
    window = base.ripple_window_steps

    env.reset()
    before = int(base.ripple_valid_steps.max())
    for _ in range(30):
        env.step(dog_a, arm_a)
    grew = int(base.ripple_valid_steps.max()) - before
    print(f"\n  ripple window {window} steps ({window * base.dt:.2f} s); "
          f"eligibility counter grew {grew} over 30 steps")
    if grew <= 0:
        failures.append(
            "the ripple window is not being filled -- _record_ref_tracking_window "
            "is not running, so the R8.2 alarm is silently off"
        )
        return failures

    frequency = base._gait_frequency_hz()
    walking = frequency > 0.1
    if not bool(walking.any()):
        print("  no env has a running gait clock; skipped the detector half")
        return failures

    steps = torch.arange(window, device=base.device, dtype=torch.float).unsqueeze(1)
    time = steps * base.dt

    def emit(series):
        base.ref_tracking_window[:] = series
        base.ripple_cursor = 0
        base.ripple_valid_steps[:] = torch.where(
            walking, torch.full_like(base.ripple_valid_steps, window),
            torch.zeros_like(base.ripple_valid_steps),
        )
        extras = {}
        base._log_gait_frequency_ripple(extras)
        return {k: float(v) for k, v in extras.items()}

    rippling = emit(0.6 + 0.2 * torch.sin(2 * math.pi * frequency.unsqueeze(0) * time))
    drifting = emit(0.6 + 0.2 * torch.sin(2 * math.pi * 0.4 * time).expand(-1, base.num_envs))
    print(f"  synthetic ripple at each env's own gait frequency: "
          f"fraction {rippling.get('perf_ref_tracking_ripple_fraction', float('nan')):.2f}, "
          f"band power {rippling.get('perf_ref_tracking_ripple_band_power', float('nan')):.2f}, "
          f"envs {rippling.get('perf_ref_tracking_ripple_env_count', 0):.0f}")
    print(f"  synthetic 0.4 Hz command drift: "
          f"fraction {drifting.get('perf_ref_tracking_ripple_fraction', float('nan')):.2f}")

    if "perf_ref_tracking_ripple_fraction" not in rippling:
        failures.append(
            "the ripple metric was not emitted even with a full window on every "
            "walking env -- it would never appear in wandb either"
        )
        return failures
    if rippling["perf_ref_tracking_ripple_fraction"] < 0.95:
        failures.append(
            f"a synthetic gait-frequency ripple registered at only "
            f"{rippling['perf_ref_tracking_ripple_fraction']:.2f}; the detector is "
            "not reading each env against its own commanded frequency"
        )
    if drifting.get("perf_ref_tracking_ripple_fraction", 1.0) > 0.05:
        failures.append(
            "a 0.4 Hz reward swing was reported as a gait-frequency ripple -- the "
            "alarm would fire on ordinary command transients"
        )

    base.ripple_valid_steps[:] = 0
    base.ref_tracking_window[:] = 0.0
    return failures


def _check_domain_schedule(env, cfg, curriculum, original_count):
    """R8.1's other two columns: randomisation width and stage-4 pushes.

    The reward schedule is only one of the three things R8.1's table asks each
    stage to change.  This drives the iteration counter across the boundaries
    and reads the **actually sampled** domain spread, because the failure mode
    here is silent: an intensity that never reaches the samplers leaves stage 1
    training at full randomisation and looks exactly like a working schedule
    from the outside.
    """
    base = env.env
    failures = []
    dog_a, arm_a = zero_actions(env, cfg)
    others = base.grouping.is_grouped & ~base.grouping.is_twin
    if not bool(others.any()):
        others = torch.ones_like(base.grouping.is_twin)

    def spread():
        base._update_domain_curriculum()
        every = torch.arange(base.num_envs, device=base.device)
        base._randomize_dof_props(every, base.cfg)
        base._randomize_rigid_body_props(every, base.cfg)
        base._nominalize_twins()
        values = base.motor_strengths[others]
        friction = base.friction_coeffs[others]
        return (float(values.max() - values.min()),
                float(friction.max() - friction.min()))

    print(f"\n  {'iter':>8}{'stage':>7}{'rand':>7}{'push':>7}"
          f"{'motor spread':>14}{'friction spread':>17}")
    observed = {}
    try:
        for iteration in (0, 6000, 6500, 7000, 12500, 13000):
            global_switch.count = iteration
            env.step(dog_a, arm_a)
            motor, friction = spread()
            observed[iteration] = (motor, friction)
            print(f"  {iteration:>8}{curriculum.stage(iteration):>7}"
                  f"{curriculum.randomization_intensity(iteration):>7.2f}"
                  f"{curriculum.disturbance_intensity(iteration):>7.2f}"
                  f"{motor:>14.4f}{friction:>17.4f}")
    finally:
        global_switch.count = original_count
        base._update_domain_curriculum()

    weak, full = observed[0], observed[7000]
    for index, name in enumerate(("motor strength", "friction")):
        if not weak[index] < full[index] * 0.8:
            failures.append(
                f"{name} spread is {weak[index]:.4f} in stage 1 against "
                f"{full[index]:.4f} at full intensity -- the randomisation "
                "schedule is not reaching the samplers"
            )
    if not bool(cfg.domain_rand.push_robots):
        failures.append(
            "domain_rand.push_robots is False, so stage 4 is a no-op -- in v1 "
            "pushes are the whole of its robustness-recovery claim"
        )
    if curriculum.disturbance_intensity(curriculum.stage_start(4) - 1) != 0.0:
        failures.append("stage-4 disturbance leaks into stage 3")
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
        # An octave with no FFT bin at all is a resolution limit, not a gap in
        # the sweep: a short segment gives coarse bins and the lowest octave can
        # fall entirely between two of them.  Reporting that as a spectral gap
        # would be blaming the signal for the measurement.
        gaps, unresolved = [], []
        for lo_hz in (0.1, 0.25, 0.5, 1.0):
            octave = spectrum[(freqs >= lo_hz) & (freqs < 2 * lo_hz)]
            if octave.numel() == 0:
                unresolved.append(f"{lo_hz}-{2 * lo_hz}Hz")
            elif float(octave.max()) <= 0.02 * float(spectrum.max()):
                gaps.append(f"{lo_hz}-{2 * lo_hz}Hz")
        if unresolved:
            print(f"      note: {', '.join(unresolved)} has no FFT bin over a "
                  f"{duration:.1f}s segment; not enough resolution to judge it")
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


def check_dr(env, cfg, steps=40):
    """Domain randomisation reaches the simulator, and sensing latency delays
    only what a sensor produces.

    Both halves of this check exist because their failure mode is silence.  A
    payload resampled into a tensor that IsaacGym never re-reads trains exactly
    like no randomisation at all, and an observation delayed by zero steps
    looks like a healthy run of a policy that is quietly still being handed the
    present.
    """
    base = env.env
    failures = []
    env.reset()
    dog_a, arm_a = zero_actions(env, cfg)

    # --- body properties actually reach the simulator ----------------------
    def sim_base_props(env_id):
        props = base.gym.get_actor_rigid_body_properties(
            base.envs[env_id], base.actor_handles[env_id]
        )
        return props[0].mass, (props[0].com.x, props[0].com.y, props[0].com.z)

    # The CoM the simulator integrates must be the CoM the critic is told
    # about: IsaacGym will not move it after prepare_sim, so the invariant is
    # that nothing else moves it either.
    probe = min(4, cfg.env.num_envs)
    print(f"  {'env':<5}{'mass':>10}{'com x':>10}{'com y':>10}{'com z':>10}")
    for env_id in range(cfg.env.num_envs):
        _, com = sim_base_props(env_id)
        for axis, value in enumerate(com):
            if abs(value - float(base.com_displacements[env_id, axis])) > 1e-6:
                failures.append(
                    f"env {env_id}: com_displacements[{axis}] "
                    f"{float(base.com_displacements[env_id, axis]):.4f} but the simulator "
                    f"is integrating {value:.4f}"
                )

    base.payloads[:probe] = torch.tensor([1.5, -1.5, 0.75, 0.0][:probe], device=base.device)
    base.refresh_actor_rigid_body_props(range(probe), cfg)
    for env_id in range(probe):
        mass, com = sim_base_props(env_id)
        print(f"  {env_id:<5}{mass:>10.4f}{com[0]:>10.4f}{com[1]:>10.4f}{com[2]:>10.4f}")
        want = base.default_body_mass + float(base.payloads[env_id])
        if abs(mass - want) > 1e-4:
            failures.append(f"env {env_id}: sim base mass {mass:.4f} != {want:.4f}")

    # Refreshing twice must not compound: the old creation-time callback cached
    # props[0].mass as the nominal mass, so a second pass would have added the
    # payload to a base that already contained it.
    mass_once, _ = sim_base_props(0)
    base.refresh_actor_rigid_body_props(range(probe), cfg)
    mass_twice, _ = sim_base_props(0)
    if abs(mass_once - mass_twice) > 1e-6:
        failures.append(
            f"refresh is not idempotent: {mass_once:.4f} -> {mass_twice:.4f}"
        )

    # --- the randomisation is on, and the twins are exempt ------------------
    com_before = base.com_displacements.clone()
    env.reset()
    for _ in range(steps):
        env.step(dog_a, arm_a)
    if not torch.equal(com_before, base.com_displacements):
        failures.append(
            "com_displacements was redrawn after the actors were created; the "
            "simulator cannot follow it there"
        )
    com_spread = float(base.com_displacements.abs().max())
    print(f"  max |base com displacement| {com_spread:.4f} m")
    if cfg.domain_rand.randomize_com_displacement and com_spread <= 0.0:
        failures.append("randomize_com_displacement is on but every env is at 0")
    if base._grouping_active():
        twins = base.is_nominal_twin
        if float(base.com_displacements[twins].abs().max()) > 0.0:
            failures.append("a nominal twin has a non-zero base com displacement")
        if float(base.stage1_ee_payload_com[twins].abs().max()) > 0.0:
            failures.append("a nominal twin carries an offset payload")
        if int(base.dog_obs_latency.delays[twins].max()) != 0:
            failures.append("a nominal twin observes with a delay")

    # --- the payload hangs off its centre of mass, not the grasp point ------
    # Forced to full intensity rather than waiting for the curriculum: at
    # iteration 0 the arm disturbance is off, so a plain rollout would exercise
    # nothing here and report success for a branch it never entered.
    previous = getattr(base, "stage1_arm_play_intensity", None)
    try:
        base.stage1_arm_play_intensity = 1.0
        everyone = torch.arange(cfg.env.num_envs, device=base.device)
        base._resample_stage1_ee_payload(everyone)
        base._nominalize_twins(everyone)
        env.step(dog_a, arm_a)
        loaded = (base.stage1_ee_payload_mass > 0.0) & (
            base.stage1_ee_payload_com.abs().sum(dim=-1) > 0.0
        )
        print(f"  payload offset exercised on {int(loaded.sum())}/{cfg.env.num_envs} envs")
        if not torch.any(loaded):
            failures.append("no env drew both a payload mass and a CoM offset at full intensity")
        else:
            base._apply_stage1_ee_payload_force()
            ee_pos = base.rigid_body_state.view(base.num_envs, -1, 13)[:, base.ee_idx, :3]
            lever = (base.stage1_payload_force_positions[:, base.ee_idx] - ee_pos)[loaded]
            offset = base.stage1_ee_payload_com[loaded]
            print(f"  payload lever arm: max {float(lever.norm(dim=-1).max()):.4f} m")
            if float(lever.norm(dim=-1).min()) <= 0.0:
                failures.append("a loaded env applies its payload at the grasp point")
            # The lever is the offset rotated into the world: a rotation, so
            # the length has to survive it exactly.
            length_error = (lever.norm(dim=-1) - offset.norm(dim=-1)).abs().max()
            if float(length_error) > 1e-5:
                failures.append(
                    f"the payload lever arm is not the sampled offset rotated ({float(length_error):.2e})"
                )
            if base._grouping_active() and float(
                base.stage1_ee_payload_com[base.is_nominal_twin].abs().max()
            ) > 0.0:
                failures.append("a nominal twin carries an offset payload at full intensity")
    finally:
        if previous is None:
            if hasattr(base, "stage1_arm_play_intensity"):
                del base.stage1_arm_play_intensity
        else:
            base.stage1_arm_play_intensity = previous

    # --- sensing latency: measured segments are late, commands are not ------
    delay = min(1, base.dog_obs_latency.max_steps)
    if delay == 0:
        failures.append("dog_obs_latency has no capacity; the delay can never bite")
    else:
        # Observation noise is added *after* the delay, so with it on the
        # comparison below measures the noise, not the lag.  Silenced for the
        # duration of this sub-check and restored afterwards: the two
        # mechanisms are independent and this gate is about the lag.
        noise_was = getattr(base.cfg.dog, "add_obs_noise", True)
        base.cfg.dog.add_obs_noise = False
        env.reset()
        base.dog_obs_latency.delays[:] = delay
        base.dog_obs_latency_fill[:] = True
        snapshots, observed = [], []
        for _ in range(steps):
            env.step(dog_a, arm_a)
            base.dog_obs_latency.delays[:] = delay          # survive the resets
            obs = env.get_dog_observations()["obs"]
            snapshots.append(base.projected_gravity.clone())
            observed.append(obs[:, :3].clone())
        base.cfg.dog.add_obs_noise = noise_was
        # Only envs whose ring holds `delay` steps of post-reset history: a
        # reset legitimately refills it with the fresh state, so an env that
        # restarted inside the window is expected to read the present.
        settled = base.episode_length_buf > delay
        print(f"  {int(settled.sum())}/{cfg.env.num_envs} envs have {delay} step(s) of history")
        if torch.any(settled):
            lag_error = (observed[-1][settled] - snapshots[-1 - delay][settled]).abs().max()
            live_error = (observed[-1][settled] - snapshots[-1][settled]).abs().max()
            print(f"  gravity obs vs t-{delay}: {float(lag_error):.2e}   vs t: {float(live_error):.2e}")
            if float(lag_error) > 1e-5:
                failures.append(
                    f"observed gravity does not match the measurement {delay} step(s) ago"
                )
            if float(live_error) <= 1e-5:
                failures.append("observed gravity is the live value: the delay is not applied")
        else:
            failures.append("no environment survived long enough to test the delay")

        # A command is known onboard without a sensor, so it must appear in the
        # very observation built after it changes.
        base.commands_dog[:, 0] = 0.4321
        obs = env.get_dog_observations()["obs"]
        commanded = obs[:, 3 * base.num_actions_loco + 3]
        expected = 0.4321 * float(base.commands_scale_dog[0])
        print(f"  vx command in obs: {float(commanded[0]):.4f} (expected {expected:.4f})")
        if abs(float(commanded[0]) - expected) > 1e-4:
            failures.append("the vx command is not reaching the observation undelayed")

    return failures


def check_terrain(env, cfg, steps=60):
    """Mild rough ground: the right tiles, the right environments on them, and
    every height still measured against the local ground.

    The failure this is really guarding is the quiet one: a nominal twin
    standing on uneven tile.  Nothing crashes, no metric looks wrong, and the
    response that every other environment is being aligned to just stops being
    a fixed target.
    """
    base = env.env
    failures = []
    if not base._terrain_roughness_active():
        print(f"  terrain.mesh_type={cfg.terrain.mesh_type}, "
              f"roughness_tiers={getattr(cfg.terrain, 'roughness_tiers', None)}: rough ground is off")
        if cfg.terrain.mesh_type in ("trimesh", "heightfield"):
            failures.append("trimesh terrain without roughness tiers: the tiles are unaddressable")
        return failures

    tiers = list(cfg.terrain.roughness_tiers)
    if float(tiers[0]) != 0.0:
        failures.append(f"tier 0 is {tiers[0]} m, not flat: the twins have nowhere nominal to stand")
    if not cfg.terrain.measure_heights:
        failures.append(
            "measure_heights is off on rough ground: every body-height quantity "
            "silently reverts to the world frame (global invariant 9)"
        )

    # --- the tiles really carry the amplitudes they were asked for ----------
    field = base.terrain.height_field_raw * cfg.terrain.vertical_scale
    per_env_pixels = int(cfg.terrain.terrain_width / cfg.terrain.horizontal_scale)
    print(f"  {'tier':<6}{'asked (m)':>12}{'tile spread (m)':>18}{'envs':>8}{'twins':>8}")
    env_tiers = base._terrain_tier_of_env()
    for tier, amplitude in enumerate(tiers):
        columns = cfg.terrain.roughness_tier_columns[tier]
        col = columns[len(columns) // 2]
        patch = field[:per_env_pixels, col * per_env_pixels:(col + 1) * per_env_pixels]
        spread = float(patch.max() - patch.min())
        selected = env_tiers == tier
        twins = int((selected & base.is_nominal_twin).sum()) if base._grouping_active() else 0
        print(f"  {tier:<6}{amplitude:>12.3f}{spread:>18.4f}{int(selected.sum()):>8}{twins:>8}")
        # random_uniform_terrain quantises to vertical_scale, so allow a step.
        if abs(spread - 2 * amplitude) > 2 * cfg.terrain.vertical_scale:
            failures.append(
                f"tier {tier} tiles span {spread:.4f} m, expected {2 * amplitude:.4f} m"
            )
        if tier > 0 and twins:
            failures.append(f"{twins} nominal twin(s) are standing on tier {tier}")

    # --- the twins are flat, and stay flat when the curriculum opens up -----
    if base._grouping_active():
        every = torch.arange(cfg.env.num_envs, device=base.device)
        base._assign_terrain_tiers(every, intensity=1.0)
        env_tiers = base._terrain_tier_of_env()
        twin_tiers = env_tiers[base.is_nominal_twin]
        others = env_tiers[~base.is_nominal_twin]
        print(f"  at intensity 1.0: twins on tiers {sorted(set(int(t) for t in twin_tiers))}, "
              f"others on {sorted(set(int(t) for t in others))}")
        if int(twin_tiers.max()) != 0:
            failures.append("a twin left flat ground when the randomisation opened up")
        if len(tiers) > 1 and int(others.max()) == 0:
            failures.append("at full intensity no environment reached a rough tier")

        base._assign_terrain_tiers(every, intensity=0.0)
        if int(base._terrain_tier_of_env().max()) != 0:
            failures.append("at intensity 0 some environment is not on flat ground")
        base._assign_terrain_tiers(every, intensity=1.0)

        # A twin walks for a whole episode.  Being *placed* on flat ground is
        # not the invariant -- staying on it is.
        _, y_reach = base._episode_reach()
        width = float(cfg.terrain.terrain_width)
        safe = base._terrain_tier_band(0, confined=True)
        stray = base.is_nominal_twin & ~torch.isin(base.terrain_types, safe)
        flat_band = cfg.terrain.roughness_tier_columns[0]
        print(f"  lateral reach in one episode {y_reach:.1f} m; flat band is "
              f"{len(flat_band) * width:.0f} m wide, {len(safe)} of its {len(flat_band)} columns "
              f"are further than that from another tier or the map edge")
        if int(stray.sum()) > 0:
            failures.append(
                f"{int(stray.sum())} twin(s) can walk out of the flat tier within one episode"
            )

    # --- heights are measured against the local ground ----------------------
    env.reset()
    dog_a, arm_a = zero_actions(env, cfg)
    for _ in range(steps):
        env.step(dog_a, arm_a)
    reference = base._terrain_reference_height()
    print(f"  terrain reference height: [{float(reference.min()):.4f}, {float(reference.max()):.4f}] m")
    if not isinstance(base.measured_heights, torch.Tensor):
        failures.append("measured_heights is still a scalar; _terrain_reference_height cannot work")
    elif float(reference.abs().max()) == 0.0:
        failures.append("every terrain reference height is 0 on rough ground")
    if base._grouping_active():
        twin_reference = reference[base.is_nominal_twin].abs().max()
        print(f"  max |reference height| under a twin: {float(twin_reference):.4f} m")
        if float(twin_reference) > cfg.terrain.vertical_scale:
            failures.append("a twin's ground is not flat")

    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", default="all",
                        choices=["r1", "r2", "r3", "r4", "r5", "r6", "r8", "conv", "dr", "terrain", "all"])
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--robot", type=str, default="go2_x5")
    parser.add_argument("--policy", type=str, default=None,
                        help="stage-1 dog checkpoint; required for the R3 check")
    parser.add_argument("--seconds", type=float, default=60.0, help="R3 rollout length")
    args = parser.parse_args()

    env, cfg = build_env(args.num_envs, args.sim_device, args.robot)
    wanted = (["conv", "r1", "r2", "r3", "r4", "r5", "r6", "r8", "dr", "terrain"]
              if args.check == "all" else [args.check])

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
        elif name == "r8":
            results[name] = check_r8(env, cfg)
        elif name == "conv":
            results[name] = check_conv(env, cfg)
        elif name == "dr":
            results[name] = check_dr(env, cfg)
        elif name == "terrain":
            results[name] = check_terrain(env, cfg)

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
