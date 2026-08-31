"""R2 acceptance tests for the prescribed reference-response model.

These are the numerical criteria spelled out in
``docs/project-design-rlmpc-v3-coding.md`` R2, plus the invariants that section
states as design decisions.  Deliberately CPU-only and IsaacGym-free: run with
``pytest go1_gym/response/test_reference.py`` in any environment that has torch.
"""

import math

import pytest
import torch

from go1_gym.response import (
    DEFAULT_CHANNELS,
    ChannelSpec,
    ReferenceModel,
    critically_damped_step_response,
    validate_channels,
)

DT = 0.02  # policy step: sim dt 0.005 x decimation 4
PITCH = next(c for c in DEFAULT_CHANNELS if c.name == "pitch")
HEIGHT = next(c for c in DEFAULT_CHANNELS if c.name == "height")


def _single_channel_model(spec, num_envs=1, dt=DT, dtype=torch.float32):
    return ReferenceModel([spec], num_envs=num_envs, dt=dt, dtype=dtype)


def _run_step(model, amplitude, num_steps):
    """Apply a constant command from rest, returning xi and xi_dot histories."""
    command = torch.full((model.num_envs, model.num_channels), amplitude, dtype=model.dtype)
    xi_hist, rate_hist = [], []
    for _ in range(num_steps):
        model.step(command)
        xi_hist.append(model.xi.clone())
        rate_hist.append(model.xi_dot.clone())
    return torch.cat(xi_hist, dim=1), torch.cat(rate_hist, dim=1)


# --------------------------------------------------------------------------
# R2 acceptance 1: agreement with the analytic critically damped step response
# --------------------------------------------------------------------------


def test_step_response_matches_analytic_solution():
    """0.4 rad pitch step, omega_n=5, zeta=1 -> analytic solution within 1e-4.

    The default pitch rate limit (0.8 rad/s) is deliberately not exercised here:
    the unsaturated peak rate of a 0.4 rad step is A*omega_n/e = 0.736 rad/s,
    so this runs the pure linear system, which is what the criterion targets.
    """
    amplitude = 0.4
    num_steps = 40  # 0.8 s
    model = _single_channel_model(PITCH)
    xi, _ = _run_step(model, amplitude, num_steps)

    t = torch.arange(1, num_steps + 1, dtype=torch.float64) * DT
    analytic = critically_damped_step_response(amplitude, PITCH.omega_n, t)

    err = (xi[0].to(torch.float64) - analytic).abs()
    assert err.max().item() < 1e-4, f"max deviation {err.max().item():.3e} exceeds 1e-4"


@pytest.mark.parametrize(
    "time_s, expected",
    [(0.1, 0.036), (0.2, 0.106), (0.4, 0.238), (0.8, 0.363)],
)
def test_step_response_reference_points(time_s, expected):
    """The four checkpoints quoted verbatim in the R2 acceptance criteria."""
    amplitude = 0.4
    num_steps = int(round(time_s / DT))
    model = _single_channel_model(PITCH)
    xi, _ = _run_step(model, amplitude, num_steps)
    # The doc quotes three decimals, so compare at that resolution.
    assert xi[0, -1].item() == pytest.approx(expected, abs=5e-4)


def test_exact_discretisation_is_step_size_independent():
    """Halving dt must not change xi at a shared wall-clock time.

    This is the property a Euler integrator does not have, and the reason the
    MPC can discretise the same continuous system at its own rate without
    disagreeing with what the policy was trained against.
    """
    amplitude = 0.4
    coarse = _single_channel_model(PITCH, dt=DT, dtype=torch.float64)
    fine = _single_channel_model(PITCH, dt=DT / 2, dtype=torch.float64)

    xi_coarse, _ = _run_step(coarse, amplitude, 40)   # 0.8 s
    xi_fine, _ = _run_step(fine, amplitude, 80)       # 0.8 s

    assert xi_coarse[0, -1].item() == pytest.approx(xi_fine[0, -1].item(), abs=1e-12)


# --------------------------------------------------------------------------
# R2 acceptance 2: rate saturation
# --------------------------------------------------------------------------


def test_rate_saturation_caps_peak_rate():
    """0.6 rad step: peak rate is held at 0.8 rad/s (unclipped it would be 1.10)."""
    amplitude = 0.6
    unclipped_peak = amplitude * PITCH.omega_n / math.e
    assert unclipped_peak == pytest.approx(1.1036, abs=1e-3), "guard on the doc's 1.10 figure"
    assert unclipped_peak > PITCH.rate_limit, "the 0.6 rad step must actually saturate"

    model = _single_channel_model(PITCH)
    _, rate = _run_step(model, amplitude, 100)

    peak = rate.abs().max().item()
    assert peak <= PITCH.rate_limit + 1e-6, f"peak rate {peak:.4f} exceeds the limit"
    assert peak == pytest.approx(PITCH.rate_limit, rel=1e-3), "the limit should be reached, not merely respected"


def test_small_step_does_not_saturate():
    """A 0.4 rad step stays strictly inside the box -- the limit must not bite early."""
    model = _single_channel_model(PITCH)
    _, rate = _run_step(model, 0.4, 100)
    peak = rate.abs().max().item()
    assert peak < PITCH.rate_limit
    assert peak == pytest.approx(0.4 * PITCH.omega_n / math.e, rel=2e-3)


def test_saturated_position_never_moves_faster_than_the_limit():
    """The continuous-time spec bounds |dxi/dt| pointwise, so over one step
    |dxi| <= rate_limit * dt.  Clipping only the endpoint of an exact linear
    step violates this by ~40% on a large command jump."""
    bound = PITCH.rate_limit * DT
    # Several amplitudes, including ones large enough that saturation both
    # starts and ends inside the window.
    for amplitude in (0.6, 1.2, 2.0):
        model = _single_channel_model(PITCH)
        xi, _ = _run_step(model, amplitude, 120)
        first = xi[0, 0].abs().item()
        increments = (xi[0, 1:] - xi[0, :-1]).abs()
        assert first <= bound * 1.01, f"A={amplitude}: first step {first:.5f} > {bound:.5f}"
        # 1% covers the single boundary-crossing step, which takes the linear
        # branch and is the one place the piecewise update is O(dt) approximate.
        # Endpoint-clipping (the wrong implementation) overshoots by ~40% here.
        assert increments.max().item() <= bound * 1.01, (
            f"A={amplitude}: max step {increments.max().item():.5f} exceeds "
            f"rate_limit*dt = {bound:.5f}"
        )


def test_saturated_segment_is_a_constant_velocity_ramp():
    """R2's stated intent: large commands degenerate into a ramp."""
    model = _single_channel_model(PITCH)
    xi, rate = _run_step(model, 1.2, 60)
    saturated = rate[0].abs() >= PITCH.rate_limit - 1e-6
    assert saturated.sum() > 10, "expected a long saturated segment"

    # Inside the saturated run the position advances by exactly limit*dt.
    idx = saturated.nonzero().flatten()
    run = idx[(idx > idx.min()) & (idx < idx.max())]
    increments = xi[0, run] - xi[0, run - 1]
    expected = PITCH.rate_limit * DT
    assert increments.min().item() == pytest.approx(expected, rel=1e-5)
    assert increments.max().item() == pytest.approx(expected, rel=1e-5)


def test_saturated_trajectory_is_step_size_independent():
    """The property that motivated exact discretisation must survive
    saturation -- otherwise training at 50 Hz and planning at another rate
    disagree exactly where the rate limit bites."""
    coarse = _single_channel_model(PITCH, dt=DT, dtype=torch.float64)
    fine = _single_channel_model(PITCH, dt=DT / 4, dtype=torch.float64)

    xi_coarse, rate_coarse = _run_step(coarse, 1.2, 50)   # 1.0 s
    xi_fine, _ = _run_step(fine, 1.2, 200)                # 1.0 s
    assert (rate_coarse[0].abs() >= PITCH.rate_limit - 1e-9).any(), "must actually saturate"

    assert xi_coarse[0, -1].item() == pytest.approx(xi_fine[0, -1].item(), abs=2e-3)


def test_saturated_response_still_converges_without_overshoot():
    """Clipping must not turn a critically damped system into an oscillator."""
    amplitude = 0.6
    model = _single_channel_model(PITCH)
    xi, _ = _run_step(model, amplitude, 300)  # 6 s
    assert xi[0, -1].item() == pytest.approx(amplitude, abs=1e-4), "unit DC gain"
    assert xi.max().item() <= amplitude + 1e-6, "critically damped reference must not overshoot"
    diffs = xi[0, 1:] - xi[0, :-1]
    assert (diffs >= -1e-7).all(), "approach to the command must be monotonic"


# --------------------------------------------------------------------------
# R2 acceptance 3 / global invariant 1: command resampling must not reset xi
# --------------------------------------------------------------------------


def test_command_change_leaves_reference_state_continuous():
    """A command jump moves the equilibrium, not the state."""
    model = _single_channel_model(PITCH)
    settle = torch.full((1, 1), 0.3, dtype=model.dtype)
    for _ in range(200):  # 4 s, fully settled
        model.step(settle)

    before = model.xi.clone()
    jumped = torch.full((1, 1), -0.3, dtype=model.dtype)
    model.step(jumped)
    after = model.xi.clone()

    jump_in_command = 0.6
    position_change = (after - before).abs().item()

    assert position_change < 0.05 * jump_in_command, (
        f"xi moved {position_change:.4f} in one step on a {jump_in_command} command jump; "
        "the reference state was reset instead of evolving"
    )
    # Guard against the two ways a reset would show up.
    assert abs(after.item() - jumped.item()) > 0.5, "xi snapped to the new command"
    assert abs(after.item()) > 0.25, "xi snapped to zero"


def test_reference_never_moves_faster_than_the_command_it_chases():
    """Sanity bound across a full jump sequence: no discontinuity anywhere."""
    model = _single_channel_model(PITCH)
    commands = [0.4, -0.4, 0.0, 0.35, -0.2]
    history = []
    for value in commands:
        cmd = torch.full((1, 1), value, dtype=model.dtype)
        for _ in range(25):  # 0.5 s per command -- jumps land mid-transient
            model.step(cmd)
            history.append(model.xi.item())

    steps = torch.tensor(history)
    increments = (steps[1:] - steps[:-1]).abs()
    # Position increments are bounded by the constrained linear dynamics; a
    # state reset would show up as an increment on the order of the command jump.
    assert increments.max().item() < 0.1 * 0.8, "a step-to-step jump in xi indicates a reset"


# --------------------------------------------------------------------------
# Reset alignment (R2 invariant: align to measurement, only on reset)
# --------------------------------------------------------------------------


def test_pending_alignment_snaps_to_measured_state_on_next_step():
    model = _single_channel_model(PITCH, num_envs=4)
    cmd = torch.full((4, 1), 0.3, dtype=model.dtype)
    for _ in range(50):
        model.step(cmd)
    assert model.xi.abs().max().item() > 0.1, "precondition: reference has moved"

    env_ids = torch.tensor([1, 3])
    model.request_alignment(env_ids)
    assert model.has_pending_alignment

    measured = torch.full((4, 1), -0.12, dtype=model.dtype)
    settled = model.xi[0, 0].item()
    model.step(cmd, measured=measured)

    assert not model.has_pending_alignment
    # Aligned envs restart from the measurement (then take one step towards cmd).
    for env in (1, 3):
        assert model.xi[env, 0].item() == pytest.approx(-0.12, abs=2e-3)
    # Untouched envs are unaffected.
    for env in (0, 2):
        assert model.xi[env, 0].item() == pytest.approx(settled, abs=1e-3)


def test_step_without_measured_raises_when_alignment_pending():
    model = _single_channel_model(PITCH, num_envs=2)
    model.request_alignment(torch.tensor([0]))
    with pytest.raises(ValueError, match="pending reference alignment"):
        model.step(torch.zeros((2, 1), dtype=model.dtype))


# --------------------------------------------------------------------------
# Channel bookkeeping
# --------------------------------------------------------------------------


def test_default_channel_table_matches_the_command_layout():
    """Channel order is (vx, vy, wyaw, height, pitch); command indices are not."""
    assert [c.name for c in DEFAULT_CHANNELS] == ["vx", "vy", "wyaw", "height", "pitch"]
    assert [c.cmd_index for c in DEFAULT_CHANNELS] == [0, 1, 2, 5, 3]
    validate_channels(DEFAULT_CHANNELS)


def test_pitch_bandwidth_must_stay_below_height_bandwidth():
    """R2 invariant, enforced rather than documented."""
    bad = (
        ChannelSpec(name="height", cmd_index=5, omega_n=5.0, rate_limit=0.25),
        ChannelSpec(name="pitch", cmd_index=3, omega_n=7.0, rate_limit=0.8),
    )
    with pytest.raises(ValueError, match="omega_n\\[pitch\\]"):
        validate_channels(bad)


def test_duplicate_command_indices_are_rejected():
    bad = (
        ChannelSpec(name="a", cmd_index=0, omega_n=5.0, rate_limit=1.0),
        ChannelSpec(name="b", cmd_index=0, omega_n=5.0, rate_limit=1.0),
    )
    with pytest.raises(ValueError, match="duplicate cmd_index"):
        validate_channels(bad)


def test_non_critical_damping_is_rejected():
    with pytest.raises(NotImplementedError, match="critically damped"):
        ReferenceModel([PITCH], num_envs=1, dt=DT, zeta=0.7)


def test_gather_commands_picks_the_right_columns():
    model = ReferenceModel(DEFAULT_CHANNELS, num_envs=3, dt=DT)
    commands_dog = torch.arange(3 * 11, dtype=torch.float32).reshape(3, 11)
    gathered = model.gather_commands(commands_dog)

    assert gathered.shape == (3, 5)
    for env in range(3):
        for channel, spec in enumerate(DEFAULT_CHANNELS):
            assert gathered[env, channel].item() == commands_dog[env, spec.cmd_index].item()


def test_gather_commands_rejects_a_too_narrow_buffer():
    """Running without --dyna_gait gives a 6-wide buffer; height sits at 5, so
    that still fits -- but anything narrower must fail loudly, not silently."""
    model = ReferenceModel(DEFAULT_CHANNELS, num_envs=2, dt=DT)
    model.gather_commands(torch.zeros(2, 6))  # ok: max index is 5
    with pytest.raises(ValueError, match="at least 6"):
        model.gather_commands(torch.zeros(2, 5))


def test_normalized_rate_maps_saturation_to_unit_magnitude():
    model = _single_channel_model(PITCH)
    _run_step(model, 0.6, 20)
    assert model.normalized_rate().abs().max().item() <= 1.0 + 1e-6
    _, rate = _run_step(model, 0.6, 1)
    del rate


def test_channels_evolve_independently_across_envs_and_channels():
    """No cross-talk: the whole value of the reference model is per-channel decoupling."""
    model = ReferenceModel(DEFAULT_CHANNELS, num_envs=2, dt=DT)
    commands = torch.zeros(2, 5)
    commands[0, 4] = 0.4  # env 0 pitch only
    commands[1, 0] = 0.5  # env 1 vx only

    for _ in range(30):
        model.step(commands)

    assert model.xi[0, 4].item() > 0.1
    assert model.xi[0, :4].abs().max().item() == 0.0
    assert model.xi[1, 0].item() > 0.1
    assert model.xi[1, 1:].abs().max().item() == 0.0


def test_reference_is_stateless_with_respect_to_batch_size_ordering():
    """Env i's trajectory must not depend on what env j was commanded."""
    solo = _single_channel_model(PITCH, num_envs=1)
    batch = _single_channel_model(PITCH, num_envs=4)

    solo_cmd = torch.full((1, 1), 0.35, dtype=solo.dtype)
    batch_cmd = torch.tensor([[0.35], [-0.4], [0.0], [0.6]], dtype=batch.dtype)

    for _ in range(60):
        solo.step(solo_cmd)
        batch.step(batch_cmd)

    assert batch.xi[0, 0].item() == pytest.approx(solo.xi[0, 0].item(), abs=1e-6)
