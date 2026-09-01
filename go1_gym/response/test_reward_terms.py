"""R4 acceptance tests: the reward terms and the sigma calibration.

Synthetic trajectories only -- these pin the *shape* of the reward functions,
which is what R4's acceptance criteria are about.
"""

import math

import pytest
import torch

from go1_gym.response import (
    DEFAULT_CHANNELS,
    DEFAULT_TARGET_DISCRIMINATION,
    ChannelSpec,
    ReferenceModel,
    calibrate_channels,
    calibrate_sigma,
    discrimination_for_sigma,
)
from go1_gym.response.reward_terms import (
    domain_consistency,
    phase_variance,
    reference_tracking,
    settled_mask,
    soft_gate_from_events,
    steady_gain,
)

DT = 0.02
PITCH = next(c for c in DEFAULT_CHANNELS if c.name == "pitch")
STEP = 0.4
#: Calibrated for the pitch channel at a 0.4 rad step; see test_pitch_sigma_*.
PITCH_SIGMA = 0.1152

#: Representative step per channel: the half-width of the sampling range, which
#: is what the calibration script derives from cfg.
CALIBRATION_AMPLITUDES = {"vx": 0.5, "vy": 0.3, "wyaw": 1.0, "height": 0.25, "pitch": 0.4}


def _trajectory(omega_n, rate_limit, steps, amplitude=STEP):
    model = ReferenceModel(
        [ChannelSpec(name="c", cmd_index=0, omega_n=omega_n, rate_limit=rate_limit)],
        num_envs=1,
        dt=DT,
        dtype=torch.float64,
    )
    command = torch.full((1, 1), float(amplitude), dtype=torch.float64)
    out = torch.empty(steps, dtype=torch.float64)
    for k in range(steps):
        model.step(command)
        out[k] = model.xi[0, 0]
    return out


def _mean_reward(candidate, reference, sigma=PITCH_SIGMA):
    sigma_row = torch.tensor([[sigma]], dtype=torch.float64)
    weights = torch.ones(1, 1, dtype=torch.float64)
    rewards = [
        reference_tracking(
            candidate[k].reshape(1, 1), reference[k].reshape(1, 1), sigma_row, weights
        )[0]
        for k in range(candidate.shape[0])
    ]
    return torch.stack(rewards)


# --------------------------------------------------------------------------
# R4 acceptance: three artificial trajectories
# --------------------------------------------------------------------------


@pytest.mark.parametrize("aggressive_multiple", [2.0, 4.0])
def test_transient_reward_orders_follow_then_sluggish_then_aggressive(aggressive_multiple):
    """R4's stated ordering, evaluated over the transient window.

    The window matters and R4 does not name one, so this uses the same window
    the sigma calibration does -- the transient.  Over a *longer* window the
    ordering of the two failure modes flips (see the next test), because an
    aggressive policy reaches the command and then sits exactly on the reference
    for the remainder, earning back what it lost.
    """
    steps = int(round(4.0 / PITCH.omega_n / DT))
    reference = _trajectory(PITCH.omega_n, PITCH.rate_limit, steps)
    transient = slice(0, steps // 2)  # 0 .. 2/omega_n

    follows = _mean_reward(reference.clone(), reference)[transient].mean()
    sluggish = _mean_reward(
        _trajectory(0.5 * PITCH.omega_n, PITCH.rate_limit, steps), reference
    )[transient].mean()
    aggressive = _mean_reward(
        _trajectory(
            aggressive_multiple * PITCH.omega_n,
            aggressive_multiple * PITCH.rate_limit,
            steps,
        ),
        reference,
    )[transient].mean()

    assert follows > sluggish > aggressive, (
        f"follows={follows:.4f} sluggish={sluggish:.4f} aggressive={aggressive:.4f}"
    )
    assert follows == pytest.approx(1.0, abs=1e-9)


def test_aggressive_policy_is_driven_to_near_zero_reward():
    """R4's worked example: a policy that settles in ~0.2 s (about 4x the
    reference bandwidth) should see its reward collapse to the 0.01 order."""
    steps = int(round(4.0 / PITCH.omega_n / DT))
    reference = _trajectory(PITCH.omega_n, PITCH.rate_limit, steps)
    aggressive = _trajectory(4.0 * PITCH.omega_n, 4.0 * PITCH.rate_limit, steps)
    rewards = _mean_reward(aggressive, reference)

    probe = int(0.1 / DT) - 1
    assert reference[probe].item() == pytest.approx(0.036, abs=1e-3)
    assert (aggressive[probe] - reference[probe]).abs().item() == pytest.approx(0.21, abs=0.03)
    assert rewards[probe].item() < 0.1
    assert rewards.min().item() < 0.02, "worst-instant reward should reach the 0.01 order"


def test_ordering_flips_over_a_long_window():
    """Documenting a real limitation rather than hiding it.

    Over a window several times the settling time, aggressive scores ABOVE
    sluggish.  The reward is symmetric in the error, so once an aggressive
    policy has converged it is indistinguishable from one that followed, while a
    sluggish one is still behind.  Aggregated over an episode there is therefore
    no extra penalty on leading versus lagging.  If that turns out to matter,
    the fix is an asymmetric error, which is a design change beyond R4 as
    written -- so this test pins the current behaviour instead of asserting the
    ordering R4 states for the transient.
    """
    steps = int(round(10.0 / PITCH.omega_n / DT))
    reference = _trajectory(PITCH.omega_n, PITCH.rate_limit, steps)
    sluggish = _mean_reward(_trajectory(0.5 * PITCH.omega_n, PITCH.rate_limit, steps), reference)
    aggressive = _mean_reward(
        _trajectory(2.0 * PITCH.omega_n, 2.0 * PITCH.rate_limit, steps), reference
    )
    assert aggressive.mean() > sluggish.mean()


def test_detrending_reduces_the_variance_of_the_tracking_reward():
    """R4 acceptance: rerun without detrending and the reward variance should
    grow noticeably.  Here the gait ripple is injected explicitly."""
    steps = int(round(6.0 / PITCH.omega_n / DT))
    reference = _trajectory(PITCH.omega_n, PITCH.rate_limit, steps)
    phase = torch.arange(steps, dtype=torch.float64) * DT * 3.0
    ripple = 0.05 * torch.sin(2 * math.pi * phase)

    measured = reference + ripple
    with_detrend = _mean_reward(measured - ripple, reference)  # perfect delta_hat
    without_detrend = _mean_reward(measured, reference)

    assert without_detrend.var() > 5 * with_detrend.var(), (
        f"var without={without_detrend.var():.3e} with={with_detrend.var():.3e}"
    )
    assert with_detrend.mean() > without_detrend.mean()


# --------------------------------------------------------------------------
# Sigma calibration
# --------------------------------------------------------------------------


def test_pitch_sigma_lands_in_the_range_the_document_quotes():
    """R4 quotes 0.08-0.12 for the pitch channel at a +-0.4 rad range.  That
    pins the upper half of the stated 30-50% discrimination band: 0.40 gives
    0.127, outside the quoted range."""
    result = calibrate_sigma("pitch", PITCH.omega_n, PITCH.rate_limit, STEP, DT)
    assert 0.08 <= result.sigma <= 0.12, result
    assert result.sigma == pytest.approx(PITCH_SIGMA, abs=5e-3)
    assert DEFAULT_TARGET_DISCRIMINATION == 0.45

    midpoint = calibrate_sigma(
        "pitch", PITCH.omega_n, PITCH.rate_limit, STEP, DT, target_discrimination=0.40
    )
    assert midpoint.sigma > 0.12, "the 0.40 midpoint should fall outside the quoted range"


def test_calibration_hits_its_target_discrimination():
    results = calibrate_channels(DEFAULT_CHANNELS, CALIBRATION_AMPLITUDES, DT)
    assert set(results) == {c.name for c in DEFAULT_CHANNELS}
    for result in results.values():
        assert result.discrimination == pytest.approx(DEFAULT_TARGET_DISCRIMINATION, abs=1e-3)
        assert 0.0 < result.sigma < result.amplitude


def test_sigma_scales_with_the_channel_amplitude():
    """The channels carry different units, which is why R4 forbids a shared
    sigma; a doubled step should roughly double the scale."""
    small = calibrate_sigma("c", 5.0, 100.0, 0.2, DT).sigma
    large = calibrate_sigma("c", 5.0, 100.0, 0.4, DT).sigma
    assert large == pytest.approx(2 * small, rel=0.05)


def test_discrimination_is_monotonic_in_sigma():
    gap = torch.linspace(0.0, 0.2, 50, dtype=torch.float64)
    values = [discrimination_for_sigma(gap, s) for s in (0.01, 0.05, 0.1, 0.5, 2.0)]
    assert all(a > b for a, b in zip(values, values[1:])), values


def test_calibration_rejects_a_missing_amplitude():
    with pytest.raises(ValueError, match="no calibration amplitude"):
        calibrate_channels(DEFAULT_CHANNELS, {"vx": 0.5}, DT)


# --------------------------------------------------------------------------
# Masks and gates
# --------------------------------------------------------------------------


def test_settled_mask_is_per_channel():
    """Thresholds are 2/omega_n and 3/omega_n, and omega_n differs per channel,
    so the slowest channel stays masked longest."""
    settle = torch.tensor([[10.0, 20.0, 30.0]])
    counter = torch.tensor([5, 15, 25, 35])
    mask = settled_mask(counter, settle)
    assert mask.shape == (4, 3)
    assert mask[0].tolist() == [0.0, 0.0, 0.0]
    assert mask[1].tolist() == [1.0, 0.0, 0.0]
    assert mask[2].tolist() == [1.0, 1.0, 0.0]
    assert mask[3].tolist() == [1.0, 1.0, 1.0]


def test_soft_gate_holds_then_reopens():
    timer = torch.zeros(2)
    triggered = torch.tensor([True, False])
    timer = soft_gate_from_events(timer, triggered, hold_steps=25)
    assert timer.tolist() == [25.0, 0.0]

    quiet = torch.tensor([False, False])
    for _ in range(24):
        timer = soft_gate_from_events(timer, quiet, hold_steps=25)
    assert timer[0].item() == 1.0
    timer = soft_gate_from_events(timer, quiet, hold_steps=25)
    assert timer.tolist() == [0.0, 0.0]


def test_soft_gate_zeroes_the_term_rather_than_granting_it():
    """Handing out full credit while disturbed would pay the policy for getting
    disturbed."""
    sigma = torch.tensor([[0.1]])
    weights = torch.ones(1, 1)
    perfect = torch.zeros(2, 1)
    open_gate = reference_tracking(perfect, perfect, sigma, weights, gate=torch.ones(2))
    shut_gate = reference_tracking(perfect, perfect, sigma, weights, gate=torch.zeros(2))
    assert open_gate.tolist() == [1.0, 1.0]
    assert shut_gate.tolist() == [0.0, 0.0]


# --------------------------------------------------------------------------
# Penalty terms
# --------------------------------------------------------------------------


def test_phase_variance_penalises_deviation_not_amplitude():
    """The point of R4.2: a large but repeatable oscillation is fine, a small
    but wandering one is not."""
    weights = torch.ones(1, 1)
    large_but_matched = phase_variance(
        torch.tensor([[0.05]]), torch.tensor([[0.05]]), weights
    )
    small_but_wrong = phase_variance(
        torch.tensor([[0.01]]), torch.tensor([[-0.01]]), weights
    )
    assert large_but_matched.item() == pytest.approx(0.0)
    assert small_but_wrong.item() > 0.0


def test_steady_gain_compares_against_the_command_not_the_reference():
    weights = torch.ones(1, 1)
    measured = torch.tensor([[0.27]])
    command = torch.tensor([[0.30]])
    value = steady_gain(measured, command, weights)
    assert value.item() == pytest.approx((0.30 - 0.27) ** 2)


def test_penalties_are_non_negative_so_the_scale_owns_the_sign():
    weights = torch.ones(4, 3)
    torch.manual_seed(0)
    assert torch.all(phase_variance(torch.randn(4, 3), torch.randn(4, 3), weights) >= 0)
    assert torch.all(steady_gain(torch.randn(4, 3), torch.randn(4, 3), weights) >= 0)


def test_masked_channels_contribute_nothing():
    weights = torch.ones(1, 3)
    mask = torch.tensor([[1.0, 0.0, 0.0]])
    deviation = torch.tensor([[0.1, 5.0, 5.0]])
    value = phase_variance(deviation, torch.zeros(1, 3), weights, mask)
    assert value.item() == pytest.approx(0.01)


def test_reference_tracking_is_bounded_and_peaks_at_perfect_tracking():
    sigma = torch.tensor([[0.1, 0.2]])
    weights = torch.tensor([[1.0, 3.0]])
    perfect = reference_tracking(torch.zeros(1, 2), torch.zeros(1, 2), sigma, weights)
    far = reference_tracking(torch.full((1, 2), 10.0), torch.zeros(1, 2), sigma, weights)
    assert perfect.item() == pytest.approx(1.0)
    assert far.item() == pytest.approx(0.0, abs=1e-12)


def test_channel_weights_are_honoured():
    sigma = torch.tensor([[0.1, 0.1]])
    detrended = torch.tensor([[0.0, 1.0]])  # channel 1 completely wrong
    reference = torch.zeros(1, 2)
    heavy_on_good = reference_tracking(
        detrended, reference, sigma, torch.tensor([[9.0, 1.0]])
    )
    heavy_on_bad = reference_tracking(
        detrended, reference, sigma, torch.tensor([[1.0, 9.0]])
    )
    assert heavy_on_good.item() == pytest.approx(0.9, abs=1e-6)
    assert heavy_on_bad.item() == pytest.approx(0.1, abs=1e-6)


# --- R5: domain consistency -------------------------------------------------


def test_domain_consistency_is_zero_when_the_response_matches_the_twin():
    detrended = torch.randn(8, 5)
    weights = torch.ones(1, 5)
    valid = torch.ones(8)
    assert torch.allclose(
        domain_consistency(detrended, detrended.clone(), weights, valid),
        torch.zeros(8),
    )


def test_domain_consistency_grows_with_the_gap_and_is_non_negative():
    weights = torch.ones(1, 5)
    valid = torch.ones(3)
    twin = torch.zeros(3, 5)
    detrended = torch.tensor([0.0, 0.1, -0.3]).unsqueeze(-1).repeat(1, 5)
    value = domain_consistency(detrended, twin, weights, valid)
    assert torch.all(value >= 0)
    assert value[0] < value[1] < value[2]
    # symmetric: overshoot and undershoot cost the same
    assert torch.allclose(
        domain_consistency(-detrended, twin, weights, valid), value
    )


def test_domain_consistency_respects_the_validity_mask():
    """Ungrouped envs, the twin itself, and desynchronised groups contribute 0."""
    weights = torch.ones(1, 5)
    detrended = torch.full((4, 5), 0.5)
    twin = torch.zeros(4, 5)
    valid = torch.tensor([0.0, 1.0, 0.0, 1.0])
    value = domain_consistency(detrended, twin, weights, valid)
    assert value.tolist() == [0.0, 1.25, 0.0, 1.25]


def test_domain_consistency_weights_channels_independently():
    valid = torch.ones(1)
    twin = torch.zeros(1, 5)
    detrended = torch.zeros(1, 5)
    detrended[0, 3] = 1.0
    weights = torch.tensor([[1.0, 1.0, 1.0, 4.0, 1.0]])
    assert float(domain_consistency(detrended, twin, weights, valid)) == 4.0


def test_domain_consistency_matches_a_grouping_broadcast_end_to_end():
    """The shape R5 specifies, assembled the way the env assembles it."""
    from go1_gym.response import EnvGrouping

    grouping = EnvGrouping(8, group_size=4)
    detrended = torch.zeros(8, 5)
    detrended[1] = 0.2          # member of group 0 drifts
    detrended[5] = 0.2          # member of group 1 drifts
    twin = grouping.broadcast_from_twin(detrended)
    weights = torch.ones(1, 5)

    value = domain_consistency(detrended, twin, weights, grouping.valid)
    assert float(value[1]) == pytest.approx(5 * 0.04)
    assert float(value[0]) == 0.0      # twins never score
    assert float(value[2]) == 0.0      # matches its twin exactly

    grouping.mark_desync(torch.tensor([3]))
    off = domain_consistency(detrended, twin, weights, grouping.valid)
    assert float(off[1]) == 0.0        # group 0 is now desynchronised
    assert float(off[5]) == pytest.approx(5 * 0.04)   # group 1 unaffected
