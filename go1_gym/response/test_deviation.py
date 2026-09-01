"""R7.1: the (g, l) response-deviation statistics.

These four numbers replace the teacher-student pathway R7.2 asked for, so what
matters most is not that they are accurate but that they are **honest**: a
statistic that reports a confident wrong value when it has no information is
worse than one that reports nothing, because the policy cannot tell the two
apart.  Most of the tests below are about the "no information" cases.
"""

import math

import pytest
import torch

from go1_gym.response import ResponseDeviationEstimator

DT = 0.02
RATE_LIMIT = [1.2, 0.8, 3.0, 0.25, 0.8]      # vx vy wyaw height pitch
COMMAND_SCALE = [0.5, 0.3, 1.0, 0.25, 0.4]
WYAW, PITCH = 2, 4


def make(num_envs=4, channels=(WYAW, PITCH), tau_s=5.0, warmup_s=0.2, **kwargs):
    return ResponseDeviationEstimator(
        num_envs, channels, RATE_LIMIT, COMMAND_SCALE, dt=DT,
        tau_s=tau_s, warmup_s=warmup_s, **kwargs
    )


def drive(estimator, steps, *, command, gain=1.0, lag=0.0, rate=0.3, num_channels=5):
    """Hold a command and feed back ``gain * command + lag`` as the measurement."""
    n = estimator.num_envs
    u = torch.zeros(n, num_channels)
    xi = torch.zeros(n, num_channels)
    xi_dot = torch.zeros(n, num_channels)
    for channel, value in command.items():
        u[:, channel] = value
        xi[:, channel] = value
        xi_dot[:, channel] = rate
    for _ in range(steps):
        y = u * gain + lag
        estimator.update(y, u, xi, xi_dot)
    return estimator.observation()


# --- the honesty cases -----------------------------------------------------


def test_an_uncommanded_channel_reports_nothing_not_zero_gain():
    """The bug this gate exists for: EMA(u^2)=0 makes the regression read
    gain 0, i.e. "this domain has no gain", which is a confident wrong answer."""
    estimator = make()
    observation = drive(estimator, 600, command={PITCH: 0.4}, gain=0.5)
    wyaw_gain, wyaw_lag = observation[0, 0], observation[0, 1]
    assert float(wyaw_gain) == 0.0
    assert float(wyaw_lag) == 0.0


def test_warmup_reports_nominal():
    estimator = make(warmup_s=2.0)
    observation = drive(estimator, 10, command={PITCH: 0.4}, gain=0.5)
    assert torch.all(observation == 0.0)
    assert not estimator.converged.any()


def test_a_held_command_never_claims_a_lag():
    """sign(xi_dot) is arbitrary when the reference is not moving, so an
    ungated EMA would integrate noise through every held command."""
    estimator = make()
    observation = drive(estimator, 600, command={PITCH: 0.4}, gain=1.0, lag=0.3, rate=0.0)
    assert float(observation[0, 3]) == 0.0        # pitch lag
    assert float(observation[0, 2]) != 0.0        # ...but the gain is still identified


def test_the_deadband_is_a_fraction_of_the_rate_limit():
    estimator = make(rate_deadband=0.5)
    below = drive(estimator, 400, command={PITCH: 0.4}, lag=0.2,
                  rate=0.4 * RATE_LIMIT[PITCH])
    assert float(below[0, 3]) == 0.0
    estimator = make(rate_deadband=0.5)
    above = drive(estimator, 400, command={PITCH: 0.4}, lag=0.2,
                  rate=0.6 * RATE_LIMIT[PITCH])
    assert float(above[0, 3]) != 0.0


# --- correctness on cases where there IS information ------------------------


def test_gain_recovers_a_known_scale_factor():
    for gain in (0.5, 1.0, 1.4):
        estimator = make()
        observation = drive(estimator, 3000, command={PITCH: 0.4}, gain=gain)
        assert float(observation[0, 2]) == pytest.approx(gain - 1.0, abs=0.02)


def test_gain_is_a_regression_so_a_zero_mean_command_still_works():
    """The plan's EMA(y)/EMA(u) is meaningless here: every decision channel is
    commanded symmetrically about zero, so the denominator tends to zero."""
    estimator = make(num_envs=1)
    u = torch.zeros(1, 5)
    xi_dot = torch.zeros(1, 5)
    total = 0.0
    for step in range(4000):
        value = 0.4 if (step // 100) % 2 == 0 else -0.4     # zero mean
        u[:, PITCH] = value
        total += value
        estimator.update(u * 0.6, u, u, xi_dot)
    assert abs(total) < 500                                  # mean really is ~0
    assert float(estimator.observation()[0, 2]) == pytest.approx(-0.4, abs=0.02)


def test_lag_sign_follows_the_direction_of_reference_motion():
    """Positive when the body is behind the reference, negative when ahead."""
    behind = make()
    value = drive(behind, 3000, command={PITCH: 0.4}, lag=-0.1, rate=0.3)
    assert float(value[0, 3]) == pytest.approx(-0.1, abs=0.02)

    ahead = make()
    value = drive(ahead, 3000, command={PITCH: 0.4}, lag=+0.1, rate=-0.3)
    assert float(value[0, 3]) == pytest.approx(-0.1, abs=0.02)


def test_the_gain_ratio_has_no_start_up_transient():
    """Worth pinning because it is not obvious and it is load-bearing.

    ``cross`` and ``square`` are EMAs with the same alpha starting from the same
    zero, so their *ratio* is already correct long before either one has
    converged -- the lags cancel.  That is what makes a 5 s time constant
    affordable here: it costs nothing at the start of an episode, and only sets
    how fast a *change* in domain is tracked.
    """
    estimator = make(warmup_s=0.2, tau_s=5.0)
    observation = drive(estimator, int(0.5 / DT), command={PITCH: 0.4}, gain=0.4)
    assert float(observation[0, 2]) == pytest.approx(-0.6, abs=0.02)


def test_the_time_constant_governs_tracking_a_change_of_domain():
    """5 s against an MPC horizon of ~1 s is what makes these parameters rather
    than states; a wrong alpha silently turns one into the other."""
    estimator = make(tau_s=5.0)
    assert estimator.alpha == pytest.approx(1.0 - math.exp(-DT / 5.0))
    drive(estimator, 3000, command={PITCH: 0.4}, gain=1.0)
    assert float(estimator.observation()[0, 2]) == pytest.approx(0.0, abs=0.02)
    # now the domain changes; one time constant should cover ~63% of the step
    observation = drive(estimator, int(5.0 / DT), command={PITCH: 0.4}, gain=0.0)
    assert float(observation[0, 2]) == pytest.approx(-0.632, abs=0.05)


# --- lifecycle -------------------------------------------------------------


def test_reset_clears_only_the_selected_envs():
    estimator = make(num_envs=4)
    drive(estimator, 600, command={PITCH: 0.4}, gain=0.5)
    estimator.reset(torch.tensor([1, 3]))
    observation = estimator.observation()
    assert torch.all(observation[[1, 3]] == 0.0)
    assert float(observation[0, 2]) == pytest.approx(-0.5, abs=0.05)
    assert not estimator.converged[[1, 3]].any()
    assert estimator.converged[[0, 2]].all()


def test_reset_accepts_an_empty_selection():
    estimator = make()
    estimator.reset(torch.zeros(0, dtype=torch.long))
    assert estimator.observation().shape == (4, 4)


def test_observation_width_is_two_per_channel():
    assert make(channels=(PITCH,)).observation().shape == (4, 2)
    assert make(channels=(WYAW, PITCH)).observation().shape == (4, 4)
    assert make(channels=(0, 1, 2, 3, 4)).observation().shape == (4, 10)


def test_bad_configuration_rejected():
    with pytest.raises(ValueError, match="tau_s"):
        make(tau_s=0.0)
    with pytest.raises(ValueError, match="rate_deadband"):
        make(rate_deadband=1.0)
