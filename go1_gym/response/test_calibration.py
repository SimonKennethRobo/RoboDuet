"""R8.2: recovering (omega_n, rate_limit) from a measured step response.

The sigma calibration (R4) is tested in test_reward_terms.py; this file covers
the other half of calibration -- turning what the robot actually achieved into
the reference model it should be asked to track.
"""

import math

import pytest
import torch

from go1_gym.response import ChannelSpec, ReferenceModel
from go1_gym.response.calibration import (
    normalised_rise_time,
    omega_from_peak_acceleration,
    omega_from_rise_time,
    percentile,
    rate_limit_from_peak_rate,
    saturation_ratio,
)

# --- R8.2: recovering the model from a measured step -------------------------


def test_omega_round_trips_through_the_analytic_peak_acceleration():
    """A * omega^2 is the peak acceleration of a critically damped step."""
    for omega in (3.0, 5.0, 8.0):
        for amplitude in (0.2, 0.4, 1.0):
            peak = amplitude * omega ** 2
            assert omega_from_peak_acceleration(peak, amplitude) == pytest.approx(omega)


def test_omega_recovered_from_a_simulated_trajectory():
    """End to end against the same integrator the env runs, not a formula."""
    dt = 0.02
    for omega in (4.0, 7.0):
        amplitude = 0.4
        model = ReferenceModel(
            [ChannelSpec(name="c", cmd_index=0, omega_n=omega, rate_limit=1e6)],
            num_envs=1, dt=dt, dtype=torch.float64,
        )
        command = torch.full((1, 1), amplitude, dtype=torch.float64)
        rates, positions = [], []
        for _ in range(400):
            model.step(command)
            rates.append(float(model.xi_dot[0, 0]))
            positions.append(float(model.xi[0, 0]))
        peak_rate = max(rates)
        assert peak_rate == pytest.approx(amplitude * omega / math.e, rel=0.02)

        # A one-step difference is biased low by exp(-omega*dt); uncorrected it
        # is wrong by ~4% at omega=4, and the bias grows with bandwidth.
        peak_acceleration = rates[0] / dt
        naive = omega_from_peak_acceleration(peak_acceleration, amplitude)
        assert naive < omega * 0.99
        corrected = omega_from_peak_acceleration(peak_acceleration, amplitude, dt=dt)
        assert corrected == pytest.approx(omega, rel=0.01)

        # The rise time needs no differentiation at all, which is why it is the
        # one the script uses on measured data.
        target = 0.5 * amplitude
        crossing = next(k for k, v in enumerate(positions) if v >= target)
        assert omega_from_rise_time((crossing + 1) * dt) == pytest.approx(omega, rel=0.03)


def test_saturation_ratio_separates_slew_bound_from_bandwidth_bound():
    omega, amplitude = 8.0, 0.5
    unsaturated = amplitude * omega / math.e
    assert saturation_ratio(unsaturated, omega, amplitude) == pytest.approx(1.0)
    assert saturation_ratio(unsaturated / 4, omega, amplitude) == pytest.approx(0.25)


def test_percentile_is_the_lower_tail_not_the_mean():
    """R8.2 forbids the mean: the reference has to be realisable in the hard
    domains, not in the average one."""
    values = [1.0] * 8 + [10.0] * 2          # mean 2.8, p20 = 1.0
    assert percentile(values, 0.2) == pytest.approx(1.0)
    assert percentile(values, 0.5) == pytest.approx(1.0)
    assert percentile(values, 1.0) == pytest.approx(10.0)
    assert percentile(values, 0.2) < sum(values) / len(values)


def test_percentile_interpolates_and_guards_its_input():
    assert percentile([0.0, 10.0], 0.5) == pytest.approx(5.0)
    with pytest.raises(ValueError, match="no samples"):
        percentile([], 0.2)
    with pytest.raises(ValueError, match="fraction"):
        percentile([1.0], 1.5)


def test_rate_limit_never_goes_negative():
    assert rate_limit_from_peak_rate(-3.0) == 0.0
    assert rate_limit_from_peak_rate(1.25) == 1.25


def test_normalised_rise_time_matches_the_analytic_response():
    for fraction in (0.1, 0.5, 0.9):
        x = normalised_rise_time(fraction)
        assert 1.0 - (1.0 + x) * math.exp(-x) == pytest.approx(fraction, abs=1e-9)
    with pytest.raises(ValueError, match="fraction"):
        normalised_rise_time(0.0)
    with pytest.raises(ValueError, match="rise_time"):
        omega_from_rise_time(0.0)
