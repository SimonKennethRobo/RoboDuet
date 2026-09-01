"""R9.1: the numerics behind the evaluation figures.

Tested against signals whose true order is known, because the paper's central
claim is comparative -- "a markedly simpler model, to the same accuracy" -- and
a comparison is only worth anything if the measuring instrument is calibrated.
"""

import math

import pytest
import torch

from go1_gym.response.analysis import (
    empirical_bode,
    fit_first_order,
    fit_second_order,
    fit_second_order_with_residual,
    model_order_curve,
    phase_conditioned_dispersion,
    phase_conditioned_mean,
    residual_fourier_coefficients,
    simulate_first_order,
    simulate_second_order,
)

DT = 0.02


def step_command(steps=600, amplitude=0.4, hold=150):
    u = torch.zeros(steps, dtype=torch.float64)
    for k in range(steps):
        u[k] = amplitude if (k // hold) % 2 == 0 else -amplitude
    return u


def gait_phase(steps, frequency=3.0, dt=DT):
    return torch.remainder(torch.arange(steps, dtype=torch.float64) * dt * frequency, 1.0)


# --- the fits recover what generated the data ------------------------------


def test_first_order_recovers_its_own_time_constant():
    u = step_command()
    for tau in (0.05, 0.2, 0.8):
        y = simulate_first_order(u, DT, tau)
        fit = fit_first_order(u, y, DT)
        assert fit.parameter == pytest.approx(tau, rel=0.05)
        assert fit.r_squared > 0.999


def test_second_order_recovers_its_own_bandwidth():
    u = step_command()
    for omega in (3.0, 5.0, 8.0):
        y = simulate_second_order(u, DT, omega)
        fit = fit_second_order(u, y, DT)
        assert fit.parameter == pytest.approx(omega, rel=0.05)
        assert fit.r_squared > 0.999


def test_r_squared_barely_separates_model_orders_on_step_data():
    """A finding that shapes the figure, pinned so it cannot be forgotten.

    A FIRST-order fit explains a second-order step response with R^2 = 0.995.
    R^2 saturates near 1 and is therefore close to useless for the claim
    "a markedly simpler model suffices to the same accuracy" -- plotted as an
    order curve it would look flat and prove nothing either way.
    """
    u = step_command()
    y = simulate_second_order(u, DT, 6.0)
    first = fit_first_order(u, y, DT)
    assert first.r_squared > 0.99
    assert first.r_squared == pytest.approx(0.995, abs=0.01)


def test_residual_energy_is_the_discriminator_r_squared_is_not():
    """Same two fits, on a log scale: ten orders of magnitude apart.

    This is why R9.1 asks for normalized residual energy alongside R^2, and why
    the order curve should be plotted on that axis.
    """
    u = step_command()
    y = simulate_second_order(u, DT, 6.0)
    first = fit_first_order(u, y, DT)
    second = fit_second_order(u, y, DT)
    assert second.normalized_residual_energy < first.normalized_residual_energy / 1e6


def test_rich_excitation_separates_the_orders_far_better_than_a_step():
    """The other half of the answer: use R6's signals, not a step.

    A step is mostly a slow rise, which any lag can follow; a sweep visits
    frequencies where the two orders differ by 20 dB/decade of roll-off.  So the
    order curve should be computed from the chirp environments' data.
    """
    steps = 6000
    t = torch.arange(steps, dtype=torch.float64) * DT
    duration = steps * DT
    sweep = (2.0 - 0.1) / duration
    u = 0.4 * torch.sin(2 * math.pi * (0.1 * t + 0.5 * sweep * t * t))
    y = simulate_second_order(u, DT, 6.0)

    on_chirp = fit_first_order(u, y, DT)
    on_step = fit_first_order(step_command(), simulate_second_order(step_command(), DT, 6.0), DT)
    assert on_chirp.r_squared < on_step.r_squared - 0.05


# --- the residual term -----------------------------------------------------


def test_the_residual_term_absorbs_a_periodic_gait_ripple():
    """A predictable ripple should not be charged against model order: the
    planner knows the phase and can feed it forward."""
    steps = 900
    u = step_command(steps)
    phase = gait_phase(steps)
    clean = simulate_second_order(u, DT, 6.0)
    ripple = 0.05 * torch.sin(2 * math.pi * 2 * phase)      # 2nd harmonic, as trot gives
    y = clean + ripple

    plain = fit_second_order(u, y, DT)
    with_residual = fit_second_order_with_residual(u, y, phase, DT)
    assert with_residual.r_squared > plain.r_squared
    assert with_residual.r_squared > 0.99


def test_the_residual_cannot_rescue_a_wrong_bandwidth():
    """Fitted after the dynamics, not jointly -- otherwise the residual would
    absorb bandwidth error and flatter the reported model order."""
    steps = 900
    u = step_command(steps)
    phase = gait_phase(steps)
    y = simulate_second_order(u, DT, 6.0)
    fit = fit_second_order_with_residual(u, y, phase, DT)
    assert fit.parameter == pytest.approx(6.0, rel=0.05)


def test_phase_conditioned_mean_is_periodic_in_phase():
    steps = 800
    phase = gait_phase(steps)
    signal = torch.sin(2 * math.pi * phase) + 0.01 * torch.randn(steps, dtype=torch.float64)
    recovered = phase_conditioned_mean(signal, phase, num_bins=16)
    assert torch.corrcoef(torch.stack((recovered, torch.sin(2 * math.pi * phase))))[0, 1] > 0.99


def test_model_order_curve_is_monotone_for_a_second_order_plant():
    steps = 900
    u = step_command(steps)
    phase = gait_phase(steps)
    y = simulate_second_order(u, DT, 6.0) + 0.03 * torch.sin(2 * math.pi * 2 * phase)
    curve = model_order_curve(u, y, phase, DT)
    assert (
        curve["first"].r_squared
        < curve["second"].r_squared
        < curve["second+residual"].r_squared
    )


# --- frequency response ----------------------------------------------------


def test_bode_recovers_a_known_first_order_corner():
    """A first-order lag is -3 dB at 1/(2 pi tau); the estimator has to find it."""
    steps = 20000
    dt = DT
    t = torch.arange(steps, dtype=torch.float64) * dt
    duration = steps * dt
    f0, f1 = 0.1, 2.0
    sweep = (f1 - f0) / duration
    u = torch.sin(2 * math.pi * (f0 * t + 0.5 * sweep * t * t))
    tau = 0.5
    y = simulate_first_order(u, dt, tau)

    bode = empirical_bode(u, y, dt, band=(f0, f1))
    corner = 1.0 / (2 * math.pi * tau)
    index = int((bode["frequency_hz"] - corner).abs().argmin())
    assert float(bode["gain_db"][index]) == pytest.approx(-3.0, abs=1.5)
    assert float(bode["phase_deg"][index]) == pytest.approx(-45.0, abs=12.0)


def test_bode_gain_falls_with_frequency_for_a_lag():
    steps = 20000
    t = torch.arange(steps, dtype=torch.float64) * DT
    duration = steps * DT
    sweep = (2.0 - 0.1) / duration
    u = torch.sin(2 * math.pi * (0.1 * t + 0.5 * sweep * t * t))
    y = simulate_first_order(u, DT, 0.5)
    bode = empirical_bode(u, y, DT)
    assert float(bode["gain_db"][0]) > float(bode["gain_db"][-1]) + 6.0


def test_bode_excludes_bins_the_chirp_never_excited():
    """R6's chirp is amplitude-tapered, so its input spectrum is deliberately
    not flat; dividing by an unexcited bin invents resonance."""
    steps = 4000
    t = torch.arange(steps, dtype=torch.float64) * DT
    u = torch.sin(2 * math.pi * 0.5 * t)            # a single tone
    y = simulate_first_order(u, DT, 0.3)
    bode = empirical_bode(u, y, DT, band=(0.1, 2.0))
    assert bode["frequency_hz"].numel() < 20
    assert float(bode["frequency_hz"].min()) == pytest.approx(0.5, abs=0.05)


# --- cross-domain dispersion ----------------------------------------------


def test_dispersion_is_zero_when_every_domain_agrees():
    phase = gait_phase(400)
    one = torch.sin(2 * math.pi * phase)
    values = torch.stack([one, one.clone(), one.clone()])
    assert phase_conditioned_dispersion(values, phase) == pytest.approx(0.0, abs=1e-12)


def test_dispersion_grows_with_cross_domain_disagreement():
    phase = gait_phase(400)
    one = torch.sin(2 * math.pi * phase)
    small = torch.stack([one, one * 1.05, one * 0.95])
    large = torch.stack([one, one * 1.5, one * 0.5])
    assert (
        phase_conditioned_dispersion(small, phase)
        < phase_conditioned_dispersion(large, phase)
    )


def test_dispersion_needs_at_least_two_domains():
    phase = gait_phase(100)
    with pytest.raises(ValueError, match="at least two domains"):
        phase_conditioned_dispersion(torch.zeros(1, 100), phase)
    with pytest.raises(ValueError, match="domains, samples"):
        phase_conditioned_dispersion(torch.zeros(100), phase)


def test_fourier_coefficients_pick_out_the_planted_harmonic():
    phase = gait_phase(2000)
    residual = 0.3 * torch.cos(2 * math.pi * 2 * phase)      # pure 2nd harmonic
    coefficients = residual_fourier_coefficients(residual, phase, harmonics=3)
    magnitude = coefficients.abs()[0]
    assert float(magnitude[1]) > 5 * float(magnitude[0])
    assert float(magnitude[1]) > 5 * float(magnitude[2])
    assert float(magnitude[1]) == pytest.approx(0.15, rel=0.1)


def test_fourier_coefficients_are_returned_per_domain():
    """Metric 3 is their cross-domain variance, so they must not be reduced."""
    phase = gait_phase(1000)
    residual = torch.stack([
        0.3 * torch.cos(2 * math.pi * 2 * phase),
        0.1 * torch.cos(2 * math.pi * 2 * phase),
    ])
    coefficients = residual_fourier_coefficients(residual, phase, harmonics=3)
    assert coefficients.shape == (2, 3)
    assert float(coefficients.abs()[0, 1]) > float(coefficients.abs()[1, 1])


# --- guards ----------------------------------------------------------------


def test_simulators_reject_nonsense_parameters():
    u = step_command(10)
    with pytest.raises(ValueError, match="tau"):
        simulate_first_order(u, DT, 0.0)
    with pytest.raises(ValueError, match="omega_n"):
        simulate_second_order(u, DT, -1.0)
