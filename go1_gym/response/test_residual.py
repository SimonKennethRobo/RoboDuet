"""R3 acceptance tests: phase-conditioned residual and zero-delay detrending.

Synthetic signals only -- CPU, no IsaacGym.  The point of these is to pin the
two properties R3 states as design decisions: the estimate must recover a
phase-locked waveform, and the phase-lookup detrend must have no delay while
the low-pass one does.
"""

import math

import pytest
import torch

from go1_gym.response import PhaseResidualEstimator

DT = 0.02
GAIT_HZ = 3.0
STEPS_PER_SECOND = int(round(1.0 / DT))


def _make(num_envs=1, num_channels=1, **kwargs):
    return PhaseResidualEstimator(num_envs=num_envs, num_channels=num_channels, dt=DT, **kwargs)


def _drive(est, seconds, signal_fn, speed=0.5, gait_hz=GAIT_HZ, active=True, phase0=0.0):
    """Run the estimator over a synthetic signal.

    ``signal_fn(t, phase) -> (num_envs, num_channels)``.  Returns the histories
    needed by the tests: measurement, phase-lookup detrend, low-pass.
    """
    n = int(seconds / DT)
    phase = torch.full((est.num_envs,), phase0)
    speed_t = torch.full((est.num_envs,), float(speed))
    active_t = torch.full((est.num_envs,), bool(active))
    meas_hist, detrended_hist, lp_hist = [], [], []
    for k in range(n):
        t = k * DT
        phase = torch.remainder(phase + DT * gait_hz, 1.0)
        sb, pb = est.speed_bin(speed_t), est.phase_bin(phase)
        y = signal_fn(t, phase)
        # Use path first: it must be read-only, so ordering cannot matter.
        detrended_hist.append(est.detrend(y, sb, pb).clone())
        est.update(y, sb, pb, active_t)
        meas_hist.append(y.clone())
        lp_hist.append(est.y_lp.clone())
    return (torch.stack(meas_hist), torch.stack(detrended_hist), torch.stack(lp_hist))


# --------------------------------------------------------------------------
# R3 acceptance 1: delta_hat vs phase is a clean periodic waveform
# --------------------------------------------------------------------------


def test_estimate_recovers_a_phase_locked_waveform():
    """30 s at a fixed command; delta_hat over phase should be dominated by the
    first one or two harmonics."""
    est = _make()

    def signal(t, phase):
        oscillation = 0.02 * torch.sin(2 * math.pi * phase) + 0.006 * torch.sin(4 * math.pi * phase)
        return (0.3 + oscillation).unsqueeze(-1)

    _drive(est, 30.0, signal)

    curve = est.delta_hat[0, est.speed_bin(torch.tensor([0.5]))[0], :, 0]
    spectrum = torch.fft.rfft(curve.double())
    power = spectrum.abs() ** 2
    ac_power = power[1:]  # drop the DC bin
    assert ac_power.sum() > 0
    first_two = ac_power[:2].sum() / ac_power.sum()
    assert first_two > 0.7, f"harmonics 1-2 carry only {first_two:.2%} of the AC power"


def test_estimate_amplitude_matches_the_injected_oscillation():
    est = _make()
    amplitude = 0.02

    def signal(t, phase):
        return (0.3 + amplitude * torch.sin(2 * math.pi * phase)).unsqueeze(-1)

    _drive(est, 40.0, signal)
    curve = est.delta_hat[0, est.speed_bin(torch.tensor([0.5]))[0], :, 0]
    recovered = (curve.max() - curve.min()).item() / 2.0
    # The lagging low-pass attenuates a 3 Hz component by only ~11% at tau=0.5s,
    # so most of the amplitude survives into the residual.
    assert recovered == pytest.approx(amplitude, rel=0.25)


# --------------------------------------------------------------------------
# R3 acceptance 2: the phase lookup has no lag, the low-pass does
# --------------------------------------------------------------------------


def test_phase_lookup_detrend_has_no_lag_while_lowpass_does():
    """Converge on a steady signal, then step the trend.

    Both are estimates of the low-frequency content: ``y - delta_hat(phase)``
    versus ``y_lp``.  The first must follow the step immediately; the second
    lags by roughly its time constant.  This is why the reward path may not use
    the low-pass.
    """
    est = _make()
    amplitude = 0.02
    step_time = 40.0
    step_size = 0.2

    def signal(t, phase):
        trend = 0.3 + (step_size if t >= step_time else 0.0)
        return (trend + amplitude * torch.sin(2 * math.pi * phase)).unsqueeze(-1)

    _, detrended, lowpass = _drive(est, step_time + 2.0, signal)

    step_index = int(step_time / DT)
    target = 0.3 + step_size
    # 0.2 s after the step -- less than half the low-pass time constant.
    probe = step_index + int(0.2 / DT)

    lookup_err = abs(detrended[probe, 0, 0].item() - target)
    lowpass_err = abs(lowpass[probe, 0, 0].item() - target)

    assert lookup_err < 0.25 * step_size, f"phase-lookup detrend lagged: err {lookup_err:.4f}"
    assert lowpass_err > 0.25 * step_size, f"low-pass did not lag: err {lowpass_err:.4f}"
    assert lowpass_err > 2 * lookup_err


def test_detrending_removes_the_oscillation_not_the_trend():
    est = _make()
    amplitude = 0.02

    def signal(t, phase):
        return (0.3 + amplitude * torch.sin(2 * math.pi * phase)).unsqueeze(-1)

    measured, detrended, _ = _drive(est, 40.0, signal)
    tail = slice(int(30.0 / DT), None)
    raw_ripple = measured[tail, 0, 0].max() - measured[tail, 0, 0].min()
    left_ripple = detrended[tail, 0, 0].max() - detrended[tail, 0, 0].min()
    assert left_ripple < 0.35 * raw_ripple, "oscillation was not removed"
    assert detrended[tail, 0, 0].mean().item() == pytest.approx(0.3, abs=0.01)


# --------------------------------------------------------------------------
# R3 invariants
# --------------------------------------------------------------------------


def test_estimate_is_per_environment():
    """Two envs with different oscillation amplitudes must not average together."""
    est = _make(num_envs=2)

    def signal(t, phase):
        amps = torch.tensor([0.02, 0.06])
        return (0.3 + amps * torch.sin(2 * math.pi * phase)).unsqueeze(-1)

    _drive(est, 40.0, signal)
    sb = est.speed_bin(torch.tensor([0.5, 0.5]))
    spans = [
        (est.delta_hat[i, sb[i], :, 0].max() - est.delta_hat[i, sb[i], :, 0].min()).item()
        for i in range(2)
    ]
    assert spans[1] == pytest.approx(3 * spans[0], rel=0.3), spans


def test_speed_bins_do_not_contaminate_each_other():
    """The oscillation's shape changes with travel speed, which is exactly why
    R3 requires speed binning."""
    est = _make()
    slow_amp, fast_amp = 0.01, 0.05

    def make_signal(amp):
        def signal(t, phase):
            return (0.3 + amp * torch.sin(2 * math.pi * phase)).unsqueeze(-1)
        return signal

    _drive(est, 40.0, make_signal(slow_amp), speed=0.05)
    _drive(est, 40.0, make_signal(fast_amp), speed=0.9)

    slow_bin = est.speed_bin(torch.tensor([0.05]))[0]
    fast_bin = est.speed_bin(torch.tensor([0.9]))[0]
    assert slow_bin != fast_bin

    def span(b):
        return (est.delta_hat[0, b, :, 0].max() - est.delta_hat[0, b, :, 0].min()).item() / 2

    assert span(slow_bin) == pytest.approx(slow_amp, rel=0.35)
    assert span(fast_bin) == pytest.approx(fast_amp, rel=0.35)


def test_use_path_never_mutates_state():
    """R3 invariant: the update path and the use path must stay separate.  A
    lookup that quietly updated would close the feedback loop this design is
    built to avoid."""
    est = _make(num_envs=4, num_channels=3)
    est.delta_hat.normal_()
    est.sample_count.fill_(7.0)
    before = (est.delta_hat.clone(), est.sample_count.clone(), est.y_lp.clone())

    sb = est.speed_bin(torch.rand(4))
    pb = est.phase_bin(torch.rand(4))
    est.detrend(torch.randn(4, 3), sb, pb)
    est.lookup(sb, pb)
    est.is_converged(sb, pb)

    assert torch.equal(est.delta_hat, before[0])
    assert torch.equal(est.sample_count, before[1])
    assert torch.equal(est.y_lp, before[2])


def test_inactive_samples_are_skipped_and_snap_the_lowpass():
    """Standing freezes the phase clock, so its samples would all pile into one
    bin; and a fresh reset leaves the low-pass holding a stale level."""
    est = _make(num_envs=2)
    speed = torch.full((2,), 0.5)
    phase = torch.full((2,), 0.25)
    sb, pb = est.speed_bin(speed), est.phase_bin(phase)

    measured = torch.full((2, 1), 0.7)
    inactive = torch.tensor([False, False])
    for _ in range(50):
        est.update(measured, sb, pb, inactive)

    assert torch.all(est.sample_count == 0), "inactive samples must not be counted"
    assert torch.all(est.delta_hat == 0), "inactive samples must not move the estimate"
    assert torch.allclose(est.y_lp, measured), "low-pass should snap while inactive"


def test_convergence_gate_opens_only_after_enough_visits():
    est = _make(min_cycles=10.0)
    speed = torch.full((1,), 0.5)
    phase = torch.full((1,), 0.3)
    sb, pb = est.speed_bin(speed), est.phase_bin(phase)
    active = torch.ones(1, dtype=torch.bool)

    assert not est.is_converged(sb, pb).item()
    for _ in range(9):
        est.update(torch.zeros(1, 1), sb, pb, active)
    assert not est.is_converged(sb, pb).item()
    est.update(torch.zeros(1, 1), sb, pb, active)
    assert est.is_converged(sb, pb).item()


def test_bins_are_independent_across_channels():
    est = _make(num_channels=2)

    def signal(t, phase):
        a = 0.02 * torch.sin(2 * math.pi * phase)
        b = torch.zeros_like(a)
        return torch.stack((0.3 + a, 0.1 + b), dim=-1)

    _drive(est, 40.0, signal)
    sb = est.speed_bin(torch.tensor([0.5]))[0]
    span0 = (est.delta_hat[0, sb, :, 0].max() - est.delta_hat[0, sb, :, 0].min()).item()
    span1 = (est.delta_hat[0, sb, :, 1].max() - est.delta_hat[0, sb, :, 1].min()).item()
    assert span0 > 0.02
    assert span1 < 1e-3


# --------------------------------------------------------------------------
# Bookkeeping
# --------------------------------------------------------------------------


def test_speed_bucketing_boundaries():
    est = _make(speed_bin_edges=(0.15, 0.35, 0.6))
    speeds = torch.tensor([0.0, 0.149, 0.15, 0.34, 0.35, 0.59, 0.6, 5.0])
    assert est.speed_bin(speeds).tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert est.num_speed_bins == 4


def test_phase_bucketing_wraps():
    est = _make(num_phase_bins=16)
    phases = torch.tensor([0.0, 0.0624, 0.0626, 0.999, 1.0, 1.5, -0.01])
    bins = est.phase_bin(phases)
    assert bins.tolist() == [0, 0, 1, 15, 0, 8, 15]
    assert bins.max().item() < 16


def test_rejects_unsorted_speed_edges():
    with pytest.raises(ValueError, match="strictly increasing"):
        _make(speed_bin_edges=(0.4, 0.2))


def test_storage_footprint_is_small():
    est = PhaseResidualEstimator(num_envs=4096, num_channels=5, dt=DT)
    megabytes = est.delta_hat.numel() * est.delta_hat.element_size() / 1e6
    assert megabytes < 8.0, f"{megabytes:.1f} MB"
