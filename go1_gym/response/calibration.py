"""Calibration of the R4.1 error scale ``sigma``, per channel.

R4 is explicit that sigma must be calibrated rather than guessed, and it gives
the procedure: build the reference trajectory and a candidate that is "twice as
fast", then pick sigma so the difference in integrated reward over the transient
window is 30-50% of the largest difference achievable.

Why that criterion.  ``exp(-e^2 / sigma^2)`` is flat for ``e << sigma`` and flat
again (at zero) for ``e >> sigma``; it only discriminates in between.  Too large
a sigma and an aggressive policy scores almost the same as one that follows the
reference, so the reward carries no signal about *how* the target was reached --
which is the entire mechanism of this method.  Too small and the reward is zero
almost everywhere, giving no gradient at all.

The units differ per channel (m/s, rad/s, m, rad), so a shared sigma is
meaningless and each channel is calibrated on its own.

Everything here is a pure computation on the reference model -- no simulator, no
policy -- so it runs in milliseconds and is unit-tested.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Sequence

import torch

from .reference import ChannelSpec, ReferenceModel


#: R4 asks for 30-50% of the maximum achievable integrated-reward gap.  0.45
#: rather than the midpoint, for two reasons that agree: it is the value at
#: which this procedure reproduces the sigma range the requirements document
#: itself quotes for the pitch channel (0.08-0.12 for a 0.4 rad step -- 0.40
#: gives 0.127, outside it), and the documented failure mode is "sigma too
#: large, no discrimination", so the smaller-sigma half of the band is the safer
#: side to sit on.
DEFAULT_TARGET_DISCRIMINATION = 0.45


@dataclass(frozen=True)
class SigmaCalibration:
    """Result for one channel."""

    name: str
    sigma: float
    amplitude: float
    window_s: float
    #: Achieved discrimination: mean over the window of
    #: ``1 - exp(-(fast - nominal)^2 / sigma^2)``, i.e. the fraction of the
    #: largest possible integrated-reward gap that this sigma actually realises.
    discrimination: float
    #: Peak deviation of the "twice as fast" candidate from the reference.
    peak_gap: float


def _trajectory(omega_n: float, rate_limit: float, amplitude: float, dt: float, steps: int):
    """Reference response to a step of ``amplitude`` from rest."""
    model = ReferenceModel(
        [ChannelSpec(name="c", cmd_index=0, omega_n=omega_n, rate_limit=rate_limit)],
        num_envs=1,
        dt=dt,
        dtype=torch.float64,
    )
    command = torch.full((1, 1), float(amplitude), dtype=torch.float64)
    out = torch.empty(steps, dtype=torch.float64)
    for k in range(steps):
        model.step(command)
        out[k] = model.xi[0, 0]
    return out


def discrimination_for_sigma(gap: torch.Tensor, sigma: float) -> float:
    """Fraction of the maximum achievable integrated-reward gap realised.

    The nominal candidate sits exactly on the reference and always scores 1, so
    the integrated difference is ``mean(1 - exp(-gap^2 / sigma^2))``.  Its
    supremum is 1 (reached as sigma -> 0), which is what "the largest possible
    difference" means in R4's wording.
    """
    if sigma <= 0.0:
        return 1.0
    return float(torch.mean(1.0 - torch.exp(-(gap ** 2) / (sigma ** 2))))


def calibrate_sigma(
    name: str,
    omega_n: float,
    rate_limit: float,
    amplitude: float,
    dt: float,
    target_discrimination: float = DEFAULT_TARGET_DISCRIMINATION,
    window_factor: float = 4.0,
    faster_multiple: float = 2.0,
    tolerance: float = 1e-6,
) -> SigmaCalibration:
    """Solve for the sigma that hits ``target_discrimination``.

    The window is ``window_factor / omega_n`` -- four time constants, by which
    point a critically damped step response has essentially settled, so the
    integral covers the transient and not a long stretch of steady state that
    would dilute it.

    The "twice as fast" candidate gets ``faster_multiple`` times the bandwidth
    **and** the same multiple of rate limit.  Scaling both is deliberate: the
    peak rate of a critically damped step response is ``A * omega_n / e``, so
    leaving the limit alone would saturate the candidate and turn the comparison
    into one about saturation rather than about bandwidth, which is what sigma
    is supposed to discriminate.
    """
    if not amplitude > 0.0:
        raise ValueError(f"amplitude must be > 0, got {amplitude}")
    window_s = window_factor / omega_n
    steps = max(2, int(round(window_s / dt)))

    nominal = _trajectory(omega_n, rate_limit, amplitude, dt, steps)
    faster = _trajectory(
        omega_n * faster_multiple, rate_limit * faster_multiple, amplitude, dt, steps
    )
    gap = (faster - nominal).abs()

    # discrimination is strictly decreasing in sigma, so bisect.
    low, high = 1e-9, max(float(gap.max()) * 50.0, 1.0)
    if discrimination_for_sigma(gap, high) > target_discrimination:
        raise RuntimeError(
            f"channel {name!r}: even sigma={high:g} discriminates more than "
            f"{target_discrimination}; the two trajectories are implausibly far apart"
        )
    while high - low > tolerance:
        mid = 0.5 * (low + high)
        if discrimination_for_sigma(gap, mid) > target_discrimination:
            low = mid
        else:
            high = mid
    sigma = 0.5 * (low + high)

    return SigmaCalibration(
        name=name,
        sigma=sigma,
        amplitude=float(amplitude),
        window_s=window_s,
        discrimination=discrimination_for_sigma(gap, sigma),
        peak_gap=float(gap.max()),
    )


def calibrate_channels(
    channels: Sequence[ChannelSpec],
    amplitudes: Dict[str, float],
    dt: float,
    target_discrimination: float = DEFAULT_TARGET_DISCRIMINATION,
    **kwargs,
) -> Dict[str, SigmaCalibration]:
    """Calibrate every channel; ``amplitudes`` is a name-keyed step size."""
    missing = [c.name for c in channels if c.name not in amplitudes]
    if missing:
        raise ValueError(f"no calibration amplitude given for {missing}")
    return {
        c.name: calibrate_sigma(
            name=c.name,
            omega_n=c.omega_n,
            rate_limit=c.rate_limit,
            amplitude=amplitudes[c.name],
            dt=dt,
            target_discrimination=target_discrimination,
            **kwargs,
        )
        for c in channels
    }
