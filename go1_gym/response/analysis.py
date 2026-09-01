"""R9.1 -- the numerics behind the evaluation figures.

The central claim this file has to support:

    a response-consistent policy can be described by a **markedly simpler**
    model to the same accuracy, without losing robustness.

"Markedly simpler" is only meaningful if model order is measured the same way
for every policy being compared, so the fitting lives here rather than inside a
plotting script: one implementation, unit-tested against signals whose true
order is known.

Three model orders, matching R9.1's list:

``first``            ``tau`` only -- a lag.
``second``           critically damped ``omega_n`` -- the reference model's form.
``second+residual``  the same, plus the gait-phase-conditioned residual R3
                     estimates.  This is the honest ceiling: the residual is
                     periodic and predictable, so a planner that knows the gait
                     phase can feed it forward, and charging the model for it
                     would understate how well the closed loop can be predicted.

Everything is pure torch on CPU-sized arrays -- no simulator, no plotting -- so
the numbers in the paper come from tested code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from .reference import ChannelSpec, ReferenceModel


@dataclass(frozen=True)
class ModelFit:
    """One fitted model."""

    order: str
    #: Fitted parameter: ``tau`` for first order, ``omega_n`` for second.
    parameter: float
    #: Fraction of the signal's variance the model explains.
    r_squared: float
    #: Residual energy over signal energy.  Reported alongside R^2 because they
    #: answer different questions: R^2 is about variance about the mean, this is
    #: about absolute magnitude, and a model can look good on one and poor on
    #: the other when the signal has a large DC component.
    normalized_residual_energy: float


def _r_squared(y: torch.Tensor, prediction: torch.Tensor) -> float:
    residual = y - prediction
    total = ((y - y.mean()) ** 2).sum()
    if float(total) <= 0.0:
        return float("nan")
    return float(1.0 - (residual ** 2).sum() / total)


def _normalized_residual_energy(y: torch.Tensor, prediction: torch.Tensor) -> float:
    energy = (y ** 2).sum()
    if float(energy) <= 0.0:
        return float("nan")
    return float(((y - prediction) ** 2).sum() / energy)


def simulate_first_order(u: torch.Tensor, dt: float, tau: float,
                         initial: Optional[float] = None) -> torch.Tensor:
    """``y[k+1] = a y[k] + (1 - a) u[k]`` with ``a = exp(-dt / tau)``."""
    if tau <= 0.0:
        raise ValueError(f"tau must be > 0, got {tau}")
    a = math.exp(-float(dt) / float(tau))
    y = torch.empty_like(u)
    state = float(u[0]) if initial is None else float(initial)
    for k in range(u.numel()):
        y[k] = state
        state = a * state + (1.0 - a) * float(u[k])
    return y


def simulate_second_order(u: torch.Tensor, dt: float, omega_n: float,
                          initial: Optional[float] = None) -> torch.Tensor:
    """Critically damped second order, using the env's own integrator.

    Reusing :class:`ReferenceModel` rather than re-deriving the recurrence is
    deliberate: if the fit and the trained-against reference ever disagreed, the
    model-order figure would be measuring that disagreement.
    """
    if omega_n <= 0.0:
        raise ValueError(f"omega_n must be > 0, got {omega_n}")
    model = ReferenceModel(
        # No rate limit while fitting: the fit is asking "what bandwidth
        # describes this signal", and a saturating reference would fold a
        # second, unfitted parameter into the answer.
        [ChannelSpec(name="c", cmd_index=0, omega_n=float(omega_n), rate_limit=1e9)],
        num_envs=1,
        dt=float(dt),
        dtype=torch.float64,
    )
    start = float(u[0]) if initial is None else float(initial)
    model.xi[:] = start
    command = torch.zeros(1, 1, dtype=torch.float64)
    y = torch.empty_like(u)
    for k in range(u.numel()):
        y[k] = model.xi[0, 0]
        command[0, 0] = float(u[k])
        model.step(command)
    return y


def _golden_section(objective, low: float, high: float, tolerance: float = 1e-4) -> float:
    """Minimise a unimodal objective on ``[low, high]`` without scipy."""
    invphi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = low, high
    c, d = b - invphi * (b - a), a + invphi * (b - a)
    fc, fd = objective(c), objective(d)
    while b - a > tolerance:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = objective(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = objective(d)
    return 0.5 * (a + b)


def fit_first_order(u: torch.Tensor, y: torch.Tensor, dt: float,
                    bounds: Tuple[float, float] = (0.01, 5.0)) -> ModelFit:
    u = u.double()
    y = y.double()

    def cost(tau):
        return float(((y - simulate_first_order(u, dt, tau)) ** 2).sum())

    tau = _golden_section(cost, *bounds)
    prediction = simulate_first_order(u, dt, tau)
    return ModelFit("first", tau, _r_squared(y, prediction),
                    _normalized_residual_energy(y, prediction))


def fit_second_order(u: torch.Tensor, y: torch.Tensor, dt: float,
                     bounds: Tuple[float, float] = (0.5, 40.0)) -> ModelFit:
    u = u.double()
    y = y.double()

    def cost(omega):
        return float(((y - simulate_second_order(u, dt, omega)) ** 2).sum())

    omega = _golden_section(cost, *bounds)
    prediction = simulate_second_order(u, dt, omega)
    return ModelFit("second", omega, _r_squared(y, prediction),
                    _normalized_residual_energy(y, prediction))


def phase_conditioned_mean(values: torch.Tensor, phase: torch.Tensor,
                           num_bins: int = 16) -> torch.Tensor:
    """Mean of ``values`` in each gait-phase bin, returned per sample."""
    index = torch.clamp((phase * num_bins).long(), 0, num_bins - 1)
    totals = torch.zeros(num_bins, dtype=values.dtype)
    counts = torch.zeros(num_bins, dtype=values.dtype)
    totals.index_add_(0, index, values)
    counts.index_add_(0, index, torch.ones_like(values))
    means = totals / torch.clamp(counts, min=1.0)
    return means[index]


def fit_second_order_with_residual(
    u: torch.Tensor, y: torch.Tensor, phase: torch.Tensor, dt: float,
    num_bins: int = 16, bounds: Tuple[float, float] = (0.5, 40.0)
) -> ModelFit:
    """Second order plus the gait-phase-conditioned residual.

    The residual is fitted **after** the dynamics rather than jointly, which
    matches how it is used: R3 estimates it from the realised motion, and the
    planner adds it as a known periodic term.  Fitting them jointly would let
    the residual absorb bandwidth error and flatter the model order.
    """
    u = u.double()
    y = y.double()
    base = fit_second_order(u, y, dt, bounds)
    prediction = simulate_second_order(u, dt, base.parameter)
    residual = phase_conditioned_mean(y - prediction, phase.double(), num_bins)
    corrected = prediction + residual
    return ModelFit("second+residual", base.parameter, _r_squared(y, corrected),
                    _normalized_residual_energy(y, corrected))


def model_order_curve(u: torch.Tensor, y: torch.Tensor, phase: torch.Tensor,
                      dt: float, num_bins: int = 16) -> Dict[str, ModelFit]:
    """R9.1 metric 4: goodness of fit against model order.

    Half of the paper's main figure -- the other half is a robustness metric on
    the same axes.  The claim is only interesting if both are plotted for the
    same policies: a simpler model at equal accuracy means nothing if
    robustness fell to get it.
    """
    return {
        "first": fit_first_order(u, y, dt),
        "second": fit_second_order(u, y, dt),
        "second+residual": fit_second_order_with_residual(u, y, phase, dt, num_bins),
    }


# ---------------------------------------------------------------------------
# Frequency response
# ---------------------------------------------------------------------------


def empirical_bode(
    u: torch.Tensor, y: torch.Tensor, dt: float,
    band: Tuple[float, float] = (0.1, 2.0),
    energy_floor: float = 0.02,
) -> Dict[str, torch.Tensor]:
    """Closed-loop frequency response from a chirp record.

    ``H(f) = Y(f) / U(f)``, evaluated only where the input actually carries
    energy.  The floor is not tidiness: outside the swept band ``U(f)`` is
    numerically tiny, and dividing by it produces enormous gains that look like
    resonance and are pure noise.  R6's chirp is amplitude-tapered to hold slew
    rate constant, so the input spectrum is deliberately *not* flat and this
    gate matters more here than for a textbook sweep.
    """
    u = u.double() - u.double().mean()
    y = y.double() - y.double().mean()
    spectrum_u = torch.fft.rfft(u)
    spectrum_y = torch.fft.rfft(y)
    freqs = torch.fft.rfftfreq(u.numel(), d=float(dt))

    magnitude = spectrum_u.abs()
    keep = magnitude > energy_floor * float(magnitude.max())
    keep &= (freqs >= band[0]) & (freqs <= band[1])

    response = spectrum_y[keep] / spectrum_u[keep]
    return {
        "frequency_hz": freqs[keep],
        "gain_db": 20.0 * torch.log10(response.abs().clamp(min=1e-12)),
        "phase_deg": torch.angle(response) * 180.0 / math.pi,
    }


# ---------------------------------------------------------------------------
# Cross-domain dispersion
# ---------------------------------------------------------------------------


def phase_conditioned_dispersion(
    values: torch.Tensor, phase: torch.Tensor, num_bins: int = 16
) -> float:
    """R9.1 metric 2: ``E_phi[Var_domain(y | phase)]``.

    ``values`` is ``(domains, samples)`` and ``phase`` is ``(samples,)`` -- the
    domains share a phase clock, which is exactly what R5's grouping and phase
    synchronisation exist to guarantee.  Without that shared clock this number
    would measure phase misalignment rather than cross-domain disagreement.
    """
    if values.ndim != 2:
        raise ValueError(f"values must be (domains, samples), got {tuple(values.shape)}")
    if values.shape[0] < 2:
        raise ValueError("cross-domain variance needs at least two domains")
    index = torch.clamp((phase.double() * num_bins).long(), 0, num_bins - 1)
    variances, weights = [], []
    for b in range(num_bins):
        selected = values[:, index == b]
        if selected.shape[1] == 0:
            continue
        # variance across domains at each sample, then averaged within the bin
        variances.append(float(selected.double().var(dim=0, unbiased=False).mean()))
        weights.append(selected.shape[1])
    if not variances:
        return float("nan")
    total = float(sum(weights))
    return float(sum(v * w for v, w in zip(variances, weights)) / total)


def residual_fourier_coefficients(
    residual: torch.Tensor, phase: torch.Tensor, harmonics: int = 3
) -> torch.Tensor:
    """Complex Fourier coefficients of a phase-indexed residual, n = 1..harmonics.

    R9.1 metric 3 asks for the **cross-domain variance** of these, so they are
    returned per input row rather than reduced here.
    """
    if residual.ndim == 1:
        residual = residual.unsqueeze(0)
    angle = 2.0 * math.pi * phase.double()
    coefficients = []
    for n in range(1, harmonics + 1):
        basis = torch.exp(-1j * n * angle)
        coefficients.append((residual.double() * basis).mean(dim=-1))
    return torch.stack(coefficients, dim=-1)
