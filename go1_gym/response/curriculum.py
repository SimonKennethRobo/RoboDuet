"""R8.1 -- the four-stage curriculum, as a weight schedule.

R8 exists because the consistency terms cannot simply be switched on.  Measured
(see the R0 plan): the phase-variance term is 95x larger for a from-scratch
policy than for a trained one, so its end-of-ramp weight applied at iteration 0
multiplies the whole reward by about 1e-3 -- no NaN, no divergence, just a
constant negative floor that never recovers.  R5's term has a second problem on
top: it is masked off while a group is desynchronised, which early in training
is essentially always, so an early weight is uninformative as well as harmful.

    stage 1  original rewards only; R3's estimator updates but feeds nothing
    stage 2  R4.1 ramps in -- the policy learns to track the reference
    stage 3  R4.2 + R4.3 + R5 ramp in -- the consistency layer
    stage 4  weights frozen, disturbances raised -- robustness recovery

**Why R4.1 is gated to stage 2 rather than left on from the start.**  It was on
from iteration 0 through steps 3-7 of this work and trained fine, so this is not
a stability fix -- it is a circularity fix.  R8.2 calibrates the reference model
from "the stage-1 purely-robust checkpoint", and a policy already shaped by
reference tracking is no longer a neutral measurement of what the hardware can
do: it has been pulled towards the very reference model the calibration is
supposed to derive.  Stage 1 has to be clean for stage 2's target to mean
anything.

The schedule is expressed as a **multiplier on the configured weight**, never as
the weight itself.  The configured value is the end-of-ramp target, which keeps
one number in one place -- and, more importantly, keeps the term *registered*.
``LeggedRobot._prepare_reward_function`` drops zero-scale terms before
registering them, so a term shipped at 0 could never be ramped up later: the
function would not exist to call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

import torch


class ResponseCurriculum:
    """Maps a training iteration to a multiplier per reward term.

    Deliberately not a state machine over measured exit criteria.  R8 lists
    both iteration counts and outcome-based exits ("R4.1 reward > 0.8",
    "consistency metrics acceptable"); this implements the iteration schedule
    and reports the metrics an operator needs to check the outcome criteria
    against.  An automatic exit on a measured threshold would silently advance
    on a noisy estimate -- and the 200-iteration reward spread measured on this
    setup is 11.6x across seeds, which is far too noisy to gate on.
    """

    def __init__(
        self,
        stage_boundaries: Sequence[int],
        ramp_iterations: int,
        term_stage: Mapping[str, int],
        term_handover: Optional[Mapping[str, int]] = None,
        handover_floor: float = 0.15,
        enabled: bool = True,
        randomization_stage: int = 3,
        randomization_floor: float = 0.3,
        disturbance_stage: int = 4,
    ) -> None:
        boundaries = [int(b) for b in stage_boundaries]
        if sorted(boundaries) != boundaries:
            raise ValueError(f"stage_boundaries must be ascending, got {boundaries}")
        if any(b < 0 for b in boundaries):
            raise ValueError(f"stage_boundaries must be non-negative, got {boundaries}")
        if int(ramp_iterations) < 0:
            raise ValueError(f"ramp_iterations must be >= 0, got {ramp_iterations}")
        self.stage_boundaries = boundaries
        self.num_stages = len(boundaries) + 1
        self.ramp_iterations = int(ramp_iterations)
        bad = {name: stage for name, stage in term_stage.items()
               if not 1 <= int(stage) <= self.num_stages}
        if bad:
            raise ValueError(
                f"term_stage entries outside 1..{self.num_stages}: {bad}"
            )
        self.term_stage = {name: int(stage) for name, stage in term_stage.items()}
        handover = dict(term_handover or {})
        bad = {name: stage for name, stage in handover.items()
               if not 1 <= int(stage) <= self.num_stages}
        if bad:
            raise ValueError(f"term_handover entries outside 1..{self.num_stages}: {bad}")
        overlap = set(handover) & set(self.term_stage)
        if overlap:
            raise ValueError(
                f"a term cannot both ramp up and hand over: {sorted(overlap)}"
            )
        if not 0.0 <= handover_floor <= 1.0:
            raise ValueError(f"handover_floor must be in [0, 1], got {handover_floor}")
        self.term_handover = {name: int(stage) for name, stage in handover.items()}
        self.handover_floor = float(handover_floor)
        self.enabled = bool(enabled)
        for label, stage in (("randomization_stage", randomization_stage),
                             ("disturbance_stage", disturbance_stage)):
            if not 1 <= int(stage) <= self.num_stages:
                raise ValueError(
                    f"{label} must be in 1..{self.num_stages}, got {stage}"
                )
        if not 0.0 <= randomization_floor <= 1.0:
            raise ValueError(
                f"randomization_floor must be in [0, 1], got {randomization_floor}"
            )
        self.randomization_stage = int(randomization_stage)
        self.randomization_floor = float(randomization_floor)
        self.disturbance_stage = int(disturbance_stage)

    # -- schedule -----------------------------------------------------------

    def stage(self, iteration: int) -> int:
        """1-based stage index at this iteration.

        With the curriculum disabled everything stays in stage 1.  That is the
        safe direction: disabling it must not hand every consistency term its
        full weight at iteration 0, which is the one configuration measured to
        destroy the reward.
        """
        if not self.enabled:
            return 1
        stage = 1
        for boundary in self.stage_boundaries:
            if iteration >= boundary:
                stage += 1
        return stage

    def stage_start(self, stage: int) -> int:
        """First iteration of a stage."""
        if stage <= 1:
            return 0
        return self.stage_boundaries[stage - 2]

    def multiplier(self, iteration: int) -> Dict[str, float]:
        """Per-term multiplier in ``[0, 1]``, linear over the ramp.

        Two kinds of term.  ``term_stage`` entries ramp UP from 0 as their stage
        begins.  ``term_handover`` entries ramp DOWN from 1 to
        ``handover_floor`` over the same window: they are the instantaneous
        posture penalties R4.1 replaces, and the handover has to be a crossfade
        rather than a step, or there is an interval -- the whole of stage 1, as
        it turned out -- with neither at full strength and nothing holding the
        body's posture.
        """
        result = {}
        for name, stage in self.term_stage.items():
            result[name] = self._ramp(iteration, stage)
        for name, stage in self.term_handover.items():
            result[name] = 1.0 - (1.0 - self.handover_floor) * self._ramp(iteration, stage)
        return result

    def _ramp(self, iteration: int, stage: int) -> float:
        """Fraction of the way through ``stage``'s ramp, in ``[0, 1]``."""
        start = self.stage_start(stage)
        if not self.enabled or iteration < start:
            return 0.0
        if self.ramp_iterations == 0:
            return 1.0
        return min(1.0, (iteration - start) / self.ramp_iterations)

    def randomization_intensity(self, iteration: int) -> float:
        """How far the domain ranges are opened, from nominal to full.

        R8.1 asks for weak randomisation in stages 1-2 and full randomisation
        from stage 3, which is the stage that first asks for cross-domain
        consistency.  The order matters: a policy that cannot yet walk on ice
        learns nothing from being asked to walk on ice *consistently*.

        Never zero.  A floor keeps some domain spread from the start, because a
        policy trained on exactly one domain and then handed the full range at
        stage 3 has to relearn locomotion at the same moment it is first asked
        for consistency -- which is the collapse R8 already warns stage 3 is
        prone to.
        """
        span = 1.0 - self.randomization_floor
        return self.randomization_floor + span * self._ramp(iteration, self.randomization_stage)

    def disturbance_intensity(self, iteration: int) -> float:
        """Stage 4's added push disturbance, ``0`` before that stage.

        Stage 4 exists to show that robustness comes back after the consistency
        weights are frozen, so its disturbance must be off while those weights
        are still moving -- otherwise a robustness change and a weight change
        land at the same time and neither can be attributed.
        """
        return self._ramp(iteration, self.disturbance_stage)

    def apply(
        self, reward_scales: Mapping[str, float], iteration: int
    ) -> Dict[str, float]:
        """Scale the terms this curriculum owns; pass everything else through.

        Returns a new dict -- ``global_switch.get_reward_scales()`` may hand
        back the very dict the trainer holds, and scaling it in place would
        compound the multiplier once per control step.
        """
        scaled = dict(reward_scales)
        for name, factor in self.multiplier(iteration).items():
            if name in scaled:
                scaled[name] = scaled[name] * factor
        return scaled

    # -- reporting ----------------------------------------------------------

    def report(self, iteration: int) -> Dict[str, float]:
        """Flat metrics for the training log."""
        metrics = {
            "curriculum_stage": float(self.stage(iteration)),
            "curriculum_randomization": self.randomization_intensity(iteration),
            "curriculum_disturbance": self.disturbance_intensity(iteration),
        }
        for name, factor in self.multiplier(iteration).items():
            metrics[f"curriculum_weight_{name}"] = float(factor)
        return metrics

    def checkpoint_iterations(self) -> Dict[str, int]:
        """Iterations R8 asks for a checkpoint at.

        The end of stage 3 and the end of stage 4 are two points on the
        robustness/predictability Pareto front, and R8 requires reporting both
        -- stage 4's existence is the evidence for the claim that consistency is
        a soft objective rather than the primary one.
        """
        marks = {}
        if self.num_stages >= 4:
            marks["stage3_end"] = self.stage_boundaries[2]
        if self.num_stages >= 3:
            marks["stage2_end"] = self.stage_boundaries[1]
        if self.num_stages >= 2:
            marks["stage1_end"] = self.stage_boundaries[0]
        return marks


def dominant_frequency(series: Sequence[float], dt: float) -> Optional[float]:
    """Frequency carrying the most power in ``series``, excluding DC.

    R8.2 names a specific diagnostic: if the posture channels were calibrated
    without binning by gait phase, the training reward develops **a periodic
    ripple at the gait frequency**.  That is a distinctive signature and a cheap
    one to watch for, so it is computed rather than left to be noticed by eye.
    """
    values = torch.as_tensor(list(series), dtype=torch.float64)
    if values.numel() < 8:
        return None
    values = values - values.mean()
    if float(values.abs().max()) == 0.0:
        return None
    spectrum = torch.fft.rfft(values).abs()
    freqs = torch.fft.rfftfreq(values.numel(), d=dt)
    spectrum[0] = 0.0
    return float(freqs[int(spectrum.argmax())])


def gait_frequency_ripple(
    series: Sequence[float], dt: float, gait_frequency: float, tolerance: float = 0.15
) -> bool:
    """True when the reward series ripples at the gait frequency.

    ``dt`` is the spacing of ``series``, which is per **iteration**, not per
    control step.  Watch it on a per-iteration reward log and the observable
    frequency band is set by that spacing, not by the simulator's.
    """
    peak = dominant_frequency(series, dt)
    if peak is None or gait_frequency <= 0.0:
        return False
    return abs(peak - gait_frequency) <= tolerance * gait_frequency


@dataclass(frozen=True)
class RippleReport:
    """What :func:`gait_frequency_ripple_batch` found, ready for a logger.

    ``fraction`` is the share of eligible environments whose reward series peaks
    inside the gait-frequency band; ``band_power`` is the mean share of their
    AC power that sits in that band.  Both are reported because they fail
    differently: ``fraction`` is a hard vote that says nothing about how strong
    the peak is, ``band_power`` is continuous but never reaches 0 (a band always
    contains some broadband power).  Read the first for "is this happening", the
    second for "is it getting worse".  ``count`` is how many environments were
    eligible at all -- a fraction over three environments is not a measurement.
    """

    fraction: float
    band_power: float
    count: int


def gait_frequency_ripple_batch(
    series: torch.Tensor,
    dt: float,
    gait_frequency: torch.Tensor,
    eligible: Optional[torch.Tensor] = None,
    tolerance: float = 0.15,
) -> RippleReport:
    """R8.2's ripple diagnostic, measured where the ripple actually lives.

    ``series`` is ``(T, E)``: the raw R4.1 term sampled **per control step, per
    environment**, oldest row first.  ``gait_frequency`` is ``(E,)`` in Hz.

    The per-iteration reward curve -- what :func:`gait_frequency_ripple` above
    reads, and what the R0 plan originally proposed watching -- cannot show this
    signal.  One logged point is a mean over thousands of environments and tens
    of steps, and the environments are not phase-locked to each other, so a
    per-environment ripple at 3 Hz averages away twice over before it is ever
    written down.  The failure R8.2 warns about is per environment and phase
    locked to *its own* gait clock, so that is the axis it has to be measured
    on; each environment is compared against its own commanded frequency rather
    than a population constant, which also covers the 2.5-3.5 Hz band R1 samples
    the frequency from.

    The band half-width is ``tolerance * f``, but never narrower than one FFT
    bin: at a 128-step window and dt = 0.02 s a bin is 0.39 Hz while 15% of
    2.5 Hz is 0.375 Hz, so without the floor the band could fall between bins
    and the check would report "clean" no matter what the policy did.
    """
    if series.ndim != 2:
        raise ValueError(f"series must be (T, E), got shape {tuple(series.shape)}")
    steps = int(series.shape[0])
    if steps < 8:
        return RippleReport(0.0, 0.0, 0)

    values = series.to(torch.float32)
    freq = gait_frequency.to(device=values.device, dtype=values.dtype).reshape(-1)
    if eligible is None:
        usable = torch.ones(values.shape[1], dtype=torch.bool, device=values.device)
    else:
        usable = eligible.to(device=values.device, dtype=torch.bool).reshape(-1)

    spectrum = torch.fft.rfft(values - values.mean(dim=0, keepdim=True), dim=0).abs()
    spectrum[0] = 0.0
    power = spectrum ** 2
    total = power.sum(dim=0)

    usable = usable & (freq > 0.0) & (total > 0.0)
    count = int(usable.sum())
    if count == 0:
        return RippleReport(0.0, 0.0, 0)

    freqs = torch.fft.rfftfreq(steps, d=dt).to(device=values.device, dtype=values.dtype)
    half_width = torch.clamp(tolerance * freq, min=1.0 / (steps * dt))
    in_band = (freqs.unsqueeze(1) - freq.unsqueeze(0)).abs() <= half_width.unsqueeze(0)
    peak_is_gait = (freqs[power.argmax(dim=0)] - freq).abs() <= half_width
    band_share = (power * in_band).sum(dim=0) / torch.clamp(total, min=1e-30)

    return RippleReport(
        fraction=float(peak_is_gait[usable].to(values.dtype).mean()),
        band_power=float(band_share[usable].mean()),
        count=count,
    )
