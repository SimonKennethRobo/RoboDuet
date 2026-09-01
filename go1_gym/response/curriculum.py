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

from typing import Dict, Mapping, Optional, Sequence


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
        enabled: bool = True,
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
        self.enabled = bool(enabled)

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
        """Per-term multiplier in ``[0, 1]``, linear over the ramp."""
        result = {}
        for name, stage in self.term_stage.items():
            start = self.stage_start(stage)
            if not self.enabled or iteration < start:
                result[name] = 0.0
            elif self.ramp_iterations == 0:
                result[name] = 1.0
            else:
                result[name] = min(1.0, (iteration - start) / self.ramp_iterations)
        return result

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
        metrics = {"curriculum_stage": float(self.stage(iteration))}
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
    import torch

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
