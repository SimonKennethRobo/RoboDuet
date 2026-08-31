"""Gait-phase-conditioned residual estimate and zero-delay detrending (R3).

During continuous locomotion the base is not a smooth platform: even with the
average velocity exactly right, trot produces a periodic z / roll / pitch
oscillation.  This module estimates that oscillation online, indexed by
``(speed bin, gait phase bin, channel)``, per environment.

It has two consumers, and R3 requires their data paths to stay separate:

* **R4.1** needs the *low-frequency* part of the measurement with **no delay**,
  so that a command transient is not smeared.  It gets it by subtracting the
  phase-indexed table: ``y - delta_hat(phase)``.  This is delay-free because the
  gait phase is a commanded, open-loop clock (``gait_indices`` is integrated
  from the gait-frequency command), so "where in the cycle am I" is known
  exactly rather than estimated from contact.
* **R4.2** needs a *target* for the oscillation, i.e. ``delta_hat`` itself.

The estimate is maintained with a lagging low-pass (``y_lp``).  That lag is
fine here -- it only slows how fast the statistic converges -- and it must not
leak into the reward path, which is the whole point of keeping the two apart:
an EMA has roughly half a gait period of phase lag, and feeding that into a
transient-tracking reward would badly distort the error signal.

Invariants this module exists to enforce (R3):

* per-environment storage -- different envs have different domains, so their
  oscillation genuinely differs; cross-env consistency is R5's job, not this
  module's;
* binning by speed -- the amplitude and shape of the oscillation change a lot
  with travel speed, and forcing one shape across all speeds is physically
  unrealisable, which would degrade into a constant negative reward floor;
* the update path and the use path are different code paths and are never
  allowed to become the same one.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch


class PhaseResidualEstimator:
    """Running estimate of ``delta_hat[env, speed_bin, phase_bin, channel]``.

    Storage is ``num_envs x num_speed_bins x num_phase_bins x num_channels``
    float32 -- 5.2 MB at 4096 envs with the default 4 x 16 x 5 layout.
    """

    def __init__(
        self,
        num_envs: int,
        num_channels: int,
        dt: float,
        speed_bin_edges: Sequence[float] = (0.15, 0.35, 0.6),
        num_phase_bins: int = 16,
        lowpass_tau_s: float = 0.5,
        estimate_cycles: float = 15.0,
        min_cycles: float = 45.0,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if num_phase_bins < 2:
            raise ValueError(f"num_phase_bins must be >= 2, got {num_phase_bins}")
        edges = list(speed_bin_edges)
        if any(b <= a for a, b in zip(edges, edges[1:])):
            raise ValueError(f"speed_bin_edges must be strictly increasing, got {edges}")
        if not lowpass_tau_s > 0.0:
            raise ValueError(f"lowpass_tau_s must be > 0, got {lowpass_tau_s}")
        if not estimate_cycles > 0.0:
            raise ValueError(f"estimate_cycles must be > 0, got {estimate_cycles}")

        self.num_envs = int(num_envs)
        self.num_channels = int(num_channels)
        self.dt = float(dt)
        self.num_phase_bins = int(num_phase_bins)
        self.num_speed_bins = len(edges) + 1
        self.device = torch.device(device)
        self.dtype = dtype

        self.speed_bin_edges = torch.tensor(edges, device=self.device, dtype=self.dtype)

        # Low-pass coefficient is per *step*; the delta_hat coefficient is per
        # *visit to a bin*, and a bin is visited about once per gait cycle.
        # Mixing the two units is an easy way to make the estimator either
        # far too slow or effectively instantaneous.
        self.lowpass_alpha = self.dt / lowpass_tau_s
        self.estimate_alpha = 1.0 - math.exp(-1.0 / float(estimate_cycles))
        self.min_samples = float(min_cycles)

        self.y_lp = torch.zeros(
            (self.num_envs, self.num_channels), device=self.device, dtype=self.dtype
        )
        self.delta_hat = torch.zeros(
            (self.num_envs, self.num_speed_bins, self.num_phase_bins, self.num_channels),
            device=self.device,
            dtype=self.dtype,
        )
        self.sample_count = torch.zeros(
            (self.num_envs, self.num_speed_bins, self.num_phase_bins),
            device=self.device,
            dtype=self.dtype,
        )
        self._env_index = torch.arange(self.num_envs, device=self.device)

    # ------------------------------------------------------------------ bins

    def speed_bin(self, speed: torch.Tensor) -> torch.Tensor:
        """Bucket a per-env scalar speed into ``[-inf, e0), [e0, e1), ..., [e_last, inf)``.

        Commanded speed, not measured.

        Using the command keeps the estimator out of a feedback loop with the
        very quantity it is detrending, and it is the value the planner knows.
        """
        # right=True gives half-open bins [-inf, e0), [e0, e1), ..., [e_last, inf),
        # i.e. a value sitting exactly on an edge belongs to the upper bin.
        # torch's default (right=False) is the other convention and would put it
        # in the lower one.
        return torch.bucketize(speed.detach(), self.speed_bin_edges, right=True)

    def phase_bin(self, phase: torch.Tensor) -> torch.Tensor:
        """Bucket a normalised gait phase in [0, 1)."""
        wrapped = torch.remainder(phase.detach(), 1.0)
        index = (wrapped * self.num_phase_bins).long()
        return torch.clamp(index, 0, self.num_phase_bins - 1)

    def _flat_index(self, speed_bin: torch.Tensor, phase_bin: torch.Tensor) -> torch.Tensor:
        return speed_bin * self.num_phase_bins + phase_bin

    # ------------------------------------------------------------- use path

    def lookup(self, speed_bin: torch.Tensor, phase_bin: torch.Tensor) -> torch.Tensor:
        """``delta_hat`` at the current bin.  Read-only -- no state changes."""
        flat = self.delta_hat.view(self.num_envs, -1, self.num_channels)
        return flat[self._env_index, self._flat_index(speed_bin, phase_bin)]

    def detrend(
        self, measured: torch.Tensor, speed_bin: torch.Tensor, phase_bin: torch.Tensor
    ) -> torch.Tensor:
        """Zero-delay low-frequency component: ``y - delta_hat(phase)``.

        This is the quantity every R4 tracking term must be computed on.  It is
        delay-free because it subtracts a phase-indexed table rather than
        filtering the signal in time.
        """
        return measured - self.lookup(speed_bin, phase_bin)

    def is_converged(self, speed_bin: torch.Tensor, phase_bin: torch.Tensor) -> torch.Tensor:
        """Whether the current bin has seen enough samples to be trusted.

        R3: rewards that depend on this estimate must stay off until it has
        converged, otherwise they are a constant negative floor early in
        training.
        """
        flat = self.sample_count.view(self.num_envs, -1)
        return flat[self._env_index, self._flat_index(speed_bin, phase_bin)] >= self.min_samples

    # ---------------------------------------------------------- update path

    def update(
        self,
        measured: torch.Tensor,
        speed_bin: torch.Tensor,
        phase_bin: torch.Tensor,
        active: Optional[torch.Tensor] = None,
    ) -> None:
        """Advance the estimate by one step.

        ``active`` marks environments whose sample is meaningful.  Where it is
        false the low-pass is *snapped* to the measurement and ``delta_hat`` is
        left alone.  Two situations need that:

        * shortly after a reset, where the low-pass still holds the previous
          episode's level and the robot is settling;
        * while standing, where ``_step_contact_targets`` forces the gait
          frequency to zero -- the phase clock stops, so every sample would
          pile into whichever bin the env happened to freeze in and corrupt it.
        """
        measured = measured.to(self.dtype)
        if active is None:
            active = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        active_c = active.unsqueeze(-1)

        # Lagging low-pass; the lag is deliberate and confined to this path.
        stepped = self.y_lp + (measured - self.y_lp) * self.lowpass_alpha
        self.y_lp = torch.where(active_c, stepped, measured)

        residual = measured - self.y_lp

        flat_delta = self.delta_hat.view(self.num_envs, -1, self.num_channels)
        flat_count = self.sample_count.view(self.num_envs, -1)
        index = self._flat_index(speed_bin, phase_bin)

        current = flat_delta[self._env_index, index]
        blended = current + self.estimate_alpha * (residual - current)
        flat_delta[self._env_index, index] = torch.where(active_c, blended, current)
        flat_count[self._env_index, index] = flat_count[self._env_index, index] + active.to(self.dtype)

    # ------------------------------------------------------------------ misc

    def reset(self, env_ids: torch.Tensor) -> None:
        """Drop everything learned for these environments.

        Deliberately NOT called on episode reset.  A phase bin is visited about
        once per gait cycle, so the estimate needs tens of seconds of walking to
        converge -- far longer than one episode.  Clearing per episode would
        mean it never converges at all.  The estimate is meant to average over
        the per-episode motor randomisation while tracking the env-persistent
        domain (mass, geometry, friction, mount), and its time constant is
        chosen for exactly that.  This entry point exists for evaluation code
        that wants a clean slate.
        """
        if env_ids.numel() == 0:
            return
        self.y_lp[env_ids] = 0.0
        self.delta_hat[env_ids] = 0.0
        self.sample_count[env_ids] = 0.0
