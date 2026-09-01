"""R7.1 -- the ``(g, l)`` response-deviation statistics.

Teacher-student was vetoed, so domain identification has to happen inside the
actor's processing of its observation history.  The observation that makes that
tractable: **the quantity the policy would have to identify can simply be
computed**, so there is no reason to make a network learn it.  Two slow
statistics per channel:

``g`` -- **how much gain this domain has**, as a deviation from 1.  A heavier
robot on slippery ground realises less body pitch per unit of pitch command;
that ratio is exactly the DC gain R4.3 exists to pin to 1, and it is the single
number an MPC would be wrong about if it assumed nominal dynamics.

``l`` -- **how much this domain lags**, signed.  The tracking error projected
onto the direction the reference is moving: positive when the body is
systematically behind the reference, negative when ahead.  This is the integral
term of a PID controller, made explicit.  "Produce the same response to the same
command under a different plant" is a problem classical control answers with
integral action; this hands the policy the same signal rather than hoping it
reconstructs one.

Marginal cost is close to zero -- R3 already maintains the low-passed
measurement for phase detrending, and the reference state is already there.

Two deviations from the formulas in the plan, both because the plan's version is
undefined on this data:

**Gain is a regression, not a ratio.**  The plan wrote
``EMA(y) / EMA(u)``.  Every decision channel is commanded symmetrically about
zero, so ``EMA(u)`` tends to zero over the 5 s window and the ratio is not
merely ill-conditioned, it is meaningless -- regularising the denominator turns
a meaningless number into a small meaningless number.  ``EMA(y*u) / EMA(u^2)``
is the least-squares gain, agrees with the ratio whenever a DC component exists,
and stays well defined when it does not.

**Lag accumulates only while the reference is actually moving.**  ``sign(xi_dot)``
is arbitrary in steady state, so an ungated EMA integrates noise for as long as
the command is held -- which, at a 10 s resample interval, is most of the time.

Like the rest of this package: pure tensors, no IsaacGym, testable on CPU.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch


class ResponseDeviationEstimator:
    """Per-env, per-channel ``(g - 1, l)``, updated once per control step.

    The time constant is deliberately long -- 5 s against an MPC horizon of
    about 1 s.  That is what lets the planner treat these as **parameters**
    rather than as states: something that moves 5x slower than the horizon is a
    constant over the horizon.  It is also why this is not the recurrent hidden
    state that was rejected: the dynamics are a fully specified EMA, the value
    is computed in the environment and *observed* rather than hidden, and it is
    reproducible on hardware from the same two inputs.
    """

    def __init__(
        self,
        num_envs: int,
        channel_index: Sequence[int],
        rate_limit: Sequence[float],
        command_scale: Sequence[float],
        dt: float,
        tau_s: float = 5.0,
        warmup_s: float = 2.0,
        rate_deadband: float = 0.05,
        excitation_fraction: float = 0.1,
        eps: float = 1e-4,
        device: str = "cpu",
    ) -> None:
        if tau_s <= 0.0:
            raise ValueError(f"tau_s must be > 0, got {tau_s}")
        if not 0.0 <= rate_deadband < 1.0:
            raise ValueError(f"rate_deadband must be in [0, 1), got {rate_deadband}")
        self.num_envs = int(num_envs)
        self.channel_index = torch.as_tensor(list(channel_index), dtype=torch.long, device=device)
        self.num_channels = int(self.channel_index.numel())
        self.dt = float(dt)
        self.eps = float(eps)
        # Deadband in absolute rate units, per selected channel: below this the
        # reference is not moving enough for the sign of its velocity to mean
        # anything.
        limits = torch.as_tensor(list(rate_limit), dtype=torch.float, device=device)
        self.rate_floor = (limits[self.channel_index] * float(rate_deadband)).unsqueeze(0)
        # Excitation floor on EMA(u^2).  Without it an UNCOMMANDED channel
        # reports gain 0, i.e. "this domain has no gain at all" -- a confident
        # wrong answer, which is precisely the failure the ratio form was
        # rejected for.  The channel has to have been driven at this fraction of
        # a representative step, in RMS, before a gain is claimed.
        scales = torch.as_tensor(list(command_scale), dtype=torch.float, device=device)
        self.excitation_floor = (
            (scales[self.channel_index] * float(excitation_fraction)) ** 2
        ).unsqueeze(0)
        self.alpha = 1.0 - math.exp(-self.dt / float(tau_s))
        self.warmup_steps = max(1, int(round(float(warmup_s) / self.dt)))
        self.device = device

        shape = (self.num_envs, self.num_channels)
        self.cross = torch.zeros(shape, device=device)      # EMA of y * u
        self.square = torch.zeros(shape, device=device)     # EMA of u * u
        self.lag = torch.zeros(shape, device=device)        # EMA of (y - xi) * sign(xi_dot)
        self.steps = torch.zeros(self.num_envs, device=device)
        self.lag_steps = torch.zeros(shape, device=device)

    # -- lifecycle ----------------------------------------------------------

    def reset(self, env_ids: torch.Tensor) -> None:
        """Clear on episode reset: the statistics describe a domain, and after a
        reset the robot's own state no longer matches what produced them."""
        if env_ids.numel() == 0:
            return
        self.cross[env_ids] = 0.0
        self.square[env_ids] = 0.0
        self.lag[env_ids] = 0.0
        self.steps[env_ids] = 0.0
        self.lag_steps[env_ids] = 0.0

    def update(
        self,
        measured: torch.Tensor,
        command: torch.Tensor,
        reference: torch.Tensor,
        reference_rate: torch.Tensor,
    ) -> None:
        """Advance both statistics by one control step.

        All four arguments are ``(E, C_all)`` in response-channel order; the
        configured subset is selected here so callers never have to know which
        channels are observed.
        """
        index = self.channel_index
        y = measured[:, index]
        u = command[:, index]
        xi = reference[:, index]
        xi_dot = reference_rate[:, index]

        self.cross += self.alpha * (y * u - self.cross)
        self.square += self.alpha * (u * u - self.square)

        moving = xi_dot.abs() > self.rate_floor
        increment = (y - xi) * torch.sign(xi_dot)
        # Advance the lag EMA only on informative samples, and hold it
        # otherwise -- an ungated EMA would decay a real lag estimate towards
        # zero through every held command, which is most of an episode.
        self.lag = torch.where(moving, self.lag + self.alpha * (increment - self.lag), self.lag)
        self.lag_steps += moving.float()
        self.steps += 1.0

    # -- read-out -----------------------------------------------------------

    @property
    def converged(self) -> torch.Tensor:
        """``(E,)`` bool: has the slow EMA seen enough of an episode to mean
        anything?  Same idea as R3's visit-count gate."""
        return self.steps >= self.warmup_steps

    def observation(self) -> torch.Tensor:
        """``(E, 2 * C)`` interleaved ``[g - 1, l]`` per observed channel.

        ``g`` is reported as a **deviation from 1** so that zero is the nominal
        value.  That is what makes the warm-up answer honest: before the EMA has
        converged the estimator emits zeros, which the policy reads as "nominal,
        no information yet" rather than as a confident wrong number.  It also
        removes the need for a separate warm-up flag in the observation vector.
        """
        gain = self.cross / (self.square + self.eps)
        # Each half carries its own "have I seen enough to answer" gate, and the
        # gates are different quantities: a gain needs the channel to have been
        # COMMANDED, a lag needs the reference to have MOVED.  A channel held at
        # zero satisfies neither, and both then report 0 == nominal.
        excited = (self.square > self.excitation_floor).float()
        moved = (self.lag_steps >= 1.0).float()
        deviation = torch.stack(((gain - 1.0) * excited, self.lag * moved), dim=-1)
        deviation = deviation * self.converged.view(-1, 1, 1).float()
        return deviation.reshape(self.num_envs, 2 * self.num_channels)
