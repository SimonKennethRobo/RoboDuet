"""R6 -- rich excitation command signals for the identification environments.

The problem R6 exists to solve: the original command resampling period is 10 s
against a 20 s episode, so a single episode contains **one** command step.  A
reference model that describes the transient is then fitted to almost no
transient data, and R8's calibration and R9's identification have nothing to
work with.  A fraction of the environments therefore stop drawing i.i.d. step
commands and instead run designed excitation signals.

Three signals, per the requirements table:

===== ============================================ =========================
PRBS  random square wave, hold ``U(0.5, 3.0)`` s   step transients (omega_n, zeta)
Chirp linear sweep ``0.1 -> 2.0`` Hz               closed-loop bandwidth / Bode
Ramp  slopes spanning ``0.2x .. 3x`` of rate_limit where the rate limit bites
===== ============================================ =========================

Two design decisions this module makes that the requirements document leaves
open, both stated here because they change what the identification data means:

**One channel per environment per episode.**  The other four decision channels
hold whatever baseline the curriculum drew at reset.  Exciting several at once
would make the resulting data MIMO, and every downstream use (the Bode plot in
R9, the per-channel omega_n in R8.2) is SISO.  The posture channels get the
larger share of the draw because they are the ones the original command
distribution barely excites at all.

**The chirp holds slew rate, not amplitude, constant.**  A fixed-amplitude
sweep is the textbook choice, but here it would drive the *reference model*
into its own rate limit: 0.5 m/s at 2 Hz asks for 6.3 m/s^2 against a 1.2 m/s^2
limit, so the top half of the sweep would measure the saturation nonlinearity
rather than the bandwidth.  Amplitude is therefore tapered as
``min(A0, slew_budget / (2*pi*f))``, which keeps the whole sweep inside the
linear region.  The cost is real and worth stating: amplitude at 2 Hz is ~15%
of amplitude at 0.1 Hz, so the high-frequency end has correspondingly worse
signal-to-noise and needs more averaging.

Like the rest of this package, no IsaacGym and no env coupling: the object owns
its tensors, and the env calls :meth:`plan` on reset and :meth:`step` once per
control step.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import torch

from .reference import ChannelSpec, validate_channels

#: Signal identifiers, used as the values of the per-env ``signal`` tensor.
PRBS = 0
CHIRP = 1
RAMP = 2
SIGNAL_NAMES = ("prbs", "chirp", "ramp")


def _weight_row(weights: Dict[str, float], names: Sequence[str], what: str) -> torch.Tensor:
    missing = [n for n in names if n not in weights]
    if missing:
        raise ValueError(f"no {what} weight given for {missing}")
    row = torch.tensor([float(weights[n]) for n in names], dtype=torch.double)
    if (row < 0).any():
        raise ValueError(f"{what} weights must be non-negative, got {weights}")
    if float(row.sum()) <= 0.0:
        raise ValueError(f"{what} weights are all zero: {weights}")
    return row / row.sum()


class ExcitationSampler:
    """Drives ``commands_dog`` for the identification subset of environments.

    ``low``/``high`` are per channel, in **response channel order**, and are the
    band the excitation is drawn from.  They are deliberately the curriculum's
    *initial active window* rather than its hard limits: identification data is
    only useful where the policy is competent, and the full limit band
    (+-1.5 m/s from iteration 0) mostly produces falls.  They are plain
    attributes, so R8 may retarget them at the curriculum's live window.

    The identification set is the **tail block** of env indices, contiguous and
    fixed for the run.  Contiguous because R5's environment grouping lands in
    the same index space and has to slot in beside this without either of them
    reshuffling; fixed because an env that alternates between the two roles
    would carry a curriculum bin it never earned.
    """

    def __init__(
        self,
        channels: Sequence[ChannelSpec],
        num_envs: int,
        dt: float,
        *,
        low: Sequence[float],
        high: Sequence[float],
        pool_envs: Optional[int] = None,
        block_multiple: int = 1,
        share_with: Optional[torch.Tensor] = None,
        env_fraction: float = 0.25,
        signal_weights: Optional[Dict[str, float]] = None,
        channel_weights: Optional[Dict[str, float]] = None,
        prbs_hold_s: Sequence[float] = (0.5, 3.0),
        chirp_hz: Sequence[float] = (0.1, 2.0),
        chirp_duration_s: float = 20.0,
        chirp_slew_fraction: float = 0.8,
        ramp_slope_multiple: Sequence[float] = (0.2, 3.0),
        device: str = "cpu",
        generator: Optional[torch.Generator] = None,
    ) -> None:
        channels = tuple(channels)
        validate_channels(channels)
        if not 0.0 <= env_fraction < 1.0:
            raise ValueError(f"env_fraction must be in [0, 1), got {env_fraction}")
        if not prbs_hold_s[0] > 0.0 or prbs_hold_s[1] < prbs_hold_s[0]:
            raise ValueError(f"prbs_hold_s must be an increasing positive pair, got {prbs_hold_s}")
        if not chirp_hz[0] > 0.0 or chirp_hz[1] <= chirp_hz[0]:
            raise ValueError(f"chirp_hz must be an increasing positive pair, got {chirp_hz}")
        # Nyquist: a 0.02 s control step cannot represent a command above 25 Hz,
        # and the reference model would alias it into a lower frequency without
        # complaining.
        if chirp_hz[1] >= 0.5 / dt:
            raise ValueError(
                f"chirp upper frequency {chirp_hz[1]} Hz is at or above Nyquist "
                f"({0.5 / dt} Hz) for dt={dt}"
            )

        self.channels = channels
        self.channel_names = tuple(c.name for c in channels)
        self.num_channels = len(channels)
        self.num_envs = int(num_envs)
        self.dt = float(dt)
        self.device = device

        self.low = torch.as_tensor(low, dtype=torch.float, device=device)
        self.high = torch.as_tensor(high, dtype=torch.float, device=device)
        if self.low.shape != (self.num_channels,) or self.high.shape != (self.num_channels,):
            raise ValueError(
                f"low/high must have one entry per channel ({self.num_channels}), "
                f"got {tuple(self.low.shape)} / {tuple(self.high.shape)}"
            )
        if (self.high <= self.low).any():
            raise ValueError(f"every channel needs high > low, got {low} / {high}")

        self.cmd_index = torch.tensor(
            [c.cmd_index for c in channels], dtype=torch.long, device=device
        )
        self.rate_limit = torch.tensor(
            [c.rate_limit for c in channels], dtype=torch.float, device=device
        )

        self._signal_p = _weight_row(
            signal_weights or {"prbs": 0.5, "chirp": 0.3, "ramp": 0.2}, SIGNAL_NAMES, "signal"
        ).to(device)
        self._channel_p = _weight_row(
            channel_weights or {n: 1.0 for n in self.channel_names},
            self.channel_names,
            "channel",
        ).to(device)

        self.prbs_hold_steps = (
            max(1, int(round(float(prbs_hold_s[0]) / self.dt))),
            max(1, int(round(float(prbs_hold_s[1]) / self.dt))),
        )
        self.chirp_f0, self.chirp_f1 = float(chirp_hz[0]), float(chirp_hz[1])
        self.chirp_steps = max(2, int(round(float(chirp_duration_s) / self.dt)))
        self.chirp_slew_fraction = float(chirp_slew_fraction)
        self.ramp_slope_multiple = (
            float(ramp_slope_multiple[0]), float(ramp_slope_multiple[1])
        )
        self._generator = generator

        # Tail block of the TRAINING pool.  ``pool_envs`` matters: this repo
        # puts the held-out evaluation environments at the tail of the full env
        # range, so a tail block of ``num_envs`` would silently excite exactly
        # the environments whose metrics are supposed to stay comparable.
        # int() floors, so a 1-env evaluation build gets an empty identification
        # set and this object becomes a no-op -- which is what play/eval scripts
        # want without needing a special case.
        self.pool_envs = self.num_envs if pool_envs is None else int(pool_envs)
        if not 0 <= self.pool_envs <= self.num_envs:
            raise ValueError(
                f"pool_envs must be in [0, {self.num_envs}], got {self.pool_envs}"
            )
        # R5 grouping, when enabled, requires the identification block to be a
        # whole number of groups: half a group excited and half not would break
        # the "one command vector per group" invariant outright.
        if block_multiple < 1:
            raise ValueError(f"block_multiple must be >= 1, got {block_multiple}")
        num_identification = int(self.pool_envs * float(env_fraction))
        num_identification = (num_identification // block_multiple) * block_multiple
        self.is_identification = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        if num_identification > 0:
            self.is_identification[self.pool_envs - num_identification:self.pool_envs] = True
        self.identification_ids = self.is_identification.nonzero(as_tuple=False).flatten()

        # R5: every env in a group must see an identical command, excitation
        # included, so the plan is drawn for the group leader and copied.  The
        # per-step state is copied too, not just the plan -- PRBS redraws and
        # the chirp clock would otherwise drift the members apart within a step
        # or two, and the drift would be invisible in the plan itself.
        if share_with is None:
            share_with = torch.arange(self.num_envs, device=device)
        else:
            share_with = share_with.to(device=device, dtype=torch.long)
            if share_with.shape != (self.num_envs,):
                raise ValueError(
                    f"share_with must have one entry per env ({self.num_envs}), "
                    f"got {tuple(share_with.shape)}"
                )
        self.share_with = share_with
        self._shares = bool((share_with != torch.arange(self.num_envs, device=device)).any())

        zeros_l = lambda: torch.zeros(self.num_envs, dtype=torch.long, device=device)  # noqa: E731
        zeros_f = lambda: torch.zeros(self.num_envs, dtype=torch.float, device=device)  # noqa: E731
        self.signal = zeros_l()
        self.channel = zeros_l()
        self.center = zeros_f()
        self.amplitude = zeros_f()
        self.env_low = zeros_f()
        self.env_high = zeros_f()
        self.prbs_hold = torch.ones(self.num_envs, dtype=torch.long, device=device)
        self.value = zeros_f()
        self.chirp_t = zeros_f()
        self.ramp_slope = zeros_f()
        # Increments on every plan draw.  (signal, channel) is NOT a plan
        # identity: with 3 signals and 5 channels a redraw lands on the same
        # pair about one time in fifteen, and anything downstream that segments
        # a recording by "same plan" then silently welds two plans together.
        self.plan_generation = zeros_l()

        if self.identification_ids.numel() > 0:
            self.plan(self.identification_ids)

    # -- planning -----------------------------------------------------------

    @property
    def active(self) -> bool:
        """False when no environment is in identification mode."""
        return bool(self.identification_ids.numel() > 0)

    def _rand(self, n: int) -> torch.Tensor:
        return torch.rand(n, device=self.device, generator=self._generator)

    def _categorical(self, probabilities: torch.Tensor, n: int) -> torch.Tensor:
        """Sample ``n`` indices from a normalised probability row."""
        cdf = torch.cumsum(probabilities, dim=0)
        draw = self._rand(n).to(cdf.dtype).unsqueeze(-1)
        return torch.clamp(
            (draw > cdf.unsqueeze(0)).sum(dim=-1), max=probabilities.numel() - 1
        )

    def plan(self, env_ids: torch.Tensor) -> None:
        """Draw a fresh excitation plan for ``env_ids``.

        Called on reset (and only there): the plan has to outlive the episode
        so that a chirp sweeps its whole band, and so the four passive channels
        are genuinely constant for the whole identification record.
        """
        if env_ids.numel() == 0:
            return
        env_ids = env_ids[self.is_identification[env_ids]]
        n = int(env_ids.numel())
        if n == 0:
            return

        signal = self._categorical(self._signal_p, n)
        channel = self._categorical(self._channel_p, n)
        low = self.low[channel]
        high = self.high[channel]

        self.signal[env_ids] = signal
        self.channel[env_ids] = channel
        self.env_low[env_ids] = low
        self.env_high[env_ids] = high
        self.center[env_ids] = 0.5 * (low + high)
        self.amplitude[env_ids] = 0.5 * (high - low)

        # PRBS: fire on the first step of the episode rather than after a hold,
        # so the transient starts immediately instead of up to 3 s in.
        self.prbs_hold[env_ids] = 0
        self.value[env_ids] = low + (high - low) * self._rand(n)

        self.chirp_t[env_ids] = 0.0

        # Ramp starts at a randomly chosen edge and travels inward, so the full
        # span is covered before the first reversal.
        at_low = self._rand(n) < 0.5
        slope_magnitude = (
            self.rate_limit[channel]
            * (
                self.ramp_slope_multiple[0]
                + (self.ramp_slope_multiple[1] - self.ramp_slope_multiple[0]) * self._rand(n)
            )
            * self.dt
        )
        ramp_start = torch.where(at_low, low, high)
        ramp_slope = torch.where(at_low, slope_magnitude, -slope_magnitude)
        is_ramp = signal == RAMP
        self.value[env_ids] = torch.where(is_ramp, ramp_start, self.value[env_ids])
        self.ramp_slope[env_ids] = ramp_slope
        self.plan_generation[env_ids] += 1

        self._share_from_leader()

    # -- per-step advance ---------------------------------------------------

    def step(self, commands: torch.Tensor) -> torch.Tensor:
        """Write this step's excitation into ``commands``; return the jump mask.

        The returned mask marks environments whose command is not in
        quasi-steady state and whose R4.2/R4.3 settle counter must therefore be
        reset.  For PRBS that is the switching steps only -- a PRBS env between
        switches *is* settled and is a legitimate steady-gain sample.  For chirp
        and ramp it is every step, because those commands never stop moving.

        Written without any data-dependent shape: all three signals are
        evaluated for every identification env and selected with ``where``.
        The obvious formulation -- ``ids[signal == CHIRP]`` and a ``numel()``
        guard per branch -- costs three GPU->CPU synchronisations on **every
        control step**, which is far more than the handful of wasted FLOPs this
        version spends on a tensor of a few hundred elements.
        """
        jumped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if not self.active:
            return jumped
        ids = self.identification_ids
        n = ids.numel()

        signal = self.signal[ids]
        is_prbs = signal == PRBS
        is_chirp = signal == CHIRP
        is_ramp = signal == RAMP
        low, high = self.env_low[ids], self.env_high[ids]
        value = self.value[ids]

        # --- PRBS ---
        hold = self.prbs_hold[ids]
        switching = is_prbs & (hold <= 0)
        lo_steps, hi_steps = self.prbs_hold_steps
        redraw = torch.randint(
            lo_steps, hi_steps + 1, (n,), device=self.device, generator=self._generator
        )
        value_prbs = torch.where(switching, low + (high - low) * self._rand(n), value)
        self.prbs_hold[ids] = torch.where(
            is_prbs, torch.where(switching, redraw, hold - 1), hold
        )

        # --- chirp ---
        duration = self.chirp_steps * self.dt
        elapsed = self.chirp_t[ids]
        tau = torch.remainder(elapsed * self.dt, duration)
        sweep = (self.chirp_f1 - self.chirp_f0) / duration
        frequency = self.chirp_f0 + sweep * tau
        phase = 2.0 * math.pi * (self.chirp_f0 * tau + 0.5 * sweep * tau * tau)
        # Constant-slew taper; see the module docstring.
        budget = self.chirp_slew_fraction * self.rate_limit[self.channel[ids]]
        amplitude = torch.minimum(
            self.amplitude[ids], budget / (2.0 * math.pi * frequency)
        )
        value_chirp = self.center[ids] + amplitude * torch.sin(phase)
        self.chirp_t[ids] = torch.where(is_chirp, elapsed + 1.0, elapsed)

        # --- ramp ---
        slope = self.ramp_slope[ids]
        proposed = value + slope
        # Reflect at the band edge rather than clamping: a clamped ramp parks at
        # the limit and stops exciting anything, which is exactly the
        # sample-starved regime R6 exists to fix.
        reflect = (proposed > high) | (proposed < low)
        self.ramp_slope[ids] = torch.where(is_ramp, torch.where(reflect, -slope, slope), slope)
        value_ramp = torch.clamp(torch.where(reflect, value, proposed), min=low, max=high)

        value = torch.where(is_prbs, value_prbs, torch.where(is_chirp, value_chirp, value_ramp))
        self.value[ids] = value
        self._share_from_leader()

        ids_all = self.identification_ids
        commands[ids_all, self.cmd_index[self.channel[ids_all]]] = self.value[ids_all]

        jumped[ids] = switching | is_chirp | is_ramp
        if self._shares:
            jumped = jumped[self.share_with] & self.is_identification
        return jumped

    def _share_from_leader(self) -> None:
        """Copy every shared env's plan AND running state from its leader."""
        if not self._shares:
            return
        source = self.share_with
        for name in (
            "signal", "channel", "center", "amplitude", "env_low", "env_high",
            "prbs_hold", "value", "chirp_t", "ramp_slope", "plan_generation",
        ):
            buffer = getattr(self, name)
            buffer[:] = buffer[source]

    # -- introspection ------------------------------------------------------

    def signal_counts(self) -> Dict[str, int]:
        """How many identification envs currently run each signal."""
        if not self.active:
            return {name: 0 for name in SIGNAL_NAMES}
        signal = self.signal[self.identification_ids]
        return {name: int((signal == i).sum()) for i, name in enumerate(SIGNAL_NAMES)}

    def channel_counts(self) -> Dict[str, int]:
        """How many identification envs currently excite each channel."""
        if not self.active:
            return {name: 0 for name in self.channel_names}
        channel = self.channel[self.identification_ids]
        return {name: int((channel == i).sum()) for i, name in enumerate(self.channel_names)}
