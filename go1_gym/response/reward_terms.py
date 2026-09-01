"""The R4 reward terms, as pure tensor functions.

Kept out of ``go1_gym/envs/rewards/rewards.py`` so the acceptance criteria --
which are statements about the *shape* of these functions, not about the
simulator -- can be tested on synthetic trajectories.  ``rewards.py`` holds the
thin ``_reward_*`` wrappers that pull the tensors out of the env.

Sign convention follows the rest of the repo: every function here returns a
NON-NEGATIVE quantity, and the configured reward scale supplies the sign.  That
matters more than usual because of how the total is composed
(``LeggedRobot.compute_reward``)::

    reward = positive_sum * exp(negative_sum / sigma_rew_neg)

R4's invariant is that the reference-tracking term enters as a *task* term
(positive, so it multiplies) while the consistency penalties enter as *aux*
factors (negative, so they attenuate).  Writing any of them additively would be
both hard to scale and liable to swamp the task reward early in training.

One arithmetic trap worth stating once: reward scales are multiplied by ``dt``
at registration, and negative terms are then divided by
``rewards.sigma_rew_neg`` inside the exponent.  With ``dt = 0.02`` and
``sigma_rew_neg = 0.02`` those cancel exactly, so **the configured weight of a
penalty is its effective coefficient in the exponent**.  That is a coincidence
of two unrelated constants, not a design; changing either silently rescales
every penalty here.
"""

from __future__ import annotations

from typing import Optional

import torch


def reference_tracking(
    detrended: torch.Tensor,
    reference: torch.Tensor,
    sigma: torch.Tensor,
    weights: torch.Tensor,
    gate: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """R4.1 -- the core term.  Positive, in ``[0, 1]``.

    Exponential error on the **detrended** measurement against the reference
    *state*, evaluated every step.  Rewarding the position at each instant
    rather than the endpoint is the whole mechanism: given a 0.4 rad step, the
    step at t = 0.1 s is asked for 0.036 rad, not 0.4, so arriving too fast is
    penalised exactly as arriving too slowly is.

    ``sigma`` and ``weights`` are ``(1, C)`` rows; ``sigma`` must come from the
    calibration in :mod:`go1_gym.response.calibration`, per channel, because the
    channels do not share units.

    ``gate`` is R4.1's soft-target mask (see :func:`soft_gate_from_events`).
    Where it is 0 the term contributes nothing.  Zero rather than one is
    deliberate: handing out full credit while disturbed would pay the policy for
    getting disturbed.  With zero it simply cannot earn consistency credit
    during recovery, which is what "prioritise stability under extreme
    disturbance" means operationally -- there is no gradient asking it to trade
    stability for consistency, in either direction.
    """
    error = detrended - reference
    per_channel = torch.exp(-(error ** 2) / (sigma ** 2))
    value = (per_channel * weights).sum(dim=-1) / weights.sum()
    if gate is not None:
        value = value * gate
    return value


def phase_variance(
    oscillation: torch.Tensor,
    delta_hat: torch.Tensor,
    weights: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """R4.2 -- penalty on the *variance* about the phase-conditioned mean.

    The distinction R4 insists on: this is not an amplitude penalty.  Two
    policies with the same 2 cm z ripple are not equivalent -- one whose
    waveform repeats identically every cycle is predictable, so the MPC can feed
    it forward and the arm can cancel it, leaving essentially no end-effector
    jitter; one whose waveform wanders leaks the whole 2 cm through.  This term
    turns the second into the first.  It does compress amplitude a little as a
    side effect, but it does not force it down, because squeezing the ripple out
    costs leg torque and buys robustness problems.

    ``oscillation`` is the instantaneous deviation from the local mean, i.e.
    ``y - y_lp`` from the residual estimator's *update* path.  Using the lagging
    low-pass is acceptable here, and only here, because this term is masked off
    through the transient and therefore only ever evaluated in quasi-steady
    state, where the lag is irrelevant.  R4.1, which lives in the transient,
    must use the zero-delay phase lookup instead.

    Scale this term with care.  Measured, it is ~95x larger for a from-scratch
    policy (0.34) than for a trained one (0.0036), so a weight sized against a
    trained policy multiplies the whole reward by ~1e-3 if switched on at
    iteration 0.  R8's slow ramp is not caution, it is a requirement.
    """
    deviation = oscillation - delta_hat
    value = (deviation ** 2) * weights
    if mask is not None:
        value = value * mask
    return value.sum(dim=-1)


def steady_gain(
    detrended: torch.Tensor,
    command: torch.Tensor,
    weights: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """R4.3 -- pin the identified model's DC gain to 1.

    Once a command has been held long enough, the detrended measurement is
    compared with the **command**, not with the reference state.  Body height is
    the channel that needs it most: a 0.30 m command that actually walks at
    0.27 m, and at 0.25 m once a 3 kg payload is added, gives the MPC a
    load-dependent gain it has no way to know about.
    """
    deviation = detrended - command
    value = (deviation ** 2) * weights
    if mask is not None:
        value = value * mask
    return value.sum(dim=-1)


def settled_mask(
    steps_since_command_change: torch.Tensor,
    settle_steps: torch.Tensor,
) -> torch.Tensor:
    """``(E, C)`` mask: has this channel's transient finished?

    Per channel, because the thresholds R4 gives are ``2 / omega_n`` (R4.2) and
    ``3 / omega_n`` (R4.3) and ``omega_n`` differs between channels -- the pitch
    channel is deliberately the slowest, so it stays masked longest.
    """
    return (steps_since_command_change.unsqueeze(-1) >= settle_steps).to(settle_steps.dtype)


def soft_gate_from_events(
    timer: torch.Tensor,
    triggered: torch.Tensor,
    hold_steps: int,
) -> torch.Tensor:
    """Advance R4.1's soft-target countdown; returns the new timer.

    ``triggered`` marks environments that just saw a large disturbance or a bad
    slip.  The gate stays shut for ``hold_steps`` afterwards.  The caller turns
    the timer into a multiplier with ``(timer == 0)``.
    """
    decremented = torch.clamp(timer - 1, min=0)
    return torch.where(triggered, torch.full_like(timer, hold_steps), decremented)


def domain_consistency(
    detrended: torch.Tensor,
    twin_detrended: torch.Tensor,
    weights: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """R5 -- penalty on responding differently to the same command in a
    different domain.  Non-negative; the scale supplies the sign.

    ``twin_detrended`` is the nominal twin's detrended response, gathered per
    env.  The comparison is against the twin and never against a within-group
    mean or variance, which R5 states as an invariant: variance has a degenerate
    minimiser -- be equally sluggish in every domain -- that collapses
    consistency and bandwidth together, while "behave in the hard domain the way
    you behave in the easy one" carries its own performance anchor.

    ``valid`` is :attr:`EnvGrouping.valid`: zero for ungrouped envs, for the
    twin itself, and for any group whose phase or command timing has drifted
    after a fall.  That last one is the R5 invariant with teeth -- without it
    the penalty is largest exactly where it is least meaningful, and a fall gets
    charged twice.

    On "the twin side must not receive gradient".  In this codebase there is no
    autograd path through the simulator at all -- these are measured tensors, so
    a ``.detach()`` here would be decoration.  What actually implements the
    requirement is that the twin's own value of this term is masked to zero:
    the policy is never paid for moving the *twin* towards a member, only for
    moving members towards the twin.  Gradient asymmetry in a model-free setting
    is a property of which environments the reward is applied to, not of the
    tensor graph.
    """
    error = detrended - twin_detrended
    return ((error ** 2) * weights).sum(dim=-1) * valid
