"""R5 -- environment grouping and the nominal twin.

R5 asks for an explicit penalty on "the same command produces a different
response in a different domain".  The mechanism: environments are grouped
(group size ~4); a group **shares** its command sequence, gait phase clock and
resample instants, and **randomises** friction, mass, payload, motor strength
and arm configuration across its members.  One member of each group is the
**nominal twin** -- flat ground, nominal mass, no payload, arm fixed, no
disturbance -- and the others are asked to match its detrended response.

Two things this module exists to make hard to get wrong.

**The twin, not the group mean.**  R5 states this as an invariant and the
reason is a degenerate solution: a within-group variance penalty is minimised
just as well by being equally sluggish in every domain, which drives
consistency and bandwidth to zero together.  Matching a *nominal twin* means
"behave in a hard domain the way you behave in the easy one", which carries its
own performance anchor.  So the comparison target is always an index into the
group, never a reduction over it.

**Termination asymmetry.**  When one member resets, its phase clock and its
physical state are no longer those of the group, so any cross-domain comparison
against it is meaningless for a while.  This is a correctness requirement, not a
refinement: without it the consistency penalty is largest exactly when it is
least meaningful, and the policy is punished for a fall it has already been
punished for.

**How long "a while" is, and why it is not "until the next resample".**  The
first implementation latched the group off until its next shared resample.  That
is correct but ruinously conservative, and the cost is arithmetic rather than
empirical: with a resample window ``W``, an episode length ``L`` and a group of
``G``, the fraction of time the term is masked is

    1 - (L / ((G+1) W)) * (1 - (1 - W/L)^(G+1))

At the shipped ``W = 500``, ``L = 1000``, ``G = 4`` that is **0.61 for a policy
that never falls at all** -- the mask is driven by episode *timeouts*, not by
falls, so no amount of training brings it down.  Measured on the 20k run it sat
at 0.71-0.82, i.e. R5 was switched off for three quarters of training.

What is actually incomparable after a reset is bounded and short: the command
and the gait phase are re-adopted from the twin on the spot (see
``LeggedRobot._adopt_group_commands`` and the phase copy at the end of
``reset_idx``), so all that remains is the robot's own start-up transient.  The
mask is therefore a fixed settling countdown, ``settle_steps``, restarted by any
member's reset.  Same invariant, ~4x the availability, and -- the part that
matters for the reward -- an availability that no longer depends on the episode
length or the resample period.

The grouped block is a prefix of the environment range.  Environments past it
(the remainder that does not fill a whole group, and the held-out evaluation
environments) keep the original per-environment resample clock and take no part
in R5 -- ``is_grouped`` is the single flag that separates the two regimes.
"""

from __future__ import annotations

from typing import Optional

import torch


class EnvGrouping:
    """Group membership, the group-level clock, and the desync flag.

    Index layout, fixed for the run::

        group g occupies envs [g*size, (g+1)*size)
        env g*size is the nominal twin of group g

    Fixed rather than reshuffled because several domain-randomisation draws
    (arm link mass/COM, the mount-TF bucket, base mass) happen **once, during
    ``_create_envs``**, and cannot be redrawn later.  Any scheme that moved the
    twin during training would silently leave the old twin non-nominal.
    """

    def __init__(
        self,
        num_envs: int,
        group_size: int = 4,
        pool_envs: Optional[int] = None,
        settle_steps: int = 50,
        device: str = "cpu",
    ) -> None:
        if group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {group_size}")
        if settle_steps < 1:
            raise ValueError(f"settle_steps must be >= 1, got {settle_steps}")
        self.num_envs = int(num_envs)
        self.group_size = int(group_size)
        # Control steps a group stays uncomparable after any member resets.  The
        # default is the 1.0 s the env config ships; it is a constructor
        # argument rather than a constant because the right value is a settling
        # time in seconds and only the env knows dt.
        self.settle_steps = int(settle_steps)
        self.device = device

        pool = self.num_envs if pool_envs is None else int(pool_envs)
        if not 0 <= pool <= self.num_envs:
            raise ValueError(f"pool_envs must be in [0, {self.num_envs}], got {pool}")
        # Only whole groups participate.  A partial trailing group would have a
        # twin whose members do not all exist.
        self.num_groups = pool // self.group_size
        self.num_grouped = self.num_groups * self.group_size

        index = torch.arange(self.num_envs, device=device)
        self.is_grouped = index < self.num_grouped
        # -1 for ungrouped envs, so an accidental use as an index raises rather
        # than silently reading the last group.
        self.group_of = torch.where(
            self.is_grouped, index // self.group_size, torch.full_like(index, -1)
        )
        self.twin_of = torch.where(
            self.is_grouped,
            (index // self.group_size) * self.group_size,
            index,
        )
        self.is_twin = self.is_grouped & (index == self.twin_of)

        self.group_clock = torch.zeros(self.num_groups, dtype=torch.long, device=device)
        # Steps remaining before the group is comparable again.  0 = comparable.
        self.group_settle = torch.zeros(self.num_groups, dtype=torch.long, device=device)

    # -- clock --------------------------------------------------------------
    #
    # Contract: call ``groups_due()`` first, then ``advance()`` at the end of
    # the step.  Checking before advancing means a freshly built grouping is
    # due on its very first step, so every group gets a real command draw
    # immediately.  Advancing first would leave every group holding zeros until
    # the first interval elapsed -- 10 s of standing still at the start of
    # training, which looks like a policy failure rather than a clock bug.

    def advance(self) -> None:
        """End of one control step.

        Called from ``_post_physics_step_callback``, which runs *before*
        ``reset_idx``: a group marked this step therefore serves its full
        ``settle_steps`` before becoming comparable again, never one short.
        """
        self.group_clock += 1
        torch.clamp_(self.group_settle.sub_(1), min=0)

    def groups_due(self, interval) -> torch.Tensor:
        """Group ids whose shared resample instant is this step.

        ``interval`` may be a scalar or a per-group tensor.  Per-group because
        the identification groups (R6) need a longer period than the rest: their
        commands must hold still for a whole excitation plan, and giving them a
        different period is a cleaner way to say that than suppressing their
        resample -- suppression leaves a group that never draws a command at all
        if its twin is also suppressed.
        """
        if self.num_groups == 0:
            return torch.zeros(0, dtype=torch.long, device=self.device)
        if not torch.is_tensor(interval):
            interval = torch.full_like(self.group_clock, int(interval))
        return (self.group_clock % interval == 0).nonzero(as_tuple=False).flatten()

    def envs_of_groups(self, group_ids: torch.Tensor) -> torch.Tensor:
        """Flatten group ids to the env ids they contain, ascending."""
        if group_ids.numel() == 0:
            return torch.zeros(0, dtype=torch.long, device=self.device)
        offsets = torch.arange(self.group_size, device=self.device)
        return (group_ids.unsqueeze(-1) * self.group_size + offsets).flatten().sort().values

    # -- synchronisation ----------------------------------------------------

    def mark_desync(self, env_ids: torch.Tensor) -> None:
        """A grouped env reset; start the group's settling countdown.

        Applies whether or not the env is the twin.  R5 spells out both cases --
        a member falling desynchronises the comparison, the twin falling removes
        the target altogether -- and one countdown per group covers both.

        Restarting rather than accumulating: two members resetting a step apart
        leaves the group off until the later one has settled, which is what the
        invariant asks for.
        """
        if env_ids.numel() == 0:
            return
        groups = self.group_of[env_ids]
        groups = groups[groups >= 0]
        if groups.numel() > 0:
            self.group_settle[groups] = self.settle_steps

    def resync(self, group_ids: torch.Tensor) -> None:
        """A group resample restarts the shared clock.

        It deliberately does **not** clear the settling countdown.  A resample
        re-establishes the shared *command*, which was never what a reset broke:
        the member that reset is still mid-transient, and unmasking it because a
        resample happened to land 3 steps later would feed exactly the start-up
        spike the mask exists to keep out of the cross-domain comparison.  The
        countdown is the sole authority on comparability.
        """
        if group_ids.numel() > 0:
            self.group_clock[group_ids] = 0

    @property
    def group_desync(self) -> torch.Tensor:
        """``(num_groups,)`` bool: is this group inside its settling window?"""
        return self.group_settle > 0

    @property
    def valid(self) -> torch.Tensor:
        """``(E,)`` float mask: may this env's R5 term contribute this step?

        Zero for ungrouped envs, for members of a desynchronised group, and for
        the twin itself -- the twin is the target, and comparing it with itself
        would add a constant zero to the reward for a quarter of the
        environments, quietly changing the reward's scale.
        """
        healthy = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.num_groups > 0:
            settling = self.group_settle[self.group_of.clamp(min=0)] > 0
            healthy = self.is_grouped & ~settling
        return (healthy & ~self.is_twin).float()

    def broadcast_from_twin(self, values: torch.Tensor) -> torch.Tensor:
        """Replace every grouped env's row with its twin's row.

        Works for ``(E,)`` and ``(E, ...)``; ungrouped rows pass through because
        ``twin_of`` is the identity there.
        """
        return values[self.twin_of]
