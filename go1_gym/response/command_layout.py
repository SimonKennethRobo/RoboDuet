"""Authoritative index map for ``commands_dog`` and the R1 channel partition.

Three different orderings coexist in this codebase and conflating them is the
easiest way to introduce a silent, wrong-channel bug:

1.  **Command order** -- the columns of ``commands_dog``.  Fixed by
    ``LeggedRobot._init_buffers``' ``commands_scale_dog`` construction.
2.  **Curriculum grid order** -- the keyword order passed to
    ``RewardThresholdCurriculum``.  Happens to be *identical* to the command
    order, which is why ``new_commands[:, i]`` can be written straight into
    ``commands_dog[:, i]``.  This module asserts that coincidence rather than
    trusting it.
3.  **Response channel order** -- the order R2 lists the five MPC decision
    channels in: forward velocity, lateral velocity, yaw rate, **body height**,
    **body pitch**.  Note height comes before pitch here, while in the command
    vector pitch (3) comes before height (5).

Everything downstream imports names from this module instead of writing
literals, so a layout change is a one-file change.
"""

from __future__ import annotations

from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# Command order: columns of commands_dog.
# Indices 0..5 always exist; 6..10 exist only with --dyna_gait
# (cfg.dog.dog_num_commands is 6 without it, 11 with it).
# ---------------------------------------------------------------------------
VX = 0
VY = 1
YAW_RATE = 2
BODY_PITCH = 3
BODY_ROLL = 4
BODY_HEIGHT = 5
GAIT_FREQUENCY = 6
FOOTSWING_HEIGHT = 7
STANCE_WIDTH = 8
STANCE_LENGTH = 9
GAIT_DURATION = 10

#: index -> canonical name, in command order.
DOG_COMMAND_NAMES: Tuple[str, ...] = (
    "vx",
    "vy",
    "yaw_rate",
    "body_pitch",
    "body_roll",
    "body_height",
    "gait_frequency",
    "footswing_height",
    "stance_width",
    "stance_length",
    "gait_duration",
)

DOG_COMMAND_INDEX: Dict[str, int] = {name: i for i, name in enumerate(DOG_COMMAND_NAMES)}

#: Keyword order used to build the curriculum grid in
#: ``LeggedRobot._init_command_distribution``.  Kept here so a test can assert
#: it still matches the command order.
CURRICULUM_KEY_ORDER: Tuple[str, ...] = (
    "x_vel",
    "y_vel",
    "yaw_vel",
    "body_pitch",
    "body_roll",
    "body_height",
    "gait_frequency",
    "footswing_height",
    "stance_width",
    "stance_length",
    "gait_duration",
)


# ---------------------------------------------------------------------------
# R1 partition of the command space.
# ---------------------------------------------------------------------------

#: The five channels the MPC actually decides, in **response channel order**
#: (R2's ordering: height before pitch).  This is the tuple ReferenceModel's
#: default channel table is built from.
DECISION_CMD_INDEX: Tuple[int, ...] = (VX, VY, YAW_RATE, BODY_HEIGHT, BODY_PITCH)
DECISION_CHANNEL_NAMES: Tuple[str, ...] = ("vx", "vy", "wyaw", "height", "pitch")

#: Physical unit suffix per decision channel.  Metric names in this repo carry
#: their units (see AGENTS.md) and these five channels do not share one.
DECISION_CHANNEL_UNITS: Dict[str, str] = {
    "vx": "mps",
    "vy": "mps",
    "wyaw": "rad_s",
    "height": "m",
    "pitch": "rad",
}

#: Trained over a narrow band, fixed at deployment, but recorded as a
#: conditioning input for the gait-phase residual model (R1 "semi-free").
#: Excluded from the adaptive curriculum: it gets a single grid bin, so the
#: curriculum cannot restrict it and sampling is plain uniform over the band.
SEMI_FREE_CMD_INDEX: Tuple[int, ...] = (GAIT_FREQUENCY,)

#: Held constant (or jittered inside a narrow band for robustness).  These keep
#: their positions in the command vector -- R1 invariant: deleting an index
#: would change the observation width and poison the comparison against the
#: unmodified policy.
FROZEN_CMD_INDEX: Tuple[int, ...] = (
    BODY_ROLL,
    FOOTSWING_HEIGHT,
    STANCE_WIDTH,
    STANCE_LENGTH,
    GAIT_DURATION,
)


def assert_partition_is_complete(num_commands: int = 11) -> None:
    """Every command column belongs to exactly one R1 group."""
    partitioned = (
        set(DECISION_CMD_INDEX) | set(SEMI_FREE_CMD_INDEX) | set(FROZEN_CMD_INDEX)
    )
    expected = set(range(num_commands))
    if partitioned != expected:
        raise AssertionError(
            f"R1 partition covers {sorted(partitioned)} but commands are {sorted(expected)}"
        )
    total = len(DECISION_CMD_INDEX) + len(SEMI_FREE_CMD_INDEX) + len(FROZEN_CMD_INDEX)
    if total != num_commands:
        raise AssertionError(f"R1 groups overlap: {total} slots for {num_commands} commands")
