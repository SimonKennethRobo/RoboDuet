"""Pure-tensor state machines for the response-consistent locomotion policy.

Everything in this package is deliberately free of IsaacGym and of any
``LeggedRobot``/``WBCEnv`` coupling: these objects hold their own tensors and
are advanced by the env, never the other way round.  That keeps the numerical
acceptance criteria in ``docs/project-design-rlmpc-v3-coding.md`` (R2/R3/R4)
testable on CPU without starting a simulator.

Note on placement: the R0 plan originally put this at
``go1_gym/envs/roboduet/response.py``.  That location cannot work -- importing
any submodule of ``go1_gym.envs.roboduet`` executes its ``__init__``, which
imports ``wbc_env`` and therefore IsaacGym, defeating the entire reason for
splitting these classes out.  ``go1_gym/__init__.py`` only does ``import os``,
so this package is reachable from a bare ``torch`` environment.
"""

from .command_layout import (
    GAIT_FREQUENCY,
    CURRICULUM_KEY_ORDER,
    DECISION_CHANNEL_NAMES,
    DECISION_CHANNEL_UNITS,
    DECISION_CMD_INDEX,
    DOG_COMMAND_INDEX,
    DOG_COMMAND_NAMES,
    FROZEN_CMD_INDEX,
    SEMI_FREE_CMD_INDEX,
    assert_partition_is_complete,
)
from .calibration import (
    DEFAULT_TARGET_DISCRIMINATION,
    SigmaCalibration,
    calibrate_channels,
    calibrate_sigma,
    discrimination_for_sigma,
)
from .excitation import (
    CHIRP,
    PRBS,
    RAMP,
    SIGNAL_NAMES,
    ExcitationSampler,
)
from .grouping import EnvGrouping
from .residual import PhaseResidualEstimator
from .reference import (
    DEFAULT_CHANNELS,
    ChannelSpec,
    ReferenceModel,
    build_channels,
    critically_damped_step_response,
    validate_channels,
)

__all__ = [
    "GAIT_FREQUENCY",
    "CURRICULUM_KEY_ORDER",
    "DECISION_CHANNEL_NAMES",
    "DECISION_CHANNEL_UNITS",
    "DECISION_CMD_INDEX",
    "DOG_COMMAND_INDEX",
    "DOG_COMMAND_NAMES",
    "FROZEN_CMD_INDEX",
    "SEMI_FREE_CMD_INDEX",
    "assert_partition_is_complete",
    "CHIRP",
    "PRBS",
    "RAMP",
    "SIGNAL_NAMES",
    "ExcitationSampler",
    "EnvGrouping",
    "DEFAULT_CHANNELS",
    "ChannelSpec",
    "PhaseResidualEstimator",
    "DEFAULT_TARGET_DISCRIMINATION",
    "SigmaCalibration",
    "calibrate_channels",
    "calibrate_sigma",
    "discrimination_for_sigma",
    "ReferenceModel",
    "build_channels",
    "critically_damped_step_response",
    "validate_channels",
]
