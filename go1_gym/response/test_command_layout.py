"""R1 acceptance tests: command-space trimming.

Runs against the *real* composed config and the *real* curriculum builder, not
a copy of either, so a drift between config and trainer shows up here.  Still
IsaacGym-free: ``go1_gym.envs.config`` and ``go1_gym.envs.base.curriculum``
both import cleanly under bare torch/numpy.
"""

import argparse

import numpy as np
import pytest

from go1_gym.envs.base.curriculum import (
    build_command_curriculum,
    command_curriculum_bounds,
    command_curriculum_kwargs,
    command_curriculum_local_range,
)
from go1_gym.envs.config import build_roboduet_config
from go1_gym.response import (
    DECISION_CHANNEL_NAMES,
    DECISION_CMD_INDEX,
    DEFAULT_CHANNELS,
    DOG_COMMAND_NAMES,
    FROZEN_CMD_INDEX,
    SEMI_FREE_CMD_INDEX,
    assert_partition_is_complete,
)
from go1_gym.response.command_layout import CURRICULUM_KEY_ORDER


def _args(**overrides):
    base = dict(
        robot="go2_x5",
        num_envs=4096,
        dyna_gait=True,
        goal_reaching=False,
        traj_tracking=False,
        arm_action_mode=None,
        no_reach_table=False,
        dyna_gait_min_frequency=0.0,
        video=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.fixture(scope="module")
def cfg():
    return build_roboduet_config(_args())


@pytest.fixture(scope="module")
def curriculum(cfg):
    return build_command_curriculum(cfg)


# --------------------------------------------------------------------------
# Layout bookkeeping
# --------------------------------------------------------------------------


def test_r1_partition_covers_every_command_slot():
    assert_partition_is_complete(num_commands=11)


def test_curriculum_dimension_order_matches_command_order(cfg):
    """The trainer writes ``new_commands[:, i]`` into ``commands_dog[:, i]``.

    That is only correct because the curriculum's keyword order happens to
    coincide with the command column order.  Pin it down.
    """
    keys = [k for k in command_curriculum_kwargs(cfg) if k != "seed"]
    assert tuple(keys) == CURRICULUM_KEY_ORDER
    assert len(keys) == cfg.dog.dog_num_commands
    # Same length and same semantic sequence as the command names.
    assert len(DOG_COMMAND_NAMES) == 11


def test_reference_channels_use_the_layout_indices():
    assert tuple(c.cmd_index for c in DEFAULT_CHANNELS) == DECISION_CMD_INDEX
    assert tuple(c.name for c in DEFAULT_CHANNELS) == DECISION_CHANNEL_NAMES
    # The ordering trap this table exists to prevent: pitch is command 3 but
    # channel 4; height is command 5 but channel 3.
    assert DECISION_CMD_INDEX[3] == 5, "channel 3 is body height, command index 5"
    assert DECISION_CMD_INDEX[4] == 3, "channel 4 is body pitch, command index 3"


# --------------------------------------------------------------------------
# R1 acceptance: frozen channels are constant, decision channels are not
# --------------------------------------------------------------------------


def test_frozen_channels_are_constant_under_curriculum_sampling(curriculum):
    """The acceptance criterion, run against real sampling rather than config."""
    samples, _ = curriculum.sample(batch_size=20_000)
    spread = samples.max(axis=0) - samples.min(axis=0)

    for index in FROZEN_CMD_INDEX:
        name = DOG_COMMAND_NAMES[index]
        assert spread[index] < 0.05, (
            f"frozen channel {name!r} (command {index}) varied by {spread[index]:.4f} "
            "across 20k samples"
        )


def test_body_roll_is_pinned_to_exactly_zero(curriculum):
    samples, _ = curriculum.sample(batch_size=5_000)
    roll = samples[:, DOG_COMMAND_NAMES.index("body_roll")]
    assert np.all(roll == 0.0), f"body roll must be identically 0, saw range {roll.min()}..{roll.max()}"


def test_gait_frequency_is_uniform_in_the_narrow_band(curriculum):
    samples, _ = curriculum.sample(batch_size=20_000)
    freq = samples[:, DOG_COMMAND_NAMES.index("gait_frequency")]
    assert freq.min() >= 2.5 - 1e-9 and freq.max() <= 3.5 + 1e-9
    # A single bin means the curriculum cannot restrict it; sampling should
    # cover the band rather than cluster at the centre.
    assert freq.max() - freq.min() > 0.9, "gait frequency should span its band"
    assert abs(freq.mean() - 3.0) < 0.02


def test_decision_channels_actually_vary(curriculum):
    samples, _ = curriculum.sample(batch_size=20_000)
    spread = samples.max(axis=0) - samples.min(axis=0)
    for index in DECISION_CMD_INDEX:
        name = DOG_COMMAND_NAMES[index]
        assert spread[index] > 0.1, f"decision channel {name!r} did not vary ({spread[index]:.4f})"


def test_posture_channels_are_real_curriculum_dimensions(cfg):
    """Before R1 pitch/height had a single bin covering their whole range, so
    the adaptive curriculum had no handle on them at all."""
    assert cfg.commands.num_bins_body_pitch > 1
    assert cfg.commands.num_bins_body_height > 1


def test_frozen_and_semi_free_channels_get_one_bin_each(cfg):
    """A single bin is how a channel is kept out of the adaptive curriculum
    while keeping its slot in the command vector (R1 invariant)."""
    bins = {
        "body_roll": cfg.commands.num_bins_body_roll,
        "footswing_height": cfg.commands.num_bins_footswing_height,
        "stance_width": cfg.commands.num_bins_stance_width,
        "stance_length": cfg.commands.num_bins_stance_length,
        "gait_duration": cfg.commands.num_bins_gait_duration,
        "gait_frequency": cfg.commands.num_bins_gait_frequency,
    }
    for name, count in bins.items():
        assert count == 1, f"{name} should have 1 curriculum bin, has {count}"


def test_command_width_is_unchanged_by_freezing(cfg):
    """R1 acceptance: freezing must not change the command vector's width.

    Frozen channels keep their slot rather than being deleted, so the command
    vector is exactly as wide as before the trimming.  This is R1's invariant
    and it still holds; the observation width is no longer part of it.
    """
    assert cfg.dog.dog_num_commands == 11


def test_observation_width_is_whatever_the_parts_table_says(cfg):
    """R1 originally pinned this at 90 to prove freezing changed nothing.

    R7 then raised it to 112 on purpose (reference state, its rate, the command
    gap, the end-effector position, and the two IMU-anchored (g, l) pairs), so
    the literal is now R7's to own.  What still has to hold -- and is the thing
    worth asserting -- is that the configured width is derived from the parts
    table rather than written down twice.
    """
    from go1_gym.envs.config.core import dog_obs_dim_parts

    assert cfg.dog.dog_num_observations == sum(dog_obs_dim_parts(cfg).values())
    assert cfg.dog.dog_num_observations == 112
    assert cfg.dog.dog_num_obs_history == 112 * cfg.dog.dog_num_observation_history


def test_curriculum_grid_is_small_enough_to_be_covered(cfg):
    """1.96M bins over 4096 envs was never going to be visited; the trimming
    is what makes the grid tractable."""
    kwargs = command_curriculum_kwargs(cfg)
    grid = int(np.prod([v[2] for k, v in kwargs.items() if k != "seed"]))
    assert grid == 21 * 3 * 21 * 5 * 1 * 5, grid
    assert grid < 50_000


# --------------------------------------------------------------------------
# Profile ownership: enabling a layout must preserve command configuration
# --------------------------------------------------------------------------


def test_dynamic_gait_preserves_profile_command_values(monkeypatch):
    """The profile owns the band; layout activation cannot silently rewrite it."""
    cfg = build_roboduet_config(_args(dyna_gait_min_frequency=0.0))
    assert cfg.commands.gait_frequency_cmd_range == [2.5, 3.5]
    assert cfg.commands.limit_gait_frequency == [2.5, 3.5]

    # The legacy CLI lower bound remains ineffective on this branch.
    cfg = build_roboduet_config(_args(dyna_gait_min_frequency=1.0))
    assert cfg.commands.gait_frequency_cmd_range == [2.5, 3.5]

    from go1_gym.envs.config.wbc import ROBODUET_PROFILE
    monkeypatch.setitem(ROBODUET_PROFILE.overrides, "commands.gait_frequency_cmd_range", [2.6, 3.4])
    monkeypatch.setitem(ROBODUET_PROFILE.overrides, "commands.limit_gait_frequency", [2.5, 3.5])
    cfg = build_roboduet_config(_args())
    assert cfg.commands.gait_frequency_cmd_range == [2.6, 3.4]
    assert cfg.commands.limit_gait_frequency == [2.5, 3.5]


def test_build_without_dynamic_gait_still_works():
    """Inactive gait defaults must not add command slots to a bare config."""
    cfg = build_roboduet_config(_args(dyna_gait=False))
    assert cfg.dog.dog_num_commands == 6
    assert cfg.commands.body_roll_range == [0.0, 0.0]
    assert cfg.commands.num_bins_body_pitch == 5


# --------------------------------------------------------------------------
# Equivalence of the extracted curriculum builders with the original inline code
# --------------------------------------------------------------------------


def _original_curriculum_kwargs(cfg):
    """Verbatim transcription of LeggedRobot._init_command_distribution as it
    stood at commit 9e7e995, before the extraction."""
    curriculum_kwargs = dict(
        seed=cfg.commands.curriculum_seed,
        x_vel=(cfg.commands.limit_vel_x[0], cfg.commands.limit_vel_x[1], cfg.commands.num_bins_vel_x),
        y_vel=(cfg.commands.limit_vel_y[0], cfg.commands.limit_vel_y[1], cfg.commands.num_bins_vel_y),
        yaw_vel=(cfg.commands.limit_vel_yaw[0], cfg.commands.limit_vel_yaw[1], cfg.commands.num_bins_vel_yaw),
        body_pitch=(cfg.commands.limit_body_pitch[0], cfg.commands.limit_body_pitch[1], cfg.commands.num_bins_body_pitch),
        body_roll=(cfg.commands.limit_body_roll[0], cfg.commands.limit_body_roll[1], cfg.commands.num_bins_body_roll),
        body_height=(cfg.commands.limit_body_height[0], cfg.commands.limit_body_height[1], cfg.commands.num_bins_body_height),
    )
    if cfg.commands.use_dynamic_gait:
        curriculum_kwargs.update(
            gait_frequency=(cfg.commands.limit_gait_frequency[0], cfg.commands.limit_gait_frequency[1], cfg.commands.num_bins_gait_frequency),
            footswing_height=(cfg.commands.limit_footswing_height[0], cfg.commands.limit_footswing_height[1], cfg.commands.num_bins_footswing_height),
            stance_width=(cfg.commands.limit_stance_width[0], cfg.commands.limit_stance_width[1], cfg.commands.num_bins_stance_width),
            stance_length=(cfg.commands.limit_stance_length[0], cfg.commands.limit_stance_length[1], cfg.commands.num_bins_stance_length),
            gait_duration=(cfg.commands.limit_gait_duration[0], cfg.commands.limit_gait_duration[1], cfg.commands.num_bins_gait_duration),
        )
    return curriculum_kwargs


def _original_bounds(cfg):
    low = np.array([
        cfg.commands.lin_vel_x[0], cfg.commands.lin_vel_y[0], cfg.commands.ang_vel_yaw[0],
        cfg.commands.body_pitch_range[0], cfg.commands.body_roll_range[0], cfg.commands.limit_body_height[0],
    ])
    high = np.array([
        cfg.commands.lin_vel_x[1], cfg.commands.lin_vel_y[1], cfg.commands.ang_vel_yaw[1],
        cfg.commands.body_pitch_range[1], cfg.commands.body_roll_range[1], cfg.commands.limit_body_height[1],
    ])
    if cfg.commands.use_dynamic_gait:
        low = np.concatenate([low, np.array([
            cfg.commands.limit_gait_frequency[0], cfg.commands.limit_footswing_height[0],
            cfg.commands.limit_stance_width[0], cfg.commands.limit_stance_length[0],
            cfg.commands.limit_gait_duration[0],
        ])])
        high = np.concatenate([high, np.array([
            cfg.commands.limit_gait_frequency[1], cfg.commands.limit_footswing_height[1],
            cfg.commands.limit_stance_width[1], cfg.commands.limit_stance_length[1],
            cfg.commands.limit_gait_duration[1],
        ])])
    return low, high


@pytest.mark.parametrize("dyna_gait", [True, False])
def test_extracted_builders_match_the_original_inline_code(dyna_gait):
    """Guards the extraction itself: a transcription slip would show up here."""
    cfg = build_roboduet_config(_args(dyna_gait=dyna_gait))

    assert command_curriculum_kwargs(cfg) == _original_curriculum_kwargs(cfg)

    low, high = command_curriculum_bounds(cfg)
    orig_low, orig_high = _original_bounds(cfg)
    assert np.array_equal(low, orig_low)
    assert np.array_equal(high, orig_high)

    expected_local = np.array([0.55, 0.55, 0.55, 1.0, 1.0, 1.0])
    if dyna_gait:
        expected_local = np.concatenate([expected_local, np.full(5, 0.55)])
    assert np.array_equal(command_curriculum_local_range(cfg), expected_local)


def test_local_range_has_one_entry_per_command_dimension(cfg):
    assert command_curriculum_local_range(cfg).shape[0] == cfg.dog.dog_num_commands


def test_initial_window_activates_a_nonempty_set_of_bins(curriculum):
    """``set_to`` asserts on an empty domain; a badly chosen frozen range (e.g.
    a limit band that excludes its own grid centroid) would trip it."""
    assert curriculum.weights.sum() > 0
