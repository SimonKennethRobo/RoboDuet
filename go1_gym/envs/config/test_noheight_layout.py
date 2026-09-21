"""No-height (layout version 3) dog observation regressions; no simulator needed.

The N-series checkpoints were trained on a frame with every base-height scalar
removed, and with the *commanded* height kept (omit_height_command=False) since
that one is known onboard without an estimator. Their snapshots also predate the
R7.1 response block, so restoring one has to recover both facts -- a frame that
merely has the right width can still be the wrong frame.
"""

from argparse import Namespace

import pytest

from go1_gym.envs.config import (
    apply_config_snapshot, build_roboduet_config, cfg_to_dict,
    recompute_observation_dims, restore_dog_observation_layout,
)
from go1_gym.envs.config.core import dog_obs_dim_parts, sum_dim_parts


def build(**kwargs):
    return build_roboduet_config(Namespace(num_envs=8, robot="go2_x5", **kwargs))


def noheight_cfg(*, keep_command=True, response=False):
    cfg = build(dyna_gait=True)
    cfg.dog.observation_layout_version = 3
    cfg.dog.omit_height = True
    cfg.dog.omit_height_command = not keep_command
    cfg.dog.observe_response_model = response
    recompute_observation_dims(cfg)
    return cfg


def test_height_omission_drops_measured_height_and_its_error():
    baseline = build(dyna_gait=True)
    baseline.dog.observe_response_model = False
    recompute_observation_dims(baseline)
    full = dog_obs_dim_parts(baseline)

    cfg = noheight_cfg(keep_command=True)
    parts = dog_obs_dim_parts(cfg)
    assert parts["body_pose_actual"] == full["body_pose_actual"] - 1
    assert parts["body_pose_error"] == full["body_pose_error"] - 1
    # The command is not a measurement, so keeping it is the whole point of
    # omit_height_command=False.
    assert parts["dog_commands"] == full["dog_commands"]
    assert cfg.dog.dog_num_observations == baseline.dog.dog_num_observations - 2


def test_omit_height_command_also_drops_the_commanded_height():
    kept = dog_obs_dim_parts(noheight_cfg(keep_command=True))
    dropped = dog_obs_dim_parts(noheight_cfg(keep_command=False))
    assert dropped["dog_commands"] == kept["dog_commands"] - 1


def test_height_omission_requires_layout_version_three():
    cfg = noheight_cfg()
    cfg.dog.observation_layout_version = 2
    with pytest.raises(ValueError, match="layout version 3"):
        dog_obs_dim_parts(cfg)


def test_unsupported_layout_version_is_rejected():
    cfg = build()
    cfg.dog.observation_layout_version = 4
    with pytest.raises(ValueError, match="Unsupported dog observation layout version"):
        dog_obs_dim_parts(cfg)


@pytest.mark.parametrize("keep_command", [True, False])
def test_restore_recovers_the_noheight_frame(keep_command):
    cfg = noheight_cfg(keep_command=keep_command)
    snapshot = cfg_to_dict(cfg)

    restored = build(dyna_gait=True)
    apply_config_snapshot(restored, snapshot)
    restore_dog_observation_layout(restored, snapshot)

    assert restored.dog.observation_layout_version == 3
    assert restored.dog.omit_height
    assert restored.dog.omit_height_command == (not keep_command)
    assert restored.dog.dog_num_observations == cfg.dog.dog_num_observations
    assert restored.dog.dog_num_obs_history == cfg.dog.dog_num_obs_history


def test_response_block_absence_is_recovered_from_the_recorded_width():
    # Checkpoints that predate the R7.1 block never recorded the flag, and
    # inheriting today's default would hand the actor 22 columns it never saw.
    cfg = noheight_cfg(response=False)
    snapshot = cfg_to_dict(cfg)
    del snapshot["dog"]["observe_response_model"]

    restored = build(dyna_gait=True)
    apply_config_snapshot(restored, snapshot)
    restore_dog_observation_layout(restored, snapshot)

    assert restored.dog.observe_response_model is False
    assert restored.dog.dog_num_observations == cfg.dog.dog_num_observations
    assert "reference_state" not in dog_obs_dim_parts(restored)


def test_present_response_block_is_recovered_from_the_recorded_width():
    cfg = noheight_cfg(response=True)
    snapshot = cfg_to_dict(cfg)
    del snapshot["dog"]["observe_response_model"]

    restored = build(dyna_gait=True)
    apply_config_snapshot(restored, snapshot)
    restore_dog_observation_layout(restored, snapshot)

    assert restored.dog.observe_response_model is True
    assert restored.dog.dog_num_observations == cfg.dog.dog_num_observations


def test_current_layout_is_unchanged_by_the_noheight_fields():
    cfg = build(dyna_gait=True)
    assert cfg.dog.omit_height is False
    assert cfg.dog.observe_response_model is True
    parts = dog_obs_dim_parts(cfg)
    assert parts["body_pose_actual"] == 3
    assert "reference_state" in parts
    assert sum_dim_parts(parts) == cfg.dog.dog_num_observations
