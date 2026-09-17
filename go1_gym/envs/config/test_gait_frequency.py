"""Gait-frequency configuration and sampling regressions; no simulator needed."""

from argparse import Namespace

import numpy as np
import pytest

from go1_gym.envs.base.curriculum import Curriculum
from go1_gym.envs.config import (
    apply_config_snapshot, build_roboduet_config, cfg_to_dict,
)
from go1_gym.envs.config.wbc import ROBODUET_PROFILE


MODES = ["dyna_gait", "goal_reaching", "traj_tracking"]


def build(**kwargs):
    return build_roboduet_config(Namespace(num_envs=8, robot="go2_x5", **kwargs))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("legacy_min", [None, 0.0, 1.0])
def test_dynamic_modes_preserve_profile_frequency_bounds(mode, legacy_min):
    baseline = build()
    args = {mode: True}
    if legacy_min is not None:
        args["dyna_gait_min_frequency"] = legacy_min
    cfg = build(**args)

    assert cfg.commands.use_dynamic_gait
    assert cfg.commands.gait_frequency_cmd_range == baseline.commands.gait_frequency_cmd_range
    assert cfg.commands.limit_gait_frequency == baseline.commands.limit_gait_frequency


@pytest.mark.parametrize("mode", MODES)
def test_edited_profile_limits_reach_frequency_sampler(monkeypatch, mode):
    # Sampling ranges and curriculum limits are independently owned settings.
    monkeypatch.setitem(ROBODUET_PROFILE.overrides, "commands.gait_frequency_cmd_range", [2.6, 3.4])
    monkeypatch.setitem(ROBODUET_PROFILE.overrides, "commands.limit_gait_frequency", [2.5, 3.5])
    cfg = build(**{mode: True})
    assert cfg.commands.gait_frequency_cmd_range == [2.6, 3.4]
    assert cfg.commands.limit_gait_frequency == [2.5, 3.5]

    # Exercise the real sampler's gait marginal. The trainer constructs this
    # dimension from limit_gait_frequency, not gait_frequency_cmd_range.
    lo, hi = cfg.commands.limit_gait_frequency
    curriculum = Curriculum(
        seed=1234, gait_frequency=(lo, hi, cfg.commands.num_bins_gait_frequency))
    curriculum.set_to(np.array([lo]), np.array([hi]))
    frequencies = curriculum.sample(4096)[0][:, 0]
    assert np.all((frequencies >= 2.5) & (frequencies <= 3.5))
    assert np.any(frequencies < 2.6)
    assert np.any(frequencies > 3.4)


def test_checkpoint_frequency_contract_is_preserved():
    cfg = build(dyna_gait=True)
    snapshot = cfg_to_dict(cfg)
    restored = build()
    apply_config_snapshot(restored, snapshot)
    assert restored.commands.limit_gait_frequency == cfg.commands.limit_gait_frequency

    # Old policies were trained with these bounds: fixing new config builds
    # must not silently change their checkpoint-backed deployment contract.
    snapshot["commands"]["gait_frequency_cmd_range"] = [0.0, 4.0]
    snapshot["commands"]["limit_gait_frequency"] = [0.0, 4.0]
    apply_config_snapshot(restored, snapshot)
    assert restored.commands.gait_frequency_cmd_range == [0.0, 4.0]
    assert restored.commands.limit_gait_frequency == [0.0, 4.0]
