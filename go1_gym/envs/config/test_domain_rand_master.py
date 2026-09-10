"""DR mode contracts, independent of IsaacGym and GPUs."""

from argparse import Namespace

import pytest

from go1_gym.envs.config import (
    apply_config_snapshot, build_roboduet_config, cfg_to_dict,
)
from go1_gym.envs.config.domain_randomization import resolve_domain_randomization


def build(**kwargs):
    return build_roboduet_config(Namespace(num_envs=8, robot="go2_x5", **kwargs))


def test_off_masks_disturbances_without_changing_task_or_layout():
    cfg = build(domain_rand_mode="none")
    cfg.domain_rand.dog_obs_frame_drop_prob = 0.5
    cfg.domain_rand.randomize_end_effector_force = True
    before = cfg_to_dict(cfg)
    effective = resolve_domain_randomization(cfg)
    dr = effective.domain_rand
    assert not dr.randomize_action_delay
    assert not dr.randomize_dog_obs_latency
    assert dr.dog_obs_frame_drop_prob == 0
    assert not dr.push_robots
    assert not dr.randomize_end_effector_force
    assert not dr.randomize_mount_position
    assert not dr.randomize_mount_rotation
    assert not effective.noise.add_noise
    for stage in (dr, dr.stage1_arm, dr.stage2_arm):
        assert all(not value for key, value in vars(stage).items()
                   if key.startswith("randomize_"))
    after = cfg_to_dict(effective)
    for section in before:
        if section not in ("domain_rand", "noise"):
            assert after[section] == before[section], section
    assert cfg_to_dict(cfg) == before  # Recipe is still serializable/reusable.
    cfg.domain_rand.mode = "sim2real"
    assert resolve_domain_randomization(cfg).domain_rand.stage1_arm.randomize_ee_payload


def test_enabled_respects_individual_opt_outs_and_builds_are_independent():
    cfg = build(domain_rand_mode="sim2real")
    cfg.domain_rand.randomize_action_delay = False
    assert resolve_domain_randomization(cfg) is cfg
    assert not resolve_domain_randomization(cfg).domain_rand.randomize_action_delay
    assert not hasattr(build().domain_rand, "enabled")
    assert resolve_domain_randomization(None) is None


def test_checkpoint_roundtrip_and_old_snapshot_fallback():
    cfg = build(domain_rand_mode="none")
    snapshot = cfg_to_dict(cfg)
    restored = build()
    apply_config_snapshot(restored, snapshot)
    assert not resolve_domain_randomization(restored).noise.add_noise
    assert restored.domain_rand.randomize_action_delay  # Stored recipe intact.
    del snapshot["domain_rand"]["mode"]
    apply_config_snapshot(restored, snapshot)
    assert restored.domain_rand.mode == "sim2real"
    assert resolve_domain_randomization(restored).domain_rand.randomize_action_delay


@pytest.mark.parametrize("traj_tracking", [False, True])
def test_benchmark_retains_task_dynamics_and_removes_hardware_uncertainty(traj_tracking):
    cfg = build(domain_rand_mode="benchmark", traj_tracking=traj_tracking)
    before = cfg_to_dict(cfg)
    effective = resolve_domain_randomization(cfg)
    dr = effective.domain_rand
    for name in ("randomize_friction", "randomize_restitution", "randomize_base_mass",
                 "randomize_com_displacement", "randomize_motor_strength",
                 "randomize_Kp_factor", "randomize_Kd_factor", "push_robots"):
        assert getattr(dr, name) == getattr(cfg.domain_rand, name)
        assert getattr(dr, name)
    assert dr.stage1_arm.randomize_ee_payload
    assert dr.stage1_arm.randomize_link_mass
    assert dr.stage1_arm.randomize_link_com
    assert dr.stage2_arm.randomize_Kp_factor
    for name in ("randomize_mount_position", "randomize_mount_rotation",
                 "randomize_motor_offset", "randomize_gravity",
                 "randomize_action_delay", "randomize_dog_obs_latency"):
        assert not getattr(dr, name)
    assert not dr.stage1_arm.randomize_motor_offset
    assert not dr.stage2_arm.randomize_motor_offset
    assert not effective.noise.add_noise
    after = cfg_to_dict(effective)
    for section in before:
        if section not in ("domain_rand", "noise"):
            assert before[section] == after[section], section
    assert cfg_to_dict(cfg) == before
    cfg.domain_rand.mode = "sim2real"
    assert resolve_domain_randomization(cfg).domain_rand.randomize_mount_position
    assert resolve_domain_randomization(cfg).noise.add_noise


@pytest.mark.parametrize("mode", ["benchmark", "sim2real", "none"])
def test_mode_snapshot_roundtrip(mode):
    cfg = build(domain_rand_mode=mode)
    restored = build()
    apply_config_snapshot(restored, cfg_to_dict(cfg))
    assert restored.domain_rand.mode == mode
    assert cfg_to_dict(resolve_domain_randomization(restored)) == cfg_to_dict(resolve_domain_randomization(cfg))


def test_default_and_invalid_modes():
    assert build().domain_rand.mode == "benchmark"
    with pytest.raises(ValueError, match="domain_rand.mode"):
        build(domain_rand_mode="typo")


@pytest.mark.parametrize("mode", ["benchmark", "sim2real", "none"])
def test_evaluation_recipe_overrides_checkpoint_and_preserves_scenario(mode):
    from go1_gym.envs.config.domain_randomization import configure_benchmark_domain_randomization
    cfg = build(domain_rand_mode=mode)
    cfg.domain_rand.stage1_arm.link_mass_range = [100, 200]
    cfg.domain_rand.randomize_friction = False
    recipe = build()
    configure_benchmark_domain_randomization(cfg, recipe)
    assert not hasattr(cfg.domain_rand, "enabled")
    assert cfg.domain_rand.stage1_arm.link_mass_range == recipe.domain_rand.stage1_arm.link_mass_range
    assert cfg.domain_rand.randomize_friction
    assert not cfg.domain_rand.randomize_action_delay
    # A scenario may explicitly test latency even though benchmark training
    # defaults to clean sensing. Environment construction must preserve it.
    cfg.domain_rand.randomize_dog_obs_latency = True
    assert resolve_domain_randomization(cfg).domain_rand.randomize_dog_obs_latency
    assert cfg_to_dict(recipe) == cfg_to_dict(build())


@pytest.mark.parametrize("legacy_enabled", [False, True])
@pytest.mark.parametrize("mode", [None, "benchmark", "sim2real", "none"])
def test_legacy_enabled_migrates_without_restoring_deleted_field(legacy_enabled, mode):
    from copy import deepcopy
    snapshot = cfg_to_dict(build())
    dr = snapshot["domain_rand"]
    dr["enabled"] = legacy_enabled
    if mode is None:
        dr.pop("mode")
    else:
        dr["mode"] = mode
    original = deepcopy(snapshot)
    cfg = build()
    apply_config_snapshot(cfg, snapshot)
    assert cfg.domain_rand.mode == ("none" if not legacy_enabled else mode or "sim2real")
    assert not hasattr(cfg.domain_rand, "enabled")
    assert "enabled" not in cfg_to_dict(cfg)["domain_rand"]
    assert snapshot == original


def test_partial_snapshot_does_not_reset_mode():
    cfg = build(domain_rand_mode="benchmark")
    apply_config_snapshot(cfg, {"domain_rand": {"push_robots": False}})
    assert cfg.domain_rand.mode == "benchmark"
    apply_config_snapshot(cfg, {"domain_rand": {"enabled": False}})
    assert cfg.domain_rand.mode == "none"
    assert not hasattr(cfg.domain_rand, "enabled")
