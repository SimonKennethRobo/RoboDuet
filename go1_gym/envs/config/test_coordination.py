"""Behavioral checks for the six-card recipes and actual sampler implementations."""
import isaacgym  # IsaacGym must be imported before torch in this repository.
import ast
import json
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from go1_gym.envs.config import apply_config_snapshot, build_roboduet_config, cfg_to_dict
from go1_gym.envs.config.coordination import DEFAULT_CONFIG, validate_coordination
from go1_gym.envs.roboduet.coordination_arm import CoordinationArm
from go1_gym.envs.roboduet.coordination_sampling import CoordinationCommands, velocity_window_range


def config(name="C", count=100):
    return build_roboduet_config(Namespace(num_envs=count, robot="go2_x5", train_stage="stage1",
                                         dyna_gait=True, experiment=name))


@pytest.mark.parametrize("name", list("ABCDEF"))
def test_recipes_resolve_and_snapshot_roundtrip(name):
    cfg = config(name)
    assert cfg.terrain.mesh_type == "trimesh" and cfg.terrain.reset_mix_hard_fraction == 0.2
    assert cfg.domain_rand.push_robots and cfg.domain_rand.max_push_vel_xy == 1.0
    assert cfg.commands.limit_body_pitch == cfg.commands.body_pitch_range == [-0.3, 0.3]
    assert cfg.commands.limit_body_roll == cfg.commands.body_roll_range == [-0.2, 0.2]
    assert cfg.commands.limit_gait_frequency == cfg.commands.gait_frequency_cmd_range == [2.0, 3.5]
    assert cfg.dog.dog_num_observations == 90 and cfg.dog.dog_num_observation_history == 30
    restored = build_roboduet_config()
    apply_config_snapshot(restored, cfg_to_dict(cfg))
    assert cfg_to_dict(restored) == cfg_to_dict(cfg)


def test_old_snapshot_disables_new_behavior_even_on_reused_config():
    old = cfg_to_dict(build_roboduet_config())
    old["commands"].pop("coordination")
    old["env"].pop("coordination_arm")
    old["rewards"].pop("attitude_command_convention")
    old.pop("coordination_experiment")
    new = config()
    apply_config_snapshot(new, old)
    assert not new.commands.coordination.enabled and not new.env.coordination_arm.enabled
    assert new.rewards.attitude_command_convention == "legacy"
    assert not new.coordination_experiment


def make_env(cfg):
    n = cfg.env.num_envs
    return SimpleNamespace(cfg=cfg, num_envs=n, device="cpu", common_step_counter=0,
        commands_dog=torch.zeros(n, 11), reset_buf=torch.zeros(n, dtype=torch.bool),
        pitch=torch.zeros(n), roll=torch.zeros(n), gravity_vec=torch.tensor([[0., 0., -1.]]).repeat(n, 1),
        command_sums={key: torch.zeros(n) for key in ("tracking_lin_vel", "tracking_ang_vel", "ep_timesteps")},
        pretrained_reward_scales={"tracking_lin_vel": 0.04, "tracking_ang_vel": 0.025},
        curriculum_thresholds={"tracking_lin_vel": 0.8, "tracking_ang_vel": 0.7},
        _update_reset_curriculum=lambda *_: None)


def test_window_schedule_and_fixed_short_cohort():
    cfg = config()
    c = cfg.commands.coordination
    assert velocity_window_range(c, 7999) == [10, 10]
    assert velocity_window_range(c, 12000) == [9, 9]
    assert velocity_window_range(c, 20000) == [7, 7]
    assert velocity_window_range(c, 24000) == [6, 10]
    sampler = CoordinationCommands(cfg, 100, "cpu", 0.02)
    env = make_env(cfg)
    ids = torch.arange(100)
    sampler.reset(env, ids, 0)
    deadlines = sampler.next_step.clone()
    # Changing iteration cannot retroactively shorten a live window.
    env.common_step_counter = 1
    sampler.after_reward(env, 24000)
    torch.testing.assert_close(sampler.next_step, deadlines)
    sampler.reset(env, ids, 24000)
    assert sampler.short.sum() == 30
    assert (sampler.window_steps[sampler.short, 0] >= 150).all()
    assert (sampler.window_steps[sampler.short, 0] <= 200).all()
    assert (sampler.window_steps[~sampler.short, 0] >= 300).all()
    assert (sampler.window_steps[:, 1] >= 400).all()
    assert (sampler.window_steps[:, 2] >= 750).all()


def test_early_reset_actual_denominator_and_pose_does_not_clear_velocity():
    cfg = config()
    cfg.commands.coordination.standing_probability = 0
    sampler = CoordinationCommands(cfg, 100, "cpu", 0.02)
    env = make_env(cfg)
    ids = torch.arange(100)
    sampler.reset(env, ids, 0)
    env.commands_dog[:, 0] = 0.5
    env.command_sums["ep_timesteps"][:] = 7
    env.command_sums["tracking_lin_vel"][:] = 7 * 0.04
    env.command_sums["tracking_ang_vel"][:] = 7 * 0.025
    captured = []
    sampler.curriculum.update = lambda bins, rewards, thresholds, **kwargs: captured.append((bins, rewards))
    env.common_step_counter = 7
    sampler._start(env, ids, 1, 0)
    assert (env.command_sums["ep_timesteps"] == 7).all() and not captured
    sampler.reset(env, ids[:5], 0)
    assert len(captured) == 1
    torch.testing.assert_close(captured[0][1][0], torch.full((5,), 0.04))
    assert (env.command_sums["ep_timesteps"][:5] == 0).all()
    assert (env.command_sums["ep_timesteps"][5:] == 7).all()
    assert (sampler.next_step[5:, 0] == 500).all()


def test_moving_frequency_restored_after_stand_without_gait_resampling():
    cfg = config()
    cfg.commands.coordination.standing_probability = 1
    sampler = CoordinationCommands(cfg, 100, "cpu", 0.02)
    env = make_env(cfg)
    ids = torch.arange(100)
    sampler.reset(env, ids, 0)
    assert (env.commands_dog[:, 6] == 0).all()
    before = sampler.walking_frequency.clone()
    cfg.commands.coordination.standing_probability = 0
    for _ in range(20):
        sampler._start(env, ids, 0, 0)
        moving = torch.norm(env.commands_dog[:, :3], dim=1) >= 0.1
        torch.testing.assert_close(env.commands_dog[moving, 6], before[moving])
        assert (env.commands_dog[moving, 6] >= 2).all()


def arm_env(cfg):
    env = make_env(cfg)
    n = cfg.env.num_envs
    env.num_actions_loco, env.num_actions_arm = 12, 6
    env.dof_pos = torch.zeros(n, 18)
    env.dof_vel = torch.zeros(n, 18)
    env.dof_pos_limits = torch.tensor([[-1., 1.]] * 18)
    env.default_dof_pos = torch.zeros(1, 18)
    env.actions = torch.zeros(n, 18)
    env.stage1_arm_fixed_dof_pos = torch.zeros(n, 6)
    env.stage1_arm_target_offset = torch.zeros(n, 6)
    env.stage1_arm_target_vel = torch.zeros(n, 6)
    env.stage1_arm_target_accel = torch.zeros(n, 6)
    env.torques = torch.zeros(n, 18)
    env.torque_limits = torch.full((18,), 20.)
    return env


def test_arm_mixture_limits_reset_and_curriculum():
    torch.manual_seed(42)
    cfg = config()
    env = arm_env(cfg)
    arm = CoordinationArm(cfg, 100, "cpu", 0.02)
    arm.reset(env, torch.arange(100))
    assert [(arm.mode == i).sum().item() for i in range(3)] == [80, 10, 10]
    arm.step(env, 0)
    assert (env.actions == 0).all()
    for step in range(1000):
        env.common_step_counter = step
        previous_q, previous_v = arm.q.clone(), arm.v.clone()
        arm.step(env, 1.0)
        assert arm.q.abs().max() <= 1.00001
        assert (arm.v.abs() <= arm.vmax + 1e-5).all()
        assert ((arm.v - previous_v).abs() <= arm.amax * 0.02 + 1e-5).all()
        torch.testing.assert_close(arm.q - previous_q, arm.v * 0.02, atol=1e-6, rtol=1e-4)
    untouched = arm.q[1:].clone()
    env.dof_pos[0, 12:] = 0.3
    arm.reset(env, torch.tensor([0]))
    assert (arm.q[0] == 0.3).all() and (arm.v[0] == 0).all()
    torch.testing.assert_close(arm.q[1:], untouched)


def test_rpy_reward_matches_combined_positive_and_negative_attitudes():
    path = Path(__file__).resolve().parents[1] / "rewards/rewards.py"
    node = next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.FunctionDef) and n.name == "_reward_orientation_control")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    env = make_env(config())
    pitch, roll = torch.linspace(-0.3, 0.3, 100), torch.linspace(0.2, -0.2, 100)
    env.commands_dog[:, 3], env.commands_dog[:, 4] = pitch, roll
    # Independent rotation matrices, using R = Ry(pitch) Rx(roll).
    gravity = []
    for p, r in zip(pitch, roll):
        ry = torch.tensor([[p.cos(), 0, p.sin()], [0, 1, 0], [-p.sin(), 0, p.cos()]])
        rx = torch.tensor([[1, 0, 0], [0, r.cos(), -r.sin()], [0, r.sin(), r.cos()]])
        gravity.append((ry @ rx).T @ torch.tensor([0., 0., -1.]))
    env.projected_gravity = torch.stack(gravity)
    reward = namespace["_reward_orientation_control"](SimpleNamespace(env=env))
    torch.testing.assert_close(reward, torch.zeros_like(reward), atol=1e-12, rtol=0)
    env.commands_dog[:, 3:5] *= -1
    assert (namespace["_reward_orientation_control"](SimpleNamespace(env=env)) > 0).all()


def test_arm_metrics_drain_after_ppo_inference_mode():
    cfg = config()
    env = arm_env(cfg)
    arm = CoordinationArm(cfg, 100, "cpu", 0.02)
    arm.reset(env, torch.arange(100))
    with torch.inference_mode():
        env.dof_vel[:, 12:] = 0.5
        arm.after_physics(env)
        env.dof_vel[:, 12:] = 0.6
        arm.after_physics(env)
    metrics = arm.metrics()
    assert metrics["ArmMotion/actual_abs_velocity_rad_s"] == pytest.approx(0.55)
    assert metrics["ArmMotion/actual_abs_acceleration_rad_s2"] == pytest.approx(5.0)
    assert metrics["ArmMotion/random_time_fraction"] == 0.8
    assert arm.metrics()["ArmMotion/actual_peak_velocity_rad_s"] == 0.0


def test_invalid_mixture_and_schedule_fail_early():
    cfg = config()
    cfg.env.coordination_arm.fractions = [0.8, 0.2, 0.1]
    with pytest.raises(ValueError, match="fractions"):
        validate_coordination(cfg)
    cfg = config()
    cfg.commands.coordination.velocity_schedule = [[0, 10], [0, 6]]
    with pytest.raises(ValueError, match="increase"):
        validate_coordination(cfg)


def test_training_json_and_explicit_cli_precedence(tmp_path):
    from scripts.auto_train import parse_args
    plan = json.loads(DEFAULT_CONFIG.read_text())
    plan["experiments"]["C"]["training"] = {"learning_rate": 0.0001, "schedule": "fixed"}
    path = tmp_path / "recipes.json"
    path.write_text(json.dumps(plan))
    args = parse_args(["--experiment", "C", "--experiment_config", str(path), "--train_stage", "stage1",
                       "--dyna_gait", "--num_envs", "32", "--num_learning_iterations", "7"])
    assert args.num_envs == 32 and args.num_learning_iterations == 7
    assert args.learning_rate == 0.0001 and args.lr_schedule == "fixed" and args.seed == 11
    assert parse_args([]).num_learning_iterations == 100000
    plan["training"]["lerning_rate"] = 0.1
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="Unknown training fields"):
        parse_args(["--experiment", "C", "--experiment_config", str(path)])
