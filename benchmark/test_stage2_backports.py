"""Run separately in isaacgym: python -m pytest benchmark/test_stage2_backports.py -q."""
import isaacgym  # noqa: F401 -- must precede torch
from types import SimpleNamespace

import pytest
import torch

from benchmark.dog_policy.cli import _batched_cmd_fn, _plan_point_batches
from benchmark.dog_policy.evaluation import Accumulator, PolicyHandle, set_vel_cmd, _apply_benchmark_env_overrides
from go1_gym.envs.config import build_roboduet_config
from go1_gym.envs.roboduet.legged_robot import LeggedRobot
from go1_gym.response.grouping import EnvGrouping
from go1_gym.utils.global_switch import global_switch


def test_benchmark_disables_training_command_writers_without_changing_observations():
    cfg = build_roboduet_config()
    width = cfg.dog.dog_num_observations
    _apply_benchmark_env_overrides(cfg, 32, 4)
    assert not cfg.response.grouping.enabled
    assert not cfg.response.excitation.enabled
    assert cfg.dog.dog_num_observations == width


@pytest.mark.parametrize("mode", ["benchmark", "sim2real", "none"])
def test_benchmark_domain_recipe_is_independent_of_checkpoint_mode(mode):
    from go1_gym.envs.config import cfg_to_dict
    from go1_gym.envs.config.domain_randomization import resolve_domain_randomization
    cfg = build_roboduet_config()
    cfg.domain_rand.mode = mode
    cfg.domain_rand.stage1_arm.link_mass_range = [99., 100.]
    _apply_benchmark_env_overrides(cfg, 32, 4)
    reference = build_roboduet_config()
    _apply_benchmark_env_overrides(reference, 32, 4)
    assert cfg_to_dict(cfg.domain_rand) == cfg_to_dict(reference.domain_rand)
    assert not cfg.domain_rand.randomize_action_delay
    assert not cfg.domain_rand.randomize_mount_position
    assert not cfg.noise.add_noise
    cfg.domain_rand.randomize_dog_obs_latency = True
    assert resolve_domain_randomization(cfg).domain_rand.randomize_dog_obs_latency


def test_batched_commands_preserve_cells_and_unowned_columns():
    commands = torch.arange(12 * 11, dtype=torch.float).reshape(12, 11)
    env = SimpleNamespace(env=SimpleNamespace(commands_dog=commands))
    untouched = commands[:, 3:].clone()
    handles = [PolicyHandle('a', None, 0, 6, 2), PolicyHandle('b', None, 6, 12, 2)]
    points = [('slow', lambda e: set_vel_cmd(e, .2, 0, 0), {}),
              ('fast', lambda e: set_vel_cmd(e, .8, 0, 0), {})]
    apply, cells = _batched_cmd_fn(env, handles, points)
    commands[:, :3] = -100
    apply(env)
    assert cells == [(0, 2), (2, 4), (6, 8), (8, 10)]
    torch.testing.assert_close(commands[:, 0], torch.tensor([.2, .2, .8, .8, .8, .8] * 2))
    torch.testing.assert_close(commands[:, 3:], untouched)


def test_batch_splits_on_global_arm_intensity_and_capacity():
    points = [(str(i), None, {'arm_intensity': intensity})
              for i, intensity in enumerate([0., 0., 0., .5, .5, 1.])]
    batches = _plan_point_batches(points, 0., 2, 'B')
    assert [(len(points), intensity) for points, intensity in batches] == [(2, 0.), (1, 0.), (2, .5), (1, 1.)]


def test_pooled_metrics_match_independent_accumulators():
    pool = Accumulator(6, 'cpu')
    separate = [Accumulator(2, 'cpu') for _ in range(3)]
    for step in range(4):
        values = torch.arange(6, dtype=torch.float) + step
        for acc, vals in [(pool, values)] + [(acc, values[2*i:2*i+2]) for i, acc in enumerate(separate)]:
            acc.add_val('value', vals)
            acc.add_count('events', vals > 4)
            acc.tick()
    for i, expected in enumerate(separate):
        actual = pool.view(2*i, 2*i+2)
        for method in ['rmse', 'mean', 'std']:
            assert getattr(actual, method)('value') == pytest.approx(getattr(expected, method)('value'))
        assert actual.total_count('events') == expected.total_count('events')


@pytest.fixture
def robot(monkeypatch):
    monkeypatch.setattr(global_switch, 'count', 8000)
    env = LeggedRobot.__new__(LeggedRobot)
    env.cfg = build_roboduet_config()
    env.device = 'cpu'
    env.num_envs = 32
    env.robustness_push_mask = torch.zeros(32, dtype=torch.bool)
    env.dt = .02
    env.cfg.domain_rand.push_interval_range = [2, 8]
    env.episode_length_buf = torch.full((32,), 10, dtype=torch.long)
    env.next_push_step = torch.full((32,), 10, dtype=torch.long)
    env.root_states = torch.zeros(32, 13)
    env.root_states[:, 6] = 1
    env.base_quat = env.root_states[:, 3:7]
    env.is_nominal_twin = torch.arange(32) % 4 == 0
    env.domain_disturbance_intensity = 0.
    env.sim = None
    env.writes = []
    env.gym = SimpleNamespace(set_actor_root_state_tensor_indexed=lambda *args: env.writes.append(args[2].clone()))
    monkeypatch.setattr('go1_gym.envs.roboduet.legged_robot.gymtorch.unwrap_tensor', lambda t: t)
    env._arm_check_termination_hook = lambda: None
    return env


def test_pushes_keep_r8_gate_and_nominal_twins(robot):
    ids = torch.arange(robot.num_envs)
    before = robot.root_states.clone()
    robot._push_robots(ids, robot.cfg)
    assert not robot.writes
    assert not robot.robustness_push_mask.any()
    assert torch.all(robot.next_push_step > robot.episode_length_buf)
    assert torch.unique(robot.next_push_step).numel() > 1
    torch.testing.assert_close(robot.root_states, before)
    robot.domain_disturbance_intensity = .5
    robot.next_push_step[:] = robot.episode_length_buf
    robot._push_robots(ids, robot.cfg)
    torch.testing.assert_close(robot.root_states[robot.is_nominal_twin], before[robot.is_nominal_twin])
    assert torch.equal(robot.writes[0].long(), ids[~robot.is_nominal_twin])
    assert torch.equal(robot.robustness_push_mask, ~robot.is_nominal_twin)
    assert robot.root_states[:, 7:9].abs().max() <= .5 * robot.cfg.domain_rand.max_push_vel_xy
    assert robot.root_states[:, 10:13].abs().max() <= .5 * robot.cfg.domain_rand.max_push_ang_vel
    robot._push_robots(ids, robot.cfg)
    assert len(robot.writes) == 1  # no duplicate impulse at the same step


def test_push_clock_restarts_for_only_reset_envs(robot):
    robot.cfg.domain_rand.push_interval_range = [7, 7]
    before = robot.next_push_step.clone()
    robot.episode_length_buf[[1, 5]] = 0
    robot._resample_push_interval(torch.tensor([1, 5]))
    assert robot.next_push_step[[1, 5]].tolist() == [7, 7]
    before[[1, 5]] = 7
    assert torch.equal(robot.next_push_step, before)


def test_twin_reset_resynchronizes_surviving_group_members(robot):
    robot.grouping = EnvGrouping(32, group_size=4, pool_envs=28, device='cpu')
    robot.is_nominal_twin = robot.grouping.is_twin
    robot.gait_indices = torch.full((32,), .6)
    robot._reset_gait_phase(torch.tensor([0, 5, 31]))
    assert robot.gait_indices[:4].tolist() == [0.] * 4
    torch.testing.assert_close(robot.gait_indices[4:31], torch.full((27,), .6))
    assert robot.gait_indices[31] == 0.


def test_attitude_grace_does_not_suppress_timeout_or_height(robot):
    robot.cfg.env.max_episode_length = 100
    robot.cfg.rewards.use_terminal_roll_pitch = True
    robot.cfg.rewards.terminal_body_ori = .5
    robot.cfg.rewards.terminal_roll_pitch_grace_s = 1.
    robot.cfg.rewards.use_terminal_body_height = True
    robot.cfg.rewards.terminal_body_height = .17
    robot.measured_heights = torch.zeros(32, 1)
    robot.root_states[:, 2] = .4
    robot.base_quat[:, 0] = torch.sin(torch.tensor(.5))
    robot.base_quat[:, 3] = torch.cos(torch.tensor(.5))
    robot.episode_length_buf[:4] = torch.tensor([50, 51, 101, 1])
    robot.root_states[3, 2] = .1
    robot.check_termination()
    assert robot.reset_buf[:4].tolist() == [False, True, True, True]
    assert robot.time_out_buf[:4].tolist() == [False, False, True, False]
    robot.cfg.rewards.use_terminal_roll_pitch = False
    robot.check_termination()
    assert robot.reset_buf[:4].tolist() == [False, False, True, True]


def test_source_reward_weights_and_raibert_presets():
    from go1_gym.envs.config import set_raibert_form, validate_raibert_form, resolve_reward_scales
    cfg = build_roboduet_config()
    assert cfg.reward_scales.raibert_heuristic == -1.
    assert cfg.wbc.reward_scales.raibert_heuristic == -1.
    assert cfg.reward_scales.feet_contact_forces == -.01
    assert cfg.reward_scales.feet_impact_vel == .4
    assert cfg.rewards.feet_impact_vel_sigma == .8
    assert not hasattr(cfg.reward_scales, 'raibert_sigma')
    assert resolve_reward_scales(cfg)['raibert_heuristic']['active']
    set_raibert_form(cfg, 'exp')
    assert cfg.reward_scales.raibert_heuristic == .4
    assert cfg.wbc.reward_scales.raibert_heuristic == .2
    validate_raibert_form(cfg)
    cfg.reward_scales.raibert_heuristic = -1.
    with pytest.raises(ValueError, match='positive'):
        validate_raibert_form(cfg)


def test_exponential_impact_rewards_soft_touchdown():
    from go1_gym.envs.rewards.rewards import Rewards
    cfg = build_roboduet_config()
    env = SimpleNamespace(cfg=cfg, num_envs=3, feet_indices=torch.arange(4),
                          prev_foot_velocities=torch.zeros(3, 4, 3),
                          contact_forces=torch.zeros(3, 4, 3))
    env.contact_forces[:, :, 2] = 10
    env.prev_foot_velocities[1, 0, 2] = -.1
    env.prev_foot_velocities[2, 0, 2] = -.2
    actual = Rewards(env)._reward_feet_impact_vel()
    torch.testing.assert_close(actual, torch.exp(-torch.tensor([0., .01, .04]) / .8))
    assert actual[0] > actual[1] > actual[2]


def test_raibert_exp_matches_quadratic_geometry():
    from go1_gym.envs.rewards.rewards import Rewards
    cfg = build_roboduet_config()
    env = SimpleNamespace(cfg=cfg, num_envs=2, device='cpu',
                          base_pos=torch.zeros(2, 3), base_quat=torch.tensor([[0.,0.,0.,1.]]*2),
                          foot_positions=torch.zeros(2,4,3), foot_indices=torch.zeros(2,4),
                          commands_dog=torch.zeros(2,6))
    rewards = Rewards(env)
    quadratic = rewards._reward_raibert_heuristic()
    cfg.rewards.raibert_form = 'exp'
    torch.testing.assert_close(rewards._reward_raibert_heuristic(), torch.exp(-quadratic / .35))


def test_push_curriculum_source_timing_and_r8_gate(robot, monkeypatch):
    for iteration, expected in [(0, 0.), (4000, .5), (8000, 1.), (16000, 1.)]:
        monkeypatch.setattr(global_switch, 'count', iteration)
        assert robot._get_push_curriculum_intensity() == expected
    robot.domain_disturbance_intensity = 0.
    robot._push_robots(torch.arange(32), robot.cfg)
    assert not robot.writes


def test_reset_mixture_preserves_training_partition_and_ramp():
    from go1_gym.envs.roboduet.robustness import FixedResetMixture
    cfg = build_roboduet_config()
    cfg.terrain.reset_mix_hard_fraction = .25
    rng = torch.random.get_rng_state().clone()
    mixture = FixedResetMixture(cfg.terrain, 40, 32, 'cpu')
    other = FixedResetMixture(cfg.terrain, 48, 32, 'cpu')
    assert torch.equal(rng, torch.random.get_rng_state())
    assert torch.equal(mixture.hard[:32], other.hard[:32])
    assert mixture.hard[:32].sum() == 8
    ids = torch.arange(32)
    early = mixture.limit(ids, 4000).flatten()
    late = mixture.limit(ids, 12000).flatten()
    torch.testing.assert_close(early, torch.full((32,), cfg.terrain.reset_mix_easy_tilt_rad))
    torch.testing.assert_close(late[mixture.hard[:32]], torch.full((8,), cfg.terrain.reset_mix_hard_tilt_rad))
    torch.testing.assert_close(late[~mixture.hard[:32]], early[~mixture.hard[:32]])


def test_old_snapshot_and_benchmark_disable_reset_mixture():
    from go1_gym.envs.config import apply_config_snapshot, cfg_to_dict
    cfg = build_roboduet_config()
    snapshot = cfg_to_dict(cfg)
    snapshot['terrain'].pop('reset_mode')
    snapshot['terrain'].pop('robustness_metrics')
    apply_config_snapshot(cfg, snapshot)
    assert cfg.terrain.reset_mode == 'legacy'
    assert not cfg.terrain.robustness_metrics
    cfg = build_roboduet_config()
    _apply_benchmark_env_overrides(cfg, 32, 4)
    assert cfg.terrain.reset_mode == 'legacy'
    assert not cfg.terrain.robustness_metrics
    assert not cfg.terrain.reset_curriculum


def test_robustness_metrics_count_failed_and_ongoing_steps_then_drain():
    from go1_gym.envs.roboduet.robustness import RobustnessMetrics
    metrics = RobustnessMetrics(torch.tensor([False, True]), .02, .04)
    error = torch.tensor([[1., 2., 3.], [3., 4., 5.]])
    yes = torch.tensor([True, True])
    no = ~yes
    metrics.update(error, torch.tensor([1, 3]), torch.tensor([False, True]), no, no, yes, no)
    values = metrics.pop()
    assert values['TrackingTime/all/sample_count'] == 2
    assert values['TrackingTime/all/vx_mae_mps'] == 2
    assert values['TrackingAge/easy/early/sample_count'] == 1
    assert values['TrackingAge/hard/late/sample_count'] == 1
    assert values['Termination/all/failures_count'] == 1
    assert metrics.pop()['TrackingTime/all/sample_count'] == 0
