"""Execute simulator methods on tensor fixtures without loading IsaacGym.

AST extraction preserves the production method bodies; simulator calls are
replaced with a recording stub so curriculum, reward and twin behavior can be
checked on CPU alongside the response unit suite.
"""
import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from go1_gym.envs.config import build_roboduet_config
from go1_gym.envs.config.core import apply_config_snapshot, cfg_to_dict
from go1_gym.response import reward_terms
from go1_gym.response.test_response_config import _args

ROOT = Path(__file__).resolve().parents[2]


def method(path, cls, name, **namespace):
    tree = ast.parse((ROOT / path).read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    node = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[node], type_ignores=[])
    scope = dict(vars(reward_terms), torch=torch, **namespace)
    exec(compile(module, str(path), 'exec'), scope)
    return scope[name]


def env_method(name, **namespace):
    return method('go1_gym/envs/roboduet/legged_robot.py', 'LeggedRobot', name, **namespace)


def cfg():
    return build_roboduet_config(_args())


def test_profile_and_old_snapshot_push_schedule():
    config = cfg()
    assert not config.domain_rand.push_use_response_curriculum
    assert config.terrain.reset_mix_hard_fraction == 0.2
    snapshot = cfg_to_dict(config)
    apply_config_snapshot(config, snapshot)
    assert not config.domain_rand.push_use_response_curriculum
    del snapshot['domain_rand']['push_use_response_curriculum']
    apply_config_snapshot(config, snapshot)
    assert config.domain_rand.push_use_response_curriculum


def test_pose_command_is_allowed_but_slow_tip_and_body_rate_trigger_recovery():
    update = env_method('_update_response_soft_gate',
                        wrap_to_pi=lambda x: (x + torch.pi) % (2 * torch.pi) - torch.pi)
    env = NS(cfg=cfg(), dt=.02, feet_indices=[0],
             contact_forces=torch.zeros(3, 1, 3), foot_velocities=torch.zeros(3, 1, 3),
             base_lin_vel=torch.zeros(3, 3), prev_base_lin_vel=torch.zeros(3, 3),
             commands_dog=torch.zeros(3, 11), pitch=torch.tensor([.4, .3, 0.]),
             roll=torch.zeros(3), base_ang_vel=torch.zeros(3, 3),
             response_soft_gate_timer=torch.zeros(3, dtype=torch.long), soft_gate_hold_steps=50)
    env.commands_dog[0, 3] = .4  # commanded lean must not disable the reward
    env.base_ang_vel[2, 0] = 2.
    update(env)
    assert env.response_soft_gate.tolist() == [1., 0., 0.]
    update(env)  # sustained imbalance renews the hold
    assert env.response_soft_gate_timer.tolist() == [0, 50, 50]
    env.pitch[1] = 0
    env.base_ang_vel.zero_()
    for _ in range(49):
        update(env)
    assert env.response_soft_gate.tolist() == [1., 0., 0.]
    update(env)
    assert env.response_soft_gate.tolist() == [1., 1., 1.]


@pytest.mark.parametrize('iteration,expected', [(0, 0.), (2000, .25), (4000, .5), (8000, 1.)])
def test_early_push_ramp_preserves_twin_and_starts_recovery(iteration, expected):
    config = cfg()
    writes = []
    ramp = env_method('_get_push_curriculum_intensity', global_switch=NS(count=iteration))
    push = env_method('_push_robots',
                      torch_rand_float=lambda lo, hi, shape, device: torch.full(shape, hi),
                      gymtorch=NS(unwrap_tensor=lambda x: x))
    env = NS(cfg=config, domain_disturbance_intensity=0., device='cpu', sim=None,
             robustness_push_mask=torch.zeros(2, dtype=torch.bool),
             episode_length_buf=torch.ones(2, dtype=torch.long), next_push_step=torch.zeros(2),
             is_nominal_twin=torch.tensor([True, False]), root_states=torch.zeros(2, 13),
             response_soft_gate_timer=torch.zeros(2, dtype=torch.long), soft_gate_hold_steps=50,
             gym=NS(set_actor_root_state_tensor_indexed=lambda *args: writes.append(args)))
    env._get_push_curriculum_intensity = lambda: ramp(env)
    env._resample_push_interval = lambda ids, cfg: None
    push(env, torch.arange(2), config)
    assert env.root_states[0].count_nonzero() == 0
    assert env.response_soft_gate_timer[0] == 0
    assert env.root_states[1, 7].item() == pytest.approx(expected)
    assert len(writes) == int(expected > 0)
    assert env.response_soft_gate_timer[1].item() == (51 if expected else 0)
    config.domain_rand.push_use_response_curriculum = True
    env.root_states.zero_()
    writes.clear()
    push(env, torch.arange(2), config)
    assert not writes  # legacy stage gate remains available


@pytest.mark.parametrize('name', ['phase_variance', 'steady_gain', 'domain_consistency'])
def test_recovery_masks_reward_without_changing_raw_diagnostics(name):
    reward = method('go1_gym/envs/rewards/rewards.py', 'Rewards', '_reward_' + name)
    env = NS(response_oscillation=torch.ones(2, 5), response_delta_hat=torch.zeros(2, 5),
             response_channel_weights=torch.ones(5), response_soft_gate=torch.tensor([0., 1.]),
             response_phase_variance_mask=torch.ones(2, 5), response_steady_gain_mask=torch.ones(2, 5),
             response_detrended=torch.ones(2, 5), response_twin_detrended=torch.zeros(2, 5),
             response_consistency_gain=1., grouping=NS(valid=torch.ones(2)),
             commands_dog=torch.zeros(2, 11),
             response_ref=NS(gather_commands=lambda cmd: torch.zeros(2, 5)))
    values = reward(NS(env=env))
    assert values[0] == 0 and values[1] > 0
    assert env.response_phase_variance_mask.min() == 1
    assert env.response_steady_gain_mask.min() == 1
    assert env.grouping.valid.min() == 1
