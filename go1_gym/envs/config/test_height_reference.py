"""Geometry, reset freshness, and compatibility of terrain-relative heights."""
import ast
from pathlib import Path
from types import SimpleNamespace

import isaacgym
import torch
from go1_gym.utils.math_utils import quat_apply_yaw
from go1_gym.utils.height_sampling import sample_triangle_heights
from go1_gym.envs.config import build_roboduet_config, apply_config_snapshot, cfg_to_dict


def make_env():
    path = Path(__file__).resolve().parents[3] / 'go1_gym/envs/roboduet/legged_robot.py'
    names = ('_uses_terrain_height', '_ground_height_at', '_body_ground_reference', '_body_height', '_foot_clearance')
    nodes = [n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = {'torch': torch, 'quat_apply_yaw': quat_apply_yaw, 'sample_triangle_heights': sample_triangle_heights}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
    env = type('HeightEnv', (), {name: ns[name] for name in names})()
    env.cfg = build_roboduet_config()
    env.cfg.terrain.border_size = 0
    env.terrain = SimpleNamespace(cfg=env.cfg.terrain)
    env.num_envs, env.device = 2, 'cpu'
    env.base_pos = torch.tensor([[1.,1.,.3],[2.,2.,.35]])
    env.base_quat = torch.tensor([[0.,0.,0.,1.]]).repeat(2,1)
    env.height_samples = torch.zeros(40,40)
    env.foot_positions = env.base_pos[:,None,:].repeat(1,4,1)
    env.foot_positions[:,:,2] = .06
    return env


def test_triangles_not_bilinear_and_border_clamping():
    # Saddle cell distinguishes the 00--11 triangles from bilinear interpolation.
    heights = torch.tensor([[0.,0.],[0.,1.]])
    xy = torch.tensor([[.75,.25],[.25,.75],[.5,.5],[-1.,-1.],[2.,2.]])
    actual = sample_triangle_heights(heights,xy,1.,1.,0.)
    torch.testing.assert_close(actual,torch.tensor([.25,.25,.5,0.,1.]))
    shifted = sample_triangle_heights(heights,xy-2.,1.,1.,2.)
    torch.testing.assert_close(actual,shifted)


def test_vertical_translation_invariance_and_reset_freshness():
    env = make_env()
    before_body, before_feet = env._body_height(), env._foot_clearance()
    env.height_samples += 20  # 20 * .005 = .1 m raised ground
    env.base_pos[:,2] += .1
    env.foot_positions[:,:,2] += .1
    torch.testing.assert_close(env._body_height(),before_body)
    torch.testing.assert_close(env._foot_clearance(),before_feet)
    # No policy-step advancement: emulate a reset writing a new root height.
    env.base_pos[0,2] += .2
    torch.testing.assert_close(env._body_height(),before_body+torch.tensor([.2,0.]))


def test_each_foot_uses_its_own_ground_and_plane_matches_legacy():
    env = make_env()
    env.height_samples[:] = torch.arange(40)[:,None] * 2  # slope .1 m/m
    env.foot_positions[0,:,0] = torch.tensor([1.,1.2,1.4,1.6])
    torch.testing.assert_close(env._foot_clearance()[0],torch.tensor([-.04,-.06,-.08,-.10]))
    env.cfg.terrain.mesh_type = 'plane'
    torch.testing.assert_close(env._body_height(),env.base_pos[:,2])
    torch.testing.assert_close(env._foot_clearance(),env.foot_positions[:,:,2])


def test_snapshot_semantics_and_partial_updates():
    cfg = build_roboduet_config()
    assert cfg.terrain.height_reference == 'terrain'
    snapshot = cfg_to_dict(cfg)
    restored = build_roboduet_config()
    apply_config_snapshot(restored,snapshot)
    assert restored.terrain.height_reference == 'terrain'
    snapshot['terrain'].pop('height_reference')
    apply_config_snapshot(restored,snapshot)
    assert restored.terrain.height_reference == 'world'
    apply_config_snapshot(cfg,{'terrain':{'num_rows':3}})
    assert cfg.terrain.height_reference == 'terrain'


def test_height_reward_translation_invariance():
    env = make_env()
    env.commands_dog = torch.zeros(2,6)
    path = Path(__file__).resolve().parents[3] / 'go1_gym/envs/rewards/rewards.py'
    node = next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.FunctionDef) and n.name=='_reward_jump')
    ns = {'torch':torch}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
    reward = lambda: ns['_reward_jump'](SimpleNamespace(env=env))
    before = reward()
    env.height_samples += 20
    env.base_pos[:,2] += .1
    torch.testing.assert_close(reward(),before)


def test_stage2_dog_loader_restores_height_semantics():
    from go1_gym.envs.config import restore_dog_observation_layout
    cfg = build_roboduet_config()
    snapshot = cfg_to_dict(cfg)
    snapshot['terrain'].pop('height_reference')
    restore_dog_observation_layout(cfg, snapshot)
    assert cfg.terrain.height_reference == 'world'
    snapshot['terrain']['height_reference'] = 'terrain'
    restore_dog_observation_layout(cfg, snapshot)
    assert cfg.terrain.height_reference == 'terrain'
