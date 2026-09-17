"""CPU contracts for the selective rlmpc DR/terrain backport."""
import ast
from pathlib import Path
from types import SimpleNamespace

import isaacgym
import numpy as np
import pytest
import torch

from go1_gym.envs.config import build_roboduet_config
from go1_gym.utils.latency import LatencyBuffer
from go1_gym.utils.terrain import Terrain, roughness_tier_columns


def test_latency_per_env_and_reset_does_not_leak_previous_episode():
    ring = LatencyBuffer(2, 1, 2, 'cpu')
    ring.delays[:] = torch.tensor([0, 2])
    for value in (10., 20., 30.):
        ring.push(torch.full((2, 1), value))
    assert ring.read().flatten().tolist() == [30., 10.]
    ring.reset_idx(torch.tensor([1]), torch.tensor([[30.], [99.]]))
    assert ring.read().flatten().tolist() == [30., 99.]


def test_terrain_amplitudes_and_column_coverage():
    cfg = build_roboduet_config().terrain
    cfg.mesh_type = 'heightfield'
    cfg.num_rows, cfg.num_cols, cfg.border_size = 2, 6, 0
    terrain = Terrain(cfg, 12)
    bands = cfg.roughness_tier_columns
    assert bands == [[0, 1, 2], [3], [4, 5]]
    for tier, band in enumerate(bands):
        heights = terrain.height_field_raw[:, band[0] * cfg.width_per_env_pixels:(band[-1]+1) * cfg.width_per_env_pixels]
        assert np.max(np.abs(heights)) * cfg.vertical_scale <= cfg.roughness_tiers[tier] + 1e-6
        assert bool(np.any(heights)) == (tier > 0)


@pytest.mark.parametrize('cols,tiers,weights', [(2,3,None),(6,3,[1,-1,1]),(6,3,[1,float('nan'),1])])
def test_invalid_terrain_bands_fail_early(cols, tiers, weights):
    with pytest.raises(ValueError):
        roughness_tier_columns(cols, tiers, weights)


def test_sensor_cache_advances_once_and_reset_refills_same_step():
    # Exercise the actual environment method without creating a simulator.
    path = Path(__file__).resolve().parents[3] / 'go1_gym/envs/roboduet/wbc_env.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_dog_measurements')
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    read = namespace['_dog_measurements']
    ring = LatencyBuffer(1, 1, 2, 'cpu')
    ring.delays[:] = 1
    env = SimpleNamespace(cfg=SimpleNamespace(domain_rand=SimpleNamespace(randomize_dog_obs_latency=True,dog_obs_latency_jitter_steps=1)),
                          dog_obs_latency=ring, dog_obs_latency_fill=torch.ones(1,dtype=torch.bool),
                          _dog_latency_step=-1, common_step_counter=0)
    env._dog_measurement_snapshot=lambda: torch.tensor([[10.]])
    first=read(env).clone()
    head=ring.head
    assert torch.equal(read(env),first)
    assert ring.head==head
    env._dog_measurement_snapshot=lambda: torch.tensor([[99.]])
    env.dog_obs_latency_fill[:]=True
    assert read(env).item()==99
    assert ring.head==head


def test_chassis_randomization_targets_trunk_not_light_base():
    path = Path(__file__).resolve().parents[3] / 'go1_gym/envs/roboduet/legged_robot.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_process_rigid_body_props')
    namespace = {'gymapi': SimpleNamespace(Vec3=lambda *xyz: xyz)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    env = SimpleNamespace(body_names=['base', 'trunk', 'ee'], ee_idx=2,
                          payloads=torch.tensor([-2.]), com_displacements=torch.tensor([[.01,.02,.03]]))
    props = [SimpleNamespace(mass=.44), SimpleNamespace(mass=6.9), SimpleNamespace(mass=.1)]
    namespace['_process_rigid_body_props'](env, props, 0)
    assert props[0].mass == .44
    assert props[1].mass == pytest.approx(4.9)
    assert env.default_body_mass == 6.9
    assert env.base_mass_body_index == 1
    assert props[2].mass == pytest.approx(.2)
