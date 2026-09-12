"""Failure injection: invalid samples cannot trigger a terrain CUDA assert."""
from types import SimpleNamespace
import isaacgym
import pytest
import torch
from go1_gym.envs.roboduet.numerical_safety import quarantine_physics
from go1_gym.utils.height_sampling import sample_triangle_heights


def test_nan_height_query_preserves_invalidity_without_invalid_index():
    heights = torch.arange(16, dtype=torch.float).reshape(4, 4)
    xy = torch.tensor([[1., 1.], [float('nan'), 1.], [1.e30, -1.e30]])
    result = sample_triangle_heights(heights, xy, 1., 1., 0.)
    assert result[0] == 5 and result[2] == 12
    assert torch.isnan(result[1])


@pytest.mark.parametrize('fault', ['root_nan', 'joint_nan', 'rigid_inf', 'contact_nan', 'root_explosion'])
def test_quarantine_changes_only_bad_environment_and_records_state(tmp_path, fault):
    root = torch.tensor([[0., 0., .3, 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]]).repeat(3, 1)
    env = SimpleNamespace(num_envs=3, num_bodies=2, root_states=root, base_init_state=root[0].clone(),
        env_origins=torch.zeros(3, 3), default_dof_pos=torch.ones(1, 4),
        dof_pos=torch.ones(3, 4), dof_vel=torch.zeros(3, 4), rigid_body_state=root[:, None].repeat(1, 2, 1),
        contact_forces=torch.zeros(3, 2, 3), numerical_fault_mask=torch.zeros(3, dtype=torch.bool),
        numerical_fault_count=0, numerical_fault_dumps=0, numerical_fault_active=False,
        cfg=SimpleNamespace(env=SimpleNamespace(numerical_max_root_speed=100., numerical_max_dof_speed=1000.)),
        numerical_fault_log_dir=str(tmp_path), common_step_counter=2, actions=torch.zeros(3, 4),
        commands_dog=torch.zeros(3, 11), torques=torch.zeros(3, 4), step_locomotion_power=torch.zeros(3))
    if fault == 'root_nan': env.root_states[1, 0] = float('nan')
    if fault == 'joint_nan': env.dof_vel[1, 0] = float('nan')
    if fault == 'rigid_inf': env.rigid_body_state[1, 1, 0] = float('inf')
    if fault == 'contact_nan': env.contact_forces[1, 0, 0] = float('nan')
    if fault == 'root_explosion': env.root_states[1, 7] = 1.e10
    before = {k: getattr(env, k)[[0, 2]].clone() for k in ('root_states', 'dof_pos', 'dof_vel', 'rigid_body_state', 'contact_forces')}
    quarantine_physics(env)
    assert env.numerical_fault_mask.tolist() == [False, True, False]
    assert env.numerical_fault_count == 1 and env.numerical_fault_active
    for key, value in before.items():
        assert torch.isfinite(getattr(env, key)).all()
        torch.testing.assert_close(getattr(env, key)[[0, 2]], value)
    assert (tmp_path / 'fault-0.pt').exists()
    quarantine_physics(env)
    assert not env.numerical_fault_active and not env.numerical_fault_mask.any()
    assert env.numerical_fault_count == 1
