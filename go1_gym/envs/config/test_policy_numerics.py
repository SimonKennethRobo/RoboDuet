"""Fault injection for opt-in PPO guards and pre-failure physical context."""
from types import SimpleNamespace
import isaacgym
import pytest
import torch
from go1_gym_learn.ppo_cse_automatic.numerical_guard import NumericalGuard
from go1_gym.envs.roboduet.numerical_safety import record_physics_context, fault_context


def make_guard(tmp_path):
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters())
    return NumericalGuard(model, optimizer, tmp_path)


def test_input_fault_records_offending_row_and_original_model(tmp_path):
    guard = make_guard(tmp_path)
    guard.phase, guard.iteration = 'rollout', 22501
    seen = []
    guard.context = lambda ids: seen.extend(ids.tolist())
    obs = torch.zeros(4, 3)
    obs[2, 1] = float('nan')
    with pytest.raises(FloatingPointError, match='rollout/inputs'):
        guard.check('inputs', obs=obs)
    dump = torch.load(tmp_path/'policy-fault.pt')
    assert seen == [2] and dump['iteration'] == 22501
    assert dump['details']['obs']['nonfinite'] == 1
    assert torch.isnan(dump['samples']['obs'][0, 1])
    for k, v in guard.model.state_dict().items():
        torch.testing.assert_close(v, dump['model'][k])


@pytest.mark.parametrize('kind', ['mean_nan', 'std_nan', 'std_negative'])
def test_distribution_fault_is_captured_before_normal_construction(tmp_path, kind):
    guard = make_guard(tmp_path)
    mean, std = torch.zeros(4, 2), torch.ones(2)
    if kind == 'mean_nan': mean[1, 0] = float('nan')
    if kind == 'std_nan': std[0] = float('nan')
    if kind == 'std_negative': std[0] = -1
    with pytest.raises(FloatingPointError):
        guard.check_distribution(mean, std)
    assert (tmp_path/'policy-fault.pt').is_file()


def test_invalid_gradient_cannot_reach_optimizer_step(tmp_path):
    guard = make_guard(tmp_path)
    before = {k: v.clone() for k, v in guard.model.state_dict().items()}
    guard.model(torch.ones(2, 3)).sum().backward()
    guard.model.weight.grad[0, 0] = float('inf')
    with pytest.raises(FloatingPointError, match='gradients'):
        guard.check_gradients()
        guard.optimizer.step()
    for k, v in before.items():
        torch.testing.assert_close(v, guard.model.state_dict()[k])


def test_finite_values_do_not_create_fault_artifacts(tmp_path):
    guard = make_guard(tmp_path)
    guard.check('inputs', obs=torch.zeros(4, 3))
    guard.check_distribution(torch.zeros(4, 2), torch.ones(2))
    guard.check_parameters()
    assert not list(tmp_path.iterdir())


def test_trace_retains_previous_steps_without_aliasing_or_other_envs():
    env = SimpleNamespace(num_envs=3, cfg=SimpleNamespace(env=SimpleNamespace(numerical_trace_steps=2)),
                          common_step_counter=0, root_states=torch.zeros(3, 13))
    for step in range(3):
        env.common_step_counter = step
        env.root_states[:, 0] = step
        record_physics_context(env)
    env.root_states[1, 0] = float('nan')
    dump = fault_context(env, torch.tensor([1]))
    assert [f['step'] for f in dump['history']] == [1, 2]
    assert [f['values']['root_states'][0, 0].item() for f in dump['history']] == [1, 2]
    assert torch.isnan(dump['current']['root_states'][0, 0])
    assert dump['history'][0]['values']['root_states'].shape == (1, 13)
