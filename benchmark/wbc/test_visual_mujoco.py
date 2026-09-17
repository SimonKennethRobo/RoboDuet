"""Compare Visual's adapter with the original training/playback source."""

import ast
import importlib.util
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from benchmark.data.run_visual_library import select_tasks, VISUAL, CHECKPOINT, STACK
from benchmark.wbc.dwbc_mujoco import VISUAL_POLICY_ORDER
from benchmark.wbc.visual_mujoco import VisualMujoco


@pytest.fixture
def visual():
    if not CHECKPOINT.is_file():
        pytest.skip("Local Visual checkpoint is required for source parity checks")
    torch.set_num_threads(1)
    sim = VisualMujoco(VISUAL, CHECKPOINT,
                      STACK / "rl_sar/src/rl_sar_zoo/go2_x5_description/mjcf/scene.xml")
    sim.data.qpos[:3] = [.1, -.2, .34]
    sim.data.qpos[3:7] = [.98, .03, -.06, .15]
    sim.data.qpos[3:7] /= np.linalg.norm(sim.data.qpos[3:7])
    sim.data.qpos[sim.qadr] = sim.default_dof_pos + np.linspace(-.01, .01, 18)
    sim.data.qvel[sim.vadr] = np.linspace(-.2, .3, 18)
    mujoco.mj_forward(sim.model, sim.data)
    sim.reset_policy()
    return sim


def native_methods():
    utils = VISUAL / "third_party/isaacgym/python/isaacgym/torch_utils.py"
    spec = importlib.util.spec_from_file_location("_visual_native_torch_utils", utils)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = VISUAL / "low-level/legged_gym/envs/manip_loco/manip_loco.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ManipLoco")
    methods = {"compute_observations", "_get_body_orientation", "_reindex_all", "_reindex_feet"}
    cls.bases, cls.decorator_list = [], []
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    namespace = vars(module).copy()
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["ManipLoco"]()


def test_full_observation_and_action_match_native_source(visual):
    sim = visual
    rng = np.random.default_rng(42)
    sim.goal_position = np.array([.5, .1, .65])
    sim.goal_quaternion = np.array([0., 0., 0., 1.])
    sim.history[:] = rng.normal(size=(10, 71))
    sim.latest_actions[:] = rng.normal(size=18)
    sim.latest_actions[12:] = 0.
    sim.command = [.3, 0., -.55, 0., 0., 0.]
    sim.gait_observation[:] = [.2, .4, -.3, .2, -.1]
    native = native_methods()
    t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32).reshape(1, -1)
    q, dq, quat, gyro, base_pos, _ = sim.read_state()
    native.cfg = SimpleNamespace(
        env=SimpleNamespace(num_gripper_joints=2, reorder_dofs=True, observe_gait_commands=True, history_len=10),
        domain_rand=SimpleNamespace(observe_priv=True))
    native.obs_scales = SimpleNamespace(ang_vel=1., dof_pos=1., dof_vel=.05)
    native.num_envs, native.stand_by, native.add_noise = 1, False, False
    native.base_pos, native.base_quat, native.base_ang_vel = t(base_pos), t(quat), t(gyro)
    native.arm_base_offset = t([.05, 0., .1])
    native.curr_ee_goal_cart_world, native.ee_goal_orn_quat = t(sim.goal_position), t(sim.goal_quaternion)
    native.dof_pos, native.dof_vel = t(np.r_[q, 0., 0.]), t(np.r_[dq, 0., 0.])
    native.default_dof_pos = t(np.r_[sim.default_dof_pos, 0., 0.])
    native.action_history_buf = t(sim.latest_actions[VISUAL_POLICY_ORDER]).reshape(1, 1, 18)
    native.foot_contacts_from_sensor = t(sim._contacts())
    native.commands, native.commands_scale = t(sim.command[:3]), torch.ones(3)
    native.gait_indices, native.clock_inputs = t(sim.gait_observation[:1]).flatten(), t(sim.gait_observation[1:])
    native.mass_params_tensor, native.friction_coeffs_tensor = torch.zeros(1, 5), torch.zeros(1, 1)
    native.motor_strength = torch.ones(1, 18)
    native.obs_history_buf = torch.tensor(sim.history[None])
    native.episode_length_buf = torch.tensor([42])
    native.compute_observations()
    captured = []
    hook = sim.actor_critic.actor.register_forward_pre_hook(lambda _module, args: captured.append(args[0].clone()))
    executed = sim.forward(*sim.read_state())
    hook.remove()
    np.testing.assert_allclose(captured[0].numpy(), native.obs_buf.numpy(), atol=5e-7)
    np.testing.assert_allclose(sim.history, native.obs_history_buf.numpy()[0], atol=5e-7)
    with torch.inference_mode():
        expected = sim.actor_critic.actor(native.obs_buf, hist_encoding=True)[0].numpy()
    expected[12:] = 0.
    np.testing.assert_allclose(executed, expected[VISUAL_POLICY_ORDER], atol=5e-5)
    assert sim.action_delay_steps == 0


def test_projection_and_follower_leave_reference_unchanged(visual):
    target = np.array([5., 1., .3])
    quat = np.array([0., 0., 0., 1.])
    reference = SimpleNamespace(at=lambda _time: (0., target, quat))
    visual.set_reference(reference, 0.)
    np.testing.assert_array_equal(target, [5., 1., .3])
    np.testing.assert_array_equal(visual.reference_position, target)
    assert visual.target_projected
    assert np.linalg.norm(visual.goal_position - target) > 1.
    assert 0 < visual.command[0] <= .02 + 1e-12
    assert abs(visual.command[2]) <= .05 + 1e-12
    visual.reset_policy()
    assert not any(visual.command)
    assert not visual.action_queue


def test_library_selection_preserves_tasks_and_rejects_unknown_names():
    tasks = [dict(task_id="a", source_trajectory_id="curriculum-a0-b0", cell_A=0, cell_B=0),
             dict(task_id="b", source_trajectory_id="random-line-000")]
    selected = select_tasks(tasks, [[0, 0]], ["a", "random-line-000"])
    assert selected == tasks
    assert selected[0] is tasks[0]
    with pytest.raises(ValueError, match="Unknown"):
        select_tasks(tasks, names=["missing"])
