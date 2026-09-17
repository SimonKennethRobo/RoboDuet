"""Raw arm/dog observation parity and frozen-target playback checks."""

import ast
from pathlib import Path
import sys
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from benchmark.wbc.cross_method_cli import RAW_BUNDLE_ROOT, DEFAULT_SCENE
from benchmark.wbc.mujoco import JOINT_NAMES, _quat_to_matrix
from benchmark.wbc.roboduet_raw_mujoco import RoboDuetRawMujoco


class Capture:
    def __init__(self, module):
        self.module, self.inputs, self.outputs = module, [], []

    def __call__(self, value):
        self.inputs.append(value.clone())
        result = self.module(value)
        self.outputs.append(result.clone())
        return result


@pytest.fixture
def raw():
    if not (RAW_BUNDLE_ROOT.parent / "parameters.pkl").is_file():
        pytest.skip("Local Raw checkpoint is required for source parity")
    torch.set_num_threads(1)
    sim = RoboDuetRawMujoco(RAW_BUNDLE_ROOT / "policy/go2_x5", "roboduet_go2_x5",
                           DEFAULT_SCENE, RAW_BUNDLE_ROOT.parent)
    sim.data.qpos[:3] = [.1, -.2, .34]
    sim.data.qpos[3:7] = [.98, .03, -.06, .15]
    sim.data.qpos[3:7] /= np.linalg.norm(sim.data.qpos[3:7])
    adr = [sim.model.jnt_qposadr[sim.model.joint(name).id] for name in JOINT_NAMES]
    sim.data.qpos[adr] = sim.default_dof_pos
    mujoco.mj_forward(sim.model, sim.data)
    sim.reset_policy()
    return sim


def native_env():
    root = RAW_BUNDLE_ROOT.parents[3]
    # Locate the source checkout independently of the run directory depth.
    while root.name != "RoboDuetRaw" and root.parent != root:
        root = root.parent
    path = root / "go1_gym/envs/automatic/__init__.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "VelocityTrackingEasyEnv")
    cls.bases, cls.decorator_list = [], []
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and
                n.name in ("get_arm_observations", "get_dog_observations", "plan")]
    for function in cls.body:
        if function.name == "plan":
            continue
        # Execute the original public observation assembly; privileged reward
        # diagnostics after this boundary are not actor inputs.
        stop = next(i for i, n in enumerate(function.body) if isinstance(n, ast.Assign) and
                    any(isinstance(t, ast.Name) and t.id == "privileged_obs_buf" for t in n.targets))
        function.body = function.body[:stop] + [ast.Return(value=ast.Name(id="obs_buf", ctx=ast.Load()))]
    common = ast.parse((root / "go1_gym/utils/common.py").read_text())
    rpy = next(n for n in common.body if isinstance(n, ast.FunctionDef) and n.name == "quaternion_to_rpy")
    module = ast.fix_missing_locations(ast.Module(body=[rpy, cls], type_ignores=[]))
    namespace = {"torch": torch, "global_switch": SimpleNamespace(switch_open=True)}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["VelocityTrackingEasyEnv"]()


def test_raw_dual_actor_inputs_and_posture_match_native(raw):
    sim = raw
    rng = np.random.default_rng(42)
    sim.history[:] = rng.normal(size=sim.history.shape)
    sim.arm_history[:] = rng.normal(size=sim.arm_history.shape)
    sim.actions[:] = rng.normal(size=18)
    old_arm_history, old_actions = sim.arm_history.copy(), sim.actions.copy()
    sim.set_reference(SimpleNamespace(at=lambda _t: (0., np.array([1.2, .2, .6]), np.array([0., 0., 0., 1.]))), 0.)
    sim.policy, sim.arm_adaptation, sim.arm_history_model, sim.arm_body = [
        Capture(module) for module in (sim.policy, sim.arm_adaptation, sim.arm_history_model, sim.arm_body)]
    q, dq, quat, gyro, pos, vel = sim.read_state()
    actions = sim.forward(q, dq, quat, gyro, pos, vel)
    native = native_env()
    def ns(value):
        return SimpleNamespace(**{k: ns(v) for k, v in value.items()}) if isinstance(value, dict) else value
    native.cfg = ns(sim.training_config)
    native.obs_scales = SimpleNamespace(dof_pos=1., dof_vel=.05)
    native.num_actions_loco, native.num_actions_arm, native.num_envs, native.device = 12, 6, 1, "cpu"
    t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32).reshape(1, -1)
    native.base_quat, native.dof_pos, native.dof_vel = t(quat), t(q), t(dq)
    native.default_dof_pos, native.actions = t(sim.default_dof_pos), t(old_actions)
    native.commands_arm_obs = t(sim.arm_commands)
    native.commands_dog = t([*sim.command[:3], 0., 0.])
    native.commands_scale_dog = t(sim.p["dog_commands_scale"])
    native.plan_actions = torch.zeros(1, 2)
    native.projected_gravity = t(_quat_to_matrix(quat).T @ np.array([0., 0., -1.]))
    clock = np.sin(2 * np.pi * (sim.gait_indices + np.array([.5, 0., 0., .5])))
    if np.linalg.norm(sim.command[:3]) < .1:
        clock[:] = 1.
    native.clock_inputs = t(clock)
    arm_obs = native.get_arm_observations()
    expected_history = np.vstack((old_arm_history[1:], arm_obs.numpy()))
    np.testing.assert_allclose(sim.arm_adaptation.inputs[0].numpy().reshape(30, 20), expected_history, atol=2e-7)
    np.testing.assert_allclose(sim.arm_history_model.inputs[0].numpy().reshape(29, 20), expected_history[:-1], atol=2e-7)
    np.testing.assert_allclose(sim.arm_body.inputs[0].numpy()[0, :20], arm_obs.numpy()[0], atol=2e-7)
    plan = sim.arm_body.outputs[0]
    native.plan(plan[:, 6:])
    np.testing.assert_allclose(native.commands_dog.numpy()[0, 3:5], sim.command[3:5], atol=2e-7)
    dog_obs = native.get_dog_observations()
    np.testing.assert_allclose(sim.policy.inputs[0].numpy()[0, -56:], dog_obs.numpy()[0], atol=2e-7)
    np.testing.assert_allclose(actions[12:], np.clip(plan.numpy()[0, :6], -10., 10.))


def test_near_motion_and_far_target_both_move_base(raw):
    q, dq, quat, gyro, pos, vel = raw.read_state()
    trunk = raw.data.xpos[raw.base].copy()
    forward = _quat_to_matrix(quat)[:, 0]
    target = trunk + .4 * forward
    target[2] = .55
    quaternion = np.array([0., 0., 0., 1.])
    raw.set_reference(SimpleNamespace(at=lambda _t: (0., target, quaternion)), 0.)
    assert np.linalg.norm(raw.command[:2]) > 0.
    near = target - .12 * forward
    saved_near = near.copy()
    for _ in range(10):
        raw.set_reference(SimpleNamespace(at=lambda _t: (0., near, quaternion)), .5)
    assert np.linalg.norm(raw.command[:2]) > .1
    np.testing.assert_array_equal(near, saved_near)
    np.testing.assert_array_equal(raw.reference_position, saved_near)
    far = trunk + 3. * forward
    far[2] = 1.4
    saved = far.copy()
    # A reversed target must respect the previous command's acceleration limit.
    for _ in range(40):
        raw.set_reference(SimpleNamespace(at=lambda _t: (0., far, quaternion)), 1.)
    np.testing.assert_array_equal(far, saved)
    np.testing.assert_array_equal(raw.reference_position, saved)
    assert raw.target_projected and raw.command[0] > 0.
    assert raw.follower.diagnostics is not None
    local = raw.goal_position - np.r_[trunk[:2], .38]
    assert .30 - 1e-8 <= np.linalg.norm(local) <= .77 + 1e-8
    raw.reset_policy()
    assert not np.any(raw.arm_history) and not np.any(raw.history) and not any(raw.command)
    assert raw.follower.diagnostics is None
    assert not np.any(raw.follower.integral_xy)


def test_native_mode_preserves_unprojected_input(raw):
    raw.base_mode, raw.target_mode = "stand", "native"
    target, quaternion = np.array([3., -2., 1.4]), np.array([0., 0., 0., 1.])
    raw.set_reference(SimpleNamespace(at=lambda _t: (0., target, quaternion)), 0.)
    np.testing.assert_array_equal(raw.goal_position, target)
    np.testing.assert_array_equal(raw.goal_quaternion, quaternion)
    assert not any(raw.command[:3])
