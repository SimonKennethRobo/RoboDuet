"""Numerical adapter-contract regressions; no trained policies or ROS needed."""
from types import SimpleNamespace

import numpy as np
import pytest
import mujoco
import torch

from benchmark.wbc.dwbc_mujoco import DwbcMujoco, VISUAL_POLICY_ORDER
from benchmark.wbc.mujoco import JOINT_NAMES, joint_order_indices, _set_mpc_dog_command, configure_position_drives
from benchmark.wbc.umi_mujoco import UMI_TO_X5_HOME, UmiMujoco
from benchmark.wbc.wb_locoman_mujoco import WbLocomanMujoco
from scripts.rl_sar_obs import RlSarObservation, effective_gait_frequency


@pytest.fixture
def plant():
    bodies = "".join(f'<body pos="0 0 .02"><joint name="{name}" axis="0 1 0"/>'
                     '<geom type="sphere" size=".01" mass=".1"/></body>' for name in JOINT_NAMES)
    actuators = "".join(f'<motor joint="{name}"/>' for name in JOINT_NAMES)
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="base_link"><freejoint/>'
        '<geom type="sphere" size=".1" mass="10"/><site name="x5_ee" pos=".2 0 .3"/>'
        + bodies + '</body></worldbody><actuator>' + actuators + '</actuator></mujoco>')
    data = mujoco.MjData(model)
    data.qpos[3:7] = [2**-.5, 0., 0., 2**-.5]
    data.qvel[3:6] = [1., 2., 3.]
    mujoco.mj_forward(model, data)
    return model, data


@pytest.mark.parametrize("cls", [DwbcMujoco, UmiMujoco, WbLocomanMujoco])
def test_body_gyro_matches_mujoco_spatial_velocity(plant, cls):
    sim = cls.__new__(cls)
    sim.model, sim.data = plant
    sim.base = sim.model.body("base_link").id
    sim.qadr = sim.qpos_adr = np.arange(7, 25)
    sim.vadr = sim.dof_adr = np.arange(6, 24)
    spatial = np.empty(6)
    mujoco.mj_objectVelocity(sim.model, sim.data, mujoco.mjtObj.mjOBJ_BODY, sim.base, spatial, 1)
    np.testing.assert_allclose(sim.read_state()[3], spatial[:3], atol=1e-12)


def test_policy_joint_order_roundtrip_by_name(plant):
    model, _ = plant
    policy_from_canonical, canonical_from_policy = joint_order_indices(model, VISUAL_POLICY_ORDER)
    physical = np.arange(18.)
    np.testing.assert_array_equal(policy_from_canonical[:6], [3, 4, 5, 0, 1, 2])
    np.testing.assert_array_equal(physical[policy_from_canonical][canonical_from_policy], physical)


def test_robot_lab_command_layout_and_separate_arm_velocity_scale():
    params = dict(num_leg_dofs=12, num_arm_dofs=6, default_dof_pos=[0.] * 18,
                  observations=["robot_lab/velocity_pose_commands", "robot_lab/arm_dof_vel"],
                  num_observations=13, base_height_target=.2, arm_dof_vel_scale=.1)
    builder = RlSarObservation(params)
    state = dict(cmd_x=.1, cmd_y=.2, cmd_yaw=.3, cmd_height=.04, cmd_roll=.05, cmd_pitch=-.06,
                 dof_vel=np.arange(18.))
    np.testing.assert_allclose(builder.term("robot_lab/velocity_pose_commands", state),
                               [.1, .2, .3, .24, .05, -.06, 0.])
    np.testing.assert_allclose(builder.term("robot_lab/arm_dof_vel", state), np.arange(12., 18.) * .1)
    assert effective_gait_frequency(params, [0., 0., 0.]) == 0.


def test_mpc_full_command_reaches_policy_with_correct_posture_order():
    sim = SimpleNamespace(p={"limit_vel_x": [-.5, .5]})
    velocity = _set_mpc_dog_command(sim, {"base_velocity_body": [.8, -.2, .3], "body_posture": [.04, -.1, .2]})
    np.testing.assert_allclose(velocity, [.5, -.2, .3])
    np.testing.assert_allclose(sim.command, [.5, -.2, .3, -.1, .2, .04])


class CaptureActor:
    def __init__(self):
        self.observations = []

    def __call__(self, obs, **kwargs):
        self.observations.append(obs.numpy().copy())
        return torch.arange(18, dtype=torch.float32)[None] + len(self.observations)


@pytest.mark.parametrize("variant,delay,prop,priv", [("dwbc", 2, 76, 24), ("visual", 1, 71, 18)])
def test_dwbc_visual_observation_and_delayed_actions(plant, variant, delay, prop, priv):
    sim = DwbcMujoco.__new__(DwbcMujoco)
    sim.model, sim.data = plant
    sim.base = sim.model.body("base_link").id
    sim.variant, sim.priv_width = variant, priv
    sim.default_dof_pos = np.zeros(18)
    sim.goal_position, sim.goal_quaternion = np.array([.4, .1, .6]), np.array([0., 0., 0., 1.])
    sim.history = np.zeros((10, prop), np.float32)
    sim.actions = np.zeros(18)
    sim.latest_actions = np.zeros(18)
    sim.command = [.12, -.23, .34, 0., 0., 0.]
    sim.action_queue = [np.zeros(18) for _ in range(delay)]
    sim._contacts = lambda: np.array([1., 0., 1., 0.])
    actor = CaptureActor()
    sim.actor_critic = SimpleNamespace(actor=actor)
    state = (np.arange(18.) * .01, np.arange(18.), np.array([0., 0., 0., 1.]),
             np.array([.1, .2, .3]), np.zeros(3), np.zeros(3))
    executed = [sim.forward(*state) for _ in range(delay + 1)]
    assert actor.observations[0].shape == (1, prop * 11 + priv)
    np.testing.assert_array_equal(executed[0], np.zeros(18))
    expected = np.arange(18.) + 1
    if variant == "visual":
        expected[12:] = 0.
        expected = expected[VISUAL_POLICY_ORDER]
        np.testing.assert_allclose(actor.observations[0][0, 5:23], state[0][VISUAL_POLICY_ORDER])
        np.testing.assert_array_equal(actor.observations[0][0, 53:57], [0., 1., 0., 1.])
        action_slice = slice(41, 53)
    else:
        action_slice = slice(45, 63)
        np.testing.assert_allclose(actor.observations[0][0, 67:70], sim.command[:3])
    np.testing.assert_array_equal(executed[-1], expected)
    # The observation sees the last GENERATED action, even while the plant
    # executes an older command from its delay buffer.
    np.testing.assert_array_equal(actor.observations[1][0, action_slice],
                                  np.arange(action_slice.stop - action_slice.start) + 1)


def test_umi_future_positions_precede_all_rotations(plant):
    sim = UmiMujoco.__new__(UmiMujoco)
    sim.model, sim.data = plant
    sim.ee = sim.model.site("x5_ee").id
    sim.default_dof_pos = np.zeros(18)
    sim.reference_time = 0.
    sim.target_offsets = [.02, .04, .06, 1.]
    sim.position_scale, sim.orientation_scale = 3., 1.5
    sim.pose_latency_frames = 3
    sim.pose_history = [(np.zeros(3), np.eye(3))]
    sim.reference = SimpleNamespace(at=lambda t: (0., np.array([t, 2*t, -t]), np.array([0., 0., 0., 1.])))
    sim.actions = np.zeros(18)
    sim.action_buffer = np.zeros((2, 18))
    sim.action_clip = 100.
    sim.actor = CaptureActor()
    q = np.arange(18.) * .01
    dq = np.arange(18.) * .1
    gyro = np.array([.1, .2, .3])
    sim.forward(q, dq, np.array([0., 0., 0., 1.]), gyro, np.zeros(3), np.zeros(3))
    obs = sim.actor.observations[0][0]
    np.testing.assert_allclose(obs[:18], q)
    np.testing.assert_allclose(obs[18:36], dq * .05)
    np.testing.assert_allclose(obs[36:39], [0., 0., -1.])
    np.testing.assert_allclose(obs[39:42], gyro * .25)
    np.testing.assert_allclose(obs[42:54], np.array([[t, 2*t, -t] for t in sim.target_offsets]).ravel() * 3)
    np.testing.assert_allclose(obs[54:78], np.tile([1.5, 0., 0., 0., 1.5, 0.], 4))


def test_umi_joint_specific_delay_uses_original_decimation_formula():
    sim = UmiMujoco.__new__(UmiMujoco)
    sim.p = {"decimation": 4}
    sim.delay_steps = np.array([4] * 12 + [3] * 6)
    sim.default_dof_pos, sim.action_scale = np.zeros(18), np.ones(18)
    sim.action_clip = 100.
    sim.action_buffer = np.array([[2.] * 18, [1.] * 18])
    np.testing.assert_array_equal(sim.control_targets(0), np.ones(18))
    np.testing.assert_array_equal(sim.control_targets(2), np.ones(18))
    np.testing.assert_array_equal(sim.control_targets(3), [1.] * 12 + [2.] * 6)


def test_umi_transfer_action_limit_is_explicit():
    sim = UmiMujoco.__new__(UmiMujoco)
    sim.reference = SimpleNamespace(at=lambda _t: (0., np.zeros(3), np.array([0., 0., 0., 1.])))
    sim.reference_time = 0.
    sim.default_dof_pos = np.zeros(18)
    sim.target_offsets = [0.] * 4
    sim.position_scale = sim.orientation_scale = 1.
    sim.pose_latency_frames = 2
    sim.pose_history = [(np.zeros(3), np.eye(3))]
    sim.actions = np.zeros(18)
    sim.action_buffer = np.zeros((2, 18))
    sim.action_clip = 100.
    sim.transfer_action_limits = np.r_[[1.] * 12, [2.] * 6]
    sim.actor = lambda _obs: torch.full((1, 18), 7.)
    result = sim.forward(np.zeros(18), np.zeros(18), np.array([0., 0., 0., 1.]),
                         np.zeros(3), np.zeros(3), np.zeros(3))
    np.testing.assert_array_equal(result, np.r_[[1.] * 12, [2.] * 6])


def test_umi_arx5_tool_frame_conjugates_common_relative_pose():
    sim = UmiMujoco.__new__(UmiMujoco)
    sim.reference_time = 0.
    sim.default_dof_pos = np.zeros(18)
    sim.target_offsets = [0.] * 4
    sim.position_scale = sim.orientation_scale = 1.
    sim.pose_latency_frames = 2
    sim.pose_history = [(np.zeros(3), np.eye(3))]
    target = np.array([.1, .2, .3])
    sim.reference = SimpleNamespace(at=lambda _t: (0., target, np.array([0., 0., 0., 1.])))
    sim.actions = np.zeros(18)
    sim.action_buffer = np.zeros((2, 18))
    sim.action_clip = 100.
    sim.tool_frame = "arx5_home"
    sim.actor = CaptureActor()
    sim.forward(np.zeros(18), np.zeros(18), np.array([0., 0., 0., 1.]),
                np.zeros(3), np.zeros(3), np.zeros(3))
    common_relative = np.eye(4)
    common_relative[:3, 3] = target
    expected = UMI_TO_X5_HOME @ common_relative @ np.linalg.inv(UMI_TO_X5_HOME)
    obs = sim.actor.observations[0][0]
    np.testing.assert_allclose(obs[42:54], np.tile(expected[:3, 3], 4), atol=1e-7)
    np.testing.assert_allclose(obs[54:78], np.tile(expected[:2, :3].reshape(-1), 4), atol=1e-7)


def test_wb_locoman_feedback_changes_between_mpc_updates():
    sim = WbLocomanMujoco.__new__(WbLocomanMujoco)
    sim.feedback = {"feedforward_nm": np.ones(18), "q_rad": np.ones(18),
                    "dq_rad_s": np.zeros(18), "kp": np.full(18, 25.), "kd": np.full(18, 1.2)}
    a = sim.compute_torque(np.zeros(18), np.zeros(18))
    b = sim.compute_torque(np.full(18, .1), np.full(18, .2))
    np.testing.assert_allclose(a - b, np.full(18, 2.74))


def test_implicit_position_drive_exposes_pd_derivative_to_engine(plant):
    model, data = plant
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    sim = SimpleNamespace(model=model, data=data, arm_position_drive=True, mapping=list(range(18)),
                          p={"rl_kp": [50.] * 18, "rl_kd": [10.] * 18})
    configure_position_drives(sim, np.full(18, 20.))
    data.qpos[19:25] = .2
    data.qvel[18:24] = .3
    data.ctrl[12:] = .4
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.actuator_force[12:], 50 * (.4 - .2) - 10 * .3)
    np.testing.assert_allclose(model.actuator_biasprm[12:, 2], -10.)
    assert model.actuator_forcelimited[12:].all()


def test_registry_uses_known_working_policy_and_native_mpc():
    from benchmark.wbc.cross_method_cli import _method_contracts
    contracts = _method_contracts()
    assert contracts["roboduet"]["policy_key"] == "robot_lab_rear_r30o_s42_11497"
    for method in ("roboduet", "ma2022"):
        assert contracts[method]["upper_controller"] == "floating_base_ocs2_mpc"
