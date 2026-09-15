"""CPU regression tests for the WBC benchmark's public contracts."""

import ctypes
from types import SimpleNamespace

import isaacgym  # noqa: F401 - must precede torch
import numpy as np
import pytest
import torch

from benchmark.compare import _summary_rows
from benchmark.wbc.scoring import aggregate_task_events, timed_trajectory_success
from benchmark.wbc.suite import finalize_task_spec, trajectory_features
from benchmark.wbc.workspace import build_workspace_grid, summarize_workspace_results


def _batch(time_law):
    return SimpleNamespace(
        gamma_p=torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]),
        gamma_quat=torch.tensor([[[0.0, 0.0, 0.0, 1.0]] * 3]),
        gamma_tangent=torch.tensor([[[1.0, 0.0, 0.0]] * 3]),
        gamma_s=torch.tensor([[0.0, 1.0, 1.0]]),
        tl_t=torch.tensor([[0.0, 0.5, 1.0]]),
        tl_s=torch.tensor([time_law]),
        tl_sdot=torch.tensor([[1.0, 1.0, 1.0]]),
        L=torch.tensor([1.0]),
        T=torch.tensor([1.0]),
        max_gamma_points=3,
        max_tl_points=3,
    )


def test_progress_without_accuracy_or_hold_is_not_success():
    result = timed_trajectory_success(
        {
            "reference_time_s": 1.0,
            "reference_duration_s": 1.0,
            "final_progress": 0.85,
            "final_ee_pos_error_m": 0.20,
            "final_ee_rot_error_rad": 1.0,
            "tracking_tube_fraction": 0.0,
            "endpoint_hold_time_s": 0.0,
        }
    )
    assert result == {"success": False, "end_reason": "criteria_not_met"}


def test_timed_trajectory_success_requires_and_accepts_full_contract():
    result = timed_trajectory_success(
        {
            "reference_time_s": 2.0,
            "reference_duration_s": 2.0,
            "final_progress": 1.0,
            "final_ee_pos_error_m": 0.01,
            "final_ee_rot_error_rad": 0.02,
            "tracking_tube_fraction": 0.9,
            "endpoint_hold_time_s": 0.5,
        }
    )
    assert result == {"success": True, "end_reason": "success"}


def test_full_time_law_changes_reference_content_hash():
    first = trajectory_features(_batch([0.0, 0.5, 1.0]), 0)
    second = trajectory_features(_batch([0.0, 0.2, 1.0]), 0)
    assert first["content_sha256"] != second["content_sha256"]


def test_trajectory_features_measure_xy_displacement():
    features = trajectory_features(_batch([0.0, 0.5, 1.0]), 0)
    assert features["xy_displacement_m"] == pytest.approx(1.0)
    assert features["xy_heading_rad"] == pytest.approx(0.0)


def test_start_aligned_task_identity_uses_full_translation_and_orientation():
    first = {"task_family": "timed_trajectory", "content_sha256": "reference"}
    second = dict(first)
    state = {"root_state_env_local": [0.0]}
    finalize_task_spec(
        first, initial_state=state, anchor_env_local=[1.0, 2.0, 3.0], deadline_s=5.0
    )
    finalize_task_spec(
        second, initial_state=state, anchor_env_local=[1.0, 2.0, 99.0], deadline_s=5.0
    )

    assert first["anchor_env_local_xyz_m"] == [1.0, 2.0, 3.0]
    assert first["task_spec_sha256"] != second["task_spec_sha256"]
    third = dict({"task_family": "timed_trajectory", "content_sha256": "reference"})
    finalize_task_spec(
        third,
        initial_state=state,
        anchor_env_local=[1.0, 2.0, 3.0],
        orientation_left_multiplier_xyzw=[0.0, 0.0, 1.0, 0.0],
        deadline_s=5.0,
    )
    assert first["task_spec_sha256"] != third["task_spec_sha256"]


def test_push_schedule_is_part_of_task_identity():
    nominal = {"task_family": "timed_trajectory", "content_sha256": "reference"}
    pushed = dict(nominal)
    kwargs = dict(
        initial_state={"root_state_env_local": [0.0]},
        anchor_env_local=[1.0, 2.0, 3.0],
        deadline_s=5.0,
    )
    finalize_task_spec(nominal, **kwargs)
    finalize_task_spec(
        pushed,
        **kwargs,
        disturbance_schedule=[
            {
                "type": "constant_force",
                "start_time_s": 1.0,
                "duration_s": 0.1,
                "body": "base",
                "frame": "environment_world",
                "force_n": [80.0, 0.0, 0.0],
            }
        ],
    )
    assert nominal["content_sha256"] == pushed["content_sha256"]
    assert nominal["task_spec_sha256"] != pushed["task_spec_sha256"]


def test_upper_controller_output_contract_rejects_mixed_and_wrong_widths():
    from benchmark.wbc.controllers import UpperControllerOutput, validate_upper_controller_output

    valid = UpperControllerOutput(
        arm_actions=torch.zeros(2, 6), plan_actions=torch.zeros(2, 9)
    )
    validate_upper_controller_output(valid, 2, 6, 9)
    with pytest.raises(ValueError, match="arm_actions"):
        validate_upper_controller_output(
            UpperControllerOutput(torch.zeros(2, 5)), 2, 6, 9
        )
    with pytest.raises(ValueError, match="cannot use plan actions and direct commands"):
        validate_upper_controller_output(
            UpperControllerOutput(
                torch.zeros(2, 6),
                plan_actions=torch.zeros(2, 9),
                base_velocity_body=torch.zeros(2, 3),
                body_posture=torch.zeros(2, 3),
            ),
            2,
            6,
            9,
        )


def test_ocs2_state_and_command_wire_contracts_are_fixed_width_and_framed():
    from benchmark.wbc.controllers import (
        _BenchmarkRequestHeader,
        _CmdMsg,
        _Header,
        _StateMsg,
        decode_ocs2_command,
        encode_ocs2_request,
        encode_ocs2_state,
    )

    base = SimpleNamespace(
        num_actions_loco=12,
        num_actions_arm=6,
        root_states=torch.tensor([[11.0, 22.0, 3.0] + [0.0] * 10]),
        env_origins=torch.tensor([[10.0, 20.0, 0.0]]),
        base_quat=torch.tensor([[0.1, 0.2, 0.3, 0.9]]),
        base_lin_vel=torch.tensor([[1.0, 2.0, 3.0]]),
        base_ang_vel=torch.tensor([[4.0, 5.0, 6.0]]),
        dof_pos=torch.arange(18, dtype=torch.float32).view(1, 18),
        dof_vel=torch.arange(18, dtype=torch.float32).view(1, 18) + 20,
    )
    payload = encode_ocs2_state(base, 0, 7)
    assert len(payload) == 224
    state = _StateMsg.from_buffer_copy(payload)
    assert state.h.seq == 7
    times = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
    poses = np.asarray(
        [
            [0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 2.0],
            [0.5, 0.0, 0.6, 0.0, 0.0, 0.0, 1.0],
            [0.6, 0.0, 0.7, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    request_payload = encode_ocs2_request(payload, 0.14, times, poses)
    request = _BenchmarkRequestHeader.from_buffer_copy(
        request_payload[: ctypes.sizeof(_BenchmarkRequestHeader)]
    )
    assert request.controller_time_s == pytest.approx(0.14)
    assert request.reference_count == 3
    assert len(request_payload) == 240 + 3 * 8 + 3 * 7 * 4
    decoded_times = np.frombuffer(request_payload, dtype="<f8", count=3, offset=240)
    decoded_poses = np.frombuffer(
        request_payload, dtype="<f4", count=21, offset=240 + 3 * 8
    ).reshape(3, 7)
    assert decoded_times.tolist() == pytest.approx(times.tolist())
    assert np.linalg.norm(decoded_poses[:, 3:], axis=1).tolist() == pytest.approx([1.0] * 3)
    state_only = encode_ocs2_request(payload, 0.16)
    assert len(state_only) == 240
    assert _BenchmarkRequestHeader.from_buffer_copy(state_only).reference_count == 0
    # Phase is an optional trailer: legacy StateMsg/reference bytes are intact.
    for times_arg, poses_arg, legacy in [(None, None, state_only), (times, poses, request_payload)]:
        framed = encode_ocs2_request(payload, .16 if times_arg is None else .14,
                                    times_arg, poses_arg, gait_phase_rad=5.7)
        assert len(framed) == len(legacy) + 8
        header = _BenchmarkRequestHeader.from_buffer_copy(framed)
        assert header.reserved == 1
        assert framed[:236] == legacy[:236]
        assert framed[240:-8] == legacy[240:]
        assert np.frombuffer(framed[-8:], dtype="<f8")[0] == pytest.approx(5.7)
    with pytest.raises(ValueError, match="finite"):
        encode_ocs2_request(payload, 0., gait_phase_rad=float("nan"))
    with pytest.raises(ValueError, match="strictly increasing"):
        encode_ocs2_request(payload, 0.0, [0.0, 0.0], poses[:2])

    command = _CmdMsg()
    command.h = _Header(0x57424301, 1, 2, 9, 1.0)
    command.base_lin_vel_body_xy[:] = (0.1, -0.2)
    command.base_ang_vel_body_z = 0.3
    command.body_height_cmd = 0.04
    command.body_pitch_cmd = 0.05
    command.body_roll_cmd = -0.06
    command.arm_q_cmd[:] = tuple(float(i) for i in range(6))
    command.arm_dq_cmd[:] = (0.0,) * 6
    command.policy_time = 0.4
    command.solver_ok = 1
    command.mode = 2
    decoded = decode_ocs2_command(bytes(command))
    assert decoded["seq"] == 9
    assert decoded["mode"] == 2
    assert decoded["solver_ok"] is True
    assert decoded["base_velocity_body"].tolist() == pytest.approx([0.1, -0.2, 0.3])
    assert decoded["body_posture"].tolist() == pytest.approx([0.04, 0.05, -0.06])


def test_ik_controller_requires_residual_mode_and_matching_task_count():
    from benchmark.wbc.controllers import IkUpperController

    controller = IkUpperController()
    base = SimpleNamespace(arm_action_mode="ik_residual")
    controller.reset(base, 0, 1, [{}])
    with pytest.raises(ValueError, match="task count"):
        controller.reset(base, 0, 1, [])
    base.arm_action_mode = "end_to_end"
    with pytest.raises(ValueError, match="ik_residual"):
        controller.reset(base, 0, 1, [{}])


def test_ocs2_controller_reset_restarts_synchronous_sequence():
    from benchmark.wbc.controllers import FloatingBaseOcs2MpcController

    controller = object.__new__(FloatingBaseOcs2MpcController)
    controller.sequence = 37
    transport = SimpleNamespace(reset_task_calls=0)
    transport.reset_task = lambda: setattr(
        transport, "reset_task_calls", transport.reset_task_calls + 1
    )
    controller.transport = transport
    batch = SimpleNamespace(
        T=torch.tensor([1.0]),
        tl_t=torch.tensor([[0.0, 0.5, 1.0, 1.0]]),
        tl_s=torch.tensor([[0.0, 0.5, 1.0, 1.0]]),
        max_tl_points=4,
        p_at=lambda s: torch.stack((s, torch.zeros_like(s), torch.ones_like(s)), dim=-1),
        quat_at=lambda s: torch.nn.functional.pad(
            torch.ones_like(s).unsqueeze(-1), (3, 0)
        ),
    )
    base = SimpleNamespace(
        arm_action_mode="end_to_end",
        dt=0.02,
        num_envs=1,
        traj_batch=batch,
        env_origins=torch.tensor([[10.0, 20.0, 0.0]]),
    )
    controller.reset(base, 0, 1, [{}])
    assert controller.sequence == 0
    assert transport.reset_task_calls == 1
    assert controller.reference_times_s.tolist() == pytest.approx([0.0, 0.5, 1.0])
    assert controller.reference_poses_xyz_xyzw.shape == (3, 7)
    assert controller.reference_poses_xyz_xyzw[0].tolist() == pytest.approx(
        [-10.0, -20.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    )


def test_system_matrix_parses_trajectory_visualization_switch():
    from benchmark.wbc.system_cli import parse_args

    args = parse_args(
        [
            "--env-logdir",
            "env",
            "--loco-logdirs",
            "dog",
            "--visualize-trajectories",
        ]
    )
    assert args.visualize_trajectories is True


def test_system_matrix_parses_async_ocs2_transport():
    from benchmark.wbc.system_cli import parse_args

    args = parse_args(
        [
            "--env-logdir",
            "env",
            "--loco-logdirs",
            "dog",
            "--ocs2-transport",
            "async",
            "--ocs2-command-timeout-s",
            "0.25",
            "--ocs2-command-mode",
            "pose_only",
        ]
    )
    assert args.ocs2_transport == "async"
    assert args.ocs2_command_timeout_s == pytest.approx(0.25)
    assert args.ocs2_command_mode == "pose_only"


def test_benchmark_trajectory_viewer_records_caps_and_resets_actual_path():
    from go1_gym.envs.roboduet.wbc_env import WBCEnv

    env = object.__new__(WBCEnv)
    env.headless = False
    env.viewer = object()
    env.cfg = SimpleNamespace(asset=SimpleNamespace(render_sphere=False))
    env.end_effector_state = torch.zeros(1, 13)
    env.configure_benchmark_trajectory_viewer(True, max_draw_points=2)
    assert env.cfg.asset.render_sphere is True

    env._record_benchmark_viewer_trajectory()
    env.end_effector_state[0, 0] = 1.0
    env._record_benchmark_viewer_trajectory()
    env.end_effector_state[0, 0] = 2.0
    drawn = env._record_benchmark_viewer_trajectory()
    assert drawn[:, 0].tolist() == pytest.approx([0.0, 2.0])

    env.reset_benchmark_viewer_trajectory()
    assert env._benchmark_executed_trajectory_world == []


def test_ground_relative_trajectory_placement_uses_pointwise_terrain_height():
    from go1_gym.envs.roboduet.wbc_env import WBCEnv

    env = object.__new__(WBCEnv)
    env._ground_height_at = lambda xy: 0.1 * xy[..., 0] + 0.2 * xy[..., 1]
    path = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 2.0, 1.5]]])
    offset = torch.tensor([[3.0, 4.0, 9.0]])

    placed = env._place_trajectory_positions(path, offset, ground_relative_z=True)

    torch.testing.assert_close(placed[0, :, :2], torch.tensor([[3.0, 4.0], [4.0, 6.0]]))
    torch.testing.assert_close(placed[0, :, 2], torch.tensor([1.1, 3.1]))


def test_custom_trajectory_placement_retains_full_xyz_offset():
    from go1_gym.envs.roboduet.wbc_env import WBCEnv

    env = object.__new__(WBCEnv)
    path = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 2.0, 1.5]]])
    offset = torch.tensor([[3.0, 4.0, 9.0]])

    placed = env._place_trajectory_positions(path, offset, ground_relative_z=False)

    torch.testing.assert_close(placed, torch.tensor([[[3.0, 4.0, 9.0], [4.0, 6.0, 10.5]]]))


def test_trajectory_placement_writes_back_after_advanced_indexing():
    from go1_gym.envs.roboduet.wbc_env import WBCEnv

    env = object.__new__(WBCEnv)
    env.device = "cpu"
    env.traj_batch = SimpleNamespace(gamma_p=torch.zeros(2, 3, 3))
    env._ground_height_at = lambda xy: torch.zeros_like(xy[..., 0])
    env._default_trajectory_anchor = lambda env_ids: torch.zeros(len(env_ids), 3)
    for name in (
        "traj_s", "traj_s_prev", "traj_sim_time", "traj_d_lat", "traj_sdot_meas",
        "traj_sdot_ref", "traj_timing_err", "traj_twist_err", "traj_dlat_sum",
        "traj_timing_abs_sum", "traj_ik_jump_max", "traj_samples", "traj_dlat_sq_sum",
        "traj_twist_err_sum", "traj_rho_sum", "traj_rho_max", "traj_rho_above_hi_count",
        "traj_rho_valid_count", "traj_base_util_sum", "traj_base_util_count",
        "traj_v_ff_sum", "traj_v_base_sum", "traj_motor_power_sum",
        "traj_manipulability_sum",
    ):
        setattr(env, name, torch.ones(2))

    env._place_and_reset_trajectories(
        torch.tensor([1]), offset=torch.tensor([[2.0, 3.0, 4.0]])
    )

    torch.testing.assert_close(
        env.traj_batch.gamma_p[1], torch.tensor([[2.0, 3.0, 4.0]] * 3)
    )


def test_task_booleans_are_aggregated_as_event_rates():
    rows = [
        {"completed": True, "fall": False},
        {"completed": False, "fall": True},
    ]
    summary = aggregate_task_events(rows)
    assert summary["completion_rate"] == pytest.approx(0.5)
    assert summary["fall_rate"] == pytest.approx(0.5)
    compared = _summary_rows({"candidate": {"wbc_trajectories": rows}})
    assert compared["candidate"]["completion_rate"] == pytest.approx(0.5)
    assert compared["candidate"]["fall_rate"] == pytest.approx(0.5)


def test_inactive_nan_is_removed_by_mask():
    from benchmark.wbc.evaluation import WBCAccumulator

    result = WBCAccumulator._masked(
        torch.tensor([float("nan"), 2.0]), torch.tensor([False, True])
    )
    assert torch.equal(result, torch.tensor([0.0, 2.0]))


def test_terminal_snapshot_survives_autoreset_tensor_mutation():
    from benchmark.wbc.evaluation import BenchmarkWBCEnv

    env = object.__new__(BenchmarkWBCEnv)
    env.num_envs = 2
    env.device = "cpu"
    env.end_effector_state = torch.arange(26, dtype=torch.float32).reshape(2, 13)
    env.arm_goal_pos_world = torch.ones(2, 3)
    env.arm_goal_quat_world = torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 2)
    env.goal_rho = torch.tensor([0.2, 0.4])
    env.goal_rho_valid = torch.tensor([True, True])
    env.traj_d_lat = torch.tensor([0.01, 0.02])
    env.traj_timing_err = torch.tensor([0.03, 0.04])
    env.traj_sdot_meas = torch.tensor([0.5, 0.6])
    env.traj_s = torch.tensor([0.7, 0.8])
    env.traj_batch = SimpleNamespace(L=torch.ones(2), T=torch.full((2,), 2.0))
    env.traj_sim_time = torch.tensor([1.0, 1.5])
    env.root_states = torch.arange(26, dtype=torch.float32).reshape(2, 13)
    env.dof_pos = torch.arange(36, dtype=torch.float32).reshape(2, 18)
    env.dof_vel = torch.arange(36, dtype=torch.float32).reshape(2, 18)
    env.torques = torch.ones(2, 18)
    env.joint_pos_target = torch.zeros(2, 18)
    env.actions = torch.zeros(2, 18)
    env.contact_forces = torch.zeros(2, 27, 3)
    env.foot_velocities = torch.zeros(2, 4, 3)
    env.step_locomotion_abs_energy_j = torch.zeros(2)
    env.step_locomotion_positive_energy_j = torch.zeros(2)
    env.goal_manipulability = torch.tensor([0.1, 0.2])
    env.goal_jacobian_sigma_min = torch.tensor([0.3, 0.4])
    env.goal_rot_jacobian_sigma_min = torch.tensor([0.5, 0.6])
    env.goal_joint_limit_distance = torch.full((2, 6), 0.2)
    env.goal_ik_step_norm_rad = torch.tensor([0.01, 0.02])
    env.goal_ik_step_saturated = torch.tensor([False, True])
    env.goal_ik_solver_valid = torch.tensor([True, True])
    env.base_feedforward_cmd = torch.zeros(2, 3)
    env.time_out_buf = torch.tensor([False, True])
    env.traj_early_term = torch.tensor([True, False])
    env.numerical_fault_mask = torch.tensor([False, False])

    env._arm_pre_reset_capture_hook(torch.tensor([1]))
    terminal_root = env.benchmark_terminal_snapshot["root_state"][1].clone()
    env.root_states[1].zero_()
    env.traj_early_term[1] = True

    assert env.benchmark_terminal_snapshot["valid"].tolist() == [False, True]
    assert torch.equal(env.benchmark_terminal_snapshot["root_state"][1], terminal_root)
    assert not bool(env.benchmark_terminal_snapshot["traj_early_term"][1])


def test_raw_trace_can_be_rescored_without_a_simulator(tmp_path):
    from benchmark.wbc.scoring import DEVELOPMENT_TIMED_TRAJECTORY_PROTOCOL
    from benchmark.wbc.trace import score_trace_archive, write_trace_archive

    time_steps = 3
    zeros = torch.zeros(time_steps, 1, dtype=torch.bool)
    reference_position = torch.tensor(
        [[[0.0, 0.0, 0.0]], [[0.1, 0.0, 0.0]], [[0.2, 0.0, 0.0]]]
    )
    actual_state = torch.zeros(time_steps, 1, 13)
    actual_state[..., :3] = reference_position
    actual_state[..., 6] = 1.0
    trace = {
        "sample_present": torch.ones(time_steps, 1, dtype=torch.bool),
        "metric_valid": torch.ones(time_steps, 1, dtype=torch.bool),
        "terminal_snapshot": zeros.clone(),
        "done": zeros.clone(),
        "timed_out": zeros.clone(),
        "trajectory_early_termination": zeros.clone(),
        "numerical_fault": zeros.clone(),
        "fall": zeros.clone(),
        "reference_time_s": torch.tensor([[0.25], [0.50], [0.75]]),
        "reference_duration_s": torch.full((time_steps, 1), 0.50),
        "reference_ee_position_m": reference_position,
        "reference_ee_quaternion_xyzw": torch.tensor(
            [[[0.0, 0.0, 0.0, 1.0]]] * time_steps
        ),
        "actual_ee_state": actual_state,
        "actual_ee_grasp_linear_velocity_mps": torch.zeros(time_steps, 1, 3),
        "base_root_state": torch.zeros(time_steps, 1, 13),
        "dof_velocity_rad_s": torch.zeros(time_steps, 1, 18),
        "actuator_command": torch.zeros(time_steps, 1, 18),
        "leg_abs_mechanical_energy_step_j": torch.zeros(time_steps, 1),
        "leg_positive_mechanical_energy_step_j": torch.zeros(time_steps, 1),
        "trajectory_progress": torch.tensor([[0.0], [0.99], [1.0]]),
        "trajectory_lateral_error_m": torch.tensor([[0.03], [0.02], [0.01]]),
        "trajectory_timing_error_m": torch.tensor([[-0.06], [0.03], [0.0]]),
        "goal_rho": torch.tensor([[0.8], [0.9], [1.1]]),
        "goal_rho_valid": torch.ones(time_steps, 1, dtype=torch.bool),
        "base_feedforward_command": torch.tensor(
            [[[0.1, 0.0, 0.0]], [[0.2, 0.0, 0.0]], [[0.3, 0.0, 0.0]]]
        ),
        "manipulability": torch.tensor([[0.4], [0.3], [0.2]]),
        "jacobian_sigma_min": torch.tensor([[0.2], [0.1], [0.05]]),
        "rot_jacobian_sigma_min": torch.tensor([[0.2], [0.04], [0.1]]),
        "joint_limit_distance_fraction": torch.tensor(
            [[[0.10] * 6], [[0.01] * 6], [[0.20] * 6]]
        ),
        "ik_step_norm_rad": torch.zeros(time_steps, 1),
        "ik_step_saturated": zeros.clone(),
        "ik_solver_valid": torch.ones(time_steps, 1, dtype=torch.bool),
    }
    accumulator = SimpleNamespace(
        protocol=dict(DEVELOPMENT_TIMED_TRAJECTORY_PROTOCOL),
        kinematic_protocol={
            "status": "development_thresholds",
            "reach_model_limit_ratio": 1.0,
            "rho_comfort_hi": 0.85,
            "joint_limit_margin_fraction": 0.02,
            "rot_jacobian_sigma_min": 0.05,
        },
        n_steps=time_steps,
        trace_control_type="M",
        trace_num_actions_loco=12,
        trace_num_actions_arm=6,
        trace_arm_action_mode="end_to_end",
        trace_reach_model="scalar_sphere_fallback",
        trace_self_collision_observability="disabled_by_asset_filter",
        trace_tensors=lambda count: {key: value[:, :count] for key, value in trace.items()},
    )
    path = tmp_path / "trace.npz"
    record = write_trace_archive(path, accumulator, ["task-1"], dt_s=0.25)
    rescored = score_trace_archive(path)[0]

    assert record["schema_version"] == "legged-manip-trace-v3"
    assert rescored["completed"] is True
    assert rescored["completion_time_s"] == pytest.approx(0.75)
    assert rescored["ee_pos_rmse_m"] == pytest.approx(0.0)
    assert rescored["n_valid_metric_samples"] == time_steps
    assert rescored["d_lat_mean_m"] == pytest.approx(0.02)
    assert rescored["timing_err_mean_m"] == pytest.approx(0.03)
    assert rescored["rho_mean"] == pytest.approx(0.9333333333)
    assert rescored["rho_above_hi_rate"] == pytest.approx(2 / 3)
    assert rescored["reach_model_outside_fraction"] == pytest.approx(1 / 3)
    assert rescored["joint_limit_near_fraction"] == pytest.approx(1 / 3)
    assert rescored["rot_singularity_near_fraction"] == pytest.approx(1 / 3)
    assert rescored["kinematic_observed_feasible_fraction"] == pytest.approx(1 / 3)
    assert rescored["ik_solver_status"] == "not_applicable"
    assert rescored["self_collision_status"] == "disabled_by_asset_filter"


def test_workspace_volume_keeps_failed_targets_in_denominator():
    manifest = build_workspace_grid(
        bounds_m=((0.0, 0.2), (0.0, 0.1), (0.0, 0.1)), spacing_m=0.1
    )
    rows = []
    for index, target in enumerate(manifest["targets"]):
        rows.append(
            {
                **target,
                "completed": index == 0,
                "self_collision_status": "disabled_by_asset_filter",
            }
        )
    summary = summarize_workspace_results(rows, scope="fixed_base")

    assert summary["n_targets"] == 2
    assert summary["completion_rate"] == pytest.approx(0.5)
    assert summary["attempted_volume_m3"] == pytest.approx(0.002)
    assert summary["reachable_volume_m3"] == pytest.approx(0.001)
    assert summary["self_collision_evaluable"] is False
