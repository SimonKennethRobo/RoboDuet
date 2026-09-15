"""Backend-neutral raw trace archive and offline timed-trajectory scorer."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from benchmark.wbc.scoring import (
    DEVELOPMENT_KINEMATIC_PROTOCOL,
    DEVELOPMENT_TIMED_TRAJECTORY_PROTOCOL,
    timed_trajectory_success,
)


TRACE_SCHEMA_VERSION = "legged-manip-trace-v3"
SUPPORTED_TRACE_SCHEMA_VERSIONS = {
    "legged-manip-trace-v1",
    "legged-manip-trace-v2",
    TRACE_SCHEMA_VERSION,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_trace_archive(path, accumulator, task_ids, dt_s: float) -> dict:
    """Persist one policy/wave trace without simulator-specific Python objects."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tensors = accumulator.trace_tensors(len(task_ids))
    arrays = {key: value.numpy() for key, value in tensors.items()}
    arrays.update(
        schema_version=np.asarray(TRACE_SCHEMA_VERSION),
        protocol_json=np.asarray(
            json.dumps(accumulator.protocol, sort_keys=True, separators=(",", ":"))
        ),
        kinematic_protocol_json=np.asarray(
            json.dumps(
                accumulator.kinematic_protocol,
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
        task_id=np.asarray(list(task_ids)),
        control_dt_s=np.asarray(float(dt_s), dtype=np.float64),
        requested_steps=np.asarray(int(accumulator.n_steps), dtype=np.int64),
        control_type=np.asarray(str(accumulator.trace_control_type)),
        num_actions_loco=np.asarray(
            int(accumulator.trace_num_actions_loco), dtype=np.int64
        ),
        num_actions_arm=np.asarray(
            int(accumulator.trace_num_actions_arm), dtype=np.int64
        ),
        arm_action_mode=np.asarray(str(accumulator.trace_arm_action_mode)),
        reach_model=np.asarray(str(accumulator.trace_reach_model)),
        self_collision_observability=np.asarray(
            str(accumulator.trace_self_collision_observability)
        ),
    )
    np.savez_compressed(target, **arrays)
    return {
        "path": str(target),
        "format": "numpy_npz_v1",
        "schema_version": TRACE_SCHEMA_VERSION,
        "sha256": _sha256_file(target),
        "task_count": len(task_ids),
        "control_dt_s": float(dt_s),
        "control_type": str(accumulator.trace_control_type),
        "arm_action_mode": str(accumulator.trace_arm_action_mode),
        "reach_model": str(accumulator.trace_reach_model),
    }


def _quat_geodesic_xyzw(reference, actual):
    reference = reference / np.clip(
        np.linalg.norm(reference, axis=-1, keepdims=True), 1e-12, None
    )
    actual = actual / np.clip(
        np.linalg.norm(actual, axis=-1, keepdims=True), 1e-12, None
    )
    dot = np.clip(np.abs(np.sum(reference * actual, axis=-1)), 0.0, 1.0)
    return 2.0 * np.arccos(dot)


def _fd_norm_stats(signal, valid, dt_s: float, order: int):
    value = signal
    mask = valid
    for _ in range(order):
        value = np.diff(value, axis=0) / dt_s
        mask = mask[1:] & mask[:-1]
    norms = np.linalg.norm(value, axis=-1)[mask]
    if not norms.size:
        return None
    return {
        "mean": float(np.mean(norms)),
        "std": float(np.std(norms)),
        "p90": float(np.quantile(norms, 0.90)),
        "p99": float(np.quantile(norms, 0.99)),
    }


def _scalar_stats(values):
    if not values.size:
        return {"std": None, "variance": None, "p95": None, "peak": None}
    return {
        "std": float(np.std(values)),
        "variance": float(np.var(values)),
        "p95": float(np.quantile(values, 0.95)),
        "peak": float(np.max(values)),
    }


def score_trace_archive(path) -> list[dict]:
    """Recompute core accuracy/events/success directly from a saved raw trace."""
    with np.load(path, allow_pickle=False) as trace:
        schema = str(trace["schema_version"].item())
        if schema not in SUPPORTED_TRACE_SCHEMA_VERSIONS:
            raise ValueError(f"unsupported trace schema: {schema}")
        protocol = json.loads(str(trace["protocol_json"].item()))
        kinematic_protocol = (
            json.loads(str(trace["kinematic_protocol_json"].item()))
            if "kinematic_protocol_json" in trace.files
            else dict(DEVELOPMENT_KINEMATIC_PROTOCOL)
        )
        dt = float(trace["control_dt_s"].item())
        requested_steps = int(trace["requested_steps"].item())
        control_type = str(trace["control_type"].item())
        num_actions_loco = int(trace["num_actions_loco"].item())
        num_actions_arm = int(trace["num_actions_arm"].item())
        arm_action_mode = (
            str(trace["arm_action_mode"].item())
            if "arm_action_mode" in trace.files
            else "unknown"
        )
        reach_model = (
            str(trace["reach_model"].item())
            if "reach_model" in trace.files
            else "not_recorded"
        )
        self_collision_status = (
            str(trace["self_collision_observability"].item())
            if "self_collision_observability" in trace.files
            else "not_recorded"
        )
        task_ids = trace["task_id"].tolist()
        rows = []
        for task_index, task_id in enumerate(task_ids):
            present = trace["sample_present"][:, task_index].astype(bool)
            valid = trace["metric_valid"][:, task_index].astype(bool)
            pos_error = np.linalg.norm(
                trace["reference_ee_position_m"][:, task_index]
                - trace["actual_ee_state"][:, task_index, :3],
                axis=-1,
            )
            rot_error = _quat_geodesic_xyzw(
                trace["reference_ee_quaternion_xyzw"][:, task_index],
                trace["actual_ee_state"][:, task_index, 3:7],
            )
            valid_indices = np.flatnonzero(valid)
            numerical_fault = bool(trace["numerical_fault"][:, task_index].any())
            fall = bool(trace["fall"][:, task_index].any())
            timed_out_event = bool(trace["timed_out"][:, task_index].any())
            early = bool(
                trace["trajectory_early_termination"][:, task_index].any()
            )
            metrics = {
                "task_id": str(task_id),
                "ee_pos_rmse_m": None,
                "ee_pos_mae_m": None,
                "ee_pos_error_p95_m": None,
                "ee_pos_error_peak_m": None,
                "ee_pos_error_residual_std_m": None,
                "ee_pos_error_residual_variance_m2": None,
                "ee_rot_rmse_rad": None,
                "ee_rot_mae_rad": None,
                "ee_rot_error_p95_rad": None,
                "ee_rot_error_peak_rad": None,
                "ee_rot_error_residual_std_rad": None,
                "ee_rot_error_residual_variance_rad2": None,
                "final_progress": None,
                "reference_time_s": None,
                "reference_duration_s": None,
                "final_ee_pos_error_m": None,
                "final_ee_rot_error_rad": None,
                "tracking_tube_fraction": None,
                "endpoint_hold_time_s": 0.0,
                "completion_time_s": None,
                "leg_abs_mechanical_power_mean_w": None,
                "leg_positive_mechanical_power_mean_w": None,
                "leg_abs_mechanical_energy_j": None,
                "leg_positive_mechanical_energy_j": None,
                "arm_abs_mechanical_power_mean_w": None,
                "whole_body_abs_mechanical_power_mean_w": None,
                "d_lat_mean_m": None,
                "timing_err_mean_m": None,
                "progress_mean": None,
                "rho_mean": None,
                "rho_max": None,
                "rho_above_hi_rate": None,
                "base_util_mean": None,
                "v_ff_xy_mean": None,
                "v_base_xy_mean": None,
                "manipulability_mean": None,
                "jacobian_sigma_min_mean": None,
                "rot_jacobian_sigma_min_mean": None,
                "joint_limit_margin_min_fraction": None,
                "reach_model_within_limit_fraction": None,
                "reach_model_outside_fraction": None,
                "joint_limit_near_fraction": None,
                "rot_singularity_near_fraction": None,
                "kinematic_observed_feasible_fraction": None,
                "ik_solver_status": (
                    "not_applicable"
                    if arm_action_mode == "end_to_end"
                    else "not_recorded"
                ),
                "ik_step_saturation_fraction": None,
                "ik_solver_invalid_fraction": None,
                "reach_model": reach_model,
                "self_collision_status": self_collision_status,
                "smoothness": {
                    "ee_accel": None,
                    "ee_jerk": None,
                    "base_accel": None,
                    "base_jerk": None,
                    "arm_joint_accel": None,
                    "arm_joint_jerk": None,
                },
                "fall": fall,
                "traj_early_term": early,
                "numerical_fault": numerical_fault,
                "n_env_steps": int(present.sum()),
                "n_valid_metric_samples": int(valid.sum()),
                "terminal_snapshot_used": bool(
                    trace["terminal_snapshot"][:, task_index].any()
                ),
            }
            if valid_indices.size:
                p = pos_error[valid]
                r = rot_error[valid]
                p_stats = _scalar_stats(p)
                r_stats = _scalar_stats(r)
                metrics.update(
                    ee_pos_rmse_m=float(np.sqrt(np.mean(np.square(p)))),
                    ee_pos_mae_m=float(np.mean(p)),
                    ee_pos_error_p95_m=p_stats["p95"],
                    ee_pos_error_peak_m=p_stats["peak"],
                    ee_pos_error_residual_std_m=p_stats["std"],
                    ee_pos_error_residual_variance_m2=p_stats["variance"],
                    ee_rot_rmse_rad=float(np.sqrt(np.mean(np.square(r)))),
                    ee_rot_mae_rad=float(np.mean(r)),
                    ee_rot_error_p95_rad=r_stats["p95"],
                    ee_rot_error_peak_rad=r_stats["peak"],
                    ee_rot_error_residual_std_rad=r_stats["std"],
                    ee_rot_error_residual_variance_rad2=r_stats["variance"],
                    tracking_tube_fraction=float(
                        np.mean(
                            (p <= float(protocol["position_tolerance_m"]))
                            & (r <= float(protocol["rotation_tolerance_rad"]))
                        )
                    ),
                )
                if "trajectory_lateral_error_m" in trace.files:
                    metrics["d_lat_mean_m"] = float(
                        np.mean(trace["trajectory_lateral_error_m"][:, task_index][valid])
                    )
                if "trajectory_timing_error_m" in trace.files:
                    metrics["timing_err_mean_m"] = float(
                        np.mean(
                            np.abs(
                                trace["trajectory_timing_error_m"][:, task_index][valid]
                            )
                        )
                    )
                metrics["progress_mean"] = float(
                    np.mean(trace["trajectory_progress"][:, task_index][valid])
                )

                if "goal_rho" in trace.files and "goal_rho_valid" in trace.files:
                    rho_valid = valid & trace["goal_rho_valid"][:, task_index].astype(bool)
                    rho = trace["goal_rho"][:, task_index][rho_valid]
                    if rho.size:
                        reach_limit = float(
                            kinematic_protocol["reach_model_limit_ratio"]
                        )
                        outside = rho > reach_limit
                        metrics.update(
                            rho_mean=float(np.mean(rho)),
                            rho_max=float(np.max(rho)),
                            reach_model_within_limit_fraction=float(np.mean(~outside)),
                            reach_model_outside_fraction=float(np.mean(outside)),
                        )
                        rho_hi = float(kinematic_protocol["rho_comfort_hi"])
                        metrics["rho_above_hi_rate"] = float(np.mean(rho > rho_hi))

                if "base_feedforward_command" in trace.files:
                    ff = np.linalg.norm(
                        trace["base_feedforward_command"][:, task_index, :2], axis=-1
                    )
                    base_speed = np.linalg.norm(
                        trace["base_root_state"][:, task_index, 7:9], axis=-1
                    )
                    metrics["v_ff_xy_mean"] = float(np.mean(ff[valid]))
                    metrics["v_base_xy_mean"] = float(np.mean(base_speed[valid]))
                    active_ff = valid & (ff > 0.05)
                    if np.any(active_ff):
                        metrics["base_util_mean"] = float(
                            np.mean(base_speed[active_ff] / ff[active_ff])
                        )

                if "manipulability" in trace.files:
                    metrics["manipulability_mean"] = float(
                        np.mean(trace["manipulability"][:, task_index][valid])
                    )
                if "jacobian_sigma_min" in trace.files:
                    metrics["jacobian_sigma_min_mean"] = float(
                        np.mean(trace["jacobian_sigma_min"][:, task_index][valid])
                    )
                if "rot_jacobian_sigma_min" in trace.files:
                    rot_sigma = trace["rot_jacobian_sigma_min"][:, task_index][valid]
                    metrics["rot_jacobian_sigma_min_mean"] = float(np.mean(rot_sigma))
                    metrics["rot_singularity_near_fraction"] = float(
                        np.mean(
                            rot_sigma
                            <= float(kinematic_protocol["rot_jacobian_sigma_min"])
                        )
                    )
                if "joint_limit_distance_fraction" in trace.files:
                    joint_margin = np.min(
                        trace["joint_limit_distance_fraction"][:, task_index], axis=-1
                    )[valid]
                    metrics["joint_limit_margin_min_fraction"] = float(
                        np.min(joint_margin)
                    )
                    metrics["joint_limit_near_fraction"] = float(
                        np.mean(
                            joint_margin
                            <= float(
                                kinematic_protocol["joint_limit_margin_fraction"]
                            )
                        )
                    )
                if (
                    "goal_rho" in trace.files
                    and "goal_rho_valid" in trace.files
                    and "rot_jacobian_sigma_min" in trace.files
                    and "joint_limit_distance_fraction" in trace.files
                ):
                    rho_all = trace["goal_rho"][:, task_index]
                    rho_valid_all = trace["goal_rho_valid"][:, task_index].astype(bool)
                    rot_all = trace["rot_jacobian_sigma_min"][:, task_index]
                    margin_all = np.min(
                        trace["joint_limit_distance_fraction"][:, task_index], axis=-1
                    )
                    infeasible = (
                        (rho_valid_all & (rho_all > float(kinematic_protocol["reach_model_limit_ratio"])))
                        | (margin_all <= float(kinematic_protocol["joint_limit_margin_fraction"]))
                        | (rot_all <= float(kinematic_protocol["rot_jacobian_sigma_min"]))
                    )
                    metrics["kinematic_observed_feasible_fraction"] = float(
                        np.mean(~infeasible[valid])
                    )
                if arm_action_mode != "end_to_end" and "ik_solver_valid" in trace.files:
                    metrics["ik_solver_status"] = "evaluated"
                    metrics["ik_solver_invalid_fraction"] = float(
                        np.mean(~trace["ik_solver_valid"][:, task_index][valid].astype(bool))
                    )
                    metrics["ik_step_saturation_fraction"] = float(
                        np.mean(
                            trace["ik_step_saturated"][:, task_index][valid].astype(bool)
                        )
                    )
                final = int(valid_indices[-1])
                metrics.update(
                    final_progress=float(trace["trajectory_progress"][final, task_index]),
                    reference_time_s=float(trace["reference_time_s"][final, task_index]),
                    reference_duration_s=float(
                        trace["reference_duration_s"][final, task_index]
                    ),
                    final_ee_pos_error_m=float(pos_error[final]),
                    final_ee_rot_error_rad=float(rot_error[final]),
                )
                within = (
                    (pos_error <= float(protocol["position_tolerance_m"]))
                    & (rot_error <= float(protocol["rotation_tolerance_rad"]))
                    & valid
                    & (
                        trace["trajectory_progress"][:, task_index]
                        >= float(protocol["endpoint_progress_min"])
                    )
                )
                hold_steps = 0
                for index in valid_indices[::-1]:
                    if not within[index]:
                        break
                    hold_steps += 1
                metrics["endpoint_hold_time_s"] = hold_steps * dt

                dof_velocity = trace["dof_velocity_rad_s"][:, task_index]
                command = trace["actuator_command"][:, task_index]
                leg_power = np.sum(
                    np.abs(
                        command[:, :num_actions_loco]
                        * dof_velocity[:, :num_actions_loco]
                    ),
                    axis=-1,
                )
                metrics["leg_abs_mechanical_power_mean_w"] = float(
                    np.mean(leg_power[valid])
                )
                leg_positive_power = np.sum(
                    np.clip(
                        command[:, :num_actions_loco]
                        * dof_velocity[:, :num_actions_loco],
                        0.0,
                        None,
                    ),
                    axis=-1,
                )
                metrics["leg_positive_mechanical_power_mean_w"] = float(
                    np.mean(leg_positive_power[valid])
                )
                metrics["leg_abs_mechanical_energy_j"] = float(
                    np.sum(
                        trace["leg_abs_mechanical_energy_step_j"][:, task_index][valid]
                    )
                )
                metrics["leg_positive_mechanical_energy_j"] = float(
                    np.sum(
                        trace["leg_positive_mechanical_energy_step_j"][:, task_index][valid]
                    )
                )
                if control_type == "P":
                    arm_slice = slice(
                        num_actions_loco, num_actions_loco + num_actions_arm
                    )
                    arm_power = np.sum(
                        np.abs(command[:, arm_slice] * dof_velocity[:, arm_slice]),
                        axis=-1,
                    )
                    whole_power = np.sum(np.abs(command * dof_velocity), axis=-1)
                    metrics["arm_abs_mechanical_power_mean_w"] = float(
                        np.mean(arm_power[valid])
                    )
                    metrics["whole_body_abs_mechanical_power_mean_w"] = float(
                        np.mean(whole_power[valid])
                    )
                else:
                    metrics["arm_abs_mechanical_power_mean_w"] = None
                    metrics["whole_body_abs_mechanical_power_mean_w"] = None

                leg_torque = command[:, :num_actions_loco][valid]
                metrics["leg_torque_rms_nm"] = float(
                    np.sqrt(np.mean(np.square(leg_torque)))
                )
                metrics["leg_torque_abs_peak_nm"] = float(
                    np.max(np.abs(leg_torque))
                )
                if "actuator_torque_limit" in trace.files:
                    limits = trace["actuator_torque_limit"][
                        :, task_index, :num_actions_loco
                    ][valid]
                    metrics["leg_torque_saturation_fraction"] = float(
                        np.mean(np.abs(leg_torque) >= 0.99 * limits)
                    )
                else:
                    metrics["leg_torque_saturation_fraction"] = None

                base_position = trace["base_root_state"][:, task_index, :3][valid]
                base_delta = base_position[-1] - base_position[0]
                base_segment = np.diff(base_position, axis=0)
                base_path = float(np.linalg.norm(base_segment[:, :2], axis=-1).sum())
                base_net = float(np.linalg.norm(base_delta[:2]))
                base_span = np.ptp(base_position[:, :2], axis=0)
                metrics.update(
                    base_xy_path_length_m=base_path,
                    base_xy_net_displacement_m=base_net,
                    base_xy_path_excess_m=base_path - base_net,
                    base_xy_bounding_box_area_m2=float(np.prod(base_span)),
                    base_z_drift_m=float(base_delta[2]),
                    base_z_drift_abs_m=float(abs(base_delta[2])),
                )

                if "foot_linear_velocity_mps" in trace.files:
                    foot_velocity = trace["foot_linear_velocity_mps"][:, task_index]
                    foot_force = trace["foot_contact_force_n"][:, task_index]
                    contact = foot_force[..., 2] > 1.0
                    contact_valid = contact & valid[:, None]
                    slip = np.linalg.norm(foot_velocity[..., :2], axis=-1)
                    metrics["foot_slip_speed_mean_contact_mps"] = (
                        float(np.mean(slip[contact_valid]))
                        if np.any(contact_valid)
                        else None
                    )
                    metrics["foot_contact_fraction"] = float(
                        np.sum(contact_valid) / (np.sum(valid) * contact.shape[-1])
                    )
                    supported = np.any(contact, axis=-1)
                    metrics["support_fraction"] = float(np.mean(supported[valid]))
                    metrics["no_support_fraction"] = float(np.mean(~supported[valid]))
                else:
                    metrics.update(
                        foot_slip_speed_mean_contact_mps=None,
                        foot_contact_fraction=None,
                        support_fraction=None,
                        no_support_fraction=None,
                    )
                metrics["smoothness"] = {
                    "ee_accel": _fd_norm_stats(
                        trace["actual_ee_grasp_linear_velocity_mps"][:, task_index],
                        valid,
                        dt,
                        1,
                    ),
                    "ee_jerk": _fd_norm_stats(
                        trace["actual_ee_grasp_linear_velocity_mps"][:, task_index],
                        valid,
                        dt,
                        2,
                    ),
                    "base_accel": _fd_norm_stats(
                        trace["base_root_state"][:, task_index, 7:10], valid, dt, 1
                    ),
                    "base_jerk": _fd_norm_stats(
                        trace["base_root_state"][:, task_index, 7:10], valid, dt, 2
                    ),
                    "arm_joint_accel": _fd_norm_stats(
                        dof_velocity[:, num_actions_loco : num_actions_loco + num_actions_arm],
                        valid,
                        dt,
                        1,
                    ),
                    "arm_joint_jerk": _fd_norm_stats(
                        dof_velocity[:, num_actions_loco : num_actions_loco + num_actions_arm],
                        valid,
                        dt,
                        2,
                    ),
                }

                running_valid = 0
                running_tube = 0
                running_hold = 0
                for index in valid_indices:
                    running_valid += 1
                    in_tube = (
                        pos_error[index] <= float(protocol["position_tolerance_m"])
                        and rot_error[index] <= float(protocol["rotation_tolerance_rad"])
                    )
                    running_tube += int(in_tube)
                    at_endpoint = in_tube and (
                        trace["trajectory_progress"][index, task_index]
                        >= float(protocol["endpoint_progress_min"])
                    )
                    running_hold = running_hold + 1 if at_endpoint else 0
                    if (
                        trace["reference_time_s"][index, task_index]
                        >= trace["reference_duration_s"][index, task_index]
                        and running_tube / running_valid
                        >= float(protocol["tracking_tube_fraction"])
                        and running_hold * dt >= float(protocol["hold_time_s"])
                    ):
                        metrics["completion_time_s"] = float((index + 1) * dt)
                        break

            deadline_reached = (
                metrics["n_env_steps"] >= requested_steps
                and metrics["completion_time_s"] is None
            )
            metrics["timed_out"] = timed_out_event or deadline_reached
            metrics["benchmark_deadline_reached"] = deadline_reached
            decision = timed_trajectory_success(metrics, protocol)
            metrics.update(
                completed=decision["success"],
                success=decision["success"],
                incomplete=not decision["success"],
                end_reason=decision["end_reason"],
            )
            rows.append(metrics)
    return rows


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Rescore a WBC raw trace archive")
    parser.add_argument("archives", nargs="+")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    payload = {
        str(Path(path)): score_trace_archive(path) for path in args.archives
    }
    encoded = json.dumps(payload, indent=2, allow_nan=False)
    if args.output:
        Path(args.output).write_text(encoded + "\n", encoding="utf-8")
    else:
        sys.stdout.write(encoded + "\n")


if __name__ == "__main__":
    main()
