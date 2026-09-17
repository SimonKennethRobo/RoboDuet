"""Backend-independent success protocol and task-result aggregation.

The IsaacGym and MuJoCo runners both emit the fields consumed here.  Keeping
the decision logic free of simulator imports makes saved traces/results
re-scorable without constructing either backend.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Optional


PROTOCOL_VERSION = "legged-manip-dev-v2"

DEVELOPMENT_TIMED_TRAJECTORY_PROTOCOL = {
    "protocol_version": PROTOCOL_VERSION,
    "task_family": "timed_trajectory",
    "status": "posthoc_calibrated_reporting_thresholds",
    "calibration_source": "formal-mujoco-ours-sota/20260916_013919",
    "position_tolerance_m": 0.15,
    "rotation_tolerance_rad": math.radians(40.0),
    "endpoint_progress_min": 0.97,
    "tracking_tube_fraction": 0.40,
    "hold_time_s": 0.50,
}

# These are development reporting thresholds, frozen into every raw archive.
# They classify observed configurations; they are not a proof that an unseen
# target has an IK solution.  The rotational Jacobian threshold matches the
# reach-table builder's documented default and remains comparable because its
# rows are dimensionless.
DEVELOPMENT_KINEMATIC_PROTOCOL = {
    "status": "development_thresholds",
    "reach_model_limit_ratio": 1.0,
    "rho_comfort_hi": 0.85,
    "joint_limit_margin_fraction": 0.02,
    "rot_jacobian_sigma_min": 0.05,
}


def timed_trajectory_success(metrics: Mapping[str, object], protocol=None) -> dict:
    """Apply the public timed-trajectory success contract to one task row."""
    protocol = dict(DEVELOPMENT_TIMED_TRAJECTORY_PROTOCOL if protocol is None else protocol)
    failure_flags = (
        ("numerical_fault", "numerical_fault"),
        ("fall", "fall"),
        ("traj_early_term", "trajectory_early_termination"),
    )
    for field, reason in failure_flags:
        if bool(metrics.get(field)):
            return {"success": False, "end_reason": reason}

    required = {
        "reference_time_s": None,
        "reference_duration_s": None,
        "final_progress": protocol["endpoint_progress_min"],
        "final_ee_pos_error_m": protocol["position_tolerance_m"],
        "final_ee_rot_error_rad": protocol["rotation_tolerance_rad"],
        "tracking_tube_fraction": protocol["tracking_tube_fraction"],
        "endpoint_hold_time_s": protocol["hold_time_s"],
    }
    for field in required:
        value = metrics.get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
            return {"success": False, "end_reason": f"missing_{field}"}

    success = (
        float(metrics["reference_time_s"]) >= float(metrics["reference_duration_s"])
        and float(metrics["final_progress"]) >= float(required["final_progress"])
        and float(metrics["final_ee_pos_error_m"]) <= float(required["final_ee_pos_error_m"])
        and float(metrics["final_ee_rot_error_rad"]) <= float(required["final_ee_rot_error_rad"])
        and float(metrics["tracking_tube_fraction"]) >= float(required["tracking_tube_fraction"])
        and float(metrics["endpoint_hold_time_s"]) >= float(required["endpoint_hold_time_s"])
    )
    if success:
        return {"success": True, "end_reason": "success"}
    if bool(metrics.get("timed_out")):
        return {"success": False, "end_reason": "timeout"}
    return {"success": False, "end_reason": "criteria_not_met"}


def aggregate_task_events(rows: Iterable[Mapping[str, object]]) -> dict:
    """Aggregate task booleans without treating them as continuous metrics."""
    items = list(rows)
    n = len(items)
    if n == 0:
        return {
            "n_episodes": 0,
            "completion_count": 0,
            "completion_rate": None,
            "fall_count": 0,
            "fall_rate": None,
            "timeout_count": 0,
            "timeout_rate": None,
            "traj_early_term_count": 0,
            "traj_early_term_rate": None,
            "numerical_fault_count": 0,
            "numerical_fault_rate": None,
        }

    def count(field: str) -> int:
        return sum(bool(row.get(field)) for row in items)

    completed = count("completed")
    falls = count("fall")
    timeouts = count("timed_out")
    early = count("traj_early_term")
    faults = count("numerical_fault")
    return {
        "n_episodes": n,
        "completion_count": completed,
        "completion_rate": completed / n,
        "fall_count": falls,
        "fall_rate": falls / n,
        "timeout_count": timeouts,
        "timeout_rate": timeouts / n,
        "traj_early_term_count": early,
        "traj_early_term_rate": early / n,
        "numerical_fault_count": faults,
        "numerical_fault_rate": faults / n,
    }


def finite_or_none(value) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(float(value)) else None
