"""Versioned, deterministic trajectory-suite support for the WBC benchmark."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch

from benchmark.wbc.scoring import DEVELOPMENT_TIMED_TRAJECTORY_PROTOCOL


SUITE_VERSION = "wbc-task-spec-v3"
TIMING_CONTRACT_VERSION = "bounded-se3-arc-time-law-v1"


def deterministic_cell_rows(bank, cell_a: int, cell_b: int) -> List[int]:
    """Return every concrete row in a cell in stable bank order."""
    start = int(bank.cell_offset[cell_a, cell_b])
    return list(range(start, start + int(bank.per_cell)))


def _valid_gamma_count(batch, row: int) -> int:
    length = batch.L[row]
    # Padding repeats the terminal arc-length value. Keep the first terminal
    # point, excluding the repeated padded tail.
    terminal = torch.nonzero(batch.gamma_s[row] >= length - 1e-7, as_tuple=False)
    return max(2, int(terminal[0].item()) + 1) if terminal.numel() else batch.max_gamma_points


def _valid_time_law_count(batch, row: int) -> int:
    duration = batch.T[row]
    terminal = torch.nonzero(batch.tl_t[row] >= duration - 1e-7, as_tuple=False)
    return max(2, int(terminal[0].item()) + 1) if terminal.numel() else batch.max_tl_points


def _hash_tensors(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().contiguous().cpu()
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def finalize_task_spec(
    task: dict,
    *,
    initial_state: dict,
    anchor_env_local,
    deadline_s: float,
    orientation_left_multiplier_xyzw=(0.0, 0.0, 0.0, 1.0),
    disturbance_schedule=None,
) -> None:
    """Attach execution-defining state and derive the stable public task ID."""
    task["initial_state"] = initial_state
    task["anchor_env_local_xyz_m"] = [float(value) for value in anchor_env_local[:3]]
    task["anchor_env_local_xy_m"] = task["anchor_env_local_xyz_m"][:2]
    task["orientation_left_multiplier_xyzw"] = [
        float(value) for value in orientation_left_multiplier_xyzw
    ]
    task["disturbance_schedule"] = list(disturbance_schedule or [])
    task["deadline_s"] = float(deadline_s)
    task["height_reference"] = "environment_local_translation"
    identity = {
        "task_family": task["task_family"],
        "reference_content_sha256": task["content_sha256"],
        "initial_state": task["initial_state"],
        "anchor_env_local_xyz_m": task["anchor_env_local_xyz_m"],
        "orientation_left_multiplier_xyzw": task[
            "orientation_left_multiplier_xyzw"
        ],
        "disturbance_schedule": task["disturbance_schedule"],
        "deadline_s": task["deadline_s"],
        "height_reference": task["height_reference"],
        "evaluation_protocol": DEVELOPMENT_TIMED_TRAJECTORY_PROTOCOL,
        "reference_frame": "environment_local_world",
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    task["task_spec_sha256"] = digest
    task["task_id"] = f"timed-trajectory-{digest[:16]}"


def refresh_suite_hash(manifest: dict) -> str:
    payload = {
        "suite_version": manifest["suite_version"],
        "reference_frame": manifest["reference_frame"],
        "initialization": manifest["initialization"],
        "evaluation_protocol": manifest["evaluation_protocol"],
        "timing_contract": manifest.get("timing_contract"),
        "reference_archive": manifest.get("reference_archive"),
        "trajectories": manifest["trajectories"],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest["suite_sha256"] = digest
    return digest


def write_reference_archive(path, base, manifest: dict) -> dict:
    """Save complete replayable references in a backend-neutral NPZ file."""
    rows = torch.as_tensor(
        [task["bank_row"] for task in manifest["trajectories"]],
        device=base.traj_bank.batch.device,
        dtype=torch.long,
    )
    batch = base.traj_bank.batch

    def array(value):
        return value[rows].detach().cpu().numpy()

    target = Path(path)
    np.savez_compressed(
        target,
        gamma_s=array(batch.gamma_s),
        gamma_p=array(batch.gamma_p),
        gamma_quat_xyzw=array(batch.gamma_quat),
        gamma_tangent=array(batch.gamma_tangent),
        path_length_m=array(batch.L),
        tl_t=array(batch.tl_t),
        tl_s=array(batch.tl_s),
        tl_sdot=array(batch.tl_sdot),
        duration_s=array(batch.T),
        gamma_points=np.asarray(
            [task["gamma_points"] for task in manifest["trajectories"]], dtype=np.int32
        ),
        time_law_points=np.asarray(
            [task["time_law_points"] for task in manifest["trajectories"]], dtype=np.int32
        ),
        task_id=np.asarray([task["task_id"] for task in manifest["trajectories"]]),
    )
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    record = {
        "path": target.name,
        "format": "numpy_npz_v1",
        "sha256": digest.hexdigest(),
        "quaternion_order": "xyzw",
        "position_z_reference": "task_anchor_environment_local_translation",
    }
    manifest["reference_archive"] = record
    refresh_suite_hash(manifest)
    return record


def trajectory_features(batch, row: int) -> Dict[str, object]:
    """Measure generated content rather than trusting generator parameters."""
    n = _valid_gamma_count(batch, row)
    p = batch.gamma_p[row, :n].detach().float().cpu()
    q = batch.gamma_quat[row, :n].detach().float().cpu()
    tangent = batch.gamma_tangent[row, :n].detach().float().cpu()
    s = batch.gamma_s[row, :n].detach().float().cpu()
    m = _valid_time_law_count(batch, row)
    tl_t = batch.tl_t[row, :m].detach().float().cpu()
    tl_s = batch.tl_s[row, :m].detach().float().cpu()
    tl_sdot = batch.tl_sdot[row, :m].detach().float().cpu()

    span = p.max(dim=0).values - p.min(dim=0).values
    xy_delta = p[-1, :2] - p[0, :2]
    dominant = ("x", "y", "z")[int(torch.argmax(span).item())]
    ds = (s[1:] - s[:-1]).clamp_min(1e-6)
    dots = (tangent[:-1] * tangent[1:]).sum(dim=-1).clamp(-1.0, 1.0)
    curvature = torch.acos(dots) / ds

    qdots = (q[:-1] * q[1:]).sum(dim=-1).abs().clamp(max=1.0)
    rotation_steps = 2.0 * torch.acos(qdots)
    duration = float(batch.T[row].detach().cpu().item())
    length = float(batch.L[row].detach().cpu().item())

    def quantile(values, qv):
        return float(torch.quantile(values, qv).item()) if values.numel() else 0.0

    def absolute_derivative(values, times):
        if values.shape[0] < 2:
            return torch.empty(0)
        delta_t = (times[1:] - times[:-1]).clamp_min(1e-8)
        if values.ndim > 1:
            delta_t = delta_t.unsqueeze(-1)
        return (values[1:] - values[:-1]) / delta_t

    # Reconstruct the Cartesian reference at the time-law samples. This keeps
    # the public manifest explicit about the difference between SE(3) arc
    # speed (metre-equivalent because rotation is weighted by lambda) and
    # physical EE linear speed.
    time_s = tl_s.clamp(min=s[0], max=s[-1])
    indices = torch.searchsorted(s, time_s, right=True).clamp(1, s.shape[0] - 1) - 1
    s0, s1 = s[indices], s[indices + 1]
    fraction = ((time_s - s0) / (s1 - s0).clamp_min(1e-9)).clamp(0.0, 1.0)
    reference_position = p[indices] + fraction.unsqueeze(-1) * (
        p[indices + 1] - p[indices]
    )
    linear_velocity = absolute_derivative(reference_position, tl_t)
    linear_speed = torch.linalg.vector_norm(linear_velocity, dim=-1)
    linear_acceleration = absolute_derivative(
        linear_velocity, 0.5 * (tl_t[1:] + tl_t[:-1])
    )
    linear_acceleration_norm = (
        torch.linalg.vector_norm(linear_acceleration, dim=-1)
        if linear_acceleration.numel()
        else torch.empty(0)
    )
    arc_acceleration = absolute_derivative(tl_sdot, tl_t).abs()

    content_hash = _hash_tensors(
        p, q, tangent, s, tl_t, tl_s, tl_sdot, batch.L[row : row + 1], batch.T[row : row + 1]
    )
    return {
        "trajectory_id": f"bank-row-{row:06d}-{content_hash[:12]}",
        "task_family": "timed_trajectory",
        "bank_row": int(row),
        "content_sha256": content_hash,
        "gamma_points": n,
        "time_law_points": m,
        "span_x_m": float(span[0]),
        "span_y_m": float(span[1]),
        "span_z_m": float(span[2]),
        "min_z_m": float(p[:, 2].min().item()),
        "max_z_m": float(p[:, 2].max().item()),
        "xy_displacement_m": float(torch.linalg.vector_norm(xy_delta).item()),
        "xy_heading_rad": float(torch.atan2(xy_delta[1], xy_delta[0]).item()),
        "dominant_axis": dominant,
        "path_length_m": length,
        "duration_s": duration,
        "timing_contract_version": TIMING_CONTRACT_VERSION,
        "se3_arc_speed_mean_mps_equiv": length / max(duration, 1e-6),
        "se3_arc_speed_p95_mps_equiv": quantile(tl_sdot, 0.95),
        "se3_arc_speed_peak_mps_equiv": float(tl_sdot.max().item()),
        "se3_arc_acceleration_abs_p95_mps2_equiv": quantile(arc_acceleration, 0.95),
        "se3_arc_acceleration_abs_peak_mps2_equiv": (
            float(arc_acceleration.max().item()) if arc_acceleration.numel() else 0.0
        ),
        "reference_linear_speed_mean_mps": (
            float(linear_speed.mean().item()) if linear_speed.numel() else 0.0
        ),
        "reference_linear_speed_p95_mps": quantile(linear_speed, 0.95),
        "reference_linear_speed_peak_mps": (
            float(linear_speed.max().item()) if linear_speed.numel() else 0.0
        ),
        "reference_linear_acceleration_p95_mps2": quantile(
            linear_acceleration_norm, 0.95
        ),
        "reference_linear_acceleration_peak_mps2": (
            float(linear_acceleration_norm.max().item())
            if linear_acceleration_norm.numel()
            else 0.0
        ),
        "curvature_p50_rad_m": quantile(curvature, 0.50),
        "curvature_p90_rad_m": quantile(curvature, 0.90),
        "curvature_p99_rad_m": quantile(curvature, 0.99),
        "curvature_max_rad_m": float(curvature.max().item()) if curvature.numel() else 0.0,
        "rotation_step_p90_rad": quantile(rotation_steps, 0.90),
        "rotation_total_rad": float(rotation_steps.sum().item()),
    }


def build_suite_manifest(base, cells: Sequence[tuple], rows_per_cell: int = 0) -> dict:
    entries = []
    for cell_a, cell_b in cells:
        _, timing = base.traj_curriculum.params_for_cell(cell_a, cell_b)
        rows = deterministic_cell_rows(base.traj_bank, cell_a, cell_b)
        if rows_per_cell > 0:
            rows = rows[:rows_per_cell]
        for row in rows:
            item = trajectory_features(base.traj_bank.batch, row)
            item.update(
                cell_A=int(cell_a),
                cell_B=int(cell_b),
                configured_minimum_duration_s=float(timing["T"]),
                configured_se3_arc_speed_limit_mps_equiv=float(timing["v_max"]),
                configured_se3_arc_acceleration_limit_mps2_equiv=(
                    float(timing["a_max"]) if timing.get("a_max") is not None else None
                ),
                configured_reference_linear_acceleration_limit_mps2=(
                    float(timing["linear_a_max"])
                    if timing.get("linear_a_max") is not None
                    else None
                ),
            )
            entries.append(item)

    manifest = {
        "suite_version": SUITE_VERSION,
        "suite_sha256": None,
        "bank_seed": int(base.cfg.wbc.goal_reaching.trajectory.bank_seed),
        "curriculum_grid": {
            "geometry_levels": int(base.traj_curriculum.nA),
            "timing_levels": int(base.traj_curriculum.nB),
        },
        "held_out_status": "unverified_against_training_bank",
        "reference_frame": "environment_local_world",
        "initialization": {
            "reset_profile": "benchmark_nominal_reset",
            "settle_controller": "zero_action",
            "candidate_policy_active_during_settle": False,
            "post_sync_materialization_steps": 1,
            "paired_policy_input_validation": True,
        },
        "evaluation_protocol": dict(DEVELOPMENT_TIMED_TRAJECTORY_PROTOCOL),
        "timing_contract": {
            "version": TIMING_CONTRACT_VERSION,
            "arc_metric": "translation_m_plus_lambda_weighted_rotation_rad",
            "T_semantics": "minimum_duration_s",
            "v_max_semantics": "hard_se3_arc_speed_limit_mps_equiv",
            "a_max_semantics": "hard_se3_arc_acceleration_limit_mps2_equiv",
            "linear_a_max_semantics": "hard_cartesian_reference_acceleration_limit_mps2",
            "duration_semantics": "extended_to_satisfy_derivative_limits",
        },
        "trajectories": entries,
        "coverage": summarize_coverage(entries),
    }
    refresh_suite_hash(manifest)
    return manifest


def summarize_coverage(entries: Iterable[dict]) -> dict:
    items = list(entries)
    axes = {axis: sum(item["dominant_axis"] == axis for item in items) for axis in ("x", "y", "z")}
    curvature = [float(item["curvature_p90_rad_m"]) for item in items]
    xy_displacements = [float(item["xy_displacement_m"]) for item in items]
    spans = {
        axis: max((float(item[f"span_{axis}_m"]) for item in items), default=0.0)
        for axis in ("x", "y", "z")
    }
    return {
        "n_trajectories": len(items),
        "dominant_axis_counts": axes,
        "max_span_m": spans,
        "curvature_p90_range_rad_m": [min(curvature, default=0.0), max(curvature, default=0.0)],
        "xy_displacement_range_m": [
            min(xy_displacements, default=0.0),
            max(xy_displacements, default=0.0),
        ],
    }


def validate_coverage(manifest: dict) -> None:
    """Reject a nominal suite that silently loses a requested feature family."""
    coverage = manifest["coverage"]
    if coverage["n_trajectories"] == 0:
        raise ValueError("WBC trajectory suite is empty")
    missing_axes = [axis for axis, count in coverage["dominant_axis_counts"].items() if count == 0]
    if missing_axes:
        raise ValueError(f"WBC suite has no dominant-axis coverage for: {', '.join(missing_axes)}")
    lo, hi = coverage["curvature_p90_range_rad_m"]
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        raise ValueError("WBC suite does not contain distinct curvature levels")
    hardest_a = int(manifest["curriculum_grid"]["geometry_levels"]) - 1
    hardest = [
        float(item["xy_displacement_m"])
        for item in manifest["trajectories"]
        if int(item["cell_A"]) == hardest_a
    ]
    if hardest and (min(hardest) < 4.9 or max(hardest) > 5.1):
        raise ValueError(
            "WBC hardest geometry level must have approximately 5 m XY displacement"
        )
    hardest_items = [
        item for item in manifest["trajectories"] if int(item["cell_A"]) == hardest_a
    ]
    if hardest_items:
        if any(abs(float(item["min_z_m"])) > 0.01 for item in hardest_items):
            raise ValueError("WBC hardest geometry level must reach ground-relative z=0 m")
        if any(abs(float(item["max_z_m"]) - 1.5) > 0.01 for item in hardest_items):
            raise ValueError("WBC hardest geometry level must reach ground-relative z=1.5 m")
        if any(float(item["curvature_p99_rad_m"]) < 2.0 for item in hardest_items):
            raise ValueError("WBC hardest geometry level lacks high-curvature segments")
    for item in manifest["trajectories"]:
        speed_limit = float(item["configured_se3_arc_speed_limit_mps_equiv"])
        if float(item["se3_arc_speed_peak_mps_equiv"]) > speed_limit + 1e-5:
            raise ValueError("WBC time law exceeds its declared SE(3) arc-speed limit")
        acceleration_limit = item["configured_se3_arc_acceleration_limit_mps2_equiv"]
        if (
            acceleration_limit is not None
            and float(item["se3_arc_acceleration_abs_peak_mps2_equiv"])
            > float(acceleration_limit) + 1e-4
        ):
            raise ValueError(
                "WBC time law exceeds its declared SE(3) arc-acceleration limit"
            )
        linear_acceleration_limit = item[
            "configured_reference_linear_acceleration_limit_mps2"
        ]
        if (
            linear_acceleration_limit is not None
            and float(item["reference_linear_acceleration_peak_mps2"])
            > float(linear_acceleration_limit) + 1e-4
        ):
            raise ValueError(
                "WBC time law exceeds its declared Cartesian reference-acceleration limit"
            )
