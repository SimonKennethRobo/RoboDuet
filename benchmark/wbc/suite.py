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


SUITE_VERSION = "wbc-task-spec-v1"


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


def finalize_task_spec(task: dict, *, initial_state: dict, anchor_env_local, deadline_s: float) -> None:
    """Attach execution-defining state and derive the stable public task ID."""
    task["initial_state"] = initial_state
    task["anchor_env_local_m"] = [float(value) for value in anchor_env_local]
    task["disturbance_schedule"] = []
    task["deadline_s"] = float(deadline_s)
    identity = {
        "task_family": task["task_family"],
        "reference_content_sha256": task["content_sha256"],
        "initial_state": task["initial_state"],
        "anchor_env_local_m": task["anchor_env_local_m"],
        "disturbance_schedule": task["disturbance_schedule"],
        "deadline_s": task["deadline_s"],
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
        "dominant_axis": dominant,
        "path_length_m": length,
        "duration_s": duration,
        "mean_speed_mps": length / max(duration, 1e-6),
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
        rows = deterministic_cell_rows(base.traj_bank, cell_a, cell_b)
        if rows_per_cell > 0:
            rows = rows[:rows_per_cell]
        for row in rows:
            item = trajectory_features(base.traj_bank.batch, row)
            item.update(cell_A=int(cell_a), cell_B=int(cell_b))
            entries.append(item)

    manifest = {
        "suite_version": SUITE_VERSION,
        "suite_sha256": None,
        "bank_seed": int(base.cfg.wbc.goal_reaching.trajectory.bank_seed),
        "held_out_status": "unverified_against_training_bank",
        "reference_frame": "environment_local_world",
        "initialization": {
            "reset_profile": "benchmark_nominal_reset",
            "settle_controller": "zero_action",
            "candidate_policy_active_during_settle": False,
        },
        "evaluation_protocol": dict(DEVELOPMENT_TIMED_TRAJECTORY_PROTOCOL),
        "trajectories": entries,
        "coverage": summarize_coverage(entries),
    }
    refresh_suite_hash(manifest)
    return manifest


def summarize_coverage(entries: Iterable[dict]) -> dict:
    items = list(entries)
    axes = {axis: sum(item["dominant_axis"] == axis for item in items) for axis in ("x", "y", "z")}
    curvature = [float(item["curvature_p90_rad_m"]) for item in items]
    spans = {
        axis: max((float(item[f"span_{axis}_m"]) for item in items), default=0.0)
        for axis in ("x", "y", "z")
    }
    return {
        "n_trajectories": len(items),
        "dominant_axis_counts": axes,
        "max_span_m": spans,
        "curvature_p90_range_rad_m": [min(curvature, default=0.0), max(curvature, default=0.0)],
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
