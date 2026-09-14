"""Versioned, deterministic trajectory-suite support for the WBC benchmark."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Dict, Iterable, List, Sequence

import torch


SUITE_VERSION = "wbc-bank-v1"


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


def trajectory_features(batch, row: int) -> Dict[str, object]:
    """Measure generated content rather than trusting generator parameters."""
    n = _valid_gamma_count(batch, row)
    p = batch.gamma_p[row, :n].detach().float().cpu()
    q = batch.gamma_quat[row, :n].detach().float().cpu()
    tangent = batch.gamma_tangent[row, :n].detach().float().cpu()
    s = batch.gamma_s[row, :n].detach().float().cpu()

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

    payload = torch.cat((p.flatten(), q.flatten(), s.flatten())).numpy().tobytes()
    content_hash = hashlib.sha256(payload).hexdigest()
    return {
        "trajectory_id": f"bank-row-{row:06d}-{content_hash[:12]}",
        "bank_row": int(row),
        "content_sha256": content_hash,
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

    suite_hash = hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "suite_version": SUITE_VERSION,
        "suite_sha256": suite_hash,
        "bank_seed": int(base.cfg.wbc.goal_reaching.trajectory.bank_seed),
        "trajectories": entries,
        "coverage": summarize_coverage(entries),
    }


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
