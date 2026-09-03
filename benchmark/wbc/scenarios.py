"""Paired, wave-based Stage-2 trajectory benchmark scenarios."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import isaacgym  # noqa: F401 - must precede torch
import torch

from benchmark.wbc.evaluation import (
    WBCPolicyHandle,
    _wbc_step_all,
    wbc_acc_to_result,
    wbc_eval_loop,
)
from benchmark.wbc.suite import build_suite_manifest, validate_coverage
from go1_gym.utils.global_switch import global_switch


def _validate_handles(base, handles):
    if not handles:
        raise ValueError("WBC evaluation requires at least one policy")
    n_per = handles[0].n_envs
    if any(h.n_envs != n_per for h in handles):
        raise ValueError("Paired WBC evaluation requires equal env counts per policy")
    expected_start = 0
    for handle in handles:
        if handle.env_start != expected_start:
            raise ValueError("WBC policy env slices must be contiguous and non-overlapping")
        expected_start = handle.env_end
    if expected_start != base.num_envs:
        raise ValueError("WBC policy env slices must cover the complete shared env pool")
    return n_per


def _load_paired_cell_trajectories(
    base, handles, cell_a: int, cell_b: int, rows: Optional[Sequence[int]] = None
):
    """Load identical concrete bank rows into corresponding policy envs."""
    n_per = _validate_handles(base, handles)
    if rows is None:
        cell_a_template = torch.full((n_per,), cell_a, device=base.device, dtype=torch.long)
        cell_b_template = torch.full((n_per,), cell_b, device=base.device, dtype=torch.long)
        row_tensor = base.traj_bank.sample_rows(
            cell_a_template, cell_b_template, base.traj_curriculum.rng
        )
    else:
        if len(rows) > n_per:
            raise ValueError("A trajectory wave cannot exceed envs per policy")
        row_tensor = torch.as_tensor(rows, device=base.device, dtype=torch.long)

    for handle in handles:
        env_ids = torch.arange(
            handle.env_start, handle.env_start + len(row_tensor), device=base.device
        )
        base.traj_curriculum.cell_A[env_ids] = cell_a
        base.traj_curriculum.cell_B[env_ids] = cell_b
        base.traj_batch.load_from_stacked(env_ids, base.traj_bank.batch, row_tensor)
        base._place_and_reset_trajectories(env_ids)
    return [int(row) for row in row_tensor.tolist()]


def run_wbc_aggregate(
    env,
    handles: List[WBCPolicyHandle],
    cells: Optional[List[tuple]] = None,
    n_steps: int = 500,
    settle_steps: int = 30,
    device: str = "cuda:0",
    suite_rows_per_cell: int = 0,
    validate_feature_coverage: bool = True,
) -> Dict[str, Dict[str, object]]:
    """Evaluate a deterministic held-out suite in paired multi-env waves.

    All policies receive the same concrete bank row in corresponding env slots.
    A suite larger than ``num_envs_per_policy`` is processed in waves, while
    each wave still advances the shared IsaacGym simulation exactly once per
    control step.
    """
    base = env.env
    global_switch.open_switch()
    if cells is None:
        cells = [
            (a, b)
            for a in range(base.traj_curriculum.nA)
            for b in range(base.traj_curriculum.nB)
        ]

    n_per = _validate_handles(base, handles)
    manifest = build_suite_manifest(base, cells, rows_per_cell=suite_rows_per_cell)
    if validate_feature_coverage:
        validate_coverage(manifest)
    elif not manifest["trajectories"]:
        raise ValueError("WBC trajectory suite is empty")
    by_row = {entry["bank_row"]: entry for entry in manifest["trajectories"]}
    out = {
        handle.name: {"wbc_aggregate": [], "wbc_trajectories": []}
        for handle in handles
    }
    dt = float(base.dt)
    all_ids = torch.arange(base.num_envs, device=base.device)

    for cell_index, (cell_a, cell_b) in enumerate(cells):
        cell_rows = [
            entry["bank_row"] for entry in manifest["trajectories"]
            if entry["cell_A"] == cell_a and entry["cell_B"] == cell_b
        ]
        waves = [cell_rows[i:i + n_per] for i in range(0, len(cell_rows), n_per)]
        for wave_index, rows in enumerate(waves):
            label = f"cell_A={cell_a}_B={cell_b}"
            print(
                f"  [cell {cell_index + 1:2d}/{len(cells)} wave {wave_index + 1}/{len(waves)}] {label}",
                end="  ", flush=True,
            )
            env.reset()
            for _ in range(settle_steps):
                _wbc_step_all(env, handles)

            loaded_rows = _load_paired_cell_trajectories(
                base, handles, cell_a, cell_b, rows=rows
            )
            env.clear_cached(all_ids)
            accs = wbc_eval_loop(
                env, handles, n_steps, device,
                valid_env_counts=[len(loaded_rows)] * len(handles),
            )

            for handle, acc in zip(handles, accs):
                aggregate = wbc_acc_to_result(
                    acc, handle.name, "wbc_aggregate", label, dt,
                    cell_a, cell_b, loaded_rows,
                )
                aggregate["wave_index"] = wave_index
                out[handle.name]["wbc_aggregate"].append(aggregate)
                for local_index, row in enumerate(loaded_rows):
                    trajectory_result = dict(by_row[row])
                    trajectory_result.update(
                        run_name=handle.name,
                        scenario="wbc_trajectories",
                        label=trajectory_result["trajectory_id"],
                        wave_index=wave_index,
                        **acc.per_env_summary(local_index, dt),
                    )
                    out[handle.name]["wbc_trajectories"].append(trajectory_result)
                print(
                    f"{handle.name}: pos_err={aggregate['ee_pos_rmse_m']:.3f}m "
                    f"complete={aggregate['completion_rate']:.1%}",
                    end="  " if len(handles) > 1 else "", flush=True,
                )
            print()

    return {"results": out, "suite_manifest": manifest}
