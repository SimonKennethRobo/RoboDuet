"""Paired, wave-based Stage-2 trajectory benchmark scenarios."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import isaacgym  # noqa: F401 - must precede torch
import torch

from benchmark.wbc.evaluation import (
    WBCPolicyHandle,
    _wbc_step_all,
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
            raise ValueError(
                "WBC policy env slices must be contiguous and non-overlapping"
            )
        expected_start = handle.env_end
    if expected_start != base.num_envs:
        raise ValueError(
            "WBC policy env slices must cover the complete shared env pool"
        )
    return n_per


def _load_paired_tasks(base, handles, tasks: Sequence[dict]):
    """Load the same flattened task list into every policy slice."""
    n_per = _validate_handles(base, handles)
    if len(tasks) > n_per:
        raise ValueError("A trajectory wave cannot exceed envs per policy")
    rows = torch.as_tensor(
        [task["bank_row"] for task in tasks], device=base.device, dtype=torch.long
    )
    cell_a = torch.as_tensor(
        [task["cell_A"] for task in tasks], device=base.device, dtype=torch.long
    )
    cell_b = torch.as_tensor(
        [task["cell_B"] for task in tasks], device=base.device, dtype=torch.long
    )
    for handle in handles:
        env_ids = torch.arange(
            handle.env_start, handle.env_start + len(tasks), device=base.device
        )
        base.traj_curriculum.cell_A[env_ids] = cell_a
        base.traj_curriculum.cell_B[env_ids] = cell_b
        base.traj_batch.load_from_stacked(env_ids, base.traj_bank.batch, rows)
        base._place_and_reset_trajectories(env_ids)
        # Settling belongs to simulator warm-up, not to the logical benchmark
        # episode. Start timeout and trajectory early-termination grace at the
        # instant the concrete task is injected.
        base.episode_length_buf[env_ids] = 0
        base.reset_buf[env_ids] = False
        base.time_out_buf[env_ids] = False
        base.reverse_buf[env_ids] = False
        base.traj_early_term[env_ids] = False
        base.traj_term_cause[env_ids] = False


def plan_wbc_waves(manifest: dict, tasks_per_policy: int) -> List[List[dict]]:
    """Pack flattened tasks into waves while keeping each cell intact.

    Keeping a complete cell in one wave preserves exact pooled RMSE and
    smoothness quantiles without a lossy merge across independent rollouts.
    """
    if tasks_per_policy <= 0:
        raise ValueError("tasks_per_policy must be positive")
    cell_blocks = {}
    for task in manifest["trajectories"]:
        cell_blocks.setdefault((task["cell_A"], task["cell_B"]), []).append(task)
    largest = max((len(block) for block in cell_blocks.values()), default=0)
    if largest > tasks_per_policy:
        raise ValueError(
            f"Total env budget provides {tasks_per_policy} envs per policy, but "
            f"the largest curriculum cell has {largest} trajectories. Increase "
            f"--total_envs to at least {largest} * number_of_compatible_methods."
        )
    waves: List[List[dict]] = []
    current: List[dict] = []
    for block in cell_blocks.values():
        if current and len(current) + len(block) > tasks_per_policy:
            waves.append(current)
            current = []
        current.extend(block)
    if current:
        waves.append(current)
    return waves


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
    """Evaluate one flattened ``cell x trajectory`` task per environment.

    Compatible methods share one simulator. Each method gets an identical
    copy of every logical task in its own environment slice, and a wave steps
    all methods and all loaded tasks together.
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
    waves = plan_wbc_waves(manifest, n_per)
    out = {
        handle.name: {"wbc_aggregate": [], "wbc_trajectories": []} for handle in handles
    }
    dt = float(base.dt)
    all_ids = torch.arange(base.num_envs, device=base.device)

    for wave_index, tasks in enumerate(waves):
        cell_keys = list(
            dict.fromkeys((task["cell_A"], task["cell_B"]) for task in tasks)
        )
        print(
            f"  [wave {wave_index + 1:2d}/{len(waves)}] "
            f"{len(tasks)} tasks, {len(cell_keys)} cell(s)",
            flush=True,
        )
        env.reset()
        for _ in range(settle_steps):
            _wbc_step_all(env, handles)

        _load_paired_tasks(base, handles, tasks)
        env.clear_cached(all_ids)
        accs = wbc_eval_loop(
            env,
            handles,
            n_steps,
            device,
            valid_env_counts=[len(tasks)] * len(handles),
        )

        for handle, acc in zip(handles, accs):
            for cell_a, cell_b in cell_keys:
                local_indices = [
                    index
                    for index, task in enumerate(tasks)
                    if task["cell_A"] == cell_a and task["cell_B"] == cell_b
                ]
                rows = [tasks[index]["bank_row"] for index in local_indices]
                label = f"cell_A={cell_a}_B={cell_b}"
                aggregate = dict(
                    run_name=handle.name,
                    scenario="wbc_aggregate",
                    label=label,
                    cell_A=cell_a,
                    cell_B=cell_b,
                    trajectory_bank_rows=rows,
                    wave_index=wave_index,
                    **acc.wbc_summary_for_indices(local_indices, dt),
                )
                out[handle.name]["wbc_aggregate"].append(aggregate)

            for local_index, task in enumerate(tasks):
                trajectory_result = dict(task)
                trajectory_result.update(
                    run_name=handle.name,
                    scenario="wbc_trajectories",
                    label=task["trajectory_id"],
                    wave_index=wave_index,
                    **acc.per_env_summary(local_index, dt),
                )
                out[handle.name]["wbc_trajectories"].append(trajectory_result)

            completed = sum(
                row["completed"]
                for row in out[handle.name]["wbc_trajectories"]
                if row["wave_index"] == wave_index
            )
            print(f"    {handle.name}: {completed}/{len(tasks)} completed", flush=True)

    expected_ids = [task["trajectory_id"] for task in manifest["trajectories"]]
    for handle in handles:
        actual_ids = [
            row["trajectory_id"] for row in out[handle.name]["wbc_trajectories"]
        ]
        if actual_ids != expected_ids or len(set(actual_ids)) != len(actual_ids):
            raise RuntimeError(
                f"{handle.name}: flattened WBC schedule lost, duplicated, or reordered tasks"
            )

    schedule = {
        "total_logical_tasks": len(manifest["trajectories"]),
        "tasks_per_policy_capacity": n_per,
        "num_waves": len(waves),
        "wave_task_counts": [len(wave) for wave in waves],
        "one_env_per_trajectory": True,
    }
    return {"results": out, "suite_manifest": manifest, "schedule": schedule}
