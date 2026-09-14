"""Paired, wave-based Stage-2 trajectory benchmark scenarios."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import isaacgym  # noqa: F401 - must precede torch
import torch
from isaacgym import gymtorch

from benchmark.wbc.evaluation import (
    WBCPolicyHandle,
    wbc_eval_loop,
)
from benchmark.wbc.suite import (
    build_suite_manifest,
    finalize_task_spec,
    refresh_suite_hash,
    validate_coverage,
)
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


def _neutral_settle_step(env):
    """Advance every slot with candidate-independent zero actions."""
    base = env.env
    if env.num_plan_actions > 0:
        env.plan(torch.zeros(base.num_envs, env.num_plan_actions, device=base.device))
    env.step(
        torch.zeros(base.num_envs, base.num_actions_loco, device=base.device),
        torch.zeros(base.num_envs, base.num_actions_arm, device=base.device),
    )


def _synchronize_paired_initial_states(base, handles, count: int):
    """Copy one canonical physical state into every policy slice.

    Root positions are translated by each environment origin; orientation,
    twist, DOF position, and DOF velocity remain identical.  Simulator writes
    follow the repository's required DOF-before-root order.
    """
    if count <= 0:
        return
    reference = torch.arange(handles[0].env_start, handles[0].env_start + count, device=base.device)
    all_ids = []
    ref_root = base.root_states[reference].clone()
    ref_dof = base.dof_state[reference].clone()
    for handle in handles:
        target = torch.arange(handle.env_start, handle.env_start + count, device=base.device)
        root = ref_root.clone()
        root[:, :3] += base.env_origins[target] - base.env_origins[reference]
        base.dof_state[target] = ref_dof
        base.root_states[target] = root
        all_ids.append(target)

        for name in (
            "actions",
            "last_actions",
            "last_last_actions",
            "commands_dog",
            "commands_arm",
            "goal_command_targets",
            "goal_command_smoothed",
            "upper_plan_actions_raw",
            "arm_policy_actions",
        ):
            value = getattr(base, name, None)
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == base.num_envs:
                value[target] = value[reference]

    env_ids = torch.cat(all_ids).to(dtype=torch.int32)
    base.gym.set_dof_state_tensor_indexed(
        base.sim,
        gymtorch.unwrap_tensor(base.dof_state),
        gymtorch.unwrap_tensor(env_ids),
        len(env_ids),
    )
    base.gym.set_actor_root_state_tensor_indexed(
        base.sim,
        gymtorch.unwrap_tensor(base.root_states),
        gymtorch.unwrap_tensor(env_ids),
        len(env_ids),
    )
    base.gym.refresh_dof_state_tensor(base.sim)
    base.gym.refresh_actor_root_state_tensor(base.sim)
    base.gym.refresh_rigid_body_state_tensor(base.sim)
    base.gym.refresh_net_contact_force_tensor(base.sim)
    base.gym.refresh_jacobian_tensors(base.sim)


def _freeze_task_execution_specs(base, handles, tasks: Sequence[dict], deadline_s: float):
    reference = torch.arange(handles[0].env_start, handles[0].env_start + len(tasks), device=base.device)
    anchors = base._default_trajectory_anchor(reference) - base.env_origins[reference]
    root_local = base.root_states[reference].clone()
    root_local[:, :3] -= base.env_origins[reference]
    for index, task in enumerate(tasks):
        state = {
            "root_state_env_local": [float(value) for value in root_local[index].tolist()],
            "dof_position_rad": [float(value) for value in base.dof_pos[reference[index]].tolist()],
            "dof_velocity_rad_s": [float(value) for value in base.dof_vel[reference[index]].tolist()],
        }
        finalize_task_spec(
            task,
            initial_state=state,
            anchor_env_local=anchors[index].tolist(),
            deadline_s=deadline_s,
        )


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
        anchors = torch.as_tensor(
            [task["anchor_env_local_m"] for task in tasks],
            device=base.device,
            dtype=base.root_states.dtype,
        ) + base.env_origins[env_ids]
        base._place_and_reset_trajectories(env_ids, offset=anchors)
        # Settling belongs to simulator warm-up, not to the logical benchmark
        # episode. Start timeout and trajectory early-termination grace at the
        # instant the concrete task is injected.
        base.episode_length_buf[env_ids] = 0
        base.reset_buf[env_ids] = False
        base.time_out_buf[env_ids] = False
        base.reverse_buf[env_ids] = False
        base.traj_early_term[env_ids] = False
        base.traj_term_cause[env_ids] = False

    # The first arm observation after injection must see the new t=0
    # reference, not the reference left by the settle episode.
    active_ids = torch.cat(
        [
            torch.arange(handle.env_start, handle.env_start + len(tasks), device=base.device)
            for handle in handles
        ]
    )
    base.arm_goal_pos_world[active_ids] = base.traj_batch.gamma_p[active_ids, 0]
    base.arm_goal_quat_world[active_ids] = base.traj_batch.gamma_quat[active_ids, 0]
    base._update_ee_task_space_error()
    base._update_goal_reaching_diagnostics()


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
    progress_callback=None,
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
            _neutral_settle_step(env)

        _synchronize_paired_initial_states(base, handles, len(tasks))
        _freeze_task_execution_specs(base, handles, tasks, deadline_s=n_steps * dt)
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
        refresh_suite_hash(manifest)
        if progress_callback is not None:
            progress_callback(
                {
                    "completed_wave_index": wave_index,
                    "results": out,
                    "suite_manifest": manifest,
                }
            )

    expected_ids = [task["trajectory_id"] for task in manifest["trajectories"]]
    refresh_suite_hash(manifest)
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
