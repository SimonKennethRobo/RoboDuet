"""Paired, wave-based Stage-2 trajectory benchmark scenarios."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import isaacgym  # noqa: F401 - must precede torch
import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_apply, quat_conjugate, quat_mul, quat_rotate_inverse

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
from benchmark.wbc.trace import write_trace_archive
from go1_gym.utils.global_switch import global_switch
from go1_gym.utils import quaternion_to_rpy


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


def _set_benchmark_nominal_initial_states(env, handles, count: int):
    """Replace reset RNG output with one frozen nominal physical state.

    IsaacGym's generic reset randomizes every DOF position and base twist, so
    merely fixing the RNG seed still changes a task when the env count or wave
    packing changes. Build the reference slice from config-defined nominal
    state, copy it across candidates, then clear policy histories before the
    candidate-independent settle phase.
    """
    if count <= 0:
        return
    base = env.env
    reference = torch.arange(
        handles[0].env_start, handles[0].env_start + count, device=base.device
    )
    dof_state = base.dof_state.view(base.num_envs, base.num_dof, 2)
    dof_state[reference, :, 0] = base.default_dof_pos.expand(count, -1)
    dof_state[reference, :, 1] = 0.0
    base.root_states[reference] = base.base_init_state.expand(count, -1)
    base.root_states[reference, :3] += base.env_origins[reference]
    _synchronize_paired_initial_states(base, handles, count)
    all_ids = torch.cat(
        [
            torch.arange(handle.env_start, handle.env_start + count, device=base.device)
            for handle in handles
        ]
    )
    env.clear_cached(all_ids)


def _synchronize_paired_initial_states(base, handles, count: int):
    """Copy one canonical physical state into every active task slot.

    Root positions are translated by each environment origin; orientation,
    twist, DOF position, and DOF velocity remain identical.  Simulator writes
    follow the repository's required DOF-before-root order.

    The source is the first active environment, rather than one source per
    task column.  IsaacGym can materialize nominally identical task columns
    with tiny slot-dependent differences.  Reusing a single source keeps the
    concrete initial state independent of wave packing and total env count.
    """
    if count <= 0:
        return
    reference = torch.arange(handles[0].env_start, handles[0].env_start + count, device=base.device)
    all_ids = []
    canonical_root = base.root_states[reference[:1]].clone()
    dof_state = base.dof_state.view(base.num_envs, base.num_dof, 2)
    canonical_dof = dof_state[reference[:1]].clone()
    for handle in handles:
        target = torch.arange(handle.env_start, handle.env_start + count, device=base.device)
        root = canonical_root.expand(count, -1).clone()
        root[:, :3] += base.env_origins[target] - base.env_origins[reference[0]]
        dof_state[target] = canonical_dof.expand(count, -1, -1)
        base.root_states[target] = root
        all_ids.append(target)

        for name in (
            "actions",
            "last_actions",
            "last_last_actions",
            "last_dof_vel",
            "last_root_vel",
            "last_joint_pos_target",
            "joint_pos_target",
            "commands_dog",
            "commands_arm",
            "goal_command_targets",
            "goal_command_smoothed",
            "upper_plan_actions_raw",
            "arm_policy_actions",
            "arm_ema_motion",
            "gait_indices",
            "feet_air_time",
            "feet_contact_time",
            "last_air_time",
            "last_contact_time",
            "_gait_last_contacts",
            "prev_ee_twist_body",
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

    # IsaacGym refreshes the acquired tensors but not the task's derived
    # body-frame caches. Rebuild them before the first policy observation.
    base.base_pos[:] = base.root_states[: base.num_envs, :3]
    base.base_quat[:] = base.root_states[: base.num_envs, 3:7]
    base.base_lin_vel[:] = quat_rotate_inverse(
        base.base_quat, base.root_states[: base.num_envs, 7:10]
    )
    base.base_ang_vel[:] = quat_rotate_inverse(
        base.base_quat, base.root_states[: base.num_envs, 10:13]
    )
    base.projected_gravity[:] = quat_rotate_inverse(base.base_quat, base.gravity_vec)
    rpy = quaternion_to_rpy(base.base_quat)
    base.roll[:], base.pitch[:], base.y[:] = rpy[:, 0], rpy[:, 1], rpy[:, 2]
    rigid = base.rigid_body_state.view(base.num_envs, base.num_bodies, 13)
    base.end_effector_state[:] = rigid[:, base.ee_idx]
    base.end_effector_state[:, :3] += quat_apply(
        base.end_effector_state[:, 3:7],
        base.ee_local_offset.expand(base.num_envs, -1),
    )
    base.foot_velocities = rigid[:, base.feet_indices, 7:10]
    base.foot_positions = rigid[:, base.feet_indices, :3]

    # Contact tensors describe the pre-write settle frame until physics advances.
    # Rigid-link tensors can have the same lag. Copy canonical derived caches so
    # the first dog/arm observations are paired; the next physics step rebuilds
    # them from the already synchronized root/DOF simulator state.
    reference_contacts = base.contact_forces[reference[:1]].clone()
    reference_ee = base.end_effector_state[reference[:1]].clone()
    reference_foot_positions = base.foot_positions[reference[:1]].clone()
    reference_foot_velocities = base.foot_velocities[reference[:1]].clone()
    for handle in handles:
        target = torch.arange(handle.env_start, handle.env_start + count, device=base.device)
        base.contact_forces[target] = reference_contacts.expand(count, -1, -1)
        translation = base.env_origins[target] - base.env_origins[reference[0]]
        base.end_effector_state[target] = reference_ee.expand(count, -1)
        base.end_effector_state[target, :3] += translation
        base.foot_positions[target] = reference_foot_positions.expand(count, -1, -1) + translation[:, None, :]
        base.foot_velocities[target] = reference_foot_velocities.expand(count, -1, -1)


def _freeze_task_execution_specs(
    base, handles, tasks: Sequence[dict], deadline_s: float, disturbance_schedule=None
):
    reference = torch.arange(handles[0].env_start, handles[0].env_start + len(tasks), device=base.device)
    canonical = reference[:1]
    initial_ee_position_local = (
        base.end_effector_state[canonical, :3] - base.env_origins[canonical]
    )
    initial_ee_quaternion = base.end_effector_state[canonical, 3:7]
    root_local = base.root_states[canonical].clone()
    root_local[:, :3] -= base.env_origins[canonical]
    state = {
        "root_state_env_local": [float(value) for value in root_local[0].tolist()],
        "dof_position_rad": [float(value) for value in base.dof_pos[canonical[0]].tolist()],
        "dof_velocity_rad_s": [float(value) for value in base.dof_vel[canonical[0]].tolist()],
    }
    # Every active slot was synchronized from this same materialized state.
    # Reuse its environment-local representation so task identity does not
    # inherit float32 origin-subtraction noise from the slot assignment.
    for task in tasks:
        row = int(task["bank_row"])
        start_position = base.traj_bank.batch.gamma_p[row, 0]
        start_quaternion = base.traj_bank.batch.gamma_quat[row, 0]
        anchor = initial_ee_position_local[0] - start_position
        orientation_alignment = quat_mul(
            initial_ee_quaternion[0:1],
            quat_conjugate(start_quaternion.unsqueeze(0)),
        )[0]
        finalize_task_spec(
            task,
            initial_state=state,
            anchor_env_local=anchor.tolist(),
            orientation_left_multiplier_xyzw=orientation_alignment.tolist(),
            deadline_s=deadline_s,
            disturbance_schedule=disturbance_schedule,
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
            [task["anchor_env_local_xyz_m"] for task in tasks],
            device=base.device,
            dtype=base.root_states.dtype,
        )
        anchors = anchors + base.env_origins[env_ids]
        orientation_alignment = torch.as_tensor(
            [task["orientation_left_multiplier_xyzw"] for task in tasks],
            device=base.device,
            dtype=base.traj_batch.gamma_quat.dtype,
        )
        reference_quaternions = base.traj_batch.gamma_quat[env_ids]
        base.traj_batch.gamma_quat[env_ids] = quat_mul(
            orientation_alignment[:, None, :]
            .expand_as(reference_quaternions)
            .reshape(-1, 4),
            reference_quaternions.reshape(-1, 4),
        ).view_as(reference_quaternions)
        base._place_and_reset_trajectories(
            env_ids, offset=anchors, ground_relative_z=False
        )
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
    if hasattr(base, "clear_benchmark_terminal_snapshots"):
        base.clear_benchmark_terminal_snapshots(active_ids)
    base.arm_goal_pos_world[active_ids] = base.traj_batch.gamma_p[active_ids, 0]
    base.arm_goal_quat_world[active_ids] = base.traj_batch.gamma_quat[active_ids, 0]
    base._update_ee_task_space_error()
    base._update_goal_reaching_diagnostics()


def _validate_paired_initial_observations(env, handles, count: int, atol=2e-5):
    """Fail before rollout if identical tasks do not yield paired policy inputs."""
    if len(handles) < 2 or count <= 0:
        return
    histories = {
        name: getattr(env, name).clone()
        for name in ("obs_history", "arm_obs_history", "dog_obs_history")
    }
    try:
        arm = env.get_arm_observations()
        dog = env.get_dog_observations()
    finally:
        for name, value in histories.items():
            getattr(env, name).copy_(value)

    reference = handles[0]
    for label, observations in (("arm", arm), ("dog", dog)):
        for key in ("obs", "obs_history"):
            expected = observations[key][reference.env_start : reference.env_start + count]
            for handle in handles[1:]:
                actual = observations[key][handle.env_start : handle.env_start + count]
                delta = torch.abs(actual - expected)
                maximum = float(delta.max().item()) if delta.numel() else 0.0
                if maximum > atol:
                    flat_index = int(torch.argmax(delta).item())
                    raise RuntimeError(
                        f"paired {label} {key} mismatch before policy takeover: "
                        f"{reference.name} vs {handle.name}, max_abs={maximum:.6g}, "
                        f"flat_index={flat_index}"
                    )


def _synchronize_paired_task_caches(base, handles, count: int):
    """Pair task-derived, body-frame caches after per-slice task injection."""
    if len(handles) < 2 or count <= 0:
        return
    reference = torch.arange(
        handles[0].env_start, handles[0].env_start + count, device=base.device
    )
    names = (
        "commands_dog",
        "goal_command_targets",
        "goal_command_smoothed",
        "base_feedforward_cmd",
        "upper_plan_actions_raw",
        "arm_policy_actions",
        "arm_target_pos_body",
        "arm_target_quat_body",
        "commands_arm_obs",
        "goal_rho",
        "goal_rho_prev",
        "goal_rho_valid",
        "goal_manipulability",
        "goal_joint_limit_distance",
        "arm_ema_motion",
    )
    for handle in handles[1:]:
        target = torch.arange(
            handle.env_start, handle.env_start + count, device=base.device
        )
        for name in names:
            value = getattr(base, name, None)
            if torch.is_tensor(value) and value.shape[0] == base.num_envs:
                value[target] = value[reference]


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
    raw_trace_dir=None,
    trace_prefix: str = "wbc",
    scenario_id: str = "nominal",
    disturbance_schedule=None,
) -> Dict[str, Dict[str, object]]:
    """Evaluate one flattened ``cell x trajectory`` task per environment.

    Compatible methods share one simulator. Each method gets an identical
    copy of every logical task in its own environment slice, and a wave steps
    all methods and all loaded tasks together.
    """
    base = env.env
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
        # Settling must be controller-independent. In particular, zero action
        # in IK mode would otherwise solve toward the pre-injection random
        # target while zero end-to-end action would hold nominal arm joints.
        global_switch.switch_flag = False
        env.reset()
        _set_benchmark_nominal_initial_states(env, handles, len(tasks))
        for _ in range(settle_steps):
            _neutral_settle_step(env)

        _synchronize_paired_initial_states(base, handles, len(tasks))
        # Indexed root/DOF writes do not immediately rebuild every articulated
        # rigid-body cache used by the arm observation. One candidate-independent
        # zero-action step materializes the synchronized state before task time 0.
        _neutral_settle_step(env)
        _synchronize_paired_initial_states(base, handles, len(tasks))
        _freeze_task_execution_specs(
            base,
            handles,
            tasks,
            deadline_s=n_steps * dt,
            disturbance_schedule=disturbance_schedule,
        )
        _load_paired_tasks(base, handles, tasks)
        _synchronize_paired_task_caches(base, handles, len(tasks))
        if hasattr(base, "reset_benchmark_viewer_trajectory"):
            base.reset_benchmark_viewer_trajectory()
        global_switch.open_switch()
        for handle in handles:
            if handle.upper_controller is not None:
                handle.upper_controller.reset(
                    base, handle.env_start, handle.env_end, tasks
                )
        env.clear_cached(all_ids)
        _validate_paired_initial_observations(env, handles, len(tasks))
        accs = wbc_eval_loop(
            env,
            handles,
            n_steps,
            device,
            valid_env_counts=[len(tasks)] * len(handles),
            record_raw_traces=raw_trace_dir is not None,
            tasks=tasks,
        )

        for handle, acc in zip(handles, accs):
            trace_record = None
            if raw_trace_dir is not None:
                safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", handle.name).strip("._")
                trace_path = (
                    Path(raw_trace_dir)
                    / f"{trace_prefix}_wave_{wave_index + 1:03d}_{safe_name or 'policy'}.npz"
                )
                trace_record = write_trace_archive(
                    trace_path,
                    acc,
                    [task["task_id"] for task in tasks],
                    dt,
                )
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
                    upper_controller=handle.controller_id,
                    locomotion_policy=handle.locomotion_policy_id,
                    scenario="wbc_aggregate",
                    robustness_scenario=scenario_id,
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
                    upper_controller=handle.controller_id,
                    locomotion_policy=handle.locomotion_policy_id,
                    scenario="wbc_trajectories",
                    robustness_scenario=scenario_id,
                    label=task["trajectory_id"],
                    wave_index=wave_index,
                    **acc.per_env_summary(local_index, dt),
                )
                if trace_record is not None:
                    trajectory_result["raw_trace"] = {
                        **trace_record,
                        "task_index": local_index,
                    }
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
