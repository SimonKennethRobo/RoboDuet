"""Backend-neutral workspace-probe task generation and aggregation.

The runner for either simulator may consume the generated target specs.  This
module deliberately owns only the public grid and scoring contract, so fixed
base, posture-enabled, and bounded whole-body probes can share identical
targets without sharing controller implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
from itertools import product
from typing import Iterable, Mapping, Sequence


WORKSPACE_SCHEMA_VERSION = "workspace-probe-v1"
WORKSPACE_SCOPES = ("fixed_base", "bounded_posture", "bounded_whole_body")


def build_workspace_grid(
    bounds_m: Sequence[Sequence[float]],
    spacing_m: float,
    orientation_xyzw: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
) -> dict:
    """Build a cell-centred Cartesian target grid in the initial shoulder frame."""
    if len(bounds_m) != 3 or any(len(axis) != 2 for axis in bounds_m):
        raise ValueError("bounds_m must contain [min, max] for x, y, z")
    spacing = float(spacing_m)
    if spacing <= 0:
        raise ValueError("spacing_m must be positive")
    orientation = [float(value) for value in orientation_xyzw]
    if len(orientation) != 4:
        raise ValueError("orientation_xyzw must have four values")

    axes = []
    for lo_value, hi_value in bounds_m:
        lo, hi = float(lo_value), float(hi_value)
        if hi <= lo:
            raise ValueError("each workspace bound must satisfy max > min")
        count = math.floor((hi - lo) / spacing + 1e-12)
        if count <= 0:
            raise ValueError("each workspace axis must contain at least one cell")
        axes.append([lo + (index + 0.5) * spacing for index in range(count)])

    voxel_volume = spacing ** 3
    targets = []
    for index, position in enumerate(product(*axes)):
        payload = {
            "schema_version": WORKSPACE_SCHEMA_VERSION,
            "frame": "initial_shoulder",
            "target_position_m": [float(value) for value in position],
            "target_orientation_xyzw": orientation,
            "voxel_volume_m3": voxel_volume,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        payload["workspace_target_id"] = f"workspace-{digest[:16]}"
        payload["workspace_index"] = index
        targets.append(payload)
    manifest = {
        "schema_version": WORKSPACE_SCHEMA_VERSION,
        "frame": "initial_shoulder",
        "bounds_m": [[float(value) for value in axis] for axis in bounds_m],
        "spacing_m": spacing,
        "voxel_volume_m3": voxel_volume,
        "targets": targets,
    }
    manifest["workspace_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return manifest


def summarize_workspace_results(
    rows: Iterable[Mapping[str, object]], scope: str
) -> dict:
    """Score one bounded workspace probe without dropping failed targets."""
    if scope not in WORKSPACE_SCOPES:
        raise ValueError(f"unsupported workspace scope: {scope}")
    items = list(rows)
    ids = [str(row.get("workspace_target_id")) for row in items]
    if any(item == "None" for item in ids) or len(set(ids)) != len(ids):
        raise ValueError("workspace rows require unique workspace_target_id values")

    completed = [bool(row.get("completed")) for row in items]
    voxel_volumes = [float(row["voxel_volume_m3"]) for row in items]
    if any(volume <= 0 for volume in voxel_volumes):
        raise ValueError("workspace voxel volumes must be positive")
    attempted_volume = sum(voxel_volumes)
    reached_volume = sum(
        volume for volume, success in zip(voxel_volumes, completed) if success
    )
    return {
        "schema_version": WORKSPACE_SCHEMA_VERSION,
        "scope": scope,
        "frame": "initial_shoulder",
        "n_targets": len(items),
        "completion_count": sum(completed),
        "completion_rate": sum(completed) / len(items) if items else None,
        "attempted_volume_m3": attempted_volume if items else None,
        "reachable_volume_m3": reached_volume if items else None,
        "self_collision_evaluable": all(
            row.get("self_collision_status") == "evaluated" for row in items
        )
        if items
        else None,
        "target_results": [dict(row) for row in items],
    }


def run_isaacgym_workspace_probe(
    env,
    handle,
    manifest: Mapping[str, object],
    scopes: Sequence[str] = WORKSPACE_SCOPES,
    n_steps: int = 100,
    settle_steps: int = 10,
    position_threshold_m: float = 0.03,
    rotation_threshold_rad: float = math.radians(5.0),
    hold_steps: int = 5,
) -> dict:
    """Execute workspace-probe-v1 through the normal controller/dog step.

    The runner is intentionally sequential: one simulator environment owns one
    static target at a time. This keeps native OCS2's single-instance contract
    explicit and, critically, retains every attempted target in the output.
    """
    import torch
    from isaacgym.torch_utils import quat_apply, quat_mul

    from benchmark.wbc.evaluation import _wbc_step_all
    from benchmark.wbc.scenarios import (
        _neutral_settle_step,
        _set_benchmark_nominal_initial_states,
        _synchronize_paired_initial_states,
    )
    from go1_gym.utils.global_switch import global_switch

    if handle.n_envs != 1 or env.env.num_envs != 1:
        raise ValueError("workspace runner currently requires one sequential IsaacGym env")
    if n_steps <= 0 or settle_steps < 0 or hold_steps <= 0:
        raise ValueError("workspace step counts must be positive")
    selected_scopes = list(scopes)
    if not selected_scopes or any(scope not in WORKSPACE_SCOPES for scope in selected_scopes):
        raise ValueError(f"workspace scopes must be selected from {WORKSPACE_SCOPES}")
    if manifest.get("schema_version") != WORKSPACE_SCHEMA_VERSION:
        raise ValueError("workspace manifest schema mismatch")

    base = env.env
    env_id = torch.tensor([0], device=base.device, dtype=torch.long)
    outputs = {}
    try:
        for scope in selected_scopes:
            rows = []
            handle.command_scope = scope
            for target in manifest["targets"]:
                base.clear_workspace_probe()
                global_switch.switch_flag = False
                env.reset()
                _set_benchmark_nominal_initial_states(env, [handle], 1)
                for _ in range(settle_steps):
                    _neutral_settle_step(env)
                _synchronize_paired_initial_states(base, [handle], 1)
                _neutral_settle_step(env)
                _synchronize_paired_initial_states(base, [handle], 1)

                shoulder_pos, shoulder_quat = base._shoulder_frame()
                local_position = torch.as_tensor(
                    target["target_position_m"],
                    device=base.device,
                    dtype=base.root_states.dtype,
                ).view(1, 3)
                local_quat = torch.as_tensor(
                    target["target_orientation_xyzw"],
                    device=base.device,
                    dtype=base.root_states.dtype,
                ).view(1, 4)
                target_position = shoulder_pos[env_id] + quat_apply(
                    shoulder_quat[env_id], local_position
                )
                target_quaternion = quat_mul(shoulder_quat[env_id], local_quat)
                base.set_workspace_probe_goal(env_id, target_position, target_quaternion)
                base.set_workspace_fixed_base(env_id, scope == "fixed_base")
                base.episode_length_buf[env_id] = 0
                base.reset_buf[env_id] = False
                base.time_out_buf[env_id] = False
                base.traj_early_term[env_id] = False
                global_switch.open_switch()
                handle.upper_controller.reset(base, 0, 1, [target])
                env.clear_cached(env_id)

                initial_root = base.root_states[0].clone()
                best_position_error = math.inf
                best_rotation_error = math.inf
                max_base_xy = 0.0
                stable = 0
                completed = False
                termination_reason = "timeout"
                samples = 0
                for _step in range(n_steps):
                    done, timed_out, early = _wbc_step_all(env, [handle])
                    pos_error = float(torch.linalg.vector_norm(base.ee_pos_err[0]).item())
                    quat_dot = torch.abs(
                        torch.dot(
                            base.end_effector_state[0, 3:7], target_quaternion[0]
                        )
                    ).clamp(0.0, 1.0)
                    rot_error = float((2.0 * torch.acos(quat_dot)).item())
                    best_position_error = min(best_position_error, pos_error)
                    best_rotation_error = min(best_rotation_error, rot_error)
                    base_delta = base.root_states[0, :2] - initial_root[:2]
                    max_base_xy = max(
                        max_base_xy, float(torch.linalg.vector_norm(base_delta).item())
                    )
                    samples += 1
                    if pos_error <= position_threshold_m and rot_error <= rotation_threshold_rad:
                        stable += 1
                        if stable >= hold_steps:
                            completed = True
                            termination_reason = "success"
                            break
                    else:
                        stable = 0
                    if bool(done[0].item()):
                        if bool(early[0].item()):
                            termination_reason = "trajectory_cutoff"
                        elif bool(timed_out[0].item()):
                            termination_reason = "simulator_timeout"
                        else:
                            termination_reason = "fall_or_reset"
                        break

                rows.append(
                    {
                        **dict(target),
                        "scope": scope,
                        "upper_controller": handle.controller_id,
                        "locomotion_policy": handle.locomotion_policy_id,
                        "completed": completed,
                        "termination_reason": termination_reason,
                        "n_control_steps": samples,
                        "best_position_error_m": best_position_error,
                        "best_rotation_error_rad": best_rotation_error,
                        "max_base_xy_displacement_m": max_base_xy,
                        "self_collision_status": "disabled_by_asset_filter",
                    }
                )
            outputs[scope] = summarize_workspace_results(rows, scope)
    finally:
        handle.command_scope = "bounded_whole_body"
        base.clear_workspace_probe()
    return {
        "schema_version": WORKSPACE_SCHEMA_VERSION,
        "workspace_sha256": manifest["workspace_sha256"],
        "upper_controller": handle.controller_id,
        "locomotion_policy": handle.locomotion_policy_id,
        "scope_semantics": {
            "fixed_base": "six-DoF root state locked at the post-settle initial state",
            "bounded_posture": "planar and yaw command locked to zero; bounded height, pitch and roll enabled",
            "bounded_whole_body": "bounded planar, yaw and posture commands enabled",
        },
        "scopes": outputs,
    }
