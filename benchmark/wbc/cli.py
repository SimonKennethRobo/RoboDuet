"""CLI for paired Stage-2 trajectory-tracking benchmarks."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import random
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import isaacgym  # noqa: F401 - must precede torch
import numpy as np
import torch

from benchmark.metadata import git_snapshot, runtime_snapshot
from benchmark.wbc.scoring import (
    DEVELOPMENT_KINEMATIC_PROTOCOL,
    DEVELOPMENT_TIMED_TRAJECTORY_PROTOCOL,
    PROTOCOL_VERSION,
)
from benchmark.wbc.evaluation import (
    WBCPolicyHandle,
    _load_cfg_from_pkl,
    describe_wbc_shared_env_group,
    group_wbc_shared_env_compatible_runs,
    load_wbc_env_benchmark,
    load_wbc_policies,
)
from benchmark.wbc.scenarios import run_wbc_aggregate
from benchmark.wbc.suite import SUITE_VERSION, write_reference_archive
from go1_gym.envs.config.core import cfg_to_dict


def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="WBC Stage-2 trajectory benchmark")
    parser.add_argument("--logdirs", nargs="+", required=True)
    parser.add_argument("--names", nargs="*", default=None)
    parser.add_argument("--ckptids", nargs="*", default=None)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--robot", type=str, default=None)
    parser.add_argument(
        "--total_envs",
        type=int,
        default=4096,
        help="Total env budget shared by compatible methods; unused capacity is not created",
    )
    parser.add_argument(
        "--num_envs_per_policy",
        type=int,
        default=None,
        help="Deprecated compatibility override; prefer --total_envs",
    )
    parser.add_argument("--num_eval_steps", type=int, default=500)
    parser.add_argument("--settle_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="benchmark/results")
    parser.add_argument("--bank_seed", type=int, default=12345)
    parser.add_argument("--bank_per_cell", type=int, default=8)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run only curriculum cell (0, 0) for a quick end-to-end GPU check",
    )
    parser.add_argument(
        "--cells",
        nargs="+",
        metavar="A,B",
        help="Evaluate only explicit curriculum cells, for example --cells 5,0 5,5",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Validate config grouping and load both checkpoints on CPU without creating IsaacGym",
    )
    parser.add_argument(
        "--suite_rows_per_cell",
        type=int,
        default=0,
        help="Stable prefix of rows per cell; 0 evaluates the complete held-out bank",
    )
    parser.add_argument(
        "--record_representative_videos",
        action="store_true",
        help="Replay selected trajectories with reference and actual EE overlays",
    )
    parser.add_argument(
        "--record_raw_traces",
        action="store_true",
        help="Write per-wave backend-neutral NPZ traces for offline rescoring",
    )
    parser.add_argument("--num_representative_videos", type=int, default=6)
    parser.add_argument(
        "--video_bank_rows",
        nargs="*",
        type=int,
        default=None,
        help="Optional explicit representative bank rows (overrides automatic selection)",
    )
    return parser.parse_args(argv)


def _selected_cells(args, n_levels_a: int, n_levels_b: int):
    if args.smoke and args.cells:
        raise ValueError("--smoke and --cells cannot be used together")
    if args.smoke:
        return [(0, 0)]
    if not args.cells:
        return None
    cells = []
    for value in args.cells:
        try:
            cell_a, cell_b = (int(part) for part in value.split(",", maxsplit=1))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid curriculum cell {value!r}; expected A,B") from exc
        if not (0 <= cell_a < n_levels_a and 0 <= cell_b < n_levels_b):
            raise ValueError(
                f"curriculum cell {(cell_a, cell_b)} is outside "
                f"0..{n_levels_a - 1} x 0..{n_levels_b - 1}"
            )
        if (cell_a, cell_b) not in cells:
            cells.append((cell_a, cell_b))
    return cells


def _validate_wbc_logdir(logdir: str, ckpt_id: str):
    path = Path(logdir)
    missing = []
    for rel in ("parameters.pkl", "checkpoints_arm", "checkpoints_dog"):
        candidate = path / rel
        if not (
            candidate.is_dir() if rel.startswith("checkpoints") else candidate.is_file()
        ):
            missing.append(rel)
    dog_name = "last_dog" if ckpt_id == "last" else ckpt_id.zfill(6)
    arm_name = "last_arm" if ckpt_id == "last" else ckpt_id.zfill(6)
    for policy, name in (("dog", dog_name), ("arm", arm_name)):
        ckpt = path / f"checkpoints_{policy}" / f"ac_weights_{name}.pt"
        if not ckpt.is_file():
            missing.append(str(ckpt.relative_to(path)))
    if missing:
        raise ValueError(
            f"{logdir}: invalid WBC candidate; missing " + ", ".join(missing)
        )


def set_benchmark_seed(seed: int, device: str):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalized(values, n, default):
    result = list((values or [])[:n])
    while len(result) < n:
        result.append(default(len(result)))
    return result


def _dump_json(path, payload):
    """Write standards-compliant JSON; non-finite metrics are a hard fault."""
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)


def _sha256_file(path):
    candidate = Path(path)
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_provenance(name, logdir, ckpt_id):
    root = Path(logdir).resolve()
    dog_name = "last_dog" if ckpt_id == "last" else ckpt_id.zfill(6)
    arm_name = "last_arm" if ckpt_id == "last" else ckpt_id.zfill(6)
    resources = {
        "parameters": root / "parameters.pkl",
        "dog_checkpoint": root / "checkpoints_dog" / f"ac_weights_{dog_name}.pt",
        "arm_checkpoint": root / "checkpoints_arm" / f"ac_weights_{arm_name}.pt",
    }
    return {
        "name": name,
        "logdir": str(root),
        "ckptid": ckpt_id,
        "resources": {
            key: {"path": str(path), "sha256": _sha256_file(path)}
            for key, path in resources.items()
        },
    }


def _git_snapshot_at(path):
    root = Path(path)
    if not root.is_dir():
        return {"path": str(root), "available": False}

    def output(*args):
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            return None

    status = output("status", "--short")
    return {
        "path": str(root.resolve()),
        "available": output("rev-parse", "--is-inside-work-tree") == "true",
        "commit": output("rev-parse", "HEAD"),
        "branch": output("branch", "--show-current"),
        "dirty": bool(status),
        "status_short": status or "",
    }


def main(argv: Optional[List[str]] = None):
    if argv is not None and "--system_matrix" in argv:
        from benchmark.wbc.system_cli import main as system_main

        system_main([value for value in argv if value != "--system_matrix"])
        return
    args = parse_args(argv)
    n_runs = len(args.logdirs)
    names = _normalized(args.names, n_runs, lambda i: Path(args.logdirs[i]).name[:24])
    raw_ckpts = _normalized(args.ckptids, n_runs, lambda _i: "last")
    ckptids = ["last" if value == "last" else value.zfill(6) for value in raw_ckpts]
    for logdir, ckpt_id in zip(args.logdirs, ckptids):
        _validate_wbc_logdir(logdir, ckpt_id)

    groups = group_wbc_shared_env_compatible_runs(args.logdirs)
    if args.total_envs <= 0:
        raise ValueError("--total_envs must be positive")
    if args.num_envs_per_policy is not None and args.num_envs_per_policy <= 0:
        raise ValueError("--num_envs_per_policy must be positive")
    if args.bank_per_cell <= 0:
        raise ValueError("--bank_per_cell must be positive")
    if args.suite_rows_per_cell < 0:
        raise ValueError("--suite_rows_per_cell cannot be negative")
    if args.num_representative_videos < 0:
        raise ValueError("--num_representative_videos cannot be negative")
    print(
        f"[WBC Benchmark] {n_runs} policy pair(s), {len(groups)} layout group(s), "
        f"requested total env budget={args.total_envs}"
    )
    print(
        f"[WBC Benchmark] seed={args.seed} bank_seed={args.bank_seed} "
        "held_out=unverified_against_training_bank"
    )

    if args.validate_only:
        for index, (logdir, ckpt_id) in enumerate(zip(args.logdirs, ckptids)):
            cfg = _load_cfg_from_pkl(logdir, robot=args.robot)
            load_wbc_policies(logdir, ckpt_id, cfg, device="cpu")
            print(
                f"[WBC Benchmark] validated {names[index]}: "
                f"dog_obs={cfg.dog.dog_num_observations} "
                f"arm_obs={cfg.arm.arm_num_observations} "
                f"arm_actions={cfg.arm.num_actions_arm_cd}"
            )
        return

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    all_results: Dict[str, Dict[str, List[dict]]] = {name: {} for name in names}
    group_metadata = []
    suite_manifests = []
    control_dts = []
    n_curriculum_cells = []
    peak_envs = 0
    representative_video_records = []
    representative_video_errors = []
    candidate_provenance = [
        _candidate_provenance(name, logdir, ckpt_id)
        for name, logdir, ckpt_id in zip(names, args.logdirs, ckptids)
    ]
    state_path = os.path.join(run_dir, "run_state.json")
    _dump_json(
        state_path,
        {
            "status": "running",
            "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "completed_groups": 0,
            "completed_waves_in_current_group": 0,
        },
    )

    for group_number, run_indices in enumerate(groups, start=1):
        set_benchmark_seed(args.seed, args.sim_device)
        group_names = [names[index] for index in run_indices]
        base_index = run_indices[0]
        preview_cfg = _load_cfg_from_pkl(args.logdirs[base_index], robot=args.robot)
        n_levels_a = int(preview_cfg.wbc.goal_reaching.trajectory.n_levels_A)
        n_levels_b = int(preview_cfg.wbc.goal_reaching.trajectory.n_levels_B)
        selected_cells = _selected_cells(args, n_levels_a, n_levels_b)
        n_cells_preview = (
            len(selected_cells) if selected_cells is not None else n_levels_a * n_levels_b
        )
        rows_per_cell = min(
            args.suite_rows_per_cell or args.bank_per_cell, args.bank_per_cell
        )
        logical_tasks = n_cells_preview * rows_per_cell
        if args.num_envs_per_policy is not None:
            envs_per_policy = min(logical_tasks, args.num_envs_per_policy)
        else:
            envs_per_policy = min(logical_tasks, args.total_envs // len(run_indices))
        envs_per_policy = (envs_per_policy // rows_per_cell) * rows_per_cell
        if envs_per_policy < rows_per_cell:
            raise ValueError(
                f"Total env budget {args.total_envs} is too small for {len(run_indices)} "
                f"compatible methods and {rows_per_cell} trajectories per cell"
            )
        total_envs = envs_per_policy * len(run_indices)
        peak_envs = max(peak_envs, total_envs)
        print(
            f"\n[WBC Benchmark] Layout group {group_number}/{len(groups)}: "
            f"{', '.join(group_names)} ({total_envs} envs)"
        )
        env, cfg = load_wbc_env_benchmark(
            args.logdirs[base_index],
            total_envs,
            envs_per_policy,
            headless=args.headless,
            device=args.sim_device,
            robot=args.robot,
            bank_seed=args.bank_seed,
            bank_per_cell=args.bank_per_cell,
        )
        base = env.env
        resolved_config_name = f"resolved_config_group_{group_number:02d}.json"
        resolved_config_path = os.path.join(run_dir, resolved_config_name)
        _dump_json(resolved_config_path, cfg_to_dict(cfg))
        control_dts.append(float(base.dt))
        n_cells = n_cells_preview
        n_curriculum_cells.append(n_cells)
        group_video_tasks = []
        try:
            handles = []
            for local_index, run_index in enumerate(run_indices):
                start = local_index * envs_per_policy
                end = start + envs_per_policy
                dog_policy, arm_policy = load_wbc_policies(
                    args.logdirs[run_index],
                    ckptids[run_index],
                    cfg,
                    device=args.sim_device,
                )
                handles.append(
                    WBCPolicyHandle(
                        names[run_index], dog_policy, arm_policy, start, end
                    )
                )
            def save_wave_progress(progress):
                partial = dict(all_results)
                for partial_name, scenarios in progress["results"].items():
                    partial[partial_name] = scenarios
                _dump_json(os.path.join(run_dir, "results.partial.json"), partial)
                _dump_json(
                    os.path.join(run_dir, "trajectory_suite.partial.json"),
                    suite_manifests + [progress["suite_manifest"]],
                )
                _dump_json(
                    state_path,
                    {
                        "status": "running",
                        "completed_groups": group_number - 1,
                        "current_group": group_number,
                        "completed_waves_in_current_group": progress["completed_wave_index"] + 1,
                    },
                )

            try:
                run_output = run_wbc_aggregate(
                    env,
                    handles,
                    n_steps=args.num_eval_steps,
                    settle_steps=args.settle_steps,
                    device=args.sim_device,
                    suite_rows_per_cell=args.suite_rows_per_cell,
                    cells=selected_cells,
                    validate_feature_coverage=selected_cells is None,
                    progress_callback=save_wave_progress,
                    raw_trace_dir=(
                        os.path.join(run_dir, "raw_traces")
                        if args.record_raw_traces
                        else None
                    ),
                    trace_prefix=f"group_{group_number:02d}",
                )
            except Exception as exc:
                _dump_json(
                    state_path,
                    {
                        "status": "failed",
                        "failed_group": group_number,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "failed_at": datetime.datetime.now().isoformat(timespec="seconds"),
                    },
                )
                raise
            reference_archive_name = f"task_references_group_{group_number:02d}.npz"
            write_reference_archive(
                os.path.join(run_dir, reference_archive_name),
                base,
                run_output["suite_manifest"],
            )
            for name, scenarios in run_output["results"].items():
                all_results[name] = scenarios
            suite_manifests.append(run_output["suite_manifest"])
            if args.record_representative_videos:
                from benchmark.wbc.video import select_representative_tasks

                group_video_tasks = select_representative_tasks(
                    run_output["suite_manifest"],
                    count=args.num_representative_videos,
                    bank_rows=args.video_bank_rows,
                )
            group_metadata.append(
                {
                    "group": group_number,
                    "runs": group_names,
                    "total_envs": total_envs,
                    "layout": describe_wbc_shared_env_group(args.logdirs[base_index]),
                    "suite_sha256": run_output["suite_manifest"]["suite_sha256"],
                    "schedule": run_output["schedule"],
                    "resolved_config": {
                        "path": resolved_config_name,
                        "sha256": _sha256_file(resolved_config_path),
                    },
                }
            )
        finally:
            base.close()
            del env
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if args.record_representative_videos:
            from benchmark.wbc.video import record_representative_videos

            set_benchmark_seed(args.seed, args.sim_device)
            try:
                representative_video_records.extend(
                    record_representative_videos(
                        [args.logdirs[index] for index in run_indices],
                        [names[index] for index in run_indices],
                        [ckptids[index] for index in run_indices],
                        group_video_tasks,
                        run_dir,
                        device=args.sim_device,
                        robot=args.robot,
                        bank_seed=args.bank_seed,
                        bank_per_cell=args.bank_per_cell,
                        n_steps=args.num_eval_steps,
                        settle_steps=args.settle_steps,
                    )
                )
            except Exception as exc:
                message = f"layout group {group_number}: {type(exc).__name__}: {exc}"
                representative_video_errors.append(message)
                print(
                    f"[WBC Benchmark] Representative video recording failed: {message}"
                )
    json_path = os.path.join(run_dir, "results.json")
    metadata_path = os.path.join(run_dir, "metadata.json")
    suite_path = os.path.join(run_dir, "trajectory_suite.json")
    _dump_json(json_path, all_results)
    _dump_json(suite_path, suite_manifests)

    if representative_video_records:
        trajectory_lookup = {
            (name, row["trajectory_id"]): row
            for name, scenarios in all_results.items()
            for row in scenarios.get("wbc_trajectories", [])
        }
        for record in representative_video_records:
            metrics = trajectory_lookup.get(
                (record["run_name"], record["trajectory_id"]), {}
            )
            record.update(
                {
                    key: metrics.get(key)
                    for key in ("ee_pos_rmse_m", "ee_rot_rmse_rad", "completed", "fall")
                }
            )
        _dump_json(
            os.path.join(run_dir, "representative_videos.json"),
            representative_video_records,
        )

    metadata = {
        "mode": "wbc",
        "protocol": PROTOCOL_VERSION,
        "evaluation_protocol": dict(DEVELOPMENT_TIMED_TRAJECTORY_PROTOCOL),
        "kinematic_reporting_protocol": dict(DEVELOPMENT_KINEMATIC_PROTOCOL),
        "task_suite_version": SUITE_VERSION,
        "names": names,
        "logdirs": args.logdirs,
        "ckptids": ckptids,
        "requested_total_envs": args.total_envs,
        "legacy_num_envs_per_policy_override": args.num_envs_per_policy,
        "peak_total_envs": peak_envs,
        "control_dt_s": control_dts,
        "seed": args.seed,
        "bank_seed": args.bank_seed,
        "held_out": None,
        "held_out_status": "unverified_against_training_bank",
        "n_eval_steps": args.num_eval_steps,
        "settle_steps": args.settle_steps,
        "suite_rows_per_cell": args.suite_rows_per_cell,
        "smoke": args.smoke,
        "selected_cells": args.cells,
        "n_curriculum_cells": n_curriculum_cells,
        "num_layout_groups": len(groups),
        "layout_groups": group_metadata,
        "representative_videos": representative_video_records,
        "representative_video_errors": representative_video_errors,
        "raw_traces_recorded": bool(args.record_raw_traces),
        "power_sampling": {
            "rate": "control_step",
            "leg_energy_integration": "physics_substep",
            "arm_and_whole_body_available_when": "control.control_type == P",
            "electrical_energy_model": False,
        },
        "source": {
            "roboduet": git_snapshot(),
            "rl_sar": _git_snapshot_at("/home/simon/Projects/Simon/wbc_rl_mpc/rl_sar"),
        },
        "runtime": runtime_snapshot(),
        "candidates": candidate_provenance,
        "command": shlex.join(sys.argv if argv is None else ["python", "-m", "benchmark", "--wbc", *argv]),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    _dump_json(metadata_path, metadata)
    _dump_json(
        state_path,
        {
            "status": "complete",
            "completed_groups": len(groups),
            "completed_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "results": os.path.basename(json_path),
            "suite": os.path.basename(suite_path),
            "metadata": os.path.basename(metadata_path),
        },
    )

    try:
        from benchmark.reports.html import write_report_bundle

        write_report_bundle(json_path)
    except Exception as exc:
        print(f"[WBC Benchmark] Skipping HTML report: {exc}")
    print(f"[WBC Benchmark] Output directory -> {run_dir}")


if __name__ == "__main__":
    main()
