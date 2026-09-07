"""CLI for paired Stage-2 trajectory-tracking benchmarks."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import isaacgym  # noqa: F401 - must precede torch
import numpy as np
import torch

from benchmark.wbc.evaluation import (
    WBCPolicyHandle,
    _load_cfg_from_pkl,
    describe_wbc_shared_env_group,
    group_wbc_shared_env_compatible_runs,
    load_wbc_env_benchmark,
    load_wbc_policies,
)
from benchmark.wbc.scenarios import run_wbc_aggregate


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
    parser.add_argument("--num_representative_videos", type=int, default=6)
    parser.add_argument(
        "--video_bank_rows",
        nargs="*",
        type=int,
        default=None,
        help="Optional explicit representative bank rows (overrides automatic selection)",
    )
    return parser.parse_args(argv)


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


def main(argv: Optional[List[str]] = None):
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
        f"held_out={args.bank_seed != 0}"
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

    for group_number, run_indices in enumerate(groups, start=1):
        set_benchmark_seed(args.seed, args.sim_device)
        group_names = [names[index] for index in run_indices]
        base_index = run_indices[0]
        preview_cfg = _load_cfg_from_pkl(args.logdirs[base_index], robot=args.robot)
        n_cells_preview = (
            1
            if args.smoke
            else int(
                preview_cfg.wbc.goal_reaching.trajectory.n_levels_A
                * preview_cfg.wbc.goal_reaching.trajectory.n_levels_B
            )
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
        control_dts.append(float(base.dt))
        n_cells = (
            1 if args.smoke else int(base.traj_curriculum.nA * base.traj_curriculum.nB)
        )
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
            run_output = run_wbc_aggregate(
                env,
                handles,
                n_steps=args.num_eval_steps,
                settle_steps=args.settle_steps,
                device=args.sim_device,
                suite_rows_per_cell=args.suite_rows_per_cell,
                cells=[(0, 0)] if args.smoke else None,
                validate_feature_coverage=not args.smoke,
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
    with open(json_path, "w") as stream:
        json.dump(all_results, stream, indent=2)
    with open(suite_path, "w") as stream:
        json.dump(suite_manifests, stream, indent=2)

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
        with open(os.path.join(run_dir, "representative_videos.json"), "w") as stream:
            json.dump(representative_video_records, stream, indent=2)

    metadata = {
        "mode": "wbc",
        "protocol": "wbc-suite-v3",
        "names": names,
        "logdirs": args.logdirs,
        "ckptids": ckptids,
        "requested_total_envs": args.total_envs,
        "legacy_num_envs_per_policy_override": args.num_envs_per_policy,
        "peak_total_envs": peak_envs,
        "control_dt_s": control_dts,
        "seed": args.seed,
        "bank_seed": args.bank_seed,
        "held_out": args.bank_seed != 0,
        "n_eval_steps": args.num_eval_steps,
        "settle_steps": args.settle_steps,
        "suite_rows_per_cell": args.suite_rows_per_cell,
        "smoke": args.smoke,
        "n_curriculum_cells": n_curriculum_cells,
        "num_layout_groups": len(groups),
        "layout_groups": group_metadata,
        "representative_videos": representative_video_records,
        "representative_video_errors": representative_video_errors,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(metadata_path, "w") as stream:
        json.dump(metadata, stream, indent=2)

    try:
        from benchmark.reports.html import write_report_bundle

        write_report_bundle(json_path)
    except Exception as exc:
        print(f"[WBC Benchmark] Skipping HTML report: {exc}")
    print(f"[WBC Benchmark] Output directory -> {run_dir}")


if __name__ == "__main__":
    main()
