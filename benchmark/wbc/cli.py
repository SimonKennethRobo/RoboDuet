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
    parser.add_argument("--num_envs_per_policy", type=int, default=16)
    parser.add_argument("--num_eval_steps", type=int, default=500)
    parser.add_argument("--settle_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="benchmark/results")
    parser.add_argument("--bank_seed", type=int, default=12345)
    parser.add_argument("--bank_per_cell", type=int, default=8)
    parser.add_argument(
        "--smoke", action="store_true",
        help="Run only curriculum cell (0, 0) for a quick end-to-end GPU check",
    )
    parser.add_argument(
        "--validate_only", action="store_true",
        help="Validate config grouping and load both checkpoints on CPU without creating IsaacGym",
    )
    parser.add_argument(
        "--suite_rows_per_cell", type=int, default=0,
        help="Stable prefix of rows per cell; 0 evaluates the complete held-out bank",
    )
    return parser.parse_args(argv)


def _validate_wbc_logdir(logdir: str, ckpt_id: str):
    path = Path(logdir)
    missing = []
    for rel in ("parameters.pkl", "checkpoints_arm", "checkpoints_dog"):
        candidate = path / rel
        if not (candidate.is_dir() if rel.startswith("checkpoints") else candidate.is_file()):
            missing.append(rel)
    dog_name = "last_dog" if ckpt_id == "last" else ckpt_id.zfill(6)
    arm_name = "last_arm" if ckpt_id == "last" else ckpt_id.zfill(6)
    for policy, name in (("dog", dog_name), ("arm", arm_name)):
        ckpt = path / f"checkpoints_{policy}" / f"ac_weights_{name}.pt"
        if not ckpt.is_file():
            missing.append(str(ckpt.relative_to(path)))
    if missing:
        raise ValueError(f"{logdir}: invalid WBC candidate; missing " + ", ".join(missing))


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
    peak_envs = args.num_envs_per_policy * max(len(group) for group in groups)
    print(
        f"[WBC Benchmark] {n_runs} policy pair(s), {len(groups)} layout group(s), "
        f"peak {peak_envs} envs"
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

    all_results: Dict[str, Dict[str, List[dict]]] = {name: {} for name in names}
    group_metadata = []
    suite_manifests = []
    control_dts = []
    n_curriculum_cells = []

    for group_number, run_indices in enumerate(groups, start=1):
        set_benchmark_seed(args.seed, args.sim_device)
        group_names = [names[index] for index in run_indices]
        total_envs = args.num_envs_per_policy * len(run_indices)
        base_index = run_indices[0]
        print(
            f"\n[WBC Benchmark] Layout group {group_number}/{len(groups)}: "
            f"{', '.join(group_names)} ({total_envs} envs)"
        )
        env, cfg = load_wbc_env_benchmark(
            args.logdirs[base_index], total_envs, args.num_envs_per_policy,
            headless=args.headless, device=args.sim_device, robot=args.robot,
            bank_seed=args.bank_seed, bank_per_cell=args.bank_per_cell,
        )
        base = env.env
        control_dts.append(float(base.dt))
        n_cells = 1 if args.smoke else int(base.traj_curriculum.nA * base.traj_curriculum.nB)
        n_curriculum_cells.append(n_cells)
        try:
            handles = []
            for local_index, run_index in enumerate(run_indices):
                start = local_index * args.num_envs_per_policy
                end = start + args.num_envs_per_policy
                dog_policy, arm_policy = load_wbc_policies(
                    args.logdirs[run_index], ckptids[run_index], cfg,
                    device=args.sim_device,
                )
                handles.append(WBCPolicyHandle(
                    names[run_index], dog_policy, arm_policy, start, end
                ))
            run_output = run_wbc_aggregate(
                env, handles, n_steps=args.num_eval_steps,
                settle_steps=args.settle_steps, device=args.sim_device,
                suite_rows_per_cell=args.suite_rows_per_cell,
                cells=[(0, 0)] if args.smoke else None,
                validate_feature_coverage=not args.smoke,
            )
            for name, scenarios in run_output["results"].items():
                all_results[name] = scenarios
            suite_manifests.append(run_output["suite_manifest"])
            group_metadata.append({
                "group": group_number,
                "runs": group_names,
                "total_envs": total_envs,
                "layout": describe_wbc_shared_env_group(args.logdirs[base_index]),
                "suite_sha256": run_output["suite_manifest"]["suite_sha256"],
            })
        finally:
            base.close()
            del env
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    json_path = os.path.join(run_dir, "results.json")
    metadata_path = os.path.join(run_dir, "metadata.json")
    suite_path = os.path.join(run_dir, "trajectory_suite.json")
    with open(json_path, "w") as stream:
        json.dump(all_results, stream, indent=2)
    with open(suite_path, "w") as stream:
        json.dump(suite_manifests, stream, indent=2)

    metadata = {
        "mode": "wbc",
        "protocol": "wbc-suite-v1",
        "names": names,
        "logdirs": args.logdirs,
        "ckptids": ckptids,
        "num_envs_per_policy": args.num_envs_per_policy,
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
