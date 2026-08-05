"""WBC (stage-2 trajectory tracking) benchmark CLI.

    python -m benchmark.cli --wbc --logdirs runs/my_run --headless
    python -m benchmark.cli --wbc --logdirs runs/run_A runs/run_B --names v1 v2 --headless
"""

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
    load_wbc_env_benchmark,
    load_wbc_policies,
)
from benchmark.wbc.scenarios import run_wbc_aggregate


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description="WBC (stage-2 trajectory tracking) benchmark")
    p.add_argument("--logdirs", nargs="+", required=True)
    p.add_argument("--names", nargs="*", default=None, help="Display name per logdir")
    p.add_argument("--ckptids", nargs="*", default=None, help="Checkpoint id per logdir (default: 'last')")
    p.add_argument("--headless", action="store_true", default=False)
    p.add_argument("--sim_device", type=str, default="cuda:0")
    p.add_argument("--robot", type=str, default=None)
    p.add_argument("--num_envs_per_policy", type=int, default=16,
                   help="Envs per policy")
    p.add_argument("--num_eval_steps", type=int, default=400,
                   help="Sim steps per curriculum cell")
    p.add_argument("--settle_steps", type=int, default=20,
                   help="Post-reset settle steps before measurement")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--output_dir", type=str, default="benchmark/results")
    p.add_argument("--bank_seed", type=int, default=12345,
                   help="Trajectory-bank seed (0 for training-set score)")
    p.add_argument("--bank_per_cell", type=int, default=8,
                   help="Trajectories per curriculum cell in the bank")
    return p.parse_args(argv)


def _validate_wbc_logdir(logdir: str, ckpt_id: str):
    path = Path(logdir)
    missing = []
    for rel in ("parameters.pkl", "checkpoints_arm", "checkpoints_dog"):
        candidate = path / rel
        if not (candidate.is_dir() if rel.startswith("checkpoints") else candidate.is_file()):
            missing.append(rel)
    dog_ckpt_name = "last_dog" if ckpt_id == "last" else ckpt_id.zfill(6)
    arm_ckpt_name = "last_arm" if ckpt_id == "last" else ckpt_id.zfill(6)
    for policy, ckpt_name in (("dog", dog_ckpt_name), ("arm", arm_ckpt_name)):
        ckpt_path = path / f"checkpoints_{policy}" / f"ac_weights_{ckpt_name}.pt"
        if not ckpt_path.is_file():
            missing.append(str(ckpt_path.relative_to(path)))
    if missing:
        raise ValueError(f"{logdir}: invalid WBC benchmark candidate; missing " + ", ".join(missing))


def set_benchmark_seed(seed: int, device: str):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)

    n_runs = len(args.logdirs)
    names = list((args.names or [])[:n_runs])
    while len(names) < n_runs:
        names.append(Path(args.logdirs[len(names)]).name[:24])

    ckptids_raw = list((args.ckptids or [])[:n_runs])
    while len(ckptids_raw) < n_runs:
        ckptids_raw.append("last")
    ckptids = [("last" if c == "last" else c.zfill(6)) for c in ckptids_raw]

    for logdir, ckpt_id in zip(args.logdirs, ckptids):
        _validate_wbc_logdir(logdir, ckpt_id)

    set_benchmark_seed(args.seed, args.sim_device)

    total_envs = args.num_envs_per_policy * n_runs
    print(f"[WBC Benchmark] {n_runs} policy pair(s) x {args.num_envs_per_policy} envs = "
          f"{total_envs} evaluated envs")
    print(f"[WBC Benchmark] seed = {args.seed}  bank_seed = {args.bank_seed}  "
          f"held_out = {args.bank_seed != 0}")

    all_results: Dict[str, Dict[str, List[dict]]] = {name: {} for name in names}

    print(f"\n[WBC Benchmark] Creating shared env from {args.logdirs[0]}")
    env, cfg = load_wbc_env_benchmark(
        args.logdirs[0], total_envs=total_envs,
        envs_per_policy=args.num_envs_per_policy,
        headless=args.headless, device=args.sim_device, robot=args.robot,
        bank_seed=args.bank_seed, bank_per_cell=args.bank_per_cell,
    )
    sim_dt = float(env.env.dt)

    try:
        print("[WBC Benchmark] Loading policies...")
        handles: List[WBCPolicyHandle] = []
        for i, (logdir, ckpt_id) in enumerate(zip(args.logdirs, ckptids)):
            s = i * args.num_envs_per_policy
            e = s + args.num_envs_per_policy
            print(f"  [{i + 1}/{n_runs}] {names[i]:24s}  envs [{s}:{e})  ckpt={ckpt_id}")
            dog_p, arm_p = load_wbc_policies(logdir, ckpt_id, cfg, device=args.sim_device)
            handles.append(WBCPolicyHandle(
                name=names[i], dog_policy=dog_p, arm_policy=arm_p,
                env_start=s, env_end=e,
            ))

        print(f"\n[WBC Benchmark] Aggregate (curriculum cells x {args.num_eval_steps} steps)...")
        aggregate = run_wbc_aggregate(
            env, handles,
            n_steps=args.num_eval_steps,
            settle_steps=args.settle_steps,
            device=args.sim_device,
        )
        for name, results in aggregate.items():
            all_results[name]["wbc_aggregate"] = results
    finally:
        env.env.close()
        del env
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Output ---
    from benchmark.dog_policy.evaluation import print_comparison_table

    flat_results: Dict[str, Dict[str, List]] = {}
    for name, scenarios in all_results.items():
        flat_results[name] = {}
        for scenario_key, rows in scenarios.items():
            flat_results[name][scenario_key] = rows
    print_comparison_table(flat_results)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    json_path = os.path.join(run_dir, "results.json")
    metadata_path = os.path.join(run_dir, "metadata.json")
    with open(json_path, "w") as fh:
        json.dump(all_results, fh, indent=2)

    metadata = dict(
        mode="wbc",
        protocol="wbc",
        names=names, logdirs=args.logdirs, ckptids=ckptids,
        num_envs_per_policy=args.num_envs_per_policy,
        total_envs=total_envs,
        control_dt_s=float(sim_dt),
        seed=args.seed, bank_seed=args.bank_seed,
        held_out=args.bank_seed != 0,
        n_eval_steps=args.num_eval_steps,
        settle_steps=args.settle_steps,
        n_curriculum_cells=base.traj_curriculum.nA * base.traj_curriculum.nB if 'base' in dir() else 36,
        timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
    )
    with open(metadata_path, "w") as fh:
        json.dump(metadata, fh, indent=2)

    try:
        from benchmark.reports.html import write_report_bundle
        write_report_bundle(json_path)
    except Exception as exc:
        print(f"[WBC Benchmark] Skipping HTML report: {exc}")

    print(f"[WBC Benchmark] Output directory -> {run_dir}")


if __name__ == "__main__":
    main()
