"""Run a frozen trajectory library with I_Q and native_ideal MPC.

Typical use from the RoboDuet repository root:

    /opt/miniconda3/envs/isaacgym/bin/python \
      benchmark/data/run_iq_native_ideal_library.py \
      --output benchmark/results/iq_native_ideal_all

The command materializes the library once, runs every task sequentially, and
writes ``batch_state.json`` after each task. Re-run with ``--resume`` and the
same output directory to skip tasks that already have a receipt.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmark.data.materialize_trajectory_library import (
    _sha256,
    freeze_policy_state,
    materialize,
)


DEFAULT_STACK = Path("/home/simon/Projects/Simon/wbc_rl_mpc")
DEFAULT_SCENE = DEFAULT_STACK / "rl_sar/src/rl_sar_zoo/go2_x5_description/mjcf/scene.xml"


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _make_suite(args, output: Path) -> Path:
    suite_dir = output / "suite"
    suite_path = suite_dir / "trajectory_suite.json"
    if suite_path.is_file():
        existing = json.loads(suite_path.read_text())
        if existing.get("initialization", {}).get("z_anchor_policy") != (
            "preserve_library_ground_relative_height"
        ):
            raise ValueError(
                f"existing suite uses the obsolete XYZ start-anchor rule: {suite_path}; "
                "choose a new --output directory"
            )
        return suite_path
    if suite_dir.exists():
        raise FileExistsError(
            f"incomplete suite directory already exists: {suite_dir}; use a new output"
        )
    policy_root = args.stack_root / "rl_sar/policy/go2_x5"
    initial_state, ee_pose = freeze_policy_state(
        args.stack_root, args.policy_key, args.scene, args.settle_s
    )
    initialization = {
        "source": "once-frozen rl_sar getup and zero-command settle",
        "policy_key": args.policy_key,
        "settle_s": args.settle_s,
        "policy_sha256": _sha256(policy_root / args.policy_key / "policy.pt"),
        "policy_config_sha256": _sha256(policy_root / args.policy_key / "config.yaml"),
        "base_config_sha256": _sha256(policy_root / "base.yaml"),
        "scene": str(args.scene),
        "scene_sha256": _sha256(args.scene),
        "canonical_ee_pose_xyz_xyzw": ee_pose.tolist(),
        "xy_anchor_policy": "align_each_trajectory_start_to_canonical_ee_xy",
        "z_anchor_policy": "preserve_library_ground_relative_height",
        "policy_history_at_replay": "reset; physical state only is frozen",
    }
    return materialize(
        args.library, suite_dir, initial_state=initial_state,
        canonical_ee_pose=ee_pose,
        completion_timeout_s=args.completion_timeout_s,
        initialization=initialization,
    )


def _receipt_summary(receipt_path: Path) -> dict:
    receipt = json.loads(receipt_path.read_text())
    result = receipt.get("result") or {}
    return {
        "status": receipt.get("status"),
        "success": result.get("success"),
        "end_reason": result.get("end_reason"),
        "fall": result.get("fall"),
        "numerical_fault": result.get("numerical_fault"),
        "recorded_steps": receipt.get("recorded_steps"),
        "ee_pos_rmse_m": result.get("ee_pos_rmse_m"),
        "ee_rot_rmse_rad": result.get("ee_rot_rmse_rad"),
    }


def _select_tasks(tasks: list[dict], cells: list[list[int]] | None) -> list[dict]:
    """Select curriculum cells in the user's requested order."""
    if not cells:
        return tasks
    by_cell = {
        (int(task["cell_A"]), int(task["cell_B"])): task
        for task in tasks
        if "cell_A" in task and "cell_B" in task
    }
    selected = []
    seen = set()
    for values in cells:
        cell = (int(values[0]), int(values[1]))
        if cell in seen:
            continue
        if cell not in by_cell:
            available = ", ".join(f"({a},{b})" for a, b in sorted(by_cell))
            raise ValueError(f"trajectory cell {cell} does not exist; available cells: {available}")
        selected.append(by_cell[cell])
        seen.add(cell)
    return selected


def run(args) -> int:
    output = args.output.resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(
            f"output already exists: {output}; use --resume or choose a new directory"
        )
    output.mkdir(parents=True, exist_ok=True)
    suite_path = _make_suite(args, output)
    suite = json.loads(suite_path.read_text())
    tasks = _select_tasks(suite["trajectories"], args.cell)
    if args.max_tasks:
        tasks = tasks[:args.max_tasks]

    state_path = output / "batch_state.json"
    state = {
        "schema_version": "iq-native-ideal-library-batch-v1",
        "status": "prepared" if args.prepare_only else "running",
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "library": str(args.library),
        "suite": str(suite_path),
        "suite_sha256": suite["suite_sha256"],
        "policy_key": args.policy_key,
        "controller": f"native_ideal/full/{args.ocs2_transport}",
        "mpc_info": str(args.mpc_info) if args.mpc_info else None,
        "reference_window_s": args.reference_window_s,
        "reference_window_dt_s": args.reference_window_dt_s,
        "reference_window_knots": int(round(
            args.reference_window_s / args.reference_window_dt_s
        )) + 1,
        "requested_tasks": len(tasks),
        "requested_cells": args.cell,
        "tasks": {},
    }
    if state_path.is_file():
        previous = json.loads(state_path.read_text())
        state["started_at"] = previous.get("started_at", state["started_at"])
        state["tasks"] = previous.get("tasks", {})
    _write_json(state_path, state)
    if args.prepare_only:
        print(json.dumps({"status": "prepared", "suite": str(suite_path),
                          "task_count": len(tasks), "output": str(output)}, indent=2))
        return 0

    runs_dir = output / "runs"
    runs_dir.mkdir(exist_ok=True)
    operational_failures = 0
    for index, task in enumerate(tasks, start=1):
        task_id = task["task_id"]
        task_dir = runs_dir / task_id
        receipt_path = task_dir / "receipt.json"
        if args.resume and receipt_path.is_file():
            state["tasks"][task_id] = {"index": index, "state": "existing",
                                             **_receipt_summary(receipt_path)}
            print(f"[{index}/{len(tasks)}] skip completed {task_id}", flush=True)
            _write_json(state_path, state)
            continue
        task_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, "-m", "benchmark.wbc.mujoco",
            "--suite", str(suite_path), "--task-id", task_id,
            "--rl-sar-root", str(args.stack_root / "rl_sar"),
            "--policy-key", args.policy_key,
            "--scene", str(args.scene),
            "--upper-controller", "floating_base_ocs2_mpc",
            "--ocs2-root", str(args.stack_root),
            "--ocs2-task-profile", "native_ideal",
            "--ocs2-command-mode", "full",
            "--ocs2-transport", args.ocs2_transport,
            "--ocs2-reference-window-s", str(args.reference_window_s),
            "--ocs2-reference-window-dt-s", str(args.reference_window_dt_s),
            "--output", str(task_dir),
        ]
        if args.max_steps:
            command.extend(["--max-steps", str(args.max_steps)])
        if args.mpc_info:
            command.extend(["--ocs2-task-file", str(args.mpc_info)])
        if args.viewer:
            command.extend(["--viewer", "--realtime"])
        print(f"[{index}/{len(tasks)}] run {task['source_trajectory_id']} -> {task_id}", flush=True)
        state["tasks"][task_id] = {
            "index": index, "source_trajectory_id": task["source_trajectory_id"],
            "state": "running", "command": command,
        }
        _write_json(state_path, state)
        with (task_dir / "run.log").open("w") as log:
            process = subprocess.Popen(
                command, cwd=Path(__file__).resolve().parents[2],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=args.wall_timeout_s)
                timeout_error = None
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
                returncode = process.returncode
                timeout_error = f"wall clock timeout after {args.wall_timeout_s:g} s"
        entry = state["tasks"][task_id]
        entry.update(returncode=returncode)
        if receipt_path.is_file():
            entry.update(state="finished", **_receipt_summary(receipt_path))
        else:
            entry.update(state="operational_failure", error="missing receipt")
        if timeout_error or returncode != 0 or not receipt_path.is_file():
            entry["state"] = "operational_failure"
            entry["error"] = timeout_error or f"runner exited {returncode}"
            operational_failures += 1
        _write_json(state_path, state)

    state["status"] = "complete" if operational_failures == 0 else "complete_with_operational_failures"
    state["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    state["operational_failures"] = operational_failures
    _write_json(state_path, state)
    print(json.dumps({"status": state["status"], "output": str(output),
                      "task_count": len(tasks), "operational_failures": operational_failures}, indent=2))
    return 0 if operational_failures == 0 else 1


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library", type=Path,
                        default=Path("benchmark/data/frozen_trajectory_library"))
    parser.add_argument("--stack-root", type=Path, default=DEFAULT_STACK)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--policy-key", default="I_Q")
    parser.add_argument("--settle-s", type=float, default=4.0)
    parser.add_argument("--completion-timeout-s", type=float, default=1.0)
    parser.add_argument("--wall-timeout-s", type=float, default=600.0,
                        help="Maximum wall time for each trajectory.")
    parser.add_argument("--reference-window-s", type=float, default=1.0,
                        help="Forward MPC reference horizon in seconds.")
    parser.add_argument("--reference-window-dt-s", type=float, default=0.02,
                        help="Sampling interval inside the MPC reference window.")
    parser.add_argument("--ocs2-transport", choices=("synchronous", "async"),
                        default="synchronous")
    parser.add_argument("--mpc-info", type=Path,
                        help="OCS2 .info task file; defaults to go2_x5_ocs2/config/task_floating.info.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--viewer", action="store_true",
                        help="Open a real-time MuJoCo window for each trajectory.")
    parser.add_argument("--max-tasks", type=int, default=0,
                        help="Run only the first N tasks; zero means all tasks.")
    parser.add_argument(
        "--cell", type=int, nargs=2, action="append", metavar=("A", "B"),
        help="Run one curriculum cell; repeat this option to select multiple cells.",
    )
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Diagnostic step limit passed to each task; zero means full duration.")
    args = parser.parse_args(argv)
    args.library = args.library.resolve()
    args.stack_root = args.stack_root.resolve()
    args.scene = args.scene.resolve()
    if args.mpc_info:
        args.mpc_info = args.mpc_info.resolve()
        if not args.mpc_info.is_file():
            parser.error(f"MPC info file does not exist: {args.mpc_info}")
    if min(args.settle_s, args.wall_timeout_s, args.reference_window_s,
           args.reference_window_dt_s) <= 0 or args.completion_timeout_s < 0:
        parser.error("settle/wall timeout must be positive; completion timeout must be non-negative")
    if args.max_tasks < 0 or args.max_steps < 0:
        parser.error("max-tasks and max-steps must be non-negative")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
