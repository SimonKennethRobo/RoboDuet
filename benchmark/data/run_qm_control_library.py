"""Run native qm_control on a selected task from the frozen trajectory library."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess


REPO = Path(__file__).resolve().parents[2]
STACK = Path("/home/simon/Projects/Simon/wbc_rl_mpc")
MPC_BASELINE = STACK / "baselines/mpc_baseline"
PYTHON = Path("/opt/miniconda3/envs/base312/bin/python")


def select_task(suite, cell, trajectory):
    tasks = suite["trajectories"]
    if trajectory:
        matches = [task for task in tasks
                   if trajectory in (task["source_trajectory_id"], task["task_id"])]
    else:
        matches = [task for task in tasks
                   if (task.get("cell_A"), task.get("cell_B")) == tuple(cell)]
    if len(matches) != 1:
        selector = trajectory or f"cell {tuple(cell)}"
        raise ValueError(f"expected exactly one frozen task for {selector}, found {len(matches)}")
    return matches[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path,
                        default=REPO / "benchmark/data/frozen_trajectory_library2")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--cell", type=int, nargs=2, default=(5, 3), metavar=("A", "B"))
    selection.add_argument("--trajectory", help="Frozen source trajectory name or TaskSpec ID")
    parser.add_argument("--scenario", action="append", choices=("nominal", "push"),
                        help="Repeat to run both; default is nominal only.")
    parser.add_argument("--output", type=Path,
                        default=REPO / "benchmark/results/qm_control_library")
    parser.add_argument("--ros-domain-id", type=int, default=93)
    parser.add_argument("--startup-timeout", type=float, default=180.)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--viewer", action="store_true",
                        help="Run at wall-clock speed with the interactive MuJoCo viewer.")
    parser.add_argument("--mpc-coupling-mode", choices=("async", "sync"), default="async",
                        help="Default async mode advances with the latest available command.")
    args = parser.parse_args(argv)

    suite_path = (args.library / "suite/trajectory_suite.json").resolve()
    suite = json.loads(suite_path.read_text())
    task = select_task(suite, args.cell, args.trajectory)
    scenarios = args.scenario or ["nominal"]
    command = [
        str(PYTHON), "-m", "benchmark.aligned_cli",
        "--roboduet-root", str(REPO),
        "--suite", str(suite_path),
        "--task-id", task["task_id"],
        "--output-dir", str(args.output.resolve()),
        "--scenarios", *scenarios,
        "--ros-domain-id", str(args.ros_domain_id),
        "--startup-timeout", str(args.startup_timeout),
        "--mpc-coupling-mode", args.mpc_coupling_mode,
    ]
    if args.prepare_only:
        command.append("--prepare-only")
    if args.viewer:
        command.append("--viewer")
    print(f"qm_control: {task['source_trajectory_id']} ({task['task_id']})", flush=True)
    print(f"duration={task['duration_s']:.6f}s deadline={task['deadline_s']:.6f}s", flush=True)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(MPC_BASELINE)
    return subprocess.run(command, cwd=MPC_BASELINE, env=environment).returncode


if __name__ == "__main__":
    raise SystemExit(main())
