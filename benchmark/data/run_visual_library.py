"""Play Visual WholeBody on the existing frozen library TaskSpecs."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
STACK = Path("/home/simon/Projects/Simon/wbc_rl_mpc")
VISUAL = STACK / "baselines/visual_wholebody"
CHECKPOINT = VISUAL / "low-level/logs/go2x5-visual-low/go2x5_low_v6_velocity_curriculum_tb_resume1000/model_24000.pt"
MA2022_DEPLOYMENT = STACK / "rl_sar/deploy/ma2022"
MA2022_BUNDLE = STACK / "rl_sar/policy/go2_x5/ma2022_student"
UMI = STACK / "baselines/umi-on-legs/mani-centric-wbc"
UMI_CHECKPOINT = UMI / "checkpoints/tossing/ours-real/model.pt"
sys.path.insert(0, str(REPO))

from benchmark.wbc.mujoco import FrozenReference, _sha256
from benchmark.wbc.cross_method_cli import RAW_BUNDLE_ROOT


def select_tasks(tasks, cells=None, names=None):
    if not cells and not names:
        return tasks
    selected = []
    for cell in cells or []:
        matches = [t for t in tasks if [t.get("cell_A"), t.get("cell_B")] == list(cell)]
        if not matches:
            raise ValueError(f"Unknown curriculum cell {cell}")
        selected.extend(matches)
    for name in names or []:
        matches = [t for t in tasks if name in (t["task_id"], t["source_trajectory_id"])]
        if not matches:
            raise ValueError(f"Unknown trajectory {name}")
        selected.extend(matches)
    return list({t["task_id"]: t for t in selected}.values())


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def stop_process(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def main(argv=None, *, method="visual"):
    if method not in ("visual", "roboduet_raw", "ma2022", "umi"):
        raise ValueError(f"Unsupported library playback method: {method}")
    parser = argparse.ArgumentParser(description=f"Play {method} on the existing frozen library TaskSpecs.")
    parser.add_argument("--library", type=Path, default=REPO / "benchmark/data/frozen_trajectory_library2")
    parser.add_argument("--output", type=Path)
    if method == "visual":
        parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
        parser.add_argument("--action-delay", type=int, choices=(0, 1), default=0,
                            help="Default 0 matches native playback; 1 is the delayed-control diagnostic.")
    elif method == "roboduet_raw":
        parser.add_argument("--run-root", type=Path, default=RAW_BUNDLE_ROOT.parent)
        parser.add_argument("--target-mode", choices=("bounded", "native"), default="bounded")
    elif method == "umi":
        parser.add_argument("--checkpoint", type=Path, default=UMI_CHECKPOINT)
        parser.add_argument("--leg-action-limit", type=float, default=.5)
        parser.add_argument("--arm-action-limit", type=float, default=4.)
        parser.add_argument("--tool-frame", choices=("native_x5", "arx5_home"), default="arx5_home")
        parser.add_argument("--mujoco-profile", choices=("common", "training_nominal"), default="training_nominal")
    parser.add_argument("--cell", type=int, nargs=2, action="append", metavar=("A", "B"))
    parser.add_argument("--trajectory", action="append", help="Source name or TaskSpec ID; repeat to select several.")
    parser.add_argument("--list", action="store_true", help="List all library trajectories without simulating.")
    parser.add_argument("--viewer", action="store_true", help="Real-time viewer; close it to advance to the next task.")
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--playback-speed", type=float, default=1.0,
                        help="Scale reference time; values below 1 are diagnostic slow playback.")
    parser.add_argument("--wall-timeout-s", type=float, default=600.)
    if method in ("visual", "roboduet_raw"):
        parser.add_argument("--base-mode", choices=("follow", "stand"), default="follow")
    args = parser.parse_args(argv)
    if args.max_tasks < 0 or args.max_steps < 0 or args.wall_timeout_s <= 0 or not 0 < args.playback_speed <= 1:
        parser.error("Step/task limits must be nonnegative and timeout positive")
    if method == "umi" and (args.leg_action_limit < 0 or args.arm_action_limit < 0):
        parser.error("UMI transfer action limits must be nonnegative")
    library = args.library.resolve()
    suite_path = library / "suite/trajectory_suite.json"
    suite = json.loads(suite_path.read_text())
    if args.list:
        for task in suite["trajectories"]:
            print(f"{task['source_trajectory_id']:24s} {task['task_id']}  {task['duration_s']:.1f} s")
        return 0
    source = suite["source_library"]
    for file, field in (("manifest.json", "manifest_sha256"), ("trajectories.npz", "archive_sha256")):
        if _sha256(library / file) != source[field]:
            raise ValueError(f"Library changed since TaskSpec materialization: {file}")
    tasks = select_tasks(suite["trajectories"], args.cell, args.trajectory)
    if args.max_tasks:
        tasks = tasks[:args.max_tasks]
    # Validate the complete selected inputs before launching any controller.
    for task in tasks:
        FrozenReference(suite_path, task["task_id"])
    if method == "visual":
        checkpoint = args.checkpoint.resolve()
        required = [checkpoint, checkpoint.with_name("run_config.json")]
        policy_arguments = ["--rl-sar-root", str(STACK / "rl_sar"), "--policy-key", "unused",
            "--policy-adapter", "visual", "--dwbc-root", str(VISUAL),
            "--dwbc-checkpoint", str(checkpoint),
            "--visual-base-mode", args.base_mode, "--visual-action-delay", str(args.action_delay)]
        policy_settings = dict(action_delay=args.action_delay, base_mode=args.base_mode)
    elif method == "roboduet_raw":
        run_root = args.run_root.resolve()
        bundle = run_root / "rl_sar"
        checkpoint = bundle / "policy/go2_x5/roboduet_go2_x5/policy.pt"
        required = [checkpoint, checkpoint.with_name("config.yaml"),
                    bundle / "policy/go2_x5/base.yaml", run_root / "parameters.pkl"]
        required.extend(run_root / "deploy_model" / name for name in
                        ("adaptation_module_latest_arm.jit", "history_latest_arm.jit", "body_latest_arm.jit"))
        policy_arguments = ["--rl-sar-root", str(bundle), "--policy-key", "roboduet_go2_x5",
            "--policy-adapter", "roboduet_raw", "--raw-run-root", str(run_root),
            "--raw-base-mode", args.base_mode, "--raw-target-mode", args.target_mode]
        policy_settings = dict(run_root=str(run_root), target_mode=args.target_mode,
                               base_mode=args.base_mode)
    elif method == "ma2022":
        checkpoint = MA2022_BUNDLE / "student_policy.pt"
        env_config = MA2022_BUNDLE / "env_cfg.json"
        deployment_config = MA2022_DEPLOYMENT / "config.yaml"
        required = [checkpoint, env_config, deployment_config,
                    MA2022_DEPLOYMENT / "adapter.py"]
        policy_arguments = [
            "--rl-sar-root", str(STACK / "rl_sar"),
            "--policy-key", "ma2022_student",
            "--policy-adapter", "ma2022",
            "--ma2022-deployment-root", str(MA2022_DEPLOYMENT),
            "--ma2022-policy", str(checkpoint),
            "--ma2022-env-config", str(env_config),
            "--ma2022-config", str(deployment_config),
            "--upper-controller", "floating_base_ocs2_mpc",
            "--ocs2-root", str(STACK),
            "--ocs2-transport", "synchronous",
            "--ocs2-task-profile", "native_ideal",
            "--ocs2-command-mode", "full",
        ]
        policy_settings = dict(
            deployment_root=str(MA2022_DEPLOYMENT),
            env_config=str(env_config),
            deployment_config=str(deployment_config),
            upper_controller="floating_base_ocs2_mpc",
            ocs2_transport="synchronous",
            ocs2_task_profile="native_ideal",
            ocs2_command_mode="full",
        )
    else:
        checkpoint = args.checkpoint.resolve()
        required = [checkpoint, checkpoint.with_name("config.pkl")]
        policy_arguments = [
            "--rl-sar-root", str(STACK / "rl_sar"),
            "--policy-key", "unused",
            "--policy-adapter", "umi",
            "--umi-checkpoint", str(checkpoint),
            "--umi-leg-action-limit", str(args.leg_action_limit),
            "--umi-arm-action-limit", str(args.arm_action_limit),
            "--umi-tool-frame", args.tool_frame,
            "--umi-mujoco-profile", args.mujoco_profile,
        ]
        policy_settings = dict(
            leg_action_limit=args.leg_action_limit,
            arm_action_limit=args.arm_action_limit,
            tool_frame=args.tool_frame,
            mujoco_profile=args.mujoco_profile,
            common_plant_transfer=True,
        )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {method} playback inputs: {missing}")
    output = (args.output or REPO / f"benchmark/results/{method}_library" /
              dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "suite"
    snapshot.mkdir()
    shutil.copy2(suite_path, snapshot / suite_path.name)
    archive = Path(suite["reference_archive"]["path"])
    if archive.is_absolute() or ".." in archive.parts:
        raise ValueError("Expected a relative reference archive within the suite directory")
    (snapshot / archive).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(suite_path.parent / archive, snapshot / archive)
    state = dict(schema_version=f"{method}-library-playback-v1", status="running", method=method,
                 suite_sha256=suite["suite_sha256"], library=str(library),
                 checkpoint=str(checkpoint), checkpoint_sha256=_sha256(checkpoint),
                 **policy_settings,
                 playback_speed=args.playback_speed,
                 max_steps=args.max_steps, viewer=args.viewer,
                 requested_tasks=len(tasks), tasks={})
    state_path = output / "batch_state.json"
    write_json(state_path, state)
    print(f"Output: {output}", flush=True)
    failed = False
    for index, task in enumerate(tasks, 1):
        name = task["source_trajectory_id"]
        task_dir = output / "runs" / name
        task_dir.mkdir(parents=True)
        command = [sys.executable, "-m", "benchmark.wbc.mujoco",
            "--suite", str(snapshot / suite_path.name), "--task-id", task["task_id"],
            *policy_arguments,
            "--scene", str(STACK / "rl_sar/src/rl_sar_zoo/go2_x5_description/mjcf/scene.xml"),
            "--reference-speed-scale", str(args.playback_speed),
            "--max-steps", str(args.max_steps), "--output", str(task_dir)]
        if args.viewer:
            command.extend(["--viewer", "--realtime"])
        row = dict(task_id=task["task_id"], status="running", command=command)
        state["tasks"][name] = row
        write_json(state_path, state)
        print(f"[{index}/{len(tasks)}] {name}", flush=True)
        with (task_dir / "run.log").open("w") as log:
            process = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                code = process.wait(timeout=args.wall_timeout_s)
            except subprocess.TimeoutExpired:
                stop_process(process)
                code = process.returncode
                row["error"] = "wall_clock_timeout"
            except KeyboardInterrupt:
                stop_process(process)
                row.update(status="interrupted", exit_code=process.returncode)
                state["status"] = "interrupted"
                write_json(state_path, state)
                return 130
        row.update(exit_code=code, status="failed")
        receipt_path = task_dir / "receipt.json"
        if code == 0 and receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text())
            row.update(status=receipt["status"], recorded_steps=receipt["recorded_steps"],
                       result=receipt["result"], receipt=str(receipt_path))
        failed |= row["status"] != "complete"
        print(f"  {row['status']}; steps={row.get('recorded_steps')}; "
              f"fall={row.get('result', {}).get('fall')}", flush=True)
        write_json(state_path, state)
    state["status"] = "failed" if failed else "complete"
    state["tracking_success_is_required"] = False
    write_json(state_path, state)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
