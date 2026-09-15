"""Run external methods on the frozen RoboDuet MuJoCo benchmark contract.

The benchmark contract and final scoring live in RoboDuet. Method-specific
processes remain in their original baseline trees and are pinned by content
hashes in every receipt. The first adapter reuses qm_control's native
SQP-MPC, QP-WBC, and common Go2+X5 MuJoCo plant.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from benchmark.wbc.scoring import (
    DEVELOPMENT_KINEMATIC_PROTOCOL,
    DEVELOPMENT_TIMED_TRAJECTORY_PROTOCOL,
)
from benchmark.wbc.trace import TRACE_SCHEMA_VERSION, score_trace_archive


METHODS = (
    "roboduet", "roboduet_raw", "umi", "visual_wholebody",
    "wb_locoman", "qm_control", "deep_whole_body_control", "ma2022",
)
DEFAULT_BASELINE_ROOT = Path(
    "/home/simon/Projects/Simon/wbc_rl_mpc/baselines/mpc_baseline"
)
WORKSPACE_ROOT = Path("/home/simon/Projects/Simon/wbc_rl_mpc")
DEFAULT_RL_SAR_ROOT = WORKSPACE_ROOT / "rl_sar"
DEFAULT_SCENE = DEFAULT_RL_SAR_ROOT / "src/rl_sar_zoo/go2_x5_description/mjcf/scene.xml"
RAW_BUNDLE_ROOT = Path(
    "/home/simon/Projects/WBC/RoboDuetRaw/runs/default_go2x5v3_noselfcollision/0905/"
    "default_go2x5v3_noselfcollision_151454_seed6444/rl_sar"
)
COMMON_PHYSICAL_FIELDS = (
    "ee_pos_rmse_m", "ee_rot_rmse_rad", "ee_pos_error_p95_m",
    "ee_pos_error_peak_m", "ee_rot_error_p95_rad", "ee_rot_error_peak_rad",
    "fall", "timed_out", "numerical_fault", "final_progress",
    "leg_torque_rms_nm", "leg_torque_abs_peak_nm",
    "leg_torque_saturation_fraction", "leg_abs_mechanical_energy_j",
    "foot_slip_speed_mean_contact_mps", "foot_contact_fraction",
    "support_fraction", "no_support_fraction", "smoothness",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _git_state(root: Path) -> dict:
    def capture(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True,
        )
        return completed.stdout.strip()

    return {
        "head": capture("rev-parse", "HEAD"),
        "branch": capture("branch", "--show-current"),
        "dirty": bool(capture("status", "--porcelain")),
    }


def _method_contracts() -> dict[str, dict]:
    """Return the auditable adapter inventory; paths are checked at call time."""
    return {
        "roboduet": {
            "backend": "roboduet_rl_sar_mujoco", "policy_key": "roboduet_stage1",
            "root": DEFAULT_RL_SAR_ROOT, "common_mujoco": True,
        },
        "roboduet_raw": {
            "backend": "roboduet_rl_sar_mujoco", "policy_key": "roboduet_go2_x5",
            "root": RAW_BUNDLE_ROOT, "common_mujoco": True,
        },
        "qm_control": {
            "backend": "qm_control_native", "root": DEFAULT_BASELINE_ROOT,
            "common_mujoco": True,
        },
        "wb_locoman": {
            "backend": "common_controller_sidecar",
            "policy_key": "unused",
            "root": DEFAULT_BASELINE_ROOT / "wb-locoman_baseline",
            "common_mujoco": True,
        },
        "ma2022": {
            "backend": "ma2022_recurrent_student",
            "policy_key": "ma2022_student",
            "root": DEFAULT_RL_SAR_ROOT / "deploy/ma2022", "common_mujoco": True,
        },
        "visual_wholebody": {
            "backend": "visual_checkpoint_mujoco", "policy_key": "unused",
            "root": WORKSPACE_ROOT / "baselines/visual_wholebody", "common_mujoco": True,
        },
        "umi": {
            "backend": "umi_checkpoint_mujoco", "policy_key": "unused",
            "root": WORKSPACE_ROOT / "baselines/umi-on-legs/mani-centric-wbc", "common_mujoco": True,
        },
        "deep_whole_body_control": {
            "backend": "dwbc_checkpoint_mujoco",
            "policy_key": "unused",
            "root": WORKSPACE_ROOT / "baselines/Deep-Whole-Body-Control", "common_mujoco": True,
        },
    }


def preflight(method: str | None = None) -> dict:
    contracts = _method_contracts()
    selected = METHODS if method is None else (method,)
    result = {}
    for name in selected:
        contract = copy.deepcopy(contracts[name])
        root = Path(contract.pop("root"))
        required = []
        if name in ("roboduet", "roboduet_raw"):
            policy = root / "policy/go2_x5" / contract["policy_key"]
            required = [root / "policy/go2_x5/base.yaml", policy / "config.yaml",
                        policy / "policy.pt", DEFAULT_SCENE]
        elif name == "qm_control":
            required = [root / "benchmark/aligned_cli.py",
                        root / "qm_control_baseline/install_aligned/setup.bash",
                        root / "mujoco_models/go2_x5_description/mjcf/scene.xml"]
        elif name == "wb_locoman":
            required = [root / "benchmark_sidecar.py", root / "controller.py"]
        elif name == "ma2022":
            required = [root / "adapter.py", root / "play.py",
                        root / "config.yaml",
                        DEFAULT_RL_SAR_ROOT / "policy/go2_x5/ma2022_student/student_policy.pt",
                        DEFAULT_RL_SAR_ROOT / "policy/go2_x5/ma2022_student/env_cfg.json"]
        elif name == "visual_wholebody":
            required = [root / "benchmark_adapter/run.py",
                        root / "low-level/logs/go2x5-visual-low/go2x5_low_v6_velocity_curriculum_tb_resume1000/model_24000.pt"]
        elif name == "umi":
            required = [root / "benchmark_adapter/run.py",
                        root / "checkpoints/tossing/ours-real/model.pt"]
        else:
            required = [root / "legged_gym/logs/go2_x5/1789388761_go2_x5_reward_fix/model_11000.pt"]
        missing = [str(path) for path in required if not path.is_file()]
        ready = bool(contract["common_mujoco"] and not missing)
        result[name] = {
            **contract, "root": str(root), "required": [str(path) for path in required],
            "missing": missing, "status": "ready" if ready else "blocked",
            "blocker": None if ready else contract.get("blocker", "missing required files"),
        }
    return {"schema_version": "cross-method-preflight-v1", "methods": result}


def normalize_trace_protocol(trace_path: Path) -> dict:
    """Replace only embedded protocol metadata, retaining backend samples."""
    backend_path = trace_path.with_name("trace.backend.npz")
    shutil.copy2(trace_path, backend_path)
    with np.load(backend_path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    schema = str(np.asarray(arrays.get("schema_version")).item())
    if schema != TRACE_SCHEMA_VERSION:
        raise ValueError(f"expected {TRACE_SCHEMA_VERSION}, got {schema!r}")
    arrays["protocol_json"] = np.asarray(json.dumps(
        DEVELOPMENT_TIMED_TRAJECTORY_PROTOCOL,
        sort_keys=True, separators=(",", ":"),
    ))
    arrays["kinematic_protocol_json"] = np.asarray(json.dumps(
        DEVELOPMENT_KINEMATIC_PROTOCOL,
        sort_keys=True, separators=(",", ":"),
    ))
    np.savez_compressed(trace_path, **arrays)
    return {
        "operation": "replace_embedded_protocol_metadata_only",
        "backend_trace": {"path": str(backend_path), "sha256": _sha256(backend_path)},
        "normalized_trace": {"path": str(trace_path), "sha256": _sha256(trace_path)},
        "sample_fields_changed": False,
    }


def _coverage(metrics: dict) -> dict:
    missing = [name for name in COMMON_PHYSICAL_FIELDS if metrics.get(name) is None]
    return {
        "schema_version": "cross-method-metric-coverage-v1",
        "computed": [name for name in COMMON_PHYSICAL_FIELDS if metrics.get(name) is not None],
        "not_applicable": ["policy_action", "joint_position_target_rad"],
        "unavailable": [
            "rho", "base_feedforward", "manipulability", "jacobian",
            "joint_limit_margin", "ik_solver_diagnostics", "reach_model",
            "self_collision",
        ],
        "missing_common_physical_fields": missing,
        "common_physical_metrics_complete": not missing,
    }


def _normalize_results(output: Path, roboduet_root: Path) -> list[dict]:
    results = []
    for scenario in ("nominal", "push"):
        scenario_dir = output / scenario
        if not scenario_dir.is_dir():
            continue
        trace_path = scenario_dir / "trace.npz"
        receipt_path = scenario_dir / "receipt.json"
        if not trace_path.is_file() or not receipt_path.is_file():
            raise RuntimeError(f"incomplete backend output for {scenario}: {scenario_dir}")
        normalization = normalize_trace_protocol(trace_path)
        metrics = score_trace_archive(trace_path)[0]
        coverage = _coverage(metrics)
        _write_json(scenario_dir / "metrics.json", {str(trace_path): [metrics]})
        _write_json(scenario_dir / "metric_coverage.json", coverage)
        receipt = json.loads(receipt_path.read_text())
        receipt.update(
            benchmark_owner="RoboDuet/benchmark/wbc",
            trace={"path": str(trace_path), "sha256": _sha256(trace_path),
                   "schema_version": TRACE_SCHEMA_VERSION},
            protocol_normalization=normalization,
            scorer={
                "implementation": "benchmark.wbc.trace.score_trace_archive",
                "trace_py_sha256": _sha256(roboduet_root / "benchmark/wbc/trace.py"),
                "scoring_py_sha256": _sha256(roboduet_root / "benchmark/wbc/scoring.py"),
            },
            metric_coverage=coverage,
            metrics=metrics,
        )
        _write_json(receipt_path, receipt)
        results.append(receipt)
    if not results:
        raise RuntimeError(f"backend produced no scenario results: {output}")
    _write_json(output / "results.json", results)
    return results


def _render_results(output: Path, results: list[dict], args, scene: Path) -> list[dict]:
    if not args.record_video:
        return results
    os.environ.setdefault("MUJOCO_GL", "egl")
    from benchmark.wbc.mujoco_video import render_trace_artifacts

    updated = []
    for receipt in results:
        scenario = receipt.get("scenario")
        if scenario is None:
            schedule = receipt.get("disturbance_schedule", [])
            scenario = "push" if schedule else "nominal"
        scenario_dir = output / scenario
        receipt["visual_artifacts"] = render_trace_artifacts(
            scenario_dir / "trace.npz", scene, scenario_dir,
            method=args.method, scenario=scenario, fps=args.video_fps,
            width=args.video_width, height=args.video_height,
        )
        _write_json(scenario_dir / "receipt.json", receipt)
        updated.append(receipt)
    _write_json(output / "results.json", updated)
    return updated


def _clean_environment(python: Path, ros_domain_id: int) -> dict[str, str]:
    return {
        "HOME": os.environ.get("HOME", str(Path.home())),
        "USER": os.environ.get("USER", ""),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "PATH": ":".join((
            str(python.parent), "/usr/local/sbin", "/usr/local/bin",
            "/usr/sbin", "/usr/bin", "/sbin", "/bin",
        )),
        "PYTHONNOUSERSITE": "1",
        "ROS_DOMAIN_ID": str(ros_domain_id),
    }


def run_qm_control(args) -> tuple[Path, list[dict]]:
    roboduet_root = Path(__file__).resolve().parents[2]
    baseline_root = Path(args.baseline_root or DEFAULT_BASELINE_ROOT).resolve()
    python = Path(args.python).resolve()
    required = (
        baseline_root / "benchmark/aligned_cli.py",
        baseline_root / "benchmark/task_contract.py",
        baseline_root / "qm_control_baseline/install_aligned/setup.bash",
        baseline_root / "qm_control_baseline/src/go2_x5_whole_body_mpc/scripts/mujoco_bridge.py",
        baseline_root / "mujoco_models/go2_x5_description/mjcf/scene.xml",
        python,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing qm_control dependency: " + ", ".join(missing))

    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    existing_outputs = {path.resolve() for path in output_root.iterdir() if path.is_dir()}
    command = [
        str(python), "-m", "benchmark.aligned_cli",
        "--roboduet-root", str(roboduet_root),
        "--suite", str(Path(args.suite).resolve()),
        "--output-dir", str(output_root),
        "--mujoco-python", str(python),
        "--ros-domain-id", str(args.ros_domain_id),
        "--scenarios", *args.scenarios,
    ]
    if args.task_id:
        command.extend(("--task-id", args.task_id))
    if args.prepare_only:
        command.append("--prepare-only")
    started = dt.datetime.now(dt.timezone.utc)
    completed = subprocess.run(
        command, cwd=baseline_root,
        env=_clean_environment(python, args.ros_domain_id), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=args.timeout_s, check=False,
    )
    candidates = [
        path for path in output_root.iterdir()
        if path.is_dir() and path.resolve() not in existing_outputs
    ]
    if not candidates:
        raise RuntimeError("qm_control adapter produced no output directory\n" + completed.stdout)
    output = max(candidates, key=lambda path: path.stat().st_mtime_ns)
    (output / "adapter.log").write_text(completed.stdout)
    if completed.returncode:
        raise RuntimeError(
            f"qm_control adapter exited {completed.returncode}; see {output / 'adapter.log'}"
        )
    results = [] if args.prepare_only else _normalize_results(output, roboduet_root)
    results = _render_results(
        output, results, args,
        baseline_root / "mujoco_models/go2_x5_description/mjcf/scene.xml",
    )
    manifest = {
        "schema_version": "cross-method-mujoco-run-v1",
        "status": "prepared" if args.prepare_only else "complete",
        "method": "qm_control",
        "started_at": started.isoformat(),
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": command,
        "suite": str(Path(args.suite).resolve()),
        "task_id": args.task_id,
        "scenarios": list(args.scenarios),
        "roboduet_git": _git_state(roboduet_root),
        "dependencies": {str(path): _sha256(path) for path in required if path != python},
        "python": str(python),
        "result_count": len(results),
    }
    _write_json(output / "cross_method_manifest.json", manifest)
    _write_json(output / "run_state.json", {"status": manifest["status"], "output": str(output)})
    return output, results


def _selected_group(suite_path: Path, task_id: str | None) -> tuple[dict, dict, int]:
    payload = json.loads(suite_path.read_text())
    wrappers = payload if isinstance(payload, list) else [payload]
    matches = []
    for wrapper in wrappers:
        if wrapper.get("robustness_scenario", "nominal") != "nominal":
            continue
        group = wrapper.get("suite", wrapper)
        for row, task in enumerate(group["trajectories"]):
            if task_id is None or task["task_id"] == task_id:
                matches.append((wrapper, group, row))
    if not matches:
        raise ValueError(f"no nominal task {task_id!r} in {suite_path}")
    if task_id is None and len(matches) != 1:
        raise ValueError("--task-id is required when the suite has multiple nominal tasks")
    return matches[0]


def _materialize_scenario_suite(
    source: Path, destination: Path, task_id: str | None, scenario: str,
) -> tuple[Path, str]:
    # Keep torch out of module import so IsaacGym-first test processes remain valid.
    from benchmark.wbc.suite import finalize_task_spec, refresh_suite_hash

    wrapper, source_group, row = _selected_group(source, task_id)
    group = copy.deepcopy(source_group)
    task = copy.deepcopy(group["trajectories"][row])
    source_task_id = task["task_id"]
    if scenario == "push":
        finalize_task_spec(
            task, initial_state=task["initial_state"],
            anchor_env_local=task["anchor_env_local_xyz_m"],
            orientation_left_multiplier_xyzw=task["orientation_left_multiplier_xyzw"],
            deadline_s=task["deadline_s"],
            disturbance_schedule=[{
                "type": "constant_force", "start_time_s": 0.1,
                "duration_s": 0.1, "body": "base",
                "frame": "environment_world", "force_n": [80.0, 0.0, 0.0],
                "magnitude_n": 80.0, "application_point": "body_center_of_mass",
            }],
        )
    group["trajectories"] = [task]
    archive_record = wrapper.get("reference_archive", source_group["reference_archive"])
    archive_source = (source.parent / archive_record["path"]).resolve()
    with np.load(archive_source, allow_pickle=False) as archive:
        old_ids = [str(value) for value in archive["task_id"].tolist()]
        archive_row = old_ids.index(source_task_id)
        arrays = {}
        for name in archive.files:
            value = archive[name]
            if value.ndim and value.shape[0] == len(old_ids):
                arrays[name] = value[archive_row:archive_row + 1]
            else:
                arrays[name] = value
        arrays["task_id"] = np.asarray([task["task_id"]])
    destination.mkdir(parents=True, exist_ok=True)
    archive_out = destination / "reference.npz"
    np.savez_compressed(archive_out, **arrays)
    record = {"path": archive_out.name, "format": "numpy_npz_v1", "sha256": _sha256(archive_out)}
    group["reference_archive"] = record
    refresh_suite_hash(group)
    suite_out = destination / "suite.json"
    _write_json(suite_out, [{"suite": group, "reference_archive": record,
                             "robustness_scenario": scenario}])
    return suite_out, task["task_id"]


def run_policy_method(args) -> tuple[Path, list[dict]]:
    contract = _method_contracts()[args.method]
    readiness = preflight(args.method)["methods"][args.method]
    if readiness["status"] != "ready":
        raise RuntimeError(f"{args.method} preflight blocked: {readiness['blocker']}; missing={readiness['missing']}")
    roboduet_root = Path(__file__).resolve().parents[2]
    python = Path(args.python or "/opt/miniconda3/envs/isaacgym/bin/python").resolve()
    output_root = Path(args.output or (
        roboduet_root / "benchmark/results/cross_method_mujoco" / args.method
    )).resolve()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = output_root / stamp
    output.mkdir(parents=True)
    started = dt.datetime.now(dt.timezone.utc)
    results = []
    commands = []
    for scenario in args.scenarios:
        scenario_dir = output / scenario
        suite_path, scenario_task_id = _materialize_scenario_suite(
            Path(args.suite).resolve(), scenario_dir, args.task_id, scenario,
        )
        policy_root = (
            contract["root"] if args.method in ("roboduet", "roboduet_raw")
            else DEFAULT_RL_SAR_ROOT
        )
        command = [
            str(python), "-m", "benchmark.wbc.mujoco",
            "--suite", str(suite_path), "--task-id", scenario_task_id,
            "--rl-sar-root", str(policy_root),
            "--policy-key", contract["policy_key"], "--scene", str(DEFAULT_SCENE),
            "--output", str(scenario_dir), "--upper-controller", "scripted_dls_ik",
        ]
        if args.method == "ma2022":
            bundle = DEFAULT_RL_SAR_ROOT / "policy/go2_x5/ma2022_student"
            command.extend([
                "--policy-adapter", "ma2022",
                "--ma2022-deployment-root", str(contract["root"]),
                "--ma2022-policy", str(bundle / "student_policy.pt"),
                "--ma2022-env-config", str(bundle / "env_cfg.json"),
                "--ma2022-config", str(contract["root"] / "config.yaml"),
            ])
        elif args.method == "wb_locoman":
            command.extend([
                "--policy-adapter", "wb_locoman",
                "--wb-locoman-root", str(contract["root"]),
                "--wb-locoman-python", "/opt/miniconda3/envs/base312/bin/python",
            ])
        elif args.method == "deep_whole_body_control":
            command.extend([
                "--policy-adapter", "dwbc", "--dwbc-root", str(contract["root"]),
                "--dwbc-checkpoint", str(contract["root"] / "legged_gym/logs/go2_x5/1789388761_go2_x5_reward_fix/model_11000.pt"),
            ])
        elif args.method == "visual_wholebody":
            command.extend([
                "--policy-adapter", "visual", "--dwbc-root", str(contract["root"]),
                "--dwbc-checkpoint", str(contract["root"] / "low-level/logs/go2x5-visual-low/go2x5_low_v6_velocity_curriculum_tb_resume1000/model_24000.pt"),
            ])
        elif args.method == "umi":
            command.extend([
                "--policy-adapter", "umi",
                "--umi-checkpoint", str(contract["root"] / "checkpoints/tossing/ours-real/model.pt"),
            ])
        commands.append(command)
        if args.prepare_only:
            continue
        completed = subprocess.run(
            command, cwd=roboduet_root, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=args.timeout_s, check=False,
        )
        (scenario_dir / "run.log").write_text(completed.stdout)
        if completed.returncode:
            raise RuntimeError(
                f"{args.method} {scenario} exited {completed.returncode}; see {scenario_dir / 'run.log'}"
            )
        receipt_path = scenario_dir / "receipt.json"
        if not receipt_path.is_file():
            raise RuntimeError(f"{args.method} {scenario} produced no receipt")
        receipt = json.loads(receipt_path.read_text())
        coverage = _coverage(receipt["result"])
        receipt.update(method=args.method, scenario=scenario,
                       benchmark_owner="RoboDuet/benchmark/wbc",
                       metric_coverage=coverage)
        _write_json(scenario_dir / "metric_coverage.json", coverage)
        _write_json(receipt_path, receipt)
        results.append(receipt)
    status = "prepared" if args.prepare_only else "complete"
    manifest = {
        "schema_version": "cross-method-mujoco-run-v1", "status": status,
        "method": args.method, "started_at": started.isoformat(),
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_suite": str(Path(args.suite).resolve()), "source_task_id": args.task_id,
        "scenarios": list(args.scenarios), "commands": commands,
        "roboduet_git": _git_state(roboduet_root), "preflight": readiness,
        "result_count": len(results),
    }
    _write_json(output / "cross_method_manifest.json", manifest)
    results = _render_results(output, results, args, DEFAULT_SCENE)
    _write_json(output / "results.json", results)
    _write_json(output / "run_state.json", {"status": status, "output": str(output)})
    return output, results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--list-methods", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--suite")
    parser.add_argument("--task-id")
    parser.add_argument("--scenarios", nargs="+", choices=("nominal", "push"),
                        default=("nominal", "push"))
    parser.add_argument("--output")
    parser.add_argument("--baseline-root")
    parser.add_argument("--python")
    parser.add_argument("--ros-domain-id", type=int, default=91)
    parser.add_argument("--timeout-s", type=float, default=480.0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=25)
    parser.add_argument("--video-width", type=int, default=960)
    parser.add_argument("--video-height", type=int, default=540)
    args = parser.parse_args(argv)
    if args.list_methods or args.preflight:
        print(json.dumps(preflight(args.method), indent=2))
        return
    if not args.method:
        parser.error("--method is required (or use --list-methods)")
    if not args.suite:
        parser.error("--suite is required for a run")
    if not 0 <= args.ros_domain_id <= 232:
        parser.error("--ros-domain-id must be in 0..232")
    if args.video_fps < 1 or args.video_width < 64 or args.video_height < 64:
        parser.error("video fps must be positive and dimensions at least 64 pixels")
    readiness = preflight(args.method)["methods"][args.method]
    if readiness["status"] != "ready":
        raise SystemExit(
            f"{args.method} is not runnable on the common MuJoCo plant: "
            f"{readiness['blocker']}; missing={readiness['missing']}"
        )
    if args.method == "qm_control":
        if args.output is None:
            args.output = "benchmark/results/cross_method_mujoco/qm_control"
        if args.python is None:
            args.python = "/opt/miniconda3/envs/base312/bin/python"
        output, results = run_qm_control(args)
    else:
        output, results = run_policy_method(args)
    print(json.dumps({"output": str(output), "results": results}, indent=2))


if __name__ == "__main__":
    main()
