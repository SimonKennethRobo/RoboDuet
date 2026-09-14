"""Run external methods on the frozen RoboDuet MuJoCo benchmark contract.

The benchmark contract and final scoring live in RoboDuet. Method-specific
processes remain in their original baseline trees and are pinned by content
hashes in every receipt. The first adapter reuses qm_control's native
SQP-MPC, QP-WBC, and common Go2+X5 MuJoCo plant.
"""

from __future__ import annotations

import argparse
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


METHODS = ("qm_control",)
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
    workspace = roboduet_root.parent
    baseline_root = Path(args.baseline_root or workspace / "baselines/mpc_baseline").resolve()
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
    candidates = [path for path in output_root.iterdir() if path.is_dir()]
    if not candidates:
        raise RuntimeError("qm_control adapter produced no output directory\n" + completed.stdout)
    output = max(candidates, key=lambda path: path.stat().st_mtime_ns)
    (output / "adapter.log").write_text(completed.stdout)
    if completed.returncode:
        raise RuntimeError(
            f"qm_control adapter exited {completed.returncode}; see {output / 'adapter.log'}"
        )
    results = [] if args.prepare_only else _normalize_results(output, roboduet_root)
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, default="qm_control")
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task-id")
    parser.add_argument("--scenarios", nargs="+", choices=("nominal", "push"),
                        default=("nominal", "push"))
    parser.add_argument("--output", default="benchmark/results/cross_method_mujoco/qm_control")
    parser.add_argument("--baseline-root")
    parser.add_argument("--python", default="/opt/miniconda3/envs/base312/bin/python")
    parser.add_argument("--ros-domain-id", type=int, default=91)
    parser.add_argument("--timeout-s", type=float, default=480.0)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    if not 0 <= args.ros_domain_id <= 232:
        parser.error("--ros-domain-id must be in 0..232")
    output, results = run_qm_control(args)
    print(json.dumps({"output": str(output), "results": results}, indent=2))


if __name__ == "__main__":
    main()
