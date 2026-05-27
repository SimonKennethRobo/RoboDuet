"""Offline CI checks for benchmark report tooling.

This script intentionally avoids IsaacGym, torch, CUDA, and checkpoint loading.
It verifies the parts that should be stable on GitHub-hosted runners:

- benchmark profile JSON can be parsed;
- saved results can generate standalone HTML reports;
- saved result directories can be compared without rerunning simulation.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _sample_results(scale: float) -> dict:
    return {
        "ci_candidate": {
            "vel_grid": [
                {
                    "label": "vx=+0.0 yaw=+0.0",
                    "lin_vel_x_rmse": 0.10 * scale,
                    "ang_vel_yaw_rmse": 0.05 * scale,
                    "tracking_lin_vel_reward": 0.90 / scale,
                    "tracking_ang_vel_reward": 0.85 / scale,
                    "base_height_mean": 0.31,
                    "fall_rate": 0.0,
                    "max_torque_mean": 12.0 * scale,
                },
                {
                    "label": "vx=+1.0 yaw=+0.0",
                    "lin_vel_x_rmse": 0.20 * scale,
                    "ang_vel_yaw_rmse": 0.08 * scale,
                    "tracking_lin_vel_reward": 0.80 / scale,
                    "tracking_ang_vel_reward": 0.75 / scale,
                    "base_height_mean": 0.30,
                    "fall_rate": 0.0,
                    "max_torque_mean": 13.0 * scale,
                },
            ],
            "arm_sweep": [
                {
                    "label": "intensity=1.00",
                    "lin_vel_x_rmse": 0.25 * scale,
                    "ang_vel_yaw_rmse": 0.09 * scale,
                    "tracking_lin_vel_reward": 0.76 / scale,
                    "tracking_ang_vel_reward": 0.70 / scale,
                    "base_height_mean": 0.30,
                    "fall_rate": 0.0,
                    "max_torque_mean": 15.0 * scale,
                }
            ],
        }
    }


def _write_result(root: Path, name: str, scale: float) -> Path:
    out = root / name
    out.mkdir(parents=True)
    (out / "results.json").write_text(json.dumps(_sample_results(scale), indent=2), encoding="utf-8")
    metadata = {
        "benchmark_protocol": "ci-static",
        "benchmark_mode": "dog_only",
        "seed": 1,
        "profile": "ci-fixture",
        "git": {"branch": "ci", "short_commit": "offline"},
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out


def _run(args: list[str]) -> None:
    subprocess.run(args, cwd=REPO_ROOT, check=True)


def _validate_profiles() -> None:
    profile_dir = REPO_ROOT / "benchmark" / "profiles"
    required = {"num_envs_per_policy", "num_eval_steps", "seed", "scenarios"}
    for path in sorted(profile_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            profile = json.load(f)
        missing = sorted(required - set(profile))
        if missing:
            raise ValueError(f"{path}: missing required profile fields: {missing}")
        if not isinstance(profile["scenarios"], list) or not profile["scenarios"]:
            raise ValueError(f"{path}: scenarios must be a non-empty list")


def main() -> None:
    _validate_profiles()
    with tempfile.TemporaryDirectory(prefix="roboduet-benchmark-ci-") as tmp:
        results_root = Path(tmp) / "results"
        baseline = _write_result(results_root, "baseline", scale=1.0)
        target = _write_result(results_root, "target", scale=0.8)

        _run([sys.executable, "-m", "benchmark.reports.html", "--results_root", str(results_root)])
        _run(
            [
                sys.executable,
                "-m",
                "benchmark.cli",
                "--compare_results",
                "--baseline",
                str(baseline),
                "--target",
                str(target),
            ]
        )

        expected = [
            results_root / "index.html",
            baseline / "index.html",
            target / "index.html",
            target / "compare_baseline_to_target.html",
        ]
        missing = [str(path) for path in expected if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"benchmark static check did not create expected artifacts: {missing}")


if __name__ == "__main__":
    main()
