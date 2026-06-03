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


def _scaled_candidate(candidate: dict, factor: float) -> dict:
    scaled = {}
    for scenario, rows in candidate.items():
        scaled[scenario] = []
        for row in rows:
            next_row = {}
            for key, value in row.items():
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and not key.startswith("cmd_")
                    and key not in {"n_env_steps", "n_falls"}
                ):
                    next_row[key] = value * factor
                else:
                    next_row[key] = value
            scaled[scenario].append(next_row)
    return scaled


def _sample_results(scale: float) -> dict:
    candidate = {
            "vel_grid": [
                {
                    "label": "vx=+0.0 vy=+0.0 yaw=+0.0",
                    "cmd_x": 0.0,
                    "cmd_y": 0.0,
                    "cmd_yaw": 0.0,
                    "lin_vel_xy_rmse": 0.11 * scale,
                    "lin_vel_x_rmse": 0.10 * scale,
                    "lin_vel_y_rmse": 0.02 * scale,
                    "ang_vel_yaw_rmse": 0.05 * scale,
                    "tracking_lin_vel_reward": 0.90 / scale,
                    "tracking_ang_vel_reward": 0.85 / scale,
                    "base_height_mean": 0.31,
                    "fall_rate_height": 0.0,
                    "fall_rate": 0.0,
                    "max_torque_mean": 12.0 * scale,
                },
                {
                    "label": "vx=+1.0 vy=+0.0 yaw=+0.0",
                    "cmd_x": 1.0,
                    "cmd_y": 0.0,
                    "cmd_yaw": 0.0,
                    "lin_vel_xy_rmse": 0.21 * scale,
                    "lin_vel_x_rmse": 0.20 * scale,
                    "lin_vel_y_rmse": 0.03 * scale,
                    "ang_vel_yaw_rmse": 0.08 * scale,
                    "tracking_lin_vel_reward": 0.80 / scale,
                    "tracking_ang_vel_reward": 0.75 / scale,
                    "base_height_mean": 0.30,
                    "fall_rate_height": 0.0,
                    "fall_rate": 0.0,
                    "max_torque_mean": 13.0 * scale,
                },
            ],
            "arm_sweep": [
                {
                    "label": "intensity=1.00",
                    "lin_vel_xy_rmse": 0.26 * scale,
                    "lin_vel_x_rmse": 0.25 * scale,
                    "lin_vel_y_rmse": 0.04 * scale,
                    "ang_vel_yaw_rmse": 0.09 * scale,
                    "tracking_lin_vel_reward": 0.76 / scale,
                    "tracking_ang_vel_reward": 0.70 / scale,
                    "base_height_mean": 0.30,
                    "fall_rate_height": 0.0,
                    "fall_rate": 0.0,
                    "max_torque_mean": 15.0 * scale,
                }
            ],
            "body_pose": [
                {
                    "label": "forward | pitch=+0.20rad",
                    "velocity_group": "forward",
                    "pose_axis": "pitch",
                    "sweep_axis": "pitch",
                    "cmd_pitch": 0.2,
                    "pitch_rmse_deg": 4.0 * scale,
                    "roll_rmse_deg": 1.0 * scale,
                    "height_rmse_m": 0.02 * scale,
                    "lin_vel_xy_rmse": 0.18 * scale,
                    "fall_rate_height": 0.0,
                    "fall_rate": 0.0,
                },
                {
                    "label": "forward | roll=-0.20rad",
                    "velocity_group": "forward",
                    "pose_axis": "roll",
                    "sweep_axis": "roll",
                    "cmd_roll": -0.2,
                    "pitch_rmse_deg": 1.0 * scale,
                    "roll_rmse_deg": 3.0 * scale,
                    "height_rmse_m": 0.03 * scale,
                    "lin_vel_xy_rmse": 0.20 * scale,
                    "fall_rate_height": 0.0,
                    "fall_rate": 0.0,
                },
                {
                    "label": "stand | height_delta=+0.10m",
                    "velocity_group": "stand",
                    "pose_axis": "height",
                    "sweep_axis": "height",
                    "cmd_height_delta": 0.1,
                    "pitch_rmse_deg": 0.8 * scale,
                    "roll_rmse_deg": 0.9 * scale,
                    "height_rmse_m": 0.04 * scale,
                    "lin_vel_xy_rmse": 0.08 * scale,
                    "fall_rate_height": 0.0,
                    "fall_rate": 0.0,
                },
            ],
            "gait": [
                {
                    "label": "gait_freq=4.0Hz",
                    "sweep_axis": "gait_freq",
                    "cmd_gait_freq": 4.0,
                    "gait_freq_rmse_hz": 0.5 * scale,
                    "stance_width_rmse_m": 0.04 * scale,
                    "stance_length_rmse_m": 0.05 * scale,
                    "gait_contact_force_cost": 0.1 * scale,
                    "gait_contact_vel_cost": 0.2 * scale,
                    "fall_rate_height": 0.0,
                    "fall_rate": 0.0,
                    "max_torque_mean": 14.0 * scale,
                },
                {
                    "label": "stance_w=0.30m",
                    "sweep_axis": "stance_width",
                    "cmd_stance_width": 0.3,
                    "gait_freq_rmse_hz": 0.6 * scale,
                    "stance_width_rmse_m": 0.03 * scale,
                    "stance_length_rmse_m": 0.06 * scale,
                    "gait_contact_force_cost": 0.12 * scale,
                    "gait_contact_vel_cost": 0.22 * scale,
                    "fall_rate_height": 0.0,
                    "fall_rate": 0.0,
                    "max_torque_mean": 14.5 * scale,
                },
                {
                    "label": "stance_l=0.35m",
                    "sweep_axis": "stance_length",
                    "cmd_stance_length": 0.35,
                    "gait_freq_rmse_hz": 0.7 * scale,
                    "stance_width_rmse_m": 0.05 * scale,
                    "stance_length_rmse_m": 0.04 * scale,
                    "gait_contact_force_cost": 0.14 * scale,
                    "gait_contact_vel_cost": 0.24 * scale,
                    "fall_rate_height": 0.0,
                    "fall_rate": 0.0,
                    "max_torque_mean": 15.0 * scale,
                },
            ],
        }
    return {
        "ci_candidate": candidate,
        "ci_candidate_alt": _scaled_candidate(candidate, 1.35),
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


def _validate_inline_scripts(path: Path) -> None:
    script = (
        'const fs=require("fs");'
        'const html=fs.readFileSync(process.argv[1],"utf8");'
        'const scripts=[...html.matchAll(/<script>([\\s\\S]*?)<\\/script>/g)].map(m=>m[1]);'
        'if(!scripts.length) throw new Error("no scripts");'
        'scripts.forEach((s)=>new Function(s));'
    )
    subprocess.run(["node", "-e", script, str(path)], cwd=REPO_ROOT, check=True)


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

    standard = json.loads((profile_dir / "dog_policy_standard.json").read_text(encoding="utf-8"))
    full = json.loads((profile_dir / "dog_policy_full.json").read_text(encoding="utf-8"))
    _validate_profile_counts(profile_dir / "dog_policy_standard.json", standard, expected_vel=27)
    _validate_profile_counts(profile_dir / "dog_policy_full.json", full, expected_vel=125)


def _validate_profile_counts(path: Path, profile: dict, expected_vel: int) -> None:
    if profile.get("benchmark_protocol") != "dog_only":
        raise ValueError(f"{path}: benchmark_protocol must be dog_only")

    cfg = profile["scenario_config"]
    vel = cfg["vel_grid"]
    vel_count = len(vel["vx"]) * len(vel["vy"]) * len(vel["yaw"])
    if vel_count != expected_vel:
        raise ValueError(f"{path}: velocity grid has {vel_count} points, expected {expected_vel}")

    pose = cfg["body_pose"]
    pose_count = len(pose["velocity_groups"]) * (len(pose["pitch"]) + len(pose["roll"]) + len(pose["height_delta"]))
    if pose_count != 56:
        raise ValueError(f"{path}: body pose grid has {pose_count} points, expected 56")

    gait = cfg["gait"]
    gait_count = len(gait["gait_freq"]) + len(gait["stance_width"]) + len(gait["stance_length"])
    if gait_count != 18:
        raise ValueError(f"{path}: gait sweeps have {gait_count} points, expected 18")


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

        html = (baseline / "index.html").read_text(encoding="utf-8")
        required_snippets = [
            "renderVelocityHeatmap",
            "renderGroupedScenarioPlot",
            "pose-summary",
            "grouped-panels",
            "value-mode",
            "baseline-delta",
            "policy-mode",
            "baseline-select",
            "Fall diagnostics",
            "velocity-group-select",
            "shared-legend",
            "table-group-controls",
            "row.dataset.tableGroup !== group",
            "ensurePointVisible",
            "centerLinkedMetricCells",
            "setActiveMetricTab",
            "activatePlotMetricPoint",
            "sortTooltipValues",
            "renderGroupedInteraction",
            "groupedEventToPointIndex",
            "data-inner-top",
            "Chart + Table",
            "Heatmap + Table",
            "SCENARIO_GUIDES",
        ]
        missing_snippets = [snippet for snippet in required_snippets if snippet not in html]
        if missing_snippets:
            raise AssertionError(f"benchmark report is missing structured visual snippets: {missing_snippets}")
        _validate_inline_scripts(baseline / "index.html")


if __name__ == "__main__":
    main()
