"""Summarize policy configurations without treating them as seed repeats."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    results = json.loads(args.results.read_text())
    metrics = ("lin_vel_x_rmse", "lin_vel_y_rmse", "ang_vel_yaw_rmse",
               "pitch_rmse_deg", "roll_rmse_deg", "height_rmse_m",
               "response_consistency_rmse", "fall_rate", "fall_rate_height")
    rows = []
    for name, scenarios in results.items():
        family = "rlmpc" if "rlmpc" in name else "robust"
        for scenario, points in scenarios.items():
            if not points:
                continue
            weights = np.array([p["n_env_steps"] for p in points], dtype=float)
            row = dict(run=name, family=family, scenario=scenario)
            for metric in metrics:
                values = np.array([p[metric] for p in points], dtype=float)
                valid = np.isfinite(values) & (weights > 0)
                if not valid.any():
                    row[metric] = float("nan")
                elif "rmse" in metric:
                    row[metric] = float(np.sqrt(np.average(values[valid] ** 2, weights=weights[valid])))
                else:
                    row[metric] = float(np.average(values[valid], weights=weights[valid]))
            rows.append(row)
    out = args.results.parent
    with (out / "series_summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "family", "scenario", *metrics])
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# RL-MPC and robust policy comparison", "",
             "Values below are mean +/- sample SD across trained policies, after pooling squared errors "
             "within each run with environment-step weights. Runs have different training configurations; "
             "this is configuration dispersion, not seed dispersion or a confidence interval.", "",
             "| Scenario | Metric | RL-MPC | Robust | RL-MPC / robust |",
             "|---|---|---:|---:|---:|"]
    for scenario in sorted(set(r["scenario"] for r in rows)):
        for metric in metrics:
            values = [np.array([r[metric] for r in rows if r["family"] == family and r["scenario"] == scenario])
                      for family in ("rlmpc", "robust")]
            if any(len(v) == 0 for v in values):
                continue
            labels = [f"{v.mean():.5g} +/- {v.std(ddof=1) if len(v)>1 else 0:.3g}" for v in values]
            ratio = values[0].mean() / values[1].mean() if values[1].mean() else float("nan")
            lines.append(f"| {scenario} | {metric} | {labels[0]} | {labels[1]} | {ratio:.3g} |")
    lines.extend(["", "## Interpretation limits", "",
                  "The local comparison contains RL-MPC runs 0-3 and robust runs 1-6. RL-MPC run 4 was absent.", "",
                  "These series differ in observation/history layout (112 x 50 versus 90 x 30), and potentially "
                  "other training settings. This is not an isolated response-reward ablation. The benchmark "
                  "uses a common scenario recipe and RNG seed but distinct layout groups do not guarantee "
                  "identical stepwise domain draws. Fall metrics retain the evaluator's definitions; "
                  "they are not push-recovery or terrain-traversal success probabilities. Body-pose aggregate "
                  "metrics pool the command sweep; consult results.json for individual axes and command points.", "",
                  "The separate chirp identification outputs use checkpoint-specific domain configurations. "
                  "The full R9 three-domain command suite, feasibility envelope and MPC validation are not "
                  "established by this benchmark.", ""])
    (out / "comparison.md").write_text("\n".join(lines))
    print(out / "comparison.md")


if __name__ == "__main__":
    main()
