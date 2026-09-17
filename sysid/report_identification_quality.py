"""Create machine-readable and plotted prediction-quality evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sysid.identify_iq_mujoco import CHANNELS, ROOT, sha, write_json
from sysid.iq_response_model import load_episodes
from sysid import identification_models as response_models

UNITS = ["m/s", "m/s", "rad/s", "m", "rad", "rad"]
SCALES = np.array([.55, .28, .65, .035, .20, .14])


def aggregate(quality):
    rows = []
    for horizon, values in quality["summary"].items():
        rmse = np.asarray(values["rmse"])
        rows.append(dict(horizon_s=float(horizon), rmse=rmse.tolist(),
                         normalized_rmse=(rmse/SCALES).tolist(),
                         mean_normalized_rmse=float(np.mean(rmse/SCALES))))
    return rows


def plot_horizons(models, destination):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for axis, channel, unit in zip(axes.flat, CHANNELS, UNITS):
        c = CHANNELS.index(channel)
        for name, quality in models.items():
            rows = aggregate(quality)
            axis.plot([r["horizon_s"] for r in rows], [r["rmse"][c] for r in rows],
                      marker="o", label=name)
        axis.set_title(channel); axis.set_ylabel(f"RMSE [{unit}]"); axis.grid(alpha=.25)
    for axis in axes[-1]: axis.set_xlabel("prediction horizon [s]")
    axes[0, 0].legend(ncol=2, fontsize=8)
    fig.tight_layout(); fig.savefig(destination.with_suffix(".png"), dpi=180)
    fig.savefig(destination.with_suffix(".pdf")); plt.close(fig)


def representative_episode(episodes):
    successful = [row for row in episodes if row[0].get("success")]
    candidates = successful or episodes
    return max(candidates, key=lambda row: len(row[1]["t"]))


def plot_trace(model_dicts, episode, destination):
    info, data = episode
    start = min(100, max(0, len(data["t"])-max(response_models.HORIZONS)-1))
    horizon = min(round(4./response_models.DT), len(data["t"])-start-1)
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    time = np.arange(horizon+1)*response_models.DT
    for axis, channel, unit in zip(axes.flat, CHANNELS, UNITS):
        c = CHANNELS.index(channel)
        axis.plot(time, data["y"][start:start+horizon+1, c], color="black", lw=2, label="measured")
        axis.plot(time, data["u"][start:start+horizon+1, c], color="gray", ls=":", label="command")
        for name, model in model_dicts.items():
            pred = response_models.predict_window(response_models.bundle_from_dict(model, CHANNELS),
                data, start, horizon, info.get("gait_frequency_hz", 2.75), info.get("stop_gait_at_stand", True))
            axis.plot(time, pred[:, c], label=name)
        axis.set_title(channel); axis.set_ylabel(unit); axis.grid(alpha=.25)
    for axis in axes[-1]: axis.set_xlabel("forecast time [s]")
    axes[0, 0].legend(ncol=2, fontsize=8)
    fig.suptitle(f"Causal forecast from one measured state: {info['id']}")
    fig.tight_layout(); fig.savefig(destination.with_suffix(".png"), dpi=180)
    fig.savefig(destination.with_suffix(".pdf")); plt.close(fig)
    return info["id"], start*response_models.DT


def main(args):
    root = Path(args.root).resolve(); models_dir = root/args.model_dir
    selection = json.loads((models_dir/"selection.json").read_text())
    model_names = sorted(selection["model_hashes"])
    model_dicts = {name: json.loads((models_dir/f"{name}.json").read_text()) for name in model_names}
    for name, digest in selection["model_hashes"].items():
        if sha(models_dir/f"{name}.json") != digest: raise ValueError(f"model changed: {name}")
    episodes = load_episodes(root, "test")
    qualities = {name: response_models.quality_metrics(
        response_models.bundle_from_dict(model, CHANNELS), episodes) for name, model in model_dicts.items()}
    output = root/args.output_dir
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"quality report exists: {output}; pass --overwrite to regenerate identical evidence")
    output.mkdir(parents=True, exist_ok=True)
    plot_horizons(qualities, output/"rmse_by_horizon")
    episode_id, forecast_start = plot_trace(model_dicts, representative_episode(episodes), output/"prediction_traces")
    report = dict(schema="policy_response_quality_v1", selected_model=selection["selected_model"],
        selection_sha256=sha(models_dir/"selection.json"), model_hashes=selection["model_hashes"],
        development_scores=selection["development_scores"],
        protocol_sha256=sha(root/"protocol.json"), channels=CHANNELS, units=UNITS,
        normalization_scales=SCALES.tolist(), test={name: dict(
            horizons=aggregate(quality), details=quality) for name, quality in qualities.items()},
        representative_trace=dict(episode=episode_id, forecast_start_s=forecast_start),
        plots=["rmse_by_horizon.png", "rmse_by_horizon.pdf", "prediction_traces.png", "prediction_traces.pdf"],
        evidence_boundary="offline causal prediction on frozen held-out MuJoCo trajectories; not proof of MPC tracking improvement")
    baseline = report["test"]["F0"]["horizons"]
    for name, model_report in report["test"].items():
        for row, base in zip(model_report["horizons"], baseline):
            row["mean_normalized_rmse_improvement_vs_F0_percent"] = float(
                100.*(base["mean_normalized_rmse"]-row["mean_normalized_rmse"])/base["mean_normalized_rmse"])
    write_json(output/"quality.json", report)
    selected = report["test"][report["selected_model"]]["horizons"]
    lines = ["# Identification quality", "", f"Selected on development set: `{report['selected_model']}`",
             f"Development score: `{selection['development_scores'][report['selected_model']]:.6g}`", "",
             "Held-out test RMSE (columns follow vx, vy, wz, height, pitch, roll):", ""]
    for row in selected:
        lines.append(f"- {row['horizon_s']:g} s: " + ", ".join(f"{v:.6g}" for v in row["rmse"])
                     + f"; normalized improvement vs F0 {row['mean_normalized_rmse_improvement_vs_F0_percent']:.2f}%")
    lines += ["", "Plots: `rmse_by_horizon.*`, `prediction_traces.*`.", "",
              "Boundary: offline causal prediction evidence only; MPC closed-loop benefit needs a separate paired benchmark.", ""]
    (output/"REPORT.md").write_text("\n".join(lines))
    print(json.dumps(dict(selected_model=report["selected_model"],
        test_horizons=report["test"][report["selected_model"]]["horizons"],
        output=str(output)), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT/"tmp/experiments/20260915_iq_identification"))
    parser.add_argument("--model-dir", default="models_v2")
    parser.add_argument("--output-dir", default="identification_quality")
    parser.add_argument("--overwrite", action="store_true")
    main(parser.parse_args())
