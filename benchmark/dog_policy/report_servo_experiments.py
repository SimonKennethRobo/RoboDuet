"""Report frozen-model prospective prediction and matched OCS2 closed-loop trials."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from benchmark.dog_policy.servo_model import load_samples, evaluate, score


def confirmation(models_path, data_root):
    models_path, root = Path(models_path), Path(data_root)
    bundle = json.loads(models_path.read_text())
    manifest, samples = load_samples(root)
    if not manifest.get("prospective_confirmation"):
        raise ValueError("Expected prospective confirmation data")
    if manifest["checkpoint_sha256"] != bundle["checkpoint_sha256"]:
        raise ValueError("Policy mismatch")
    results = {name: evaluate(bundle["models"]["coupled" if name == "coupled" else "separate"],
                            samples, "test", ideal_arm=name == "base_only")
               for name in ("base_only", "separate", "coupled")}
    lines = ["# Prospective arm servo confirmation", "",
        "24 new trajectories: two new postures, new phases and chirped frequencies. "
        "Models were frozen before collection; none of these samples were used for fitting or selection.",
        "100/300/500/1000 ms predictions start from measured state once, then advance the model with recorded commands.", "",
        "| Model | Horizon (ms) | Arm aggregate RMSE (deg) | J2 (deg) | J3 (deg) |",
        "|---|---:|---:|---:|---:|"]
    for name, metrics in results.items():
        for h, r in metrics.items():
            a = np.array(r["rmse"])
            lines.append(f"| {name} | {h} | {np.rad2deg(np.sqrt(np.mean(a[5:]**2))):.3f} | "
                         f"{np.rad2deg(a[6]):.3f} | {np.rad2deg(a[7]):.3f} |")
    # Paired trial-level gains, keeping overlapping windows within their trial.
    bootstrap = {}
    rng = np.random.default_rng(4201)
    for candidate in ("separate", "coupled"):
        baseline = "base_only" if candidate == "separate" else "separate"
        bootstrap[candidate] = {}
        for h in results[candidate]:
            a = np.asarray(results[baseline][h]["per_trial_mse"])[:, 5:].mean(axis=1)
            b = np.asarray(results[candidate][h]["per_trial_mse"])[:, 5:].mean(axis=1)
            idx = rng.integers(0, len(a), (2000, len(a)))
            delta = 1. - np.sqrt(b[idx].mean(axis=1) / a[idx].mean(axis=1))
            bootstrap[candidate][h] = dict(baseline=baseline,
                relative_rmse_reduction=float(1.-np.sqrt(b.mean()/a.mean())),
                trial_bootstrap_95_percentile=np.percentile(delta, [2.5,97.5]).tolist())
    ratio = score(results["coupled"]) / score(results["separate"])
    lines += ["", f"Coupled/separate normalized aggregate multi-horizon MSE ratio: {ratio:.4f}.",
        "The separate servo model improves short/medium-horizon aggregate arm prediction here, but regresses at 1 s. "
        "J3 does not improve consistently. There is no evidence to promote the explicit coupling candidate.",
        "The bootstrap in confirmation.json resamples whole excitation trials; it does not establish generalization beyond the two new posture centers.",
        "All results use nominal flat ground and one frozen policy/drive configuration. The joint model is a local affine approximation, not a complete robot dynamics model."]
    result = dict(models_sha256=hashlib.sha256(models_path.read_bytes()).hexdigest(),
                  data_sha256=hashlib.sha256((root / "excitation.npz").read_bytes()).hexdigest(),
                  selected_on_validation=bundle["selected_on_validation"], metrics=results, bootstrap=bootstrap)
    (root / "confirmation.json").write_text(json.dumps(result, indent=2) + "\n")
    (root / "report.md").write_text("\n".join(lines) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), constrained_layout=True)
    for name, metrics in results.items():
        horizons = [int(h) for h in metrics]
        rmse = np.asarray([r["rmse"] for r in metrics.values()])
        values = [np.sqrt(np.mean(rmse[:, 5:]**2, axis=1)), rmse[:, 6], rmse[:, 7]]
        for ax, value in zip(axes, values):
            ax.plot(horizons, np.rad2deg(value), marker="o", label=name)
    for ax, title in zip(axes, ["All six arm joints", "J2", "J3"]):
        ax.set(title=title, xlabel="Prediction horizon (ms)", ylabel="Joint angle RMSE (deg)")
        ax.grid(alpha=.25)
    axes[0].legend(frameon=False)
    fig.savefig(root / "prediction.png", dpi=180)
    fig.savefig(root / "prediction.pdf")
    plt.close(fig)
    print("\n".join(lines))


def closed_loop(root):
    root = Path(root)
    results = [json.loads(p.read_text()) for p in sorted(root.glob("*/metrics.json"))]
    lines = ["# Measured OCS2 servo-model comparison", "",
        "OCS2 SqpSolver / HPIPM drives the same frozen RL policy and persistent arm target interface in IsaacGym. "
        "Base-only means identified base + ideal arm velocity; separate and coupled use the frozen identified servo models.",
        "Identical URDF FK, 1 s horizon, EE/input costs, command/slew/joint-target bounds, task, seed and startup in each triplet.",
        "These bounds are the shared screening limits, not an identified feasibility envelope. "
        "Synchronous simulation does not emulate computation/communication latency.", "",
        "| Task | Seed | Model | Complete | Position RMSE (cm) | Orientation RMSE (deg) |",
        "|---|---:|---|---|---:|---:|"]
    for r in results:
        pos = f"{100*r['position_rmse_m']:.3f}" if r["completed"] else "n/a"
        rot = f"{r['orientation_rmse_deg']:.3f}" if r["completed"] else "n/a"
        lines.append(f"| {r['task']} | {r['seed']} | {r['mode']} | {r['completed']} | {pos} | {rot} |")
    pairs, prediction_checks = [], []
    for r in results:
        trial = root / f"{r['task']}_{r['seed']}_{r['mode']}"
        solves = json.loads((trial / "solver.json").read_text())
        errors = [np.asarray(a["predicted_state"])[8:14] - np.asarray(b["measured_state"])[8:14]
                  for a, b in zip(solves[:-1], solves[1:]) if a["ok"] and b["ok"]]
        if errors:
            rmse = np.rad2deg(np.sqrt(np.mean(np.asarray(errors)**2, axis=0)))
            prediction_checks.append(dict(task=r["task"], seed=r["seed"], mode=r["mode"],
                joint_100ms_rmse_deg=rmse.tolist(), aggregate_rmse_deg=float(np.sqrt(np.mean(rmse**2))),
                min_constraint_margin=min(s["constraint_margin"] for s in solves if s["ok"]),
                max_dynamics_defect=max(s["max_dynamics_defect"] for s in solves if s["ok"])))
    for task, seed in sorted({(r["task"], r["seed"]) for r in results}):
        group = {r["mode"]: r for r in results if (r["task"], r["seed"]) == (task, seed)}
        if set(group) != {"base_only", "separate", "coupled"}:
            continue
        manifests = {name: json.loads((root / f"{task}_{seed}_{name}" / "manifest.json").read_text()) for name in group}
        keys = ("checkpoint_sha256", "asset_sha256", "models_sha256", "solver_executable_sha256", "initial_state", "initial_ee",
                "command_low", "command_high", "q_low", "q_high", "mpc_period", "horizon", "orientation_weight", "push_scale",
                "config", "arm_drive_properties", "constraint_handling", "max_sqp_iterations")
        for key in keys:
            if any(m[key] != manifests["base_only"][key] for m in manifests.values()):
                raise ValueError(f"Unmatched {task}/{seed}: {key}")
        requests = {name: json.loads((root / f"{task}_{seed}_{name}" / "first_request.json").read_text()) for name in group}
        for key in ("state", "dt", "horizon", "roll", "chain", "references", "ee_weights", "R", "D", "terminal_weight",
                    "constraint_C", "constraint_D", "constraint_e", "state_constraint_F", "state_constraint_h"):
            if any(r[key] != requests["base_only"][key] for r in requests.values()):
                raise ValueError(f"Unmatched OCS2 problem {task}/{seed}: {key}")
        if all(r["completed"] for r in group.values()):
            pairs.append(dict(task=task, seed=seed, relative_position_change_separate=
                              group["separate"]["position_rmse_m"]/group["base_only"]["position_rmse_m"]-1.,
                              relative_orientation_change_separate=
                              group["separate"]["orientation_rmse_deg"]/group["base_only"]["orientation_rmse_deg"]-1.))
    lines += ["", f"{len(results)} recorded trials, {len(pairs)} complete matched triplets. "
              "Incomplete trajectories are not used to claim RMSE gains.",
              "The legacy SLSQP hold diagnostics used a different arm interface and local EE approximation; "
              "do not attribute cross-protocol differences to OCS2 or to servo identification alone."]
    lines += ["", "| Task | Seed | Model | Closed-loop 100 ms arm prediction RMSE (deg) | J2 (deg) | J3 (deg) |",
              "|---|---:|---|---:|---:|---:|"]
    for r in prediction_checks:
        lines.append(f"| {r['task']} | {r['seed']} | {r['mode']} | {r['aggregate_rmse_deg']:.3f} | "
                     f"{r['joint_100ms_rmse_deg'][1]:.3f} | {r['joint_100ms_rmse_deg'][2]:.3f} |")
    lines += ["", "These closed-loop prediction errors come from each controller's own visited states/actions; "
              "they are diagnostic, not a matched-input prediction test. Use the prospective excitation report for the latter."]
    (root / "summary.json").write_text(json.dumps(dict(trials=results, matched_triplets=pairs,
                                                       prediction_checks=prediction_checks), indent=2) + "\n")
    (root / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="data/identification/arm_servo_20260911/models.json")
    p.add_argument("--confirmation")
    p.add_argument("--closed-loop")
    args = p.parse_args()
    if args.confirmation: confirmation(args.models, args.confirmation)
    if args.closed_loop: closed_loop(args.closed_loop)
    if not (args.confirmation or args.closed_loop): p.error("select a report")
