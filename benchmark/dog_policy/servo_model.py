"""Discrete response identification with trajectory-level validation and test splits.

z = [body vx, vy, wz, world height, pitch, q(6), previous q(6), previous target(6)].
w = [five dog commands, arm position target at interval end(6)].
The arm target is ramped at the policy tick by the shared persistent interface.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

CHANNELS = ["vx", "vy", "wyaw", "height", "pitch"]


def load_samples(root, period=.1):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    data = np.load(root / "excitation.npz")
    stride = round(period / manifest["dt"])
    if not np.isclose(stride * manifest["dt"], period):
        raise ValueError("Model period must be an integer number of policy ticks")
    # Target at the last tick is the endpoint of the persistent target ramp.
    ticks = np.arange(stride, len(data["q"]) - stride, stride)
    z = np.concatenate([data["response"][ticks], data["q"][ticks], data["q"][ticks - stride], data["q_target"][ticks - 1]], axis=-1)
    zn = np.concatenate([data["response"][ticks + stride], data["q"][ticks + stride], data["q"][ticks], data["q_target"][ticks + stride - 1]], axis=-1)
    u = np.concatenate([data["command"][ticks], data["q_target"][ticks + stride - 1]], axis=-1)
    previous_target = data["q_target"][ticks - 1]
    valid = np.stack([data["valid"][t:t+stride].all(axis=0) for t in ticks])
    splits = np.array([s["split"] for s in manifest["trials"]])
    return manifest, dict(z=z, zn=zn, u=u, previous_target=previous_target, valid=valid, splits=splits, ticks=ticks)


def fit_model(samples, coupled, order, ridge):
    z, u, zn = (samples[k][:, samples["splits"] == "train"] for k in ("z", "u", "zn"))
    valid = samples["valid"][:, samples["splits"] == "train"]
    z, u, zn = z[valid], u[valid], zn[valid]
    A, B, c = np.zeros((23, 23)), np.zeros((23, 11)), np.zeros(23)
    for j in range(11):
        # Feature differences encode equilibrium and translational invariance.
        if j < 5:
            features = [z[:, j], u[:, j], np.ones(len(z))]
            maps = [("z", j), ("u", j), ("c", 0)]
            if coupled:
                features.extend([z[:, 5+i] for i in range(6)] + [z[:, 5+i]-z[:, 11+i] for i in range(6)])
                maps.extend([("z", 5+i) for i in range(6)] + [("delta", i) for i in range(6)])
            y = zn[:, j]
        else:
            k = j - 5
            features = [z[:, 17+k] - z[:, j], u[:, j] - z[:, 17+k], np.ones(len(z))]
            maps = [("error", k), ("target_delta", k), ("c", 0)]
            # Linear posture-dependent load within the sampled arm workspace.
            # This is internal arm structure, present in BOTH model classes.
            features.extend([z[:, 5+i] for i in range(6)])
            maps.extend([("z", 5+i) for i in range(6)])
            if order == 2:
                features.append(z[:, j] - z[:, 11+k])
                maps.append(("delta", k))
            if coupled:
                features.extend([z[:, i] for i in range(5)])
                maps.extend([("z", i) for i in range(5)])
            y = zn[:, j] - z[:, j]
            A[j, j] = 1.
        X = np.column_stack(features)
        scale = np.maximum(X.std(axis=0), .01)
        scale[[m[0] == "c" for m in maps]] = 1.
        Xs = X / scale
        penalty = np.eye(X.shape[1]) * ridge * len(X)
        penalty[[m[0] == "c" for m in maps], [m[0] == "c" for m in maps]] = 0.
        coeff = np.linalg.solve(Xs.T @ Xs + penalty, Xs.T @ y) / scale
        for value, (kind, k) in zip(coeff, maps):
            if kind == "z": A[j, k] += value
            elif kind == "u": B[j, k] += value
            elif kind == "c": c[j] += value
            elif kind == "error":
                A[j, 17+k] += value
                A[j, 5+k] -= value
            elif kind == "target_delta":
                B[j, 5+k] += value
                A[j, 17+k] -= value
            elif kind == "delta":
                A[j, 5+k] += value
                A[j, 11+k] -= value
    A[11:17, 5:11] = np.eye(6)
    B[17:23, 5:11] = np.eye(6)
    radius = float(np.max(np.abs(np.linalg.eigvals(A))))
    return dict(A=A.tolist(), B=B.tolist(), c=c.tolist(), coupled=coupled, order=order,
                ridge=ridge, spectral_radius=radius, period_s=.1)


def predict(model, z, u):
    return z @ np.asarray(model["A"]).T + u @ np.asarray(model["B"]).T + model["c"]


def evaluate(model, samples, split, ideal_arm=False):
    mask = samples["splits"] == split
    z, u, zn, valid, prev = (samples[k][:, mask] for k in ("z", "u", "zn", "valid", "previous_target"))
    horizons = {}
    for h in (1, 3, 5, 10):
        starts = np.arange(0, len(z)-h+1, 2)
        state = z[starts].copy()
        keep = np.ones(state.shape[:2], dtype=bool)
        for k in range(h):
            old = state
            state = predict(model, state, u[starts+k])
            if ideal_arm:
                state[..., 5:11] = old[..., 5:11] + u[starts+k, :, 5:11] - prev[starts+k]
            keep &= valid[starts+k]
        err = state[..., :11] - zn[starts+h-1, :, :11]
        err[~keep] = np.nan
        mse = np.nanmean(err**2, axis=0)
        horizons[str(h*100)] = dict(rmse=np.sqrt(np.nanmean(err**2, axis=(0, 1))).tolist(),
                                   per_trial_mse=mse.tolist(), transitions=int(keep.sum()))
    return horizons


def score(metrics):
    norm = np.array([.25, .18, .3, .025, .12] + [.15]*6)
    return float(np.mean([np.mean((np.asarray(v["rmse"]) / norm)**2) for v in metrics.values()]))


def fit_and_report(root):
    root = Path(root)
    manifest, samples = load_samples(root)
    chosen, selection = {}, {}
    for name, coupled in (("separate", False), ("coupled", True)):
        candidates = []
        for order in (1, 2):
            for ridge in (0., 1e-4, .001, .01, .1):
                model = fit_model(samples, coupled, order, ridge)
                # Reject unstable free-running dynamics and wrong-sign servo gains.
                good = bool(model["spectral_radius"] < 1. and np.all(np.diag(np.array(model["B"])[5:11, 5:11]) > 0))
                val = score(evaluate(model, samples, "validation")) if good else None
                candidates.append(dict(order=order, ridge=ridge, stable=good,
                                       spectral_radius=model["spectral_radius"], validation_score=val))
        feasible = [c for c in candidates if c["stable"]]
        if not feasible:
            raise RuntimeError(f"No stable positive-gain {name} model; inspect interface/excitation before proceeding")
        best = min(feasible, key=lambda c: c["validation_score"])
        chosen[name] = fit_model(samples, coupled, best["order"], best["ridge"])
        selection[name] = candidates
    metrics = {name: {split: evaluate(model, samples, split) for split in ("train", "validation", "test")}
               for name, model in chosen.items()}
    metrics["base_only"] = {split: evaluate(chosen["separate"], samples, split, ideal_arm=True)
                            for split in ("validation", "test")}
    # Freeze selection before examining test performance.
    ratio_val = score(metrics["coupled"]["validation"]) / score(metrics["separate"]["validation"])
    selected = "coupled" if ratio_val < .9 else "separate"
    bundle = dict(version="arm-servo-affine-v2", models=chosen, selected_on_validation=selected,
                  selection=selection, metrics=metrics,
                  checkpoint_sha256=manifest["checkpoint_sha256"], asset_sha256=manifest["asset_sha256"],
                  data_sha256=hashlib.sha256((root / "excitation.npz").read_bytes()).hexdigest(),
                  state_layout=CHANNELS + [f"q{i}" for i in range(6)] + [f"previous_q{i}" for i in range(6)] + [f"previous_target{i}" for i in range(6)],
                  input_layout=CHANNELS + [f"q_target_end{i}" for i in range(6)],
                  period_s=.1, interface=manifest["interface"],
                  scope="Nominal plane, fixed policy/drive, sampled postures and excitation amplitudes; not hardware or a safety envelope")
    (root / "models.json").write_text(json.dumps(bundle, indent=2, allow_nan=False) + "\n")
    lines = ["# Arm servo identification and coupling check", "",
        f"{len(manifest['trials'])} independent trials; {manifest['surviving_trials']} survived. "
        "36 training trials / 12 validation trials / 24 test trials, with disjoint postures and phases.",
        "The MPC is outside the identification boundary. No hold-push samples are fitted.",
        f"Applied-target readback maximum error: {manifest['max_applied_target_error_rad']:.3g} rad.", "",
        "Model: z_next = A z + B w + c. Arm rows model target error, target increment, posture-dependent load and optional position increment; "
        "coupled rows additionally use cross base/arm states. Arm target remains an explicit integrator state in MPC.",
        "Model order and ridge use validation rollouts at 100/300/500/1000 ms; unstable/wrong-sign fits are rejected.",
        "This pilot test was consulted during model-class development; final confirmation requires fresh trajectories. "
        "Each prediction is initialized once; no future measured state is injected.", "",
        "| Model | Horizon (ms) | Arm aggregate RMSE (deg) | J2 (deg) | J3 (deg) | vx (m/s) | height (m) | pitch (deg) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ("base_only", "separate", "coupled"):
        for horizon, result in metrics[name]["test"].items():
            r = np.array(result["rmse"])
            lines.append(f"| {name} | {horizon} | {np.rad2deg(np.sqrt(np.mean(r[5:]**2))):.3f} | "
                         f"{np.rad2deg(r[6]):.3f} | {np.rad2deg(r[7]):.3f} | {r[0]:.4f} | {r[3]:.4f} | {np.rad2deg(r[4]):.3f} |")
    ratio_test = score(metrics["coupled"]["test"]) / score(metrics["separate"]["test"])
    lines += ["", f"Normalized multi-horizon MSE ratio coupled/separate: validation={ratio_val:.4f}, test={ratio_test:.4f}.",
              f"Validation rule: select coupling only with >10% aggregate MSE reduction. Selected: `{selected}`.",
              "This is a model-class check within the sampled nominal domain; correlations do not establish a mechanical causal law.",
              "The measured test per-trial errors in models.json expose regressions hidden by aggregate improvements.",
              "Independent arm-only, base-only and combined excitations are included, but only six posture centers were sampled."]
    (root / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root")
    fit_and_report(p.parse_args().root)
