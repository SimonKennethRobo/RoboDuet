"""Fit and validate the paper's first-order + phase/speed residual model.

Fits use disjoint complete training trajectories. Validation rolls the model
forward from ONE measured initial response/phase and uses only future commands;
future measured speed or phase is never supplied to the predictor.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, lsq_linear
from scipy.signal import lfilter

from identify_iq_mujoco import CHANNELS, ROOT, sha, write_json

DT = .02
HORIZONS = [5, 15, 30, 50]


def load_episodes(root, split):
    episodes = []
    for receipt in sorted((root/"raw").glob(f"{split}_*.json")):
        info = json.loads(receipt.read_text())
        path = receipt.with_suffix(".npz")
        if sha(path) != info["sha256"]:
            raise ValueError(f"raw data hash mismatch: {path}")
        arrays = dict(np.load(path))
        # Failed prefixes remain available, but never treat a reset as a
        # continuation. Keep both their inclusion and failure counts explicit.
        episodes.append((info, arrays))
    return episodes


def gait_basis(data, harmonic):
    speed = np.sqrt(np.sum(data["y"][:, :2]**2, axis=1) + 1e-6)
    phase = data["phase"]
    sn, cs = np.sin(harmonic*phase), np.cos(harmonic*phase)
    return np.column_stack([sn, cs, speed*sn, speed*cs])


def regression(episodes, channel, tau, harmonic=0):
    a = np.exp(-DT/tau)
    designs, targets = [], []
    for _, data in episodes:
        y, u = data["y"][:, channel], data["u"][:, channel]
        if not np.isfinite(y).all():
            continue
        filtered = np.r_[0., lfilter([1-a], [1., -a], u)]
        basis = gait_basis(data, harmonic) if harmonic else None
        if harmonic:
            # Hold the new command's gate on BOTH endpoints of each interval.
            # Integrating delta_dot introduces no artificial impulse when the
            # optimizer changes commands at a multiple-shooting boundary.
            walk = np.linalg.norm(data["u"][:-1, :3], axis=1) >= .1
            force = walk[:, None]*(basis[1:]-a*basis[:-1])
            residual_filter = np.vstack([np.zeros(4), lfilter([1.], [1., -a], force, axis=0)])
        for h in HORIZONS:
            starts = np.arange(0, len(y)-h, 5)
            if not len(starts):
                continue
            decay = a**h
            command = filtered[starts+h] - decay*filtered[starts]
            columns = [command, np.full(len(starts), 1-decay)]
            if harmonic:
                columns.extend((residual_filter[starts+h]-decay*residual_filter[starts]).T)
            designs.append(np.column_stack(columns))
            targets.append(y[starts+h]-decay*y[starts])
    return np.concatenate(designs), np.concatenate(targets)


def coefficients(parameters):
    gain, bias, a0, a1, phase = parameters[:5]
    return np.array([gain, bias, a0*np.cos(phase), a0*np.sin(phase),
                     a1*np.cos(phase), a1*np.sin(phase)])


def fit_channel(episodes, channel, harmonic=0):
    best = None
    for tau in np.geomspace(.04, 2.5, 35):
        x, target = regression(episodes, channel, tau, harmonic)
        beta = np.linalg.lstsq(x, target, rcond=None)[0]
        loss = np.mean((x@beta-target)**2)
        if best is None or loss < best[0]:
            best = loss, tau, beta
    _, tau, beta = best
    if not harmonic:
        def error(p):
            x, target = regression(episodes, channel, p[2])
            return x@p[:2]-target
        result = least_squares(error, np.r_[beta, tau],
            bounds=([-5., -2., .025], [5., 2., 3.]), max_nfev=50)
        gain, bias, tau = result.x
        return dict(gain=float(gain), bias=float(bias), tau_s=float(tau), delay_s=0.,
                    fit_rmse=float(np.sqrt(np.mean(result.fun**2))))
    # The paper uses ONE phase offset for a0 and a1. Project the unconstrained
    # sine/cosine estimate to a shared phase, then optimize that exact model.
    mat = np.array([beta[2:4], beta[4:6]])
    left, singular, right = np.linalg.svd(mat, full_matrices=False)
    amps, phase = left[:, 0]*singular[0], np.arctan2(right[0, 1], right[0, 0])
    p0 = np.r_[beta[:2], amps, phase, tau]
    lower = [-5., -2., -.5, -1., -4*np.pi, .025]
    upper = [5., 2., .5, 1., 4*np.pi, 3.]
    p0 = np.clip(p0, np.array(lower)+1e-8, np.array(upper)-1e-8)
    def error(p):
        x, target = regression(episodes, channel, p[5], harmonic)
        return x@coefficients(p)-target
    result = least_squares(error, p0, bounds=(lower, upper), max_nfev=65,
                           ftol=1e-8, xtol=1e-8, gtol=1e-8)
    gain, bias, a0, a1, phase, tau = result.x
    return dict(gain=float(gain), bias=float(bias), tau_s=float(tau), delay_s=0.,
        residual=dict(amplitude=float(a0), speed_amplitude=float(a1),
                      harmonic=int(harmonic), phase_offset=float(phase)),
        fit_rmse=float(np.sqrt(np.mean(result.fun**2))))


def delta(model, velocity, phase, walk):
    result = np.zeros(6)
    if not walk:
        return result
    speed = np.sqrt(float(velocity[0]**2+velocity[1]**2)+1e-6)
    for c in range(3, 6):
        residual = model[CHANNELS[c]].get("residual")
        if residual:
            result[c] = (residual["amplitude"]+residual["speed_amplitude"]*speed)*np.sin(
                residual["harmonic"]*phase+residual["phase_offset"])
    return result


def predict(model, y0, phase0, previous_walk, commands, gait_frequency=2.75):
    gain = np.array([model[c]["gain"] for c in CHANNELS])
    bias = np.array([model[c]["bias"] for c in CHANNELS])
    decay = np.exp(-DT/np.array([model[c]["tau_s"] for c in CHANNELS]))
    y = np.asarray(y0).copy()
    phase = phase0
    result = [y.copy()]
    for u in commands:
        walk = np.linalg.norm(u[:3]) >= .1
        nominal = y-delta(model, y[:2], phase, walk)
        nominal = decay*nominal + (1-decay)*(gain*u+bias)
        phase += 2*np.pi*gait_frequency*DT*walk
        y = nominal + delta(model, nominal[:2], phase, walk)
        result.append(y.copy())
    return np.asarray(result)


def evaluate(model, episodes):
    errors = {h: [] for h in HORIZONS}
    for info, data in episodes:
        for start in range(0, len(data["t"])-max(HORIZONS), 10):
            previous_walk = start > 0 and np.linalg.norm(data["u"][start-1, :3]) >= .1
            prediction = predict(model, data["y"][start], data["phase"][start],
                previous_walk, data["u"][start:start+max(HORIZONS)])
            for h in HORIZONS:
                errors[h].append(prediction[h]-data["y"][start+h])
    return {str(round(h*DT, 2)): dict(zip(CHANNELS,
                np.sqrt(np.mean(np.array(values)**2, axis=0)).tolist()))
            for h, values in errors.items()}


def fit(args):
    root = Path(args.root).resolve()
    output = root/args.model_dir
    output.mkdir(exist_ok=True)
    if (output/"selection.json").exists():
        raise FileExistsError("selection is already frozen; use a new experiment version")
    train = load_episodes(root, "train")
    dev = load_episodes(root, "development")
    if len(train) != 36 or len(dev) != 18:
        raise ValueError(f"expected 36 training/18 development episodes, got {len(train)}/{len(dev)}")
    nominal = {name: fit_channel(train, c) for c, name in enumerate(CHANNELS)}
    write_json(output/"F0.json", nominal)
    candidates = {}
    selection = {}
    combined = dict(nominal)
    for c in range(3, 6):
        name = CHANNELS[c]
        rows = []
        for harmonic in [1, 2, 3, 4]:
            channel = fit_channel(train, c, harmonic)
            candidate = {**nominal, name: channel}
            metrics = evaluate(candidate, dev)
            score = np.mean([metrics[str(h)][name]**2 for h in [.1, .3, .6, 1.]])
            rows.append(dict(harmonic=harmonic, development_mse=float(score), model=channel))
            print(name, harmonic, score, flush=True)
        best = min(rows, key=lambda row: row["development_mse"])
        selection[name] = best
        combined[name] = best["model"]
        candidates[name] = rows
    write_json(output/"F1_gait.json", combined)
    nominal_dev, gait_dev = evaluate(nominal, dev), evaluate(combined, dev)
    # Integration exports both ablations even if residual fails to improve.
    # Selection does not inspect test data.
    write_json(output/"selection.json", dict(schema="iq_response_selection_v1",
        protocol_sha256=sha(root/"protocol.json"), fitted_on=[e[0]["id"] for e in train],
        selected_on=[e[0]["id"] for e in dev], selected_harmonics=selection,
        candidates=candidates, development=dict(F0=nominal_dev, F1_gait=gait_dev),
        source_sha256=sha(__file__)))
    test = load_episodes(root, "test")
    if len(test) != 18:
        raise ValueError("test collection incomplete")
    # Frozen models selected above; testing has no feedback into fitting.
    report = dict(schema="iq_response_validation_v1", model_hashes={
        "F0": sha(output/"F0.json"), "F1_gait": sha(output/"F1_gait.json")},
        claim="future-command-only multi-horizon prediction on disjoint MuJoCo episodes",
        horizons_s=[.1, .3, .6, 1.], units=["m/s", "m/s", "rad/s", "m", "rad", "rad"],
        test=dict(F0=evaluate(nominal, test), F1_gait=evaluate(combined, test)),
        coverage={split: dict(episodes=len(es), failures=sum(not e[0]["success"] for e in es),
            failed_ids=[e[0]["id"] for e in es if not e[0]["success"]])
            for split, es in [("train", train), ("development", dev), ("test", test)]})
    for arm in ["fixed", "moving"]:
        es = [e for e in test if e[0]["arm"] == arm]
        report["test_"+arm] = dict(F0=evaluate(nominal, es), F1_gait=evaluate(combined, es))
    write_json(root/(args.model_dir+"_prediction_validation.json"), report)
    print(json.dumps(report, indent=2))


def validate_frozen(args):
    root = Path(args.root).resolve()
    data_root = Path(args.data_root).resolve() if args.data_root else root
    models = {name: json.loads((root/args.model_dir/f"{name}.json").read_text())
              for name in ["F0", "F1_gait"]}
    hashes = {name: sha(root/args.model_dir/f"{name}.json") for name in models}
    protocol = json.loads((data_root/"protocol.json").read_text())
    if protocol.get("frozen_models", hashes) != hashes:
        raise ValueError("validation protocol was frozen against different models")
    episodes = load_episodes(data_root, "test")
    expected = [e["id"] for e in protocol["episodes"] if e["split"] == "test"]
    if sorted(e[0]["id"] for e in episodes) != sorted(expected) or not episodes:
        raise ValueError("held-out collection incomplete")
    report = dict(schema="iq_frozen_prediction_validation_v2", model_hashes=hashes,
        selection_sha256=sha(root/args.model_dir/"selection.json"),
        evaluator_sha256=sha(__file__), data_root=str(data_root),
        protocol_sha256=sha(data_root/"protocol.json"), speed_epsilon=1e-6,
        claim="single measured initial response and phase, then future commands only; no test feedback into fit/selection",
        units=["m/s", "m/s", "rad/s", "m", "rad", "rad"],
        coverage=dict(episodes=len(episodes), failures=sum(not e[0]["success"] for e in episodes),
                      failed_ids=[e[0]["id"] for e in episodes if not e[0]["success"]]),
        test={name: evaluate(model, episodes) for name, model in models.items()})
    for arm in ["fixed", "moving"]:
        group = [e for e in episodes if e[0]["arm"] == arm]
        if group:
            report["test_"+arm] = {name: evaluate(model, group) for name,model in models.items()}
    destination = root/args.report_name
    if destination.exists():
        raise FileExistsError("validation report exists; choose a fresh --report-name")
    write_json(destination, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT/"tmp/experiments/20260915_iq_identification"))
    parser.add_argument("--model-dir", default="models_v2")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--data-root")
    parser.add_argument("--report-name", default="independent_prediction_validation.json")
    args = parser.parse_args()
    validate_frozen(args) if args.evaluate_only else fit(args)
