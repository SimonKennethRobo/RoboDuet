"""Grouped holdout, causal 0.2/0.5/1.0 s prediction of recorded response channels.

No simulator imports. Physical +RPY commands, native body-frame planar velocity,
fitted gain/bias/bandwidth/delay, and phase extrapolation from window start.
This evaluates five channels, not a complete SE(3) rollout: old exports omit
the roll command. Existing datasets also have different domain recipes.
"""
import argparse
import csv
import gc
import json
import pickle
from pathlib import Path

import numpy as np
from scipy import optimize, signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CHANNELS = ("vx", "vy", "wyaw", "height", "pitch")
ORDERS = ("ideal", "first", "second", "second+gait")
HORIZONS = (0.2, 0.5, 1.0)
VERSION = "grouped-horizon-v2-body-command-contract"


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean_json(value), indent=2, allow_nan=False))
    temporary.replace(path)


def response_basis(u, dt, parameter, order):
    """Exact ZOH filters, returning input, initial-position and rate terms."""
    t = np.arange(u.shape[-1]) * dt
    if order == "first":
        a = np.exp(-dt / parameter)
        return signal.lfilter([0, 1 - a], [1, -a], u, axis=-1), np.exp(-t / parameter), np.zeros_like(t)
    e = np.exp(-parameter * dt)
    s = parameter * dt
    b = [0, 1 - e * (1 + s), e * e - e * (1 - s)]
    h = (1 + parameter * t) * np.exp(-parameter * t)
    return signal.lfilter(b, [1, -2 * e, e * e], u, axis=-1), h, t * np.exp(-parameter * t)


def delayed(u, delay):
    return np.r_[np.repeat(u[0], delay), u][:len(u)] if delay else u


def phase_basis(phase, speed, speed_terms=True):
    base = np.stack([fn(2 * np.pi * n * phase) for n in (1, 2, 3)
                     for fn in (np.sin, np.cos)], axis=-1)
    return np.concatenate((base, base * speed[..., None]), axis=-1) if speed_terms else base


def load_dataset(path):
    meta = json.loads(path.with_suffix(".json").read_text())
    with open(meta["provenance"]["config_snapshot"], "rb") as handle:
        cfg = pickle.load(handle)["Cfg"]
    keys = ("command", "measured", "gait_phase", "gait_frequency_hz", "is_standing",
            "reset", "plan_generation", "excitation_channel", "excitation_signal",
            "is_identification", "group_of")
    with np.load(path) as archive:
        data = {k: archive[k] for k in keys}
    # The recorded local velocity command is scored against body-frame velocity
    # by this policy environment. Converting only y to heading frame changes
    # the input-output contract. A full pose predictor must rotate its predicted
    # body twist with predicted attitude, not future measured attitude.
    if cfg["commands"].get("global_reference", False):
        raise ValueError("This evaluator requires the recorded local body-frame command contract")
    legacy = "response" not in cfg and cfg["dog"]["dog_num_observations"] == 90
    if legacy:
        data["command"][:, :, 4] *= -1
    data["speed"] = np.linalg.norm(data["measured"][:, :, :2], axis=-1)
    return data, meta, cfg, legacy


def get_records(data, channel, dt):
    rows = []
    for env in np.flatnonzero(data["is_identification"]):
        generation = data["plan_generation"][:, env]
        change = np.diff(generation) != 0
        change |= np.diff(data["excitation_channel"][:, env]) != 0
        change |= np.diff(data["excitation_signal"][:, env]) != 0
        edges = np.r_[0, np.flatnonzero(change) + 1, len(generation)]
        group = int(data["group_of"][env])
        group = group if group >= 0 else 100000 + int(env)
        for lo, hi in zip(edges[:-1], edges[1:]):
            if data["excitation_channel"][lo, env] != channel:
                continue
            valid = (data["reset"][lo:hi, env] == 0)
            valid &= np.isfinite(data["measured"][lo:hi, env, channel])
            valid &= np.isfinite(data["command"][lo:hi, env, channel])
            boundaries = np.flatnonzero(np.diff(np.r_[False, valid, False]))
            for start, end in zip(boundaries[::2], boundaries[1::2]):
                start, end = lo + start, lo + end
                if (end - start) * dt < 4:
                    continue
                u = data["command"][start:end, env, channel].astype(float)
                if np.std(u) < 1e-5:
                    continue
                rows.append(dict(env=int(env), group=group, start=int(start), end=int(end),
                                 signal=int(data["excitation_signal"][start, env]),
                                 generation=int(generation[start]), u=u,
                                 y=data["measured"][start:end, env, channel].astype(float),
                                 phase=data["gait_phase"][start:end, env].astype(float),
                                 frequency=data["gait_frequency_hz"][start:end, env].astype(float),
                                 moving=data["is_standing"][start:end, env] == 0,
                                 speed=data["speed"][start:end, env].astype(float)))
    return rows


def fit_lag(rows, dt, order):
    warmup = max(1, round(1 / dt))
    bounds = (0.02, 3.0) if order == "first" else (0.8, 35.0)
    best = None
    # Delay is selected on training records only. Report it as an effective lag,
    # not identified actuator latency, because exports are post-step samples.
    for delay in (0, 1, 2, 3, 5):
        def objective(log_parameter, full=False):
            parameter = np.exp(log_parameter)
            designs, targets = [], []
            for row in rows:
                xu, h, _ = response_basis(delayed(row["u"], delay), dt, parameter, order)
                designs.append(np.column_stack((xu[warmup:], 1 - h[warmup:])))
                targets.append(row["y"][warmup:] - h[warmup:] * row["y"][0])
            X, y = np.concatenate(designs), np.concatenate(targets)
            gain_bias = np.linalg.lstsq(X, y, rcond=None)[0]
            mse = float(np.mean((X @ gain_bias - y) ** 2))
            return (mse, gain_bias) if full else mse
        grid = np.linspace(*np.log(bounds), 24)
        scores = [objective(x) for x in grid]
        idx = int(np.argmin(scores))
        opt = optimize.minimize_scalar(objective, bounds=(grid[max(0, idx - 1)],
                                                          grid[min(len(grid) - 1, idx + 1)]), method="bounded")
        mse, (gain, bias) = objective(opt.x, full=True)
        if best is None or mse < best["training_mse"]:
            best = dict(order=order, parameter=float(np.exp(opt.x)), gain=float(gain), bias=float(bias),
                        delay_steps=delay, delay_s=delay * dt, training_mse=mse,
                        parameter_name="tau_s" if order == "first" else "omega_n_rad_s",
                        parameter_bounds=list(bounds))
    return best


def fit_gait(rows, dt, model):
    warmup = max(1, round(1 / dt))
    designs, errors = [], []
    for row in rows:
        xu, h, _ = response_basis(delayed(row["u"], model["delay_steps"]), dt, model["parameter"], "second")
        prediction = model["gain"] * xu + model["bias"] * (1 - h) + row["y"][0] * h
        keep = row["moving"].copy()
        keep[:warmup] = False
        designs.append(phase_basis(row["phase"], row["speed"])[keep])
        errors.append((row["y"] - prediction)[keep])
    X, y = np.concatenate(designs), np.concatenate(errors)
    if len(X) < 12:
        return dict(status="unavailable: too few moving training samples", coefficients=[0.] * 6,
                    speed_terms=False, rank=0, samples=len(X))
    rank = int(np.linalg.matrix_rank(X))
    speed_terms = rank == 12
    if not speed_terms:
        X = X[:, :6]
    penalty = max(float(np.trace(X.T @ X)) / X.shape[1] * 1e-6, 1e-10)
    coefficients = np.linalg.solve(X.T @ X + penalty * np.eye(X.shape[1]), X.T @ y)
    return dict(status="fitted", coefficients=coefficients.tolist(), speed_terms=speed_terms,
                rank=int(np.linalg.matrix_rank(X)), samples=len(X))


def evaluate_record(row, dt, models):
    steps = [round(h / dt) for h in HORIZONS]
    max_h = max(steps)
    starts = np.arange(6, len(row["u"]) - max_h, max(1, round(0.2 / dt)))
    if len(starts) == 0:
        return []
    offsets = np.arange(max_h + 1)
    indices = starts[:, None] + offsets
    truth = row["y"][indices]
    y0 = row["y"][starts]
    derivative = (y0 - row["y"][starts - 5]) / (5 * dt)
    output = []
    for order in ORDERS:
        if order == "ideal":
            prediction = row["u"][indices]
        else:
            base_order = "second" if order == "second+gait" else order
            model = models[base_order]
            u = row["u"][indices - model["delay_steps"]]
            xu, h, hd = response_basis(u, dt, model["parameter"], base_order)
            initial, initial_rate = y0.copy(), derivative.copy()
            correction = 0.
            if order == "second+gait":
                gait = models["gait"]
                # Forecast phase using only the current clock. Never use future
                # measured gait phase or future measured speed as predictor input.
                future_phase = row["phase"][starts, None] + row["frequency"][starts, None] * offsets * dt
                current_speed = np.broadcast_to(row["speed"][starts, None], future_phase.shape)
                correction = phase_basis(future_phase, current_speed, gait["speed_terms"]) @ gait["coefficients"]
                correction *= row["moving"][starts, None]
                initial -= correction[:, 0]
                initial_rate -= (correction[:, 1] - correction[:, 0]) / dt
            prediction = (model["gain"] * xu + model["bias"] * (1 - h)
                          + initial[:, None] * h + initial_rate[:, None] * hd + correction)
        for horizon, step in zip(HORIZONS, steps):
            error = prediction[:, step] - truth[:, step]
            output.append(dict(model=order, horizon_s=horizon, mse=float(np.mean(error ** 2)),
                               absolute_error_p95=float(np.percentile(np.abs(error), 95)),
                               windows=len(starts), group=row["group"], env=row["env"],
                               generation=row["generation"], start=row["start"], signal=row["signal"]))
    return output


def aggregate(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["group"], []).append(row["mse"])
    values = np.array([np.mean(v) for v in grouped.values()])
    if not len(values):
        return {"status": "no held-out records"}
    rng = np.random.RandomState(17)
    bootstrap = np.sqrt(rng.choice(values, (1000, len(values)), replace=True).mean(axis=1))
    return dict(rmse=float(np.sqrt(values.mean())), group_bootstrap_95ci=np.percentile(bootstrap, [2.5, 97.5]).tolist(),
                groups=len(values), records=len(rows), windows=sum(r["windows"] for r in rows))


def analyze(path):
    data, meta, cfg, legacy = load_dataset(path)
    dt = meta["dt_s"]
    result = dict(version=VERSION, dataset=str(path), policy=meta["policy"], dt_s=dt,
                  command_pitch_sign=-1 if legacy else 1, channels={},
                  training_configuration={"response": cfg.get("response"),
                                          "commands": cfg.get("commands"),
                                          "reset_mix_hard_fraction": cfg["terrain"].get("reset_mix_hard_fraction"),
                                          "max_push_ang_vel": cfg["domain_rand"].get("max_push_ang_vel")},
                  limitations=["Checkpoint-specific domain recipes; not a matched reward ablation.",
                               "Five-channel prediction, not full SE(3); original data omit roll commands.",
                               "Native body-frame velocity command/response; full MPC frame conversion remains separate.",
                               "Effective delay includes post-step logging alignment uncertainty.",
                               "Gait frequency and residual amplitude use window-start values."])
    detailed = []
    for channel, name in enumerate(CHANNELS):
        records = get_records(data, channel, dt)
        train = [r for r in records if r["group"] % 3 != 0 and r["signal"] == 1 and len(r["u"]) * dt >= 8]
        test = [r for r in records if r["group"] % 3 == 0]
        train_groups = sorted(set(r["group"] for r in train))
        test_groups = sorted(set(r["group"] for r in test))
        info = dict(train_groups=train_groups, test_groups=test_groups, train_chirps=len(train), test_records=len(test))
        if len(train_groups) < 2 or not test:
            info["status"] = "insufficient independent training chirps or test groups"
        else:
            models = {order: fit_lag(train, dt, order) for order in ("first", "second")}
            models["gait"] = fit_gait(train, dt, models["second"])
            measurements = [r for record in test for r in evaluate_record(record, dt, models)]
            for row in measurements:
                row.update(run=path.stem, channel=name)
            detailed.extend(measurements)
            summaries = {}
            for order in ORDERS:
                summaries[order] = {str(h): aggregate([r for r in measurements if r["model"] == order and r["horizon_s"] == h])
                                    for h in HORIZONS}
            info.update(status="ok", models=models, prediction=summaries)
        result["channels"][name] = info
        print(path.stem, name, info["status"], flush=True)
    del data
    gc.collect()
    return result, detailed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--output", default="data/identification/prediction_horizons")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for path in map(Path, args.datasets):
        target = out / (path.stem + "_prediction.json")
        if target.exists():
            cached = json.loads(target.read_text())
            if cached.get("version") == VERSION:
                results[path.stem] = cached
                print("resuming cached", path.stem, flush=True)
                continue
        result, details = analyze(path)
        if details:
            with (out / (path.stem + "_trials.csv")).open("w") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(details[0]))
                writer.writeheader()
                writer.writerows(details)
        save_json(target, result)
        results[path.stem] = result
    save_json(out / "summary.json", results)
    lines = ["# Held-out response prediction", "",
             "Each run is a different configuration, not a seed repeat. Lower RMSE is better. "
             "Fit: complete chirp records in training groups. Test: disjoint complete groups, including "
             "PRBS, chirp and ramp. Nominal twins cannot cross the train/test boundary. "
             "Group bootstrap intervals and per-trial data are in JSON/CSV.", "",
             "| Run | Channel | Horizon (s) | Ideal | First | Second | Second + gait |", "|---|---|---:|---:|---:|---:|---:|"]
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharex=True)
    for name, result in results.items():
        family_row = 0 if "rlmpc" in name else 1
        for c, channel in enumerate(CHANNELS):
            info = result["channels"][channel]
            if info["status"] != "ok":
                lines.append(f"| {name} | {channel} | - | insufficient data | - | - | - |")
                continue
            for h in HORIZONS:
                values = [info["prediction"][m][str(h)].get("rmse", float("nan")) for m in ORDERS]
                lines.append(f"| {name} | {channel} | {h} | " + " | ".join(f"{v:.5g}" for v in values) + " |")
            axes[family_row, c].plot(HORIZONS, [info["prediction"]["second+gait"][str(h)]["rmse"] for h in HORIZONS],
                                     "o-", label=name)
    for row in range(2):
        for c, channel in enumerate(CHANNELS):
            axes[row, c].set(title=("RL-MPC " if row == 0 else "Robust ") + channel,
                             xlabel="Open-loop horizon (s)", ylabel="RMSE (channel units)")
            axes[row, c].grid(alpha=.2)
        axes[row, -1].legend(fontsize=5)
    fig.tight_layout()
    fig.savefig(out / "prediction_horizons.png", dpi=180)
    lines.extend(["", "## Scope", "",
                  "No future measured state is injected into prediction windows. Second-order rate is "
                  "initialized from the preceding 0.1 s. Future recorded commands are replayed as candidate "
                  "inputs. Phase is extrapolated from its initial value/frequency, not taken from future logs.", "",
                  "Legacy 90D pitch commands are mapped to physical +RPY. Fits have independent gain, bias, "
                  "bandwidth and effective discrete delay. An unavailable gait fit falls back to the base "
                  "second-order model and is explicitly flagged in JSON.", "",
                  "These exports differ in training-domain recipe and roll-command support. The old archive "
                  "does not contain the roll command, so this report does not claim full SE(3) prediction "
                  "or matched-domain causality. No identified parameters are automatically deployed to MPC.", ""])
    (out / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
