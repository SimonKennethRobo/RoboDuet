"""Fit held-out chirp response models and export model-order/Bode figures."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import signal, optimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CHANNELS = ("vx", "vy", "wyaw", "height", "pitch")


def predict(u, y0, dt, parameter, order):
    if order == "first":
        a = np.exp(-dt / parameter)
        return signal.lfilter([0, 1 - a], [1, -a], u - y0) + y0
    b, a, _ = signal.cont2discrete(
        ([parameter ** 2], [1, 2 * parameter, parameter ** 2]), dt
    )
    return signal.lfilter(b.ravel(), a, u - y0) + y0


def harmonics(phase):
    return np.column_stack([
        fn(2 * np.pi * n * phase)
        for n in (1, 2, 3) for fn in (np.sin, np.cos)
    ])


def records(data, channel, dt):
    result = []
    for env in np.flatnonzero(data["is_identification"]):
        generation = data["plan_generation"][:, env]
        edges = np.r_[0, np.flatnonzero(np.diff(generation)) + 1, len(generation)]
        for lo, hi in zip(edges[:-1], edges[1:]):
            # Reset samples contain the new episode's state, not its predecessor.
            valid = ((data["reset"][lo:hi, env] == 0)
                     & (data["excitation_signal"][lo:hi, env] == 1)
                     & (data["excitation_channel"][lo:hi, env] == channel))
            boundaries = np.flatnonzero(np.diff(np.r_[False, valid, False]))
            for start, end in zip(boundaries[::2], boundaries[1::2]):
                start, end = lo + start, lo + end
                if (end - start) * dt < 8:
                    continue
                u = data["command"][start:end, env, channel].astype(float)
                y = data["measured"][start:end, env, channel].astype(float)
                phase = data["gait_phase"][start:end, env].astype(float)
                if not np.isfinite(u).all() or not np.isfinite(y).all() or np.std(u) < 1e-5:
                    continue
                result.append(dict(env=int(env), start=int(start), u=u, y=y,
                                   phase=phase,
                                   moving=data["is_standing"][start:end, env] == 0))
    return result


def fit_channel(rows, dt):
    envs = sorted(set(row["env"] for row in rows))
    if len(envs) < 2:
        return {"status": "insufficient independent chirp environments", "segments": len(rows)}
    test_envs = set(envs[::3])
    train = [r for r in rows if r["env"] not in test_envs]
    test = [r for r in rows if r["env"] in test_envs]
    warmup = max(1, round(1 / dt))
    fits = {}
    for order, bounds in (("first", (0.01, 5)), ("second", (0.5, 40))):
        def cost(log_parameter):
            parameter = np.exp(log_parameter)
            return sum(np.sum((r["y"][warmup:] - predict(
                r["u"], r["y"][0], dt, parameter, order)[warmup:]) ** 2) for r in train)
        grid = np.linspace(*np.log(bounds), 32)
        best = int(np.argmin([cost(x) for x in grid]))
        low, high = grid[max(0, best - 1)], grid[min(len(grid) - 1, best + 1)]
        parameter = float(np.exp(optimize.minimize_scalar(
            cost, bounds=(low, high), method="bounded").x))
        fits[order] = {"parameter": parameter,
                       "parameter_name": "tau_s" if order == "first" else "omega_n_rad_s"}
    omega = fits["second"]["parameter"]
    designs, residuals = [], []
    for row in train:
        mask = row["moving"].copy()
        mask[:warmup] = False
        designs.append(harmonics(row["phase"])[mask])
        residuals.append((row["y"] - predict(row["u"], row["y"][0], dt, omega, "second"))[mask])
    design = np.concatenate(designs)
    coefficients = (np.linalg.lstsq(design, np.concatenate(residuals), rcond=None)[0]
                    if len(design) else np.zeros(6))
    fits["second+residual"] = dict(fits["second"], fourier_coefficients=coefficients.tolist(),
                                    harmonic_design_rank=int(np.linalg.matrix_rank(design)) if len(design) else 0,
                                    residual_status="fitted" if len(design) else "unavailable: no moving training samples")
    for order, fit in fits.items():
        errors, energy, centered_energy = [], [], []
        for row in test:
            prediction = predict(row["u"], row["y"][0], dt, fit["parameter"],
                                 "first" if order == "first" else "second")
            if order == "second+residual":
                prediction += (harmonics(row["phase"]) @ coefficients) * row["moving"]
            y = row["y"][warmup:]
            errors.extend((y - prediction[warmup:]) ** 2)
            energy.extend(y ** 2)
            centered_energy.extend((y - y.mean()) ** 2)
        fit.update(rmse=float(np.sqrt(np.mean(errors))),
                   normalized_residual_energy=float(np.sum(errors) / max(np.sum(energy), 1e-15)),
                   centered_normalized_residual_energy=float(np.sum(errors) / max(np.sum(centered_energy), 1e-15)))
    frequency = np.linspace(0.1, 2, 80)
    gains, phases = [], []
    for row in test:
        u, y = row["u"] - row["u"].mean(), row["y"] - row["y"].mean()
        U, Y = np.fft.rfft(u), np.fft.rfft(y)
        f = np.fft.rfftfreq(len(u), dt)
        mask = (f >= 0.1) & (f <= 2) & (np.abs(U) > 0.02 * np.max(np.abs(U)))
        if mask.sum() < 3:
            continue
        H = Y[mask] / U[mask]
        gains.append(np.interp(frequency, f[mask], 20 * np.log10(np.maximum(abs(H), 1e-12)), left=np.nan, right=np.nan))
        phases.append(np.interp(frequency, f[mask], np.unwrap(np.angle(H)) * 180 / np.pi, left=np.nan, right=np.nan))
    bode = {"frequency_hz": frequency.tolist()}
    for key, values in (("gain_db", gains), ("phase_deg", phases)):
        bode[key] = np.nanpercentile(values, [10, 50, 90], axis=0).tolist() if values else []
    return dict(status="ok", train_segments=len(train), test_segments=len(test),
                train_envs=sorted(set(r["env"] for r in train)), test_envs=sorted(test_envs),
                models=fits, bode=bode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--output", default="data/identification/analysis")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for path in map(Path, args.datasets):
        meta = json.loads(path.with_suffix(".json").read_text())
        with np.load(path) as archive:
            # NpzFile decompresses on every __getitem__; materialize once.
            data = {key: archive[key] for key in (
                "is_identification", "plan_generation", "reset", "excitation_signal",
                "excitation_channel", "command", "measured", "gait_phase", "is_standing"
            )}
            result = {"dataset": str(path), "policy": meta["policy"], "channels": {}}
            for c, name in enumerate(CHANNELS):
                result["channels"][name] = fit_channel(records(data, c, meta["dt_s"]), meta["dt_s"])
                print(path.stem, name, result["channels"][name]["status"], flush=True)
        summaries[path.stem] = result
        (out / (path.stem + "_fits.json")).write_text(json.dumps(result, indent=2))
    fig, axes = plt.subplots(1, 5, figsize=(19, 4))
    bode_fig, bode_axes = plt.subplots(2, 5, figsize=(19, 7), sharex="col")
    for name, result in summaries.items():
        for c, channel in enumerate(CHANNELS):
            row = result["channels"][channel]
            if row["status"] != "ok":
                continue
            axes[c].plot(range(3), [row["models"][m]["normalized_residual_energy"] for m in
                                     ("first", "second", "second+residual")], "o-", label=name)
            for axis, key in zip(bode_axes[:, c], ("gain_db", "phase_deg")):
                if row["bode"][key]:
                    low, mid, high = row["bode"][key]
                    line, = axis.plot(row["bode"]["frequency_hz"], mid, label=name)
                    axis.fill_between(row["bode"]["frequency_hz"], low, high, color=line.get_color(), alpha=.10)
    for c, channel in enumerate(CHANNELS):
        axes[c].set(title=channel, yscale="log", xticks=range(3), xticklabels=["1st", "2nd", "+gait"])
        axes[c].grid(alpha=.2)
        bode_axes[0, c].set(title=channel, ylabel="Gain (dB)")
        bode_axes[1, c].set(xlabel="Frequency (Hz)", ylabel="Phase (deg)")
    axes[0].set_ylabel("Held-out normalized residual energy")
    axes[-1].legend(fontsize=5)
    fig.tight_layout()
    bode_fig.tight_layout()
    fig.savefig(out / "model_order.png", dpi=180)
    bode_fig.savefig(out / "bode.png", dpi=180)
    (out / "summary.json").write_text(json.dumps(summaries, indent=2))
    (out / "README.md").write_text(
        "# Closed-loop identification\n\n"
        "First-order lag and critically damped second-order models have fixed unit DC gain. "
        "The third model adds three gait harmonics learned only on training environments. "
        "Chirp records are split by plan_generation and reset; segments shorter than 8 s are excluded. "
        "Every third environment is held out. The first second initializes each record and is not scored. "
        "Energy is normalized by raw measured signal energy; centered-energy ratios are also in JSON. "
        "Height is the offset from the configured nominal height, as recorded by _response_measured.\n\n"
        "Bode bands are empirical 10/50/90 percentiles over held-out chirp segments, not confidence intervals. "
        "The datasets restore each policy's training domain configuration, so these are not matched-domain "
        "reward ablations. Sampling is after env.step; sub-step timing and command replans limit delay inference. "
        "These fits do not establish a feasibility envelope, rough-terrain success, push recovery, "
        "or the complete three-domain R9 acceptance suite.\n")


if __name__ == "__main__":
    main()
