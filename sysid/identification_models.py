"""Low-order closed-loop response models for identification and evaluation."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

DT = .02
HORIZONS_S = (.1, .3, .6, 1.0)
HORIZONS = tuple(round(h / DT) for h in HORIZONS_S)
DELAY_STEPS = (0, 1, 2, 3, 5)


def transition(model, dt=DT):
    """Return the exact ZOH state and scalar-target matrices."""
    if model["order"] == "first":
        a = math.exp(-dt / model["tau_s"])
        return np.array([[a]]), np.array([1. - a])
    if model["order"] == "second":
        w = model["natural_frequency_rad_s"]
        zeta = model.get("damping_ratio", 1.)
        if abs(zeta - 1.) > 1e-12:
            raise ValueError("Only critically damped second-order models are deployable")
        s, e = w * dt, math.exp(-w * dt)
        ad = e * np.array([[1. + s, dt], [-w*w*dt, 1. - s]])
        return ad, np.array([1. - ad[0, 0], -ad[1, 0]])
    raise ValueError(f"Unsupported response order: {model['order']}")


def causal_rate(values, start, dt=DT, samples=5):
    begin = max(0, start - samples + 1)
    y = np.asarray(values[begin:start+1], dtype=float)
    if len(y) < 2:
        return 0.
    # Closed-form least-squares slope on a uniform grid. This is called in the
    # inner parameter-search loop, where np.polyfit's repeated SVD dominates a
    # full 72-episode fit by orders of magnitude.
    index = np.arange(len(y), dtype=float)
    centered = index-index.mean()
    return float(centered@y / (dt*(centered@centered)))


def response_weights(order, parameter, horizon, dt=DT):
    template = ({"order": "first", "tau_s": parameter} if order == "first" else
                {"order": "second", "natural_frequency_rad_s": parameter, "damping_ratio": 1.})
    ad, bd = transition(template, dt)
    c = np.zeros(ad.shape[0]); c[0] = 1.
    initial = c @ np.linalg.matrix_power(ad, horizon)
    weights = np.array([c @ np.linalg.matrix_power(ad, horizon-1-j) @ bd
                        for j in range(horizon)])
    return initial, weights


def regression_groups(episodes, channel, order, parameter, delay_steps):
    groups = []
    for _, data in episodes:
        y, u = np.asarray(data["y"][:, channel]), np.asarray(data["u"][:, channel])
        if len(y) <= max(HORIZONS) + 100 or not np.isfinite(y).all() or not np.isfinite(u).all():
            continue
        for horizon in HORIZONS:
            starts = np.arange(100, len(y)-horizon, 10)
            initial, weights = response_weights(order, parameter, horizon)
            indices = starts[:, None] + np.arange(horizon)[None, :] - delay_steps
            indices = np.maximum(indices, 0)
            command = u[indices] @ weights
            bias = np.full(len(starts), weights.sum())
            base = initial[0] * y[starts]
            if order == "second":
                rates = np.array([causal_rate(y, int(start)) for start in starts])
                base += initial[1] * rates
            groups.append((np.column_stack([command, bias]), y[starts+horizon]-base))
    if not groups:
        raise ValueError("No complete finite trajectories available for fitting")
    return groups


def candidate_loss(episodes, channel, order, parameter, delay_steps, full=False):
    groups = regression_groups(episodes, channel, order, parameter, delay_steps)
    design = np.concatenate([x / math.sqrt(len(target)) for x, target in groups])
    target = np.concatenate([target / math.sqrt(len(target)) for x, target in groups])
    gain_bias = np.linalg.lstsq(design, target, rcond=None)[0]
    group_mse = np.array([np.mean((x @ gain_bias - target)**2) for x, target in groups])
    loss = float(group_mse.mean())
    return (loss, gain_bias, group_mse) if full else loss


def fit_channel(episodes, channel, order, delays=DELAY_STEPS):
    bounds = (.025, 3.) if order == "first" else (.5, 40.)
    grid = np.geomspace(*bounds, 28)
    best = None
    for delay in delays:
        values = [candidate_loss(episodes, channel, order, p, delay) for p in grid]
        index = int(np.argmin(values))
        lo, hi = grid[max(0, index-1)], grid[min(len(grid)-1, index+1)]
        optimum = minimize_scalar(lambda logp: candidate_loss(
            episodes, channel, order, math.exp(logp), delay),
            bounds=(math.log(lo), math.log(hi)), method="bounded")
        parameter = min((grid[index], math.exp(optimum.x)),
                        key=lambda p: candidate_loss(episodes, channel, order, p, delay))
        loss, (gain, bias), group_mse = candidate_loss(
            episodes, channel, order, parameter, delay, full=True)
        result = dict(order=order, gain=float(gain), bias=float(bias),
                      delay_s=float(delay*DT), delay_steps=int(delay),
                      fit_rmse=float(math.sqrt(loss)),
                      trajectory_horizon_rmse=np.sqrt(group_mse).tolist(),
                      objective="equal trajectory and horizon endpoint MSE",
                      horizons_s=list(HORIZONS_S), window_start_s=2., window_stride_s=.2)
        if order == "first":
            result["tau_s"] = float(parameter)
            result["parameter_at_bound"] = bool(parameter < bounds[0]*1.01 or parameter > bounds[1]/1.01)
        else:
            result.update(natural_frequency_rad_s=float(parameter), damping_ratio=1.,
                          parameter_at_bound=bool(parameter < bounds[0]*1.01 or parameter > bounds[1]/1.01))
        if best is None or loss < best[0]:
            best = loss, result
    return best[1]


def residual_value(model, velocity, phase, walking):
    if not walking or "residual" not in model:
        return 0.
    residual = model["residual"]
    speed = math.sqrt(float(velocity[0]**2 + velocity[1]**2) + 1e-6)
    return ((residual["amplitude"] + residual["speed_amplitude"] * speed) *
            math.sin(residual["harmonic"] * phase + residual["phase_offset"]))


def predict_window(models, data, start, horizon, frequency, stop_at_stand=True):
    """Causal forecast: one measured initial state, past rate, future commands."""
    command = np.asarray(data["u"])
    actual = np.asarray(data["y"])
    phase = float(data["phase"][start])
    state = []
    for channel, model in enumerate(models):
        walking = not stop_at_stand or np.linalg.norm(command[max(0, start-1), :3]) >= .1
        y0 = actual[start, channel] - residual_value(model, actual[start, :2], phase, walking)
        if model["order"] == "first":
            state.append(np.array([y0]))
        else:
            state.append(np.array([y0, causal_rate(actual[:, channel], start)]))
    result = np.empty((horizon+1, len(models)))
    result[0] = actual[start]
    for k in range(horizon):
        walking = not stop_at_stand or np.linalg.norm(command[start+k, :3]) >= .1
        for channel, model in enumerate(models):
            delay = int(model.get("delay_steps", round(model.get("delay_s", 0.) / DT)))
            index = max(0, start+k-delay)
            target = model["gain"] * command[index, channel] + model["bias"]
            ad, bd = transition(model)
            state[channel] = ad @ state[channel] + bd * target
        phase += 2*np.pi*frequency*DT*walking
        for channel, model in enumerate(models):
            result[k+1, channel] = state[channel][0] + residual_value(
                model, np.array([state[0][0], state[1][0]]), phase, walking)
    return result


def quality_metrics(model_bundle, episodes):
    errors = {h: [] for h in HORIZONS}
    per_episode = []
    for info, data in episodes:
        episode_errors = {h: [] for h in HORIZONS}
        for start in range(100, len(data["t"])-max(HORIZONS), 10):
            prediction = predict_window(model_bundle, data, start, max(HORIZONS),
                info.get("gait_frequency_hz", 2.75), info.get("stop_gait_at_stand", True))
            for horizon in HORIZONS:
                error = prediction[horizon] - data["y"][start+horizon]
                errors[horizon].append(error); episode_errors[horizon].append(error)
        per_episode.append(dict(id=info["id"], success=info["success"],
            rmse={str(HORIZONS_S[HORIZONS.index(h)]): np.sqrt(np.mean(np.square(episode_errors[h]), axis=0)).tolist()
                  for h in HORIZONS if episode_errors[h]}))
    summary = {}
    for horizon, values in errors.items():
        error = np.asarray(values)
        summary[str(HORIZONS_S[HORIZONS.index(horizon)])] = dict(
            rmse=np.sqrt(np.mean(error**2, axis=0)).tolist(),
            mae=np.mean(np.abs(error), axis=0).tolist(), bias=np.mean(error, axis=0).tolist())
    return dict(summary=summary, per_episode=per_episode)


def bundle_from_dict(value, channels):
    return [value[name] for name in channels]
