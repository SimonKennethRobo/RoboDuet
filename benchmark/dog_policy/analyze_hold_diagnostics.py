"""Analyze paired hold experiments without importing IsaacGym or the policy."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def rms(x, axis=0):
    return np.sqrt(np.mean(np.asarray(x) ** 2, axis=axis))


def analyze_trial(path):
    manifest = json.loads((path / "manifest.json").read_text())
    metrics = json.loads((path / "metrics.json").read_text())
    initial = json.loads((path / "initial_state.json").read_text())
    solves = json.loads((path / "solver.json").read_text())
    with np.load(path / "trajectory.npz", allow_pickle=False) as f:
        a = f["values"].copy()
        measured_velocity = f["arm_measured_velocity"].copy()
    result = {"name": path.name, "manifest": manifest, "metrics": metrics,
              "initial": initial, "segments": [], "one_step": {}}
    if not len(a):
        return result
    state = a[:, 4:18]
    q = state[:, 8:14]
    u = a[:, 33:44]
    angular_error = (Rotation.from_quat(a[:, 29:33]) *
                     Rotation.from_quat(a[:, 22:26]).inv()).as_rotvec()
    low, high = np.array(manifest["input_lower"]), np.array(manifest["input_upper"])
    qlow, qhigh = np.array(manifest["arm_position_lower"]), np.array(manifest["arm_position_upper"])
    saturation = (u <= low + 1e-4) | (u >= high - 1e-4)
    for start, end in ((0, 24), (0, 6), (6, 12), (12, 18), (18, 24)):
        mask = (a[:, 0] >= start) & (a[:, 0] < end)
        if not mask.any():
            continue
        result["segments"].append({
            "interval_s": [start, end], "samples": int(mask.sum()),
            "position_rmse_m": float(rms(a[mask, 1])),
            "orientation_rmse_deg": float(np.rad2deg(rms(a[mask, 2]))),
            "orientation_rotvec_xyz_rmse_deg": np.rad2deg(rms(angular_error[mask])).tolist(),
            "orientation_rotvec_xyz_mean_deg": np.rad2deg(angular_error[mask].mean(axis=0)).tolist(),
            "base_pitch_roll_mean_deg": np.rad2deg(np.c_[state[mask, 4], a[mask, 18]].mean(axis=0)).tolist(),
            "base_pitch_roll_std_deg": np.rad2deg(np.c_[state[mask, 4], a[mask, 18]].std(axis=0)).tolist(),
            "command_saturation_fraction": saturation[mask].mean(axis=0).tolist(),
            "command_mean": u[mask].mean(axis=0).tolist(),
            "arm_target_tracking_rmse_deg": np.rad2deg(rms(a[mask, 44:50] - q[mask])).tolist(),
            "arm_position_min_rad": q[mask].min(axis=0).tolist(),
            "arm_position_max_rad": q[mask].max(axis=0).tolist(),
            "arm_position_boundary_fraction": ((q[mask] <= qlow + 0.02) |
                                                 (q[mask] >= qhigh - 0.02)).mean(axis=0).tolist(),
            "arm_actual_velocity_rms_rad_s": rms(measured_velocity[mask]).tolist(),
        })
    commanded, realized, arm_errors, base_errors = [], [], [], []
    ee_errors, kinematic_errors, arm_angular_errors = [], [], []
    period = manifest["mpc_period_s"]
    for d in solves:
        if "predicted_state" not in d or not d["ok"]:
            continue
        target_time = d["time"] + period
        index = int(np.searchsorted(a[:, 0], target_time - 1e-6))
        if index >= len(a) or abs(a[index, 0] - target_time) > 1e-5 or a[index, 3]:
            continue
        before, predicted, actual = np.array(d["state"]), np.array(d["predicted_state"]), state[index]
        command, jac = np.array(d["command"]), np.array(d["arm_jacobian"])
        commanded.append(command[5:])
        realized.append((actual[8:14] - before[8:14]) / period)
        arm_errors.append(actual[8:14] - predicted[8:14])
        base_errors.append(actual[[2, 4, 5, 6, 7]] - predicted[[2, 4, 5, 6, 7]])
        actual_rotation = (Rotation.from_quat(a[index, 22:26]) *
                           Rotation.from_quat(d["ee_quaternion_xyzw"]).inv()).as_rotvec()
        ee_errors.append(actual_rotation - np.array(d["predicted_ee_delta"])[3:])
        delta = actual - before
        delta[3] = (delta[3] + np.pi) % (2 * np.pi) - np.pi
        yaw, pitch = before[3:5]
        pitch_axis = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        roll_axis = np.array([np.cos(yaw) * np.cos(pitch),
                              np.sin(yaw) * np.cos(pitch), -np.sin(pitch)])
        reconstructed = (np.array([0.0, 0.0, delta[3]]) + pitch_axis * delta[4] +
                         roll_axis * (a[index, 18] - d["roll"]) + jac[3:] @ delta[8:14])
        kinematic_errors.append(actual_rotation - reconstructed)
        arm_angular_errors.append(jac[3:] @ (actual[8:14] - predicted[8:14]))
    if commanded:
        commanded, realized = np.asarray(commanded), np.asarray(realized)
        active = np.abs(commanded) > 0.05
        denominator = np.sum(np.where(active, commanded ** 2, 0.0), axis=0)
        effective_gain = np.divide(np.sum(np.where(active, commanded * realized, 0.0), axis=0),
                                   denominator, out=np.zeros(6), where=denominator > 1e-9)
        result["one_step"] = {
            "intervals": len(commanded),
            "arm_velocity_effective_gain_diagnostic_only": effective_gain.tolist(),
            "arm_position_prediction_rmse_deg": np.rad2deg(rms(arm_errors)).tolist(),
            "base_prediction_rmse_z_pitch_vx_vy_wz": rms(base_errors).tolist(),
            "ee_angular_prediction_rmse_deg": float(np.rad2deg(rms(np.linalg.norm(ee_errors, axis=1)))),
            "ee_angular_reconstruction_using_actual_motion_rmse_deg":
                float(np.rad2deg(rms(np.linalg.norm(kinematic_errors, axis=1)))),
            "arm_mismatch_angular_projection_rmse_deg":
                float(np.rad2deg(rms(np.linalg.norm(arm_angular_errors, axis=1)))),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    trials = {p.parent.name: analyze_trial(p.parent) for p in sorted(args.root.glob("*/metrics.json"))}
    comparisons = []
    specifications = [
        ("effect_of_push_ideal", "baseline_ideal_no_push", "baseline_ideal_push"),
        ("effect_of_push_identified", "baseline_identified_no_push", "baseline_identified_push"),
        ("effect_of_model_without_push", "baseline_ideal_no_push", "baseline_identified_no_push"),
        ("effect_of_model_with_push", "baseline_ideal_push", "baseline_identified_push"),
    ]
    for intervention in ("pose_envelope", "orientation90"):
        for push in ("push", "no_push"):
            specifications.append(("effect_of_" + intervention + "_" + push,
                                   "baseline_identified_" + push, intervention + "_identified_" + push))
    for name, first, second in specifications:
        if first not in trials or second not in trials:
            continue
        x, y = trials[first], trials[second]
        initial_delta = np.array(x["initial"]["state"]) - np.array(y["initial"]["state"])
        pair = {"name": name, "baseline": first, "intervention": second,
                "initial_state_max_abs_difference": float(np.max(np.abs(initial_delta))),
                "initial_ee_position_difference_m": float(np.linalg.norm(
                    np.array(x["initial"]["ee_position"]) - y["initial"]["ee_position"])),
                "both_full_duration": x["metrics"]["completed"] and y["metrics"]["completed"],
                "orientation_change_deg": None, "orientation_relative_change": None}
        if pair["both_full_duration"]:
            old, new = x["segments"][0]["orientation_rmse_deg"], y["segments"][0]["orientation_rmse_deg"]
            pair["orientation_change_deg"] = new - old
            pair["orientation_relative_change"] = new / max(old, 1e-12) - 1
        comparisons.append(pair)
    payload = {"scope": "Seed-29 diagnostics of the existing local-linear MPC prototype, not OCS2",
               "trials": trials, "comparisons": comparisons}
    (args.root / "diagnosis.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = ["# Hold-pose mechanism diagnostics", "",
             payload["scope"], "",
             "Each intervention changes only push amplitude, pose command bounds, or orientation cost weight.",
             "No controller-interface or arm-actuation correction was made.", "",
             "| Trial | Full duration | Position RMSE (m) | Orientation RMSE (deg) |",
             "|---|---|---:|---:|"]
    for name, trial in trials.items():
        segment = trial["segments"][0] if trial["segments"] else {}
        lines.append("| {} | {} | {} | {} |".format(name, trial["metrics"]["completed"],
                     segment.get("position_rmse_m", "n/a"), segment.get("orientation_rmse_deg", "n/a")))
    lines += ["", "Stopped-run errors cover only their recorded durations; do not compare them to full runs.", "",
              "## Paired interventions", "",
              "| Comparison | Initial state max difference | Both full duration | Orientation change (deg) |",
              "|---|---:|---|---:|"]
    for pair in comparisons:
        lines.append("| {} | {:.3g} | {} | {} |".format(pair["name"],
                     pair["initial_state_max_abs_difference"], pair["both_full_duration"],
                     pair["orientation_change_deg"]))
    lines += ["", "Detailed segments, rotation-vector components, saturation, arm tracking and one-step errors are in diagnosis.json.",
              "Effective arm gains are closed-loop descriptive ratios, not validated system-identification models.",
              "Angular mismatch projections are local kinematic diagnostics, not additive causal error budgets."]
    (args.root / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"root": str(args.root), "trials": len(trials), "comparisons": comparisons}, indent=2), flush=True)


if __name__ == "__main__":
    main()
