"""Paired, measured IsaacGym closed-loop MPC screening (not the OCS2 dummy)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import xml.etree.ElementTree as ET

import isaacgym  # noqa: F401 -- before torch
from isaacgym import gymapi, gymtorch
import numpy as np
from scipy.optimize import LinearConstraint, minimize
from scipy.spatial.transform import Rotation
import torch

from benchmark.dog_policy.evaluation import (
    _apply_benchmark_env_overrides, _load_cfg_from_pkl,
    configure_stage1, load_dog_policy_for_benchmark,
)
from go1_gym.envs.config import cfg_to_dict, configure_privileged_obs_dims
from go1_gym.envs.roboduet.wbc_env import WBCEnv
from go1_gym.envs.roboduet.wbc_env_wrapper import HistoryWrapper


VERSION = "isaacgym-local-linear-response-mpc-v1"
CHANNELS = ("vx", "vy", "wyaw", "height", "pitch")
TASKS = ("reach", "line", "circle", "hold_push")


class MPCEnv(WBCEnv):
    def _arm_decimation_hook(self):
        # No training-generated arm/payload disturbances. This experiment owns
        # a prescribed world-frame force schedule, repeated every physics step.
        forces = getattr(self, "mpc_forces", None)
        if forces is not None:
            self.gym.apply_rigid_body_force_tensors(
                self.sim, gymtorch.unwrap_tensor(forces), None, gymapi.ENV_SPACE,
            )


def target_at(task, t, anchor, initial_rotation):
    t = max(float(t), 0.0)
    yaw = 0.0
    if task == "reach":
        a = min(t / 3.0, 1.0)
        a = a * a * (3.0 - 2.0 * a)
        offset = a * np.array([0.65, 0.15, 0.08])
    elif task == "line":
        offset = np.array([0.06 * t, 0.0, 0.03 * np.sin(2.0 * np.pi * t / 8.0)])
    elif task == "circle":
        angle = 2.0 * np.pi * t / 12.0
        offset = np.array([0.35 * np.sin(angle) + 0.02 * t,
                           0.35 * (1.0 - np.cos(angle)), 0.03 * np.sin(angle)])
        yaw = 0.15 * np.sin(angle)
    else:
        offset = np.zeros(3)
    return anchor + offset, Rotation.from_rotvec([0.0, 0.0, yaw]) * initial_rotation


def lag_coefficients(model, period):
    """Exact first-order ZOH, including a delay shorter than one MPC interval.

    Return coefficients for final response and integrated response, respectively,
    multiplying (current response, current command, previous command, constant).
    """
    tau, gain, bias = model["parameter"], model["gain"], model["bias"]
    delay = model["delay_s"]
    if tau <= 0 or not 0 <= delay <= period:
        raise ValueError("This controller requires 0 <= fitted delay <= MPC period and tau > 0")
    a = np.exp(-period / tau)
    a_new = np.exp(-(period - delay) / tau)
    final = np.array([a, gain * (1.0 - a_new), gain * (a_new - a), bias * (1.0 - a)])
    integral = np.array([tau * (1.0 - a),
                         gain * (period - delay - tau * (1.0 - a_new)),
                         gain * (delay - tau * (a_new - a)),
                         bias * (period - tau * (1.0 - a))])
    return final, integral


class ResponseMPC:
    """Receding-horizon convex QP with measured, frozen local EE kinematics.

    State: world x/y/z/yaw/pitch, body vx/vy/wz, six arm q, previous 11 inputs.
    Inputs: vx/vy/wz/height-offset/pitch followed by six arm joint velocities.
    Roll is not an input because this checkpoint was trained with roll fixed.
    """

    def __init__(self, mode, models, q_low, q_high, command_low, command_high,
                 slew, nominal_height, period=0.1, horizon=10,
                 orientation_weight=30.0, diagnostics=False):
        self.mode, self.models = mode, models
        self.orientation_weight, self.diagnostics = orientation_weight, diagnostics
        self.dt, self.horizon = period, horizon
        self.nx, self.nu = 25, 11
        self.q_low, self.q_high = q_low, q_high
        self.height = nominal_height
        arm_speed = np.array([0.8, 0.8, 0.8, 1.2, 1.2, 1.2])
        self.low = np.r_[command_low, -arm_speed]
        self.high = np.r_[command_high, arm_speed]
        self.slew = np.r_[slew * period, 2.0 * arm_speed]
        self.plan = np.zeros((horizon, self.nu))
        self.previous = np.zeros(self.nu)
        self.difference = np.eye(horizon * self.nu)
        self.difference[self.nu:, :-self.nu] -= np.eye((horizon - 1) * self.nu)
        self.bounds = list(zip(np.tile(self.low, horizon), np.tile(self.high, horizon)))

    def dynamics(self, rotation, pitch, roll):
        A, B, c = np.eye(self.nx), np.zeros((self.nx, self.nu)), np.zeros(self.nx)
        velocity_integrals = []
        for i, channel in enumerate(CHANNELS):
            state_index = 5 + i if i < 3 else (2 if i == 3 else 4)
            if self.mode == "ideal":
                final = np.array([0.0, 1.0, 0.0, 0.0])
                integral = np.array([0.0, self.dt, 0.0, 0.0])
            else:
                final, integral = lag_coefficients(self.models[channel], self.dt)
            A[state_index] = 0.0
            A[state_index, state_index] = final[0]
            A[state_index, 14 + i] = final[2]
            B[state_index, i] = final[1]
            c[state_index] = final[3]
            if i == 3:
                c[state_index] += (1.0 - final[0]) * self.height
            if i < 3:
                velocity_integrals.append(integral)

        # Translate measured body-frame velocity through the full measured
        # orientation, not through a silently substituted heading frame.
        for world_axis in (0, 1):
            for i in (0, 1):
                integral = velocity_integrals[i] * rotation[world_axis, i]
                A[world_axis, 5 + i] += integral[0]
                B[world_axis, i] += integral[1]
                A[world_axis, 14 + i] += integral[2]
                c[world_axis] += integral[3]
        denominator = np.cos(pitch) * np.cos(roll)
        if abs(denominator) < 0.2:
            raise ValueError("Base tilt exceeds local Euler-model operating region")
        integral = velocity_integrals[2] / denominator
        A[3, 7] += integral[0]
        B[3, 2] += integral[1]
        A[3, 16] += integral[2]
        c[3] += integral[3]
        coupling = np.sin(roll) / denominator
        A[3] += coupling * A[4]
        A[3, 4] -= coupling
        B[3] += coupling * B[4]
        c[3] += coupling * c[4]
        B[8:14, 5:11] = np.eye(6) * self.dt
        A[14:] = 0.0
        B[14:] = np.eye(self.nu)
        return A, B, c

    def solve(self, state, rotation, roll, ee_position, ee_rotation, arm_jacobian,
              task, now, anchor, target_rotation):
        began = time.perf_counter()
        A, B, c = self.dynamics(rotation, state[4], roll)
        C = np.zeros((6, self.nx))
        C[:3, :3] = np.eye(3)
        yaw_axis = np.array([0.0, 0.0, 1.0])
        pitch_axis = np.array([-np.sin(state[3]), np.cos(state[3]), 0.0])
        lever = ee_position - state[:3]
        for column, axis in ((3, yaw_axis), (4, pitch_axis)):
            C[:3, column] = np.cross(axis, lever)
            C[3:, column] = axis
        C[:, 8:14] = arm_jacobian
        baseline = np.r_[state, self.previous]
        predicted = baseline.copy()
        influence = np.zeros((self.nx, self.horizon * self.nu))
        matrices, residuals, q_matrices, q_low, q_high = [], [], [], [], []
        for k in range(self.horizon):
            predicted = A @ predicted + c
            influence = A @ influence
            influence[:, k * self.nu:(k + 1) * self.nu] += B
            desired_position, desired_rotation = target_at(
                task, now + (k + 1) * self.dt, anchor, target_rotation,
            )
            desired = np.r_[desired_position - ee_position,
                            (desired_rotation * ee_rotation.inv()).as_rotvec()]
            weight = np.sqrt(np.r_[np.full(3, 200.0), np.full(3, self.orientation_weight)])
            if k == self.horizon - 1:
                weight *= np.sqrt(3.0)
            matrices.append((C @ influence) * weight[:, None])
            residuals.append((C @ (predicted - baseline) - desired) * weight)
            q_matrices.append(influence[8:14])
            q_low.extend(self.q_low - predicted[8:14])
            q_high.extend(self.q_high - predicted[8:14])
        M, residual = np.concatenate(matrices), np.concatenate(residuals)
        r = np.tile(np.r_[1.0, 1.5, 0.4, 0.01, 0.05, np.full(6, 0.02)], self.horizon)
        dr = np.tile(np.r_[0.3, 0.3, 0.1, 5.0, 2.0, np.full(6, 0.01)], self.horizon)
        difference_target = np.zeros(self.horizon * self.nu)
        difference_target[:self.nu] = self.previous
        D = self.difference
        H = M.T @ M + np.diag(r) + D.T @ (dr[:, None] * D)
        g = M.T @ residual - D.T @ (dr * difference_target)
        constraints = np.vstack([np.concatenate(q_matrices), D])
        lower = np.r_[q_low, difference_target - np.tile(self.slew, self.horizon)]
        upper = np.r_[q_high, difference_target + np.tile(self.slew, self.horizon)]
        initial = np.vstack([self.plan[1:], self.plan[-1:]]).reshape(-1)
        result = minimize(
            lambda u: 0.5 * u @ H @ u + g @ u, initial,
            jac=lambda u: H @ u + g, method="SLSQP", bounds=self.bounds,
            constraints=[LinearConstraint(constraints, lower, upper)],
            options={"maxiter": 60, "ftol": 1e-6, "disp": False},
        )
        ok = bool(result.success and np.isfinite(result.x).all())
        if ok:
            margins = constraints @ result.x
            ok = bool(np.max(lower - margins) < 1e-5 and np.max(margins - upper) < 1e-5)
        if ok:
            self.plan = result.x.reshape(self.horizon, self.nu)
            command = self.plan[0].copy()
        else:
            # A failed solve is logged, not hidden by pretending it converged.
            command = self.previous.copy()
            command[:3] = 0.0
            command[5:] = 0.0
            self.plan[:] = command
        self.previous = command.copy()
        diagnostic = {"ok": ok, "seconds": time.perf_counter() - began,
                      "iterations": int(result.nit), "message": str(result.message)}
        if self.diagnostics:
            next_state = A @ baseline + B @ command + c
            diagnostic.update({
                "time": float(now), "state": state.tolist(), "roll": float(roll),
                "ee_position": ee_position.tolist(),
                "ee_quaternion_xyzw": ee_rotation.as_quat().tolist(),
                "command": command.tolist(), "arm_jacobian": arm_jacobian.tolist(),
                "predicted_state": next_state[:14].tolist(),
                "predicted_ee_delta": (C @ (next_state - baseline)).tolist(),
            })
        return command, diagnostic


def measured_state(base):
    root = base.root_states[0].detach().cpu().numpy().copy()
    rotation = Rotation.from_quat(root[3:7])
    yaw, pitch, roll = rotation.as_euler("ZYX")
    velocity = base.base_lin_vel[0].detach().cpu().numpy()
    yaw_rate = float(base.base_ang_vel[0, 2])
    arm_q = base.dof_pos[0, 12:18].detach().cpu().numpy().copy()
    state = np.r_[root[:3], yaw, pitch, velocity[:2], yaw_rate, arm_q]
    ee = base.end_effector_state[0].detach().cpu().numpy().copy()
    return state, rotation, roll, ee[:3], Rotation.from_quat(ee[3:7])


def run_trial(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)
    configure_stage1(0.0)
    cfg = _load_cfg_from_pkl(args.logdir)
    _apply_benchmark_env_overrides(cfg, total_envs=1, envs_per_policy=1)
    cfg.domain_rand.mode = "none"
    cfg.terrain.reset_curriculum = False
    for key in ("z_init_range", "yaw_init_range", "pitch_init_range", "roll_init_range"):
        setattr(cfg.terrain, key, 0.0)
    cfg.env.record_video = False
    cfg.env.keep_arm_fixed = False
    cfg.env.stage1_arm_curriculum = False
    cfg.env.stage1_arm_init_dof_pos_noise = 0.0
    cfg.env.episode_length_s = args.seconds + 10.0
    cfg.env.arm_policy_enabled = False
    nominal_q = np.array([0.0, 0.6, 0.6, 0.0, 0.0, 0.0])
    for i, value in enumerate(nominal_q, 1):
        cfg.init_state.default_joint_angles["x5_joint" + str(i)] = float(value)
    configure_privileged_obs_dims(cfg)
    data = json.loads(Path(args.identification).read_text())
    key = Path(args.logdir).name + "_last"
    models = {channel: data[key]["channels"][channel]["models"]["first"] for channel in CHANNELS}
    actual_asset = cfg.asset.file.format(MINI_GYM_ROOT_DIR=str(Path(__file__).resolve().parents[2]))
    joints = {j.get("name"): j for j in ET.parse(actual_asset).getroot().findall("joint")}
    q_low = np.array([float(joints["x5_joint" + str(i)].find("limit").get("lower")) for i in range(1, 7)]) + 0.03
    q_high = np.array([float(joints["x5_joint" + str(i)].find("limit").get("upper")) for i in range(1, 7)]) - 0.03
    action_span = cfg.normalization.clip_actions * cfg.control.action_scale
    q_low = np.maximum(q_low, nominal_q - action_span)
    q_high = np.minimum(q_high, nominal_q + action_span)
    limits = [cfg.commands.limit_vel_x, cfg.commands.limit_vel_y, cfg.commands.limit_vel_yaw,
              cfg.commands.limit_body_height, cfg.commands.limit_body_pitch]
    low = np.maximum(np.array(limits)[:, 0], [-0.6, -0.4, -0.8, -0.06, -0.2])
    high = np.minimum(np.array(limits)[:, 1], [0.6, 0.4, 0.8, 0.04, 0.2])
    if args.pose_envelope == "checkpoint":
        low[3:] = np.array(limits)[3:, 0]
        high[3:] = np.array(limits)[3:, 1]
    slew = np.array([cfg.response.rate_limit[ch] for ch in CHANNELS])
    controller = ResponseMPC(args.mode, models, q_low, q_high, low, high, slew,
                             cfg.rewards.base_height_target,
                             orientation_weight=args.orientation_weight, diagnostics=args.diagnostics)
    manifest = {"version": VERSION, "mode": args.mode, "task": args.task, "seed": args.seed,
                "seconds": args.seconds, "logdir": str(Path(args.logdir).resolve()),
                "identification": str(Path(args.identification).resolve()), "models": models,
                "mpc_period_s": controller.dt, "horizon_s": controller.dt * controller.horizon,
                "command_lower": low.tolist(), "command_upper": high.tolist(),
                "input_lower": controller.low.tolist(), "input_upper": controller.high.tolist(),
                "arm_position_lower": q_low.tolist(), "arm_position_upper": q_high.tolist(),
                "push_scale": args.push_scale, "pose_envelope": args.pose_envelope,
                "orientation_weight": args.orientation_weight, "diagnostics": args.diagnostics,
                "scope": "Nominal flat-ground local-linear MPC; no OCS2, terrain or identified feasibility envelope",
                "checkpoint_sha256": hashlib.sha256((Path(args.logdir) / "checkpoints_dog/ac_weights_last_dog.pt").read_bytes()).hexdigest()}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "config.json").write_text(json.dumps(cfg_to_dict(cfg), indent=2) + "\n")
    base = MPCEnv(sim_device=args.device, headless=True, cfg=cfg, graphics_device_id=-1)
    env = HistoryWrapper(base)
    records, solves, failures = [], [], []
    measured_arm_velocities, measured_base_angular_velocities = [], []
    try:
        policy = load_dog_policy_for_benchmark(args.logdir, "last", cfg, device=args.device)
        policy_steps = round(controller.dt / base.dt)
        if not np.isclose(policy_steps * base.dt, controller.dt):
            raise ValueError("MPC period must be an integer number of policy steps")
        base.mpc_forces = torch.zeros((1, base.num_bodies, 3), device=base.device)
        trunk = -1
        for body_name in ("base", "trunk"):
            trunk = base.gym.find_actor_rigid_body_handle(
                base.envs[0], base.actor_handles[0], body_name)
            if trunk >= 0:
                break
        if trunk < 0:
            raise ValueError(
                "Cannot locate base/trunk for the prescribed force schedule; "
                f"available bodies: {base.body_names}")
        gait = [np.mean(cfg.commands.limit_gait_frequency),
                np.mean(cfg.commands.limit_footswing_height),
                np.mean(cfg.commands.limit_stance_width),
                np.mean(cfg.commands.limit_stance_length),
                np.mean(cfg.commands.limit_gait_duration)]

        def apply(command, q_target):
            base.commands_dog[0, :3] = torch.as_tensor(command[:3], device=base.device, dtype=torch.float)
            base.commands_dog[0, 3] = float(command[4])
            base.commands_dog[0, 4] = 0.0
            base.commands_dog[0, 5] = float(command[3])
            base.commands_dog[0, 6:11] = torch.as_tensor(gait, device=base.device, dtype=torch.float)
            # Stage 1's canonical wrapper deliberately uses arm_fake_actions.
            # Here it carries external MPC targets, with the training arm
            # curriculum and kinematic arm locking explicitly disabled above.
            env.arm_fake_actions[0] = torch.as_tensor(
                (q_target - nominal_q) / cfg.control.action_scale,
                device=base.device, dtype=torch.float,
            )
            with torch.no_grad():
                dog_action = policy(env.get_dog_observations())
                return env.step(dog_action, env.arm_fake_actions)[2]

        env.reset()
        warmup_falls = 0
        for _ in range(round(3.0 / base.dt)):
            warmup_falls += int(apply(np.zeros(11), nominal_q)[0])
        state, rotation, roll, anchor, target_rotation = measured_state(base)
        initial = {"state": state.tolist(), "ee_position": anchor.tolist(),
                   "ee_quaternion_xyzw": target_rotation.as_quat().tolist(), "warmup_falls": warmup_falls}
        (output / "initial_state.json").write_text(json.dumps(initial, indent=2) + "\n")
        if warmup_falls:
            failures.append("warmup_reset")
        command = np.zeros(11)
        q_target = state[8:14].copy()
        consecutive_solver_failures = 0
        for step in range(round(args.seconds / base.dt)):
            if failures:
                break
            now = step * base.dt
            if step % policy_steps == 0:
                state, rotation, roll, ee_position, ee_rotation = measured_state(base)
                base.gym.refresh_jacobian_tensors(base.sim)
                jac, _ = base._arm_jacobian()
                jac = jac[0].detach().cpu().numpy().astype(float).copy()
                offset = ee_rotation.apply(base.ee_local_offset[0].detach().cpu().numpy())
                jac[:3] += np.cross(jac[3:].T, offset).T
                command, diagnostic = controller.solve(
                    state, rotation.as_matrix(), roll, ee_position, ee_rotation, jac,
                    args.task, now, anchor, target_rotation,
                )
                solves.append(diagnostic)
                consecutive_solver_failures = 0 if diagnostic["ok"] else consecutive_solver_failures + 1
                q_target = np.clip(state[8:14], q_low, q_high)
                if consecutive_solver_failures >= 10:
                    failures.append("ten_consecutive_solver_failures")
                    break
            base.mpc_forces.zero_()
            if args.task == "hold_push":
                for event, event_time in enumerate((6.0, 12.0, 18.0)):
                    if event_time <= now < event_time + 0.25:
                        base.mpc_forces[0, trunk, 1] = args.push_scale * (20.0 if event % 2 == 0 else -20.0)
            q_target = np.clip(q_target + command[5:] * base.dt, q_low, q_high)
            done = bool(apply(command, q_target)[0])
            next_time = (step + 1) * base.dt
            state, rotation, roll, ee_position, ee_rotation = measured_state(base)
            desired_position, desired_rotation = target_at(args.task, next_time, anchor, target_rotation)
            position_error = float(np.linalg.norm(ee_position - desired_position))
            rotation_error = float((desired_rotation * ee_rotation.inv()).magnitude())
            records.append(np.r_[next_time, position_error, rotation_error, float(done),
                                 state, roll, ee_position, ee_rotation.as_quat(), desired_position,
                                 desired_rotation.as_quat(), command, q_target])
            if args.diagnostics:
                measured_arm_velocities.append(base.dof_vel[0, 12:18].detach().cpu().numpy().copy())
                measured_base_angular_velocities.append(base.base_ang_vel[0].detach().cpu().numpy().copy())
            if done:
                failures.append("environment_reset")
            elif not np.isfinite(records[-1]).all():
                failures.append("nonfinite_state")
            if step % 250 == 0:
                print("PROGRESS mode={} task={} seed={} t={:.2f}s ee={:.4f}m rot={:.4f}rad solve={:.3f}s".format(
                    args.mode, args.task, args.seed, next_time, position_error, rotation_error, solves[-1]["seconds"]), flush=True)
        array = np.asarray(records)
        extra = {}
        if args.diagnostics:
            extra = {"arm_measured_velocity": np.asarray(measured_arm_velocities),
                     "base_angular_velocity": np.asarray(measured_base_angular_velocities)}
        np.savez_compressed(output / "trajectory.npz", values=array, **extra,
                            columns=np.array(["time", "ee_position_error", "ee_rotation_error", "reset"] +
                                ["state_" + str(i) for i in range(14)] + ["roll"] +
                                ["ee_p_" + str(i) for i in range(3)] + ["ee_q_" + str(i) for i in range(4)] +
                                ["target_p_" + str(i) for i in range(3)] + ["target_q_" + str(i) for i in range(4)] +
                                ["input_" + str(i) for i in range(11)] + ["arm_target_" + str(i) for i in range(6)]))
        metrics = {"mode": args.mode, "task": args.task, "seed": args.seed,
                   "completed": not failures, "failures": failures,
                   "duration_s": len(records) * base.dt,
                   "position_rmse_until_stop_m": float(np.sqrt(np.mean(array[:, 1] ** 2))) if records else None,
                   "orientation_rmse_until_stop_rad": float(np.sqrt(np.mean(array[:, 2] ** 2))) if records else None,
                   "position_p95_until_stop_m": float(np.percentile(array[:, 1], 95)) if records else None,
                   "final_position_error_m": float(array[-1, 1]) if records else None,
                   "solver_calls": len(solves), "solver_failures": sum(not d["ok"] for d in solves),
                   "solve_time_median_s": float(np.median([d["seconds"] for d in solves])) if solves else None,
                   "solve_time_p95_s": float(np.percentile([d["seconds"] for d in solves], 95)) if solves else None,
                   "solve_deadline_misses": sum(d["seconds"] > controller.dt for d in solves)}
        (output / "solver.json").write_text(json.dumps(solves, indent=2) + "\n")
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print("TRIAL_RESULT " + json.dumps(metrics), flush=True)
    finally:
        base.close()


def summarize(root):
    root = Path(root)
    entries = [json.loads(p.read_text()) for p in sorted(root.glob("*/metrics.json"))]
    lines = ["# Measured IsaacGym closed-loop MPC screening", "",
             "Same RL-MPC_2 policy, costs and constraints; only prediction dynamics differ.",
             "Nominal plane, frozen commanded roll, local EE Jacobian; not an OCS2/terrain/feasibility-envelope validation.",
             "Stopped runs are failures. Their truncated RMSE is not used to claim a gain.", "",
             "| Task | Seed | Ideal completed | Identified completed | Ideal position RMSE (m) | Identified position RMSE (m) | Change |",
             "|---|---:|---|---|---:|---:|---:|"]
    paired = []
    for task, seed in sorted({(e["task"], e["seed"]) for e in entries}):
        group = {e["mode"]: e for e in entries if e["task"] == task and e["seed"] == seed}
        if set(group) != {"ideal", "identified"}:
            continue
        ideal, identified = group["ideal"], group["identified"]
        change = None
        if ideal["completed"] and identified["completed"]:
            change = identified["position_rmse_until_stop_m"] / max(ideal["position_rmse_until_stop_m"], 1e-12) - 1.0
        paired.append({"task": task, "seed": seed, "ideal": ideal, "identified": identified,
                       "position_relative_change": change})
        error_a = "n/a" if not ideal["completed"] else "{:.5f}".format(ideal["position_rmse_until_stop_m"])
        error_b = "n/a" if not identified["completed"] else "{:.5f}".format(identified["position_rmse_until_stop_m"])
        change_text = "n/a" if change is None else "{:+.1%}".format(change)
        lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            task, seed, ideal["completed"], identified["completed"], error_a, error_b, change_text))
    lines += ["", "See each trial's solver.json and metrics.json for solver failures, orientation errors and timing.",
              "The simulator advances synchronously; solve deadline misses are recorded, not hidden or emulated as deployment latency."]
    (root / "summary.json").write_text(json.dumps({"version": VERSION, "trials": entries, "pairs": paired}, indent=2) + "\n")
    (root / "report.md").write_text("\n".join(lines) + "\n")
    print("SUMMARY trials={} pairs={} output={}".format(len(entries), len(paired), root), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("ideal", "identified"))
    parser.add_argument("--task", choices=TASKS, default="reach")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--seconds", type=float, default=24.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--logdir", default="runs/stage1_rlmpc_benchmark_2_223425")
    parser.add_argument("--identification", default="data/identification/balanced_validation/prediction/summary.json")
    parser.add_argument("--output")
    parser.add_argument("--summarize")
    parser.add_argument("--push-scale", type=float, default=1.0)
    parser.add_argument("--pose-envelope", choices=("screening", "checkpoint"), default="screening")
    parser.add_argument("--orientation-weight", type=float, default=30.0)
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()
    if args.push_scale < 0 or args.orientation_weight <= 0:
        parser.error("--push-scale must be nonnegative and --orientation-weight must be positive")
    if args.summarize:
        summarize(args.summarize)
    elif args.mode and args.output and args.seconds > 0:
        run_trial(args)
    else:
        parser.error("Provide --mode and --output, with a positive --seconds")


if __name__ == "__main__":
    main()
