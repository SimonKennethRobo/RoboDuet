"""OCS2 SQP/HPIPM closed loop on the measured RL + arm-servo IsaacGym plant."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import selectors
import subprocess
import time

from benchmark.dog_policy.servo_runtime import ServoPlant, DEFAULT_POLICY, NOMINAL_Q
from benchmark.dog_policy.closed_loop_mpc import target_at
import numpy as np
from scipy.spatial.transform import Rotation
import torch

ROOT = Path(__file__).resolve().parents[2]
NX, NU, TARGET_INDEX, PREVIOUS_INDEX = 37, 11, 20, 26
Z_INDEX = [5, 6, 7, 2, 4] + list(range(8, 26))


def kinematic_chain(plant):
    parents = {j.find("child").get("link"): j for j in plant.urdf.findall("joint")}
    link = plant.base.body_names[plant.base.ee_idx]
    chain = []
    while link in parents:
        joint = parents[link]
        origin = joint.find("origin")
        T = np.eye(4)
        if origin is not None:
            T[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            T[:3, :3] = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
        name = joint.get("name")
        index = int(name[len("x5_joint"):]) - 1 if name.startswith("x5_joint") and name[len("x5_joint"):].isdigit() else -1
        if joint.get("type") != "fixed" and not 0 <= index < 6:
            raise ValueError(f"Unexpected moving joint in EE chain: {name}")
        axis = joint.find("axis")
        axis = np.fromstring(axis.get("xyz"), sep=" ") if axis is not None else np.array([0., 0., 1.])
        chain.append(dict(origin=T.tolist(), axis=axis.tolist(), index=index))
        link = joint.find("parent").get("link")
    chain.reverse()
    T = np.eye(4)
    T[:3, 3] = plant.base.ee_local_offset[0].cpu().numpy()
    chain.append(dict(origin=T.tolist(), axis=[0., 0., 1.], index=-1))
    return chain


def prediction_matrices(model, state, roll, mode):
    dt = model["period_s"]
    Az, Bz, cz = (np.asarray(model[k]) for k in ("A", "B", "c"))
    A, B, c = np.eye(NX), np.zeros((NX, NU)), np.zeros(NX)
    W = np.zeros((11, NX))
    W[5:, TARGET_INDEX:TARGET_INDEX+6] = np.eye(6)
    V = np.zeros((11, NU))
    V[:5, :5] = np.eye(5)
    V[5:, 5:] = dt * np.eye(6)
    A[Z_INDEX] = 0.
    A[np.ix_(Z_INDEX, Z_INDEX)] = Az
    A[Z_INDEX] += Bz @ W
    B[Z_INDEX] = Bz @ V
    c[Z_INDEX] = cz
    if mode == "base_only":
        A[8:14] = 0.
        A[8:14, 8:14] = np.eye(6)
        B[8:14] = 0.
        B[8:14, 5:] = dt * np.eye(6)
        c[8:14] = 0.
    rotation = Rotation.from_euler("ZYX", [state[3], state[4], roll]).as_matrix()
    for j in (0, 1):
        for k in (0, 1):
            weight = .5 * dt * rotation[j, k]
            A[j, 5+k] += weight
            A[j] += weight * A[5+k]
            B[j] += weight * B[5+k]
            c[j] += weight * c[5+k]
    denom = np.cos(state[4]) * np.cos(roll)
    if abs(denom) < .2:
        raise ValueError("Euler yaw prediction outside operating domain")
    weight = .5 * dt / denom
    A[3, 7] += weight
    A[3] += weight * A[7]
    B[3] += weight * B[7]
    c[3] += weight * c[7]
    coupling = np.sin(roll) / denom
    A[3] += coupling * A[4]
    A[3, 4] -= coupling
    B[3] += coupling * B[4]
    c[3] += coupling * c[4]
    A[PREVIOUS_INDEX:] = 0.
    B[PREVIOUS_INDEX:] = np.eye(NU)
    return A, B, c


def constraint_matrices(plant, dt):
    speed = np.array([.8, .8, .8, 1.2, 1.2, 1.2])
    low, high = np.r_[plant.low, -speed], np.r_[plant.high, speed]
    C = np.zeros((22, NX))
    D = np.vstack([np.eye(NU), -np.eye(NU)])
    e = np.r_[-low, high]
    slew = np.array([plant.cfg.response.rate_limit[k] for k in ("vx", "vy", "wyaw", "height", "pitch")]) * dt
    S = np.zeros((5, NX)); S[:, PREVIOUS_INDEX:PREVIOUS_INDEX+5] = np.eye(5)
    U = np.eye(NU)[:5]
    C = np.vstack([C, -S, S]); D = np.vstack([D, U, -U]); e = np.r_[e, slew, slew]
    F = np.zeros((12, NX))
    F[:6, TARGET_INDEX:TARGET_INDEX+6] = np.eye(6)
    F[6:, TARGET_INDEX:TARGET_INDEX+6] = -np.eye(6)
    h = np.r_[-plant.q_low, plant.q_high]
    return C, D, e, F, h


class OCS2Service:
    def __init__(self, executable, stderr_path, timeout=60.):
        self.stderr = open(stderr_path, "w")
        env = os.environ.copy()
        # Linked OCS2 libraries use the system C++ runtime; don't inject Conda.
        env.pop("LD_LIBRARY_PATH", None)
        self.process = subprocess.Popen([str(Path(executable).resolve())], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.stderr, text=True, bufsize=1, env=env)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.timeout = timeout

    def solve(self, request):
        self.process.stdin.write(json.dumps(request, allow_nan=False) + "\n")
        self.process.stdin.flush()
        if not self.selector.select(self.timeout):
            raise TimeoutError("OCS2 solver did not answer; inspect ocs2_stderr.log")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"OCS2 exited ({self.process.poll()}); inspect ocs2_stderr.log")
        return json.loads(line)

    def close(self):
        self.selector.close()
        self.process.stdin.close()
        try:
            self.process.wait(timeout=3.)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try: self.process.wait(timeout=3.)
            except subprocess.TimeoutExpired:
                self.process.kill(); self.process.wait()
        self.stderr.close()


def state_vector(measured, previous_q, q_target, previous_input):
    return np.r_[measured["root"][0, :3], measured["ypr"][0, :2], measured["response"][0, :3],
                 measured["q"][0], previous_q, q_target, previous_input]


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    bundle = json.loads(Path(args.models).read_text())
    model = bundle["models"]["coupled" if args.mode == "coupled" else "separate"]
    plant = ServoPlant(seed=args.seed, seconds=args.seconds, logdir=args.logdir, device=args.device)
    service = None
    records, solves, failures = [], [], []
    dt = model["period_s"]
    ticks = round(dt / plant.dt)
    try:
        if plant.manifest["checkpoint_sha256"] != bundle["checkpoint_sha256"] or plant.manifest["asset_sha256"] != bundle["asset_sha256"]:
            raise ValueError("Identified policy or URDF differs from the measured plant")
        if not np.isclose(ticks * plant.dt, dt):
            raise ValueError("MPC period must match the policy tick grid")
        C, D, e, F, h = constraint_matrices(plant, dt)
        for _ in range(round(3. / plant.dt)):
            done, _ = plant.step(np.zeros(5), target=NOMINAL_Q)
            if done.any(): raise RuntimeError("Reset during warmup")
        m = plant.state()
        anchor, target_rotation = m["ee"][0, :3].copy(), Rotation.from_quat(m["ee"][0, 3:7])
        previous_q = m["q"][0].copy()
        previous_input = np.zeros(11)
        chain = kinematic_chain(plant)
        manifest = dict(plant.manifest, solver="OCS2 SqpSolver / HPIPM", mode=args.mode,
            model_file=str(Path(args.models).resolve()), models_sha256=hashlib.sha256(Path(args.models).read_bytes()).hexdigest(),
            solver_executable_sha256=hashlib.sha256(Path(args.solver).read_bytes()).hexdigest(),
            task=args.task, seconds=args.seconds, mpc_period=dt, horizon=args.horizon,
            envelope="same screening command/slew/joint-target bounds in all modes; not an identified safety envelope",
            constraint_handling="explicit relaxed barriers mu=.001 delta=1e-6 plus full-horizon feasibility rejection",
            max_sqp_iterations=120,
            solver_acceptance="finite feasible solution with dynamics defect <1e-5; not an optimality certificate",
            kinematics="URDF analytic T_world_base * FK(q); measured roll frozen over each horizon",
            dynamics="identified 100ms grid map; fixed-grid Euler adapter; measured state at every MPC update",
            simulation="synchronous IsaacGym, not dummy rollout; no real-time latency emulation",
            orientation_weight=args.orientation_weight, push_scale=args.push_scale,
            initial_state=state_vector(m, previous_q, plant.q_target[0], previous_input).tolist(), initial_ee=m["ee"][0].tolist())
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        service = OCS2Service(args.solver, output / "ocs2_stderr.log")
        base = plant.base
        base.mpc_forces = torch.zeros((1, base.num_bodies, 3), device=base.device)
        trunk = next((base.body_names.index(k) for k in ("base", "trunk") if k in base.body_names), None)
        if trunk is None: raise ValueError("Missing trunk/base body")
        max_fk_position, max_fk_rotation = 0., 0.
        for k in range(round(args.seconds / dt)):
            now = k * dt
            m = plant.state()
            state = state_vector(m, previous_q, plant.q_target[0], previous_input)
            roll = float(m["ypr"][0, 2])
            A, B, c = prediction_matrices(model, state, roll, args.mode)
            refs = [np.r_[p, r.as_quat()].tolist() for p, r in
                    [target_at(args.task, now + i*dt, anchor, target_rotation) for i in range(args.horizon+1)]]
            request = dict(A=A.tolist(), B=B.tolist(), c=c.tolist(), state=state.tolist(), dt=dt,
                horizon=args.horizon, roll=roll, chain=chain, references=refs, iterations=120,
                ee_weights=[200.]*3 + [args.orientation_weight]*3,
                R=[1., 1.5, .4, .01, .05] + [.02]*6,
                D=[.3, .3, .1, 5., 2.] + [.01]*6, terminal_weight=.3,
                previous_input_index=PREVIOUS_INDEX, constraint_C=C.tolist(), constraint_D=D.tolist(), constraint_e=e.tolist(),
                state_constraint_F=F.tolist(), state_constraint_h=h.tolist())
            if k == 0:
                (output / "first_request.json").write_text(json.dumps(request) + "\n")
            diagnostic = service.solve(request)
            diagnostic.update(time=now, measured_state=state.tolist())
            solves.append(diagnostic)
            if not diagnostic["ok"]:
                (output / "rejected_request.json").write_text(json.dumps(request) + "\n")
                failures.append("ocs2_rejected_solution: " + diagnostic.get("error", f"margin={diagnostic.get('constraint_margin')}"))
                break
            fk = np.asarray(diagnostic["fk_at_state"])
            fk_position = float(np.linalg.norm(fk[:3] - m["ee"][0, :3]))
            fk_rotation = float((Rotation.from_quat(fk[3:]) * Rotation.from_quat(m["ee"][0, 3:]).inv()).magnitude())
            max_fk_position, max_fk_rotation = max(max_fk_position, fk_position), max(max_fk_rotation, fk_rotation)
            if fk_position > .003 or fk_rotation > .01:
                failures.append(f"URDF/measured EE mismatch: {fk_position} m, {fk_rotation} rad")
                break
            command = np.array(diagnostic["command"])
            previous_q = m["q"][0].copy()
            for i in range(ticks):
                t = now + i * plant.dt
                base.mpc_forces.zero_()
                if args.task == "hold_push":
                    for event, start in enumerate((6., 12., 18.)):
                        if start <= t < start+.25:
                            base.mpc_forces[0, trunk, 1] = args.push_scale * (20. if event % 2 == 0 else -20.)
                done, applied = plant.step(command[:5], velocity=command[5:])
                measured = plant.state()
                p, r = target_at(args.task, t+plant.dt, anchor, target_rotation)
                poserr = float(np.linalg.norm(measured["ee"][0, :3]-p))
                roterr = float((Rotation.from_quat(measured["ee"][0, 3:])*r.inv()).magnitude())
                records.append(dict(time=t+plant.dt, position_error=poserr, rotation_error=roterr,
                    q=measured["q"][0].tolist(), dq=measured["dq"][0].tolist(), response=measured["response"][0].tolist(),
                    ee=measured["ee"][0].tolist(), command=command.tolist(), q_target=applied[0].tolist(), reset=bool(done[0])))
                if done.any():
                    failures.append("environment_reset")
                    break
            previous_input = command
            if failures: break
            if k % 25 == 0:
                print(f"OCS2 mode={args.mode} task={args.task} t={now+dt:.1f}s EE={poserr:.4f}m rot={np.rad2deg(roterr):.2f}deg", flush=True)
        metrics = dict(mode=args.mode, task=args.task, seed=args.seed, solver="OCS2 SqpSolver / HPIPM",
            completed=not failures, failures=failures, duration_s=len(records)*plant.dt, solver_calls=len(solves),
            solver_failures=sum(not s["ok"] for s in solves),
            position_rmse_m=float(np.sqrt(np.mean([r["position_error"]**2 for r in records]))) if records else None,
            orientation_rmse_deg=float(np.rad2deg(np.sqrt(np.mean([r["rotation_error"]**2 for r in records])))) if records else None,
            max_fk_position_error_m=max_fk_position, max_fk_orientation_error_rad=max_fk_rotation)
        (output / "solver.json").write_text(json.dumps(solves, indent=2) + "\n")
        (output / "trajectory.json").write_text(json.dumps(records) + "\n")
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print("TRIAL_RESULT " + json.dumps(metrics), flush=True)
        if failures: raise RuntimeError(str(failures))
    finally:
        if service is not None: service.close()
        plant.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--models", default="data/identification/arm_servo_20260911/models.json")
    p.add_argument("--mode", choices=("base_only", "separate", "coupled"), default="separate")
    p.add_argument("--task", choices=("reach", "line", "circle", "hold_push"), default="hold_push")
    p.add_argument("--seconds", type=float, default=24.)
    p.add_argument("--seed", type=int, default=29)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--orientation-weight", type=float, default=30.)
    p.add_argument("--push-scale", type=float, default=1.)
    p.add_argument("--solver", default=str(ROOT / "benchmark/ocs2_servo/build/servo_ocs2"))
    p.add_argument("--logdir", default=DEFAULT_POLICY)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    if args.seconds <= 0 or args.horizon < 2 or args.orientation_weight <= 0 or args.push_scale < 0:
        p.error("invalid duration/horizon/weight/push scale")
    run(args)
