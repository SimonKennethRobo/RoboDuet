"""Export I_Q fitted task files and run their native OCS2/MuJoCo closed loop."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from identify_iq_mujoco import IdentificationPlant, ROOT, STACK, CHANNELS, sha, write_json
sys.path.insert(0, str(ROOT))
from benchmark.wbc.controllers import NativeOcs2Transport, encode_ocs2_state_values

LIMITS = np.array([.55, .28, .65, .035, .20, .14])


def block_span(text, key):
    match = re.search(r"(?m)^\s*"+re.escape(key)+r"\s*\{", text)
    if not match:
        raise ValueError(f"missing task block {key}")
    start = text.index("{", match.start())
    depth = 1
    for stop in range(start+1, len(text)):
        depth += (text[stop] == "{") - (text[stop] == "}")
        if depth == 0:
            return match.start(), stop+1
    raise ValueError("unbalanced task")


def export(root):
    out = root/"mpc"
    out.mkdir(exist_ok=True)
    source = STACK/"go2_x5_ocs2/config/task_floating.info"
    text = source.read_text()
    # A mount-height state must not be constrained by the old trunk-height
    # [0.2,0.35] box. All three compared controllers share this correction.
    start, end = block_span(text, "basePositionLimits")
    text = text[:start]+"\nbasePositionLimits\n{\n activate true\n mu .1\n delta .001\n limits\n {\n z { lower .27 upper .47 }\n pitch { lower -.30 upper .30 }\n roll { lower -.20 upper .20 }\n }\n}\n"+text[end:]
    text = re.sub(r"recompileLibraries\s+true", "recompileLibraries false", text)
    # Frozen initial values are superseded by the first measured observation.
    # Preserve original costs/arm geometry and keep self collision enabled.
    (out/"task_I_Q_ideal.info").write_text(text)
    manifests = {}
    for label, filename in [("first_order", "F0.json"), ("gait", "F1_gait.json")]:
        model_path = root/"models_v2"/filename
        model = json.loads(model_path.read_text())
        task = re.sub(r"manipulatorModelType\s+3", "manipulatorModelType 4", text)
        blocks = list(re.finditer(r"fullyActuatedFloatingArmManipulator\s*\{[^{}]*\}", task))
        if len(blocks) != 4:
            raise ValueError("expected initial state, input cost, lower/upper command blocks")
        values = [[0, 0, .4, 0, 0, 0, 0, 0, 0, 0], None, -LIMITS, LIMITS]
        for index, match in reversed(list(enumerate(blocks))):
            if index == 1:
                entries = "\n".join(f" ({j},{j}) {w}" for j, w in enumerate([.2,.2,.15,.3,.4,.4]))
            else:
                entries = "\n".join(f" ({j},0) {v}" for j,v in enumerate(values[index]))
            task = task[:match.end()]+"\n policyAwareFloatingArmManipulator\n {\n"+entries+"\n }\n"+task[match.end():]
        task = task.replace("inputCost\n{", "inputCost\n{\n physicalMotion true\n physicalMotionWeights\n {\n"+
            "\n".join(f" ({j},0) {w}" for j,w in enumerate([.2,.2,.3,.15,.4,.4]+[.01]*6))+"\n }\n", 1)
        task += "\n; I_Q model independently fitted in MuJoCo; mounting-plane coordinates.\npolicyResponseModel\n{\n transportDelays false\n bodyFrameVelocity false\n stopGaitAtStand true\n heightOffset 0\n gaitFrequency 2.75\n nominal\n {\n"
        for name in CHANNELS:
            c = model[name]
            task += f" {name} {{ gain {c['gain']:.17g} timeConstant {c['tau_s']:.17g} bias {c['bias']:.17g} delay 0 }}\n"
        task += " }\n gaitResidual\n {\n activate "+("true" if label == "gait" else "false")+"\n"
        if label == "gait":
            for name, field in [("height", "z"), ("pitch", "pitch"), ("roll", "roll")]:
                r = model[name]["residual"]
                task += f" {field} {{ amplitude {r['amplitude']:.17g} speedAmplitude {r['speed_amplitude']:.17g} harmonic {r['harmonic']} phaseOffset {r['phase_offset']:.17g} }}\n"
        task += " }\n}\nidentifiedDeployment\n{\n"
        for bound, sign in [("commandMin", -1), ("commandMax", 1)]:
            task += " "+bound+"\n {\n"+"\n".join(f" {name} {sign*limit}" for name,limit in zip(CHANNELS,LIMITS))+"\n }\n"
        task += "}\n"
        path = out/f"task_I_Q_{label}.info"
        path.write_text(task)
        manifests[label] = dict(model=str(model_path), model_sha256=sha(model_path),
            task=str(path), task_sha256=sha(path), state_dim=16, input_dim=12)
    write_json(out/"manifest.json", dict(models=manifests, source_task=str(source),
        source_task_sha256=sha(source), ideal_task_sha256=sha(out/"task_I_Q_ideal.info"),
        policy_sha256=sha(STACK/"rl_sar/policy/go2_x5/I_Q/policy.pt"),
        command_limits=LIMITS.tolist(), scope="flat ground, fixed 2.75Hz gait; limits are sampled ranges, not a learned feasibility envelope"))
    print(json.dumps(manifests, indent=2))


def reference(initial, seconds, scenario):
    t = np.linspace(0., seconds+1., int((seconds+1)/.05)+1)
    p = np.broadcast_to(initial["ee"], (len(t),3)).copy()
    q0 = Rotation.from_matrix(initial["ee_rotation"].reshape(3,3))
    rotation = np.zeros((len(t),3))
    if scenario != "hold":
        fraction = np.clip(t/seconds, 0, 1)
        smooth = 10*fraction**3-15*fraction**4+6*fraction**5
        distance = {"curve": .35, "long_curve": .55, "walking_curve": 1.2}[scenario]
        p[:,0] += distance*smooth
        p[:,1] += .045*np.sin(2*np.pi*fraction)*np.sin(np.pi*fraction)**2
        p[:,2] += .025*np.sin(2*np.pi*fraction)*np.sin(np.pi*fraction)**2
        rotation[:,1] = .12*np.sin(2*np.pi*fraction)*np.sin(np.pi*fraction)**2
    quat = (Rotation.from_rotvec(rotation)*q0).as_quat()
    return t, np.column_stack([p,quat])


def run(args):
    root = Path(args.root).resolve()
    deployment = json.loads((root/"mpc/manifest.json").read_text())
    inputs = json.loads((root/"input_manifest.json").read_text())["files"]
    for path in [STACK/"rl_sar/policy/go2_x5/base.yaml", STACK/"rl_sar/policy/go2_x5/I_Q/config.yaml",
                 STACK/"rl_sar/policy/go2_x5/I_Q/policy.pt", *Path(args.scene).parent.glob("*.xml")]:
        if str(path) not in inputs or sha(path) != inputs[str(path)]:
            raise ValueError(f"identified plant input changed: {path}")
    output = Path(args.output).resolve() if args.output else root/"closed_loop"/f"{args.scenario}_{args.controller}_s{args.seed}"
    if (output/"receipt.json").exists():
        raise FileExistsError("finished trial exists; use a new output")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    plant = IdentificationPlant(STACK/"rl_sar/policy/go2_x5", args.scene, args.seed)
    init = plant.snapshot()
    times, poses = reference(init, args.seconds, args.scenario)
    np.savez_compressed(output/"reference.npz", times=times, poses=poses)
    task = root/"mpc"/f"task_I_Q_{args.controller}.info"
    expected_task = (deployment["ideal_task_sha256"] if args.controller == "ideal"
                     else deployment["models"][args.controller]["task_sha256"])
    if sha(task) != expected_task:
        raise ValueError("MPC task differs from the frozen exported model")
    runtime_hashes = {str(p): sha(p) for p in [Path(__file__), ROOT/"scripts/identify_iq_mujoco.py",
        ROOT/"scripts/sim2sim_mujoco.py", ROOT/"scripts/rl_sar_obs.py", ROOT/"benchmark/wbc/controllers.py",
        STACK/"ros2_ws/install/go2_x5_ocs2_bridge/lib/go2_x5_ocs2_bridge/wbc_benchmark_sync"]}
    transport = None
    viewer = None
    if args.viewer:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(plant.sim.model, plant.sim.data)
        viewer.cam.distance = 2.
    trace = []
    fail = plant.failure()
    begin = time.monotonic()
    try:
        transport = NativeOcs2Transport(output/"ocs2", STACK, mode="synchronous",
            task_profile="native_ideal", task_file=task, timeout_s=300.)
        for k in range(round(args.seconds/.02)):
            if fail or (viewer and not viewer.is_running()):
                break
            state = plant.snapshot()
            payload = encode_ocs2_state_values(seq=k, base_pos_world=state["base_position"],
                base_quat_xyzw=state["quaternion"], base_lin_vel_body=state["body_velocity"],
                base_ang_vel_body=state["gyro"], arm_q=state["q"][12:], arm_dq=state["dq"][12:],
                leg_q=state["q"][:12], leg_dq=state["dq"][:12])
            tic = time.monotonic()
            command = transport.exchange(payload, times if k == 0 else None,
                poses if k == 0 else None, gait_phase_rad=state["phase"])
            solve_wall = time.monotonic()-tic
            u_raw = np.r_[command["base_velocity_body"], command["body_posture"]]
            u = np.clip(u_raw, -LIMITS, LIMITS)
            arm = np.asarray(command["arm_q_cmd"])
            qdot = np.asarray(command["arm_dq_cmd"])
            if not np.isfinite(np.r_[u,arm,qdot]).all():
                raise FloatingPointError("nonfinite MPC command")
            fraction = np.clip((k*.02)/.05, 0, len(times)-1)
            lower = min(int(fraction),len(times)-2)
            alpha = fraction-lower
            target = (1-alpha)*poses[lower]+alpha*poses[lower+1]
            target[3:] /= np.linalg.norm(target[3:])
            rot_error = (Rotation.from_quat(target[3:]).inv()*Rotation.from_matrix(state["ee_rotation"].reshape(3,3))).magnitude()
            trace.append(dict(**state, t=k*.02, command=u, raw_command=u_raw,
                arm_target=arm, arm_velocity=qdot, target=target,
                position_error=float(np.linalg.norm(state["ee"]-target[:3])),
                orientation_error=float(rot_error), solve_wall_s=solve_wall))
            plant.step(u, arm, qdot)
            fail = plant.failure()
            if viewer:
                viewer.sync()
                time.sleep(max(0.,begin+(k+1)*.02-time.monotonic()))
    except Exception as error:
        fail = f"{type(error).__name__}: {error}"
    finally:
        if transport:
            transport.close()
        if viewer:
            viewer.close()
        if trace:
            arrays = {key: np.asarray([row[key] for row in trace]) for key in trace[0]}
            np.savez_compressed(output/"trace.npz", **arrays)
        receipt = dict(controller=args.controller, scenario=args.scenario, seed=args.seed,
            requested_steps=round(args.seconds/.02), steps=len(trace), failure=fail,
            success=fail is None and len(trace)==round(args.seconds/.02),
            physics_dt_s=.0025, policy_dt_s=.02, control_dt_s=.005,
            wall_seconds=time.monotonic()-begin, task_sha256=sha(task),
            runtime_hashes=runtime_hashes, reference_sha256=sha(output/"reference.npz"),
            transport_statistics=transport.runtime_stats if transport else {},
            policy_sha256=sha(STACK/"rl_sar/policy/go2_x5/I_Q/policy.pt"),
            phase_source="actual RlSarMujoco policy clock at measured state boundary",
            protocol="simulator-time native C++ SQP/MRT, Python rl_sar mirror, original explicit torque PD",
            arm_model="ideal dq integration plus existing MRT position lookahead; actual plant has PD/target slew",
            frames="trunk origin velocity -> heading response; arm mount pose for FK")
        if trace:
            receipt.update(position_rmse_m=float(np.sqrt(np.mean(arrays["position_error"]**2))),
                orientation_rmse_rad=float(np.sqrt(np.mean(arrays["orientation_error"]**2))),
                base_displacement_m=float(np.linalg.norm(arrays["base_position"][-1,:2]-arrays["base_position"][0,:2])),
                command_clipped_steps=int(np.count_nonzero(np.any(np.abs(arrays["command"]-arrays["raw_command"])>1e-6,axis=1))),
                trace_sha256=sha(output/"trace.npz"))
        write_json(output/"receipt.json", receipt)
        print(json.dumps(receipt, indent=2), flush=True)
    if not receipt["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["export", "run"])
    parser.add_argument("--root", default=str(ROOT/"tmp/experiments/20260915_iq_identification"))
    parser.add_argument("--scene", default=str(STACK/"rl_sar/src/rl_sar_zoo/go2_x5_description/mjcf/scene.xml"))
    parser.add_argument("--controller", choices=["ideal", "first_order", "gait"], default="first_order")
    parser.add_argument("--scenario", choices=["hold", "curve", "long_curve", "walking_curve"], default="curve")
    parser.add_argument("--seconds", type=float, default=16.)
    parser.add_argument("--seed", type=int, default=9101)
    parser.add_argument("--output")
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    if args.mode == "export":
        export(Path(args.root).resolve())
    else:
        run(args)
