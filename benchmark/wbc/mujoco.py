"""Run a frozen WBC TaskSpec through the rl_sar MuJoCo policy boundary.

The exported dog policy controls the 12 leg joints. The upper controller can
be either the deterministic DLS baseline or the independent native OCS2
process used by the IsaacGym system matrix. The runner reads the same suite
JSON/reference NPZ and emits the same trace schema as the IsaacGym path.
"""

from __future__ import annotations

import argparse
import atexit
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mujoco
import numpy as np

from benchmark.wbc.scoring import (
    DEVELOPMENT_KINEMATIC_PROTOCOL,
    DEVELOPMENT_TIMED_TRAJECTORY_PROTOCOL,
)
from benchmark.wbc.trace import TRACE_SCHEMA_VERSION, score_trace_archive
from benchmark.wbc.suite import refresh_suite_hash


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tree(root: Path) -> Tuple[str, int]:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest(), len(files)


def _quat_xyzw_from_matrix(matrix: np.ndarray) -> np.ndarray:
    quat_wxyz = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat_wxyz, matrix.reshape(-1))
    return np.roll(quat_wxyz, -1)


def _quat_slerp(q0: np.ndarray, q1: np.ndarray, fraction: float) -> np.ndarray:
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        value = q0 + fraction * (q1 - q0)
        return value / np.linalg.norm(value)
    angle = math.acos(np.clip(dot, -1.0, 1.0))
    return (
        math.sin((1.0 - fraction) * angle) * q0
        + math.sin(fraction * angle) * q1
    ) / math.sin(angle)


def _quat_mul_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return np.array(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        dtype=np.float64,
    )


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rotation_error(desired_xyzw: np.ndarray, current_matrix: np.ndarray) -> np.ndarray:
    relative = _quat_to_matrix(desired_xyzw) @ current_matrix.T
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    angle = math.acos(cosine)
    vee = np.array(
        [relative[2, 1] - relative[1, 2], relative[0, 2] - relative[2, 0], relative[1, 0] - relative[0, 1]]
    )
    if angle < 1e-7:
        return 0.5 * vee
    return angle * vee / (2.0 * math.sin(angle))


def _forward_project(reference, previous_arc, position, quaternion, window=0.15, samples=16):
    """Mirror RoboDuet's monotonic SE(3) progress projection."""
    candidates = np.linspace(
        previous_arc,
        min(previous_arc + window, float(reference.gamma_s[-1])),
        samples,
    )
    current_rotation = _quat_to_matrix(quaternion)
    costs = []
    lateral = []
    for arc in candidates:
        index = int(np.clip(np.searchsorted(reference.gamma_s, arc, side="right") - 1, 0, len(reference.gamma_s) - 2))
        fraction = float((arc - reference.gamma_s[index]) / max(reference.gamma_s[index + 1] - reference.gamma_s[index], 1e-12))
        point = reference.gamma_p[index] + fraction * (reference.gamma_p[index + 1] - reference.gamma_p[index])
        target_quaternion = _quat_slerp(reference.gamma_q[index], reference.gamma_q[index + 1], fraction)
        position_error = float(np.linalg.norm(point - position))
        rotation_error = float(np.linalg.norm(_rotation_error(target_quaternion, current_rotation)))
        costs.append(math.sqrt(position_error**2 + (0.15 * rotation_error) ** 2))
        lateral.append(position_error)
    best = int(np.argmin(costs))
    return float(candidates[best]), float(lateral[best])


class FrozenReference:
    def __init__(self, suite_path: Path, task_id: Optional[str] = None):
        records = json.loads(suite_path.read_text())
        if isinstance(records, dict):
            records = [records]
        groups = [record.get("suite", record) for record in records]
        matches = []
        for group_index, group in enumerate(groups):
            for row_index, task in enumerate(group["trajectories"]):
                if task_id is None or task["task_id"] == task_id:
                    matches.append((group_index, row_index, group, task))
        if not matches:
            raise ValueError(f"Task {task_id!r} was not found in {suite_path}")
        if task_id is None and len(matches) != 1:
            raise ValueError("--task-id is required when the suite contains multiple tasks")
        self.group_index, row, self.group, self.task = matches[0]
        copied_group = copy.deepcopy(self.group)
        expected_suite_hash = copied_group.get("suite_sha256")
        if refresh_suite_hash(copied_group) != expected_suite_hash:
            raise ValueError(f"Suite hash mismatch: {suite_path}")
        identity = {
            "task_family": self.task["task_family"],
            "reference_content_sha256": self.task["content_sha256"],
            "initial_state": self.task["initial_state"],
            "disturbance_schedule": self.task["disturbance_schedule"],
            "deadline_s": self.task["deadline_s"],
            "height_reference": self.task["height_reference"],
            "evaluation_protocol": self.group["evaluation_protocol"],
            "reference_frame": self.group["reference_frame"],
        }
        if "anchor_env_local_xyz_m" in self.task:
            identity["anchor_env_local_xyz_m"] = self.task["anchor_env_local_xyz_m"]
            identity["orientation_left_multiplier_xyzw"] = self.task[
                "orientation_left_multiplier_xyzw"
            ]
        else:
            identity["anchor_env_local_xy_m"] = self.task["anchor_env_local_xy_m"]
        task_digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if task_digest != self.task["task_spec_sha256"]:
            raise ValueError(f"TaskSpec hash mismatch: {self.task['task_id']}")
        if self.task["task_id"] != f"timed-trajectory-{task_digest[:16]}":
            raise ValueError(f"Task ID does not match TaskSpec hash: {self.task['task_id']}")
        archive = suite_path.parent / self.group["reference_archive"]["path"]
        expected = self.group["reference_archive"]["sha256"]
        if _sha256(archive) != expected:
            raise ValueError(f"Reference archive hash mismatch: {archive}")
        with np.load(archive, allow_pickle=False) as data:
            task_ids = [str(value) for value in data["task_id"].tolist()]
            archive_row = task_ids.index(self.task["task_id"])
            ng = int(data["gamma_points"][archive_row])
            nt = int(data["time_law_points"][archive_row])
            self.gamma_s = data["gamma_s"][archive_row, :ng].astype(np.float64)
            self.gamma_p = data["gamma_p"][archive_row, :ng].astype(np.float64)
            self.gamma_q = data["gamma_quat_xyzw"][archive_row, :ng].astype(np.float64)
            self.tl_t = data["tl_t"][archive_row, :nt].astype(np.float64)
            self.tl_s = data["tl_s"][archive_row, :nt].astype(np.float64)
            self.duration = float(data["duration_s"][archive_row])
        if "anchor_env_local_xyz_m" in self.task:
            anchor = np.asarray(self.task["anchor_env_local_xyz_m"], dtype=np.float64)
            self.gamma_p += anchor
            alignment = np.asarray(
                self.task["orientation_left_multiplier_xyzw"], dtype=np.float64
            )
            alignment /= np.linalg.norm(alignment)
            self.gamma_q = np.stack(
                [_quat_mul_xyzw(alignment, quat) for quat in self.gamma_q]
            )
            self.gamma_q /= np.linalg.norm(self.gamma_q, axis=1, keepdims=True)
        else:
            anchor = np.asarray(self.task["anchor_env_local_xy_m"], dtype=np.float64)
            self.gamma_p[:, :2] += anchor
        self.archive_path = archive

    def timed_poses(self) -> np.ndarray:
        """Return the complete xyz+xyzw reference sampled at every time-law knot."""
        return np.stack(
            [np.concatenate(self.at(knot_time)[1:]) for knot_time in self.tl_t]
        ).astype(np.float32)

    def at(self, time_s: float):
        time_s = float(np.clip(time_s, 0.0, self.duration))
        arc = float(np.interp(time_s, self.tl_t, self.tl_s))
        index = int(np.clip(np.searchsorted(self.gamma_s, arc, side="right") - 1, 0, len(self.gamma_s) - 2))
        fraction = float((arc - self.gamma_s[index]) / max(self.gamma_s[index + 1] - self.gamma_s[index], 1e-12))
        position = self.gamma_p[index] + fraction * (self.gamma_p[index + 1] - self.gamma_p[index])
        quaternion = _quat_slerp(self.gamma_q[index], self.gamma_q[index + 1], fraction)
        return arc, position, quaternion


def _add_scalar_column(store, name, value, dtype=np.float32):
    store.setdefault(name, []).append(np.asarray([value], dtype=dtype))


def _validated_push_events(schedule) -> list[dict]:
    """Validate the backend-neutral base-force subset used by this runner."""
    events = []
    for event in schedule:
        expected = {
            "type": "constant_force",
            "body": "base",
            "frame": "environment_world",
            "application_point": "body_center_of_mass",
        }
        mismatches = [
            f"{key}={event.get(key)!r} (expected {value!r})"
            for key, value in expected.items() if event.get(key) != value
        ]
        force = np.asarray(event.get("force_n"), dtype=np.float64)
        start = float(event.get("start_time_s", -1.0))
        duration = float(event.get("duration_s", -1.0))
        if mismatches or force.shape != (3,) or not np.isfinite(force).all():
            detail = "; ".join(mismatches) or f"invalid force_n={force!r}"
            raise ValueError(f"unsupported MuJoCo disturbance: {detail}")
        if start < 0.0 or duration <= 0.0:
            raise ValueError("push start_time_s must be nonnegative and duration_s positive")
        declared = float(event.get("magnitude_n", np.linalg.norm(force)))
        if not np.isclose(declared, np.linalg.norm(force), rtol=1e-6, atol=1e-9):
            raise ValueError("push magnitude_n does not match force_n")
        events.append({"start": start, "stop": start + duration, "force": force})
    return events


def _push_force_at(events, time_s: float) -> np.ndarray:
    force = np.zeros(3, dtype=np.float64)
    for event in events:
        if event["start"] <= time_s < event["stop"]:
            force += event["force"]
    return force


def _foot_observation(model, data) -> tuple[np.ndarray, np.ndarray]:
    """Return world linear velocity and contact load for FL, FR, RL, RR."""
    names = ("FL", "FR", "RL", "RR")
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{name}_foot") for name in names]
    geom_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in names]
    if min(body_ids + geom_ids) < 0:
        raise ValueError("MJCF must expose FL/FR/RL/RR foot bodies and geoms")
    velocity = np.zeros((4, 3), dtype=np.float64)
    for row, body in enumerate(body_ids):
        spatial = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, spatial, 0)
        velocity[row] = spatial[3:]
    force = np.zeros((4, 3), dtype=np.float64)
    lookup = {geom: row for row, geom in enumerate(geom_ids)}
    for index in range(data.ncon):
        contact = data.contact[index]
        rows = {lookup[geom] for geom in (int(contact.geom1), int(contact.geom2)) if geom in lookup}
        if not rows:
            continue
        local = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, index, local)
        # The scorer needs normal support load; keep it positive independent of
        # whether the foot is geom1 or geom2 in MuJoCo's contact pair.
        vertical_load = abs(float(local[0] * contact.frame[2]))
        for row in rows:
            force[row, 2] += vertical_load
    return velocity, force


def _set_mpc_dog_command(sim, command: dict) -> np.ndarray:
    """Apply the physical OCS2 command at rl_sar's six-command boundary."""
    velocity = np.asarray(command["base_velocity_body"], dtype=np.float64).copy()
    posture = np.asarray(command["body_posture"], dtype=np.float64).copy()
    for index, key in enumerate(("limit_vel_x", "limit_vel_y", "limit_vel_yaw")):
        if key in sim.p:
            velocity[index] = np.clip(velocity[index], *sim.p[key])
    for index, key in enumerate(("limit_body_height", "limit_body_pitch", "limit_body_roll")):
        if key in sim.p:
            posture[index] = np.clip(posture[index], *sim.p[key])
    # rl_sar command order is vx, vy, yaw, pitch, roll, height offset.
    sim.command = [
        velocity[0], velocity[1], velocity[2], posture[1], posture[2], posture[0]
    ]
    return velocity


def _draw_trajectory(viewer, reference: FrozenReference, executed, target, base_position):
    """Draw orange reference and green executed paths in a passive MuJoCo viewer."""
    reference_points = reference.gamma_p
    if len(reference_points) > 101:
        indices = np.linspace(0, len(reference_points) - 1, 101).astype(int)
        reference_points = reference_points[indices]
    actual_points = np.asarray(executed[-201:], dtype=np.float64)
    with viewer.lock():
        viewer.cam.lookat[:] = base_position
        scene = viewer.user_scn
        scene.ngeom = 0

        def line(start, end, rgba):
            if scene.ngeom >= scene.maxgeom:
                return
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom, mujoco.mjtGeom.mjGEOM_LINE,
                np.zeros(3), np.zeros(3), np.eye(3).reshape(-1), rgba,
            )
            mujoco.mjv_connector(
                geom, mujoco.mjtGeom.mjGEOM_LINE, 3.0, start, end
            )
            scene.ngeom += 1

        for points, color in (
            (reference_points, np.array([1.0, 0.55, 0.0, 1.0], np.float32)),
            (actual_points, np.array([0.1, 1.0, 0.1, 1.0], np.float32)),
        ):
            for start, end in zip(points[:-1], points[1:]):
                line(start, end, color)
        if scene.ngeom < scene.maxgeom:
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.018, 0.018, 0.018]), target, np.eye(3).reshape(-1),
                np.array([1.0, 0.9, 0.1, 1.0], np.float32),
            )
            scene.ngeom += 1
    viewer.sync()


def run(args) -> Tuple[Path, dict]:
    rl_sar_root = Path(args.rl_sar_root).resolve()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from sim2sim_mujoco import RlSarMujoco  # pylint: disable=import-outside-toplevel

    suite_path = Path(args.suite).resolve()
    reference = FrozenReference(suite_path, args.task_id)
    push_events = _validated_push_events(reference.task["disturbance_schedule"])
    scene = Path(args.scene).resolve()
    robot_dir = rl_sar_root / "policy" / "go2_x5"
    if args.policy_adapter == "umi":
        from benchmark.wbc.umi_mujoco import UmiMujoco  # pylint: disable=import-outside-toplevel

        sim = UmiMujoco(args.umi_checkpoint, scene)
    elif args.policy_adapter in ("dwbc", "visual"):
        from benchmark.wbc.dwbc_mujoco import DwbcMujoco  # pylint: disable=import-outside-toplevel

        sim = DwbcMujoco(args.dwbc_root, args.dwbc_checkpoint, scene,
                         variant="visual" if args.policy_adapter == "visual" else "dwbc")
    elif args.policy_adapter == "wb_locoman":
        from benchmark.wbc.wb_locoman_mujoco import WbLocomanMujoco  # pylint: disable=import-outside-toplevel

        sim = WbLocomanMujoco(args.wb_locoman_root, args.wb_locoman_python, scene)
    elif args.policy_adapter == "ma2022":
        from benchmark.wbc.ma2022_mujoco import (  # pylint: disable=import-outside-toplevel
            Ma2022Mujoco, load_config,
        )

        sim = Ma2022Mujoco(
            args.ma2022_deployment_root, args.ma2022_policy,
            args.ma2022_env_config, scene,
            load_config(Path(args.ma2022_config).resolve()),
        )
    else:
        sim = RlSarMujoco(robot_dir, args.policy_key, scene, seed=args.seed)
    sim.model.opt.timestep = float(args.physics_dt)
    sim.substeps = max(1, round(sim.control_dt / sim.model.opt.timestep))

    site = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_SITE, "x5_ee")
    base = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    if site < 0 or base < 0:
        raise ValueError("MJCF must contain base_link and x5_ee")
    joints = [int(sim.model.actuator_trnid[sim.mapping[i], 0]) for i in range(sim.n)]
    qpos_adr = [int(sim.model.jnt_qposadr[joint]) for joint in joints]
    dof_adr = [int(sim.model.jnt_dofadr[joint]) for joint in joints]

    state = reference.task["initial_state"]
    root = np.asarray(state["root_state_env_local"], dtype=np.float64)
    dof_position = np.asarray(state["dof_position_rad"], dtype=np.float64)[: sim.n]
    dof_velocity = np.asarray(state["dof_velocity_rad_s"], dtype=np.float64)[: sim.n]
    mujoco.mj_resetData(sim.model, sim.data)
    sim.data.qpos[:3] = root[:3]
    sim.data.qpos[3:7] = np.roll(root[3:7], 1)
    sim.data.qvel[:6] = root[7:13]
    for i in range(sim.n):
        sim.data.qpos[qpos_adr[i]] = dof_position[i]
        sim.data.qvel[dof_adr[i]] = dof_velocity[i]
    mujoco.mj_forward(sim.model, sim.data)
    sim.history.fill(0.0)
    sim.actions.fill(0.0)
    if hasattr(sim, "reset_policy"):
        sim.reset_policy()
    sim.gait_indices = 0.0
    sim.command = [0.0] * 6

    policy_dt = sim.policy_dt
    requested_steps = min(
        int(math.ceil(float(reference.task["deadline_s"]) / policy_dt)),
        args.max_steps if args.max_steps > 0 else 2**31 - 1,
    )
    kp = np.asarray(sim.p["rl_kp"], dtype=np.float64)
    kd = np.asarray(sim.p["rl_kd"], dtype=np.float64)
    torque_limits = np.asarray(sim.p["torque_limits"], dtype=np.float64)
    q_target = dof_position.copy()
    trace: Dict[str, List[np.ndarray]] = {}
    arm_dofs = np.asarray(dof_adr[12:18], dtype=np.int32)
    arm_joints = joints[12:18]
    measured_arc = 0.0
    use_mpc = args.upper_controller == "floating_base_ocs2_mpc"
    transport = None
    viewer = None
    executed_path = []
    if use_mpc:
        from benchmark.wbc.controllers import (  # pylint: disable=import-outside-toplevel
            NativeOcs2Transport,
        )

        if abs(policy_dt - 0.02) > 1e-9:
            raise ValueError("native OCS2 MuJoCo testing requires a 50 Hz dog policy")
        transport = NativeOcs2Transport(
            Path(args.output).resolve() / "ocs2_runtime",
            Path(args.ocs2_root),
            timeout_s=args.ocs2_timeout_s,
            mode=args.ocs2_transport,
            command_timeout_s=args.ocs2_command_timeout_s,
            command_mode=args.ocs2_command_mode,
        )
        transport.reset_task()
        atexit.register(transport.close)
        reference_poses = reference.timed_poses()
    if args.viewer:
        from mujoco import viewer as mj_viewer  # pylint: disable=import-outside-toplevel

        viewer = mj_viewer.launch_passive(sim.model, sim.data)
        viewer.cam.distance = 2.1
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
    realtime_origin = time.monotonic()

    for step in range(requested_steps):
        if viewer is not None and not viewer.is_running():
            break
        q, dq, quat, gyro, base_pos, lin_vel = sim.read_state()
        reference_time = min((step + 1) * policy_dt, reference.duration)
        arc, goal_position, goal_quaternion = reference.at(reference_time)
        if hasattr(sim, "set_reference"):
            sim.set_reference(reference, reference_time)
        mpc_command = None
        base_feedforward = np.zeros(3, dtype=np.float64)
        if use_mpc:
            from benchmark.wbc.controllers import (  # pylint: disable=import-outside-toplevel
                encode_ocs2_state_values,
            )

            wire_sequence = transport.wire_sequence(step)
            state_payload = encode_ocs2_state_values(
                seq=wire_sequence,
                base_pos_world=sim.data.xpos[base],
                base_quat_xyzw=quat,
                base_lin_vel_body=lin_vel,
                base_ang_vel_body=gyro,
                arm_q=q[12:18],
                arm_dq=dq[12:18],
                leg_q=q[:12],
                leg_dq=dq[:12],
            )
            mpc_command = transport.exchange(
                state_payload,
                reference.tl_t if step == 0 else None,
                reference_poses if step == 0 else None,
            )
            base_feedforward = _set_mpc_dog_command(sim, mpc_command)
        actions = sim.forward(q, dq, quat, gyro, base_pos, lin_vel)
        q_target = sim.compute_output(actions)

        jacp = np.zeros((3, sim.model.nv), dtype=np.float64)
        jacr = np.zeros((3, sim.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(sim.model, sim.data, jacp, jacr, site)
        jacobian = np.vstack((jacp[:, arm_dofs], args.orientation_weight * jacr[:, arm_dofs]))
        current_rotation = sim.data.site_xmat[site].reshape(3, 3).copy()
        error = np.concatenate(
            (goal_position - sim.data.site_xpos[site], args.orientation_weight * _rotation_error(goal_quaternion, current_rotation))
        )
        normal = jacobian @ jacobian.T + args.ik_damping**2 * np.eye(6)
        if getattr(sim, "controls_arm", False):
            arm_q = q_target[12:18].copy()
            raw_step_norm = float(np.linalg.norm(arm_q - q[12:18]))
            saturated = False
            ik_valid = True
        elif getattr(sim, "direct_torque", False):
            arm_q = q[12:18].copy()
            raw_step_norm = 0.0
            saturated = False
            ik_valid = False
        elif use_mpc:
            arm_q = np.asarray(mpc_command["arm_q_cmd"], dtype=np.float64).copy()
            step_q = arm_q - q[12:18]
            raw_step_norm = float(np.linalg.norm(step_q))
            saturated = False
            ik_valid = bool(mpc_command["solver_ok"])
        else:
            ik_valid = True
            try:
                step_q = jacobian.T @ np.linalg.solve(normal, error)
                ik_valid = bool(np.isfinite(step_q).all())
            except np.linalg.LinAlgError:
                step_q, ik_valid = np.zeros(6), False
            raw_step_norm = float(np.linalg.norm(step_q)) if ik_valid else math.inf
            saturated = raw_step_norm > args.max_ik_step_rad
            if saturated:
                step_q *= args.max_ik_step_rad / max(raw_step_norm, 1e-12)
            if not ik_valid:
                step_q.fill(0.0)
            arm_q = q[12:18] + step_q
        margins = np.empty(6, dtype=np.float64)
        for i, joint in enumerate(arm_joints):
            low, high = sim.model.jnt_range[joint]
            arm_q[i] = np.clip(arm_q[i], low, high)
            margins[i] = min(arm_q[i] - low, high - arm_q[i]) / max(high - low, 1e-12)
        q_target[12:18] = arm_q

        leg_abs_energy = 0.0
        leg_positive_energy = 0.0
        push_impulse = np.zeros(3, dtype=np.float64)
        for _ in range(int(sim.p["decimation"])):
            for _ in range(sim.substeps):
                push_force = _push_force_at(push_events, float(sim.data.time))
                sim.data.xfrc_applied[base, :3] = push_force
                push_impulse += push_force * sim.model.opt.timestep
                q_now, dq_now, *_ = sim.read_state()
                torque = np.clip(
                    actions if getattr(sim, "direct_torque", False)
                    else kp * (q_target - q_now) - kd * dq_now,
                    -torque_limits, torque_limits,
                )
                for i in range(sim.n):
                    sim.data.ctrl[sim.mapping[i]] = torque[i]
                mujoco.mj_step(sim.model, sim.data)
                power = torque[:12] * dq_now[:12]
                leg_abs_energy += float(np.abs(power).sum()) * sim.model.opt.timestep
                leg_positive_energy += float(np.maximum(power, 0.0).sum()) * sim.model.opt.timestep
        sim.data.xfrc_applied[base, :3] = 0.0

        mujoco.mj_forward(sim.model, sim.data)
        q, dq, quat, gyro, base_pos, lin_vel = sim.read_state()
        actual_position = sim.data.site_xpos[site].copy()
        actual_quaternion = _quat_xyzw_from_matrix(sim.data.site_xmat[site].reshape(3, 3))
        foot_velocity, foot_force = _foot_observation(sim.model, sim.data)
        spatial_velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(sim.model, sim.data, mujoco.mjtObj.mjOBJ_SITE, site, spatial_velocity, 0)
        measured_arc, lateral_error = _forward_project(
            reference, measured_arc, actual_position, actual_quaternion
        )
        numerical_fault = not all(
            np.isfinite(value).all() for value in (sim.data.qpos, sim.data.qvel, actual_position, actual_quaternion)
        )
        base_rotation = sim.data.xmat[base].reshape(3, 3)
        upright = float(base_rotation[2, 2])
        fall = bool(base_pos[2] < 0.17 or upright < 0.5)

        jac_singular = np.linalg.svd(jacobian, compute_uv=False)
        rot_singular = np.linalg.svd(jacr[:, arm_dofs], compute_uv=False)
        manipulability = float(np.sqrt(max(np.linalg.det(jacobian @ jacobian.T), 0.0)))
        actual_state = np.concatenate((actual_position, actual_quaternion, spatial_velocity[3:], spatial_velocity[:3]))
        base_state = np.concatenate((sim.data.xpos[base], quat, sim.data.qvel[:3], sim.data.qvel[3:6]))

        for name, value, dtype in (
            ("sample_present", True, bool), ("metric_valid", not numerical_fault, bool),
            ("terminal_snapshot", False, bool), ("done", fall or numerical_fault, bool),
            ("timed_out", step + 1 == requested_steps, bool),
            ("trajectory_early_termination", False, bool),
            ("numerical_fault", numerical_fault, bool), ("fall", fall, bool),
            ("reference_time_s", reference_time, np.float32),
            ("reference_duration_s", reference.duration, np.float32),
            ("trajectory_progress", min(measured_arc / max(reference.gamma_s[-1], 1e-12), 1.0), np.float32),
            ("trajectory_lateral_error_m", lateral_error, np.float32),
            ("trajectory_timing_error_m", abs(measured_arc - arc), np.float32),
            ("goal_rho", 0.0, np.float32), ("goal_rho_valid", False, bool),
            ("manipulability", manipulability, np.float32),
            ("jacobian_sigma_min", jac_singular[-1], np.float32),
            ("rot_jacobian_sigma_min", rot_singular[-1], np.float32),
            ("ik_step_norm_rad", raw_step_norm if np.isfinite(raw_step_norm) else 0.0, np.float32),
            ("ik_step_saturated", saturated, bool), ("ik_solver_valid", ik_valid, bool),
            ("leg_abs_mechanical_energy_step_j", leg_abs_energy, np.float32),
            ("leg_positive_mechanical_energy_step_j", leg_positive_energy, np.float32),
        ):
            _add_scalar_column(trace, name, value, dtype)
        trace.setdefault("reference_ee_position_m", []).append(goal_position[None].astype(np.float32))
        trace.setdefault("reference_ee_quaternion_xyzw", []).append(goal_quaternion[None].astype(np.float32))
        trace.setdefault("actual_ee_state", []).append(actual_state[None].astype(np.float32))
        trace.setdefault("actual_ee_grasp_linear_velocity_mps", []).append(spatial_velocity[3:][None].astype(np.float32))
        trace.setdefault("environment_origin_m", []).append(np.zeros((1, 3), np.float32))
        trace.setdefault("base_root_state", []).append(base_state[None].astype(np.float32))
        trace.setdefault("dof_position_rad", []).append(q[None].astype(np.float32))
        trace.setdefault("dof_velocity_rad_s", []).append(dq[None].astype(np.float32))
        trace.setdefault("actuator_command", []).append(torque[None].astype(np.float32))
        trace.setdefault("joint_position_target_rad", []).append(q_target[None].astype(np.float32))
        trace.setdefault("policy_action", []).append(actions[None].astype(np.float32))
        trace.setdefault("actuator_torque_limit", []).append(torque_limits[None].astype(np.float32))
        trace.setdefault("base_feedforward_command", []).append(
            base_feedforward[None].astype(np.float32)
        )
        trace.setdefault("joint_limit_distance_fraction", []).append(margins[None].astype(np.float32))
        applied_push_force = push_impulse / policy_dt
        trace.setdefault("benchmark_push_active", []).append(
            np.asarray([bool(np.linalg.norm(push_impulse) > 0.0)], dtype=bool)
        )
        trace.setdefault("benchmark_push_force_world_n", []).append(
            applied_push_force[None].astype(np.float32)
        )
        trace.setdefault("foot_linear_velocity_mps", []).append(
            foot_velocity[None].astype(np.float32)
        )
        trace.setdefault("foot_contact_force_n", []).append(
            foot_force[None].astype(np.float32)
        )
        executed_path.append(actual_position.copy())
        if viewer is not None:
            _draw_trajectory(viewer, reference, executed_path, goal_position, sim.data.xpos[base])
        if (args.realtime or viewer is not None) and not (
            use_mpc and args.ocs2_transport == "async"
        ):
            time.sleep(max(0.0, realtime_origin + (step + 1) * policy_dt - time.monotonic()))
        if fall or numerical_fault:
            break

    if viewer is not None:
        viewer.close()
    if transport is not None:
        transport.close()
        atexit.unregister(transport.close)
    if hasattr(sim, "close"):
        sim.close()

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "trace.npz"
    arrays = {name: np.stack(values) for name, values in trace.items()}
    arrays.update(
        schema_version=np.asarray(TRACE_SCHEMA_VERSION),
        protocol_json=np.asarray(json.dumps(DEVELOPMENT_TIMED_TRAJECTORY_PROTOCOL, sort_keys=True, separators=(",", ":"))),
        kinematic_protocol_json=np.asarray(json.dumps(DEVELOPMENT_KINEMATIC_PROTOCOL, sort_keys=True, separators=(",", ":"))),
        task_id=np.asarray([reference.task["task_id"]]), control_dt_s=np.asarray(policy_dt, np.float64),
        requested_steps=np.asarray(requested_steps, np.int64),
        control_type=np.asarray("T" if getattr(sim, "direct_torque", False) else "P"),
        num_actions_loco=np.asarray(12, np.int64), num_actions_arm=np.asarray(6, np.int64),
        arm_action_mode=np.asarray(
            "direct_torque" if getattr(sim, "direct_torque", False) else args.upper_controller
        ), reach_model=np.asarray("not_available"),
        self_collision_observability=np.asarray("not_recorded"),
    )
    np.savez_compressed(trace_path, **arrays)
    result = score_trace_archive(trace_path)[0]
    resource_tree_hash, resource_file_count = _sha256_tree(scene.parent)
    observation_mirror = Path(__file__).resolve().parents[2] / "scripts" / "rl_sar_obs.py"
    sim_loop_mirror = Path(__file__).resolve().parents[2] / "scripts" / "sim2sim_mujoco.py"
    production_observation = rl_sar_root / "src" / "rl_sar" / "library" / "core" / "rl_sdk" / "rl_sdk.cpp"
    receipt = {
        "status": "complete" if len(trace["sample_present"]) else "failed",
        "backend": (
            "mujoco-python-umi-boundary-v1"
            if args.policy_adapter == "umi"
            else "mujoco-python-visual-wholebody-boundary-v1"
            if args.policy_adapter == "visual"
            else "mujoco-python-dwbc-boundary-v1"
            if args.policy_adapter == "dwbc"
            else "mujoco-python-wb-locoman-sidecar-v1"
            if args.policy_adapter == "wb_locoman"
            else "mujoco-python-ma2022-boundary-v1"
            if args.policy_adapter == "ma2022"
            else "mujoco-python-rl-sar-boundary-v1"
        ),
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "controller_adapter": (
            "umi_on_legs_learned_joint_targets"
            if args.policy_adapter == "umi"
            else "visual_wholebody_policy_plus_scripted_dls_ik_arm"
            if args.policy_adapter == "visual"
            else "deep_whole_body_control_learned_joint_targets"
            if args.policy_adapter == "dwbc"
            else
            "wb_locoman_fatrop_direct_torque"
            if args.policy_adapter == "wb_locoman"
            else
            "ma2022_recurrent_student_plus_scripted_dls_ik_arm"
            if args.policy_adapter == "ma2022"
            else "rl_sar_dog_policy_plus_native_floating_base_ocs2_mpc"
            if use_mpc else "rl_sar_dog_policy_plus_scripted_dls_ik_arm"
        ),
        "evidence_scope": (
            "closed_loop_mujoco_native_ocs2_implementation_test"
            if use_mpc
            else "implementation_smoke_not_learned_manipulation_policy"
        ),
        "initial_state_adapter": {
            "root": "isaac_xyzw_to_mujoco_wxyz",
            "joint_order": (
                "named_FL_FR_RL_RR_X5" if args.policy_adapter != "rl_sar"
                else "rl_sar_joint_mapping"
            ),
            "task_dof_count": len(state["dof_position_rad"]),
            "simulated_dof_count": sim.n,
            "ignored_task_dofs": max(0, len(state["dof_position_rad"]) - sim.n),
        },
        "task_id": reference.task["task_id"],
        "disturbance_schedule": reference.task["disturbance_schedule"],
        "suite_sha256": reference.group["suite_sha256"],
        "reference_archive_sha256": _sha256(reference.archive_path),
        "scene": {
            "path": str(scene),
            "sha256": _sha256(scene),
            "resource_root": str(scene.parent),
            "resource_tree_sha256": resource_tree_hash,
            "resource_file_count": resource_file_count,
        },
        "policy": ({
            "adapter": "umi_on_legs_official_actor",
            "model_sha256": _sha256(Path(args.umi_checkpoint)),
            "adapter_sha256": _sha256(Path(__file__).with_name("umi_mujoco.py")),
        } if args.policy_adapter == "umi" else {
            "adapter": ("visual_wholebody_checkpoint" if args.policy_adapter == "visual"
                        else "deep_whole_body_control_checkpoint"),
            "model_sha256": _sha256(Path(args.dwbc_checkpoint)),
            "adapter_sha256": _sha256(Path(__file__).with_name("dwbc_mujoco.py")),
        } if args.policy_adapter in ("dwbc", "visual") else {
            "adapter": "wb_locoman_fatrop_sidecar",
            "sidecar_sha256": _sha256(Path(args.wb_locoman_root) / "benchmark_sidecar.py"),
            "controller_sha256": _sha256(Path(args.wb_locoman_root) / "controller.py"),
            "solver_mode": "runtime_fatrop",
        } if args.policy_adapter == "wb_locoman" else {
            "adapter": "ma2022_recurrent_student",
            "model_sha256": _sha256(sim.policy_path),
            "env_config_sha256": _sha256(sim.env_config_path),
            "deployment_config_sha256": _sha256(sim.config_path),
            "deployment_adapter_sha256": _sha256(sim.adapter_source),
        } if args.policy_adapter == "ma2022" else {
            "adapter": "rl_sar",
            "key": args.policy_key,
            "model_sha256": _sha256(robot_dir / args.policy_key / "policy.pt"),
            "config_sha256": _sha256(robot_dir / args.policy_key / "config.yaml"),
            "base_config_sha256": _sha256(robot_dir / "base.yaml"),
        }),
        "observation_boundary": {
            "implementation": "python_mirror_of_rl_sar_compute_observation",
            "mirror_sha256": _sha256(observation_mirror),
            "simulation_loop_mirror_sha256": _sha256(sim_loop_mirror),
            "production_rl_sdk_sha256": (
                _sha256(production_observation) if production_observation.is_file() else None
            ),
            "production_rl_sdk_path": (
                str(production_observation) if production_observation.is_file()
                else "not_packaged_with_policy_bundle"
            ),
        },
        "mujoco_version": mujoco.__version__,
        "physics_dt_s": sim.model.opt.timestep,
        "policy_dt_s": policy_dt,
        "requested_steps": requested_steps,
        "recorded_steps": len(trace["sample_present"]),
        "trace": {"path": str(trace_path), "sha256": _sha256(trace_path), "schema_version": TRACE_SCHEMA_VERSION},
        "result": result,
    }
    if use_mpc:
        receipt["ocs2"] = {
            "root": str(Path(args.ocs2_root).resolve()),
            "transport": args.ocs2_transport,
            "command_mode": args.ocs2_command_mode,
            "complete_reference_knots": int(len(reference.tl_t)),
            "runtime_stats": transport.runtime_stats,
            "runner_source": {
                "path": str(transport.runner_source),
                "sha256": _sha256(transport.runner_source),
            },
            "runner_executable": {
                "path": str(transport.runner_executable),
                "sha256": _sha256(transport.runner_executable),
            },
        }
        receipt["trace_field_semantics"] = {
            "ik_solver_valid": "native OCS2 solver_ok compatibility field",
            "ik_step_norm_rad": "MPC absolute arm target minus measured arm position",
            "ik_step_saturated": "false; joint-limit clipping is recorded separately",
        }
    (output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return trace_path, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task-id")
    parser.add_argument("--rl-sar-root", required=True)
    parser.add_argument("--policy-key", required=True)
    parser.add_argument("--policy-adapter", choices=("rl_sar", "ma2022", "wb_locoman", "dwbc", "visual", "umi"), default="rl_sar")
    parser.add_argument("--ma2022-deployment-root")
    parser.add_argument("--ma2022-policy")
    parser.add_argument("--ma2022-env-config")
    parser.add_argument("--ma2022-config")
    parser.add_argument("--wb-locoman-root")
    parser.add_argument("--wb-locoman-python", default="/opt/miniconda3/envs/base312/bin/python")
    parser.add_argument("--dwbc-root")
    parser.add_argument("--dwbc-checkpoint")
    parser.add_argument("--umi-checkpoint")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--physics-dt", type=float, default=0.0025)
    parser.add_argument("--ik-damping", type=float, default=0.05)
    parser.add_argument("--orientation-weight", type=float, default=0.25)
    parser.add_argument("--max-ik-step-rad", type=float, default=0.08)
    parser.add_argument(
        "--upper-controller",
        choices=("scripted_dls_ik", "floating_base_ocs2_mpc"),
        default="scripted_dls_ik",
    )
    parser.add_argument("--ocs2-root", default="/home/simon/Projects/Simon/wbc_rl_mpc")
    parser.add_argument(
        "--ocs2-transport", choices=("synchronous", "async"), default="async"
    )
    parser.add_argument("--ocs2-timeout-s", type=float, default=90.0)
    parser.add_argument("--ocs2-command-timeout-s", type=float, default=0.5)
    parser.add_argument(
        "--ocs2-command-mode", choices=("full", "pose_only"), default="full"
    )
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    args = parser.parse_args()
    trace_path, receipt = run(args)
    print(json.dumps({"trace": str(trace_path), "receipt": str(Path(args.output).resolve() / "receipt.json"), "result": receipt["result"]}, indent=2))


if __name__ == "__main__":
    main()
