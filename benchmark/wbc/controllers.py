"""Explicit upper-controller adapters for the IsaacGym WBC benchmark."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import torch


UPPER_CONTROLLER_CONTRACT_VERSION = "wbc-upper-controller-v4"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class UpperControllerOutput:
    arm_actions: torch.Tensor
    plan_actions: Optional[torch.Tensor] = None
    base_velocity_body: Optional[torch.Tensor] = None
    body_posture: Optional[torch.Tensor] = None


def validate_upper_controller_output(output: UpperControllerOutput, n: int, n_arm: int, n_plan: int):
    """Fail fast on action width, units boundary and non-finite controller output."""
    if output.arm_actions.shape != (n, n_arm):
        raise ValueError(
            f"upper controller arm_actions must have shape {(n, n_arm)}, "
            f"got {tuple(output.arm_actions.shape)}"
        )
    tensors = [output.arm_actions]
    if output.plan_actions is not None:
        if output.plan_actions.shape != (n, n_plan):
            raise ValueError(
                f"upper controller plan_actions must have shape {(n, n_plan)}, "
                f"got {tuple(output.plan_actions.shape)}"
            )
        tensors.append(output.plan_actions)
    direct = output.base_velocity_body is not None or output.body_posture is not None
    if direct:
        if output.base_velocity_body is None or output.body_posture is None:
            raise ValueError("external base velocity and body posture must be supplied together")
        if output.base_velocity_body.shape != (n, 3):
            raise ValueError("base_velocity_body must be [vx_mps, vy_mps, yaw_rate_rad_s]")
        if output.body_posture.shape != (n, 3):
            raise ValueError("body_posture must be [height_m, pitch_rad, roll_rad]")
        tensors.extend((output.base_velocity_body, output.body_posture))
    if output.plan_actions is not None and direct:
        raise ValueError("an upper controller cannot use plan actions and direct commands together")
    if any(not torch.isfinite(value).all() for value in tensors):
        raise ValueError("upper controller output contains NaN or Inf")


class IkUpperController:
    """Pure DLS-IK arm controller plus the environment's deterministic base staging."""

    controller_id = "ik"
    required_arm_action_mode = "ik_residual"

    def __init__(self):
        adapter_source = Path(__file__).resolve()
        self.provenance = {
            "controller_id": self.controller_id,
            "contract_version": UPPER_CONTROLLER_CONTRACT_VERSION,
            "implementation": "WBCEnv._solve_arm_dls_ik_step + WBCEnv.plan base staging",
            "reference_frame": "environment_local_world",
            "base_command_frame": "body",
            "arm_command": "zero residual around DLS IK joint target",
            "update_rate": "one update per IsaacGym control step",
            "adapter_source": {
                "path": str(adapter_source),
                "sha256": _sha256_file(adapter_source),
            },
        }

    def reset(self, base, env_start: int, env_end: int, tasks: Sequence[Mapping[str, object]]):
        if base.arm_action_mode != self.required_arm_action_mode:
            raise ValueError(
                f"ik requires arm.action_mode={self.required_arm_action_mode}, "
                f"got {base.arm_action_mode}"
            )
        if len(tasks) != env_end - env_start:
            raise ValueError("IK controller reset task count does not match env slice")

    def step(self, base, env_start: int, env_end: int) -> UpperControllerOutput:
        n = env_end - env_start
        return UpperControllerOutput(
            arm_actions=torch.zeros(n, base.num_actions_arm, device=base.device),
            plan_actions=torch.zeros(n, base.num_plan_actions, device=base.device),
        )

    def close(self):
        pass


class _Header(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("version", ctypes.c_uint16),
        ("msg_id", ctypes.c_uint16),
        ("seq", ctypes.c_uint64),
        ("stamp", ctypes.c_double),
    ]


class _StateMsg(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ("h", _Header),
        ("base_pos_world", ctypes.c_float * 3),
        ("base_quat_wxyz", ctypes.c_float * 4),
        ("base_lin_vel_body", ctypes.c_float * 3),
        ("base_ang_vel_body", ctypes.c_float * 3),
        ("arm_q", ctypes.c_float * 6),
        ("arm_dq", ctypes.c_float * 6),
        ("leg_q", ctypes.c_float * 12),
        ("leg_dq", ctypes.c_float * 12),
        ("fsm_state_id", ctypes.c_uint8),
        ("bridge_enabled", ctypes.c_uint8),
        ("policy_ok", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8),
    ]


class _CmdMsg(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ("h", _Header),
        ("base_lin_vel_body_xy", ctypes.c_float * 2),
        ("base_ang_vel_body_z", ctypes.c_float),
        ("body_height_cmd", ctypes.c_float),
        ("body_pitch_cmd", ctypes.c_float),
        ("body_roll_cmd", ctypes.c_float),
        ("arm_q_cmd", ctypes.c_float * 6),
        ("arm_dq_cmd", ctypes.c_float * 6),
        ("gripper_cmd", ctypes.c_float),
        ("policy_time", ctypes.c_double),
        ("solve_time_ms", ctypes.c_float),
        ("solver_ok", ctypes.c_uint8),
        ("mode", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 2),
    ]


class _BenchmarkRequestHeader(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ("state", _StateMsg),
        ("controller_time_s", ctypes.c_double),
        ("reference_count", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


assert ctypes.sizeof(_StateMsg) == 224
assert ctypes.sizeof(_CmdMsg) == 116
assert ctypes.sizeof(_BenchmarkRequestHeader) == 240


def encode_ocs2_request(
    state_payload: bytes,
    controller_time_s: float,
    reference_times_s=None,
    reference_poses_xyz_xyzw=None,
    gait_phase_rad=None,
) -> bytes:
    """Pack one state request and, on task reset, the complete SE(3) reference."""
    state = _StateMsg.from_buffer_copy(state_payload)
    header = _BenchmarkRequestHeader()
    header.state = state
    header.controller_time_s = float(controller_time_s)
    if not np.isfinite(header.controller_time_s):
        raise ValueError("OCS2 controller time must be finite")
    trailer = b""
    if gait_phase_rad is not None:
        if not np.isfinite(gait_phase_rad):
            raise ValueError("OCS2 measured gait phase must be finite")
        header.reserved = 1  # optional little-endian float64 after reference data
        trailer = np.asarray([gait_phase_rad], dtype="<f8").tobytes()
    if reference_times_s is None and reference_poses_xyz_xyzw is None:
        return bytes(header) + trailer
    if reference_times_s is None or reference_poses_xyz_xyzw is None:
        raise ValueError("OCS2 reference times and poses must be supplied together")
    times = np.asarray(reference_times_s, dtype="<f8")
    poses = np.asarray(reference_poses_xyz_xyzw, dtype="<f4").copy()
    if times.ndim != 1 or poses.shape != (times.size, 7):
        raise ValueError("OCS2 full reference must have [K] times and [K,7] xyz+xyzw poses")
    if not 2 <= times.size <= 4096:
        raise ValueError("OCS2 full reference must contain between 2 and 4096 knots")
    if not np.isfinite(times).all() or not np.isfinite(poses).all():
        raise ValueError("OCS2 full reference contains NaN or Inf")
    if times[0] < 0.0 or np.any(np.diff(times) <= 0.0):
        raise ValueError("OCS2 full reference times must be nonnegative and strictly increasing")
    qnorm = np.linalg.norm(poses[:, 3:], axis=1)
    if np.any(qnorm < 1e-8):
        raise ValueError("OCS2 full reference contains a zero quaternion")
    poses[:, 3:] /= qnorm[:, None]
    header.reference_count = times.size
    return bytes(header) + times.tobytes(order="C") + poses.tobytes(order="C") + trailer


def encode_ocs2_state_values(
    *,
    seq: int,
    base_pos_world,
    base_quat_xyzw,
    base_lin_vel_body,
    base_ang_vel_body,
    arm_q,
    arm_dq,
    leg_q,
    leg_dq,
) -> bytes:
    """Encode simulator-neutral state arrays using the native bridge contract."""
    state = _StateMsg()
    state.h = _Header(0x57424301, 1, 1, int(seq), time.monotonic())

    def assign(field, values, width):
        array = np.asarray(values, dtype=np.float32).reshape(-1)
        if array.shape != (width,) or not np.isfinite(array).all():
            raise ValueError(f"OCS2 state field must be finite width {width}")
        getattr(state, field)[:] = array.tolist()

    assign("base_pos_world", base_pos_world, 3)
    assign("base_quat_wxyz", np.asarray(base_quat_xyzw)[[3, 0, 1, 2]], 4)
    assign("base_lin_vel_body", base_lin_vel_body, 3)
    assign("base_ang_vel_body", base_ang_vel_body, 3)
    assign("arm_q", arm_q, 6)
    assign("arm_dq", arm_dq, 6)
    assign("leg_q", leg_q, 12)
    assign("leg_dq", leg_dq, 12)
    state.fsm_state_id = 2
    state.bridge_enabled = 1
    state.policy_ok = 1
    return bytes(state)


def encode_ocs2_state(base, env_index: int, seq: int) -> bytes:
    """Encode one IsaacGym state using the native OCS2 bridge's wire contract."""
    if base.num_actions_loco != 12 or base.num_actions_arm != 6:
        raise ValueError("native OCS2 bridge requires 12 leg and 6 arm DOFs")
    root_local = base.root_states[env_index, :3] - base.env_origins[env_index]
    quat_xyzw = base.base_quat[env_index]
    arm = slice(base.num_actions_loco, base.num_actions_loco + base.num_actions_arm)
    return encode_ocs2_state_values(
        seq=seq,
        base_pos_world=root_local.detach().cpu().numpy(),
        base_quat_xyzw=quat_xyzw.detach().cpu().numpy(),
        base_lin_vel_body=base.base_lin_vel[env_index].detach().cpu().numpy(),
        base_ang_vel_body=base.base_ang_vel[env_index].detach().cpu().numpy(),
        arm_q=base.dof_pos[env_index, arm].detach().cpu().numpy(),
        arm_dq=base.dof_vel[env_index, arm].detach().cpu().numpy(),
        leg_q=base.dof_pos[env_index, :12].detach().cpu().numpy(),
        leg_dq=base.dof_vel[env_index, :12].detach().cpu().numpy(),
    )


def decode_ocs2_command(payload: bytes) -> dict:
    if len(payload) != ctypes.sizeof(_CmdMsg):
        raise ValueError(f"OCS2 command payload is {len(payload)} bytes, expected 116")
    command = _CmdMsg.from_buffer_copy(payload)
    if (command.h.magic, command.h.version, command.h.msg_id) != (0x57424301, 1, 2):
        raise ValueError("OCS2 command header does not match protocol v1 MSG_CMD")
    values = np.r_[
        command.base_lin_vel_body_xy,
        command.base_ang_vel_body_z,
        command.body_height_cmd,
        command.body_pitch_cmd,
        command.body_roll_cmd,
        command.arm_q_cmd,
        command.arm_dq_cmd,
    ].astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("OCS2 command contains NaN or Inf")
    return {
        "seq": int(command.h.seq),
        "solver_ok": bool(command.solver_ok),
        "mode": int(command.mode),
        "base_velocity_body": values[:3],
        "body_posture": values[3:6],
        "arm_q_cmd": values[6:12],
        "arm_dq_cmd": values[12:18],
        "policy_time": float(command.policy_time),
        "solve_time_ms": float(command.solve_time_ms),
    }


class NativeOcs2Transport:
    """Own a native C++ OCS2 process and its benchmark ZMQ transport."""

    def __init__(
        self,
        output_dir: Path,
        stack_root: Path,
        timeout_s: float = 90.0,
        mode: str = "synchronous",
        command_timeout_s: float = 0.5,
        command_mode: str = "full",
        task_profile: str = "legacy_benchmark",
        base_height_target: float = 0.3,
        task_file=None,
        arm_plan: bool = False,
    ):
        import zmq

        if mode not in ("synchronous", "async"):
            raise ValueError(f"unsupported native OCS2 transport: {mode}")
        if command_mode not in ("full", "pose_only"):
            raise ValueError(f"unsupported native OCS2 command mode: {command_mode}")
        if timeout_s <= 0.0 or command_timeout_s <= 0.0:
            raise ValueError("OCS2 timeouts must be positive")
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stack_root = Path(stack_root).resolve()
        self.timeout_s = float(timeout_s)
        self.command_timeout_s = float(command_timeout_s)
        self.mode = mode
        if task_profile not in ("native_ideal", "legacy_benchmark"):
            raise ValueError(f"unknown OCS2 task profile: {task_profile}")
        self.task_profile = task_profile
        self.task_file = Path(task_file).resolve() if task_file else None
        self.arm_plan = bool(arm_plan)
        if self.arm_plan and mode != "synchronous":
            raise ValueError("correlated Ma2022 arm plans require synchronous transport")
        self.base_height_target = float(base_height_target)
        self.command_mode = command_mode
        self.expected_command_mode = 2 if command_mode == "full" else 1
        self.control_dt_s = 0.02
        self.delay_compensation_s = 0.01
        token = hashlib.sha256(str(self.output_dir).encode()).hexdigest()[:12]
        self.context = zmq.Context()
        self.request = None
        self.state = None
        self.command = None
        if mode == "synchronous":
            self.request_endpoint = f"ipc:///tmp/roboduet_ocs2_sync_{token}"
            self.request = self.context.socket(zmq.REQ)
            self.request.setsockopt(zmq.LINGER, 0)
            self.request.bind(self.request_endpoint)
        else:
            self.state_endpoint = f"ipc:///tmp/roboduet_ocs2_state_{token}"
            self.command_endpoint = f"ipc:///tmp/roboduet_ocs2_cmd_{token}"
            self.state = self.context.socket(zmq.PUB)
            self.state.setsockopt(zmq.SNDHWM, 1)
            self.state.setsockopt(zmq.LINGER, 0)
            self.state.bind(self.state_endpoint)
            self.command = self.context.socket(zmq.SUB)
            self.command.setsockopt(zmq.CONFLATE, 1)
            self.command.setsockopt(zmq.SUBSCRIBE, b"")
            self.command.setsockopt(zmq.LINGER, 0)
            self.command.connect(self.command_endpoint)
        self.task_epoch = 0
        self.wall_origin_s = None
        self.last_command = None
        self.last_command_wall_s = None
        self.runtime_stats = {
            "task_initializations": 0,
            "commands_received": 0,
            "commands_reused": 0,
            "max_command_lag_steps": 0,
            "max_solve_time_ms": 0.0,
        }
        self.processes = []
        self._start_processes()

    def _start_processes(self):
        ros_setup = "/opt/ros/jazzy/setup.zsh"
        ws_setup = self.stack_root / "ros2_ws/install/setup.zsh"
        config = self.output_dir / "bridge.yaml"
        source_task = self.task_file or (
            self.stack_root
            / "go2_x5_ocs2"
            / "config"
            / "task_floating.info"
        )
        task = self.output_dir / "task_floating_benchmark.info"
        task_text = source_task.read_text(encoding="utf-8")
        runtime_task_text = task_text
        if self.task_profile == "legacy_benchmark":
            # Historical matrix-only modifications. Native reproduction must
            # neither apply nor require these substitutions to match.
            collision_marker = "selfCollision\n{\n  ; activate self-collision constraint\n  activate  true"
            if collision_marker not in task_text:
                raise ValueError("cannot locate selfCollision activation in OCS2 task")
            runtime_task_text = task_text.replace(collision_marker, collision_marker[:-4] + "false", 1)
            substitutions = (
                (r"(?m)^(\s*nThreads\s+)3(\s*)$", r"\g<1>1\g<2>", 0, 2),
                (r"(?m)^(\s*weight\s+)1\.0(\s*)$", r"\g<1>0.0\g<2>", 1, 1),
                (r"(?m)^(\s*muPosition\s+)10\.0(\s*)$", r"\g<1>50.0\g<2>", 0, 2),
                (r"(?m)^(\s*muOrientation\s+)5\.0(\s*)$", r"\g<1>25.0\g<2>", 0, 2),
            )
            for pattern, replacement, count, expected in substitutions:
                runtime_task_text, changed = re.subn(pattern, replacement, runtime_task_text, count=count)
                if changed != expected:
                    raise ValueError(f"legacy benchmark substitution {pattern}: expected {expected}, got {changed}")
        task.write_text(runtime_task_text, encoding="utf-8")
        self.source_task = source_task
        self.runtime_task = task
        self.bridge_config = config
        runner_name = (
            "wbc_benchmark_sync" if self.mode == "synchronous" else "wbc_benchmark_async"
        )
        self.runner_source = (
            self.stack_root / "go2_x5_ocs2_bridge" / "tools" / f"{runner_name}.cpp"
        )
        self.urdf_file = (
            self.stack_root / "go2_x5_description" / "urdf" / "arx5_ac1_floating.urdf"
        )
        kernel = (
            self.stack_root / "ros2_ws" / "install" / "ocs2_mobile_manipulator"
            / "lib" / "libocs2_mobile_manipulator.a"
        )
        digest = hashlib.sha256(
            f"roboduet-{self.mode}-v1".encode() + task.read_bytes() + self.urdf_file.read_bytes()
            + kernel.read_bytes()
        ).hexdigest()[:16]
        self.codegen_dir = Path("/tmp") / f"roboduet_ocs2_{self.mode}_{os.getuid()}" / digest
        executable = (
            self.stack_root / "ros2_ws" / "install" / "go2_x5_ocs2_bridge"
            / "lib" / "go2_x5_ocs2_bridge" / runner_name
        )
        self.runner_executable = executable
        if not executable.is_file():
            raise FileNotFoundError(
                f"native OCS2 runner is not built: {executable}; rebuild go2_x5_ocs2_bridge"
            )
        if self.mode == "synchronous":
            endpoint_args = shlex.quote(self.request_endpoint)
            implementation = "in_process_sqp_mrt_request_reply"
            endpoint_config = f"    requestEndpoint: \"{self.request_endpoint}\"\n"
        else:
            endpoint_args = (
                f"{shlex.quote(self.state_endpoint)} {shlex.quote(self.command_endpoint)}"
            )
            implementation = "independent_process_latest_state_latest_command"
            endpoint_config = (
                f"    stateEndpoint: \"{self.state_endpoint}\"\n"
                f"    cmdEndpoint: \"{self.command_endpoint}\"\n"
            )
        config.write_text(
            "/**:\n  ros__parameters:\n"
            f"    implementation: {implementation}\n"
            f"{endpoint_config}"
            f"    controlDt: {self.control_dt_s}\n"
            f"    commandMode: {self.command_mode}\n"
            f"    baseHeightTarget: {self.base_height_target}\n"
            f"    taskProfile: {self.task_profile}\n",
            encoding="utf-8",
        )
        commands = [
            (
                f"native_ocs2_{self.mode}",
                f"source {ros_setup} && source {ws_setup} && "
                f"{shlex.quote(str(executable))} {shlex.quote(str(task))} "
                f"{shlex.quote(str(self.codegen_dir))} {shlex.quote(str(self.urdf_file))} "
                f"{endpoint_args} {shlex.quote(self.command_mode)} "
                f"{self.base_height_target} {shlex.quote(self.task_profile)}",
            ),
        ]
        for name, command in commands:
            log = (self.output_dir / f"{name}.log").open("wb")
            process = subprocess.Popen(
                ["zsh", "-lc", command],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env={**{key: value for key, value in os.environ.items() if key != "WBC_BENCHMARK_ARM_PLAN"},
                     **({"WBC_BENCHMARK_ARM_PLAN": "1"} if self.arm_plan else {})},
            )
            self.processes.append((name, process, log))

    def reset_task(self):
        if self.mode != "async":
            return
        import zmq

        self.task_epoch += 1
        if self.task_epoch >= 2**32:
            raise OverflowError("OCS2 async task epoch exhausted")
        while self.command.poll(timeout=0, flags=zmq.POLLIN):
            self.command.recv(flags=zmq.DONTWAIT)
        self.last_command = None
        self.last_command_wall_s = None
        self.wall_origin_s = None
        self.runtime_stats["task_initializations"] += 1

    def wire_sequence(self, logical_sequence: int) -> int:
        if self.mode == "synchronous":
            return int(logical_sequence)
        if not 0 <= logical_sequence < 2**32:
            raise OverflowError("OCS2 async task step exceeds uint32")
        return (self.task_epoch << 32) | int(logical_sequence)

    def _check_processes(self):
        for name, process, _log in self.processes:
            if process.poll() is not None:
                raise RuntimeError(f"{name} exited with code {process.returncode}")

    def _accept_async_command(self, payload: bytes, expected_seq: int):
        decoded = decode_ocs2_command(payload)
        expected_task = expected_seq >> 32
        command_task = decoded["seq"] >> 32
        expected_step = expected_seq & 0xFFFFFFFF
        command_step = decoded["seq"] & 0xFFFFFFFF
        if command_task != expected_task or command_step > expected_step:
            return False
        if not decoded["solver_ok"] or decoded["mode"] != self.expected_command_mode:
            raise RuntimeError("OCS2 async command reports an invalid solver or command mode")
        if self.command_mode == "pose_only" and not np.allclose(
            decoded["base_velocity_body"][:2], 0.0, atol=1e-8
        ):
            raise RuntimeError("OCS2 pose_only command contains planar translation")
        self.last_command = decoded
        self.last_command_wall_s = time.monotonic()
        lag = expected_step - command_step
        self.runtime_stats["commands_received"] += 1
        self.runtime_stats["max_command_lag_steps"] = max(
            self.runtime_stats["max_command_lag_steps"], lag
        )
        self.runtime_stats["max_solve_time_ms"] = max(
            self.runtime_stats["max_solve_time_ms"], decoded["solve_time_ms"]
        )
        return True

    def _exchange_async(self, request: bytes, expected_seq: int) -> dict:
        import zmq

        logical_sequence = expected_seq & 0xFFFFFFFF
        if logical_sequence == 0:
            deadline = time.monotonic() + self.timeout_s
            while time.monotonic() < deadline:
                self._check_processes()
                self.state.send(request, flags=zmq.DONTWAIT)
                if self.command.poll(timeout=20, flags=zmq.POLLIN):
                    if self._accept_async_command(self.command.recv(), expected_seq):
                        self.wall_origin_s = time.monotonic()
                        return self.last_command
                time.sleep(0.01)
            raise TimeoutError("OCS2 async process did not acknowledge the complete trajectory")

        target_wall_s = self.wall_origin_s + logical_sequence * self.control_dt_s
        remaining_s = target_wall_s - time.monotonic()
        if remaining_s > 0.0:
            time.sleep(remaining_s)
        self._check_processes()
        self.state.send(request, flags=zmq.DONTWAIT)
        if self.command.poll(timeout=0, flags=zmq.POLLIN):
            self._accept_async_command(self.command.recv(flags=zmq.DONTWAIT), expected_seq)
        if self.last_command is None:
            raise RuntimeError("OCS2 async transport has no initialized command")
        age_s = time.monotonic() - self.last_command_wall_s
        if age_s > self.command_timeout_s:
            raise TimeoutError(
                f"OCS2 async command is stale ({age_s:.3f} s > {self.command_timeout_s:.3f} s)"
            )
        if self.last_command["seq"] != expected_seq:
            self.runtime_stats["commands_reused"] += 1
            lag = logical_sequence - (self.last_command["seq"] & 0xFFFFFFFF)
            self.runtime_stats["max_command_lag_steps"] = max(
                self.runtime_stats["max_command_lag_steps"], lag
            )
        return self.last_command

    def exchange(
        self,
        state_payload: bytes,
        reference_times_s=None,
        reference_poses_xyz_xyzw=None,
        gait_phase_rad=None,
    ) -> dict:
        import zmq

        state = _StateMsg.from_buffer_copy(state_payload)
        expected_seq = int(state.h.seq)
        logical_sequence = (
            expected_seq if self.mode == "synchronous" else expected_seq & 0xFFFFFFFF
        )
        controller_time_s = logical_sequence * self.control_dt_s
        request = encode_ocs2_request(
            state_payload,
            controller_time_s,
            reference_times_s,
            reference_poses_xyz_xyzw,
            gait_phase_rad=gait_phase_rad,
        )
        if self.mode == "async":
            return self._exchange_async(request, expected_seq)
        self.request.send(request)
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            self._check_processes()
            if self.request.poll(timeout=10, flags=zmq.POLLIN):
                payload = self.request.recv()
                if self.arm_plan:
                    width = ctypes.sizeof(_CmdMsg)
                    decoded = decode_ocs2_command(payload[:width])
                    plan = json.loads(payload[width:])
                    if (plan.get("schema") != "ma2022_arm_plan_v1" or not plan.get("valid")
                            or plan.get("seq") != expected_seq):
                        raise RuntimeError("invalid or uncorrelated native arm horizon")
                    for key in ("q", "dq", "ddq"):
                        values = np.asarray(plan[key], dtype=float)
                        if values.shape != (5, 6) or not np.isfinite(values).all():
                            raise RuntimeError(f"invalid native arm plan {key}")
                    decoded["arm_plan"] = plan
                else:
                    decoded = decode_ocs2_command(payload)
                correlated = decoded["seq"] == expected_seq
                synchronous_time = abs(
                    decoded["policy_time"]
                    - (controller_time_s + self.delay_compensation_s)
                ) <= 1e-9
                if (correlated and synchronous_time
                        and decoded["solver_ok"]
                        and decoded["mode"] == self.expected_command_mode):
                    self.runtime_stats["commands_received"] += 1
                    self.runtime_stats["task_initializations"] += int(logical_sequence == 0)
                    self.runtime_stats["max_solve_time_ms"] = max(self.runtime_stats["max_solve_time_ms"], decoded["solve_time_ms"])
                    return decoded
                raise RuntimeError("OCS2 reply violates the synchronous request contract")
            time.sleep(0.002)
        raise TimeoutError("OCS2 did not reply to the simulator request")

    def close(self):
        for _name, process, _log in reversed(self.processes):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for _name, process, log in reversed(self.processes):
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            log.close()
        for socket in (self.request, self.state, self.command):
            if socket is not None:
                socket.close(linger=0)
        self.context.term()


class FloatingBaseOcs2MpcController:
    controller_id = "floating_base_ocs2_mpc"
    required_arm_action_mode = "end_to_end"

    def __init__(self, transport, source_root: Path):
        self.transport = transport
        self.source_root = Path(source_root).resolve()
        self.sequence = 0
        is_async = self.transport.mode == "async"
        self.provenance = {
            "controller_id": self.controller_id,
            "contract_version": UPPER_CONTROLLER_CONTRACT_VERSION,
            "implementation": (
                "independent native C++ OCS2 SQP/MRT process over latest-value ZMQ"
                if is_async
                else "in-process native C++ OCS2 SQP/MRT request/reply runner"
            ),
            "controller_profile": f"native_ideal_{self.transport.command_mode}",
            "command_mode": self.transport.command_mode,
            "reference_frame": "environment_local_world",
            "base_position_source_frame": "IsaacGym base link origin",
            "ocs2_mount_offset_from_base_m": [0.05, 0.0, 0.10],
            "base_command_frame": "body",
            "base_command_units": ["m/s", "m/s", "rad/s", "m", "rad", "rad"],
            "arm_command": "absolute joint position radians",
            "update_rate": (
                "independent MPC updates from conflated latest state; IsaacGym paced at 50 Hz"
                if is_async
                else "one exchange per IsaacGym control step"
            ),
            "synchronization": {
                "mode": (
                    "wall_clock_latest_state_latest_command"
                    if is_async
                    else "simulator_time_request_response"
                ),
                "control_dt_s": self.transport.control_dt_s,
                "state_command_sequence_correlation": (
                    "same_task_latest_at_or_before_step" if is_async else "exact"
                ),
                "policy_observation_time_correlation": (
                    "asynchronous_cached_command" if is_async else "exact"
                ),
                "solver_threads": 1,
                "reference_delivery": "complete_task_trajectory_on_sequence_zero",
                "reference_time_frame": "task_relative_simulator_time",
                "steady_state_command_timeout_s": (
                    self.transport.command_timeout_s if is_async else None
                ),
            },
            "source_root": str(self.source_root),
            "adapter_source": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256_file(Path(__file__).resolve()),
            },
            "resources": {
                "source_task": {
                    "path": str(self.transport.source_task),
                    "sha256": _sha256_file(self.transport.source_task),
                },
                "runtime_task_benchmark_profile": {
                    "path": str(self.transport.runtime_task),
                    "sha256": _sha256_file(self.transport.runtime_task),
                },
                "bridge_config": {
                    "path": str(self.transport.bridge_config),
                    "sha256": _sha256_file(self.transport.bridge_config),
                },
                "runner_source": {
                    "path": str(self.transport.runner_source),
                    "sha256": _sha256_file(self.transport.runner_source),
                },
                "runner_executable": {
                    "path": str(self.transport.runner_executable),
                    "sha256": _sha256_file(self.transport.runner_executable),
                },
                "robot_urdf": {
                    "path": str(self.transport.urdf_file),
                    "sha256": _sha256_file(self.transport.urdf_file),
                },
            },
            "runtime_stats": self.transport.runtime_stats,
        }

    def reset(self, base, env_start: int, env_end: int, tasks: Sequence[Mapping[str, object]]):
        if env_end - env_start != 1:
            raise ValueError("native OCS2 adapter currently requires one sequential IsaacGym env")
        if len(tasks) != 1:
            raise ValueError("native OCS2 reset requires exactly one task")
        if base.arm_action_mode != self.required_arm_action_mode:
            raise ValueError(
                f"floating_base_ocs2_mpc requires arm.action_mode={self.required_arm_action_mode}, "
                f"got {base.arm_action_mode}"
            )
        if abs(float(base.dt) - 0.02) > 1e-8:
            raise ValueError("native OCS2 benchmark contract requires a 50 Hz (0.02 s) control step")
        if base.num_envs != 1 or env_start != 0:
            raise ValueError("native OCS2 full-reference extraction requires the sequential env 0")
        tb = base.traj_batch
        duration = float(tb.T[env_start].item())
        count = int((tb.tl_t[env_start] < duration - 1e-7).sum().item()) + 1
        count = max(2, min(count, tb.max_tl_points))
        times = tb.tl_t[env_start, :count]
        arc_positions = tb.tl_s[:, :count]
        positions = tb.p_at(arc_positions)[env_start]
        quaternions = tb.quat_at(arc_positions)[env_start]
        origin = base.env_origins[env_start]
        positions = positions - origin.unsqueeze(0)
        self.reference_times_s = times.detach().cpu().numpy().astype(np.float64)
        self.reference_poses_xyz_xyzw = (
            torch.cat((positions, quaternions), dim=-1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        self.sequence = 0
        self.transport.reset_task()

    def step(self, base, env_start: int, env_end: int) -> UpperControllerOutput:
        sequence = self.sequence
        state_payload = encode_ocs2_state(
            base, env_start, self.transport.wire_sequence(sequence)
        )
        command = self.transport.exchange(
            state_payload,
            self.reference_times_s if sequence == 0 else None,
            self.reference_poses_xyz_xyzw if sequence == 0 else None,
        )
        self.sequence += 1
        q_default = base.default_dof_pos[
            :, base.num_actions_loco : base.num_actions_loco + base.num_actions_arm
        ][0]
        arm_scale = float(base.cfg.arm.end_to_end.action_scale)
        arm_action = (
            torch.as_tensor(command["arm_q_cmd"], device=base.device, dtype=q_default.dtype)
            - q_default
        ) / arm_scale
        return UpperControllerOutput(
            arm_actions=arm_action.view(1, -1),
            base_velocity_body=torch.as_tensor(
                command["base_velocity_body"], device=base.device, dtype=q_default.dtype
            ).view(1, 3),
            body_posture=torch.as_tensor(
                command["body_posture"], device=base.device, dtype=q_default.dtype
            ).view(1, 3),
        )

    def close(self):
        self.transport.close()
