"""WBC env/policy loading, extended Accumulator, and eval loop.

Shares the same config-loading and env-override path as the dog-only benchmark
so layout compatibility grouping and the HTML report pipeline work without
duplication.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import isaacgym  # noqa: F401 - must precede torch
import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_apply, quat_conjugate, quat_mul, quat_rotate_inverse

from benchmark.dog_policy.evaluation import (
    Accumulator,
    CRITICAL_COMPAT_CFG_PATHS,
    _cfg_value,
    _read_cfg_dict,
    _load_cfg_from_pkl,
    _apply_benchmark_env_overrides,
)
from benchmark.wbc.scoring import (
    DEVELOPMENT_KINEMATIC_PROTOCOL,
    DEVELOPMENT_TIMED_TRAJECTORY_PROTOCOL,
    aggregate_task_events,
    timed_trajectory_success,
)
from go1_gym.envs.config import configure_privileged_obs_dims
from go1_gym.envs.roboduet.wbc_env import WBCEnv, dog_cmd_idx
from go1_gym.envs.roboduet.wbc_env_wrapper import HistoryWrapper


# ---------------------------------------------------------------------------


class BenchmarkWBCEnv(WBCEnv):
    """WBC task with reset-safe terminal snapshots for benchmark scoring."""

    _TERMINAL_TENSOR_SOURCES = {
        "end_effector_state": "end_effector_state",
        "goal_pos_world": "arm_goal_pos_world",
        "goal_quat_world": "arm_goal_quat_world",
        "goal_rho": "goal_rho",
        "goal_rho_valid": "goal_rho_valid",
        "traj_d_lat": "traj_d_lat",
        "traj_timing_err": "traj_timing_err",
        "traj_sdot_meas": "traj_sdot_meas",
        "traj_s": "traj_s",
        "traj_length": "traj_batch.L",
        "traj_duration": "traj_batch.T",
        "traj_sim_time": "traj_sim_time",
        "root_state": "root_states",
        "dof_position": "dof_pos",
        "dof_velocity": "dof_vel",
        "torques": "torques",
        "joint_position_target": "joint_pos_target",
        "actions": "actions",
        "contact_forces": "contact_forces",
        "foot_velocities": "foot_velocities",
        "leg_abs_energy_j": "step_locomotion_abs_energy_j",
        "leg_positive_energy_j": "step_locomotion_positive_energy_j",
        "manipulability": "goal_manipulability",
        "jacobian_sigma_min": "goal_jacobian_sigma_min",
        "rot_jacobian_sigma_min": "goal_rot_jacobian_sigma_min",
        "joint_limit_distance": "goal_joint_limit_distance",
        "ik_step_norm_rad": "goal_ik_step_norm_rad",
        "ik_step_saturated": "goal_ik_step_saturated",
        "ik_solver_valid": "goal_ik_solver_valid",
        "base_feedforward_cmd": "base_feedforward_cmd",
        "time_out": "time_out_buf",
        "traj_early_term": "traj_early_term",
        "numerical_fault": "numerical_fault_mask",
        "benchmark_push_active": "benchmark_push_active",
        "benchmark_push_force_world_n": "benchmark_push_force_world_n",
    }

    @staticmethod
    def _resolve_tensor_source(env, path):
        value = env
        for part in path.split("."):
            value = getattr(value, part)
        return value

    def _ensure_terminal_snapshot_buffers(self):
        if hasattr(self, "benchmark_terminal_snapshot"):
            return
        self._ensure_benchmark_push_buffers()
        snapshot = {
            "valid": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        }
        for name, source in self._TERMINAL_TENSOR_SOURCES.items():
            value = self._resolve_tensor_source(self, source)
            snapshot[name] = torch.empty_like(value[: self.num_envs])
        self.benchmark_terminal_snapshot = snapshot

    def clear_benchmark_terminal_snapshots(self, env_ids=None):
        self._ensure_terminal_snapshot_buffers()
        if env_ids is None:
            self.benchmark_terminal_snapshot["valid"].zero_()
        else:
            self.benchmark_terminal_snapshot["valid"][env_ids] = False

    def _arm_pre_reset_capture_hook(self, env_ids):
        if len(env_ids) == 0:
            return
        self._ensure_terminal_snapshot_buffers()
        snapshot = self.benchmark_terminal_snapshot
        for name, source in self._TERMINAL_TENSOR_SOURCES.items():
            value = self._resolve_tensor_source(self, source)
            snapshot[name][env_ids] = value[env_ids]
        snapshot["valid"][env_ids] = True

    def _ensure_benchmark_push_buffers(self):
        if hasattr(self, "benchmark_push_forces"):
            return
        if hasattr(self, "rigid_body_state"):
            shape_source = self.rigid_body_state[:, :3].reshape(
                self.num_envs, self.num_bodies, 3
            )
        else:
            # Supports lightweight contract fixtures; a real env always owns
            # rigid_body_state before the first reset capture.
            shape_source = self.contact_forces
        self.benchmark_push_forces = torch.zeros_like(shape_source)
        self.benchmark_push_torques = torch.zeros_like(self.benchmark_push_forces)
        self.benchmark_push_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.benchmark_push_force_world_n = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device
        )

    def set_benchmark_push(self, env_ids, force_world_n):
        """Set the deterministic base-COM force for the next control step."""
        self._ensure_benchmark_push_buffers()
        self.benchmark_push_forces.zero_()
        self.benchmark_push_torques.zero_()
        self.benchmark_push_active.zero_()
        self.benchmark_push_force_world_n.zero_()
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
        if not len(env_ids):
            return
        force = torch.as_tensor(
            force_world_n, device=self.device, dtype=self.benchmark_push_forces.dtype
        )
        if force.shape != (len(env_ids), 3) or not torch.isfinite(force).all():
            raise ValueError("benchmark push force must be finite with shape (n_envs, 3)")
        self.benchmark_push_forces[env_ids, 0] = force
        self.benchmark_push_active[env_ids] = torch.linalg.vector_norm(force, dim=-1) > 0
        self.benchmark_push_force_world_n[env_ids] = force

    def _arm_decimation_hook(self):
        super()._arm_decimation_hook()
        self._apply_workspace_fixed_base_constraint()
        self._ensure_benchmark_push_buffers()
        if not bool(self.benchmark_push_active.any().item()):
            return
        ok = self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.benchmark_push_forces),
            gymtorch.unwrap_tensor(self.benchmark_push_torques),
            gymapi.ENV_SPACE,
        )
        if not ok:
            raise RuntimeError("IsaacGym rejected deterministic benchmark push")

    def set_workspace_probe_goal(self, env_ids, position_world, quaternion_world):
        """Install a static workspace target without changing training config."""
        if not hasattr(self, "workspace_probe_active"):
            self.workspace_probe_active = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self.workspace_probe_position_world = torch.zeros(
                self.num_envs, 3, device=self.device
            )
            self.workspace_probe_quaternion_world = torch.zeros(
                self.num_envs, 4, device=self.device
            )
            self.workspace_probe_quaternion_world[:, 3] = 1.0
            self.workspace_fixed_base_active = torch.zeros_like(
                self.workspace_probe_active
            )
            self.workspace_fixed_root_state = torch.zeros_like(self.root_states)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
        position = torch.as_tensor(
            position_world, device=self.device, dtype=self.root_states.dtype
        )
        quaternion = torch.as_tensor(
            quaternion_world, device=self.device, dtype=self.root_states.dtype
        )
        if position.shape != (len(env_ids), 3) or quaternion.shape != (len(env_ids), 4):
            raise ValueError("workspace goal must have position (n,3) and xyzw quaternion (n,4)")
        if not torch.isfinite(position).all() or not torch.isfinite(quaternion).all():
            raise ValueError("workspace goal must be finite")
        norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
        if bool((norm < 1e-8).any().item()):
            raise ValueError("workspace goal quaternion has zero norm")
        self.workspace_probe_active[env_ids] = True
        self.workspace_probe_position_world[env_ids] = position
        self.workspace_probe_quaternion_world[env_ids] = quaternion / norm
        self._apply_workspace_probe_goal()

    def set_workspace_fixed_base(self, env_ids, enabled=True):
        if not hasattr(self, "workspace_probe_active"):
            raise RuntimeError("set workspace goal before enabling its base constraint")
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
        self.workspace_fixed_base_active[env_ids] = bool(enabled)
        if enabled:
            self.workspace_fixed_root_state[env_ids] = self.root_states[env_ids]
            self.workspace_fixed_root_state[env_ids, 7:13] = 0.0

    def clear_workspace_probe(self):
        if hasattr(self, "workspace_probe_active"):
            self.workspace_probe_active.zero_()
            self.workspace_fixed_base_active.zero_()

    def _apply_workspace_fixed_base_constraint(self):
        if not hasattr(self, "workspace_fixed_base_active"):
            return
        env_ids = self.workspace_fixed_base_active.nonzero(as_tuple=False).flatten()
        if not len(env_ids):
            return
        self.root_states[env_ids] = self.workspace_fixed_root_state[env_ids]
        ids_i32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(ids_i32),
            len(ids_i32),
        )
        self.base_pos[env_ids] = self.root_states[env_ids, :3]
        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.base_lin_vel[env_ids] = 0.0
        self.base_ang_vel[env_ids] = 0.0
        self.projected_gravity[env_ids] = quat_rotate_inverse(
            self.base_quat[env_ids], self.gravity_vec[env_ids]
        )

    def _apply_workspace_probe_goal(self):
        if not hasattr(self, "workspace_probe_active"):
            return
        env_ids = self.workspace_probe_active.nonzero(as_tuple=False).flatten()
        if not len(env_ids):
            return
        self.arm_goal_pos_world[env_ids] = self.workspace_probe_position_world[env_ids]
        self.arm_goal_quat_world[env_ids] = self.workspace_probe_quaternion_world[env_ids]
        self._update_ee_task_space_error()
        self._update_goal_reaching_diagnostics()

    def _arm_post_physics_hook(self):
        self._apply_workspace_fixed_base_constraint()
        super()._arm_post_physics_hook()
        # WBCEnv advances its trajectory reference in this hook. Workspace
        # probes replace that reference with their immutable static target.
        self._apply_workspace_probe_goal()


# ---------------------------------------------------------------------------
# WBC-specific env loading
# ---------------------------------------------------------------------------

WBC_COMPAT_PATHS = [
    "sim.dt",
    "sim.substeps",
    "control.control_type",
    "control.action_scale",
    "control.hip_scale_reduction",
    "control.decimation",
    "dog.control.stiffness_leg",
    "dog.control.damping_leg",
    "arm.control.stiffness_arm",
    "arm.control.damping_arm",
    "arm.arm_num_observations",
    "arm.arm_num_observation_history",
    "arm.num_actions_arm_cd",
    "arm.arm_num_privileged_obs",
    "arm.action_mode",
    "arm.use_adaptation_module",
    "env.arm_observe_dog_state",
    "arm.end_to_end.action_scale",
    "arm.waypoint.anchor",
    "arm.waypoint.pos_scale",
    "arm.waypoint.rot_scale",
    "arm.ik.damping",
    "arm.ik.step_gain",
    "arm.ik.max_step_rad",
    "arm.ik.residual_scale",
    "arm.ik.pos_weight",
    "arm.ik.rot_weight",
    "arm.ik.ee_local_pos",
    "arm.checkpoint_observation_layout",
    "wbc.goal_reaching.target_mode",
    "wbc.goal_reaching.rho_star",
    "wbc.goal_reaching.rho_lo",
    "wbc.goal_reaching.rho_hi",
    "wbc.goal_reaching.reach_table_path",
    "wbc.goal_reaching.delta_vel_limit",
    "wbc.goal_reaching.response_time_s",
    "wbc.goal_reaching.base_nom_filter_hz",
    "wbc.goal_reaching.command_smoothing_alpha",
    "wbc.goal_reaching.bypass_post_processing",
    "wbc.goal_reaching.posture_rate_limit",
    "wbc.goal_reaching.gait_rate_limit",
    "wbc.goal_reaching.high_speed_posture_scale",
    "wbc.goal_reaching.high_speed_threshold",
    "wbc.goal_reaching.trajectory.preview_points",
    "wbc.goal_reaching.trajectory.preview_horizon",
    "wbc.goal_reaching.trajectory.max_gamma_points",
    "wbc.goal_reaching.trajectory.max_tl_points",
    "wbc.goal_reaching.trajectory.update_s_window",
    "wbc.goal_reaching.trajectory.n_levels_A",
    "wbc.goal_reaching.trajectory.n_levels_B",
    "wbc.goal_reaching.trajectory.anchor_offset_body",
    "wbc.goal_reaching.command_channels",
    "commands.limit_vel_x",
    "commands.limit_vel_y",
    "commands.limit_vel_yaw",
    "commands.limit_body_height",
    "commands.limit_body_pitch",
    "commands.limit_body_roll",
    "commands.limit_gait_frequency",
    "commands.limit_stance_width",
    "commands.limit_stance_length",
]


WBC_SHARED_ENV_COMPAT_PATHS = tuple(
    dict.fromkeys(list(CRITICAL_COMPAT_CFG_PATHS) + WBC_COMPAT_PATHS)
)


def _freeze_compat_value(value):
    if isinstance(value, dict):
        return tuple(
            sorted((key, _freeze_compat_value(item)) for key, item in value.items())
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_compat_value(item) for item in value)
    return value


def wbc_shared_env_compatibility_key(logdir: str):
    cfg_dict = _read_cfg_dict(logdir)
    return tuple(
        _freeze_compat_value(_cfg_value(cfg_dict, path))
        for path in WBC_SHARED_ENV_COMPAT_PATHS
    )


def group_wbc_shared_env_compatible_runs(logdirs: List[str]) -> List[List[int]]:
    groups: List[List[int]] = []
    key_to_group = {}
    for index, logdir in enumerate(logdirs):
        key = wbc_shared_env_compatibility_key(logdir)
        group_index = key_to_group.get(key)
        if group_index is None:
            group_index = len(groups)
            key_to_group[key] = group_index
            groups.append([])
        groups[group_index].append(index)
    return groups


def describe_wbc_shared_env_group(logdir: str) -> Dict[str, object]:
    cfg_dict = _read_cfg_dict(logdir)
    return {path: _cfg_value(cfg_dict, path) for path in WBC_SHARED_ENV_COMPAT_PATHS}


def load_wbc_env_benchmark(
    logdir: str,
    total_envs: int,
    envs_per_policy: int,
    headless: bool = True,
    device: str = "cuda:0",
    robot: Optional[str] = None,
    bank_seed: int = 12345,
    bank_per_cell: Optional[int] = None,
    record_video: bool = False,
    arm_action_mode: Optional[str] = None,
    initial_arm_joint_positions: Optional[Sequence[float]] = None,
):
    """Build the checkpoint's env at ``total_envs`` with evaluation overrides.

    The bank is reseeded by default so the benchmark scores held-out
    trajectories. ``bank_per_cell`` shrinks the pre-generated trajectory bank
    (CPU/scipy-bound at construction time).
    """
    cfg = _load_cfg_from_pkl(logdir, robot=robot)
    if arm_action_mode is not None:
        cfg.arm.action_mode = str(arm_action_mode)
    if initial_arm_joint_positions is not None:
        values = tuple(float(value) for value in initial_arm_joint_positions)
        joint_names = tuple(f"x5_joint{index}" for index in range(1, 7))
        if len(values) != len(joint_names):
            raise ValueError("initial arm joint positions must contain six radians values")
        defaults = cfg.init_state.default_joint_angles
        missing = [name for name in joint_names if name not in defaults]
        if missing:
            raise ValueError(
                "system benchmark arm home requires Go2-X5 joints; missing "
                + ", ".join(missing)
            )
        for name, value in zip(joint_names, values):
            defaults[name] = value
    _apply_benchmark_env_overrides(cfg, total_envs, envs_per_policy)
    cfg.env.record_video = bool(record_video)
    if record_video:
        cfg.env.recording_overlay_trajectory = True
        cfg.env.recording_overlay_text = True

    # Candidate slices start from identical nominal reset states. Per-env asset
    # buckets still repeat with envs_per_policy, preserving paired mount props.
    for attr in (
        "x_init_range",
        "y_init_range",
        "z_init_range",
        "yaw_init_range",
        "roll_init_range",
        "pitch_init_range",
    ):
        if hasattr(cfg.terrain, attr):
            setattr(cfg.terrain, attr, 0.0)
    cfg.terrain.reset_curriculum = False
    cfg.env.stage1_arm_init_dof_pos_noise = 0.0
    for stage_name in ("stage1_arm", "stage2_arm"):
        stage = getattr(cfg.domain_rand, stage_name, None)
        if stage is not None:
            for attr in (
                "randomize_Kp_factor",
                "randomize_Kd_factor",
                "randomize_motor_strength",
                "randomize_motor_offset",
                "randomize_ee_payload",
                "randomize_link_mass",
                "randomize_link_com",
            ):
                if hasattr(stage, attr):
                    setattr(stage, attr, False)
    cfg.domain_rand.randomize_mount_position = False
    cfg.domain_rand.randomize_mount_rotation = False
    # The benchmark owns its per-task deadline through n_eval_steps. Saved
    # training configs usually retain a 20 s episode horizon, which can be
    # shorter than a bounded long-path time law and would otherwise create an
    # unrelated simulator timeout before the frozen TaskSpec deadline.
    cfg.env.episode_length_s = 10000

    mode = str(getattr(cfg.wbc.goal_reaching, "target_mode", "static"))
    if mode != "trajectory":
        raise ValueError(
            f"{logdir} is not a trajectory-tracking run (target_mode='{mode}'). "
            f"Use --dog_only for locomotion-only runs."
        )

    cfg.wbc.goal_reaching.trajectory.bank_seed = int(bank_seed)
    # Old trajectory checkpoints commonly persist the previous 1536-point
    # capacity. Benchmark references now include 5 m planar travel, a 0--1.5 m
    # ground-relative height range and high-curvature segments; keeping the
    # old value would silently truncate the hardest tasks during bank load.
    cfg.wbc.goal_reaching.trajectory.max_gamma_points = max(
        2048, int(cfg.wbc.goal_reaching.trajectory.max_gamma_points)
    )
    if bank_per_cell is not None:
        cfg.wbc.goal_reaching.trajectory.bank_per_cell = int(bank_per_cell)

    configure_privileged_obs_dims(cfg)
    env = BenchmarkWBCEnv(sim_device=device, headless=headless, cfg=cfg)
    return HistoryWrapper(env), cfg


def load_wbc_policies(logdir: str, ckpt_id: str, cfg, device: str = "cuda:0"):
    """(dog_policy, arm_policy), both resident on ``device``."""
    from scripts.load_policy import load_arm_policy, load_dog_policy

    ckpt_id = "last" if str(ckpt_id) == "last" else str(ckpt_id).zfill(6)
    dog_policy = load_dog_policy(logdir, ckpt_id, cfg, device=device)
    arm_policy = load_arm_policy(logdir, ckpt_id, cfg, device=device)
    return dog_policy, arm_policy


# ---------------------------------------------------------------------------
# WBC Accumulator — extends the dog-only one with SE(3) tracking, rho, energy
# ---------------------------------------------------------------------------


class WBCAccumulator(Accumulator):
    """Per-policy Welford accumulator for WBC metrics.

    Extends the dog-only Accumulator with EE tracking error, reach utilisation,
    motor power, and the velocity history needed to compute finite-difference
    acceleration and jerk offline.
    """

    def __init__(
        self, num_envs: int, n_steps: int, device: str, record_trace: bool = False
    ):
        super().__init__(num_envs, device)
        self.n_steps = n_steps
        self.exposure_steps = torch.zeros(num_envs, device=device)
        self.episode_count = torch.ones(num_envs, device=device)
        self.final_progress = torch.zeros(num_envs, device=device)
        self.final_reference_time_s = torch.zeros(num_envs, device=device)
        self.reference_duration_s = torch.zeros(num_envs, device=device)
        self.final_ee_pos_error = torch.full((num_envs,), torch.nan, device=device)
        self.final_ee_rot_error = torch.full((num_envs,), torch.nan, device=device)
        self.tracking_tube_steps = torch.zeros(num_envs, device=device)
        self.endpoint_hold_steps = torch.zeros(num_envs, device=device)
        self.completion_time_s = torch.full((num_envs,), torch.nan, device=device)
        self.terminal_samples = torch.zeros(num_envs, device=device)
        self.protocol = dict(DEVELOPMENT_TIMED_TRAJECTORY_PROTOCOL)
        self.kinematic_protocol = dict(DEVELOPMENT_KINEMATIC_PROTOCOL)
        self.record_trace = bool(record_trace)
        self._trace = {} if self.record_trace else None
        self.trace_control_type = None
        self.trace_num_actions_loco = None
        self.trace_num_actions_arm = None
        self.trace_arm_action_mode = None
        self.trace_reach_model = None
        self.trace_self_collision_observability = None
        # Full velocity vectors are retained so direction changes contribute to
        # acceleration and jerk. A validity mask prevents finite differences
        # from crossing a terminal/reset boundary.
        self._ee_velocities: List[torch.Tensor] = []
        self._base_velocities: List[torch.Tensor] = []
        self._arm_dof_velocities: List[torch.Tensor] = []
        self._ee_position_errors: List[torch.Tensor] = []
        self._ee_rotation_errors: List[torch.Tensor] = []
        self._base_positions: List[torch.Tensor] = []
        self._leg_torques: List[torch.Tensor] = []
        self._leg_torque_limits: List[torch.Tensor] = []
        self._foot_slip_speeds: List[torch.Tensor] = []
        self._foot_contacts: List[torch.Tensor] = []
        self._smoothness_valid: List[torch.Tensor] = []

    @staticmethod
    def _masked(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.where(mask.to(device=value.device), value, torch.zeros_like(value))

    def _add_sq(self, key: str, sq: torch.Tensor, mask: torch.Tensor):
        self._get(f"{key}_sq").add_(self._masked(sq, mask))
        self._get(f"{key}_samples").add_(mask.float())

    def _add_val(self, key: str, value: torch.Tensor, mask: torch.Tensor):
        self._get(f"{key}_sum").add_(self._masked(value, mask))
        self._get(f"{key}_sq").add_(self._masked(value.pow(2), mask))
        self._get(f"{key}_samples").add_(mask.float())

    def _add_count(self, key: str, value: torch.Tensor, mask: torch.Tensor):
        self._get(key).add_(self._masked(value.float(), mask))

    def _add_sum(self, key: str, value: torch.Tensor, mask: torch.Tensor):
        self._get(key).add_(self._masked(value, mask))

    def _add_max(self, key: str, value: torch.Tensor, mask: torch.Tensor):
        stat_key = f"{key}_max"
        if stat_key not in self._stats:
            self._stats[stat_key] = torch.full(
                (self.n,), -torch.inf, device=self.dev, dtype=value.dtype
            )
        candidate = torch.where(mask, value, torch.full_like(value, -torch.inf))
        torch.maximum(self._stats[stat_key], candidate, out=self._stats[stat_key])

    def _add_min(self, key: str, value: torch.Tensor, mask: torch.Tensor):
        stat_key = f"{key}_min"
        if stat_key not in self._stats:
            self._stats[stat_key] = torch.full(
                (self.n,), torch.inf, device=self.dev, dtype=value.dtype
            )
        candidate = torch.where(mask, value, torch.full_like(value, torch.inf))
        torch.minimum(self._stats[stat_key], candidate, out=self._stats[stat_key])

    def _trace_step(self, **values):
        if not self.record_trace:
            return
        for key, value in values.items():
            if not torch.is_tensor(value):
                value = torch.as_tensor(value, device=self.dev)
            self._trace.setdefault(key, []).append(value.detach().clone())

    def trace_tensors(self, count=None):
        """Return time-major tensors suitable for a backend-neutral archive."""
        if not self.record_trace:
            raise RuntimeError("raw trace recording was not enabled")
        width = self.n if count is None else int(count)
        return {
            key: torch.stack(values, dim=0)[:, :width].detach().cpu()
            for key, values in self._trace.items()
        }

    def _indices(self, indices=None):
        if indices is None:
            return torch.arange(self.n, device=self.dev, dtype=torch.long)
        return torch.as_tensor(indices, device=self.dev, dtype=torch.long)

    def _maximum(self, key: str, indices=None):
        values = self._stats.get(f"{key}_max")
        if values is None:
            return None
        values = values[self._indices(indices)]
        finite = values[torch.isfinite(values)]
        return float(finite.max().cpu()) if finite.numel() else None

    def _minimum(self, key: str, indices=None):
        values = self._stats.get(f"{key}_min")
        if values is None:
            return None
        values = values[self._indices(indices)]
        finite = values[torch.isfinite(values)]
        return float(finite.min().cpu()) if finite.numel() else None

    def add_wbc_step(
        self,
        env: WBCEnv,
        s: int,
        e: int,
        active: torch.Tensor,
        done: torch.Tensor,
        timed_out: torch.Tensor,
        traj_early_term: torch.Tensor,
    ) -> torch.Tensor:
        """Accumulate one step without crossing an episode reset boundary.

        IsaacGym auto-resets inside ``env.step``. Continuous state tensors for
        a done env can therefore already describe its next episode, so those
        samples are excluded. Terminal events are still counted once, against
        the pre-step ``active`` mask, and the caller then deactivates the env.
        """
        active = active.to(device=env.device, dtype=torch.bool)
        self.trace_control_type = str(env.cfg.control.control_type)
        self.trace_num_actions_loco = int(env.num_actions_loco)
        self.trace_num_actions_arm = int(env.num_actions_arm)
        self.trace_arm_action_mode = str(env.arm_action_mode)
        self.trace_reach_model = (
            "directional_table" if env.reach_table is not None else "scalar_sphere_fallback"
        )
        self.trace_self_collision_observability = (
            "disabled_by_asset_filter"
            if int(env.cfg.asset.self_collisions) != 0
            else "not_identifiable_from_net_contact_force"
        )
        self.kinematic_protocol["rho_comfort_hi"] = float(
            env.cfg.wbc.goal_reaching.rho_hi
        )
        done = done.to(device=env.device, dtype=torch.bool)
        timed_out = timed_out.to(device=env.device, dtype=torch.bool)
        traj_early_term = traj_early_term.to(device=env.device, dtype=torch.bool)
        snapshot = getattr(env, "benchmark_terminal_snapshot", None)
        terminal_valid = (
            done & snapshot["valid"][s:e]
            if snapshot is not None
            else torch.zeros_like(done)
        )
        sample_present = active & (~done | terminal_valid)
        measure = sample_present.clone()
        self.exposure_steps.add_(active.float())
        self.terminal_samples.add_((active & terminal_valid).float())

        def sampled(current: torch.Tensor, key: str) -> torch.Tensor:
            value = current[s:e]
            if snapshot is None:
                return value
            terminal = snapshot[key][s:e]
            selector = terminal_valid
            while selector.ndim < value.ndim:
                selector = selector.unsqueeze(-1)
            return torch.where(selector, terminal, value)

        ee = sampled(env.end_effector_state, "end_effector_state")  # (n, 13)
        goal_p = sampled(env.arm_goal_pos_world, "goal_pos_world")  # (n, 3)
        goal_q = sampled(env.arm_goal_quat_world, "goal_quat_world")  # (n, 4)

        # EE position / orientation error
        pos_err2 = (goal_p - ee[:, :3]).pow(2).sum(dim=-1)
        self._add_sq("ee_pos_err", pos_err2, measure)
        self._add_val("ee_pos_err_l1", torch.sqrt(pos_err2.clamp_min(1e-12)), measure)

        q_err = quat_mul(goal_q, quat_conjugate(ee[:, 3:7]))
        w = q_err[:, 3].abs().clamp(max=1.0)
        rot_err = 2.0 * torch.acos(w)

        arm_slice = slice(
            env.num_actions_loco, env.num_actions_loco + env.num_actions_arm
        )
        rho_g = sampled(env.goal_rho, "goal_rho")
        valid_g = sampled(env.goal_rho_valid, "goal_rho_valid")
        traj_d_lat = sampled(env.traj_d_lat, "traj_d_lat")
        traj_timing_err = sampled(env.traj_timing_err, "traj_timing_err")
        traj_sdot_meas = sampled(env.traj_sdot_meas, "traj_sdot_meas")
        traj_s = sampled(env.traj_s, "traj_s")
        traj_length = sampled(env.traj_batch.L, "traj_length")
        traj_duration = sampled(env.traj_batch.T, "traj_duration")
        traj_sim_time = sampled(env.traj_sim_time, "traj_sim_time")
        root_state = sampled(env.root_states, "root_state")
        dof_velocity = sampled(env.dof_vel, "dof_velocity")
        dof_position = sampled(env.dof_pos, "dof_position")
        torques = sampled(env.torques, "torques")
        joint_position_target = sampled(
            env.joint_pos_target, "joint_position_target"
        )
        actions = sampled(env.actions, "actions")
        contact_forces = sampled(env.contact_forces, "contact_forces")
        foot_velocities = sampled(env.foot_velocities, "foot_velocities")
        leg_abs_energy_j = sampled(
            env.step_locomotion_abs_energy_j, "leg_abs_energy_j"
        )
        leg_positive_energy_j = sampled(
            env.step_locomotion_positive_energy_j, "leg_positive_energy_j"
        )
        manipulability = sampled(env.goal_manipulability, "manipulability")
        jacobian_sigma_min = sampled(
            env.goal_jacobian_sigma_min, "jacobian_sigma_min"
        )
        rot_jacobian_sigma_min = sampled(
            env.goal_rot_jacobian_sigma_min, "rot_jacobian_sigma_min"
        )
        joint_limit_distance = sampled(
            env.goal_joint_limit_distance, "joint_limit_distance"
        )
        ik_step_norm_rad = sampled(env.goal_ik_step_norm_rad, "ik_step_norm_rad")
        ik_step_saturated = sampled(
            env.goal_ik_step_saturated, "ik_step_saturated"
        )
        ik_solver_valid = sampled(env.goal_ik_solver_valid, "ik_solver_valid")
        base_feedforward_cmd = sampled(
            env.base_feedforward_cmd, "base_feedforward_cmd"
        )
        benchmark_push_active = sampled(
            env.benchmark_push_active, "benchmark_push_active"
        )
        benchmark_push_force = sampled(
            env.benchmark_push_force_world_n, "benchmark_push_force_world_n"
        )
        captured_fault = (
            sampled(env.numerical_fault_mask, "numerical_fault").bool()
            if snapshot is not None
            else env.numerical_fault_mask[s:e]
        )
        finite = (
            torch.isfinite(ee).all(dim=-1)
            & torch.isfinite(goal_p).all(dim=-1)
            & torch.isfinite(goal_q).all(dim=-1)
            & torch.isfinite(traj_d_lat)
            & torch.isfinite(traj_timing_err)
            & torch.isfinite(traj_sdot_meas)
            & torch.isfinite(traj_s)
            & torch.isfinite(traj_length)
            & torch.isfinite(root_state[:, 7:13]).all(dim=-1)
            & torch.isfinite(dof_velocity).all(dim=-1)
            & torch.isfinite(torques).all(dim=-1)
            & torch.isfinite(manipulability)
            & torch.isfinite(jacobian_sigma_min)
            & torch.isfinite(rot_jacobian_sigma_min)
            & torch.isfinite(joint_limit_distance).all(dim=-1)
            & torch.isfinite(ik_step_norm_rad)
            & (~valid_g | torch.isfinite(rho_g))
            & ~captured_fault
        )
        numerical_fault = measure & ~finite
        self._add_count("numerical_fault", numerical_fault, active)
        measure &= finite
        progress = (traj_s / traj_length.clamp_min(1e-6)).clamp(max=1.0)
        ee_offset_world = quat_apply(
            ee[:, 3:7], env.ee_local_offset.expand(e - s, -1)
        )
        ee_velocity = ee[:, 7:10] + torch.cross(
            ee[:, 10:13], ee_offset_world, dim=-1
        )

        self._trace_step(
            sample_present=sample_present,
            metric_valid=measure,
            terminal_snapshot=terminal_valid,
            done=active & done,
            timed_out=active & timed_out,
            trajectory_early_termination=active & traj_early_term,
            numerical_fault=active & numerical_fault,
            fall=active & done & ~timed_out & ~traj_early_term & ~numerical_fault,
            reference_time_s=traj_sim_time,
            reference_duration_s=traj_duration,
            reference_ee_position_m=goal_p,
            reference_ee_quaternion_xyzw=goal_q,
            actual_ee_state=ee,
            actual_ee_grasp_linear_velocity_mps=ee_velocity,
            environment_origin_m=env.env_origins[s:e],
            base_root_state=root_state,
            dof_position_rad=dof_position,
            dof_velocity_rad_s=dof_velocity,
            actuator_command=torques,
            joint_position_target_rad=joint_position_target,
            policy_action=actions,
            foot_contact_force_n=contact_forces[:, env.feet_indices],
            foot_linear_velocity_mps=foot_velocities,
            actuator_torque_limit=env.torque_limits.unsqueeze(0).expand(e - s, -1),
            leg_abs_mechanical_energy_step_j=leg_abs_energy_j,
            leg_positive_mechanical_energy_step_j=leg_positive_energy_j,
            trajectory_progress=progress,
            trajectory_lateral_error_m=traj_d_lat,
            trajectory_timing_error_m=traj_timing_err,
            goal_rho=rho_g,
            goal_rho_valid=valid_g,
            base_feedforward_command=base_feedforward_cmd,
            manipulability=manipulability,
            jacobian_sigma_min=jacobian_sigma_min,
            rot_jacobian_sigma_min=rot_jacobian_sigma_min,
            joint_limit_distance_fraction=joint_limit_distance,
            ik_step_norm_rad=ik_step_norm_rad,
            ik_step_saturated=ik_step_saturated,
            ik_solver_valid=ik_solver_valid,
            benchmark_push_active=benchmark_push_active,
            benchmark_push_force_world_n=benchmark_push_force,
        )

        # _add_val already records both sum and squared sum.
        self._add_val("ee_rot_err", rot_err, measure)

        # Trajectory progress
        self._add_val("d_lat", traj_d_lat, measure)
        self._add_val("timing_err_abs", traj_timing_err.abs(), measure)
        self._add_val("sdot_meas", traj_sdot_meas, measure)
        self._add_val("progress", progress, measure)
        self.final_progress[:] = torch.where(measure, progress, self.final_progress)
        self.final_reference_time_s[:] = torch.where(
            measure, traj_sim_time, self.final_reference_time_s
        )
        self.reference_duration_s[:] = torch.where(
            measure, traj_duration, self.reference_duration_s
        )
        self.final_ee_pos_error[:] = torch.where(
            measure, torch.sqrt(pos_err2.clamp_min(0.0)), self.final_ee_pos_error
        )
        self.final_ee_rot_error[:] = torch.where(measure, rot_err, self.final_ee_rot_error)
        within_tube = (
            (self.final_ee_pos_error <= float(self.protocol["position_tolerance_m"]))
            & (self.final_ee_rot_error <= float(self.protocol["rotation_tolerance_rad"]))
        )
        self.tracking_tube_steps.add_((measure & within_tube).float())
        at_endpoint = (
            progress >= float(self.protocol["endpoint_progress_min"])
        ) & within_tube
        self.endpoint_hold_steps[:] = torch.where(
            measure & at_endpoint,
            self.endpoint_hold_steps + 1.0,
            torch.where(measure, torch.zeros_like(self.endpoint_hold_steps), self.endpoint_hold_steps),
        )
        self.steps.add_(measure.float())
        tube_fraction = self.tracking_tube_steps / self.steps.clamp_min(1.0)
        success_now = (
            measure
            & (traj_sim_time >= traj_duration)
            & (progress >= float(self.protocol["endpoint_progress_min"]))
            & (tube_fraction >= float(self.protocol["tracking_tube_fraction"]))
            & (
                self.endpoint_hold_steps * float(env.dt)
                >= float(self.protocol["hold_time_s"])
            )
            & ~torch.isfinite(self.completion_time_s)
        )
        elapsed = self.exposure_steps * float(env.dt)
        self.completion_time_s[:] = torch.where(
            success_now, elapsed, self.completion_time_s
        )

        # Reach utilisation
        rho_measure = measure & valid_g
        self._add_val("rho", rho_g, rho_measure)
        self._add_max("rho", rho_g, rho_measure)
        self._add_count("rho_valid", valid_g, measure)
        rho_hi = float(env.cfg.wbc.goal_reaching.rho_hi)
        self._add_count("rho_above_hi", valid_g & (rho_g > rho_hi), measure)

        # Base utilisation
        ff_norm = torch.linalg.vector_norm(base_feedforward_cmd[:, :2], dim=-1)
        # root_states stores world-frame base-origin velocity.  Differencing
        # body-frame base_lin_vel would mix physical acceleration with frame
        # rotation.
        base_velocity = root_state[:, 7:10]
        base_norm = torch.linalg.vector_norm(base_velocity[:, :2], dim=-1)
        self._add_val("v_ff_xy", ff_norm, measure)
        self._add_val("v_base_xy", base_norm, measure)
        base_active = ff_norm > 0.05
        ratio = base_norm / ff_norm.clamp_min(1e-6)
        self._add_val("util_ratio", ratio, measure & base_active)

        # In mixed "M" control, env.torques contains leg torques followed by
        # arm position targets.  Only the leg slice is a physical torque.  Arm
        # and whole-body mechanical power are available only when all DOFs use
        # torque/PD control ("P").
        leg_power = (
            torques[:, : env.num_actions_loco]
            * dof_velocity[:, : env.num_actions_loco]
        )
        self._add_val("leg_abs_mechanical_power", leg_power.abs().sum(dim=-1), measure)
        self._add_val("leg_positive_mechanical_power", leg_power.clamp_min(0.0).sum(dim=-1), measure)
        self._add_sum("leg_abs_mechanical_energy_j", leg_abs_energy_j, measure)
        self._add_sum(
            "leg_positive_mechanical_energy_j", leg_positive_energy_j, measure
        )
        if str(env.cfg.control.control_type) == "P":
            arm_power = torques[:, arm_slice] * dof_velocity[:, arm_slice]
            whole_power = torques * dof_velocity
            self._add_val("arm_abs_mechanical_power", arm_power.abs().sum(dim=-1), measure)
            self._add_val("whole_abs_mechanical_power", whole_power.abs().sum(dim=-1), measure)

        # Full vectors for physically meaningful finite differences.
        self._ee_velocities.append(ee_velocity.clone())
        self._base_velocities.append(base_velocity.clone())
        self._arm_dof_velocities.append(dof_velocity[:, arm_slice].clone())
        self._ee_position_errors.append(torch.sqrt(pos_err2.clamp_min(0.0)).clone())
        self._ee_rotation_errors.append(rot_err.clone())
        self._base_positions.append(root_state[:, :3].clone())
        self._leg_torques.append(torques[:, : env.num_actions_loco].clone())
        self._leg_torque_limits.append(
            env.torque_limits[: env.num_actions_loco]
            .unsqueeze(0)
            .expand(e - s, -1)
            .clone()
        )
        foot_contact = contact_forces[:, env.feet_indices, 2] > 1.0
        self._foot_contacts.append(foot_contact.clone())
        self._foot_slip_speeds.append(
            torch.linalg.vector_norm(foot_velocities[:, :, :2], dim=-1).clone()
        )
        self._smoothness_valid.append(measure.clone())

        # A timeout is not a fall, and a tracking cutoff is reported separately.
        fall = done & ~timed_out & ~traj_early_term
        self._add_count("fall", fall, active)
        self._add_count("timed_out", timed_out, active)
        self._add_count("traj_early_term", traj_early_term, active)

        # Manipulability
        self._add_val("manipulability", manipulability, measure)
        self._add_val("jacobian_sigma_min", jacobian_sigma_min, measure)
        self._add_val("rot_jacobian_sigma_min", rot_jacobian_sigma_min, measure)
        joint_margin = joint_limit_distance.min(dim=-1).values
        self._add_val("joint_limit_margin", joint_margin, measure)
        self._add_min("joint_limit_margin", joint_margin, measure)
        self._add_count(
            "joint_limit_near",
            joint_margin <= float(self.kinematic_protocol["joint_limit_margin_fraction"]),
            measure,
        )
        self._add_count(
            "rot_singularity_near",
            rot_jacobian_sigma_min
            <= float(self.kinematic_protocol["rot_jacobian_sigma_min"]),
            measure,
        )
        self._add_count(
            "reach_model_outside",
            valid_g & (rho_g > float(self.kinematic_protocol["reach_model_limit_ratio"])),
            measure,
        )
        observed_infeasible = (
            (valid_g & (rho_g > float(self.kinematic_protocol["reach_model_limit_ratio"])))
            | (
                joint_margin
                <= float(self.kinematic_protocol["joint_limit_margin_fraction"])
            )
            | (
                rot_jacobian_sigma_min
                <= float(self.kinematic_protocol["rot_jacobian_sigma_min"])
            )
        )
        self._add_count("kinematic_observed_infeasible", observed_infeasible, measure)
        ik_applicable = self.trace_arm_action_mode != "end_to_end"
        if ik_applicable:
            self._add_count("ik_step_saturated", ik_step_saturated, measure)
            self._add_count("ik_solver_invalid", ~ik_solver_valid, measure)

        return numerical_fault | success_now

    # -- smoothness summary (offline, called once per scenario point) -------

    def _fd_norm_stats(
        self, signal_list: List[torch.Tensor], dt: float, order: int, indices=None
    ):
        """Differentiate vectors, then summarize their Euclidean norms.

        The validity mask is differentiated alongside the signal so no sample
        crosses an auto-reset boundary or includes an already inactive env.
        """
        if len(signal_list) < order + 1:
            return None
        sig = torch.stack(signal_list, dim=0)  # (T, N, D)
        valid = torch.stack(self._smoothness_valid, dim=0)  # (T, N)
        selected = self._indices(indices)
        sig = sig[:, selected]
        valid = valid[:, selected]
        for _ in range(order):
            sig = (sig[1:] - sig[:-1]) / dt
            valid = valid[1:] & valid[:-1]
        norms = torch.linalg.vector_norm(sig, dim=-1)
        values = norms[valid]
        if values.numel() == 0:
            return None
        return dict(
            mean=float(values.mean().cpu()),
            std=float(values.std(unbiased=False).cpu()),
            p90=float(values.float().quantile(0.90).cpu()),
            p99=float(values.float().quantile(0.99).cpu()),
        )

    def smoothness_summary(self, dt: float, indices=None) -> dict:
        return dict(
            ee_accel=self._fd_norm_stats(self._ee_velocities, dt, 1, indices),
            ee_jerk=self._fd_norm_stats(self._ee_velocities, dt, 2, indices),
            base_accel=self._fd_norm_stats(self._base_velocities, dt, 1, indices),
            base_jerk=self._fd_norm_stats(self._base_velocities, dt, 2, indices),
            arm_joint_accel=self._fd_norm_stats(
                self._arm_dof_velocities, dt, 1, indices
            ),
            arm_joint_jerk=self._fd_norm_stats(
                self._arm_dof_velocities, dt, 2, indices
            ),
        )

    def _scalar_series_stats(self, signal_list, indices=None):
        if not signal_list:
            return None
        values = torch.stack(signal_list, dim=0)[:, self._indices(indices)]
        valid = torch.stack(self._smoothness_valid, dim=0)[:, self._indices(indices)]
        values = values[valid]
        if not values.numel():
            return None
        return {
            "std": float(values.std(unbiased=False).cpu()),
            "variance": float(values.var(unbiased=False).cpu()),
            "p95": float(values.float().quantile(0.95).cpu()),
            "peak": float(values.max().cpu()),
        }

    def _effort_summary(self, indices=None):
        if not self._leg_torques:
            return None
        selected = self._indices(indices)
        torque = torch.stack(self._leg_torques, dim=0)[:, selected]
        limits = torch.stack(self._leg_torque_limits, dim=0)[:, selected]
        valid = torch.stack(self._smoothness_valid, dim=0)[:, selected]
        expanded_valid = valid.unsqueeze(-1).expand_as(torque)
        values = torque[expanded_valid]
        valid_limits = limits[expanded_valid]
        if not values.numel():
            return None
        return {
            "leg_torque_rms_nm": float(torch.sqrt(values.square().mean()).cpu()),
            "leg_torque_abs_peak_nm": float(values.abs().max().cpu()),
            "leg_torque_saturation_fraction": float(
                (values.abs() >= 0.99 * valid_limits).float().mean().cpu()
            ),
        }

    def _contact_summary(self, indices=None):
        if not self._foot_contacts:
            return None
        selected = self._indices(indices)
        contact = torch.stack(self._foot_contacts, dim=0)[:, selected]
        slip = torch.stack(self._foot_slip_speeds, dim=0)[:, selected]
        valid = torch.stack(self._smoothness_valid, dim=0)[:, selected]
        contact_valid = contact & valid.unsqueeze(-1)
        return {
            "foot_slip_speed_mean_contact_mps": (
                float(slip[contact_valid].mean().cpu()) if contact_valid.any() else None
            ),
            "foot_contact_fraction": float(
                contact_valid.float().sum().cpu()
                / (valid.float().sum().clamp_min(1.0).cpu() * contact.shape[-1])
            ),
            "support_fraction": float(
                ((contact.any(dim=-1) & valid).float().sum() / valid.float().sum().clamp_min(1.0)).cpu()
            ),
            "no_support_fraction": float(
                (((~contact.any(dim=-1)) & valid).float().sum() / valid.float().sum().clamp_min(1.0)).cpu()
            ),
        }

    def _base_path_summary(self, index: int):
        if not self._base_positions:
            return None
        position = torch.stack(self._base_positions, dim=0)[:, index]
        valid = torch.stack(self._smoothness_valid, dim=0)[:, index]
        position = position[valid]
        if not position.numel():
            return None
        displacement = position[-1] - position[0]
        segment = position[1:] - position[:-1]
        path_length = (
            torch.linalg.vector_norm(segment[:, :2], dim=-1).sum()
            if segment.numel()
            else torch.zeros((), device=position.device)
        )
        span = position[:, :2].max(dim=0).values - position[:, :2].min(dim=0).values
        return {
            "base_xy_path_length_m": float(path_length.cpu()),
            "base_xy_net_displacement_m": float(
                torch.linalg.vector_norm(displacement[:2]).cpu()
            ),
            "base_xy_path_excess_m": float(
                (path_length - torch.linalg.vector_norm(displacement[:2])).cpu()
            ),
            "base_xy_bounding_box_area_m2": float((span[0] * span[1]).cpu()),
            "base_z_drift_m": float(displacement[2].cpu()),
            "base_z_drift_abs_m": float(displacement[2].abs().cpu()),
        }

    def _sample_mean(self, key: str):
        total = self._stats.get(f"{key}_sum", torch.zeros(1, device=self.dev)).sum()
        count = self._stats.get(f"{key}_samples", torch.zeros(1, device=self.dev)).sum()
        return float((total / count).cpu()) if count.item() > 0 else None

    def _sample_rmse(self, key: str):
        total = self._stats.get(f"{key}_sq", torch.zeros(1, device=self.dev)).sum()
        count = self._stats.get(f"{key}_samples", torch.zeros(1, device=self.dev)).sum()
        return float(torch.sqrt(total / count).cpu()) if count.item() > 0 else None

    def per_env_summary(self, index: int, dt: float) -> dict:
        def mean(key):
            total = self._stats.get(f"{key}_sum", torch.zeros(self.n, device=self.dev))[
                index
            ]
            count = self._stats.get(
                f"{key}_samples", torch.zeros(self.n, device=self.dev)
            )[index]
            return float((total / count).cpu()) if count.item() > 0 else None

        def rmse(key):
            total = self._stats.get(f"{key}_sq", torch.zeros(self.n, device=self.dev))[
                index
            ]
            count = self._stats.get(
                f"{key}_samples", torch.zeros(self.n, device=self.dev)
            )[index]
            return float(torch.sqrt(total / count).cpu()) if count.item() > 0 else None

        def count(key):
            return int(
                self._stats.get(key, torch.zeros(self.n, device=self.dev))[index].item()
            )

        samples = int(self.steps[index].item())
        pos_stats = self._scalar_series_stats(self._ee_position_errors, [index]) or {}
        rot_stats = self._scalar_series_stats(self._ee_rotation_errors, [index]) or {}
        effort = self._effort_summary([index]) or {}
        contact = self._contact_summary([index]) or {}
        base_path = self._base_path_summary(index) or {}
        completion_time = self.completion_time_s[index]
        deadline_reached = (
            int(self.exposure_steps[index].item()) >= self.n_steps
            and not torch.isfinite(completion_time)
        )
        metrics = dict(
            ee_pos_rmse_m=rmse("ee_pos_err"),
            ee_pos_mae_m=mean("ee_pos_err_l1"),
            ee_pos_error_p95_m=pos_stats.get("p95"),
            ee_pos_error_peak_m=pos_stats.get("peak"),
            ee_pos_error_residual_std_m=pos_stats.get("std"),
            ee_pos_error_residual_variance_m2=pos_stats.get("variance"),
            ee_rot_rmse_rad=rmse("ee_rot_err"),
            ee_rot_mae_rad=mean("ee_rot_err"),
            ee_rot_error_p95_rad=rot_stats.get("p95"),
            ee_rot_error_peak_rad=rot_stats.get("peak"),
            ee_rot_error_residual_std_rad=rot_stats.get("std"),
            ee_rot_error_residual_variance_rad2=rot_stats.get("variance"),
            d_lat_mean_m=mean("d_lat"),
            timing_err_mean_m=mean("timing_err_abs"),
            progress_mean=mean("progress"),
            rho_mean=mean("rho"),
            rho_max=self._maximum("rho", [index]),
            rho_above_hi_rate=(
                float(count("rho_above_hi")) / count("rho_valid")
                if count("rho_valid") > 0
                else None
            ),
            base_util_mean=mean("util_ratio"),
            v_ff_xy_mean=mean("v_ff_xy"),
            v_base_xy_mean=mean("v_base_xy"),
            leg_abs_mechanical_power_mean_w=mean("leg_abs_mechanical_power"),
            leg_positive_mechanical_power_mean_w=mean("leg_positive_mechanical_power"),
            leg_abs_mechanical_energy_j=float(
                self._stats.get(
                    "leg_abs_mechanical_energy_j", torch.zeros(self.n, device=self.dev)
                )[index].item()
            ),
            leg_positive_mechanical_energy_j=float(
                self._stats.get(
                    "leg_positive_mechanical_energy_j",
                    torch.zeros(self.n, device=self.dev),
                )[index].item()
            ),
            arm_abs_mechanical_power_mean_w=mean("arm_abs_mechanical_power"),
            whole_body_abs_mechanical_power_mean_w=mean("whole_abs_mechanical_power"),
            manipulability_mean=mean("manipulability"),
            jacobian_sigma_min_mean=mean("jacobian_sigma_min"),
            rot_jacobian_sigma_min_mean=mean("rot_jacobian_sigma_min"),
            joint_limit_margin_min_fraction=self._minimum(
                "joint_limit_margin", [index]
            ),
            final_progress=float(self.final_progress[index].item()) if samples else None,
            reference_time_s=(
                float(self.final_reference_time_s[index].item()) if samples else None
            ),
            reference_duration_s=(
                float(self.reference_duration_s[index].item()) if samples else None
            ),
            final_ee_pos_error_m=(
                float(self.final_ee_pos_error[index].item()) if samples else None
            ),
            final_ee_rot_error_rad=(
                float(self.final_ee_rot_error[index].item()) if samples else None
            ),
            tracking_tube_fraction=(
                float(self.tracking_tube_steps[index].item()) / samples if samples else None
            ),
            endpoint_hold_time_s=float(self.endpoint_hold_steps[index].item()) * dt,
            completion_time_s=(
                float(completion_time.item()) if torch.isfinite(completion_time) else None
            ),
            fall=bool(count("fall")),
            timed_out=bool(count("timed_out")) or deadline_reached,
            benchmark_deadline_reached=deadline_reached,
            traj_early_term=bool(count("traj_early_term")),
            numerical_fault=bool(count("numerical_fault")),
            n_env_steps=int(self.exposure_steps[index].item()),
            n_valid_metric_samples=samples,
            terminal_snapshot_used=bool(self.terminal_samples[index].item()),
        )
        metrics.update(effort)
        metrics.update(contact)
        metrics.update(base_path)
        metrics.update(self._kinematic_task_summary(index))
        decision = timed_trajectory_success(metrics, self.protocol)
        metrics.update(
            completed=decision["success"],
            success=decision["success"],
            incomplete=not decision["success"],
            end_reason=decision["end_reason"],
        )
        return metrics

    def _kinematic_task_summary(self, index: int) -> dict:
        samples = int(self.steps[index].item())
        rho_valid = int(
            self._stats.get("rho_valid", torch.zeros(self.n, device=self.dev))[
                index
            ].item()
        )

        def fraction(key, denominator):
            if denominator <= 0:
                return None
            count = self._stats.get(key, torch.zeros(self.n, device=self.dev))[
                index
            ]
            return float(count.item()) / denominator

        ik_applicable = self.trace_arm_action_mode != "end_to_end"
        return {
            "reach_model": self.trace_reach_model,
            "reach_model_within_limit_fraction": (
                None
                if rho_valid <= 0
                else 1.0 - fraction("reach_model_outside", rho_valid)
            ),
            "reach_model_outside_fraction": fraction(
                "reach_model_outside", rho_valid
            ),
            "joint_limit_near_fraction": fraction("joint_limit_near", samples),
            "rot_singularity_near_fraction": fraction(
                "rot_singularity_near", samples
            ),
            "kinematic_observed_feasible_fraction": (
                None
                if samples <= 0
                else 1.0
                - fraction("kinematic_observed_infeasible", samples)
            ),
            "ik_solver_status": "evaluated" if ik_applicable else "not_applicable",
            "ik_step_saturation_fraction": (
                fraction("ik_step_saturated", samples) if ik_applicable else None
            ),
            "ik_solver_invalid_fraction": (
                fraction("ik_solver_invalid", samples) if ik_applicable else None
            ),
            "self_collision_status": self.trace_self_collision_observability,
        }

    def wbc_summary(self, dt: float) -> dict:
        """All WBC metrics for this scenario point, as a plain dict."""
        return self.wbc_summary_for_indices(None, dt)

    def wbc_summary_for_indices(self, indices, dt: float) -> dict:
        """Pooled summary for an arbitrary subset of this wave env slots."""
        selected = self._indices(indices)

        def stat_sum(key):
            values = self._stats.get(key, torch.zeros(self.n, device=self.dev))
            return float(values[selected].sum().item())

        def sample_mean(key):
            count = stat_sum(f"{key}_samples")
            return stat_sum(f"{key}_sum") / count if count > 0 else None

        def sample_rmse(key):
            count = stat_sum(f"{key}_samples")
            return (stat_sum(f"{key}_sq") / count) ** 0.5 if count > 0 else None

        per_task = [self.per_env_summary(int(index), dt) for index in selected.tolist()]
        events = aggregate_task_events(per_task)
        rho_valid = stat_sum("rho_valid")
        def task_mean(key):
            values = [row.get(key) for row in per_task]
            values = [float(value) for value in values if value is not None]
            return sum(values) / len(values) if values else None

        pos_stats = self._scalar_series_stats(self._ee_position_errors, selected) or {}
        rot_stats = self._scalar_series_stats(self._ee_rotation_errors, selected) or {}
        effort = self._effort_summary(selected) or {}
        contact = self._contact_summary(selected) or {}
        summary = dict(
            ee_pos_rmse_m=sample_rmse("ee_pos_err"),
            ee_pos_mae_m=sample_mean("ee_pos_err_l1"),
            ee_pos_error_p95_m=pos_stats.get("p95"),
            ee_pos_error_peak_m=pos_stats.get("peak"),
            ee_pos_error_residual_std_m=pos_stats.get("std"),
            ee_pos_error_residual_variance_m2=pos_stats.get("variance"),
            ee_rot_rmse_rad=sample_rmse("ee_rot_err"),
            ee_rot_mae_rad=sample_mean("ee_rot_err"),
            ee_rot_error_p95_rad=rot_stats.get("p95"),
            ee_rot_error_peak_rad=rot_stats.get("peak"),
            ee_rot_error_residual_std_rad=rot_stats.get("std"),
            ee_rot_error_residual_variance_rad2=rot_stats.get("variance"),
            d_lat_mean_m=sample_mean("d_lat"),
            timing_err_mean_m=sample_mean("timing_err_abs"),
            progress_mean=sample_mean("progress"),
            rho_mean=sample_mean("rho"),
            rho_max=self._maximum("rho", selected),
            rho_above_hi_rate=(stat_sum("rho_above_hi") / rho_valid if rho_valid > 0 else None),
            base_util_mean=sample_mean("util_ratio"),
            v_ff_xy_mean=sample_mean("v_ff_xy"),
            v_base_xy_mean=sample_mean("v_base_xy"),
            leg_abs_mechanical_power_mean_w=sample_mean("leg_abs_mechanical_power"),
            leg_positive_mechanical_power_mean_w=sample_mean("leg_positive_mechanical_power"),
            leg_abs_mechanical_energy_mean_j=(
                stat_sum("leg_abs_mechanical_energy_j") / len(selected)
                if len(selected)
                else None
            ),
            leg_positive_mechanical_energy_mean_j=(
                stat_sum("leg_positive_mechanical_energy_j") / len(selected)
                if len(selected)
                else None
            ),
            arm_abs_mechanical_power_mean_w=sample_mean("arm_abs_mechanical_power"),
            whole_body_abs_mechanical_power_mean_w=sample_mean("whole_abs_mechanical_power"),
            manipulability_mean=sample_mean("manipulability"),
            jacobian_sigma_min_mean=sample_mean("jacobian_sigma_min"),
            rot_jacobian_sigma_min_mean=sample_mean("rot_jacobian_sigma_min"),
            joint_limit_margin_min_fraction=self._minimum(
                "joint_limit_margin", selected
            ),
            reach_model_outside_fraction=(
                stat_sum("reach_model_outside") / rho_valid
                if rho_valid > 0
                else None
            ),
            reach_model_within_limit_fraction=(
                1.0 - stat_sum("reach_model_outside") / rho_valid
                if rho_valid > 0
                else None
            ),
            joint_limit_near_fraction=(
                stat_sum("joint_limit_near") / stat_sum("joint_limit_margin_samples")
                if stat_sum("joint_limit_margin_samples") > 0
                else None
            ),
            rot_singularity_near_fraction=(
                stat_sum("rot_singularity_near")
                / stat_sum("rot_jacobian_sigma_min_samples")
                if stat_sum("rot_jacobian_sigma_min_samples") > 0
                else None
            ),
            kinematic_observed_feasible_fraction=(
                1.0
                - stat_sum("kinematic_observed_infeasible")
                / stat_sum("joint_limit_margin_samples")
                if stat_sum("joint_limit_margin_samples") > 0
                else None
            ),
            ik_solver_status=(
                "not_applicable"
                if self.trace_arm_action_mode == "end_to_end"
                else "evaluated"
            ),
            ik_step_saturation_fraction=(
                None
                if self.trace_arm_action_mode == "end_to_end"
                or stat_sum("joint_limit_margin_samples") <= 0
                else stat_sum("ik_step_saturated")
                / stat_sum("joint_limit_margin_samples")
            ),
            ik_solver_invalid_fraction=(
                None
                if self.trace_arm_action_mode == "end_to_end"
                or stat_sum("joint_limit_margin_samples") <= 0
                else stat_sum("ik_solver_invalid")
                / stat_sum("joint_limit_margin_samples")
            ),
            reach_model=self.trace_reach_model,
            self_collision_status=self.trace_self_collision_observability,
            smoothness=self.smoothness_summary(dt, selected),
            n_env_steps=int(self.exposure_steps[selected].sum().item()),
            n_valid_metric_samples=int(self.steps[selected].sum().item()),
            terminal_snapshot_count=int(self.terminal_samples[selected].sum().item()),
            base_xy_path_length_mean_m=task_mean("base_xy_path_length_m"),
            base_xy_net_displacement_mean_m=task_mean("base_xy_net_displacement_m"),
            base_xy_path_excess_mean_m=task_mean("base_xy_path_excess_m"),
            base_xy_bounding_box_area_mean_m2=task_mean(
                "base_xy_bounding_box_area_m2"
            ),
            base_z_drift_abs_mean_m=task_mean("base_z_drift_abs_m"),
        )
        summary.update(effort)
        summary.update(contact)
        summary.update(events)
        summary["incomplete_rate"] = (
            None if events["completion_rate"] is None else 1.0 - events["completion_rate"]
        )
        return summary


# ---------------------------------------------------------------------------
# WBC eval loop (accumulator-based, for bank-trajectory scenarios)
# ---------------------------------------------------------------------------


@dataclass
class WBCPolicyHandle:
    """One locomotion policy + upper controller and its env slice.

    ``arm_policy`` retains the legacy learned-arm path. New system benchmarks
    set ``upper_controller`` explicitly so controller and locomotion identity
    remain independent result dimensions.
    """

    name: str
    dog_policy: Callable
    arm_policy: Callable
    env_start: int
    env_end: int
    upper_controller: object = None
    controller_id: str = "learned_arm_policy"
    locomotion_policy_id: str = "unspecified"
    command_scope: str = "bounded_whole_body"

    @property
    def n_envs(self) -> int:
        return self.env_end - self.env_start


def _wbc_step_all(
    env: HistoryWrapper,
    handles: List[WBCPolicyHandle],
):
    """One coordinated step: arm inference for all handles (collect plans),
    call plan() once, dog inference for all handles, then one env.step().

    plan() writes to ``commands_dog`` and ``base_feedforward_cmd``, which are
    env-wide — it must be called once with all policies' plan outputs so no
    policy silently overwrites another's command.
    """
    base = env.env
    gpu = base.device
    n_arm = base.num_actions_arm
    n_dog = base.num_actions_loco
    N = base.num_envs
    full_arm_actions = torch.zeros(N, n_arm, device=gpu)
    plan_actions = (
        torch.zeros(N, env.num_plan_actions, device=gpu)
        if env.num_plan_actions > 0
        else None
    )
    full_dog_actions = torch.zeros(N, n_dog, device=gpu)

    with torch.no_grad():
        # Upper-controller inference: collect physical + plan/direct commands.
        arm_obs = None
        direct_outputs = []
        for h in handles:
            s, e = h.env_start, h.env_end
            if h.upper_controller is None:
                if arm_obs is None:
                    arm_obs = env.get_arm_observations()
                arm_obs_slice = {k: v[s:e] for k, v in arm_obs.items()}
                actions_arm = h.arm_policy(arm_obs_slice).to(gpu)
                controller_plan = actions_arm[..., n_arm:]
                actions_arm = actions_arm[..., :n_arm]
            else:
                from benchmark.wbc.controllers import validate_upper_controller_output

                output = h.upper_controller.step(base, s, e)
                validate_upper_controller_output(
                    output, h.n_envs, n_arm, env.num_plan_actions
                )
                actions_arm = output.arm_actions.to(gpu)
                controller_plan = output.plan_actions
                if output.base_velocity_body is not None:
                    direct_outputs.append((h, output))
            full_arm_actions[s:e] = actions_arm[..., :n_arm]
            if plan_actions is not None and controller_plan is not None:
                plan_actions[s:e] = controller_plan.to(gpu)

        if plan_actions is not None:
            env.plan(plan_actions)
        for handle, output in direct_outputs:
            env_ids = torch.arange(handle.env_start, handle.env_end, device=gpu)
            base.apply_external_upper_commands(
                env_ids,
                output.base_velocity_body,
                output.body_posture,
            )
        for handle in handles:
            if handle.command_scope == "bounded_whole_body":
                continue
            if handle.command_scope not in ("fixed_base", "bounded_posture"):
                raise ValueError(f"unsupported upper-controller command scope: {handle.command_scope}")
            env_ids = torch.arange(handle.env_start, handle.env_end, device=gpu)
            posture = torch.stack(
                (
                    base.commands_dog[env_ids, dog_cmd_idx["body_height"]],
                    base.commands_dog[env_ids, dog_cmd_idx["body_pitch"]],
                    base.commands_dog[env_ids, dog_cmd_idx["body_roll"]],
                ),
                dim=-1,
            )
            if handle.command_scope == "fixed_base":
                posture.zero_()
            base.apply_external_upper_commands(
                env_ids,
                torch.zeros(handle.n_envs, 3, device=gpu),
                posture,
            )

        # Dog inference: after plan() has written commands_dog
        dog_obs = env.get_dog_observations()
        for h in handles:
            s, e = h.env_start, h.env_end
            dog_obs_slice = {k: v[s:e] for k, v in dog_obs.items()}
            full_dog_actions[s:e] = h.dog_policy(dog_obs_slice).to(gpu)

        _, _, done, _ = env.step(full_dog_actions, full_arm_actions)

    early_term = getattr(
        base, "traj_early_term", torch.zeros(N, device=gpu, dtype=torch.bool)
    )
    timed_out = base.time_out_buf.clone()
    snapshot = getattr(base, "benchmark_terminal_snapshot", None)
    if snapshot is not None:
        terminal = done & snapshot["valid"]
        timed_out = torch.where(terminal, snapshot["time_out"], timed_out)
        early_term = torch.where(
            terminal, snapshot["traj_early_term"], early_term
        )
    return done.clone(), timed_out, early_term.clone()


def wbc_eval_loop(
    env: HistoryWrapper,
    handles: List[WBCPolicyHandle],
    n_steps: int,
    device: str,
    valid_env_counts: Optional[List[int]] = None,
    record_raw_traces: bool = False,
    tasks: Optional[List[dict]] = None,
) -> List[WBCAccumulator]:
    """Evaluate one prepared trajectory episode per environment.

    The caller owns reset, settling and trajectory assignment. Once an env
    terminates it is removed from the active mask, so its auto-reset episode
    cannot leak into the current scenario's statistics.
    """
    base = env.env
    gpu = base.device
    n_per = handles[0].n_envs
    accs = [
        WBCAccumulator(n_per, n_steps, device=gpu, record_trace=record_raw_traces)
        for _ in handles
    ]
    if valid_env_counts is None:
        valid_env_counts = [h.n_envs for h in handles]
    if len(valid_env_counts) != len(handles):
        raise ValueError("valid_env_counts must match handles")
    active = []
    for h, valid_count, acc in zip(handles, valid_env_counts, accs):
        mask = torch.arange(h.n_envs, device=gpu) < int(valid_count)
        active.append(mask)
        acc.episode_count[:] = mask.float()

    for _step_i in range(n_steps):
        if hasattr(base, "set_benchmark_push"):
            push_ids = []
            push_values = []
            if tasks is not None:
                for handle in handles:
                    for local_index, task in enumerate(tasks):
                        elapsed = _step_i * float(base.dt)
                        for event in task.get("disturbance_schedule", []):
                            start = float(event["start_time_s"])
                            duration = float(event["duration_s"])
                            if start <= elapsed < start + duration:
                                if event.get("frame") != "environment_world":
                                    raise ValueError("benchmark currently supports environment_world push frame")
                                if event.get("body") != "base":
                                    raise ValueError("benchmark currently supports base-body pushes")
                                push_ids.append(handle.env_start + local_index)
                                push_values.append(event["force_n"])
            base.set_benchmark_push(push_ids, push_values)
        done, timed_out, early_term = _wbc_step_all(env, handles)
        for i, (h, acc) in enumerate(zip(handles, accs)):
            s, e = h.env_start, h.env_end
            finished = acc.add_wbc_step(
                base,
                s,
                e,
                active=active[i],
                done=done[s:e],
                timed_out=timed_out[s:e],
                traj_early_term=early_term[s:e],
            )
            active[i] &= ~done[s:e] & ~finished
        if not any(bool(mask.any().item()) for mask in active):
            break

    return accs


# ---------------------------------------------------------------------------
# WBC scenario result (extends dog-only ScenarioResult)
# ---------------------------------------------------------------------------


def wbc_acc_to_result(
    acc: WBCAccumulator,
    name: str,
    scenario: str,
    label: str,
    dt: float,
    cell_a: int = 0,
    cell_b: int = 0,
    trajectory_bank_rows: Optional[List[int]] = None,
) -> dict:
    """``WBCAccumulator`` -> flat dict ready for JSON serialisation."""
    w = acc.wbc_summary(dt)
    return dict(
        run_name=name,
        scenario=scenario,
        label=label,
        cell_A=cell_a,
        cell_B=cell_b,
        trajectory_bank_rows=list(trajectory_bank_rows or []),
        **w,
    )
