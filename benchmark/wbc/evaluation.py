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
from isaacgym.torch_utils import quat_apply, quat_conjugate, quat_mul

from benchmark.dog_policy.evaluation import (
    Accumulator,
    CRITICAL_COMPAT_CFG_PATHS,
    _cfg_value,
    _read_cfg_dict,
    _load_cfg_from_pkl,
    _apply_benchmark_env_overrides,
)
from go1_gym.envs.config import configure_privileged_obs_dims
from go1_gym.envs.roboduet.wbc_env import WBCEnv
from go1_gym.envs.roboduet.wbc_env_wrapper import HistoryWrapper


# ---------------------------------------------------------------------------
# WBC-specific env loading
# ---------------------------------------------------------------------------

WBC_COMPAT_PATHS = [
    "arm.arm_num_observations",
    "arm.arm_num_observation_history",
    "arm.num_actions_arm_cd",
    "arm.arm_num_privileged_obs",
    "arm.action_mode",
    "arm.checkpoint_observation_layout",
    "wbc.goal_reaching.target_mode",
    "wbc.goal_reaching.rho_star",
    "wbc.goal_reaching.rho_lo",
    "wbc.goal_reaching.rho_hi",
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
    "wbc.goal_reaching.trajectory.max_gamma_points",
    "wbc.goal_reaching.trajectory.max_tl_points",
    "wbc.goal_reaching.trajectory.update_s_window",
    "wbc.goal_reaching.trajectory.n_levels_A",
    "wbc.goal_reaching.trajectory.n_levels_B",
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


WBC_SHARED_ENV_COMPAT_PATHS = tuple(dict.fromkeys(
    list(CRITICAL_COMPAT_CFG_PATHS) + WBC_COMPAT_PATHS
))


def _freeze_compat_value(value):
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze_compat_value(item)) for key, item in value.items()))
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
):
    """Build the checkpoint's env at ``total_envs`` with evaluation overrides.

    The bank is reseeded by default so the benchmark scores held-out
    trajectories. ``bank_per_cell`` shrinks the pre-generated trajectory bank
    (CPU/scipy-bound at construction time).
    """
    cfg = _load_cfg_from_pkl(logdir, robot=robot)
    _apply_benchmark_env_overrides(cfg, total_envs, envs_per_policy)

    # Candidate slices start from identical nominal reset states. Per-env asset
    # buckets still repeat with envs_per_policy, preserving paired mount props.
    for attr in ("x_init_range", "y_init_range", "z_init_range",
                 "yaw_init_range", "roll_init_range", "pitch_init_range"):
        if hasattr(cfg.terrain, attr):
            setattr(cfg.terrain, attr, 0.0)
    cfg.terrain.reset_curriculum = False
    cfg.env.stage1_arm_init_dof_pos_noise = 0.0
    for stage_name in ("stage1_arm", "stage2_arm"):
        stage = getattr(cfg.domain_rand, stage_name, None)
        if stage is not None:
            for attr in ("randomize_Kp_factor", "randomize_Kd_factor",
                         "randomize_motor_strength", "randomize_motor_offset",
                         "randomize_ee_payload"):
                if hasattr(stage, attr):
                    setattr(stage, attr, False)

    mode = str(getattr(cfg.wbc.goal_reaching, "target_mode", "static"))
    if mode != "trajectory":
        raise ValueError(
            f"{logdir} is not a trajectory-tracking run (target_mode='{mode}'). "
            f"Use --dog_only for locomotion-only runs."
        )

    cfg.wbc.goal_reaching.trajectory.bank_seed = int(bank_seed)
    if bank_per_cell is not None:
        cfg.wbc.goal_reaching.trajectory.bank_per_cell = int(bank_per_cell)

    configure_privileged_obs_dims(cfg)
    env = WBCEnv(sim_device=device, headless=headless, cfg=cfg)
    return HistoryWrapper(env), cfg


def load_wbc_policies(
    logdir: str, ckpt_id: str, cfg, device: str = "cuda:0"
):
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

    def __init__(self, num_envs: int, n_steps: int, device: str):
        super().__init__(num_envs, device)
        self.n_steps = n_steps
        self.exposure_steps = torch.zeros(num_envs, device=device)
        self.episode_count = torch.ones(num_envs, device=device)
        self.final_progress = torch.zeros(num_envs, device=device)
        self.success_progress = 0.8
        # Full velocity vectors are retained so direction changes contribute to
        # acceleration and jerk. A validity mask prevents finite differences
        # from crossing a terminal/reset boundary.
        self._ee_velocities: List[torch.Tensor] = []
        self._base_velocities: List[torch.Tensor] = []
        self._arm_dof_velocities: List[torch.Tensor] = []
        self._smoothness_valid: List[torch.Tensor] = []

    @staticmethod
    def _masked(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return value * mask.to(device=value.device, dtype=value.dtype)

    def _add_sq(self, key: str, sq: torch.Tensor, mask: torch.Tensor):
        self._get(f"{key}_sq").add_(self._masked(sq, mask))
        self._get(f"{key}_samples").add_(mask.float())

    def _add_val(self, key: str, value: torch.Tensor, mask: torch.Tensor):
        self._get(f"{key}_sum").add_(self._masked(value, mask))
        self._get(f"{key}_sq").add_(self._masked(value.pow(2), mask))
        self._get(f"{key}_samples").add_(mask.float())

    def _add_count(self, key: str, value: torch.Tensor, mask: torch.Tensor):
        self._get(key).add_(self._masked(value.float(), mask))

    def _add_max(self, key: str, value: torch.Tensor, mask: torch.Tensor):
        stat_key = f"{key}_max"
        if stat_key not in self._stats:
            self._stats[stat_key] = torch.full(
                (self.n,), -torch.inf, device=self.dev, dtype=value.dtype
            )
        candidate = torch.where(mask, value, torch.full_like(value, -torch.inf))
        torch.maximum(self._stats[stat_key], candidate, out=self._stats[stat_key])

    def _maximum(self, key: str) -> float:
        values = self._stats.get(f"{key}_max")
        if values is None:
            return 0.0
        finite = values[torch.isfinite(values)]
        return float(finite.max().cpu()) if finite.numel() else 0.0

    def add_wbc_step(
        self,
        env: WBCEnv,
        s: int,
        e: int,
        active: torch.Tensor,
        done: torch.Tensor,
        timed_out: torch.Tensor,
        traj_early_term: torch.Tensor,
    ):
        """Accumulate one step without crossing an episode reset boundary.

        IsaacGym auto-resets inside ``env.step``. Continuous state tensors for
        a done env can therefore already describe its next episode, so those
        samples are excluded. Terminal events are still counted once, against
        the pre-step ``active`` mask, and the caller then deactivates the env.
        """
        active = active.to(device=env.device, dtype=torch.bool)
        done = done.to(device=env.device, dtype=torch.bool)
        timed_out = timed_out.to(device=env.device, dtype=torch.bool)
        traj_early_term = traj_early_term.to(device=env.device, dtype=torch.bool)
        measure = active & ~done
        self.exposure_steps.add_(active.float())

        ee = env.end_effector_state[s:e]            # (n, 13)
        goal_p = env.arm_goal_pos_world[s:e]         # (n, 3)
        goal_q = env.arm_goal_quat_world[s:e]        # (n, 4)

        # EE position / orientation error
        pos_err2 = (goal_p - ee[:, :3]).pow(2).sum(dim=-1)
        self._add_sq("ee_pos_err", pos_err2, measure)
        self._add_val("ee_pos_err_l1", torch.sqrt(pos_err2.clamp_min(1e-12)), measure)

        q_err = quat_mul(goal_q, quat_conjugate(ee[:, 3:7]))
        w = q_err[:, 3].abs().clamp(max=1.0)
        rot_err = 2.0 * torch.acos(w)
        # _add_val already records both sum and squared sum.
        self._add_val("ee_rot_err", rot_err, measure)

        # Trajectory progress
        self._add_val("d_lat", env.traj_d_lat[s:e], measure)
        self._add_val("timing_err_abs", env.traj_timing_err[s:e].abs(), measure)
        self._add_val("sdot_meas", env.traj_sdot_meas[s:e], measure)
        progress = (env.traj_s[s:e] / env.traj_batch.L[s:e].clamp_min(1e-6)).clamp(max=1.0)
        self._add_val("progress", progress, measure)
        self.final_progress[:] = torch.where(measure, progress, self.final_progress)
        self.success_progress = float(env.cfg.wbc.goal_reaching.trajectory.success_progress)

        # Reach utilisation
        rho_g = env.goal_rho[s:e]
        valid_g = env.goal_rho_valid[s:e]
        rho_measure = measure & valid_g
        self._add_val("rho", rho_g, rho_measure)
        self._add_max("rho", rho_g, rho_measure)
        self._add_count("rho_valid", valid_g, measure)
        rho_hi = float(env.cfg.wbc.goal_reaching.rho_hi)
        self._add_count("rho_above_hi", valid_g & (rho_g > rho_hi), measure)

        # Base utilisation
        ff_norm = torch.linalg.vector_norm(env.base_feedforward_cmd[s:e, :2], dim=-1)
        base_velocity = env.base_lin_vel[s:e, :3]
        base_norm = torch.linalg.vector_norm(base_velocity[:, :2], dim=-1)
        self._add_val("v_ff_xy", ff_norm, measure)
        self._add_val("v_base_xy", base_norm, measure)
        base_active = ff_norm > 0.05
        ratio = base_norm / ff_norm.clamp_min(1e-6)
        self._add_val("util_ratio", ratio, measure & base_active)

        # Motor power
        arm_slice = slice(env.num_actions_loco, env.num_actions_loco + env.num_actions_arm)
        power = (env.torques[s:e, arm_slice] * env.dof_vel[s:e, arm_slice]).abs().sum(dim=-1)
        self._add_val("motor_power", power, measure)

        # Full vectors for physically meaningful finite differences.
        ee_offset_world = quat_apply(
            ee[:, 3:7], env.ee_local_offset.expand(e - s, -1)
        )
        ee_velocity = ee[:, 7:10] + torch.cross(ee[:, 10:13], ee_offset_world, dim=-1)
        self._ee_velocities.append(ee_velocity.clone())
        self._base_velocities.append(base_velocity.clone())
        self._arm_dof_velocities.append(env.dof_vel[s:e, arm_slice].clone())
        self._smoothness_valid.append(measure.clone())

        # A timeout is not a fall, and a tracking cutoff is reported separately.
        fall = done & ~timed_out & ~traj_early_term
        self._add_count("fall", fall, active)
        self._add_count("timed_out", timed_out, active)
        self._add_count("traj_early_term", traj_early_term, active)

        # Manipulability
        self._add_val("manipulability", env.goal_manipulability[s:e], measure)

        self.steps.add_(measure.float())

    # -- smoothness summary (offline, called once per scenario point) -------

    def _fd_norm_stats(self, signal_list: List[torch.Tensor], dt: float, order: int):
        """Differentiate vectors, then summarize their Euclidean norms.

        The validity mask is differentiated alongside the signal so no sample
        crosses an auto-reset boundary or includes an already inactive env.
        """
        if len(signal_list) < order + 1:
            return None
        sig = torch.stack(signal_list, dim=0)  # (T, N, D)
        valid = torch.stack(self._smoothness_valid, dim=0)  # (T, N)
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

    def smoothness_summary(self, dt: float) -> dict:
        return dict(
            ee_accel=self._fd_norm_stats(self._ee_velocities, dt, 1),
            ee_jerk=self._fd_norm_stats(self._ee_velocities, dt, 2),
            base_accel=self._fd_norm_stats(self._base_velocities, dt, 1),
            base_jerk=self._fd_norm_stats(self._base_velocities, dt, 2),
            arm_joint_accel=self._fd_norm_stats(self._arm_dof_velocities, dt, 1),
            arm_joint_jerk=self._fd_norm_stats(self._arm_dof_velocities, dt, 2),
        )

    def _sample_mean(self, key: str) -> float:
        total = self._stats.get(f"{key}_sum", torch.zeros(1, device=self.dev)).sum()
        count = self._stats.get(f"{key}_samples", torch.zeros(1, device=self.dev)).sum()
        return float((total / count.clamp_min(1.0)).cpu())

    def _sample_rmse(self, key: str) -> float:
        total = self._stats.get(f"{key}_sq", torch.zeros(1, device=self.dev)).sum()
        count = self._stats.get(f"{key}_samples", torch.zeros(1, device=self.dev)).sum()
        return float(torch.sqrt(total / count.clamp_min(1.0)).cpu())

    def per_env_summary(self, index: int, dt: float) -> dict:
        def mean(key):
            total = self._stats.get(f"{key}_sum", torch.zeros(self.n, device=self.dev))[index]
            count = self._stats.get(f"{key}_samples", torch.zeros(self.n, device=self.dev))[index]
            return float((total / count.clamp_min(1.0)).cpu())

        def rmse(key):
            total = self._stats.get(f"{key}_sq", torch.zeros(self.n, device=self.dev))[index]
            count = self._stats.get(f"{key}_samples", torch.zeros(self.n, device=self.dev))[index]
            return float(torch.sqrt(total / count.clamp_min(1.0)).cpu())

        def count(key):
            return int(self._stats.get(key, torch.zeros(self.n, device=self.dev))[index].item())

        completed = bool(self.final_progress[index].item() >= self.success_progress) and not bool(
            count("fall") + count("traj_early_term")
        )
        return dict(
            ee_pos_rmse_m=rmse("ee_pos_err"),
            ee_pos_mae_m=mean("ee_pos_err_l1"),
            ee_rot_rmse_rad=rmse("ee_rot_err"),
            ee_rot_mae_rad=mean("ee_rot_err"),
            d_lat_mean_m=mean("d_lat"),
            timing_err_mean_m=mean("timing_err_abs"),
            progress_mean=mean("progress"),
            rho_mean=mean("rho"),
            base_util_mean=mean("util_ratio"),
            motor_power_mean_w=mean("motor_power"),
            fall=bool(count("fall")),
            timed_out=bool(count("timed_out")),
            traj_early_term=bool(count("traj_early_term")),
            completed=completed,
            incomplete=not completed,
            n_env_steps=int(self.exposure_steps[index].item()),
        )

    def wbc_summary(self, dt: float) -> dict:
        """All WBC metrics for this scenario point, as a plain dict."""
        episodes = max(float(self.episode_count.sum().item()), 1.0)
        completed = float((self.final_progress >= self.success_progress).float().mul(
            1.0 - self._stats.get("fall", torch.zeros(self.n, device=self.dev)).clamp(max=1.0)
        ).mul(
            1.0 - self._stats.get("traj_early_term", torch.zeros(self.n, device=self.dev)).clamp(max=1.0)
        ).sum().item())
        return dict(
            ee_pos_rmse_m=self._sample_rmse("ee_pos_err"),
            ee_pos_mae_m=self._sample_mean("ee_pos_err_l1"),
            ee_rot_rmse_rad=self._sample_rmse("ee_rot_err"),
            ee_rot_mae_rad=self._sample_mean("ee_rot_err"),
            d_lat_mean_m=self._sample_mean("d_lat"),
            timing_err_mean_m=self._sample_mean("timing_err_abs"),
            progress_mean=self._sample_mean("progress"),
            rho_mean=self._sample_mean("rho"),
            rho_max=self._maximum("rho"),
            rho_above_hi_rate=(self.total_count("rho_above_hi") / max(self.total_count("rho_valid"), 1)),
            base_util_mean=self._sample_mean("util_ratio"),
            v_ff_xy_mean=self._sample_mean("v_ff_xy"),
            v_base_xy_mean=self._sample_mean("v_base_xy"),
            motor_power_mean_w=self._sample_mean("motor_power"),
            manipulability_mean=self._sample_mean("manipulability"),
            fall_rate=(self.total_count("fall") / episodes),
            timeout_rate=(self.total_count("timed_out") / episodes),
            traj_early_term_rate=(self.total_count("traj_early_term") / episodes),
            completion_rate=(completed / episodes),
            incomplete_rate=(max(episodes - completed, 0.0) / episodes),
            n_episodes=int(episodes),
            smoothness=self.smoothness_summary(dt),
            n_env_steps=int(self.exposure_steps.sum().item()),
        )


# ---------------------------------------------------------------------------
# WBC eval loop (accumulator-based, for bank-trajectory scenarios)
# ---------------------------------------------------------------------------


@dataclass
class WBCPolicyHandle:
    """One policy pair + its env slice."""
    name: str
    dog_policy: Callable
    arm_policy: Callable
    env_start: int
    env_end: int

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
    plan_actions = torch.zeros(N, env.num_plan_actions, device=gpu) if env.num_plan_actions > 0 else None
    full_dog_actions = torch.zeros(N, n_dog, device=gpu)

    with torch.no_grad():
        # Arm inference: collect physical + plan actions
        arm_obs = env.get_arm_observations()
        for h in handles:
            s, e = h.env_start, h.env_end
            arm_obs_slice = {k: v[s:e] for k, v in arm_obs.items()}
            actions_arm = h.arm_policy(arm_obs_slice).to(gpu)
            full_arm_actions[s:e] = actions_arm[..., :n_arm]
            if plan_actions is not None:
                plan_actions[s:e] = actions_arm[..., n_arm:]

        if plan_actions is not None:
            env.plan(plan_actions)

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
    return done.clone(), base.time_out_buf.clone(), early_term.clone()


def wbc_eval_loop(
    env: HistoryWrapper,
    handles: List[WBCPolicyHandle],
    n_steps: int,
    device: str,
    valid_env_counts: Optional[List[int]] = None,
) -> List[WBCAccumulator]:
    """Evaluate one prepared trajectory episode per environment.

    The caller owns reset, settling and trajectory assignment. Once an env
    terminates it is removed from the active mask, so its auto-reset episode
    cannot leak into the current scenario's statistics.
    """
    base = env.env
    gpu = base.device
    n_per = handles[0].n_envs
    accs = [WBCAccumulator(n_per, n_steps, device=gpu) for _ in handles]
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
        done, timed_out, early_term = _wbc_step_all(env, handles)
        for i, (h, acc) in enumerate(zip(handles, accs)):
            s, e = h.env_start, h.env_end
            acc.add_wbc_step(
                base,
                s,
                e,
                active=active[i],
                done=done[s:e],
                timed_out=timed_out[s:e],
                traj_early_term=early_term[s:e],
            )
            active[i] &= ~done[s:e]
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
