"""WBC env/policy loading, extended Accumulator, and eval loop.

Shares the same config-loading and env-override path as the dog-only benchmark
so layout compatibility grouping and the HTML report pipeline work without
duplication.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import isaacgym  # noqa: F401 - must precede torch
import torch
from isaacgym.torch_utils import quat_conjugate, quat_mul

from benchmark.dog_policy.evaluation import (
    Accumulator,
    PolicyHandle,
    ScenarioResult,
    _acc_to_result,
    _load_cfg_from_pkl,
    _apply_benchmark_env_overrides,
    configure_stage1,
    detect_command_layout,
)
from go1_gym.envs.config import configure_privileged_obs_dims
from go1_gym.envs.roboduet.wbc_env import WBCEnv
from go1_gym.envs.roboduet.wbc_env_wrapper import HistoryWrapper
from go1_gym.utils.global_switch import global_switch


# ---------------------------------------------------------------------------
# WBC-specific env loading
# ---------------------------------------------------------------------------

WBC_COMPAT_PATHS = [
    "arm.arm_num_observations",
    "arm.arm_num_observation_history",
    "arm.num_actions_arm_cd",
    "arm.arm_num_privileged_obs",
    "wbc.goal_reaching.target_mode",
    "wbc.goal_reaching.rho_star",
    "wbc.goal_reaching.rho_lo",
    "wbc.goal_reaching.rho_hi",
    "wbc.goal_reaching.trajectory.preview_points",
    "wbc.goal_reaching.trajectory.max_gamma_points",
    "wbc.goal_reaching.trajectory.n_levels_A",
    "wbc.goal_reaching.trajectory.n_levels_B",
]


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
        # Velocity buffers for offline finite-difference smoothness.
        # shape (n_steps, num_envs) — these are the signals, not accumulators.
        self._ee_vel_norms: List[torch.Tensor] = []
        self._base_vel_norms: List[torch.Tensor] = []
        self._arm_dof_vel_norms: List[torch.Tensor] = []
        self._motor_power: List[torch.Tensor] = []

    def add_wbc_step(self, env: WBCEnv, s: int, e: int):
        """Accumulate one step of WBC-specific metrics for envs [s:e)."""
        ee = env.end_effector_state[s:e]            # (n, 13)
        goal_p = env.arm_goal_pos_world[s:e]         # (n, 3)
        goal_q = env.arm_goal_quat_world[s:e]        # (n, 4)

        # EE position / orientation error
        pos_err2 = (goal_p - ee[:, :3]).pow(2).sum(dim=-1)
        self.add_sq("ee_pos_err", pos_err2)
        self.add_val("ee_pos_err_l1", torch.sqrt(pos_err2.clamp_min(1e-12)))

        q_err = quat_mul(goal_q, quat_conjugate(ee[:, 3:7]))
        w = q_err[:, 3].abs().clamp(max=1.0)
        rot_err = 2.0 * torch.acos(w)
        self.add_val("ee_rot_err", rot_err)
        self.add_sq("ee_rot_err", rot_err)

        # Trajectory progress
        self.add_val("d_lat", env.traj_d_lat[s:e])
        self.add_val("timing_err_abs", env.traj_timing_err[s:e].abs())
        self.add_val("sdot_meas", env.traj_sdot_meas[s:e])
        progress = (env.traj_s[s:e] / env.traj_batch.L[s:e].clamp_min(1e-6)).clamp(max=1.0)
        self.add_val("progress", progress)

        # Reach utilisation
        rho_g = env.goal_rho[s:e]
        valid_g = env.goal_rho_valid[s:e]
        self.add_val("rho", rho_g)
        self.add_val("rho_max", rho_g)
        self.add_count("rho_valid", valid_g)
        rho_hi = float(env.cfg.wbc.goal_reaching.rho_hi)
        self.add_count("rho_above_hi", valid_g & (rho_g > rho_hi))

        # Base utilisation
        ff_norm = torch.linalg.vector_norm(env.base_feedforward_cmd[s:e, :2], dim=-1)
        base_norm = torch.linalg.vector_norm(env.base_lin_vel[s:e, :2], dim=-1)
        self.add_val("v_ff_xy", ff_norm)
        self.add_val("v_base_xy", base_norm)
        active = ff_norm > 0.05
        ratio = base_norm / ff_norm.clamp_min(1e-6)
        self.add_val("util_ratio", ratio * active.float())

        # Motor power
        arm_slice = slice(env.num_actions_loco, env.num_actions_loco + env.num_actions_arm)
        power = (env.torques[s:e, arm_slice] * env.dof_vel[s:e, arm_slice]).abs().sum(dim=-1)
        self.add_val("motor_power", power)

        # Velocity norms (for offline smoothness)
        self._ee_vel_norms.append(torch.linalg.vector_norm(ee[:, 7:10], dim=-1).clone())
        self._base_vel_norms.append(base_norm.clone())
        self._arm_dof_vel_norms.append(
            torch.linalg.vector_norm(env.dof_vel[s:e, arm_slice], dim=-1).clone()
        )
        self._motor_power.append(power.clone())

        # Fall tracking
        self.add_count("fall", env.reset_buf[s:e])
        self.add_count("timed_out", env.time_out_buf[s:e])
        self.add_count(
            "traj_early_term",
            getattr(env, "traj_early_term", torch.zeros(env.num_envs, device=env.device, dtype=torch.bool))[s:e],
        )

        # Manipulability
        self.add_val("manipulability", env.goal_manipulability[s:e])

        self.tick()

    # -- smoothness summary (offline, called once per scenario point) -------

    def _fd_norm_stats(self, signal_list: List[torch.Tensor], dt: float, order: int):
        """Finite-difference derivative of ``order``, then mean/std over all
        (step, env). Returns the scalar norm at each (T, N) cell."""
        if len(signal_list) < order + 1:
            return None
        sig = torch.stack(signal_list, dim=0)  # (T, N)
        for _ in range(order):
            sig = (sig[1:] - sig[:-1]) / dt
        return dict(
            mean=float(sig.abs().mean().cpu()),
            std=float(sig.abs().std().cpu()),
            p90=float(sig.abs().flatten().float().quantile(0.90).cpu()),
            p99=float(sig.abs().flatten().float().quantile(0.99).cpu()),
        )

    def smoothness_summary(self, dt: float) -> dict:
        return dict(
            ee_accel=self._fd_norm_stats(self._ee_vel_norms, dt, 1),
            ee_jerk=self._fd_norm_stats(self._ee_vel_norms, dt, 2),
            base_accel=self._fd_norm_stats(self._base_vel_norms, dt, 1),
            base_jerk=self._fd_norm_stats(self._base_vel_norms, dt, 2),
            arm_joint_accel=self._fd_norm_stats(self._arm_dof_vel_norms, dt, 1),
            arm_joint_jerk=self._fd_norm_stats(self._arm_dof_vel_norms, dt, 2),
        )

    def wbc_summary(self, dt: float) -> dict:
        """All WBC metrics for this scenario point, as a plain dict."""
        return dict(
            ee_pos_rmse_m=self.rmse("ee_pos_err"),
            ee_pos_mae_m=self.mean("ee_pos_err_l1"),
            ee_rot_rmse_rad=self.rmse("ee_rot_err"),
            ee_rot_mae_rad=self.mean("ee_rot_err"),
            d_lat_mean_m=self.mean("d_lat"),
            timing_err_mean_m=self.mean("timing_err_abs"),
            progress_mean=self.mean("progress"),
            rho_mean=self.mean("rho"),
            rho_max=self.mean("rho_max"),
            rho_above_hi_rate=(self.total_count("rho_above_hi") / max(self.total_count("rho_valid"), 1)),
            base_util_mean=self.mean("util_ratio"),
            v_ff_xy_mean=self.mean("v_ff_xy"),
            v_base_xy_mean=self.mean("v_base_xy"),
            motor_power_mean_w=self.mean("motor_power"),
            manipulability_mean=self.mean("manipulability"),
            fall_rate=(self.total_count("fall") / max(self.steps.mean().item(), 1)),
            timeout_rate=(self.total_count("timed_out") / max(self.steps.mean().item(), 1)),
            traj_early_term_rate=(self.total_count("traj_early_term") / max(self.steps.mean().item(), 1)),
            smoothness=self.smoothness_summary(dt),
            n_env_steps=int(self.steps.sum().item()),
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
        for h in handles:
            s, e = h.env_start, h.env_end
            arm_obs = env.get_arm_observations()
            arm_obs_slice = {k: v[s:e] for k, v in arm_obs.items()}
            actions_arm = h.arm_policy(arm_obs_slice).to(gpu)
            full_arm_actions[s:e] = actions_arm[..., :n_arm]
            if plan_actions is not None:
                plan_actions[s:e] = actions_arm[..., n_arm:]

        if plan_actions is not None:
            env.plan(plan_actions)

        # Dog inference: after plan() has written commands_dog
        for h in handles:
            s, e = h.env_start, h.env_end
            dog_obs = env.get_dog_observations()
            dog_obs_slice = {k: v[s:e] for k, v in dog_obs.items()}
            full_dog_actions[s:e] = h.dog_policy(dog_obs_slice).to(gpu)

        env.step(full_dog_actions, full_arm_actions)


def wbc_eval_loop(
    env: HistoryWrapper,
    handles: List[WBCPolicyHandle],
    n_steps: int,
    device: str,
    settle_steps: int = 0,
) -> List[WBCAccumulator]:
    """Step all env groups; accumulate WBC metrics per-policy group.

    Returns one ``WBCAccumulator`` per handle, in the same order.
    """
    global_switch.open_switch()
    env.reset()

    base = env.env
    gpu = base.device
    n_per = handles[0].n_envs
    accs = [WBCAccumulator(n_per, n_steps, device=gpu) for _ in handles]

    for _ in range(settle_steps):
        _wbc_step_all(env, handles)

    for _step_i in range(n_steps):
        _wbc_step_all(env, handles)
        for h, acc in zip(handles, accs):
            acc.add_wbc_step(base, h.env_start, h.env_end)

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
) -> dict:
    """``WBCAccumulator`` -> flat dict ready for JSON serialisation."""
    w = acc.wbc_summary(dt)
    return dict(
        run_name=name,
        scenario=scenario,
        label=label,
        cell_A=cell_a,
        cell_B=cell_b,
        **w,
    )
