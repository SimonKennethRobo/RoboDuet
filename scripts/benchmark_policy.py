"""Stage-1 dog-policy benchmark — headless, no joystick.

Evaluates one or more checkpoints on four scenario families and writes a JSON
summary.  When multiple logdirs are given the results are also printed as a
side-by-side comparison table.

Scenarios
---------
A  Velocity command grid
   5 x-vel × 3 yaw-vel points; arm held at ``--arm_intensity``.
   Tracks: lin_vel_x, lin_vel_y, ang_vel_yaw RMSE + fall rate.

B  Arm-disturbance robustness sweep
   Fixed forward velocity (1.0 m/s) at arm intensities [0, 0.25, 0.5, 0.75, 1.0].

C  Body-pose command tracking
   Sweeps pitch, roll, and height commands individually; compares commanded vs
   actual body state.  Only run when the checkpoint uses ≥6 command dims.

D  Gait-parameter tracking
   Sweeps gait_frequency, footswing_height, and stance_width commands against
   measured gait state.  Only run when ``use_dynamic_gait=True``.

Usage::

    # single run
    python scripts/benchmark_policy_stage1.py \\
        --logdirs runs/my_run \\
        --ckptids last \\
        --headless

    # multi-run comparison
    python scripts/benchmark_policy_stage1.py \\
        --logdirs runs/run_A runs/run_B \\
        --names v1 v2 \\
        --ckptids last last \\
        --headless --num_envs 32 --num_eval_steps 2000

Stage-2 hook: ``--stage2`` is reserved for hybrid policy evaluation (not yet
implemented).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle as pkl
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

import isaacgym  # noqa: F401 – must precede torch
import torch

from go1_gym.envs import *  # noqa: F403
from go1_gym.envs.roboduet import HistoryWrapper, WBCEnv
from go1_gym.envs.roboduet.wbc_env_config import (
    ROBODUET_DEFAULTS,
    RoboDuetCfg as Cfg,
    configure_external_recipes,
    configure_privileged_obs_dims,
    env_obs_dim_parts,
    arm_obs_dim_parts,
    dog_obs_dim_parts,
    sum_dim_parts,
)
from go1_gym.utils.global_switch import global_switch
from go1_gym.utils.math_utils import quat_apply_yaw
from isaacgym.torch_utils import quat_conjugate
from scripts.load_policy import (
    _ensure_asset_file,
    _ensure_play_cfg_defaults,
    _recompute_play_dims,
    load_dog_policy,
)

# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

# Scenario A: (x_vel, y_vel, yaw_vel)
VEL_GRID: List[tuple] = [
    (xv, 0.0, yaw)
    for xv in [-0.5, 0.0, 0.5, 1.0, 1.5]
    for yaw in [-1.0, 0.0, 1.0]
]

ARM_INTENSITY_SWEEP = [0.0, 0.25, 0.5, 0.75, 1.0]   # Scenario B
FORWARD_CMD = (1.0, 0.0, 0.0)                          # x, y, yaw for B/D

# Scenario C: body-pose command sweep
PITCH_CMDS = [-0.3, -0.15, 0.0, 0.15, 0.3]  # rad
ROLL_CMDS = [-0.3, -0.15, 0.0, 0.15, 0.3]   # rad
HEIGHT_DELTA_CMDS = [-0.05, 0.0, 0.05, 0.10]  # m relative to base_height_target

# Scenario D: gait-parameter command sweep
GAIT_FREQ_CMDS = [1.5, 2.0, 2.5, 3.0, 3.5]  # Hz
FOOTSWING_HEIGHT_CMDS = [0.04, 0.08, 0.12, 0.16]  # m
STANCE_WIDTH_CMDS = [0.25, 0.30, 0.35, 0.40]  # m

# ---------------------------------------------------------------------------
# Command layout detection
# ---------------------------------------------------------------------------

class CommandLayout(NamedTuple):
    n_dims: int
    has_body_pitch: bool   # cmd[:,3]
    has_body_roll: bool    # cmd[:,4]
    has_body_height: bool  # cmd[:,5]
    has_dynamic_gait: bool # cmd[:,6..10]
    base_height_target: float


def detect_command_layout(cfg) -> CommandLayout:
    n = getattr(cfg.dog, "dog_num_commands", 3)
    use_dg = getattr(cfg.commands, "use_dynamic_gait", False)
    bh = float(getattr(cfg.rewards, "base_height_target", 0.34))
    return CommandLayout(
        n_dims=n,
        has_body_pitch=n >= 4,
        has_body_roll=n >= 5,
        has_body_height=n >= 6,
        has_dynamic_gait=use_dg and n >= 7,
        base_height_target=bh,
    )

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    # identification
    run_name: str
    scenario: str          # "vel_grid" | "arm_sweep" | "body_pose" | "gait"
    label: str             # human-readable key for this point
    # command set-points
    cmd_x: float = 0.0
    cmd_y: float = 0.0
    cmd_yaw: float = 0.0
    cmd_pitch: float = 0.0
    cmd_roll: float = 0.0
    cmd_height_delta: float = 0.0
    cmd_gait_freq: float = 0.0
    cmd_footswing_height: float = 0.0
    cmd_stance_width: float = 0.0
    arm_intensity: float = 0.0
    # sample counts
    n_env_steps: int = 0
    n_falls: int = 0
    # velocity tracking
    lin_vel_x_rmse: float = float("nan")
    lin_vel_y_rmse: float = float("nan")
    ang_vel_yaw_rmse: float = float("nan")
    # body-pose tracking (nan when not applicable)
    pitch_rmse_deg: float = float("nan")
    roll_rmse_deg: float = float("nan")
    height_rmse_m: float = float("nan")
    # gait tracking (nan when not applicable)
    gait_freq_rmse_hz: float = float("nan")
    footswing_height_rmse_m: float = float("nan")
    stance_width_rmse_m: float = float("nan")
    # stability
    fall_rate: float = float("nan")
    base_height_mean: float = float("nan")
    base_height_std: float = float("nan")
    roll_deg_rms: float = float("nan")
    pitch_deg_rms: float = float("nan")
    # effort
    max_torque_mean: float = float("nan")


# ---------------------------------------------------------------------------
# Running accumulator (one float pair per metric per env)
# ---------------------------------------------------------------------------

class Accumulator:
    """Maintains sum / sum-of-squares for RMSE and mean/std computation."""

    def __init__(self, num_envs: int, device: str = "cpu"):
        self.n = num_envs
        self.dev = device
        self._stats: Dict[str, torch.Tensor] = {}
        self.steps = torch.zeros(num_envs, device=device)

    def _get(self, key: str) -> torch.Tensor:
        if key not in self._stats:
            self._stats[key] = torch.zeros(self.n, device=self.dev)
        return self._stats[key]

    def add_sq_err(self, key: str, pred: torch.Tensor, target: torch.Tensor):
        self._get(f"{key}_sq") .add_((pred - target).pow(2))

    def add_val(self, key: str, val: torch.Tensor):
        self._get(f"{key}_sum").add_(val)
        self._get(f"{key}_sq" ).add_(val.pow(2))

    def add_count(self, key: str, mask: torch.Tensor):
        self._get(key).add_(mask.float())

    def tick(self):
        self.steps.add_(1.0)

    def rmse(self, key: str) -> float:
        sq = self._stats.get(f"{key}_sq", torch.zeros(1))
        return float((sq / self.steps.clamp(min=1)).mean().sqrt())

    def mean(self, key: str) -> float:
        s = self._stats.get(f"{key}_sum", torch.zeros(1))
        return float((s / self.steps.clamp(min=1)).mean())

    def std(self, key: str) -> float:
        s = self._stats.get(f"{key}_sum", torch.zeros(1))
        sq = self._stats.get(f"{key}_sq", torch.zeros(1))
        n = self.steps.clamp(min=1)
        var = (sq / n - (s / n).pow(2)).clamp(min=0)
        return float(var.mean().sqrt())

    def total_count(self, key: str) -> int:
        return int(self._stats.get(key, torch.zeros(1)).sum().item())

    def reset(self):
        self._stats.clear()
        self.steps.zero_()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_cfg_from_pkl(logdir: str, robot: Optional[str] = None) -> Cfg:
    configure_external_recipes(Cfg)
    checkpoint_asset_file = None
    with open(logdir + "/parameters.pkl", "rb") as f:
        pkl_cfg = pkl.load(f)
        cfg_dict = pkl_cfg["Cfg"]
        for key, value in cfg_dict.items():
            if not hasattr(Cfg, key):
                continue
            if key in ["dog", "arm", "hybrid"]:
                for k2, v2 in cfg_dict[key].items():
                    if not isinstance(v2, dict):
                        setattr(getattr(Cfg, key), k2, v2)
                    else:
                        for k3, v3 in v2.items():
                            setattr(getattr(getattr(Cfg, key), k2), k3, v3)
            elif isinstance(value, dict):
                for k2, v2 in value.items():
                    if key == "asset" and k2 == "file":
                        checkpoint_asset_file = v2
                    setattr(getattr(Cfg, key), k2, v2)
            else:
                setattr(Cfg, key, value)

    _ensure_play_cfg_defaults(Cfg)
    _ensure_asset_file(Cfg, robot=robot, checkpoint_asset_file=checkpoint_asset_file)
    _recompute_play_dims(Cfg)
    return Cfg


def _apply_benchmark_env_overrides(cfg, num_envs: int):
    cfg.terrain.mesh_type = "plane"
    cfg.terrain.teleport_robots = False
    for attr in [
        "push_robots", "randomize_friction", "randomize_gravity",
        "randomize_restitution", "randomize_motor_offset",
        "randomize_motor_strength", "randomize_friction_indep",
        "randomize_ground_friction", "randomize_base_mass",
        "randomize_Kd_factor", "randomize_Kp_factor",
        "randomize_joint_friction", "randomize_com_displacement",
        "randomize_end_effector_force",
    ]:
        if hasattr(cfg.domain_rand, attr):
            setattr(cfg.domain_rand, attr, False)
    cfg.env.num_envs = num_envs
    cfg.env.num_recording_envs = 0
    cfg.terrain.num_rows = max(5, int(math.ceil(math.sqrt(num_envs))) + 1)
    cfg.terrain.num_cols = cfg.terrain.num_rows
    cfg.terrain.border_size = 0
    cfg.terrain.center_robots = True
    cfg.terrain.center_span = 1
    cfg.asset.render_sphere = False
    cfg.env.episode_length_s = 20.0
    cfg.commands.resampling_time = 1e9
    cfg.control.control_type = "M"
    cfg.rewards.use_terminal_body_height = True
    cfg.rewards.use_terminal_roll = True
    cfg.rewards.use_terminal_pitch = True
    cfg.hybrid.rewards.use_terminal_body_height = True
    cfg.hybrid.rewards.use_terminal_roll = True
    cfg.hybrid.rewards.use_terminal_pitch = True
    cfg.arm.commands.T_traj = [20000, 30000]
    cfg.env.stage1_arm_curriculum = True


def load_env_benchmark(
    logdir: str,
    num_envs: int,
    headless: bool = True,
    device: str = "cuda:0",
    robot: Optional[str] = None,
):
    cfg = _load_cfg_from_pkl(logdir, robot=robot)
    _apply_benchmark_env_overrides(cfg, num_envs)
    env = WBCEnv(sim_device=device, headless=headless, cfg=cfg)
    env = HistoryWrapper(env)
    return env, cfg


# ---------------------------------------------------------------------------
# Stage-1 global switch helpers
# ---------------------------------------------------------------------------

def configure_stage1(arm_intensity: float, ramp_iters: int = 1000):
    global_switch.switch_flag = False
    global_switch.count = 0
    global_switch.stage1_arm_ramp_iterations = ramp_iters
    global_switch.stage1_count = int(max(0.0, min(1.0, arm_intensity)) * ramp_iters)
    global_switch.pretrained_to_hybrid_start = 10_000_000
    global_switch.pretrained_to_hybrid_end = 10_000_001


# ---------------------------------------------------------------------------
# Command setters
# ---------------------------------------------------------------------------

def _set_cmd(env: HistoryWrapper, idx: int, value: float):
    b = env.env
    if b.commands_dog.shape[1] > idx:
        b.commands_dog[:, idx] = value


def set_vel_cmd(env: HistoryWrapper, x: float, y: float, yaw: float):
    b = env.env
    b.commands_dog[:, 0] = x
    b.commands_dog[:, 1] = y
    b.commands_dog[:, 2] = yaw


def set_pose_cmd(env: HistoryWrapper, pitch: float, roll: float, height_delta: float):
    _set_cmd(env, 3, pitch)
    _set_cmd(env, 4, roll)
    _set_cmd(env, 5, height_delta)


def set_gait_cmd(
    env: HistoryWrapper,
    gait_freq: Optional[float] = None,
    footswing_height: Optional[float] = None,
    stance_width: Optional[float] = None,
):
    if gait_freq is not None:
        _set_cmd(env, 6, gait_freq)
    if footswing_height is not None:
        _set_cmd(env, 7, footswing_height)
    if stance_width is not None:
        _set_cmd(env, 8, stance_width)


# ---------------------------------------------------------------------------
# Per-step metric extraction
# ---------------------------------------------------------------------------

def _actual_height(env: HistoryWrapper) -> torch.Tensor:
    b = env.env
    return torch.mean(b.root_states[:, 2].unsqueeze(1) - b.measured_heights, dim=1).cpu()


def _actual_stance_width(env: HistoryWrapper) -> torch.Tensor:
    """Approximate stance width from foot Y-spread in yaw-only body frame."""
    b = env.env
    foot_pos = b.foot_positions  # [N, 4, 3] world frame
    base_pos = b.root_states[:, :3]  # [N, 3]
    base_quat = b.base_quat  # [N, 4]
    rel = foot_pos - base_pos.unsqueeze(1)  # [N, 4, 3]
    rel_body = torch.stack([
        quat_apply_yaw(quat_conjugate(base_quat), rel[:, i, :])
        for i in range(4)
    ], dim=1)  # [N, 4, 3]
    # Y-spread across all four feet (≈ stance width)
    y_max = rel_body[:, :, 1].max(dim=1).values
    y_min = rel_body[:, :, 1].min(dim=1).values
    return (y_max - y_min).cpu()


def _actual_swing_height(env: HistoryWrapper) -> torch.Tensor:
    """Mean foot height of airborne feet, or nan-like 0 when all grounded."""
    b = env.env
    in_contact = (b.contact_forces[:, b.feet_indices, 2] > 1.0)  # [N, 4]
    in_swing = ~in_contact
    foot_z = b.foot_positions[:, :, 2]  # [N, 4]
    # Per-env: mean Z of swinging feet; 0 when all feet on ground
    swing_z = (foot_z * in_swing.float()).sum(dim=1)  # [N]
    n_swing = in_swing.float().sum(dim=1).clamp(min=1)
    return (swing_z / n_swing).cpu()


def _contact_transition_mask(
    env: HistoryWrapper, prev_contact: torch.Tensor
) -> torch.Tensor:
    """Returns per-env count of contact-state transitions (any leg) this step."""
    b = env.env
    curr = (b.contact_forces[:, b.feet_indices, 2] > 1.0)  # [N, 4]
    changed = (curr != prev_contact).any(dim=1)  # [N]
    prev_contact[:] = curr
    return changed.float().cpu()


# ---------------------------------------------------------------------------
# Core eval loop
# ---------------------------------------------------------------------------

def _eval_loop(
    env: HistoryWrapper,
    dog_policy,
    layout: CommandLayout,
    n_steps: int,
    arm_intensity: float,
    device: str,
    # Optional per-step command overrides (called before each step)
    cmd_fn=None,
) -> Accumulator:
    """Run ``n_steps`` steps and return a filled Accumulator."""
    configure_stage1(arm_intensity)
    env.reset()
    if cmd_fn:
        cmd_fn(env)

    num_envs = env.env.num_envs
    acc = Accumulator(num_envs, device="cpu")
    prev_contact = torch.zeros(num_envs, 4, dtype=torch.bool)

    for _ in range(n_steps):
        if cmd_fn:
            cmd_fn(env)

        with torch.no_grad():
            dog_obs = env.get_dog_observations()
            actions_dog = dog_policy(dog_obs).to(device)

        env.step(actions_dog, env.arm_fake_actions)

        base = env.env
        vel = base.base_lin_vel   # [N, 3]
        ang = base.base_ang_vel   # [N, 3]
        cmd = base.commands_dog   # [N, D]

        height = _actual_height(env)

        acc.add_sq_err("vx",  vel[:, 0].cpu(), cmd[:, 0].cpu())
        acc.add_sq_err("vy",  vel[:, 1].cpu(), cmd[:, 1].cpu())
        acc.add_sq_err("yaw", ang[:, 2].cpu(), cmd[:, 2].cpu())

        if layout.has_body_pitch:
            pitch_cmd_deg = cmd[:, 3].cpu() * (180 / math.pi)
            pitch_act_deg = base.pitch.cpu() * (180 / math.pi)
            acc.add_sq_err("pitch_deg", pitch_act_deg, pitch_cmd_deg)

        if layout.has_body_roll:
            roll_cmd_deg = cmd[:, 4].cpu() * (180 / math.pi)
            roll_act_deg = base.roll.cpu() * (180 / math.pi)
            acc.add_sq_err("roll_deg", roll_act_deg, roll_cmd_deg)

        if layout.has_body_height:
            target_h = (cmd[:, 5] + layout.base_height_target).cpu()
            acc.add_sq_err("height", height, target_h)

        if layout.has_dynamic_gait:
            freq_cmd = cmd[:, 6].cpu()
            transitions = _contact_transition_mask(env, prev_contact)
            # transitions per step / (2 * dt) ≈ step frequency per leg pair
            actual_freq = transitions / (2.0 * base.dt)
            acc.add_sq_err("gait_freq", actual_freq, freq_cmd)

            sw_cmd = cmd[:, 7].cpu()
            sw_act = _actual_swing_height(env)
            acc.add_sq_err("swing_h", sw_act, sw_cmd)

            sw_width_cmd = cmd[:, 8].cpu()
            sw_width_act = _actual_stance_width(env)
            acc.add_sq_err("stance_w", sw_width_act, sw_width_cmd)

        acc.add_val("height",    height)
        acc.add_val("roll_deg",  base.roll.cpu() * (180 / math.pi))
        acc.add_val("pitch_deg", base.pitch.cpu() * (180 / math.pi))

        dog_t = base.torques[:, :12] if base.torques.shape[1] >= 12 else base.torques
        acc.add_val("max_torque", dog_t.abs().max(dim=1).values.cpu())

        fell = (base.reset_buf & ~base.time_out_buf).float().cpu()
        acc.add_count("fell", fell)

        acc.tick()

    return acc


def _acc_to_result(
    acc: Accumulator,
    layout: CommandLayout,
    run_name: str,
    scenario: str,
    label: str,
    n_steps: int,
    **kwargs,
) -> ScenarioResult:
    n_env_steps = int(acc.steps.sum().item())
    r = ScenarioResult(
        run_name=run_name,
        scenario=scenario,
        label=label,
        n_env_steps=n_env_steps,
        n_falls=acc.total_count("fell"),
        fall_rate=acc.total_count("fell") / max(1, n_env_steps),
        lin_vel_x_rmse=acc.rmse("vx"),
        lin_vel_y_rmse=acc.rmse("vy"),
        ang_vel_yaw_rmse=acc.rmse("yaw"),
        base_height_mean=acc.mean("height"),
        base_height_std=acc.std("height"),
        roll_deg_rms=acc.rmse("roll_deg") if not layout.has_body_roll else float("nan"),
        pitch_deg_rms=acc.rmse("pitch_deg") if not layout.has_body_pitch else float("nan"),
        max_torque_mean=acc.mean("max_torque"),
    )
    if layout.has_body_pitch:
        r.pitch_rmse_deg = acc.rmse("pitch_deg")
    if layout.has_body_roll:
        r.roll_rmse_deg = acc.rmse("roll_deg")
    if layout.has_body_height:
        r.height_rmse_m = acc.rmse("height")
    if layout.has_dynamic_gait:
        r.gait_freq_rmse_hz = acc.rmse("gait_freq")
        r.footswing_height_rmse_m = acc.rmse("swing_h")
        r.stance_width_rmse_m = acc.rmse("stance_w")
    for k, v in kwargs.items():
        if hasattr(r, k):
            setattr(r, k, v)
    return r


# ---------------------------------------------------------------------------
# Scenario runners
# ---------------------------------------------------------------------------

def run_scenario_a(
    env, dog_policy, layout: CommandLayout, n_steps: int,
    arm_intensity: float, device: str, run_name: str,
) -> List[ScenarioResult]:
    results = []
    print(f"\n[A] Velocity grid  arm_intensity={arm_intensity:.2f}  "
          f"{len(VEL_GRID)} points × {n_steps} steps")
    for i, (xv, yv, yaw) in enumerate(VEL_GRID):
        label = f"vx={xv:+.1f} yaw={yaw:+.1f}"
        print(f"  [{i+1:2d}/{len(VEL_GRID)}] {label}", end="  ", flush=True)

        def cmd_fn(e, _x=xv, _y=yv, _yaw=yaw):
            set_vel_cmd(e, _x, _y, _yaw)

        acc = _eval_loop(env, dog_policy, layout, n_steps, arm_intensity, device, cmd_fn)
        r = _acc_to_result(
            acc, layout, run_name, "vel_grid", label, n_steps,
            cmd_x=xv, cmd_y=yv, cmd_yaw=yaw, arm_intensity=arm_intensity,
        )
        results.append(r)
        print(f"vx_rmse={r.lin_vel_x_rmse:.4f}  yaw_rmse={r.ang_vel_yaw_rmse:.4f}  "
              f"fall%={r.fall_rate*100:.1f}")
    return results


def run_scenario_b(
    env, dog_policy, layout: CommandLayout, n_steps: int,
    device: str, run_name: str,
) -> List[ScenarioResult]:
    results = []
    xv, yv, yaw = FORWARD_CMD
    print(f"\n[B] Arm-disturbance sweep  cmd=({xv},{yv},{yaw})  "
          f"{len(ARM_INTENSITY_SWEEP)} levels × {n_steps} steps")
    for intensity in ARM_INTENSITY_SWEEP:
        print(f"  arm_intensity={intensity:.2f}", end="  ", flush=True)

        def cmd_fn(e, _x=xv, _y=yv, _yaw=yaw):
            set_vel_cmd(e, _x, _y, _yaw)

        acc = _eval_loop(env, dog_policy, layout, n_steps, intensity, device, cmd_fn)
        r = _acc_to_result(
            acc, layout, run_name, "arm_sweep", f"intensity={intensity:.2f}", n_steps,
            cmd_x=xv, cmd_y=yv, cmd_yaw=yaw, arm_intensity=intensity,
        )
        results.append(r)
        print(f"vx_rmse={r.lin_vel_x_rmse:.4f}  yaw_rmse={r.ang_vel_yaw_rmse:.4f}  "
              f"fall%={r.fall_rate*100:.1f}")
    return results


def run_scenario_c(
    env, dog_policy, layout: CommandLayout, n_steps: int,
    arm_intensity: float, device: str, run_name: str,
) -> List[ScenarioResult]:
    if not layout.has_body_pitch:
        print("\n[C] Body-pose tracking  SKIPPED (fewer than 4 command dims)")
        return []
    results = []
    xv, yv, yaw = FORWARD_CMD
    print(f"\n[C] Body-pose tracking  arm_intensity={arm_intensity:.2f}")

    # Pitch sweep
    print("  Pitch sweep:")
    for p in PITCH_CMDS:
        def cmd_fn(e, _x=xv, _y=yv, _yaw=yaw, _p=p):
            set_vel_cmd(e, _x, _y, _yaw)
            set_pose_cmd(e, pitch=_p, roll=0.0, height_delta=0.0)
        acc = _eval_loop(env, dog_policy, layout, n_steps, arm_intensity, device, cmd_fn)
        label = f"pitch={p:+.2f}rad"
        print(f"    {label}", end="  ", flush=True)
        r = _acc_to_result(
            acc, layout, run_name, "body_pose", label, n_steps,
            cmd_x=xv, cmd_y=yv, cmd_yaw=yaw,
            cmd_pitch=p, arm_intensity=arm_intensity,
        )
        results.append(r)
        print(f"pitch_rmse={r.pitch_rmse_deg:.3f}°  fall%={r.fall_rate*100:.1f}")

    if layout.has_body_roll:
        print("  Roll sweep:")
        for ro in ROLL_CMDS:
            def cmd_fn(e, _x=xv, _y=yv, _yaw=yaw, _ro=ro):
                set_vel_cmd(e, _x, _y, _yaw)
                set_pose_cmd(e, pitch=0.0, roll=_ro, height_delta=0.0)
            acc = _eval_loop(env, dog_policy, layout, n_steps, arm_intensity, device, cmd_fn)
            label = f"roll={ro:+.2f}rad"
            print(f"    {label}", end="  ", flush=True)
            r = _acc_to_result(
                acc, layout, run_name, "body_pose", label, n_steps,
                cmd_x=xv, cmd_y=yv, cmd_yaw=yaw,
                cmd_roll=ro, arm_intensity=arm_intensity,
            )
            results.append(r)
            print(f"roll_rmse={r.roll_rmse_deg:.3f}°  fall%={r.fall_rate*100:.1f}")

    if layout.has_body_height:
        print("  Height sweep:")
        bh = layout.base_height_target
        for hd in HEIGHT_DELTA_CMDS:
            def cmd_fn(e, _x=xv, _y=yv, _yaw=yaw, _hd=hd):
                set_vel_cmd(e, _x, _y, _yaw)
                set_pose_cmd(e, pitch=0.0, roll=0.0, height_delta=_hd)
            acc = _eval_loop(env, dog_policy, layout, n_steps, arm_intensity, device, cmd_fn)
            target_h = bh + hd
            label = f"h_target={target_h:.3f}m"
            print(f"    {label}", end="  ", flush=True)
            r = _acc_to_result(
                acc, layout, run_name, "body_pose", label, n_steps,
                cmd_x=xv, cmd_y=yv, cmd_yaw=yaw,
                cmd_height_delta=hd, arm_intensity=arm_intensity,
            )
            results.append(r)
            print(f"height_rmse={r.height_rmse_m:.4f}m  fall%={r.fall_rate*100:.1f}")

    return results


def run_scenario_d(
    env, dog_policy, layout: CommandLayout, n_steps: int,
    arm_intensity: float, device: str, run_name: str,
) -> List[ScenarioResult]:
    if not layout.has_dynamic_gait:
        print("\n[D] Gait-parameter tracking  SKIPPED (use_dynamic_gait=False)")
        return []
    results = []
    xv, yv, yaw = FORWARD_CMD
    print(f"\n[D] Gait-parameter tracking  arm_intensity={arm_intensity:.2f}")

    print("  Gait-frequency sweep:")
    for gf in GAIT_FREQ_CMDS:
        def cmd_fn(e, _x=xv, _y=yv, _yaw=yaw, _gf=gf):
            set_vel_cmd(e, _x, _y, _yaw)
            set_gait_cmd(e, gait_freq=_gf)
        acc = _eval_loop(env, dog_policy, layout, n_steps, arm_intensity, device, cmd_fn)
        label = f"gait_freq={gf:.1f}Hz"
        print(f"    {label}", end="  ", flush=True)
        r = _acc_to_result(
            acc, layout, run_name, "gait", label, n_steps,
            cmd_x=xv, cmd_y=yv, cmd_yaw=yaw,
            cmd_gait_freq=gf, arm_intensity=arm_intensity,
        )
        results.append(r)
        print(f"freq_rmse={r.gait_freq_rmse_hz:.4f}Hz  fall%={r.fall_rate*100:.1f}")

    print("  Footswing-height sweep:")
    for sh in FOOTSWING_HEIGHT_CMDS:
        def cmd_fn(e, _x=xv, _y=yv, _yaw=yaw, _sh=sh):
            set_vel_cmd(e, _x, _y, _yaw)
            set_gait_cmd(e, footswing_height=_sh)
        acc = _eval_loop(env, dog_policy, layout, n_steps, arm_intensity, device, cmd_fn)
        label = f"swing_h={sh:.2f}m"
        print(f"    {label}", end="  ", flush=True)
        r = _acc_to_result(
            acc, layout, run_name, "gait", label, n_steps,
            cmd_x=xv, cmd_y=yv, cmd_yaw=yaw,
            cmd_footswing_height=sh, arm_intensity=arm_intensity,
        )
        results.append(r)
        print(f"swing_h_rmse={r.footswing_height_rmse_m:.4f}m  fall%={r.fall_rate*100:.1f}")

    print("  Stance-width sweep:")
    for sw in STANCE_WIDTH_CMDS:
        def cmd_fn(e, _x=xv, _y=yv, _yaw=yaw, _sw=sw):
            set_vel_cmd(e, _x, _y, _yaw)
            set_gait_cmd(e, stance_width=_sw)
        acc = _eval_loop(env, dog_policy, layout, n_steps, arm_intensity, device, cmd_fn)
        label = f"stance_w={sw:.2f}m"
        print(f"    {label}", end="  ", flush=True)
        r = _acc_to_result(
            acc, layout, run_name, "gait", label, n_steps,
            cmd_x=xv, cmd_y=yv, cmd_yaw=yaw,
            cmd_stance_width=sw, arm_intensity=arm_intensity,
        )
        results.append(r)
        print(f"width_rmse={r.stance_width_rmse_m:.4f}m  fall%={r.fall_rate*100:.1f}")

    return results


# ---------------------------------------------------------------------------
# Multi-run comparison table
# ---------------------------------------------------------------------------

def _fmt(v: float, fmt: str = ".4f") -> str:
    return "—" if math.isnan(v) else f"{v:{fmt}}"


def print_comparison_table(all_results: Dict[str, Dict[str, List[ScenarioResult]]]):
    run_names = list(all_results.keys())
    sep = "=" * 80

    for scenario in ("vel_grid", "arm_sweep", "body_pose", "gait"):
        scenario_labels = {
            "vel_grid":  "A — Velocity command grid",
            "arm_sweep": "B — Arm-disturbance sweep",
            "body_pose": "C — Body-pose tracking",
            "gait":      "D — Gait-parameter tracking",
        }[scenario]

        # collect labels across all runs for this scenario
        labels: List[str] = []
        for rn in run_names:
            for r in all_results[rn].get(scenario, []):
                if r.label not in labels:
                    labels.append(r.label)
        if not labels:
            continue

        print(f"\n{sep}")
        print(f"Scenario {scenario_labels}")
        print(sep)

        # choose columns based on scenario
        if scenario == "vel_grid":
            cols = [("vx_rmse", "lin_vel_x_rmse"), ("yaw_rmse", "ang_vel_yaw_rmse"),
                    ("h(m)", "base_height_mean"), ("fall%", "fall_rate")]
        elif scenario == "arm_sweep":
            cols = [("vx_rmse", "lin_vel_x_rmse"), ("yaw_rmse", "ang_vel_yaw_rmse"),
                    ("fall%", "fall_rate")]
        elif scenario == "body_pose":
            cols = [("pitch°rmse", "pitch_rmse_deg"), ("roll°rmse", "roll_rmse_deg"),
                    ("h_rmse(m)", "height_rmse_m"), ("fall%", "fall_rate")]
        else:  # gait
            cols = [("freq_rmse", "gait_freq_rmse_hz"),
                    ("swh_rmse", "footswing_height_rmse_m"),
                    ("width_rmse", "stance_width_rmse_m"), ("fall%", "fall_rate")]

        # build lookup: run → label → result
        lookup: Dict[str, Dict[str, ScenarioResult]] = {
            rn: {r.label: r for r in all_results[rn].get(scenario, [])}
            for rn in run_names
        }

        col_w = max(12, max(len(cn) for cn, _ in cols) + 2)
        label_w = max(22, max(len(lb) for lb in labels) + 2)
        run_w = max(8, max(len(rn) for rn in run_names) + 2)

        # header
        header = f"{'label':<{label_w}}"
        for rn in run_names:
            for cn, _ in cols:
                header += f"  {rn[:run_w-2]+'/'+cn:<{col_w}}"
        print(header)
        print("-" * len(header))

        for lb in labels:
            row = f"{lb:<{label_w}}"
            for rn in run_names:
                res = lookup[rn].get(lb)
                for _, attr in cols:
                    if res is None:
                        row += f"  {'—':<{col_w}}"
                    else:
                        v = getattr(res, attr)
                        if attr == "fall_rate":
                            row += f"  {v*100:>{col_w-2}.1f}%  "
                        else:
                            row += f"  {_fmt(v):<{col_w}}"
            print(row)

    print(f"\n{sep}")


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_results(
    all_results: Dict[str, Dict[str, List[ScenarioResult]]],
    output_path: str,
):
    serialisable = {
        run_name: {
            scenario: [asdict(r) for r in results]
            for scenario, results in scenarios.items()
        }
        for run_name, scenarios in all_results.items()
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"\n[Benchmark] Results saved → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Stage-1 dog-policy benchmark")
    p.add_argument("--logdirs", nargs="+", required=True,
                   help="One or more training-run directories to evaluate and compare")
    p.add_argument("--names", nargs="*", default=None,
                   help="Short display names for each logdir (defaults to dir name)")
    p.add_argument("--ckptids", nargs="*", default=None,
                   help="Checkpoint id per logdir ('last' or zero-padded int). "
                        "Defaults to 'last' for all.")
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--no_headless", dest="headless", action="store_false")
    p.add_argument("--sim_device", type=str, default="cuda:0")
    p.add_argument("--robot", type=str, default="go2", choices=["go1", "go2"])
    p.add_argument("--num_envs", type=int, default=32)
    p.add_argument("--num_eval_steps", type=int, default=2000,
                   help="Sim steps per scenario point (per env)")
    p.add_argument("--arm_intensity", type=float, default=1.0,
                   help="Arm disturbance for scenarios A/C/D (0=fixed arm, 1=full)")
    p.add_argument("--output_dir", type=str, default="benchmark_results")
    p.add_argument("--skip_a", action="store_true", help="Skip velocity grid (Scenario A)")
    p.add_argument("--skip_b", action="store_true", help="Skip arm sweep (Scenario B)")
    p.add_argument("--skip_c", action="store_true", help="Skip body-pose tracking (Scenario C)")
    p.add_argument("--skip_d", action="store_true", help="Skip gait tracking (Scenario D)")
    # Stage-2 reserved interface
    p.add_argument("--stage2", action="store_true",
                   help="[Reserved] Evaluate in stage-2 hybrid mode (not yet implemented)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.stage2:
        print("[WARNING] --stage2 is reserved and not yet implemented; proceeding as stage-1.")

    n_runs = len(args.logdirs)
    names = (args.names or [])[:n_runs]
    while len(names) < n_runs:
        names.append(Path(args.logdirs[len(names)]).name[:24])

    ckptids_raw = (args.ckptids or [])[:n_runs]
    while len(ckptids_raw) < n_runs:
        ckptids_raw.append("last")
    ckptids = [("last" if c == "last" else c.zfill(6)) for c in ckptids_raw]

    # Use first logdir's config for the env (assert compat for the rest)
    print(f"[Benchmark] Creating env from {args.logdirs[0]}")
    env, cfg = load_env_benchmark(
        logdir=args.logdirs[0],
        num_envs=args.num_envs,
        headless=args.headless,
        device=args.sim_device,
        robot=args.robot,
    )
    configure_privileged_obs_dims(cfg)
    layout = detect_command_layout(cfg)

    print(f"[Benchmark] Command layout: {layout.n_dims} dims  "
          f"pose={'✓' if layout.has_body_pitch else '✗'}  "
          f"dyn_gait={'✓' if layout.has_dynamic_gait else '✗'}  "
          f"base_h_target={layout.base_height_target:.3f}m")

    all_results: Dict[str, Dict[str, List[ScenarioResult]]] = {}

    for run_name, logdir, ckpt_id in zip(names, args.logdirs, ckptids):
        print(f"\n{'='*60}")
        print(f"[Benchmark] Run: {run_name}  logdir={logdir}  ckpt={ckpt_id}")
        print(f"{'='*60}")

        dog_policy = load_dog_policy(logdir, ckpt_id, cfg)
        run_results: Dict[str, List[ScenarioResult]] = {}

        if not args.skip_a:
            run_results["vel_grid"] = run_scenario_a(
                env, dog_policy, layout, args.num_eval_steps,
                args.arm_intensity, args.sim_device, run_name,
            )

        if not args.skip_b:
            run_results["arm_sweep"] = run_scenario_b(
                env, dog_policy, layout, args.num_eval_steps,
                args.sim_device, run_name,
            )

        if not args.skip_c:
            run_results["body_pose"] = run_scenario_c(
                env, dog_policy, layout, args.num_eval_steps,
                args.arm_intensity, args.sim_device, run_name,
            )

        if not args.skip_d:
            run_results["gait"] = run_scenario_d(
                env, dog_policy, layout, args.num_eval_steps,
                args.arm_intensity, args.sim_device, run_name,
            )

        all_results[run_name] = run_results

    print_comparison_table(all_results)

    # Build output filename from all run names
    run_tag = "_vs_".join(names[:3]) + (f"_+{n_runs-3}" if n_runs > 3 else "")
    output_path = os.path.join(args.output_dir, f"stage1_{run_tag}.json")
    save_results(all_results, output_path)


if __name__ == "__main__":
    main()
