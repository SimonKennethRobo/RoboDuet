"""Dog-only policy benchmark with GPU-parallel multi-policy evaluation.

All policies under comparison share **one** IsaacGym simulation.
``total_envs = num_envs_per_policy × N_policies``.
Each policy owns a contiguous slice of envs; ``env.step()`` advances every
slice simultaneously, so the expensive GPU step is paid only once per
scenario point regardless of how many policies are compared.

Scenarios
---------
A  Velocity command grid
   Profile-configurable vx/vy/yaw grid; arm held at ``--arm_intensity``.

B  Arm-disturbance robustness sweep
   Fixed forward velocity (1.0 m/s) at arm intensities [0, 0.25, 0.5, 0.75, 1.0].

C  Body-pose command tracking  (requires dog_num_commands ≥ 4)
   Sweeps pitch, roll, and height-delta commands; reports both angle RMSE and
   the training orientation-control error where available.

D  Gait-parameter tracking  (dog_num_commands >= 9)
   Sweeps gait_frequency, stance_width, and stance_length where supported; reports
   command RMSE plus training-aligned contact and foot-placement diagnostics.

Usage::

    # single run
    python -m benchmark.dog_policy.cli \\
        --logdirs runs/my_run --ckptids last --headless

    # multi-run GPU-parallel comparison (N policies, one shared sim)
    python -m benchmark.dog_policy.cli \\
        --logdirs runs/run_A runs/run_B runs/run_C \\
        --names v1 v2 v3 --ckptids last last 040000 \\
        --headless --num_envs_per_policy 32 --num_eval_steps 2000

Stage-2 hook: ``--stage2`` is reserved (not yet implemented).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import isaacgym  # noqa: F401 – must precede torch
import numpy as np
import torch

from benchmark.candidates import discover_run_logdirs
from benchmark.dog_policy.evaluation import (
    CommandLayout,
    HistoryWrapper,
    PolicyHandle,
    ScenarioResult,
    _acc_to_result,
    _eval_loop_parallel,
    detect_command_layout,
    load_dog_policy_for_benchmark,
    load_env_benchmark,
    print_comparison_table,
    read_dog_num_commands,
    save_metadata,
    save_results,
    set_gait_cmd,
    set_pose_cmd,
    set_vel_cmd,
    validate_shared_env_compatibility,
)
from benchmark.metadata import build_benchmark_metadata

# ---------------------------------------------------------------------------
# Scenario command grids
# ---------------------------------------------------------------------------

VEL_GRID: List[tuple] = [(xv, 0.0, yaw) for xv in [-0.5, 0.0, 0.5, 1.0, 1.5] for yaw in [-1.0, 0.0, 1.0]]

ARM_INTENSITY_SWEEP = [0.0, 0.25, 0.5, 0.75, 1.0]
FORWARD_CMD = (1.0, 0.0, 0.0)
VELOCITY_GROUPS = {
    "stand": (0.0, 0.0, 0.0),
    "forward": (0.5, 0.0, 0.0),
    "lateral": (0.0, 0.5, 0.0),
    "turn": (0.0, 0.0, 0.5),
}
FIXED_GAIT_CMD = {
    "gait_freq": 4.0,
    "stance_width": 0.30,
    "stance_length": 0.35,
    "footswing_height": 0.06,
    "gait_duration": 0.5,
}
ZERO_POSE_CMD = {"pitch": 0.0, "roll": 0.0, "height_delta": 0.0}

PITCH_CMDS = [-0.3, -0.15, 0.0, 0.15, 0.3]  # rad
ROLL_CMDS = [-0.3, -0.15, 0.0, 0.15, 0.3]  # rad
HEIGHT_DELTA_CMDS = [-0.05, 0.0, 0.05, 0.10]  # m
GAIT_FREQ_CMDS = [1.5, 2.0, 2.5, 3.0, 3.5]  # Hz
FOOTSWING_HEIGHT_CMDS = [0.04, 0.08, 0.12, 0.16]  # m
STANCE_WIDTH_CMDS = [0.25, 0.30, 0.35, 0.40]  # m
STANCE_LENGTH_CMDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]  # m

# ---------------------------------------------------------------------------
# Scenario runners - each returns {run_name: [ScenarioResult]}
# ---------------------------------------------------------------------------

ResultsMap = Dict[str, List[ScenarioResult]]
Point = tuple  # (label, cmd_fn, extra_kwargs)


def _empty_results(handles: List[PolicyHandle]) -> ResultsMap:
    return {h.name: [] for h in handles}


def _scenario_cfg(scenario_config: Optional[dict], scenario: str) -> dict:
    if not scenario_config:
        return {}
    return scenario_config.get(scenario, {})


def _list_cfg(cfg: dict, key: str, default: List[float]) -> List[float]:
    values = cfg.get(key, default)
    return [float(v) for v in values]


def _cmd_tuple(cfg: dict, key: str, default: tuple) -> tuple:
    values = cfg.get(key, default)
    if isinstance(values, dict):
        return (
            float(values.get("vx", values.get("x", default[0]))),
            float(values.get("vy", values.get("y", default[1]))),
            float(values.get("yaw", default[2])),
        )
    return tuple(float(v) for v in values)


def _fixed_gait_cfg(cfg: dict) -> dict:
    merged = dict(FIXED_GAIT_CMD)
    merged.update(cfg.get("fixed_gait", {}))
    return merged


def _fixed_pose_cfg(cfg: dict) -> dict:
    merged = dict(ZERO_POSE_CMD)
    merged.update(cfg.get("fixed_pose", {}))
    return merged


def _apply_fixed_gait(env: HistoryWrapper, gait: dict):
    set_gait_cmd(
        env,
        gait_freq=float(gait.get("gait_freq", FIXED_GAIT_CMD["gait_freq"])),
        footswing_height=float(gait.get("footswing_height", FIXED_GAIT_CMD["footswing_height"])),
        stance_width=float(gait.get("stance_width", FIXED_GAIT_CMD["stance_width"])),
        stance_length=float(gait.get("stance_length", FIXED_GAIT_CMD["stance_length"])),
        gait_duration=float(gait.get("gait_duration", FIXED_GAIT_CMD["gait_duration"])),
    )


def _apply_fixed_pose(env: HistoryWrapper, pose: dict):
    set_pose_cmd(
        env,
        float(pose.get("pitch", ZERO_POSE_CMD["pitch"])),
        float(pose.get("roll", ZERO_POSE_CMD["roll"])),
        float(pose.get("height_delta", ZERO_POSE_CMD["height_delta"])),
    )


def _command_fn(
    velocity: tuple,
    gait: Optional[dict] = None,
    pose: Optional[dict] = None,
    extra_cmd: Optional[Callable] = None,
) -> Callable:
    xv, yv, yaw = velocity

    def cmd_fn(env):
        set_vel_cmd(env, xv, yv, yaw)
        if gait is not None:
            _apply_fixed_gait(env, gait)
        if pose is not None:
            _apply_fixed_pose(env, pose)
        if extra_cmd is not None:
            extra_cmd(env)

    return cmd_fn


def _forward_cmd_fn(extra_cmd: Optional[Callable] = None) -> Callable:
    return _command_fn(FORWARD_CMD, extra_cmd=extra_cmd)


def _fmt_metric(results: List[ScenarioResult], metrics: List[tuple]) -> str:
    parts = []
    for r in results:
        vals = []
        for label, attr, scale, fmt in metrics:
            raw = getattr(r, attr)
            if raw is None:
                vals.append(f"{label}=—")
                continue
            val = raw * scale
            suffix = "%" if fmt.endswith("%") else ""
            fmt_spec = fmt[:-1] if suffix else fmt
            vals.append(f"{label}={val:{fmt_spec}}{suffix}")
        parts.append(f"{r.run_name}: " + " ".join(vals))
    return "  ".join(parts)


def _run_points(
    env: HistoryWrapper,
    handles: List[PolicyHandle],
    layout: CommandLayout,
    n_steps: int,
    default_arm_intensity: Optional[float],
    device: str,
    points: List[Point],
    scenario: str,
    header: str,
    fmt_fn: Callable,
    indent: str = "  ",
) -> ResultsMap:
    out = _empty_results(handles)
    print(header)
    total = len(points)
    for i, (label, cmd_fn, extra_kw) in enumerate(points):
        arm_intensity = extra_kw.get("arm_intensity", default_arm_intensity)
        if arm_intensity is None:
            raise ValueError(f"{scenario}/{label}: arm_intensity was not provided")
        print(f"{indent}[{i + 1:2d}/{total}] {label}", end="  ", flush=True)
        accs = _eval_loop_parallel(env, handles, layout, n_steps, arm_intensity, device, cmd_fn)
        point_results = []
        for h, acc in zip(handles, accs):
            result = _acc_to_result(acc, layout, h.name, scenario, label, n_steps, **extra_kw)
            out[h.name].append(result)
            point_results.append(result)
        print(fmt_fn(point_results))
    return out


def _run_sweeps(
    env: HistoryWrapper,
    handles: List[PolicyHandle],
    layout: CommandLayout,
    n_steps: int,
    arm_intensity: float,
    device: str,
    scenario: str,
    header: str,
    sweeps: List[tuple],
) -> ResultsMap:
    out = _empty_results(handles)
    print(header)
    for name, points, fmt_fn in sweeps:
        partial = _run_points(
            env,
            handles,
            layout,
            n_steps,
            arm_intensity,
            device,
            points,
            scenario,
            f"  {name} sweep:",
            fmt_fn,
            indent="    ",
        )
        for run_name, results in partial.items():
            out[run_name].extend(results)
    return out


def run_scenario_a(
    env: HistoryWrapper,
    handles: List[PolicyHandle],
    layout: CommandLayout,
    n_steps: int,
    arm_intensity: float,
    device: str,
    scenario_config: Optional[dict] = None,
) -> ResultsMap:
    cfg = _scenario_cfg(scenario_config, "vel_grid")
    fixed_gait = _fixed_gait_cfg(cfg)
    fixed_pose = _fixed_pose_cfg(cfg)
    vx_values = _list_cfg(cfg, "vx", sorted({xv for xv, _, _ in VEL_GRID}))
    vy_values = _list_cfg(cfg, "vy", sorted({yv for _, yv, _ in VEL_GRID}))
    yaw_values = _list_cfg(cfg, "yaw", sorted({yaw for _, _, yaw in VEL_GRID}))
    grid = [(xv, yv, yaw) for xv in vx_values for yv in vy_values for yaw in yaw_values]
    points = [
        (
            f"vx={xv:+.1f} vy={yv:+.1f} yaw={yaw:+.1f}",
            _command_fn((xv, yv, yaw), fixed_gait, fixed_pose),
            dict(
                cmd_x=xv,
                cmd_y=yv,
                cmd_yaw=yaw,
                cmd_pitch=float(fixed_pose["pitch"]),
                cmd_roll=float(fixed_pose["roll"]),
                cmd_height_delta=float(fixed_pose["height_delta"]),
                cmd_gait_freq=float(fixed_gait["gait_freq"]),
                cmd_footswing_height=float(fixed_gait["footswing_height"]),
                cmd_stance_width=float(fixed_gait["stance_width"]),
                cmd_stance_length=float(fixed_gait["stance_length"]),
                cmd_gait_duration=float(fixed_gait["gait_duration"]),
                arm_intensity=arm_intensity,
            ),
        )
        for xv, yv, yaw in grid
    ]
    return _run_points(
        env,
        handles,
        layout,
        n_steps,
        arm_intensity,
        device,
        points,
        "vel_grid",
        f"\n[A] Velocity grid  arm_intensity={arm_intensity:.2f}  "
        f"{len(points)} points x {n_steps} steps  {len(handles)} policies in parallel",
        lambda rs: _fmt_metric(
            rs,
            [
                ("vx", "lin_vel_x_rmse", 1.0, ".4f"),
                ("vy", "lin_vel_y_rmse", 1.0, ".4f"),
                ("yaw", "ang_vel_yaw_rmse", 1.0, ".4f"),
                ("fall_h", "fall_rate_height", 100.0, ".1f%"),
            ],
        ),
    )


def run_scenario_b(
    env: HistoryWrapper,
    handles: List[PolicyHandle],
    layout: CommandLayout,
    n_steps: int,
    device: str,
    scenario_config: Optional[dict] = None,
) -> ResultsMap:
    cfg = _scenario_cfg(scenario_config, "arm_sweep")
    fixed_gait = _fixed_gait_cfg(cfg)
    fixed_pose = _fixed_pose_cfg(cfg)
    xv, yv, yaw = _cmd_tuple(cfg, "fixed_velocity", FORWARD_CMD)
    intensities = _list_cfg(cfg, "arm_intensity", ARM_INTENSITY_SWEEP)
    disturbance_seed = cfg.get("disturbance_seed")
    points = [
        (
            f"intensity={intensity:.2f}",
            _command_fn((xv, yv, yaw), fixed_gait, fixed_pose),
            dict(
                cmd_x=xv,
                cmd_y=yv,
                cmd_yaw=yaw,
                cmd_pitch=float(fixed_pose["pitch"]),
                cmd_roll=float(fixed_pose["roll"]),
                cmd_height_delta=float(fixed_pose["height_delta"]),
                cmd_gait_freq=float(fixed_gait["gait_freq"]),
                cmd_footswing_height=float(fixed_gait["footswing_height"]),
                cmd_stance_width=float(fixed_gait["stance_width"]),
                cmd_stance_length=float(fixed_gait["stance_length"]),
                cmd_gait_duration=float(fixed_gait["gait_duration"]),
                arm_intensity=intensity,
                disturbance_seed=disturbance_seed,
            ),
        )
        for intensity in intensities
    ]
    return _run_points(
        env,
        handles,
        layout,
        n_steps,
        None,
        device,
        points,
        "arm_sweep",
        f"\n[B] Arm-disturbance sweep  cmd=({xv},{yv},{yaw})  "
        f"{len(points)} levels x {n_steps} steps  {len(handles)} policies in parallel",
        lambda rs: _fmt_metric(
            rs,
            [
                ("xy", "lin_vel_xy_rmse", 1.0, ".4f"),
                ("vx", "lin_vel_x_rmse", 1.0, ".4f"),
                ("fall_h", "fall_rate_height", 100.0, ".1f%"),
            ],
        ),
    )


def run_scenario_c(
    env: HistoryWrapper,
    handles: List[PolicyHandle],
    layout: CommandLayout,
    n_steps: int,
    arm_intensity: float,
    device: str,
    scenario_config: Optional[dict] = None,
) -> ResultsMap:
    if not layout.has_body_pitch:
        print("\n[C] Body-pose tracking  SKIPPED (dog_num_commands < 4)")
        return _empty_results(handles)

    cfg = _scenario_cfg(scenario_config, "body_pose")
    fixed_gait = _fixed_gait_cfg(cfg)
    velocity_groups = cfg.get("velocity_groups", VELOCITY_GROUPS)
    pitch_cmds = _list_cfg(cfg, "pitch", PITCH_CMDS)
    roll_cmds = _list_cfg(cfg, "roll", ROLL_CMDS)
    height_delta_cmds = _list_cfg(cfg, "height_delta", HEIGHT_DELTA_CMDS)

    def _base_kw(group_name: str, vel: tuple, pose_axis: str, pose: dict) -> dict:
        xv, yv, yaw = vel
        return dict(
            cmd_x=xv,
            cmd_y=yv,
            cmd_yaw=yaw,
            cmd_pitch=float(pose.get("pitch", 0.0)),
            cmd_roll=float(pose.get("roll", 0.0)),
            cmd_height_delta=float(pose.get("height_delta", 0.0)),
            cmd_gait_freq=float(fixed_gait["gait_freq"]),
            cmd_footswing_height=float(fixed_gait["footswing_height"]),
            cmd_stance_width=float(fixed_gait["stance_width"]),
            cmd_stance_length=float(fixed_gait["stance_length"]),
            cmd_gait_duration=float(fixed_gait["gait_duration"]),
            arm_intensity=arm_intensity,
            velocity_group=group_name,
            pose_axis=pose_axis,
            sweep_axis=pose_axis,
        )

    sweeps = []
    for group_name, vel_value in velocity_groups.items():
        vel = _cmd_tuple({"velocity": vel_value}, "velocity", FORWARD_CMD)
        sweeps.append(
            (
                f"{group_name} Pitch",
                [
                    (
                        f"{group_name} | pitch={pitch:+.2f}rad",
                        _command_fn(vel, fixed_gait, {"pitch": pitch, "roll": 0.0, "height_delta": 0.0}),
                        _base_kw(group_name, vel, "pitch", {"pitch": pitch}),
                    )
                    for pitch in pitch_cmds
                ],
                lambda rs: _fmt_metric(rs, [("pitch", "pitch_rmse_deg", 1.0, ".3f"), ("fall_h", "fall_rate_height", 100.0, ".1f%")]),
            )
        )
        if layout.has_body_roll:
            sweeps.append(
                (
                    f"{group_name} Roll",
                    [
                        (
                            f"{group_name} | roll={roll:+.2f}rad",
                            _command_fn(vel, fixed_gait, {"pitch": 0.0, "roll": roll, "height_delta": 0.0}),
                            _base_kw(group_name, vel, "roll", {"roll": roll}),
                        )
                        for roll in roll_cmds
                    ],
                    lambda rs: _fmt_metric(
                        rs, [("roll", "roll_rmse_deg", 1.0, ".3f"), ("fall_h", "fall_rate_height", 100.0, ".1f%")]
                    ),
                )
            )
        if layout.has_body_height:
            sweeps.append(
                (
                    f"{group_name} Height",
                    [
                        (
                            f"{group_name} | height_delta={delta:+.2f}m",
                            _command_fn(vel, fixed_gait, {"pitch": 0.0, "roll": 0.0, "height_delta": delta}),
                            _base_kw(group_name, vel, "height", {"height_delta": delta}),
                        )
                        for delta in height_delta_cmds
                    ],
                    lambda rs: _fmt_metric(rs, [("h", "height_rmse_m", 1.0, ".4f"), ("fall_h", "fall_rate_height", 100.0, ".1f%")]),
                )
            )
    return _run_sweeps(
        env,
        handles,
        layout,
        n_steps,
        arm_intensity,
        device,
        "body_pose",
        f"\n[C] Body-pose tracking  arm_intensity={arm_intensity:.2f}  {len(handles)} policies in parallel",
        sweeps,
    )


def run_scenario_d(
    env: HistoryWrapper,
    handles: List[PolicyHandle],
    layout: CommandLayout,
    n_steps: int,
    arm_intensity: float,
    device: str,
    scenario_config: Optional[dict] = None,
) -> ResultsMap:
    if not layout.has_dynamic_gait:
        print("\n[D] Gait-parameter tracking  SKIPPED (requires dog_num_commands >= 9)")
        return _empty_results(handles)

    cfg = _scenario_cfg(scenario_config, "gait")
    fixed_gait = _fixed_gait_cfg(cfg)
    fixed_pose = _fixed_pose_cfg(cfg)
    xv, yv, yaw = _cmd_tuple(cfg, "fixed_velocity", (0.5, 0.0, 0.0))
    freq_cmds = _list_cfg(cfg, "gait_freq", GAIT_FREQ_CMDS)
    width_cmds = _list_cfg(cfg, "stance_width", STANCE_WIDTH_CMDS)
    length_cmds = _list_cfg(cfg, "stance_length", STANCE_LENGTH_CMDS)
    base_kw = dict(
        cmd_x=xv,
        cmd_y=yv,
        cmd_yaw=yaw,
        cmd_pitch=float(fixed_pose["pitch"]),
        cmd_roll=float(fixed_pose["roll"]),
        cmd_height_delta=float(fixed_pose["height_delta"]),
        cmd_footswing_height=float(fixed_gait["footswing_height"]),
        cmd_gait_duration=float(fixed_gait["gait_duration"]),
        arm_intensity=arm_intensity,
    )
    sweeps = [
        (
            "Gait-frequency",
            [
                (
                    f"gait_freq={freq:.1f}Hz",
                    _command_fn((xv, yv, yaw), fixed_gait, fixed_pose, lambda e, _f=freq: set_gait_cmd(e, gait_freq=_f)),
                    dict(
                        base_kw,
                        cmd_gait_freq=freq,
                        cmd_stance_width=float(fixed_gait["stance_width"]),
                        cmd_stance_length=float(fixed_gait["stance_length"]),
                        sweep_axis="gait_freq",
                    ),
                )
                for freq in freq_cmds
            ],
            lambda rs: _fmt_metric(
                rs,
                [
                    ("contact_f", "gait_contact_force_cost", 1.0, ".4f"),
                    ("contact_v", "gait_contact_vel_cost", 1.0, ".4f"),
                    ("fall_h", "fall_rate_height", 100.0, ".1f%"),
                ],
            ),
        ),
        (
            "Stance-width",
            [
                (
                    f"stance_w={width:.2f}m",
                    _command_fn((xv, yv, yaw), fixed_gait, fixed_pose, lambda e, _w=width: set_gait_cmd(e, stance_width=_w)),
                    dict(
                        base_kw,
                        cmd_gait_freq=float(fixed_gait["gait_freq"]),
                        cmd_stance_width=width,
                        cmd_stance_length=float(fixed_gait["stance_length"]),
                        sweep_axis="stance_width",
                    ),
                )
                for width in width_cmds
            ],
            lambda rs: _fmt_metric(
                rs, [("stance_w", "stance_width_rmse_m", 1.0, ".4f"), ("fall_h", "fall_rate_height", 100.0, ".1f%")]
            ),
        ),
    ]
    if layout.has_stance_length:
        sweeps.append(
            (
                "Stance-length",
                [
                    (
                        f"stance_l={length:.2f}m",
                        _command_fn((xv, yv, yaw), fixed_gait, fixed_pose, lambda e, _l=length: set_gait_cmd(e, stance_length=_l)),
                        dict(
                            base_kw,
                            cmd_gait_freq=float(fixed_gait["gait_freq"]),
                            cmd_stance_width=float(fixed_gait["stance_width"]),
                            cmd_stance_length=length,
                            sweep_axis="stance_length",
                        ),
                    )
                    for length in length_cmds
                ],
                lambda rs: _fmt_metric(
                    rs, [("stance_l", "stance_length_rmse_m", 1.0, ".4f"), ("fall_h", "fall_rate_height", 100.0, ".1f%")]
                ),
            )
        )
    return _run_sweeps(
        env,
        handles,
        layout,
        n_steps,
        arm_intensity,
        device,
        "gait",
        f"\n[D] Gait-parameter tracking  arm_intensity={arm_intensity:.2f}  {len(handles)} policies in parallel",
        sweeps,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


SCENARIO_FLAGS = {
    "vel_grid": "skip_a",
    "arm_sweep": "skip_b",
    "body_pose": "skip_c",
    "gait": "skip_d",
}

def _load_json_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _cli_option_was_provided(option: str) -> bool:
    import sys

    prefix = option + "="
    return option in sys.argv[1:] or any(arg.startswith(prefix) for arg in sys.argv[1:])


def _apply_profile(args):
    args.scenario_config = {}
    args.profile_data = {}
    if not args.profile:
        return

    profile = _load_json_config(args.profile)
    args.profile_data = profile
    simple_fields = [
        "headless",
        "sim_device",
        "robot",
        "num_envs_per_policy",
        "num_eval_steps",
        "seed",
        "arm_intensity",
        "output_dir",
        "benchmark_protocol",
    ]
    for field in simple_fields:
        option = "--" + field
        if field in profile and not _cli_option_was_provided(option):
            setattr(args, field, profile[field])

    scenarios = profile.get("scenarios")
    if scenarios is not None:
        unknown = sorted(set(scenarios) - set(SCENARIO_FLAGS))
        if unknown:
            raise ValueError(f"{args.profile}: unknown benchmark scenarios: {', '.join(unknown)}")
        for scenario, skip_attr in SCENARIO_FLAGS.items():
            skip_option = "--" + skip_attr
            if not _cli_option_was_provided(skip_option):
                setattr(args, skip_attr, scenario not in scenarios)
    args.scenario_config = profile.get("scenario_config", {})


def _discover_candidate_logdirs(candidate_dir: str) -> List[Path]:
    root = Path(candidate_dir)
    logdirs = []
    for logdir in discover_run_logdirs(root):
        has_dog = (logdir / "checkpoints_dog").is_dir()
        if has_dog:
            logdirs.append(logdir)

    if not logdirs:
        raise ValueError(f"{candidate_dir}: no dog-policy candidates found")
    return logdirs


def _dog_checkpoint_path(logdir: Path, ckpt_id: str) -> Path:
    ckpt_id_ = "last_dog" if ckpt_id == "last" else ckpt_id.zfill(6)
    return logdir / "checkpoints_dog" / f"ac_weights_{ckpt_id_}.pt"


def _validate_dog_logdir(logdir: str, ckpt_id: str):
    path = Path(logdir)
    missing = []
    for rel in ("parameters.pkl", "params.txt", "checkpoints_dog"):
        candidate = path / rel
        if rel == "checkpoints_dog":
            exists = candidate.is_dir()
        else:
            exists = candidate.is_file()
        if not exists:
            missing.append(rel)

    ckpt_path = _dog_checkpoint_path(path, ckpt_id)
    if not ckpt_path.is_file():
        missing.append(str(ckpt_path.relative_to(path)))

    if missing:
        raise ValueError(
            f"{logdir}: invalid dog-policy benchmark candidate; missing " + ", ".join(missing)
        )


def _validate_dog_logdirs(logdirs: List[str], ckptids: List[str]):
    for logdir, ckpt_id in zip(logdirs, ckptids):
        _validate_dog_logdir(logdir, ckpt_id)


def _apply_candidate_dir(args):
    if not args.candidate_dir:
        if not args.logdirs:
            raise ValueError("Provide either --logdirs or --candidate_dir")
        return

    logdirs = _discover_candidate_logdirs(args.candidate_dir)
    args.logdirs = [str(path) for path in logdirs]
    args.names = [path.name[:24] for path in logdirs]


def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description="Dog-only policy benchmark (GPU-parallel)")
    p.add_argument("--logdirs", nargs="+", default=None)
    p.add_argument("--names", nargs="*", default=None, help="Display name per logdir (default: directory name)")
    p.add_argument("--ckptids", nargs="*", default=None, help="Checkpoint id per logdir (default: 'last' for all)")
    p.add_argument("--candidate_dir", type=str, default=None, help="Run-like candidate root directory")
    p.add_argument("--profile", type=str, default=None, help="JSON benchmark profile path")
    p.add_argument("--headless", action="store_true", default=False)
    p.add_argument("--sim_device", type=str, default="cuda:0")
    p.add_argument("--robot", type=str, default="go2", choices=["go1", "go2"])
    p.add_argument(
        "--num_envs_per_policy",
        type=int,
        default=32,
        help="Envs per policy. Total envs = num_envs_per_policy × N_policies.",
    )
    p.add_argument("--num_eval_steps", type=int, default=100, help="Sim steps per scenario point (per env)")
    p.add_argument("--seed", type=int, default=1, help="Benchmark RNG seed, independent from training seed")
    p.add_argument("--arm_intensity", type=float, default=1.0, help="Arm disturbance for scenarios A/C/D")
    p.add_argument("--output_dir", type=str, default="benchmark/results")
    p.add_argument("--benchmark_protocol", type=str, default="dog_only")
    p.add_argument("--skip_a", action="store_true")
    p.add_argument("--skip_b", action="store_true")
    p.add_argument("--skip_c", action="store_true")
    p.add_argument("--skip_d", action="store_true")
    p.add_argument("--stage2", action="store_true", help="[Reserved] Stage-2 hybrid evaluation (not yet implemented)")
    args = p.parse_args(argv)
    args.scenario_config = {}
    args.profile_data = {}
    _apply_profile(args)
    _apply_candidate_dir(args)
    return args


def set_benchmark_seed(seed: int, device: str):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _profile_requires_full_gait_layout(args) -> bool:
    scenario_config = args.scenario_config or {}
    for scenario, skip_attr in SCENARIO_FLAGS.items():
        if getattr(args, skip_attr):
            continue
        cfg = scenario_config.get(scenario, {})
        if not isinstance(cfg, dict):
            continue
        if "stance_length" in cfg or "gait_duration" in cfg:
            return True
        fixed_gait = cfg.get("fixed_gait")
        if isinstance(fixed_gait, dict) and (
            "stance_length" in fixed_gait or "gait_duration" in fixed_gait
        ):
            return True
    return False


def _validate_layout_for_profile(args, layout: CommandLayout):
    if not _profile_requires_full_gait_layout(args):
        return
    if not layout.has_gait_duration:
        raise ValueError(
            "Profile gait scenario requires dog_num_commands >= 11 "
            "(indices 9=stance_length and 10=gait_duration). "
            f"Current runtime layout has dog_num_commands={layout.n_dims}. "
            "Use the smoke profile or a checkpoint trained with the full dog command layout."
        )


def _validate_checkpoint_layout_for_profile(args, base_logdir: str):
    if not _profile_requires_full_gait_layout(args):
        return
    n_dims = read_dog_num_commands(base_logdir)
    if n_dims < 11:
        raise ValueError(
            "Profile gait scenario requires dog_num_commands >= 11 "
            "(indices 9=stance_length and 10=gait_duration). "
            f"Base checkpoint parameters.pkl has dog_num_commands={n_dims}. "
            "Use the smoke profile or a checkpoint trained with the full dog command layout."
        )


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)

    if args.stage2:
        print("[WARNING] --stage2 is reserved and not yet implemented.")

    n_runs = len(args.logdirs)
    names = list((args.names or [])[:n_runs])
    while len(names) < n_runs:
        names.append(Path(args.logdirs[len(names)]).name[:24])

    ckptids_raw = list((args.ckptids or [])[:n_runs])
    while len(ckptids_raw) < n_runs:
        ckptids_raw.append("last")
    ckptids = [("last" if c == "last" else c.zfill(6)) for c in ckptids_raw]
    _validate_dog_logdirs(args.logdirs, ckptids)

    num_envs_per_policy = args.num_envs_per_policy
    total_envs = num_envs_per_policy * n_runs

    print(f"[Benchmark] {n_runs} policies × {num_envs_per_policy} envs = {total_envs} total envs")
    print(f"[Benchmark] Seed = {args.seed}")
    set_benchmark_seed(args.seed, args.sim_device)
    validate_shared_env_compatibility(args.logdirs[0], args.logdirs[1:])
    _validate_checkpoint_layout_for_profile(args, args.logdirs[0])
    print(f"[Benchmark] Creating shared env from {args.logdirs[0]}")

    env, cfg = load_env_benchmark(
        logdir=args.logdirs[0],
        total_envs=total_envs,
        envs_per_policy=num_envs_per_policy,
        headless=args.headless,
        device=args.sim_device,
        robot=args.robot,
    )
    layout = detect_command_layout(cfg)
    _validate_layout_for_profile(args, layout)

    print(
        f"[Benchmark] Layout: {layout.n_dims} cmd dims  "
        f"pose={'on' if layout.has_body_pitch else 'off'}  "
        f"gait_metrics={'on' if layout.has_dynamic_gait else 'off'}  "
        f"stance_length={'on' if layout.has_stance_length else 'off'}  "
        f"base_h={layout.base_height_target:.3f}m"
    )

    # Build one PolicyHandle per run; each owns a contiguous env slice.
    # The shared-sim benchmark intentionally requires identical obs/control
    # semantics across all compared checkpoints.
    print("[Benchmark] Loading policies...")
    handles: List[PolicyHandle] = []
    for i, (name, logdir, ckpt_id) in enumerate(zip(names, args.logdirs, ckptids)):
        s = i * num_envs_per_policy
        e = s + num_envs_per_policy
        print(f"  [{i + 1}/{n_runs}] {name:24s}  envs [{s}:{e})  ckpt={ckpt_id}")
        policy = load_dog_policy_for_benchmark(logdir, ckpt_id, cfg)
        handles.append(PolicyHandle(name=name, policy=policy, env_start=s, env_end=e))

    all_results: Dict[str, Dict[str, List[ScenarioResult]]] = {h.name: {} for h in handles}

    def _merge(scenario_key: str, per_policy: ResultsMap):
        for name, results in per_policy.items():
            all_results[name][scenario_key] = results

    if not args.skip_a:
        _merge(
            "vel_grid",
            run_scenario_a(
                env,
                handles,
                layout,
                args.num_eval_steps,
                args.arm_intensity,
                args.sim_device,
                args.scenario_config,
            ),
        )

    if not args.skip_b:
        _merge("arm_sweep", run_scenario_b(env, handles, layout, args.num_eval_steps, args.sim_device, args.scenario_config))

    if not args.skip_c:
        _merge(
            "body_pose",
            run_scenario_c(
                env,
                handles,
                layout,
                args.num_eval_steps,
                args.arm_intensity,
                args.sim_device,
                args.scenario_config,
            ),
        )

    if not args.skip_d:
        _merge("gait", run_scenario_d(env, handles, layout, args.num_eval_steps, args.arm_intensity, args.sim_device, args.scenario_config))

    print_comparison_table(all_results)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    json_path = os.path.join(run_dir, "results.json")
    metadata_path = os.path.join(run_dir, "metadata.json")
    metadata = build_benchmark_metadata(
        mode="dog_only",
        protocol=args.benchmark_protocol,
        profile=args.profile or "cli",
        candidate_dir=args.candidate_dir or "cli",
        names=names,
        logdirs=args.logdirs,
        ckptids=ckptids,
        args=args,
        total_envs=total_envs,
        control_dt_s=getattr(env.env, "dt", "unknown"),
        layout=layout,
        command_argv=sys.argv,
    )
    save_results(all_results, json_path)
    save_metadata(metadata, metadata_path)
    try:
        from benchmark.reports.html import write_report_bundle

        write_report_bundle(json_path)
    except Exception as exc:
        print(f"[Benchmark] Skipping HTML report: {exc}")
    print(f"[Benchmark] Output directory → {run_dir}")


if __name__ == "__main__":
    main()
