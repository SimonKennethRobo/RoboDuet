"""Stage-1 dog-policy benchmark with GPU-parallel multi-policy evaluation.

All policies under comparison share **one** IsaacGym simulation.
``total_envs = num_envs_per_policy × N_policies``.
Each policy owns a contiguous slice of envs; ``env.step()`` advances every
slice simultaneously, so the expensive GPU step is paid only once per
scenario point regardless of how many policies are compared.

Scenarios
---------
A  Velocity command grid
   5 x-vel × 3 yaw-vel points; arm held at ``--arm_intensity``.

B  Arm-disturbance robustness sweep
   Fixed forward velocity (1.0 m/s) at arm intensities [0, 0.25, 0.5, 0.75, 1.0].

C  Body-pose command tracking  (requires dog_num_commands ≥ 4)
   Sweeps pitch, roll, and height-delta commands; reports both angle RMSE and
   the training orientation-control error where available.

D  Gait-parameter tracking  (dog_num_commands >= 9)
   Sweeps gait_frequency, footswing_height, and stance_width; reports the
   training-aligned contact, foot-clearance, and Raibert foot-placement costs.

Usage::

    # single run
    python scripts/benchmark_policy.py \\
        --logdirs runs/my_run --ckptids last --headless

    # multi-run GPU-parallel comparison (N policies, one shared sim)
    python scripts/benchmark_policy.py \\
        --logdirs runs/run_A runs/run_B runs/run_C \\
        --names v1 v2 v3 --ckptids last last 040000 \\
        --headless --num_envs_per_policy 32 --num_eval_steps 2000

Stage-2 hook: ``--stage2`` is reserved (not yet implemented).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

import isaacgym  # noqa: F401 – must precede torch

from scripts.benchmark_policy_common import (
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
    save_markdown_report,
    save_results,
    save_visualizations,
    set_gait_cmd,
    set_pose_cmd,
    set_vel_cmd,
    validate_shared_env_compatibility,
)

# ---------------------------------------------------------------------------
# Scenario command grids
# ---------------------------------------------------------------------------

VEL_GRID: List[tuple] = [(xv, 0.0, yaw) for xv in [-0.5, 0.0, 0.5, 1.0, 1.5] for yaw in [-1.0, 0.0, 1.0]]

ARM_INTENSITY_SWEEP = [0.0, 0.25, 0.5, 0.75, 1.0]
FORWARD_CMD = (1.0, 0.0, 0.0)

PITCH_CMDS = [-0.3, -0.15, 0.0, 0.15, 0.3]  # rad
ROLL_CMDS = [-0.3, -0.15, 0.0, 0.15, 0.3]  # rad
HEIGHT_DELTA_CMDS = [-0.05, 0.0, 0.05, 0.10]  # m
GAIT_FREQ_CMDS = [1.5, 2.0, 2.5, 3.0, 3.5]  # Hz
FOOTSWING_HEIGHT_CMDS = [0.04, 0.08, 0.12, 0.16]  # m
STANCE_WIDTH_CMDS = [0.25, 0.30, 0.35, 0.40]  # m

# ---------------------------------------------------------------------------
# Scenario runners - each returns {run_name: [ScenarioResult]}
# ---------------------------------------------------------------------------

ResultsMap = Dict[str, List[ScenarioResult]]
Point = tuple  # (label, cmd_fn, extra_kwargs)


def _empty_results(handles: List[PolicyHandle]) -> ResultsMap:
    return {h.name: [] for h in handles}


def _forward_cmd_fn(extra_cmd: Optional[Callable] = None) -> Callable:
    xv, yv, yaw = FORWARD_CMD

    def cmd_fn(env):
        set_vel_cmd(env, xv, yv, yaw)
        if extra_cmd is not None:
            extra_cmd(env)

    return cmd_fn


def _fmt_metric(results: List[ScenarioResult], metrics: List[tuple]) -> str:
    parts = []
    for r in results:
        vals = []
        for label, attr, scale, fmt in metrics:
            val = getattr(r, attr) * scale
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
) -> ResultsMap:
    points = [
        (
            f"vx={xv:+.1f} yaw={yaw:+.1f}",
            lambda e, _x=xv, _y=yv, _yaw=yaw: set_vel_cmd(e, _x, _y, _yaw),
            dict(cmd_x=xv, cmd_y=yv, cmd_yaw=yaw, arm_intensity=arm_intensity),
        )
        for xv, yv, yaw in VEL_GRID
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
                ("yaw", "ang_vel_yaw_rmse", 1.0, ".4f"),
                ("fall", "fall_rate", 100.0, ".1f%"),
            ],
        ),
    )


def run_scenario_b(
    env: HistoryWrapper,
    handles: List[PolicyHandle],
    layout: CommandLayout,
    n_steps: int,
    device: str,
) -> ResultsMap:
    xv, yv, yaw = FORWARD_CMD
    points = [
        (
            f"intensity={intensity:.2f}",
            lambda e, _x=xv, _y=yv, _yaw=yaw: set_vel_cmd(e, _x, _y, _yaw),
            dict(cmd_x=xv, cmd_y=yv, cmd_yaw=yaw, arm_intensity=intensity),
        )
        for intensity in ARM_INTENSITY_SWEEP
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
        lambda rs: _fmt_metric(rs, [("vx", "lin_vel_x_rmse", 1.0, ".4f"), ("fall", "fall_rate", 100.0, ".1f%")]),
    )


def run_scenario_c(
    env: HistoryWrapper,
    handles: List[PolicyHandle],
    layout: CommandLayout,
    n_steps: int,
    arm_intensity: float,
    device: str,
) -> ResultsMap:
    if not layout.has_body_pitch:
        print("\n[C] Body-pose tracking  SKIPPED (dog_num_commands < 4)")
        return _empty_results(handles)

    xv, yv, yaw = FORWARD_CMD
    base_kw = dict(cmd_x=xv, cmd_y=yv, cmd_yaw=yaw, arm_intensity=arm_intensity)
    sweeps = [
        (
            "Pitch",
            [
                (
                    f"pitch={pitch:+.2f}rad",
                    _forward_cmd_fn(lambda e, _p=pitch: set_pose_cmd(e, _p, 0.0, 0.0)),
                    dict(base_kw, cmd_pitch=pitch),
                )
                for pitch in PITCH_CMDS
            ],
            lambda rs: _fmt_metric(rs, [("pitch", "pitch_rmse_deg", 1.0, ".3f"), ("fall", "fall_rate", 100.0, ".1f%")]),
        )
    ]
    if layout.has_body_roll:
        sweeps.append(
            (
                "Roll",
                [
                    (
                        f"roll={roll:+.2f}rad",
                        _forward_cmd_fn(lambda e, _r=roll: set_pose_cmd(e, 0.0, _r, 0.0)),
                        dict(base_kw, cmd_roll=roll),
                    )
                    for roll in ROLL_CMDS
                ],
                lambda rs: _fmt_metric(
                    rs, [("roll", "roll_rmse_deg", 1.0, ".3f"), ("fall", "fall_rate", 100.0, ".1f%")]
                ),
            )
        )
    if layout.has_body_height:
        sweeps.append(
            (
                "Height",
                [
                    (
                        f"h_target={layout.base_height_target + delta:.3f}m",
                        _forward_cmd_fn(lambda e, _d=delta: set_pose_cmd(e, 0.0, 0.0, _d)),
                        dict(base_kw, cmd_height_delta=delta),
                    )
                    for delta in HEIGHT_DELTA_CMDS
                ],
                lambda rs: _fmt_metric(rs, [("h", "height_rmse_m", 1.0, ".4f"), ("fall", "fall_rate", 100.0, ".1f%")]),
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
) -> ResultsMap:
    if not layout.has_dynamic_gait:
        print("\n[D] Gait-parameter tracking  SKIPPED (requires dog_num_commands >= 9)")
        return _empty_results(handles)

    xv, yv, yaw = FORWARD_CMD
    base_kw = dict(cmd_x=xv, cmd_y=yv, cmd_yaw=yaw, arm_intensity=arm_intensity)
    sweeps = [
        (
            "Gait-frequency",
            [
                (
                    f"gait_freq={freq:.1f}Hz",
                    _forward_cmd_fn(lambda e, _f=freq: set_gait_cmd(e, gait_freq=_f)),
                    dict(base_kw, cmd_gait_freq=freq),
                )
                for freq in GAIT_FREQ_CMDS
            ],
            lambda rs: _fmt_metric(
                rs,
                [
                    ("contact_f", "gait_contact_force_cost", 1.0, ".4f"),
                    ("contact_v", "gait_contact_vel_cost", 1.0, ".4f"),
                    ("fall", "fall_rate", 100.0, ".1f%"),
                ],
            ),
        ),
        (
            "Footswing-height",
            [
                (
                    f"swing_h={height:.2f}m",
                    _forward_cmd_fn(lambda e, _h=height: set_gait_cmd(e, footswing_height=_h)),
                    dict(base_kw, cmd_footswing_height=height),
                )
                for height in FOOTSWING_HEIGHT_CMDS
            ],
            lambda rs: _fmt_metric(
                rs, [("clearance", "foot_clearance_rmse_m", 1.0, ".4f"), ("fall", "fall_rate", 100.0, ".1f%")]
            ),
        ),
        (
            "Stance-width",
            [
                (
                    f"stance_w={width:.2f}m",
                    _forward_cmd_fn(lambda e, _w=width: set_gait_cmd(e, stance_width=_w)),
                    dict(base_kw, cmd_stance_width=width),
                )
                for width in STANCE_WIDTH_CMDS
            ],
            lambda rs: _fmt_metric(
                rs, [("raibert", "raibert_rmse_m", 1.0, ".4f"), ("fall", "fall_rate", 100.0, ".1f%")]
            ),
        ),
    ]
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
    if not args.profile:
        return

    profile = _load_json_config(args.profile)
    simple_fields = [
        "headless",
        "sim_device",
        "robot",
        "num_envs_per_policy",
        "num_eval_steps",
        "arm_intensity",
        "output_dir",
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


def _discover_candidate_logdirs(candidate_dir: str, mode: str) -> List[Path]:
    root = Path(candidate_dir)
    if not root.exists():
        raise ValueError(f"{candidate_dir}: candidate directory does not exist")
    if not root.is_dir():
        raise ValueError(f"{candidate_dir}: candidate path is not a directory")

    logdirs = []
    for params_path in sorted(root.rglob("parameters.pkl")):
        logdir = params_path.parent
        has_dog = (logdir / "checkpoints_dog").is_dir()
        has_arm = (logdir / "checkpoints_arm").is_dir()
        if mode == "dog_only" and has_dog:
            logdirs.append(logdir)
        elif mode == "arm_only" and has_arm:
            logdirs.append(logdir)
        elif mode == "hybrid" and has_dog and has_arm:
            logdirs.append(logdir)

    if not logdirs:
        raise ValueError(f"{candidate_dir}: no {mode} candidates found")
    return logdirs


def _benchmark_mode(args) -> str:
    if not args.dog_only and not args.arm_only and not args.hybrid:
        args.dog_only = True
    selected = [name for name in ("dog_only", "arm_only", "hybrid") if getattr(args, name)]
    if len(selected) != 1:
        raise ValueError("Select exactly one benchmark mode: --dog_only, --arm_only, or --hybrid")
    return selected[0]


def _apply_candidate_dir(args):
    mode = _benchmark_mode(args)
    if mode != "dog_only":
        raise NotImplementedError(f"{mode} benchmark mode is reserved but not implemented yet")

    if not args.candidate_dir:
        if not args.logdirs:
            raise ValueError("Provide either --logdirs or --candidate_dir")
        return

    logdirs = _discover_candidate_logdirs(args.candidate_dir, mode)
    args.logdirs = [str(path) for path in logdirs]
    args.names = [path.name[:24] for path in logdirs]


def parse_args():
    p = argparse.ArgumentParser(description="Dog-only policy benchmark (GPU-parallel)")
    p.add_argument("--logdirs", nargs="+", default=None)
    p.add_argument("--names", nargs="*", default=None, help="Display name per logdir (default: directory name)")
    p.add_argument("--ckptids", nargs="*", default=None, help="Checkpoint id per logdir (default: 'last' for all)")
    p.add_argument("--candidate_dir", type=str, default=None, help="Run-like candidate root directory")
    p.add_argument("--profile", type=str, default=None, help="JSON benchmark profile path")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dog_only", action="store_true", default=False, help="Evaluate dog policies only")
    mode.add_argument("--arm_only", action="store_true", help="[Reserved] Evaluate arm policies only")
    mode.add_argument("--hybrid", action="store_true", help="[Reserved] Evaluate dog+arm policy pairs")
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
    p.add_argument("--arm_intensity", type=float, default=1.0, help="Arm disturbance for scenarios A/C/D")
    p.add_argument("--output_dir", type=str, default="benchmark/results")
    p.add_argument("--skip_a", action="store_true")
    p.add_argument("--skip_b", action="store_true")
    p.add_argument("--skip_c", action="store_true")
    p.add_argument("--skip_d", action="store_true")
    p.add_argument("--stage2", action="store_true", help="[Reserved] Stage-2 hybrid evaluation (not yet implemented)")
    args = p.parse_args()
    _apply_profile(args)
    _apply_candidate_dir(args)
    return args


def main():
    args = parse_args()

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

    num_envs_per_policy = args.num_envs_per_policy
    total_envs = num_envs_per_policy * n_runs

    print(f"[Benchmark] {n_runs} policies × {num_envs_per_policy} envs = {total_envs} total envs")
    validate_shared_env_compatibility(args.logdirs[0], args.logdirs[1:])
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

    print(
        f"[Benchmark] Layout: {layout.n_dims} cmd dims  "
        f"pose={'on' if layout.has_body_pitch else 'off'}  "
        f"gait_metrics={'on' if layout.has_dynamic_gait else 'off'}  "
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
            "vel_grid", run_scenario_a(env, handles, layout, args.num_eval_steps, args.arm_intensity, args.sim_device)
        )

    if not args.skip_b:
        _merge("arm_sweep", run_scenario_b(env, handles, layout, args.num_eval_steps, args.sim_device))

    if not args.skip_c:
        _merge(
            "body_pose", run_scenario_c(env, handles, layout, args.num_eval_steps, args.arm_intensity, args.sim_device)
        )

    if not args.skip_d:
        _merge("gait", run_scenario_d(env, handles, layout, args.num_eval_steps, args.arm_intensity, args.sim_device))

    print_comparison_table(all_results)

    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    json_path = os.path.join(run_dir, "results.json")
    markdown_path = os.path.join(run_dir, "report.md")
    plots_dir = os.path.join(run_dir, "plots")
    metadata = {
        "candidate_dir": args.candidate_dir or "cli",
        "benchmark_mode": _benchmark_mode(args),
        "profile": args.profile or "cli",
        "runs": ", ".join(names),
        "num_envs_per_policy": args.num_envs_per_policy,
        "total_envs": total_envs,
        "num_eval_steps": args.num_eval_steps,
        "control_dt_s": getattr(env.env, "dt", "unknown"),
        "headless": args.headless,
        "dog_num_commands": getattr(cfg.dog, "dog_num_commands", "unknown"),
        "use_dynamic_gait": getattr(cfg.commands, "use_dynamic_gait", "unknown"),
        "scenario_d_enabled": layout.has_dynamic_gait and not args.skip_d,
    }
    save_results(all_results, json_path)
    save_visualizations(all_results, plots_dir)
    save_markdown_report(all_results, markdown_path, metadata=metadata, plot_dir=plots_dir)
    print(f"[Benchmark] Output directory → {run_dir}")


if __name__ == "__main__":
    main()
