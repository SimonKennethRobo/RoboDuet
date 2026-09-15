"""Controller x locomotion-policy IsaacGym benchmark matrix."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shlex
import sys
from pathlib import Path

import isaacgym  # noqa: F401
import torch

from benchmark.metadata import git_snapshot, runtime_snapshot
from benchmark.wbc.cli import _dump_json, _sha256_file, set_benchmark_seed
from benchmark.wbc.controllers import (
    FloatingBaseOcs2MpcController,
    IkUpperController,
    NativeOcs2Transport,
    UPPER_CONTROLLER_CONTRACT_VERSION,
)
from benchmark.wbc.evaluation import (
    WBCPolicyHandle,
    _load_cfg_from_pkl,
    load_wbc_env_benchmark,
)
from benchmark.wbc.scenarios import run_wbc_aggregate
from benchmark.wbc.scoring import DEVELOPMENT_TIMED_TRAJECTORY_PROTOCOL, PROTOCOL_VERSION
from benchmark.wbc.suite import SUITE_VERSION, write_reference_archive
from benchmark.wbc.workspace import (
    WORKSPACE_SCOPES,
    build_workspace_grid,
    run_isaacgym_workspace_probe,
)
from go1_gym.envs.config.core import cfg_to_dict
from scripts.load_policy import load_dog_policy


SYSTEM_ARM_HOME_RAD = (0.0, 0.9, 0.9, 0.0, 0.0, 0.0)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Complete IsaacGym controller x locomotion matrix")
    parser.add_argument("--env-logdir", required=True)
    parser.add_argument("--loco-logdirs", nargs="+", required=True)
    parser.add_argument("--loco-names", nargs="*", default=None)
    parser.add_argument("--loco-ckptids", nargs="*", default=None)
    parser.add_argument(
        "--upper-controllers",
        nargs="+",
        choices=("floating_base_ocs2_mpc", "ik"),
        default=("floating_base_ocs2_mpc", "ik"),
    )
    parser.add_argument(
        "--scenarios", nargs="+", choices=("nominal", "push"), default=("nominal", "push")
    )
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument(
        "--visualize-trajectories",
        action="store_true",
        help="Draw the reference path (orange) and executed EE path (green) in IsaacGym.",
    )
    parser.add_argument("--sim_device", default="cuda:0")
    parser.add_argument("--robot", default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--bank_seed", type=int, default=12345)
    parser.add_argument("--bank_per_cell", type=int, default=1)
    parser.add_argument("--cells", nargs="+", default=("0,0",))
    parser.add_argument("--suite_rows_per_cell", type=int, default=1)
    parser.add_argument("--num_eval_steps", type=int, default=20)
    parser.add_argument("--settle_steps", type=int, default=10)
    parser.add_argument("--record_raw_traces", action="store_true", default=True)
    parser.add_argument("--output_dir", default="benchmark/results/system_matrix")
    parser.add_argument(
        "--ocs2-root", default="/home/simon/Projects/Simon/wbc_rl_mpc"
    )
    parser.add_argument("--ocs2-timeout-s", type=float, default=90.0)
    parser.add_argument(
        "--ocs2-transport",
        choices=("synchronous", "async"),
        default="synchronous",
        help=(
            "synchronous gives deterministic benchmark steps; async runs MPC in an "
            "independent latest-state/latest-command process paced at real time"
        ),
    )
    parser.add_argument(
        "--ocs2-command-timeout-s",
        type=float,
        default=0.5,
        help="Maximum wall-clock age of a cached MPC command in async mode.",
    )
    parser.add_argument(
        "--ocs2-command-mode",
        choices=("full", "pose_only"),
        default="full",
        help=(
            "full enables planar velocity; pose_only suppresses vx/vy while retaining "
            "yaw rate, body posture and arm targets"
        ),
    )
    parser.add_argument("--push-start-s", type=float, default=0.10)
    parser.add_argument("--push-duration-s", type=float, default=0.10)
    parser.add_argument("--push-force-n", nargs=3, type=float, default=(80.0, 0.0, 0.0))
    parser.add_argument("--workspace", action="store_true")
    parser.add_argument(
        "--workspace-scopes", nargs="+", choices=WORKSPACE_SCOPES, default=WORKSPACE_SCOPES
    )
    parser.add_argument(
        "--workspace-bounds-m", nargs=6, type=float,
        default=(0.30, 0.40, -0.05, 0.05, -0.05, 0.05),
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX", "Z_MIN", "Z_MAX"),
    )
    parser.add_argument("--workspace-spacing-m", type=float, default=0.10)
    parser.add_argument("--workspace-steps", type=int, default=100)
    parser.add_argument("--workspace-hold-steps", type=int, default=5)
    parser.add_argument("--validate_only", action="store_true")
    return parser.parse_args(argv)


def _normalize(values, size, default):
    result = list(values or [])
    if len(result) > size:
        raise ValueError("too many per-policy values")
    while len(result) < size:
        result.append(default(len(result)))
    return result


def _validate_dog_bundle(path, ckpt_id):
    root = Path(path)
    name = "last_dog" if ckpt_id == "last" else ckpt_id.zfill(6)
    required = [root / "parameters.pkl", root / "checkpoints_dog" / f"ac_weights_{name}.pt"]
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise ValueError("invalid locomotion policy bundle; missing " + ", ".join(missing))


def _parse_cells(values, cfg):
    result = []
    for value in values:
        a, b = (int(item) for item in value.split(",", 1))
        if not (0 <= a < cfg.wbc.goal_reaching.trajectory.n_levels_A):
            raise ValueError(f"cell A out of range: {a}")
        if not (0 <= b < cfg.wbc.goal_reaching.trajectory.n_levels_B):
            raise ValueError(f"cell B out of range: {b}")
        result.append((a, b))
    return list(dict.fromkeys(result))


def _controller(controller_id, base, output_dir, args):
    if controller_id == "ik":
        return IkUpperController()
    transport = NativeOcs2Transport(
        output_dir / "ocs2_runtime",
        Path(args.ocs2_root),
        timeout_s=args.ocs2_timeout_s,
        mode=args.ocs2_transport,
        command_timeout_s=args.ocs2_command_timeout_s,
        command_mode=args.ocs2_command_mode,
    )
    return FloatingBaseOcs2MpcController(transport, Path(args.ocs2_root))


def _disturbance(args, scenario):
    if scenario == "nominal":
        return []
    force = [float(value) for value in args.push_force_n]
    magnitude = sum(value * value for value in force) ** 0.5
    return [
        {
            "type": "constant_force",
            "start_time_s": float(args.push_start_s),
            "duration_s": float(args.push_duration_s),
            "body": "base",
            "frame": "environment_world",
            "force_n": force,
            "magnitude_n": magnitude,
            "application_point": "body_center_of_mass",
        }
    ]


def _policy_provenance(name, path, ckpt_id):
    root = Path(path).resolve()
    filename = "ac_weights_last_dog.pt" if ckpt_id == "last" else f"ac_weights_{ckpt_id.zfill(6)}.pt"
    return {
        "locomotion_policy_id": name,
        "logdir": str(root),
        "ckptid": ckpt_id,
        "parameters_sha256": _sha256_file(root / "parameters.pkl"),
        "dog_checkpoint_sha256": _sha256_file(root / "checkpoints_dog" / filename),
    }


def main(argv=None):
    args = parse_args(argv)
    n_loco = len(args.loco_logdirs)
    names = _normalize(args.loco_names, n_loco, lambda i: Path(args.loco_logdirs[i]).name[:24])
    ckptids = [
        "last" if value == "last" else value.zfill(6)
        for value in _normalize(args.loco_ckptids, n_loco, lambda _i: "last")
    ]
    if len(set(names)) != len(names):
        raise ValueError("locomotion policy names must be unique")
    if args.num_eval_steps <= 0 or args.suite_rows_per_cell <= 0:
        raise ValueError("evaluation steps and suite rows must be positive")
    if args.ocs2_timeout_s <= 0.0 or args.ocs2_command_timeout_s <= 0.0:
        raise ValueError("OCS2 timeouts must be positive")
    if args.visualize_trajectories and args.headless:
        raise ValueError(
            "--visualize-trajectories requires the IsaacGym viewer; remove --headless"
        )
    for path, ckpt_id in zip(args.loco_logdirs, ckptids):
        _validate_dog_bundle(path, ckpt_id)

    preview = _load_cfg_from_pkl(args.env_logdir, robot=args.robot)
    cells = _parse_cells(args.cells, preview)
    for path, ckpt_id, name in zip(args.loco_logdirs, ckptids, names):
        load_dog_policy(path, ckpt_id, preview, device="cpu")
        print(f"[system benchmark] validated locomotion policy {name}")
    if args.validate_only:
        print("[system benchmark] fixed controller contract selection validated")
        return

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    state_path = run_dir / "run_state.json"
    _dump_json(state_path, {"status": "running", "completed_matrix_cells": 0})
    results = {f"{controller}::{name}": {"wbc_aggregate": [], "wbc_trajectories": []}
               for controller in args.upper_controllers for name in names}
    suite_records = []
    workspace_records = []
    workspace_manifest = None
    if args.workspace:
        bounds = [args.workspace_bounds_m[i : i + 2] for i in range(0, 6, 2)]
        workspace_manifest = build_workspace_grid(bounds, args.workspace_spacing_m)
    controller_records = {}
    expected_identity = {}
    completed = 0
    try:
        for controller_id in args.upper_controllers:
            action_mode = "ik_residual" if controller_id == "ik" else "end_to_end"
            for loco_path, loco_name, ckpt_id in zip(args.loco_logdirs, names, ckptids):
                pair_name = f"{controller_id}::{loco_name}"
                set_benchmark_seed(args.seed, args.sim_device)
                env, cfg = load_wbc_env_benchmark(
                    args.env_logdir,
                    total_envs=1,
                    envs_per_policy=1,
                    headless=args.headless,
                    device=args.sim_device,
                    robot=args.robot,
                    bank_seed=args.bank_seed,
                    bank_per_cell=args.bank_per_cell,
                    arm_action_mode=action_mode,
                    initial_arm_joint_positions=SYSTEM_ARM_HOME_RAD,
                )
                base = env.env
                base.configure_benchmark_trajectory_viewer(args.visualize_trajectories)
                controller = None
                try:
                    dog_policy = load_dog_policy(loco_path, ckpt_id, cfg, device=args.sim_device)
                    cell_dir = run_dir / "runtime" / controller_id / loco_name
                    controller = _controller(controller_id, base, cell_dir, args)
                    controller_records.setdefault(controller_id, controller.provenance)
                    handle = WBCPolicyHandle(
                        pair_name,
                        dog_policy,
                        None,
                        0,
                        1,
                        upper_controller=controller,
                        controller_id=controller_id,
                        locomotion_policy_id=loco_name,
                    )
                    _dump_json(
                        run_dir / f"resolved_config_{controller_id}_{loco_name}.json",
                        cfg_to_dict(cfg),
                    )
                    for scenario in args.scenarios:
                        output = run_wbc_aggregate(
                            env,
                            [handle],
                            cells=cells,
                            n_steps=args.num_eval_steps,
                            settle_steps=args.settle_steps,
                            device=args.sim_device,
                            suite_rows_per_cell=args.suite_rows_per_cell,
                            validate_feature_coverage=False,
                            raw_trace_dir=run_dir / "raw_traces" if args.record_raw_traces else None,
                            trace_prefix=f"{controller_id}_{loco_name}_{scenario}",
                            scenario_id=scenario,
                            disturbance_schedule=_disturbance(args, scenario),
                        )
                        for kind, rows in output["results"][pair_name].items():
                            results[pair_name][kind].extend(rows)
                        reference_name = f"references_{controller_id}_{loco_name}_{scenario}.npz"
                        record = write_reference_archive(
                            run_dir / reference_name, base, output["suite_manifest"]
                        )
                        manifest = output["suite_manifest"]
                        identity = [
                            (
                                task["content_sha256"],
                                task["initial_state"],
                                task["anchor_env_local_xy_m"],
                                task["disturbance_schedule"],
                            )
                            for task in manifest["trajectories"]
                        ]
                        if scenario in expected_identity and identity != expected_identity[scenario]:
                            raise RuntimeError(f"frozen TaskSpec identity drifted in scenario {scenario}")
                        expected_identity.setdefault(scenario, identity)
                        suite_records.append(
                            {
                                "upper_controller": controller_id,
                                "locomotion_policy": loco_name,
                                "robustness_scenario": scenario,
                                "suite": manifest,
                                "reference_archive": record,
                            }
                        )
                        completed += 1
                        _dump_json(run_dir / "results.partial.json", results)
                        _dump_json(
                            state_path,
                            {"status": "running", "completed_matrix_cells": completed},
                        )
                    if workspace_manifest is not None:
                        workspace_records.append(
                            run_isaacgym_workspace_probe(
                                env,
                                handle,
                                workspace_manifest,
                                scopes=args.workspace_scopes,
                                n_steps=args.workspace_steps,
                                settle_steps=args.settle_steps,
                                hold_steps=args.workspace_hold_steps,
                            )
                        )
                        _dump_json(run_dir / "workspace_results.json", workspace_records)
                finally:
                    if controller is not None:
                        controller.close()
                    base.close()
                    del env
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        _dump_json(run_dir / "results.json", results)
        _dump_json(run_dir / "trajectory_suite.json", suite_records)
        if workspace_manifest is not None:
            _dump_json(run_dir / "workspace_manifest.json", workspace_manifest)
            _dump_json(run_dir / "workspace_results.json", workspace_records)
        metadata = {
            "mode": "wbc_system_matrix",
            "protocol": PROTOCOL_VERSION,
            "task_suite_version": SUITE_VERSION,
            "upper_controller_contract": UPPER_CONTROLLER_CONTRACT_VERSION,
            "evaluation_protocol": dict(DEVELOPMENT_TIMED_TRAJECTORY_PROTOCOL),
            "upper_controllers": list(args.upper_controllers),
            "locomotion_policies": [
                _policy_provenance(name, path, ckpt)
                for name, path, ckpt in zip(names, args.loco_logdirs, ckptids)
            ],
            "controller_provenance": controller_records,
            "scenarios": list(args.scenarios),
            "push_schedule": _disturbance(args, "push"),
            "workspace_probe": {
                "enabled": workspace_manifest is not None,
                "schema_version": workspace_manifest["schema_version"] if workspace_manifest else None,
                "workspace_sha256": workspace_manifest["workspace_sha256"] if workspace_manifest else None,
                "scopes": list(args.workspace_scopes) if workspace_manifest else [],
            },
            "trajectory_visualization": {
                "enabled": bool(args.visualize_trajectories),
                "reference_color_rgb": [1.0, 0.55, 0.0],
                "executed_color_rgb": [0.1, 1.0, 0.1],
                "viewer_env_index": 0,
            },
            "source": {"roboduet": git_snapshot()},
            "runtime": runtime_snapshot(),
            "command": shlex.join(sys.argv if argv is None else ["python", "-m", "benchmark", "--wbc", "--system_matrix", *argv]),
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        _dump_json(run_dir / "metadata.json", metadata)
        _dump_json(
            state_path,
            {
                "status": "complete",
                "completed_matrix_cells": completed,
                "completed_at": datetime.datetime.now().isoformat(timespec="seconds"),
            },
        )
        from benchmark.reports.html import write_report_bundle

        write_report_bundle(run_dir / "results.json")
        print(f"[system benchmark] output -> {run_dir}")
    except Exception as exc:
        _dump_json(
            state_path,
            {
                "status": "failed",
                "completed_matrix_cells": completed,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise


if __name__ == "__main__":
    main()
