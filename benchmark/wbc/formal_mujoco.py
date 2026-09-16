"""Launch the frozen cross-method benchmark on one deterministic rough MuJoCo plant."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading

import numpy as np

from benchmark.wbc.cross_method_cli import DEFAULT_SCENE, METHODS, preflight


ENVIRONMENT_ID = "formal-mujoco-our-sota"
DEFAULT_LIBRARY = Path(__file__).resolve().parents[1] / "data/frozen_trajectory_library2/suite/trajectory_suite.json"
DEFAULT_WBC_RL_MPC_ROOT = Path(os.environ.get(
    "WBC_RL_MPC_ROOT", "/home/simon/Projects/Simon/wbc_rl_mpc",
))
DEFAULT_ROBODUET_OCS2_TASK = (
    DEFAULT_WBC_RL_MPC_ROOT
    / "go2_x5_ocs2/config/robot_lab_rear_r30o_s42_11497/task_sota.info"
)
FROZEN_ROBODUET_IDENTITY = {
    "policy_key": "robot_lab_rear_r30o_s42_11497",
    "policy_sha256": "dfc2bd1bb34e8490730d8d84f2e3bfd90eed29045c2b519726372f92100d2ba7",
    "config_sha256": "9edf2219aab2ad2ff265d8da6e6c6856070ace7762218257ccebd50f89cc0563",
    "ocs2_task_sha256": "8c36bf47fe59353f0984d27fcc46477ed9e3f70850f01963a6a030b201c3b938",
    "ocs2_transport": "synchronous",
    "ocs2_command_mode": "full",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest(), len(files)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def mild_rough_heightfield(seed: int, resolution: int, peak_height_m: float) -> np.ndarray:
    """Return deterministic rolling terrain with zero height at the spawn centre."""
    if resolution < 9 or resolution % 2 == 0:
        raise ValueError("terrain resolution must be an odd integer >= 9")
    if not 0.0 < peak_height_m <= 0.10:
        raise ValueError("terrain peak height must be in (0, 0.10] m")
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(resolution, resolution))
    for _ in range(5):
        padded = np.pad(noise, 1, mode="reflect")
        noise = sum(
            padded[i:i + resolution, j:j + resolution]
            for i in range(3) for j in range(3)
        ) / 9.0
    axis = np.linspace(-1.0, 1.0, resolution)
    xx, yy = np.meshgrid(axis, axis)
    phase = rng.uniform(-np.pi, np.pi, size=3)
    field = (
        0.55 * np.sin(2.0 * np.pi * (1.8 * xx + 0.35 * yy) + phase[0])
        + 0.35 * np.sin(2.0 * np.pi * (-0.45 * xx + 1.45 * yy) + phase[1])
        + 0.20 * np.sin(2.0 * np.pi * (2.7 * xx - 1.1 * yy) + phase[2])
        + 0.20 * noise / max(float(np.std(noise)), 1e-12)
    )
    centre = resolution // 2
    field -= field[centre, centre]
    positive = float(field.max())
    negative = abs(float(field.min()))
    field = np.where(
        field >= 0.0,
        field / max(positive, 1e-12) * peak_height_m,
        field / max(negative, 1e-12) * peak_height_m,
    )
    field[centre, centre] = 0.0
    return field.astype(np.float64)


def materialize_environment(
    output: Path, *, seed: int = 20260915, resolution: int = 129,
    half_extent_m: float = 8.0, peak_height_m: float = 0.10,
    slide_friction: float = 0.8,
) -> tuple[Path, dict]:
    """Create the single scene consumed by every benchmark method."""
    if half_extent_m < 6.0:
        raise ValueError("terrain half extent must cover the 5 m trajectory library")
    if not 0.1 <= slide_friction <= 2.0:
        raise ValueError("slide friction must be in [0.1, 2.0]")
    source_scene = DEFAULT_SCENE.resolve()
    source_robot = source_scene.with_name("go2_x5.xml")
    source_assets = source_scene.parent / "assets"
    if not source_robot.is_file() or not source_assets.is_dir():
        raise FileNotFoundError("common Go2-X5 MuJoCo resources are incomplete")
    output.mkdir(parents=True, exist_ok=True)

    heights = mild_rough_heightfield(seed, resolution, peak_height_m)
    height_path = output / "heightfield_m.npy"
    np.save(height_path, heights, allow_pickle=False)
    minimum, maximum = float(heights.min()), float(heights.max())
    span = maximum - minimum
    normalized = (heights - minimum) / span

    robot_text = source_robot.read_text()
    absolute_assets = str(source_assets.resolve())
    robot_text, replacements = re.subn(
        r'meshdir="[^"]+"', f'meshdir="{absolute_assets}"', robot_text, count=1,
    )
    if replacements != 1:
        raise ValueError("could not pin the common robot mesh directory")
    robot_out = output / "go2_x5.xml"
    robot_out.write_text(robot_text)

    elevation = "\n      ".join(
        " ".join(f"{value:.9f}" for value in row) for row in normalized
    )
    scene_out = output / "scene.xml"
    scene_out.write_text(f'''<mujoco model="go2_x5 formal paper rolling scene">
  <include file="go2_x5.xml"/>
  <option timestep="0.0025" integrator="implicitfast" gravity="0 0 -9.81"/>
  <statistic center="0 0 0.1" extent="8"/>
  <visual>
    <quality shadowsize="4096"/>
    <headlight diffuse="0.38 0.38 0.38" ambient="0.22 0.22 0.22" specular="0 0 0"/>
    <rgba haze="0.44 0.52 0.57 1"/>
    <global azimuth="-135" elevation="-16"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.56 0.66 0.73" rgb2="0.12 0.16 0.20" width="512" height="3072"/>
    <material name="groundplane" rgba="0.45 0.48 0.43 1" specular="0.025" shininess="0.06" reflectance="0"/>
    <hfield name="formal_visible_rolling" nrow="{resolution}" ncol="{resolution}"
      size="{half_extent_m:.9f} {half_extent_m:.9f} {span:.9f} 0.05"
      elevation="{elevation}"/>
  </asset>
  <worldbody>
    <light name="terrain_key" pos="-4 -6 5" dir="0.45 0.65 -0.65" directional="true"
      diffuse="0.86 0.81 0.73" specular="0.04 0.04 0.04" castshadow="true"/>
    <light name="terrain_fill" pos="4 3 6" dir="-0.4 -0.3 -1" directional="true"
      diffuse="0.30 0.36 0.42" specular="0 0 0" castshadow="false"/>
    <geom name="floor" type="hfield" hfield="formal_visible_rolling" pos="0 0 {minimum:.9f}"
      material="groundplane" friction="{slide_friction:.9f} 0.02 0.01" condim="6"/>
  </worldbody>
</mujoco>
''')
    assets_hash, asset_count = _tree_sha256(source_assets)
    manifest = {
        "schema_version": "formal-mujoco-environment-v1",
        "environment_id": ENVIRONMENT_ID,
        "scene": {"path": str(scene_out), "sha256": _sha256(scene_out)},
        "source_scene": {"path": str(source_scene), "sha256": _sha256(source_scene)},
        "robot_xml": {"source_sha256": _sha256(source_robot), "materialized_sha256": _sha256(robot_out)},
        "mesh_assets": {"root": str(source_assets.resolve()), "tree_sha256": assets_hash, "file_count": asset_count},
        "physics": {"timestep_s": 0.0025, "integrator": "implicitfast", "gravity_mps2": [0.0, 0.0, -9.81]},
        "terrain": {
            "kind": "mujoco_rolling_heightfield", "seed": seed,
            "resolution": [resolution, resolution],
            "half_extent_xy_m": [half_extent_m, half_extent_m], "peak_bound_m": peak_height_m,
            "actual_min_m": minimum, "actual_max_m": maximum, "rms_m": float(np.sqrt(np.mean(heights ** 2))),
            "spawn_height_m": float(heights[resolution // 2, resolution // 2]),
            "heightfield_sha256": _sha256(height_path), "slide_friction": slide_friction,
            "torsional_friction": 0.02, "rolling_friction": 0.01, "contact_dimensions": 6,
            "appearance": {
                "surface": "matte_sage_stone", "rgba": [0.45, 0.48, 0.43, 1.0],
                "texture": "none", "reflectance": 0.0,
                "lighting": "soft_neutral_low_angle_key_with_cool_fill",
            },
        },
        "disturbance": {
            "scenarios": ["nominal", "push"], "body": "base", "application_point": "body_center_of_mass",
            "frame": "environment_world", "force_n": [80.0, 0.0, 0.0], "start_time_s": 0.1,
            "duration_s": 0.1, "target_impulse_ns": [8.0, 0.0, 0.0],
        },
        "common_scene_required": True,
        "method_specific_dynamics_allowed_for": ["umi"],
        "method_specific_dynamics_exceptions": {
            "umi": {
                "profile": "training_nominal",
                "leg_damping_nms_per_rad": 0.1,
                "leg_frictionloss_nm": 0.025,
                "foot_slide_friction": 1.0,
                "scope": "in_memory_umi_adapter_only",
            },
        },
    }
    _write_json(output / "environment.json", manifest)
    return scene_out, manifest


def _load_tasks(suite_path: Path) -> tuple[dict, list[dict]]:
    suite = json.loads(suite_path.read_text())
    if isinstance(suite, list):
        if len(suite) != 1:
            raise ValueError("formal entry expects one frozen nominal suite")
        suite = suite[0].get("suite", suite[0])
    tasks = suite["trajectories"]
    if not tasks:
        raise ValueError("suite contains no tasks")
    return suite, tasks


def _roboduet_identity(readiness: dict, task_file: Path) -> dict:
    required = [Path(path) for path in readiness["methods"]["roboduet"]["required"]]
    policy = next(path for path in required if path.name == "policy.pt")
    config = next(path for path in required if path.name == "config.yaml")
    identity = {
        **FROZEN_ROBODUET_IDENTITY,
        "policy": {"path": str(policy.resolve()), "sha256": _sha256(policy)},
        "config": {"path": str(config.resolve()), "sha256": _sha256(config)},
        "ocs2_task": {"path": str(task_file.resolve()), "sha256": _sha256(task_file)},
    }
    actual = {
        "policy_sha256": identity["policy"]["sha256"],
        "config_sha256": identity["config"]["sha256"],
        "ocs2_task_sha256": identity["ocs2_task"]["sha256"],
    }
    mismatches = {
        key: {"expected": FROZEN_ROBODUET_IDENTITY[key], "actual": value}
        for key, value in actual.items() if value != FROZEN_ROBODUET_IDENTITY[key]
    }
    if mismatches:
        raise ValueError(f"frozen RoboDuet identity mismatch: {mismatches}")
    return identity


def select_shard_tasks(tasks: list[dict], shard_count: int, shard_index: int) -> list[dict]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("shard requires count >= 1 and 0 <= index < count")
    return [task for index, task in enumerate(tasks) if index % shard_count == shard_index]


def _validate_roboduet_receipts(records: dict, expected_sha256: str) -> str | None:
    for scenario, record in records.items():
        receipt = json.loads(Path(record["receipt"]).read_text())
        ocs2 = receipt.get("ocs2", {})
        source = ocs2.get("source_task_sha256")
        runtime = ocs2.get("runtime_task_sha256")
        if source != expected_sha256 or runtime != expected_sha256 or source != runtime:
            return f"{scenario} OCS2 task receipt mismatch: source={source}, runtime={runtime}"
    return None


def _summary(jobs: list[dict]) -> dict:
    counts = {name: 0 for name in ("pending", "running", "complete", "failed", "interrupted")}
    for job in jobs:
        counts[job["status"]] = counts.get(job["status"], 0) + 1
    return counts


def _campaign_results(state: dict) -> dict:
    rows = []
    for job in state["jobs"]:
        records = job.get("scenario_results", {})
        for scenario in state["scenarios"]:
            record = records.get(scenario, {})
            rows.append({
                "method": job["method"], "task_id": job["task_id"],
                "source_trajectory_id": job.get("source_trajectory_id"), "scenario": scenario,
                "job_status": job["status"], "exit_code": job.get("exit_code"),
                "receipt": record.get("receipt"), "receipt_status": record.get("status"),
                "result": record.get("result"),
            })
    return {
        "schema_version": "formal-mujoco-results-v1", "campaign_status": state["status"],
        "environment_id": state["environment"]["environment_id"],
        "environment_scene_sha256": state["environment"]["scene"]["sha256"],
        "suite_sha256": state["suite_sha256"], "rows": rows,
    }


def _collect_scenario_results(job_dir: Path, scenarios) -> dict:
    records = {}
    for scenario in scenarios:
        receipts = sorted((job_dir / "backend").glob(f"*/{scenario}/receipt.json"))
        if not receipts:
            continue
        receipt_path = max(receipts, key=lambda path: path.stat().st_mtime_ns)
        receipt = json.loads(receipt_path.read_text())
        records[scenario] = {
            "receipt": str(receipt_path), "status": receipt.get("status"),
            "result": receipt.get("result", receipt.get("metrics")),
        }
    return records


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--scenarios", nargs="+", choices=("nominal", "push"), default=("nominal", "push"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ros-domain-base", type=int, default=120)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--output-root", type=Path, default=Path("benchmark/results/formal_mujoco_paper_rolling_v3"))
    parser.add_argument("--campaign-root", type=Path,
                        help="Exact shared campaign root; environment is materialized once here")
    parser.add_argument("--prepare-campaign", action="store_true",
                        help="Create only the immutable shared manifest/environment")
    parser.add_argument("--node-name", default=socket.gethostname())
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-tasks", type=int, default=0, help="Bounded smoke; 0 selects all frozen tasks")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--run-python", type=Path, default=Path(sys.executable),
                        help="Python used by method runners")
    parser.add_argument("--video-python", type=Path, default=Path(sys.executable),
                        help="Python used by the EGL video renderer")
    parser.add_argument("--roboduet-ocs2-task-file", type=Path, default=DEFAULT_ROBODUET_OCS2_TASK)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument("--terrain-seed", type=int, default=20260915)
    parser.add_argument("--terrain-resolution", type=int, default=129)
    parser.add_argument("--terrain-half-extent", type=float, default=8.0)
    parser.add_argument("--terrain-peak-height", type=float, default=0.10)
    parser.add_argument("--ground-friction", type=float, default=0.8)
    args = parser.parse_args(argv)
    if args.workers < 1 or args.workers > 16:
        parser.error("--workers must be in 1..16")
    if not 0 <= args.ros_domain_base <= 232 or args.ros_domain_base + args.workers - 1 > 232:
        parser.error("ROS domain range exceeds 0..232")
    if args.timeout_s <= 0 or args.max_tasks < 0:
        parser.error("timeouts must be positive and task limit nonnegative")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("shard requires count >= 1 and 0 <= index < count")
    if "/" in args.node_name or args.node_name in ("", ".", ".."):
        parser.error("--node-name must be one safe path component")
    for label, path in (("run Python", args.run_python), ("video Python", args.video_python),
                        ("RoboDuet OCS2 task", args.roboduet_ocs2_task_file)):
        if not path.resolve().is_file():
            parser.error(f"{label} does not exist: {path}")

    suite_path = args.suite.resolve()
    suite, all_tasks = _load_tasks(suite_path)
    tasks = select_shard_tasks(all_tasks, args.shard_count, args.shard_index)
    if args.max_tasks:
        tasks = tasks[:args.max_tasks]
    readiness = preflight()
    blocked = [name for name in args.methods if readiness["methods"][name]["status"] != "ready"]
    if blocked:
        raise RuntimeError(f"formal benchmark preflight blocked: {blocked}")
    roboduet_identity = _roboduet_identity(readiness, args.roboduet_ocs2_task_file.resolve())

    shared_contract = {
        "schema_version": "formal-mujoco-shared-campaign-v1",
        "suite": str(suite_path), "suite_file_sha256": _sha256(suite_path),
        "suite_sha256": suite["suite_sha256"], "task_count": len(all_tasks),
        "methods": list(args.methods), "scenarios": list(args.scenarios),
        "roboduet": roboduet_identity,
    }

    if args.prepare_campaign:
        if not args.campaign_root or args.resume:
            parser.error("--prepare-campaign requires --campaign-root and cannot use --resume")
        campaign_root = args.campaign_root.resolve()
        campaign_root.mkdir(parents=True, exist_ok=False)
        _, environment = materialize_environment(
            campaign_root / "environment", seed=args.terrain_seed,
            resolution=args.terrain_resolution, half_extent_m=args.terrain_half_extent,
            peak_height_m=args.terrain_peak_height, slide_friction=args.ground_friction,
        )
        shared_contract["environment"] = environment
        shared_contract["created_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        _write_json(campaign_root / "campaign_manifest.json", shared_contract)
        print(f"Prepared immutable Formal MuJoCo campaign: {campaign_root}", flush=True)
        return 0

    if args.resume:
        run_dir = args.resume.resolve()
        state_path = run_dir / "formal_state.json"
        state = json.loads(state_path.read_text())
        if (state["suite_sha256"] != suite["suite_sha256"] or
                state["methods"] != list(args.methods) or
                state.get("roboduet") != roboduet_identity or
                state.get("shard") != {"count": args.shard_count, "index": args.shard_index}):
            raise ValueError("resume suite, methods, RoboDuet identity, or shard does not match")
        scene = Path(state["environment"]["scene"]["path"])
        if _sha256(scene) != state["environment"]["scene"]["sha256"]:
            raise ValueError("resume scene hash does not match campaign state")
        jobs = state["jobs"]
        for job in jobs:
            if job["status"] in ("running", "interrupted") or (args.rerun_failed and job["status"] == "failed"):
                job["status"] = "pending"
    else:
        if args.campaign_root:
            campaign_root = args.campaign_root.resolve()
            manifest = json.loads((campaign_root / "campaign_manifest.json").read_text())
            for key in ("suite_file_sha256", "suite_sha256", "task_count", "methods", "scenarios", "roboduet"):
                if manifest.get(key) != shared_contract[key]:
                    raise ValueError(f"shared campaign manifest mismatch for {key}")
            environment = manifest["environment"]
            scene = Path(environment["scene"]["path"])
            if _sha256(scene) != environment["scene"]["sha256"]:
                raise ValueError("shared scene hash does not match campaign manifest")
            run_dir = campaign_root / "nodes" / args.node_name
            run_dir.mkdir(parents=True, exist_ok=False)
        else:
            stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = (args.output_root.resolve() / stamp)
            run_dir.mkdir(parents=True, exist_ok=False)
            scene, environment = materialize_environment(
                run_dir / "environment", seed=args.terrain_seed, resolution=args.terrain_resolution,
                half_extent_m=args.terrain_half_extent, peak_height_m=args.terrain_peak_height,
                slide_friction=args.ground_friction,
            )
        jobs = []
        for task in tasks:
            for method in args.methods:
                jobs.append({
                    "index": len(jobs), "method": method, "task_id": task["task_id"],
                    "source_trajectory_id": task.get("source_trajectory_id"), "status": "pending",
                })
        state = {
            "schema_version": "formal-mujoco-campaign-v1", "status": "prepared",
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(), "run_dir": str(run_dir),
            "suite": str(suite_path), "suite_file_sha256": _sha256(suite_path),
            "suite_sha256": suite["suite_sha256"], "methods": list(args.methods),
            "scenarios": list(args.scenarios), "workers": args.workers,
            "ros_domain_base": args.ros_domain_base, "environment": environment,
            "node_name": args.node_name,
            "shard": {"count": args.shard_count, "index": args.shard_index},
            "roboduet": roboduet_identity,
            "run_python": str(args.run_python.resolve()),
            "video_python": str(args.video_python.resolve()),
            "preflight": readiness, "jobs": jobs,
        }
    state_path = run_dir / "formal_state.json"
    state["summary"] = _summary(jobs)
    _write_json(state_path, state)
    _write_json(run_dir / "formal_results.json", _campaign_results(state))
    print(f"Formal MuJoCo campaign: {run_dir}", flush=True)
    print(f"Jobs: {len(jobs)} method-task pairs; scenarios={list(args.scenarios)}; workers={args.workers}", flush=True)
    if args.dry_run:
        return 0

    pending = queue.Queue()
    for job in jobs:
        if job["status"] == "pending":
            pending.put(job)
    lock = threading.Lock()
    stop = threading.Event()
    active: dict[int, subprocess.Popen] = {}

    def persist():
        state["summary"] = _summary(jobs)
        state["status"] = "running" if state["summary"]["running"] or state["summary"]["pending"] else (
            "failed" if state["summary"]["failed"] else "complete")
        state["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        _write_json(state_path, state)
        _write_json(run_dir / "formal_results.json", _campaign_results(state))

    def worker(slot: int):
        domain = args.ros_domain_base + slot
        while not stop.is_set():
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            job_dir = run_dir / "jobs" / job["method"] / job["task_id"]
            job_dir.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable, "-m", "benchmark.wbc.cross_method_cli",
                "--method", job["method"], "--suite", str(suite_path), "--task-id", job["task_id"],
                "--scenarios", *args.scenarios, "--scene", str(scene),
                "--output", str(job_dir / "backend"), "--ros-domain-id", str(domain),
                "--timeout-s", str(args.timeout_s), "--ocs2-transport", "synchronous",
                "--qm-mpc-coupling-mode", "sync", "--umi-mujoco-profile", "training_nominal",
                "--python", str(args.run_python.resolve()),
                "--video-python", str(args.video_python.resolve()),
            ]
            if job["method"] == "roboduet":
                command.extend(["--roboduet-ocs2-task-file",
                                str(args.roboduet_ocs2_task_file.resolve())])
            if args.record_video:
                command.append("--record-video")
            with lock:
                job.update(status="running", worker_slot=slot, ros_domain_id=domain,
                           started_at=dt.datetime.now(dt.timezone.utc).isoformat(), command=command)
                persist()
            environment = os.environ.copy()
            environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
            with (job_dir / "run.log").open("w") as log:
                process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[2], env=environment,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                with lock:
                    active[slot] = process
                code = process.wait()
                with lock:
                    active.pop(slot, None)
            scenario_results = _collect_scenario_results(job_dir, args.scenarios)
            receipt_error = None
            if job["method"] == "roboduet" and code == 0:
                receipt_error = _validate_roboduet_receipts(
                    scenario_results, roboduet_identity["ocs2_task_sha256"],
                )
            with lock:
                job.update(status="complete" if code == 0 and not receipt_error else "failed", exit_code=code,
                           finished_at=dt.datetime.now(dt.timezone.utc).isoformat(), log=str(job_dir / "run.log"),
                           scenario_results=scenario_results)
                if receipt_error:
                    job["receipt_validation_error"] = receipt_error
                _write_json(job_dir / "job.json", job)
                persist()
                print(f"[{state['summary']['complete'] + state['summary']['failed']}/{len(jobs)}] "
                      f"{job['method']} {job['source_trajectory_id']} -> {job['status']}", flush=True)
            pending.task_done()

    def terminate(_signum, _frame):
        stop.set()
        with lock:
            for process in active.values():
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)

    signal.signal(signal.SIGINT, terminate)
    signal.signal(signal.SIGTERM, terminate)
    threads = [threading.Thread(target=worker, args=(slot,), daemon=False) for slot in range(args.workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    with lock:
        if stop.is_set():
            for job in jobs:
                if job["status"] == "running":
                    job["status"] = "interrupted"
        persist()
    return 1 if state["summary"]["failed"] or state["summary"]["interrupted"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
