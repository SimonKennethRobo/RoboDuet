"""Policy bundle checks and immutable experiment-local snapshots."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np
import yaml

from scripts.rl_sar_obs import RlSarObservation

CANONICAL_CHANNELS = ["vx", "vy", "wz", "height", "pitch", "roll"]
SNAPSHOT_SCHEMA = "rl-sar-policy-bundle-snapshot-v1"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def policy_argument_key(policy):
    """Return the deployment key encoded by a key/directory/policy.pt argument."""
    path = Path(policy).expanduser()
    if path.name == "policy.pt":
        return path.parent.name
    return path.name


def verify_bundle_snapshot(experiment_root, expected_key=None):
    """Verify and return an existing experiment-local policy bundle."""
    root = Path(experiment_root).resolve() / "policy_bundle"
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SNAPSHOT_SCHEMA:
        raise ValueError(f"unsupported policy bundle snapshot: {manifest.get('schema')!r}")
    key = manifest.get("policy")
    if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", key):
        raise ValueError("invalid policy key in bundle snapshot")
    if expected_key is not None and key != expected_key:
        raise ValueError(f"bundle snapshot belongs to {key}, requested {expected_key}")
    robot_dir = root / "go2_x5"
    expected_files = {
        "go2_x5/base.yaml", f"go2_x5/{key}/config.yaml", f"go2_x5/{key}/policy.pt"
    }
    if set(manifest.get("files", {})) != expected_files:
        raise ValueError("bundle snapshot manifest has an unexpected file set")
    for relative, digest in manifest["files"].items():
        path = root / relative
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"bundle snapshot changed: {path}")
    return robot_dir, key, manifest


def snapshot_policy_bundle(robot_dir, policy_key, experiment_root):
    """Atomically copy the exact RL-SAR deployment bundle into an experiment."""
    source = Path(robot_dir).expanduser().resolve()
    experiment_root = Path(experiment_root).expanduser().resolve()
    target = experiment_root / "policy_bundle"
    if target.exists():
        snap_robot, key, manifest = verify_bundle_snapshot(experiment_root, policy_key)
        if Path(manifest["source_robot_dir"]).resolve() != source:
            raise ValueError(
                f"bundle snapshot source differs: {manifest['source_robot_dir']} != {source}"
            )
        return snap_robot, key, manifest
    experiment_root.mkdir(parents=True, exist_ok=True)
    required = {
        "go2_x5/base.yaml": source / "base.yaml",
        f"go2_x5/{policy_key}/config.yaml": source / policy_key / "config.yaml",
        f"go2_x5/{policy_key}/policy.pt": source / policy_key / "policy.pt",
    }
    for path in required.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    staging = Path(tempfile.mkdtemp(prefix=".policy_bundle.", dir=experiment_root))
    try:
        files = {}
        for relative, path in required.items():
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            files[relative] = _sha256(destination)
        manifest = {
            "schema": SNAPSHOT_SCHEMA,
            "policy": policy_key,
            "source_robot_dir": str(source),
            "source_policy_dir": str(source / policy_key),
            "files": files,
        }
        _write_json(staging / "manifest.json", manifest)
        staging.replace(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return verify_bundle_snapshot(experiment_root, policy_key)


def command_contract(params):
    """Map supported observation layouts to canonical MPC command channels."""
    observations = params["observations"]
    if "robot_lab/velocity_pose_commands" in observations:
        return dict(command_observation="robot_lab/velocity_pose_commands",
                    policy_command_observation_dim=7,
                    command_channels=list(CANONICAL_CHANNELS))
    if "roboduet/dog_commands" in observations:
        width = len(params.get("dog_commands_scale", []))
        raw = ["vx", "vy", "wz", "pitch", "roll", "height"]
        if params.get("omit_height", False):
            raw.remove("height")
        # Anything after the physical channels is gait metadata, not another
        # actuator command. A short vector genuinely removes trailing channels.
        visible = raw[:min(width, len(raw))]
        channels = [name for name in CANONICAL_CHANNELS if name in visible]
        if not channels:
            raise ValueError("roboduet/dog_commands exposes no controllable base channels")
        return dict(command_observation="roboduet/dog_commands",
                    policy_command_observation_dim=width,
                    command_channels=channels)
    raise ValueError("Expected supported base commands in observations: robot_lab/velocity_pose_commands or roboduet/dog_commands")


def resolve_bundle(policy, robot_dir):
    """Accept a deployment key, a bundle directory, or its exported policy.pt."""
    path = Path(policy).expanduser()
    if path.is_file():
        if path.name != "policy.pt":
            raise ValueError("Use an exported policy.pt with config.yaml, not training weights")
        path = path.parent
    elif not path.is_dir():
        path = Path(robot_dir).expanduser() / policy
    path = path.resolve()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", path.name):
        raise ValueError("Policy key must contain only letters, digits, '_', '-' or '.'")
    for required in [path / "policy.pt", path / "config.yaml", path.parent / "base.yaml"]:
        if not required.is_file():
            raise FileNotFoundError(f"Missing RL-SAR deployment file: {required}")
    return path.parent, path.name


def load_bundle_contract(robot_dir, policy_key):
    robot_dir = Path(robot_dir).resolve()
    base = yaml.safe_load((robot_dir / "base.yaml").read_text())[robot_dir.name]
    config = yaml.safe_load((robot_dir / policy_key / "config.yaml").read_text())
    params = {**base, **config[f"{robot_dir.name}/{policy_key}"]}
    if (params.get("num_leg_dofs"), params.get("num_arm_dofs"), params.get("num_of_dofs")) != (12, 6, 18):
        raise ValueError("Identification currently supports Go2-X5: 12 leg + 6 arm joints")
    if not np.isclose(float(params["dt"]), .005) or int(params["decimation"]) != 4:
        raise ValueError("Identification currently requires 5 ms control / 20 ms policy timing")
    commands = command_contract(params)
    try:
        RlSarObservation(params)
    except KeyError as error:
        raise ValueError(f"Observation adapter does not support {error}; export a compatible bundle or extend the adapter") from error
    history = params["observations_history"]
    if not history or any(int(i) != i or i < 0 for i in history):
        raise ValueError("observations_history must contain nonnegative integer indices")
    frequency = float(params.get("gait_frequency", 0.))
    if not np.isfinite(frequency) or frequency < 0:
        raise ValueError("gait_frequency must be finite and nonnegative")
    return dict(policy=policy_key, robot_dir=str(robot_dir),
                policy_dt_s=.02, control_dt_s=.005,
                num_observations=int(params["num_observations"]),
                history_indices=list(history),
                gait_frequency_hz=frequency, supports_gait_phase=frequency > 0.,
                stop_gait_at_stand=bool(params.get("use_dynamic_gait", False)),
                **commands)
