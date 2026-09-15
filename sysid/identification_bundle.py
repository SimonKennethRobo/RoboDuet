"""Policy bundle checks shared by collection and the identification pipeline."""
from pathlib import Path
import re

import numpy as np
import yaml

from scripts.rl_sar_obs import RlSarObservation

CANONICAL_CHANNELS = ["vx", "vy", "wz", "height", "pitch", "roll"]


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
