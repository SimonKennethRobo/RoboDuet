"""Opt-in Stage-1 demo experiments; tuning values live in configs/coordination_6gpu.json."""
import json
import math
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs/coordination_6gpu.json"
TRAINING_KEYS = {"seed", "num_envs", "num_learning_iterations", "num_steps_per_env", "num_mini_batches",
                 "save_interval", "learning_rate", "schedule", "desired_kl", "entropy_coef"}

# Always register these fields so parameters.pkl can restore an experiment.
# Defaults preserve existing training and old checkpoint behavior.
DEFAULTS = {
    "coordination_experiment": {},
    "env.quarantine_invalid_physics": False,
    "env.numerical_max_root_speed": 100.0,
    "env.numerical_max_dof_speed": 1000.0,
    "rewards.attitude_command_convention": "legacy",
    "commands.coordination.enabled": False,
    "commands.coordination.velocity_schedule": [[0, 10.0], [8000, 10.0], [16000, 8.0], [24000, 6.0]],
    "commands.coordination.final_velocity_range_s": [6.0, 10.0],
    "commands.coordination.pose_range_s": [8.0, 12.0],
    "commands.coordination.gait_range_s": [15.0, 20.0],
    "commands.coordination.short_fraction": 0.0,
    "commands.coordination.short_start_iteration": 20000,
    "commands.coordination.short_range_s": [3.0, 4.0],
    "commands.coordination.standing_probability": 0.1,
    "commands.coordination.low_speed_probability": 0.0,
    "commands.coordination.low_speed_ranges": [[-0.3, 0.3], [-0.2, 0.2], [-0.4, 0.4]],
    "commands.coordination.transition_window_s": 1.0,
    "env.coordination_arm.enabled": False,
    "env.coordination_arm.fractions": [0.7, 0.2, 0.1],
    "env.coordination_arm.random_speed_scales": [0.35, 0.7, 1.0],
    "env.coordination_arm.structured_frequency_hz": [0.12, 0.45],
    "env.coordination_arm.workspace_fraction": 0.8,
    "env.coordination_arm.target_gain": 6.0,
    "env.coordination_arm.max_joint_velocity": [5.0] * 6,
    "env.coordination_arm.max_joint_acceleration": [10.0] * 6,
    "env.coordination_arm.seed": 1234,
}


def read_experiment(name, path=None):
    path = Path(path or DEFAULT_CONFIG).resolve()
    with path.open() as stream:
        plan = json.load(stream)
    if name not in plan["experiments"]:
        raise ValueError(f"Unknown experiment {name!r}; choose {list(plan['experiments'])}")
    candidate = plan["experiments"][name]
    training = dict(plan["training"], **candidate.get("training", {}))
    unknown = set(training) - TRAINING_KEYS
    if unknown:
        raise ValueError(f"Unknown training fields: {sorted(unknown)}")
    for key in ("num_envs", "num_learning_iterations", "num_steps_per_env", "num_mini_batches", "save_interval"):
        if type(training[key]) is not int or training[key] <= 0:
            raise ValueError(f"training.{key} must be a positive integer")
    if training["schedule"] not in ("adaptive", "fixed"):
        raise ValueError("training.schedule must be adaptive or fixed")
    for key in ("learning_rate", "desired_kl", "entropy_coef"):
        if not math.isfinite(training[key]) or training[key] < 0 or (key != "entropy_coef" and training[key] == 0):
            raise ValueError(f"invalid training.{key}")
    return {
        "name": name, "source": str(path), "description": candidate["description"],
        "overrides": dict(plan["common"], **candidate.get("overrides", {})),
        "training": training,
    }


def configure_experiment(cfg, args):
    from .core import apply_cfg_overrides
    apply_cfg_overrides(cfg, DEFAULTS, allow_new=True)
    name = getattr(args, "experiment", None)
    if name is None:
        return
    if getattr(args, "train_stage", "stage1") != "stage1" or not getattr(args, "dyna_gait", False):
        raise ValueError("Coordination experiments require --train_stage stage1 --dyna_gait")
    if any(getattr(args, key, None) for key in ("stage1_ckpt_path", "stage2_ckpt_path", "resume")):
        raise ValueError("These experiments train from scratch with corrected attitude semantics; do not load v2 weights")
    experiment = read_experiment(name, getattr(args, "experiment_config", None))
    apply_cfg_overrides(cfg, experiment["overrides"])
    cfg.coordination_experiment = experiment


def validate_coordination(cfg):
    if cfg.rewards.attitude_command_convention not in ("legacy", "rpy"):
        raise ValueError("attitude_command_convention must be legacy or rpy")
    c, a = cfg.commands.coordination, cfg.env.coordination_arm
    if c.enabled:
        schedule = c.velocity_schedule
        if not schedule or schedule[0][0] != 0:
            raise ValueError("velocity_schedule must start at iteration 0")
        if any(len(row) != 2 or not math.isfinite(row[0]) or row[0] < 0 or not math.isfinite(row[1]) or row[1] <= 0 for row in schedule):
            raise ValueError("invalid velocity_schedule iteration/time")
        if any(right[0] <= left[0] for left, right in zip(schedule, schedule[1:])):
            raise ValueError("velocity_schedule iterations must increase")
        for key in ("final_velocity_range_s", "pose_range_s", "gait_range_s", "short_range_s"):
            lo, hi = getattr(c, key)
            if not (0 < lo <= hi and math.isfinite(hi)):
                raise ValueError(f"invalid {key}")
        if not 0 <= c.short_fraction <= 1 or not 0 <= c.standing_probability <= 1:
            raise ValueError("command fractions must be in [0, 1]")
        if not 0 <= c.low_speed_probability <= 1 - c.standing_probability:
            raise ValueError("low-speed and standing probabilities must sum to at most one")
        if len(c.low_speed_ranges) != 3:
            raise ValueError("low_speed_ranges needs vx, vy, yaw ranges")
        for (lo, hi), (limit_lo, limit_hi) in zip(c.low_speed_ranges, (cfg.commands.limit_vel_x, cfg.commands.limit_vel_y, cfg.commands.limit_vel_yaw)):
            if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi) or (c.low_speed_probability > 0 and not limit_lo <= lo <= hi <= limit_hi):
                raise ValueError("low_speed_ranges must be finite and within command limits")
        if c.short_start_iteration < 0 or c.transition_window_s <= 0:
            raise ValueError("invalid command timing")
        if not cfg.commands.use_dynamic_gait or cfg.commands.limit_gait_frequency[0] <= 0:
            raise ValueError("coordination sampling needs dynamic gait with a positive moving frequency")
        if hasattr(cfg, "response") and (cfg.response.grouping.enabled or cfg.response.excitation.enabled):
            raise ValueError("coordination command sampling requires response grouping/excitation disabled")
    if a.enabled:
        if len(a.fractions) != 3 or any(x < 0 for x in a.fractions) or not math.isclose(sum(a.fractions), 1.0):
            raise ValueError("arm fractions must be [random, structured, hold] summing to 1")
        if not 0 < a.workspace_fraction <= 1 or a.target_gain <= 0:
            raise ValueError("invalid arm workspace/gain")
        for key in ("max_joint_velocity", "max_joint_acceleration"):
            values = getattr(a, key)
            if len(values) != cfg.arm.num_actions_arm or any(not math.isfinite(x) or x <= 0 for x in values):
                raise ValueError(f"{key} must contain one positive bound per arm joint")
        if len(a.random_speed_scales) != 3 or any(not 0 < x <= 1 for x in a.random_speed_scales):
            raise ValueError("random_speed_scales needs three scales in (0,1]")
        lo, hi = a.structured_frequency_hz
        if not 0 < lo <= hi or not math.isfinite(hi):
            raise ValueError("invalid structured_frequency_hz")
