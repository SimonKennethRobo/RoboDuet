"""Unified configuration schema, composition, layout, build and serialization.

All task builds return independent ``ConfigNode`` trees. Profiles are applied
in an explicit order and never mutate module-level classes or previous builds.
"""

from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Mapping

from .legged_robot import LeggedRobotDefaults


class ConfigNode:
    """Mutable attribute tree used by the simulator and learners."""

    def get(self, name, default=None):
        return getattr(self, name, default)

    def __repr__(self):
        return "ConfigNode({})".format(", ".join(sorted(key for key in vars(self) if not key.startswith("_"))))


@dataclass(frozen=True)
class ConfigProfile:
    name: str
    overrides: Mapping[str, Any]
    allow_new: bool = False


def _template_to_node(template):
    node = ConfigNode()
    for key, value in vars(template).items():
        if key.startswith("_") or callable(value) and not isinstance(value, type):
            continue
        if isinstance(value, type):
            value = _template_to_node(value)
        else:
            value = deepcopy(value)
        setattr(node, key, value)
    return node


def _config_value_to_data(value):
    if isinstance(value, ConfigNode):
        return cfg_to_dict(value)
    if is_dataclass(value):
        return {field.name: _config_value_to_data(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): _config_value_to_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_config_value_to_data(item) for item in value]
    if isinstance(value, set):
        return sorted(_config_value_to_data(item) for item in value)
    return value


def cfg_to_dict(cfg):
    """Serialize a complete config tree into plain Python containers."""

    result = {}
    for key, value in vars(cfg).items():
        if key.startswith("_"):
            continue
        result[key] = _config_value_to_data(value)
    return result


def set_cfg_value(cfg, path, value, *, allow_new=False):
    """Set one dotted path with schema validation and mutable-value isolation."""

    parts = path.split(".")
    target = cfg
    for index, part in enumerate(parts[:-1]):
        if not hasattr(target, part):
            if not allow_new:
                prefix = ".".join(parts[: index + 1])
                raise KeyError("Unknown config namespace: {}".format(prefix))
            setattr(target, part, ConfigNode())
        target = getattr(target, part)
        if not isinstance(target, ConfigNode):
            raise KeyError("Config path traverses a leaf value: {}".format(".".join(parts[: index + 1])))
    if not allow_new and not hasattr(target, parts[-1]):
        raise KeyError("Unknown config field: {}".format(path))
    setattr(target, parts[-1], deepcopy(value))


def apply_cfg_overrides(cfg, overrides, *, allow_new=False):
    """Apply a mapping of dotted config paths."""

    for path, value in overrides.items():
        set_cfg_value(cfg, path, value, allow_new=allow_new)


def apply_config_snapshot(cfg, snapshot, *, strict=True):
    """Restore a nested config snapshot onto an existing schema."""

    for key, value in snapshot.items():
        if not hasattr(cfg, key):
            if strict:
                raise KeyError("Unknown config field in snapshot: {}".format(key))
            setattr(cfg, key, ConfigNode() if isinstance(value, dict) else deepcopy(value))
        current = getattr(cfg, key)
        if isinstance(current, ConfigNode) and isinstance(value, dict):
            apply_config_snapshot(current, value, strict=strict)
        else:
            setattr(cfg, key, deepcopy(value))


def build_config(*profiles):
    """Compose independent config instances from the base template and profiles."""

    cfg = _template_to_node(LeggedRobotDefaults)
    provenance = {}
    for profile in profiles:
        apply_cfg_overrides(cfg, profile.overrides, allow_new=profile.allow_new)
        for path in profile.overrides:
            provenance[path] = profile.name
    cfg._profiles = tuple(profile.name for profile in profiles)
    cfg._provenance = provenance
    return cfg


def build_go1_config():
    from .go1 import GO1_PROFILE

    return build_config(GO1_PROFILE)


def build_wtw_config():
    from .go1 import GO1_PROFILE
    from .wtw import WTW_PROFILE

    return build_config(GO1_PROFILE, WTW_PROFILE)


def _derive_wbc_rewards(cfg, factors):
    cfg.wbc.reward_scales.tracking_lin_vel = factors["tracking_lin_vel"] * cfg.reward_scales.tracking_lin_vel
    cfg.wbc.reward_scales.tracking_ang_vel = factors["tracking_ang_vel"] * cfg.reward_scales.tracking_ang_vel
    cfg.wbc.reward_scales.arm_energy = factors["arm_energy"]
    cfg.wbc.reward_scales.arm_dof_vel = factors["arm_dof_vel"] * cfg.reward_scales.dof_vel
    cfg.wbc.reward_scales.arm_dof_acc = factors["arm_dof_acc"] * cfg.reward_scales.dof_acc
    cfg.wbc.reward_scales.arm_action_rate = factors["arm_action_rate"] * cfg.reward_scales.action_rate
    cfg.wbc.reward_scales.arm_action_smoothness_1 = (
        factors["arm_action_smoothness_1"] * cfg.reward_scales.action_smoothness_1
    )
    cfg.wbc.reward_scales.arm_action_smoothness_2 = factors["arm_action_smoothness_2"] * cfg.reward_scales.action_smoothness_2


@dataclass(frozen=True)
class RoboDuetRuntimeOptions:
    num_envs: int
    robot: str
    use_rot6d: bool = True
    traj_track: bool = False
    dyna_gait: bool = False
    dyna_gait_min_frequency: float = 0.0
    stage1_arm_curriculum: bool = True

    @classmethod
    def from_args(cls, args):
        return cls(
            num_envs=args.num_envs,
            robot=args.robot,
            use_rot6d=getattr(args, "use_rot6d", True),
            traj_track=getattr(args, "traj_track", False),
            dyna_gait=getattr(args, "dyna_gait", False),
            dyna_gait_min_frequency=getattr(args, "dyna_gait_min_frequency", 0.0),
            stage1_arm_curriculum=not getattr(args, "no_stage1_arm_curriculum", False),
        )


@dataclass
class RoboDuetLayout:
    arm_cmd: int
    dog_cmd: int
    arm_action_cd: int

    @classmethod
    def from_cfg(cls, cfg):
        return cls(
            arm_cmd=cfg.arm.arm_num_commands,
            dog_cmd=cfg.dog.dog_num_commands,
            arm_action_cd=cfg.arm.num_actions_arm_cd,
        )

    def finalize(self, cfg):
        cfg.arm.arm_num_commands = self.arm_cmd
        cfg.arm.num_actions_arm_cd = self.arm_action_cd
        cfg.dog.dog_num_commands = self.dog_cmd
        recompute_observation_dims(cfg)


def sum_dim_parts(parts):
    return sum(parts.values())


def env_obs_dim_parts(cfg):
    parts = {
        "dof_pos": cfg.env.num_actions,
        "dog_dof_vel": cfg.dog.num_actions_loco,
        "actions": cfg.env.num_actions,
        "dog_velocity_commands": 3,
        "arm_task": cfg.arm.arm_num_commands + 2,
    }
    if cfg.commands.use_dynamic_gait:
        parts["dynamic_gait_commands"] = cfg.dog.dog_num_commands - 3
    if cfg.env.observe_two_prev_actions:
        parts["two_prev_actions"] = cfg.env.num_actions
    if cfg.env.observe_timing_parameter:
        parts["timing_parameter"] = 1
    if cfg.env.observe_clock_inputs:
        parts["clock_inputs"] = 4
    if cfg.env.observe_vel:
        parts["base_velocity"] = 6
    if cfg.env.observe_only_ang_vel:
        parts["base_ang_vel"] = 3
    if cfg.env.observe_only_lin_vel:
        parts["base_lin_vel"] = 3
    if cfg.env.observe_yaw:
        parts["heading"] = 1
    if cfg.env.observe_contact_states:
        parts["contact_states"] = 4
    if cfg.wbc.trajectory.enabled:
        parts.update(
            {
                "arm_dof_vel": cfg.arm.num_actions_arm,
                "base_height": 1,
                "ee_pose_body": 9,
                "ee_twist_body": 6,
                "trajectory_window": len(cfg.wbc.trajectory.window_offsets) * 9,
                "trajectory_progress_index": 1,
            }
        )
    return parts


def arm_obs_dim_parts(cfg):
    if cfg.wbc.trajectory.enabled:
        parts = {
            "arm_dof_pos": cfg.arm.num_actions_arm,
            "arm_dof_vel": cfg.arm.num_actions_arm,
            "arm_actions": cfg.arm.num_actions_arm_cd,
            "base_height": 1,
            "base_ang_vel": 3,
            "dog_velocity_commands": 3,
            "trajectory_completion_time_command": 1,
            "ee_pose_body": 9,
            "ee_twist_body": 6,
            "trajectory_window": len(cfg.wbc.trajectory.window_offsets) * 9,
            "trajectory_progress_index": 1,
        }
        if cfg.commands.use_dynamic_gait:
            parts["dog_body_pose_commands"] = 3
            parts["dynamic_gait_commands"] = cfg.dog.dog_num_commands - 6
        return parts

    parts = {
        "arm_dof_pos": cfg.arm.num_actions_arm,
        "arm_dof_vel": cfg.arm.num_actions_arm,
        "arm_actions": cfg.arm.num_actions_arm_cd,
        "arm_commands": cfg.arm.arm_num_commands,
        "base_roll_pitch": 2,
    }
    if cfg.commands.use_dynamic_gait:
        parts["dog_body_pose_commands"] = 3
        parts["dynamic_gait_commands"] = cfg.dog.dog_num_commands - 6
    if cfg.env.observe_two_prev_actions:
        parts["two_prev_actions"] = cfg.env.num_actions
    return parts


def dog_obs_dim_parts(cfg):
    parts = {
        "projected_gravity": 3,
        "dog_dof_pos": cfg.dog.num_actions_loco,
        "dog_dof_vel": cfg.dog.num_actions_loco,
        "dog_actions": cfg.dog.num_actions_loco,
        "dog_commands": cfg.dog.dog_num_commands,
        "arm_commands": cfg.arm.arm_num_commands,
        "base_roll_pitch": 2,
        "arm_dof_pos": cfg.arm.num_actions_arm,
        "arm_dof_vel": cfg.arm.num_actions_arm,
    }
    if cfg.env.observe_two_prev_actions:
        parts["two_prev_actions"] = cfg.env.num_actions
    if cfg.env.observe_timing_parameter:
        parts["timing_parameter"] = 1
    if cfg.env.observe_clock_inputs:
        parts["clock_inputs"] = 4
    if cfg.env.observe_vel:
        parts["base_velocity"] = 6
    if cfg.env.observe_only_ang_vel:
        parts["base_ang_vel"] = 3
    if cfg.env.observe_only_lin_vel:
        parts["base_lin_vel"] = 3
    if cfg.env.observe_yaw:
        parts["heading"] = 1
    if cfg.env.observe_contact_states:
        parts["contact_states"] = 4
    if cfg.wbc.trajectory.enabled:
        parts["ee_pose_body"] = 9
    return parts


def privileged_obs_dim_parts(cfg, dof_dim, policy=None):
    parts = {}
    if cfg.env.priv_observe_friction:
        parts["friction"] = 1
    if cfg.env.priv_observe_ground_friction:
        parts["ground_friction"] = 1
    if cfg.env.priv_observe_restitution:
        parts["restitution"] = 1
    if cfg.env.priv_observe_base_mass:
        parts["base_mass"] = 1
    if cfg.env.priv_observe_com_displacement:
        parts["com_displacement"] = 3
    if cfg.env.priv_observe_motor_strength:
        parts["motor_strength"] = dof_dim
    if cfg.env.priv_observe_motor_offset:
        parts["motor_offset"] = dof_dim
    if cfg.env.priv_observe_Kp_factor:
        parts["kp_factor"] = dof_dim
    if cfg.env.priv_observe_Kd_factor:
        parts["kd_factor"] = dof_dim
    if cfg.env.priv_observe_joint_friction:
        parts["dof_friction"] = dof_dim
    if getattr(cfg.env, "priv_observe_dof_damping", False):
        parts["dof_damping"] = dof_dim
    if cfg.env.priv_observe_body_height:
        parts["body_height"] = 1
    if cfg.env.priv_observe_gravity:
        parts["gravity"] = 3
    if cfg.env.priv_observe_body_velocity or cfg.env.priv_observe_vel:
        parts["base_velocity"] = 6
    if cfg.env.priv_observe_clock_inputs:
        parts["clock_inputs"] = 4
    if cfg.env.priv_observe_desired_contact_states:
        parts["desired_contact_states"] = 4
    if cfg.env.priv_observe_high_freq_goal:
        parts["high_freq_goal"] = 6
    if getattr(cfg.env, "priv_observe_arm_mount_tf", False):
        parts["arm_mount_tf"] = 6
    if policy == "dog":
        parts["arm_dof_pos"] = cfg.arm.num_actions_arm
        parts["arm_dof_vel"] = cfg.arm.num_actions_arm
    return parts


def recompute_observation_dims(cfg):
    cfg.env.num_observations = sum_dim_parts(env_obs_dim_parts(cfg))
    cfg.env.num_obs_history = cfg.env.num_observation_history * cfg.env.num_observations
    cfg.arm.arm_num_observations = sum_dim_parts(arm_obs_dim_parts(cfg))
    cfg.arm.arm_num_obs_history = cfg.arm.arm_num_observation_history * cfg.arm.arm_num_observations
    cfg.dog.dog_num_observations = sum_dim_parts(dog_obs_dim_parts(cfg))
    cfg.dog.dog_num_obs_history = cfg.dog.dog_num_observation_history * cfg.dog.dog_num_observations


def configure_privileged_obs_dims(cfg):
    from .wbc import ROBODUET_OVERRIDES

    dog_parts = privileged_obs_dim_parts(cfg, cfg.dog.num_actions_loco, policy="dog")
    base_arm_action_dim = ROBODUET_OVERRIDES["arm.num_actions_arm_cd"]
    arm_parts = privileged_obs_dim_parts(cfg, base_arm_action_dim, policy="arm")
    if cfg.wbc.trajectory.enabled:
        arm_parts["foot_contact_states"] = 4
        arm_parts["full_trajectory"] = cfg.wbc.trajectory.num_waypoints * 9

    dog_dim = sum_dim_parts(dog_parts)
    arm_dim = sum_dim_parts(arm_parts)
    cfg.env.num_privileged_obs = dog_dim
    cfg.env.arm_num_privileged_obs = arm_dim
    cfg.env.dog_num_privileged_obs = dog_dim
    cfg.arm.arm_num_privileged_obs = arm_dim
    cfg.dog.dog_num_privileged_obs = dog_dim


def derive_privileged_normalization_ranges(cfg):
    """Cover every enabled dog/arm Kp/Kd randomization with one stable range."""

    randomization_profiles = (
        cfg.domain_rand,
        cfg.domain_rand.stage1_arm,
        cfg.domain_rand.stage2_arm,
    )
    for factor in ("Kp", "Kd"):
        if not getattr(cfg.env, f"priv_observe_{factor}_factor", False):
            continue
        enabled_ranges = [
            getattr(profile, f"{factor}_factor_range")
            for profile in randomization_profiles
            if getattr(profile, f"randomize_{factor}_factor", False)
        ]
        if not enabled_ranges:
            continue

        lower = min(float(value[0]) for value in enabled_ranges)
        upper = max(float(value[1]) for value in enabled_ranges)
        if lower >= upper:
            raise ValueError(f"Invalid enabled {factor} factor ranges: {enabled_ranges}")

        path = f"normalization.{factor}_factor_range"
        setattr(cfg.normalization, f"{factor}_factor_range", [lower, upper])
        cfg._provenance[path] = "derived:domain_rand"


def enable_rot6d(cfg, layout):
    from .wbc import FEATURE_LAYOUT

    cfg.use_rot6d = True
    layout.arm_cmd += FEATURE_LAYOUT["rot6d_command_dims"]


def enable_traj_track(cfg, layout, traj_track_reward_scale=5.0):
    from .wbc import FEATURE_LAYOUT, TRAJECTORY_REWARD_CONFIG

    cfg.wbc.trajectory.enabled = True
    layout.arm_action_cd = cfg.arm.num_actions_arm + FEATURE_LAYOUT["trajectory_plan_action_dims"]

    cfg.wbc.reward_scales.arm_manip_commands_tracking_combine = 0.0
    cfg.wbc.reward_scales.vis_manip_commands_tracking_lpy = 0.0
    cfg.wbc.reward_scales.vis_manip_commands_tracking_rpy = 0.0
    cfg.wbc.reward_scales.traj_track = traj_track_reward_scale
    for name, value in TRAJECTORY_REWARD_CONFIG.items():
        setattr(cfg.wbc.reward_scales, name, value)


def enable_dyna_gait(cfg, layout, min_frequency=0.0):
    from .wbc import DYNAMIC_GAIT_BIN_CONFIG, FEATURE_LAYOUT

    cfg.commands.use_dynamic_gait = True
    cfg.commands.gait_frequency_cmd_range = [min_frequency, cfg.commands.gait_frequency_cmd_range[1]]
    cfg.commands.limit_gait_frequency = deepcopy(cfg.commands.gait_frequency_cmd_range)
    cfg.commands.limit_footswing_height = deepcopy(cfg.commands.footswing_height_range)
    cfg.commands.limit_gait_duration = deepcopy(cfg.commands.gait_duration_cmd_range)
    cfg.commands.limit_stance_width = deepcopy(cfg.commands.stance_width_range)
    cfg.commands.limit_stance_length = deepcopy(cfg.commands.stance_length_range)

    layout.dog_cmd += FEATURE_LAYOUT["dynamic_gait_command_dims"]
    layout.arm_action_cd = cfg.arm.num_actions_arm + FEATURE_LAYOUT["dynamic_gait_plan_action_dims"]
    if cfg.wbc.trajectory.enabled:
        layout.arm_action_cd += FEATURE_LAYOUT["trajectory_plan_action_dims"]
    cfg.env.observe_gait_commands = True
    apply_cfg_overrides(cfg, DYNAMIC_GAIT_BIN_CONFIG)


def configure_robot_asset(cfg, robot):
    from .wbc import ROBOT_ASSET_FILES

    try:
        cfg.asset.file = ROBOT_ASSET_FILES[robot]
    except KeyError as exc:
        raise ValueError("Unknown robot {!r}; expected one of {}".format(robot, sorted(ROBOT_ASSET_FILES))) from exc


def validate_roboduet_cfg(cfg):
    required_fields = (
        ("env.num_observations", cfg.env.num_observations),
        ("env.num_obs_history", cfg.env.num_obs_history),
        ("arm.arm_num_observations", cfg.arm.arm_num_observations),
        ("arm.arm_num_obs_history", cfg.arm.arm_num_obs_history),
        ("dog.dog_num_observations", cfg.dog.dog_num_observations),
        ("dog.dog_num_obs_history", cfg.dog.dog_num_obs_history),
        ("arm.num_actions_arm_cd", cfg.arm.num_actions_arm_cd),
    )
    missing = [name for name, value in required_fields if value is None]
    if missing:
        raise ValueError("RoboDuet config has unset required fields: {}".format(", ".join(missing)))


def build_roboduet_config(args=None, *, options=None, traj_track_reward_scale=5.0, debug=False):
    """Build one finalized RoboDuet config without mutating another build."""

    from .go1 import GO1_PROFILE
    from .wbc import WBC_REWARD_FACTORS, ROBODUET_PROFILE
    from .wtw import WTW_PROFILE

    if options is None:
        options = RoboDuetRuntimeOptions.from_args(args) if args is not None else RoboDuetRuntimeOptions(4096, "go2")
    cfg = build_config(GO1_PROFILE, WTW_PROFILE, ROBODUET_PROFILE)
    _derive_wbc_rewards(cfg, WBC_REWARD_FACTORS)
    derive_privileged_normalization_ranges(cfg)
    cfg.env.num_envs = options.num_envs
    cfg.env.stage1_arm_curriculum = options.stage1_arm_curriculum

    layout = RoboDuetLayout.from_cfg(cfg)
    if options.use_rot6d:
        enable_rot6d(cfg, layout)
    else:
        cfg.use_rot6d = False

    if options.traj_track:
        enable_traj_track(cfg, layout, traj_track_reward_scale=traj_track_reward_scale)

    if options.dyna_gait:
        enable_dyna_gait(cfg, layout, min_frequency=options.dyna_gait_min_frequency)

    layout.finalize(cfg)
    configure_privileged_obs_dims(cfg)
    configure_robot_asset(cfg, options.robot)
    validate_roboduet_cfg(cfg)

    if debug:
        cfg.domain_rand.randomize_mount_pos = False

    return cfg
