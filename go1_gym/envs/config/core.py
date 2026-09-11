"""Unified configuration schema, composition, layout, build and serialization.

All task builds return independent ``ConfigNode`` trees. Profiles are applied
in an explicit order and never mutate module-level classes or previous builds.
"""

import warnings
import math
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Mapping

from .legged_robot import LeggedRobotDefaults
from .domain_randomization import domain_randomization_mode

# Arm action interface selector (cfg.arm.action_mode). Defined here rather
# than in wbc.py because both the CLI plumbing and validate_roboduet_cfg need
# it and wbc.py is only imported lazily from inside this module's builders.
# See the arm.action_mode block in wbc.py for what each mode means.
ARM_ACTION_MODES = ("ik_residual", "ik_waypoint", "end_to_end")


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


def apply_config_snapshot(cfg, snapshot, *, strict=True, drop_unknown=False):
    """Restore a nested config snapshot onto an existing schema.

    ``drop_unknown`` skips snapshot fields absent from the current schema
    instead of raising or creating them. Use it when restoring an old
    checkpoint after a field was deleted from the config (e.g. a since-removed
    dead param): the field cannot apply to the current code and must not be
    resurrected into the live cfg, so it is dropped with a warning.
    """

    # Full pre-mixture checkpoint snapshots must not inherit a newly enabled
    # reset mode from today's wbc.py. Partial config updates keep their meaning.
    if "env" in snapshot and "terrain" in snapshot and hasattr(cfg, "terrain"):
        cfg.terrain.height_reference = snapshot["terrain"].get("height_reference", "world")
        if "reset_mode" not in snapshot["terrain"]:
            cfg.terrain.reset_mode = "legacy"
        if "robustness_metrics" not in snapshot["terrain"]:
            cfg.terrain.robustness_metrics = False
    # Migrate the removed master flag without mutating the checkpoint data.
    # A disabled legacy flag took precedence even when a mode was present.
    if "domain_rand" in snapshot:
        snapshot = dict(snapshot)
        dr = dict(snapshot["domain_rand"])
        legacy_enabled = dr.pop("enabled", None)
        if legacy_enabled is False:
            dr["mode"] = "none"
        elif "mode" not in dr and (legacy_enabled is not None or "env" in snapshot):
            dr["mode"] = "sim2real"
        if "env" in snapshot:
            # Full pre-feature snapshots keep their original sensing/load model.
            dr.setdefault("randomize_dog_obs_latency", False)
            dr.setdefault("dog_obs_latency_steps_range", [0, 0])
            dr.setdefault("dog_obs_latency_jitter_steps", 0)
            if "stage1_arm" in dr:
                dr["stage1_arm"] = dict(dr["stage1_arm"])
                dr["stage1_arm"].setdefault("ee_payload_com_offset_range", [0.0, 0.0, 0.0])
        snapshot["domain_rand"] = dr
    if "env" in snapshot and "terrain" in snapshot and "roughness_tiers" not in snapshot["terrain"]:
        snapshot = dict(snapshot)
        snapshot["terrain"] = dict(snapshot["terrain"], roughness_tiers=[])
    for key, value in snapshot.items():
        if not hasattr(cfg, key):
            if drop_unknown:
                warnings.warn(
                    "Dropping config field absent from current schema: {}".format(key)
                )
                continue
            if strict:
                raise KeyError("Unknown config field in snapshot: {}".format(key))
            setattr(cfg, key, ConfigNode() if isinstance(value, dict) else deepcopy(value))
        current = getattr(cfg, key)
        if isinstance(current, ConfigNode) and isinstance(value, dict):
            apply_config_snapshot(current, value, strict=strict, drop_unknown=drop_unknown)
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
    dyna_gait: bool = False
    # Legacy compatibility input; gait-frequency bounds belong to the profile.
    dyna_gait_min_frequency: float = 0.0
    stage1_arm_curriculum: bool = True
    goal_reaching: bool = False
    traj_tracking: bool = False
    # How the actor's 6 arm action dims are decoded into joint targets:
    # 'ik_residual' | 'ik_waypoint' | 'end_to_end'. None (the default) keeps
    # whatever STAGE2_OVERRIDES in wbc.py sets, so editing that table works;
    # a value here overrides it. See ARM_ACTION_MODES.
    arm_action_mode: str = None
    # False forces rho / v_ff back onto the legacy reach_radius sphere instead
    # of the M2 direction-dependent table -- the ablation the design doc's
    # A.3 calls for (2D table vs sphere approximation).
    reach_table: bool = True
    # 'clock' | 'clock_free' | None. None keeps whatever the wbc.py override
    # tables set (currently 'clock'); a value swaps the gait reward table via
    # set_gait_reward_mode. Pair 'clock_free' with --no_clock_inputs.
    gait_reward_mode: str = None
    # 'quadratic' | 'exp' | None. None keeps the config default (currently
    # 'quadratic', the legacy multiplicative cost). 'exp' makes
    # raibert_heuristic a bounded additive reward -- see set_raibert_form.
    raibert_form: str = None
    domain_rand_mode: str = None

    @classmethod
    def from_args(cls, args):
        return cls(
            domain_rand_mode=getattr(args, "domain_rand_mode", None),
            num_envs=args.num_envs,
            robot=args.robot,
            use_rot6d=getattr(args, "use_rot6d", True),
            dyna_gait=getattr(args, "dyna_gait", False),
            dyna_gait_min_frequency=getattr(args, "dyna_gait_min_frequency", 0.0),
            stage1_arm_curriculum=not getattr(args, "no_stage1_arm_curriculum", False),
            goal_reaching=getattr(args, "goal_reaching", False),
            traj_tracking=getattr(args, "traj_tracking", False),
            arm_action_mode=getattr(args, "arm_action_mode", None),
            reach_table=not getattr(args, "no_reach_table", False),
            gait_reward_mode=(
                "clock_free" if getattr(args, "clock_free_gait", False)
                else getattr(args, "gait_reward_mode", None)
            ),
            raibert_form=(
                "exp" if getattr(args, "raibert_exp", False)
                else getattr(args, "raibert_form", None)
            ),
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
    return parts


def arm_obs_dim_parts(cfg):
    """Single source of truth for actor-facing arm observations.

    Goal-reaching mode adds whole-body state and coordination diagnostics for
    the 12D upper actor. The default layout stays compatible with the legacy
    6D DLS-residual policy.
    """
    if getattr(getattr(cfg.wbc, "goal_reaching", None), "enabled", False):
        parts = {
            "ee_pos_err_body": 3,
            "ee_rot_err_body": 3,
            "ee_twist_body": 6,
            "ee_target_pos_body": 3,
            "ee_target_rot6d_body": 6,
            "arm_dof_pos": cfg.arm.num_actions_arm,
            "arm_dof_vel": cfg.arm.num_actions_arm,
            "base_roll_pitch_height": 3,
            "base_twist": 6,
            "dog_vel_residual": 3,
            "dog_contact_states": 4,
            "base_feedforward": 3,
            "arm_actions": cfg.arm.num_actions_arm_cd,
        }
        if getattr(cfg.wbc.goal_reaching, "target_mode", "static") == "trajectory":
            # Moving-trajectory extras (actor-facing). K preview points, each
            # carrying pos(3)+rot6d(6)+tangent(3)+sdot(1)+rho(1)=14, plus
            # progress/timing scalars, a tau phase encoding, and the reach
            # urgency pair. Total = 8 + K*14 (= 134 at K=9).
            K = int(cfg.wbc.goal_reaching.trajectory.preview_points)
            parts["traj_progress_scalars"] = 4  # s_norm, timing_err, sdot, sdot_ref
            parts["traj_tau_enc"] = 2
            parts["traj_urgency"] = 2  # urgency, s_to_viol
            parts["traj_preview"] = K * 14
        return parts

    parts = {
        "ee_pos_err": 3,
        "ee_rot_err_axis_angle": 3,
        "ee_target_pos_body": 3,
        "ee_target_rot6d_body": 6,
        "arm_dof_pos": cfg.arm.num_actions_arm,
        "arm_dof_vel": cfg.arm.num_actions_arm,
        "arm_actions": cfg.arm.num_actions_arm_cd,
    }
    if cfg.env.observe_two_prev_actions:
        parts["two_prev_actions"] = cfg.env.num_actions
    if getattr(cfg.env, "arm_observe_dog_state", False):
        parts["dog_contact_states"] = 4
        parts["dog_vel_residual"] = 3
    return parts


def dog_obs_term_present(cfg, switch):
    """Old checkpoints retain zero slots; new policies omit disabled terms."""
    version = cfg.dog.observation_layout_version
    if version not in (1, 2):
        raise ValueError(f"Unsupported dog observation layout version: {version}")
    return version == 1 or bool(getattr(cfg.dog, switch))


def restore_dog_observation_layout(cfg, snapshot):
    """Restore actor input semantics, including pre-versioned checkpoints.

    Also used when loading a stage-1 dog into stage-2 training. Do not inherit
    today's observation switches for a saved actor, even if widths coincide.
    Call before creating the environment/history buffers.
    """
    cfg.terrain.height_reference = snapshot.get("terrain", {}).get("height_reference", "world")
    dog = snapshot.get("dog", {})
    # Width equality alone cannot prove command/rotation semantics agree.
    for section, names in (
        ("dog", ("dog_num_commands", "num_actions_loco")),
        ("arm", ("arm_num_commands", "num_actions_arm")),
        ("env", ("observe_two_prev_actions", "observe_timing_parameter",
                 "observe_yaw", "observe_contact_states")),
        ("wbc", ("use_vision",)),
    ):
        saved = snapshot.get(section, {})
        for name in names:
            if name in saved and saved[name] != getattr(getattr(cfg, section), name):
                raise ValueError(f"Dog checkpoint requires {section}.{name}={saved[name]}; "
                                 "the runtime observation layout differs")
    if "use_rot6d" in snapshot and snapshot["use_rot6d"] != cfg.use_rot6d:
        raise ValueError("Dog checkpoint and runtime use different rot6d representations")
    cfg.dog.observation_layout_version = dog.get("observation_layout_version", 1)
    cfg.dog.observe_clock_inputs = dog.get(
        "observe_clock_inputs", snapshot.get("env", {}).get("observe_clock_inputs", True)
    )
    for name in ("observe_lin_vel", "observe_pose_actual", "observe_track_error"):
        setattr(cfg.dog, name, dog.get(name, True))
    if "dog_num_observation_history" in dog:
        cfg.dog.dog_num_observation_history = dog["dog_num_observation_history"]
    recompute_observation_dims(cfg)
    recorded = dog.get("dog_num_observations")
    if recorded is not None and int(recorded) != cfg.dog.dog_num_observations:
        raise ValueError(
            f"Dog checkpoint observation layout mismatch: saved={recorded}, "
            f"reconstructed={cfg.dog.dog_num_observations}. Check command widths, "
            "rot6d and observation switches against parameters.pkl."
        )
    recorded_history = dog.get("dog_num_obs_history")
    if recorded_history is not None and int(recorded_history) != cfg.dog.dog_num_obs_history:
        raise ValueError("Dog checkpoint history width disagrees with frame width and history length")


def dog_obs_dim_parts(cfg):
    parts = {
        "projected_gravity": 3,
        "dog_dof_pos": cfg.dog.num_actions_loco,
        "dog_dof_vel": cfg.dog.num_actions_loco,
        "dog_actions": cfg.dog.num_actions_loco,
        "dog_commands": cfg.dog.dog_num_commands,
        "arm_commands": cfg.arm.arm_num_commands,
        "arm_dof_pos": cfg.arm.num_actions_arm,
        "arm_dof_vel": cfg.arm.num_actions_arm,
    }
    if cfg.env.observe_two_prev_actions:
        parts["two_prev_actions"] = cfg.env.num_actions
    if cfg.env.observe_timing_parameter:
        parts["timing_parameter"] = 1
    if cfg.dog.observe_clock_inputs:
        parts["clock_inputs"] = 4
    parts["base_ang_vel"] = 3
    if dog_obs_term_present(cfg, "observe_lin_vel"):
        parts["base_lin_vel"] = 3
    if dog_obs_term_present(cfg, "observe_pose_actual"):
        parts["body_pose_actual"] = 3
    if dog_obs_term_present(cfg, "observe_track_error"):
        parts["body_pose_error"] = 3
        parts["velocity_error"] = 3
    if cfg.env.observe_yaw:
        parts["heading"] = 1
    if cfg.env.observe_contact_states:
        parts["contact_states"] = 4
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
    if cfg.env.priv_observe_com_displacement and (
        policy != "dog" or cfg.dog.priv_observe_com_displacement
    ):
        parts["com_displacement"] = 3
    if getattr(cfg.env, "priv_observe_stage1_ee_payload_mass", False):
        parts["stage1_ee_payload_mass"] = 1
    if policy == "dog" and cfg.dog.priv_observe_motor_strength:
        parts["dog_motor_strength"] = 1
    if policy == "dog" and cfg.dog.priv_observe_motor_offset:
        parts["dog_motor_offset"] = dof_dim
    if cfg.env.priv_observe_motor_strength:
        parts["motor_strength"] = dof_dim
    if cfg.env.priv_observe_motor_offset:
        parts["motor_offset"] = dof_dim
    if cfg.env.priv_observe_Kp_factor:
        parts["kp_factor"] = 1 if policy == "dog" else dof_dim
    if cfg.env.priv_observe_Kd_factor:
        parts["kd_factor"] = 1 if policy == "dog" else dof_dim
    if cfg.env.priv_observe_joint_friction and (
        policy != "dog" or cfg.dog.priv_observe_joint_friction
    ):
        parts["dof_friction"] = dof_dim
    if getattr(cfg.env, "priv_observe_dof_damping", False) and (
        policy != "dog" or cfg.dog.priv_observe_dof_damping
    ):
        parts["dof_damping"] = dof_dim
    if cfg.env.priv_observe_body_height:
        parts["body_height"] = 1
    if cfg.env.priv_observe_gravity:
        parts["gravity"] = 3
    if policy == "dog" and cfg.dog.priv_observe_gravity:
        parts["dog_gravity"] = 3
    if cfg.env.priv_observe_body_velocity or cfg.env.priv_observe_vel:
        parts["base_velocity"] = 6
    if cfg.env.priv_observe_clock_inputs:
        parts["clock_inputs"] = 4
    if cfg.env.priv_observe_desired_contact_states:
        parts["desired_contact_states"] = 4
    if policy == "dog" and cfg.dog.priv_observe_contact_states:
        parts["dog_contact_states"] = 4
    if cfg.env.priv_observe_high_freq_goal:
        parts["high_freq_goal"] = 6
    if getattr(cfg.env, "priv_observe_arm_mount_tf", False):
        parts["arm_mount_tf"] = 6
    if policy == "dog":
        if cfg.dog.priv_observe_arm_dynamics:
            parts["arm_kp_factor"] = cfg.arm.num_actions_arm
            parts["arm_kd_factor"] = cfg.arm.num_actions_arm
            parts["arm_motor_strength"] = cfg.arm.num_actions_arm
            parts["arm_motor_offset"] = cfg.arm.num_actions_arm
            parts["arm_link_mass_scale"] = cfg.arm.num_privileged_links
            parts["arm_link_com_offset"] = 3 * cfg.arm.num_privileged_links
        parts["arm_dof_pos"] = cfg.arm.num_actions_arm
        parts["arm_dof_vel"] = cfg.arm.num_actions_arm
    if policy == "arm" and getattr(getattr(cfg.wbc, "goal_reaching", None), "enabled", False):
        parts["goal_manipulability"] = 1
        parts["goal_joint_limit_distance"] = cfg.arm.num_actions_arm
        parts["goal_rho"] = 1
        parts["arm_ema_motion"] = cfg.arm.num_actions_arm
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

    dog_dim = sum_dim_parts(dog_parts)
    arm_dim = sum_dim_parts(arm_parts)
    cfg.env.num_privileged_obs = dog_dim
    cfg.env.arm_num_privileged_obs = arm_dim
    cfg.env.dog_num_privileged_obs = dog_dim
    cfg.arm.arm_num_privileged_obs = arm_dim
    cfg.dog.dog_num_privileged_obs = dog_dim


def derive_privileged_normalization_ranges(cfg):
    """Align privileged normalization ranges with enabled randomizations."""

    direct_ranges = {
        "friction_range": "friction_range",
        "restitution_range": "restitution_range",
        "added_mass_range": "added_mass_range",
        "motor_strength_range": "motor_strength_range",
        "motor_offset_range": "motor_offset_range",
        "gravity_range": "gravity_range",
    }
    for normalization_name, randomization_name in direct_ranges.items():
        value = deepcopy(getattr(cfg.domain_rand, randomization_name))
        setattr(cfg.normalization, normalization_name, value)
        cfg._provenance[f"normalization.{normalization_name}"] = "derived:domain_rand"
    for factor in ("Kp", "Kd"):
        value = deepcopy(getattr(cfg.domain_rand, f"{factor}_factor_range"))
        setattr(cfg.normalization, f"dog_{factor}_factor_range", value)
        cfg._provenance[f"normalization.dog_{factor}_factor_range"] = "derived:domain_rand"
    if getattr(cfg.env, "priv_observe_stage1_ee_payload_mass", False):
        value = deepcopy(cfg.domain_rand.stage1_arm.ee_payload_mass_range)
        cfg.normalization.stage1_ee_payload_mass_range = value
        cfg._provenance["normalization.stage1_ee_payload_mass_range"] = "derived:domain_rand"

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

    arm_profiles = (cfg.domain_rand.stage1_arm, cfg.domain_rand.stage2_arm)

    def derive_arm_range(source_field, target_field, enabled_field, *, symmetric=False):
        values = [getattr(profile, source_field) for profile in arm_profiles if getattr(profile, enabled_field, False)]
        if not values:
            return
        if symmetric:
            radius = max(abs(float(value)) for value in values)
            result = [-radius, radius]
        else:
            result = [min(float(value[0]) for value in values), max(float(value[1]) for value in values)]
        setattr(cfg.normalization, target_field, result)
        cfg._provenance[f"normalization.{target_field}"] = "derived:domain_rand"

    derive_arm_range("motor_strength_range", "arm_motor_strength_range", "randomize_motor_strength")
    derive_arm_range("motor_offset_range", "arm_motor_offset_range", "randomize_motor_offset", symmetric=True)
    derive_arm_range("Kp_factor_range", "arm_Kp_factor_range", "randomize_Kp_factor")
    derive_arm_range("Kd_factor_range", "arm_Kd_factor_range", "randomize_Kd_factor")
    derive_arm_range("link_mass_range", "arm_link_mass_scale_range", "randomize_link_mass")
    derive_arm_range("link_com_range", "arm_link_com_offset_range", "randomize_link_com", symmetric=True)


def enable_rot6d(cfg, layout):
    from .wbc import FEATURE_LAYOUT

    cfg.use_rot6d = True
    layout.arm_cmd += FEATURE_LAYOUT["rot6d_command_dims"]


def enable_dyna_gait(cfg, layout):
    from .wbc import DYNAMIC_GAIT_BIN_CONFIG, FEATURE_LAYOUT

    cfg.commands.use_dynamic_gait = True
    # Keep the profile's sampling range and curriculum limits independent.
    # A layout toggle must not reset the configured frequency lower bound.
    cfg.commands.limit_footswing_height = deepcopy(cfg.commands.footswing_height_range)
    cfg.commands.limit_gait_duration = deepcopy(cfg.commands.gait_duration_cmd_range)
    cfg.commands.limit_stance_width = deepcopy(cfg.commands.stance_width_range)
    cfg.commands.limit_stance_length = deepcopy(cfg.commands.stance_length_range)

    layout.dog_cmd += FEATURE_LAYOUT["dynamic_gait_command_dims"]
    cfg.env.observe_gait_commands = True
    apply_cfg_overrides(cfg, DYNAMIC_GAIT_BIN_CONFIG)


def enable_goal_reaching(cfg, layout):
    from .wbc import FEATURE_LAYOUT, GOAL_REACHING_REWARD_SCALES

    if not cfg.commands.use_dynamic_gait:
        raise ValueError("goal_reaching requires dynamic gait commands")
    cfg.wbc.goal_reaching.enabled = True
    layout.arm_action_cd = cfg.arm.num_actions_arm + FEATURE_LAYOUT["goal_reaching_plan_action_dims"]
    for name, scale in GOAL_REACHING_REWARD_SCALES.items():
        setattr(cfg.wbc.reward_scales, name, scale)


def enable_traj_tracking(cfg, layout):
    """Switch goal reaching from a static point to a moving SE(3) trajectory.

    A strict superset on top of ``enable_goal_reaching`` (which must have run
    first): flips ``target_mode`` to 'trajectory', applies the trajectory
    overrides, and swaps the static goal_reaching reward set for the
    trajectory-tracking one (Group T tracking terms + the carried-over
    reachability/smoothness terms; the static-only terms are zeroed).
    """
    from .wbc import TRAJ_TRACKING_OVERRIDES, TRAJ_TRACKING_REWARD_SCALES

    apply_cfg_overrides(cfg, TRAJ_TRACKING_OVERRIDES, allow_new=True)
    for name, scale in TRAJ_TRACKING_REWARD_SCALES.items():
        setattr(cfg.wbc.reward_scales, name, scale)


def validate_arm_action_mode(cfg):
    """Check cfg.arm.action_mode, wherever it came from -- the wbc.py override
    table, a --arm_action_mode flag, or a restored checkpoint snapshot."""
    mode = getattr(cfg.arm, "action_mode", None)
    if mode not in ARM_ACTION_MODES:
        raise ValueError(
            "arm.action_mode is {!r}; expected one of {}".format(mode, sorted(ARM_ACTION_MODES))
        )
    if mode == "ik_waypoint" and cfg.arm.num_actions_arm != 6:
        # The 6 dims are read as dpos(3) + axis-angle drot(3), not per-joint.
        raise ValueError(
            "arm.action_mode='ik_waypoint' needs exactly 6 arm action dims "
            "(an SE(3) waypoint offset); arm.num_actions_arm is {}".format(cfg.arm.num_actions_arm)
        )


# ============================================================
# cfg.reward_scales and cfg.wbc.reward_scales are two CO-EQUAL SIBLING
# tables, not a namespace/override pair -- despite "wbc." reading like a
# sub-namespace the way every other wbc.* field is. They are read by
# global_switch.get_reward_scales(): "stage 1" (a.k.a. "pretrained") reads
# cfg.reward_scales; "stage 2" (a.k.a. "wbc") reads cfg.wbc.reward_scales;
# a --train_stage two_stage run interpolates between them; a --train_stage
# stage1 run pins the switch threshold past the end of training, so it ALWAYS
# reads cfg.reward_scales and cfg.wbc.reward_scales is never consulted for
# its VALUES at all.
#
# But both tables still matter for stage-1-only training, for a second and
# entirely different reason: LeggedRobot._prepare_reward_function's merge
# step decides, ONCE, which reward names get computed *at all* (a single
# reward_names list shared by every stage -- get_reward_scales only ever
# picks which VALUES those names use). A name absent from cfg.wbc.reward_
# scales inherits cfg.reward_scales' value into the registration check; a
# name PRESENT there at exactly 0.0 does not, and is dropped for every
# stage regardless of cfg.reward_scales. Before the fix below, that dropped
# raibert_heuristic silently for six weeks (2026-07-26 to 2026-09-05, commit
# 15a4581) purely because wbc.py had it at -0.0 to mean "off for stage 2" --
# a --train_stage stage1 run was affected exactly as much as a two_stage one,
# and nothing in the log said so.
#
# Use resolve_reward_scales(cfg) to answer "is X actually computed, and with
# what value in each stage" from a plain cfg object -- no simulator, no env,
# no live run needed. Every ad-hoc reimplementation of this merge (there have
# been several, each risking drifting from the real algorithm) should go
# through this function instead.
def resolve_reward_scales(cfg):
    """Non-mutating preview of _prepare_reward_function's registration logic.

    Returns {name: {"stage1": float, "stage2": float | None, "active": bool}}
    for every key appearing in either cfg.reward_scales or cfg.wbc.reward_
    scales. "stage2" is None when the name is absent from cfg.wbc.reward_
    scales (i.e. it would inherit "stage1" into the registration check, per
    the comment above). "active" is a single flag, not one per stage: exactly
    one reward_names list is built and walked in every stage, so a name is
    either computed everywhere or nowhere.

    dt does not affect any of this (scaling by a positive dt never changes
    whether a value is zero), so this reports the raw config scales, not the
    dt-scaled values LeggedRobot._parse_cfg/_prepare_reward_function compute
    at env creation.
    """
    stage1 = {k: v for k, v in vars(cfg.reward_scales).items() if not k.startswith("_")}
    stage2 = {k: v for k, v in vars(cfg.wbc.reward_scales).items() if not k.startswith("_")}

    # Matches _prepare_reward_function's fixed pop condition directly, rather
    # than replaying its dict-mutation sequence (an earlier version of this
    # function did that and reproduced the *pre-fix* bug: dict-mutation order
    # matters and is easy to get subtly wrong, so state the invariant instead
    # of re-deriving it procedurally).
    return {
        name: {
            "stage1": stage1.get(name, 0.0),
            "stage2": stage2.get(name),
            "active": stage1.get(name, 0.0) != 0 or stage2.get(name, 0.0) != 0,
        }
        for name in set(stage1) | set(stage2)
    }


# Gait shaping, selected by rewards.gait_reward_mode. The tables write
# cfg.reward_scales.* -- _prepare_reward_function copies any name absent from
# cfg.wbc.reward_scales into the stage-2 table at the same value, so one entry
# drives both stages. A nonzero entry is deliberately *not* mirrored into
# cfg.wbc.reward_scales: that table carries stage-2-specific tuning (e.g.
# raibert_heuristic at -1.0 rather than stage 1's -10.0) which must survive a
# mode switch. Zero entries are mirrored, because a name left nonzero there
# would keep a clock-based term alive in stage 2 after clock_free turned it
# off in stage 1 -- see set_gait_reward_mode.
GAIT_REWARD_MODES = {
    # Scored against _step_contact_targets' absolute phase. Requires
    # dog.observe_clock_inputs, or the actor is graded on a target it cannot
    # observe. These are the stock wtw.py values.
    "clock": {
        "tracking_contacts_shaped_force": 4.0,
        "tracking_contacts_shaped_vel": 4.0,
        "feet_clearance_cmd_linear": -30.0,
        "raibert_heuristic": -10.0,
        "gait_sync": 0.0,
        "feet_air_time_variance": 0.0,
        "joint_mirror": 0.0,
        "feet_stance_width": 0.0,
        "feet_swing_height": 0.0,
    },
    # Contact stopwatches, foot geometry and joint symmetry only -- nothing
    # reads foot_indices or desired_contact_states. Scales are chosen against
    # the ji22 shaping in use here (only_positive_rewards_ji22_style with
    # sigma_rew_neg=0.02), where the total is rew_pos * exp(rew_neg / 0.02):
    # a negative term costs a factor exp(scale * dt * value / 0.02), so at
    # dt=0.02 a per-step product of -0.02 already costs 37% of the reward.
    # gait_sync and feet_stance_width are bounded positives and land in
    # rew_pos, so they cannot close that gate at all; the three costs below
    # are sized to stay well inside it. Raise joint_mirror toward -1.0/-2.0
    # if the symmetry effect is too weak, and watch rew_total for the
    # collapse signature.
    "clock_free": {
        "tracking_contacts_shaped_force": 0.0,
        "tracking_contacts_shaped_vel": 0.0,
        "feet_clearance_cmd_linear": 0.0,
        "raibert_heuristic": 0.0,
        "gait_sync": 2.0,
        "feet_air_time_variance": -2.0,
        "joint_mirror": -0.5,
        "feet_stance_width": 1.0,
        "feet_swing_height": -20.0,
    },
}


def set_gait_reward_mode(cfg, mode):
    """Swap the gait shaping between the clock-based and clock-free tables.

    ``mode=None`` leaves cfg alone, so the override tables in wbc.py stay the
    source of truth and the CLI flag is a genuine override (same convention as
    set_arm_action_mode).

    Changes no observation or action dimension, so it is safe to flip on a
    resume -- but the two tables optimise different objectives, and a policy
    trained under one is not comparable to one trained under the other.
    """
    if mode is None:
        return
    if mode not in GAIT_REWARD_MODES:
        raise ValueError(
            f"Unknown rewards.gait_reward_mode {mode!r}; expected one of {sorted(GAIT_REWARD_MODES)}"
        )
    cfg.rewards.gait_reward_mode = mode
    for name, scale in GAIT_REWARD_MODES[mode].items():
        setattr(cfg.reward_scales, name, scale)
        # Disabling has to reach stage 2 as well: wbc.reward_scales overrides
        # the stage-1 value, so leaving it nonzero would keep a clock-based
        # term scoring the WBC policy against a phase the actor cannot see.
        # Enabling deliberately does not, so stage-2-specific tuning survives.
        if scale == 0.0 and hasattr(cfg.wbc.reward_scales, name):
            setattr(cfg.wbc.reward_scales, name, 0.0)


# Paired (form, stage-1 scale, stage-2 scale) presets for
# _reward_raibert_heuristic. The sign is part of the form, not a free
# parameter: 'quadratic' returns a cost and 'exp' returns a bounded reward.
RAIBERT_FORMS = {
    "quadratic": {"reward_scales": -10.0, "wbc": -1.0},
    # exp lands in rew_buf_pos and adds rather than gates. Calibrated (with
    # rewards.raibert_sigma=0.35) against the real rew_pos budget of a healthy
    # run -- tracking_lin_vel + tracking_ang_vel summed to ~18.4 over an
    # episode in stage1_sim2real_abl_14 -- so that a mostly-well-placed foot
    # (reward around 0.5-0.6/step, see raibert_sigma's comment) contributes
    # roughly 20% of that, not enough to dominate velocity tracking, still
    # enough to matter. Stage 2 gets half: it has EE tracking to leave room
    # for. Both numbers are a calibration by formula, not yet confirmed by an
    # actual 'exp'-form training run -- no run in stage1_sim2real_abl_1..15
    # used this form, they were all 'quadratic' regardless of what the table
    # said (see resolve_reward_scales).
    "exp": {"reward_scales": 0.4, "wbc": 0.2},
}


def set_raibert_form(cfg, form):
    """Switch _reward_raibert_heuristic between the cost and bounded forms.

    ``form=None`` leaves cfg alone. Anything else rewrites both the form and
    the matching scales together -- flipping the form without flipping the
    sign would reward bad foot placement, and nothing downstream would catch
    it, so the two are never settable independently through this path.
    """
    if form is None:
        return
    if form not in RAIBERT_FORMS:
        raise ValueError(f"Unknown rewards.raibert_form {form!r}; expected one of {sorted(RAIBERT_FORMS)}")
    cfg.rewards.raibert_form = form
    scales = RAIBERT_FORMS[form]
    # Only rescale a term that is actually live: gait_reward_mode='clock_free'
    # zeroes raibert on purpose, and that must not be undone here.
    if cfg.reward_scales.raibert_heuristic != 0.0:
        cfg.reward_scales.raibert_heuristic = scales["reward_scales"]
    if getattr(cfg.wbc.reward_scales, "raibert_heuristic", 0.0) != 0.0:
        cfg.wbc.reward_scales.raibert_heuristic = scales["wbc"]


def validate_raibert_form(cfg):
    """Reject a raibert form whose scale has the wrong sign.

    'quadratic' returns a cost and 'exp' returns a bounded reward, so a scale
    carried over from the other form flips the objective: the policy would be
    paid to put its feet in the wrong place. That trains quietly to a
    plausible-looking reward curve, so it is an error, not a warning.
    """
    form = getattr(cfg.rewards, "raibert_form", "quadratic")
    if form not in RAIBERT_FORMS:
        raise ValueError(f"Unknown rewards.raibert_form {form!r}; expected one of {sorted(RAIBERT_FORMS)}")
    wanted = "positive" if form == "exp" else "negative"
    for label, scale in (
        ("reward_scales.raibert_heuristic", cfg.reward_scales.raibert_heuristic),
        ("wbc.reward_scales.raibert_heuristic", getattr(cfg.wbc.reward_scales, "raibert_heuristic", 0.0)),
    ):
        if scale == 0.0:
            continue  # disabled; sign is meaningless
        if (form == "exp") != (scale > 0):
            raise ValueError(
                f"rewards.raibert_form='{form}' needs a {wanted} {label}, got {scale}. "
                f"The 'exp' form returns a bounded reward in [0, 1] and the 'quadratic' "
                f"form returns an unbounded cost -- use set_raibert_form so the two stay paired."
            )


def validate_gait_reward_mode(cfg):
    """Warn when the gait shaping and the clock observation disagree.

    Not an error: 'clock' shaping without clock_inputs is exactly the ablation
    stage1_sim2real_abl_2/8/9/10 ran, and 'clock_free' shaping with the clock
    still observed is a harmless superset. But the first combination grades the
    actor against a phase it cannot see, which is easy to do by accident and
    hard to spot in the logs, so say so out loud.
    """
    mode = getattr(cfg.rewards, "gait_reward_mode", "clock")
    observes_clock = bool(getattr(cfg.dog, "observe_clock_inputs", True))
    if mode == "clock" and not observes_clock:
        print(
            "[config] WARNING: rewards.gait_reward_mode='clock' but "
            "dog.observe_clock_inputs is off -- tracking_contacts_shaped_*, "
            "feet_clearance_cmd_linear and raibert_heuristic all score against "
            "the gait phase, which the actor cannot observe. Use "
            "--clock_free_gait, or turn the clock observation back on."
        )


def set_arm_action_mode(cfg, mode):
    """Override how the actor's 6 arm action dims become joint position targets.

    ``mode=None`` is "leave it alone" -- the value already in cfg (from
    STAGE2_OVERRIDES in wbc.py, or a restored snapshot) stands. That is what
    makes the override table the default source of truth and --arm_action_mode
    a genuine override, rather than the CLI's own default silently winning
    every build.

    Every mode keeps the same 6-wide arm action head, so this changes no
    observation/action dimension and no checkpoint shape -- only WBCEnv's
    decoding (``_apply_stage2_arm_action``). It is therefore safe to flip on a
    resume, though the resulting policy is of course not transferable across
    modes.
    """
    if mode is not None:
        cfg.arm.action_mode = mode
    validate_arm_action_mode(cfg)


def configure_robot_asset(cfg, robot):
    from .wbc import ROBOT_ASSET_FILES, ROBOT_ARM_SPEC

    try:
        cfg.asset.file = ROBOT_ASSET_FILES[robot]
    except KeyError as exc:
        raise ValueError("Unknown robot {!r}; expected one of {}".format(robot, sorted(ROBOT_ASSET_FILES))) from exc

    spec = ROBOT_ARM_SPEC[robot]
    cfg.asset.ee_body_name = spec["ee_body_name"]
    cfg.arm.ik.ee_local_pos = list(spec["ee_local_pos"])
    cfg.domain_rand.mount_joint_name = spec["mount_joint_name"]
    # M2 reachability table for this robot (scripts/build_reach_table.py). The
    # env falls back to the scalar reach_radius sphere if the file is absent,
    # so this path pointing at nothing is a supported configuration.
    cfg.wbc.goal_reaching.reach_table_path = (
        "{MINI_GYM_ROOT_DIR}/resources/reach_tables/" + f"{robot}_2d.pt"
    )


def validate_roboduet_cfg(cfg):
    domain_randomization_mode(cfg)
    if cfg.terrain.height_reference not in ("world", "terrain"):
        raise ValueError("terrain.height_reference must be world or terrain")
    if cfg.terrain.height_reference == "terrain":
        if cfg.terrain.mesh_type == "trimesh" and cfg.terrain.slope_treshold is not None:
            raise ValueError("terrain height queries require terrain.slope_treshold=None (unshifted triangles)")
        if cfg.terrain.mesh_type not in ("plane", "trimesh", "heightfield"):
            raise ValueError("terrain height reference requires a ground surface")
    for name in ("priv_observe_ground_friction", "priv_observe_ground_friction_per_foot"):
        if getattr(cfg.env, name, False):
            raise ValueError(f"env.{name} is unsupported; use env.priv_observe_friction for actor friction.")
    validate_reset_mixture(cfg.terrain)
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
    # A checkpoint snapshot restore (apply_config_snapshot) writes arm.action_mode
    # straight from the pickle, bypassing set_arm_action_mode -- so re-check here,
    # which every build and every load_env path runs through.
    validate_arm_action_mode(cfg)


def validate_reset_mixture(t):
    if t.reset_mode not in ("legacy", "fixed_mixture"):
        raise ValueError("terrain.reset_mode must be legacy or fixed_mixture")
    if not 0 <= t.reset_mix_hard_fraction <= 1:
        raise ValueError("reset_mix_hard_fraction must be in [0, 1]")
    for name in ("reset_mix_easy_tilt_rad", "reset_mix_hard_tilt_rad",
                 "reset_mix_yaw_rad", "reset_mix_z_m", "robustness_early_window_s"):
        value = getattr(t, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError("terrain.{} must be finite and nonnegative".format(name))
    if t.reset_mix_easy_tilt_rad > t.reset_mix_hard_tilt_rad or t.reset_mix_hard_tilt_rad > math.pi:
        raise ValueError("reset tilt ranges must satisfy 0 <= easy <= hard <= pi")
    for name in ("reset_mix_start_iteration", "reset_mix_ramp_iterations", "reset_mix_seed"):
        value = getattr(t, name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("terrain.{} must be a nonnegative integer".format(name))
    if t.reset_mix_ramp_iterations == 0 or t.robustness_early_window_s == 0:
        raise ValueError("reset ramp iterations and metrics early window must be positive")


def build_roboduet_config(args=None, *, options=None, debug=False):
    """Build one finalized RoboDuet config without mutating another build."""

    from .go1 import GO1_PROFILE
    from .wbc import WBC_REWARD_FACTORS, ROBODUET_PROFILE
    from .wtw import WTW_PROFILE

    if options is None:
        options = RoboDuetRuntimeOptions.from_args(args) if args is not None else RoboDuetRuntimeOptions(4096, "go2")
    cfg = build_config(GO1_PROFILE, WTW_PROFILE, ROBODUET_PROFILE)
    if options.domain_rand_mode is not None:
        cfg.domain_rand.mode = options.domain_rand_mode
    _derive_wbc_rewards(cfg, WBC_REWARD_FACTORS)
    derive_privileged_normalization_ranges(cfg)
    cfg.env.num_envs = options.num_envs
    cfg.env.stage1_arm_curriculum = options.stage1_arm_curriculum

    layout = RoboDuetLayout.from_cfg(cfg)
    if options.use_rot6d:
        enable_rot6d(cfg, layout)
    else:
        cfg.use_rot6d = False

    goal_reaching = options.goal_reaching or options.traj_tracking
    if options.dyna_gait or goal_reaching:
        enable_dyna_gait(cfg, layout)
    if goal_reaching:
        enable_goal_reaching(cfg, layout)
    if options.traj_tracking:
        enable_traj_tracking(cfg, layout)
    set_arm_action_mode(cfg, options.arm_action_mode)
    set_gait_reward_mode(cfg, options.gait_reward_mode)
    validate_gait_reward_mode(cfg)
    set_raibert_form(cfg, options.raibert_form)
    validate_raibert_form(cfg)
    if not options.reach_table:
        cfg.wbc.goal_reaching.reach_table_path = ""

    layout.finalize(cfg)
    configure_privileged_obs_dims(cfg)
    configure_robot_asset(cfg, options.robot)
    validate_roboduet_cfg(cfg)

    if debug:
        cfg.domain_rand.randomize_mount_position = False
        cfg.domain_rand.randomize_mount_rotation = False

    return cfg
