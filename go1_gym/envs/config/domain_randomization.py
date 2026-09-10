"""Resolve the DR mode before actors or observation buffers are created."""

from copy import deepcopy


DOMAIN_RAND_MODES = ("benchmark", "sim2real", "none")

# Hardware uncertainties absent from the current simulator benchmark protocol.
_BENCHMARK_DISABLED_FLAGS = (
    "randomize_mount_position", "randomize_mount_rotation",
    "randomize_gravity", "randomize_motor_offset",
    "randomize_action_delay", "randomize_dog_obs_latency",
    "randomize_lag_timesteps",
)


def domain_randomization_mode(cfg):
    mode = getattr(cfg.domain_rand, "mode", "sim2real")
    if mode not in DOMAIN_RAND_MODES:
        raise ValueError(f"domain_rand.mode must be one of {DOMAIN_RAND_MODES}, got {mode!r}")
    return mode


# Explicit scope: do not recursively disable arbitrary future randomizations.
# Reset/task sampling and arm movement also help the simulation task itself.
_DYNAMICS_FLAGS = (
    "randomize_rigids_after_start", "randomize_friction",
    "randomize_restitution", "randomize_base_mass", "randomize_com_displacement",
    "randomize_motor_strength", "randomize_motor_offset",
    "randomize_Kp_factor", "randomize_Kd_factor", "randomize_gravity",
    "randomize_mount_position", "randomize_mount_rotation",
    "randomize_end_effector_force", "push_robots", "randomize_action_delay",
    "randomize_dog_obs_latency",
    # Legacy config-only flags, currently without consumers in RoboDuet.
    "randomize_lag_timesteps", "randomize_friction_indep",
)
_ARM_FLAGS = (
    "randomize_Kp_factor", "randomize_Kd_factor", "randomize_motor_strength",
    "randomize_motor_offset", "randomize_link_mass", "randomize_link_com",
    "randomize_ee_payload",
)


def resolve_domain_randomization(cfg):
    """Resolve training mode without destroying the caller's individual recipe.

    Legacy checkpoint flags are migrated to modes by the snapshot loader.
    Modes apply before actor construction, never live. Ranges, normalization,
    task curricula and observation layouts are preserved.
    """
    if cfg is None:
        return None
    mode = domain_randomization_mode(cfg)
    if mode == "sim2real" or getattr(cfg, "_domain_rand_scenario_owned", False):
        return cfg
    effective = deepcopy(cfg)
    dr = effective.domain_rand
    flags = _DYNAMICS_FLAGS if mode == "none" else _BENCHMARK_DISABLED_FLAGS
    for name in flags:
        if hasattr(dr, name):
            setattr(dr, name, False)
    for stage in ("stage1_arm", "stage2_arm"):
        arm = getattr(dr, stage, None)
        for name in (_ARM_FLAGS if mode == "none" else ("randomize_motor_offset",)):
            if hasattr(arm, name):
                setattr(arm, name, False)
    dr.dog_obs_frame_drop_prob = 0.0
    dr.dog_obs_latency_jitter_steps = 0
    effective.noise.add_noise = False
    return effective


def configure_benchmark_domain_randomization(cfg, recipe):
    """Use one evaluator-owned recipe regardless of checkpoint training mode.

    Call after checkpoint restoration and before scenario overrides/actor
    creation. Subsequent scenario settings must survive the env's resolver.
    This standardizes config, not paired random draws across policy rows.
    """
    common = deepcopy(recipe)
    common.domain_rand.mode = "benchmark"
    common = resolve_domain_randomization(common)
    # Robot identity belongs to the selected asset, not the DR recipe.
    if hasattr(cfg.domain_rand, "mount_joint_name"):
        common.domain_rand.mount_joint_name = cfg.domain_rand.mount_joint_name
    cfg.domain_rand = common.domain_rand
    cfg.noise = common.noise
    cfg.noise_scales = deepcopy(common.noise_scales)
    cfg._domain_rand_scenario_owned = True
