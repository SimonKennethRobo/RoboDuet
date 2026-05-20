"""Task configuration helpers shared by training and benchmark entrypoints."""

from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.go1.wtw_config import config_wtw

from .asset_config import config_asset


def apply_base_task_config(cfg, args):
    config_go1(cfg)
    config_wtw(cfg)
    config_asset(cfg)

    cfg.commands.distributional_commands = False
    cfg.domain_rand.lag_timesteps = 6
    cfg.domain_rand.randomize_lag_timesteps = False
    cfg.control.control_type = "M"

    cfg.domain_rand.added_mass_range = [-2.0, 2.0]
    cfg.env.observe_two_prev_actions = False
    cfg.commands.body_roll_range = [-0.4, 0.4]
    cfg.commands.limit_body_roll = [-0.4, 0.4]
    cfg.commands.body_pitch_range = [-0.4, 0.4]
    cfg.commands.limit_body_pitch = [-0.4, 0.4]

    cfg.env.num_envs = args.num_envs
    cfg.env.keep_arm_fixed = True
    cfg.env.stage1_arm_curriculum = not getattr(args, "no_stage1_arm_curriculum", False)
    cfg.env.stage1_arm_fixed_fraction = getattr(args, "stage1_arm_fixed_fraction", 0.1)
    cfg.env.stage1_arm_max_accel = getattr(args, "stage1_arm_max_accel", 2.0)
    cfg.env.stage1_arm_max_vel = getattr(args, "stage1_arm_max_vel", 1.0)
    cfg.env.stage1_arm_max_offset = getattr(args, "stage1_arm_max_offset", 0.35)
    cfg.env.stage1_arm_accel_resample_time_s = getattr(args, "stage1_arm_accel_resample_time_s", 0.5)

    cfg.terrain.mesh_type = "plane"
    if cfg.terrain.mesh_type == "plane":
        cfg.terrain.teleport_robots = False

    cfg.control.update_obs_freq = 20
    cfg.env.num_actions = 18
    cfg.env.num_observations = 63
    cfg.env.num_obs_history = cfg.env.num_observation_history * cfg.env.num_observations

    cfg.hybrid.reward_scales.tracking_lin_vel = 0.7 * cfg.reward_scales.tracking_lin_vel
    cfg.hybrid.reward_scales.tracking_ang_vel = 0.5 * cfg.reward_scales.tracking_ang_vel
    cfg.hybrid.reward_scales.arm_energy = -0.00004
    cfg.reward_scales.loco_energy = -0.00004
    cfg.reward_scales.jump = -0.00
    cfg.rewards.terminal_body_height = 0.28
    cfg.rewards.use_terminal_body_height = True

    cfg.commands.T_force_range = [2, 4.0]
    cfg.domain_rand.randomize_end_effector_force = False
    cfg.commands.add_force_thres = 0.3
    cfg.domain_rand.max_force = 15
    cfg.domain_rand.max_force_offset = 0.01

    cfg.env.priv_observe_vel = False
    cfg.commands.global_reference = False
    cfg.env.priv_observe_high_freq_goal = False
    cfg.dog.dog_num_privileged_obs = 2
    cfg.arm.arm_num_privileged_obs = 9
    cfg.env.num_privileged_obs = 9

    cfg.asset.render_sphere = True
    cfg.hybrid.use_vision = False
    cfg.rewards.manip_weight_lpy = 3
    cfg.rewards.manip_weight_rpy = 1
    cfg.hybrid.reward_scales.arm_dof_vel = 10 * cfg.reward_scales.dof_vel
    cfg.hybrid.reward_scales.arm_dof_acc = 10 * cfg.reward_scales.dof_acc
    cfg.hybrid.reward_scales.arm_action_rate = 10 * cfg.reward_scales.action_rate
    cfg.hybrid.reward_scales.arm_action_smoothness_1 = 5 * cfg.reward_scales.action_smoothness_1
    cfg.hybrid.reward_scales.arm_action_smoothness_2 = 5 * cfg.reward_scales.action_smoothness_2


def apply_rot6d_config(cfg):
    cfg.use_rot6d = True
    cfg.env.num_observations += 3
    cfg.env.num_obs_history = cfg.env.num_observation_history * cfg.env.num_observations
    cfg.arm.arm_num_observations += 3
    cfg.arm.arm_num_obs_history = cfg.arm.arm_num_observations * cfg.arm.arm_num_observation_history
    cfg.arm.arm_num_commands += 3
    cfg.dog.dog_num_observations += 3
    cfg.dog.dog_num_obs_history = cfg.dog.dog_num_observations * cfg.dog.dog_num_observation_history


def apply_traj_track_config(cfg, traj_track_reward_scale=5.0):
    cfg.arm.trajectory.enabled = True
    traj_window_dims = len(cfg.arm.trajectory.window_offsets) * 9
    cfg.arm.num_actions_arm_cd = cfg.arm.num_actions_arm + 3
    cfg.arm.arm_num_observations = 12 + 1 + 4 + 3 + 3 + 9 + 6 + traj_window_dims + 1
    cfg.arm.arm_num_obs_history = cfg.arm.arm_num_observations * cfg.arm.arm_num_observation_history
    cfg.dog.dog_num_observations += 9
    cfg.dog.dog_num_obs_history = cfg.dog.dog_num_observations * cfg.dog.dog_num_observation_history
    cfg.env.num_observations += 1 + 4 + 9 + 6 + traj_window_dims + 1
    cfg.env.num_obs_history = cfg.env.num_observation_history * cfg.env.num_observations
    cfg.hybrid.reward_scales.arm_manip_commands_tracking_combine = 0.0
    cfg.hybrid.reward_scales.vis_manip_commands_tracking_lpy = 0.0
    cfg.hybrid.reward_scales.vis_manip_commands_tracking_rpy = 0.0
    cfg.hybrid.reward_scales.traj_track = traj_track_reward_scale
    cfg.hybrid.reward_scales.trajectory_current_tracking = 1.0
    cfg.hybrid.reward_scales.trajectory_completion_time = 0.5
    cfg.hybrid.reward_scales.arm_delta_vel_cmd = -0.05
    cfg.hybrid.reward_scales.ee_smoothness = -1e-4


def apply_dyna_gait_config(cfg, min_frequency=0.0):
    cfg.commands.use_dynamic_gait = True
    cfg.commands.gait_frequency_cmd_range = [
        min_frequency,
        cfg.commands.gait_frequency_cmd_range[1],
    ]
    cfg.commands.limit_gait_frequency = cfg.commands.gait_frequency_cmd_range
    cfg.commands.limit_footswing_height = cfg.commands.footswing_height_range
    cfg.commands.limit_gait_duration = cfg.commands.gait_duration_cmd_range
    cfg.commands.limit_stance_width = cfg.commands.stance_width_range
    cfg.commands.limit_stance_length = cfg.commands.stance_length_range

    num_new_gait_dims = 5
    cfg.dog.dog_num_commands += num_new_gait_dims
    cfg.dog.dog_num_observations += num_new_gait_dims
    cfg.dog.dog_num_obs_history = cfg.dog.dog_num_observations * cfg.dog.dog_num_observation_history

    plan_action_dims = 7 + (3 if cfg.arm.trajectory.enabled else 0)
    cfg.arm.num_actions_arm_cd = cfg.arm.num_actions_arm + plan_action_dims
    cfg.arm.arm_num_observations += num_new_gait_dims
    cfg.arm.arm_num_obs_history = cfg.arm.arm_num_observations * cfg.arm.arm_num_observation_history

    cfg.env.num_observations += num_new_gait_dims + 2
    cfg.env.num_obs_history = cfg.env.num_observation_history * cfg.env.num_observations
    cfg.env.observe_gait_commands = True

    cfg.commands.num_bins_gait_frequency = 11
    cfg.commands.num_bins_footswing_height = 5
    cfg.commands.num_bins_gait_duration = 3
    cfg.commands.num_bins_stance_width = 3
    cfg.commands.num_bins_stance_length = 3


def apply_robot_asset(cfg, robot):
    if robot == "go1":
        cfg.asset.file = "{MINI_GYM_ROOT_DIR}/resources/robots/arx5p2Go1/urdf/arx5p2Go1.urdf"
    elif robot == "go2":
        cfg.asset.file = "{MINI_GYM_ROOT_DIR}/resources/robots/go2/urdf/arx5go2.urdf"


def configure_task_from_args(cfg, args, traj_track_reward_scale=5.0):
    apply_base_task_config(cfg, args)
    cfg.use_rot6d = getattr(args, "use_rot6d", False)
    if cfg.use_rot6d:
        apply_rot6d_config(cfg)

    if getattr(args, "traj_track", False):
        apply_traj_track_config(cfg, traj_track_reward_scale=traj_track_reward_scale)
    else:
        cfg.arm.trajectory.enabled = False

    if getattr(args, "dyna_gait", False):
        apply_dyna_gait_config(cfg, min_frequency=getattr(args, "dyna_gait_min_frequency", 0.0))
    else:
        cfg.commands.use_dynamic_gait = False

    apply_robot_asset(cfg, args.robot)
