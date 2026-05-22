"""RoboDuet WBC task configuration.

This module is the source of truth for RoboDuet-specific defaults. The
`legged_robot_config.py` module only provides the base Cfg schema.
"""

from dataclasses import dataclass, field
import math

from params_proto import PrefixProto

from go1_gym.envs.go1.go1_config import config_robot
from go1_gym.envs.go1.wtw_config import config_wtw

from .asset_config import config_asset
from .legged_robot_config import LeggedRobotCfg


@dataclass(frozen=True)
class HybridRewardTerminationConfig:
    terminal_body_height: float = 0.17
    use_terminal_body_height: bool = True
    use_terminal_roll: bool = False
    use_terminal_pitch: bool = False
    terminal_body_roll: float = 0.10
    terminal_body_pitch: float = 0.2
    terminal_body_pitch_roll: float = 80.0 / 180.0 * math.pi
    headupdown_thres: float = 0.1


@dataclass(frozen=True)
class HybridRewardScaleConfig:
    jump: float = 5
    arm_manip_commands_tracking_combine: float = 1.0
    vis_manip_commands_tracking_lpy: float = 1.0
    vis_manip_commands_tracking_rpy: float = 1.0
    orientation_heuristic: float = -2.0
    orientation_control: float = -10.0
    hip_action_l2: float = -0.05
    raibert_heuristic: float = -0.0
    arm_control_smoothness_1: float = -0.1
    arm_control_limits: float = -5.0
    traj_track: float = 0.0
    trajectory_current_tracking: float = 0.0
    trajectory_completion_time: float = 0.0
    arm_delta_vel_cmd: float = 0.0
    ee_smoothness: float = 0.0
    arm_contact: float = -1.0


@dataclass(frozen=True)
class HybridConfig:
    num_actions: int = 18
    plan_vel: bool = False
    use_vision: bool = False
    rewards: HybridRewardTerminationConfig = HybridRewardTerminationConfig()
    reward_scales: HybridRewardScaleConfig = HybridRewardScaleConfig()


@dataclass(frozen=True)
class ArmCommandConfig:
    l: tuple = (0.3, 0.77)
    p: tuple = (-math.pi * 0.45, math.pi * 0.45)
    y: tuple = (-math.pi / 2.0, math.pi / 2.0)
    roll_ee: tuple = (-math.pi * 0.45, math.pi * 0.45)
    pitch_ee: tuple = (-math.radians(60.0), math.radians(60.0))
    yaw_ee: tuple = (-math.radians(75.0), math.radians(75.0))
    T_traj: tuple = (2.0, 3.0)
    T_force_range: tuple = (1.0, 4.0)
    add_force_thres: float = 0.3


@dataclass(frozen=True)
class ArmTrajectoryConfig:
    enabled: bool = False
    traj_type: list = field(default_factory=lambda: ["line", "s_curve"])
    window_offsets: tuple = (0, 1, 2, 4, 8, 16, 32, 64)
    num_waypoints: int = 96
    start_radius: float = 0.0
    length_range: tuple = (0.10, 0.45)
    s_curve_amplitude_range: tuple = (0.02, 0.12)
    s_curve_frequency: float = 1.0
    circle_radius: float = 0.5
    circle_turns: float = 1.0
    completion_time_range: tuple = (2.0, 5.0)
    completion_pos_threshold: float = 0.05
    completion_rot_threshold: float = 0.25
    curriculum_levels: int = 6
    curriculum_success_threshold: float = 0.6
    delta_vel_limit: tuple = (0.4, 0.25, 0.6)
    user_cmd_mode: str = "zero"
    user_lin_vel_x: tuple = (-0.3, 0.3)
    user_lin_vel_y: tuple = (-0.2, 0.2)
    user_ang_vel_yaw: tuple = (-0.4, 0.4)
    pos_error_scale: float = 4.0
    rot_error_scale: float = 1.0
    completion_time_sigma: float = 0.35


@dataclass(frozen=True)
class ArmConfig:
    num_actions_arm: int = 6
    arm_num_privileged_obs: int = 50
    arm_num_observation_history: int = 30
    arm_num_observations: int = 20
    arm_num_commands: int = 6
    num_actions_arm_cd: int = 8
    use_adaptation_module: bool = False
    commands: ArmCommandConfig = ArmCommandConfig()
    trajectory: ArmTrajectoryConfig = ArmTrajectoryConfig()
    obs_scales: tuple = (("l", 1.0), ("p", 1.0), ("y", 1.0), ("wx", 1.0), ("wy", 1.0), ("wz", 1.0))
    stiffness_arm: dict = None
    damping_arm: dict = None


@dataclass(frozen=True)
class DogConfig:
    num_actions_loco: int = 12
    dog_num_privileged_obs: int = 66
    dog_num_observation_history: int = 30
    dog_num_observations: int = 68  # +12 vs old 56: arm joint pos (6) + vel (6)
    dog_num_commands: int = 5
    dog_actions: int = 12
    use_adaptation_module: bool = False
    stiffness_leg: dict = None
    damping_leg: dict = None


@dataclass(frozen=True)
class EnvConfig:
    keep_arm_fixed: bool = True
    num_actions: int = 18
    num_observations: int = 63
    num_privileged_obs: int = 66
    dog_num_privileged_obs: int = 66
    arm_num_privileged_obs: int = 50
    priv_observe_arm_mount_tf: bool = True
    priv_observe_friction: bool = True
    priv_observe_ground_friction: bool = False
    priv_observe_restitution: bool = True
    priv_observe_base_mass: bool = True
    priv_observe_com_displacement: bool = True
    priv_observe_motor_strength: bool = False
    priv_observe_motor_offset: bool = False
    priv_observe_joint_friction: bool = True
    priv_observe_Kp_factor: bool = True
    priv_observe_Kd_factor: bool = True
    priv_observe_dof_damping: bool = True
    priv_observe_body_height: bool = False
    priv_observe_gravity: bool = False
    priv_observe_body_velocity: bool = False
    priv_observe_clock_inputs: bool = False
    priv_observe_desired_contact_states: bool = False
    priv_observe_vel: bool = True
    priv_observe_high_freq_goal: bool = False
    observe_two_prev_actions: bool = False
    record_video: bool = False
    recording_width_px: int = 500
    recording_height_px: int = 320
    recording_mode: str = "COLOR"
    num_recording_envs: int = 1
    recording_frame_stride: int = 1
    recording_overlay_text: bool = True
    recording_overlay_trajectory: bool = True
    debug_viz: bool = False
    all_agents_share: bool = False


@dataclass(frozen=True)
class CommandConfig:
    global_reference: bool = False
    distributional_commands: bool = False
    body_roll_range: tuple = (-0.4, 0.4)
    limit_body_roll: tuple = (-0.4, 0.4)
    body_pitch_range: tuple = (-0.4, 0.4)
    limit_body_pitch: tuple = (-0.4, 0.4)
    T_force_range: tuple = (2.0, 4.0)
    add_force_thres: float = 0.3


@dataclass(frozen=True)
class ControlConfig:
    update_obs_freq: int = 20
    control_type: str = "M"


@dataclass(frozen=True)
class TerrainConfig:
    mesh_type: str = "plane"


@dataclass(frozen=True)
class ArmDomainRandConfig:
    """Per-stage arm domain randomization (applied every episode reset)."""

    randomize_Kp_factor: bool = True
    Kp_factor_range: tuple = (0.9, 1.1)
    randomize_Kd_factor: bool = True
    Kd_factor_range: tuple = (0.9, 1.1)
    randomize_motor_strength: bool = True
    motor_strength_range: tuple = (0.85, 1.15)
    randomize_motor_offset: bool = True
    motor_offset_range: float = 0.025
    randomize_link_mass: bool = True
    link_mass_range: tuple = (0.85, 1.15)
    randomize_link_com: bool = True
    link_com_range: float = 0.01


@dataclass(frozen=True)
class DomainRandConfig:
    lag_timesteps: int = 6
    randomize_lag_timesteps: bool = False
    randomize_action_delay: bool = True
    added_mass_range: tuple = (-2.0, 2.0)
    randomize_end_effector_force: bool = False
    max_force: float = 15
    max_force_offset: float = 0.01
    randomize_mount_pos: bool = True
    mount_pos_range: tuple = ((-0.05, 0.05), (-0.02, 0.02), (-0.05, 0.05))
    mount_pos_buckets: int = 16
    mount_pos_bucket_seed: int = 1234
    mount_joint_name: str = "zarx5p2_mount"

    stage1_arm: ArmDomainRandConfig = ArmDomainRandConfig(
        Kp_factor_range=(0.5, 1.5),
        Kd_factor_range=(0.2, 2.0),
        motor_strength_range=(0.7, 1.3),
        motor_offset_range=0.05,
        link_mass_range=(0.1, 2),
        link_com_range=0.1,
    )
    stage2_arm: ArmDomainRandConfig = ArmDomainRandConfig()


@dataclass(frozen=True)
class RewardConfig:
    terminal_body_height: float = 0.17
    use_terminal_body_height: bool = True
    manip_weight_lpy: float = 3
    manip_weight_rpy: float = 1


@dataclass(frozen=True)
class BaseRewardScaleConfig:
    jump: float = 10
    loco_energy: float = -0.00004


@dataclass(frozen=True)
class HybridRewardScaleOverrideConfig:
    tracking_lin_vel_multiplier: float = 0.7
    tracking_ang_vel_multiplier: float = 0.5
    arm_energy: float = -0.00004
    arm_dof_vel_multiplier: float = 10.0
    arm_dof_acc_multiplier: float = 10.0
    arm_action_rate_multiplier: float = 10.0
    arm_action_smoothness_1_multiplier: float = 5.0
    arm_action_smoothness_2_multiplier: float = 5.0


@dataclass(frozen=True)
class Stage1ArmDisturbanceConfig:
    fixed_fraction: float = 0.1
    saturation_fraction: float = 0.8
    accel_resample_time_s: float = 0.01  # 100 Hz, larger than actual ctrl freq
    zero_accel_probability: float = 0.3
    zero_vel_probability: float = 0.1
    max_accel: float = 10.0
    max_vel: float = 5.0
    max_offset: float = 999
    init_dof_pos_noise: float = 1  # ±rad additive noise on arm joints at reset


@dataclass(frozen=True)
class DynaGaitFeatureConfig:
    num_gait_dims: int = 5
    plan_action_dims: int = 7
    num_bins_gait_frequency: int = 11
    num_bins_footswing_height: int = 5
    num_bins_gait_duration: int = 3
    num_bins_stance_width: int = 3
    num_bins_stance_length: int = 3


@dataclass(frozen=True)
class TrajTrackFeatureConfig:
    dog_obs_dims: int = 9
    arm_delta_vel_action_dims: int = 3
    env_extra_obs_without_window: int = 1 + 4 + 9 + 6 + 1
    arm_obs_without_window: int = 12 + 1 + 4 + 3 + 3 + 9 + 6 + 1
    trajectory_current_tracking_scale: float = 1.0
    trajectory_completion_time_scale: float = 0.5
    arm_delta_vel_cmd_scale: float = -0.05
    ee_smoothness_scale: float = -1e-4


@dataclass(frozen=True)
class AssetConfig:
    render_sphere: bool = True
    go1_file: str = "{MINI_GYM_ROOT_DIR}/resources/robots/arx5p2Go1/urdf/arx5p2Go1.urdf"
    go2_file: str = "{MINI_GYM_ROOT_DIR}/resources/robots/go2/urdf/arx5go2.urdf"


@dataclass(frozen=True)
class RoboDuetDefaults:
    hybrid: HybridConfig = HybridConfig()
    arm: ArmConfig = ArmConfig(
        stiffness_arm={
            "zarx": 50.0,
            "zarx_j1": 40.0,
            "zarx_j2": 70.0,
            "zarx_j3": 70.0,
            "zarx_j4": 25.0,
            "zarx_j5": 25.0,
            "zarx_j6": 25.0,
            "zarx_j7": 50.0,
            "zarx_j8": 50.0,
        },
        damping_arm={
            "zarx": 20.0,
            "zarx_j1": 3.0,
            "zarx_j2": 15.0,
            "zarx_j3": 15.0,
            "zarx_j4": 2.0,
            "zarx_j5": 2.0,
            "zarx_j6": 2.0,
            "zarx_j7": 20.0,
            "zarx_j8": 20.0,
        },
    )
    dog: DogConfig = DogConfig(
        stiffness_leg={"joint": 35.0},
        damping_leg={"joint": 1.0},
    )
    env: EnvConfig = EnvConfig()
    commands: CommandConfig = CommandConfig()
    control: ControlConfig = ControlConfig()
    terrain: TerrainConfig = TerrainConfig()
    domain_rand: DomainRandConfig = DomainRandConfig()
    rewards: RewardConfig = RewardConfig()
    reward_scales: BaseRewardScaleConfig = BaseRewardScaleConfig()
    hybrid_reward_overrides: HybridRewardScaleOverrideConfig = HybridRewardScaleOverrideConfig()
    stage1_arm_disturbance: Stage1ArmDisturbanceConfig = Stage1ArmDisturbanceConfig()
    dyna_gait: DynaGaitFeatureConfig = DynaGaitFeatureConfig()
    traj_track: TrajTrackFeatureConfig = TrajTrackFeatureConfig()
    asset: AssetConfig = AssetConfig()


@dataclass(frozen=True)
class RoboDuetRuntimeOptions:
    num_envs: int
    robot: str
    use_rot6d: bool = False
    traj_track: bool = False
    dyna_gait: bool = False
    dyna_gait_min_frequency: float = 0.0
    stage1_arm_curriculum: bool = True

    @classmethod
    def from_args(cls, args):
        return cls(
            num_envs=args.num_envs,
            robot=args.robot,
            use_rot6d=getattr(args, "use_rot6d", False),
            traj_track=getattr(args, "traj_track", False),
            dyna_gait=getattr(args, "dyna_gait", False),
            dyna_gait_min_frequency=getattr(args, "dyna_gait_min_frequency", 0.0),
            stage1_arm_curriculum=not getattr(args, "no_stage1_arm_curriculum", False),
        )


@dataclass
class RoboDuetLayout:
    env_obs: int
    arm_obs: int
    dog_obs: int
    arm_cmd: int
    dog_cmd: int
    arm_action_cd: int

    @classmethod
    def from_cfg(cls, cfg):
        return cls(
            env_obs=cfg.env.num_observations,
            arm_obs=cfg.arm.arm_num_observations,
            dog_obs=cfg.dog.dog_num_observations,
            arm_cmd=cfg.arm.arm_num_commands,
            dog_cmd=cfg.dog.dog_num_commands,
            arm_action_cd=cfg.arm.num_actions_arm_cd,
        )

    def finalize(self, cfg):
        cfg.env.num_observations = self.env_obs
        cfg.env.num_obs_history = cfg.env.num_observation_history * self.env_obs
        cfg.arm.arm_num_observations = self.arm_obs
        cfg.arm.arm_num_obs_history = cfg.arm.arm_num_observation_history * self.arm_obs
        cfg.arm.arm_num_commands = self.arm_cmd
        cfg.arm.num_actions_arm_cd = self.arm_action_cd
        cfg.dog.dog_num_observations = self.dog_obs
        cfg.dog.dog_num_obs_history = cfg.dog.dog_num_observation_history * self.dog_obs
        cfg.dog.dog_num_commands = self.dog_cmd


ROBODUET_DEFAULTS = RoboDuetDefaults()


class RoboDuetCfg(LeggedRobotCfg):
    use_rot6d = False

    class hybrid(PrefixProto, cli=False):
        num_actions = ROBODUET_DEFAULTS.hybrid.num_actions
        plan_vel = ROBODUET_DEFAULTS.hybrid.plan_vel
        use_vision = ROBODUET_DEFAULTS.hybrid.use_vision

        class rewards(PrefixProto, cli=False):
            terminal_body_height = ROBODUET_DEFAULTS.hybrid.rewards.terminal_body_height
            use_terminal_body_height = ROBODUET_DEFAULTS.hybrid.rewards.use_terminal_body_height
            use_terminal_roll = ROBODUET_DEFAULTS.hybrid.rewards.use_terminal_roll
            use_terminal_pitch = ROBODUET_DEFAULTS.hybrid.rewards.use_terminal_pitch
            terminal_body_roll = ROBODUET_DEFAULTS.hybrid.rewards.terminal_body_roll
            terminal_body_pitch = ROBODUET_DEFAULTS.hybrid.rewards.terminal_body_pitch
            terminal_body_pitch_roll = ROBODUET_DEFAULTS.hybrid.rewards.terminal_body_pitch_roll
            headupdown_thres = ROBODUET_DEFAULTS.hybrid.rewards.headupdown_thres

        class reward_scales(PrefixProto, cli=False):
            jump = ROBODUET_DEFAULTS.hybrid.reward_scales.jump
            arm_manip_commands_tracking_combine = (
                ROBODUET_DEFAULTS.hybrid.reward_scales.arm_manip_commands_tracking_combine
            )
            vis_manip_commands_tracking_lpy = ROBODUET_DEFAULTS.hybrid.reward_scales.vis_manip_commands_tracking_lpy
            vis_manip_commands_tracking_rpy = ROBODUET_DEFAULTS.hybrid.reward_scales.vis_manip_commands_tracking_rpy
            orientation_heuristic = ROBODUET_DEFAULTS.hybrid.reward_scales.orientation_heuristic
            orientation_control = ROBODUET_DEFAULTS.hybrid.reward_scales.orientation_control
            hip_action_l2 = ROBODUET_DEFAULTS.hybrid.reward_scales.hip_action_l2
            raibert_heuristic = ROBODUET_DEFAULTS.hybrid.reward_scales.raibert_heuristic
            arm_control_smoothness_1 = ROBODUET_DEFAULTS.hybrid.reward_scales.arm_control_smoothness_1
            arm_control_limits = ROBODUET_DEFAULTS.hybrid.reward_scales.arm_control_limits
            traj_track = ROBODUET_DEFAULTS.hybrid.reward_scales.traj_track
            trajectory_current_tracking = ROBODUET_DEFAULTS.hybrid.reward_scales.trajectory_current_tracking
            trajectory_completion_time = ROBODUET_DEFAULTS.hybrid.reward_scales.trajectory_completion_time
            arm_delta_vel_cmd = ROBODUET_DEFAULTS.hybrid.reward_scales.arm_delta_vel_cmd
            ee_smoothness = ROBODUET_DEFAULTS.hybrid.reward_scales.ee_smoothness
            arm_contact = ROBODUET_DEFAULTS.hybrid.reward_scales.arm_contact

    class arm(PrefixProto, cli=False):
        num_actions_arm = ROBODUET_DEFAULTS.arm.num_actions_arm
        arm_num_privileged_obs = ROBODUET_DEFAULTS.arm.arm_num_privileged_obs
        arm_num_observation_history = ROBODUET_DEFAULTS.arm.arm_num_observation_history
        arm_num_observations = ROBODUET_DEFAULTS.arm.arm_num_observations
        arm_num_obs_history = arm_num_observations * arm_num_observation_history
        arm_num_commands = ROBODUET_DEFAULTS.arm.arm_num_commands
        num_actions_arm_cd = ROBODUET_DEFAULTS.arm.num_actions_arm_cd

        class commands(PrefixProto, cli=False):
            l = list(ROBODUET_DEFAULTS.arm.commands.l)
            p = list(ROBODUET_DEFAULTS.arm.commands.p)
            y = list(ROBODUET_DEFAULTS.arm.commands.y)
            roll_ee = list(ROBODUET_DEFAULTS.arm.commands.roll_ee)
            pitch_ee = list(ROBODUET_DEFAULTS.arm.commands.pitch_ee)
            yaw_ee = list(ROBODUET_DEFAULTS.arm.commands.yaw_ee)
            T_traj = list(ROBODUET_DEFAULTS.arm.commands.T_traj)
            T_force_range = list(ROBODUET_DEFAULTS.arm.commands.T_force_range)
            add_force_thres = ROBODUET_DEFAULTS.arm.commands.add_force_thres

        class trajectory(PrefixProto, cli=False):
            enabled = ROBODUET_DEFAULTS.arm.trajectory.enabled
            traj_type = list(ROBODUET_DEFAULTS.arm.trajectory.traj_type)
            window_offsets = list(ROBODUET_DEFAULTS.arm.trajectory.window_offsets)
            num_waypoints = ROBODUET_DEFAULTS.arm.trajectory.num_waypoints
            start_radius = ROBODUET_DEFAULTS.arm.trajectory.start_radius
            length_range = list(ROBODUET_DEFAULTS.arm.trajectory.length_range)
            s_curve_amplitude_range = list(ROBODUET_DEFAULTS.arm.trajectory.s_curve_amplitude_range)
            s_curve_frequency = ROBODUET_DEFAULTS.arm.trajectory.s_curve_frequency
            circle_radius = ROBODUET_DEFAULTS.arm.trajectory.circle_radius
            circle_turns = ROBODUET_DEFAULTS.arm.trajectory.circle_turns
            completion_time_range = list(ROBODUET_DEFAULTS.arm.trajectory.completion_time_range)
            completion_pos_threshold = ROBODUET_DEFAULTS.arm.trajectory.completion_pos_threshold
            completion_rot_threshold = ROBODUET_DEFAULTS.arm.trajectory.completion_rot_threshold
            delta_vel_limit = list(ROBODUET_DEFAULTS.arm.trajectory.delta_vel_limit)
            user_cmd_mode = ROBODUET_DEFAULTS.arm.trajectory.user_cmd_mode
            user_lin_vel_x = list(ROBODUET_DEFAULTS.arm.trajectory.user_lin_vel_x)
            user_lin_vel_y = list(ROBODUET_DEFAULTS.arm.trajectory.user_lin_vel_y)
            user_ang_vel_yaw = list(ROBODUET_DEFAULTS.arm.trajectory.user_ang_vel_yaw)
            pos_error_scale = ROBODUET_DEFAULTS.arm.trajectory.pos_error_scale
            rot_error_scale = ROBODUET_DEFAULTS.arm.trajectory.rot_error_scale
            completion_time_sigma = ROBODUET_DEFAULTS.arm.trajectory.completion_time_sigma
            curriculum_levels = ROBODUET_DEFAULTS.arm.trajectory.curriculum_levels
            curriculum_success_threshold = ROBODUET_DEFAULTS.arm.trajectory.curriculum_success_threshold

        class obs_scales(PrefixProto, cli=False):
            l = 1.0
            p = 1.0
            y = 1.0
            wx = 1.0
            wy = 1.0
            wz = 1.0

        class control(PrefixProto, cli=False):
            stiffness_arm = dict(ROBODUET_DEFAULTS.arm.stiffness_arm)
            damping_arm = dict(ROBODUET_DEFAULTS.arm.damping_arm)

    class dog(PrefixProto, cli=False):
        num_actions_loco = ROBODUET_DEFAULTS.dog.num_actions_loco
        dog_num_privileged_obs = ROBODUET_DEFAULTS.dog.dog_num_privileged_obs
        dog_num_observation_history = ROBODUET_DEFAULTS.dog.dog_num_observation_history
        dog_num_observations = ROBODUET_DEFAULTS.dog.dog_num_observations
        dog_num_obs_history = dog_num_observations * dog_num_observation_history
        dog_num_commands = ROBODUET_DEFAULTS.dog.dog_num_commands
        dog_actions = ROBODUET_DEFAULTS.dog.dog_actions

        class control(PrefixProto, cli=False):
            stiffness_leg = dict(ROBODUET_DEFAULTS.dog.stiffness_leg)
            damping_leg = dict(ROBODUET_DEFAULTS.dog.damping_leg)

    class env(LeggedRobotCfg.env):
        num_observations = ROBODUET_DEFAULTS.env.num_observations
        num_privileged_obs = ROBODUET_DEFAULTS.env.num_privileged_obs
        num_actions = ROBODUET_DEFAULTS.env.num_actions
        keep_arm_fixed = ROBODUET_DEFAULTS.env.keep_arm_fixed
        dog_num_privileged_obs = ROBODUET_DEFAULTS.env.dog_num_privileged_obs
        arm_num_privileged_obs = ROBODUET_DEFAULTS.env.arm_num_privileged_obs
        priv_observe_vel = ROBODUET_DEFAULTS.env.priv_observe_vel
        priv_observe_high_freq_goal = ROBODUET_DEFAULTS.env.priv_observe_high_freq_goal
        observe_two_prev_actions = ROBODUET_DEFAULTS.env.observe_two_prev_actions
        record_video = ROBODUET_DEFAULTS.env.record_video
        recording_width_px = ROBODUET_DEFAULTS.env.recording_width_px
        recording_height_px = ROBODUET_DEFAULTS.env.recording_height_px
        recording_mode = ROBODUET_DEFAULTS.env.recording_mode
        num_recording_envs = ROBODUET_DEFAULTS.env.num_recording_envs
        recording_frame_stride = ROBODUET_DEFAULTS.env.recording_frame_stride
        recording_overlay_text = ROBODUET_DEFAULTS.env.recording_overlay_text
        recording_overlay_trajectory = ROBODUET_DEFAULTS.env.recording_overlay_trajectory
        debug_viz = ROBODUET_DEFAULTS.env.debug_viz
        all_agents_share = ROBODUET_DEFAULTS.env.all_agents_share
        stage1_arm_curriculum = True
        stage1_arm_fixed_fraction = ROBODUET_DEFAULTS.stage1_arm_disturbance.fixed_fraction
        stage1_arm_saturation_fraction = ROBODUET_DEFAULTS.stage1_arm_disturbance.saturation_fraction
        stage1_arm_init_dof_pos_noise = ROBODUET_DEFAULTS.stage1_arm_disturbance.init_dof_pos_noise
        stage1_arm_accel_resample_time_s = ROBODUET_DEFAULTS.stage1_arm_disturbance.accel_resample_time_s
        stage1_arm_zero_accel_probability = ROBODUET_DEFAULTS.stage1_arm_disturbance.zero_accel_probability
        stage1_arm_zero_vel_probability = ROBODUET_DEFAULTS.stage1_arm_disturbance.zero_vel_probability
        stage1_arm_max_accel = ROBODUET_DEFAULTS.stage1_arm_disturbance.max_accel
        stage1_arm_max_vel = ROBODUET_DEFAULTS.stage1_arm_disturbance.max_vel
        stage1_arm_max_offset = ROBODUET_DEFAULTS.stage1_arm_disturbance.max_offset


Cfg = RoboDuetCfg


def configure_external_recipes(cfg):
    config_robot(cfg)
    config_wtw(cfg)
    config_asset(cfg)


def _materialize_value(value):
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return dict(value)
    return value


def _copy_dataclass_attrs(target, source, names=None):
    if names is None:
        names = source.__dataclass_fields__.keys()
    for name in names:
        setattr(target, name, _materialize_value(getattr(source, name)))


def materialize_base_cfg(cfg, defaults, options):
    _copy_dataclass_attrs(cfg.hybrid, defaults.hybrid, names=("num_actions", "plan_vel", "use_vision"))
    _copy_dataclass_attrs(cfg.hybrid.rewards, defaults.hybrid.rewards)
    _copy_dataclass_attrs(cfg.hybrid.reward_scales, defaults.hybrid.reward_scales)

    _copy_dataclass_attrs(
        cfg.arm,
        defaults.arm,
        names=(
            "num_actions_arm",
            "arm_num_privileged_obs",
            "arm_num_observation_history",
            "arm_num_observations",
            "arm_num_commands",
            "num_actions_arm_cd",
            "use_adaptation_module",
        ),
    )
    _copy_dataclass_attrs(cfg.arm.commands, defaults.arm.commands)
    _copy_dataclass_attrs(cfg.arm.trajectory, defaults.arm.trajectory)
    for name, value in defaults.arm.obs_scales:
        setattr(cfg.arm.obs_scales, name, value)
    cfg.arm.control.stiffness_arm = dict(defaults.arm.stiffness_arm)
    cfg.arm.control.damping_arm = dict(defaults.arm.damping_arm)

    _copy_dataclass_attrs(
        cfg.dog,
        defaults.dog,
        names=(
            "num_actions_loco",
            "dog_num_privileged_obs",
            "dog_num_observation_history",
            "dog_num_observations",
            "dog_num_commands",
            "dog_actions",
            "use_adaptation_module",
        ),
    )
    cfg.dog.control.stiffness_leg = dict(defaults.dog.stiffness_leg)
    cfg.dog.control.damping_leg = dict(defaults.dog.damping_leg)

    cfg.env.num_envs = options.num_envs
    _copy_dataclass_attrs(cfg.env, defaults.env)
    cfg.env.stage1_arm_curriculum = options.stage1_arm_curriculum
    cfg.env.stage1_arm_fixed_fraction = defaults.stage1_arm_disturbance.fixed_fraction
    cfg.env.stage1_arm_saturation_fraction = defaults.stage1_arm_disturbance.saturation_fraction
    cfg.env.stage1_arm_init_dof_pos_noise = defaults.stage1_arm_disturbance.init_dof_pos_noise
    cfg.env.stage1_arm_accel_resample_time_s = defaults.stage1_arm_disturbance.accel_resample_time_s
    cfg.env.stage1_arm_zero_accel_probability = defaults.stage1_arm_disturbance.zero_accel_probability
    cfg.env.stage1_arm_zero_vel_probability = defaults.stage1_arm_disturbance.zero_vel_probability
    cfg.env.stage1_arm_max_accel = defaults.stage1_arm_disturbance.max_accel
    cfg.env.stage1_arm_max_vel = defaults.stage1_arm_disturbance.max_vel
    cfg.env.stage1_arm_max_offset = defaults.stage1_arm_disturbance.max_offset

    _copy_dataclass_attrs(cfg.commands, defaults.commands)
    _copy_dataclass_attrs(cfg.control, defaults.control)
    _copy_dataclass_attrs(cfg.domain_rand, defaults.domain_rand)
    _copy_dataclass_attrs(cfg.rewards, defaults.rewards)
    _copy_dataclass_attrs(cfg.reward_scales, defaults.reward_scales)
    cfg.normalization.Kp_factor_range = [0.5, 1.5]
    cfg.normalization.Kd_factor_range = [0.2, 2.0]
    cfg.normalization.dof_damping_range = [0.0, 10.0]

    cfg.terrain.mesh_type = defaults.terrain.mesh_type
    if cfg.terrain.mesh_type == "plane":
        cfg.terrain.teleport_robots = False

    cfg.asset.render_sphere = defaults.asset.render_sphere

    rewards = defaults.hybrid_reward_overrides
    cfg.hybrid.reward_scales.tracking_lin_vel = rewards.tracking_lin_vel_multiplier * cfg.reward_scales.tracking_lin_vel
    cfg.hybrid.reward_scales.tracking_ang_vel = rewards.tracking_ang_vel_multiplier * cfg.reward_scales.tracking_ang_vel
    cfg.hybrid.reward_scales.arm_energy = rewards.arm_energy
    cfg.hybrid.reward_scales.arm_dof_vel = rewards.arm_dof_vel_multiplier * cfg.reward_scales.dof_vel
    cfg.hybrid.reward_scales.arm_dof_acc = rewards.arm_dof_acc_multiplier * cfg.reward_scales.dof_acc
    cfg.hybrid.reward_scales.arm_action_rate = rewards.arm_action_rate_multiplier * cfg.reward_scales.action_rate
    cfg.hybrid.reward_scales.arm_action_smoothness_1 = (
        rewards.arm_action_smoothness_1_multiplier * cfg.reward_scales.action_smoothness_1
    )
    cfg.hybrid.reward_scales.arm_action_smoothness_2 = (
        rewards.arm_action_smoothness_2_multiplier * cfg.reward_scales.action_smoothness_2
    )


def _privileged_obs_dim(cfg, dof_dim):
    dim = 0
    if cfg.env.priv_observe_friction:
        dim += 1
    if cfg.env.priv_observe_ground_friction:
        dim += 1
    if cfg.env.priv_observe_restitution:
        dim += 1
    if cfg.env.priv_observe_base_mass:
        dim += 1
    if cfg.env.priv_observe_com_displacement:
        dim += 3
    if cfg.env.priv_observe_motor_strength:
        dim += dof_dim
    if cfg.env.priv_observe_motor_offset:
        dim += dof_dim
    if cfg.env.priv_observe_Kp_factor:
        dim += dof_dim
    if cfg.env.priv_observe_Kd_factor:
        dim += dof_dim
    if cfg.env.priv_observe_joint_friction:
        dim += dof_dim
    if getattr(cfg.env, "priv_observe_dof_damping", False):
        dim += dof_dim
    if cfg.env.priv_observe_body_height:
        dim += 1
    if cfg.env.priv_observe_gravity:
        dim += 3
    if cfg.env.priv_observe_body_velocity or cfg.env.priv_observe_vel:
        dim += 6
    if cfg.env.priv_observe_clock_inputs:
        dim += 4
    if cfg.env.priv_observe_desired_contact_states:
        dim += 4
    if cfg.env.priv_observe_high_freq_goal:
        dim += 6
    if getattr(cfg.env, "priv_observe_arm_mount_tf", False):
        dim += 6

    return dim


def configure_privileged_obs_dims(cfg):
    dog_dim = _privileged_obs_dim(cfg, cfg.dog.num_actions_loco)
    arm_dim = _privileged_obs_dim(cfg, ROBODUET_DEFAULTS.arm.num_actions_arm_cd)
    if cfg.arm.trajectory.enabled:
        arm_dim += cfg.arm.trajectory.num_waypoints * 9

    cfg.env.num_privileged_obs = dog_dim
    cfg.env.arm_num_privileged_obs = arm_dim
    cfg.env.dog_num_privileged_obs = dog_dim
    cfg.arm.arm_num_privileged_obs = arm_dim
    cfg.dog.dog_num_privileged_obs = dog_dim


def enable_rot6d(cfg, layout):
    cfg.use_rot6d = True
    layout.env_obs += 3
    layout.arm_obs += 3
    layout.arm_cmd += 3
    layout.dog_obs += 3


def enable_traj_track(cfg, layout, defaults, traj_track_reward_scale=5.0):
    cfg.arm.trajectory.enabled = True
    traj_window_dims = len(defaults.arm.trajectory.window_offsets) * 9
    feature = defaults.traj_track

    layout.arm_action_cd = defaults.arm.num_actions_arm + feature.arm_delta_vel_action_dims
    layout.arm_obs = feature.arm_obs_without_window + traj_window_dims
    layout.dog_obs += feature.dog_obs_dims
    layout.env_obs += feature.env_extra_obs_without_window + traj_window_dims

    cfg.hybrid.reward_scales.arm_manip_commands_tracking_combine = 0.0
    cfg.hybrid.reward_scales.vis_manip_commands_tracking_lpy = 0.0
    cfg.hybrid.reward_scales.vis_manip_commands_tracking_rpy = 0.0
    cfg.hybrid.reward_scales.traj_track = traj_track_reward_scale
    cfg.hybrid.reward_scales.trajectory_current_tracking = feature.trajectory_current_tracking_scale
    cfg.hybrid.reward_scales.trajectory_completion_time = feature.trajectory_completion_time_scale
    cfg.hybrid.reward_scales.arm_delta_vel_cmd = feature.arm_delta_vel_cmd_scale
    cfg.hybrid.reward_scales.ee_smoothness = feature.ee_smoothness_scale


def disable_traj_track(cfg):
    cfg.arm.trajectory.enabled = False


def enable_dyna_gait(cfg, layout, defaults, min_frequency=0.0):
    feature = defaults.dyna_gait

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

    layout.dog_cmd += feature.num_gait_dims
    layout.dog_obs += feature.num_gait_dims
    layout.arm_action_cd = defaults.arm.num_actions_arm + feature.plan_action_dims
    if cfg.arm.trajectory.enabled:
        layout.arm_action_cd += defaults.traj_track.arm_delta_vel_action_dims
    layout.arm_obs += feature.num_gait_dims
    layout.env_obs += feature.num_gait_dims + 2
    cfg.env.observe_gait_commands = True

    cfg.commands.num_bins_gait_frequency = feature.num_bins_gait_frequency
    cfg.commands.num_bins_footswing_height = feature.num_bins_footswing_height
    cfg.commands.num_bins_gait_duration = feature.num_bins_gait_duration
    cfg.commands.num_bins_stance_width = feature.num_bins_stance_width
    cfg.commands.num_bins_stance_length = feature.num_bins_stance_length


def disable_dyna_gait(cfg):
    cfg.commands.use_dynamic_gait = False


def configure_robot_asset(cfg, defaults, robot):
    if robot == "go1":
        cfg.asset.file = defaults.asset.go1_file
    elif robot == "go2":
        cfg.asset.file = defaults.asset.go2_file


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


def configure_task_from_args(cfg, args, traj_track_reward_scale=5.0):
    defaults = ROBODUET_DEFAULTS
    options = RoboDuetRuntimeOptions.from_args(args)

    configure_external_recipes(cfg)
    materialize_base_cfg(cfg, defaults, options)

    layout = RoboDuetLayout.from_cfg(cfg)
    if options.use_rot6d:
        enable_rot6d(cfg, layout)
    else:
        cfg.use_rot6d = False

    if options.traj_track:
        enable_traj_track(cfg, layout, defaults, traj_track_reward_scale=traj_track_reward_scale)
    else:
        disable_traj_track(cfg)

    if options.dyna_gait:
        enable_dyna_gait(cfg, layout, defaults, min_frequency=options.dyna_gait_min_frequency)
    else:
        disable_dyna_gait(cfg)

    layout.finalize(cfg)
    configure_privileged_obs_dims(cfg)
    configure_robot_asset(cfg, defaults, options.robot)
    validate_roboduet_cfg(cfg)
