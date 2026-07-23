"""RoboDuet-specific configuration delta.

``ROBODUET_OVERRIDES`` contains only fields that RoboDuet adds or changes from
the composed LeggedRobot -> Go1 -> WTW profiles. Inherited values stay in their
own profile instead of being repeated here. Keys use full ``Cfg`` paths, so
RoboDuet can still replace any inherited field by adding one explicit entry.

Derived observation sizes and runtime feature switches are intentionally kept
out of this table.  They are calculated in ``core.py`` from these
source values and the command-line options.
"""

import math

from .core import ConfigProfile


ROBOT_ASSET_FILES = {
    "go1": "{MINI_GYM_ROOT_DIR}/resources/robots/arx5p2Go1/urdf/arx5p2Go1.urdf",
    "go2": "{MINI_GYM_ROOT_DIR}/resources/robots/go2/urdf/arx5go2.urdf",
    "go2_x5": "{MINI_GYM_ROOT_DIR}/resources/robots/go2_x5_v3/urdf/go2_x5.urdf",
}

# Per-robot arm wiring applied in core.configure_robot_asset. ee_body_name is
# the tracked end-effector link; ee_local_pos shifts the tracked point from that
# link's origin to the grasp point (only the x5 arm's gripper_center needs it);
# mount_joint_name is the fixed joint whose transform mount randomization varies.
ROBOT_ARM_SPEC = {
    "go1": {
        "ee_body_name": "zarx_body6",
        "ee_local_pos": [0.0, 0.0, 0.0],
        "mount_joint_name": "zarx5p2_mount",
    },
    "go2": {
        "ee_body_name": "zarx_body6",
        "ee_local_pos": [0.0, 0.0, 0.0],
        "mount_joint_name": "zarx5p2_mount",
    },
    "go2_x5": {
        "ee_body_name": "x5_link6",
        "ee_local_pos": [0.1424, 0.0, 0.0001057],
        "mount_joint_name": "arm_mount_joint",
    },
}


# One flat delta table makes RoboDuet's changes to inherited profiles explicit.
ROBODUET_OVERRIDES = {
    # Robot initial state and low-level control.
    "init_state.default_joint_angles": {
        "FL_hip_joint": 0.1,
        "RL_hip_joint": 0.1,
        "FR_hip_joint": -0.1,
        "RR_hip_joint": -0.1,
        "FL_thigh_joint": 0.8,
        "RL_thigh_joint": 1.0,
        "FR_thigh_joint": 0.8,
        "RR_thigh_joint": 1.0,
        "FL_calf_joint": -1.5,
        "RL_calf_joint": -1.5,
        "FR_calf_joint": -1.5,
        "RR_calf_joint": -1.5,
        "widow_waist": 0.0,
        "widow_shoulder": 0.0,
        "widow_elbow": 0.0,
        "forearm_roll": 0.0,
        "widow_wrist_angle": 0.0,
        "widow_wrist_rotate": 0.0,
        "widow_forearm_roll": 0.0,
        "gripper": 0.0,
        "widow_left_finger": 0.0,
        "widow_right_finger": 0.0,
        "zarx_j1": 0.0,
        "zarx_j2": 0.0,
        "zarx_j3": 0.0,
        "zarx_j4": 0.0,
        "zarx_j5": 0.0,
        "zarx_j6": 0.0,
        "zarx_j7": 0.0,
        "zarx_j8": 0.0,
        "x5_joint1": 0.0,
        "x5_joint2": 0.0,
        "x5_joint3": 0.0,
        "x5_joint4": 0.0,
        "x5_joint5": 0.0,
        "x5_joint6": 0.0,
        "x5_joint8": 0.0,
        "x5_gripper_joint": 0.0,
    },
    # Arm PD gains, looked up by exact DOF name (see _process_dof_props and the
    # arm branch of _init_buffers' gain derivation). Union of every registered
    # arm's DOF names -- unmatched keys are simply never hit, so one table
    # serves the arx/zarx arm (go1, go2) and the x5 arm (go2_x5).
    "arm.control.stiffness_arm": {
        "zarx_j1": 40.0,
        "zarx_j2": 70.0,
        "zarx_j3": 70.0,
        "zarx_j4": 25.0,
        "zarx_j5": 25.0,
        "zarx_j6": 25.0,
        "zarx_j7": 50.0,
        "zarx_j8": 50.0,
        "x5_joint1": 40.0,
        "x5_joint2": 70.0,
        "x5_joint3": 70.0,
        "x5_joint4": 25.0,
        "x5_joint5": 25.0,
        "x5_joint6": 25.0,
        "x5_joint8": 50.0,
        "x5_gripper_joint": 50.0,
    },
    "arm.control.damping_arm": {
        "zarx_j1": 3.0,
        "zarx_j2": 15.0,
        "zarx_j3": 15.0,
        "zarx_j4": 2.0,
        "zarx_j5": 2.0,
        "zarx_j6": 2.0,
        "zarx_j7": 20.0,
        "zarx_j8": 20.0,
        "x5_joint1": 3.0,
        "x5_joint2": 15.0,
        "x5_joint3": 15.0,
        "x5_joint4": 2.0,
        "x5_joint5": 2.0,
        "x5_joint6": 2.0,
        "x5_joint8": 20.0,
        "x5_gripper_joint": 20.0,
    },
    "control.control_type": "M",
    "control.stiffness": {"joint": 35.0, "widow": 5.0, "zarx": 5.0, "zarx_j3": 20.0},
    "control.update_obs_freq": 20,  # only effective when use_vision is True
    "asset.penalize_contacts_on": ["base", "trunk", "wrist", "thigh", "calf", "Head"],
    "asset.terminate_after_contacts_on": [""],
    "asset.self_collisions": 1,  # 1 = disable, 0 = enable
    "asset.render_sphere": True,
    # Environment and policy layout source values. Observation widths are
    # derived later and therefore are not editable constants here.
    "env.keep_arm_fixed": True,
    "env.num_actions": 18,
    "env.priv_observe_arm_mount_tf": True,
    "env.priv_observe_base_mass": True,
    "env.priv_observe_com_displacement": True,
    "env.priv_observe_Kp_factor": True,
    "env.priv_observe_Kd_factor": True,
    "env.priv_observe_dof_damping": True,
    "env.priv_observe_vel": True,
    "env.priv_observe_high_freq_goal": False,  # 是否向 privileged observation 添加未降采样的目标相对 EE 位姿
    "env.observe_two_prev_actions": False,  # add last last actions
    "env.record_video": False,
    "env.recording_width_px": 500,
    "env.recording_height_px": 320,
    "env.num_recording_envs": 1,
    "env.recording_frame_stride": 2,
    "env.recording_overlay_text": True,
    "env.recording_overlay_trajectory": True,
    "env.debug_viz": False,
    # Disabled only by the dual-policy runner for pure stage-1 training. Two-stage,
    # stage-2, unified training and play keep the arm-policy path available.
    "env.arm_policy_enabled": True,
    # Stage-1 arm disturbance curriculum.
    "env.stage1_arm_ramp_iterations": 20000,
    "env.stage1_arm_fixed_fraction": 0.1,
    "env.stage1_arm_saturation_fraction": 0.8,
    "env.stage1_arm_accel_resample_time_s": 0.01,
    "env.stage1_arm_zero_accel_probability": 0.3,
    "env.stage1_arm_zero_vel_probability": 0.005,
    "env.stage1_arm_max_accel": 10.0,
    "env.stage1_arm_max_vel": 5.0,
    "env.stage1_arm_init_dof_pos_noise": 1.0,
    # Dog command distribution and limits.
    "commands.body_roll_range": [-0.4, 0.4],
    "commands.limit_body_roll": [-0.4, 0.4],
    "commands.T_force_range": [2.0, 4.0],  # only effective when randomize_end_effector_force=True
    "commands.add_force_thres": 0.3,
    # Locomotion and arm rewards.
    "rewards.terminal_body_height": 0.17,
    "reward_scales.loco_energy": -0.00004,
    # Penalizes the leg policy's base-velocity response for drifting from a
    # first-order reference model of commands_dog, so an upstream planner
    # (v_ff) can rely on a predictable, fixed time-constant leg response
    # regardless of payload/posture disturbance.
    "reward_scales.response_consistency": -0.05,
    # Cross-policy channel: let the arm policy see the dog's gait phase,
    # foot contact state, and (v_actual - v_cmd) tracking residual.
    "env.arm_observe_dog_state": True,
    # Dog policy/controller layout.
    "dog.num_actions_loco": 12,
    "dog.dog_num_observation_history": 30,
    "dog.dog_num_commands": 6,
    "dog.dog_actions": 12,
    "dog.use_adaptation_module": False,
    "dog.add_obs_noise": True,
    "dog.observe_lin_vel": True,
    "dog.observe_pose_actual": True,
    "dog.observe_track_error": True,
    # Dog-critic-only privileged dynamics. Shared leg factors are represented
    # compactly; arm factors remain per joint because they are sampled so.
    "dog.priv_observe_motor_strength": True,
    "dog.priv_observe_motor_offset": True,
    "dog.priv_observe_gravity": True,
    "dog.priv_observe_contact_states": True,
    "dog.priv_observe_arm_dynamics": True,
    # These shared env-side fields remain available to the arm critic but are
    # omitted from dog privileged obs because they are fixed in this profile.
    "dog.priv_observe_com_displacement": False,
    "dog.priv_observe_joint_friction": False,
    "dog.priv_observe_dof_damping": False,
    "dog.control.stiffness_leg": {"joint": 35.0},
    "dog.control.damping_leg": {"joint": 1.0},
    # Arm commands and controller.
    "arm.num_actions_arm": 6,
    "arm.num_privileged_links": 8,
    "arm.arm_num_observation_history": 60,
    "arm.arm_num_commands": 6,  # slot exposed to the DOG policy's obs
    # No plan-action channel in the DLS-IK architecture (v_ff/posture output
    # from the arm policy is deferred -- see project-design-v3.md §5). Arm
    # action == Δq residual only, so num_actions_arm_cd == num_actions_arm.
    "arm.num_actions_arm_cd": 6,
    "arm.use_adaptation_module": False,
    "arm.target.pos_range": [[0.0, 0.55], [-0.4, 0.4], [0.25, 0.9]],
    "arm.target.roll_ee": [-math.radians(60.0), math.radians(60.0)],
    "arm.target.pitch_ee": [-math.radians(75.0), math.radians(75.0)],
    "arm.target.yaw_ee": [-math.radians(90.0), math.radians(90.0)],
    "arm.target.resample_time_s": [2.0, 3.0],
    "arm.ik.damping": 0.1,
    "arm.ik.step_gain": 1.0,
    "arm.ik.max_step_rad": 0.5,
    "arm.ik.residual_scale": 0.07,
    "arm.ik.ee_local_pos": [0.1424, 0.0, 0.0001057],
    "arm.obs_scales.l": 1.0,
    "arm.obs_scales.p": 1.0,
    "arm.obs_scales.y": 1.0,
    "arm.obs_scales.wx": 1.0,
    "arm.obs_scales.wy": 1.0,
    "arm.obs_scales.wz": 1.0,
    # WBC planning and reward configuration.
    #
    # Stage-2 MVP (project-design-v3.md, DLS-IK-only slice): base does not
    # move, arm policy outputs a Δq residual on top of a per-step DLS-IK
    # correction toward a static per-episode SE(3) target sampled in the base
    # frame. v_ff/ρ, γ(s)/s_ref(t), multi-critic and the safety filter are
    # deliberately out of scope for this round.
    "wbc.use_vision": False,
    "wbc.rewards.terminal_body_height": 0.17,
    "wbc.rewards.use_terminal_body_height": True,
    "wbc.rewards.use_terminal_roll": False,  # whether to terminate episode when roll exceeds threshold
    "wbc.rewards.use_terminal_pitch": False,
    "wbc.rewards.terminal_body_roll": 0.10,
    "wbc.rewards.terminal_body_pitch": 0.2,
    # exp(-err^2 / sigma) tracking sigmas for the DLS-IK task-space reward
    # (see _reward_ee_pos_tracking / _reward_ee_rot_tracking).
    "rewards.ee_pos_tracking_sigma": 0.02,  # (m^2), ~14cm error -> reward=0.5
    "rewards.ee_rot_tracking_sigma": 0.25,  # (rad^2), ~35deg error -> reward=0.5
    "wbc.reward_scales.jump": 5.0,
    "wbc.reward_scales.ee_pos_tracking": 4.0,
    "wbc.reward_scales.ee_rot_tracking": 1.0,
    "wbc.reward_scales.hip_action_l2": -0.05,
    "wbc.reward_scales.raibert_heuristic": -0.0,
    "wbc.reward_scales.arm_control_limits": -0.0001,
    "wbc.reward_scales.ee_smoothness": -1e-4,
    "wbc.reward_scales.arm_contact": -1.0,
    # Domain randomization.
    "domain_rand.dog_obs_frame_drop_prob": 0.02,
    "domain_rand.added_mass_range": [-2.0, 2.0],
    "domain_rand.randomize_lag_timesteps": False,
    "domain_rand.randomize_end_effector_force": False,
    "domain_rand.max_force": 15.0,
    "domain_rand.max_force_offset": 0.01,
    "domain_rand.randomize_mount_position": True,
    "domain_rand.mount_position_range": [[-0.05, 0.05], [-0.02, 0.02], [-0.05, 0.05]],
    "domain_rand.randomize_mount_rotation": True,
    "domain_rand.mount_rpy_range": [
        [-0.05236, 0.05236],
        [-0.05236, 0.05236],
        [-0.08727, 0.08727],
    ],
    "domain_rand.mount_tf_buckets": 16,
    "domain_rand.mount_tf_bucket_seed": 1234,
    "domain_rand.stage1_arm.randomize_Kp_factor": True,
    "domain_rand.stage1_arm.Kp_factor_range": [0.5, 1.5],
    "domain_rand.stage1_arm.randomize_Kd_factor": True,
    "domain_rand.stage1_arm.Kd_factor_range": [0.2, 2.0],
    "domain_rand.stage1_arm.randomize_motor_strength": True,
    "domain_rand.stage1_arm.motor_strength_range": [0.7, 1.3],
    "domain_rand.stage1_arm.randomize_motor_offset": True,
    "domain_rand.stage1_arm.motor_offset_range": 0.05,
    "domain_rand.stage1_arm.randomize_link_mass": True,
    "domain_rand.stage1_arm.link_mass_range": [0.1, 2.0],
    "domain_rand.stage1_arm.randomize_link_com": True,
    "domain_rand.stage1_arm.link_com_range": 0.1,
    # Simulated payload: a per-episode random mass held at the EE, applied
    # as a sustained gravity-aligned force at the EE rigid body (not a
    # rigid-body mass edit, which IsaacGym only allows at actor creation).
    # Scaled by the same stage1 curriculum intensity as the rest of the arm
    # disturbance (see _get_stage1_arm_curriculum_intensity).
    "domain_rand.stage1_arm.randomize_ee_payload": True,
    "domain_rand.stage1_arm.ee_payload_mass_range": [0.0, 1.5],  # kg
    "env.priv_observe_stage1_ee_payload_mass": True,
    "domain_rand.stage2_arm.randomize_Kp_factor": True,
    "domain_rand.stage2_arm.Kp_factor_range": [0.9, 1.1],
    "domain_rand.stage2_arm.randomize_Kd_factor": True,
    "domain_rand.stage2_arm.Kd_factor_range": [0.9, 1.1],
    "domain_rand.stage2_arm.randomize_motor_strength": True,
    "domain_rand.stage2_arm.motor_strength_range": [0.85, 1.15],
    "domain_rand.stage2_arm.randomize_motor_offset": True,
    "domain_rand.stage2_arm.motor_offset_range": 0.025,
    "domain_rand.stage2_arm.randomize_link_mass": False,
    "domain_rand.stage2_arm.link_mass_range": [0.9, 1.1],
    "domain_rand.stage2_arm.randomize_link_com": False,
    "domain_rand.stage2_arm.link_com_range": 0.01,
}


# Non-Cfg constants used while deriving WBC rewards and feature layouts.
WBC_REWARD_FACTORS = {
    "tracking_lin_vel": 0.7,
    "tracking_ang_vel": 0.5,
    "arm_energy": -0.00004,
    "arm_dof_vel": 0.01,
    "arm_dof_acc": 0.01,
    "arm_action_rate": 0.001,
    "arm_action_smoothness_1": 0.001,
    "arm_action_smoothness_2": 0.001,
}


FEATURE_LAYOUT = {
    "rot6d_command_dims": 3,
    "dynamic_gait_command_dims": 5,
}


DYNAMIC_GAIT_BIN_CONFIG = {
    "commands.num_bins_gait_frequency": 11,
    "commands.num_bins_footswing_height": 5,
    "commands.num_bins_gait_duration": 3,
    "commands.num_bins_stance_width": 3,
    "commands.num_bins_stance_length": 3,
}


ROBODUET_PROFILE = ConfigProfile(
    name="roboduet",
    overrides=ROBODUET_OVERRIDES,
    allow_new=True,
)
