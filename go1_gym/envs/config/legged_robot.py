# License: see [LICENSE, LICENSES/legged_gym/LICENSE]
"""Framework-level legged-robot default profile.

The nested classes are an immutable template only. ``core.build_config``
materializes them into an independent ``ConfigNode`` tree for every build.
"""


class LeggedRobotDefaults:
    class env:
        num_envs = 4096
        num_observations = 48
        num_scalar_observations = 42
        # if not None a privilige_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_privileged_obs = None
        privileged_future_horizon = 1
        num_actions = 12
        num_observation_history = 15
        env_spacing = 3.  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 20  # episode length in seconds
        observe_vel = True
        observe_only_ang_vel = False
        observe_only_lin_vel = False
        observe_yaw = False
        observe_contact_states = False
        observe_command = True
        observe_height_command = False
        observe_gait_commands = False
        observe_timing_parameter = False
        observe_clock_inputs = False
        observe_two_prev_actions = False
        observe_imu = False

        priv_observe_friction = True
        priv_observe_friction_indep = True
        priv_observe_ground_friction = False
        priv_observe_ground_friction_per_foot = False
        priv_observe_restitution = True
        priv_observe_base_mass = True
        priv_observe_com_displacement = True
        priv_observe_motor_strength = False
        priv_observe_motor_offset = False
        priv_observe_joint_friction = True
        priv_observe_Kp_factor = True
        priv_observe_Kd_factor = True
        priv_observe_contact_forces = False
        priv_observe_contact_states = False
        priv_observe_body_velocity = False
        priv_observe_foot_height = False
        priv_observe_body_height = False
        priv_observe_gravity = False
        priv_observe_terrain_type = False
        priv_observe_clock_inputs = False
        priv_observe_doubletime_clock_inputs = False
        priv_observe_halftime_clock_inputs = False
        priv_observe_desired_contact_states = False
        priv_observe_dummy_variable = False

    class terrain:
        mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 0  # 25 # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        terrain_noise_magnitude = 0.1
        # rough terrain only:
        terrain_smoothness = 0.005
        measure_heights = True
        # 1mx1.6m rectangle (without center line)
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        min_init_terrain_level = 0
        max_init_terrain_level = 5  # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces
        difficulty_scale = 1.
        x_init_range = 1.
        y_init_range = 1.
        z_init_range = 1.
        yaw_init_range = 0.
        roll_init_range = 3.14
        pitch_init_range = 3.14
        reset_curriculum = False
        reset_curriculum_initial_fraction = 0.0
        reset_curriculum_tracking_threshold = 0.5
        reset_curriculum_tracking_ema_alpha = 0.05
        reset_curriculum_stability_iterations = 100
        reset_curriculum_growth_iterations = 5000
        x_init_offset = 0.
        y_init_offset = 0.
        teleport_robots = True
        teleport_thresh = 2.0
        max_platform_height = 0.2
        center_robots = False
        center_span = 5

    class commands:
        use_dynamic_gait = False
        command_curriculum = False
        max_reverse_curriculum = 1.
        max_forward_curriculum = 1.
        yaw_command_curriculum = False
        max_yaw_curriculum = 1.
        exclusive_command_sampling = False
        num_commands = 3
        resampling_time = 10.  # time before command are changed[s]
        subsample_gait = False
        gait_interval_s = 10.  # time between resampling gait params
        vel_interval_s = 10.
        jump_interval_s = 20.  # time between jumps
        jump_duration_s = 0.1  # duration of jump
        jump_height = 0.3
        heading_command = True  # if true: compute ang vel command from heading error
        global_reference = False # if true: vel obs are in global frame
        observe_accel = False
        curriculum_type = "RewardThresholdCurriculum"
        lipschitz_threshold = 0.9

        num_lin_vel_bins = 20
        lin_vel_step = 0.3
        num_ang_vel_bins = 20
        ang_vel_step = 0.3
        distribution_update_extension_distance = 1
        curriculum_seed = 100

        lin_vel_x = [-1.0, 1.0]  # min max [m/s]
        lin_vel_y = [-1.0, 1.0]  # min max [m/s]
        ang_vel_yaw = [-1, 1]  # min max [rad/s]
        body_height_cmd = [-0.05, 0.05]
        impulse_height_commands = False

        limit_vel_x = [-10.0, 10.0]
        limit_vel_y = [-0.6, 0.6]
        limit_vel_yaw = [-10.0, 10.0]
        limit_body_height = [-0.05, 0.05]
        limit_gait_phase = [0, 0.01]
        limit_gait_offset = [0, 0.01]
        limit_gait_bound = [0, 0.01]
        limit_gait_frequency = [2.0, 2.01]
        limit_gait_duration = [0.49, 0.5]
        limit_footswing_height = [0.06, 0.061]
        limit_body_pitch = [0.0, 0.01]
        limit_body_roll = [0.0, 0.01]
        limit_aux_reward_coef = [0.0, 0.01]
        limit_compliance = [0.0, 0.01]
        limit_stance_width = [0.0, 0.01]
        limit_stance_length = [0.0, 0.01]

        num_bins_vel_x = 25
        num_bins_vel_y = 3
        num_bins_vel_yaw = 25
        num_bins_body_height = 1
        num_bins_gait_frequency = 11
        num_bins_gait_phase = 11
        num_bins_gait_offset = 2
        num_bins_gait_bound = 2
        num_bins_gait_duration = 3
        num_bins_footswing_height = 1
        num_bins_body_pitch = 1
        num_bins_body_roll = 1
        num_bins_aux_reward_coef = 1
        num_bins_compliance = 1
        num_bins_compliance = 1
        num_bins_stance_width = 1
        num_bins_stance_length = 1

        heading = [-3.14, 3.14]

        gait_phase_cmd_range = [0.0, 0.01]
        gait_offset_cmd_range = [0.0, 0.01]
        gait_bound_cmd_range = [0.0, 0.01]
        gait_frequency_cmd_range = [2.0, 2.01]
        gait_duration_cmd_range = [0.49, 0.5]
        footswing_height_range = [0.06, 0.061]
        body_pitch_range = [0.0, 0.01]
        body_roll_range = [0.0, 0.01]
        aux_reward_coef_range = [0.0, 0.01]
        compliance_range = [0.0, 0.01]
        stance_width_range = [0.0, 0.01]
        stance_length_range = [0.0, 0.01]

        exclusive_phase_offset = True
        binary_phases = False
        pacing_offset = False
        balance_gait_distribution = True
        gaitwise_curricula = True

    class curriculum_thresholds:
        tracking_lin_vel = 0.8  # closer to 1 is tighter
        tracking_ang_vel = 0.5
        tracking_contacts_shaped_force = 0.8  # closer to 1 is tighter
        tracking_contacts_shaped_vel = 0.8

    class init_state:
        pos = [0.0, 0.0, 1.]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        # target angles when action = 0.0
        default_joint_angles = {"joint_a": 0., "joint_b": 0.}

    class control:
        control_type = 'actuator_net' #'P'  # P: position, V: velocity, T: torques
        # PD gains live in dog.control.stiffness_leg / arm.control.stiffness_arm.
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        hip_scale_reduction = 1.0
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset:
        file = ""
        foot_name = "None"  # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = []
        arm_contact_bodies = []
        terminate_after_contacts_on = []
        disable_gravity = False
        # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        collapse_fixed_joints = True
        fix_base_link = False  # fixe the base of the robot
        default_dof_drive_mode = 3  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        # replace collision cylinders with capsules, leads to faster/more stable simulation
        replace_cylinder_with_capsule = True
        flip_visual_attachments = True  # Some .obj meshes must be flipped from y-up to z-up

        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        rand_interval_s = 10
        randomize_rigids_after_start = True
        randomize_friction = True
        friction_range = [0.5, 1.25]  # increase range
        randomize_restitution = False
        restitution_range = [0, 1.0]
        randomize_base_mass = False
        # add link masses, increase range, randomize inertia, randomize joint properties
        added_mass_range = [-1., 1.]
        randomize_com_displacement = False
        # add link masses, increase range, randomize inertia, randomize joint properties
        com_displacement_range = [-0.15, 0.15]
        randomize_motor_strength = False
        motor_strength_range = [0.9, 1.1]
        randomize_Kp_factor = False
        Kp_factor_range = [0.9, 1.1]
        randomize_Kd_factor = False
        Kd_factor_range = [0.9, 1.1]
        gravity_rand_interval_s = 7
        gravity_impulse_duration = 1.0
        randomize_gravity = False
        gravity_range = [-1.0, 1.0]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.
        max_push_ang_vel = 0.6
        randomize_action_delay = True
        randomize_lag_timesteps = True
        lag_timesteps = 6
        # Per-step, per-env probability that get_dog_observations() re-delivers
        # the previous step's (already-noised) observation instead of a fresh
        # one -- simulates a dropped sensor/comms frame. 0 = disabled.
        dog_obs_frame_drop_prob = 0.0

    class rewards:
        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards_ji22_style = False
        sigma_rew_neg = 5
        reward_container_name = "Rewards"
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        tracking_sigma_lat = 0.25  # tracking reward = exp(-error^2/sigma)
        tracking_sigma_long = 0.25  # tracking reward = exp(-error^2/sigma)
        tracking_sigma_yaw = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1.  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 1.
        max_contact_force = 100.  # forces above this value are penalized
        use_terminal_body_height = False
        terminal_body_height = 0.20
        use_terminal_foot_height = False
        terminal_foot_height = -0.005
        use_terminal_roll_pitch = False
        terminal_body_ori = 0.5
        kappa_gait_probs = 0.07
        gait_force_sigma = 50.
        gait_vel_sigma = 0.5
        footswing_height = 0.09
        # Time constant (s) of the first-order low-pass reference model that
        # response_consistency tracks: v_ref += (v_cmd - v_ref) * dt / T.
        response_consistency_T = 0.4
        # Which gait shaping is active. 'clock' = the stock terms scored
        # against _step_contact_targets' phase (tracking_contacts_shaped_
        # force/vel, feet_clearance_cmd_linear, raibert_heuristic).
        # 'clock_free' = the contact-stopwatch/symmetry terms ported from
        # robot_lab, for training with dog.observe_clock_inputs off. Set via
        # config.core.set_gait_reward_mode, which rewrites the reward scales;
        # writing this field alone changes nothing.
        gait_reward_mode = "clock"
        # _reward_gait_sync: exp(-squared timing error / sigma), with each
        # squared term clipped at max_err^2 so one badly out-of-phase foot
        # saturates instead of zeroing the whole product.
        gait_sync_sigma = 0.5
        gait_sync_max_err = 0.2
        # _reward_feet_air_time_variance: cap (s) on each measured phase
        # duration before the variance is taken.
        gait_air_time_clip = 0.5
        # _reward_feet_stance_width: exp(-lateral error^2 / sigma).
        gait_stance_width_sigma = 0.25
        # _reward_feet_swing_height: body-frame target foot height (m, negative
        # = below the base) and the gain of the tanh(|v_xy|) swing detector
        # that replaces the clock's (1 - desired_contact_states) weight.
        gait_swing_height_target = -0.25
        gait_swing_tanh_mult = 2.0

    class reward_scales:
        termination = -0.0
        tracking_lin_vel = 1.0
        tracking_ang_vel = 0.5
        response_consistency = 0.0
        lin_vel_z = -2.0
        ang_vel_xy = -0.05
        orientation = -0.
        torques = -0.00001
        dof_vel = -0.
        dof_acc = -2.5e-7
        collision = -1.
        arm_contact = -1.
        action_rate = -0.01
        tracking_contacts_shaped_force = 0.
        tracking_contacts_shaped_vel = 0.
        jump = 0.0
        dof_pos_limits = 0.0
        feet_contact_forces = 0.
        feet_slip = 0.
        feet_clearance_cmd_linear = 0.
        dof_pos = 0.
        action_smoothness_1 = 0.
        action_smoothness_2 = 0.
        feet_impact_vel = 0.0
        raibert_heuristic = 0.0
        # Clock-free gait shaping (rewards.gait_reward_mode == 'clock_free').
        # Zero here; set_gait_reward_mode fills in the live values.
        gait_sync = 0.0
        feet_air_time_variance = 0.0
        joint_mirror = 0.0
        feet_stance_width = 0.0
        feet_swing_height = 0.0

    class normalization:
        clip_observations = 100.
        clip_actions = 100.

        friction_range = [0.05, 4.5]
        ground_friction_range = [0.05, 4.5]
        restitution_range = [0, 1.0]
        added_mass_range = [-1., 3.]
        com_displacement_range = [-0.1, 0.1]
        motor_strength_range = [0.9, 1.1]
        motor_offset_range = [-0.05, 0.05]
        Kp_factor_range = [0.9, 1.1]
        Kd_factor_range = [0.9, 1.1]
        joint_friction_range = [0.0, 0.7]
        dof_damping_range = [0.0, 10.0]
        contact_force_range = [0.0, 50.0]
        contact_state_range = [0.0, 1.0]
        body_velocity_range = [-6.0, 6.0]
        foot_height_range = [0.0, 0.15]
        body_height_range = [0.0, 0.60]
        gravity_range = [-1.0, 1.0]
        motion = [-0.01, 0.01]

    class obs_scales:
        lin_vel = 2.0
        ang_vel = 0.25
        dof_pos = 1.0
        dof_vel = 0.05
        imu = 0.1
        height_measurements = 5.0
        friction_measurements = 1.0
        body_height_cmd = 1.0
        gait_phase_cmd = 1.0
        gait_freq_cmd = 1.0
        gait_offset_cmd = 1.0
        gait_bound_cmd = 1.0
        gait_duration_cmd = 1.0
        footswing_height_cmd = 0.15
        body_pitch_cmd = 1.
        body_roll_cmd = 1.
        aux_reward_cmd = 1.0
        compliance_cmd = 1.0
        stance_width_cmd = 1.0
        stance_length_cmd = 1.0
        segmentation_image = 1.0
        rgb_image = 1.0
        depth_image = 1.0

    class noise:
        add_noise = True
        noise_level = 1.0  # scales other values

    class noise_scales:
        dof_pos = 0.01
        dof_vel = 1.5
        lin_vel = 0.1
        ang_vel = 0.2
        imu = 0.1
        gravity = 0.05
        contact_states = 0.05
        height_measurements = 0.1
        friction_measurements = 0.0
        segmentation_image = 0.0
        rgb_image = 0.0
        depth_image = 0.0

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        use_gpu_pipeline = True

        # IsaacGym's ``gymutil.parse_sim_config`` requires this leaf to be a
        # mapping, rather than an attribute namespace.
        physx = {
            "num_threads": 10,
            "solver_type": 1,  # 0: pgs, 1: tgs
            "num_position_iterations": 4,
            "num_velocity_iterations": 1,
            "contact_offset": 0.01,  # [m]
            "rest_offset": 0.0,  # [m]
            "bounce_threshold_velocity": 0.5,
            "max_depenetration_velocity": 1.0,
            "max_gpu_contact_pairs": 2 ** 23,
            "default_buffer_size_multiplier": 5,
            "contact_collection": 2,  # 0: never, 1: last sub-step, 2: all sub-steps
        }
