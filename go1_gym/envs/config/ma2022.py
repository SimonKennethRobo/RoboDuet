"""Ma et al., RA-L 2022 locomotion recipe (editable, robot-specific defaults).

Ma-specific wrench/noise/network settings remain Go2 adaptations. Reward
terms follow reference [10], Supplement S7; Ma stability multipliers are local.
"""

from dataclasses import dataclass, field

from .core import ConfigNode, RoboDuetRuntimeOptions, build_roboduet_config


@dataclass
class MaTrainingConfig:
    # World-frame [Fx,Fy,Fz,Tx,Ty,Tz], N and Nm. Negative z models arm weight.
    wrench_min: tuple = (-20., -20., -60., -6., -6., -4.)
    wrench_max: tuple = (20., 20., 0., 6., 6., 4.)
    beta_range: tuple = (0.001, 0.01)
    prediction_times: tuple = (0., 0.2, 0.4, 0.6, 0.8)
    wrench_scale: tuple = (0.05, 0.05, 0.02, 0.2, 0.2, 0.25)
    # Radii of uniformly sampled 3-D balls; componentwise acceleration gains.
    force_gain_radius: float = 1.0
    torque_gain_radius: float = 0.05
    disturbance_std: tuple = (1., 1., 1., 0.1, 0.1, 0.1)
    # Student-only normalized input noise. Offset/scale are per episode.
    prediction_noise_std: float = 0.05
    prediction_bias_std: float = 0.05
    prediction_scale_std: float = 0.05
    proprio_noise_std: float = 0.01
    scan_noise_std: float = 0.02  # metres, before scan normalization
    scan_scale: float = 5.0
    # Phase increments and cubic foot lift follow reference [10] S5.
    # Go2 lift height, joint residual scale and IK geometry are adaptations.
    # c = initial ** (exponent ** iteration), set once per training iteration.
    # [10] updates c <- c**0.98 per episode; a per-episode update couples the
    # penalty ramp to episode length (falls shorten episodes, which raises c,
    # which rewards falling). 0.997/iteration reaches c = 0.9 near 1000 iters.
    reward_curriculum_initial: float = 0.1
    reward_curriculum_exponent: float = 0.997
    # Weight on the q-ddot**2 part of joint_motion, relative to [10] S7 (1.0).
    # Here q-ddot is a 20-ms finite difference: at 1.0 the nominal gait alone
    # costs ~-70/step against <= +2.25 of tracking reward, and the policy
    # learns to fall. 0.01 puts the nominal gait near -0.7/step.
    joint_acceleration_coef: float = 0.01
    # Clip the per-step reward (all terms except termination) at zero, as in
    # this repository's only_positive_rewards. Otherwise a noisy early policy
    # earns ~-5/step (value ~-500), and falling (-10 once) is the best option.
    only_positive_rewards: bool = True
    # Upper bound on the Gaussian action std. Actions are clipped to +-3, so a
    # larger std changes nothing in the env while the entropy bonus rewards it
    # (v3_1 saturated at the old e**2 = 7.39 bound).
    max_action_std: float = 1.0
    # Ma Fig. 5 ablation switch. False zeroes the 30 wrench-prediction entries
    # of the teacher and student wrench observations; the wrench is still
    # applied and the commands/base twist entries are unchanged.
    observe_wrench_prediction: bool = True
    stability_multiplier: float = 2.0  # Ma III-D1: higher weight; factor not published
    knee_limit: float = -0.1  # Go2 calf convention; prevents knee reversal
    gait_frequency: float = 2.0
    # Radians per policy action. With actions clipped to +-3 this bounds the
    # per-step phase change to ~0.05 cycles, comparable to the nominal
    # 0.04-cycle advance, so the phase remains a gait clock.
    phase_increment_scale: float = 0.1
    swing_height: float = 0.06
    residual_scale: float = 0.25
    hidden_dim: int = 128
    embedding_dim: int = 32
    learning_rate: float = 3e-4
    rollout_steps: int = 24
    ppo_epochs: int = 5
    minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 1.0
    max_grad_norm: float = 1.0
    save_interval: int = 500
    loss_weights: dict = field(default_factory=lambda: {
        "action": 1., "embedding": 1., "privileged": 1.,
        "scan": 1., "w1": 1., "w2": 1.,
    })


def build_ma_config(num_envs=4096, robot="go2", terrain="trimesh"):
    if robot != "go2":
        raise ValueError("The available bare robot asset for Ma training is Go2")
    cfg = build_roboduet_config(options=RoboDuetRuntimeOptions(num_envs, robot))
    cfg.asset.file = "{MINI_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2_description.urdf"
    cfg.asset.render_sphere = False
    cfg.asset.arm_contact_bodies = []
    cfg.asset.terminate_after_contacts_on = ["base", "trunk"]
    cfg.env.num_actions = cfg.dog.num_actions_loco = 12
    cfg.arm.num_actions_arm = 0
    cfg.env.arm_policy_enabled = False
    cfg.env.stage1_arm_curriculum = False
    cfg.env.record_video = False
    cfg.env.episode_length_s = 20.
    cfg.control.control_type = "P"
    cfg.control.action_scale = 1.
    cfg.control.hip_scale_reduction = 1.
    cfg.control.decimation = 4
    cfg.sim.dt = 0.005
    # Thousands of independent robots share terrain tiles. Give PhysX's
    # broadphase sufficient pair buffers for their overlapping AABBs.
    cfg.sim.physx["default_buffer_size_multiplier"] = 32
    cfg.dog.control.stiffness_leg = {"joint": 40.}
    cfg.dog.control.damping_leg = {"joint": 1.}
    cfg.init_state.pos = [0., 0., 0.34]
    cfg.init_state.default_joint_angles = {
        f"{leg}_{joint}_joint": angle
        for leg in ("FL", "FR", "RL", "RR")
        for joint, angle in (("hip", 0.1 if leg.endswith("L") else -0.1),
                             ("thigh", 0.8), ("calf", -1.5))
    }
    cfg.terrain.mesh_type = terrain
    cfg.terrain.measure_heights = True
    cfg.terrain.measured_points_x = [-0.6, -0.3, 0., 0.3, 0.6]
    cfg.terrain.measured_points_y = [-0.4, -0.2, 0., 0.2, 0.4]
    cfg.terrain.roughness_tiers = [0., 0.02, 0.04]
    cfg.terrain.roughness_tier_weights = [0.4, 0.3, 0.3]
    cfg.terrain.curriculum = False
    cfg.terrain.reset_curriculum = False
    cfg.terrain.reset_mode = "legacy"
    cfg.terrain.robustness_metrics = False
    cfg.terrain.z_init_range = 0.
    cfg.terrain.roll_init_range = cfg.terrain.pitch_init_range = 0.
    cfg.terrain.yaw_init_range = 3.14
    # Distribute large batches over 200 tiles rather than crowding 4096
    # actors onto 30 tiles (which overflowed GPU aggregate-pair buffers).
    cfg.terrain.num_rows = 10
    cfg.terrain.num_cols = 20
    cfg.terrain.max_init_terrain_level = 2
    cfg.response.grouping.enabled = False
    cfg.response.excitation.enabled = False
    cfg.response.curriculum.enabled = False
    # Start from an explicit dynamics allowlist, including nested arm flags.
    # The parent builder supplies simulator schema, not the Ma DR recipe.
    def disable_randomization(node):
        for key, value in vars(node).items():
            if isinstance(value, ConfigNode):
                disable_randomization(value)
            elif key.startswith("randomize_") or key in ("push_robots", "push_curriculum"):
                setattr(node, key, False)
    disable_randomization(cfg.domain_rand)
    cfg.domain_rand.dog_obs_frame_drop_prob = 0.
    cfg.domain_rand.dog_obs_latency_jitter_steps = 0
    cfg.domain_rand.mode = "sim2real"
    for name in ("randomize_mount_position", "randomize_mount_rotation", "push_robots",
                 "randomize_end_effector_force", "randomize_gravity",
                 "randomize_rigids_after_start", "randomize_action_delay"):
        setattr(cfg.domain_rand, name, False)
    cfg.domain_rand.randomize_base_mass = False
    cfg.domain_rand.randomize_com_displacement = False
    cfg.domain_rand.randomize_friction = True
    cfg.domain_rand.friction_range = [0.4, 1.5]
    cfg.domain_rand.randomize_restitution = False
    cfg.domain_rand.randomize_motor_strength = False
    cfg.domain_rand.motor_strength_range = [1., 1.]
    cfg.domain_rand.Kp_factor_range = [1., 1.]
    cfg.domain_rand.Kd_factor_range = [1., 1.]
    cfg.domain_rand.motor_offset_range = [0., 0.]
    cfg.commands.use_dynamic_gait = False
    cfg.commands.command_curriculum = False
    # Parent physical metrics read the full command buffer; only [:3] enters
    # the Ma policy. Remaining slots carry fixed posture/gait defaults.
    cfg.dog.dog_num_commands = 11
    cfg.commands.resampling_time = 5.
    cfg.commands.lin_vel_x = [-0.5, 0.5]
    cfg.commands.lin_vel_y = [-0.3, 0.3]
    cfg.commands.ang_vel_yaw = [-1., 1.]
    cfg.rewards.only_positive_rewards = False
    cfg.rewards.only_positive_rewards_ji22_style = False
    cfg.rewards.base_height_target = 0.30
    cfg.rewards.use_terminal_body_height = True
    cfg.rewards.terminal_body_height = 0.16
    cfg.rewards.use_terminal_roll_pitch = True
    # v3.2 teachers settled into a crouch leaning ~0.56 rad in roll, which a
    # 1.0-rad limit never terminates. 0.5 rad ends that posture as a fall.
    cfg.rewards.terminal_body_ori = 0.5
    cfg.rewards.terminal_roll_pitch_grace_s = 0.
    # Signed terms are computed by the task-owned reward kernel. Positive
    # coefficients from [10] S7; orientation is the extra Ma III-D1 term.
    scales = dict(tracking_lin_vel=0.75, tracking_ang_vel=0.75,
                  orthogonal_velocity=0.75, body_motion=1., orientation=1.,
                  foot_clearance=0.003, collision=0.1, joint_motion=0.001,
                  knee_limit=0.08, target_smoothness=0.003,
                  torques=1e-6, foot_slip=0.003,
                  # Not in [10] S7: falls end the episode, and without a
                  # penalty an all-negative return makes falling optimal.
                  termination=10.)
    cfg.reward_scales = ConfigNode()
    cfg.wbc.reward_scales = ConfigNode()
    for name, value in scales.items():
        setattr(cfg.reward_scales, name, value)
        setattr(cfg.wbc.reward_scales, name, value)
    cfg.asset.ee_body_name = None
    return cfg
