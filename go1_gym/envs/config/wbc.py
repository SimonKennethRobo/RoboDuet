"""RoboDuet-specific configuration delta.

``ROBODUET_OVERRIDES`` contains only fields that RoboDuet adds or changes from
the composed LeggedRobot -> Go1 -> WTW profiles. Inherited values stay in their
own profile instead of being repeated here. Keys use full ``Cfg`` paths, so
RoboDuet can still replace any inherited field by adding one explicit entry.

Derived observation sizes and runtime feature switches are intentionally kept
out of this table.  They are calculated in ``core.py`` from these
source values and the command-line options.

Below the robot/arm wiring tables, parameters are grouped by **which
stage/mode you'd touch them to tune**, not just by subsystem:

- ``COMMON_OVERRIDES``: infra shared by every stage (robot pose, PD gains,
  low-level control, dog policy, arm policy layout, base domain rand).
- ``STAGE1_OVERRIDES``: only read while pretraining the dog with the arm as
  a disturbance source (``env.stage1_arm_*`` curriculum, ``domain_rand.stage1_arm.*``).
- ``STAGE2_OVERRIDES``: shared substrate both stage-2 submodes below sit on
  top of (the ``arm.action_mode`` action interface, DLS-IK controller, EE
  tracking reward/sigma, WBC termination, ``domain_rand.stage2_arm.*``).
- ``STAGE2_IK_OVERRIDES``: the default stage-2 submode (no ``--goal_reaching``)
  -- absolute-box SE(3) EE target sampling.
- ``GOAL_REACHING_OVERRIDES`` / ``GOAL_REACHING_REWARD_SCALES``: the
  ``--goal_reaching`` submode. Both the sampling/shaping params and the
  reward weights that only make sense for this submode live together here --
  tune this one section, not two.

All groups, including ``GOAL_REACHING_REWARD_SCALES`` at 0.0, are merged into
one ``ROBODUET_OVERRIDES`` and applied unconditionally at config-build time.
That's safe because ``LeggedRobot._prepare_reward_function`` now drops any
``wbc.reward_scales.*`` entry that's exactly 0 before registering ``_reward_*``
callbacks, so a name sitting at its 0.0 default is never registered/called --
only ``core.enable_goal_reaching()`` overwriting it with a real value (behind
``--goal_reaching``) makes it live. See the docstring above
``GOAL_REACHING_OVERRIDES``.

Parameters carry short section headers only; no per-parameter comments.
Every field's meaning, current value rationale and tuning direction lives in
``docs/PARAM_TUNING.md``.
"""

import math

from .core import ConfigProfile

ROBOT_ASSET_FILES = {
    "go1": "{MINI_GYM_ROOT_DIR}/resources/robots/arx5p2Go1/urdf/arx5p2Go1.urdf",
    "go2": "{MINI_GYM_ROOT_DIR}/resources/robots/go2/urdf/arx5go2.urdf",
    "go2_x5": "{MINI_GYM_ROOT_DIR}/resources/robots/go2_x5_v3/urdf/go2_x5.urdf",
}

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


# ============================================================
# COMMON: shared by every stage/mode -- robot pose, PD gains, low-level
# control & asset, env infra + generic critic priv-obs switches, dog
# policy, arm policy layout (everything except target sampling, which is
# stage-2-submode-specific below), and base/mount domain randomization.
# ============================================================
COMMON_OVERRIDES = {
    # Main training target: perturbed simulation benchmark.
    # sim2real retains the full recipe; none is a nominal-dynamics ablation.
    "domain_rand.mode": "benchmark",
    # robot init pose (leg + arm joint angles)
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
    # low-level control & asset
    "control.control_type": "M",
    "control.update_obs_freq": 20,
    "asset.penalize_contacts_on": ["base", "trunk", "wrist", "thigh", "calf", "Head"],
    "asset.terminate_after_contacts_on": [""],
    "asset.self_collisions": 1,
    "asset.render_sphere": True,
    # env: policy layout & infrastructure
    "env.num_actions": 18,
    "env.keep_arm_fixed": True,
    "env.arm_policy_enabled": True,
    "env.observe_two_prev_actions": False,
    "env.arm_observe_dog_state": True,
    "env.record_video": False,
    "env.recording_width_px": 500,
    "env.recording_height_px": 320,
    "env.num_recording_envs": 1,
    "env.recording_frame_stride": 2,
    "env.recording_overlay_text": True,
    "env.recording_overlay_trajectory": True,
    "env.debug_viz": False,
    # privileged observation switches
    "env.priv_observe_base_mass": True,
    "env.priv_observe_com_displacement": True,
    "env.priv_observe_Kp_factor": True,
    "env.priv_observe_Kd_factor": True,
    "env.priv_observe_dof_damping": True,
    "env.priv_observe_vel": True,
    "env.priv_observe_arm_mount_tf": True,
    "env.priv_observe_high_freq_goal": False,
    # R1 command defaults: keep every channel slot; only five channels enter
    # the adaptive curriculum. Height is a delta from base_height_target.
    "commands.body_roll_range": [0.0, 0.0],
    "commands.limit_body_roll": [0.0, 0.0],
    "commands.num_bins_body_roll": 1,
    "commands.num_bins_body_pitch": 1,
    "commands.num_bins_body_height": 1,
    "commands.body_height_cmd": [-0.10, 0.05],
    "commands.limit_body_height": [-0.10, 0.05],
    "commands.gait_frequency_cmd_range": [2.8, 3.0],
    "commands.limit_gait_frequency": [2.8, 3.0],
    "commands.num_bins_gait_frequency": 1,
    "commands.footswing_height_range": [0.04, 0.041],
    "commands.limit_footswing_height": [0.04, 0.041],
    "commands.num_bins_footswing_height": 1,
    "commands.stance_width_range": [0.28, 0.32],
    "commands.limit_stance_width": [0.28, 0.32],
    "commands.num_bins_stance_width": 1,
    "commands.stance_length_range": [0.43, 0.46],
    "commands.limit_stance_length": [0.43, 0.46],
    "commands.num_bins_stance_length": 1,
    "commands.gait_duration_cmd_range": [0.49, 0.5],
    "commands.limit_gait_duration": [0.49, 0.5],
    "commands.num_bins_gait_duration": 1,
    "commands.T_force_range": [2.0, 4.0],
    "commands.add_force_thres": 0.3,
    # legs / dog policy
    "dog.num_actions_loco": 12,
    "dog.dog_actions": 12,
    "dog.dog_num_observation_history": 30,
    "dog.dog_num_commands": 6,
    "dog.use_adaptation_module": False,
    "dog.add_obs_noise": True,
    "dog.observe_lin_vel": True,
    "dog.observe_pose_actual": True,
    "dog.observe_track_error": True,
    "dog.priv_observe_motor_strength": True,
    "dog.priv_observe_motor_offset": True,
    "dog.priv_observe_gravity": True,
    "dog.priv_observe_contact_states": True,
    "dog.priv_observe_arm_dynamics": True,
    "dog.priv_observe_com_displacement": False,
    "dog.priv_observe_joint_friction": False,
    "dog.priv_observe_dof_damping": False,
    "dog.control.stiffness_leg": {"joint": 35.0},
    "dog.control.damping_leg": {"joint": 1.0},
    "arm.control.stiffness_arm": {
        "zarx_j1": 40.0,
        "zarx_j2": 70.0,
        "zarx_j3": 70.0,
        "zarx_j4": 25.0,
        "zarx_j5": 25.0,
        "zarx_j6": 25.0,
        "zarx_j7": 50.0,
        "zarx_j8": 50.0,
        "x5_joint1": 50.0,
        "x5_joint2": 50.0,
        "x5_joint3": 80.0,
        "x5_joint4": 30.0,
        "x5_joint5": 20.0,
        "x5_joint6": 20.0,
        "x5_joint8": 1000.0,
        "x5_gripper_joint": 1000.0,
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
        "x5_joint1": 5.0,
        "x5_joint2": 10.0,
        "x5_joint3": 10.0,
        "x5_joint4": 2.5,
        "x5_joint5": 2.0,
        "x5_joint6": 1.0,
        "x5_joint8": 100.0,
        "x5_gripper_joint": 100.0,
    },
    # arm policy layout (target sampling lives in the stage-2 submode blocks)
    "arm.num_actions_arm": 6,
    "arm.num_actions_arm_cd": 6,
    "arm.num_privileged_links": 8,
    "arm.arm_num_observation_history": 60,
    "arm.arm_num_commands": 6,
    "arm.use_adaptation_module": False,

    "rewards.terminal_body_height": 0.17,
    # v3-stage2 reward and attitude termination defaults.
    "rewards.raibert_form": "quadratic",
    "rewards.raibert_sigma": 0.35,
    "reward_scales.raibert_heuristic": -3.0,
    "rewards.terminal_body_ori": math.radians(60.0),
    "rewards.terminal_roll_pitch_grace_s": 1.0,

    "reward_scales.feet_impact_vel": 0.4,
    "rewards.feet_impact_vel_sigma": 0.8,
    "reward_scales.feet_contact_forces": -0.01,
    # ---- R4.4: hand the instantaneous posture terms over to R4.1 -------------
    # They guard a real failure mode: a policy can swing the body violently
    # inside a gait cycle and still look correct to R4.1, which sees only the
    # detrended (oscillation-removed) signal.  So they are kept and the
    # curriculum fades them as R4.1 ramps in, rather than being cut outright.
    #
    # The first version set 15% here, which was wrong twice.  It assumed R4.1
    # was always on -- R8 then gated R4.1 to stage 2, leaving all of stage 1
    # with posture at 15% and nothing replacing it.  And it applied to ROLL,
    # which R4.1 never covers: the decision channels are vx/vy/wyaw/height/pitch
    # and R1 freezes roll out of the command space entirely.
    #
    # orientation_control is now roll only and is NEVER faded.  pitch_control
    # and jump (body height, despite the name) fade, because R4.1 covers those.
    # Weights MEASURED, not restored from history.  Setting these back to the
    # pre-R4.4 values (-5.0 / 10.0) because "that is what they used to be"
    # collapsed a 1500-iteration run: rew_orientation_control reached -36, the
    # total reward sat at 0.001-0.07, and roll RMS ran to 1.0-1.6 rad.  That is
    # the same failure R4.2 was measured to have -- a negative weight sized for
    # one context applied in another -- and the old value was measured against a
    # reward set that had none of R1-R8 in it.
    #
    # -0.75 is the value the R5/R6 runs actually trained under, reaching roll
    # RMS 0.124 by 40k iterations. Split across the two axes it keeps the same
    # total attitude penalty as before this change.
    "reward_scales.orientation_control": -5.0,
    "reward_scales.pitch_control": -5.0,
    "reward_scales.jump": 10.0,
    # ---- R4.1/4.2/4.3 -------------------------------------------------------
    # ref_tracking is the task term (positive -> multiplies); the other two are
    # aux factors (negative -> attenuate). Weights are set from measured term
    # magnitudes; see docs/rlmpc-v3-r0-plan.md.
    # These four are END-OF-RAMP TARGETS, not starting weights.  R8's curriculum
    # multiplies each by a factor that starts at 0, so the term is registered
    # from the beginning (LeggedRobot._prepare_reward_function drops zero-scale
    # terms outright, which would make a later ramp impossible) but contributes
    # nothing until its stage begins.
    "reward_scales.ref_tracking": 2.0,
    # Gated to stage 3 by the curriculum. This is not caution, it is measured:
    # enabling them at their end-of-ramp weight from iteration 0 destroys the
    # reward signal. Against a TRAINED policy the raw phase-variance term is
    # 0.0036; against a from-scratch policy it is 0.34 -- 95x larger -- and at
    # weight -20 that puts the exponent at -6.8 to -8.6, multiplying the whole
    # reward by ~1e-3. That is exactly the "consistency reward is a constant
    # negative floor" failure the requirements document lists, and it is why R8
    # mandates a slow ramp rather than a switch.
    #
    # Note the R3 convergence gate does NOT protect against this: it opens after
    # ~45 visits per phase bin, a few seconds of walking, long before the policy
    # is any good. It guards an unconverged ESTIMATE, not an untrained POLICY.
    #
    # End-of-ramp targets, from the measured magnitudes (weight =
    # -ln(attenuation) / mean_term_value, ~10% attenuation against a trained
    # policy): phase_variance -20.0, steady_gain -0.5.
    "reward_scales.phase_variance": -20.0,
    "reward_scales.steady_gain": -0.5,
    # ---- R5 ------------------------------------------------------------------
    # Same reasoning, same default. R5's term is structurally the same shape as
    # R4.2 -- a squared deviation entering the exponent -- and it is switched on
    # in the same stage-3 ramp, so switching it on at iteration 0 would fail the
    # same way.
    #
    # It has a second problem R4.2 does not: availability.  A group of 4 is
    # desynchronised if ANY member resets before the group's next shared
    # resample, so if each env has probability p of resetting inside a 500-step
    # window, the group is usable only (1-p)^4 of the time.  Measured over
    # training, perf_group_desync_fraction sits near 0 while most envs are still
    # in their first episode, jumps to 1.0 the moment resets become common (the
    # first timeout wave, around iteration 42 at 24 steps/iteration), and only
    # then decays -- 0.32-0.37 by iteration 80.  So for a long stretch the term
    # is simply absent, and a weight applied there is uninformative rather than
    # merely harmful.
    #
    # The practical gate for R8 stage 3 is therefore this metric coming down and
    # staying down, NOT an iteration count. Group size trades directly against
    # it: 4 members gives 3 comparisons per twin but needs a 4x lower per-env
    # fall rate for the same availability.
    # End-of-ramp target -3.0, measured not guessed: a walking policy gives a
    # mean raw term of 0.0351 over the envs the mask leaves active, and
    # -ln(0.90) / 0.0351 = 3.0 for the same 10% attenuation the other two are
    # sized at (scripts/check_response_runtime.py --check r4 --policy <ckpt>).
    #
    # Note the term is averaged over ACTIVE envs only.  Twins, ungrouped envs
    # and desynchronised groups are masked out -- measured, that is 31.5% of
    # samples even for a policy that walks -- and averaging over everything
    # would understate the term by about a third and bake that into the weight.
    # -0.5, not -3.0.  The term is now divided by its own availability (see
    # response.grouping.availability_tau_s), so this number means "the weight if
    # the mask were always open" and the fix becomes a change to WHEN the term
    # is applied rather than to how hard it bites.
    #
    # Derivation: the 20k run scored -3.0 against a stage-3 mask that was open
    # 0.178 of the time, i.e. an always-on equivalent of -0.53.  Rounded down
    # rather than up because that run says the term was, if anything, already
    # too hard -- the stage-3 ramp took early termination from 0.05 to 0.67 and
    # halved episode length before stage 4 recovered it.
    "reward_scales.domain_consistency": -0.5,
    "reward_scales.loco_energy": -0.00004,
    "domain_rand.push_robots": True,
    "domain_rand.max_push_vel_xy": 1.0,
    "domain_rand.max_push_ang_vel": 1.0,
    "domain_rand.push_interval_s_range": [1.0, 8.0],
    "domain_rand.push_curriculum": True,
    "domain_rand.push_use_response_curriculum": False,
    "domain_rand.push_curriculum_initial_fraction": 0.0,
    "domain_rand.push_curriculum_growth_iterations": 8000,
    "terrain.reset_curriculum_tracking_threshold": 0.35,
    "terrain.reset_curriculum_growth_iterations": 15000,

    "terrain.mesh_type": "plane",
    "terrain.measure_heights": False,
    "terrain.roughness_tiers": [0.0, 0.02, 0.04],
    # Half the map is flat: a twin has to stay on it for a whole episode,
    # and at commands.limit_vel_y an episode covers 20 m.
    "terrain.roughness_tier_weights": [0.5, 0.25, 0.25],
    "terrain.center_robots": False,  # use all num_rows x num_cols tiles
    "terrain.reset_mode": "fixed_mixture",
    "terrain.reset_mix_hard_fraction": 0.2,
    "terrain.reset_mix_seed": 1234,  # partition RNG, independent of training RNG
    "terrain.reset_mix_easy_tilt_rad": math.radians(18.0),
    "terrain.reset_mix_hard_tilt_rad": math.radians(45.0),
    "terrain.reset_mix_yaw_rad": 0.314,
    "terrain.reset_mix_z_m": 0.05,
    "terrain.reset_mix_start_iteration": 4000,
    "terrain.reset_mix_ramp_iterations": 8000,
    "terrain.robustness_metrics": True,  # raw step sums in robustness.jsonl + wandb
    "terrain.robustness_early_window_s": 2.0,

    "domain_rand.dog_obs_frame_drop_prob": 0.05,
    # Sensing latency, in policy steps of 0.02 s.  0-1 step covers the 5-20 ms
    # a real IMU + encoder + driver + inference chain adds on top of the
    # control period; widen it in the domain-randomisation sweep rather than
    # here, so a run's latency exposure stays readable from its parameters.pkl.
    "domain_rand.randomize_dog_obs_latency": True,
    "domain_rand.dog_obs_latency_steps_range": [0, 1],
    "domain_rand.dog_obs_latency_jitter_steps": 0,
    "domain_rand.added_mass_range": [-2.0, 2.0],
    "domain_rand.randomize_lag_timesteps": False,
    # Global invariant 12.  Base CoM was the one body-level domain parameter
    # still switched off, which left every training robot perfectly balanced
    # about its geometric centre -- the single condition a real quadruped with
    # a 3 kg arm bolted on top is never in.  +-5 cm, not the WTW default of
    # +-15 cm: that was sized for a bare Go1, and 15 cm is beyond the base's
    # own half-length here.
    "domain_rand.randomize_com_displacement": True,
    "domain_rand.com_displacement_range": [-0.05, 0.05],
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
}


# ============================================================
# STAGE 1: dog-only pretraining while the arm is a disturbance source, not
# policy-driven. Only these keys are stage-1-exclusive; PD gains, dog
# policy layout etc. above are shared infra, not stage-1 params.
# ============================================================
STAGE1_OVERRIDES = {
    "env.priv_observe_stage1_ee_payload_mass": True,
    "env.stage1_arm_ramp_iterations": 20000,
    "env.stage1_arm_fixed_fraction": 0.1,
    "env.stage1_arm_saturation_fraction": 0.8,
    "env.stage1_arm_accel_resample_time_s": 0.01,
    "env.stage1_arm_zero_accel_probability": 0.3,
    "env.stage1_arm_zero_vel_probability": 0.005,
    "env.stage1_arm_max_accel": 10.0,
    "env.stage1_arm_max_vel": 5.0,
    "env.stage1_arm_init_dof_pos_noise": 1.0,
    # domain randomization: stage-1 arm (incl. EE payload)
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
    "domain_rand.stage1_arm.randomize_ee_payload": True,
    "domain_rand.stage1_arm.ee_payload_mass_range": [0.0, 1.5],
    # Per-axis half-width, in the EE frame, of the offset between the grasp
    # point and the payload's centre of mass (global invariant 12).  Sized on
    # the objects this arm is meant to carry: a 10 cm reach along the tool
    # axis, less across it.
    "domain_rand.stage1_arm.ee_payload_com_offset_range": [0.10, 0.05, 0.05],
}


# ============================================================
# STAGE 2 (shared substrate): both stage-2 submodes below -- legacy IK box
# reaching and --goal_reaching -- sit on top of this. The DLS-IK
# controller, EE tracking reward/sigma and WBC termination are identical
# in both submodes; only target *sampling* differs (see the two blocks
# further down).
# ============================================================
STAGE2_OVERRIDES = {
    # ---- arm action interface: what the actor's first 6 action dims MEAN ----
    # All three modes keep the same 6-wide arm action head (and therefore the
    # same obs/action layout and checkpoint shapes); only the decoding from
    # action to joint position target differs.
    #   'ik_residual'  -- (default, legacy) DLS-IK drives the EE to the task
    #                     target and the action is a per-joint delta_q residual
    #                     scaled by arm.ik.residual_scale.
    #   'ik_waypoint'  -- the action IS an intermediate EE waypoint (dpos(3),
    #                     axis-angle drot(3), base frame) offset from
    #                     arm.waypoint.anchor; DLS-IK solves for THAT pose
    #                     instead of the task target. No joint residual.
    #   'end_to_end'   -- no IK at all: the action is the arm joint position
    #                     target in the usual (target - default)/scale form.
    "arm.action_mode": "end_to_end",
    # 'ik_waypoint' only. anchor='target' makes the action a bounded detour
    # around the task/trajectory reference (zero action == the pure-IK
    # baseline, which is what the termination thresholds are calibrated
    # against); anchor='ee' makes it a bounded EE displacement command from
    # where the EE is now, i.e. the policy fully owns the EE path and IK is
    # only the velocity resolver.
    "arm.waypoint.anchor": "target",
    # PER-AXIS half-widths of the tanh-bounded offset box (m / rad), in the
    # BASE frame -- not a radius. The offset's norm reaches sqrt(3)x these.
    "arm.waypoint.pos_scale": 0.15,
    "arm.waypoint.rot_scale": 0.50,
    # 'end_to_end' only: joint-target scale for the arm slice, replacing the
    # shared control.action_scale (0.25) that the legs use.
    "arm.end_to_end.action_scale": 0.25,
    # arm DLS-IK controller (drives the arm in both 'ik_*' action modes and in
    # both stage-2 submodes; unused under 'end_to_end')
    "arm.ik.damping": 0.1,
    "arm.ik.step_gain": 1.0,
    "arm.ik.max_step_rad": 0.5,
    "arm.ik.residual_scale": 0.07,
    "arm.ik.pos_weight": 1.0,
    "arm.ik.rot_weight": 3.0,
    "arm.ik.ee_local_pos": [0.1424, 0.0, 0.0001057],
    # rewards: WBC / arm task-space
    "wbc.use_vision": False,
    "rewards.ee_pos_tracking_sigma": 0.02,
    "rewards.ee_rot_tracking_sigma": 0.25,
    "wbc.reward_scales.ee_pos_tracking": 3.0,
    "wbc.reward_scales.ee_rot_tracking": 1.0,
    "wbc.reward_scales.arm_control_limits": -0.0001,
    "wbc.reward_scales.ee_smoothness": -1e-4,
    "wbc.reward_scales.arm_contact": -1.0,
    "wbc.reward_scales.jump": 5.0,
    "wbc.reward_scales.hip_action_l2": -0.05,
    "wbc.reward_scales.raibert_heuristic": -1.0,
    "wbc.rewards.terminal_body_height": 0.17,
    "wbc.rewards.use_terminal_body_height": True,
    "wbc.rewards.use_terminal_roll": False,
    "wbc.rewards.use_terminal_pitch": False,
    "wbc.rewards.terminal_body_roll": 0.10,
    "wbc.rewards.terminal_body_pitch": 0.2,
    # domain randomization: stage-2 arm (tighter than stage-1 -- stage-2
    # wants cm-level precision, see docs/PARAM_TUNING.md §11)
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


# Also stage-2-shared, but NOT an override dict like the sections above --
# core._derive_wbc_rewards(cfg, WBC_REWARD_FACTORS) multiplies each factor by
# the matching cfg.reward_scales.* (dog/pretrained) value to derive
# cfg.wbc.reward_scales.{tracking_lin_vel,tracking_ang_vel,arm_energy,
# arm_dof_vel,arm_dof_acc,arm_action_rate,arm_action_smoothness_1,_2} at
# build time, so the arm-side reward stays proportional to its dog-side
# counterpart instead of needing separate hand-tuning. It can't be a
# "wbc.reward_scales.X": value entry in STAGE2_OVERRIDES because the value
# isn't a literal -- it's a ratio applied to another config field.
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


# ============================================================
# STAGE 2 - IK submode (default, no --goal_reaching): absolute box SE(3)
# EE target sampled in the base frame every resample, tracked purely by
# DLS-IK (STAGE2_OVERRIDES above) + a small policy residual. This is the
# ONLY thing exclusive to this submode -- everything else it needs is
# shared STAGE2_OVERRIDES.
# ============================================================
STAGE2_IK_OVERRIDES = {
    "arm.target.pos_range": [[0.0, 0.55], [-0.4, 0.4], [0.25, 0.9]],
    "arm.target.roll_ee": [-math.radians(20.0), math.radians(20.0)],
    "arm.target.pitch_ee": [-math.radians(20.0), math.radians(20.0)],
    "arm.target.yaw_ee": [-math.radians(20.0), math.radians(20.0)],
    "arm.target.resample_time_s": [2.0, 3.0],
}


# ============================================================
# GOAL REACHING (--goal_reaching) submode: static world-frame SE(3) goal,
# upper actor outputs Δq_arm(6) + Δv_base(3) + posture(3). Params AND
# reward weights that only matter for this submode live together in this
# section -- tune GOAL_REACHING_OVERRIDES / GOAL_REACHING_REWARD_SCALES,
# nothing else.
#
# GOAL_REACHING_OVERRIDES (the params) is folded into the always-applied
# ROBODUET_OVERRIDES below, because WBCEnv._resample_arm_target unconditionally
# evaluates `self.cfg.arm.target`/reads `getattr(self.cfg.wbc, "goal_reaching",
# None)` every resample regardless of the flag, and code that actually acts on
# goal-reaching state (WBCEnv.plan(), _update_goal_reaching_diagnostics()) is
# itself gated behind _goal_reaching_enabled(), so the section is cheap and
# harmless to keep always-present.
#
# GOAL_REACHING_REWARD_SCALES (the weights) is ALSO folded in, at 0.0 --
# same pattern as e.g. wbc.reward_scales.raibert_heuristic elsewhere in this
# file. That's safe, not wasteful: LeggedRobot._prepare_reward_function drops
# any wbc.reward_scales.* entry that's exactly 0 before registering _reward_*
# callbacks, so _reward_goal_pos_l2/_reward_reachability_barrier/etc. are
# simply never called while the scale sits at its 0.0 default. core.
# enable_goal_reaching() overwrites these with real values only when
# --goal_reaching is passed, which is what actually makes them live.
# ============================================================
GOAL_REACHING_OVERRIDES = {
    "wbc.goal_reaching.enabled": False,
    # x/y are a body-relative offset sampled at goal resample time; z is an
    # absolute world-frame height (e.g. [0.25, 0.90] = 0.25m-0.90m above the
    # ground), independent of the robot's own height.
    "wbc.goal_reaching.pos_range": [[-0.4, 0.4], [-0.2, 0.2], [0.05, 1.2]],
    "wbc.goal_reaching.roll_ee": [-math.radians(20.0), math.radians(20.0)],
    "wbc.goal_reaching.pitch_ee": [-math.radians(20.0), math.radians(20.0)],
    "wbc.goal_reaching.yaw_ee": [-math.radians(20.0), math.radians(20.0)],
    "wbc.goal_reaching.resample_time_s": [4.0, 6.0],
    "wbc.goal_reaching.delta_vel_limit": [0.30, 0.30, 0.60],
    "wbc.goal_reaching.rho_star": 0.60,
    "wbc.goal_reaching.rho_lo": 0.35,
    "wbc.goal_reaching.rho_hi": 0.85,
    # Direction-dependent reach R_max(u) (M2). The path is filled in per robot
    # by core.configure_robot_asset; build the file with
    # scripts/build_reach_table.py. Set to "" to force the legacy sphere.
    "wbc.goal_reaching.reach_table_path": "",
    # Fallback reach when no table is loaded: rho degrades to r / reach_radius,
    # i.e. a direction-independent sphere. Measured against the real go2_x5
    # envelope this is off by up to 57% (0.26 m straight down vs 0.85 m up),
    # and being constant it makes rho nearly blind to base pitch/height --
    # the very channel §4.5 expects to steer rho with.
    "wbc.goal_reaching.reach_radius": 0.60,
    "wbc.goal_reaching.response_time_s": 0.50,
    "wbc.goal_reaching.base_nom_filter_hz": 1.50,
    "wbc.goal_reaching.command_smoothing_alpha": 0.20,
    # When True, stage2→stage1 plan commands are passed through directly:
    # no high-speed posture scaling, no rate limiting, no low-pass filtering
    # (command smoothing + base feedforward filter both become identity).
    "wbc.goal_reaching.bypass_post_processing": True,
    "wbc.goal_reaching.command_channels.vx": True,
    "wbc.goal_reaching.command_channels.vy": True,
    "wbc.goal_reaching.command_channels.yaw": True,
    "wbc.goal_reaching.command_channels.height": True,
    "wbc.goal_reaching.command_channels.pitch": True,
    "wbc.goal_reaching.command_channels.roll": True,
    "wbc.goal_reaching.command_channels.gait_freq": False,
    "wbc.goal_reaching.command_channels.stance_width": False,
    "wbc.goal_reaching.command_channels.stance_length": False,
    "wbc.goal_reaching.posture_rate_limit": [0.05, 0.10, 0.10],
    "wbc.goal_reaching.gait_rate_limit": [0.50, 0.05, 0.05],
    "wbc.goal_reaching.high_speed_posture_scale": 0.50,
    "wbc.goal_reaching.high_speed_threshold": 0.80,
    "wbc.goal_reaching.fixed_gait_frequency": 4.0,
    "wbc.goal_reaching.fixed_footswing_height": 0.06,
    "wbc.goal_reaching.fixed_stance_width": 0.35,
    "wbc.goal_reaching.success_pos_threshold": 0.05,
    "wbc.goal_reaching.success_rot_threshold": 0.25,
    "wbc.goal_reaching.stay_sector_radius": 0.60,
    "wbc.goal_reaching.stay_sector_half_angle": math.radians(90.0),
    # 'static' = the classic discrete-goal-sequence behavior; 'trajectory' =
    # the moving SE(3) path tracking mode (set by enable_traj_tracking).
    "wbc.goal_reaching.target_mode": "static",
    # Trajectory-mode defaults live here (always in the schema, even in static
    # mode) so a saved trajectory-run config round-trips through load_env's
    # snapshot restore without the subtree being dropped. enable_traj_tracking
    # only flips target_mode to 'trajectory'.
    "wbc.goal_reaching.trajectory.preview_horizon": 0.5,  # L_h (m)
    "wbc.goal_reaching.trajectory.preview_points": 9,  # K
    "wbc.goal_reaching.trajectory.update_s_window": 0.15,  # forward search window (m)
    "wbc.goal_reaching.trajectory.timing_tau": 0.10,  # timing deadzone (m of arc length)
    # body-frame point the origin-centered path is translated to at reset (a
    # comfortable reachable spot in front of the shoulder): rho_star*reach in
    # front, slightly up.
    "wbc.goal_reaching.trajectory.anchor_offset_body": [0.36, 0.0, 0.10],
    # trajectory bank / batch sizing (see modules/curriculum.py TrajectoryBank).
    # max_gamma_points bounds L/ds_grid; hardest cell L~10m at ds_grid=0.01 with
    # per-sample variation, so 1536 leaves headroom.
    "wbc.goal_reaching.trajectory.max_gamma_points": 1536,
    "wbc.goal_reaching.trajectory.max_tl_points": 512,
    "wbc.goal_reaching.trajectory.bank_per_cell": 64,
    # M10 curriculum grid
    "wbc.goal_reaching.trajectory.n_levels_A": 6,
    "wbc.goal_reaching.trajectory.n_levels_B": 6,
    "wbc.goal_reaching.trajectory.curriculum_ema_alpha": 0.05,
    "wbc.goal_reaching.trajectory.curriculum_success_threshold": 0.70,
    "wbc.goal_reaching.trajectory.curriculum_fail_threshold": 0.30,
    # episode-success criteria (per design doc M10)
    "wbc.goal_reaching.trajectory.success_progress": 0.80,  # traversed fraction of L
    "wbc.goal_reaching.trajectory.success_dlat": 0.08,  # mean lateral err (m)
    "wbc.goal_reaching.trajectory.success_timing": 0.15,  # mean |timing_err| (m)
    "wbc.goal_reaching.trajectory.ik_jump_threshold": 0.50,  # max ||delta_q_ik|| (rad)
    # early-termination thresholds (design doc §8.4). Set an entry to 0 to
    # disable that one condition. These are NOT a performance bar -- they
    # declare an episode unrecoverable, so that samples whose s projection
    # (and every traj_* quantity derived from it) has stopped meaning anything
    # don't keep filling the buffer. The success_* criteria above are what
    # score tracking quality; these are an order of magnitude looser.
    #
    # Calibrated against the zero-action pure-DLS-IK baseline (no policy
    # residual, easiest curriculum cell, 256 envs x 375 steps), whose
    # distributions are:
    #     d_lat   p50 0.15  p90 0.60  p99 1.03  max 1.35
    #     timing  p50 0.13  p90 0.62  p99 1.35  max 2.09
    # Anything at or below that baseline's p90 would terminate the reference
    # controller itself, so each threshold sits at roughly its p99.
    #
    # d_lat is absolute because it is a spatial error: 1.00 m ~ 1.67 *
    # reach_radius, which means the same thing in every curriculum cell.
    "wbc.goal_reaching.trajectory.terminate_d_lat": 1.00,  # lateral err (m)
    # timing is a FRACTION of the path length, not metres. |s - s_ref| is an
    # arc length bounded by L, and L spans 2.0 m (easiest cell) to 9.0 m
    # (hardest) -- a fixed metre threshold would mean 75% of the path on the
    # easy end and 17% on the hard end, i.e. effectively disabled early and
    # strict late, by accident rather than by design. As a fraction it is
    # cell-independent, and because the time law normalizes L to T seconds,
    # timing_err / L is exactly "fraction of the episode's duration behind
    # schedule": 0.70 ~ 5.6 s of lag at the default T = 8 s, in every cell.
    # 0.70 is the same ~p99-of-baseline calibration as terminate_d_lat (the
    # baseline's p99 lag is 1.35 m on the easiest cell, whose L is 2.0 m).
    "wbc.goal_reaching.trajectory.terminate_timing_frac": 0.70,
    # Grace period after a reset during which none of the above fire. The arm
    # starts the episode wherever the reset pose left it, not on the path, so
    # d_lat is legitimately large for the first fraction of a second; without
    # this the env could reset-loop instead of ever running an episode.
    "wbc.goal_reaching.trajectory.terminate_grace_s": 0.5,
}

GOAL_REACHING_REWARD_SCALES = {
    "goal_pos_l2": -2.0,
    "reachability_barrier": -0.2,
    "manipulability": 0.05,
    "joint_limit_barrier": -0.02,
    "arm_ema_motion": -0.05,
    "rho_rate": -0.02,
    "upper_action_rate": -0.02,
    "delta_vel_magnitude": -0.05,
    "posture_command_rate": -0.02,
    "stay_still_in_reach_sector": -0.05,
}

# ============================================================
# Trajectory-tracking sub-mode (target_mode='trajectory').
# Applied by enable_traj_tracking() AFTER enable_goal_reaching(). The
# trajectory.* defaults already live in GOAL_REACHING_OVERRIDES (so they're
# always in the schema and round-trip through load_env); this only flips the
# mode switch.
# ============================================================
TRAJ_TRACKING_OVERRIDES = {
    "wbc.goal_reaching.target_mode": "trajectory",
}

# Group T tracking rewards (new) + the reachability/smoothness terms carried
# over unchanged from goal_reaching. goal_pos_l2 and stay_still_in_reach_sector
# are explicitly zeroed here: they're superseded by traj_lateral_err / the base
# should move to track. Primary tracking term (traj_lateral_err) kept at the
# same magnitude goal_pos_l2 had, so this drops into the existing tuning.
TRAJ_TRACKING_REWARD_SCALES = {
    "goal_pos_l2": 0.0,
    "stay_still_in_reach_sector": 0.0,
    "traj_progress": 1.0,
    "traj_lateral_err": -2.0,
    "traj_timing": -0.5,
    "traj_twist_err": -0.2,
    "reachability_barrier": -0.2,
    "manipulability": 0.05,
    "joint_limit_barrier": -0.02,
    "arm_ema_motion": -0.05,
    "rho_rate": -0.02,
    "upper_action_rate": -0.02,
    "delta_vel_magnitude": -0.05,
    "posture_command_rate": -0.02,
}


# ============================================================
# R2 -- prescribed reference-response model.
#
# omega_n / rate_limit are STARTING VALUES from the requirements document.
# R8.2's calibration measures the 20th percentile of what a stage-1 policy can
# actually sustain across the full randomisation range and overwrites them; a
# reference the plant cannot realise in the hard domains forces the policy to
# choose between missing the reference and going unstable, which is the main
# way this method loses robustness.
#
# cmd_index is NOT here on purpose: it is a fact about the commands_dog layout
# (go1_gym/response/command_layout.py), not a tunable.
# ============================================================
RESPONSE_MODEL_OVERRIDES = {
    "response.channel_order": ["vx", "vy", "wyaw", "height", "pitch"],
    "response.omega_n": {"vx": 8.0, "vy": 6.0, "wyaw": 8.0, "height": 7.0, "pitch": 5.0},
    "response.rate_limit": {"vx": 1.2, "vy": 0.8, "wyaw": 3.0, "height": 0.25, "pitch": 0.8},
    # ---- R3: gait-phase-conditioned residual estimate -----------------------
    # Speed binning is an R3 invariant, not a refinement: the oscillation's
    # amplitude and shape change strongly with travel speed, and forcing one
    # shape across all speeds is physically unrealisable -- the dependent reward
    # would degrade into a constant negative floor.
    "response.residual.speed_bin_edges": [0.15, 0.35, 0.6],
    "response.residual.num_phase_bins": 16,
    # Update-path low-pass. Long enough to leave the ~3 Hz gait ripple in the
    # residual (attenuated only ~11% at tau=0.5 s), short enough to track the
    # slow trend. Its lag is confined to this path and never reaches a reward.
    "response.residual.lowpass_tau_s": 0.5,
    # In units of *visits to a bin*, and a bin is visited about once per gait
    # cycle -- so 15 cycles is ~5 s at 3 Hz, comfortably longer than the gait
    # period as R3 requires.
    "response.residual.estimate_cycles": 15.0,
    # Convergence gate for the rewards that consume the estimate (R3: they must
    # stay off until it has converged).
    "response.residual.min_cycles": 45.0,
    # Steps after a reset during which samples are discarded and the low-pass is
    # snapped: the robot is still settling and the filter holds a stale level.
    "response.residual.warmup_steps": 25,
    # ---- R4: reward terms ---------------------------------------------------
    # sigma is CALIBRATED, never guessed: scripts/calibrate_reward_sigma.py
    # solves for the value that makes an "omega_n doubled" candidate differ from
    # the reference by 45% of the largest achievable integrated-reward gap.
    # Re-run it whenever omega_n, rate_limit or the command ranges change.
    "response.reward.sigma": {
        "vx": 0.1546,
        "vy": 0.0870,
        "wyaw": 0.2883,
        "height": 0.0921,
        "pitch": 0.1152,
    },
    "response.reward.channel_weights": {
        "vx": 1.0,
        "vy": 1.0,
        "wyaw": 1.0,
        "height": 1.0,
        "pitch": 1.0,
    },
    # Step size each channel is calibrated at: the half-width of its sampling
    # range. Kept in config so the calibration script and the training run
    # cannot disagree about what "a representative step" means.
    "response.reward.calibration_amplitudes": {
        "vx": 0.5,
        "vy": 0.3,
        "wyaw": 1.0,
        "height": 0.25,
        "pitch": 0.4,
    },
    # R4.2 is not counted for 2/omega_n after a command step, R4.3 not for
    # 3/omega_n. Per channel, because omega_n differs -- pitch is deliberately
    # the slowest and therefore stays masked longest.
    "response.reward.phase_variance_settle_factor": 2.0,
    "response.reward.steady_gain_settle_factor": 3.0,
    # R4.1's soft target: consistency is dropped for this long after a large
    # disturbance or a bad slip, so the policy is never asked to trade stability
    # for predictability.
    "response.reward.soft_gate_hold_s": 1.0,
    "response.reward.soft_gate_pose_error_rad": 0.25,
    "response.reward.soft_gate_ang_vel_rad_s": 1.5,
    # Tail thresholds, not typical values. Each trigger holds the gate shut for
    # 25 steps, so a trigger firing on x% of steps shuts the gate for ~25x% of
    # the time -- measured: a 0.5 m/s slip threshold fires on 2.05% of steps and
    # ends up shutting the gate 36% of the time, which would leave the
    # consistency term inactive most of the time. Measured slip percentiles
    # while walking: p50 0.057, p90 0.139, p99 0.740, p99.9 1.626 m/s (the tail
    # is touchdown/liftoff transients, not real slipping), and horizontal
    # acceleration p99.9 9.6 m/s^2.
    "response.reward.soft_gate_slip_speed": 1.5,
    "response.reward.soft_gate_accel": 20.0,
    # ---- R8.2's miscalibration alarm ---------------------------------------
    # A reference model calibrated without binning the posture channels by gait
    # phase is achievable at average phase and unachievable at the unfavourable
    # ones, and R8.2 says the signature of that is a ripple in the reward AT THE
    # GAIT FREQUENCY. It is measured per environment on the per-STEP R4.1 term
    # (see gait_frequency_ripple_batch): the per-iteration reward curve averages
    # over thousands of environments that are not phase-locked to each other, so
    # the ripple is gone twice over before it reaches the logger.
    #
    # 128 steps = 2.56 s = ~8 gait cycles at 3 Hz, with a 0.39 Hz bin. Shorter
    # windows cannot resolve the 2.5-3.5 Hz band the frequency is sampled from;
    # longer ones are harder to keep uninterrupted, since a reset, a shut soft
    # gate or a stretch of standing all void the window.
    "response.diagnostics.ripple_window_steps": 128,
    "response.diagnostics.ripple_tolerance": 0.15,
    # A warning threshold, not a failure threshold. Chance level for "the peak
    # happens to land in the band" is roughly one bin in the 0-25 Hz half-band,
    # so a sustained 0.5 is far above coincidence -- but it is an instruction to
    # go and re-run the calibration, never something the training loop acts on
    # by itself.
    "response.diagnostics.ripple_warn_fraction": 0.5,
    "response.diagnostics.ripple_warn_iterations": 20,
}


# ============================================================
# R6 -- rich excitation for the identification environments.
#
# The original setup resamples commands every 10 s into a 20 s episode: one
# command step per episode, which is nowhere near enough transient data to fit
# a reference model to, let alone to calibrate one (R8.2) or to draw a Bode
# plot from (R9).
#
# The two R6 invariants are enforced elsewhere, in LeggedRobot:
#   * identification envs are filtered out of curriculum.update() and
#     _update_reset_curriculum() -- otherwise their (deliberately) low tracking
#     reward would be read as "this command bin is too hard" and the curriculum
#     would shrink the command range for everybody;
#   * the curriculum's progress keys stay the four original tracking terms, and
#     _prepare_reward_function asserts no consistency reward ever takes one of
#     those names.
# ============================================================
RESPONSE_EXCITATION_OVERRIDES = {
    "response.excitation.enabled": True,
    # Tail block of the TRAINING envs (the eval envs sit past them and are left
    # alone).  25% is the requirements figure.
    "response.excitation.env_fraction": 0.25,
    "response.excitation.signal_weights": {"prbs": 0.5, "chirp": 0.3, "ramp": 0.2},
    # Posture channels get 55% of the draw between them against 45% for the
    # three velocity channels, i.e. ~3x the per-channel share.  The requirements
    # document singles them out: the original command distribution barely steps
    # body height or pitch at all, so they are the channels whose reference
    # model is least supported by data.
    "response.excitation.channel_weights": {
        "vx": 0.15,
        "vy": 0.15,
        "wyaw": 0.15,
        "height": 0.275,
        "pitch": 0.275,
    },
    # Straight from the requirements table.  Note the arithmetic: a hold drawn
    # from U(0.5, 3.0) s averages 1.75 s, so a 20 s episode gets ~11 PRBS
    # switches, not the >= 20 the acceptance criterion asks for.  The two
    # numbers in the requirements document are not simultaneously satisfiable
    # at a 20 s episode; the interval is kept as specified and the shortfall is
    # reported.  Narrowing this to [0.5, 1.5] is the one-line fix if the >= 20
    # figure turns out to be the binding one.
    "response.excitation.prbs_hold_s": [0.5, 3.0],
    "response.excitation.chirp_hz": [0.1, 2.0],
    # One full sweep per episode.  Anything shorter and the low-frequency end
    # of the sweep does not complete a single cycle.
    "response.excitation.chirp_duration_s": 20.0,
    # Fraction of the channel's rate limit the chirp is allowed to demand.  Below
    # 1.0 so the *reference model* never saturates during a sweep -- a saturated
    # reference turns the Bode measurement into a measurement of the saturation.
    "response.excitation.chirp_slew_fraction": 0.8,
    "response.excitation.ramp_slope_multiple": [0.2, 3.0],
}


# ============================================================
# R5 -- environment grouping and the nominal twin.
#
# A group shares its command sequence, gait phase clock and resample instants;
# its members differ only in domain (friction, mass, payload, motor strength,
# arm configuration).  One member -- the group leader -- is held at nominal
# values so the others have something to be consistent WITH.
#
# R5 states the choice of a twin over a within-group variance penalty as an
# invariant, and the reason is a degenerate solution: variance is minimised
# just as well by being equally sluggish everywhere, which collapses
# consistency and bandwidth together.
#
# The twin block is fixed for the run and cannot be reshuffled: arm link
# mass/COM, the mount-TF bucket and base mass are drawn once during
# _create_envs and never redrawn.
# ============================================================
RESPONSE_GROUPING_OVERRIDES = {
    "response.grouping.enabled": True,
    # 4 is the requirements figure. It buys 3 cross-domain comparisons per
    # nominal env; larger groups amortise the twin better but cut the number of
    # distinct command sequences the curriculum sees by the same factor.
    "response.grouping.group_size": 4,
    # How long a group stays uncomparable after any member resets.
    #
    # This replaces "latched off until the next shared resample", which was
    # correct but cost 0.61 availability for a policy that never falls -- the
    # mask was driven by episode timeouts (episode_length_s = 20 against
    # resampling_time = 10) rather than by falls, so it could not come down with
    # training.  Measured over the 20k run it sat at 0.71-0.82; the settling
    # window puts it near 0.85 open.  EnvGrouping's docstring has the
    # arithmetic.
    #
    # 1.0 s, not 0.5: the command and gait phase are re-adopted from the twin at
    # the reset instant, so what is being waited out is only the robot's own
    # start-up transient -- but the slowest decision channel is pitch at
    # omega_n = 5.0, i.e. 4/omega_n = 0.8 s to settle, and a window shorter than
    # that feeds the transient into the cross-domain comparison as if it were a
    # domain difference.
    "response.grouping.settle_s": 1.0,
    # Availability normalisation for R5's reward. tau is deliberately much
    # longer than the settling window so the gain is a slow schedule rather than
    # a second source of reward noise.
    "response.grouping.availability_tau_s": 20.0,
    # Ceiling on 1/availability. The loop is adverse -- falling lowers
    # availability, which would raise the penalty -- so the gain is allowed to
    # compensate down to 33% availability and no further.
    "response.grouping.availability_gain_max": 3.0,
}


# ============================================================
# R7 -- observations.
#
# The five decision channels' reference state is handed to the policy rather
# than left to be reconstructed.  The hard argument for it: the pitch channel's
# reference takes 4/omega_n = 0.8 s to settle, and the ORIGINAL history window
# was 30 x 0.02 = 0.6 s -- too short to contain one complete step response, so
# reconstructing xi from the command history was not merely hard, it was
# information-theoretically impossible.
#
# The history window goes to 50 steps (1.0 s) for the same reason, and because
# teacher-student was vetoed: domain identification now has to happen inside the
# actor's processing of this window, and 1.0 s is the shortest window that
# covers both jobs.  It is the upper end of the 0.5-1.0 s that R7.2 itself
# quotes for the proprioceptive history.
# ============================================================
RESPONSE_OBS_OVERRIDES = {
    "dog.dog_num_observation_history": 50,
    # (g, l) -- only for the channels with an absolute sensing anchor.  See the
    # deployment contract: pitch and yaw rate come from the IMU, whose gravity
    # reference does not drift.  vx/vy/height would need leg-odometry state
    # estimation, which is biased and degrades exactly when the robot slips --
    # and a 5 s EMA is the worst possible thing to feed a slow bias into.
    "response.deviation.channels": ["wyaw", "pitch"],
    # 5 s, against an MPC horizon of ~1 s.  Being 5x slower than the horizon is
    # what lets the planner treat these as constant PARAMETERS rather than as
    # states it has to predict.
    "response.deviation.tau_s": 5.0,
    "response.deviation.warmup_s": 2.0,
    # Below this fraction of the channel's rate limit the reference is not
    # moving enough for sign(xi_dot) to carry information.
    "response.deviation.rate_deadband": 0.05,
    # ...and below this fraction of a representative command amplitude (in RMS)
    # the channel has not been driven hard enough to identify a gain at all.
    "response.deviation.excitation_fraction": 0.1,
}


# ============================================================
# R8.1 -- the four-stage curriculum.
#
# The reward scales above are end-of-ramp TARGETS; this schedule supplies the
# multiplier that gets them there.  Expressing "off" as a multiplier rather than
# as a zero scale is what makes the ramp possible at all: a zero-scale term is
# never registered, so it could not be turned on later.
#
# R4.1 is gated to stage 2 even though it trains fine from iteration 0.  The
# reason is circularity, not stability: R8.2 calibrates the reference model from
# the stage-1 checkpoint, and a policy already pulled towards a reference model
# is not a neutral measurement of what the hardware can do.
# ============================================================
RESPONSE_CURRICULUM_OVERRIDES = {
    "response.curriculum.enabled": True,
    # Stage 1 ends / 2 ends / 3 ends, in iterations.
    "response.curriculum.stage_boundaries": [3000, 6000, 12000],
    # R8: "stage 3 is the one most likely to collapse, the weights must rise
    # slowly"; 1000 iterations is the figure it suggests.
    "response.curriculum.ramp_iterations": 1000,
    "response.curriculum.term_stage": {
        "ref_tracking": 2,
        "phase_variance": 3,
        "steady_gain": 3,
        "domain_consistency": 3,
    },
    # Faded DOWN over the same window R4.1 ramps up, so posture is never
    # unguarded.  Roll is deliberately absent: R4.1 has no roll channel.
    "response.curriculum.term_handover": {
        "pitch_control": 2,
        "jump": 2,
    },
    "response.curriculum.handover_floor": 0.15,
    # ---- R8.1's other two columns: randomisation and disturbance ------------
    # Stage 3 is where cross-domain consistency is first asked for, so it is
    # also where the domain ranges open fully.  The order is the point: a policy
    # that cannot yet walk on ice learns nothing from being asked to walk on ice
    # *consistently*.
    "response.curriculum.randomization_stage": 3,
    # Not zero. A policy trained on a single domain and then handed the full
    # range at stage 3 would have to relearn locomotion at the very moment it is
    # first asked for consistency -- which is the stage-3 collapse R8 warns
    # about, arriving by a second route.
    "response.curriculum.randomization_floor": 0.3,
    # Stage 4 = robustness recovery. Its disturbance must stay off while the
    # consistency weights are still moving, otherwise a robustness change and a
    # weight change land together and neither can be attributed.
    "response.curriculum.disturbance_stage": 4,
    # Friction/restitution/base payload reach the simulator only through a
    # per-env refresh call. Doing that every reset costs +94% per step
    # (measured 46.6 -> 90.6 ms at 4096 envs); doing it when the intensity has
    # actually moved costs 2.2 s a pass, about twenty times a run. So the
    # schedule is applied on transitions, and this is how big a move counts.
    "response.curriculum.refresh_threshold": 0.05,
}


ROBODUET_OVERRIDES = {
    **RESPONSE_MODEL_OVERRIDES,
    **RESPONSE_EXCITATION_OVERRIDES,
    **RESPONSE_GROUPING_OVERRIDES,
    **COMMON_OVERRIDES,
    **STAGE1_OVERRIDES,
    **STAGE2_OVERRIDES,
    **STAGE2_IK_OVERRIDES,
    **GOAL_REACHING_OVERRIDES,
    # Last, so it wins over the stage-1 table's history length of 30.
    **RESPONSE_OBS_OVERRIDES,
    **RESPONSE_CURRICULUM_OVERRIDES,
    **{f"wbc.reward_scales.{name}": 0.0 for name in {*GOAL_REACHING_REWARD_SCALES, *TRAJ_TRACKING_REWARD_SCALES}},
}


FEATURE_LAYOUT = {
    "rot6d_command_dims": 3,
    "dynamic_gait_command_dims": 5,
    "goal_reaching_plan_action_dims": 9,
}


ROBODUET_PROFILE = ConfigProfile(
    name="roboduet",
    overrides=ROBODUET_OVERRIDES,
    allow_new=True,
)
