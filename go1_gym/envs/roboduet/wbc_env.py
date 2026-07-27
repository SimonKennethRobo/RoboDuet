import cv2
import isaacgym

assert isaacgym
import numpy as np
import pytorch3d.transforms as pt3d
import torch
from isaacgym import gymapi, gymtorch, gymutil
from isaacgym.torch_utils import quat_apply, quat_from_euler_xyz, quat_mul, quat_rotate, to_torch, torch_rand_float

from go1_gym.envs.config import ConfigNode
from go1_gym.utils.global_switch import global_switch
from go1_gym.utils.math_utils import (
    ee_twist_body_6d,
    get_scale_shift,
    pose_world_to_body_9d,
    quat_conjugate,
    quat_error_axis_angle,
    quat_to_angle,
    quat_xyzw_to_rot6d,
)

from .legged_robot import LeggedRobot, quaternion_to_rpy
from .utils import ObservationBuilder, clip_observation
dog_cmd_idx = {
    "x_vel": 0,
    "y_vel": 1,
    "yaw_vel": 2,
    "body_pitch": 3,
    "body_roll": 4,
    "body_height": 5,
    "gait_frequency": 6,
    "footswing_height": 7,
    "stance_width": 8,
    "stance_length": 9,
    "gait_duration": 10,
    # named slices
    "velocity": slice(0, 3),
    "body_pose": slice(3, 6),
    "gait_params": slice(6, 11),
}

goal_plan_channel_names = ("vx", "vy", "yaw", "height", "pitch", "roll")


class WBCEnv(LeggedRobot):
    def __init__(
        self,
        sim_device,
        headless,
        num_envs=None,
        cfg: ConfigNode = None,
        eval_cfg: ConfigNode = None,
        initial_dynamics_dict=None,
        physics_engine="SIM_PHYSX",
        graphics_device_id=None,
    ):

        if num_envs is not None:
            cfg.env.num_envs = num_envs

        sim_params = gymapi.SimParams()
        gymutil.parse_sim_config(vars(cfg.sim), sim_params)
        super().__init__(
            cfg,
            sim_params,
            physics_engine,
            sim_device,
            headless,
            eval_cfg,
            initial_dynamics_dict,
            graphics_device_id,
        )

    # ============================================================
    # Geometry / EE helpers (formerly in LeggedRobot)
    # ============================================================

    def quat_to_angle(self, quat):
        return quat_to_angle(quat.to(self.device))

    def _pose_world_to_body_9d(self, pos_world, quat_world, env_ids=None):
        if env_ids is None:
            base_pos = self.base_pos
            base_quat = self.base_quat
        else:
            base_pos = self.base_pos[env_ids]
            base_quat = self.base_quat[env_ids]
        return pose_world_to_body_9d(pos_world, quat_world, base_pos, base_quat)

    def get_ee_pose_body_9d(self, env_ids=None):
        if env_ids is None:
            return self._pose_world_to_body_9d(self.end_effector_state[:, :3], self.end_effector_state[:, 3:7])
        return self._pose_world_to_body_9d(
            self.end_effector_state[env_ids, :3],
            self.end_effector_state[env_ids, 3:7],
            env_ids,
        )

    def get_ee_twist_body(self):
        return ee_twist_body_6d(self.end_effector_state, self.root_states, self.base_quat, self.num_envs)

    def _arm_target_pos_world(self, env_ids=None):
        """World-frame position of the current target.

        Legacy 6D reaching keeps a base-relative target. Whole-body goal
        reaching locks the sampled target in world coordinates so moving the
        base changes reachability instead of dragging the goal along.
        """
        if self._goal_reaching_enabled():
            return self.arm_goal_pos_world if env_ids is None else self.arm_goal_pos_world[env_ids]
        if env_ids is None:
            return self.base_pos + quat_apply(self.base_quat, self.arm_target_pos_body)
        return self.base_pos[env_ids] + quat_apply(self.base_quat[env_ids], self.arm_target_pos_body[env_ids])

    def _arm_target_quat_world(self, env_ids=None):
        if self._goal_reaching_enabled():
            return self.arm_goal_quat_world if env_ids is None else self.arm_goal_quat_world[env_ids]
        if env_ids is None:
            return quat_mul(self.base_quat, self.arm_target_quat_body)
        return quat_mul(self.base_quat[env_ids], self.arm_target_quat_body[env_ids])

    def _goal_reaching_enabled(self):
        goal_cfg = getattr(self.cfg.wbc, "goal_reaching", None)
        return bool(
            goal_cfg is not None
            and getattr(goal_cfg, "enabled", False)
            and getattr(self, "num_plan_actions", 0) == 6
        )

    def _traj_tracking_enabled(self):
        goal_cfg = getattr(self.cfg.wbc, "goal_reaching", None)
        return bool(
            self._goal_reaching_enabled()
            and getattr(goal_cfg, "target_mode", "static") == "trajectory"
        )

    def _get_object_pose_in_ee(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        xyz = self._arm_target_pos_world(env_ids)
        dxyz = xyz - self.end_effector_state[:, 0:3]
        self.obj_pose_in_ee[:] = quat_apply(quat_conjugate(self.end_effector_state[:, 3:7]), dxyz)
        return self.obj_pose_in_ee[:]

    def _get_object_abg_in_ee(self):
        rot_in_world = self._arm_target_quat_world()
        rot_in_ee = quat_mul(quat_conjugate(self.end_effector_state[:, 3:7]), rot_in_world)
        self.obj_abg_in_ee[:] = self.quat_to_angle(rot_in_ee)
        return self.obj_abg_in_ee[:]

    # ============================================================
    # Arm / trajectory command resampling
    # ============================================================

    def _resample_arm_target(self, env_ids):
        """Sample a static SE(3) target from the active task's independent box.

        Legacy arm reaching reads ``arm.target``. Whole-body goal reaching
        reads ``wbc.goal_reaching`` and then locks the sampled pose in world
        coordinates. No reachability gating is applied.

        For goal reaching, ``pos_range``'s x/y are still a body-relative
        offset (rotated into world by the base heading at sample time), but
        z is an absolute world-frame height -- e.g. [0.0, 0.5] samples a
        goal between 0m and 0.5m above the ground, independent of the
        robot's own height.
        """
        if len(env_ids) == 0:
            return
        target_cfg = self.cfg.arm.target
        goal_cfg = getattr(self.cfg.wbc, "goal_reaching", None)
        active_target_cfg = goal_cfg if self._goal_reaching_enabled() else target_cfg
        pos_range = torch.tensor(active_target_cfg.pos_range, dtype=torch.float, device=self.device)
        rand_pos = torch.rand((len(env_ids), 3), device=self.device)
        self.arm_target_pos_body[env_ids] = pos_range[:, 0] + (pos_range[:, 1] - pos_range[:, 0]) * rand_pos

        roll = torch_rand_float(
            active_target_cfg.roll_ee[0],
            active_target_cfg.roll_ee[1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(-1)
        pitch = torch_rand_float(
            active_target_cfg.pitch_ee[0],
            active_target_cfg.pitch_ee[1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(-1)
        yaw = torch_rand_float(
            active_target_cfg.yaw_ee[0],
            active_target_cfg.yaw_ee[1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(-1)
        zero_vec = torch.zeros_like(roll)
        q1 = quat_from_euler_xyz(zero_vec, zero_vec, yaw)
        q2 = quat_from_euler_xyz(zero_vec, pitch, zero_vec)
        q3 = quat_from_euler_xyz(roll, zero_vec, zero_vec)
        self.arm_target_quat_body[env_ids] = quat_mul(q1, quat_mul(q2, q3)).reshape(-1, 4)

        if self._goal_reaching_enabled():
            xy_offset_world = quat_apply(self.base_quat[env_ids], self.arm_target_pos_body[env_ids])[:, :2]
            self.arm_goal_pos_world[env_ids, :2] = self.base_pos[env_ids, :2] + xy_offset_world
            self.arm_goal_pos_world[env_ids, 2] = self.arm_target_pos_body[env_ids, 2]
            self.arm_goal_quat_world[env_ids] = quat_mul(
                self.base_quat[env_ids], self.arm_target_quat_body[env_ids]
            )
            # The new goal is a discontinuity, not physical rho motion.
            # Re-initialize rho_rate on the next diagnostic update.
            self.goal_rho_valid[env_ids] = False

        resample_lo, resample_hi = active_target_cfg.resample_time_s
        self.arm_target_resample_steps[env_ids] = torch.randint(
            int(resample_lo / self.dt), int(resample_hi / self.dt) + 1, (len(env_ids),), device=self.device
        )
        self.arm_time_buf[env_ids] = 0

        if self.cfg.wbc.use_vision:
            self._get_object_pose_in_ee()
            self._get_object_abg_in_ee()

    def format_dog_commands(self) -> str:
        """One-line human-readable summary of the active dog commands."""
        c = self.commands_dog[0]
        n_cmd = self.commands_dog.shape[1]
        parts = [
            f"vx={float(c[0]):+.2f}",
            f"vy={float(c[1]):+.2f}",
            f"wz={float(c[2]):+.2f}",
        ]
        if n_cmd > 5:
            parts += [
                f"pitch={float(c[3]):+.2f}",
                f"roll={float(c[4]):+.2f}",
                f"dh={float(c[5]):+.3f}",
            ]
        if n_cmd > 6:
            parts.append(f"freq={float(c[6]):.2f}")
        if n_cmd > 10:
            parts += [
                f"swing={float(c[7]):.3f}",
                f"sw={float(c[8]):.3f}",
                f"sl={float(c[9]):.3f}",
                f"dur={float(c[10]):.2f}",
            ]
        return "  ".join(parts)

    @staticmethod
    def _map_unit_action(action, limits):
        lo, hi = float(limits[0]), float(limits[1])
        return 0.5 * (lo + hi) + 0.5 * (hi - lo) * torch.clamp(action, -1.0, 1.0)

    def _rate_limit_command(self, target, previous, max_delta):
        return previous + torch.clamp(target - previous, -float(max_delta), float(max_delta))

    def _smooth_goal_commands(self, indices, values):
        alpha = float(self.cfg.wbc.goal_reaching.command_smoothing_alpha)
        self.goal_command_targets[:, indices] = values
        smoothed = alpha * values + (1.0 - alpha) * self.goal_command_smoothed[:, indices]
        self.goal_command_smoothed[:, indices] = smoothed
        self.commands_dog[:, indices] = smoothed

    def _reset_goal_commands(self, env_ids):
        if not self._goal_reaching_enabled() or len(env_ids) == 0:
            return
        neutral = {
            dog_cmd_idx["x_vel"]: 0.0,
            dog_cmd_idx["y_vel"]: 0.0,
            dog_cmd_idx["yaw_vel"]: 0.0,
            dog_cmd_idx["body_pitch"]: 0.0,
            dog_cmd_idx["body_roll"]: 0.0,
            dog_cmd_idx["body_height"]: 0.0,
            dog_cmd_idx["gait_frequency"]: self.cfg.wbc.goal_reaching.fixed_gait_frequency,
            dog_cmd_idx["footswing_height"]: self.cfg.wbc.goal_reaching.fixed_footswing_height,
            dog_cmd_idx["stance_width"]: self.cfg.wbc.goal_reaching.fixed_stance_width,
            dog_cmd_idx["stance_length"]: 0.5 * sum(self.cfg.commands.limit_stance_length),
            dog_cmd_idx["gait_duration"]: 0.49,
        }
        for index, value in neutral.items():
            if index < self.commands_dog.shape[1]:
                self.commands_dog[env_ids, index] = value
        self.goal_command_targets[env_ids] = self.commands_dog[env_ids]
        self.goal_command_smoothed[env_ids] = self.commands_dog[env_ids]
        self.base_feedforward_cmd[env_ids] = 0.0
        self.delta_velocity_cmd[env_ids] = 0.0
        self.upper_plan_actions_raw[env_ids] = 0.0

    def _compute_point_goal_base_nom(self):
        cfg = self.cfg.wbc.goal_reaching
        mount_offset_body = self.arm_mount_tfs[:, :3]
        shoulder_world = self.base_pos + quat_apply(self.base_quat, mount_offset_body)
        displacement_world = self.arm_goal_pos_world - shoulder_world
        distance = displacement_world.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        direction_world = displacement_world / distance
        desired_reach = float(cfg.rho_star) * float(cfg.reach_radius)
        shoulder_target_world = self.arm_goal_pos_world - direction_world * desired_reach
        base_target_world = shoulder_target_world - quat_apply(self.base_quat, mount_offset_body)
        velocity_world = (base_target_world - self.base_pos) / max(float(cfg.response_time_s), 1e-3)
        velocity_world[:, 2] = 0.0

        rc = 1.0 / (2.0 * np.pi * max(float(cfg.base_nom_filter_hz), 1e-3))
        alpha = self.dt / (rc + self.dt)
        velocity_body = quat_apply(quat_conjugate(self.base_quat), velocity_world)
        target_body = quat_apply(quat_conjugate(self.base_quat), displacement_world)
        yaw_rate = torch.atan2(target_body[:, 1], target_body[:, 0]) / max(float(cfg.response_time_s), 1e-3)
        raw = torch.stack((velocity_body[:, 0], velocity_body[:, 1], yaw_rate), dim=-1)
        self.base_feedforward_cmd[:] = alpha * raw + (1.0 - alpha) * self.base_feedforward_cmd

    def plan(self, upper_action):
        """Apply the document-aligned upper-policy coordination channels.

        ``upper_action`` is either the full 12D actor output or its final 6D
        plan slice: dv(3), posture(h/pitch/roll). Gait frequency, swing height
        and stance width are fixed configurable commands.
        This must run before dog observations/inference for the current step.
        """
        if not self._goal_reaching_enabled():
            return
        if upper_action.shape[-1] == self.cfg.arm.num_actions_arm_cd:
            self.arm_policy_actions[:] = upper_action
            plan = upper_action[:, self.num_actions_arm :]
        elif upper_action.shape[-1] == self.num_plan_actions:
            plan = upper_action
            self.arm_policy_actions[:, self.num_actions_arm :] = plan
        else:
            raise ValueError(
                f"Expected {self.cfg.arm.num_actions_arm_cd} full or {self.num_plan_actions} plan actions, "
                f"got {upper_action.shape[-1]}"
            )

        plan = plan * self.goal_command_channel_mask
        self.arm_policy_actions[:, self.num_actions_arm :] = plan
        self.upper_plan_actions_raw[:] = plan
        bounded = torch.clamp(plan, -1.0, 1.0)
        self._compute_point_goal_base_nom()

        dv_limit = torch.tensor(
            self.cfg.wbc.goal_reaching.delta_vel_limit,
            dtype=self.commands_dog.dtype,
            device=self.device,
        ).view(1, 3)
        velocity_mask = self.goal_command_channel_mask[:, :3]
        self.delta_velocity_cmd[:] = bounded[:, :3] * dv_limit
        velocity_target = (self.base_feedforward_cmd + self.delta_velocity_cmd) * velocity_mask
        velocity_limits = (
            self.cfg.commands.limit_vel_x,
            self.cfg.commands.limit_vel_y,
            self.cfg.commands.limit_vel_yaw,
        )
        for column, limits in enumerate(velocity_limits):
            velocity_target[:, column].clamp_(float(limits[0]), float(limits[1]))
        self._smooth_goal_commands([0, 1, 2], velocity_target)

        posture_specs = (
            (dog_cmd_idx["body_height"], self.cfg.commands.limit_body_height),
            (dog_cmd_idx["body_pitch"], self.cfg.commands.limit_body_pitch),
            (dog_cmd_idx["body_roll"], self.cfg.commands.limit_body_roll),
        )
        speed = torch.linalg.vector_norm(self.commands_dog[:, :2], dim=-1)
        high_speed = torch.clamp(
            speed / max(float(self.cfg.wbc.goal_reaching.high_speed_threshold), 1e-3), 0.0, 1.0
        )
        posture_scale = 1.0 - high_speed * (1.0 - float(self.cfg.wbc.goal_reaching.high_speed_posture_scale))
        posture_values = []
        for column, (index, limits) in enumerate(posture_specs):
            if self.goal_command_channel_enabled[3 + column]:
                target = self._map_unit_action(bounded[:, 3 + column] * posture_scale, limits)
                target = self._rate_limit_command(
                    target,
                    self.goal_command_smoothed[:, index],
                    self.cfg.wbc.goal_reaching.posture_rate_limit[column],
                )
            else:
                target = torch.zeros_like(bounded[:, 3 + column])
            posture_values.append(target)
        self._smooth_goal_commands(
            [spec[0] for spec in posture_specs], torch.stack(posture_values, dim=-1)
        )

        fixed_commands = {
            dog_cmd_idx["gait_frequency"]: self.cfg.wbc.goal_reaching.fixed_gait_frequency,
            dog_cmd_idx["footswing_height"]: self.cfg.wbc.goal_reaching.fixed_footswing_height,
            dog_cmd_idx["stance_width"]: self.cfg.wbc.goal_reaching.fixed_stance_width,
        }
        for index, value in fixed_commands.items():
            self.goal_command_targets[:, index] = value
            self.goal_command_smoothed[:, index] = value
            self.commands_dog[:, index] = value

    # ============================================================
    # EE force / arm action curriculum
    # ============================================================

    def resample_force(self, env_ids):
        self.ee_forces[env_ids, self.ee_idx] = torch_rand_float(
            -self.cfg.domain_rand.max_force, self.cfg.domain_rand.max_force, (len(env_ids), 3), device=self.device
        )
        time_range = (self.cfg.commands.T_force_range[1] - self.cfg.commands.T_force_range[0]) / self.dt
        time_interval = torch.randint(
            0,
            int(time_range + 1),
            (len(env_ids),),
            device=self.device,
        ).to(dtype=self.T_force.dtype)
        self.T_force[env_ids] = (
            torch.ones_like(self.T_force[env_ids]) * self.cfg.commands.T_force_range[0] + time_interval * self.dt
        )
        self.force_time_buf[env_ids] = torch.zeros_like(self.force_time_buf[env_ids])
        self.add_force_flag[env_ids] = torch.rand_like(self.add_force_flag[env_ids])
        self.ee_forces[env_ids, self.ee_idx] *= (
            self.add_force_flag[env_ids] > self.cfg.commands.add_force_thres
        ).reshape(-1, 1)

    def add_continue_force(self):
        if not self.cfg.domain_rand.randomize_end_effector_force:
            return
        self.force_positions = self.rigid_body_state[..., :3].clone().reshape(self.num_envs, -1, 3)
        self.ee_forces[:, self.ee_idx] = 10
        self.ee_forces[:, 0] = 20

        offset = torch_rand_float(
            -self.cfg.domain_rand.max_force_offset,
            self.cfg.domain_rand.max_force_offset,
            (self.num_envs, 3),
            device=self.device,
        )
        self.force_positions[:, self.ee_idx] += offset

        assert self.gym.apply_rigid_body_force_at_pos_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.ee_forces.reshape(-1, 3)),
            gymtorch.unwrap_tensor(self.force_positions.reshape(-1, 3)),
        ), "Failed to apply force at position."

    def _stage1_arm_curriculum_active(self):
        return bool(getattr(self.cfg.env, "stage1_arm_curriculum", False)) and not global_switch.switch_open

    def _get_stage1_arm_curriculum_intensity(self):
        if not self._stage1_arm_curriculum_active():
            return 0.0
        play_intensity = getattr(self, "stage1_arm_play_intensity", None)
        if play_intensity is not None:
            return min(1.0, max(0.0, float(play_intensity)))
        ramp_iters = max(
            1,
            int(getattr(global_switch, "stage1_arm_ramp_iterations", global_switch.pretrained_to_wbc_start)),
        )
        stage1_iter = getattr(global_switch, "stage1_count", global_switch.count)
        progress = min(1.0, max(0.0, stage1_iter / ramp_iters))
        fixed_fraction = min(1.0, max(0.0, float(self.cfg.env.stage1_arm_fixed_fraction)))
        saturation_fraction = min(
            1.0, max(fixed_fraction, float(getattr(self.cfg.env, "stage1_arm_saturation_fraction", 1.0)))
        )
        if progress <= fixed_fraction:
            return 0.0
        if progress >= saturation_fraction:
            return 1.0
        return (progress - fixed_fraction) / max(1e-6, saturation_fraction - fixed_fraction)

    def _apply_stage1_arm_curriculum_actions(self):
        if not self._stage1_arm_curriculum_active():
            return
        intensity = self._get_stage1_arm_curriculum_intensity()
        self.stage1_arm_curriculum_intensity = intensity
        arm_slice = slice(self.num_actions_loco, self.num_actions_loco + self.num_actions_arm)
        arm_default = self.default_dof_pos[:, arm_slice]

        if intensity <= 0.0:
            self.stage1_arm_target_offset.zero_()
            self.stage1_arm_target_vel.zero_()
            self.stage1_arm_target_accel.zero_()
            # Set actions so pos_target matches stage1_arm_fixed_dof_pos.
            # pos_target = action_scale * action + default_dof_pos
            # action = (fixed_pos - default_arm) / action_scale
            self.actions[:, arm_slice] = (self.stage1_arm_fixed_dof_pos - arm_default) / self.cfg.control.action_scale
            return

        resample_steps = max(1, int(self.cfg.env.stage1_arm_accel_resample_time_s / self.dt))
        if self.common_step_counter % resample_steps == 0:
            accel = torch.rand_like(self.stage1_arm_target_accel) * 2.0 - 1.0
            self.stage1_arm_target_accel[:] = accel * self.cfg.env.stage1_arm_max_accel * intensity

        step_accel = self.stage1_arm_target_accel
        zero_accel_prob = min(1.0, max(0.0, float(getattr(self.cfg.env, "stage1_arm_zero_accel_probability", 0.0))))
        if zero_accel_prob > 0.0:
            use_zero_accel = torch.rand(self.num_envs, 1, device=self.device) < zero_accel_prob
            step_accel = torch.where(use_zero_accel, torch.zeros_like(step_accel), step_accel)

        self.stage1_arm_target_vel += step_accel * self.dt
        max_vel = self.cfg.env.stage1_arm_max_vel * intensity
        self.stage1_arm_target_vel[:] = torch.clamp(self.stage1_arm_target_vel, -max_vel, max_vel)

        zero_vel_prob = min(1.0, max(0.0, float(getattr(self.cfg.env, "stage1_arm_zero_vel_probability", 0.0))))
        if zero_vel_prob > 0.0:
            use_zero_vel = torch.rand(self.num_envs, 1, device=self.device) < zero_vel_prob
            self.stage1_arm_target_vel[:] = torch.where(
                use_zero_vel, torch.zeros_like(self.stage1_arm_target_vel), self.stage1_arm_target_vel
            )

        self.stage1_arm_target_offset += self.stage1_arm_target_vel * self.dt

        raw_target = arm_default + self.stage1_arm_target_offset
        lower = self.dof_pos_limits[arm_slice, 0].unsqueeze(0)
        upper = self.dof_pos_limits[arm_slice, 1].unsqueeze(0)
        hit_limit = ((raw_target < lower) & (self.stage1_arm_target_vel < 0.0)) | (
            (raw_target > upper) & (self.stage1_arm_target_vel > 0.0)
        )
        self.stage1_arm_target_vel[hit_limit] *= -1
        target = torch.clamp(raw_target, lower, upper)

        self.actions[:, arm_slice] = (target - arm_default) / self.cfg.control.action_scale

    def _resample_stage1_ee_payload(self, env_ids):
        """Sample a per-env EE payload mass for the episode. Scaled by the
        same stage1 curriculum intensity as the rest of the arm disturbance
        (see _get_stage1_arm_curriculum_intensity), so payload weight ramps
        in alongside arm motion/DR rather than jumping in at full strength."""
        stage1_arm_cfg = self.cfg.domain_rand.stage1_arm
        if not (
            self._stage1_arm_curriculum_active()
            and getattr(stage1_arm_cfg, "randomize_ee_payload", False)
        ):
            self.stage1_ee_payload_mass[env_ids] = 0.0
            return
        intensity = self._get_stage1_arm_curriculum_intensity()
        lo, hi = getattr(stage1_arm_cfg, "ee_payload_mass_range", [0.0, 0.0])
        hi = lo + (hi - lo) * intensity
        self.stage1_ee_payload_mass[env_ids] = torch_rand_float(
            lo, hi, (len(env_ids), 1), device=self.device
        ).squeeze(-1)

    def _apply_stage1_ee_payload_force(self):
        """Applies the sampled EE payload as a sustained downward force at
        the EE rigid body's current position, every physics substep (forces
        set via apply_rigid_body_force_at_pos_tensors only last one
        substep). A force at the moving EE reproduces the lever-arm-varying
        base disturbance a real carried payload would cause; IsaacGym only
        allows rigid-body *mass* edits at actor creation, so mass isn't a
        practical way to re-randomize this every episode."""
        stage1_arm_cfg = self.cfg.domain_rand.stage1_arm
        if not (
            self._stage1_arm_curriculum_active()
            and getattr(stage1_arm_cfg, "randomize_ee_payload", False)
        ):
            return
        self.stage1_payload_forces[:, self.ee_idx, 2] = -self.stage1_ee_payload_mass * 9.81
        self.stage1_payload_force_positions[:] = self.rigid_body_state[..., :3].clone().reshape(self.num_envs, -1, 3)
        assert self.gym.apply_rigid_body_force_at_pos_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.stage1_payload_forces.reshape(-1, 3)),
            gymtorch.unwrap_tensor(self.stage1_payload_force_positions.reshape(-1, 3)),
        ), "Failed to apply stage1 EE payload force."

    def _keep_arm_fixed(self):
        arm_slice = slice(self.num_actions_loco, self.num_actions_loco + self.num_actions_arm)

        if global_switch.switch_open:
            # Stage2: arm is controlled by policy, nothing to fix.
            return
        elif self._stage1_arm_curriculum_active():
            intensity = self._get_stage1_arm_curriculum_intensity()
            if intensity > 0.0:
                # Curriculum active: arm moves via position targets from _apply_stage1_arm_curriculum_actions.
                # Fix DOFs beyond the arm (none exist, but harmless).
                idx = self.num_actions_loco + self.num_actions_arm
                self.dof_pos[:, idx:] = self.default_dof_pos[:, idx:]
                self.dof_vel[:, idx:] = 0.0
            else:
                # Fixed phase: hold the per-env randomized reset position.
                self.dof_pos[:, arm_slice] = self.stage1_arm_fixed_dof_pos
                self.dof_vel[:, arm_slice] = 0.0
        else:
            # No curriculum: fix arm at per-env randomized reset position.
            self.dof_pos[:, arm_slice] = self.stage1_arm_fixed_dof_pos
            self.dof_vel[:, arm_slice] = 0.0

        ret = self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))
        assert ret, "[ERROR] Failed to set dof state."

    # ============================================================
    # LeggedRobot hook overrides
    # ============================================================

    def _arm_init_buffers_hook(self):
        self.arm_time_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.force_time_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.num_plan_actions = self.cfg.arm.num_actions_arm_cd - self.num_actions_arm
        if self.num_plan_actions not in (0, 6):
            raise ValueError(
                "Upper policy must use either the legacy 6D arm layout or the "
                f"12D goal-reaching layout; got {self.cfg.arm.num_actions_arm_cd} actions"
            )
        if self.num_plan_actions:
            channel_cfg = getattr(self.cfg.wbc.goal_reaching, "command_channels", None)
            self.goal_command_channel_enabled = tuple(
                bool(getattr(channel_cfg, name, True)) for name in goal_plan_channel_names
            )
            self.goal_command_channel_mask = torch.tensor(
                self.goal_command_channel_enabled,
                dtype=torch.float,
                device=self.device,
            ).view(1, -1)
        else:
            self.goal_command_channel_enabled = ()
            self.goal_command_channel_mask = torch.zeros(1, 0, dtype=torch.float, device=self.device)

        self.stage1_arm_target_offset = torch.zeros(
            self.num_envs, self.num_actions_arm, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.stage1_arm_target_vel = torch.zeros_like(self.stage1_arm_target_offset)
        self.stage1_arm_target_accel = torch.zeros_like(self.stage1_arm_target_offset)
        self.stage1_arm_curriculum_intensity = 0.0

        self.end_effector_state = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.ee_idx]

        # DLS-IK MVP (stage2, project-design-v3.md scope trimmed to this
        # round): static per-episode SE(3) target in the base frame, and the
        # IsaacGym Jacobian used to solve for it every control step.
        self.arm_target_pos_body = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.arm_target_quat_body = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.arm_target_quat_body[:, 3] = 1.0
        self.arm_goal_pos_world = torch.zeros_like(self.arm_target_pos_body)
        self.arm_goal_quat_world = torch.zeros_like(self.arm_target_quat_body)
        self.arm_goal_quat_world[:, 3] = 1.0
        self.arm_target_resample_steps = torch.full(
            (self.num_envs,), 10**9, dtype=torch.long, device=self.device, requires_grad=False
        )
        # Fixed EE grasp offset in the x5_link6 frame (URDF gripper_center; see
        # arm.ik.ee_local_pos). end_effector_state's position is shifted by this
        # every step so IK / reward / obs all track the actual grasp point.
        self.ee_local_offset = torch.tensor(
            self.cfg.arm.ik.ee_local_pos, dtype=torch.float, device=self.device
        ).unsqueeze(0)
        self.ee_pos_err = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.ee_rot_err_axis_angle = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        # Dog-facing summary of the arm target (pos_body(3) + axis_angle(3)
        # of arm_target_quat_body from identity), populated every step in
        # _update_ee_task_space_error. MUST stay 6-wide -- see
        # arm.arm_num_commands comment in wbc.py (stage1 checkpoint is frozen
        # on dog_num_observations=82).
        self.commands_arm_obs = torch.zeros(self.num_envs, self.cfg.arm.arm_num_commands, dtype=torch.float, device=self.device, requires_grad=False)
        # Raw (pre-IK-combine) policy action for the arm, kept only for the
        # arm_control_limits saturation reward -- see _apply_stage2_arm_ik_action.
        self.arm_residual_raw = torch.zeros(self.num_envs, self.num_actions_arm, dtype=torch.float, device=self.device, requires_grad=False)
        self.arm_policy_actions = torch.zeros(
            self.num_envs,
            self.cfg.arm.num_actions_arm_cd,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_arm_policy_actions = torch.zeros_like(self.arm_policy_actions)
        self.upper_plan_actions_raw = torch.zeros(
            self.num_envs, self.num_plan_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.base_feedforward_cmd = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.delta_velocity_cmd = torch.zeros_like(self.base_feedforward_cmd)
        self.goal_command_targets = self.commands_dog.clone()
        self.goal_command_smoothed = self.commands_dog.clone()
        self.arm_ema_motion = torch.zeros(
            self.num_envs, self.num_actions_arm, dtype=torch.float, device=self.device
        )
        self.goal_rho = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.goal_rho_prev = torch.zeros_like(self.goal_rho)
        self.goal_rho_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.goal_manipulability = torch.zeros_like(self.goal_rho)
        self.goal_joint_limit_distance = torch.zeros(
            self.num_envs, self.num_actions_arm, dtype=torch.float, device=self.device
        )

        # Jacobian MUST be acquired before the first gym.simulate() call --
        # acquiring it lazily later silently returns an all-zero tensor
        # forever (confirmed empirically; not documented). This hook runs
        # from _init_buffers(), before any reset()/simulate(), so it's safe
        # here. Shape (num_envs, num_bodies, 6, 6 + num_dof): row=body index
        # directly (floating base), cols = [6 free-root][num_dof joint cols
        # in dof_names order]. See docs/programming/tensors.rst "Jacobians".
        jac_tensor = self.gym.acquire_jacobian_tensor(self.sim, "anymal")
        self.jacobian = gymtorch.wrap_tensor(jac_tensor)

        # Object-relative-to-EE buffers, only populated/consumed when
        # cfg.wbc.use_vision is True (disabled by default; kept for that
        # legacy vision path, now sourced from arm_target_pos/quat_body).
        self.obj_obs_pose_in_ee = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.obj_pose_in_ee = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.obj_obs_abg_in_ee = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.obj_abg_in_ee = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)

        self.T_force = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.add_force_flag = torch.rand(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.ee_forces = torch.zeros_like(self.rigid_body_state[:, :3]).reshape(self.num_envs, -1, 3)
        self.force_positions = torch.zeros_like(self.rigid_body_state[:, :3]).reshape(self.num_envs, -1, 3)

        # Stage-1 simulated EE payload: per-env mass (kg) resampled every
        # episode in _resample_stage1_ee_payload, applied as a sustained
        # downward force at the EE body in _apply_stage1_ee_payload_force.
        self.stage1_ee_payload_mass = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.stage1_payload_forces = torch.zeros_like(self.rigid_body_state[:, :3]).reshape(self.num_envs, -1, 3)
        self.stage1_payload_force_positions = torch.zeros_like(self.rigid_body_state[:, :3]).reshape(self.num_envs, -1, 3)

        self.prev_ee_twist_body = torch.zeros(
            self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False
        )

        # Per-env arm target position when arm is fixed (intensity=0). Populated at reset
        # with randomized init noise so _keep_arm_fixed holds a varied pose, not always 0.
        arm_default = self.default_dof_pos[:, self.num_actions_loco : self.num_actions_loco + self.num_actions_arm]
        self.stage1_arm_fixed_dof_pos = arm_default.expand(self.num_envs, -1).clone()

        # Arm rigid-body domain rand buffers are initialized while actor body props
        # are processed in _create_envs(), so per-env link mass/COM randomization is
        # applied before the sim starts and is not rewritten during episode reset.
        if not hasattr(self, "arm_link_mass_scales"):
            arm_body_names = [n for n in self.body_names if any(k in n.lower() for k in ("x5_link", "zarx_body"))]
            self.arm_body_indices = [self.body_names.index(n) for n in arm_body_names]
            n_arm_bodies = len(self.arm_body_indices)
            props0 = self.gym.get_actor_rigid_body_properties(self.envs[0], self.actor_handles[0])
            self.arm_default_link_masses = torch.tensor(
                [props0[i].mass for i in self.arm_body_indices],
                dtype=torch.float,
                device=self.device,
            )
            self.arm_default_link_coms = torch.tensor(
                [[props0[i].com.x, props0[i].com.y, props0[i].com.z] for i in self.arm_body_indices],
                dtype=torch.float,
                device=self.device,
            )
            self.arm_link_mass_scales = torch.ones(self.num_envs, n_arm_bodies, dtype=torch.float, device=self.device)
            self.arm_link_com_offsets = torch.zeros(
                self.num_envs, n_arm_bodies, 3, dtype=torch.float, device=self.device
            )

        # See _dog_obs_layout / get_dog_observations.
        dog_obs_layout = self._dog_obs_layout()
        self.dog_obs_noise_scale_vec = torch.cat(
            [torch.ones(w) * scale for _, w, scale, _ in dog_obs_layout if w > 0]
        ).to(self.device)
        self.dog_obs_droppable_segments = []
        cursor = 0
        for _, w, _, droppable in dog_obs_layout:
            if droppable and w > 0:
                self.dog_obs_droppable_segments.append((cursor, cursor + w))
            cursor += w
        self.dog_last_delivered_obs = torch.zeros(
            self.num_envs, self.cfg.dog.dog_num_observations, dtype=torch.float, device=self.device
        )

        self._traj_tracking_init_hook()


    def _traj_tracking_init_hook(self):
        """Build the trajectory batch, pre-generated bank and M10 curriculum,
        and preallocate the per-env progress/error buffers. Inert unless
        target_mode='trajectory'."""
        if not self._traj_tracking_enabled():
            return
        from modules.trajectory import TrajectoryBatch, mat_to_quat
        from modules.curriculum import CurriculumManager, TrajectoryBank

        self._traj_mat_to_quat = mat_to_quat
        tcfg = self.cfg.wbc.goal_reaching.trajectory
        self._traj_max_g = int(tcfg.max_gamma_points)
        self._traj_max_t = int(tcfg.max_tl_points)

        self.traj_curriculum = CurriculumManager(
            self.num_envs, self.device,
            n_levels_A=int(tcfg.n_levels_A), n_levels_B=int(tcfg.n_levels_B),
            success_threshold=float(tcfg.curriculum_success_threshold),
            fail_threshold=float(tcfg.curriculum_fail_threshold),
            ema_alpha=float(tcfg.curriculum_ema_alpha),
        )
        self.traj_bank = TrajectoryBank(
            self.traj_curriculum, per_cell=int(tcfg.bank_per_cell),
            max_gamma_points=self._traj_max_g, max_tl_points=self._traj_max_t,
            device=self.device,
        )
        self.traj_batch = TrajectoryBatch(
            self.num_envs, self._traj_max_g, self._traj_max_t, device=self.device
        )

        z = lambda: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.traj_s = z()
        self.traj_s_prev = z()
        self.traj_sim_time = z()
        self.traj_d_lat = z()
        self.traj_sdot_meas = z()
        self.traj_timing_err = z()
        self.traj_twist_err = z()
        # episode success accumulators (mean over the episode)
        self.traj_dlat_sum = z()
        self.traj_timing_abs_sum = z()
        self.traj_ik_jump_max = z()
        self.traj_samples = z()
        self._traj_anchor_offset = torch.tensor(
            list(tcfg.anchor_offset_body), dtype=torch.float, device=self.device
        ).view(1, 3)


    def _arm_pre_step_hook(self):
        self._apply_stage1_arm_curriculum_actions()
        self._apply_stage2_arm_ik_action()

    def _arm_decimation_hook(self):
        self.add_continue_force()
        self._apply_stage1_ee_payload_force()

    def _arm_post_sim_hook(self):
        if self.cfg.env.keep_arm_fixed:
            self._keep_arm_fixed()

    def _arm_post_physics_hook(self):
        self.arm_time_buf += 1
        self.force_time_buf += 1
        self.end_effector_state[:] = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.ee_idx]
        # Shift the tracked position from x5_link6's origin out to the grasp
        # point (gripper_center); orientation/velocity stay link6's.
        self.end_effector_state[:, :3] += quat_apply(
            self.end_effector_state[:, 3:7], self.ee_local_offset.expand(self.num_envs, -1)
        )
        if global_switch.switch_open:
            self.gym.refresh_jacobian_tensors(self.sim)
            if self._traj_tracking_enabled():
                self._advance_trajectory_target()
            self._update_ee_task_space_error()
            if self._goal_reaching_enabled():
                self._update_goal_reaching_diagnostics()
            if self._traj_tracking_enabled():
                self._update_trajectory_progress()
            self.prev_ee_twist_body[:] = self.get_ee_twist_body()
            if self._traj_tracking_enabled():
                # Trajectory mode ends by timeout/fall/completion, not by the
                # static per-timer/reach-success target resample.
                return
            due = self.arm_time_buf >= self.arm_target_resample_steps
            if self._goal_reaching_enabled():
                goal_cfg = self.cfg.wbc.goal_reaching
                reached = (
                    torch.linalg.vector_norm(self.ee_pos_err, dim=-1)
                    < float(goal_cfg.success_pos_threshold)
                ) & (
                    torch.linalg.vector_norm(self.ee_rot_err_axis_angle, dim=-1)
                    < float(goal_cfg.success_rot_threshold)
                )
                due |= reached
            if torch.any(due):
                self._resample_arm_target(due.nonzero(as_tuple=False).flatten())

    def _advance_trajectory_target(self):
        """Advance reference time and write the current moving SE(3) reference
        into arm_goal_pos/quat_world, which every downstream reader (v_ff,
        rho diagnostics, IK task error, obs) already consumes."""
        self.traj_sim_time += self.dt
        s_ref_now = self.traj_batch.s_ref(self.traj_sim_time)
        self.arm_goal_pos_world[:] = self.traj_batch.p_at(s_ref_now)
        self.arm_goal_quat_world[:] = self._traj_mat_to_quat(self.traj_batch.R_at(s_ref_now))

    def _update_trajectory_progress(self):
        """Update the EE's arc-length progress (forward-window projection) and
        the tracking error buffers consumed by the traj_* rewards / obs."""
        self.traj_s_prev[:] = self.traj_s
        from modules.trajectory import quat_to_mat, update_s_batch

        ee_R = quat_to_mat(self.end_effector_state[:, 3:7])  # xyzw
        self.traj_s[:], self.traj_d_lat[:] = update_s_batch(
            self.traj_s, self.end_effector_state[:, :3], ee_R, self.traj_batch,
            window=float(self.cfg.wbc.goal_reaching.trajectory.update_s_window),
        )
        self.traj_sdot_meas[:] = (self.traj_s - self.traj_s_prev) / self.dt
        s_ref_now = self.traj_batch.s_ref(self.traj_sim_time)
        self.traj_timing_err[:] = self.traj_s - s_ref_now

        # EE spatial (grasp-point) velocity in world vs. reference tangent*sdot_ref
        v_origin = self.end_effector_state[:, 7:10]
        w = self.end_effector_state[:, 10:13]
        d_world = quat_apply(self.end_effector_state[:, 3:7], self.ee_local_offset.expand(self.num_envs, -1))
        v_grasp = v_origin + torch.cross(w, d_world, dim=-1)
        tangent = self.traj_batch.tangent_at(self.traj_s)
        sdot_ref = self.traj_batch.sdot_ref(self.traj_sim_time)
        v_ref = tangent * sdot_ref.unsqueeze(-1)
        self.traj_twist_err[:] = torch.linalg.vector_norm(v_grasp - v_ref, dim=-1)

        # episode success accumulators
        arm_slice = slice(self.num_actions_loco, self.num_actions_loco + self.num_actions_arm)
        step_dq = torch.linalg.vector_norm(self.dof_vel[:, arm_slice] * self.dt, dim=-1)
        self.traj_ik_jump_max[:] = torch.maximum(self.traj_ik_jump_max, step_dq)
        self.traj_dlat_sum += self.traj_d_lat
        self.traj_timing_abs_sum += self.traj_timing_err.abs()
        self.traj_samples += 1.0

    def _arm_check_termination_hook(self):
        pass

    def _arm_reset_hook(self, env_ids):
        # Absolute box target: independent of the post-reset arm pose, so sample
        # it right here rather than deferring to the first post-physics step.
        if not self._goal_reaching_enabled():
            self._resample_arm_target(env_ids)
        # stage1_arm_target_offset / vel / accel are re-initialised in
        # _arm_post_reset_refresh_hook (after the randomised dof_pos is known).
        self._resample_stage1_ee_payload(env_ids)
        self.dog_last_delivered_obs[env_ids] = 0.0
        self.prev_ee_twist_body[env_ids] = 0.0

    def _ensure_arm_rigid_body_rand_buffers(self, props):
        if hasattr(self, "arm_link_mass_scales"):
            return
        arm_body_names = [n for n in self.body_names if any(k in n.lower() for k in ("x5_link", "zarx_body"))]
        self.arm_body_indices = [self.body_names.index(n) for n in arm_body_names]
        n_arm = len(self.arm_body_indices)
        self.arm_default_link_masses = torch.tensor(
            [props[i].mass for i in self.arm_body_indices],
            dtype=torch.float,
            device=self.device,
        )
        self.arm_default_link_coms = torch.tensor(
            [[props[i].com.x, props[i].com.y, props[i].com.z] for i in self.arm_body_indices],
            dtype=torch.float,
            device=self.device,
        )
        self.arm_link_mass_scales = torch.ones(self.num_envs, n_arm, dtype=torch.float, device=self.device)
        self.arm_link_com_offsets = torch.zeros(self.num_envs, n_arm, 3, dtype=torch.float, device=self.device)

    def _sample_arm_rigid_body_props(self, env_id):
        if not self.arm_body_indices:
            return
        arm_dr = self.cfg.domain_rand.stage1_arm if not global_switch.switch_open else self.cfg.domain_rand.stage2_arm
        n_arm = len(self.arm_body_indices)
        if arm_dr.randomize_link_mass:
            lo, hi = arm_dr.link_mass_range
            self.arm_link_mass_scales[env_id] = torch.rand(n_arm, device=self.device) * (hi - lo) + lo
        if arm_dr.randomize_link_com:
            r = arm_dr.link_com_range
            self.arm_link_com_offsets[env_id] = torch.rand(n_arm, 3, device=self.device) * 2 * r - r

    def _process_rigid_body_props(self, props, env_id):
        props = super()._process_rigid_body_props(props, env_id)
        self._ensure_arm_rigid_body_rand_buffers(props)
        self._sample_arm_rigid_body_props(env_id)

        for k, body_idx in enumerate(self.arm_body_indices):
            props[body_idx].mass = self.arm_default_link_masses[k].item() * self.arm_link_mass_scales[env_id, k].item()
            com = self.arm_default_link_coms[k] + self.arm_link_com_offsets[env_id, k]
            props[body_idx].com = gymapi.Vec3(com[0].item(), com[1].item(), com[2].item())
        return props

    def _arm_post_dof_reset_hook(self, env_ids):
        if len(env_ids) == 0:
            return

        # Add additive noise to arm joint positions before the single reset-time
        # set_dof_state_tensor_indexed() call in _reset_dofs().  Issuing a second
        # DOF write after root reset can desynchronise IsaacGym's articulation cache.
        noise = getattr(self.cfg.env, "stage1_arm_init_dof_pos_noise", 0.0)
        if noise <= 0.0:
            return

        arm_slice = slice(self.num_actions_loco, self.num_actions_loco + self.num_actions_arm)
        self.dof_pos[env_ids, arm_slice] += torch_rand_float(
            -noise,
            noise,
            (len(env_ids), self.num_actions_arm),
            device=self.device,
        )
        self.dof_pos[env_ids] = torch.clamp(
            self.dof_pos[env_ids],
            self.dof_pos_limits[:, 0],
            self.dof_pos_limits[:, 1],
        )

    def _arm_post_reset_refresh_hook(self, env_ids):
        if len(env_ids) == 0:
            return

        # Bookkeeping only.  Do not write DOF/root/rigid-body state here; reset root
        # pose must remain the final sim-state write for GUI and PhysX consistency.
        arm_slice = slice(self.num_actions_loco, self.num_actions_loco + self.num_actions_arm)
        self.stage1_arm_fixed_dof_pos[env_ids] = self.dof_pos[env_ids, arm_slice].clone()

        arm_default = self.default_dof_pos[:, arm_slice]  # (1, num_actions_arm)
        self.stage1_arm_target_offset[env_ids] = self.stage1_arm_fixed_dof_pos[env_ids] - arm_default
        self.stage1_arm_target_vel[env_ids] = 0.0
        self.stage1_arm_target_accel[env_ids] = 0.0
        if self._goal_reaching_enabled():
            if self._traj_tracking_enabled():
                self._reset_trajectories(env_ids)
            else:
                self._resample_arm_target(env_ids)
            self._reset_goal_commands(env_ids)
            self.arm_policy_actions[env_ids] = 0.0
            self.last_arm_policy_actions[env_ids] = 0.0
            self.arm_ema_motion[env_ids] = 0.0
            self.goal_rho[env_ids] = 0.0
            self.goal_rho_prev[env_ids] = 0.0
            self.goal_rho_valid[env_ids] = False

    def _reset_trajectories(self, env_ids):
        """For the reset envs: score the finished episode for the curriculum,
        pick a new (harder-as-mastered) cell, gather a fresh trajectory from
        the bank, anchor it to a reachable spot in front of the shoulder, and
        clear the per-env progress state."""
        if len(env_ids) == 0:
            return
        tcfg = self.cfg.wbc.goal_reaching.trajectory

        # 1. report the finished episode to the curriculum (train envs only,
        #    and only those that actually ran an episode). Uses the OLD cell,
        #    before sample_cells overwrites it.
        train_done = (env_ids < self.num_train_envs) & (self.traj_samples[env_ids] > 0)
        if torch.any(train_done):
            rep_ids = env_ids[train_done]
            samples = self.traj_samples[rep_ids].clamp_min(1.0)
            progress = self.traj_s[rep_ids] / self.traj_batch.L[rep_ids].clamp_min(1e-6)
            mean_dlat = self.traj_dlat_sum[rep_ids] / samples
            mean_timing = self.traj_timing_abs_sum[rep_ids] / samples
            success = (
                (progress > float(tcfg.success_progress))
                & (mean_dlat < float(tcfg.success_dlat))
                & (mean_timing < float(tcfg.success_timing))
                & (self.traj_ik_jump_max[rep_ids] < float(tcfg.ik_jump_threshold))
                & self.time_out_buf[rep_ids]  # finished by timeout, not a fall
            )
            self.traj_curriculum.report_result(rep_ids, success)

        # 2. sample a new cell for every reset env, gather a bank trajectory
        self.traj_curriculum.sample_cells(env_ids)
        rows = self.traj_bank.sample_rows(
            self.traj_curriculum.cell_A[env_ids],
            self.traj_curriculum.cell_B[env_ids],
            self.traj_curriculum.rng,
        )
        self.traj_batch.load_from_stacked(env_ids, self.traj_bank.batch, rows)

        # 3. anchor the origin-centered path in front of the shoulder (world
        #    axes -- the base yaws to follow via v_ff)
        mount_offset_body = self.arm_mount_tfs[env_ids, :3]
        shoulder_world = self.base_pos[env_ids] + quat_apply(self.base_quat[env_ids], mount_offset_body)
        anchor = shoulder_world + quat_apply(
            self.base_quat[env_ids], self._traj_anchor_offset.expand(len(env_ids), -1)
        )
        self.traj_batch.gamma_p[env_ids] += anchor.unsqueeze(1)

        # 4. clear per-env progress state + episode accumulators
        for buf in (
            self.traj_s, self.traj_s_prev, self.traj_sim_time, self.traj_d_lat,
            self.traj_sdot_meas, self.traj_timing_err, self.traj_twist_err,
            self.traj_dlat_sum, self.traj_timing_abs_sum, self.traj_ik_jump_max,
            self.traj_samples,
        ):
            buf[env_ids] = 0.0

    def _randomize_arm_dof_props(self, env_ids):
        """Override arm DOF slice in Kp/Kd/strength/offset buffers with stage-specific ranges."""
        if len(env_ids) == 0:
            return
        arm_dr = self.cfg.domain_rand.stage1_arm if not global_switch.switch_open else self.cfg.domain_rand.stage2_arm
        arm_slice = slice(self.num_actions_loco, self.num_actions_loco + self.num_actions_arm)
        n = len(env_ids)

        if arm_dr.randomize_Kp_factor:
            lo, hi = arm_dr.Kp_factor_range
            self.Kp_factors[env_ids, arm_slice] = (
                torch.rand(n, self.num_actions_arm, device=self.device) * (hi - lo) + lo
            )
        if arm_dr.randomize_Kd_factor:
            lo, hi = arm_dr.Kd_factor_range
            self.Kd_factors[env_ids, arm_slice] = (
                torch.rand(n, self.num_actions_arm, device=self.device) * (hi - lo) + lo
            )
        if arm_dr.randomize_motor_strength:
            lo, hi = arm_dr.motor_strength_range
            self.motor_strengths[env_ids, arm_slice] = (
                torch.rand(n, self.num_actions_arm, device=self.device) * (hi - lo) + lo
            )
        if arm_dr.randomize_motor_offset:
            r = arm_dr.motor_offset_range
            self.motor_offsets[env_ids, arm_slice] = torch_rand_float(
                -r, r, (n, self.num_actions_arm), device=self.device
            )

    def _arm_post_dof_randomization_hook(self, env_ids):
        self._randomize_arm_dof_props(env_ids)

    def _randomize_arm_rigid_body_props(self, env_ids):
        """Deprecated: arm link mass/COM DR is applied per-env during actor creation."""
        return

    def _arm_resample_commands_train_hook(self, env_ids):
        if self._goal_reaching_enabled() and global_switch.switch_open:
            self._reset_goal_commands(env_ids)
            return True
        return False

    def _get_privileged_dof_slice(self, policy):
        if policy == "dog":
            return slice(0, self.num_actions_loco)
        if policy == "arm":
            # Privileged physics terms index robot DOFs, not actor outputs.
            # Coordination channels after the first six actions have no
            # corresponding joints and must never widen this slice.
            return slice(self.num_actions_loco, self.num_actions_loco + self.num_actions_arm)
        raise ValueError(f"Unknown privileged observation policy: {policy}")

    def _get_physics_privileged_observations(self, policy):
        privileged_obs_buf = torch.empty(self.num_envs, 0, device=self.device)
        dof_slice = self._get_privileged_dof_slice(policy)

        if self.cfg.env.priv_observe_friction:
            scale, shift = get_scale_shift(self.cfg.normalization.friction_range)
            privileged_obs_buf = torch.cat(
                (privileged_obs_buf, (self.friction_coeffs[:, 0].unsqueeze(1) - shift) * scale),
                dim=1,
            )

        if self.cfg.env.priv_observe_ground_friction:
            self.ground_friction_coeffs = self._get_ground_frictions(range(self.num_envs))
            scale, shift = get_scale_shift(self.cfg.normalization.ground_friction_range)
            privileged_obs_buf = torch.cat(
                (privileged_obs_buf, (self.ground_friction_coeffs.unsqueeze(1) - shift) * scale),
                dim=1,
            )

        if self.cfg.env.priv_observe_restitution:
            scale, shift = get_scale_shift(self.cfg.normalization.restitution_range)
            privileged_obs_buf = torch.cat(
                (privileged_obs_buf, (self.restitutions[:, 0].unsqueeze(1) - shift) * scale),
                dim=1,
            )

        if self.cfg.env.priv_observe_base_mass:
            scale, shift = get_scale_shift(self.cfg.normalization.added_mass_range)
            privileged_obs_buf = torch.cat(
                (privileged_obs_buf, (self.payloads.unsqueeze(1) - shift) * scale),
                dim=1,
            )

        if self.cfg.env.priv_observe_com_displacement and (
            policy != "dog" or self.cfg.dog.priv_observe_com_displacement
        ):
            scale, shift = get_scale_shift(self.cfg.normalization.com_displacement_range)
            privileged_obs_buf = torch.cat(
                (privileged_obs_buf, (self.com_displacements - shift) * scale),
                dim=1,
            )

        if getattr(self.cfg.env, "priv_observe_stage1_ee_payload_mass", False):
            scale, shift = get_scale_shift(self.cfg.normalization.stage1_ee_payload_mass_range)
            privileged_obs_buf = torch.cat(
                (privileged_obs_buf, (self.stage1_ee_payload_mass.unsqueeze(1) - shift) * scale),
                dim=1,
            )

        if policy == "dog" and self.cfg.dog.priv_observe_motor_strength:
            scale, shift = get_scale_shift(self.cfg.normalization.motor_strength_range)
            # Global dog motor strength is sampled once per env and broadcast
            # across all leg joints, so one scalar carries the full information.
            dog_motor_strength = self.motor_strengths[:, dof_slice.start : dof_slice.start + 1]
            privileged_obs_buf = torch.cat(
                (privileged_obs_buf, (dog_motor_strength - shift) * scale), dim=1
            )

        if policy == "dog" and self.cfg.dog.priv_observe_motor_offset:
            scale, shift = get_scale_shift(self.cfg.normalization.motor_offset_range)
            privileged_obs_buf = torch.cat(
                (privileged_obs_buf, (self.motor_offsets[:, dof_slice] - shift) * scale), dim=1
            )

        if self.cfg.env.priv_observe_motor_strength:
            scale, shift = get_scale_shift(self.cfg.normalization.motor_strength_range)
            privileged_obs_buf = torch.cat(
                (privileged_obs_buf, (self.motor_strengths[:, dof_slice] - shift) * scale),
                dim=1,
            )

        if self.cfg.env.priv_observe_motor_offset:
            scale, shift = get_scale_shift(self.cfg.normalization.motor_offset_range)
            privileged_obs_buf = torch.cat(
                (privileged_obs_buf, (self.motor_offsets[:, dof_slice] - shift) * scale),
                dim=1,
            )

        if self.cfg.env.priv_observe_Kp_factor:
            kp_range = (
                self.cfg.normalization.dog_Kp_factor_range
                if policy == "dog"
                else self.cfg.normalization.Kp_factor_range
            )
            scale, shift = get_scale_shift(kp_range)
            kp_factors = self.Kp_factors[:, dof_slice]
            if policy == "dog":
                # Global leg Kp is one per-env scalar broadcast to 12 joints.
                kp_factors = kp_factors[:, :1]
            privileged_obs_buf = torch.cat(
                (privileged_obs_buf, (kp_factors - shift) * scale),
                dim=1,
            )

        if self.cfg.env.priv_observe_Kd_factor:
            kd_range = (
                self.cfg.normalization.dog_Kd_factor_range
                if policy == "dog"
                else self.cfg.normalization.Kd_factor_range
            )
            scale, shift = get_scale_shift(kd_range)
            kd_factors = self.Kd_factors[:, dof_slice]
            if policy == "dog":
                # Global leg Kd is one per-env scalar broadcast to 12 joints.
                kd_factors = kd_factors[:, :1]
            privileged_obs_buf = torch.cat(
                (privileged_obs_buf, (kd_factors - shift) * scale),
                dim=1,
            )

        if self.cfg.env.priv_observe_joint_friction and (
            policy != "dog" or self.cfg.dog.priv_observe_joint_friction
        ):
            scale, shift = get_scale_shift(self.cfg.normalization.joint_friction_range)
            privileged_obs_buf = torch.cat(
                (privileged_obs_buf, (self.dof_frictions[:, dof_slice] - shift) * scale),
                dim=1,
            )

        if getattr(self.cfg.env, "priv_observe_dof_damping", False) and (
            policy != "dog" or self.cfg.dog.priv_observe_dof_damping
        ):
            scale, shift = get_scale_shift(self.cfg.normalization.dof_damping_range)
            privileged_obs_buf = torch.cat(
                (privileged_obs_buf, (self.dof_dampings[:, dof_slice] - shift) * scale),
                dim=1,
            )

        if self.cfg.env.priv_observe_body_height:
            scale, shift = get_scale_shift(self.cfg.normalization.body_height_range)
            privileged_obs_buf = torch.cat(
                (privileged_obs_buf, ((self.root_states[: self.num_envs, 2]).view(self.num_envs, -1) - shift) * scale),
                dim=1,
            )

        if self.cfg.env.priv_observe_gravity:
            scale, shift = get_scale_shift(self.cfg.normalization.gravity_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf, (self.gravities - shift) * scale), dim=1)

        if policy == "dog" and self.cfg.dog.priv_observe_gravity:
            scale, shift = get_scale_shift(self.cfg.normalization.gravity_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf, (self.gravities - shift) * scale), dim=1)

        if self.cfg.env.priv_observe_body_velocity or self.cfg.env.priv_observe_vel:
            if self.cfg.commands.global_reference:
                lin_vel = self.root_states[: self.num_envs, 7:10]
            else:
                lin_vel = self.base_lin_vel
            privileged_obs_buf = torch.cat(
                (
                    privileged_obs_buf,
                    lin_vel * self.obs_scales.lin_vel,
                    self.base_ang_vel * self.obs_scales.ang_vel,
                ),
                dim=1,
            )

        if self.cfg.env.priv_observe_clock_inputs:
            privileged_obs_buf = torch.cat((privileged_obs_buf, self.clock_inputs), dim=1)

        if self.cfg.env.priv_observe_desired_contact_states:
            privileged_obs_buf = torch.cat((privileged_obs_buf, self.desired_contact_states), dim=1)

        if policy == "dog" and self.cfg.dog.priv_observe_contact_states:
            foot_contact_states = (self.contact_forces[:, self.feet_indices, 2] > 1.0).float()
            privileged_obs_buf = torch.cat((privileged_obs_buf, foot_contact_states), dim=1)

        if self.cfg.env.priv_observe_high_freq_goal:
            privileged_obs_buf = torch.cat(
                (privileged_obs_buf, self.obj_pose_in_ee.clone(), self.obj_abg_in_ee.clone()),
                dim=1,
            )

        if getattr(self.cfg.env, "priv_observe_arm_mount_tf", False):
            privileged_obs_buf = torch.cat((privileged_obs_buf, self.arm_mount_tfs), dim=1)

        if policy == "dog":
            arm_slice = slice(self.num_actions_loco, self.num_actions_loco + self.num_actions_arm)
            if self.cfg.dog.priv_observe_arm_dynamics:
                scale, shift = get_scale_shift(self.cfg.normalization.arm_Kp_factor_range)
                arm_kp = (self.Kp_factors[:, arm_slice] - shift) * scale
                scale, shift = get_scale_shift(self.cfg.normalization.arm_Kd_factor_range)
                arm_kd = (self.Kd_factors[:, arm_slice] - shift) * scale
                scale, shift = get_scale_shift(self.cfg.normalization.arm_motor_strength_range)
                arm_motor_strength = (self.motor_strengths[:, arm_slice] - shift) * scale
                scale, shift = get_scale_shift(self.cfg.normalization.arm_motor_offset_range)
                arm_motor_offset = (self.motor_offsets[:, arm_slice] - shift) * scale

                expected_links = int(self.cfg.arm.num_privileged_links)
                actual_links = self.arm_link_mass_scales.shape[1]
                if actual_links != expected_links:
                    raise AssertionError(
                        f"arm privileged link count ({expected_links}) != runtime arm link count ({actual_links})"
                    )
                scale, shift = get_scale_shift(self.cfg.normalization.arm_link_mass_scale_range)
                arm_link_mass_scale = (self.arm_link_mass_scales - shift) * scale
                scale, shift = get_scale_shift(self.cfg.normalization.arm_link_com_offset_range)
                arm_link_com_offset = (self.arm_link_com_offsets.reshape(self.num_envs, -1) - shift) * scale
                privileged_obs_buf = torch.cat(
                    (
                        privileged_obs_buf,
                        arm_kp,
                        arm_kd,
                        arm_motor_strength,
                        arm_motor_offset,
                        arm_link_mass_scale,
                        arm_link_com_offset,
                    ),
                    dim=1,
                )

            arm_dof_pos = (self.dof_pos[:, arm_slice] - self.default_dof_pos[:, arm_slice]) * self.obs_scales.dof_pos
            arm_dof_vel = self.dof_vel[:, arm_slice] * self.obs_scales.dof_vel
            privileged_obs_buf = torch.cat((privileged_obs_buf, arm_dof_pos, arm_dof_vel), dim=1)

        return privileged_obs_buf

    def _arm_post_callback_hook(self):
        # Periodic arm target resample now lives in _arm_post_physics_hook
        # (arm_time_buf >= arm_target_resample_steps), since target sampling
        # is FK-independent (pure base-frame Cartesian, no deferred-resample
        # dance needed -- see _resample_arm_target).
        if self.cfg.domain_rand.randomize_end_effector_force:
            traj_ids = (self.force_time_buf % (self.T_force / self.dt).long() == 0).nonzero(as_tuple=False).flatten()
            self.resample_force(traj_ids)

    def _arm_observation_hook(self, obs_buf, roll, pitch, yaw):
        # Vision branch reads from self.obj_pose_in_ee / self.obj_abg_in_ee which need refreshing
        # before compute_observations consumes them.
        if self.cfg.wbc.use_vision:
            self._get_object_pose_in_ee()
            self._get_object_abg_in_ee()

        n_cmd_dims = self.cfg.dog.dog_num_commands if self.cfg.commands.use_dynamic_gait else 3

        if self.cfg.wbc.use_vision:
            env_ids = (
                (self.episode_length_buf % int((1.0 / self.cfg.control.update_obs_freq) / self.dt + 0.5) == 0)
                .nonzero(as_tuple=False)
                .flatten()
            )
            self.obj_obs_pose_in_ee[env_ids] = self.obj_pose_in_ee[env_ids].clone()
            self.obj_obs_abg_in_ee[env_ids] = self.obj_abg_in_ee[env_ids].clone()
            obs_buf = torch.cat(
                (
                    obs_buf,
                    (self.commands_dog * self.commands_scale_dog)[:, :n_cmd_dims],
                    self.obj_obs_pose_in_ee[:]
                    if global_switch.switch_open
                    else torch.zeros_like(self.obj_obs_pose_in_ee[:]),
                    self.obj_obs_abg_in_ee[:]
                    if global_switch.switch_open
                    else torch.zeros_like(self.obj_obs_abg_in_ee[:]),
                    roll.unsqueeze(1),
                    pitch.unsqueeze(1),
                ),
                dim=-1,
            )
        else:
            obs_buf = torch.cat(
                (
                    obs_buf,
                    (self.commands_dog * self.commands_scale_dog)[:, :n_cmd_dims],
                    self.commands_arm_obs[:]
                    if global_switch.switch_open
                    else torch.zeros_like(self.commands_arm_obs[:]),
                    roll.unsqueeze(1),
                    pitch.unsqueeze(1),
                ),
                dim=-1,
            )

        return obs_buf

    def _arm_observation_traj_hook(self, obs_buf):
        # Legacy single-policy obs path (compute_observations in
        # legged_robot.py); not exercised by the dual-policy training loop.
        return obs_buf

    def _arm_step_end_hook(self):
        if self._goal_reaching_enabled():
            self.last_arm_policy_actions[:] = self.arm_policy_actions
            self.goal_rho_prev[:] = self.goal_rho

    def _arm_init_performance_metrics_hook(self):
        for name in ("ee_position_sq_error", "ee_orientation_sq_error", "ee_tracking_samples"):
            self.performance_metric_sums[name] = torch.zeros(
                self.num_envs,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )

    def _arm_update_performance_metrics_hook(self):
        if not global_switch.switch_open:
            return

        sums = self.performance_metric_sums
        sums["ee_position_sq_error"] += torch.sum(torch.square(self.ee_pos_err), dim=-1)
        sums["ee_orientation_sq_error"] += torch.sum(torch.square(self.ee_rot_err_axis_angle), dim=-1)
        sums["ee_tracking_samples"] += 1.0

    def _arm_log_performance_metrics_hook(self, train_env_ids, episode_steps):
        samples = self.performance_metric_sums["ee_tracking_samples"][train_env_ids]
        valid = samples > 0
        if not torch.any(valid):
            return

        position_rmse = torch.sqrt(
            self.performance_metric_sums["ee_position_sq_error"][train_env_ids]
            / torch.clamp(samples, min=1.0)
        )
        orientation_rmse = torch.sqrt(
            self.performance_metric_sums["ee_orientation_sq_error"][train_env_ids]
            / torch.clamp(samples, min=1.0)
        )
        extras = self.extras["train/episode"]
        extras["perf_ee_position_rmse_m"] = self._mean_valid_metric(position_rmse, valid)
        extras["perf_ee_orientation_rmse_rad"] = self._mean_valid_metric(orientation_rmse, valid)

        if self._traj_tracking_enabled():
            st = self.traj_curriculum.stats()
            for key, value in st.items():
                extras["traj_curriculum_" + key] = torch.as_tensor(
                    float(value), device=self.device
                )

    def _arm_privileged_obs_hook(self, privileged_obs_buf):
        return self._get_physics_privileged_observations("dog")

    # ------------------------------------------------------------------
    # Viewer / video overlays
    # ------------------------------------------------------------------

    def _draw_ee_ori_coord(self):
        grasper_offset = torch.tensor([0.1, 0, 0], dtype=torch.float, device=self.device).reshape(1, -1)
        grasper_in_world = (
            self.end_effector_state[0, :3] + quat_rotate(self.end_effector_state[0:1, 3:7], grasper_offset)[0]
        )
        x, y, z = grasper_in_world[0], grasper_in_world[1], grasper_in_world[2]
        ee_quat = self.end_effector_state[0, 3:7]
        self.draw_sphere_and_axes((x, y, z), ee_quat, 0.02, (1, 1, 0))

    def _draw_command_ori_coord(self):
        target = self._arm_target_pos_world(torch.tensor([0], device=self.device))[0]
        quat_world = self._arm_target_quat_world(torch.tensor([0], device=self.device))[0]
        self.draw_sphere_and_axes((target[0].item(), target[1].item(), target[2].item()), quat_world, 0.02, (0, 1, 1))

    def _draw_viewer_polyline(self, points, color, env_id=0):
        if points.shape[0] < 2:
            return
        vertices = np.empty((points.shape[0] - 1, 2, 3), dtype=np.float32)
        vertices[:, 0, :] = points[:-1]
        vertices[:, 1, :] = points[1:]
        colors = np.tile(np.array([color], dtype=np.float32), (vertices.shape[0], 1))
        self.gym.add_lines(
            self.viewer,
            self.envs[env_id],
            vertices.shape[0],
            vertices.reshape(-1, 3),
            colors,
        )

    def _draw_policy_trajectory(self, env_id=0):
        if self.headless or self.viewer is None:
            return
        env_ids = torch.tensor([env_id], device=self.device)
        target = self._arm_target_pos_world(env_ids)[0]
        target_quat = self._arm_target_quat_world(env_ids)[0]
        self.draw_sphere_and_axes(
            (target[0].item(), target[1].item(), target[2].item()),
            target_quat,
            0.035,
            (0.0, 1.0, 1.0),
            scale=0.12,
        )
        ee_to_target = (
            torch.stack((self.end_effector_state[env_id, :3], target), dim=0).detach().cpu().numpy().astype(np.float32)
        )
        self._draw_viewer_polyline(ee_to_target, (1.0, 0.1, 0.1), env_id)

    def _arm_draw_overlay_hook(self):
        self._draw_ee_ori_coord()
        self._draw_command_ori_coord()
        self._draw_policy_trajectory()

    def _policy_command_overlay_panels(self, env_id=0):
        """Return (left_lines, right_lines) for two-panel overlay."""

        def vals(tensor, count):
            return tensor[env_id, : min(count, tensor.shape[1])].detach().cpu().tolist()

        arm_open = global_switch.switch_open
        dog = vals(self.commands_dog, min(11, self.commands_dog.shape[1]))

        # ---- LEFT: commands ----
        left = []
        left.append(
            f"vx={dog[dog_cmd_idx['x_vel']]:+.2f}  vy={dog[dog_cmd_idx['y_vel']]:+.2f}  yaw={dog[dog_cmd_idx['yaw_vel']]:+.2f}"
        )

        if self.cfg.commands.use_dynamic_gait and self.commands_dog.shape[1] >= 11:
            left.append(
                f"pitch={dog[dog_cmd_idx['body_pitch']]:+.2f}  roll={dog[dog_cmd_idx['body_roll']]:+.2f}  h_cmd={dog[dog_cmd_idx['body_height']]:+.2f}"
            )
            left.append(
                f"freq={dog[dog_cmd_idx['gait_frequency']]:.2f}  swing={dog[dog_cmd_idx['footswing_height']]:.2f}"
            )
            left.append(
                f"width={dog[dog_cmd_idx['stance_width']]:.2f}  len={dog[dog_cmd_idx['stance_length']]:.2f}  dur={dog[dog_cmd_idx['gait_duration']]:.2f}"
            )
        elif self.commands_dog.shape[1] >= 6:
            left.append(
                f"pitch={dog[dog_cmd_idx['body_pitch']]:+.2f}  roll={dog[dog_cmd_idx['body_roll']]:+.2f}  h_cmd={dog[dog_cmd_idx['body_height']]:+.2f}"
            )

        if arm_open:
            tgt = vals(self.arm_target_pos_body, 3)
            left.append(f"target x={tgt[0]:+.2f} y={tgt[1]:+.2f} z={tgt[2]:+.2f}")
            err = vals(self.ee_pos_err, 3)
            left.append(f"pos_err x={err[0]:+.2f} y={err[1]:+.2f} z={err[2]:+.2f}")

        # ---- RIGHT: status ----
        right = []
        base_z = self.root_states[env_id, 2].item()
        right.append(f"z={base_z:.2f}  pitch={self.pitch[env_id].item():+.2f}  roll={self.roll[env_id].item():+.2f}")

        # contact = (self.contact_forces[env_id, self.feet_indices, 2] > 1.0).cpu().tolist()
        # contact_str = "".join("█" if c else "░" for c in contact)
        # right.append(f"feet: {contact_str}  (FL FR RL RR)")

        if arm_open:
            arm_action = vals(
                self.actions[:, self.num_actions_loco : self.num_actions_loco + self.num_actions_arm],
                self.num_actions_arm,
            )
            right.append("arm: " + " ".join(f"{v:+.2f}" for v in arm_action))

        stage = "wbc" if arm_open else "stage1"
        right.append(f"[{stage}]")

        return left, right

    @staticmethod
    def _draw_panel(frame, lines, x0, y0, font, font_scale, line_height, alpha=0.55):
        """Draw a semi-transparent panel and text on a RGBA frame."""
        if not lines:
            return
        (tw, th), _ = cv2.getTextSize(max(lines, key=len), font, font_scale, 1)
        pad = 6
        w = tw + pad * 2
        h = line_height * len(lines) + pad
        x1, y1 = x0 + w, y0 + h

        # alpha blend dark background
        roi = frame[y0:y1, x0:x1]
        bg = np.zeros_like(roi)
        bg[:, :, 3] = 255
        frame[y0:y1, x0:x1] = (roi * (1 - alpha) + bg * alpha).astype(np.uint8)

        for i, text in enumerate(lines):
            y = y0 + pad + i * line_height + line_height - 4
            cv2.putText(frame, text, (x0 + pad, y), font, font_scale, (0, 0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, text, (x0 + pad, y), font, font_scale, (230, 230, 230, 255), 1, cv2.LINE_AA)

    def _overlay_policy_text(self, frame, env_id=0):
        left_lines, right_lines = self._policy_command_overlay_panels(env_id)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.32
        line_height = 13
        W = frame.shape[1]

        # left panel: top-left
        self._draw_panel(frame, left_lines, 4, 4, font, font_scale, line_height)

        # right panel: top-right (estimate width from longest line)
        if right_lines:
            (tw, _), _ = cv2.getTextSize(max(right_lines, key=len), font, font_scale, 1)
            x0_right = max(W // 2, W - tw - 20)
            self._draw_panel(frame, right_lines, x0_right, 4, font, font_scale, line_height)

    def _project_world_points_to_camera(self, points_world, env_handle, camera_handle):
        view = np.asarray(
            self.gym.get_camera_view_matrix(self.sim, env_handle, camera_handle), dtype=np.float32
        ).reshape(4, 4)
        proj = np.asarray(
            self.gym.get_camera_proj_matrix(self.sim, env_handle, camera_handle), dtype=np.float32
        ).reshape(4, 4)
        points_h = np.concatenate((points_world, np.ones((points_world.shape[0], 1), dtype=np.float32)), axis=1)
        clip = points_h @ view @ proj
        w = clip[:, 3:4]
        valid = np.abs(w[:, 0]) > 1e-5
        ndc = np.zeros((points_world.shape[0], 3), dtype=np.float32)
        ndc[valid] = clip[valid, :3] / w[valid]
        width, height = self.camera_props.width, self.camera_props.height
        pixels = np.empty((points_world.shape[0], 2), dtype=np.int32)
        pixels[:, 0] = ((ndc[:, 0] + 1.0) * 0.5 * width).astype(np.int32)
        pixels[:, 1] = ((1.0 - ndc[:, 1]) * 0.5 * height).astype(np.int32)
        valid &= np.isfinite(ndc).all(axis=1)
        valid &= (pixels[:, 0] >= -width) & (pixels[:, 0] <= 2 * width)
        valid &= (pixels[:, 1] >= -height) & (pixels[:, 1] <= 2 * height)
        return pixels, valid

    def _overlay_policy_trajectory(self, frame, env_id, env_handle, camera_handle):
        if not self.cfg.env.recording_overlay_trajectory:
            return
        target = self._arm_target_pos_world(torch.tensor([env_id], device=self.device))[0]
        target_np = target.detach().cpu().numpy()[None, :].astype(np.float32)
        try:
            pixels, valid = self._project_world_points_to_camera(target_np, env_handle, camera_handle)
        except Exception:
            return
        if valid[0]:
            cv2.circle(frame, tuple(pixels[0]), 5, (0, 255, 255, 255), -1, cv2.LINE_AA)

    def _arm_render_overlay_hook(self, frame, env_id, env_handle, camera_handle):
        self._overlay_policy_trajectory(frame, env_id, env_handle, camera_handle)
        if self.cfg.env.recording_overlay_text:
            self._overlay_policy_text(frame, env_id)


    def _arm_jacobian(self):
        """(num_envs, 6, num_actions_arm) world-frame Jacobian of the EE body
        w.r.t. the arm's actuated joint columns only (legs/floating-root
        columns dropped -- "IK in base frame" means the base is treated as
        fixed for this solve, which is exactly what dropping those columns
        does). See _arm_init_buffers_hook for the acquisition gotcha and the
        column-layout citation.

        Row/column layout depends on cfg.asset.fix_base_link: a floating
        base (the normal WBCEnv/dog+arm case) has num_links == num_bodies
        (row = body index directly) and num_dof+6 columns (the first 6 are
        the free-root dof, so arm columns start at num_actions_loco + 6); a
        fixed base (see scripts/test_arm_ik.py, which welds the trunk to
        isolate the arm from leg/base dynamics for unit testing) has
        num_links == num_bodies - 1 (row = body index - 1, no free-root dof
        at all) -- confirmed empirically for both, not just from docs."""
        arm_slice = slice(self.num_actions_loco, self.num_actions_loco + self.num_actions_arm)
        if self.cfg.asset.fix_base_link:
            row = self.ee_idx - 1
            col_slice = arm_slice
        else:
            row = self.ee_idx
            col_slice = slice(6 + self.num_actions_loco, 6 + self.num_actions_loco + self.num_actions_arm)
        return self.jacobian[:, row, :, col_slice], arm_slice

    def _update_ee_task_space_error(self):
        target_pos_world = self._arm_target_pos_world()
        target_quat_world = self._arm_target_quat_world()
        self.ee_pos_err[:] = target_pos_world - self.end_effector_state[:, :3]
        self.ee_rot_err_axis_angle[:] = quat_error_axis_angle(target_quat_world, self.end_effector_state[:, 3:7])

        if self._goal_reaching_enabled():
            base_inv = quat_conjugate(self.base_quat)
            self.arm_target_pos_body[:] = quat_apply(base_inv, target_pos_world - self.base_pos)
            self.arm_target_quat_body[:] = quat_mul(base_inv, target_quat_world)

        if self.cfg.use_rot6d:
            target_ori_body = quat_xyzw_to_rot6d(self.arm_target_quat_body)
        else:
            identity_quat = torch.zeros_like(self.arm_target_quat_body)
            identity_quat[:, 3] = 1.0
            target_ori_body = quat_error_axis_angle(self.arm_target_quat_body, identity_quat)
        self.commands_arm_obs[:] = torch.cat((self.arm_target_pos_body, target_ori_body), dim=-1)

    def _update_goal_reaching_diagnostics(self):
        mount_offset_body = self.arm_mount_tfs[:, :3]
        shoulder_world = self.base_pos + quat_apply(self.base_quat, mount_offset_body)
        reach = max(float(self.cfg.wbc.goal_reaching.reach_radius), 1e-3)
        self.goal_rho[:] = torch.linalg.vector_norm(self.arm_goal_pos_world - shoulder_world, dim=-1) / reach
        first_sample = ~self.goal_rho_valid
        self.goal_rho_prev[first_sample] = self.goal_rho[first_sample]
        self.goal_rho_valid[:] = True

        J, arm_slice = self._arm_jacobian()
        singular_values = torch.linalg.svdvals(J)
        self.goal_manipulability[:] = torch.prod(singular_values, dim=-1)

        q = self.dof_pos[:, arm_slice]
        lo = self.dof_pos_limits[arm_slice, 0].unsqueeze(0)
        hi = self.dof_pos_limits[arm_slice, 1].unsqueeze(0)
        span = (hi - lo).clamp_min(1e-6)
        self.goal_joint_limit_distance[:] = torch.minimum(q - lo, hi - q).clamp_min(0.0) / span
        self.arm_ema_motion[:] = 0.95 * self.arm_ema_motion + 0.05 * (
            self.dof_vel[:, arm_slice] * self.dt
        )

    def _solve_arm_dls_ik_step(self):
        """One damped-least-squares differential correction toward the
        current target, using this step's Jacobian and task-space error
        (already refreshed in _arm_post_physics_hook). Not a converged
        Newton solve -- FK only updates via an actual physics step, so this
        is a per-control-step proportional correction that converges over
        several steps in simulated time, same as a real robot's IK loop
        would. See project-design-v3.md §2.8 / §5.1 discussion of DLS.

        The raw DLS solution is clamped to max_step_rad (per-env norm, so
        direction is preserved): the Jacobian is only a first-order estimate
        and the low-level PD cannot track a large joint jump within one
        control period anyway, so for a large task-space error the
        unclamped solution is both inaccurate and untrackable -- see
        arm.ik.max_step_rad's comment in wbc.py for how this was found."""
        J, _ = self._arm_jacobian()  # (N, 6, num_actions_arm), rows 0:3 linear / 3:6 angular
        # Transport the linear rows from x5_link6's origin (where IsaacGym
        # reports the Jacobian) out to the tracked grasp point, so the linear
        # Jacobian is consistent with the offset ee_pos we drive to. For a point
        # at world offset d from the origin, v_point = v_origin + omega x d, i.e.
        # J_v_point = J_v - skew(d) J_w  (== J_v + (J_w columns) x d). Without
        # this the solver "moves" the origin while the grasp point swings on the
        # 0.14 m lever, leaving a rotation-coupled position floor.
        d_world = quat_apply(self.end_effector_state[:, 3:7], self.ee_local_offset.expand(self.num_envs, -1))
        J_w = J[:, 3:6, :]  # (N, 3, num_actions_arm)
        J = J.clone()
        J[:, 0:3, :] = J[:, 0:3, :] + torch.cross(J_w, d_world.unsqueeze(-1).expand_as(J_w), dim=1)
        err = torch.cat((self.ee_pos_err, self.ee_rot_err_axis_angle), dim=-1).unsqueeze(-1)  # (N, 6, 1)
        # Task-space weighting: trade off position vs orientation tracking by
        # scaling both the Jacobian rows and the error by sqrt(weight). This
        # solves the weighted damped least squares min ||W^.5 (J dq - err)||^2
        # + lam^2 ||dq||^2, so a larger pos_weight makes the solver spend the
        # arm's (coupled, 6-DoF) joint budget reducing position error first,
        # and vice versa. Equal weights reproduce the unweighted solve exactly.
        # See arm.ik.pos_weight / rot_weight in docs/PARAM_TUNING.md.
        sqrt_w = torch.tensor(
            [self.cfg.arm.ik.pos_weight] * 3 + [self.cfg.arm.ik.rot_weight] * 3,
            dtype=torch.float, device=self.device,
        ).sqrt().view(1, 6, 1)
        J = J * sqrt_w
        err = err * sqrt_w
        lam2 = self.cfg.arm.ik.damping ** 2
        JJt = torch.bmm(J, J.transpose(1, 2)) + lam2 * torch.eye(6, device=self.device).unsqueeze(0)
        delta_q = torch.bmm(J.transpose(1, 2), torch.linalg.solve(JJt, err)).squeeze(-1)  # (N, num_actions_arm)
        delta_q = delta_q * self.cfg.arm.ik.step_gain
        norm = delta_q.norm(dim=-1, keepdim=True)
        max_step = self.cfg.arm.ik.max_step_rad
        return delta_q * torch.clamp(max_step / torch.clamp(norm, min=1e-8), max=1.0)

    def _apply_stage2_arm_ik_action(self):
        """Combine q_ik (DLS-IK toward arm_target_pos/quat_body) with the
        policy's Δq residual (tanh-limited) into self.actions[:, arm_slice],
        in the (target - default)/action_scale form _compute_torques already
        expects -- same trick _apply_stage1_arm_curriculum_actions uses."""
        if not global_switch.switch_open:
            return
        _, arm_slice = self._arm_jacobian()
        delta_q_ik = self._solve_arm_dls_ik_step()
        self.arm_residual_raw[:] = self.actions[:, arm_slice]
        self.arm_policy_actions[:, : self.num_actions_arm] = self.arm_residual_raw
        delta_q_residual = torch.tanh(self.arm_residual_raw) * self.cfg.arm.ik.residual_scale
        q_target = self.dof_pos[:, arm_slice] + delta_q_ik + delta_q_residual
        arm_default = self.default_dof_pos[:, arm_slice]
        self.actions[:, arm_slice] = (q_target - arm_default) / self.cfg.control.action_scale

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs

    def _arm_dog_state_obs_terms(self):
        """Cross-policy channel gated by cfg.env.arm_observe_dog_state: foot
        contact state and the leg policy's (v_actual - v_cmd) tracking
        residual, so the arm/upper policy has some sense of how well the
        legs are keeping up. Dims must match core.arm_obs_dim_parts'
        dog_contact_states / dog_vel_residual entries."""
        if not getattr(self.cfg.env, "arm_observe_dog_state", False):
            return None, None
        contact_states = (self.contact_forces[:, self.feet_indices, 2] > 1.0).float()
        vel_residual = torch.cat(
            (
                self.base_lin_vel[:, :2] - self.commands_dog[:, :2],
                self.base_ang_vel[:, 2:3] - self.commands_dog[:, 2:3],
            ),
            dim=-1,
        )
        return contact_states, vel_residual

    def _arm_traj_obs_terms(self):
        """Actor-facing trajectory-tracking extras: progress/timing scalars, a
        tau phase encoding, reach urgency, and a K-point look-ahead preview
        (position/rot6d/tangent in base frame + reference speed + rho). Width
        = 8 + K*14, matching core.arm_obs_dim_parts' traj_* entries."""
        tcfg = self.cfg.wbc.goal_reaching.trajectory
        N = self.num_envs
        K = int(tcfg.preview_points)
        L = self.traj_batch.L.clamp_min(1e-6)

        s_norm = (self.traj_s / L).unsqueeze(-1)
        sdot_ref = self.traj_batch.sdot_ref(self.traj_sim_time)
        scalars = torch.stack(
            (self.traj_s / L, self.traj_timing_err, self.traj_sdot_meas, sdot_ref), dim=-1
        )
        phase = 2.0 * np.pi * s_norm
        tau_enc = torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)

        s_k, p_k, R_k, sdot_k = self.traj_batch.sample_preview(
            self.traj_s, float(tcfg.preview_horizon), K
        )
        # base-frame transforms (flatten the K axis for quat ops)
        base_inv = quat_conjugate(self.base_quat)
        base_inv_k = base_inv.unsqueeze(1).expand(N, K, 4).reshape(N * K, 4)
        p_rel = (p_k - self.base_pos.unsqueeze(1)).reshape(N * K, 3)
        p_k_body = quat_apply(base_inv_k, p_rel).reshape(N, K, 3)
        q_k = self._traj_mat_to_quat(R_k).reshape(N * K, 4)
        q_k_body = quat_mul(base_inv_k, q_k)
        rot6d_k = quat_xyzw_to_rot6d(q_k_body).reshape(N, K, 6)
        tangent_k = self.traj_batch.tangent_at(s_k).reshape(N * K, 3)
        tangent_k_body = quat_apply(base_inv_k, tangent_k).reshape(N, K, 3)

        # closed-form rho along the preview (reuse the diagnostics formula)
        mount_offset_body = self.arm_mount_tfs[:, :3]
        shoulder_world = self.base_pos + quat_apply(self.base_quat, mount_offset_body)
        reach = max(float(self.cfg.wbc.goal_reaching.reach_radius), 1e-3)
        rho_k = torch.linalg.vector_norm(p_k - shoulder_world.unsqueeze(1), dim=-1) / reach
        rho_hi = float(self.cfg.wbc.goal_reaching.rho_hi)
        urgency = (rho_k - rho_hi).clamp_min(0.0).max(dim=-1).values
        violated = rho_k > rho_hi
        any_viol = violated.any(dim=-1)
        first_viol = violated.float().argmax(dim=-1).float() / max(K, 1)
        s_to_viol = torch.where(any_viol, first_viol, torch.ones_like(first_viol))
        urgency_pair = torch.stack((urgency, s_to_viol), dim=-1)

        preview = torch.cat(
            (p_k_body, rot6d_k, tangent_k_body, sdot_k.unsqueeze(-1), rho_k.unsqueeze(-1)), dim=-1
        ).reshape(N, K * 14)
        return torch.cat((scalars, tau_enc, urgency_pair, preview), dim=-1)

    def get_arm_observations(self):
        """DLS-IK MVP task-space observations -- see core.arm_obs_dim_parts
        for the authoritative width breakdown this must match exactly."""
        arm_slice = slice(self.num_actions_loco, self.num_actions_loco + self.num_actions_arm)
        target_rot6d_body = quat_xyzw_to_rot6d(self.arm_target_quat_body)

        if self._goal_reaching_enabled():
            base_inv = quat_conjugate(self.base_quat)
            ee_pos_err_body = quat_apply(base_inv, self.ee_pos_err)
            ee_rot_err_body = quat_apply(base_inv, self.ee_rot_err_axis_angle)
            ee_twist_body = self.get_ee_twist_body()
            contact_states, vel_residual = self._arm_dog_state_obs_terms()
            common_terms = (
                ee_pos_err_body,
                ee_rot_err_body,
                ee_twist_body,
                self.arm_target_pos_body,
                target_rot6d_body,
                (self.dof_pos[:, arm_slice] - self.default_dof_pos[:, arm_slice])
                * self.obs_scales.dof_pos,
                self.dof_vel[:, arm_slice] * self.obs_scales.dof_vel,
                torch.stack((self.roll, self.pitch, self.base_pos[:, 2]), dim=-1),
                self.base_lin_vel,
                self.base_ang_vel,
                vel_residual,
            )
            if getattr(self.cfg.arm, "checkpoint_observation_layout", "current") == "goal_reaching_extended_v1":
                phase = 2.0 * np.pi * self.gait_indices
                gait_phase = torch.stack((torch.sin(phase), torch.cos(phase)), dim=-1)
                obs_buf = torch.cat(
                    common_terms
                    + (
                        gait_phase,
                        contact_states,
                        self.goal_manipulability.unsqueeze(-1),
                        self.goal_joint_limit_distance,
                        self.goal_rho.unsqueeze(-1),
                        self.arm_ema_motion,
                        self.base_feedforward_cmd,
                        self.arm_policy_actions,
                    ),
                    dim=-1,
                )
            else:
                obs_buf = torch.cat(
                    common_terms
                    + (
                        contact_states,
                        self.base_feedforward_cmd,
                        self.arm_policy_actions,
                    ),
                    dim=-1,
                )
        else:
            obs_buf = torch.cat(
                (
                    self.ee_pos_err,
                    self.ee_rot_err_axis_angle,
                    self.arm_target_pos_body,
                    target_rot6d_body,
                    (self.dof_pos[:, arm_slice] - self.default_dof_pos[:, arm_slice])
                    * self.obs_scales.dof_pos,
                    self.dof_vel[:, arm_slice] * self.obs_scales.dof_vel,
                    self.actions[:, arm_slice],
                ),
                dim=-1,
            )

        if self._traj_tracking_enabled():
            obs_buf = torch.cat((obs_buf, self._arm_traj_obs_terms()), dim=-1)

        if self.cfg.env.observe_two_prev_actions:
            obs_buf = torch.cat((obs_buf, self.last_actions), dim=-1)

        if self.cfg.wbc.use_vision:
            env_ids = (
                (self.episode_length_buf % int((1.0 / self.cfg.control.update_obs_freq) / self.dt + 0.5) == 0)
                .nonzero(as_tuple=False)
                .flatten()
            )
            self.obj_obs_pose_in_ee[env_ids] = self.obj_pose_in_ee[env_ids].clone()
            self.obj_obs_abg_in_ee[env_ids] = self.obj_abg_in_ee[env_ids].clone()

        privileged_obs_buf = self._get_physics_privileged_observations("arm")
        if self._goal_reaching_enabled():
            privileged_obs_buf = torch.cat(
                (
                    privileged_obs_buf,
                    self.goal_manipulability.unsqueeze(-1),
                    self.goal_joint_limit_distance,
                    self.goal_rho.unsqueeze(-1),
                    self.arm_ema_motion,
                ),
                dim=-1,
            )

        assert privileged_obs_buf.shape[1] == self.cfg.arm.arm_num_privileged_obs, (
            f"arm num_privileged_obs ({self.cfg.arm.arm_num_privileged_obs}) \
                           != the number of privileged observations ({privileged_obs_buf.shape[1]}),\
                               you will discard data from the student!"
        )

        # return clipped obs, clipped states (None), rewards, dones and infos
        obs_builder = ObservationBuilder(self, "arm", self.cfg.arm.arm_num_observations)
        if self._goal_reaching_enabled():
            obs_builder.add(obs_buf)
        else:
            obs_builder.add(obs_buf, *self._arm_dog_state_obs_terms())
        obs_buf = obs_builder.build()
        if privileged_obs_buf is not None:
            privileged_obs_buf = clip_observation(self, privileged_obs_buf)

        return obs_buf, privileged_obs_buf

    def _dog_obs_layout(self):
        """Ordered (name, width, noise_scale, droppable) description of
        get_dog_observations()'s actor-facing segments. Single source of
        truth for the post-concatenation independent-noise vector and the
        per-segment frame-drop offsets built once in _arm_init_buffers_hook.
        Velocity/pose measurement noise is injected earlier so tracking
        errors can share the same noisy actual values. [NOTE] this layout
        must still be updated by hand if get_dog_observations() changes.

        droppable=True marks an actual sensor reading (IMU, encoders,
        contacts, ...) that can independently simulate a dropped frame.
        Commanded/internal segments (dog_commands, our own last action,
        gait clock, ...) are never dropped -- we always know those exactly,
        there's no sensor to fail."""
        cfg = self.cfg
        ns = cfg.noise_scales
        level = cfg.noise.noise_level
        s = self.obs_scales

        layout = [
            ("projected_gravity", 3, ns.gravity * level, True),
            ("dog_dof_pos", self.num_actions_loco, ns.dof_pos * level * s.dof_pos, True),
            ("dog_dof_vel", self.num_actions_loco, ns.dof_vel * level * s.dof_vel, True),
            ("dog_actions", self.num_actions_loco, 0.0, False),
        ]
        if cfg.wbc.use_vision:
            layout += [
                ("dog_commands", cfg.dog.dog_num_commands, 0.0, False),
                ("obj_pose_in_ee", 3, 0.0, False),
                ("obj_abg_in_ee", 3, 0.0, False),
            ]
        else:
            layout += [
                ("dog_commands", cfg.dog.dog_num_commands, 0.0, False),
                ("arm_commands", cfg.arm.arm_num_commands, 0.0, False),
            ]
        if cfg.env.observe_two_prev_actions:
            layout.append(("two_prev_actions", self.num_actions_loco, 0.0, False))
        if cfg.env.observe_timing_parameter:
            layout.append(("timing_parameter", 1, 0.0, False))
        if cfg.env.observe_clock_inputs:
            layout.append(("clock_inputs", 4, 0.0, False))

        # Velocity and pose measurements are noised explicitly in
        # get_dog_observations() so their error terms can reuse exactly the
        # same noisy actual values. Keep their independent noise scales zero
        # here to avoid adding a second noise sample after concatenation.
        layout.append(("base_ang_vel", 3, 0.0, True))
        layout.append(("base_lin_vel", 3, 0.0, True))
        # Fixed tracking slot. Pose actual and tracking errors are independently
        # zero-filled by their dog-policy switches. Only the measured actual
        # pose gets sensor noise; errors receive no independent noise.
        layout.append(("body_pose_actual", 3, 0.0, False))
        layout.append(("body_pose_error", 3, 0.0, False))
        layout.append(("velocity_error", 3, 0.0, False))

        if cfg.env.observe_yaw:
            layout.append(("heading", 1, 0.0, False))
        if cfg.env.observe_contact_states:
            layout.append(("contact_states", 4, ns.contact_states * level, True))

        layout.append(("arm_dof_pos", self.num_actions_arm, ns.dof_pos * level * s.dof_pos, True))
        layout.append(("arm_dof_vel", self.num_actions_arm, ns.dof_vel * level * s.dof_vel, True))
        return layout

    def get_dog_observations(self):
        """Computes observations"""
        obs_buf = torch.cat(
            (
                self.projected_gravity,
                (self.dof_pos[:, : self.num_actions_loco] - self.default_dof_pos[:, : self.num_actions_loco])
                * self.obs_scales.dof_pos,
                self.dof_vel[:, : self.num_actions_loco] * self.obs_scales.dof_vel,
                self.actions[:, : self.num_actions_loco],
            ),
            dim=-1,
        )

        if self.cfg.wbc.use_vision:
            # "obj_obs_pose_in_ee" and "obj_obs_abg_in_ee" are already updated in the "get_arm_observations()"
            obs_buf = torch.cat(
                (
                    obs_buf,
                    (self.commands_dog * self.commands_scale_dog)[:, : self.cfg.dog.dog_num_commands],
                    (self.obj_obs_pose_in_ee[:])
                    if global_switch.switch_open
                    else torch.zeros_like(self.obj_obs_pose_in_ee[:]),
                    (self.obj_obs_abg_in_ee[:])
                    if global_switch.switch_open
                    else torch.zeros_like(self.obj_obs_abg_in_ee[:]),
                ),
                dim=-1,
            )
        else:
            idx = self.cfg.arm.arm_num_commands
            obs_buf = torch.cat(
                (
                    obs_buf,
                    (self.commands_dog * self.commands_scale_dog)[:, : self.cfg.dog.dog_num_commands],
                    (self.commands_arm_obs[:, :idx])
                    if global_switch.switch_open
                    else torch.zeros_like(self.commands_arm_obs[:, :idx]),
                ),
                dim=-1,
            )

        if self.cfg.env.observe_two_prev_actions:
            obs_buf = torch.cat((obs_buf, self.last_actions), dim=-1)

        if self.cfg.env.observe_timing_parameter:
            obs_buf = torch.cat((obs_buf, self.gait_indices.unsqueeze(1)), dim=-1)

        if self.cfg.env.observe_clock_inputs:
            obs_buf = torch.cat((obs_buf, self.clock_inputs), dim=-1)

        # Generate each measured actual once, then reuse it in the associated
        # tracking error. This keeps actual + error == command even with noise.
        lin_vel_actual = (
            self.root_states[: self.num_envs, 7:10]
            if self.cfg.commands.global_reference
            else self.base_lin_vel
        )
        ang_vel_measured = self.base_ang_vel * self.obs_scales.ang_vel
        lin_vel_measured = lin_vel_actual * self.obs_scales.lin_vel
        tracking_lin_vel_measured = self.base_lin_vel * self.obs_scales.lin_vel
        pose_measured = torch.stack(
            (
                self.base_pos[:, 2] * self.obs_scales.body_height_cmd,
                self.pitch * self.obs_scales.body_pitch_cmd,
                self.roll * self.obs_scales.body_roll_cmd,
            ),
            dim=-1,
        )
        # getattr keeps configs restored from checkpoints created before this
        # dog-specific switch compatible with the previous noise-on behavior.
        dog_obs_noise_enabled = self.cfg.noise.add_noise and getattr(
            self.cfg.dog, "add_obs_noise", True
        )
        if dog_obs_noise_enabled:
            noise_level = self.cfg.noise.noise_level
            ang_vel_noise = torch.randn_like(ang_vel_measured) * (
                self.cfg.noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
            )
            lin_vel_noise = torch.randn_like(lin_vel_measured) * (
                self.cfg.noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
            )
            pose_noise_scale = pose_measured.new_tensor(
                [
                    self.cfg.noise_scales.gravity * noise_level * self.obs_scales.body_height_cmd,
                    self.cfg.noise_scales.gravity * noise_level * self.obs_scales.body_pitch_cmd,
                    self.cfg.noise_scales.gravity * noise_level * self.obs_scales.body_roll_cmd,
                ]
            )
            ang_vel_measured = ang_vel_measured + ang_vel_noise
            lin_vel_measured = lin_vel_measured + lin_vel_noise
            tracking_lin_vel_measured = tracking_lin_vel_measured + lin_vel_noise
            pose_measured = pose_measured + torch.randn_like(pose_measured) * pose_noise_scale

        # Fixed width regardless of dog.observe_lin_vel: ang_vel is always
        # real; lin_vel's slot always exists but is zeros when the switch is
        # off, so toggling it never changes dog_num_observations.
        if self.cfg.dog.observe_lin_vel:
            lin_vel_term = lin_vel_measured
        else:
            lin_vel_term = torch.zeros(self.num_envs, 3, device=self.device)
        obs_buf = torch.cat((obs_buf, ang_vel_measured, lin_vel_term), dim=-1)

        # Fixed 9-wide tracking slot: pose actual is controlled independently
        # from pose/velocity errors. Errors are command - actual.
        if self.cfg.dog.observe_pose_actual:
            pose_actual = pose_measured
        else:
            pose_actual = torch.zeros(self.num_envs, 3, device=self.device)
        if self.cfg.dog.observe_track_error:
            height_target = float(self.cfg.rewards.base_height_target) + self.commands_dog[:, dog_cmd_idx["body_height"]]
            pose_target = torch.stack(
                (
                    height_target * self.obs_scales.body_height_cmd,
                    self.commands_dog[:, dog_cmd_idx["body_pitch"]] * self.obs_scales.body_pitch_cmd,
                    self.commands_dog[:, dog_cmd_idx["body_roll"]] * self.obs_scales.body_roll_cmd,
                ),
                dim=-1,
            )
            pose_error = pose_target - pose_measured
            velocity_error = torch.cat(
                (
                    self.commands_dog[:, :2] * self.obs_scales.lin_vel - tracking_lin_vel_measured[:, :2],
                    self.commands_dog[:, 2:3] * self.obs_scales.ang_vel - ang_vel_measured[:, 2:3],
                ),
                dim=-1,
            )
        else:
            pose_error = torch.zeros(self.num_envs, 3, device=self.device)
            velocity_error = torch.zeros(self.num_envs, 3, device=self.device)
        track_obs = torch.cat((pose_actual, pose_error, velocity_error), dim=-1)
        obs_buf = torch.cat((obs_buf, track_obs), dim=-1)

        if self.cfg.env.observe_yaw:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0]).unsqueeze(1)
            # heading_error = torch.clip(0.5 * wrap_to_pi(heading), -1., 1.).unsqueeze(1)
            obs_buf = torch.cat((obs_buf, heading), dim=-1)

        if self.cfg.env.observe_contact_states:
            obs_buf = torch.cat(
                (obs_buf, (self.contact_forces[:, self.feet_indices, 2] > 1.0).view(self.num_envs, -1) * 1.0), dim=1
            )

        arm_slice = slice(self.num_actions_loco, self.num_actions_loco + self.num_actions_arm)
        arm_pos = (self.dof_pos[:, arm_slice] - self.default_dof_pos[:, arm_slice]) * self.obs_scales.dof_pos
        arm_vel = self.dof_vel[:, arm_slice] * self.obs_scales.dof_vel
        obs_buf = torch.cat((obs_buf, arm_pos, arm_vel), dim=-1)

        if dog_obs_noise_enabled:
            # Per-element zero-mean Gaussian sensor noise. The configured
            # dog_obs_noise_scale_vec values are interpreted as standard
            # deviations in the already-scaled observation space.
            obs_buf = obs_buf + torch.randn_like(obs_buf) * self.dog_obs_noise_scale_vec

        privileged_obs_buf = self._get_physics_privileged_observations("dog")

        assert privileged_obs_buf.shape[1] == self.cfg.dog.dog_num_privileged_obs, (
            f"dog num_privileged_obs ({self.cfg.dog.dog_num_privileged_obs}) \
                           != the number of privileged observations ({privileged_obs_buf.shape[1]}),\
                               you will discard data from the student!"
        )

        # return clipped obs, clipped states (None), rewards, dones and infos
        obs_builder = ObservationBuilder(self, "dog", self.cfg.dog.dog_num_observations)
        obs_builder.add(obs_buf)
        obs_buf = obs_builder.build()

        # Simulated dropped sensor frames: each droppable segment (a real
        # sensor reading -- IMU, encoders, contacts, ... -- see
        # _dog_obs_layout) independently, per env, has a chance of
        # re-delivering its own last value instead of this step's fresh
        # one. Commanded/internal segments are never dropped.
        drop_prob = float(getattr(self.cfg.domain_rand, "dog_obs_frame_drop_prob", 0.0))
        if drop_prob > 0.0 and self.dog_obs_droppable_segments:
            dropped = (
                torch.rand(self.num_envs, len(self.dog_obs_droppable_segments), device=self.device) < drop_prob
            )
            for i, (start, end) in enumerate(self.dog_obs_droppable_segments):
                mask = dropped[:, i].unsqueeze(-1)
                obs_buf[:, start:end] = torch.where(mask, self.dog_last_delivered_obs[:, start:end], obs_buf[:, start:end])
        self.dog_last_delivered_obs = obs_buf.clone()

        if privileged_obs_buf is not None:
            privileged_obs_buf = clip_observation(self, privileged_obs_buf)

        return obs_buf, privileged_obs_buf
