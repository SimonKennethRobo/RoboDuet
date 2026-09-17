"""PD target generator for random motion, joint-space harmonics, and pose holds.

This is a disturbance generator, not a Cartesian trajectory controller. World
frame EE paths are evaluated through the existing MPC/robot closed loop.
"""
import math
import numpy as np
import torch


class CoordinationArm:
    names = ("random", "structured", "hold")

    def __init__(self, cfg, num_envs, device, dt):
        self.cfg, self.c, self.dt, self.device = cfg, cfg.env.coordination_arm, dt, device
        self.n = num_envs
        self.joints = cfg.arm.num_actions_arm
        order = np.random.RandomState(self.c.seed).permutation(num_envs)
        random_count = round(num_envs * self.c.fractions[0])
        structured_count = min(num_envs - random_count, round(num_envs * self.c.fractions[1]))
        self.mode = torch.full((num_envs,), 2, dtype=torch.long, device=device)
        self.mode[torch.as_tensor(order[:random_count], device=device)] = 0
        self.mode[torch.as_tensor(order[random_count:random_count + structured_count], device=device)] = 1
        self.speed_scale = torch.ones(num_envs, 1, device=device)
        for index, subset in enumerate(np.array_split(order[:random_count], 3)):
            self.speed_scale[torch.as_tensor(subset, device=device)] = self.c.random_speed_scales[index]
        self.q = torch.zeros(num_envs, self.joints, device=device)
        self.v = torch.zeros_like(self.q)
        self.accel = torch.zeros_like(self.q)
        self.phase = torch.zeros_like(self.q)
        self.frequency = torch.zeros_like(self.q)
        self.center = torch.zeros_like(self.q)
        self.amplitude = torch.zeros_like(self.q)
        self.hold = torch.zeros_like(self.q)
        self.next_accel_step = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.vmax = torch.tensor(self.c.max_joint_velocity, device=device).clamp(max=cfg.env.stage1_arm_max_vel)
        self.amax = torch.tensor(self.c.max_joint_acceleration, device=device).clamp(max=cfg.env.stage1_arm_max_accel)
        self.count = torch.zeros(3, device=device)
        self.moving_count = torch.zeros(3, device=device)
        self.target_speed_sum = torch.zeros((), device=device)
        self.actual_speed_sum = torch.zeros((), device=device)
        self.actual_accel_sum = torch.zeros((), device=device)
        self.actual_accel_samples = torch.zeros((), device=device)
        self.actual_speed_peak = torch.zeros((), device=device)
        self.actual_accel_peak = torch.zeros((), device=device)
        self.saturated_sum = torch.zeros((), device=device)
        self.samples = 0
        self.previous_actual_velocity = torch.zeros_like(self.q)
        self.actual_velocity_valid = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(self, env, ids):
        arm = slice(env.num_actions_loco, env.num_actions_loco + env.num_actions_arm)
        self.q[ids] = env.dof_pos[ids, arm]
        self.v[ids] = 0
        self.accel[ids] = 0
        self.actual_velocity_valid[ids] = False
        lower, upper = env.dof_pos_limits[arm, 0], env.dof_pos_limits[arm, 1]
        middle = (lower + upper) / 2
        radius = (upper - lower) / 2 * self.c.workspace_fraction
        self.center[ids] = middle + (torch.rand(len(ids), self.joints, device=self.device) * 2 - 1) * radius * 0.25
        self.amplitude[ids] = radius * (0.3 + 0.4 * torch.rand(len(ids), self.joints, device=self.device))
        self.hold[ids] = middle + (torch.rand(len(ids), self.joints, device=self.device) * 2 - 1) * radius
        self.phase[ids] = torch.rand(len(ids), self.joints, device=self.device) * 2 * math.pi
        low, high = self.c.structured_frequency_hz
        base = low + (high - low) * torch.rand(len(ids), 1, device=self.device)
        # Mixed 1:1 and 1:2 joint harmonics produce circles/figure-eight-like
        # joint paths with randomized phase; no claim of Cartesian path shape.
        self.frequency[ids] = base * torch.randint(1, 3, (len(ids), self.joints), device=self.device)
        period = max(1, math.ceil(self.cfg.env.stage1_arm_accel_resample_time_s / self.dt))
        self.next_accel_step[ids] = env.common_step_counter + torch.randint(0, period, (len(ids),), device=self.device)

    def step(self, env, intensity):
        arm = slice(env.num_actions_loco, env.num_actions_loco + env.num_actions_arm)
        default = env.default_dof_pos[:, arm]
        if intensity <= 0:
            self.q[:] = env.stage1_arm_fixed_dof_pos
            self.v.zero_()
            self.accel.zero_()
        else:
            self.phase += 2 * math.pi * self.frequency * self.dt * intensity
            vmax = self.vmax * intensity * self.speed_scale
            amax = self.amax * intensity * self.speed_scale
            due = (env.common_step_counter >= self.next_accel_step).nonzero().flatten()
            if len(due):
                self.accel[due] = (torch.rand(len(due), self.joints, device=self.device) * 2 - 1) * amax[due]
                period = max(1, math.ceil(self.cfg.env.stage1_arm_accel_resample_time_s / self.dt))
                self.next_accel_step[due] = env.common_step_counter + period
            accel = self.accel.clone()
            zero = torch.rand(self.n, 1, device=self.device) < self.cfg.env.stage1_arm_zero_accel_probability
            accel = torch.where(zero, 0.0, accel)
            desired_v = self.v + accel * self.dt
            stop = torch.rand(self.n, 1, device=self.device) < self.cfg.env.stage1_arm_zero_vel_probability
            desired_v = torch.where(stop, 0.0, desired_v)
            target = self.center + self.amplitude * intensity * torch.sin(self.phase)
            structured_v = self.c.target_gain * (target - self.q)
            hold_v = self.c.target_gain * (self.hold - self.q)
            desired_v = torch.where((self.mode == 1)[:, None], structured_v, desired_v)
            desired_v = torch.where((self.mode == 2)[:, None], hold_v, desired_v)
            lower, upper = env.dof_pos_limits[arm, 0], env.dof_pos_limits[arm, 1]
            # Account for both this step's travel and subsequent braking.
            positive_limit = torch.sqrt((amax * self.dt) ** 2 + 2 * amax * (upper - self.q).clamp(min=0)) - amax * self.dt
            negative_limit = torch.sqrt((amax * self.dt) ** 2 + 2 * amax * (self.q - lower).clamp(min=0)) - amax * self.dt
            desired_v = torch.maximum(torch.minimum(desired_v, torch.minimum(vmax, positive_limit)),
                                      -torch.minimum(vmax, negative_limit))
            self.v += torch.maximum(torch.minimum(desired_v - self.v, amax * self.dt), -amax * self.dt)
            self.q[:] = torch.maximum(torch.minimum(self.q + self.v * self.dt, upper), lower)
        env.stage1_arm_target_offset[:] = self.q - default
        env.stage1_arm_target_vel[:] = self.v
        env.stage1_arm_target_accel[:] = self.accel
        env.actions[:, arm] = (self.q - default) / env.cfg.control.action_scale

    def after_physics(self, env):
        arm = slice(env.num_actions_loco, env.num_actions_loco + env.num_actions_arm)
        actual = env.dof_vel[:, arm]
        moving = actual.abs().amax(dim=1) > 0.1
        for index in range(3):
            mask = self.mode == index
            self.count[index] += mask.sum()
            self.moving_count[index] += (mask & moving).sum()
        self.target_speed_sum += self.v.abs().sum()
        self.actual_speed_sum += actual.abs().sum()
        acceleration = (actual - self.previous_actual_velocity).abs() / self.dt * self.actual_velocity_valid[:, None]
        self.actual_accel_sum += acceleration.sum()
        self.actual_accel_samples += self.actual_velocity_valid.sum() * self.joints
        # PPO collects inside inference_mode but drains metrics outside it.
        # Preserve the original normal buffers instead of replacing them with
        # inference tensors, which cannot be zeroed by the logger afterwards.
        self.actual_speed_peak.copy_(torch.maximum(self.actual_speed_peak, actual.abs().max()))
        self.actual_accel_peak.copy_(torch.maximum(self.actual_accel_peak, acceleration.max()))
        self.previous_actual_velocity[:] = actual
        self.actual_velocity_valid[:] = True
        torque = env.torques[:, arm].abs()
        self.saturated_sum += (torque >= 0.98 * env.torque_limits[arm]).sum()
        self.samples += actual.numel()

    def metrics(self):
        result = {}
        total = max(1, self.count.sum().item())
        for index, name in enumerate(self.names):
            count = self.count[index].item()
            result[f"ArmMotion/{name}_time_fraction"] = count / total
            result[f"ArmMotion/{name}_moving_fraction"] = self.moving_count[index].item() / max(1, count)
        for name, value in (("target_abs_velocity_rad_s", self.target_speed_sum),
                            ("actual_abs_velocity_rad_s", self.actual_speed_sum),
                            ("torque_saturation_fraction", self.saturated_sum)):
            result[f"ArmMotion/{name}"] = value.item() / max(1, self.samples)
            value.zero_()
        result["ArmMotion/actual_abs_acceleration_rad_s2"] = self.actual_accel_sum.item() / max(1, self.actual_accel_samples.item())
        result["ArmMotion/actual_peak_velocity_rad_s"] = self.actual_speed_peak.item()
        result["ArmMotion/actual_peak_acceleration_rad_s2"] = self.actual_accel_peak.item()
        for value in (self.actual_accel_sum, self.actual_accel_samples, self.actual_speed_peak, self.actual_accel_peak):
            value.zero_()
        self.count.zero_()
        self.moving_count.zero_()
        self.samples = 0
        return result
