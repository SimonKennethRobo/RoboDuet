"""Stage-1 command windows with independent clocks and a velocity-only curriculum.

Pose/gait are sampled uniformly. Their updates never clear velocity statistics
or assign a joint-command bin for values that were not actually applied.
"""
import math
import json
from pathlib import Path
import numpy as np
import torch

from go1_gym.envs.base.curriculum import RewardThresholdCurriculum
from go1_gym.file_io import optional_output


def velocity_window_range(config, iteration):
    schedule = config.velocity_schedule
    if iteration >= schedule[-1][0]:
        return config.final_velocity_range_s
    for (start, first), (end, last) in zip(schedule, schedule[1:]):
        if iteration < end:
            value = first + (last - first) * max(0.0, (iteration - start) / (end - start))
            return [value, value]
    return [schedule[0][1]] * 2


def log_coordination_iteration(env, log_dir, iteration, wandb_dict):
    values = {}
    for name in ("coordination_commands", "coordination_arm"):
        sampler = getattr(env, name, None)
        if sampler is not None:
            values.update(sampler.metrics())
    if values:
        if getattr(env.cfg.env, 'quarantine_invalid_physics', False):
            values['Numerics/invalid_envs_total'] = env.numerical_fault_count
        wandb_dict.update(values)
        with optional_output("coordination.jsonl"), (Path(log_dir) / "coordination.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(iteration=int(iteration), **values), allow_nan=False) + "\n")


class CoordinationCommands:
    groups = ("velocity", "pose", "gait")

    def __init__(self, cfg, num_envs, device, dt):
        self.cfg, self.c, self.device, self.dt = cfg, cfg.commands.coordination, device, dt
        self.num_envs = num_envs
        c = cfg.commands
        self.curriculum = RewardThresholdCurriculum(
            seed=c.curriculum_seed,
            x_vel=(*c.limit_vel_x, c.num_bins_vel_x),
            y_vel=(*c.limit_vel_y, c.num_bins_vel_y),
            yaw_vel=(*c.limit_vel_yaw, c.num_bins_vel_yaw),
        )
        self.curriculum.set_to(np.array([c.lin_vel_x[0], c.lin_vel_y[0], c.ang_vel_yaw[0]]),
                               np.array([c.lin_vel_x[1], c.lin_vel_y[1], c.ang_vel_yaw[1]]))
        if not self.curriculum.weights.any():
            raise ValueError("velocity curriculum initial ranges contain no bins")
        self.bins = np.zeros(num_envs, dtype=np.int64)
        self.next_step = torch.zeros(num_envs, 3, dtype=torch.long, device=device)
        self.started_step = self.next_step.clone()
        self.window_steps = self.next_step.clone()
        self.active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.walking_frequency = torch.full((num_envs,), sum(c.limit_gait_frequency) / 2, device=device)
        # A fixed, shuffled environment cohort gives a physical-step fraction,
        # independent of early resets or how many short windows are sampled.
        order = np.random.RandomState(c.curriculum_seed + 71).permutation(num_envs)
        self.short = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.short[torch.as_tensor(order[:round(num_envs * self.c.short_fraction)], device=device)] = True
        self.total_steps = 0
        self.transition_steps = torch.zeros((), device=device)
        self.standing_steps = torch.zeros((), device=device)
        self.low_speed_steps = torch.zeros((), device=device)
        self.duration_sum = torch.zeros(3, device=device)
        self.window_count = torch.zeros(3, device=device)
        self.change_sum = torch.zeros(3, device=device)
        self.pitch_error_sq = torch.zeros((), device=device)
        self.roll_error_sq = torch.zeros((), device=device)

    def _uniform(self, count, bounds):
        bounds = torch.tensor(bounds, device=self.device, dtype=torch.float)
        return bounds[:, 0] + torch.rand(count, len(bounds), device=self.device) * (bounds[:, 1] - bounds[:, 0])

    def _finish_velocity(self, env, ids):
        if getattr(env, 'numerical_fault_active', False):
            ids = ids[~env.numerical_fault_mask[ids]]
        ids = ids[self.active[ids] & (env.command_sums["ep_timesteps"][ids] > 0)]
        if ids.numel() == 0:
            return
        # The last physics step still used the old command and is included.
        steps = env.command_sums["ep_timesteps"][ids]
        rewards, thresholds, tracking, scales = [], [], {}, {}
        for key in ("tracking_lin_vel", "tracking_ang_vel", "tracking_contacts_shaped_force", "tracking_contacts_shaped_vel"):
            scale = env.pretrained_reward_scales.get(key, 0.0)
            if key not in env.command_sums or scale <= 0:
                continue
            value = env.command_sums[key][ids] / steps
            rewards.append(value)
            thresholds.append(env.curriculum_thresholds[key] * scale)
            if key in ("tracking_lin_vel", "tracking_ang_vel"):
                tracking[key], scales[key] = value, scale
        # Explicit stand windows do not expand the moving-command curriculum.
        moving = torch.norm(env.commands_dog[ids, :3], dim=1) >= 0.1
        if rewards and moving.any():
            self.curriculum.update(self.bins[ids[moving].cpu().numpy()],
                                   [r[moving] for r in rewards], thresholds,
                                   local_range=np.array([0.55, 0.55, 0.55]))
        env._update_reset_curriculum(tracking, scales)

    def _start(self, env, ids, group, iteration):
        if ids.numel() == 0:
            return
        step = env.common_step_counter
        active = self.active[ids]
        self.duration_sum[group] += ((step - self.started_step[ids, group]) * self.dt * active).sum()
        self.window_count[group] += active.sum()
        c = self.cfg.commands
        if group == 0:
            self._finish_velocity(env, ids)
            values, bins = self.curriculum.sample(len(ids))
            values = torch.as_tensor(values, dtype=env.commands_dog.dtype, device=self.device)
            category = torch.rand(len(ids), device=self.device)
            stand = category < self.c.standing_probability
            if self.c.low_speed_probability > 0:
                low = (category >= self.c.standing_probability) & (category < self.c.standing_probability + self.c.low_speed_probability)
                values[low] = self._uniform(int(low.sum()), self.c.low_speed_ranges)
            values[stand] = 0.0
            # Keep indices consistent even for deliberately overridden standing commands.
            coordinates = ((values.cpu().numpy() - self.curriculum.lows) /
                           np.array(list(self.curriculum.bin_sizes.values()))).astype(np.int64)
            coordinates = np.clip(coordinates, 0, np.array(list(self.curriculum.ls.values())) - 1)
            bins = np.ravel_multi_index(coordinates.T, tuple(self.curriculum.ls.values()))
            self.bins[ids.cpu().numpy()] = bins
            command_slice = slice(0, 3)
            bounds = velocity_window_range(self.c, iteration)
            duration = self._uniform(len(ids), [bounds])[:, 0]
            if iteration >= self.c.short_start_iteration:
                duration = torch.where(self.short[ids], self._uniform(len(ids), [self.c.short_range_s])[:, 0], duration)
            # Only the velocity clock owns command_sums and curriculum scores.
            for sums in env.command_sums.values():
                sums[ids] = 0
        elif group == 1:
            command_slice = slice(3, 6)
            values = self._uniform(len(ids), [c.limit_body_pitch, c.limit_body_roll, c.limit_body_height])
            duration = self._uniform(len(ids), [self.c.pose_range_s])[:, 0]
        else:
            command_slice = slice(6, 11)
            values = self._uniform(len(ids), [c.limit_gait_frequency, c.limit_footswing_height,
                                               c.limit_stance_width, c.limit_stance_length, c.limit_gait_duration])
            self.walking_frequency[ids] = values[:, 0]
            duration = self._uniform(len(ids), [self.c.gait_range_s])[:, 0]
        self.change_sum[group] += (torch.norm(values - env.commands_dog[ids, command_slice], dim=1) * active).sum()
        env.commands_dog[ids, command_slice] = values
        moving = torch.norm(env.commands_dog[ids, :3], dim=1) >= 0.1
        env.commands_dog[ids, 6] = torch.where(moving, self.walking_frequency[ids], 0.0)
        self.started_step[ids, group] = step
        self.window_steps[ids, group] = torch.ceil(duration / self.dt).long().clamp(min=1)
        self.next_step[ids, group] = step + self.window_steps[ids, group]

    def reset(self, env, ids, iteration):
        for group in range(3):
            self._start(env, ids, group, iteration)
        self.active[ids] = True

    def after_reward(self, env, iteration):
        # Log physical time, including failed and truncated episodes.
        self.total_steps += self.num_envs
        age = env.common_step_counter - self.started_step[:, 0]
        self.transition_steps += (age * self.dt <= self.c.transition_window_s).sum()
        moving = torch.norm(env.commands_dog[:, :3], dim=1) >= 0.1
        bounds = torch.as_tensor(self.c.low_speed_ranges, device=self.device)
        low = ((env.commands_dog[:, :3] >= bounds[:, 0]) & (env.commands_dog[:, :3] <= bounds[:, 1])).all(dim=1)
        self.standing_steps += (~moving).sum()
        self.low_speed_steps += (low & moving).sum()
        self.pitch_error_sq += ((env.pitch - env.commands_dog[:, 3]) ** 2).sum()
        self.roll_error_sq += ((env.roll - env.commands_dog[:, 4]) ** 2).sum()
        for group in range(3):
            ids = ((env.common_step_counter >= self.next_step[:, group]) & ~env.reset_buf & self.active).nonzero().flatten()
            self._start(env, ids, group, iteration)

    def metrics(self):
        result = {
            "Commands/velocity_short_cohort_fraction": self.short.float().mean().item(),
            "Commands/velocity_transition_time_fraction": self.transition_steps.item() / max(1, self.total_steps),
            "Commands/standing_time_fraction": self.standing_steps.item() / max(1, self.total_steps),
            "Commands/low_speed_time_fraction": self.low_speed_steps.item() / max(1, self.total_steps),
            "Performance/Dog/pitch_command_rmse_rad": math.sqrt(self.pitch_error_sq.item() / max(1, self.total_steps)),
            "Performance/Dog/roll_command_rmse_rad": math.sqrt(self.roll_error_sq.item() / max(1, self.total_steps)),
        }
        for index, name in enumerate(self.groups):
            count = max(1, self.window_count[index].item())
            result[f"Commands/{name}_realized_window_s"] = self.duration_sum[index].item() / count
            result[f"Commands/{name}_planned_window_s"] = self.window_steps[:, index].float().mean().item() * self.dt
            result[f"Commands/{name}_change_l2_mean"] = self.change_sum[index].item() / count
        self.total_steps = 0
        for value in (self.transition_steps, self.standing_steps, self.low_speed_steps, self.duration_sum, self.window_count, self.change_sum,
                      self.pitch_error_sq, self.roll_error_sq):
            value.zero_()
        return result
