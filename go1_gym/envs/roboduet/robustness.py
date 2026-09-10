"""Fixed reset cohorts and step-weighted diagnostics; no simulator API calls."""
import math

from go1_gym.file_io import optional_output

import torch


class FixedResetMixture:
    def __init__(self, terrain, num_envs, num_train_envs, device):
        self.cfg = terrain
        self.enabled = terrain.reset_mode == "fixed_mixture"
        self.hard = torch.zeros(num_envs, dtype=torch.bool, device=device)
        generator = torch.Generator(device="cpu").manual_seed(terrain.reset_mix_seed)
        # Partition train/eval independently so eval env count cannot change
        # training membership. A dedicated RNG leaves policy/DR RNG untouched.
        if self.enabled:
            for lo, hi in ((0, num_train_envs), (num_train_envs, num_envs)):
                count = int(round((hi - lo) * terrain.reset_mix_hard_fraction))
                ids = torch.randperm(hi - lo, generator=generator)[:count] + lo
                self.hard[ids.to(device)] = True
        self.episode_tilt_limit = torch.zeros(num_envs, device=device)
        self.episode_tilt = torch.zeros(num_envs, device=device)

    def limit(self, env_ids, iteration):
        t = self.cfg
        u = min(1.0, max(0.0, (iteration - t.reset_mix_start_iteration) / t.reset_mix_ramp_iterations))
        hard_limit = t.reset_mix_easy_tilt_rad + (t.reset_mix_hard_tilt_rad - t.reset_mix_easy_tilt_rad) * u
        value = t.reset_mix_easy_tilt_rad + self.hard[env_ids].float() * (hard_limit - t.reset_mix_easy_tilt_rad)
        self.episode_tilt_limit[env_ids] = value
        return value[:, None]


class RobustnessMetrics:
    """Raw sums per group/age window, including ongoing and failed episodes.

    Drained at rollout boundaries, not reset boundaries. Old Performance/* is
    intentionally independent and keeps its historical episode-weighted meaning.
    """
    groups = ("all", "easy", "hard")
    ages = ("all", "early", "late")
    axes = ("vx", "vy", "yaw")
    events = ("episodes", "failures", "early_failures", "height", "orientation", "pushes")

    def __init__(self, hard, dt, early_window_s):
        self.hard = hard
        self.dt = dt
        self.early_steps = max(1, int(math.ceil(early_window_s / dt)))
        self.weights = torch.stack((torch.ones_like(hard), ~hard, hard)).float()
        self.sums = torch.zeros(3, 3, 7, device=hard.device)
        self.counts = torch.zeros(3, len(self.events), device=hard.device)
        self.tilt_sums = torch.zeros(3, 3, device=hard.device)  # count, actual tilt, sampled limit

    def update(self, error, steps, done, timeout, height_failure, orientation_failure, pushed):
        early = steps <= self.early_steps
        features = torch.cat((error.abs(), error.square(), torch.ones_like(error[:, :1])), dim=1)
        for i, age in enumerate((torch.ones_like(early), early, ~early)):
            self.sums[:, i] += self.weights @ (features * age[:, None])
        failed = done & ~timeout
        events = torch.stack((done, failed, failed & early, done & height_failure,
                              done & orientation_failure, pushed), dim=1).float()
        self.counts += self.weights @ events

    def record_reset(self, ids, tilt, limit):
        if ids.numel():
            values = torch.stack((torch.ones_like(tilt), tilt, limit), dim=1)
            self.tilt_sums += self.weights[:, ids] @ values

    def pop(self):
        sums = self.sums.cpu().tolist()
        counts = self.counts.cpu().tolist()
        tilts = self.tilt_sums.cpu().tolist()
        self.sums.zero_()
        self.counts.zero_()
        self.tilt_sums.zero_()
        result = {"Robustness/metrics_version": 1}
        for g, group in enumerate(self.groups):
            for a, age in enumerate(self.ages):
                prefix = f"TrackingTime/{group}" if age == "all" else f"TrackingAge/{group}/{age}"
                vals = sums[g][a]
                count = vals[6]
                result[prefix + "/sample_count"] = count
                for j, axis in enumerate(self.axes):
                    result[f"{prefix}/{axis}_abs_error_sum"] = vals[j]
                    result[f"{prefix}/{axis}_sq_error_sum"] = vals[j + 3]
                    if count:
                        unit = "rad_s" if axis == "yaw" else "mps"
                        result[f"{prefix}/{axis}_mae_{unit}"] = vals[j] / count
                        result[f"{prefix}/{axis}_rmse_{unit}"] = math.sqrt(vals[j + 3] / count)
            for name, value in zip(self.events, counts[g]):
                result[f"Termination/{group}/{name}_count"] = value
            count, tilt, limit = tilts[g]
            result[f"ResetMix/{group}/reset_count"] = count
            result[f"ResetMix/{group}/step_count"] = sums[g][0][6]
            if count:
                result[f"ResetMix/{group}/initial_tilt_mean_rad"] = tilt / count
                result[f"ResetMix/{group}/initial_limit_mean_rad"] = limit / count
        return result


def log_robustness_iteration(env, log_dir, iteration, wandb_dict):
    """One JSONL row per rollout; raw sums allow correct offline window pooling."""
    import json
    from pathlib import Path
    pop = getattr(env, "pop_robustness_metrics", None)
    if pop is None:
        return
    values = pop()
    if not values:
        return
    from go1_gym.logging_metrics import robustness_metrics
    wandb_dict.update(robustness_metrics(values))
    row = {"iteration": int(iteration), **values}
    with optional_output("robustness.jsonl"), (Path(log_dir) / "robustness.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
