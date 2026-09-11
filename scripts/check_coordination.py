#!/usr/bin/env python3
"""Bounded IsaacGym integration check; zero dog actions do not evaluate policy quality."""
import isaacgym
import argparse
import json
from pathlib import Path

import torch
from go1_gym.envs.config import build_roboduet_config
from go1_gym.envs.roboduet.utils import StageSchedule
from go1_gym.envs.roboduet.wbc_env import WBCEnv
from go1_gym.envs.roboduet.wbc_env_wrapper import HistoryWrapper
from go1_gym.utils import global_switch, set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=list("ABCDEF"), default="C")
    parser.add_argument("--experiment_config", default=None)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--iteration", type=int, default=25000)
    parser.add_argument("--sim_device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path("tmp/coordination_check.json"))
    args = parser.parse_args()
    if args.steps < 1 or args.num_envs < 1 or args.iteration < 0:
        parser.error("steps and num_envs must be positive; iteration must be nonnegative")
    args.robot, args.train_stage, args.dyna_gait = "go2_x5", "stage1", True
    set_seed(11)
    cfg = build_roboduet_config(args)
    cfg.env.arm_policy_enabled = False
    cfg.env.record_video = False
    cfg.asset.render_sphere = False
    StageSchedule("stage1", max(35000, args.iteration + 1), 8000, cfg.env.stage1_arm_ramp_iterations).configure(global_switch)
    global_switch.init_sigmoid_lr()
    global_switch.count = global_switch.stage1_count = args.iteration
    env = HistoryWrapper(WBCEnv(sim_device=args.sim_device, headless=True, cfg=cfg))
    try:
        env.reset()
        dog = torch.zeros(args.num_envs, cfg.dog.num_actions_loco, device=env.device)
        arm = torch.zeros(args.num_envs, cfg.arm.num_actions_arm, device=env.device)
        resets = 0
        minimum_frequency = float("inf")
        maximum_frequency = 0.0
        with torch.no_grad():
            for _ in range(args.steps):
                reward_dog, reward_arm, done, _ = env.step(dog, arm)
                observations = env.get_dog_observations()
                for value in (reward_dog, reward_arm, env.root_states, env.dof_pos, env.dof_vel,
                              observations["obs"], observations["obs_history"]):
                    assert torch.isfinite(value).all(), "nonfinite rollout value"
                assert observations["obs"].shape == (args.num_envs, 90)
                assert observations["obs_history"].shape == (args.num_envs, 2700)
                moving = torch.norm(env.commands_dog[:, :3], dim=1) >= 0.1
                frequency = env.commands_dog[moving, 6]
                if frequency.numel():
                    minimum_frequency = min(minimum_frequency, frequency.min().item())
                    maximum_frequency = max(maximum_frequency, frequency.max().item())
                    assert (frequency >= cfg.commands.limit_gait_frequency[0]).all()
                    assert (frequency <= cfg.commands.limit_gait_frequency[1]).all()
                resets += done.sum().item()
        result = dict(experiment=args.experiment, simulated_iteration=args.iteration, steps=args.steps,
                      num_envs=args.num_envs, terrain=cfg.terrain.mesh_type,
                      arm_intensity=env.stage1_arm_curriculum_intensity,
                      hard_cohort_fraction=env.reset_mixture.hard.float().mean().item(),
                      moving_frequency_min_hz=minimum_frequency, moving_frequency_max_hz=maximum_frequency,
                      resets_with_untrained_zero_actions=resets,
                      commands=env.coordination_commands.metrics(), arm=env.coordination_arm.metrics())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps(result, indent=2, allow_nan=False))
    finally:
        env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()
