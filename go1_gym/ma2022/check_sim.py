"""Bounded integration check: python -m go1_gym.ma2022.check_sim."""

import argparse
from dataclasses import replace
import tempfile
from pathlib import Path

import isaacgym  # Must precede torch.
import torch

from go1_gym.envs.config.ma2022 import MaTrainingConfig, build_ma_config
from go1_gym.ma2022.env import MaLocomotionEnv
from go1_gym.ma2022.models import Student, Teacher
from go1_gym.ma2022.training import (
    export_student, load_checkpoint, save_checkpoint, student_iteration, teacher_iteration,
)
from go1_gym.utils.global_switch import global_switch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terrain", choices=("plane", "trimesh"), default="plane")
    parser.add_argument("--sim_device", default="cuda:0")
    args = parser.parse_args()
    torch.manual_seed(42)
    cfg = build_ma_config(4, terrain=args.terrain)
    recipe = replace(MaTrainingConfig(), hidden_dim=16, embedding_dim=8,
                     rollout_steps=4, ppo_epochs=1, minibatches=1,
                     disturbance_std=(0.,)*6)
    global_switch.switch_flag = False
    global_switch.count = 0
    global_switch.pretrained_to_wbc_start = 10**12
    env = MaLocomotionEnv(cfg, recipe, args.sim_device, headless=True)
    try:
        assert env.reset()["proprio"].shape == (4, 76)
        ids = torch.tensor([0, 2], device=env.device)
        other = torch.tensor([1, 3], device=env.device)
        env.step(torch.zeros(4, 16, device=env.device))
        # Neither inherited DR flags nor global reward switching may change
        # the Ma dynamics/reward contract.
        env.cfg.domain_rand.randomize_Kp_factor = True
        env._randomize_dof_props(torch.arange(4, device=env.device), env.cfg)
        assert torch.all(env.Kp_factors == 1.) and torch.all(env.motor_offsets == 0.)
        env.compute_reward()
        expected_reward = env.rew_buf_dog.clone()
        original_get_scales = global_switch.get_reward_scales
        global_switch.get_reward_scales = lambda: {"tracking_lin_vel": 1e9}
        try:
            env.compute_reward()
            torch.testing.assert_close(env.rew_buf_dog, expected_reward)
        finally:
            global_switch.get_reward_scales = original_get_scales
        before = env.wrench.knots.clone()
        env.reset_idx(ids)
        torch.testing.assert_close(env.wrench.knots[other], before[other])
        assert not torch.equal(env.wrench.knots[ids], before[ids])
        torch.testing.assert_close(env.previous_twist[ids], env.root_states[ids, 7:13])
        assert (env.policy_actions[ids] == 0).all()
        assert (env.previous_policy_actions[ids] == 0).all()
        # A reset cannot create an artificial finite-difference impulse.
        env._wrench_substep = 0
        env._arm_decimation_hook()
        torch.testing.assert_close(env.applied_wrench[ids], env.wrench.evaluate([0.])[:, 0][ids])
        # Force the auto-reset path and verify terminal observations survive.
        env.episode_length_buf[:] = env.cfg.env.max_episode_length
        obs, _, done, _ = env.step(torch.zeros(4, 16, device=env.device))
        assert done.all() and env.terminal_timeout.any()
        assert (env.episode_length_buf == 0).all()
        assert (env.last_actions == 0).all()
        torch.testing.assert_close(obs["proprio"][:, 3:6],
                                   env.observations()["proprio"][:, 3:6])
        teacher = Teacher(env.observation_dims, recipe).to(env.device)
        optimizer = torch.optim.Adam(teacher.parameters(), lr=recipe.learning_rate)
        before = teacher.actor[0].weight.detach().clone()
        obs, metrics = teacher_iteration(env, teacher, optimizer, recipe, obs)
        assert not torch.equal(before, teacher.actor[0].weight)
        teacher.requires_grad_(False)
        student = Student(env.observation_dims, recipe).to(env.device)
        student_optimizer = torch.optim.Adam(student.parameters(), lr=recipe.learning_rate)
        states = (torch.zeros(4, recipe.hidden_dim, device=env.device),
                  torch.zeros(4, recipe.hidden_dim, device=env.device))
        reset = torch.ones(4, dtype=torch.bool, device=env.device)
        before = student.wrench_rnn.weight_ih.detach().clone()
        obs, states, reset, losses = student_iteration(
            env, student, teacher, student_optimizer, recipe, obs, states, reset)
        assert not torch.equal(before, student.wrench_rnn.weight_ih)
        with tempfile.TemporaryDirectory(prefix="ma2022_check_") as directory:
            path = Path(directory)
            save_checkpoint(path / "student.pt", student, student_optimizer, 1, "student", env, recipe)
            checkpoint = load_checkpoint(path / "student.pt")
            assert checkpoint["env_cfg"]["reward_scales"]["orientation"] == 1.
            assert "roboduet_ma2022_" not in checkpoint["env_cfg"]["asset"]["file"]
            export_student(student, path / "student_jit.pt")
            scripted = torch.jit.load(str(path / "student_jit.pt"), map_location=env.device)
            result = scripted(obs["student_proprio"], obs["student_wrench"], obs["student_scan"],
                              *states, reset)
            assert result[0].shape == (4, 16) and torch.isfinite(result[0]).all()
        print(f"PASS: {args.terrain}: forces, selective reset, timeouts, PPO, distillation, checkpoint, JIT")
        print(f"Teacher: {metrics}; student: {losses}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
