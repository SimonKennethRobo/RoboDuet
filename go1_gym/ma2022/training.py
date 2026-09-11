"""PPO teacher and on-policy truncated-BPTT student distillation."""

import math
import os
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import torch
from torch.distributions import Normal

from .models import Teacher, Student, DeployedStudent, distillation_losses


FORMAT = "ma2022-locomotion-v3"


def physical_metrics(env):
    """Pre-reset physical measurements, independent of reward weights."""
    p = env.terminal_observation["proprio"]
    return {
        "velocity_error_mps": (p[:, 3:5] - p[:, 9:11]).norm(dim=-1).mean().item(),
        "base_tilt_rad": torch.acos((-p[:, 2]).clamp(-1, 1)).mean().item(),
        "roll_pitch_rate_rad_s": p[:, 6:8].norm(dim=-1).mean().item(),
        "termination_fraction": (env.reset_buf & ~env.terminal_timeout).float().mean().item(),
    }


def gae(rewards, values, next_values, dones, timeouts, gamma, lam):
    """next_values must come from terminal observations, not auto-reset obs."""
    result = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        bootstrap = (~dones[t] | timeouts[t]).float()
        delta = rewards[t] + gamma * next_values[t] * bootstrap - values[t]
        running = delta + gamma * lam * (~dones[t]).float() * running
        result[t] = running
    return result, result + values


def save_checkpoint(path, model, optimizer, iteration, stage, env, cfg):
    """Critical writes propagate errors; publish only a complete checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(format=FORMAT, stage=stage, iteration=iteration,
                    model=model.state_dict(), optimizer=optimizer.state_dict(),
                    recipe=asdict(cfg), env_cfg=env.source_config, dims=env.observation_dims), temporary)
    os.replace(temporary, path)


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("format") != FORMAT:
        raise ValueError("Expected a Ma2022 v3 checkpoint; v1/v2 and RoboDuet checkpoints use a different MDP")
    return checkpoint


def teacher_iteration(env, model, optimizer, cfg, obs):
    transitions = []
    metrics = {}
    for _ in range(cfg.rollout_steps):
        with torch.no_grad():
            mean, value, _, _ = model(obs)
            distribution = Normal(mean, model.log_std.clamp(-5, math.log(cfg.max_action_std)).exp())
            action = distribution.sample()
            log_prob = distribution.log_prob(action).sum(-1)
            next_obs, reward, done, _ = env.step(action)
            for key, metric_value in physical_metrics(env).items():
                metrics[key] = metrics.get(key, 0.) + metric_value / cfg.rollout_steps
            next_value = model(env.terminal_observation)[1]
            transitions.append((obs, action, log_prob, value, reward.clone(),
                                done.clone(), env.terminal_timeout.clone(), next_value))
            obs = next_obs
    observations = {k: torch.stack([r[0][k] for r in transitions]).flatten(0, 1)
                    for k in ("proprio", "wrench", "scan", "privileged")}
    actions, old_log_prob, values, rewards, dones, timeouts, next_values = (
        torch.stack([row[i] for row in transitions]) for i in range(1, 8))
    advantages, returns = gae(rewards, values, next_values, dones, timeouts, cfg.gamma, cfg.gae_lambda)
    advantages = advantages.flatten()
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
    returns, actions, old_log_prob = returns.flatten(), actions.flatten(0, 1), old_log_prob.flatten()
    count = actions.shape[0]
    totals = []
    for _ in range(cfg.ppo_epochs):
        for ids in torch.randperm(count, device=actions.device).tensor_split(min(cfg.minibatches, count)):
            mean, value, _, _ = model({k: v[ids] for k, v in observations.items()})
            distribution = Normal(mean, model.log_std.clamp(-5, math.log(cfg.max_action_std)).exp())
            log_prob = distribution.log_prob(actions[ids]).sum(-1)
            ratio = torch.exp(log_prob - old_log_prob[ids])
            policy_loss = -torch.minimum(ratio * advantages[ids],
                                        ratio.clamp(1-cfg.clip_ratio, 1+cfg.clip_ratio) * advantages[ids]).mean()
            value_loss = (value - returns[ids]).square().mean()
            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * distribution.entropy().sum(-1).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite teacher loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            totals.append(loss.detach())
    metrics.update(loss=torch.stack(totals).mean().item(), reward=rewards.mean().item())
    return obs, metrics


def student_iteration(env, student, teacher, optimizer, cfg, obs, states, reset):
    # Student controls the environment so imitation covers its own failures.
    # Carry history across optimizer updates, detach only at BPTT boundaries.
    h_w, h_b = (h.detach() for h in states)
    accumulated = {}
    physical = {}
    reward_sum = 0.
    for _ in range(cfg.rollout_steps):
        out = student(obs["student_proprio"], obs["student_wrench"], obs["student_scan"], h_w, h_b, reset)
        with torch.no_grad():
            target = teacher(obs)
        terms = distillation_losses(out, target, obs)
        for key, value in terms.items():
            accumulated[key] = accumulated.get(key, 0.) + value / cfg.rollout_steps
        h_w, h_b = out[1:3]
        with torch.no_grad():
            obs, reward, reset, _ = env.step(out[0].detach())
            for key, value in physical_metrics(env).items():
                physical[key] = physical.get(key, 0.) + value / cfg.rollout_steps
            reset = reset.clone()
            reward_sum += reward.mean().item() / cfg.rollout_steps
    loss = sum(cfg.loss_weights[key] * value for key, value in accumulated.items())
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite student loss")
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.max_grad_norm)
    optimizer.step()
    metrics = {key: value.detach().item() for key, value in accumulated.items()}
    metrics["loss"], metrics["reward"] = loss.item(), reward_sum
    metrics.update(physical)
    return obs, (h_w.detach(), h_b.detach()), reset, metrics


def export_student(student, path):
    # Explicit states avoid hidden process-global history in deployment.
    deployed = DeployedStudent(deepcopy(student)).eval().cpu()
    scripted = torch.jit.script(deployed)
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    scripted.save(str(temporary))
    os.replace(temporary, path)
