"""Measured plant and persistent arm target interface shared by ID and OCS2.

The reference checkout is read-only. No simulator/policy source is modified.
"""
from pathlib import Path
import hashlib
import random
import xml.etree.ElementTree as ET

import isaacgym  # noqa: F401
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from benchmark.dog_policy.closed_loop_mpc import MPCEnv
from benchmark.dog_policy.evaluation import (
    _apply_benchmark_env_overrides, _load_cfg_from_pkl, configure_stage1,
    load_dog_policy_for_benchmark,
)
from go1_gym.envs.config import cfg_to_dict, configure_privileged_obs_dims
from go1_gym.envs.roboduet.wbc_env_wrapper import HistoryWrapper

DEFAULT_POLICY = "runs/stage1_rlmpc_benchmark_2_223425"
NOMINAL_Q = np.array([0., .6, .6, 0., 0., 0.])


class ServoPlant:
    def __init__(self, num_envs=1, seed=29, seconds=30., logdir=DEFAULT_POLICY, device="cuda:0"):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.set_num_threads(1)
        configure_stage1(0.)
        cfg = _load_cfg_from_pkl(logdir)
        _apply_benchmark_env_overrides(cfg, total_envs=num_envs, envs_per_policy=num_envs)
        cfg.domain_rand.mode = "none"
        cfg.terrain.reset_curriculum = False
        for key in ("z_init_range", "yaw_init_range", "pitch_init_range", "roll_init_range"):
            setattr(cfg.terrain, key, 0.)
        cfg.env.record_video = False
        cfg.env.keep_arm_fixed = False
        cfg.env.stage1_arm_curriculum = False
        cfg.env.stage1_arm_init_dof_pos_noise = 0.
        cfg.env.episode_length_s = seconds + 15.
        cfg.env.arm_policy_enabled = False
        for i, value in enumerate(NOMINAL_Q, 1):
            cfg.init_state.default_joint_angles["x5_joint" + str(i)] = float(value)
        configure_privileged_obs_dims(cfg)
        self.cfg = cfg
        self.asset = Path(cfg.asset.file.format(MINI_GYM_ROOT_DIR=str(Path(__file__).resolve().parents[2])))
        self.urdf = ET.parse(self.asset).getroot()
        joints = {j.get("name"): j for j in self.urdf.findall("joint")}
        self.q_low = np.array([float(joints[f"x5_joint{i}"].find("limit").get("lower")) for i in range(1, 7)]) + .03
        self.q_high = np.array([float(joints[f"x5_joint{i}"].find("limit").get("upper")) for i in range(1, 7)]) - .03
        span = cfg.normalization.clip_actions * cfg.control.action_scale
        self.q_low = np.maximum(self.q_low, NOMINAL_Q - span)
        self.q_high = np.minimum(self.q_high, NOMINAL_Q + span)
        limits = np.array([cfg.commands.limit_vel_x, cfg.commands.limit_vel_y, cfg.commands.limit_vel_yaw,
                           cfg.commands.limit_body_height, cfg.commands.limit_body_pitch])
        self.low = np.maximum(limits[:, 0], [-.6, -.4, -.8, -.06, -.2])
        self.high = np.minimum(limits[:, 1], [.6, .4, .8, .04, .2])
        self.base = MPCEnv(sim_device=device, headless=True, cfg=cfg, graphics_device_id=-1)
        self.env = HistoryWrapper(self.base)
        self.dt = self.base.dt
        self.num_envs = num_envs
        self.policy = load_dog_policy_for_benchmark(logdir, "last", cfg, device=device)
        self.gait = torch.tensor([np.mean(cfg.commands.limit_gait_frequency),
            np.mean(cfg.commands.limit_footswing_height), np.mean(cfg.commands.limit_stance_width),
            np.mean(cfg.commands.limit_stance_length), np.mean(cfg.commands.limit_gait_duration)], device=device)
        self.q_target = np.tile(NOMINAL_Q, (num_envs, 1))
        self.last_target_error = 0.
        self.env.reset()
        props = self.base.gym.get_actor_dof_properties(self.base.envs[0], self.base.actor_handles[0])
        self.manifest = {
            "policy": str(Path(logdir).resolve()),
            "checkpoint_sha256": hashlib.sha256((Path(logdir) / "checkpoints_dog/ac_weights_last_dog.pt").read_bytes()).hexdigest(),
            "asset": str(self.asset), "asset_sha256": hashlib.sha256(self.asset.read_bytes()).hexdigest(),
            "dt": self.dt, "seed": seed, "num_envs": num_envs,
            "interface": "persistent q_target += dt * velocity; clamp to joint/action bounds; no MPC-cycle reanchor",
            "control_type": cfg.control.control_type, "config": cfg_to_dict(cfg),
            "q_low": self.q_low.tolist(), "q_high": self.q_high.tolist(),
            "command_low": self.low.tolist(), "command_high": self.high.tolist(),
            "arm_drive_properties": {key: props[key][12:18].tolist() for key in props.dtype.names},
        }

    def step(self, command, velocity=None, target=None):
        if target is not None:
            self.q_target = np.clip(np.broadcast_to(target, self.q_target.shape), self.q_low, self.q_high).copy()
        elif velocity is not None:
            self.q_target = np.clip(self.q_target + self.dt * velocity, self.q_low, self.q_high)
        command = np.clip(np.broadcast_to(command, (self.num_envs, 5)), self.low, self.high)
        b = self.base
        b.commands_dog[:, :3] = torch.as_tensor(command[:, :3], device=b.device, dtype=torch.float)
        b.commands_dog[:, 3] = torch.as_tensor(command[:, 4], device=b.device, dtype=torch.float)
        b.commands_dog[:, 4] = 0.
        b.commands_dog[:, 5] = torch.as_tensor(command[:, 3], device=b.device, dtype=torch.float)
        b.commands_dog[:, 6:11] = self.gait
        self.env.arm_fake_actions[:] = torch.as_tensor((self.q_target - NOMINAL_Q) / self.cfg.control.action_scale,
                                                       device=b.device, dtype=torch.float)
        with torch.no_grad():
            action = self.policy(self.env.get_dog_observations())
            done = self.env.step(action, self.env.arm_fake_actions)[2].cpu().numpy().astype(bool)
        applied = b.joint_pos_target[:, 12:18].cpu().numpy().copy()
        self.last_target_error = float(np.max(np.abs(applied - self.q_target)))
        if self.last_target_error > 2e-6 and not done.any():
            raise RuntimeError(f"Arm interface target mismatch: {self.last_target_error}")
        return done, applied

    def state(self):
        b = self.base
        root = b.root_states.detach().cpu().numpy().copy()
        rpy = Rotation.from_quat(root[:, 3:7]).as_euler("ZYX")
        q = b.dof_pos[:, 12:18].cpu().numpy().copy()
        dq = b.dof_vel[:, 12:18].cpu().numpy().copy()
        response = np.column_stack([b.base_lin_vel[:, :2].cpu().numpy(),
            b.base_ang_vel[:, 2].cpu().numpy(), root[:, 2], rpy[:, 1]])
        return dict(q=q, dq=dq, response=response, root=root, ypr=rpy,
                    ee=b.end_effector_state[:, :7].cpu().numpy().copy(),
                    base_angular_velocity=b.base_ang_vel.cpu().numpy().copy())

    def close(self):
        self.base.close()
