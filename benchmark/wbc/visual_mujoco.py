"""Visual Go2-X5 playback policy, native persistent IK and shared omni PID.

The follower is a benchmark auxiliary controller. The native workspace
projection changes only controller inputs;
FrozenReference and the benchmark scorer retain the original world target.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

from benchmark.wbc.dwbc_mujoco import DwbcMujoco, JOINT_NAMES
from benchmark.wbc.omni_waypoint_follower import OmniFollowerConfig, configure_follower, follow_reference


class VisualMujoco(DwbcMujoco):
    def __init__(self, root, checkpoint, scene, *, base_mode="follow", action_delay=0):
        if base_mode not in ("follow", "stand") or action_delay not in (0, 1):
            raise ValueError("Visual requires base_mode follow/stand and action_delay 0/1")
        super().__init__(root, checkpoint, scene, variant="visual")
        self.config_path = Path(checkpoint).resolve().with_name("run_config.json")
        self.training_config = json.loads(self.config_path.read_text())["env_cfg"]
        cfg = self.training_config
        if (cfg["env"]["num_proprio"], cfg["env"]["num_priv"], cfg["env"]["history_len"]) != (71, 18, 10):
            raise ValueError("Unsupported Visual checkpoint observation contract")
        scales = cfg["normalization"]["obs_scales"]
        if [scales[k] for k in ("ang_vel", "lin_vel", "dof_pos", "dof_vel")] != [1., 1., 1., .05]:
            raise ValueError("Unsupported Visual checkpoint observation scales")
        self.default_dof_pos = np.asarray([cfg["init_state"]["default_joint_angles"][n] for n in JOINT_NAMES])
        self.scale = np.asarray(cfg["control"]["action_scale"])
        self.control_dt = float(cfg["sim"]["dt"])
        self.p["decimation"] = int(cfg["control"]["decimation"])
        self.policy_dt = self.control_dt * self.p["decimation"]
        self.p["rl_kp"] = [cfg["control"]["stiffness"]["joint"]] * 12 + cfg["control"]["arm_drive_stiffness"]
        self.p["rl_kd"] = [cfg["control"]["damping"]["joint"]] * 12 + cfg["control"]["arm_drive_damping"]
        # The native play entry starts global_steps at zero, independently of
        # checkpoint iteration. ManipLoco.step uses the newest action until
        # global_steps reaches 10000 * 24. Delay=1 is an explicit ablation.
        self.action_delay_steps = int(action_delay)
        self.base_mode = base_mode
        self.workspace_source = self.root / "low-level/legged_gym/utils/ee_base_follower.py"
        spec = importlib.util.spec_from_file_location("_benchmark_visual_workspace", self.workspace_source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self.project_arm_target = module.EEBaseFollower.project_arm_target
        # This native checkpoint sampled vy=0 during training. Keep the omni
        # interface, but bound its untrained lateral/backward inputs tightly.
        configure_follower(self, OmniFollowerConfig(
            max_speed_mps=.30, max_lateral_speed_mps=.02, max_backward_speed_mps=.03,
            max_acceleration_mps2=.4, max_yaw_rate_rps=.40, max_yaw_acceleration_rps2=.75))
        self.workspace = module.ArmWorkspace()
        self.gait_observation = np.zeros(5)
        self.target_projected = False
        self.reset_policy()

    def reset_policy(self):
        super().reset_policy()
        self.action_queue = [np.zeros(18) for _ in range(self.action_delay_steps)]
        self.follower.reset()
        self.gait_indices = 0.0
        self.gait_observation.fill(0.0)
        self.command = [0.0] * 6

    def set_reference(self, reference, time_s):
        _, position, quaternion = reference.at(time_s)
        self.reference_position = np.asarray(position).copy()
        self.goal_position = self.reference_position.copy()
        self.goal_quaternion = np.asarray(quaternion).copy()
        if self.base_mode == "follow":
            q, dq, quat, gyro, base_pos, lin_vel = self.read_state()
            follow_reference(self, reference, time_s)
            yaw = self._euler(quat)[2]
            center_cfg = self.training_config["goal_ee"]["sphere_center"]
            cx, cy = center_cfg["x_offset"], center_cfg["y_offset"]
            center = np.array([base_pos[0] + np.cos(yaw)*cx - np.sin(yaw)*cy,
                               base_pos[1] + np.sin(yaw)*cx + np.cos(yaw)*cy,
                               center_cfg["z_invariant_offset"]])
            projected, safe = self.project_arm_target(position, center, quat, self.workspace)
            self.goal_position = projected[0].numpy().copy()
            self.target_projected = not bool(safe[0])

    def forward(self, *state):
        action = super().forward(*state)
        # The current observation sees the clock produced by the previous
        # physics step; advance once per policy step, as in ManipLoco.
        cfg = self.training_config
        walking = (np.linalg.norm(self.command[:2]) > cfg["commands"]["lin_vel_x_clip"] or
                   abs(self.command[2]) > cfg["commands"]["ang_vel_yaw_clip"])
        self.gait_indices = ((self.gait_indices + self.policy_dt * cfg["env"]["frequencies"]) % 1.
                             if walking else 0.)
        self.gait_observation = np.r_[self.gait_indices,
            np.sin(2 * np.pi * (self.gait_indices + np.array([.5, 0., 0., .5])))]
        return action

    def _contacts(self):
        # Native foot sensors threshold the force norm at 1.5 N. A geometric
        # contact can have zero force while separating and is not sufficient.
        ids = {self.model.geom(name).id: i for i, name in enumerate(("FL", "FR", "RL", "RR"))}
        forces = np.zeros((4, 3))
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            wrench = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, wrench)
            world = contact.frame.reshape(3, 3).T @ wrench[:3]
            for geom, sign in ((int(contact.geom1), -1), (int(contact.geom2), 1)):
                if geom in ids:
                    forces[ids[geom]] += sign * world
        return (np.linalg.norm(forces, axis=1) > 1.5).astype(float)
