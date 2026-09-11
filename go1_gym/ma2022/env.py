"""Bare-quadruped IsaacGym task; no arm asset, policy or MPC at training time."""

import os
from copy import deepcopy
import tempfile
import xml.etree.ElementTree as ET

from isaacgym import gymapi, gymtorch, gymutil
from isaacgym.torch_utils import quat_rotate_inverse
import torch

from go1_gym.envs.config import cfg_to_dict
from go1_gym import MINI_GYM_ROOT_DIR
from go1_gym.envs.roboduet.legged_robot import LeggedRobot
from .wrench import WrenchSequence, world_to_body
from .rewards import reward_terms, swing_lift


class MaLocomotionEnv(LeggedRobot):
    policy_action_dim = 16

    def __init__(self, cfg, recipe, sim_device="cuda:0", headless=True):
        if not 0 < recipe.reward_curriculum_initial <= 1:
            raise ValueError("reward_curriculum_initial must be in (0,1]")
        if not 0 < recipe.reward_curriculum_exponent <= 1:
            raise ValueError("reward_curriculum_exponent must be in (0,1]")
        self.recipe = recipe
        # Save the task recipe before simulator-derived values are populated.
        self.source_config = cfg_to_dict(cfg)
        cfg = deepcopy(cfg)
        self._prepare_asset(cfg)
        params = gymapi.SimParams()
        gymutil.parse_sim_config(vars(cfg.sim), params)
        params.physx.use_gpu = sim_device.startswith("cuda")
        params.use_gpu_pipeline = sim_device.startswith("cuda")
        super().__init__(cfg, params, gymapi.SIM_PHYSX, sim_device, headless)
        if abs(self.dt - 0.02) > 1e-8:
            raise ValueError("Ma2022's wrench update and action contract require a 20-ms policy step")
        if self.num_dof != 12 or len(self.feet_indices) != 4:
            raise ValueError("Ma training requires a bare quadruped with 12 DOFs and 4 feet")
        self.wrench = WrenchSequence(self.num_envs, recipe, self.device)
        self.force_buffer = torch.zeros(self.num_envs, self.num_bodies, 3, device=self.device)
        self.torque_buffer = torch.zeros_like(self.force_buffer)
        self.previous_twist = self.root_states[:, 7:13].clone()
        self.applied_wrench = torch.zeros(self.num_envs, 6, device=self.device)
        self.phase = torch.zeros(self.num_envs, 4, device=self.device)
        self.policy_actions = torch.zeros(self.num_envs, 16, device=self.device)
        self.previous_policy_actions = torch.zeros_like(self.policy_actions)
        self.base_body = self.body_names.index("base" if "base" in self.body_names else "trunk")
        self.base_com = torch.tensor([
            [p[self.base_body].com.x, p[self.base_body].com.y, p[self.base_body].com.z]
            for p in (self.gym.get_actor_rigid_body_properties(e, a)
                      for e, a in zip(self.envs, self.actor_handles))
        ], device=self.device)
        # Resolve joint ordering by name rather than assume URDF ordering.
        self.leg_joint_indices = torch.tensor([
            [self.dof_names.index(f"{leg}_{joint}_joint") for joint in ("hip", "thigh", "calf")]
            for leg in ("FL", "FR", "RL", "RR")
        ], device=self.device)
        self.link_length = 0.213
        self.leg_collision_indices = torch.tensor([
            i for i, name in enumerate(self.body_names) if "thigh" in name or "calf" in name
        ], device=self.device)
        # Reference [10] S5: 52 samples around each foot.
        offsets = []
        for count, radius in zip((6, 8, 10, 12, 16), (.08, .16, .26, .36, .48)):
            angle = torch.arange(count, device=self.device)*2*torch.pi/count
            offsets.append(torch.stack((radius*angle.cos(), radius*angle.sin()), -1))
        self.foot_scan_offsets = torch.cat(offsets)
        self.reward_curriculum = torch.full((self.num_envs,), recipe.reward_curriculum_initial, device=self.device)
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.observation_dims = {k: v.shape[-1] for k, v in self.observations().items()}

    def reset_idx(self, ids):
        super().reset_idx(ids)
        # Both explicit reset and automatic reset must start finite differences
        # in the new episode; never compare a target against the old episode.
        self.last_dof_vel[ids] = self.dof_vel[ids]
        self.joint_pos_target[ids] = self.dof_pos[ids]
        self.last_joint_pos_target[ids] = self.dof_pos[ids]
        self.last_last_joint_pos_target[ids] = self.dof_pos[ids]

    def _randomize_dof_props(self, env_ids, cfg):
        # No inherited reset-time or periodic actuator randomization.
        # Friction is the only enabled dynamics DR in this recipe.
        for name, nominal in (("motor_strengths", 1.), ("motor_offsets", 0.),
                              ("Kp_factors", 1.), ("Kd_factors", 1.)):
            getattr(self, name)[env_ids] = nominal

    def _prepare_reward_function(self):
        # Keep parent reset/log bookkeeping buffers, but never register its
        # reward container or consult global_switch for the learning objective.
        self.reward_names = list(vars(self.cfg.reward_scales))
        self.pretrained_reward_scales = dict(vars(self.cfg.reward_scales))
        self.wbc_reward_scales = dict(self.pretrained_reward_scales)
        def buffers(names, value=0.):
            return {k: torch.full((self.num_envs,), value, device=self.device) for k in names}
        names = self.reward_names + ["total"]
        self.episode_sums = buffers(names)
        self.episode_sums_eval = buffers(names, -1.)
        self.command_sums = buffers(self.reward_names + ["lin_vel_raw", "ang_vel_raw",
                                   "lin_vel_residual", "ang_vel_residual", "ep_timesteps"])

    def _foot_heights(self):
        points = self.foot_positions[:, :, None, :2] + self.foot_scan_offsets[None, None]
        if self.cfg.terrain.mesh_type == "plane":
            heights = torch.zeros(points.shape[:-1], device=self.device)
        else:
            grid = ((points + self.terrain.cfg.border_size) / self.terrain.cfg.horizontal_scale).long()
            x = grid[..., 0].clamp(0, self.height_samples.shape[0]-2)
            y = grid[..., 1].clamp(0, self.height_samples.shape[1]-2)
            heights = torch.minimum(torch.minimum(self.height_samples[x, y], self.height_samples[x+1, y]),
                                    self.height_samples[x, y+1])*self.terrain.cfg.vertical_scale
        return heights-self.foot_positions[:, :, None, 2]

    def compute_reward(self):
        feet_state = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices]
        terms = reward_terms(
            command=self.commands_dog, linear=self.base_lin_vel, angular=self.base_ang_vel,
            gravity=self.projected_gravity, dof_pos=self.dof_pos, dof_vel=self.dof_vel,
            previous_dof_vel=self.last_dof_vel, target=self.joint_pos_target,
            last_target=self.last_joint_pos_target, previous_target=self.last_last_joint_pos_target,
            torque=self.torques, feet_velocity=feet_state[:, :, 7:10],
            feet_contact=self.contact_forces[:, self.feet_indices].norm(dim=-1)>1.,
            leg_collision=self.contact_forces[:, self.leg_collision_indices].norm(dim=-1)>1.,
            foot_heights=self._foot_heights(), phase=self.phase, knee_indices=self.leg_joint_indices[:, 2],
            knee_limit=self.recipe.knee_limit, dt=self.dt, curriculum=self.reward_curriculum,
            stability_multiplier=self.recipe.stability_multiplier)
        if set(terms) != set(self.reward_names):
            raise ValueError("Ma reward recipe does not match the task reward kernel")
        self.rew_buf_dog.zero_()
        self.rew_buf_arm.zero_()
        for name, term in terms.items():
            weighted = self.pretrained_reward_scales[name]*term
            self.rew_buf_dog += weighted
            self.episode_sums[name] += weighted
            self.command_sums[name] += weighted
        self.episode_sums["total"] += self.rew_buf_dog

    def _prepare_asset(self, cfg):
        # The checked-in bare Go2 URDF uses ROS package:// mesh URIs. Resolve
        # those into an isolated generated asset, leaving source assets intact.
        source = cfg.asset.file.format(MINI_GYM_ROOT_DIR=MINI_GYM_ROOT_DIR)
        asset_root = os.path.dirname(source)
        if not os.path.isfile(source):
            raise FileNotFoundError(f"Bare robot asset is missing: {source}")
        tree = ET.parse(source)
        for mesh in tree.findall(".//mesh"):
            uri = mesh.get("filename", "")
            if uri.startswith("package://"):
                relative = uri[len("package://"):].split("/", 1)[1]
                mesh.set("filename", os.path.abspath(os.path.join(asset_root, "..", relative)))
            elif uri:
                mesh.set("filename", os.path.abspath(os.path.join(asset_root, uri)))
        self._asset_temp = tempfile.TemporaryDirectory(prefix="roboduet_ma2022_")
        path = os.path.join(self._asset_temp.name, "bare_robot.urdf")
        tree.write(path)
        cfg.asset.file = path

    def _generate_arm_mount_asset_files(self, asset_root, asset_file):
        self.arm_mount_bucket_tfs = torch.zeros(1, 6).numpy()
        return [asset_file]

    def _get_noise_scale_vec(self, cfg):
        return torch.zeros(cfg.env.num_observations, device=self.device)

    def _process_rigid_body_props(self, props, env_id):
        # The WBC base implementation adds camera mass at an end-effector
        # index. A bare quadruped has neither that link nor that camera.
        self.default_body_mass = props[0].mass
        if self.cfg.domain_rand.randomize_base_mass:
            props[0].mass += float(self.payloads[env_id])
        if self.cfg.domain_rand.randomize_com_displacement:
            delta = self.com_displacements[env_id]
            props[0].com = gymapi.Vec3(props[0].com.x + float(delta[0]),
                                       props[0].com.y + float(delta[1]),
                                       props[0].com.z + float(delta[2]))
        return props

    def compute_observations(self):
        # This task has a separate, versioned teacher/student input contract.
        pass

    def _render_headless(self):
        recorder = getattr(self, "video_recorder", None)
        if recorder is not None:
            recorder.capture(self)

    def get_observations(self):
        return self.observations()

    def get_privileged_observations(self):
        return self.observations()["privileged"]

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.observations()

    def _resample_commands(self, ids):
        if not len(ids):
            return
        self.commands_dog[ids] = 0
        self.commands_dog[ids, 6:] = torch.tensor(
            [self.recipe.gait_frequency, self.recipe.swing_height, 0.3, 0.4, 0.5], device=self.device)
        for col, name in enumerate(("lin_vel_x", "lin_vel_y", "ang_vel_yaw")):
            lo, hi = getattr(self.cfg.commands, name)
            self.commands_dog[ids, col] = lo + torch.rand(len(ids), device=self.device) * (hi - lo)
        self.steps_since_command_change[ids] = 0
        for values in self.command_sums.values():
            values[ids] = 0

    def _arm_post_reset_refresh_hook(self, ids):
        if not hasattr(self, "wrench"):
            return
        if hasattr(self, "reward_curriculum"):
            completed = ids[self.episode_length_buf[ids] > 0]
            self.reward_curriculum[completed] = self.reward_curriculum[completed].pow(
                self.recipe.reward_curriculum_exponent)
        self.wrench.reset(ids)
        self.previous_twist[ids] = self.root_states[ids, 7:13]
        self.applied_wrench[ids] = 0
        self.phase[ids] = torch.tensor([0., 0.5, 0.5, 0.], device=self.device)
        self.policy_actions[ids] = 0
        self.previous_policy_actions[ids] = 0

    def _arm_decimation_hook(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        velocity = self.root_states[:, 7:13]
        acceleration = (velocity - self.previous_twist) / self.sim_params.dt
        self.previous_twist.copy_(velocity)
        # The polynomial advances once per 20-ms policy step; evaluate within
        # it at each physics substep rather than holding an impulse for 5 ms.
        predictable = self.wrench.evaluate([self._wrench_substep * self.sim_params.dt])[:, 0]
        self.applied_wrench = predictable + self.wrench.disturbance(acceleration)
        self._wrench_substep += 1
        self.force_buffer.zero_()
        self.torque_buffer.zero_()
        # LOCAL_SPACE is the instantaneous body frame. Apply F at COM with
        # the equivalent moment of a wrench about the base link origin:
        # tau_COM = tau_origin - r_COM x F. This avoids a hidden moment shift.
        q = self.base_quat
        force = world_to_body(q, self.applied_wrench[:, :3])
        torque = world_to_body(q, self.applied_wrench[:, 3:])
        self.force_buffer[:, self.base_body] = force
        self.torque_buffer[:, self.base_body] = torque - torch.cross(self.base_com, force, dim=-1)
        applied = self.gym.apply_rigid_body_force_tensors(
            self.sim, gymtorch.unwrap_tensor(self.force_buffer),
            gymtorch.unwrap_tensor(self.torque_buffer), gymapi.LOCAL_SPACE)
        if not applied:
            raise RuntimeError("IsaacGym rejected the base wrench force tensors")

    def _arm_post_physics_hook(self):
        # Advance prediction to the next policy observation before any reset.
        self.wrench.advance(self.dt)

    def _arm_step_end_hook(self):
        # Parent end-of-step bookkeeping copies the terminating action after
        # reset_idx. Do not let smoothness rewards span two episodes.
        ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.actions[ids] = 0
        self.last_actions[ids] = 0
        self.last_last_actions[ids] = 0
        self.last_dof_vel[ids] = self.dof_vel[ids]
        self.joint_pos_target[ids] = self.dof_pos[ids]
        self.last_joint_pos_target[ids] = self.dof_pos[ids]
        self.last_last_joint_pos_target[ids] = self.dof_pos[ids]

    def check_termination(self):
        super().check_termination()
        contact = self.contact_forces[:, self.termination_contact_indices].norm(dim=-1)
        self.reset_buf |= (contact > 1.).any(dim=1)
        self.terminal_observation = self.observations()
        self.terminal_timeout = self.time_out_buf & ~(
            (contact > 1.).any(dim=1) | self.body_height_buf | self.roll_pitch_buf)

    def _joint_targets(self, action):
        # Reference [10] S5 phase increment and cubic foot lift, with a
        # two-link sagittal IK and lift amplitude adapted to Unitree.
        self.phase = (self.phase + self.dt*self.recipe.gait_frequency +
                      self.recipe.phase_increment_scale*action[:, :4]/(2*torch.pi)) % 1.
        nominal = self.default_dof_pos[:, self.leg_joint_indices]
        thigh, calf = nominal[..., 1], nominal[..., 2]
        length = self.link_length
        x = -length * (torch.sin(thigh) + torch.sin(thigh + calf))
        z = -length * (torch.cos(thigh) + torch.cos(thigh + calf))
        z = z + self.recipe.swing_height * swing_lift(self.phase)
        knee = -torch.acos(((x*x + z*z - 2*length*length) / (2*length*length)).clamp(-0.999, 0.999))
        hip = torch.atan2(-x.expand_as(z), -z) - torch.atan2(torch.sin(knee), 1 + torch.cos(knee))
        target = self.default_dof_pos.expand(self.num_envs, -1).clone()
        target[:, self.leg_joint_indices[:, 1]] = hip
        target[:, self.leg_joint_indices[:, 2]] = knee
        return target + self.recipe.residual_scale * action[:, 4:]

    def step(self, action):
        if action.shape != (self.num_envs, 16) or not torch.isfinite(action).all():
            raise ValueError("Expected finite (num_envs,16) phase/residual actions")
        self.previous_policy_actions.copy_(self.policy_actions)
        self.policy_actions.copy_(action.clamp(-3, 3))
        targets = self._joint_targets(self.policy_actions)
        self._wrench_substep = 0
        reward, _, done, info = super().step(targets - self.default_dof_pos)
        return self.observations(), reward, done, info

    def observations(self):
        # Compute base-local state from root tensor, including immediately
        # after reset (the parent's cached velocity/gravity still describes
        # the terminal state at that boundary).
        q = self.base_quat
        linear = quat_rotate_inverse(q, self.root_states[:, 7:10])
        angular = quat_rotate_inverse(q, self.root_states[:, 10:13])
        gravity = quat_rotate_inverse(q, self.gravity_vec)
        proprio = torch.cat((gravity, linear, angular, self.commands_dog[:, :3],
                             self.dof_pos - self.default_dof_pos, self.dof_vel * 0.05,
                             self.policy_actions, self.previous_policy_actions,
                             torch.sin(2*torch.pi*self.phase), torch.cos(2*torch.pi*self.phase)), dim=-1)
        heights = self._get_heights(torch.arange(self.num_envs, device=self.device), self.cfg)
        scan = (self.base_pos[:, 2:3] - heights - self.cfg.rewards.base_height_target).clamp(-1, 1)
        scan = scan * self.recipe.scan_scale
        contacts = self.contact_forces[:, self.feet_indices].flatten(1) * 0.01
        contacts = torch.where((self.episode_length_buf > 0)[:, None], contacts, torch.zeros_like(contacts))
        # Actuator factors are fixed at nominal in this recipe, so they carry
        # no information and are not privileged inputs or decoder targets.
        privileged = torch.cat((self.friction_coeffs[:, :1], contacts), dim=-1)
        clean_wrench = torch.cat((self.wrench.prediction(q), self.commands_dog[:, :3], linear, angular), dim=-1)
        student_proprio = proprio.clone()
        # Commands, own past actions and phase clocks are known internal
        # states. Noise belongs only on the measured gravity/twist/joints.
        student_proprio[:, :9] += torch.randn_like(proprio[:, :9]) * self.recipe.proprio_noise_std
        student_proprio[:, 12:36] += torch.randn_like(proprio[:, 12:36]) * self.recipe.proprio_noise_std
        # No privileged labels are passed through either student RNN input.
        student_wrench = torch.cat((self.wrench.prediction(q, noisy=True),
                                    student_proprio[:, 9:12], student_proprio[:, 3:9]), dim=-1)
        applied = torch.cat((world_to_body(q, self.applied_wrench[:, :3]),
                             world_to_body(q, self.applied_wrench[:, 3:])), dim=-1) * self.wrench.scale
        return dict(proprio=proprio, wrench=clean_wrench, scan=scan, privileged=privileged,
                    student_proprio=student_proprio, student_wrench=student_wrench,
                    student_scan=scan + torch.randn_like(scan) * self.recipe.scan_noise_std * self.recipe.scan_scale,
                    applied_wrench=applied, gain=self.wrench.gain_target())

    def close(self):
        if self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
        self._asset_temp.cleanup()
