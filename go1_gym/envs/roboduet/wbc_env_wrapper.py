import isaacgym

assert isaacgym
import sys

import gym
import torch
from isaacgym import gymapi

from go1_gym.envs.roboduet.wbc_env import WBCEnv, dog_cmd_idx
from go1_gym.envs.roboduet.wbc_env_config import Cfg
from go1_gym.utils.global_switch import global_switch


class EvaluationWrapper(WBCEnv):
    def __init__(
        self,
        sim_device,
        headless,
        num_envs=None,
        prone=False,
        deploy=False,
        cfg: Cfg = None,
        eval_cfg: Cfg = None,
        initial_dynamics_dict=None,
        physics_engine="SIM_PHYSX",
    ):

        super().__init__(
            sim_device,
            headless,
            num_envs=num_envs,
            prone=prone,
            deploy=deploy,
            cfg=cfg,
            eval_cfg=eval_cfg,
            initial_dynamics_dict=initial_dynamics_dict,
            physics_engine=physics_engine,
        )

    def update_arm_commands(self, target_lpy, target_rpy):
        self.commands_arm_obs[:, :3] = target_lpy
        self._set_arm_orientation_obs(target_rpy[:, 0], target_rpy[:, 1], target_rpy[:, 2])


class KeyboardWrapper(WBCEnv):
    def __init__(self, sim_device, headless, cfg):
        super().__init__(sim_device, headless, cfg=cfg)

        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_NUMPAD_8, "move forward")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_NUMPAD_5, "move backward")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_NUMPAD_4, "move left")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_NUMPAD_6, "move right")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_NUMPAD_7, "turn left")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_NUMPAD_9, "turn right")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_U, "arm up")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_O, "arm down")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_I, "arm forward")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_K, "arm backward")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_J, "arm left")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_L, "arm right")

        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_W, "arm pitch down")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_S, "arm pitch up")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_A, "arm roll left")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_D, "arm roll right")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Q, "arm yaw left")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_E, "arm yaw right")

        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "reset")

    def render_gui(self, sync_frame_time=True):
        if self.viewer:
            if self.fixed_cam:  # fixed camera to tracking the robot
                cam_target = gymapi.Vec3(self.root_states[0, 0], self.root_states[0, 1], self.root_states[0, 2])
                cam_pos = cam_target + gymapi.Vec3(1, 1, 1)
                self.gym.viewer_camera_look_at(self.viewer, self.envs[0], cam_pos, cam_target)

            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync
                elif evt.action == "fixed_cam" and evt.value > 0:
                    self.fixed_cam = not self.fixed_cam

                # for demo
                elif evt.action == "save_image" and evt.value > 0:
                    self.gym.step_graphics(self.sim)
                    self.gym.render_all_camera_sensors(self.sim)
                    cam_target = gymapi.Vec3(self.root_states[0, 0], self.root_states[0, 1], self.root_states[0, 2])
                    cam_pos = cam_target + gymapi.Vec3(0.8, 0.8, 0.8)
                    self.gym.set_camera_location(self.rendering_camera, self.envs[0], cam_pos, cam_target)
                    video_frame = self.gym.get_camera_image(
                        self.sim, self.envs[0], self.rendering_camera, gymapi.IMAGE_COLOR
                    )
                    video_frame = video_frame.reshape((self.camera_props.height, self.camera_props.width, 4))
                    import matplotlib.pyplot as plt

                    # Save the image as now.png
                    plt.imsave("now.png", video_frame)

                elif evt.action == "move forward" and evt.value > 0:
                    self.commands_dog[0, dog_cmd_idx["x_vel"]] += 0.1
                elif evt.action == "move backward" and evt.value > 0:
                    self.commands_dog[0, dog_cmd_idx["x_vel"]] -= 0.1
                elif evt.action == "move left" and evt.value > 0:
                    self.commands_dog[0, dog_cmd_idx["y_vel"]] += 0.1
                    self.commands_dog[0, dog_cmd_idx["y_vel"]] = torch.clip(
                        self.commands_dog[0, dog_cmd_idx["y_vel"]], -0.5, 0.5
                    )
                elif evt.action == "move right" and evt.value > 0:
                    self.commands_dog[0, dog_cmd_idx["y_vel"]] -= 0.1
                    self.commands_dog[0, dog_cmd_idx["y_vel"]] = torch.clip(
                        self.commands_dog[0, dog_cmd_idx["y_vel"]], -0.5, 0.5
                    )
                elif evt.action == "turn left" and evt.value > 0:
                    self.commands_dog[0, dog_cmd_idx["yaw_vel"]] += 0.1
                elif evt.action == "turn right" and evt.value > 0:
                    self.commands_dog[0, dog_cmd_idx["yaw_vel"]] -= 0.1
                elif evt.action == "arm up" and evt.value > 0:
                    self.commands_arm[0, 1] += 0.1
                elif evt.action == "arm down" and evt.value > 0:
                    self.commands_arm[0, 1] -= 0.1
                elif evt.action == "arm forward" and evt.value > 0:
                    self.commands_arm[0, 0] += 0.05
                    self.commands_arm[0, 0] = torch.clip(self.commands_arm[0, 0], 0.2, 0.8)
                elif evt.action == "arm backward" and evt.value > 0:
                    self.commands_arm[0, 0] -= 0.05
                    self.commands_arm[0, 0] = torch.clip(self.commands_arm[0, 0], 0.2, 0.8)
                elif evt.action == "arm left" and evt.value > 0:
                    self.commands_arm[0, 2] += 0.1
                elif evt.action == "arm right" and evt.value > 0:
                    self.commands_arm[0, 2] -= 0.1
                elif evt.action == "arm pitch down" and evt.value > 0:
                    self.commands_arm[0, 4] += 0.1
                elif evt.action == "arm pitch up" and evt.value > 0:
                    self.commands_arm[0, 4] -= 0.1
                elif evt.action == "arm roll left" and evt.value > 0:
                    self.commands_arm[0, 3] += 0.1
                elif evt.action == "arm roll right" and evt.value > 0:
                    self.commands_arm[0, 3] -= 0.1
                elif evt.action == "arm yaw left" and evt.value > 0:
                    self.commands_arm[0, 5] += 0.1
                elif evt.action == "arm yaw right" and evt.value > 0:
                    self.commands_arm[0, 5] -= 0.1

                elif evt.action == "reset" and evt.value > 0:
                    self.reset()
                    self.commands_dog[0, dog_cmd_idx["velocity"]] = 0

                elif (
                    evt.action
                    in [
                        "move forward",
                        "move backward",
                        "turn left",
                        "move left",
                        "move right",
                        "turn right",
                        "arm up",
                        "arm down",
                        "arm forward",
                        "arm backward",
                        "arm left",
                        "arm right",
                        "arm pitch down",
                        "arm pitch up",
                        "arm roll left",
                        "arm roll right",
                        "arm yaw left",
                        "arm yaw right",
                    ]
                    and evt.value == 0
                ):
                    print(
                        f"x_vel: {self.commands_dog[0, dog_cmd_idx['x_vel']]:.2f}, "
                        f"y_vel: {self.commands_dog[0, dog_cmd_idx['y_vel']]:.2f}, "
                        f"yaw_vel: {self.commands_dog[0, dog_cmd_idx['yaw_vel']]:.2f}, "
                        f"l: {self.commands_arm[0, 0]:.2f}, "
                        f"p: {self.commands_arm[0, 1]:.2f}, "
                        f"yaw: {self.commands_arm[0, 2]:.2f}, "
                        f"roll: {self.commands_arm[0, 3]:.2f}, "
                        f"pitch: {self.commands_arm[0, 4]:.2f}, "
                        f"yaw: {self.commands_arm[0, 5]:.2f}"
                    )

        # fetch results
        if self.device != "cpu":
            self.gym.fetch_results(self.sim, True)

        # step graphics
        if self.enable_viewer_sync:
            self.gym.step_graphics(self.sim)
            self._draw_viewer_overlays()
            self.gym.draw_viewer(self.viewer, self.sim, True)
            if sync_frame_time:
                self.gym.sync_frame_time(self.sim)
        else:
            self._draw_viewer_overlays()
            self.gym.poll_viewer_events(self.viewer)

        self.update_arm_commands()

    def update_arm_commands(self):
        self.sync_arm_commands_to_obs(env_ids=slice(0, 1))


class HistoryWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env: WBCEnv = env
        cfg: Cfg = self.env.cfg
        self.obs_history_length = self.env.cfg.env.num_observation_history

        self.num_obs_history = self.obs_history_length * self.num_obs
        self.obs_history = torch.zeros(
            self.env.num_envs, self.num_obs_history, dtype=torch.float, device=self.env.device, requires_grad=False
        )

        self.dog_obs_history = torch.zeros(
            self.env.num_envs,
            cfg.dog.dog_num_obs_history,
            dtype=torch.float,
            device=self.env.device,
            requires_grad=False,
        )

        self.arm_obs_history = torch.zeros(
            self.env.num_envs,
            cfg.arm.arm_num_obs_history,
            dtype=torch.float,
            device=self.env.device,
            requires_grad=False,
        )

        self.arm_fake_actions = torch.zeros(
            self.env.num_envs, self.env.num_actions_arm, dtype=torch.float, device=self.env.device, requires_grad=False
        )

    def plan(self, obs):
        return self.env.plan(obs)

    def step(self, action_dog, action_arm):

        if not global_switch.switch_open:
            action_arm = self.arm_fake_actions

        action = torch.concat([action_dog, action_arm], dim=-1)

        rew_dog, rew_arm, done, info = self.env.step(action)

        return rew_dog, rew_arm, done, info

    def get_observations(self):
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        self.obs_history = torch.cat((self.obs_history[:, self.env.num_obs :], obs), dim=-1)
        return {"obs": obs, "privileged_obs": privileged_obs, "obs_history": self.obs_history}

    def get_dog_observations(self):
        obs, privileged_obs = self.env.get_dog_observations()
        self.dog_obs_history = torch.cat(
            (self.dog_obs_history[:, self.env.cfg.dog.dog_num_observations :], obs), dim=-1
        )
        return {"obs": obs, "privileged_obs": privileged_obs, "obs_history": self.dog_obs_history}

    def get_dog_observations_hand(self, pose_in_ee):
        obs, privileged_obs = self.env.get_dog_observations()
        obs[:, 44:50] = pose_in_ee
        self.dog_obs_history = torch.cat(
            (self.dog_obs_history[:, self.env.cfg.dog.dog_num_observations :], obs), dim=-1
        )
        return {"obs": obs, "privileged_obs": privileged_obs, "obs_history": self.dog_obs_history}

    def get_arm_observations_hand(self, pose_in_ee):
        obs, privileged_obs = self.env.get_arm_observations()
        obs[:, 12:18] = pose_in_ee
        self.arm_obs_history = torch.cat(
            (self.arm_obs_history[:, self.env.cfg.arm.arm_num_observations :], obs), dim=-1
        )
        return {"obs": obs, "privileged_obs": privileged_obs, "obs_history": self.arm_obs_history}

    def get_arm_observations(self):
        obs, privileged_obs = self.env.get_arm_observations()
        self.arm_obs_history = torch.cat(
            (self.arm_obs_history[:, self.env.cfg.arm.arm_num_observations :], obs), dim=-1
        )
        return {"obs": obs, "privileged_obs": privileged_obs, "obs_history": self.arm_obs_history}

    def reset_idx(self, env_ids):  # it might be a problem that this isn't getting called!!
        ret = super().reset_idx(env_ids)
        self.obs_history[env_ids, :] = 0
        self.arm_obs_history[env_ids, :] = 0
        self.dog_obs_history[env_ids, :] = 0
        return ret

    def clear_cached(self, env_ids):
        self.obs_history[env_ids, :] = 0
        self.arm_obs_history[env_ids, :] = 0
        self.dog_obs_history[env_ids, :] = 0

    def reset(self):
        ret = super().reset()
        self.obs_history[:, :] = 0
        self.arm_obs_history[:, :] = 0
        self.dog_obs_history[:, :] = 0
        return ret

    def __getattr__(self, name):
        return getattr(self.env, name)


class KeyboardStage1Wrapper(WBCEnv):
    """Keyboard wrapper for stage-1 (dog-only) play.

    Key layout
    ----------
    w / s  — x_vel  +/-
    a / d  — y_vel  +/-
    q / e  — yaw_vel +/-
    j / l  — body_roll +/-
    i / k  — body_pitch +/-
    t / g  — body_height +/-     (t increases, g decreases)
    [ / ]  — gait_frequency +/-   (no-op when use_dynamic_gait=False)
    u / o  — stance_width +/-     (no-op when use_dynamic_gait=False)
    SPACE  — reset vel to zero
    """

    _VEL_STEP = 0.1
    _POSE_STEP = 0.05
    _HEIGHT_STEP = 0.05
    _GAIT_FREQ_STEP = 0.5
    _STANCE_STEP = 0.05

    def __init__(self, sim_device, headless, cfg):
        super().__init__(sim_device, headless, cfg=cfg)

        bindings = [
            (gymapi.KEY_W, "dog_vx_up"),
            (gymapi.KEY_S, "dog_vx_down"),
            (gymapi.KEY_A, "dog_vy_up"),
            (gymapi.KEY_D, "dog_vy_down"),
            (gymapi.KEY_Q, "dog_yaw_up"),
            (gymapi.KEY_E, "dog_yaw_down"),
            (gymapi.KEY_J, "dog_roll_up"),
            (gymapi.KEY_L, "dog_roll_down"),
            (gymapi.KEY_I, "dog_pitch_up"),
            (gymapi.KEY_K, "dog_pitch_down"),
            (gymapi.KEY_T, "dog_height_up"),
            (gymapi.KEY_G, "dog_height_down"),
            (gymapi.KEY_LEFT_BRACKET, "dog_freq_up"),
            (gymapi.KEY_RIGHT_BRACKET, "dog_freq_down"),
            (gymapi.KEY_U, "dog_sw_up"),
            (gymapi.KEY_O, "dog_sw_down"),
            (gymapi.KEY_SPACE, "dog_vel_zero"),
            (gymapi.KEY_M, "dog_reset"),
        ]
        for key, action in bindings:
            self.gym.subscribe_viewer_keyboard_event(self.viewer, key, action)

    def _n_cmd(self):
        return self.commands_dog.shape[1]

    def _add_dog(self, idx, delta, lo, hi):
        if idx >= self._n_cmd():
            return
        val = float(self.commands_dog[0, idx]) + delta
        self.commands_dog[:, idx] = max(lo, min(hi, val))

    def _print_state(self):
        print(self.format_dog_commands(), flush=True)

    def render_gui(self, sync_frame_time=True):
        if self.viewer:
            if self.fixed_cam:
                cam_target = gymapi.Vec3(self.root_states[0, 0], self.root_states[0, 1], self.root_states[0, 2])
                cam_pos = cam_target + gymapi.Vec3(1, 1, 1)
                self.gym.viewer_camera_look_at(self.viewer, self.envs[0], cam_pos, cam_target)

            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            _STAGE1_ACTIONS = {
                "dog_vx_up",
                "dog_vx_down",
                "dog_vy_up",
                "dog_vy_down",
                "dog_yaw_up",
                "dog_yaw_down",
                "dog_roll_up",
                "dog_roll_down",
                "dog_pitch_up",
                "dog_pitch_down",
                "dog_height_up",
                "dog_height_down",
                "dog_freq_up",
                "dog_freq_down",
                "dog_sw_up",
                "dog_sw_down",
                "dog_vel_zero",
                "dog_reset",
            }

            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync
                elif evt.action == "fixed_cam" and evt.value > 0:
                    self.fixed_cam = not self.fixed_cam

                elif evt.action not in _STAGE1_ACTIONS:
                    continue

                elif evt.value == 0:
                    self._print_state()
                    continue

                # key-down handlers
                elif evt.action == "dog_vx_up":
                    self._add_dog(dog_cmd_idx["x_vel"], self._VEL_STEP, -1.5, 1.5)
                elif evt.action == "dog_vx_down":
                    self._add_dog(dog_cmd_idx["x_vel"], -self._VEL_STEP, -1.5, 1.5)
                elif evt.action == "dog_vy_up":
                    self._add_dog(dog_cmd_idx["y_vel"], self._VEL_STEP, -0.5, 0.5)
                elif evt.action == "dog_vy_down":
                    self._add_dog(dog_cmd_idx["y_vel"], -self._VEL_STEP, -0.5, 0.5)
                elif evt.action == "dog_yaw_up":
                    self._add_dog(dog_cmd_idx["yaw_vel"], self._VEL_STEP, -1.5, 1.5)
                elif evt.action == "dog_yaw_down":
                    self._add_dog(dog_cmd_idx["yaw_vel"], -self._VEL_STEP, -1.5, 1.5)
                elif evt.action == "dog_roll_up":
                    self._add_dog(dog_cmd_idx["body_roll"], self._POSE_STEP, -0.4, 0.4)
                elif evt.action == "dog_roll_down":
                    self._add_dog(dog_cmd_idx["body_roll"], -self._POSE_STEP, -0.4, 0.4)
                elif evt.action == "dog_pitch_up":
                    self._add_dog(dog_cmd_idx["body_pitch"], self._POSE_STEP, -0.4, 0.4)
                elif evt.action == "dog_pitch_down":
                    self._add_dog(dog_cmd_idx["body_pitch"], -self._POSE_STEP, -0.4, 0.4)
                elif evt.action == "dog_height_up":
                    self._add_dog(dog_cmd_idx["body_height"], self._HEIGHT_STEP, -0.2, 0.2)
                elif evt.action == "dog_height_down":
                    self._add_dog(dog_cmd_idx["body_height"], -self._HEIGHT_STEP, -0.2, 0.2)
                elif evt.action == "dog_freq_up":
                    self._add_dog(dog_cmd_idx["gait_frequency"], self._GAIT_FREQ_STEP, 1.0, 4.0)
                elif evt.action == "dog_freq_down":
                    self._add_dog(dog_cmd_idx["gait_frequency"], -self._GAIT_FREQ_STEP, 1.0, 4.0)
                elif evt.action == "dog_sw_up":
                    self._add_dog(dog_cmd_idx["stance_width"], self._STANCE_STEP, 0.2, 0.5)
                elif evt.action == "dog_sw_down":
                    self._add_dog(dog_cmd_idx["stance_width"], -self._STANCE_STEP, 0.2, 0.5)
                elif evt.action == "dog_vel_zero":
                    self.commands_dog[:, dog_cmd_idx["velocity"]] = 0.0
                    self.commands_dog[:, dog_cmd_idx["body_pose"]] = 0.0
                elif evt.action == "dog_reset":
                    self.reset()
                    self.commands_dog[:, dog_cmd_idx["velocity"]] = 0.0
                    self.commands_dog[:, dog_cmd_idx["body_pose"]] = 0.0

        if self.device != "cpu":
            self.gym.fetch_results(self.sim, True)

        if self.enable_viewer_sync:
            self.gym.step_graphics(self.sim)
            self._draw_viewer_overlays()
            self.gym.draw_viewer(self.viewer, self.sim, True)
            if sync_frame_time:
                self.gym.sync_frame_time(self.sim)
        else:
            self._draw_viewer_overlays()
            self.gym.poll_viewer_events(self.viewer)

        # self.update_arm_commands()
