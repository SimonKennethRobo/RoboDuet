"""CPU tensor/serialization regression tests; no simulator is created.

Run in the isaacgym environment with python -m unittest scripts.verify_dog_observation_layout -v.
"""
import isaacgym  # noqa: F401 -- must precede torch
import io
import itertools
import pickle
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
import yaml

from go1_gym.envs.config import (
    RoboDuetRuntimeOptions, apply_config_snapshot, build_roboduet_config,
    cfg_to_dict, recompute_observation_dims, restore_dog_observation_layout,
)
from go1_gym.envs.roboduet.wbc_env import WBCEnv
from go1_gym.utils.global_switch import global_switch
from scripts import export_rl_sar
from scripts.load_policy import load_dog_policy, _validate_checkpoint_layout
from scripts.rl_sar_obs import RlSarObservation
from scripts.verify_rl_sar_obs import raw_state_from_env
from types import SimpleNamespace

SWITCHES = ('observe_clock_inputs', 'observe_lin_vel', 'observe_pose_actual', 'observe_track_error')


def config(flags=(True, True, True, True), dynamic=True, rot6d=True):
    cfg = build_roboduet_config(options=RoboDuetRuntimeOptions(1, 'go2', dyna_gait=dynamic, use_rot6d=rot6d))
    for key, flag in zip(SWITCHES, flags):
        setattr(cfg.dog, key, flag)
    recompute_observation_dims(cfg)
    return cfg


def tensor_env(cfg):
    env = WBCEnv.__new__(WBCEnv)
    env.cfg, env.num_envs, env.device = cfg, 1, 'cpu'
    env.num_actions_loco, env.num_actions_arm = 12, 6
    env.obs_scales = cfg.obs_scales
    env.projected_gravity = torch.tensor([[0., 0., -1.]])
    joints, _, _ = export_rl_sar.joint_order_lists(cfg, 'go2_x5')
    env.default_dof_pos = torch.tensor([export_rl_sar.default_dof_pos(cfg, joints)])
    env.dof_pos = env.default_dof_pos + torch.arange(18).reshape(1, 18) * .01
    env.dof_vel = torch.arange(18).reshape(1, 18) * .1
    env.actions = torch.arange(18).reshape(1, 18) * .02
    env.last_actions = env.actions + .1
    extra, scale = export_rl_sar.dog_command_layout(cfg)
    env.commands_dog = torch.tensor([[.4, -.2, .3, .1, -.1, .02] + extra])
    env.commands_scale_dog = torch.tensor(scale)
    env.commands_arm_obs = torch.arange(cfg.arm.arm_num_commands).reshape(1, -1) * .03
    env.base_quat = torch.tensor([[0., 0., 0., 1.]])
    env.base_lin_vel = torch.tensor([[.2, -.1, .04]])
    env.base_ang_vel = torch.tensor([[.03, .04, .2]])
    env.base_pos = torch.tensor([[0., 0., .35]])
    env.pitch, env.roll = torch.zeros(1), torch.zeros(1)
    env.root_states = torch.zeros(1, 13)
    env.root_states[:, 7:10] = env.base_lin_vel
    env.dt = cfg.sim.dt * cfg.control.decimation
    env.gait_indices = torch.tensor([.12])
    for key in ('clock_inputs', 'doubletime_clock_inputs', 'halftime_clock_inputs', 'desired_contact_states'):
        setattr(env, key, torch.zeros(1, 4))
    env._step_contact_targets()
    cfg.domain_rand.dog_obs_frame_drop_prob = 0.
    env._get_physics_privileged_observations = lambda policy: torch.zeros(1, cfg.dog.dog_num_privileged_obs)
    return env


class DogObservationTests(unittest.TestCase):
    def setUp(self):
        self.switch = global_switch.switch_flag
        global_switch.switch_flag = False

    def tearDown(self):
        global_switch.switch_flag = self.switch

    def test_all_switch_combinations_match_selected_full_observation(self):
        for dynamic, rot6d, stage2 in itertools.product((False, True), repeat=3):
            global_switch.switch_flag = stage2
            full_env = tensor_env(config(dynamic=dynamic, rot6d=rot6d))
            full, _ = full_env.get_dog_observations()
            segments, cursor = {}, 0
            for name, width, _, _ in full_env._dog_obs_layout():
                segments[name] = full[:, cursor:cursor + width]
                cursor += width
            for flags in itertools.product((False, True), repeat=4):
                with self.subTest(dynamic=dynamic, rot6d=rot6d, stage2=stage2, flags=flags):
                    cfg = config(flags, dynamic, rot6d)
                    env = tensor_env(cfg)
                    actual, _ = env.get_dog_observations()
                    layout = env._dog_obs_layout()
                    expected = torch.cat([segments[name] for name, _, _, _ in layout], dim=1)
                    torch.testing.assert_close(actual, expected)
                    self.assertEqual(actual.shape[1], cfg.dog.dog_num_observations)
                    self.assertEqual(sum(w for _, w, _, _ in layout), actual.shape[1])
                    self.assertEqual(export_rl_sar.expected_obs_width(cfg, export_rl_sar.observation_terms(cfg)), actual.shape[1])
                    self.assertEqual(cfg.dog.dog_num_obs_history, 30 * actual.shape[1])

    def test_noise_and_frame_drop_offsets_follow_compact_layout(self):
        cfg = config((False, False, True, True))
        env = tensor_env(cfg)
        cfg.noise.add_noise = cfg.dog.add_obs_noise = True
        layout = env._dog_obs_layout()
        env.dog_obs_noise_scale_vec = torch.cat([torch.ones(w) * scale for _, w, scale, _ in layout])
        cursor, env.dog_obs_droppable_segments = 0, []
        for _, width, _, droppable in layout:
            if droppable:
                env.dog_obs_droppable_segments.append((cursor, cursor + width))
            cursor += width
        first, _ = env.get_dog_observations()
        cfg.domain_rand.dog_obs_frame_drop_prob = 1.
        env.dof_pos += .4
        second, _ = env.get_dog_observations()
        for start, end in env.dog_obs_droppable_segments:
            torch.testing.assert_close(first[:, start:end], second[:, start:end])
        self.assertTrue(torch.isfinite(second).all())

    def test_legacy_snapshot_keeps_zero_slots(self):
        cfg = config((False, False, False, False))
        cfg.dog.observation_layout_version = 1
        recompute_observation_dims(cfg)
        snapshot = cfg_to_dict(cfg)
        del snapshot['dog']['observation_layout_version']
        del snapshot['dog']['observe_clock_inputs']
        snapshot['env']['observe_clock_inputs'] = False
        restored = config()
        apply_config_snapshot(restored, snapshot)
        restore_dog_observation_layout(restored, snapshot)
        self.assertEqual(restored.dog.dog_num_observations, 86)
        env = tensor_env(restored)
        obs, _ = env.get_dog_observations()
        cursor = 0
        for name, width, _, _ in env._dog_obs_layout():
            if name in ('base_lin_vel', 'body_pose_actual', 'body_pose_error', 'velocity_error'):
                self.assertEqual(torch.count_nonzero(obs[:, cursor:cursor + width]).item(), 0)
            cursor += width

    def test_load_export_and_replay(self):
        for flags, legacy in [((False, False, False, False), False),
                              ((True, False, False, True), False),
                              ((True, False, False, False), True)]:
            with self.subTest(flags=flags, legacy=legacy), tempfile.TemporaryDirectory() as temp:
                cfg = config(flags)
                cfg.dog.observation_layout_version = 1 if legacy else 2
                recompute_observation_dims(cfg)
                snapshot = cfg_to_dict(cfg)
                if legacy:
                    del snapshot['dog']['observation_layout_version']
                    del snapshot['dog']['observe_clock_inputs']
                root = Path(temp)
                (root / 'checkpoints_dog').mkdir()
                with open(root / 'parameters.pkl', 'wb') as handle:
                    pickle.dump({'Cfg': snapshot}, handle)
                module = export_rl_sar._load_dog_ac_module()
                module.DogAC_Args.actor_hidden_dims = [16, 8]
                module.DogAC_Args.critic_hidden_dims = [16, 8]
                with redirect_stdout(io.StringIO()):
                    model = module.DogActorCritic(cfg.dog.dog_num_observations, cfg.dog.dog_num_privileged_obs,
                                                 cfg.dog.dog_num_obs_history, 12, use_adaptation_module=False)
                torch.save(model.state_dict(), root / 'checkpoints_dog/ac_weights_last_dog.pt')
                rebuilt, _ = export_rl_sar.load_runtime_cfg(root, 'go2')
                with redirect_stdout(io.StringIO()):
                    policy = load_dog_policy(root, 'last', rebuilt)
                    output = export_rl_sar.export(root, robot='go2_x5', quiet=True)
                params = next(iter(yaml.safe_load((output / 'config.yaml').read_text()).values()))
                env = tensor_env(rebuilt)
                actual, _ = env.get_dog_observations()
                state = raw_state_from_env(SimpleNamespace(env=env), env.commands_dog[0, :6].tolist())
                replay = RlSarObservation(params).assemble(state)
                np.testing.assert_allclose(replay, actual[0].numpy(), atol=1e-6)
                history = torch.randn(1, rebuilt.dog.dog_num_obs_history)
                with torch.no_grad():
                    expected = model.actor_body(history)
                    torch.testing.assert_close(policy({'obs_history': history}), expected)
                    torch.testing.assert_close(torch.jit.load(str(output / 'policy.pt'))(history), expected)
                wrong = config((False, False, False, True))
                with self.assertRaisesRegex(ValueError, 'incompatible'):
                    _validate_checkpoint_layout(model.state_dict(), 'dog', wrong)

    def test_stage2_restores_saved_dog_switches(self):
        from scripts.auto_train import apply_dog_checkpoint_command_limits
        trained = config((False, True, False, True))
        trained.dog.dog_num_observation_history = 12
        recompute_observation_dims(trained)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with open(root / 'parameters.pkl', 'wb') as handle:
                pickle.dump({'Cfg': cfg_to_dict(trained)}, handle)
            runtime = config((True, False, True, False))
            with redirect_stdout(io.StringIO()):
                apply_dog_checkpoint_command_limits(runtime, str(root / 'checkpoints_dog/weights.pt'))
            self.assertEqual(runtime.dog.dog_num_observations, trained.dog.dog_num_observations)
            self.assertEqual(runtime.dog.dog_num_obs_history, trained.dog.dog_num_obs_history)
            for key in SWITCHES:
                self.assertEqual(getattr(runtime.dog, key), getattr(trained.dog, key))

    def test_corrupt_recorded_width_is_rejected(self):
        cfg = config((False,) * 4)
        snapshot = cfg_to_dict(cfg)
        snapshot['dog']['dog_num_observations'] += 3
        with self.assertRaisesRegex(ValueError, 'layout mismatch'):
            restore_dog_observation_layout(cfg, snapshot)


if __name__ == '__main__':
    unittest.main()
