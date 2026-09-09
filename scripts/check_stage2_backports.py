"""IsaacGym smoke checks for the selected v3-stage2 backports.

Run in isaacgym with PYTHONPATH=.: python scripts/check_stage2_backports.py
Zero actions validate integration only, not trained-policy performance.
"""
import isaacgym  # noqa: F401
import torch
from argparse import Namespace

from benchmark.dog_policy.cli import _batched_cmd_fn
from benchmark.dog_policy.evaluation import (
    PolicyHandle, _apply_benchmark_env_overrides, _eval_loop_parallel,
    detect_command_layout, set_vel_cmd,
)
from go1_gym.envs.config import build_roboduet_config
from go1_gym.envs.roboduet.wbc_env import WBCEnv
from go1_gym.envs.roboduet.wbc_env_wrapper import HistoryWrapper
from scripts.check_response_runtime import _check_domain_schedule, zero_actions
from go1_gym.utils.global_switch import global_switch


def config():
    return build_roboduet_config(Namespace(
        robot='go2_x5', num_envs=16, dyna_gait=True, goal_reaching=False,
        traj_tracking=False, arm_action_mode=None, no_reach_table=False,
        dyna_gait_min_frequency=0., video=False,
    ))


def main():
    cfg = config()
    cfg.env.num_envs = 16
    cfg.env.record_video = False
    cfg.env.arm_policy_enabled = False
    # One asset bucket keeps this interface smoke inexpensive. R5's full
    # domain checks remain in check_response_runtime.py --check r5.
    cfg.domain_rand.mount_tf_buckets = 1
    env = HistoryWrapper(WBCEnv(sim_device='cuda:0', headless=True, cfg=cfg))
    b = env.env
    try:
        for name, scale in {'raibert_heuristic': -1., 'feet_contact_forces': -.01, 'feet_impact_vel': .4}.items():
            assert name in b.reward_names
            assert abs(b.pretrained_reward_scales[name] / b.dt - scale) < 1e-6
            assert abs(b.wbc_reward_scales[name] / b.dt - scale) < 1e-6
        print('PASS: Raibert/contact/impact terms registered with source weights in both policy stages')
        env.reset()
        b.gait_indices[:] = .6
        twin = b.is_nominal_twin.nonzero().flatten()[:1]
        b.reset_idx(twin)
        members = b.grouping.envs_of_groups(b.grouping.group_of[twin])
        torch.testing.assert_close(b.gait_indices[members], b.gait_indices[b.grouping.twin_of[members]])
        failures = _check_domain_schedule(env, cfg, b.response_curriculum, int(global_switch.count))
        assert not failures, failures
        ids = torch.arange(b.num_envs, device=b.device)
        b.next_push_step[:] = b.episode_length_buf
        before = b.root_states.clone()
        b.domain_disturbance_intensity = 0.
        b._push_robots(ids, cfg)
        torch.testing.assert_close(b.root_states, before)
        assert torch.all(b.next_push_step > b.episode_length_buf)
        b.next_push_step[:] = b.episode_length_buf
        b.domain_disturbance_intensity = 1.
        global_switch.count = cfg.domain_rand.push_curriculum_growth_iterations
        b._push_robots(ids, cfg)
        torch.testing.assert_close(b.root_states[b.is_nominal_twin], before[b.is_nominal_twin])
        assert torch.any(b.root_states[~b.is_nominal_twin, 7:9] != before[~b.is_nominal_twin, 7:9])
        env.step(*zero_actions(env, cfg))
        assert torch.isfinite(b.root_states).all()
        print('PASS: R8 domain/push schedule, actual indexed push write, nominal twin exclusion, twin reset phase')
    finally:
        b.gym.destroy_sim(b.sim)

    cfg = config()
    _apply_benchmark_env_overrides(cfg, 16, 4)
    cfg.domain_rand.mount_tf_buckets = 1
    cfg.env.record_video = False
    cfg.env.arm_policy_enabled = False
    env = HistoryWrapper(WBCEnv(sim_device='cuda:0', headless=True, cfg=cfg))
    b = env.env
    try:
        policy = lambda obs: torch.zeros(obs['obs'].shape[0], b.num_actions_loco, device=b.device)
        handles = [PolicyHandle('a', policy, 0, 8, 4), PolicyHandle('b', policy, 8, 16, 4)]
        points = [('slow', lambda e: set_vel_cmd(e, .2, 0, 0), {}),
                  ('fast', lambda e: set_vel_cmd(e, .8, 0, 0), {})]
        command, cells = _batched_cmd_fn(env, handles, points)
        expected = torch.tensor([.2] * 4 + [.8] * 4 + [.2] * 4 + [.8] * 4, device=b.device)
        def checked_command(e):
            command(e)
            torch.testing.assert_close(e.env.commands_dog[:, 0], expected)
        # Keep this short enough to avoid zero-policy falls/reset transients.
        accs = _eval_loop_parallel(env, handles, detect_command_layout(cfg), 8, 0., b.device,
                                   checked_command, metric_groups=cells)
        torch.testing.assert_close(b.commands_dog[:, 0], expected)
        assert len(accs) == 4
        assert all(torch.all(acc.steps == 8) for acc in accs)
        assert all(acc.rmse('response_consistency') >= 0 for acc in accs)
        assert not b.is_identification_env.any()
        assert not b.grouping.is_grouped.any()
        print('PASS: two policies x two command cells, R2 metrics, no training command overwrite')
    finally:
        b.gym.destroy_sim(b.sim)


if __name__ == '__main__':
    main()
