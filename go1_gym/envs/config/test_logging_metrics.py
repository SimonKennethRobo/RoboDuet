"""Presentation schema tests; no simulator or remote W&B service required."""
import ast
from pathlib import Path
import pytest
from go1_gym.logging_metrics import episode_metric_name, robustness_metrics, ppo_metrics, configure_wandb


@pytest.mark.parametrize('key,prefix', [
    ('perf_vx_mae_mps','Performance/Dog/Tracking/'),
    ('perf_base_height_signed_error_m','Performance/Dog/Tracking/'),
    ('perf_foot_slip_speed_mps','Performance/Dog/Stability/'),
    ('perf_locomotion_power_w','Performance/Dog/Efficiency/'),
    ('perf_ee_position_rmse_m','Performance/Arm/Tracking/'),
    ('perf_waypoint_pos_offset_m','Performance/Arm/Tracking/'),
    ('perf_traj_success_rate','Performance/Arm/Trajectory/'),
    ('perf_early_termination_rate','Episode/Shared/'),
    ('perf_episode_duration_s','Episode/Shared/'),
    ('rew_jump','Reward/Dog/base_height_'),
    ('rew_arm_action_rate','Reward/Arm/'),
    ('rew_total','Episode/Shared/'),
    ('terrain_level','Curriculum/Terrain/'),
    ('traj_early_termination_rate','Performance/Arm/Termination/'),
    ('unrecognized_counter','Diagnostics/Episode/'),
])
def test_category(key,prefix):
    assert episode_metric_name(key).startswith(prefix)


def test_rollout_and_episode_statistics_are_distinct_and_raw_sums_not_uploaded():
    values = {'TrackingTime/all/vx_rmse_mps':.3,'TrackingTime/all/vx_sq_error_sum':9.,
              'TrackingAge/hard/early/vx_mae_mps':.4,
              'Termination/hard/failures_count':2,'Termination/hard/episodes_count':4,
              'Termination/easy/failures_count':0,'Termination/easy/episodes_count':0}
    result=robustness_metrics(values)
    assert result['Performance/Dog/Tracking/vx_rmse_mps_rollout']==.3
    assert result['Diagnostics/Dog/hard_early_vx_mae_mps']==.4
    assert result['Diagnostics/Termination/hard_failure_fraction_rollout']==.5
    assert not any('easy' in key or 'sum' in key for key in result)
    assert 'TrackingTime/all/vx_sq_error_sum' in values  # Local record untouched.
    assert episode_metric_name('perf_vx_mae_mps').endswith('_episode_mean')


def test_reset_metadata_and_cohort_metrics_can_be_logged_together():
    values = {
        'ResetMix/configured_hard_fraction': 0.2,
        'ResetMix/hard/initial_tilt_mean_rad': 0.4,
        'ResetMix/hard/initial_limit_mean_rad': 0.5,
        'ResetMix/hard/reset_count': 8,
        'Termination/hard/failures_count': 2,
        'Termination/hard/episodes_count': 8,
    }
    original = values.copy()
    assert robustness_metrics(values) == {
        'Curriculum/Reset/configured_hard_fraction': 0.2,
        'Curriculum/Reset/hard_initial_tilt_mean_rad': 0.4,
        'Curriculum/Reset/hard_initial_limit_mean_rad': 0.5,
        'Diagnostics/Termination/hard_failure_fraction_rollout': 0.25,
    }
    assert values == original


@pytest.mark.parametrize('key', [
    'ResetMix/future_metadata', 'ResetMix/hard/nested/metric',
    'Termination/future_metadata', 'Termination/hard/nested/failures_count',
    'TrackingTime/vx_mae_mps', 'TrackingAge/hard/vx_mae_mps',
    'TrackingAge/hard/early/nested/vx_mae_mps',
])
def test_unknown_robustness_key_layouts_do_not_break_logging(key):
    assert robustness_metrics({key: 1.0}) == {}


def test_raibert_sigma_is_a_reward_parameter_not_a_reward_term():
    from go1_gym.envs.config import build_roboduet_config
    cfg = build_roboduet_config()
    assert cfg.rewards.raibert_sigma == 0.35
    assert not hasattr(cfg.reward_scales, 'raibert_sigma')


def test_frozen_or_disabled_modules_do_not_log_fake_losses():
    kwargs=dict(action_std=.2,value_loss=0.,surrogate_loss=0.,adaptation_loss=0.)
    assert ppo_metrics('Dog',trainable=False,**kwargs)=={'PPO/Dog/trainable':0,'PPO/Dog/action_std_mean':.2}
    result=ppo_metrics('Dog',trainable=True,**kwargs)
    assert 'PPO/Dog/value_loss' in result
    assert 'PPO/Dog/adaptation_loss' not in result
    assert 'PPO/Dog/adaptation_loss' in ppo_metrics('Dog',trainable=True,adaptation_enabled=True,**kwargs)


def test_schema_config_and_axis():
    class Config(dict):
        def update(self,values,**kwargs): super().update(values)
    class Run:
        config=Config()
        definitions=[]
        def define_metric(self,*args,**kwargs): self.definitions.append((args,kwargs))
    run=Run()
    configure_wandb(run)
    assert run.config['logging_schema_version']==2
    assert (('*',),{'step_metric':'Runtime/iteration'}) in run.definitions
    configure_wandb(None)


def test_both_runners_use_common_router_and_have_no_legacy_names():
    root=Path(__file__).resolve().parents[3]
    for runner in ('ppo_cse_automatic','ppo_cse_unified'):
        source=(root/'go1_gym_learn'/runner/'__init__.py').read_text()
        ast.parse(source)
        assert 'episode_metric_name(key)' in source
        assert 'configure_wandb(wandb.run)' in source
        assert 'Train_Loss/' not in source
        assert 'Train_Reward_episode/' not in source
