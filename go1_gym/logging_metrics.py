"""W&B schema v2. Presentation only: reward and episode aggregation stay unchanged."""
LOGGING_SCHEMA_VERSION = 2

# These are episode-weighted physical metrics, not rollout-pooled RMSEs.
_TRACKING = {'vx_mae_mps', 'vy_mae_mps', 'lin_vel_rmse_mps', 'yaw_rate_mae_rad_s',
             'yaw_rate_rmse_rad_s', 'base_height_rmse_m', 'base_height_signed_error_m'}
_STABILITY = {'roll_rms_rad', 'pitch_rms_rad', 'vertical_velocity_rms_mps',
              'horizontal_angular_velocity_rms_rad_s', 'foot_slip_speed_mps'}
_ARM_PREFIXES = ('ee_', 'arm_', 'goal_', 'traj_', 'manip_', 'vis_manip_')


def episode_metric_name(key):
    """Map environment episode keys; unknown diagnostics must never become rewards."""
    if key in ('global_switch_pretrained_start', 'global_switch_pretrained_end',
               'global_switch_stage1_ramp_iters'):
        return None  # Static schedule is already in run.config.
    if key == 'perf_episode_count':
        return None
    if key.startswith('perf_'):
        name = key[5:]
        if name == 'early_termination_rate':
            return 'Episode/Shared/early_termination_fraction'
        if name == 'episode_duration_s':
            return 'Episode/Shared/duration_mean_s'
        if name in _TRACKING:
            name = name.replace('base_height_signed_error_m', 'base_height_bias_m')
            return 'Performance/Dog/Tracking/' + name + '_episode_mean'
        if name in _STABILITY:
            return 'Performance/Dog/Stability/' + name + '_episode_mean'
        if name == 'locomotion_power_w':
            return 'Performance/Dog/Efficiency/mechanical_power_w_episode_mean'
        if name.startswith('ee_'):
            return 'Performance/Arm/Tracking/' + name + '_episode_mean'
        if name.startswith('traj_'):
            metric = name[5:]
            metric = {'dlat_rmse_m': 'lateral_error_rmse_m',
                      'dlat_mean_m': 'lateral_error_mean_m',
                      'timing_err_mean_m': 'timing_distance_error_mean_m',
                      'twist_err_mean_mps': 'twist_error_mean_mps',
                      'success_rate': 'success_fraction',
                      'early_term_rate': 'early_termination_fraction',
                      'rho_above_hi_rate': 'reach_ratio_above_limit_fraction',
                      'rho_mean': 'reach_ratio_mean', 'rho_max': 'reach_ratio_max',
                      'v_ff_xy_mean_mps': 'feedforward_speed_xy_mean_mps',
                      'v_base_xy_mean_mps': 'base_speed_xy_mean_mps',
                      }.get(metric, metric)
            return 'Performance/Arm/Trajectory/' + metric + '_episode_mean'
        if name.startswith(('arm_', 'goal_', 'waypoint_')):
            return 'Performance/Arm/Tracking/' + name + '_episode_mean'
        return 'Diagnostics/Performance/' + name
    if key.startswith('rew_'):
        name = key[4:]
        if name == 'total':
            return 'Episode/Shared/return_mean_reset_batches'
        if name.startswith('vis_'):
            return 'Diagnostics/Reward/' + name + '_return_mean_reset_batches'
        owner = 'Arm' if name.startswith(_ARM_PREFIXES) else 'Dog'
        name = {'jump': 'base_height'}.get(name, name)
        return f'Reward/{owner}/{name}_return_mean_reset_batches'
    exact = {
        'stage1_arm_curriculum_intensity': 'Curriculum/Arm/disturbance_intensity',
        'command_curriculum_weight': 'Curriculum/Command/bin_weight',
        'terrain_level': 'Curriculum/Terrain/row_index_mean',
    }
    if key in exact:
        return exact[key]
    for prefix, section in (
        ('curriculum_threshold_', 'Curriculum/Command/threshold_'),
        ('stage2_base_unlock_', 'Curriculum/BaseUnlock/'),
        ('reset_curriculum_', 'Curriculum/Reset/'),
        ('traj_curriculum_', 'Curriculum/Trajectory/'),
        ('global_switch_', 'Runtime/Stage/'),
        ('traj_early_termination', 'Performance/Arm/Termination/early_termination'),
    ):
        if key.startswith(prefix):
            return section + key[len(prefix):]
    return 'Diagnostics/Episode/' + key


def robustness_metrics(values):
    """Publish readable ratios; raw sufficient statistics stay in robustness.jsonl."""
    result = {}
    for key, value in values.items():
        parts = key.split('/')
        if key == 'ResetMix/configured_hard_fraction':
            result['Curriculum/Reset/configured_hard_fraction'] = value
            continue
        if key.startswith(('TrackingTime/', 'TrackingAge/')):
            expected_parts = 3 if parts[0] == 'TrackingTime' else 4
            if len(parts) != expected_parts:
                continue  # Unknown layouts remain available in the raw JSONL.
            if '_mae_' not in key and '_rmse_' not in key:
                continue
            if parts[0] == 'TrackingTime' and parts[1] == 'all':
                name = parts[-1].replace('yaw_', 'yaw_rate_')
                result['Performance/Dog/Tracking/' + name + '_rollout'] = value
            else:
                cohort = parts[1]
                age = 'all' if parts[0] == 'TrackingTime' else parts[2]
                result[f'Diagnostics/Dog/{cohort}_{age}_' + parts[-1]] = value
        elif key.startswith('Termination/'):
            if len(parts) != 3:
                continue
            _, group, metric = parts
            if metric != 'failures_count':
                continue
            total = values.get(f'Termination/{group}/episodes_count', 0)
            if total:
                result[f'Diagnostics/Termination/{group}_failure_fraction_rollout'] = value / total
        elif key.startswith('ResetMix/'):
            if len(parts) != 3:
                continue
            _, group, metric = parts
            if metric in ('initial_tilt_mean_rad', 'initial_limit_mean_rad'):
                result[f'Curriculum/Reset/{group}_{metric}'] = value
    return result


def ppo_metrics(owner, *, trainable, action_std, value_loss, surrogate_loss,
                adaptation_enabled=False, adaptation_loss=None):
    prefix = f'PPO/{owner}/'
    result = {prefix + 'trainable': int(trainable), prefix + 'action_std_mean': action_std}
    if trainable:
        result[prefix + 'value_loss'] = value_loss
        result[prefix + 'surrogate_loss'] = surrogate_loss
        if adaptation_enabled and adaptation_loss is not None:
            result[prefix + 'adaptation_loss'] = adaptation_loss
    return result


def configure_wandb(run):
    if run is None:
        return
    run.config.update({'logging_schema_version': LOGGING_SCHEMA_VERSION}, allow_val_change=True)
    run.define_metric('Runtime/iteration')
    run.define_metric('*', step_metric='Runtime/iteration')
    run.define_metric('Diagnostics/*', hidden=True)
