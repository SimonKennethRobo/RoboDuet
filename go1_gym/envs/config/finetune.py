"""Explicit weight-only Stage-1 fine-tuning with a validated policy contract."""
import hashlib
import pickle
from pathlib import Path
import re


def validate_finetune(cfg, checkpoint):
    from .core import cfg_to_dict
    checkpoint = Path(checkpoint).resolve()
    match = re.fullmatch(r'ac_weights_(\d+)\.pt', checkpoint.name)
    if not checkpoint.is_file() or not match:
        raise ValueError('Fine-tuning requires a numbered ac_weights_<iteration>.pt checkpoint')
    params = checkpoint.parent.parent / 'parameters.pkl'
    with params.open('rb') as stream:
        saved = pickle.load(stream)
    source = saved['Cfg']
    target = cfg_to_dict(cfg)
    if source['rewards'].get('attitude_command_convention', 'legacy') != 'rpy':
        raise ValueError('Fine-tuning requires corrected RPY training; legacy policy is incompatible')
    if target['rewards']['attitude_command_convention'] != 'rpy':
        raise ValueError('Fine-tuning target must preserve RPY semantics')
    mismatches = []
    for group in ('dog', 'control', 'obs_scales', 'normalization'):
        for key, value in source[group].items():
            if target[group].get(key) != value:
                mismatches.append(f'{group}.{key}')
    for group, key in [('terrain', 'height_reference'), ('rewards', 'base_height_target'),
                       ('asset', 'file'), ('init_state', 'default_joint_angles'),
                       ('commands', 'use_dynamic_gait')]:
        if target[group].get(key) != source[group].get(key):
            mismatches.append(f'{group}.{key}')
    for key, value in source['commands'].items():
        if key.startswith('limit_') and target['commands'].get(key) != value:
            mismatches.append(f'commands.{key}')
    if mismatches:
        raise ValueError('Fine-tuning policy contract mismatch: ' + ', '.join(mismatches))
    return dict(checkpoint=str(checkpoint), source_iteration=int(match.group(1)),
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                parameters_sha256=hashlib.sha256(params.read_bytes()).hexdigest(),
                optimizer_restored=False, curriculum_restored=False,
                curriculum_initial_iteration=int(match.group(1)))
