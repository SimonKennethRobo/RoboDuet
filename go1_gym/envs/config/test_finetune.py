import pickle
from types import SimpleNamespace
import isaacgym
import pytest
from go1_gym.envs.config import build_roboduet_config, cfg_to_dict
from go1_gym.envs.config.finetune import validate_finetune
from scripts.auto_train import parse_args, configure_train_stage
from go1_gym.utils import global_switch


def test_weight_only_finetune_contract_and_schedule(tmp_path):
    args = parse_args(['--train_stage', 'stage1', '--dyna_gait', '--experiment', 'G',
                      '--experiment_config', 'configs/coordination_demo_finetune.json'])
    cfg = build_roboduet_config(args)
    checkpoint = tmp_path / 'checkpoints_dog/ac_weights_020000.pt'
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b'validation fixture; runner checks actual tensors')
    snapshot = {'Cfg': cfg_to_dict(cfg)}
    parameters = tmp_path / 'parameters.pkl'
    parameters.write_bytes(pickle.dumps(snapshot))
    meta = validate_finetune(cfg, checkpoint)
    assert meta['source_iteration'] == 20000 and not meta['optimizer_restored']
    args.stage1_finetune_iteration = meta['source_iteration']
    configure_train_stage(args, cfg)
    assert global_switch.count == global_switch.stage1_count == 20000
    assert global_switch.pretrained_to_wbc_start == 28001 and not global_switch.switch_open
    snapshot['Cfg']['rewards']['attitude_command_convention'] = 'legacy'
    parameters.write_bytes(pickle.dumps(snapshot))
    with pytest.raises(ValueError, match='RPY'):
        validate_finetune(cfg, checkpoint)
    snapshot['Cfg']['rewards']['attitude_command_convention'] = 'rpy'
    snapshot['Cfg']['dog']['observe_clock_inputs'] = False
    parameters.write_bytes(pickle.dumps(snapshot))
    with pytest.raises(ValueError, match='observe_clock_inputs'):
        validate_finetune(cfg, checkpoint)
