"""Regression checks for policy identity, fitted parameters and gait timing."""
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sysid.identification_bundle import load_bundle_contract, resolve_bundle
from sysid.identify_iq_mujoco import CHANNELS, collect, commands, protocol, sha, write_json
from sysid.identify_policy import check_frozen_fit
from sysid.iq_response_model import evaluate, fit, predict
from sysid.run_iq_mpc import export
from sysid import identification_models as response_models


@pytest.fixture
def bundle(tmp_path):
    robot = tmp_path / "go2_x5"
    policy = robot / "test_policy"
    policy.mkdir(parents=True)
    (policy / "policy.pt").write_bytes(b"fixture-policy")
    base = dict(num_of_dofs=18, num_leg_dofs=12, num_arm_dofs=6,
                default_dof_pos=[0.] * 18, dt=.005, decimation=4)
    config = dict(num_observations=9, observations=["gravity_vec", "roboduet/dog_commands"],
                  observations_history=[1, 0], dog_commands_scale=[1.] * 6,
                  gait_frequency=3.25, use_dynamic_gait=False)
    (robot / "base.yaml").write_text(yaml.safe_dump({"go2_x5": base}))
    (policy / "config.yaml").write_text(yaml.safe_dump({"go2_x5/test_policy": config}))
    return robot, policy


def test_accepts_bundle_key_directory_and_policy_file(bundle):
    robot, policy = bundle
    for argument in ["test_policy", str(policy), str(policy / "policy.pt")]:
        assert resolve_bundle(argument, robot) == (robot, "test_policy")
    contract = load_bundle_contract(robot, "test_policy")
    assert contract["num_observations"] == 9
    assert contract["gait_frequency_hz"] == 3.25
    assert contract["stop_gait_at_stand"] is False


@pytest.mark.parametrize("overrides,match", [
    ({"decimation": 2}, "20 ms"),
    ({"observations": ["unsupported_term"]}, "commands in observations"),
    ({"observations": ["roboduet/dog_commands", "unsupported_term"]}, "Observation adapter"),
])
def test_rejects_unsupported_contract_before_collection(bundle, overrides, match):
    robot, policy = bundle
    path = policy / "config.yaml"
    config = yaml.safe_load(path.read_text())
    config["go2_x5/test_policy"].update(overrides)
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match=match):
        load_bundle_contract(robot, "test_policy")


def test_contract_supports_five_channel_no_height_policy(bundle):
    robot, policy = bundle
    path = policy / "config.yaml"; config = yaml.safe_load(path.read_text())
    params = config["go2_x5/test_policy"]
    params.update(omit_height=True, dog_commands_scale=[1.]*5, num_observations=8)
    path.write_text(yaml.safe_dump(config))
    contract = load_bundle_contract(robot, "test_policy")
    assert contract["command_channels"] == ["vx", "vy", "wz", "pitch", "roll"]


def test_contract_supports_robot_lab_seven_value_command_term_without_gait(bundle):
    robot, policy = bundle
    path = policy / "config.yaml"; config = yaml.safe_load(path.read_text())
    params = config["go2_x5/test_policy"]
    params.update(num_observations=7, observations=["robot_lab/velocity_pose_commands"])
    params.pop("dog_commands_scale"); params.pop("gait_frequency")
    path.write_text(yaml.safe_dump(config))
    contract = load_bundle_contract(robot, "test_policy")
    assert contract["policy_command_observation_dim"] == 7
    assert contract["command_channels"] == CHANNELS
    assert contract["supports_gait_phase"] is False


def test_collection_rejects_different_policy_on_resume(bundle, tmp_path):
    robot, _ = bundle
    output = tmp_path / "experiment"
    output.mkdir()
    write_json(output / "protocol.json", {"policy": "different_policy"})
    args = argparse.Namespace(output=str(output), robot_dir=str(robot), policy_key="test_policy", scene="unused")
    with pytest.raises(ValueError, match="different policy"):
        collect(args)


def model_with_residual():
    model = {name: dict(gain=1., bias=0., tau_s=.2, delay_s=0., fit_rmse=0.) for name in CHANNELS}
    model["height"]["bias"] = .36
    for name in CHANNELS[3:]:
        model[name]["residual"] = dict(amplitude=.01, speed_amplitude=0., harmonic=1, phase_offset=0.)
    return model


def test_prediction_uses_bundle_frequency_and_fixed_clock_at_stand():
    model = model_with_residual()
    commands = np.zeros((80, 6))
    initial = np.array([0., 0., 0., .36, 0., 0.])
    result = predict(model, initial, 0., False, commands, gait_frequency=3.25, stop_gait_at_stand=False)
    expected_z = .36 + .01 * np.sin(2*np.pi*3.25*.02*np.arange(81))
    np.testing.assert_allclose(result[:, 3], expected_z, atol=1e-12)
    info = dict(gait_frequency_hz=3.25, stop_gait_at_stand=False)
    data = dict(y=result, t=.02*np.arange(81), u=np.zeros((81, 6)), phase=2*np.pi*3.25*.02*np.arange(81))
    errors = evaluate(model, [(info, data)])
    assert max(value for horizon in errors.values() for value in horizon.values()) < 1e-12
    stopped = predict(model, initial, 0., False, commands, gait_frequency=3.25, stop_gait_at_stand=True)
    np.testing.assert_allclose(stopped[:, 3], .36)


def test_v2_protocol_has_disjoint_splits_and_joint_excitation():
    spec = protocol()
    assert spec["schema"] == "policy_mujoco_identification_v2"
    assert {split: sum(e["split"] == split for e in spec["episodes"])
            for split in ["train", "development", "test"]} == {
                "train": 36, "development": 18, "test": 18}
    assert all(any(e["split"] == split and e["channel"] == -1 for e in spec["episodes"])
               for split in ["train", "development", "test"])


def test_v2_protocol_only_excites_available_channels():
    available = ["vx", "vy", "wz", "pitch", "roll"]
    spec = protocol(available)
    assert spec["command_limits"][3] == 0.
    assert {split: sum(e["split"] == split for e in spec["episodes"])
            for split in ["train", "development", "test"]} == {
                "train": 32, "development": 17, "test": 17}
    joint = next(e for e in spec["episodes"] if e["channel"] == -1)
    _, command = commands(joint)
    assert np.all(command[:, 3] == 0.)


def test_critical_second_order_exact_zoh_step_response():
    model = dict(order="second", natural_frequency_rad_s=5., damping_ratio=1.)
    ad, bd = response_models.transition(model)
    state = np.zeros(2)
    for _ in range(10):
        state = ad@state + bd
    t = 10*response_models.DT
    expected = 1.-np.exp(-5*t)*(1.+5*t)
    assert state[0] == pytest.approx(expected, abs=1e-12)


@pytest.fixture
def fitted_experiment(bundle, tmp_path):
    robot, policy = bundle
    root = tmp_path / "experiment"
    models = root / "models_v2"
    models.mkdir(parents=True)
    spec = {**load_bundle_contract(robot, "test_policy"), "episodes": []}
    write_json(root / "protocol.json", spec)
    paths = [robot / "base.yaml", policy / "config.yaml", policy / "policy.pt"]
    write_json(root / "input_manifest.json", dict(files={str(p): sha(p) for p in paths}))
    gait = model_with_residual()
    nominal = copy.deepcopy(gait)
    for name in CHANNELS[3:]:
        nominal[name].pop("residual")
    write_json(models / "F0.json", nominal)
    delayed = copy.deepcopy(nominal)
    for name in CHANNELS[:5]:
        delayed[name].update(order="first", delay_s=.02, delay_steps=1)
    second = {name: dict(gain=1., bias=.36 if name == "height" else 0., order="second",
        natural_frequency_rad_s=5., damping_ratio=1., delay_s=0., delay_steps=0) for name in CHANNELS}
    write_json(models / "F1.json", delayed)
    write_json(models / "F1_gait.json", gait)
    write_json(models / "F2.json", second)
    names = ["F0", "F1", "F1_gait", "F2"]
    write_json(models / "selection.json", dict(protocol_sha256=sha(root/"protocol.json"),
               selected_model="F2", model_hashes={name: sha(models/f"{name}.json") for name in names}))
    stack = tmp_path / "stack"
    task = stack / "go2_x5_ocs2/config/task_floating.info"
    task.parent.mkdir(parents=True)
    task.write_text("""model_information { manipulatorModelType 3 }
model_settings { recompileLibraries true }
basePositionLimits { activate true }
initialState { fullyActuatedFloatingArmManipulator { (0,0) 0 } }
inputCost
{
 fullyActuatedFloatingArmManipulator { (0,0) 1 }
}
lower { fullyActuatedFloatingArmManipulator { (0,0) -1 } }
upper { fullyActuatedFloatingArmManipulator { (0,0) 1 } }
""")
    return root, stack, policy


def test_export_embeds_selected_policy_and_measured_model(fitted_experiment):
    root, stack, policy = fitted_experiment
    export(root, stack)
    tasks = root / "mpc"
    gait = (tasks / "task_test_policy_gait.info").read_text()
    nominal = (tasks / "task_test_policy_first_order.info").read_text()
    assert "manipulatorModelType 4" in gait
    assert "gaitFrequency 3.25" in gait
    assert "stopGaitAtStand false" in gait
    assert "height { gain 1 timeConstant 0.20000000000000001 bias 0.35999999999999999" in gait
    assert "gaitResidual\n {\n activate false" in nominal
    assert "amplitude" not in nominal
    manifest = json.loads((tasks / "manifest.json").read_text())
    assert manifest["policy"] == "test_policy"
    assert manifest["policy_sha256"] == sha(policy / "policy.pt")
    assert manifest["models"]["second_order"]["state_dim"] == 22
    assert manifest["models"]["selected"]["model_name"] == "F2"
    assert "responseOrder 2" in (tasks / "task_test_policy_selected.info").read_text()
    assert not (tasks / "task_I_Q_gait.info").exists()


@pytest.mark.parametrize("changed", ["bundle", "model"])
def test_export_rejects_policy_or_model_changed_since_fit(fitted_experiment, changed):
    root, stack, policy = fitted_experiment
    path = policy / "policy.pt" if changed == "bundle" else root / "models_v2/F0.json"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs|modified"):
        export(root, stack)
    assert not (root / "mpc/manifest.json").exists()


def test_fit_does_not_freeze_incomplete_data(tmp_path):
    write_json(tmp_path / "protocol.json", dict(episodes=[]))
    with pytest.raises(ValueError, match="Incomplete train"):
        fit(argparse.Namespace(root=str(tmp_path), model_dir="models_v2"))
    assert not (tmp_path / "models_v2/selection.json").exists()


def test_resume_rejects_tampered_fitted_model(fitted_experiment):
    root, _, _ = fitted_experiment
    path = root / "models_v2/selection.json"
    selection = json.loads(path.read_text())
    selection["raw_hashes"] = {"train_000": "unused"}
    write_json(path, selection)
    (root / "models_v2/F0.json").write_text("{}")
    with pytest.raises(ValueError, match="Frozen model changed"):
        check_frozen_fit(root, sha)
