import json
from argparse import Namespace

import numpy as np

from benchmark.wbc.cross_method_cli import METHODS, normalize_trace_protocol
from benchmark.wbc.scoring import (
    DEVELOPMENT_KINEMATIC_PROTOCOL,
    DEVELOPMENT_TIMED_TRAJECTORY_PROTOCOL,
)
from benchmark.wbc.trace import TRACE_SCHEMA_VERSION
from benchmark.wbc.mujoco_video import load_replay_trace, render_trace_artifacts, video_sample_indices


def test_normalize_trace_protocol_preserves_sample_fields(tmp_path):
    trace = tmp_path / "trace.npz"
    sample = np.arange(12, dtype=np.float32).reshape(2, 1, 6)
    np.savez_compressed(
        trace, schema_version=np.asarray(TRACE_SCHEMA_VERSION),
        protocol_json=np.asarray('{"old":true}'),
        kinematic_protocol_json=np.asarray('{"old":true}'),
        dof_position_rad=sample,
    )

    receipt = normalize_trace_protocol(trace)

    assert receipt["sample_fields_changed"] is False
    assert (tmp_path / "trace.backend.npz").is_file()
    with np.load(trace, allow_pickle=False) as normalized:
        np.testing.assert_array_equal(normalized["dof_position_rad"], sample)
        assert json.loads(str(normalized["protocol_json"].item())) == (
            DEVELOPMENT_TIMED_TRAJECTORY_PROTOCOL
        )
        assert json.loads(str(normalized["kinematic_protocol_json"].item())) == (
            DEVELOPMENT_KINEMATIC_PROTOCOL
        )


def test_registry_contains_every_handoff_method():
    assert set(METHODS) == {
        "roboduet", "roboduet_raw", "umi", "visual_wholebody",
        "wb_locoman", "qm_control", "deep_whole_body_control", "ma2022",
    }


def test_ocs2_task_override_is_only_added_to_roboduet(tmp_path, monkeypatch):
    from benchmark.wbc import cross_method_cli as cli
    task_file = tmp_path / "task_sota.info"
    task_file.write_text("task")
    scene = tmp_path / "scene.xml"
    scene.write_text("scene")
    suite = tmp_path / "suite.json"
    suite.write_text("suite")
    output = tmp_path / "output"
    monkeypatch.setattr(cli, "_method_contracts", lambda: {
        "roboduet": {"root": tmp_path, "policy_key": "p", "upper_controller": "floating_base_ocs2_mpc"},
    })
    monkeypatch.setattr(cli, "preflight", lambda _: {"methods": {"roboduet": {"status": "ready"}}})
    monkeypatch.setattr(cli, "_materialize_scenario_suite", lambda *args: (suite, "T0"))
    args = Namespace(
        method="roboduet", python=str(tmp_path / "python"), scene=str(scene), output=str(output),
        suite=str(suite), task_id="T0", scenarios=("nominal",), prepare_only=True,
        timeout_s=1, ocs2_transport="synchronous", roboduet_ocs2_task_file=str(task_file),
        umi_mujoco_profile="common", record_video=False,
    )
    _, _ = cli.run_policy_method(args)
    manifest = json.loads(next(output.glob("*/cross_method_manifest.json")).read_text())
    command = manifest["commands"][0]
    assert command[command.index("--ocs2-task-file") + 1] == str(task_file)


def test_normalization_only_requires_selected_scenarios(tmp_path, monkeypatch):
    from benchmark.wbc import cross_method_cli as cli
    nominal, push = tmp_path / "nominal", tmp_path / "push"
    nominal.mkdir()
    push.mkdir()  # aligned_cli prepares BOTH folders, even in nominal-only runs.
    (nominal / "trace.npz").write_bytes(b"test")
    (nominal / "receipt.json").write_text("{}")
    monkeypatch.setattr(cli, "normalize_trace_protocol", lambda _: {})
    monkeypatch.setattr(cli, "score_trace_archive", lambda _: [{}])
    monkeypatch.setattr(cli, "_sha256", lambda _: "test-hash")
    rows = cli._normalize_results(tmp_path, tmp_path, ("nominal",))
    assert len(rows) == 1


def test_replay_trace_and_video_sampling(tmp_path):
    trace = tmp_path / "trace.npz"
    samples = 6
    np.savez_compressed(
        trace,
        base_root_state=np.zeros((samples, 1, 13), dtype=np.float32),
        dof_position_rad=np.zeros((samples, 1, 18), dtype=np.float32),
        actual_ee_state=np.zeros((samples, 1, 13), dtype=np.float32),
        reference_ee_position_m=np.zeros((samples, 1, 3), dtype=np.float32),
        reference_time_s=np.arange(samples, dtype=np.float32)[:, None] * .02,
        control_dt_s=np.asarray(.02),
    )
    replay = load_replay_trace(trace)
    assert replay["base_root_state"].shape == (samples, 13)
    np.testing.assert_array_equal(video_sample_indices(samples, .02, 25), [0, 2, 4])


def test_video_artifact_contract_disables_target_base_pose(tmp_path, monkeypatch):
    trace = tmp_path / "trace.npz"
    samples = 2
    np.savez_compressed(
        trace, base_root_state=np.zeros((samples, 1, 13)),
        dof_position_rad=np.zeros((samples, 1, 18)), actual_ee_state=np.zeros((samples, 1, 13)),
        reference_ee_position_m=np.zeros((samples, 1, 3)),
        reference_time_s=np.arange(samples)[:, None] * .02, control_dt_s=np.asarray(.02),
    )
    scene = tmp_path / "scene.xml"
    scene.write_text("scene")
    from benchmark.wbc import mujoco_video
    monkeypatch.setattr(mujoco_video, "_tracking_plot", lambda _trace, path, *_: path.write_bytes(b"plot"))
    monkeypatch.setattr(mujoco_video, "_replay_video", lambda _trace, _scene, path, *_args, **_kwargs: path.write_bytes(b"video"))
    artifact = render_trace_artifacts(trace, scene, tmp_path / "artifacts", method="m", scenario="nominal")
    assert artifact["overlays"]["target_base_pose"] is False
