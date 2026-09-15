import json

import numpy as np

from benchmark.wbc.cross_method_cli import METHODS, normalize_trace_protocol
from benchmark.wbc.scoring import (
    DEVELOPMENT_KINEMATIC_PROTOCOL,
    DEVELOPMENT_TIMED_TRAJECTORY_PROTOCOL,
)
from benchmark.wbc.trace import TRACE_SCHEMA_VERSION
from benchmark.wbc.mujoco_video import load_replay_trace, video_sample_indices


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
