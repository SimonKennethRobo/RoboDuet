import json

import numpy as np

from benchmark.wbc.cross_method_cli import normalize_trace_protocol
from benchmark.wbc.scoring import (
    DEVELOPMENT_KINEMATIC_PROTOCOL,
    DEVELOPMENT_TIMED_TRAJECTORY_PROTOCOL,
)
from benchmark.wbc.trace import TRACE_SCHEMA_VERSION


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
