import json

import numpy as np

from benchmark.data.materialize_trajectory_library import materialize
from benchmark.data.run_iq_native_ideal_library import _select_tasks
from benchmark.wbc.mujoco import FrozenReference


def test_materialize_library_binds_every_task_to_one_pose(tmp_path):
    root = tmp_path / "library"
    root.mkdir()
    gamma_p = np.asarray([[[0.1, -0.2, 0.3], [0.4, 0.0, 0.5]]], np.float32)
    gamma_q = np.asarray([[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]], np.float32)
    arrays = {
        "trajectory_id": np.asarray(["sample"]), "family": np.asarray(["line"]),
        "gamma_points": np.asarray([2], np.int32), "time_law_points": np.asarray([2], np.int32),
        "gamma_s": np.asarray([[0.0, 0.4]], np.float32), "gamma_p": gamma_p,
        "gamma_quat_xyzw": gamma_q, "gamma_tangent": np.asarray([[[1, 0, 0], [1, 0, 0]]], np.float32),
        "path_length_m": np.asarray([0.4], np.float32), "tl_t": np.asarray([[0.0, 2.0]], np.float32),
        "tl_s": np.asarray([[0.0, 0.4]], np.float32), "tl_sdot": np.asarray([[0.2, 0.2]], np.float32),
        "duration_s": np.asarray([2.0], np.float32),
    }
    content_arrays = [arrays[name][0] for name in
                      ("gamma_s", "gamma_p", "gamma_quat_xyzw", "tl_t", "tl_s", "tl_sdot")]
    from benchmark.data.materialize_trajectory_library import _content_digest, _sha256
    np.savez_compressed(root / "trajectories.npz", **arrays)
    record = {"trajectory_id": "sample", "family": "line", "content_sha256": _content_digest(content_arrays),
              "duration_s": 2.0, "gamma_points": 2, "time_law_points": 2, "path_length_m": 0.4}
    manifest = {"schema_version": "roboduet-frozen-trajectory-library-v4", "sample_count": 1,
                "trajectories": [record], "artifacts": {"trajectories.npz": {"sha256": _sha256(root / "trajectories.npz")}}}
    (root / "manifest.json").write_text(json.dumps(manifest))
    initial = {"root_state_env_local": [0.0] * 13, "dof_position_rad": [0.0] * 18,
               "dof_velocity_rad_s": [0.0] * 18}
    target = np.asarray([0.6, 0.1, 0.7, 0.0, 0.0, 0.0, 1.0])
    suite_path = materialize(root, tmp_path / "out", initial_state=initial,
                             canonical_ee_pose=target, completion_timeout_s=1.0,
                             initialization={"source": "test"})
    payload = json.loads(suite_path.read_text())
    assert len(payload["trajectories"]) == 1
    assert payload["trajectories"][0]["deadline_s"] == 3.0
    frozen = FrozenReference(suite_path, payload["trajectories"][0]["task_id"])
    np.testing.assert_allclose(frozen.at(0.0)[1], [target[0], target[1], 0.3], atol=1e-7)
    assert payload["trajectories"][0]["anchor_env_local_xyz_m"][2] == 0.0


def test_select_tasks_uses_requested_cell_order_and_deduplicates():
    tasks = [
        {"task_id": "a0b0", "cell_A": 0, "cell_B": 0},
        {"task_id": "a0b1", "cell_A": 0, "cell_B": 1},
        {"task_id": "random-line"},
    ]
    selected = _select_tasks(tasks, [[0, 1], [0, 0], [0, 1]])
    assert [task["task_id"] for task in selected] == ["a0b1", "a0b0"]
