import hashlib
import json

import numpy as np
import pytest

# This repository's IsaacGym extension must load before any path imports torch.
import isaacgym  # noqa: F401

pytest.importorskip("mujoco")

from benchmark.wbc.mujoco import (
    FrozenReference,
    _forward_project,
    _push_force_at,
    _validated_push_events,
)
from benchmark.wbc.scoring import DEVELOPMENT_TIMED_TRAJECTORY_PROTOCOL
from benchmark.wbc.suite import refresh_suite_hash


def _file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_suite(tmp_path):
    task = {
        "task_family": "timed_trajectory",
        "content_sha256": "reference-content",
        "initial_state": {
            "root_state_env_local": [0.0] * 13,
            "dof_position_rad": [0.0] * 18,
            "dof_velocity_rad_s": [0.0] * 18,
        },
        "anchor_env_local_xy_m": [2.0, -1.0],
        "disturbance_schedule": [],
        "deadline_s": 2.0,
        "height_reference": "terrain_surface_at_reference_xy",
    }
    identity = {
        "task_family": task["task_family"],
        "reference_content_sha256": task["content_sha256"],
        "initial_state": task["initial_state"],
        "anchor_env_local_xy_m": task["anchor_env_local_xy_m"],
        "disturbance_schedule": task["disturbance_schedule"],
        "deadline_s": task["deadline_s"],
        "height_reference": task["height_reference"],
        "evaluation_protocol": DEVELOPMENT_TIMED_TRAJECTORY_PROTOCOL,
        "reference_frame": "environment_local_world",
    }
    task["task_spec_sha256"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    task["task_id"] = f"timed-trajectory-{task['task_spec_sha256'][:16]}"
    archive = tmp_path / "reference.npz"
    np.savez_compressed(
        archive,
        gamma_s=np.array([[0.0, 1.0]], np.float32),
        gamma_p=np.array([[[0.0, 0.0, 0.5], [1.0, 0.0, 0.5]]], np.float32),
        gamma_quat_xyzw=np.array([[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]], np.float32),
        tl_t=np.array([[0.0, 1.0]], np.float32),
        tl_s=np.array([[0.0, 1.0]], np.float32),
        duration_s=np.array([1.0], np.float32),
        gamma_points=np.array([2], np.int32),
        time_law_points=np.array([2], np.int32),
        task_id=np.array([task["task_id"]]),
    )
    group = {
        "suite_version": "wbc-task-spec-v2",
        "reference_frame": "environment_local_world",
        "initialization": {},
        "evaluation_protocol": DEVELOPMENT_TIMED_TRAJECTORY_PROTOCOL,
        "timing_contract": {},
        "trajectories": [task],
        "reference_archive": {
            "path": archive.name,
            "format": "numpy_npz_v1",
            "sha256": _file_hash(archive),
        },
    }
    refresh_suite_hash(group)
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps([group]))
    return suite


def _write_wrapped_v3_suite(tmp_path):
    task = {
        "task_family": "timed_trajectory",
        "content_sha256": "reference-content-v3",
        "initial_state": {
            "root_state_env_local": [0.0] * 13,
            "dof_position_rad": [0.0] * 18,
            "dof_velocity_rad_s": [0.0] * 18,
        },
        "anchor_env_local_xyz_m": [2.0, -1.0, 0.5],
        "anchor_env_local_xy_m": [2.0, -1.0],
        "orientation_left_multiplier_xyzw": [0.0, 0.0, 2**-0.5, 2**-0.5],
        "disturbance_schedule": [],
        "deadline_s": 2.0,
        "height_reference": "environment_local_translation",
    }
    identity = {
        "task_family": task["task_family"],
        "reference_content_sha256": task["content_sha256"],
        "initial_state": task["initial_state"],
        "anchor_env_local_xyz_m": task["anchor_env_local_xyz_m"],
        "orientation_left_multiplier_xyzw": task["orientation_left_multiplier_xyzw"],
        "disturbance_schedule": task["disturbance_schedule"],
        "deadline_s": task["deadline_s"],
        "height_reference": task["height_reference"],
        "evaluation_protocol": DEVELOPMENT_TIMED_TRAJECTORY_PROTOCOL,
        "reference_frame": "environment_local_world",
    }
    task["task_spec_sha256"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    task["task_id"] = f"timed-trajectory-{task['task_spec_sha256'][:16]}"
    archive = tmp_path / "reference_v3.npz"
    np.savez_compressed(
        archive,
        gamma_s=np.array([[0.0, 1.0]], np.float32),
        gamma_p=np.array([[[0.0, 0.0, 0.5], [1.0, 0.0, 0.5]]], np.float32),
        gamma_quat_xyzw=np.array([[[0.0, 0.0, 0.0, 1.0]] * 2], np.float32),
        tl_t=np.array([[0.0, 1.0]], np.float32),
        tl_s=np.array([[0.0, 1.0]], np.float32),
        duration_s=np.array([1.0], np.float32),
        gamma_points=np.array([2], np.int32),
        time_law_points=np.array([2], np.int32),
        task_id=np.array([task["task_id"]]),
    )
    record = {
        "path": archive.name,
        "format": "numpy_npz_v1",
        "sha256": _file_hash(archive),
    }
    group = {
        "suite_version": "wbc-task-spec-v3",
        "reference_frame": "environment_local_world",
        "initialization": {},
        "evaluation_protocol": DEVELOPMENT_TIMED_TRAJECTORY_PROTOCOL,
        "timing_contract": {},
        "trajectories": [task],
        "reference_archive": record,
    }
    refresh_suite_hash(group)
    suite = tmp_path / "suite_v3.json"
    suite.write_text(json.dumps([{"suite": group, "reference_archive": record}]))
    return suite


def test_frozen_reference_checks_hashes_and_applies_anchor(tmp_path):
    reference = FrozenReference(_write_suite(tmp_path))
    arc, position, quaternion = reference.at(0.5)
    assert arc == pytest.approx(0.5)
    assert position == pytest.approx([2.5, -1.0, 0.5])
    assert quaternion == pytest.approx([0.0, 0.0, 0.0, 1.0])


def test_frozen_reference_rolling_window_has_fixed_size_and_holds_endpoint(tmp_path):
    reference = FrozenReference(_write_suite(tmp_path))
    times, poses = reference.rolling_window(0.8, horizon_s=1.0, sample_dt_s=0.2)
    np.testing.assert_allclose(times, [0.8, 1.0, 1.2, 1.4, 1.6, 1.8])
    assert poses.shape == (6, 7)
    np.testing.assert_allclose(poses[1:, :3], [[3.0, -1.0, 0.5]] * 5, atol=1e-7)


def test_progress_projection_is_monotonic_and_window_bounded(tmp_path):
    reference = FrozenReference(_write_suite(tmp_path))
    arc, lateral = _forward_project(
        reference,
        previous_arc=0.25,
        position=np.array([3.0, -1.0, 0.5]),
        quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
    )
    assert 0.25 <= arc <= 0.40
    assert lateral == pytest.approx(0.60)


def test_mujoco_push_schedule_has_half_open_timing_and_adds_events():
    events = _validated_push_events([
        {
            "type": "constant_force", "start_time_s": 1.0,
            "duration_s": 0.25, "body": "base",
            "frame": "environment_world", "force_n": [32.0, 0.0, 0.0],
            "magnitude_n": 32.0, "application_point": "body_center_of_mass",
        },
        {
            "type": "constant_force", "start_time_s": 1.1,
            "duration_s": 0.1, "body": "base",
            "frame": "environment_world", "force_n": [0.0, 5.0, 0.0],
            "magnitude_n": 5.0, "application_point": "body_center_of_mass",
        },
    ])
    np.testing.assert_allclose(_push_force_at(events, 0.999), [0.0, 0.0, 0.0])
    np.testing.assert_allclose(_push_force_at(events, 1.15), [32.0, 5.0, 0.0])
    np.testing.assert_allclose(_push_force_at(events, 1.25), [0.0, 0.0, 0.0])


def test_mujoco_push_schedule_rejects_non_common_body():
    with pytest.raises(ValueError, match="unsupported MuJoCo disturbance"):
        _validated_push_events([{
            "type": "constant_force", "start_time_s": 1.0,
            "duration_s": 0.25, "body": "arm",
            "frame": "environment_world", "force_n": [1.0, 0.0, 0.0],
            "application_point": "body_center_of_mass",
        }])


def test_frozen_reference_reads_system_matrix_v3_wrapper_and_full_transform(tmp_path):
    reference = FrozenReference(_write_wrapped_v3_suite(tmp_path))
    _arc, position, quaternion = reference.at(0.5)
    assert position == pytest.approx([2.5, -1.0, 1.0])
    assert quaternion == pytest.approx([0.0, 0.0, 2**-0.5, 2**-0.5])
    poses = reference.timed_poses()
    assert poses.shape == (2, 7)
    np.testing.assert_allclose(
        poses[:, :3], [[2.0, -1.0, 1.0], [3.0, -1.0, 1.0]], atol=1e-7
    )


def test_reference_archive_tampering_is_rejected(tmp_path):
    suite = _write_suite(tmp_path)
    (tmp_path / "reference.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Reference archive hash mismatch"):
        FrozenReference(suite)
