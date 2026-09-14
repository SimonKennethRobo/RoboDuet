"""CPU regression tests for the WBC benchmark's public contracts."""

from types import SimpleNamespace

import isaacgym  # noqa: F401 - must precede torch
import pytest
import torch

from benchmark.compare import _summary_rows
from benchmark.wbc.scoring import aggregate_task_events, timed_trajectory_success
from benchmark.wbc.suite import trajectory_features


def _batch(time_law):
    return SimpleNamespace(
        gamma_p=torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]),
        gamma_quat=torch.tensor([[[0.0, 0.0, 0.0, 1.0]] * 3]),
        gamma_tangent=torch.tensor([[[1.0, 0.0, 0.0]] * 3]),
        gamma_s=torch.tensor([[0.0, 1.0, 1.0]]),
        tl_t=torch.tensor([[0.0, 0.5, 1.0]]),
        tl_s=torch.tensor([time_law]),
        tl_sdot=torch.tensor([[1.0, 1.0, 1.0]]),
        L=torch.tensor([1.0]),
        T=torch.tensor([1.0]),
        max_gamma_points=3,
        max_tl_points=3,
    )


def test_progress_without_accuracy_or_hold_is_not_success():
    result = timed_trajectory_success(
        {
            "reference_time_s": 1.0,
            "reference_duration_s": 1.0,
            "final_progress": 0.85,
            "final_ee_pos_error_m": 0.20,
            "final_ee_rot_error_rad": 1.0,
            "tracking_tube_fraction": 0.0,
            "endpoint_hold_time_s": 0.0,
        }
    )
    assert result == {"success": False, "end_reason": "criteria_not_met"}


def test_timed_trajectory_success_requires_and_accepts_full_contract():
    result = timed_trajectory_success(
        {
            "reference_time_s": 2.0,
            "reference_duration_s": 2.0,
            "final_progress": 1.0,
            "final_ee_pos_error_m": 0.01,
            "final_ee_rot_error_rad": 0.02,
            "tracking_tube_fraction": 0.9,
            "endpoint_hold_time_s": 0.5,
        }
    )
    assert result == {"success": True, "end_reason": "success"}


def test_full_time_law_changes_reference_content_hash():
    first = trajectory_features(_batch([0.0, 0.5, 1.0]), 0)
    second = trajectory_features(_batch([0.0, 0.2, 1.0]), 0)
    assert first["content_sha256"] != second["content_sha256"]


def test_task_booleans_are_aggregated_as_event_rates():
    rows = [
        {"completed": True, "fall": False},
        {"completed": False, "fall": True},
    ]
    summary = aggregate_task_events(rows)
    assert summary["completion_rate"] == pytest.approx(0.5)
    assert summary["fall_rate"] == pytest.approx(0.5)
    compared = _summary_rows({"candidate": {"wbc_trajectories": rows}})
    assert compared["candidate"]["completion_rate"] == pytest.approx(0.5)
    assert compared["candidate"]["fall_rate"] == pytest.approx(0.5)


def test_inactive_nan_is_removed_by_mask():
    from benchmark.wbc.evaluation import WBCAccumulator

    result = WBCAccumulator._masked(
        torch.tensor([float("nan"), 2.0]), torch.tensor([False, True])
    )
    assert torch.equal(result, torch.tensor([0.0, 2.0]))
