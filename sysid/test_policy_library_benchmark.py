"""Small CPU checks for the paired deployment boundary and failure accounting."""
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from benchmark.wbc.mujoco import _set_mpc_dog_command
from sysid.run_policy_library_benchmark import report, write_json


def check_matched_limits_preserve_command_order_and_policy_bounds():
    sim = SimpleNamespace(p={"limit_vel_x": [-.2, .2], "limit_body_pitch": [-.1, .1]})
    command = {"base_velocity_body": [1., -1., 1.], "body_posture": [.3, -.3, .3]}
    velocity = _set_mpc_dog_command(sim, command, [.55, .28, .65, .035, .2, 0.])
    np.testing.assert_allclose(velocity, [.2, -.28, .65])
    np.testing.assert_allclose(sim.command, [.2, -.28, .65, -.1, 0., .035])
    np.testing.assert_allclose(command["body_posture"], [.3, -.3, .3])
    with unittest.TestCase().assertRaises(ValueError):
        _set_mpc_dog_command(sim, command, [1., 1., 1., 1., float("nan"), 1.])


def check_report_keeps_unscored_failure_in_denominator(tmp_path):
    directory = tmp_path / "runs/policy/ideal/test"
    directory.mkdir(parents=True)
    write_json(directory / "execution.json", dict(policy="policy", mode="ideal", task_id="task", result={},
        operational_failure=True, model="ideal", trajectory="test", returncode=1, recorded_steps=0))
    task = dict(task_id="task", source_trajectory_id="test", duration_s=1., deadline_s=2.)
    plan = dict(policies=[dict(policy="policy", selected_model="F1")], tasks=[task], requested_trials=2, seed=1)
    data = report(tmp_path, plan)
    row = data["groups"][0]
    assert row["requested"] == 1
    assert row["operational_failures"] == 1
    assert row["successes"] == 0
    assert row["ee_pos_rmse_m"] is None


class PairedBenchmarkTests(unittest.TestCase):
    def test_command_boundary(self):
        check_matched_limits_preserve_command_order_and_policy_bounds()

    def test_failure_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            check_report_keeps_unscored_failure_in_denominator(Path(directory))


if __name__ == "__main__":
    unittest.main()
