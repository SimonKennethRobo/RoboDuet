"""Contract checks for the ID split, persistent target state and native OCS2 solve."""
import json
import os
from pathlib import Path
import subprocess
import unittest

from benchmark.dog_policy.ocs2_servo_mpc import prediction_matrices, Z_INDEX, NX, NU
from benchmark.dog_policy.servo_model import fit_model
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


class ServoContracts(unittest.TestCase):
    def test_fit_never_uses_validation_or_test_rows(self):
        rng = np.random.default_rng(32)
        samples = {k: rng.normal(size=(30, 6, width)) for k, width in (("z",23), ("zn",23), ("u",11))}
        samples.update(valid=np.ones((30,6), dtype=bool), splits=np.array(["train"]*3 + ["validation", "test", "test"]))
        before = fit_model(samples, True, 2, .01)
        for k in ("z", "zn", "u"):
            samples[k][:, 3:] *= 1000.
        after = fit_model(samples, True, 2, .01)
        for k in ("A", "B", "c"):
            np.testing.assert_array_equal(before[k], after[k])

    def test_target_state_is_persistent_and_matches_identified_grid(self):
        rng = np.random.default_rng(12)
        model = dict(A=(np.eye(23)*.8).tolist(), B=(rng.normal(size=(23,11))*.03).tolist(), c=np.zeros(23).tolist(), period_s=.1)
        model["A"][17:] = np.zeros((6,23)).tolist()
        model["B"][17:] = np.c_[np.zeros((6,5)), np.eye(6)].tolist()
        x = rng.normal(size=NX)*.1
        u = rng.normal(size=NU)*.1
        A,B,c = prediction_matrices(model, x, .03, "separate")
        expected = np.array(model["A"]) @ x[Z_INDEX] + np.array(model["B"]) @ np.r_[u[:5], x[20:26]+.1*u[5:]]
        np.testing.assert_allclose((A@x+B@u+c)[Z_INDEX], expected, atol=1e-12)
        np.testing.assert_allclose((A@x+B@u+c)[20:26], x[20:26]+.1*u[5:], atol=1e-12)
        self.assertGreater(np.max(np.abs(x[20:26]-x[8:14])), .01)

    def native_request(self):
        A, B = np.eye(NX), np.zeros((NX,NU))
        B[0,0] = .1
        A[26:] = 0.; B[26:] = np.eye(NU)
        B[20:26,5:] = np.eye(6)*.1
        F = np.zeros((12,NX)); F[:6,20:26]=np.eye(6); F[6:,20:26]=-np.eye(6)
        return dict(A=A.tolist(), B=B.tolist(), c=np.zeros(NX).tolist(), state=np.zeros(NX).tolist(),
            dt=.1, horizon=3, roll=0., chain=[], references=[[1.,0.,0.,0.,0.,0.,1.]]*4,
            ee_weights=[200.]*3+[30.]*3, R=[.02]*NU, D=[.01]*NU, terminal_weight=.3,
            previous_input_index=26, constraint_C=np.zeros((22,NX)).tolist(),
            constraint_D=np.vstack([np.eye(NU),-np.eye(NU)]).tolist(), constraint_e=[.2]*22,
            state_constraint_F=F.tolist(), state_constraint_h=[1.]*12)

    def native_solve(self, request):
        binary = ROOT / "benchmark/ocs2_servo/build/servo_ocs2"
        self.assertTrue(binary.is_file(), "Build benchmark/ocs2_servo first")
        env = os.environ.copy(); env.pop("LD_LIBRARY_PATH", None)
        result = subprocess.run([str(binary)], input=json.dumps(request)+"\n", text=True,
                                capture_output=True, check=True, timeout=10., env=env)
        return json.loads(result.stdout)

    def test_native_ocs2_active_bound_and_discrete_rollout(self):
        result = self.native_solve(self.native_request())
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["solver"], "OCS2 SqpSolver / HPIPM")
        self.assertGreater(result["command"][0], .199)
        self.assertLessEqual(result["command"][0], .2)
        self.assertAlmostEqual(result["predicted_state"][0], .1*result["command"][0], places=8)
        self.assertLess(result["max_dynamics_defect"], 1e-8)
        self.assertGreater(result["constraint_margin"], -1e-7)

    def test_native_infeasible_problem_is_not_accepted(self):
        request = self.native_request()
        request["state"][20] = 2.
        result = self.native_solve(request)
        self.assertFalse(result["ok"])


if __name__ == "__main__":
    unittest.main()
