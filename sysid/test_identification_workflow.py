"""CPU-only tests for policy snapshots and batch policy lists."""
import json
from pathlib import Path
import tempfile
import unittest

from sysid.identification_bundle import (
    snapshot_policy_bundle,
    verify_bundle_snapshot,
)
from sysid.run_policy_identification_benchmark import load_policy_file


class IdentificationWorkflowTests(unittest.TestCase):
    def test_snapshot_is_complete_and_independent_of_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            robot = root / "source/go2_x5"
            policy = robot / "candidate"
            policy.mkdir(parents=True)
            (robot / "base.yaml").write_text("base")
            (policy / "config.yaml").write_text("config")
            (policy / "policy.pt").write_bytes(b"weights")
            experiment = root / "experiment"
            frozen, key, manifest = snapshot_policy_bundle(robot, "candidate", experiment)
            self.assertEqual(key, "candidate")
            self.assertEqual((frozen / "candidate/policy.pt").read_bytes(), b"weights")
            self.assertEqual(len(manifest["files"]), 3)
            (policy / "policy.pt").write_bytes(b"new source weights")
            verified, _, _ = verify_bundle_snapshot(experiment, "candidate")
            self.assertEqual((verified / "candidate/policy.pt").read_bytes(), b"weights")

    def test_snapshot_detects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = root / "source/go2_x5/candidate"
            policy.mkdir(parents=True)
            (policy.parent / "base.yaml").write_text("base")
            (policy / "config.yaml").write_text("config")
            (policy / "policy.pt").write_bytes(b"weights")
            experiment = root / "experiment"
            frozen, _, _ = snapshot_policy_bundle(policy.parent, "candidate", experiment)
            (frozen / "candidate/config.yaml").write_text("changed")
            with self.assertRaisesRegex(ValueError, "snapshot changed"):
                verify_bundle_snapshot(experiment, "candidate")

    def test_policy_files_accept_text_and_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text = root / "policies.txt"
            text.write_text("# candidates\nI_Q\n\ncoord_I_26499\n")
            self.assertEqual(load_policy_file(text), ["I_Q", "coord_I_26499"])
            data = root / "policies.json"
            data.write_text(json.dumps(["I_Q", "/tmp/policy.pt"]))
            self.assertEqual(load_policy_file(data), ["I_Q", "/tmp/policy.pt"])


if __name__ == "__main__":
    unittest.main()
