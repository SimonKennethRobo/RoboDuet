import unittest
from types import SimpleNamespace

from go1_gym.envs.config import (
    apply_config_snapshot,
    build_config,
    build_roboduet_config,
    build_wtw_config,
    cfg_to_dict,
    set_cfg_value,
)
from go1_gym.envs.config.go1 import GO1_PROFILE
from go1_gym.envs.config.roboduet import ROBODUET_OVERRIDES
from go1_gym.envs.config.wtw import WTW_PROFILE


def _args(*, traj_track=False, dyna_gait=False):
    return SimpleNamespace(
        num_envs=17,
        robot="go2",
        use_rot6d=True,
        traj_track=traj_track,
        dyna_gait=dyna_gait,
        dyna_gait_min_frequency=0.25,
        no_stage1_arm_curriculum=False,
    )


class UnifiedConfigTest(unittest.TestCase):
    def test_roboduet_profile_contains_no_redundant_overrides(self):
        inherited_cfg = build_config(GO1_PROFILE, WTW_PROFILE)
        redundant = []

        for path, override_value in ROBODUET_OVERRIDES.items():
            inherited_value = inherited_cfg
            for part in path.split("."):
                if not hasattr(inherited_value, part):
                    break
                inherited_value = getattr(inherited_value, part)
            else:
                if type(inherited_value) is type(override_value) and inherited_value == override_value:
                    redundant.append(path)

        self.assertEqual(redundant, [])

    def test_feature_layout_dimensions(self):
        expected = {
            (False, False): (66, 31, 72, 8, 6),
            (True, False): (161, 117, 81, 9, 6),
            (False, True): (74, 43, 77, 12, 11),
            (True, True): (169, 131, 86, 15, 11),
        }

        for (traj_track, dyna_gait), dimensions in expected.items():
            with self.subTest(traj_track=traj_track, dyna_gait=dyna_gait):
                cfg = build_roboduet_config(_args(traj_track=traj_track, dyna_gait=dyna_gait))
                actual = (
                    cfg.env.num_observations,
                    cfg.arm.arm_num_observations,
                    cfg.dog.dog_num_observations,
                    cfg.arm.num_actions_arm_cd,
                    cfg.dog.dog_num_commands,
                )
                self.assertEqual(actual, dimensions)

    def test_builds_are_isolated(self):
        first = build_roboduet_config(_args(dyna_gait=True))
        second = build_roboduet_config(_args())

        first.commands.limit_vel_x[0] = -99.0
        first.arm.trajectory.traj_type.append("line")

        self.assertEqual(second.commands.limit_vel_x, [-1.0, 1.0])
        self.assertEqual(second.arm.trajectory.traj_type, ["point"])
        self.assertFalse(second.commands.use_dynamic_gait)
        self.assertEqual(second.commands.gait_frequency_cmd_range, [1.0, 6.0])
        self.assertIsInstance(second.sim.physx, dict)

    def test_wtw_profile_is_explicit_and_independent(self):
        cfg = build_wtw_config()

        self.assertEqual(cfg._profiles, ("go1", "wtw"))
        self.assertEqual(cfg.commands.num_commands, 15)
        self.assertEqual(cfg.terrain.mesh_type, "plane")
        self.assertEqual(cfg._provenance["terrain.mesh_type"], "wtw")
        self.assertFalse(hasattr(cfg, "arm"))

    def test_snapshot_round_trip_contains_all_sections(self):
        source = build_roboduet_config(_args(traj_track=True, dyna_gait=True))
        snapshot = cfg_to_dict(source)
        restored = build_roboduet_config(_args())
        apply_config_snapshot(restored, snapshot)

        self.assertEqual(cfg_to_dict(restored), snapshot)
        self.assertEqual(snapshot["terrain"]["reset_curriculum_start_threshold"], 0.7)
        self.assertTrue(snapshot["asset"]["file"].endswith("arx5go2.urdf"))
        self.assertEqual(snapshot["arm"]["trajectory"]["traj_type"], ["point"])

    def test_runtime_override_rejects_unknown_path(self):
        cfg = build_roboduet_config(_args())
        with self.assertRaises(KeyError):
            set_cfg_value(cfg, "commands.typo_limit", 1.0)


if __name__ == "__main__":
    unittest.main()
