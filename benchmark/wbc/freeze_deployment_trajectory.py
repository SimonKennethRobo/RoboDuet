"""Freeze a deployment YAML and one settled plant state into a diagnostic TaskSpec.

This creates a NEW task; it never reanchors an existing benchmark per method.
The original ROS TrajectorySpec performs the one-time world/EE anchoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import yaml

from benchmark.wbc.mujoco import FrozenReference, _quat_xyzw_from_matrix, _sha256, joint_order_indices
from benchmark.wbc.scoring import DEVELOPMENT_TIMED_TRAJECTORY_PROTOCOL
from benchmark.wbc.suite import SUITE_VERSION, finalize_task_spec, refresh_suite_hash


def freeze(args):
    root = Path(args.stack_root).resolve()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    sys.path.insert(0, str(root / "go2_x5_ocs2_bridge/scripts"))
    from sim2sim_mujoco import RlSarMujoco
    from task_space_trajectory import TrajectorySpec

    source = Path(args.trajectory).resolve()
    spec = TrajectorySpec(yaml.safe_load(source.read_text()))
    scene = root / "rl_sar/src/rl_sar_zoo/go2_x5_description/mjcf/scene.xml"
    policy_root = root / "rl_sar/policy/go2_x5"
    sim = RlSarMujoco(policy_root, args.policy_key, scene, seed=0)
    # Same get-up boundary as the deployment mirror, then a zero-command settle.
    sim.run(4.0, [0.0] * 6, getup_seconds=2.0)
    mujoco.mj_forward(sim.model, sim.data)
    base, ee = sim.model.body("base_link").id, sim.model.site("x5_ee").id
    if sim.data.xpos[base, 2] < .17 or sim.data.xmat[base].reshape(3, 3)[2, 2] < .5:
        raise RuntimeError("cannot freeze a fallen initial state")
    _order, canonical = joint_order_indices(sim.model, sim.mapping)
    q, dq, quat, *_ = sim.read_state()
    rotation = sim.data.xmat[base].reshape(3, 3)
    initial = {
        "root_state_env_local": np.r_[sim.data.qpos[:3], quat, sim.data.qvel[:3],
                                        rotation @ sim.data.qvel[3:6]].tolist(),
        "dof_position_rad": q[canonical].tolist(),
        "dof_velocity_rad_s": dq[canonical].tolist(),
    }
    measured_pose = np.r_[sim.data.site_xpos[ee],
                          _quat_xyzw_from_matrix(sim.data.site_xmat[ee].reshape(3, 3))]
    bound = spec.bind(0.0, measured_pose.tolist())
    poses, times = np.asarray(bound.poses), np.asarray(bound.times)
    dp = np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1)
    angles = 2 * np.arccos(np.clip(np.abs(np.sum(poses[1:, 3:] * poses[:-1, 3:], axis=1)), 0, 1))
    distance = np.sqrt(dp**2 + (.15 * angles)**2)
    # Stationary lead-in knots remain in the time law but not the arc geometry.
    distance[distance < 1e-10] = 0.0
    arc = np.r_[0., np.cumsum(distance)]
    keep = np.r_[True, distance > 0.0]
    if np.count_nonzero(keep) < 2:
        raise ValueError("diagnostic trajectory must contain motion")
    arrays = dict(gamma_s=arc[keep][None], gamma_p=poses[keep, :3][None],
                  gamma_quat_xyzw=poses[keep, 3:][None], tl_t=times[None], tl_s=arc[None],
                  tl_sdot=np.gradient(arc, times)[None],
                  gamma_points=np.asarray([keep.sum()]), time_law_points=np.asarray([len(times)]),
                  duration_s=np.asarray([bound.motion_end]))
    digest = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        digest.update(key.encode())
        digest.update(str(value.shape).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.tobytes())
    task = dict(task_family="timed_trajectory", content_sha256=digest.hexdigest(),
                duration_s=bound.motion_end, gamma_points=int(keep.sum()),
                time_law_points=len(times), path_length_m=float(arc[-1]),
                source_trajectory={"path": str(source), "sha256": _sha256(source),
                                   "parser_sha256": _sha256(root / "go2_x5_ocs2_bridge/scripts/task_space_trajectory.py"),
                                   "bound_measured_pose": measured_pose.tolist()})
    finalize_task_spec(task, initial_state=initial, anchor_env_local=[0., 0., 0.],
                       deadline_s=bound.motion_end + spec.completion_timeout)
    arrays["task_id"] = np.asarray([task["task_id"]])
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    archive = output / "reference.npz"
    np.savez_compressed(archive, **arrays)
    suite = dict(suite_version=SUITE_VERSION, reference_frame="environment_local_world",
                 initialization={"source": "once-frozen deployment getup 2s + locomotion settle 2s",
                                 "policy_key": args.policy_key,
                                 "policy_sha256": _sha256(policy_root / args.policy_key / "policy.pt"),
                                 "policy_config_sha256": _sha256(policy_root / args.policy_key / "config.yaml"),
                                 "scene_sha256": _sha256(scene),
                                 "policy_history_at_replay": "reset; physical state only is frozen"},
                 evaluation_protocol=DEVELOPMENT_TIMED_TRAJECTORY_PROTOCOL,
                 timing_contract={"version": "deployment-yaml-knots-v1", "diagnostic_only": True},
                 reference_archive={"path": archive.name, "sha256": _sha256(archive)},
                 trajectories=[task])
    refresh_suite_hash(suite)
    target = output / "suite.json"
    target.write_text(json.dumps(suite, indent=2) + "\n")
    frozen = FrozenReference(target)
    for t in np.linspace(0, bound.motion_end, 1001):
        _, p, quat = frozen.at(t)
        expected = np.asarray(bound.sample(t))
        np.testing.assert_allclose(p, expected[:3], atol=1e-8)
        if abs(np.dot(quat, expected[3:])) < 1.0 - 1e-8:
            raise AssertionError("frozen orientation differs from deployment parser")
    print(json.dumps({"suite": str(target), "task_id": task["task_id"],
                      "yaml_replay_equivalence_samples": 1001}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack-root", default="/home/simon/Projects/Simon/wbc_rl_mpc")
    parser.add_argument("--policy-key", default="robot_lab_rear_r30o_s42_11497")
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--output", required=True)
    freeze(parser.parse_args())


if __name__ == "__main__":
    main()
