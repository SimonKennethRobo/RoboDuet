"""Bind a frozen trajectory library to one canonical MuJoCo initial state.

The input library contains geometry and time laws only.  This command settles
one selected rl_sar policy once, freezes that physical state, and anchors every
library trajectory so its first EE pose equals the measured canonical EE pose.
The resulting suite/reference pair is directly consumable by
``benchmark.wbc.mujoco`` and ``benchmark.wbc.cross_method_cli``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmark.wbc.mujoco import (
    FrozenReference,
    _quat_mul_xyzw,
    _quat_xyzw_from_matrix,
    _sha256,
    joint_order_indices,
)
from benchmark.wbc.scoring import DEVELOPMENT_TIMED_TRAJECTORY_PROTOCOL
from benchmark.wbc.suite import (
    SUITE_VERSION,
    TIMING_CONTRACT_VERSION,
    finalize_task_spec,
    refresh_suite_hash,
)

LIBRARY_SCHEMA = "roboduet-frozen-trajectory-library-v4"


def _content_digest(arrays: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        value = np.ascontiguousarray(value)
        digest.update(str(value.shape).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _load_library(library: Path) -> tuple[dict, dict[str, np.ndarray]]:
    manifest_path = library / "manifest.json"
    archive_path = library / "trajectories.npz"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != LIBRARY_SCHEMA:
        raise ValueError(
            f"expected {LIBRARY_SCHEMA}, got {manifest.get('schema_version')!r}"
        )
    expected = manifest.get("artifacts", {}).get("trajectories.npz", {}).get("sha256")
    if not expected or _sha256(archive_path) != expected:
        raise ValueError(f"trajectory archive hash mismatch: {archive_path}")
    with np.load(archive_path, allow_pickle=False) as source:
        arrays = {name: source[name].copy() for name in source.files}
    identifiers = [str(value) for value in arrays["trajectory_id"].tolist()]
    records = manifest.get("trajectories", [])
    if identifiers != [record["trajectory_id"] for record in records]:
        raise ValueError("manifest/archive trajectory order differs")
    if len(identifiers) != int(manifest.get("sample_count", -1)):
        raise ValueError("manifest/archive trajectory count differs")
    for row, record in enumerate(records):
        ng = int(arrays["gamma_points"][row])
        nt = int(arrays["time_law_points"][row])
        content = _content_digest([
            arrays["gamma_s"][row, :ng],
            arrays["gamma_p"][row, :ng],
            arrays["gamma_quat_xyzw"][row, :ng],
            arrays["tl_t"][row, :nt],
            arrays["tl_s"][row, :nt],
            arrays["tl_sdot"][row, :nt],
        ])
        if content != record["content_sha256"]:
            raise ValueError(f"trajectory content hash mismatch: {identifiers[row]}")
    return manifest, arrays


def freeze_policy_state(stack_root: Path, policy_key: str, scene: Path, settle_s: float) -> tuple[dict, np.ndarray]:
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    sys.path.insert(0, str(scripts))
    from sim2sim_mujoco import RlSarMujoco  # pylint: disable=import-outside-toplevel

    policy_root = stack_root / "rl_sar/policy/go2_x5"
    sim = RlSarMujoco(policy_root, policy_key, scene, seed=0)
    sim.run(settle_s, [0.0] * 6, getup_seconds=min(2.0, settle_s))
    mujoco.mj_forward(sim.model, sim.data)
    base = sim.model.body("base_link").id
    ee = sim.model.site("x5_ee").id
    if sim.data.xpos[base, 2] < 0.17 or sim.data.xmat[base].reshape(3, 3)[2, 2] < 0.5:
        raise RuntimeError("cannot freeze a fallen canonical initial state")
    _policy_from_canonical, canonical_from_policy = joint_order_indices(sim.model, sim.mapping)
    q, dq, quat, *_ = sim.read_state()
    base_rotation = sim.data.xmat[base].reshape(3, 3)
    initial_state = {
        "root_state_env_local": np.r_[
            sim.data.qpos[:3], quat, sim.data.qvel[:3], base_rotation @ sim.data.qvel[3:6]
        ].tolist(),
        "dof_position_rad": q[canonical_from_policy].tolist(),
        "dof_velocity_rad_s": dq[canonical_from_policy].tolist(),
    }
    ee_pose = np.r_[
        sim.data.site_xpos[ee],
        _quat_xyzw_from_matrix(sim.data.site_xmat[ee].reshape(3, 3)),
    ]
    if hasattr(sim, "close"):
        sim.close()
    return initial_state, ee_pose


def materialize(
    library: Path,
    output: Path,
    *,
    initial_state: dict,
    canonical_ee_pose: np.ndarray,
    completion_timeout_s: float,
    initialization: dict,
) -> Path:
    library = library.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    manifest, arrays = _load_library(library)
    canonical_ee_pose = np.asarray(canonical_ee_pose, dtype=np.float64)
    if canonical_ee_pose.shape != (7,) or not np.isfinite(canonical_ee_pose).all():
        raise ValueError("canonical EE pose must be finite xyz+xyzw")
    canonical_ee_pose[3:] /= np.linalg.norm(canonical_ee_pose[3:])

    tasks = []
    for row, source in enumerate(manifest["trajectories"]):
        first_position = arrays["gamma_p"][row, 0].astype(np.float64)
        first_quaternion = arrays["gamma_quat_xyzw"][row, 0].astype(np.float64)
        first_quaternion /= np.linalg.norm(first_quaternion)
        inverse_first = np.r_[-first_quaternion[:3], first_quaternion[3]]
        orientation_alignment = _quat_mul_xyzw(canonical_ee_pose[3:], inverse_first)
        task = dict(source)
        task.update(
            task_family="timed_trajectory",
            source_trajectory_id=source["trajectory_id"],
            timing_contract_version=TIMING_CONTRACT_VERSION,
        )
        anchor = canonical_ee_pose[:3] - first_position
        # Library Z is an absolute ground-relative EE height. Translating it
        # to the settled EE start height can push large downward sweeps below
        # the ground (for example curriculum A4). Only XY is start-aligned.
        anchor[2] = 0.0
        finalize_task_spec(
            task,
            initial_state=initial_state,
            anchor_env_local=anchor,
            orientation_left_multiplier_xyzw=orientation_alignment,
            deadline_s=float(source["duration_s"]) + completion_timeout_s,
        )
        tasks.append(task)

    output.mkdir(parents=True)
    arrays["task_id"] = np.asarray([task["task_id"] for task in tasks])
    reference_path = output / "reference.npz"
    np.savez_compressed(reference_path, **arrays)
    reference_record = {
        "path": reference_path.name,
        "format": "numpy_npz_v1",
        "sha256": _sha256(reference_path),
        "quaternion_order": "xyzw",
        "position_z_reference": "task_anchor_environment_local_translation",
    }
    suite = {
        "suite_version": SUITE_VERSION,
        "reference_frame": "environment_local_world",
        "initialization": initialization,
        "evaluation_protocol": DEVELOPMENT_TIMED_TRAJECTORY_PROTOCOL,
        "timing_contract": {
            "version": TIMING_CONTRACT_VERSION,
            "completion_timeout_s": completion_timeout_s,
        },
        "reference_archive": reference_record,
        "trajectories": tasks,
        "source_library": {
            "path": str(library),
            "schema_version": manifest["schema_version"],
            "manifest_sha256": _sha256(library / "manifest.json"),
            "archive_sha256": _sha256(library / "trajectories.npz"),
        },
    }
    refresh_suite_hash(suite)
    suite_path = output / "trajectory_suite.json"
    suite_path.write_text(json.dumps(suite, indent=2, allow_nan=False) + "\n")

    for task in tasks:
        frozen = FrozenReference(suite_path, task["task_id"])
        _, position, quaternion = frozen.at(0.0)
        np.testing.assert_allclose(position[:2], canonical_ee_pose[:2], atol=1e-7)
        source_row = next(
            index for index, identifier in enumerate(arrays["trajectory_id"].tolist())
            if str(identifier) == task["source_trajectory_id"]
        )
        np.testing.assert_allclose(position[2], arrays["gamma_p"][source_row, 0, 2], atol=1e-7)
        if float(frozen.gamma_p[:, 2].min()) < -1e-7:
            raise AssertionError(f"ground-relative reference enters the ground: {task['task_id']}")
        if abs(float(np.dot(quaternion, canonical_ee_pose[3:]))) < 1.0 - 1e-7:
            raise AssertionError(f"initial orientation anchor mismatch: {task['task_id']}")
    return suite_path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library", default="benchmark/data/frozen_trajectory_library",
        help="Directory containing manifest.json and trajectories.npz.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--stack-root", default="/home/simon/Projects/Simon/wbc_rl_mpc")
    parser.add_argument("--policy-key", default="I_Q")
    parser.add_argument(
        "--scene",
        default="/home/simon/Projects/Simon/wbc_rl_mpc/rl_sar/src/rl_sar_zoo/go2_x5_description/mjcf/scene.xml",
    )
    parser.add_argument("--settle-s", type=float, default=4.0)
    parser.add_argument("--completion-timeout-s", type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.settle_s <= 0 or args.completion_timeout_s < 0:
        parser.error("--settle-s must be positive and --completion-timeout-s non-negative")

    stack_root = Path(args.stack_root).resolve()
    scene = Path(args.scene).resolve()
    policy_root = stack_root / "rl_sar/policy/go2_x5"
    for required in (scene, policy_root / "base.yaml", policy_root / args.policy_key / "config.yaml",
                     policy_root / args.policy_key / "policy.pt"):
        if not required.is_file():
            raise FileNotFoundError(required)
    initial_state, ee_pose = freeze_policy_state(
        stack_root, args.policy_key, scene, args.settle_s
    )
    initialization = {
        "source": "once-frozen rl_sar getup and zero-command settle",
        "policy_key": args.policy_key,
        "settle_s": args.settle_s,
        "policy_sha256": _sha256(policy_root / args.policy_key / "policy.pt"),
        "policy_config_sha256": _sha256(policy_root / args.policy_key / "config.yaml"),
        "base_config_sha256": _sha256(policy_root / "base.yaml"),
        "scene": str(scene),
        "scene_sha256": _sha256(scene),
        "canonical_ee_pose_xyz_xyzw": ee_pose.tolist(),
        "xy_anchor_policy": "align_each_trajectory_start_to_canonical_ee_xy",
        "z_anchor_policy": "preserve_library_ground_relative_height",
        "policy_history_at_replay": "reset; physical state only is frozen",
    }
    suite_path = materialize(
        Path(args.library), Path(args.output), initial_state=initial_state,
        canonical_ee_pose=ee_pose, completion_timeout_s=args.completion_timeout_s,
        initialization=initialization,
    )
    payload = json.loads(suite_path.read_text())
    print(json.dumps({
        "suite": str(suite_path),
        "reference": str(suite_path.parent / "reference.npz"),
        "trajectory_count": len(payload["trajectories"]),
        "suite_sha256": payload["suite_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
