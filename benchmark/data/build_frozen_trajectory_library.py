#!/usr/bin/env python3
"""Build and verify a simulator-free benchmark trajectory library from source.

By default the 6x6 curriculum grid is regenerated with RoboDuet's current
``CurriculumManager`` and ``TrajectoryFactory``. Deterministic random line and
circle families are then added. An existing gallery can be imported explicitly
with ``--import-grid`` for compatibility, but no prior data is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

def _find_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "modules/trajectory.py").is_file() and (candidate / "benchmark/wbc").is_dir():
            return candidate
    raise RuntimeError("could not locate the RoboDuet repository root")


ROOT = _find_repo_root()
sys.path.insert(0, str(ROOT))

from modules.curriculum import CurriculumManager
from modules.trajectory import mat_to_quat, quat_slerp, quat_to_mat
from modules.trajectory_generator import TimingGenerator, TrajectoryFactory


DEFAULT_OUTPUT = ROOT / "benchmark/data/frozen_trajectory_library_v4"
DEFAULT_ROBOT_SCENE = Path(
    "/home/simon/Projects/Simon/wbc_rl_mpc/rl_sar/"
    "src/rl_sar_zoo/go2_x5_description/mjcf/scene.xml"
)
SCHEMA = "roboduet-frozen-trajectory-library-v4"
LEGACY_SCHEMAS = {
    "roboduet-frozen-trajectory-library-v1",
    "roboduet-frozen-trajectory-library-v2",
    "roboduet-frozen-trajectory-library-v3",
}
SEED = 20260915
GRID_SEED = 12345
LINE_COUNT = 16
CIRCLE_COUNT = 16
Z_BOUNDS_M = (0.05, 1.0)
MAX_LINE_LENGTH_M = 3.0
MAX_CIRCLE_RADIUS_M = 3.0
LAMBDA_M_PER_RAD = 0.15


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=SEED, help="Seed for random line/circle families.")
    parser.add_argument("--grid-seed", type=int, default=GRID_SEED, help="Curriculum-grid seed.")
    parser.add_argument("--levels-a", type=int, default=6, help="Geometry difficulty levels.")
    parser.add_argument("--levels-b", type=int, default=6, help="Timing difficulty levels.")
    parser.add_argument("--line-count", type=int, default=LINE_COUNT)
    parser.add_argument("--circle-count", type=int, default=CIRCLE_COUNT)
    parser.add_argument(
        "--robot-scene", type=Path, default=DEFAULT_ROBOT_SCENE,
        help="Go2+X5 MJCF used for the 1:1 scale-reference model in Rerun.",
    )
    parser.add_argument(
        "--import-grid", type=Path,
        help="Optional old gallery directory containing manifest.json and trajectories.npz.",
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.shape).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _smoothstep(value: np.ndarray) -> np.ndarray:
    return value * value * (3.0 - 2.0 * value)


def _random_quaternions(rng: np.random.Generator, count: int) -> np.ndarray:
    # Normalized four-dimensional Gaussian samples are uniform on S^3 and
    # therefore induce uniform rotations on SO(3).
    quat = rng.normal(size=(count, 4))
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    for index in range(1, count):
        if np.dot(quat[index - 1], quat[index]) < 0.0:
            quat[index] *= -1.0
    return quat.astype(np.float32)


def _smooth_random_scalar(
    rng: np.random.Generator,
    progress: np.ndarray,
    knot_count: int,
    bounds: tuple[float, float],
    *,
    periodic: bool,
) -> tuple[np.ndarray, np.ndarray]:
    values = rng.uniform(bounds[0], bounds[1], size=knot_count)
    if periodic:
        values[-1] = values[0]
    scaled = progress * (knot_count - 1)
    segment = np.minimum(np.floor(scaled).astype(int), knot_count - 2)
    blend = _smoothstep(scaled - segment)
    output = values[segment] * (1.0 - blend) + values[segment + 1] * blend
    return output.astype(np.float32), values.astype(np.float32)


def _smooth_random_quaternion(
    rng: np.random.Generator,
    progress: np.ndarray,
    knot_count: int,
    *,
    periodic: bool,
) -> tuple[np.ndarray, np.ndarray]:
    knots = _random_quaternions(rng, knot_count)
    if periodic:
        knots[-1] = knots[0]
        if np.dot(knots[-2], knots[-1]) < 0.0:
            knots[-1] *= -1.0
    scaled = progress * (knot_count - 1)
    segment = np.minimum(np.floor(scaled).astype(int), knot_count - 2)
    blend = _smoothstep(scaled - segment).astype(np.float32)
    q0 = torch.from_numpy(knots[segment])
    q1 = torch.from_numpy(knots[segment + 1])
    quat = quat_slerp(q0, q1, torch.from_numpy(blend)).numpy().astype(np.float32)
    return quat, knots


def _primitive_values(count: int, upper: float, rng: np.random.Generator) -> np.ndarray:
    # Stratification covers the full range reproducibly; the final sample is
    # pinned to the requested upper bound so the library exercises it exactly.
    edges = np.linspace(0.25, upper, count + 1)
    values = rng.uniform(edges[:-1], edges[1:])
    values[-1] = upper
    rng.shuffle(values)
    return values


def _arc_parameterize(position: np.ndarray, quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the benchmark SE(3) arc coordinate without changing geometry."""
    translation_step = np.linalg.norm(np.diff(position.astype(np.float64), axis=0), axis=1)
    quat_dot = np.sum(quat[:-1].astype(np.float64) * quat[1:].astype(np.float64), axis=1)
    rotation_step = 2.0 * np.arccos(np.clip(np.abs(quat_dot), 0.0, 1.0))
    step = np.sqrt(translation_step**2 + (LAMBDA_M_PER_RAD * rotation_step) ** 2)
    s_m = np.concatenate(([0.0], np.cumsum(step))).astype(np.float32)
    tangent = np.gradient(position.astype(np.float64), axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    return s_m, tangent.astype(np.float32)


def _make_primitive(
    family: str,
    index: int,
    size: float,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    periodic = family == "circle"
    raw_count = 721 if periodic else 301
    progress = np.linspace(0.0, 1.0, raw_count, dtype=np.float32)
    start_xy = rng.uniform(-1.0, 1.0, size=2)

    if family == "line":
        heading = rng.uniform(-math.pi, math.pi)
        direction = np.array([math.cos(heading), math.sin(heading)])
        xy = start_xy + progress[:, None] * size * direction
        primitive = {
            "line_length_m": float(size),
            "heading_rad": float(heading),
        }
    elif family == "circle":
        phase = rng.uniform(-math.pi, math.pi)
        direction_sign = int(rng.choice([-1, 1]))
        angle = phase + direction_sign * 2.0 * math.pi * progress
        center_xy = start_xy - size * np.array([math.cos(phase), math.sin(phase)])
        xy = center_xy + size * np.stack((np.cos(angle), np.sin(angle)), axis=1)
        primitive = {
            "circle_radius_m": float(size),
            "phase_rad": float(phase),
            "direction": "counterclockwise" if direction_sign > 0 else "clockwise",
            "center_xy_m": center_xy.tolist(),
        }
    else:
        raise ValueError(f"unsupported primitive family: {family}")

    height, height_knots = _smooth_random_scalar(
        rng, progress, 9, Z_BOUNDS_M, periodic=periodic
    )
    quat, orientation_knots = _smooth_random_quaternion(
        rng, progress, 9, periodic=periodic
    )
    position = np.column_stack((xy, height)).astype(np.float32)
    s_m, tangent = _arc_parameterize(position, quat)
    path_length = float(s_m[-1])
    timing = TimingGenerator().generate(
        path_length, f_max=0.10, v_max=0.35, T=8.0, dt=0.02,
        seed=seed + 1_000_000, a_max=0.40,
    )
    p = position
    q = quat
    identifier = f"random-{family}-{index:03d}"
    record = {
        "trajectory_id": identifier,
        "family": family,
        "seed": seed,
        "position_m": p,
        "quaternion_xyzw": q,
        "s_m": s_m,
        "tangent": tangent,
        "time_s": timing.t_grid.numpy().astype(np.float32),
        "s_ref_m": timing.s_of_t.numpy().astype(np.float32),
        "speed_mps": timing.sdot_of_t.numpy().astype(np.float32),
        "path_length_m": path_length,
        "duration_s": float(timing.T),
        "height_control_knots_m": height_knots.tolist(),
        "orientation_control_knots_xyzw": orientation_knots.tolist(),
        **primitive,
    }
    record["content_sha256"] = _array_digest(
        record["s_m"], p, q, record["time_s"], record["s_ref_m"], record["speed_mps"]
    )
    return record


def _generate_grid(
    seed: int,
    levels_a: int,
    levels_b: int,
) -> tuple[list[dict], dict]:
    """Generate one deterministic trajectory for every curriculum cell."""
    curriculum = CurriculumManager(
        num_envs=1,
        device="cpu",
        n_levels_A=levels_a,
        n_levels_B=levels_b,
        seed=seed,
    )
    factory = TrajectoryFactory()
    records = []
    for a in range(levels_a):
        for b in range(levels_b):
            row = a * levels_b + b
            geometry, timing_parameters = curriculum.params_for_cell(a, b)
            geometry_seed = seed + row
            timing_seed = seed + 100_000 + row
            gamma, timing = factory.generate(
                dict(geometry, seed=geometry_seed),
                dict(timing_parameters, seed=timing_seed),
            )
            position = gamma.p.detach().cpu().numpy().astype(np.float32)
            quat = mat_to_quat(gamma.R.detach().cpu()).numpy().astype(np.float32)
            record = {
                "trajectory_id": f"curriculum-a{a}-b{b}",
                "family": "curriculum",
                "cell_A": a,
                "cell_B": b,
                "geometry_seed": geometry_seed,
                "timing_seed": timing_seed,
                "geometry_params": geometry,
                "timing_params": timing_parameters,
                "position_m": position,
                "quaternion_xyzw": quat,
                "s_m": gamma.s_grid.detach().cpu().numpy().astype(np.float32),
                "tangent": gamma.tangent.detach().cpu().numpy().astype(np.float32),
                "time_s": timing.t_grid.detach().cpu().numpy().astype(np.float32),
                "s_ref_m": timing.s_of_t.detach().cpu().numpy().astype(np.float32),
                "speed_mps": timing.sdot_of_t.detach().cpu().numpy().astype(np.float32),
                "path_length_m": float(gamma.L),
                "duration_s": float(timing.T),
            }
            record["content_sha256"] = _array_digest(
                record["s_m"], position, quat, record["time_s"],
                record["s_ref_m"], record["speed_mps"],
            )
            records.append(record)
    return records, {
        "mode": "generated_from_roboduet_source",
        "generator": "modules.trajectory_generator.TrajectoryFactory",
        "curriculum": "modules.curriculum.CurriculumManager.params_for_cell",
        "seed": seed,
        "levels_A": levels_a,
        "levels_B": levels_b,
        "sample_count": len(records),
    }


def _load_grid(source: Path) -> tuple[list[dict], dict]:
    manifest_path = source / "manifest.json"
    archive_path = source / "trajectories.npz"
    manifest = json.loads(manifest_path.read_text())
    records = []
    with np.load(archive_path, allow_pickle=False) as data:
        for sample in manifest["samples"]:
            a, b = int(sample["cell_A"]), int(sample["cell_B"])
            key = f"a{a}_b{b}"
            position = data[f"{key}_position_m"].astype(np.float32)
            rotation = data[f"{key}_rotation_matrix"].astype(np.float32)
            quat = mat_to_quat(torch.from_numpy(rotation)).numpy().astype(np.float32)
            s_m = data[f"{key}_s_m"].astype(np.float32)
            tangent = np.gradient(position, axis=0)
            tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-8)
            record = {
                "trajectory_id": f"curriculum-a{a}-b{b}",
                "family": "curriculum",
                "cell_A": a,
                "cell_B": b,
                "position_m": position,
                "quaternion_xyzw": quat,
                "s_m": s_m,
                "tangent": tangent.astype(np.float32),
                "time_s": data[f"{key}_time_s"].astype(np.float32),
                "s_ref_m": data[f"{key}_s_ref_m"].astype(np.float32),
                "speed_mps": data[f"{key}_speed_mps"].astype(np.float32),
                "path_length_m": float(s_m[-1]),
                "duration_s": float(data[f"{key}_time_s"][-1]),
            }
            record["content_sha256"] = _array_digest(
                record["s_m"], position, quat, record["time_s"],
                record["s_ref_m"], record["speed_mps"],
            )
            records.append(record)
    source_record = {
        "mode": "imported_existing_gallery",
        "path": str(source.resolve()),
        "schema_version": manifest["schema_version"],
        "manifest_sha256": _sha256(manifest_path),
        "trajectories_npz_sha256": _sha256(archive_path),
        "sample_count": len(records),
    }
    return records, source_record


def _pack_archive(records: list[dict], target: Path) -> None:
    count = len(records)
    max_gamma = max(len(record["s_m"]) for record in records)
    max_time = max(len(record["time_s"]) for record in records)
    position = np.zeros((count, max_gamma, 3), dtype=np.float32)
    quaternion = np.zeros((count, max_gamma, 4), dtype=np.float32)
    tangent = np.zeros((count, max_gamma, 3), dtype=np.float32)
    s_m = np.zeros((count, max_gamma), dtype=np.float32)
    time_s = np.zeros((count, max_time), dtype=np.float32)
    s_ref = np.zeros((count, max_time), dtype=np.float32)
    speed = np.zeros((count, max_time), dtype=np.float32)

    for row, record in enumerate(records):
        ng, nt = len(record["s_m"]), len(record["time_s"])
        position[row, :ng] = record["position_m"]
        position[row, ng:] = record["position_m"][-1]
        quaternion[row, :ng] = record["quaternion_xyzw"]
        quaternion[row, ng:] = record["quaternion_xyzw"][-1]
        tangent[row, :ng] = record["tangent"]
        tangent[row, ng:] = record["tangent"][-1]
        s_m[row, :ng] = record["s_m"]
        s_m[row, ng:] = record["s_m"][-1]
        time_s[row, :nt] = record["time_s"]
        time_s[row, nt:] = record["time_s"][-1]
        s_ref[row, :nt] = record["s_ref_m"]
        s_ref[row, nt:] = record["s_ref_m"][-1]
        speed[row, :nt] = record["speed_mps"]

    np.savez_compressed(
        target,
        trajectory_id=np.asarray([record["trajectory_id"] for record in records]),
        family=np.asarray([record["family"] for record in records]),
        gamma_points=np.asarray([len(record["s_m"]) for record in records], dtype=np.int32),
        time_law_points=np.asarray([len(record["time_s"]) for record in records], dtype=np.int32),
        gamma_s=s_m,
        gamma_p=position,
        gamma_quat_xyzw=quaternion,
        gamma_tangent=tangent,
        path_length_m=np.asarray([record["path_length_m"] for record in records], dtype=np.float32),
        tl_t=time_s,
        tl_s=s_ref,
        tl_sdot=speed,
        duration_s=np.asarray([record["duration_s"] for record in records], dtype=np.float32),
    )


def _gallery_offset(record: dict, family_index: int) -> np.ndarray:
    if record["family"] == "curriculum":
        return np.array([record["cell_A"] * 6.0, record["cell_B"] * 6.0, 0.0])
    column, row = family_index % 4, family_index // 4
    base_x = 42.0 if record["family"] == "line" else 98.0
    return np.array([base_x + column * 13.0, row * 13.0, 0.0])


def _load_scale_robot(scene_path: Path) -> tuple[list[dict], dict]:
    """Compile the MJCF and merge visual geoms by material in base coordinates."""
    import mujoco

    scene_path = scene_path.resolve()
    if not scene_path.is_file():
        raise FileNotFoundError(f"Go2+X5 scene not found: {scene_path}")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    if base_id < 0:
        raise ValueError(f"MJCF has no base_link body: {scene_path}")
    base_position = data.xpos[base_id].copy()
    base_rotation = data.xmat[base_id].reshape(3, 3).copy()
    grouped: dict[str, dict] = {}
    visual_geom_count = 0
    for geom_id in range(model.ngeom):
        if (
            int(model.geom_group[geom_id]) != 2
            or int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH)
        ):
            continue
        visual_geom_count += 1
        mesh_id = int(model.geom_dataid[geom_id])
        vertex_start = int(model.mesh_vertadr[mesh_id])
        vertex_count = int(model.mesh_vertnum[mesh_id])
        face_start = int(model.mesh_faceadr[mesh_id])
        face_count = int(model.mesh_facenum[mesh_id])
        vertices = np.asarray(
            model.mesh_vert[vertex_start:vertex_start + vertex_count], dtype=np.float32
        )
        faces = np.asarray(
            model.mesh_face[face_start:face_start + face_count], dtype=np.uint32
        )
        geom_rotation = data.geom_xmat[geom_id].reshape(3, 3)
        world_vertices = vertices @ geom_rotation.T + data.geom_xpos[geom_id]
        base_vertices = (world_vertices - base_position) @ base_rotation
        material_id = int(model.geom_matid[geom_id])
        if material_id >= 0:
            material_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_MATERIAL, material_id
            ) or f"material_{material_id}"
            rgba = np.asarray(model.mat_rgba[material_id], dtype=np.float32)
        else:
            material_name = "geom_rgba"
            rgba = np.asarray(model.geom_rgba[geom_id], dtype=np.float32)
        group = grouped.setdefault(
            material_name,
            {"vertices": [], "faces": [], "rgba": rgba},
        )
        offset = sum(len(value) for value in group["vertices"])
        group["vertices"].append(base_vertices.astype(np.float32))
        group["faces"].append(faces + offset)

    if not grouped:
        raise ValueError(f"MJCF has no group-2 visual mesh geoms: {scene_path}")
    groups = []
    all_vertices = []
    digest = hashlib.sha256()
    for material_name, values in sorted(grouped.items()):
        vertices = np.concatenate(values["vertices"], axis=0)
        faces = np.concatenate(values["faces"], axis=0)
        rgba = np.clip(np.rint(values["rgba"] * 255.0), 0, 255).astype(np.uint8)
        digest.update(material_name.encode("utf-8"))
        digest.update(_array_digest(vertices, faces, rgba).encode("ascii"))
        groups.append({
            "name": material_name,
            "vertices": vertices,
            "faces": faces,
            "rgba": rgba.tolist(),
        })
        all_vertices.append(vertices)
    vertices = np.concatenate(all_vertices, axis=0)
    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    metadata = {
        "model": "Go2+X5",
        "scene": str(scene_path),
        "scene_sha256": _sha256(scene_path),
        "compiled_visual_mesh_sha256": digest.hexdigest(),
        "visual_geom_count": visual_geom_count,
        "material_group_count": len(groups),
        "vertex_count": int(sum(len(group["vertices"]) for group in groups)),
        "triangle_count": int(sum(len(group["faces"]) for group in groups)),
        "bounds_base_frame_m": {
            "min": bounds_min.tolist(),
            "max": bounds_max.tolist(),
        },
        "instance_scale": 1.0,
        "placement": "0.75 m in negative gallery-y from each trajectory start; feet on z=0",
    }
    return groups, metadata


def _write_rrd(
    records: list[dict],
    target: Path,
    robot_mesh_groups: list[dict],
    robot_metadata: dict,
) -> tuple[str, dict]:
    import rerun as rr

    rr.init("roboduet_frozen_trajectory_library", spawn=False)
    rr.save(str(target))
    rr.log("library", rr.ViewCoordinates.RIGHT_HAND_Z_UP, timeless=True)
    colors = {
        "curriculum": [70, 140, 255, 255],
        "line": [255, 160, 40, 255],
        "circle": [80, 220, 120, 255],
    }
    family_counts = {"curriculum": 0, "line": 0, "circle": 0}
    placed = []
    robot_positions = []
    robot_ground_offset = -float(robot_metadata["bounds_base_frame_m"]["min"][2])
    for record in records:
        family = record["family"]
        index = family_counts[family]
        family_counts[family] += 1
        offset = _gallery_offset(record, index)
        path = record["position_m"] + offset
        entity = f"library/{family}/{record['trajectory_id']}"
        rr.log(entity + "/path", rr.LineStrips3D([path], radii=0.018, colors=[colors[family]]), timeless=True)
        rr.log(
            entity + "/label",
            rr.Points3D([path[0]], radii=0.045, colors=[colors[family]],
                        labels=[record["trajectory_id"]], show_labels=True),
            timeless=True,
        )
        stride = max(1, len(path) // 18)
        origins = path[::stride]
        rotation = quat_to_mat(torch.from_numpy(record["quaternion_xyzw"][::stride])).numpy()
        for axis, name, color in (
            (0, "x_axis", [255, 70, 70, 255]),
            (1, "y_axis", [70, 255, 70, 255]),
            (2, "z_axis", [70, 140, 255, 255]),
        ):
            rr.log(
                entity + "/orientation/" + name,
                rr.Arrows3D(origins=origins, vectors=rotation[:, :, axis] * 0.12,
                            radii=0.004, colors=[color]),
                timeless=True,
        )
        placed.append((record, offset, entity))
        robot_positions.append([
            float(path[0, 0]),
            float(path[0, 1] - 0.75),
            robot_ground_offset,
        ])

    # Cover the complete gallery with a thin solid slab whose top surface is
    # exactly z=0. A one-metre grid makes the physical scale easy to inspect.
    gallery_points = np.concatenate(
        [record["position_m"] + offset for record, offset, _ in placed], axis=0
    )
    ground_min = np.floor(gallery_points[:, :2].min(axis=0)) - 2.0
    ground_max = np.ceil(gallery_points[:, :2].max(axis=0)) + 2.0
    ground_center = (ground_min + ground_max) * 0.5
    ground_half_size = (ground_max - ground_min) * 0.5
    rr.log(
        "library/ground/surface",
        rr.Boxes3D(
            centers=[[ground_center[0], ground_center[1], -0.01]],
            half_sizes=[[ground_half_size[0], ground_half_size[1], 0.01]],
            colors=[[72, 74, 78, 255]],
            fill_mode="solid",
        ),
        timeless=True,
    )
    grid_lines = []
    for x in np.arange(ground_min[0], ground_max[0] + 0.5, 1.0):
        grid_lines.append([[x, ground_min[1], 0.001], [x, ground_max[1], 0.001]])
    for y in np.arange(ground_min[1], ground_max[1] + 0.5, 1.0):
        grid_lines.append([[ground_min[0], y, 0.001], [ground_max[0], y, 0.001]])
    rr.log(
        "library/ground/grid_1m",
        rr.LineStrips3D(grid_lines, radii=0.003, colors=[[125, 128, 134, 255]]),
        timeless=True,
    )

    # One mesh payload per material and one pose per trajectory. Mesh3D's
    # visualizer instances the same 1:1 model at every InstancePoses3D pose.
    robot_positions = np.asarray(robot_positions, dtype=np.float32)
    for group in robot_mesh_groups:
        rr.log(
            f"library/scale_reference/go2_x5/{group['name']}",
            rr.Mesh3D(
                vertex_positions=group["vertices"],
                triangle_indices=group["faces"],
                albedo_factor=group["rgba"],
            ),
            rr.InstancePoses3D(translations=robot_positions),
            timeless=True,
        )

    # A normalized 12-second playback keeps this geometry viewer compact even
    # when large circles have much longer bounded physical time laws.
    for frame in range(241):
        phase = frame / 240.0
        rr.set_time_seconds("preview_time", phase * 12.0)
        for record, offset, entity in placed:
            s_value = phase * record["path_length_m"]
            index = int(np.clip(np.searchsorted(record["s_m"], s_value, side="right") - 1,
                                0, len(record["s_m"]) - 2))
            delta = max(float(record["s_m"][index + 1] - record["s_m"][index]), 1e-12)
            fraction = (s_value - float(record["s_m"][index])) / delta
            point = record["position_m"][index] * (1.0 - fraction) + record["position_m"][index + 1] * fraction
            rr.log(entity + "/reference", rr.Points3D([point + offset], radii=0.07,
                                                       colors=[colors[record["family"]]]))
    rr.disconnect()
    ground_metadata = {
        "surface_z_m": 0.0,
        "slab_thickness_m": 0.02,
        "grid_spacing_m": 1.0,
        "xy_bounds_m": {
            "min": ground_min.tolist(),
            "max": ground_max.tolist(),
        },
        "robot_visual_min_z_m": 0.0,
    }
    return str(getattr(rr, "__version__", "unknown")), ground_metadata


def _public_record(record: dict) -> dict:
    excluded = {
        "position_m", "quaternion_xyzw", "s_m", "tangent",
        "time_s", "s_ref_m", "speed_mps",
    }
    output = {key: value for key, value in record.items() if key not in excluded}
    output["gamma_points"] = len(record["s_m"])
    output["time_law_points"] = len(record["time_s"])
    output["min_height_m"] = float(record["position_m"][:, 2].min())
    output["max_height_m"] = float(record["position_m"][:, 2].max())
    return output


def _write_readme(output: Path, grid_mode: str, grid_count: int, line_count: int, circle_count: int) -> None:
    (output / "README.md").write_text(
        f"""# Frozen trajectory library v4

Simulator-free trajectory data for cross-method inspection and later TaskSpec
materialization. This library was built with grid mode `{grid_mode}` and contains
{grid_count} curriculum trajectories, {line_count} seeded random lines and
{circle_count} seeded random circles.

The Rerun gallery places a full-scale static Go2+X5 model beside the start of
every trajectory. The model is compiled from the common benchmark MJCF and
instanced at scale 1.0, so it provides a direct visual reference for distance,
height and workspace scale. A solid ground slab has its top surface at z=0,
with a one-metre grid drawn just above it. The lowest model visual vertex is
placed at z=0 so the robot feet meet the displayed ground.

For the random primitives, the XY projection is exactly a line or a circle.
Height and SO(3) orientation vary smoothly between seeded random control poses
at every path sample. The longest line is {MAX_LINE_LENGTH_M:g} m and the
largest circle radius is {MAX_CIRCLE_RADIUS_M:g} m. These are geometry references only: a benchmark run must still bind a
canonical physical initial state, anchor, deadline and disturbance schedule to
create a TaskSpec.

Open the complete gallery with the matching Rerun 0.19 viewer:

```bash
/opt/miniconda3/envs/isaacgym/bin/rerun \\
  {output}/trajectory_library.rrd
```

`trajectories.npz` is the unified padded archive; use `gamma_points` and
`time_law_points` to slice valid samples. `manifest.json` records all seeds,
bounds, generator parameters, per-trajectory hashes and artifact hashes.

Generate the complete library again from RoboDuet source into a new empty directory:

```bash
cd /home/simon/Projects/WBC/RoboDuet
/opt/miniconda3/envs/isaacgym/bin/python \\
  benchmark/data/build_frozen_trajectory_library.py \\
  --output benchmark/data/frozen_trajectory_library_v4_new
```
"""
    )


def verify(output: Path) -> dict:
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    schema = manifest["schema_version"]
    if schema != SCHEMA and schema not in LEGACY_SCHEMAS:
        raise ValueError("unexpected library schema")
    if schema == SCHEMA:
        scale_model = manifest.get("scale_reference_model", {})
        if scale_model.get("model") != "Go2+X5":
            raise ValueError("missing Go2+X5 scale-reference metadata")
        if float(scale_model.get("instance_scale", 0.0)) != 1.0:
            raise ValueError("Go2+X5 scale-reference model is not full scale")
        for field in (
            "visual_geom_count",
            "material_group_count",
            "vertex_count",
            "triangle_count",
        ):
            if int(scale_model.get(field, 0)) <= 0:
                raise ValueError(f"invalid scale-reference field: {field}")
        ground = manifest.get("rerun", {}).get("ground", {})
        if float(ground.get("surface_z_m", float("nan"))) != 0.0:
            raise ValueError("Rerun ground surface is not at z=0")
        if float(ground.get("robot_visual_min_z_m", float("nan"))) != 0.0:
            raise ValueError("Go2+X5 visual mesh is not grounded at z=0")
    for name, expected in manifest["artifacts"].items():
        path = output / name
        if not path.is_file() or _sha256(path) != expected["sha256"]:
            raise ValueError(f"artifact hash mismatch: {path}")
    with np.load(output / "trajectories.npz", allow_pickle=False) as data:
        families = data["family"].tolist()
        ids = data["trajectory_id"].tolist()
        if len(ids) != len(set(ids)):
            raise ValueError("trajectory IDs are not unique")
        counts = {family: families.count(family) for family in set(families)}
        expected_counts = {key: int(value) for key, value in manifest["family_counts"].items()}
        if counts != expected_counts:
            raise ValueError(f"unexpected family counts: {counts}")
        items = {item["trajectory_id"]: item for item in manifest["trajectories"]}
        lines = [item for item in items.values() if item["family"] == "line"]
        circles = [item for item in items.values() if item["family"] == "circle"]
        max_line_length = float(manifest["constraints"]["line_length_m"]["max"])
        max_circle_radius = float(manifest["constraints"]["circle_radius_m"]["max"])
        z_min = float(manifest["constraints"]["height_m"]["min"])
        z_max = float(manifest["constraints"]["height_m"]["max"])
        if max(item["line_length_m"] for item in lines) != max_line_length:
            raise ValueError(f"line maximum is not exactly {max_line_length:g} m")
        if max(item["circle_radius_m"] for item in circles) != max_circle_radius:
            raise ValueError(f"circle radius maximum is not exactly {max_circle_radius:g} m")
        for row, (identifier, family) in enumerate(zip(ids, families)):
            gamma_count = int(data["gamma_points"][row])
            time_count = int(data["time_law_points"][row])
            position = data["gamma_p"][row, :gamma_count].astype(np.float64)
            quat = data["gamma_quat_xyzw"][row, :gamma_count].astype(np.float64)
            arrays = (
                data["gamma_s"][row, :gamma_count],
                data["gamma_p"][row, :gamma_count],
                data["gamma_quat_xyzw"][row, :gamma_count],
                data["tl_t"][row, :time_count],
                data["tl_s"][row, :time_count],
                data["tl_sdot"][row, :time_count],
            )
            if _array_digest(*arrays) != items[identifier]["content_sha256"]:
                raise ValueError(f"trajectory content hash mismatch: {identifier}")
            if np.max(np.abs(np.linalg.norm(quat, axis=1) - 1.0)) > 1e-5:
                raise ValueError(f"non-unit quaternion: {identifier}")
            if family not in ("line", "circle"):
                continue
            item = items[identifier]
            min_height, max_height = float(position[:, 2].min()), float(position[:, 2].max())
            if not (z_min - 1e-4 <= min_height <= max_height <= z_max + 1e-4):
                raise ValueError(f"height bounds violated by {identifier}")
            if max_height - min_height < 0.05:
                raise ValueError(f"height does not vary enough: {identifier}")
            orientation_span = 2.0 * np.arccos(
                np.clip(np.abs(quat @ quat[0]), 0.0, 1.0)
            ).max()
            if orientation_span < 0.1:
                raise ValueError(f"orientation does not vary enough: {identifier}")
            if family == "line":
                delta = position[-1, :2] - position[0, :2]
                planar_length = float(np.linalg.norm(delta))
                residual = np.abs(np.cross(delta, position[:, :2] - position[0, :2]))
                residual /= max(planar_length, 1e-12)
                if residual.max() > 1e-5 or planar_length > max_line_length + 1e-5:
                    raise ValueError(f"line geometry constraint violated by {identifier}")
            else:
                center = np.asarray(item["center_xy_m"], dtype=np.float64)
                radius = np.linalg.norm(position[:, :2] - center, axis=1)
                if (
                    np.max(np.abs(radius - float(item["circle_radius_m"]))) > 1e-5
                    or radius.max() > max_circle_radius + 1e-5
                ):
                    raise ValueError(f"circle geometry constraint violated by {identifier}")
                quat_closure = min(
                    np.linalg.norm(quat[0] - quat[-1]),
                    np.linalg.norm(quat[0] + quat[-1]),
                )
                if max(np.linalg.norm(position[0] - position[-1]), quat_closure) > 1e-5:
                    raise ValueError(f"circle pose does not close: {identifier}")
    return {"ok": True, "trajectory_count": len(manifest["trajectories"]), "family_counts": counts}


def build(
    output: Path,
    *,
    seed: int,
    grid_seed: int,
    levels_a: int,
    levels_b: int,
    line_count: int,
    circle_count: int,
    import_grid: Path | None,
    robot_scene: Path,
) -> None:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if min(levels_a, levels_b, line_count, circle_count) <= 0:
        raise ValueError("levels and family counts must all be positive")
    output.mkdir(parents=True)

    imported_artifacts = {}
    if import_grid is None:
        records, grid_record = _generate_grid(grid_seed, levels_a, levels_b)
    else:
        source = import_grid.resolve()
        for name in ("manifest.json", "trajectories.npz"):
            if not (source / name).is_file():
                raise FileNotFoundError(source / name)
        records, grid_record = _load_grid(source)
        source_copy = output / "source_grid"
        source_copy.mkdir()
        shutil.copy2(source / "manifest.json", source_copy / "manifest.json")
        shutil.copy2(source / "trajectories.npz", source_copy / "trajectories.npz")
        imported_artifacts = {
            "source_grid/manifest.json": {"sha256": _sha256(source_copy / "manifest.json")},
            "source_grid/trajectories.npz": {"sha256": _sha256(source_copy / "trajectories.npz")},
        }

    grid_count = len(records)
    rng = np.random.default_rng(seed)
    line_lengths = _primitive_values(line_count, MAX_LINE_LENGTH_M, rng)
    circle_radii = _primitive_values(circle_count, MAX_CIRCLE_RADIUS_M, rng)
    for index, length in enumerate(line_lengths):
        records.append(_make_primitive("line", index, float(length), seed + 10_000 + index * 10))
    for index, radius in enumerate(circle_radii):
        records.append(_make_primitive("circle", index, float(radius), seed + 20_000 + index * 10))

    archive_path = output / "trajectories.npz"
    rrd_path = output / "trajectory_library.rrd"
    _pack_archive(records, archive_path)
    robot_mesh_groups, robot_metadata = _load_scale_robot(robot_scene)
    rerun_version, ground_metadata = _write_rrd(
        records, rrd_path, robot_mesh_groups, robot_metadata
    )
    _write_readme(output, grid_record["mode"], grid_count, line_count, circle_count)
    script_hash = _sha256(Path(__file__).resolve())
    manifest = {
        "schema_version": SCHEMA,
        "status": "frozen",
        "seed": seed,
        "generator": {
            "path": str(Path(__file__).resolve()),
            "sha256": script_hash,
        },
        "curriculum_grid": grid_record,
        "scale_reference_model": robot_metadata,
        "sample_count": len(records),
        "family_counts": {
            "curriculum": grid_count,
            "line": line_count,
            "circle": circle_count,
        },
        "constraints": {
            "line_length_m": {"min": 0.25, "max": MAX_LINE_LENGTH_M},
            "circle_radius_m": {"min": 0.25, "max": MAX_CIRCLE_RADIUS_M},
            "height_m": {"min": Z_BOUNDS_M[0], "max": Z_BOUNDS_M[1]},
            "height_sampling": "seeded random control knots with smoothstep interpolation",
            "orientation_sampling": "seeded uniform SO(3) control quaternions with smooth SLERP",
            "circle_closure": "position, height and orientation close at progress 1",
            "se3_arc_lambda_m_per_rad": LAMBDA_M_PER_RAD,
        },
        "archive_schema": {
            "padding": "terminal samples repeated; slice with gamma_points/time_law_points",
            "quaternion_order": "xyzw",
            "height_reference": "geometry-local z; bind an execution anchor when materializing TaskSpec",
        },
        "rerun": {
            "sdk_version": rerun_version,
            "viewer": "/opt/miniconda3/envs/isaacgym/bin/rerun",
            "preview_timeline_s": 12.0,
            "ground": ground_metadata,
        },
        "trajectories": [_public_record(record) for record in records],
        "artifacts": {
            "README.md": {"sha256": _sha256(output / "README.md")},
            "trajectories.npz": {"sha256": _sha256(archive_path)},
            "trajectory_library.rrd": {"sha256": _sha256(rrd_path)},
            **imported_artifacts,
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(verify(output), indent=2))
    print(f"output: {output}")


def main() -> None:
    args = _parse_args()
    if args.verify_only:
        print(json.dumps(verify(args.output.resolve()), indent=2))
    else:
        build(
            args.output,
            seed=args.seed,
            grid_seed=args.grid_seed,
            levels_a=args.levels_a,
            levels_b=args.levels_b,
            line_count=args.line_count,
            circle_count=args.circle_count,
            import_grid=args.import_grid,
            robot_scene=args.robot_scene,
        )


if __name__ == "__main__":
    main()
