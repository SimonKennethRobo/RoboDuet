"""Render comparable MuJoCo replay videos and trajectory-tracking plots."""

from __future__ import annotations

import hashlib
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw


JOINT_NAMES = [
    f"{leg}_{part}_joint" for leg in ("FL", "FR", "RL", "RR")
    for part in ("hip", "thigh", "calf")
] + [f"x5_joint{i}" for i in range(1, 7)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_replay_trace(trace_path: Path) -> dict[str, np.ndarray | float]:
    """Load the single-environment state needed for deterministic replay."""
    with np.load(trace_path, allow_pickle=False) as source:
        required = (
            "base_root_state", "dof_position_rad", "actual_ee_state",
            "reference_ee_position_m", "reference_time_s",
        )
        missing = [name for name in required if name not in source.files]
        if missing:
            raise ValueError(f"trace lacks replay fields: {missing}")
        arrays = {name: np.asarray(source[name]) for name in required}
        control_dt = float(np.asarray(source["control_dt_s"]).item())
    for name, value in arrays.items():
        if value.ndim >= 2 and value.shape[1] == 1:
            arrays[name] = value[:, 0]
    count = len(arrays["reference_time_s"])
    if count == 0 or any(len(value) != count for value in arrays.values()):
        raise ValueError("trace replay fields must have the same nonzero sample count")
    if arrays["base_root_state"].shape[1] != 13:
        raise ValueError("base_root_state must contain xyz, xyzw, and six velocities")
    if arrays["dof_position_rad"].shape[1] < len(JOINT_NAMES):
        raise ValueError("trace must contain all 18 Go2+X5 joint positions")
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("trace replay fields contain non-finite values")
    return {**arrays, "control_dt_s": control_dt}


def video_sample_indices(sample_count: int, control_dt_s: float, fps: int) -> np.ndarray:
    if sample_count < 1 or control_dt_s <= 0 or fps < 1:
        raise ValueError("sample_count, control_dt_s, and fps must be positive")
    duration = (sample_count - 1) * control_dt_s
    frame_count = max(1, int(np.floor(duration * fps)) + 1)
    return np.minimum(
        np.rint(np.arange(frame_count) / (fps * control_dt_s)).astype(int),
        sample_count - 1,
    )


def _tracking_plot(trace: dict, output: Path, method: str, scenario: str) -> None:
    reference = trace["reference_ee_position_m"]
    actual = trace["actual_ee_state"][:, :3]
    time_s = trace["reference_time_s"].reshape(-1)
    error = np.linalg.norm(actual - reference, axis=1)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    axes[0, 0].plot(reference[:, 0], reference[:, 1], "--", label="reference")
    axes[0, 0].plot(actual[:, 0], actual[:, 1], label="actual")
    axes[0, 0].set(xlabel="x [m]", ylabel="y [m]", title="EE path: top view")
    axes[0, 0].axis("equal"); axes[0, 0].grid(True); axes[0, 0].legend()
    axes[0, 1].plot(reference[:, 0], reference[:, 2], "--", label="reference")
    axes[0, 1].plot(actual[:, 0], actual[:, 2], label="actual")
    axes[0, 1].set(xlabel="x [m]", ylabel="z [m]", title="EE path: side view")
    axes[0, 1].grid(True); axes[0, 1].legend()
    for axis, label in enumerate("xyz"):
        axes[1, 0].plot(time_s, reference[:, axis], "--", alpha=.8)
        axes[1, 0].plot(time_s, actual[:, axis], label=label)
    axes[1, 0].set(xlabel="reference time [s]", ylabel="position [m]", title="EE position vs time")
    axes[1, 0].grid(True); axes[1, 0].legend()
    axes[1, 1].plot(time_s, error, color="tab:red")
    axes[1, 1].set(xlabel="reference time [s]", ylabel="error norm [m]", title="EE position tracking error")
    axes[1, 1].grid(True)
    fig.suptitle(f"{method} / {scenario} | RMSE {np.sqrt(np.mean(error**2)):.3f} m")
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _add_sphere(scene, position, radius, rgba) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom, mujoco.mjtGeom.mjGEOM_SPHERE,
        np.asarray([radius, radius, radius]), np.asarray(position),
        np.eye(3).reshape(-1), np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def _add_path(scene, points, rgba, max_points=90) -> None:
    stride = max(1, int(np.ceil(len(points) / max_points)))
    for point in points[::stride]:
        _add_sphere(scene, point, 0.008, rgba)


def _replay_video(trace: dict, scene_path: Path, output: Path, method: str,
                  scenario: str, fps: int, width: int, height: int) -> None:
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    qpos_addresses = [int(model.jnt_qposadr[model.joint(name).id]) for name in JOINT_NAMES]
    indices = video_sample_indices(
        len(trace["reference_time_s"]), trace["control_dt_s"], fps,
    )
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 2.2
    camera.azimuth = 135
    camera.elevation = -20
    renderer = mujoco.Renderer(model, height=height, width=width, max_geom=500)
    reference = trace["reference_ee_position_m"]
    actual = trace["actual_ee_state"][:, :3]
    try:
        with imageio.get_writer(
            output, fps=fps, codec="libx264", quality=8,
            pixelformat="yuv420p", macro_block_size=None,
        ) as writer:
            for sample in indices:
                root = trace["base_root_state"][sample]
                data.qpos[:3] = root[:3]
                data.qpos[3:7] = np.roll(root[3:7], 1)
                data.qpos[qpos_addresses] = trace["dof_position_rad"][sample, :18]
                mujoco.mj_forward(model, data)
                camera.lookat[:] = root[:3] + np.asarray([0.2, 0.0, 0.25])
                renderer.update_scene(data, camera=camera)
                _add_path(renderer.scene, reference, [1.0, .75, .05, .75])
                _add_path(renderer.scene, actual[:sample + 1], [.05, .85, 1.0, .9])
                _add_sphere(renderer.scene, reference[sample], .025, [1.0, .75, .05, 1.0])
                _add_sphere(renderer.scene, actual[sample], .022, [.05, .85, 1.0, 1.0])
                frame = renderer.render()
                image = Image.fromarray(frame)
                draw = ImageDraw.Draw(image)
                error = float(np.linalg.norm(actual[sample] - reference[sample]))
                lines = [
                    f"{method} | {scenario}",
                    f"t={float(trace['reference_time_s'][sample]):.2f}s  EE error={error:.3f}m",
                    "reference: yellow   actual: cyan",
                ]
                draw.rounded_rectangle((12, 12, 430, 83), radius=8, fill=(0, 0, 0, 175))
                draw.multiline_text((24, 21), "\n".join(lines), fill="white", spacing=4)
                writer.append_data(np.asarray(image))
    finally:
        renderer.close()


def render_trace_artifacts(trace_path, scene_path, output_dir, *, method: str,
                           scenario: str, fps: int = 25, width: int = 960,
                           height: int = 540) -> dict:
    """Create a synchronized MP4 and a static tracking diagnostic from a trace."""
    trace_path = Path(trace_path).resolve()
    scene_path = Path(scene_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trace = load_replay_trace(trace_path)
    plot_path = output_dir / "trajectory_tracking.png"
    video_path = output_dir / "mujoco_tracking.mp4"
    _tracking_plot(trace, plot_path, method, scenario)
    _replay_video(trace, scene_path, video_path, method, scenario, fps, width, height)
    return {
        "schema_version": "mujoco-trace-replay-artifacts-v1",
        "replay_semantics": "recorded_state_trace_replayed_in_hashed_common_mujoco_scene",
        "source_trace": {"path": str(trace_path), "sha256": _sha256(trace_path)},
        "video": {"path": str(video_path), "sha256": _sha256(video_path),
                  "fps": fps, "width": width, "height": height},
        "trajectory_plot": {"path": str(plot_path), "sha256": _sha256(plot_path)},
    }
