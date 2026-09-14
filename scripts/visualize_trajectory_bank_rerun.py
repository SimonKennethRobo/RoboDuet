#!/usr/bin/env python3
"""Generate a curriculum-grid trajectory sample and record it with Rerun.

The script is simulator-free: it uses the same ``TrajectoryFactory`` and
``CurriculumManager.params_for_cell`` as the live WBC trajectory bank, then
writes the generated tensors, a JSON manifest, and a self-contained ``.rrd``.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from modules.curriculum import CurriculumManager
from modules.trajectory_generator import TrajectoryFactory


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: benchmark/results/trajectory_samples/<timestamp>).",
    )
    parser.add_argument("--seed", type=int, default=12345, help="Base trajectory seed.")
    parser.add_argument("--levels-a", type=int, default=6, help="Geometry difficulty levels.")
    parser.add_argument("--levels-b", type=int, default=6, help="Timing difficulty levels.")
    parser.add_argument("--spacing", type=float, default=6.0, help="Gallery cell spacing in metres.")
    parser.add_argument("--playback-hz", type=float, default=20.0, help="Animated reference-point rate.")
    parser.add_argument("--orientation-stride", type=int, default=24, help="Path samples between pose triads.")
    parser.add_argument("--spawn", action="store_true", help="Open the Rerun viewer after recording.")
    return parser.parse_args()


def _color(a: int, b: int, n_a: int, n_b: int) -> list[int]:
    """Blue-to-red geometry tint, dark-to-bright timing tint."""
    ga = a / max(1, n_a - 1)
    tb = b / max(1, n_b - 1)
    brightness = 0.55 + 0.45 * tb
    return [
        int(255 * brightness * ga),
        int(120 * brightness),
        int(255 * brightness * (1.0 - ga)),
        255,
    ]


def main() -> None:
    args = _parse_args()
    if args.levels_a <= 0 or args.levels_b <= 0:
        raise ValueError("--levels-a and --levels-b must be positive")
    if args.playback_hz <= 0:
        raise ValueError("--playback-hz must be positive")

    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("benchmark/results/trajectory_samples") / stamp
    output_dir.mkdir(parents=True, exist_ok=False)

    curriculum = CurriculumManager(
        num_envs=1,
        device="cpu",
        n_levels_A=args.levels_a,
        n_levels_B=args.levels_b,
        seed=args.seed,
    )
    factory = TrajectoryFactory()
    samples = []
    arrays: dict[str, np.ndarray] = {}

    for a in range(args.levels_a):
        for b in range(args.levels_b):
            row = a * args.levels_b + b
            geom, timing = curriculum.params_for_cell(a, b)
            geom_seed = args.seed + row
            timing_seed = args.seed + 100000 + row
            gamma, time_law = factory.generate(
                dict(geom, seed=geom_seed),
                dict(timing, seed=timing_seed),
            )
            key = f"a{a}_b{b}"
            p = gamma.p.detach().cpu()
            q = gamma.R.detach().cpu()
            xy_delta = p[-1, :2] - p[0, :2]
            xy_span = p[:, :2].max(dim=0).values - p[:, :2].min(dim=0).values
            ds = (gamma.s_grid[1:] - gamma.s_grid[:-1]).clamp_min(1e-6)
            tangent_dot = (gamma.tangent[:-1] * gamma.tangent[1:]).sum(dim=-1).clamp(-1.0, 1.0)
            curvature = torch.acos(tangent_dot) / ds
            arrays[f"{key}_s_m"] = gamma.s_grid.detach().cpu().numpy()
            arrays[f"{key}_position_m"] = p.numpy()
            arrays[f"{key}_rotation_matrix"] = q.numpy()
            arrays[f"{key}_time_s"] = time_law.t_grid.detach().cpu().numpy()
            arrays[f"{key}_s_ref_m"] = time_law.s_of_t.detach().cpu().numpy()
            arrays[f"{key}_speed_mps"] = time_law.sdot_of_t.detach().cpu().numpy()
            samples.append(
                {
                    "cell_A": a,
                    "cell_B": b,
                    "geometry_seed": geom_seed,
                    "timing_seed": timing_seed,
                    "path_length_m": gamma.L,
                    "duration_s": time_law.T,
                    "xy_displacement_m": float(torch.linalg.vector_norm(xy_delta).item()),
                    "span_x_m": float(xy_span[0].item()),
                    "span_y_m": float(xy_span[1].item()),
                    "min_z_m": float(p[:, 2].min().item()),
                    "max_z_m": float(p[:, 2].max().item()),
                    "curvature_p99_rad_m": float(torch.quantile(curvature, 0.99).item()),
                    "curvature_max_rad_m": float(curvature.max().item()),
                    "geometry_params": geom,
                    "timing_params": timing,
                    "gamma_points": int(p.shape[0]),
                    "time_law_points": int(time_law.t_grid.shape[0]),
                    "gamma": gamma,
                    "time_law": time_law,
                }
            )

    npz_path = output_dir / "trajectories.npz"
    np.savez_compressed(npz_path, **arrays)

    manifest = {
        "schema_version": "roboduet-trajectory-gallery-v1",
        "generator": "modules.trajectory_generator.TrajectoryFactory",
        "curriculum": "modules.curriculum.CurriculumManager.params_for_cell",
        "seed": args.seed,
        "levels_A": args.levels_a,
        "levels_B": args.levels_b,
        "sample_count": len(samples),
        "gallery_spacing_m": args.spacing,
        "playback_hz": args.playback_hz,
        "height_reference": "terrain_surface_at_reference_xy",
        "samples": [
            {key: value for key, value in sample.items() if key not in ("gamma", "time_law")}
            for sample in samples
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    import rerun as rr

    rrd_path = output_dir / "trajectory_gallery.rrd"
    rr.init("roboduet_trajectory_gallery", spawn=False)
    if args.spawn:
        rr.spawn()
    rr.save(str(rrd_path))
    rr.log("gallery", rr.ViewCoordinates.RIGHT_HAND_Z_UP, timeless=True)

    try:
        import rerun.blueprint as rrb

        rr.send_blueprint(
            rrb.Blueprint(
                rrb.Spatial3DView(origin="gallery", name="WBC trajectory curriculum 6x6"),
                collapse_panels=True,
            )
        )
    except Exception as exc:
        print(f"[rerun] blueprint skipped: {exc}")

    axis_colors = ([255, 70, 70, 255], [70, 255, 70, 255], [70, 140, 255, 255])
    for sample in samples:
        a, b = sample["cell_A"], sample["cell_B"]
        gamma = sample["gamma"]
        offset = torch.tensor([a * args.spacing, b * args.spacing, 0.0])
        path = gamma.p.detach().cpu() + offset
        color = _color(a, b, args.levels_a, args.levels_b)
        entity = f"gallery/cell_A{a}/cell_B{b}"
        rr.log(
            f"{entity}/path",
            rr.LineStrips3D([path.numpy()], radii=0.012, colors=[color]),
            timeless=True,
        )
        rr.log(
            f"{entity}/label",
            rr.Points3D(
                [path[0].numpy()],
                radii=0.04,
                colors=[color],
                labels=[
                    f"A={a} B={b}  L={gamma.L:.2f}m  "
                    f"z=[{sample['min_z_m']:.2f},{sample['max_z_m']:.2f}]m  "
                    f"k99={sample['curvature_p99_rad_m']:.1f}/m"
                ],
                show_labels=True,
            ),
            timeless=True,
        )
        margin = 0.10
        x0, y0 = (path[:, :2].min(dim=0).values - margin).tolist()
        x1, y1 = (path[:, :2].max(dim=0).values + margin).tolist()
        ground_outline = torch.tensor(
            [[x0, y0, 0.0], [x1, y0, 0.0], [x1, y1, 0.0], [x0, y1, 0.0], [x0, y0, 0.0]]
        )
        rr.log(
            f"{entity}/ground_reference",
            rr.LineStrips3D([ground_outline.numpy()], radii=0.004, colors=[[90, 90, 90, 180]]),
            timeless=True,
        )

        stride = max(1, args.orientation_stride)
        idx = torch.arange(0, path.shape[0], stride)
        origins = path[idx]
        rotations = gamma.R.detach().cpu()[idx]
        axis_length = 0.10
        for axis, name in enumerate(("x_axis", "y_axis", "z_axis")):
            vectors = rotations[:, :, axis] * axis_length
            rr.log(
                f"{entity}/orientation/{name}",
                rr.Arrows3D(
                    origins=origins.numpy(),
                    vectors=vectors.numpy(),
                    radii=0.004,
                    colors=[axis_colors[axis]],
                ),
                timeless=True,
            )

    max_duration = max(float(sample["time_law"].T) for sample in samples)
    frame_count = int(math.floor(max_duration * args.playback_hz)) + 1
    for frame in range(frame_count):
        t = min(max_duration, frame / args.playback_hz)
        rr.set_time_seconds("time", t)
        for sample in samples:
            a, b = sample["cell_A"], sample["cell_B"]
            gamma = sample["gamma"]
            time_law = sample["time_law"]
            query_t = torch.tensor([min(t, float(time_law.T))])
            s_ref = time_law.s_ref(query_t)
            pos = gamma.p_at(s_ref)[0] + torch.tensor([a * args.spacing, b * args.spacing, 0.0])
            color = _color(a, b, args.levels_a, args.levels_b)
            rr.log(
                f"gallery/cell_A{a}/cell_B{b}/reference",
                rr.Points3D([pos.numpy()], radii=0.055, colors=[color]),
            )

    print(f"generated {len(samples)} trajectories")
    print(f"rrd: {rrd_path.resolve()}")
    print(f"data: {npz_path.resolve()}")
    print(f"manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
