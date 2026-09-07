"""Representative trajectory selection and IsaacGym video replay."""

from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np


_FEATURE_KEYS = (
    "span_x_m",
    "span_y_m",
    "span_z_m",
    "path_length_m",
    "curvature_p90_rad_m",
    "rotation_total_rad",
)


def _medoid(tasks: Sequence[dict]) -> dict:
    values = np.asarray(
        [[float(task[key]) for key in _FEATURE_KEYS] for task in tasks],
        dtype=np.float64,
    )
    center = np.median(values, axis=0)
    scale = np.ptp(values, axis=0)
    scale[scale < 1e-9] = 1.0
    distance = np.square((values - center) / scale).sum(axis=1)
    best = min(
        range(len(tasks)), key=lambda index: (distance[index], tasks[index]["bank_row"])
    )
    return dict(tasks[best])


def select_representative_tasks(
    manifest: dict, count: int = 6, bank_rows: Optional[Sequence[int]] = None
) -> List[dict]:
    """Choose deterministic, policy-independent representative trajectories."""
    entries = list(manifest.get("trajectories", []))
    if not entries or count <= 0:
        return []
    by_row = {int(task["bank_row"]): task for task in entries}
    if bank_rows:
        missing = [row for row in bank_rows if int(row) not in by_row]
        if missing:
            raise ValueError(
                f"Representative bank rows are outside the suite: {missing}"
            )
        selected = [dict(by_row[int(row)]) for row in bank_rows]
        if len({task["bank_row"] for task in selected}) != len(selected):
            raise ValueError("--video_bank_rows must not contain duplicates")
        return selected
    cells = sorted({(task["cell_A"], task["cell_B"]) for task in entries})
    diagonal = [cell for cell in cells if cell[0] == cell[1]]
    candidates = diagonal or cells
    take = min(int(count), len(candidates))
    positions = np.linspace(0, len(candidates) - 1, take).round().astype(int)
    selected = []
    for index in positions:
        cell = candidates[index]
        selected.append(
            _medoid(
                [task for task in entries if (task["cell_A"], task["cell_B"]) == cell]
            )
        )
    return selected


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "policy"


def _write_mp4(frames, output: Path, fps: float) -> Path:
    import imageio.v2 as imageio

    if not frames:
        raise RuntimeError(f"No camera frames were captured for {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(output),
        fps=float(fps),
        codec="libx264",
        quality=7,
        macro_block_size=None,
        ffmpeg_log_level="error",
    )
    try:
        for frame in frames:
            array = np.asarray(frame, dtype=np.uint8)
            writer.append_data(array[..., :3])
    finally:
        writer.close()
    poster = output.with_suffix(".jpg")
    imageio.imwrite(str(poster), np.asarray(frames[0], dtype=np.uint8)[..., :3])
    return poster


def record_representative_videos(
    logdirs: Sequence[str],
    names: Sequence[str],
    ckptids: Sequence[str],
    tasks: Sequence[dict],
    run_dir: str,
    device: str = "cuda:0",
    robot: Optional[str] = None,
    bank_seed: int = 12345,
    bank_per_cell: int = 8,
    n_steps: int = 500,
    settle_steps: int = 20,
) -> List[dict]:
    """Replay selected tasks one at a time and encode one video per method/task."""
    if not tasks:
        return []
    import torch

    from benchmark.wbc.evaluation import (
        WBCPolicyHandle,
        _wbc_step_all,
        load_wbc_env_benchmark,
        load_wbc_policies,
    )
    from benchmark.wbc.scenarios import _load_paired_tasks
    from go1_gym.utils.global_switch import global_switch

    output_root = Path(run_dir)
    env, cfg = load_wbc_env_benchmark(
        logdirs[0],
        1,
        1,
        headless=True,
        device=device,
        robot=robot,
        bank_seed=bank_seed,
        bank_per_cell=bank_per_cell,
        record_video=True,
    )
    base = env.env
    global_switch.open_switch()
    all_ids = torch.arange(base.num_envs, device=base.device)
    fps = 1.0 / (
        float(base.dt) * max(1, int(getattr(cfg.env, "recording_frame_stride", 1)))
    )
    records = []
    try:
        for logdir, name, ckptid in zip(logdirs, names, ckptids):
            dog_policy, arm_policy = load_wbc_policies(
                logdir, ckptid, cfg, device=device
            )
            handle = WBCPolicyHandle(name, dog_policy, arm_policy, 0, 1)
            for task in tasks:
                replay_seed = (int(bank_seed) * 1009 + int(task["bank_row"])) % (2**31)
                random.seed(replay_seed)
                np.random.seed(replay_seed)
                torch.manual_seed(replay_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(replay_seed)
                print(
                    f"  [video] {name}: cell ({task['cell_A']}, {task['cell_B']}) row {task['bank_row']}",
                    flush=True,
                )
                env.reset()
                for _ in range(settle_steps):
                    _wbc_step_all(env, [handle])
                _load_paired_tasks(base, [handle], [task])
                env.clear_cached(all_ids)
                base._recording_ee_trace = []
                base.video_frames = []
                base.complete_video_frames = []
                base.record_now = True
                video_done = False
                video_timed_out = False
                video_early_term = False
                replay_steps = 0
                for replay_steps in range(1, n_steps + 1):
                    done, timed_out, early_term = _wbc_step_all(env, [handle])
                    if bool(done[0].item()):
                        video_done = True
                        video_timed_out = bool(timed_out[0].item())
                        video_early_term = bool(early_term[0].item())
                        break
                frames = list(base.complete_video_frames or base.video_frames)
                captured_frames = len(frames)
                minimum_frames = max(1, int(round(fps * 2.0)))
                if frames and len(frames) < minimum_frames:
                    frames.extend([frames[-1]] * (minimum_frames - len(frames)))
                base.pause_recording()
                stem = f"A{task['cell_A']}_B{task['cell_B']}_row{task['bank_row']:06d}"
                video_path = output_root / "videos" / _safe_name(name) / f"{stem}.mp4"
                poster_path = _write_mp4(frames, video_path, fps)
                records.append(
                    {
                        "run_name": name,
                        "trajectory_id": task["trajectory_id"],
                        "bank_row": task["bank_row"],
                        "cell_A": task["cell_A"],
                        "cell_B": task["cell_B"],
                        "dominant_axis": task["dominant_axis"],
                        "span_x_m": task["span_x_m"],
                        "span_y_m": task["span_y_m"],
                        "span_z_m": task["span_z_m"],
                        "path_length_m": task["path_length_m"],
                        "curvature_p90_rad_m": task["curvature_p90_rad_m"],
                        "video_path": os.path.relpath(video_path, output_root),
                        "poster_path": os.path.relpath(poster_path, output_root),
                        "n_frames": len(frames),
                        "captured_frames": captured_frames,
                        "replay_steps": replay_steps,
                        "video_terminated": video_done,
                        "video_timed_out": video_timed_out,
                        "video_traj_early_term": video_early_term,
                        "video_fall": video_done
                        and not video_timed_out
                        and not video_early_term,
                        "fps": fps,
                        "replay_seed": replay_seed,
                    }
                )
            del dog_policy, arm_policy
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        base.close()
        del env
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return records
