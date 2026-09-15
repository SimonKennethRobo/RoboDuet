"""Copy one frozen trajectory into a new full-duration TaskSpec suite."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmark.wbc.suite import finalize_task_spec, refresh_suite_hash


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_task(source, task_id: str, output, deadline_s: float) -> tuple[Path, str]:
    source = Path(source).resolve()
    output = Path(output).resolve()
    payload = json.loads(source.read_text())
    wrappers = payload if isinstance(payload, list) else [payload]
    matches = []
    for wrapper in wrappers:
        group = wrapper.get("suite", wrapper)
        for row, task in enumerate(group.get("trajectories", [])):
            if task.get("task_id") == task_id:
                matches.append((wrapper, group, row, task))
    if len(matches) != 1:
        raise ValueError(f"expected one source task {task_id!r}, found {len(matches)}")
    wrapper, source_group, row, source_task = matches[0]
    group = copy.deepcopy(source_group)
    task = copy.deepcopy(source_task)
    anchor = task.get("anchor_env_local_xyz_m")
    if anchor is None:
        anchor = [*task["anchor_env_local_xy_m"], 0.0]
    finalize_task_spec(
        task,
        initial_state=task["initial_state"],
        anchor_env_local=anchor,
        orientation_left_multiplier_xyzw=task.get(
            "orientation_left_multiplier_xyzw", (0.0, 0.0, 0.0, 1.0),
        ),
        deadline_s=deadline_s,
        disturbance_schedule=task.get("disturbance_schedule", []),
    )
    group["trajectories"] = [task]
    source_record = wrapper.get("reference_archive", source_group["reference_archive"])
    archive_source = (source.parent / source_record["path"]).resolve()
    if _sha256(archive_source) != source_record["sha256"]:
        raise ValueError(f"source reference hash mismatch: {archive_source}")
    with np.load(archive_source, allow_pickle=False) as archive:
        old_ids = [str(value) for value in archive["task_id"].tolist()]
        archive_row = old_ids.index(source_task["task_id"])
        arrays = {}
        for name in archive.files:
            value = archive[name]
            arrays[name] = (
                value[archive_row:archive_row + 1]
                if value.ndim and value.shape[0] == len(old_ids) else value
            )
        arrays["task_id"] = np.asarray([task["task_id"]])
    output.mkdir(parents=True, exist_ok=False)
    archive_out = output / "reference.npz"
    np.savez_compressed(archive_out, **arrays)
    record = {
        "path": archive_out.name,
        "format": "numpy_npz_v1",
        "sha256": _sha256(archive_out),
        "quaternion_order": source_record.get("quaternion_order", "xyzw"),
        "position_z_reference": source_record.get(
            "position_z_reference", "task_anchor_environment_local_translation",
        ),
    }
    group["reference_archive"] = record
    refresh_suite_hash(group)
    result = [{
        "suite": group,
        "reference_archive": record,
        "robustness_scenario": "nominal",
        "derived_from": {
            "source_suite": str(source),
            "source_suite_sha256": _sha256(source),
            "source_task_id": source_task["task_id"],
            "source_task_spec_sha256": source_task["task_spec_sha256"],
            "operation": "deadline_only",
        },
    }]
    suite_out = output / "trajectory_suite.json"
    suite_out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return suite_out, task["task_id"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--deadline-s", required=True, type=float)
    args = parser.parse_args(argv)
    if args.deadline_s <= 0:
        parser.error("--deadline-s must be positive")
    suite, task_id = freeze_task(args.source, args.task_id, args.output, args.deadline_s)
    print(json.dumps({"suite": str(suite), "task_id": task_id}, indent=2))


if __name__ == "__main__":
    main()
