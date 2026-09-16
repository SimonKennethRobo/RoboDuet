"""Read-only validation and aggregation for a distributed Formal MuJoCo campaign."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmark.wbc.formal_mujoco import _load_tasks, _sha256, _write_json


CONTRACT_KEYS = ("suite_sha256", "methods", "scenarios", "environment", "roboduet")


def aggregate(campaign_root: Path) -> dict:
    root = campaign_root.resolve()
    manifest = json.loads((root / "campaign_manifest.json").read_text())
    suite_path = Path(manifest["suite"])
    suite, tasks = _load_tasks(suite_path)
    if _sha256(suite_path) != manifest["suite_file_sha256"] or suite["suite_sha256"] != manifest["suite_sha256"]:
        raise ValueError("frozen suite no longer matches the campaign manifest")
    scene = Path(manifest["environment"]["scene"]["path"])
    if _sha256(scene) != manifest["environment"]["scene"]["sha256"]:
        raise ValueError("scene no longer matches the campaign manifest")

    state_paths = sorted((root / "nodes").glob("*/formal_state.json"))
    if not state_paths:
        raise ValueError("campaign has no node states")
    rows = []
    seen_shards = set()
    shard_count = None
    for path in state_paths:
        state = json.loads(path.read_text())
        for key in CONTRACT_KEYS:
            if state.get(key) != manifest.get(key):
                raise ValueError(f"{path}: campaign contract mismatch for {key}")
        shard = state["shard"]
        shard_count = shard["count"] if shard_count is None else shard_count
        if shard["count"] != shard_count or shard["index"] in seen_shards:
            raise ValueError(f"{path}: inconsistent or duplicate shard")
        seen_shards.add(shard["index"])
        node_rows = json.loads((path.parent / "formal_results.json").read_text())["rows"]
        for row in node_rows:
            rows.append({**row, "node": state["node_name"], "shard_index": shard["index"]})

    if seen_shards != set(range(shard_count)):
        raise ValueError(f"missing shards: {sorted(set(range(shard_count)) - seen_shards)}")
    expected = {
        (method, task["task_id"], scenario)
        for task in tasks for method in manifest["methods"] for scenario in manifest["scenarios"]
    }
    actual = [(row["method"], row["task_id"], row["scenario"]) for row in rows]
    duplicates = sorted({key for key in actual if actual.count(key) > 1})
    missing = sorted(expected - set(actual))
    unexpected = sorted(set(actual) - expected)
    if duplicates or missing or unexpected:
        raise ValueError(
            f"scenario key validation failed: duplicates={len(duplicates)}, "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )

    task_metadata = {task["task_id"]: task for task in tasks}
    for row in rows:
        task = task_metadata[row["task_id"]]
        row["difficulty"] = {
            "family": task.get("family"), "cell_A": task.get("cell_A"), "cell_B": task.get("cell_B"),
        }
    by_method = {}
    by_method_and_difficulty = {}
    numeric = defaultdict(lambda: defaultdict(list))
    for row in rows:
        method = row["method"]
        result = row.get("result") or {}
        for key, value in result.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value):
                numeric[method][key].append(float(value))
    for method in manifest["methods"]:
        selected = [row for row in rows if row["method"] == method]
        def count_flag(name):
            return sum(bool((row.get("result") or {}).get(name)) for row in selected)
        by_method[method] = {
            "scenario_count": len(selected),
            "complete_count": sum(row["job_status"] == "complete" for row in selected),
            "receipt_complete_count": sum(row.get("receipt_status") == "complete" for row in selected),
            "strict_success_count": sum(bool((row.get("result") or {}).get("success")) for row in selected),
            "fall_count": count_flag("fall"), "timeout_count": count_flag("timed_out"),
            "numerical_fault_count": count_flag("numerical_fault"),
            "metric_means": {key: float(np.mean(values)) for key, values in sorted(numeric[method].items())},
        }
        cells = sorted({(row["difficulty"]["cell_A"], row["difficulty"]["cell_B"]) for row in selected})
        for cell_a, cell_b in cells:
            cell_rows = [row for row in selected if (row["difficulty"]["cell_A"], row["difficulty"]["cell_B"]) == (cell_a, cell_b)]
            cell_numeric = defaultdict(list)
            for row in cell_rows:
                for key, value in (row.get("result") or {}).items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value):
                        cell_numeric[key].append(float(value))
            by_method_and_difficulty[f"{method}/A{cell_a}/B{cell_b}"] = {
                "scenario_count": len(cell_rows),
                "strict_success_count": sum(bool((row.get("result") or {}).get("success")) for row in cell_rows),
                "failure_count": sum(row["job_status"] != "complete" or row.get("receipt_status") != "complete" for row in cell_rows),
                "metric_means": {key: float(np.mean(values)) for key, values in sorted(cell_numeric.items())},
            }
    failures = [row for row in rows if row["job_status"] != "complete" or row.get("receipt_status") != "complete"]
    return {
        "schema_version": "formal-mujoco-distributed-results-v1",
        "campaign_root": str(root), "campaign_manifest_sha256": _sha256(root / "campaign_manifest.json"),
        "scenario_count": len(rows), "expected_scenario_count": len(expected),
        "job_count": len(rows) // len(manifest["scenarios"]),
        "failure_count": len(failures), "failures": failures,
        "by_method": by_method, "by_method_and_difficulty": by_method_and_difficulty,
        "rows": rows,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = aggregate(args.campaign_root)
    output = args.output or args.campaign_root / "formal_results.json"
    _write_json(output.resolve(), result)
    print(json.dumps({key: result[key] for key in (
        "campaign_root", "scenario_count", "expected_scenario_count", "failure_count",
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
