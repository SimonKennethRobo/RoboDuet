import json

import pytest

from benchmark.wbc import aggregate_formal_mujoco as aggregation


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _campaign(tmp_path, *, omit_key=None, duplicate=False):
    suite_path = tmp_path / "suite.json"
    suite_path.write_text("{}")
    contract = {
        "suite": str(suite_path), "suite_file_sha256": "hash",
        "suite_sha256": "suite", "task_count": 2,
        "methods": ["a", "b"], "scenarios": ["nominal", "push"],
        "environment": {"scene": {"path": str(tmp_path / "scene.xml"), "sha256": "hash"}},
        "roboduet": {"ocs2_task_sha256": "task"},
    }
    (tmp_path / "scene.xml").write_text("scene")
    _write(tmp_path / "campaign_manifest.json", contract)
    tasks = [{"task_id": "T0"}, {"task_id": "T1"}]
    for shard in range(2):
        node = tmp_path / "nodes" / f"n{shard}"
        state = {**contract, "node_name": f"n{shard}", "shard": {"count": 2, "index": shard}}
        _write(node / "formal_state.json", state)
        rows = []
        for method in contract["methods"]:
            for scenario in contract["scenarios"]:
                key = (method, tasks[shard]["task_id"], scenario)
                if key == omit_key:
                    continue
                rows.append({
                    "method": method, "task_id": tasks[shard]["task_id"], "scenario": scenario,
                    "job_status": "failed" if key == ("b", "T1", "push") else "complete",
                    "receipt_status": "failed" if key == ("b", "T1", "push") else "complete",
                    "result": None if key == ("b", "T1", "push") else {"fall": False, "ee_pos_rmse_m": 0.1},
                })
        if duplicate and shard == 0:
            rows.append(dict(rows[0]))
        _write(node / "formal_results.json", {"rows": rows})
    return tasks


def test_aggregate_preserves_failures_and_requires_unique_complete_keys(tmp_path, monkeypatch):
    tasks = _campaign(tmp_path)
    monkeypatch.setattr(aggregation, "_load_tasks", lambda _: ({"suite_sha256": "suite"}, tasks))
    monkeypatch.setattr(aggregation, "_sha256", lambda _: "hash")
    result = aggregation.aggregate(tmp_path)
    assert result["scenario_count"] == result["expected_scenario_count"] == 8
    assert result["failure_count"] == 1
    assert result["failures"][0]["method"] == "b"


@pytest.mark.parametrize("duplicate", [False, True])
def test_aggregate_rejects_missing_or_duplicate_scenario_keys(tmp_path, monkeypatch, duplicate):
    tasks = _campaign(tmp_path, omit_key=None if duplicate else ("a", "T0", "nominal"), duplicate=duplicate)
    monkeypatch.setattr(aggregation, "_load_tasks", lambda _: ({"suite_sha256": "suite"}, tasks))
    monkeypatch.setattr(aggregation, "_sha256", lambda _: "hash")
    with pytest.raises(ValueError, match="scenario key validation failed"):
        aggregation.aggregate(tmp_path)
