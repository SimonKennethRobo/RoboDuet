"""Rescore a formal MuJoCo campaign under the current reporting protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from benchmark.wbc.scoring import DEVELOPMENT_TIMED_TRAJECTORY_PROTOCOL
from benchmark.wbc.trace import score_trace_archive


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rescore_formal_results(source: Path) -> dict:
    payload = json.loads(source.read_text())
    rows = []
    for row in payload["rows"]:
        receipt_path = Path(row["receipt"])
        receipt = json.loads(receipt_path.read_text())
        trace_path = Path(receipt["trace"]["path"])
        rescored = score_trace_archive(
            trace_path,
            protocol_override=DEVELOPMENT_TIMED_TRAJECTORY_PROTOCOL,
        )
        if len(rescored) != 1:
            raise ValueError(f"expected one result in {trace_path}, got {len(rescored)}")
        rows.append({**row, "result": rescored[0]})
    return {
        "schema_version": "formal-mujoco-rescored-results-v1",
        "source_results": {"path": str(source), "sha256": _sha256(source)},
        "protocol": dict(DEVELOPMENT_TIMED_TRAJECTORY_PROTOCOL),
        "campaign_status": payload["campaign_status"],
        "environment_id": payload["environment_id"],
        "environment_scene_sha256": payload["environment_scene_sha256"],
        "suite_sha256": payload["suite_sha256"],
        "rows": rows,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = rescore_formal_results(args.source.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
