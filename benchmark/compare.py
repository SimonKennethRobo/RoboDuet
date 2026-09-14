"""Compare two saved benchmark result directories."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from html import escape
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple


PRIMARY_METRICS = [
    ("ee_pos_rmse_m", "EE pos RMSE", "lower"),
    ("ee_rot_rmse_rad", "EE rot RMSE", "lower"),
    ("completion_rate", "completion rate", "higher"),
    ("lin_vel_x_rmse", "vx RMSE", "lower"),
    ("ang_vel_yaw_rmse", "yaw RMSE", "lower"),
    ("fall_rate", "fall rate", "lower"),
    ("tracking_lin_vel_reward", "lin reward", "higher"),
    ("tracking_ang_vel_reward", "yaw reward", "higher"),
    ("base_height_mean", "base height", "neutral"),
    ("max_torque_mean", "max torque", "lower"),
]


def _result_file(path: Path) -> Path:
    return path if path.name == "results.json" else path / "results.json"


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_metadata(results_path: Path) -> Dict[str, Any]:
    metadata_path = results_path.with_name("metadata.json")
    if not metadata_path.is_file():
        return {}
    try:
        return _load_json(metadata_path)
    except (OSError, json.JSONDecodeError):
        return {}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _finite_values(rows: Iterable[dict], key: str) -> List[float]:
    return [float(row[key]) for row in rows if _is_number(row.get(key))]


def _metric_average(rows: Iterable[dict], metric: str) -> Optional[float]:
    values = _finite_values(rows, metric)
    if not values:
        return None
    return mean(values)


def _preferred_result_rows(scenarios):
    if scenarios.get("wbc_trajectories"):
        return scenarios["wbc_trajectories"]
    return [row for scenario_rows in scenarios.values() for row in scenario_rows]


def _summary_rows(results: Dict[str, Dict[str, List[dict]]]) -> Dict[str, Dict[str, Any]]:
    summaries = {}
    for run_name, scenarios in results.items():
        rows = _preferred_result_rows(scenarios)
        summary = {
            "candidate": run_name,
            "scenarios": len(scenarios),
            "points": len(rows),
        }
        for metric, _, _ in PRIMARY_METRICS:
            summary[metric] = _metric_average(rows, metric)
        summaries[run_name] = summary
    return summaries


def _aggregate_summary(results: Dict[str, Dict[str, List[dict]]], label: str) -> Dict[str, Any]:
    all_rows = []
    scenario_count = 0
    for scenarios in results.values():
        scenario_count += len(scenarios)
        all_rows.extend(_preferred_result_rows(scenarios))
    summary = {
        "candidate": label,
        "scenarios": scenario_count,
        "points": len(all_rows),
    }
    for metric, _, _ in PRIMARY_METRICS:
        summary[metric] = _metric_average(all_rows, metric)
    return summary


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and math.isnan(value):
        return "-"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            return "-"
        return f"{float(value):.{digits}f}"
    return escape(str(value))


def _delta(base: Any, target: Any) -> str:
    if not _is_number(base) or not _is_number(target):
        return "-"
    diff = float(target) - float(base)
    pct = ""
    if abs(float(base)) > 1e-12:
        pct = f" ({diff / abs(float(base)) * 100:+.1f}%)"
    return f"{diff:+.4f}{pct}"


def _verdict(base: Any, target: Any, direction: str) -> str:
    if direction == "neutral" or not _is_number(base) or not _is_number(target):
        return "na"
    diff = float(target) - float(base)
    if abs(diff) < 1e-12:
        return "unchanged"
    improved = diff > 0 if direction == "higher" else diff < 0
    return "improved" if improved else "regressed"


def _metadata_label(path: Path, metadata: Dict[str, Any]) -> str:
    git = metadata.get("git", {}) if isinstance(metadata.get("git"), dict) else {}
    parts = [
        path.parent.name,
        str(metadata.get("benchmark_protocol") or metadata.get("benchmark_mode") or ""),
        str(git.get("short_commit") or metadata.get("git_commit") or ""),
    ]
    return " | ".join(part for part in parts if part)


def _comparison_rows(
    baseline: Dict[str, Dict[str, Any]],
    target: Dict[str, Dict[str, Any]],
    baseline_label: str,
    target_label: str,
) -> List[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    common = sorted(set(baseline) & set(target))
    if common:
        return [(name, baseline[name], target[name]) for name in common]
    if len(baseline) == 1 and len(target) == 1:
        b_row = next(iter(baseline.values()))
        t_row = next(iter(target.values()))
        return [(f"{baseline_label} -> {target_label}", b_row, t_row)]
    return []


def _render_table(rows: List[Tuple[str, Dict[str, Any], Dict[str, Any]]]) -> str:
    headers = ["candidate", "metric", "baseline", "target", "delta", "status"]
    head = "".join(f"<th>{escape(header)}</th>" for header in headers)
    body = []
    for candidate, base_row, target_row in rows:
        for metric, label, direction in PRIMARY_METRICS:
            delta = _delta(base_row.get(metric), target_row.get(metric))
            status = _verdict(base_row.get(metric), target_row.get(metric), direction)
            status_label = "n/a" if status == "na" else status
            body.append(
                "<tr>"
                f"<td>{escape(candidate)}</td>"
                f"<td>{escape(label)}</td>"
                f"<td>{_fmt(base_row.get(metric))}</td>"
                f"<td>{_fmt(target_row.get(metric))}</td>"
                f'<td class="{escape(status)}">{escape(delta)}</td>'
                f'<td class="{escape(status)}">{escape(status_label)}</td>'
                "</tr>"
            )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _render_html(
    *,
    baseline_path: Path,
    target_path: Path,
    baseline_metadata: Dict[str, Any],
    target_metadata: Dict[str, Any],
    rows: List[Tuple[str, Dict[str, Any], Dict[str, Any]]],
) -> str:
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    baseline_label = _metadata_label(baseline_path, baseline_metadata)
    target_label = _metadata_label(target_path, target_metadata)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RoboDuet Benchmark Comparison</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #1f2933;
      --muted: #667085;
      --border: #d9dee7;
      --accent: #0f766e;
      --good: #047857;
      --bad: #b42318;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    header {{
      padding: 28px 32px 18px;
      border-bottom: 1px solid var(--border);
      background: #ffffff;
    }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    .meta {{ color: var(--muted); display: flex; flex-wrap: wrap; gap: 12px 22px; }}
    main {{ padding: 24px 32px 40px; max-width: 1280px; margin: 0 auto; }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 18px;
      margin-bottom: 18px;
      overflow-x: auto;
    }}
    .sources {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; }}
    .source {{
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
      background: #fbfcfe;
    }}
    .source label {{ display: block; color: var(--muted); font-size: 12px; font-weight: 650; }}
    .source code {{ overflow-wrap: anywhere; }}
    table {{ width: 100%; border-collapse: collapse; min-width: 860px; }}
    th, td {{ padding: 9px 10px; border-bottom: 1px solid var(--border); text-align: right; white-space: nowrap; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    th {{ color: var(--muted); font-weight: 600; background: #fbfcfe; }}
    .improved {{ color: var(--good); font-weight: 650; }}
    .regressed {{ color: var(--bad); font-weight: 650; }}
    .flat, .unchanged, .na {{ color: var(--muted); }}
    a {{ color: var(--accent); text-decoration: none; font-weight: 650; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <header>
    <h1>RoboDuet Benchmark Comparison</h1>
    <div class="meta">
      <span>generated: {escape(generated)}</span>
      <span>baseline: {escape(baseline_label)}</span>
      <span>target: {escape(target_label)}</span>
    </div>
  </header>
  <main>
    <section class="panel sources">
      <div class="source">
        <label>baseline</label>
        <p>{escape(baseline_label)}</p>
        <code>{escape(str(baseline_path))}</code>
      </div>
      <div class="source">
        <label>target</label>
        <p>{escape(target_label)}</p>
        <code>{escape(str(target_path))}</code>
      </div>
    </section>
    <section class="panel">
      <h2>Metric Changes</h2>
      {_render_table(rows) if rows else "<p>No comparable candidate names found. Use one-candidate result directories or matching candidate names.</p>"}
    </section>
  </main>
</body>
</html>
"""


def compare_results(baseline: Path, target: Path, output: Optional[Path] = None) -> Path:
    baseline_path = _result_file(baseline)
    target_path = _result_file(target)
    baseline_results = _load_json(baseline_path)
    target_results = _load_json(target_path)
    baseline_metadata = _load_metadata(baseline_path)
    target_metadata = _load_metadata(target_path)

    baseline_rows = _summary_rows(baseline_results)
    target_rows = _summary_rows(target_results)
    rows = _comparison_rows(baseline_rows, target_rows, baseline_path.parent.name, target_path.parent.name)
    if not rows:
        rows = [
            (
                "aggregate",
                _aggregate_summary(baseline_results, "baseline"),
                _aggregate_summary(target_results, "target"),
            )
        ]

    output_path = output or target_path.with_name(f"compare_{baseline_path.parent.name}_to_{target_path.parent.name}.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        _render_html(
            baseline_path=baseline_path,
            target_path=target_path,
            baseline_metadata=baseline_metadata,
            target_metadata=target_metadata,
            rows=rows,
        ),
        encoding="utf-8",
    )
    return output_path


def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Compare two saved benchmark result directories")
    parser.add_argument("--baseline", required=True, help="Baseline result directory or results.json")
    parser.add_argument("--target", required=True, help="Target result directory or results.json")
    parser.add_argument("--output", default=None, help="Output HTML path")
    parser.add_argument("--output_dir", default=None, help="Output directory for comparison report HTML")
    args = parser.parse_args(argv)
    if args.output and args.output_dir:
        parser.error("--output and --output_dir cannot be used together")
    return args


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    output = None
    if args.output:
        output = Path(args.output)
    elif args.output_dir:
        baseline_name = _result_file(Path(args.baseline)).parent.name
        target_name = _result_file(Path(args.target)).parent.name
        output = Path(args.output_dir) / f"compare_{baseline_name}_to_{target_name}.html"
    output = compare_results(
        Path(args.baseline),
        Path(args.target),
        output,
    )
    print(f"[Benchmark] Comparison report saved -> {output}")


if __name__ == "__main__":
    main()
