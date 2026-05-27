"""Generate a standalone HTML report from benchmark results.json."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from html import escape
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple


SCENARIO_TITLES = {
    "vel_grid": "Velocity Grid",
    "arm_sweep": "Arm Disturbance Sweep",
    "body_pose": "Body Pose Tracking",
    "gait": "Gait Tracking",
}

SCENARIO_ORDER = ["vel_grid", "arm_sweep", "body_pose", "gait"]

PRIMARY_METRICS = [
    ("lin_vel_x_rmse", "vx RMSE", "lower"),
    ("ang_vel_yaw_rmse", "yaw RMSE", "lower"),
    ("fall_rate", "fall rate", "lower"),
    ("tracking_lin_vel_reward", "lin reward", "higher"),
    ("tracking_ang_vel_reward", "yaw reward", "higher"),
    ("base_height_mean", "base height", "neutral"),
    ("max_torque_mean", "max torque", "lower"),
]

DETAIL_METRICS = {
    "vel_grid": [
        ("vx RMSE", "lin_vel_x_rmse"),
        ("yaw RMSE", "ang_vel_yaw_rmse"),
        ("lin reward", "tracking_lin_vel_reward"),
        ("yaw reward", "tracking_ang_vel_reward"),
        ("base height", "base_height_mean"),
        ("fall rate", "fall_rate"),
    ],
    "arm_sweep": [
        ("vx RMSE", "lin_vel_x_rmse"),
        ("yaw RMSE", "ang_vel_yaw_rmse"),
        ("lin reward", "tracking_lin_vel_reward"),
        ("yaw reward", "tracking_ang_vel_reward"),
        ("base height", "base_height_mean"),
        ("max torque", "max_torque_mean"),
        ("fall rate", "fall_rate"),
    ],
    "body_pose": [
        ("pitch RMSE deg", "pitch_rmse_deg"),
        ("roll RMSE deg", "roll_rmse_deg"),
        ("orientation ctl", "orientation_control_rmse"),
        ("height RMSE m", "height_rmse_m"),
        ("base height", "base_height_mean"),
        ("fall rate", "fall_rate"),
    ],
    "gait": [
        ("contact force", "gait_contact_force_cost"),
        ("contact vel", "gait_contact_vel_cost"),
        ("clearance m", "foot_clearance_rmse_m"),
        ("raibert m", "raibert_rmse_m"),
        ("max torque", "max_torque_mean"),
        ("fall rate", "fall_rate"),
    ],
}


def _load_results(path: Path) -> Dict[str, Dict[str, List[dict]]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_metadata(results_path: Path) -> Dict[str, Any]:
    path = results_path.with_name("metadata.json")
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _finite_values(rows: Iterable[dict], key: str) -> List[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if _is_number(value):
            values.append(float(value))
    return values


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


def _scenario_names(results: Dict[str, Dict[str, List[dict]]]) -> List[str]:
    names = set()
    for scenarios in results.values():
        names.update(scenarios)
    ordered = [name for name in SCENARIO_ORDER if name in names]
    ordered.extend(sorted(names - set(ordered)))
    return ordered


def _metric_average(rows: Iterable[dict], metric: str) -> Optional[float]:
    values = _finite_values(rows, metric)
    if not values:
        return None
    return mean(values)


def _table(headers: List[str], rows: List[List[Tuple[str, str]]], sortable: bool = True) -> str:
    th_attr = ' data-sortable="1"' if sortable else ""
    head = "".join(f"<th{th_attr}>{escape(label)}</th>" for label in headers)
    body = []
    for row in rows:
        cells = "".join(f'<td class="{klass}">{value}</td>' for value, klass in row)
        body.append(f"<tr>{cells}</tr>")
    table_class = ' class="sortable"' if sortable else ""
    return f"<table{table_class}><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _summary_table(results: Dict[str, Dict[str, List[dict]]], scenarios: List[str]) -> str:
    headers = ["scope", "points"] + [label for _, label, _ in PRIMARY_METRICS]
    sections = []
    for run_name, scenario_map in results.items():
        body = []
        for scenario in scenarios:
            scenario_rows = scenario_map.get(scenario, [])
            cells = [
                (f"<strong>{escape(SCENARIO_TITLES.get(scenario, scenario))}</strong>", ""),
                (_fmt(len(scenario_rows), 0), ""),
            ]
            for metric, _, _ in PRIMARY_METRICS:
                cells.append((_fmt(_metric_average(scenario_rows, metric)), ""))
            body.append(cells)

        title = f"<h3>{escape(run_name)}</h3>"
        sections.append(title + _table(headers, body))
    note = '<p class="summary-note">Metric columns are means over the benchmark test points in each scope.</p>'
    return note + "".join(sections)


def _heat_value(value: Any, min_value: float, max_value: float, invert: bool = False) -> str:
    if not _is_number(value) or max_value <= min_value:
        return ""
    t = (float(value) - min_value) / (max_value - min_value)
    if invert:
        t = 1.0 - t
    hue = 145 - int(145 * t)
    return f' style="background:hsl({hue} 70% 90%)"'


def _metric_prefers_higher(metric: str) -> bool:
    return "reward" in metric or metric.endswith("_rew")


def _compact_axis_label(scenario: str, label: str) -> str:
    if scenario == "vel_grid":
        return label.replace(" yaw=", "/y").replace("vx=", "vx")
    if scenario == "arm_sweep":
        return label.replace("intensity=", "a")
    if scenario == "body_pose":
        return (
            label.replace("pitch=", "p")
            .replace("roll=", "r")
            .replace("h_target=", "h")
            .replace("rad", "")
            .replace("m", "")
        )
    if scenario == "gait":
        return (
            label.replace("gait_freq=", "f")
            .replace("swing_h=", "sw")
            .replace("stance_w=", "w")
            .replace("Hz", "")
            .replace("m", "")
        )
    return label


def _scenario_plot_data(results: Dict[str, Dict[str, List[dict]]], scenario: str, metric_defs: List[Tuple[str, str]]) -> str:
    payload = {
        "metrics": [{"label": label, "key": metric} for label, metric in metric_defs],
        "series": [],
    }
    for run_name, scenario_map in results.items():
        rows = scenario_map.get(scenario, [])
        if not rows:
            continue
        payload["series"].append(
            {
                "name": run_name,
                "labels": [str(row.get("label", "-")) for row in rows],
                "axis_labels": [_compact_axis_label(scenario, str(row.get("label", "-"))) for row in rows],
                "values": {
                    metric: [row.get(metric) if _is_number(row.get(metric)) else None for row in rows]
                    for _, metric in metric_defs
                },
            }
        )
    return escape(json.dumps(payload, separators=(",", ":")), quote=True)


def _scenario_plot(results: Dict[str, Dict[str, List[dict]]], scenario: str, metric_defs: List[Tuple[str, str]]) -> str:
    data = _scenario_plot_data(results, scenario, metric_defs)
    buttons = []
    for index, (label, metric) in enumerate(metric_defs):
        active = " active" if index == 0 else ""
        buttons.append(
            f'<button type="button" class="metric-tab{active}" data-metric="{escape(metric)}">{escape(label)}</button>'
        )
    return (
        f'<div class="metric-plot" data-plot="{data}">'
        '<div class="metric-tabs">'
        + "".join(buttons)
        + "</div>"
        '<div class="metric-chart" aria-label="metric chart"></div>'
        "</div>"
    )


def _scenario_detail_sections(results: Dict[str, Dict[str, List[dict]]], scenarios: List[str]) -> str:
    sections = []
    for scenario in scenarios:
        metric_defs = DETAIL_METRICS.get(scenario)
        if not metric_defs:
            continue
        run_sections = []
        heat_metrics = [metric for _, metric in metric_defs[:3]]
        for run_name, scenario_map in results.items():
            rows = scenario_map.get(scenario, [])
            if not rows:
                continue
            mins = {m: min(_finite_values(rows, m), default=0.0) for m in heat_metrics}
            maxs = {m: max(_finite_values(rows, m), default=0.0) for m in heat_metrics}
            headers = ["test point"] + [label for label, _ in metric_defs]
            body = []
            for row in rows:
                cells = [(escape(str(row.get("label", "-"))), "")]
                for _, metric in metric_defs:
                    value = _fmt(row.get(metric))
                    if metric in heat_metrics:
                        style = _heat_value(
                            row.get(metric),
                            mins[metric],
                            maxs[metric],
                            invert=_metric_prefers_higher(metric),
                        )
                        cells.append((f"<span{style}>{value}</span>", "metric-cell"))
                    else:
                        cells.append((value, ""))
                body.append(cells)
            run_sections.append(f"<h3>{escape(run_name)}</h3>{_table(headers, body)}")
        if run_sections:
            title = SCENARIO_TITLES.get(scenario, scenario)
            sections.append(
                f'<section class="panel" id="detail-{escape(scenario)}"><h2>{escape(title)}</h2>'
                + _scenario_plot(results, scenario, metric_defs)
                + "".join(run_sections)
                + "</section>"
            )
    if not sections:
        return ""
    return "".join(sections)


def _candidate_cards(results: Dict[str, Dict[str, List[dict]]]) -> str:
    cards = []
    for run_name, scenarios in results.items():
        points = sum(len(rows) for rows in scenarios.values())
        scenario_names = _scenario_names({run_name: scenarios})
        scenario_text = ", ".join(SCENARIO_TITLES.get(name, name) for name in scenario_names)
        all_rows = [row for scenario_rows in scenarios.values() for row in scenario_rows]
        cards.append(
            '<div class="card">'
            f"<h3>{escape(run_name)}</h3>"
            f"<p>{escape(scenario_text)}</p>"
            '<div class="card-stats">'
            f'<div class="stat"><span>{points}</span><label>points</label></div>'
            f'<div class="stat"><span>{_fmt(_metric_average(all_rows, "fall_rate"))}</span><label>fall rate mean</label></div>'
            f'<div class="stat"><span>{_fmt(_metric_average(all_rows, "lin_vel_x_rmse"))}</span><label>vx RMSE mean</label></div>'
            f'<div class="stat"><span>{_fmt(_metric_average(all_rows, "ang_vel_yaw_rmse"))}</span><label>yaw RMSE mean</label></div>'
            f'<div class="stat"><span>{_fmt(_metric_average(all_rows, "tracking_lin_vel_reward"))}</span><label>lin reward mean</label></div>'
            f'<div class="stat"><span>{_fmt(_metric_average(all_rows, "tracking_ang_vel_reward"))}</span><label>yaw reward mean</label></div>'
            f'<div class="stat"><span>{_fmt(_metric_average(all_rows, "base_height_mean"))}</span><label>base height mean</label></div>'
            f'<div class="stat"><span>{_fmt(_metric_average(all_rows, "max_torque_mean"))}</span><label>max torque mean</label></div>'
            '<div class="stat stat-empty" aria-hidden="true"></div>'
            "</div>"
            "</div>"
        )
    return "".join(cards)


def _nested_get(data: Dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _metadata_panel(metadata: Dict[str, Any]) -> str:
    if not metadata:
        return ""
    fields = [
        ("benchmark_mode", "benchmark_mode", None),
        ("benchmark_protocol", "benchmark_protocol", None),
        ("profile", "profile", None),
        ("candidate_dir", "candidate_dir", None),
        ("generated_at", "generated_at", None),
        ("git_commit", "git.short_commit", "git_commit"),
        ("git_branch", "git.branch", None),
        ("git_dirty", "git.dirty", None),
        ("robot", "robot", None),
        ("sim_device", "sim_device", None),
        ("seed", "seed", None),
        ("num_envs_per_policy", "num_envs_per_policy", None),
        ("total_envs", "total_envs", None),
        ("num_eval_steps", "num_eval_steps", None),
        ("ckptids", "ckptids", None),
        ("logdirs", "logdirs", None),
        ("control_dt_s", "control_dt_s", None),
        ("command", "benchmark.command", None),
        ("runtime_python", "runtime.python", None),
        ("runtime_torch", "runtime.torch", None),
        ("cuda_device", "runtime.cuda_device_name", None),
    ]
    rows = []
    for label, path, fallback in fields:
        value = _nested_get(metadata, path)
        if value is None and fallback is not None:
            value = metadata.get(fallback)
        if value is None:
            continue
        if isinstance(value, list):
            value = ", ".join(str(item) for item in value)
        rows.append(f"<tr><th>{escape(label)}</th><td>{escape(str(value))}</td></tr>")
    if not rows:
        return ""
    return '<section class="panel metadata-panel" id="metadata"><h2>Metadata</h2><table><tbody>' + "".join(rows) + "</tbody></table></section>"


def _rel_link(target: Path, base_dir: Path) -> str:
    return escape(os.path.relpath(target, base_dir))


def _result_links(source: Path, output_path: Path, limit: int = 16) -> List[dict]:
    root = _find_results_root(source)
    if not root.is_dir():
        return []

    entries = []
    result_paths = sorted(
        root.rglob("results.json"),
        key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
        reverse=True,
    )
    for results_path in result_paths:
        entry = _read_index_entry(results_path, root)
        if entry is None:
            continue
        entries.append(
            {
                "name": entry["name"],
                "href": _rel_link(entry["report"], output_path.parent),
                "active": results_path.resolve() == source.resolve(),
                "candidate": ", ".join(entry["candidates"][:2]),
                "points": entry["points"],
            }
        )
        if len(entries) >= limit:
            break
    return entries


def _side_nav(source: Path, output_path: Path, scenarios: List[str], has_metadata: bool) -> str:
    section_links = [("Summary", "#summary")]
    if has_metadata:
        section_links.append(("Metadata", "#metadata"))
    for scenario in scenarios:
        if scenario in DETAIL_METRICS:
            section_links.append((SCENARIO_TITLES.get(scenario, scenario), f"#detail-{scenario}"))
    sections = "".join(f'<a href="{escape(href)}">{escape(label)}</a>' for label, href in section_links)
    result_items = []
    for entry in _result_links(source, output_path):
        active = " active" if entry["active"] else ""
        title = entry["candidate"] or entry["name"]
        result_items.append(
            f'<a class="result-link{active}" href="{entry["href"]}">'
            f'<span>{escape(entry["name"])}</span>'
            f'<small>{escape(title)} · {entry["points"]} pts</small>'
            "</a>"
        )
    results_block = (
        '<div class="nav-group nav-results"><h3>Results</h3>'
        + "".join(result_items)
        + "</div>"
        if result_items
        else ""
    )
    return (
        '<aside class="side-nav">'
        '<div class="nav-group"><h3>Report</h3>'
        f"{sections}"
        "</div>"
        f"{results_block}"
        "</aside>"
    )


def _render_html(results: Dict[str, Dict[str, List[dict]]], source: Path, output_path: Path) -> str:
    scenarios = _scenario_names(results)
    metadata = _load_metadata(source)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RoboDuet Benchmark Report</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #1f2933;
      --muted: #667085;
      --border: #d9dee7;
      --accent: #0f766e;
      --best: #e7f7ef;
      --worst: #fff0ed;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    header {{
      max-width: 1520px;
      margin: 24px auto 0;
      padding: 0 32px;
    }}
    .header-inner {{
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #ffffff;
      padding: 22px 24px;
    }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    h2 {{ margin: 0 0 16px; font-size: 18px; }}
    h3 {{ margin: 16px 0 10px; font-size: 15px; }}
    .meta {{ color: var(--muted); display: flex; flex-wrap: wrap; gap: 12px 22px; }}
    a, footer a {{
      color: var(--accent);
      text-decoration: none;
      font-weight: 650;
    }}
    a:hover, footer a:hover {{ text-decoration: underline; }}
    .layout {{
      display: grid;
      grid-template-columns: 220px minmax(0, 1fr);
      gap: 20px;
      max-width: 1520px;
      margin: 0 auto;
      padding: 24px 32px 40px;
    }}
    .side-nav {{
      position: sticky;
      top: 18px;
      align-self: start;
      display: grid;
      gap: 12px;
      max-height: calc(100vh - 36px);
      overflow-y: auto;
    }}
    .nav-group {{
      padding: 12px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #ffffff;
    }}
    .nav-group h3 {{
      margin: 0 0 8px;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0;
    }}
    .nav-group a {{
      display: block;
      border-radius: 6px;
      padding: 7px 8px;
      color: var(--text);
      text-decoration: none;
      font-weight: 650;
    }}
    .nav-group a:hover {{
      background: #eef6f4;
      color: var(--accent);
      text-decoration: none;
    }}
    .nav-group a.active {{
      background: #e4f2ef;
      color: var(--accent);
    }}
    .nav-results {{
      max-height: 46vh;
      overflow-y: auto;
    }}
    .result-link span {{
      display: block;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .result-link small {{
      display: block;
      color: var(--muted);
      font-size: 11px;
      font-weight: 500;
      line-height: 1.3;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .result-link:hover small, .result-link.active small {{
      color: var(--accent);
    }}
    main {{ min-width: 0; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; margin-bottom: 18px; }}
    .card, .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
    }}
    .card {{ padding: 16px; }}
    .card h3 {{ margin-top: 0; }}
    .card p {{ color: var(--muted); }}
    .card-stats {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }}
    .stat span {{ font-size: 22px; font-weight: 700; color: var(--accent); }}
    .stat label {{ display: block; color: var(--muted); }}
    .stat-empty {{ visibility: hidden; }}
    .panel {{ padding: 18px; margin-bottom: 18px; overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; min-width: 780px; }}
    th, td {{ padding: 9px 10px; border-bottom: 1px solid var(--border); text-align: right; white-space: nowrap; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ color: var(--muted); font-weight: 600; background: #fbfcfe; }}
    th[data-sortable="1"] {{ cursor: pointer; user-select: none; }}
    th[data-sortable="1"]::after {{ content: " ↕"; color: #98a2b3; font-weight: 400; }}
    .metadata-panel table {{ min-width: 0; }}
    .metadata-panel th {{ width: 220px; text-align: left; }}
    .metadata-panel td {{ text-align: left; white-space: normal; word-break: break-word; }}
    .summary-note {{ color: var(--muted); margin: 0 0 12px; }}
    td.best {{ background: var(--best); color: #047857; font-weight: 650; }}
    td.worst {{ background: var(--worst); color: #b42318; }}
    td.metric-cell span {{ display: block; margin: -9px -10px; padding: 9px 10px; }}
    .metric-plot {{
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #fbfcfe;
      padding: 12px;
      margin-bottom: 18px;
    }}
    .metric-tabs {{
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 10px;
    }}
    .metric-tab {{
      border: 1px solid var(--border);
      border-radius: 6px;
      background: #ffffff;
      color: var(--text);
      font: inherit;
      font-weight: 650;
      padding: 6px 9px;
      cursor: pointer;
    }}
    .metric-tab:hover, .metric-tab.active {{
      background: #e4f2ef;
      color: var(--accent);
    }}
    .metric-chart {{
      min-height: 280px;
      overflow-x: auto;
    }}
    .metric-chart svg {{
      display: block;
      width: 100%;
      min-width: 720px;
      height: auto;
    }}
    .chart-axis {{ stroke: #98a2b3; stroke-width: 1; }}
    .chart-grid {{ stroke: #d9dee7; stroke-width: 1; }}
    .chart-label {{ fill: var(--muted); font-size: 11px; }}
    .chart-line {{ fill: none; stroke-width: 2.4; }}
    .chart-point {{ stroke: #ffffff; stroke-width: 1.5; }}
    .chart-legend {{ fill: var(--text); font-size: 12px; }}
    footer {{ color: var(--muted); padding: 0 32px 30px; max-width: 1520px; margin: 0 auto; }}
    @media (max-width: 900px) {{
      .layout {{
        grid-template-columns: 1fr;
        padding: 16px;
      }}
      .side-nav {{
        position: static;
        display: block;
      }}
      .nav-group {{
        margin-bottom: 10px;
      }}
      .nav-group a {{
        display: inline-block;
        margin: 0 4px 6px 0;
      }}
      header {{ margin-top: 16px; padding: 0 16px; }}
      .header-inner {{ padding: 18px; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="header-inner">
      <h1>RoboDuet Benchmark Report</h1>
      <div class="meta">
        <span>source: {escape(str(source))}</span>
        <span>generated: {escape(generated)}</span>
        <span>candidates: {len(results)}</span>
        <span>scenarios: {len(scenarios)}</span>
      </div>
    </div>
  </header>
  <div class="layout">
    {_side_nav(source, output_path, scenarios, bool(metadata))}
    <main>
      <section class="cards">{_candidate_cards(results)}</section>
      <section class="panel" id="summary"><h2>Summary - Metric Mean</h2>{_summary_table(results, scenarios)}</section>
      {_metadata_panel(metadata)}
      {_scenario_detail_sections(results, scenarios)}
    </main>
  </div>
  <footer>Heatmap colors are normalized within each candidate and scenario table; green indicates better relative values for highlighted metrics, red indicates worse relative values.</footer>
  <script>
    (() => {{
      document.querySelectorAll("table.sortable").forEach((table) => {{
        const headers = Array.from(table.querySelectorAll("th"));
        headers.forEach((header, colIndex) => {{
          header.addEventListener("click", () => {{
            const tbody = table.tBodies[0];
            const rows = Array.from(tbody.rows);
            const direction = header.dataset.direction === "asc" ? "desc" : "asc";
            headers.forEach((item) => delete item.dataset.direction);
            header.dataset.direction = direction;
            rows.sort((a, b) => {{
              const av = a.cells[colIndex]?.textContent.trim() || "";
              const bv = b.cells[colIndex]?.textContent.trim() || "";
              const an = Number(av);
              const bn = Number(bv);
              const bothNumeric = Number.isFinite(an) && Number.isFinite(bn);
              const cmp = bothNumeric ? an - bn : av.localeCompare(bv);
              return direction === "asc" ? cmp : -cmp;
            }});
            rows.forEach((row) => tbody.appendChild(row));
          }});
        }});
      }});

      const colors = ["#0f766e", "#7c3aed", "#dc6803", "#2563eb", "#c026d3", "#16a34a"];

      function finite(values) {{
        return values.filter((value) => Number.isFinite(value));
      }}

      function fmt(value) {{
        if (!Number.isFinite(value)) return "-";
        return value.toFixed(4);
      }}

      function renderMetricPlot(plot, metricKey) {{
        const data = JSON.parse(plot.dataset.plot || "{{}}");
        const metric = (data.metrics || []).find((item) => item.key === metricKey) || (data.metrics || [])[0];
        const chart = plot.querySelector(".metric-chart");
        if (!metric || !chart) return;

        const series = (data.series || []).map((item, i) => ({{
          name: item.name,
          labels: item.labels || [],
          axisLabels: item.axis_labels || item.labels || [],
          values: (item.values && item.values[metric.key] ? item.values[metric.key] : []).map((value) =>
            value === null ? NaN : Number(value)
          ),
          color: colors[i % colors.length],
        }}));
        const allValues = series.flatMap((item) => finite(item.values));
        if (!allValues.length) {{
          chart.innerHTML = "<p class='summary-note'>No numeric values for this metric.</p>";
          return;
        }}

        let yMin = Math.min(...allValues);
        let yMax = Math.max(...allValues);
        if (yMin === yMax) {{
          yMin -= 1;
          yMax += 1;
        }}
        const pad = (yMax - yMin) * 0.08;
        yMin -= pad;
        yMax += pad;

        const width = 820;
        const height = 280;
        const margin = {{ left: 58, right: 24, top: series.length > 1 ? 34 : 18, bottom: 58 }};
        const innerW = width - margin.left - margin.right;
        const innerH = height - margin.top - margin.bottom;
        const maxPoints = Math.max(...series.map((item) => item.values.length));
        const x = (index) => margin.left + (maxPoints <= 1 ? innerW / 2 : (index / (maxPoints - 1)) * innerW);
        const y = (value) => margin.top + (1 - (value - yMin) / (yMax - yMin)) * innerH;
        const labels = series[0]?.labels || [];
        const axisLabels = series[0]?.axisLabels || labels;
        const labelStep = Math.max(1, Math.ceil(maxPoints / 8));
        const ticks = [0, 0.25, 0.5, 0.75, 1].map((t) => yMin + t * (yMax - yMin));

        const grid = ticks.map((tick) => {{
          const yy = y(tick);
          return `<line class="chart-grid" x1="${{margin.left}}" y1="${{yy}}" x2="${{width - margin.right}}" y2="${{yy}}"></line>` +
            `<text class="chart-label" x="${{margin.left - 8}}" y="${{yy + 4}}" text-anchor="end">${{fmt(tick)}}</text>`;
        }}).join("");

        const xLabels = Array.from({{ length: maxPoints }}, (_, i) => {{
          if (i % labelStep !== 0 && i !== maxPoints - 1) return "";
          const label = axisLabels[i] || labels[i] || String(i + 1);
          return `<text class="chart-label" x="${{x(i)}}" y="${{height - 16}}" text-anchor="middle">` +
            `${{label}}<title>${{labels[i] || label}}</title></text>`;
        }}).join("");

        const paths = series.map((item) => {{
          const points = item.values
            .map((value, i) => Number.isFinite(value) ? `${{x(i)}},${{y(value)}}` : null)
            .filter(Boolean);
          const circles = item.values.map((value, i) => {{
            if (!Number.isFinite(value)) return "";
            return `<circle class="chart-point" cx="${{x(i)}}" cy="${{y(value)}}" r="4" fill="${{item.color}}">` +
              `<title>${{item.name}} | ${{labels[i] || i}} | ${{metric.label}}: ${{fmt(value)}}</title></circle>`;
          }}).join("");
          return `<polyline class="chart-line" stroke="${{item.color}}" points="${{points.join(" ")}}"></polyline>${{circles}}`;
        }}).join("");

        const legend = series.length <= 1 ? "" : series.map((item, i) => {{
          const lx = margin.left + i * 170;
          return `<circle cx="${{lx}}" cy="18" r="5" fill="${{item.color}}"></circle>` +
            `<text class="chart-legend" x="${{lx + 9}}" y="22">${{item.name}}</text>`;
        }}).join("");

        chart.innerHTML = `<svg viewBox="0 0 ${{width}} ${{height}}" role="img" aria-label="${{metric.label}}">` +
          legend +
          grid +
          `<line class="chart-axis" x1="${{margin.left}}" y1="${{margin.top + innerH}}" x2="${{width - margin.right}}" y2="${{margin.top + innerH}}"></line>` +
          `<line class="chart-axis" x1="${{margin.left}}" y1="${{margin.top}}" x2="${{margin.left}}" y2="${{margin.top + innerH}}"></line>` +
          paths +
          xLabels +
          `</svg>`;
      }}

      document.querySelectorAll(".metric-plot").forEach((plot) => {{
        const firstButton = plot.querySelector(".metric-tab");
        if (firstButton) renderMetricPlot(plot, firstButton.dataset.metric);
        plot.querySelectorAll(".metric-tab").forEach((button) => {{
          button.addEventListener("click", () => {{
            plot.querySelectorAll(".metric-tab").forEach((item) => item.classList.remove("active"));
            button.classList.add("active");
            renderMetricPlot(plot, button.dataset.metric);
          }});
        }});
      }});

      const scrollKey = "roboduet-benchmark-scroll-y";
      document.querySelectorAll(".result-link").forEach((link) => {{
        link.addEventListener("click", () => {{
          sessionStorage.setItem(scrollKey, String(window.scrollY));
        }});
      }});
      const savedScroll = sessionStorage.getItem(scrollKey);
      if (savedScroll !== null) {{
        sessionStorage.removeItem(scrollKey);
        const y = Number(savedScroll);
        if (Number.isFinite(y)) {{
          requestAnimationFrame(() => window.scrollTo(0, y));
          window.addEventListener("load", () => window.scrollTo(0, y), {{ once: true }});
        }}
      }}
    }})();
  </script>
</body>
</html>
"""


def _find_results_root(path: Path) -> Path:
    for parent in path.resolve().parents:
        if parent.name == "results":
            return parent
    return path.parent


def _read_index_entry(results_path: Path, root: Path) -> Optional[dict]:
    try:
        results = _load_results(results_path)
    except (OSError, json.JSONDecodeError):
        return None

    run_dir = results_path.parent
    scenarios = _scenario_names(results)
    points = sum(len(rows) for scenario_map in results.values() for rows in scenario_map.values())
    candidates = list(results)
    report_path = run_dir / "index.html"
    if not report_path.exists() and (run_dir / "report.html").exists():
        report_path = run_dir / "report.html"
    return {
        "name": os.path.relpath(run_dir, root),
        "results": results_path,
        "report": report_path,
        "candidates": candidates,
        "scenarios": scenarios,
        "points": points,
    }


def _render_index(entries: List[dict], root: Path) -> str:
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not entries:
        latest_href = ""
        latest_name = "No benchmark results found"
        refresh = ""
        script = ""
    else:
        latest = entries[0]
        latest_report = latest["report"] if latest["report"].exists() else latest["results"]
        latest_href = _rel_link(latest_report, root)
        latest_name = latest["name"]
        refresh = f'<meta http-equiv="refresh" content="0; url={latest_href}">'
        script = f'<script>window.location.replace("{latest_href}");</script>'

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  {refresh}
  <title>RoboDuet Benchmark Report</title>
  <style>
    body {{ margin: 0; font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f6f7f9; color: #1f2933; }}
    header {{ padding: 28px 32px 18px; background: #fff; border-bottom: 1px solid #d9dee7; }}
    main {{ padding: 24px 32px 40px; max-width: 760px; margin: 0 auto; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    .meta, p {{ color: #667085; }}
    .panel {{ background: #fff; border: 1px solid #d9dee7; border-radius: 8px; padding: 18px; }}
    a {{ color: #0f766e; text-decoration: none; font-weight: 600; margin-right: 10px; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <header>
    <h1>RoboDuet Benchmark Report</h1>
    <div class="meta">generated: {escape(generated)} | result sets: {len(entries)}</div>
  </header>
  <main>
    <section class="panel">
      <h2>Opening latest report</h2>
      <p>{escape(latest_name)}</p>
      {f'<p><a href="{latest_href}">Open report</a></p>' if latest_href else ''}
    </section>
  </main>
  {script}
</body>
</html>
"""


def _write_index_for_root(root: Path):
    entries = []
    result_paths = sorted(
        root.rglob("results.json"),
        key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
        reverse=True,
    )
    for path in result_paths:
        entry = _read_index_entry(path, root)
        if entry is not None:
            entries.append(entry)
    for entry in entries:
        report_path = entry["report"]
        if not report_path.exists():
            results = _load_results(entry["results"])
            report_path.write_text(_render_html(results, entry["results"], report_path), encoding="utf-8")
    index_path = root / "index.html"
    if entries:
        latest_results = entries[0]["results"]
        latest = _load_results(latest_results)
        index_path.write_text(_render_html(latest, latest_results, index_path), encoding="utf-8")
        print(f"[Benchmark] Latest report index saved -> {index_path}")
    else:
        index_path.write_text(_render_index(entries, root), encoding="utf-8")
        print(f"[Benchmark] Empty results index saved -> {index_path}")


def _write_index(results_path: Path):
    _write_index_for_root(_find_results_root(results_path))


def _write_report(results_path: Path, output_path: Path):
    results = _load_results(results_path)
    html = _render_html(results, results_path, output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"[Benchmark] HTML report saved -> {output_path}")


def write_report_bundle(results_path: str):
    path = Path(results_path)
    _write_report(path, path.with_name("index.html"))
    _write_index(path)


def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Generate standalone HTML report from benchmark results.json")
    parser.add_argument("--results", default=None, help="Path to benchmark results.json")
    parser.add_argument(
        "--results_root",
        default=None,
        help="Generate index.html for every results.json under this benchmark results root",
    )
    parser.add_argument("--output", default=None, help="Output HTML path. Defaults to index.html next to results.json")
    parser.add_argument(
        "--no_index",
        action="store_true",
        help="Do not update the benchmark results index.html next to the results root",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    if args.results_root:
        root = Path(args.results_root)
        result_paths = sorted(root.rglob("results.json"))
        if not result_paths:
            raise ValueError(f"{root}: no results.json files found")
        if args.output:
            raise ValueError("--output cannot be used with --results_root")
        for results_path in result_paths:
            _write_report(results_path, results_path.with_name("index.html"))
        if not args.no_index:
            _write_index_for_root(root)
        return

    if not args.results:
        raise ValueError("Provide either --results or --results_root")

    results_path = Path(args.results)
    output_path = Path(args.output) if args.output else results_path.with_name("index.html")
    _write_report(results_path, output_path)
    if not args.no_index:
        _write_index(results_path)


if __name__ == "__main__":
    main()
