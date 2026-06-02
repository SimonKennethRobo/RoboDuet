"""Generate a standalone HTML report from benchmark results.json."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
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
        ("pitch RMS deg", "pitch_deg_rms"),
        ("roll RMSE deg", "roll_rmse_deg"),
        ("roll RMS deg", "roll_deg_rms"),
        ("orientation ctl", "orientation_control_rmse"),
        ("height RMSE m", "height_rmse_m"),
        ("base height", "base_height_mean"),
        ("fall rate", "fall_rate"),
    ],
    "gait": [
        ("freq RMSE Hz", "gait_freq_rmse_hz"),
        ("swing h RMSE m", "footswing_height_rmse_m"),
        ("stance w RMSE m", "stance_width_rmse_m"),
        ("contact force", "gait_contact_force_cost"),
        ("contact vel", "gait_contact_vel_cost"),
        ("clearance m", "foot_clearance_rmse_m"),
        ("raibert m", "raibert_rmse_m"),
        ("max torque", "max_torque_mean"),
        ("fall rate", "fall_rate"),
    ],
}

NEUTRAL_HEATMAP_METRICS = {"base_height_mean"}


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


def _metric_has_heatmap(metric: str) -> bool:
    return metric not in NEUTRAL_HEATMAP_METRICS


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
        heat_metrics = [metric for _, metric in metric_defs if _metric_has_heatmap(metric)]
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


def _json_for_script(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def _comparison_result_payload(source: Path, output_path: Path, limit: int = 16) -> List[dict]:
    root = _find_results_root(source)
    if not root.is_dir():
        return []

    payload = []
    result_paths = sorted(
        root.rglob("results.json"),
        key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
        reverse=True,
    )
    for results_path in result_paths:
        entry = _read_index_entry(results_path, root)
        if entry is None:
            continue
        report_path = entry["report"] if entry["report"].exists() else results_path.with_name("index.html")
        active = results_path.resolve() == source.resolve()
        payload.append(
            {
                "name": entry["name"],
                "href": os.path.relpath(report_path, output_path.parent),
                "active": active,
                "candidates": entry["candidates"],
                "scenarios": entry["scenarios"],
                "points": entry["points"],
                "results": entry["loaded_results"],
                "metadata": _load_metadata(results_path),
            }
        )
        if len(payload) >= limit:
            break
    return payload


def _comparison_series_payload(comparison_entries: List[dict]) -> List[dict]:
    active_entries = [entry for entry in comparison_entries if entry["active"]]
    visible_entries = active_entries or comparison_entries[:1]

    candidates = []
    for entry in visible_entries:
        for candidate in entry["results"]:
            candidates.append(candidate)
    duplicate_candidates = {name for name, count in Counter(candidates).items() if count > 1}

    payload = []
    for entry in visible_entries:
        for candidate, scenarios in entry["results"].items():
            points = sum(len(rows) for rows in scenarios.values())
            display_name = f'{entry["name"]} / {candidate}' if candidate in duplicate_candidates else candidate
            payload.append(
                {
                    "name": display_name,
                    "candidate": candidate,
                    "report": entry["name"],
                    "href": entry["href"],
                    "active": False,
                    "scenarios": entry["scenarios"],
                    "points": points,
                    "results": scenarios,
                    "metadata": entry["metadata"],
                }
            )
    return payload


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


def _side_nav(
    source: Path,
    output_path: Path,
    scenarios: List[str],
    has_metadata: bool,
    comparison_series: List[dict],
) -> str:
    section_links = [("Summary", "#summary")]
    if has_metadata:
        section_links.append(("Metadata", "#metadata"))
    for scenario in scenarios:
        if scenario in DETAIL_METRICS:
            section_links.append((SCENARIO_TITLES.get(scenario, scenario), f"#detail-{scenario}"))
    sections = "".join(f'<a href="{escape(href)}">{escape(label)}</a>' for label, href in section_links)
    result_items = []
    for index, entry in enumerate(comparison_series):
        active = " active" if entry["active"] else ""
        title = entry["report"]
        result_items.append(
            f'<div class="compare-result{active}">'
            f'<input type="checkbox" class="compare-toggle" data-compare-index="{index}" '
            f'aria-label="Compare {escape(entry["name"])}">'
            f'<a class="result-link{active}" href="{escape(entry["href"])}">'
            f'<span>{escape(entry["name"])}</span>'
            f'<small>{escape(title)} · {entry["points"]} pts</small>'
            "</a></div>"
        )
    results_block = (
        '<div class="nav-group nav-results"><div class="nav-title-row"><h3>Results</h3>'
        '<div class="nav-actions"><button type="button" class="nav-action" id="select-all-results">Select all</button>'
        '<button type="button" class="nav-action" id="clear-all-results">Clear all</button></div></div>'
        + "".join(result_items)
        + "</div>"
        if result_items
        else ""
    )
    return (
        '<aside class="side-nav">'
        '<div class="nav-group nav-report"><h3>Report</h3>'
        f'<div id="report-nav-links">{sections}</div>'
        "</div>"
        f"{results_block}"
        "</aside>"
    )


def _report_script(comparison_series: List[dict]) -> str:
    return (
        """<script>
    (() => {
      const COMPARE_RESULTS = __COMPARE_RESULTS__;
      const PRIMARY_METRICS = __PRIMARY_METRICS__;
      const DETAIL_METRICS = __DETAIL_METRICS__;
      const SCENARIO_TITLES = __SCENARIO_TITLES__;
      const SCENARIO_ORDER = __SCENARIO_ORDER__;
      const HOME_SCENARIO = "home_preview";
      const HOME_METRICS = [
        ["vx RMSE", "lin_vel_x_rmse"],
        ["yaw RMSE", "ang_vel_yaw_rmse"],
        ["lin reward", "tracking_lin_vel_reward"],
      ];
      const HOME_RESULTS = {
        "demo A": {
          [HOME_SCENARIO]: [
            { label: "vx=0.5 yaw=0.0", lin_vel_x_rmse: 0.24, ang_vel_yaw_rmse: 0.18, tracking_lin_vel_reward: 0.86 },
            { label: "vx=1.0 yaw=-0.5", lin_vel_x_rmse: 0.31, ang_vel_yaw_rmse: 0.36, tracking_lin_vel_reward: 0.74 },
            { label: "vx=1.0 yaw=0.5", lin_vel_x_rmse: 0.29, ang_vel_yaw_rmse: 0.27, tracking_lin_vel_reward: 0.79 },
          ],
        },
        "demo B": {
          [HOME_SCENARIO]: [
            { label: "vx=0.5 yaw=0.0", lin_vel_x_rmse: 0.19, ang_vel_yaw_rmse: 0.21, tracking_lin_vel_reward: 0.89 },
            { label: "vx=1.0 yaw=-0.5", lin_vel_x_rmse: 0.27, ang_vel_yaw_rmse: 0.30, tracking_lin_vel_reward: 0.81 },
            { label: "vx=1.0 yaw=0.5", lin_vel_x_rmse: 0.34, ang_vel_yaw_rmse: 0.25, tracking_lin_vel_reward: 0.76 },
          ],
        },
      };
      const METADATA_FIELDS = [
        ["benchmark_mode", "benchmark_mode", null],
        ["benchmark_protocol", "benchmark_protocol", null],
        ["profile", "profile", null],
        ["candidate_dir", "candidate_dir", null],
        ["generated_at", "generated_at", null],
        ["git_commit", "git.short_commit", "git_commit"],
        ["git_branch", "git.branch", null],
        ["git_dirty", "git.dirty", null],
        ["robot", "robot", null],
        ["sim_device", "sim_device", null],
        ["seed", "seed", null],
        ["num_envs_per_policy", "num_envs_per_policy", null],
        ["total_envs", "total_envs", null],
        ["num_eval_steps", "num_eval_steps", null],
        ["ckptids", "ckptids", null],
        ["logdirs", "logdirs", null],
        ["control_dt_s", "control_dt_s", null],
        ["command", "benchmark.command", null],
        ["runtime_python", "runtime.python", null],
        ["runtime_torch", "runtime.torch", null],
        ["cuda_device", "runtime.cuda_device_name", null],
      ];

      const colors = ["#0f766e", "#7c3aed", "#dc6803", "#2563eb", "#c026d3", "#16a34a"];
      let pinnedInteraction = null;

      function esc(value) {
        return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;",
        }[ch]));
      }

      function cssEscape(value) {
        if (window.CSS && typeof CSS.escape === "function") return CSS.escape(String(value));
        return String(value).replace(/["\\\\]/g, "\\\\$&");
      }

      function isNumber(value) {
        return typeof value === "number" && Number.isFinite(value);
      }

      function fmt(value, digits = 4) {
        if (value === null || value === undefined) return "-";
        if (typeof value === "number") return Number.isFinite(value) ? value.toFixed(digits) : "-";
        return esc(value);
      }

      function finiteValues(rows, key) {
        return (rows || []).map((row) => row[key]).filter(isNumber);
      }

      function minFinite(rows, key) {
        const values = finiteValues(rows, key);
        return values.length ? Math.min(...values) : 0;
      }

      function maxFinite(rows, key) {
        const values = finiteValues(rows, key);
        return values.length ? Math.max(...values) : 0;
      }

      function metricAverage(rows, key) {
        const values = finiteValues(rows, key);
        if (!values.length) return null;
        return values.reduce((sum, value) => sum + value, 0) / values.length;
      }

      function scenarioNames(results) {
        const names = new Set();
        Object.values(results).forEach((scenarioMap) => {
          Object.keys(scenarioMap || {}).forEach((name) => names.add(name));
        });
        const ordered = SCENARIO_ORDER.filter((name) => names.has(name));
        Array.from(names).sort().forEach((name) => {
          if (!ordered.includes(name)) ordered.push(name);
        });
        return ordered;
      }

      function metricPrefersHigher(metric) {
        return metric.includes("reward") || metric.endsWith("_rew");
      }

      function metricHasHeatmap(metric) {
        return metric !== "base_height_mean";
      }

      function metricIsNeutral(metric) {
        return metric === "points" || !metricHasHeatmap(metric);
      }

      function resultColor(name, index = 0) {
        return colors[index % colors.length];
      }

      function heatValue(value, minValue, maxValue, invert = false) {
        if (!isNumber(value) || maxValue <= minValue) return "";
        let t = (value - minValue) / (maxValue - minValue);
        if (invert) t = 1 - t;
        const hue = 145 - Math.floor(145 * t);
        return `background:hsl(${hue} 62% 96%)`;
      }

      function rankForValue(values, value, metric) {
        if (values.length <= 1 || metricIsNeutral(metric) || !isNumber(value)) return "";
        const bestValue = metricPrefersHigher(metric) ? Math.max(...values) : Math.min(...values);
        const worstValue = metricPrefersHigher(metric) ? Math.min(...values) : Math.max(...values);
        if (value === bestValue && bestValue !== worstValue) return "best";
        if (value === worstValue && bestValue !== worstValue) return "worst";
        return "";
      }

      function metricCellValue(value, rank, heatStyle = "") {
        const badge = rank ? `<em class="cell-rank-badge">${rank}</em>` : "";
        const styleAttr = heatStyle ? ` style="${heatStyle}"` : "";
        return `<span${styleAttr}><strong>${fmt(value)}</strong>${badge}</span>`;
      }

      function table(headers, rows, sortable = true) {
        const thAttr = sortable ? ' data-sortable="1"' : "";
        const tableClass = sortable ? ' class="sortable"' : "";
        return `<table${tableClass}><thead><tr>${headers.map((label) => `<th${thAttr}>${esc(label)}</th>`).join("")}</tr></thead>` +
          `<tbody>${rows.map((row) => `<tr>${row.map(([value, klass]) => `<td class="${klass || ""}">${value}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
      }

      function attrsToString(attrs = {}) {
        return Object.entries(attrs)
          .filter(([, value]) => value !== null && value !== undefined)
          .map(([key, value]) => ` ${key}="${esc(value)}"`)
          .join("");
      }

      function dataCell(value, klass = "", sortValue = null, attrs = {}) {
        const sortAttr = sortValue === null || sortValue === undefined ? "" : ` data-sort-value="${esc(sortValue)}"`;
        return `<td class="${klass}"${sortAttr}${attrsToString(attrs)}>${value}</td>`;
      }

      function groupedTable(stubLabel, groups, rows, options = {}) {
        const tableAttrs = attrsToString({
          "data-scenario": options.scenario || null,
          "data-compare-kind": options.kind || null,
        });
        const firstHeader = `<tr><th class="sticky-col" rowspan="2" data-sortable="1" data-sort-column="0">${esc(stubLabel)}</th>` +
          groups.map((group) => `<th class="metric-group metric-col-start metric-col-end" colspan="${group.columns.length}"><strong>${esc(group.label)}</strong></th>`).join("") +
          "</tr>";
        let colIndex = 1;
        const secondHeader = "<tr>" + groups.map((group) => group.columns.map((column, columnIndex) => {
          const colorStyle = column.color ? ` style="background:${esc(column.color)}"` : "";
          const boundaryClass = `${columnIndex === 0 ? " metric-col-start" : ""}${columnIndex === group.columns.length - 1 ? " metric-col-end" : ""}`;
          const header = `<th class="result-subhead${boundaryClass}" data-sortable="1" data-sort-column="${colIndex}"${colorStyle}><span title="${esc(column.label)}">${esc(column.label)}</span></th>`;
          colIndex += 1;
          return header;
        }).join("")).join("") + "</tr>";
        const body = rows.map((row) => {
          const cells = [dataCell(row.stub, "sticky-col row-head", row.stubSort || row.stub, row.attrs || {})];
          row.groups.forEach((group) => {
            group.cells.forEach((cell, cellIndex) => {
              const boundaryClass = `${cellIndex === 0 ? " metric-col-start" : ""}${cellIndex === group.cells.length - 1 ? " metric-col-end" : ""}`;
              cells.push(dataCell(cell.value, `${cell.klass || ""}${boundaryClass}`, cell.sortValue, cell.attrs || {}));
            });
          });
          return `<tr>${cells.join("")}</tr>`;
        }).join("");
        return `<div class="compare-table-shell"><table class="sortable compare-wide-table"${tableAttrs}><thead>${firstHeader}${secondHeader}</thead><tbody>${body}</tbody></table></div>`;
      }

      function runNames(results) {
        return Object.keys(results || {});
      }

      function rowByLabel(rows) {
        return Object.fromEntries((rows || []).map((row) => [String(row.label ?? "-"), row]));
      }

      function compactAxisLabel(scenario, label) {
        if (scenario === "vel_grid") return label.replace(" yaw=", "/y").replace("vx=", "vx");
        if (scenario === "arm_sweep") return label.replace("intensity=", "a");
        if (scenario === "body_pose") {
          return label.replace("pitch=", "p").replace("roll=", "r").replace("h_target=", "h").replace("rad", "").replace("m", "");
        }
        if (scenario === "gait") {
          return label.replace("gait_freq=", "f").replace("swing_h=", "sw").replace("stance_w=", "w").replace("Hz", "").replace("m", "");
        }
        return label;
      }

      function flattenSelected(selected) {
        const flattened = {};
        selected.forEach((series) => {
          flattened[series.name] = series.results || {};
        });
        return flattened;
      }

      function selectedResults() {
        const selected = [];
        document.querySelectorAll(".compare-toggle").forEach((input) => {
          if (input.checked) selected.push(COMPARE_RESULTS[Number(input.dataset.compareIndex)]);
        });
        return selected.filter(Boolean);
      }

      function renderCards(results) {
        const compact = runNames(results).length > 3;
        return Object.entries(results).map(([runName, scenarios]) => {
          const points = Object.values(scenarios || {}).reduce((sum, rows) => sum + rows.length, 0);
          const scenarioText = scenarioNames({ [runName]: scenarios }).map((name) => SCENARIO_TITLES[name] || name).join(", ");
          const allRows = Object.values(scenarios || {}).flat();
          const stats = [
            ["points", fmt(points, 0)],
            ["fall rate mean", fmt(metricAverage(allRows, "fall_rate"))],
            ["vx RMSE mean", fmt(metricAverage(allRows, "lin_vel_x_rmse"))],
            ["yaw RMSE mean", fmt(metricAverage(allRows, "ang_vel_yaw_rmse"))],
            ["lin reward mean", fmt(metricAverage(allRows, "tracking_lin_vel_reward"))],
            ["yaw reward mean", fmt(metricAverage(allRows, "tracking_ang_vel_reward"))],
            ["base height mean", fmt(metricAverage(allRows, "base_height_mean"))],
            ["max torque mean", fmt(metricAverage(allRows, "max_torque_mean"))],
          ];
          const visibleStats = compact ? stats.slice(0, 4) : stats;
          return `<div class="card${compact ? " card-compact" : ""}"><h3>${esc(runName)}</h3><p>${esc(scenarioText)}</p><div class="card-stats">` +
            visibleStats.map(([label, value]) => `<div class="stat"><span>${value}</span><label>${esc(label)}</label></div>`).join("") +
            (compact ? "" : '<div class="stat stat-empty" aria-hidden="true"></div>') +
            "</div></div>";
        }).join("");
      }

      function renderSummaryCompare(results, scenarios) {
        const names = runNames(results);
        const groups = [
          { label: "points", columns: names.map((name, index) => ({ label: name, color: resultColor(name, index) })) },
          ...PRIMARY_METRICS.map(([, label]) => ({
            label,
            columns: names.map((name, index) => ({ label: name, color: resultColor(name, index) })),
          })),
        ];
        const rows = scenarios.map((scenario) => {
          const groupsForRow = [];
          groupsForRow.push({
            cells: names.map((name) => {
              const count = (results[name]?.[scenario] || []).length;
              return { value: fmt(count, 0), sortValue: count };
            }),
          });
          PRIMARY_METRICS.forEach(([metric]) => {
            const values = names.map((name) => metricAverage(results[name]?.[scenario] || [], metric)).filter(isNumber);
            groupsForRow.push({
              cells: names.map((name) => {
                const value = metricAverage(results[name]?.[scenario] || [], metric);
                const rank = rankForValue(values, value, metric);
                return {
                  value: metricCellValue(value, rank),
                  klass: rank ? `cell-${rank}` : "",
                  sortValue: isNumber(value) ? value : "",
                };
              }),
            });
          });
          return {
            stub: `<strong>${esc(scenario === HOME_SCENARIO ? "Mini Report" : SCENARIO_TITLES[scenario] || scenario)}</strong>`,
            stubSort: scenario === HOME_SCENARIO ? "Mini Report" : SCENARIO_TITLES[scenario] || scenario,
            groups: groupsForRow,
          };
        });
        return groupedTable("scope", groups, rows, { kind: "summary" });
      }

      function renderSummary(results, scenarios) {
        const intro = '<p class="summary-note">Metric columns are means over the benchmark test points in each scope.</p>';
        return intro + renderSummaryCompare(results, scenarios);
      }

      function nestedGet(data, path) {
        return path.split(".").reduce((current, part) => (
          current && typeof current === "object" && part in current ? current[part] : undefined
        ), data);
      }

      function renderMetadata(selected) {
        if (!selected.length || !selected.some((entry) => entry.metadata && Object.keys(entry.metadata).length)) return "";
        if (selected.length === 1) {
          const metadata = selected[0].metadata || {};
          const rows = METADATA_FIELDS.map(([label, path, fallback]) => {
            let value = nestedGet(metadata, path);
            if (value === undefined && fallback) value = metadata[fallback];
            if (value === undefined || value === null) return "";
            if (Array.isArray(value)) value = value.join(", ");
            return `<tr><th>${esc(label)}</th><td>${esc(value)}</td></tr>`;
          }).join("");
          return rows ? `<section class="panel metadata-panel" id="metadata"><h2>Metadata</h2><table><tbody>${rows}</tbody></table></section>` : "";
        }
        const headers = ["field"].concat(selected.map((entry) => entry.name));
        const rows = METADATA_FIELDS.map(([label, path, fallback]) => {
          const values = selected.map((entry) => {
            let value = nestedGet(entry.metadata || {}, path);
            if (value === undefined && fallback) value = (entry.metadata || {})[fallback];
            if (Array.isArray(value)) value = value.join(", ");
            return value === undefined || value === null ? "-" : String(value);
          });
          if (values.every((value) => value === "-")) return null;
          const unique = new Set(values);
          return [[esc(label), ""]].concat(values.map((value) => [esc(value), unique.size > 1 ? "metadata-diff" : ""]));
        }).filter(Boolean);
        return `<section class="panel metadata-panel" id="metadata"><h2>Metadata</h2>${table(headers, rows)}</section>`;
      }

      function scenarioPlotData(results, scenario, metricDefs) {
        return {
          metrics: metricDefs.map(([label, key]) => ({ label, key })),
          series: Object.entries(results).map(([runName, scenarioMap], index) => {
            const rows = scenarioMap[scenario] || [];
            return {
              name: runName,
              color: resultColor(runName, index),
              labels: rows.map((row) => String(row.label ?? "-")),
              axis_labels: rows.map((row) => compactAxisLabel(scenario, String(row.label ?? "-"))),
              values: Object.fromEntries(metricDefs.map(([, metric]) => [
                metric,
                rows.map((row) => isNumber(row[metric]) ? row[metric] : null),
              ])),
            };
          }).filter((item) => item.labels.length),
        };
      }

      function renderScenarioPlot(results, scenario, metricDefs) {
        const buttons = metricDefs.map(([label, metric], index) => {
          const active = index === 0 ? " active" : "";
          return `<button type="button" class="metric-tab${active}" data-metric="${esc(metric)}">${esc(label)}</button>`;
        }).join("");
        const data = esc(JSON.stringify(scenarioPlotData(results, scenario, metricDefs)));
        return `<div class="metric-plot" data-scenario="${esc(scenario)}" data-plot="${data}"><div class="metric-tabs">${buttons}</div><div class="metric-chart" aria-label="metric chart"></div></div>`;
      }

      function renderScenarioCompare(results, scenario, metricDefs) {
        const names = runNames(results);
        const groups = metricDefs.map(([label, metric]) => ({
          label,
          metric,
          columns: names.map((name, index) => ({ label: name, result: name, color: resultColor(name, index) })),
        }));
        const rowMaps = Object.fromEntries(names.map((name) => [name, rowByLabel(results[name]?.[scenario] || [])]));
        const labels = new Set();
        names.forEach((name) => {
          Object.keys(rowMaps[name]).forEach((label) => {
            labels.add(label);
          });
        });
        const heatMetrics = metricDefs.map(([, metric]) => metric).filter(metricHasHeatmap);
        const heatRanges = Object.fromEntries(heatMetrics.map((metric) => {
          const values = names.flatMap((name) => finiteValues(results[name]?.[scenario] || [], metric));
          return [metric, {
            min: values.length ? Math.min(...values) : 0,
            max: values.length ? Math.max(...values) : 0,
          }];
        }));
        const rows = Array.from(labels).map((label, pointIndex) => ({
          stub: esc(label),
          stubSort: label,
          attrs: {
            "data-scenario": scenario,
            "data-point-index": pointIndex,
            "data-point-label": label,
          },
          groups: metricDefs.map(([, metric]) => {
            const metricValues = names
              .map((name) => rowMaps[name][label]?.[metric])
              .filter(isNumber);
            return {
              cells: names.map((name) => {
                const row = rowMaps[name][label] || {};
                const rawValue = row[metric];
                const rank = rankForValue(metricValues, rawValue, metric);
                const attrs = {
                  "data-scenario": scenario,
                  "data-metric": metric,
                  "data-point-index": pointIndex,
                  "data-point-label": label,
                  "data-result": name,
                };
                if (heatMetrics.includes(metric)) {
                  const range = heatRanges[metric];
                  const style = heatValue(rawValue, range.min, range.max, metricPrefersHigher(metric));
                  return {
                    value: metricCellValue(rawValue, rank, style),
                    klass: `metric-cell${rank ? ` cell-${rank}` : ""}`,
                    sortValue: isNumber(rawValue) ? rawValue : "",
                    attrs,
                  };
                }
                return {
                  value: metricCellValue(rawValue, rank),
                  klass: rank ? `cell-${rank}` : "",
                  sortValue: isNumber(rawValue) ? rawValue : "",
                  attrs,
                };
              }),
            };
          }),
        }));
        return groupedTable("test point", groups, rows, { kind: "scenario", scenario });
      }

      function renderScenarioSections(results, scenarios) {
        return scenarios.map((scenario) => {
          const metricDefs = DETAIL_METRICS[scenario];
          if (!metricDefs) return "";
          const tables = renderScenarioCompare(results, scenario, metricDefs);
          if (!tables) return "";
          return `<section class="panel" id="detail-${esc(scenario)}"><h2>${esc(SCENARIO_TITLES[scenario] || scenario)}</h2>` +
            renderScenarioPlot(results, scenario, metricDefs) + tables + "</section>";
        }).join("");
      }

      function renderHome() {
        const previewTable = renderScenarioCompare(HOME_RESULTS, HOME_SCENARIO, HOME_METRICS);
        const previewPlot = renderScenarioPlot(HOME_RESULTS, HOME_SCENARIO, HOME_METRICS);
        return `
          <section class="cards home-cards" id="home-overview">
            <div class="card home-card"><h3>1. Pick results</h3><p>Use the left Results panel to open one run, compare several runs, or select everything.</p><div class="home-arrow">Results -> report</div></div>
            <div class="card home-card"><h3>2. Compare metrics</h3><p>Metric groups share test points, so each row compares the same condition across results.</p><div class="home-arrow">metric -> sub-columns</div></div>
            <div class="card home-card"><h3>3. Link chart and table</h3><p>Hover the chart or table to highlight the same 1xN result group in both views.</p><div class="home-arrow">chart <-> table</div></div>
          </section>
          <section class="panel home-panel" id="home-pick">
            <h2>Pick Results</h2>
            <div class="home-steps">
              <div><strong>Select all</strong><span>Compare every report currently available in this results folder.</span></div>
              <div><strong>Clear all</strong><span>Return here without leaving the page.</span></div>
              <div><strong>Checkboxes</strong><span>Build a focused comparison set manually.</span></div>
              <div><strong>Result name</strong><span>Open one report in the same workspace.</span></div>
            </div>
          </section>
          <section class="panel home-panel" id="home-summary">
            <h2>Summary Preview</h2>
            <p class="summary-note">This mini report uses example values only. It shows the same grouped layout used by real benchmark data.</p>
            ${renderSummaryCompare(HOME_RESULTS, [HOME_SCENARIO])}
          </section>
          <section class="panel home-panel" id="home-chart">
            <h2>Chart Interaction</h2>
            <p class="summary-note">Hover or click a point. The vertical guide chooses a test point and highlights the matching result cells.</p>
            ${previewPlot}
          </section>
          <section class="panel home-panel" id="home-table">
            <h2>Metric Table</h2>
            <p class="summary-note">Best and worst are computed inside each test-point metric group. The outline marks the 1xN comparison slice.</p>
            ${previewTable}
          </section>`;
      }

      function bindSortableTables() {
        document.querySelectorAll("table.sortable").forEach((tableElement) => {
          const headers = Array.from(tableElement.querySelectorAll("th[data-sortable='1']"));
          headers.forEach((header) => {
            header.addEventListener("click", () => {
              const tbody = tableElement.tBodies[0];
              const rows = Array.from(tbody.rows);
              const direction = header.dataset.direction === "asc" ? "desc" : "asc";
              headers.forEach((item) => delete item.dataset.direction);
              header.dataset.direction = direction;
              const colIndex = Number(header.dataset.sortColumn ?? header.cellIndex);
              rows.sort((a, b) => {
                const av = a.cells[colIndex]?.dataset.sortValue ?? a.cells[colIndex]?.textContent.trim() ?? "";
                const bv = b.cells[colIndex]?.dataset.sortValue ?? b.cells[colIndex]?.textContent.trim() ?? "";
                const an = Number(av);
                const bn = Number(bv);
                const bothNumeric = Number.isFinite(an) && Number.isFinite(bn);
                const cmp = bothNumeric ? an - bn : av.localeCompare(bv);
                return direction === "asc" ? cmp : -cmp;
              });
              rows.forEach((row) => tbody.appendChild(row));
            });
          });
        });
      }

      function finite(values) {
        return values.filter((value) => Number.isFinite(value));
      }

      function resetLinkedHighlights(scope = document) {
        scope.querySelectorAll(".linked-highlight, .linked-group, .linked-start, .linked-middle, .linked-end, .row-linked, .linked-pinned").forEach((item) => {
          item.classList.remove("linked-highlight", "linked-group", "linked-start", "linked-middle", "linked-end", "row-linked", "linked-pinned");
        });
      }

      function clearLinkedHighlights(force = false) {
        if (pinnedInteraction && !force) return;
        pinnedInteraction = null;
        resetLinkedHighlights();
      }

      function centerMetricGroup(cells) {
        if (!cells.length) return;
        const shell = cells[0].closest(".compare-table-shell");
        if (!shell || shell.scrollWidth <= shell.clientWidth + 2) return;
        const shellRect = shell.getBoundingClientRect();
        const firstRect = cells[0].getBoundingClientRect();
        const lastRect = cells[cells.length - 1].getBoundingClientRect();
        if (firstRect.left >= shellRect.left && lastRect.right <= shellRect.right) return;
        const groupCenter = (firstRect.left + lastRect.right) / 2 - shellRect.left + shell.scrollLeft;
        const nextLeft = Math.max(0, groupCenter - shell.clientWidth / 2);
        shell.scrollTo({ left: nextLeft, behavior: "smooth" });
      }

      function centerMetricColumn(scenario, metricKey) {
        const selector = `[data-scenario="${cssEscape(scenario)}"][data-metric="${cssEscape(metricKey)}"]`;
        const cells = Array.from(document.querySelectorAll(selector));
        if (!cells.length) return;
        const firstPoint = cells[0].dataset.pointIndex;
        centerMetricGroup(cells.filter((cell) => cell.dataset.pointIndex === firstPoint));
      }

      function setLinkedHighlight(scenario, metricKey, pointIndex, pinned = false, options = {}) {
        pinnedInteraction = pinned ? { scenario, metricKey, pointIndex } : null;
        resetLinkedHighlights();
        const selector = `[data-scenario="${cssEscape(scenario)}"][data-metric="${cssEscape(metricKey)}"][data-point-index="${pointIndex}"]`;
        const cells = Array.from(document.querySelectorAll(selector));
        cells.forEach((cell, index) => {
          const positionClass = index === 0 ? "linked-start" : index === cells.length - 1 ? "linked-end" : "linked-middle";
          cell.classList.add("linked-highlight", "linked-group", positionClass);
          if (cells.length === 1) cell.classList.add("linked-end");
          if (pinned) cell.classList.add("linked-pinned");
        });
        if (options.center) centerMetricGroup(cells);
        document.querySelectorAll(`[data-scenario="${cssEscape(scenario)}"][data-point-index="${pointIndex}"]`).forEach((cell) => {
          if (cell.classList.contains("sticky-col")) cell.classList.add("row-linked");
        });
      }

      function clearChartHighlights(plot) {
        plot.querySelectorAll(".chart-crosshair, .chart-tooltip-group").forEach((item) => item.remove());
        plot.querySelectorAll(".chart-point.is-active").forEach((item) => item.classList.remove("is-active"));
      }

      function renderMetricPlot(plot, metricKey) {
        const data = JSON.parse(plot.dataset.plot || "{}");
        const metric = (data.metrics || []).find((item) => item.key === metricKey) || (data.metrics || [])[0];
        const chart = plot.querySelector(".metric-chart");
        if (!metric || !chart) return;

        const series = (data.series || []).map((item, i) => ({
          name: item.name,
          labels: item.labels || [],
          axisLabels: item.axis_labels || item.labels || [],
          values: (item.values && item.values[metric.key] ? item.values[metric.key] : []).map((value) =>
            value === null ? NaN : Number(value)
          ),
          color: item.color || resultColor(item.name, i),
        }));
        const allValues = series.flatMap((item) => finite(item.values));
        if (!allValues.length) {
          chart.innerHTML = "<p class='summary-note'>No numeric values for this metric.</p>";
          return;
        }

        let yMin = Math.min(...allValues);
        let yMax = Math.max(...allValues);
        if (yMin === yMax) {
          yMin -= 1;
          yMax += 1;
        }
        const pad = (yMax - yMin) * 0.08;
        yMin -= pad;
        yMax += pad;

        const width = 820;
        const height = 280;
        const margin = { left: 58, right: 24, top: series.length > 1 ? 34 : 18, bottom: 58 };
        const innerW = width - margin.left - margin.right;
        const innerH = height - margin.top - margin.bottom;
        const maxPoints = Math.max(...series.map((item) => item.values.length));
        const x = (index) => margin.left + (maxPoints <= 1 ? innerW / 2 : (index / (maxPoints - 1)) * innerW);
        const y = (value) => margin.top + (1 - (value - yMin) / (yMax - yMin)) * innerH;
        const labels = series[0]?.labels || [];
        const axisLabels = series[0]?.axisLabels || labels;
        const labelStep = Math.max(1, Math.ceil(maxPoints / 8));
        const ticks = [0, 0.25, 0.5, 0.75, 1].map((t) => yMin + t * (yMax - yMin));

        const grid = ticks.map((tick) => {
          const yy = y(tick);
          return `<line class="chart-grid" x1="${margin.left}" y1="${yy}" x2="${width - margin.right}" y2="${yy}"></line>` +
            `<text class="chart-label" x="${margin.left - 8}" y="${yy + 4}" text-anchor="end">${fmt(tick)}</text>`;
        }).join("");

        const xLabels = Array.from({ length: maxPoints }, (_, i) => {
          if (i % labelStep !== 0 && i !== maxPoints - 1) return "";
          const label = axisLabels[i] || labels[i] || String(i + 1);
          return `<text class="chart-label" x="${x(i)}" y="${height - 16}" text-anchor="middle">` +
            `${esc(label)}<title>${esc(labels[i] || label)}</title></text>`;
        }).join("");

        const paths = series.map((item) => {
          const points = item.values
            .map((value, i) => Number.isFinite(value) ? `${x(i)},${y(value)}` : null)
            .filter(Boolean);
          const circles = item.values.map((value, i) => {
            if (!Number.isFinite(value)) return "";
            return `<circle class="chart-point" data-point-index="${i}" data-result="${esc(item.name)}" cx="${x(i)}" cy="${y(value)}" r="4" fill="${item.color}">` +
              `<title>${esc(item.name)} | ${esc(labels[i] || i)} | ${esc(metric.label)}: ${fmt(value)}</title></circle>`;
          }).join("");
          return `<polyline class="chart-line" stroke="${item.color}" points="${points.join(" ")}"></polyline>${circles}`;
        }).join("");

        const legend = series.length <= 1 ? "" : series.map((item, i) => {
          const lx = margin.left + i * 170;
          return `<circle cx="${lx}" cy="18" r="5" fill="${item.color}"></circle>` +
            `<text class="chart-legend" x="${lx + 9}" y="22">${esc(item.name)}</text>`;
        }).join("");

        chart.innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(metric.label)}">` +
          legend +
          grid +
          `<line class="chart-axis" x1="${margin.left}" y1="${margin.top + innerH}" x2="${width - margin.right}" y2="${margin.top + innerH}"></line>` +
          `<line class="chart-axis" x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${margin.top + innerH}"></line>` +
          paths +
          xLabels +
          `</svg>`;

        const svg = chart.querySelector("svg");
        const scenario = plot.dataset.scenario || "";
        const state = { pinnedIndex: null };

        function renderInteraction(pointIndex, pinned = false, options = {}) {
          if (!Number.isFinite(pointIndex) || pointIndex < 0 || pointIndex >= maxPoints) return;
          clearChartHighlights(plot);
          setLinkedHighlight(scenario, metric.key, pointIndex, pinned, options);
          const xx = x(pointIndex);
          svg.querySelectorAll(`.chart-point[data-point-index="${pointIndex}"]`).forEach((point) => {
            point.classList.add("is-active");
          });
          const values = series.map((item) => ({
            name: item.name,
            color: item.color,
            value: item.values[pointIndex],
          })).filter((item) => Number.isFinite(item.value));
          const titleText = `${metric.label} | ${labels[pointIndex] || pointIndex}`;
          const tooltipW = Math.max(260, Math.min(390, 36 + titleText.length * 6.4));
          const rowH = 17;
          const tooltipH = 30 + values.length * rowH;
          const tx = xx > width - margin.right - tooltipW ? Math.max(margin.left + 8, xx - tooltipW - 14) : xx + 14;
          const ty = Math.max(28, Math.min(height - tooltipH - 8, margin.top + 8));
          const valueRows = values.map((item, i) => {
            const yy = ty + 32 + i * rowH;
            return `<circle cx="${tx + 12}" cy="${yy - 4}" r="4" fill="${item.color}"></circle>` +
              `<text class="chart-tooltip-text" x="${tx + 22}" y="${yy}">${esc(item.name)}: ${fmt(item.value)}</text>`;
          }).join("");
          const tooltip = `<g class="chart-tooltip-group">` +
            `<line class="chart-crosshair" x1="${xx}" y1="${margin.top}" x2="${xx}" y2="${margin.top + innerH}"></line>` +
            `<rect class="chart-tooltip" fill-opacity="0.82" x="${tx}" y="${ty}" width="${tooltipW}" height="${tooltipH}" rx="6"></rect>` +
            `<text class="chart-tooltip-title" x="${tx + 10}" y="${ty + 18}">${esc(titleText)}</text>` +
            valueRows +
            `</g>`;
          svg.insertAdjacentHTML("beforeend", tooltip);
        }

        function eventToPointIndex(event) {
          const rect = svg.getBoundingClientRect();
          const viewX = ((event.clientX - rect.left) / rect.width) * width;
          const clamped = Math.max(margin.left, Math.min(width - margin.right, viewX));
          if (maxPoints <= 1) return 0;
          return Math.round(((clamped - margin.left) / innerW) * (maxPoints - 1));
        }

        svg.addEventListener("mousemove", (event) => {
          if (state.pinnedIndex !== null) return;
          renderInteraction(eventToPointIndex(event), false, { center: true });
        });
        svg.addEventListener("mouseleave", () => {
          if (state.pinnedIndex !== null) return;
          clearChartHighlights(plot);
          clearLinkedHighlights();
        });
        svg.addEventListener("click", (event) => {
          const index = eventToPointIndex(event);
          state.pinnedIndex = state.pinnedIndex === index ? null : index;
          if (state.pinnedIndex === null) {
            clearChartHighlights(plot);
            clearLinkedHighlights(true);
          } else {
            renderInteraction(state.pinnedIndex, true, { center: true });
          }
        });
        plot.activatePoint = (pointIndex, pinned = false, options = {}) => {
          state.pinnedIndex = pinned ? pointIndex : null;
          renderInteraction(pointIndex, pinned, options);
        };
        plot.currentPointIndex = () => state.pinnedIndex;
      }

      function bindMetricPlots() {
        document.querySelectorAll(".metric-plot").forEach((plot) => {
          const firstButton = plot.querySelector(".metric-tab");
          if (firstButton) renderMetricPlot(plot, firstButton.dataset.metric);
          plot.querySelectorAll(".metric-tab").forEach((button) => {
            button.addEventListener("click", () => {
              plot.querySelectorAll(".metric-tab").forEach((item) => item.classList.remove("active"));
              button.classList.add("active");
              const pinnedIndex = typeof plot.currentPointIndex === "function" ? plot.currentPointIndex() : null;
              clearLinkedHighlights();
              renderMetricPlot(plot, button.dataset.metric);
              if (pinnedIndex !== null) {
                plot.activatePoint?.(pinnedIndex, true, { center: true });
              } else {
                centerMetricColumn(plot.dataset.scenario || "", button.dataset.metric);
              }
            });
          });
        });
      }

      function bindTableChartLinks() {
        document.querySelectorAll("td[data-scenario][data-metric][data-point-index]").forEach((cell) => {
          const scenario = cell.dataset.scenario;
          const metric = cell.dataset.metric;
          const pointIndex = Number(cell.dataset.pointIndex);
          cell.addEventListener("mouseenter", () => {
            const plot = document.querySelector(`.metric-plot[data-scenario="${cssEscape(scenario)}"]`);
            const activeMetric = plot?.querySelector(".metric-tab.active")?.dataset.metric;
            if (!plot || activeMetric !== metric) {
              setLinkedHighlight(scenario, metric, pointIndex);
              return;
            }
            plot.activatePoint?.(pointIndex, false, { center: false });
          });
          cell.addEventListener("mouseleave", () => {
            const plot = document.querySelector(`.metric-plot[data-scenario="${cssEscape(scenario)}"]`);
            if (pinnedInteraction) return;
            if (plot) clearChartHighlights(plot);
            clearLinkedHighlights(true);
          });
          cell.addEventListener("click", () => {
            const plot = document.querySelector(`.metric-plot[data-scenario="${cssEscape(scenario)}"]`);
            const activeMetric = plot?.querySelector(".metric-tab.active")?.dataset.metric;
            if (!plot || activeMetric !== metric) {
              setLinkedHighlight(scenario, metric, pointIndex, true);
              return;
            }
            plot.activatePoint?.(pointIndex, true, { center: false });
          });
        });
      }

      function navLink(label, href) {
        return `<a href="${esc(href)}">${esc(label)}</a>`;
      }

      function renderReportNav(scenarios, hasMetadata, isHome) {
        const container = document.getElementById("report-nav-links");
        if (!container) return;
        if (isHome) {
          container.innerHTML = [
            navLink("Overview", "#home-overview"),
            navLink("Pick Results", "#home-pick"),
            navLink("Summary Preview", "#home-summary"),
            navLink("Chart Interaction", "#home-chart"),
            navLink("Metric Table", "#home-table"),
          ].join("");
          return;
        }
        const links = [navLink("Summary", "#summary")];
        if (hasMetadata) links.push(navLink("Metadata", "#metadata"));
        scenarios.forEach((scenario) => {
          if (DETAIL_METRICS[scenario]) links.push(navLink(SCENARIO_TITLES[scenario] || scenario, `#detail-${scenario}`));
        });
        container.innerHTML = links.join("");
      }

      function syncResultSelectionState() {
        document.querySelectorAll(".compare-result").forEach((row) => {
          const input = row.querySelector(".compare-toggle");
          const active = Boolean(input && input.checked);
          row.classList.toggle("active", active);
          row.querySelector(".result-link")?.classList.toggle("active", active);
        });
      }

      function renderReport() {
        const selected = selectedResults();
        const results = flattenSelected(selected);
        const scenarios = scenarioNames(results);
        const isHome = !Object.keys(results).length;
        syncResultSelectionState();
        const main = document.getElementById("report-main");
        if (!main) return;
        renderReportNav(scenarios, selected.some((entry) => entry.metadata && Object.keys(entry.metadata).length), isHome);
        clearLinkedHighlights(true);
        if (isHome) {
          main.innerHTML = renderHome();
        } else {
          main.innerHTML =
            `<section class="cards">${renderCards(results)}</section>` +
            `<section class="panel" id="summary"><h2>Summary - Metric Mean</h2>${renderSummary(results, scenarios)}</section>` +
            renderMetadata(selected) +
            renderScenarioSections(results, scenarios);
        }
        bindSortableTables();
        bindMetricPlots();
        bindTableChartLinks();
      }

      function bindReportNavLinks() {
        const nav = document.querySelector(".side-nav");
        if (!nav) return;
        nav.addEventListener("click", (event) => {
          const link = event.target.closest(".nav-report a[href^='#']");
          if (!link) return;
          const href = link.getAttribute("href");
          if (!href || href === "#") return;
          const target = document.querySelector(href);
          if (!target) return;
          event.preventDefault();
          target.scrollIntoView({ behavior: "smooth", block: "start" });
          history.pushState(null, "", href);
        });
      }

      document.querySelectorAll(".compare-toggle").forEach((input) => {
        input.addEventListener("change", renderReport);
      });

      const selectAllButton = document.getElementById("select-all-results");
      if (selectAllButton) {
        selectAllButton.addEventListener("click", () => {
          document.querySelectorAll(".compare-toggle").forEach((input) => { input.checked = true; });
          renderReport();
          document.getElementById("report-main")?.scrollIntoView({ behavior: "smooth", block: "start" });
        });
      }

      const clearAllButton = document.getElementById("clear-all-results");
      if (clearAllButton) {
        clearAllButton.addEventListener("click", () => {
          document.querySelectorAll(".compare-toggle").forEach((input) => { input.checked = false; });
          renderReport();
          document.getElementById("home-overview")?.scrollIntoView({ behavior: "smooth", block: "start" });
        });
      }

      document.querySelectorAll(".result-link").forEach((link) => {
        link.addEventListener("click", (event) => {
          event.preventDefault();
          document.querySelectorAll(".compare-toggle").forEach((input) => { input.checked = false; });
          const input = link.closest(".compare-result")?.querySelector(".compare-toggle");
          if (input) input.checked = true;
          renderReport();
          document.getElementById("report-main")?.scrollIntoView({ behavior: "smooth", block: "start" });
        });
      });
      renderReport();
      bindReportNavLinks();
    })();
  </script>"""
        .replace("__COMPARE_RESULTS__", _json_for_script(comparison_series))
        .replace("__PRIMARY_METRICS__", _json_for_script(PRIMARY_METRICS))
        .replace("__DETAIL_METRICS__", _json_for_script(DETAIL_METRICS))
        .replace("__SCENARIO_TITLES__", _json_for_script(SCENARIO_TITLES))
        .replace("__SCENARIO_ORDER__", _json_for_script(SCENARIO_ORDER))
    )


def _render_html(results: Dict[str, Dict[str, List[dict]]], source: Path, output_path: Path) -> str:
    scenarios = _scenario_names(results)
    metadata = _load_metadata(source)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    comparison_entries = _comparison_result_payload(source, output_path)
    comparison_series = _comparison_series_payload(comparison_entries)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RoboDuet Benchmark Report</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f3f5f8;
      --panel: #ffffff;
      --panel-2: #f8fafc;
      --header: #344054;
      --header-2: #475467;
      --text: #1f2933;
      --muted: #667085;
      --border: #d0d5dd;
      --accent: #0f766e;
      --accent-soft: #e6f4f1;
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
      background: linear-gradient(180deg, #ffffff, var(--panel-2));
      padding: 22px 24px;
      box-shadow: 0 1px 2px rgb(16 24 40 / 6%);
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
      background: var(--panel);
      box-shadow: 0 1px 2px rgb(16 24 40 / 5%);
    }}
    .nav-group h3 {{
      margin: 0 0 8px;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0;
    }}
    .nav-title-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 8px;
    }}
    .nav-title-row h3 {{
      margin: 0;
    }}
    .nav-actions {{
      display: flex;
      gap: 4px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }}
    .nav-action {{
      border: 1px solid var(--border);
      border-radius: 5px;
      background: #ffffff;
      color: var(--header);
      cursor: pointer;
      font: inherit;
      font-size: 11px;
      font-weight: 700;
      line-height: 1.2;
      padding: 4px 6px;
    }}
    .nav-action:hover {{
      background: var(--accent-soft);
      border-color: #b7d7d1;
      color: var(--accent);
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
      background: var(--accent-soft);
      color: var(--accent);
      text-decoration: none;
    }}
    .nav-group a.active {{
      background: var(--accent-soft);
      color: var(--accent);
    }}
    .nav-results {{
      max-height: 46vh;
      overflow-y: auto;
    }}
    .compare-result {{
      display: grid;
      grid-template-columns: auto minmax(0, 1fr);
      gap: 6px;
      align-items: start;
      border-radius: 6px;
    }}
    .compare-result input {{
      margin: 11px 0 0 2px;
    }}
    .compare-result.active {{
      background: var(--accent-soft);
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
      box-shadow: 0 1px 2px rgb(16 24 40 / 5%);
    }}
    .card {{ padding: 14px 16px; }}
    .card h3 {{ margin-top: 0; color: var(--header); }}
    .card p {{ color: var(--muted); margin-bottom: 12px; }}
    .card-stats {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }}
    .stat span {{ font-size: 22px; font-weight: 700; color: var(--accent); }}
    .stat label {{ display: block; color: var(--muted); }}
    .stat-empty {{ visibility: hidden; }}
    .panel {{ padding: 18px; margin-bottom: 18px; overflow: hidden; scroll-margin-top: 18px; }}
    .home-card p {{
      min-height: 42px;
    }}
    .home-arrow {{
      display: inline-block;
      border: 1px solid #b7d7d1;
      border-radius: 6px;
      background: var(--accent-soft);
      color: var(--accent);
      font-size: 12px;
      font-weight: 800;
      padding: 5px 7px;
    }}
    .home-panel h2 {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .home-steps {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 10px;
    }}
    .home-steps div {{
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--panel-2);
      padding: 12px;
    }}
    .home-steps strong {{
      display: block;
      color: var(--header);
      margin-bottom: 4px;
    }}
    .home-steps span {{
      color: var(--muted);
      font-size: 13px;
    }}
    table {{ width: 100%; border-collapse: collapse; min-width: 780px; }}
    th, td {{ padding: 9px 10px; border-bottom: 1px solid var(--border); text-align: right; white-space: nowrap; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ color: var(--muted); font-weight: 600; background: #fbfcfe; }}
    th[data-sortable="1"] {{ cursor: pointer; user-select: none; }}
    th[data-sortable="1"]::after {{ content: " ↕"; color: #98a2b3; font-weight: 400; }}
    th[data-direction="asc"]::after {{ content: " ↑"; color: var(--accent); }}
    th[data-direction="desc"]::after {{ content: " ↓"; color: var(--accent); }}
    .compare-table-shell {{
      overflow-x: auto;
      overflow-y: hidden;
      margin-top: 12px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #ffffff;
      isolation: isolate;
      box-shadow: inset 0 0 0 1px rgb(255 255 255 / 60%);
    }}
    .compare-wide-table {{
      min-width: 1120px;
      table-layout: auto;
    }}
    .compare-wide-table th {{
      vertical-align: bottom;
      color: #ffffff;
      font-weight: 750;
      background: var(--header);
    }}
    .compare-wide-table .metric-group {{
      text-align: center;
      border-left: 1px solid #667085;
      border-right: 1px solid #667085;
      background: var(--header);
    }}
    .compare-wide-table .metric-col-start {{
      border-left: 2px solid #667085;
    }}
    .compare-wide-table .metric-col-end {{
      border-right: 1px solid #98a2b3;
    }}
    .compare-wide-table thead tr:nth-child(2) th {{
      background: var(--header-2);
      color: #ffffff;
      font-weight: 700;
    }}
    .compare-wide-table thead tr:nth-child(2) th.result-subhead {{
      color: #ffffff;
      font-weight: 800;
      text-shadow: 0 1px 1px rgb(0 0 0 / 28%);
      box-shadow: inset 0 -2px 0 rgb(255 255 255 / 28%);
      max-width: 132px;
    }}
    .compare-wide-table thead tr:nth-child(2) th.result-subhead span {{
      display: block;
      max-width: 120px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      margin: 0 auto;
    }}
    .compare-wide-table .sticky-col {{
      position: sticky;
      left: 0;
      z-index: 4;
      background: #f2f4f7;
      color: var(--text);
      font-weight: 750;
      box-shadow: 2px 0 0 var(--border);
      min-width: 170px;
      max-width: 260px;
      white-space: normal;
    }}
    .compare-wide-table td.sticky-col {{ position: sticky; }}
    .compare-wide-table th.sticky-col {{
      z-index: 6;
      background: var(--header);
      color: #ffffff;
    }}
    .compare-wide-table td.sticky-col::after {{
      content: "";
      position: absolute;
      top: 0;
      right: -1px;
      bottom: 0;
      width: 1px;
      background: var(--border);
    }}
    .compare-wide-table td {{
      min-width: 92px;
      position: relative;
    }}
    .metadata-panel table {{ min-width: 0; }}
    .metadata-panel th {{ width: 220px; text-align: left; }}
    .metadata-panel td {{ text-align: left; white-space: normal; word-break: break-word; }}
    .metadata-panel td.metadata-diff {{ background: #fff7ed; }}
    .summary-note {{ color: var(--muted); margin: 0 0 12px; }}
    td.best {{ background: var(--best); color: #047857; font-weight: 650; }}
    td.worst {{ background: var(--worst); color: #b42318; }}
    td.metric-cell span, td.cell-best span, td.cell-worst span {{
      display: block;
      margin: -9px -10px;
      padding: 9px 10px;
      position: relative;
      z-index: 1;
    }}
    td.cell-best span, td.cell-worst span {{
      font-weight: 800;
      position: relative;
    }}
    td.cell-best {{
      background: #ecfdf3 !important;
      color: #027a48;
    }}
    td.cell-worst {{
      background: #fff1f3 !important;
      color: #b42318;
    }}
    td.cell-best span {{
      background: #ecfdf3 !important;
    }}
    td.cell-worst span {{
      background: #fff1f3 !important;
    }}
    .cell-rank-badge {{
      display: inline-block;
      margin-left: 6px;
      padding: 1px 4px;
      border-radius: 4px;
      font-size: 10px;
      font-style: normal;
      font-weight: 800;
      line-height: 1.2;
      text-transform: uppercase;
      vertical-align: 1px;
    }}
    td.cell-best .cell-rank-badge {{
      background: #039855;
      color: #ffffff;
    }}
    td.cell-worst .cell-rank-badge {{
      background: #d92d20;
      color: #ffffff;
    }}
    td.linked-group {{
      background: #ecfdf3 !important;
      position: relative;
      z-index: 2;
    }}
    td.linked-group::after {{
      content: "";
      position: absolute;
      inset: 0;
      z-index: 3;
      pointer-events: none;
      border-top: 2px solid var(--accent);
      border-bottom: 2px solid var(--accent);
    }}
    td.linked-start::after {{
      border-left: 2px solid var(--accent);
      border-radius: 5px 0 0 5px;
    }}
    td.linked-end::after {{
      border-right: 2px solid var(--accent);
      border-radius: 0 5px 5px 0;
    }}
    td.linked-start.linked-end::after {{
      border: 2px solid var(--accent);
      border-radius: 5px;
    }}
    td.linked-pinned.linked-group::after {{
      border-top-width: 3px;
      border-bottom-width: 3px;
      border-color: #0f766e;
    }}
    td.linked-pinned.linked-start::after {{
      border-left-width: 3px;
    }}
    td.linked-pinned.linked-end::after {{
      border-right-width: 3px;
    }}
    td.row-linked, .compare-wide-table tr.row-linked td.sticky-col {{
      outline: 2px solid var(--accent);
      outline-offset: -2px;
      background: #ecfdf3 !important;
    }}
    .metric-plot {{
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--panel-2);
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
      background: var(--accent-soft);
      color: var(--accent);
    }}
    .metric-chart {{
      min-height: 280px;
      overflow: hidden;
      position: relative;
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
    .chart-point.is-active {{ stroke: #111827; stroke-width: 2.5; }}
    .chart-crosshair {{ stroke: #475467; stroke-width: 1.2; stroke-dasharray: 4 4; pointer-events: none; }}
    .chart-legend {{ fill: var(--text); font-size: 12px; }}
    .chart-tooltip {{
      fill: #ffffff;
      fill-opacity: 0.82;
      stroke: var(--border);
      stroke-width: 1;
      filter: drop-shadow(0 4px 8px rgb(16 24 40 / 16%));
    }}
    .chart-tooltip-title {{ fill: var(--text); font-size: 12px; font-weight: 750; }}
    .chart-tooltip-text {{ fill: var(--text); font-size: 12px; font-weight: 650; }}
    .chart-tooltip-muted {{ fill: var(--muted); font-size: 11px; }}
    footer {{ color: var(--muted); padding: 0 32px 30px; max-width: 1520px; margin: 0 auto; }}
    @media (prefers-reduced-motion: reduce) {{
      html {{ scroll-behavior: auto; }}
    }}
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
    {_side_nav(source, output_path, scenarios, bool(metadata), comparison_series)}
    <main id="report-main">
      <section class="cards">{_candidate_cards(results)}</section>
      <section class="panel" id="summary"><h2>Summary - Metric Mean</h2>{_summary_table(results, scenarios)}</section>
      {_metadata_panel(metadata)}
      {_scenario_detail_sections(results, scenarios)}
    </main>
  </div>
  <footer>Heatmap colors are normalized within each candidate and scenario table; green indicates better relative values for highlighted metrics, red indicates worse relative values.</footer>
  {_report_script(comparison_series)}
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
        "loaded_results": results,
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
            results = entry["loaded_results"]
            report_path.write_text(_render_html(results, entry["results"], report_path), encoding="utf-8")
    index_path = root / "index.html"
    if entries:
        latest_results = entries[0]["results"]
        latest = entries[0]["loaded_results"]
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
