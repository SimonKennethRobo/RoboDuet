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
    "wbc_aggregate": "WBC Curriculum Aggregate",
    "wbc_trajectories": "WBC Per-Trajectory Results",
}

SCENARIO_ORDER = ["vel_grid", "arm_sweep", "body_pose", "gait", "wbc_aggregate", "wbc_trajectories"]

PRIMARY_METRICS = [
    ("ee_pos_rmse_m", "EE pos RMSE", "lower"),
    ("ee_rot_rmse_rad", "EE rot RMSE", "lower"),
    ("completion_rate", "completion rate", "higher"),
    ("lin_vel_xy_rmse", "xy RMSE", "lower"),
    ("lin_vel_x_rmse", "vx RMSE", "lower"),
    ("lin_vel_y_rmse", "vy RMSE", "lower"),
    ("ang_vel_yaw_rmse", "yaw RMSE", "lower"),
    ("fall_rate_height", "height fall rate", "lower"),
    ("fall_rate", "fall rate", "lower"),
    ("tracking_lin_vel_reward", "lin reward", "higher"),
    ("tracking_ang_vel_reward", "yaw reward", "higher"),
    ("base_height_mean", "base height", "neutral"),
    ("max_torque_mean", "max torque", "lower"),
]

DETAIL_METRICS = {
    "vel_grid": [
        ("xy RMSE", "lin_vel_xy_rmse"),
        ("vx RMSE", "lin_vel_x_rmse"),
        ("vy RMSE", "lin_vel_y_rmse"),
        ("yaw RMSE", "ang_vel_yaw_rmse"),
        ("height fall", "fall_rate_height"),
        ("lin reward", "tracking_lin_vel_reward"),
        ("yaw reward", "tracking_ang_vel_reward"),
        ("base height", "base_height_mean"),
        ("max torque", "max_torque_mean"),
    ],
    "arm_sweep": [
        ("xy RMSE", "lin_vel_xy_rmse"),
        ("vx RMSE", "lin_vel_x_rmse"),
        ("vy RMSE", "lin_vel_y_rmse"),
        ("yaw RMSE", "ang_vel_yaw_rmse"),
        ("height fall", "fall_rate_height"),
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
        ("xy RMSE", "lin_vel_xy_rmse"),
        ("vx RMSE", "lin_vel_x_rmse"),
        ("vy RMSE", "lin_vel_y_rmse"),
        ("base height", "base_height_mean"),
        ("height fall", "fall_rate_height"),
        ("fall rate", "fall_rate"),
    ],
    "gait": [
        ("freq RMSE Hz", "gait_freq_rmse_hz"),
        ("swing h RMSE m", "footswing_height_rmse_m"),
        ("stance w RMSE m", "stance_width_rmse_m"),
        ("stance l RMSE m", "stance_length_rmse_m"),
        ("contact force", "gait_contact_force_cost"),
        ("contact vel", "gait_contact_vel_cost"),
        ("clearance m", "foot_clearance_rmse_m"),
        ("raibert m", "raibert_rmse_m"),
        ("max torque", "max_torque_mean"),
        ("height fall", "fall_rate_height"),
        ("fall rate", "fall_rate"),
    ],
    "wbc_aggregate": [
        ("EE pos RMSE m", "ee_pos_rmse_m"),
        ("EE rot RMSE rad", "ee_rot_rmse_rad"),
        ("lateral error m", "d_lat_mean_m"),
        ("timing error m", "timing_err_mean_m"),
        ("progress", "progress_mean"),
        ("completion", "completion_rate"),
        ("fall rate", "fall_rate"),
        ("motor power W", "motor_power_mean_w"),
    ],
    "wbc_trajectories": [
        ("EE pos RMSE m", "ee_pos_rmse_m"),
        ("EE rot RMSE rad", "ee_rot_rmse_rad"),
        ("lateral error m", "d_lat_mean_m"),
        ("timing error m", "timing_err_mean_m"),
        ("progress", "progress_mean"),
        ("motor power W", "motor_power_mean_w"),
        ("span x m", "span_x_m"),
        ("span y m", "span_y_m"),
        ("span z m", "span_z_m"),
        ("curvature p90 rad/m", "curvature_p90_rad_m"),
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


def _preferred_result_rows(scenarios):
    if scenarios.get("wbc_trajectories"):
        return scenarios["wbc_trajectories"]
    return [row for scenario_rows in scenarios.values() for row in scenario_rows]


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
                "rows": rows,
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
        all_rows = _preferred_result_rows(scenarios)
        cards.append(
            '<div class="card">'
            f"<h3>{escape(run_name)}</h3>"
            f"<p>{escape(scenario_text)}</p>"
            '<div class="card-stats">'
            f'<div class="stat"><span>{points}</span><label>points</label></div>'
            f'<div class="stat"><span>{_fmt(_metric_average(all_rows, "fall_rate"))}</span><label>fall rate mean</label></div>'
            f'<div class="stat"><span>{_fmt(_metric_average(all_rows, "lin_vel_xy_rmse"))}</span><label>xy RMSE mean</label></div>'
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
      const HOME_HEATMAP_SCENARIO = "home_heatmap_preview";
      const HOME_METRICS = [
        ["vx RMSE", "lin_vel_x_rmse"],
        ["yaw RMSE", "ang_vel_yaw_rmse"],
        ["lin reward", "tracking_lin_vel_reward"],
      ];
      const HOME_HEATMAP_METRICS = [
        ["xy RMSE", "lin_vel_xy_rmse"],
        ["height fall", "fall_rate_height"],
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
      const HOME_HEATMAP_RESULTS = {
        "demo A": {
          [HOME_HEATMAP_SCENARIO]: [
            { label: "vx=-1.0 vy=-1.0 yaw=0.0", cmd_x: -1, cmd_y: -1, cmd_yaw: 0, lin_vel_xy_rmse: 0.42, fall_rate_height: 0 },
            { label: "vx=+0.0 vy=-1.0 yaw=0.0", cmd_x: 0, cmd_y: -1, cmd_yaw: 0, lin_vel_xy_rmse: 0.31, fall_rate_height: 0 },
            { label: "vx=+1.0 vy=-1.0 yaw=0.0", cmd_x: 1, cmd_y: -1, cmd_yaw: 0, lin_vel_xy_rmse: 0.49, fall_rate_height: 0 },
            { label: "vx=-1.0 vy=+0.0 yaw=0.0", cmd_x: -1, cmd_y: 0, cmd_yaw: 0, lin_vel_xy_rmse: 0.34, fall_rate_height: 0 },
            { label: "vx=+0.0 vy=+0.0 yaw=0.0", cmd_x: 0, cmd_y: 0, cmd_yaw: 0, lin_vel_xy_rmse: 0.12, fall_rate_height: 0 },
            { label: "vx=+1.0 vy=+0.0 yaw=0.0", cmd_x: 1, cmd_y: 0, cmd_yaw: 0, lin_vel_xy_rmse: 0.28, fall_rate_height: 0 },
            { label: "vx=-1.0 vy=+1.0 yaw=0.0", cmd_x: -1, cmd_y: 1, cmd_yaw: 0, lin_vel_xy_rmse: 0.53, fall_rate_height: 0.02 },
            { label: "vx=+0.0 vy=+1.0 yaw=0.0", cmd_x: 0, cmd_y: 1, cmd_yaw: 0, lin_vel_xy_rmse: 0.36, fall_rate_height: 0 },
            { label: "vx=+1.0 vy=+1.0 yaw=0.0", cmd_x: 1, cmd_y: 1, cmd_yaw: 0, lin_vel_xy_rmse: 0.58, fall_rate_height: 0.03 },
          ],
        },
        "demo B": {
          [HOME_HEATMAP_SCENARIO]: [
            { label: "vx=-1.0 vy=-1.0 yaw=0.0", cmd_x: -1, cmd_y: -1, cmd_yaw: 0, lin_vel_xy_rmse: 0.38, fall_rate_height: 0 },
            { label: "vx=+0.0 vy=-1.0 yaw=0.0", cmd_x: 0, cmd_y: -1, cmd_yaw: 0, lin_vel_xy_rmse: 0.29, fall_rate_height: 0 },
            { label: "vx=+1.0 vy=-1.0 yaw=0.0", cmd_x: 1, cmd_y: -1, cmd_yaw: 0, lin_vel_xy_rmse: 0.44, fall_rate_height: 0 },
            { label: "vx=-1.0 vy=+0.0 yaw=0.0", cmd_x: -1, cmd_y: 0, cmd_yaw: 0, lin_vel_xy_rmse: 0.30, fall_rate_height: 0 },
            { label: "vx=+0.0 vy=+0.0 yaw=0.0", cmd_x: 0, cmd_y: 0, cmd_yaw: 0, lin_vel_xy_rmse: 0.15, fall_rate_height: 0 },
            { label: "vx=+1.0 vy=+0.0 yaw=0.0", cmd_x: 1, cmd_y: 0, cmd_yaw: 0, lin_vel_xy_rmse: 0.24, fall_rate_height: 0 },
            { label: "vx=-1.0 vy=+1.0 yaw=0.0", cmd_x: -1, cmd_y: 1, cmd_yaw: 0, lin_vel_xy_rmse: 0.46, fall_rate_height: 0 },
            { label: "vx=+0.0 vy=+1.0 yaw=0.0", cmd_x: 0, cmd_y: 1, cmd_yaw: 0, lin_vel_xy_rmse: 0.32, fall_rate_height: 0 },
            { label: "vx=+1.0 vy=+1.0 yaw=0.0", cmd_x: 1, cmd_y: 1, cmd_yaw: 0, lin_vel_xy_rmse: 0.50, fall_rate_height: 0.01 },
          ],
        },
      };
      const SCENARIO_GUIDES = {
        vel_grid: {
          title: "这个场景测什么 / What this scenario measures",
          body: "速度网格检查不同 vx/vy/yaw 命令下的跟踪误差和稳定性热区。Velocity grid shows tracking error and stability hot spots across commanded vx, vy, and yaw.",
          read: "怎么看 / How to read: 先看 heatmap 中颜色更深的格子，再切换 metric 比较策略；height fall 为 0 表示本次 profile 没有触发高度终止。Start from darker heatmap cells, then switch metrics/policies; zero height fall means no height terminal event in this profile."
        },
        arm_sweep: {
          title: "这个场景测什么 / What this scenario measures",
          body: "手臂扰动扫描检查 arm intensity 增大时，狗策略的速度跟踪是否退化。Arm disturbance sweep checks whether locomotion tracking degrades as arm disturbance intensity increases.",
          read: "怎么看 / How to read: 关注曲线是否随 intensity 上升而变差，以及 fall 诊断是否出现非零事件。Look for worse values as intensity rises and any non-zero fall diagnostics."
        },
        body_pose: {
          title: "这个场景测什么 / What this scenario measures",
          body: "身体姿态跟踪检查 pitch/roll/height 命令在不同运动组下是否可控。Body pose tracking checks pitch, roll, and height command tracking across movement groups.",
          read: "怎么看 / How to read: 先看 summary heatmap 定位异常组，再用 velocity group selector 查看单组折线；越低的 RMSE 通常越好。Use the summary heatmap first, then inspect one velocity group at a time; lower RMSE is usually better."
        },
        gait: {
          title: "这个场景测什么 / What this scenario measures",
          body: "步态跟踪检查动态 gait command 对接触、摆腿、站距和 Raibert 足端指标的影响。Gait tracking shows how dynamic gait commands affect contact, clearance, stance, and Raibert metrics.",
          read: "怎么看 / How to read: 三个子图分别看 gait frequency、stance width、stance length sweep；共享 legend 对应策略颜色。Read the three sweeps separately; the shared legend maps colors to policies."
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

      function sortTooltipValues(values) {
        return values
          .map((item, index) => ({ ...item, index }))
          .sort((a, b) => {
            if (b.value !== a.value) return b.value - a.value;
            return a.index - b.index;
          });
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

      function tableGroupLabel(row, scenario) {
        if (scenario === "vel_grid" && isNumber(row.cmd_yaw)) return `yaw ${fmt(row.cmd_yaw, 1)}`;
        if (scenario === "body_pose") return row.velocity_group ? String(row.velocity_group) : "body pose";
        if (scenario === "gait") {
          const labels = { gait_freq: "gait frequency", stance_width: "stance width", stance_length: "stance length" };
          return labels[row.sweep_axis] || row.sweep_axis || "gait";
        }
        return "";
      }

      function groupSortKey(groupLabel, scenario) {
        if (scenario === "vel_grid") {
          const value = Number(String(groupLabel).replace("yaw", "").trim());
          return Number.isFinite(value) ? value : 999;
        }
        if (scenario === "body_pose") {
          const order = ["stand", "forward", "lateral", "turn"];
          const index = order.indexOf(groupLabel);
          return index >= 0 ? index : 999;
        }
        if (scenario === "gait") {
          const order = ["gait frequency", "stance width", "stance length"];
          const index = order.indexOf(groupLabel);
          return index >= 0 ? index : 999;
        }
        return 0;
      }

      function groupedTable(stubLabel, groups, rows, options = {}) {
        const orderedRows = [...rows].sort((a, b) => {
          const ag = a.groupLabel || "";
          const bg = b.groupLabel || "";
          if (ag !== bg) {
            const cmp = groupSortKey(ag, options.scenario || "") - groupSortKey(bg, options.scenario || "");
            if (cmp !== 0) return cmp;
            return ag.localeCompare(bg);
          }
          return String(a.stubSort || a.stub).localeCompare(String(b.stubSort || b.stub));
        });
        const rowGroups = Array.from(new Set(orderedRows.map((row) => row.groupLabel).filter(Boolean)));
        const collapsible = options.kind === "scenario" && rowGroups.length > 1 && rows.length > 20;
        const tableAttrs = attrsToString({
          "data-scenario": options.scenario || null,
          "data-compare-kind": options.kind || null,
          "data-collapsible": collapsible ? "1" : null,
        });
        const stubSortAttr = collapsible ? "" : ' data-sortable="1" data-sort-column="0"';
        const firstHeader = `<tr><th class="sticky-col" rowspan="2"${stubSortAttr}>${esc(stubLabel)}</th>` +
          groups.map((group) => `<th class="metric-group metric-col-start metric-col-end" colspan="${group.columns.length}"><strong>${esc(group.label)}</strong></th>`).join("") +
          "</tr>";
        let colIndex = 1;
        const secondHeader = "<tr>" + groups.map((group) => group.columns.map((column, columnIndex) => {
          const colorStyle = column.color ? ` style="background:${esc(column.color)}"` : "";
          const boundaryClass = `${columnIndex === 0 ? " metric-col-start" : ""}${columnIndex === group.columns.length - 1 ? " metric-col-end" : ""}`;
          const sortAttr = collapsible ? "" : ` data-sortable="1" data-sort-column="${colIndex}"`;
          const header = `<th class="result-subhead${boundaryClass}"${sortAttr}${colorStyle}><span title="${esc(column.label)}">${esc(column.label)}</span></th>`;
          colIndex += 1;
          return header;
        }).join("")).join("") + "</tr>";
        const groupControls = !collapsible ? "" : `<div class="table-group-controls"><button type="button" data-table-action="expand">Expand all</button><button type="button" data-table-action="collapse">Collapse all</button></div>`;
        const groupHeaders = new Set();
        const totalColumns = 1 + groups.reduce((sum, group) => sum + group.columns.length, 0);
        const body = orderedRows.map((row) => {
          const groupLabel = row.groupLabel || "";
          let prefix = "";
          if (collapsible && groupLabel && !groupHeaders.has(groupLabel)) {
            groupHeaders.add(groupLabel);
            const open = groupHeaders.size === 1;
            prefix = `<tr class="table-group-row" data-group="${esc(groupLabel)}"><td class="table-group-cell sticky-col">` +
              `<button type="button" class="table-group-toggle" data-group="${esc(groupLabel)}" aria-expanded="${open ? "true" : "false"}">` +
              `<span class="table-group-caret">${open ? "−" : "+"}</span>${esc(groupLabel)} <small>${orderedRows.filter((item) => item.groupLabel === groupLabel).length} points</small>` +
              `</button></td><td class="table-group-fill" colspan="${totalColumns - 1}"></td></tr>`;
          }
          const hiddenClass = collapsible && groupHeaders.size > 1 ? " is-collapsed" : "";
          const rowAttrs = attrsToString({
            "data-table-group": collapsible && groupLabel ? groupLabel : null,
          });
          const cells = [dataCell(row.stub, "sticky-col row-head", row.stubSort || row.stub, row.attrs || {})];
          row.groups.forEach((group) => {
            group.cells.forEach((cell, cellIndex) => {
              const boundaryClass = `${cellIndex === 0 ? " metric-col-start" : ""}${cellIndex === group.cells.length - 1 ? " metric-col-end" : ""}`;
              cells.push(dataCell(cell.value, `${cell.klass || ""}${boundaryClass}`, cell.sortValue, cell.attrs || {}));
            });
          });
          return `${prefix}<tr class="metric-row${hiddenClass}"${rowAttrs}>${cells.join("")}</tr>`;
        }).join("");
        return `${groupControls}<div class="compare-table-shell"><table class="${collapsible ? "" : "sortable "}compare-wide-table"${tableAttrs}><thead>${firstHeader}${secondHeader}</thead><tbody>${body}</tbody></table></div>`;
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
            ["xy RMSE mean", fmt(metricAverage(allRows, "lin_vel_xy_rmse"))],
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
              rows: rows.map((row, pointIndex) => ({ ...row, __pointIndex: pointIndex })),
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
        const dataObject = scenarioPlotData(results, scenario, metricDefs);
        const policyNames = dataObject.series.map((item) => item.name);
        const policyOptions = policyNames.map((name) => `<option value="${esc(name)}">${esc(name)}</option>`).join("");
        const multiPolicyClass = policyNames.length > 1 ? "" : " is-hidden";
        const velocityGroups = Array.from(new Set(dataObject.series.flatMap((item) => item.rows.map((row) => row.velocity_group).filter(Boolean))));
        const preferredGroup = velocityGroups.includes("forward") ? "forward" : velocityGroups[0];
        const groupOptions = velocityGroups.map((name) => `<option value="${esc(name)}"${name === preferredGroup ? " selected" : ""}>${esc(name)}</option>`).join("");
        const groupControl = scenario === "body_pose" && velocityGroups.length > 1
          ? `<label class="plot-select velocity-group-wrap">velocity group <select class="velocity-group-select">${groupOptions}</select></label>`
          : "";
        const controls = `
          <div class="plot-controls">
            <div class="metric-tabs">${buttons}</div>
            <div class="plot-control-row">
              <div class="segmented value-mode" data-control="value-mode" aria-label="value mode">
                <button type="button" class="active" data-value-mode="absolute">absolute</button>
                <button type="button" data-value-mode="baseline-delta">baseline delta</button>
              </div>
              <div class="segmented policy-mode" data-control="policy-mode" aria-label="policy mode">
                <button type="button" class="active" data-policy-mode="all">all</button>
                <button type="button" data-policy-mode="single">single</button>
              </div>
              <label class="plot-select baseline-select-wrap${multiPolicyClass}">baseline
                <select class="baseline-select">${policyOptions}</select>
              </label>
              <label class="plot-select single-policy-wrap is-hidden">policy
                <select class="single-policy-select">${policyOptions}</select>
              </label>
              ${groupControl}
            </div>
          </div>`;
        const data = esc(JSON.stringify(dataObject));
        return `<div class="metric-plot" data-scenario="${esc(scenario)}" data-plot="${data}">${controls}<div class="metric-chart" aria-label="metric chart"></div></div>`;
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
        const rows = Array.from(labels).map((label, pointIndex) => {
          const firstRow = names.map((name) => rowMaps[name][label]).find(Boolean) || {};
          return {
            stub: esc(label),
            stubSort: label,
            groupLabel: tableGroupLabel(firstRow, scenario),
            attrs: {
              "data-scenario": scenario,
              "data-point-index": pointIndex,
              "data-point-label": label,
              "data-velocity-group": firstRow.velocity_group || null,
              "data-sweep-axis": firstRow.sweep_axis || firstRow.pose_axis || null,
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
                  "data-velocity-group": firstRow.velocity_group || null,
                  "data-sweep-axis": firstRow.sweep_axis || firstRow.pose_axis || null,
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
          };
        });
        return groupedTable("test point", groups, rows, { kind: "scenario", scenario });
      }

      function scenarioGuide(scenario) {
        const guide = SCENARIO_GUIDES[scenario];
        if (!guide) return "";
        return `<div class="scenario-guide"><h3>${esc(guide.title)}</h3><p>${esc(guide.body)}</p><p>${esc(guide.read)}</p></div>`;
      }

      function fallDiagnostics(results, scenario) {
        const names = runNames(results);
        if (!names.length) return "";
        const cards = names.map((name) => {
          const rows = results[name]?.[scenario] || [];
          const envSteps = rows.reduce((sum, row) => sum + (isNumber(row.n_env_steps) ? row.n_env_steps : 0), 0);
          const falls = rows.reduce((sum, row) => sum + (isNumber(row.n_falls) ? row.n_falls : Math.round((row.fall_rate || 0) * (row.n_env_steps || 0))), 0);
          const heightFalls = rows.reduce((sum, row) => sum + (isNumber(row.fall_rate_height) && isNumber(row.n_env_steps) ? Math.round(row.fall_rate_height * row.n_env_steps) : 0), 0);
          const note = falls === 0 && heightFalls === 0
            ? "本 profile 未触发 terminal fall；zero is not proof of no risk."
            : "检测到非零 fall event；non-zero terminal event observed.";
          return `<div class="fall-card"><strong>${esc(name)}</strong><span>fall ${falls} / ${envSteps || "-"} env-steps</span><span>height fall ${heightFalls} / ${envSteps || "-"} env-steps</span><small>${esc(note)}</small></div>`;
        }).join("");
        return `<div class="fall-diagnostics"><h3>Fall 诊断 / Fall diagnostics</h3><p>0 表示本次 profile 下没有触发 terminal event，不代表绝对无跌倒风险。Zero means no terminal event was observed in this profile, not guaranteed absence of risk.</p><div class="fall-cards">${cards}</div></div>`;
      }

      function renderScenarioSections(results, scenarios) {
        return scenarios.map((scenario) => {
          const metricDefs = DETAIL_METRICS[scenario];
          if (!metricDefs) return "";
          const tables = renderScenarioCompare(results, scenario, metricDefs);
          if (!tables) return "";
          return `<section class="panel" id="detail-${esc(scenario)}"><h2>${esc(SCENARIO_TITLES[scenario] || scenario)}</h2>` +
            scenarioGuide(scenario) + fallDiagnostics(results, scenario) + renderScenarioPlot(results, scenario, metricDefs) + tables + "</section>";
        }).join("");
      }

      function renderHome() {
        const previewTable = renderScenarioCompare(HOME_RESULTS, HOME_SCENARIO, HOME_METRICS);
        const previewPlot = renderScenarioPlot(HOME_RESULTS, HOME_SCENARIO, HOME_METRICS);
        const previewHeatmap = renderScenarioPlot(HOME_HEATMAP_RESULTS, HOME_HEATMAP_SCENARIO, HOME_HEATMAP_METRICS);
        const previewHeatmapTable = renderScenarioCompare(HOME_HEATMAP_RESULTS, HOME_HEATMAP_SCENARIO, HOME_HEATMAP_METRICS);
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
            <h2>Chart + Table Interaction</h2>
            <p class="summary-note">Hover or click a line-chart point. The chart and its metric table highlight the same test point.</p>
            ${previewPlot}
            ${previewTable}
          </section>
          <section class="panel home-panel" id="home-heatmap">
            <h2>Heatmap + Table MVP</h2>
            <p class="summary-note">Velocity grid heatmaps show command-space hot spots. Hover or click a cell to highlight its own metric table.</p>
            ${previewHeatmap}
            ${previewHeatmapTable}
          </section>`;
      }

      function bindSortableTables() {
        document.querySelectorAll("table.sortable").forEach((tableElement) => {
          if (tableElement.dataset.collapsible === "1") return;
          const headers = Array.from(tableElement.querySelectorAll("th[data-sortable='1']"));
          headers.forEach((header) => {
            header.addEventListener("click", () => {
              const tbody = tableElement.tBodies[0];
              const rows = Array.from(tbody.rows).filter((row) => !row.classList.contains("table-group-row"));
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

      function setTableGroup(shell, group, open) {
        shell.querySelectorAll("tr[data-table-group]").forEach((row) => {
          if (row.dataset.tableGroup !== group) return;
          row.classList.toggle("is-collapsed", !open);
        });
        const button = Array.from(shell.querySelectorAll(".table-group-toggle")).find((item) => item.dataset.group === group);
        if (button) {
          button.setAttribute("aria-expanded", open ? "true" : "false");
          const caret = button.querySelector(".table-group-caret");
          if (caret) caret.textContent = open ? "−" : "+";
        }
      }

      function bindCollapsibleTables() {
        document.querySelectorAll(".compare-table-shell").forEach((shell) => {
          shell.querySelectorAll(".table-group-toggle").forEach((button) => {
            button.addEventListener("click", () => {
              const group = button.dataset.group || "";
              const open = button.getAttribute("aria-expanded") !== "true";
              setTableGroup(shell, group, open);
            });
          });
          const controls = shell.previousElementSibling;
          if (!controls || !controls.classList.contains("table-group-controls")) return;
          controls.querySelector("[data-table-action='expand']")?.addEventListener("click", () => {
            shell.querySelectorAll(".table-group-toggle").forEach((button) => setTableGroup(shell, button.dataset.group || "", true));
          });
          controls.querySelector("[data-table-action='collapse']")?.addEventListener("click", () => {
            shell.querySelectorAll(".table-group-toggle").forEach((button) => setTableGroup(shell, button.dataset.group || "", false));
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

      function centerLinkedMetricCells(scenario, metricKey, pointIndex) {
        const selector = `[data-scenario="${cssEscape(scenario)}"][data-metric="${cssEscape(metricKey)}"][data-point-index="${pointIndex}"]`;
        const cells = Array.from(document.querySelectorAll(selector));
        centerMetricGroup(cells);
        return cells.length > 0;
      }

      function ensurePointVisible(scenario, pointIndex) {
        const selector = `[data-scenario="${cssEscape(scenario)}"][data-point-index="${pointIndex}"]`;
        const target = document.querySelector(selector);
        const row = target?.closest("tr[data-table-group]");
        const group = row?.dataset.tableGroup;
        const shell = row?.closest(".compare-table-shell");
        if (!group || !shell) return;
        setTableGroup(shell, group, true);
      }

      function setLinkedHighlight(scenario, metricKey, pointIndex, pinned = false, options = {}) {
        pinnedInteraction = pinned ? { scenario, metricKey, pointIndex } : null;
        ensurePointVisible(scenario, pointIndex);
        resetLinkedHighlights();
        const selector = `[data-scenario="${cssEscape(scenario)}"][data-metric="${cssEscape(metricKey)}"][data-point-index="${pointIndex}"]`;
        const cells = Array.from(document.querySelectorAll(selector));
        cells.forEach((cell, index) => {
          const positionClass = index === 0 ? "linked-start" : index === cells.length - 1 ? "linked-end" : "linked-middle";
          cell.classList.add("linked-highlight", "linked-group", positionClass);
          if (cells.length === 1) cell.classList.add("linked-end");
          if (pinned) cell.classList.add("linked-pinned");
        });
        if (options.center) centerLinkedMetricCells(scenario, metricKey, pointIndex);
        document.querySelectorAll(`[data-scenario="${cssEscape(scenario)}"][data-point-index="${pointIndex}"]`).forEach((cell) => {
          if (cell.classList.contains("sticky-col")) cell.classList.add("row-linked");
        });
      }

      function clearChartHighlights(plot) {
        plot.querySelectorAll(".chart-crosshair, .chart-tooltip-group").forEach((item) => item.remove());
        plot.querySelectorAll(".chart-point.is-active").forEach((item) => item.classList.remove("is-active"));
        plot.querySelectorAll(".heat-cell.is-active").forEach((item) => item.classList.remove("is-active"));
        plot.querySelectorAll(".grouped-point.is-active, .pose-summary-cell.is-active").forEach((item) => item.classList.remove("is-active"));
      }

      function activePlotMetric(plot) {
        return plot.querySelector(".metric-tab.active")?.dataset.metric || plot.querySelector(".metric-tab")?.dataset.metric || "";
      }

      function setActiveMetricTab(plot, metricKey) {
        if (!plot || !metricKey) return false;
        const button = plot.querySelector(`.metric-tab[data-metric="${cssEscape(metricKey)}"]`);
        if (!button) return false;
        if (button.classList.contains("active")) return true;
        plot.querySelectorAll(".metric-tab").forEach((item) => item.classList.remove("active"));
        button.classList.add("active");
        renderMetricPlot(plot, metricKey);
        return true;
      }

      function activatePlotMetricPoint(scenario, metricKey, pointIndex, pinned = false, options = {}) {
        const plot = document.querySelector(`.metric-plot[data-scenario="${cssEscape(scenario)}"]`);
        if (!plot) {
          setLinkedHighlight(scenario, metricKey, pointIndex, pinned, options);
          return false;
        }
        if (!setActiveMetricTab(plot, metricKey)) {
          setLinkedHighlight(scenario, metricKey, pointIndex, pinned, options);
          return false;
        }
        plot.activatePoint?.(pointIndex, pinned, options);
        return true;
      }

      function plotControlState(plot, data) {
        const series = data.series || [];
        const firstName = series[0]?.name || "";
        const valueMode = plot.querySelector("[data-value-mode].active")?.dataset.valueMode || "absolute";
        const policyMode = plot.querySelector("[data-policy-mode].active")?.dataset.policyMode || "all";
        const baselineSelect = plot.querySelector(".baseline-select");
        const singleSelect = plot.querySelector(".single-policy-select");
        const velocityGroupSelect = plot.querySelector(".velocity-group-select");
        if (baselineSelect && !series.some((item) => item.name === baselineSelect.value)) baselineSelect.value = firstName;
        if (singleSelect && !series.some((item) => item.name === singleSelect.value)) singleSelect.value = firstName;
        return {
          valueMode,
          policyMode,
          baselineName: baselineSelect?.value || firstName,
          singleName: singleSelect?.value || firstName,
          velocityGroup: velocityGroupSelect?.value || "",
        };
      }

      function syncPlotControlVisibility(plot) {
        const data = JSON.parse(plot.dataset.plot || "{}");
        const state = plotControlState(plot, data);
        const hasMultiple = (data.series || []).length > 1;
        plot.querySelector(".baseline-select-wrap")?.classList.toggle("is-hidden", !hasMultiple);
        plot.querySelector(".single-policy-wrap")?.classList.toggle("is-hidden", state.policyMode !== "single");
      }

      function selectedPlotSeries(plot, data, metricKey, options = {}) {
        const rawSeries = data.series || [];
        const state = plotControlState(plot, data);
        const baseline = rawSeries.find((item) => item.name === state.baselineName) || rawSeries[0];
        const visible = state.policyMode === "single"
          ? rawSeries.filter((item) => item.name === state.singleName)
          : rawSeries;
        return visible.map((item) => {
          const sourceIndex = Math.max(0, rawSeries.findIndex((candidate) => candidate.name === item.name));
          let rows = (item.rows || []).map((row, index) => {
            const current = row[metricKey];
            const baseValue = baseline?.rows?.[index]?.[metricKey];
            const value = state.valueMode === "baseline-delta"
              ? (isNumber(current) && isNumber(baseValue) ? current - baseValue : null)
              : (isNumber(current) ? current : null);
            return { ...row, [metricKey]: value };
          });
          if (!options.ignoreVelocityGroup && (plot.dataset.scenario || "") === "body_pose" && state.velocityGroup) {
            rows = rows.filter((row) => row.velocity_group === state.velocityGroup);
          }
          const values = rows.map((row) => row[metricKey]);
          return {
            ...item,
            color: item.color || resultColor(item.name, sourceIndex),
            rows,
            values: { ...(item.values || {}), [metricKey]: values },
          };
        });
      }

      function renderVelocityHeatmap(plot, metricKey, metric) {
        const data = JSON.parse(plot.dataset.plot || "{}");
        const chart = plot.querySelector(".metric-chart");
        const scenario = plot.dataset.scenario || "";
        const series = selectedPlotSeries(plot, data, metricKey).map((item, i) => ({
          name: item.name,
          color: item.color || resultColor(item.name, i),
          rows: item.rows || [],
          labels: item.labels || [],
        })).filter((item) => item.rows.length);
        if (!series.length || !series.some((item) => item.rows.some((row) => isNumber(row.cmd_y)))) {
          return false;
        }
        const isVelocityScenario = scenario === "vel_grid" || scenario === HOME_HEATMAP_SCENARIO;
        const allValues = series.flatMap((item) => item.rows.map((row) => row[metricKey]).filter(isNumber));
        if (!allValues.length) {
          chart.innerHTML = "<p class='summary-note'>No numeric values for this metric.</p>";
          return true;
        }
        const minV = Math.min(...allValues);
        const maxV = Math.max(...allValues);
        const vxVals = Array.from(new Set(series.flatMap((item) => item.rows.map((row) => row.cmd_x).filter(isNumber)))).sort((a, b) => a - b);
        const vyVals = Array.from(new Set(series.flatMap((item) => item.rows.map((row) => row.cmd_y).filter(isNumber)))).sort((a, b) => b - a);
        const yawVals = Array.from(new Set(series.flatMap((item) => item.rows.map((row) => row.cmd_yaw).filter(isNumber)))).sort((a, b) => a - b);
        const lowerIsBetter = !metricPrefersHigher(metricKey);
        const heat = (value) => {
          if (!isNumber(value) || maxV <= minV) return "#f2f4f7";
          let t = (value - minV) / (maxV - minV);
          if (!lowerIsBetter) t = 1 - t;
          const hue = 145 - Math.floor(145 * t);
          return `hsl(${hue} 66% 88%)`;
        };
        const cellByYaw = (item, yaw) => {
          const byKey = new Map();
          item.rows.forEach((row, index) => {
            if (Number(row.cmd_yaw) !== Number(yaw)) return;
            byKey.set(`${row.cmd_x}|${row.cmd_y}`, { row, index: row.__pointIndex ?? index });
          });
          return byKey;
        };
        const blocks = series.map((item) => {
          const facets = yawVals.map((yaw) => {
            const map = cellByYaw(item, yaw);
            const grid = vyVals.map((vy) => vxVals.map((vx) => {
              const entry = map.get(`${vx}|${vy}`);
              if (!entry) return `<div class="heat-cell heat-empty"></div>`;
              const value = entry.row[metricKey];
              const label = entry.row.label ?? `vx=${vx} vy=${vy} yaw=${yaw}`;
              return `<button type="button" class="heat-cell" style="background:${heat(value)}" data-point-index="${entry.index}" data-result="${esc(item.name)}" title="${esc(item.name)} | ${esc(label)} | ${esc(metric.label)}: ${fmt(value)}">` +
                `<span>${fmt(value, 3)}</span></button>`;
            }).join("")).join("");
            const xLabels = vxVals.map((vx) => `<span>${fmt(vx, 1)}</span>`).join("");
            const yLabels = vyVals.map((vy) => `<span>${fmt(vy, 1)}</span>`).join("");
            return `<div class="heat-facet"><h4>yaw ${fmt(yaw, 1)}</h4><div class="heat-body">` +
              `<div class="heat-y">${yLabels}</div><div class="heat-grid" style="grid-template-columns: repeat(${vxVals.length}, minmax(54px, 1fr));">${grid}</div>` +
              `<div class="heat-x" style="grid-template-columns: repeat(${vxVals.length}, minmax(54px, 1fr));">${xLabels}</div></div></div>`;
          }).join("");
          return `<div class="heat-policy"><h3>${esc(item.name)}</h3><div class="heat-facets">${facets}</div></div>`;
        }).join("");
        chart.innerHTML = `<div class="heatmap-chart"><div class="heat-axis-title">${isVelocityScenario ? "x = vx, y = vy, facets = yaw" : "heatmap"}</div>${blocks}</div>`;
        const state = { pinnedIndex: null };
        function activate(pointIndex, pinned = false, options = {}) {
          clearChartHighlights(plot);
          setLinkedHighlight(scenario, metric.key, pointIndex, pinned, options);
          chart.querySelectorAll(`.heat-cell[data-point-index="${pointIndex}"]`).forEach((cell) => cell.classList.add("is-active"));
        }
        chart.querySelectorAll(".heat-cell[data-point-index]").forEach((cell) => {
          const pointIndex = Number(cell.dataset.pointIndex);
          cell.addEventListener("mouseenter", () => {
            if (state.pinnedIndex !== null) return;
            activate(pointIndex, false, { center: true });
          });
          cell.addEventListener("mouseleave", () => {
            if (state.pinnedIndex !== null) return;
            clearChartHighlights(plot);
            clearLinkedHighlights();
          });
          cell.addEventListener("click", () => {
            state.pinnedIndex = state.pinnedIndex === pointIndex ? null : pointIndex;
            if (state.pinnedIndex === null) {
              clearChartHighlights(plot);
              clearLinkedHighlights(true);
            } else {
              activate(state.pinnedIndex, true, { center: true });
            }
          });
        });
        plot.activatePoint = (pointIndex, pinned = false, options = {}) => {
          state.pinnedIndex = pinned ? pointIndex : null;
          activate(pointIndex, pinned, options);
        };
        plot.currentPointIndex = () => state.pinnedIndex;
        return true;
      }

      function rowAxisValue(row, axis) {
        const keys = {
          pitch: "cmd_pitch",
          roll: "cmd_roll",
          height: "cmd_height_delta",
          gait_freq: "cmd_gait_freq",
          stance_width: "cmd_stance_width",
          stance_length: "cmd_stance_length",
        };
        const value = row[keys[axis]];
        return isNumber(value) ? value : null;
      }

      function poseAxisMetric(axis) {
        if (axis === "pitch") return "pitch_rmse_deg";
        if (axis === "roll") return "roll_rmse_deg";
        if (axis === "height") return "height_rmse_m";
        return "orientation_control_rmse";
      }

      function groupRowsByAxis(series, scenario) {
        const axes = scenario === "body_pose" ? ["pitch", "roll", "height"] : ["gait_freq", "stance_width", "stance_length"];
        const presentAxes = axes.filter((axis) => series.some((item) => item.rows.some((row) => row.sweep_axis === axis || row.pose_axis === axis || rowAxisValue(row, axis) !== null)));
        return presentAxes.length ? presentAxes : axes;
      }

      function renderGroupedLinePanel(scenario, metric, axis, series, title) {
        const rowsBySeries = series.map((item) => ({
          ...item,
          points: item.rows
            .map((row, index) => ({ row, index: row.__pointIndex ?? index, x: rowAxisValue(row, axis), y: row[metric.key] }))
            .filter((point) => point.x !== null && isNumber(point.y) && (
              rowAxisValue(point.row, axis) !== null &&
              (scenario === "gait" ? (point.row.sweep_axis || axis) === axis : (point.row.pose_axis || point.row.sweep_axis || axis) === axis)
            ))
            .sort((a, b) => a.x - b.x),
        })).filter((item) => item.points.length);
        if (!rowsBySeries.length) return "";
        const allY = rowsBySeries.flatMap((item) => item.points.map((point) => point.y));
        const allX = rowsBySeries.flatMap((item) => item.points.map((point) => point.x));
        let yMin = Math.min(...allY);
        let yMax = Math.max(...allY);
        let xMin = Math.min(...allX);
        let xMax = Math.max(...allX);
        if (yMin === yMax) { yMin -= 1; yMax += 1; }
        if (xMin === xMax) { xMin -= 1; xMax += 1; }
        const yPad = (yMax - yMin) * 0.08;
        yMin -= yPad;
        yMax += yPad;
        const width = 620;
        const height = 250;
        const margin = { left: 54, right: 22, top: 18, bottom: 42 };
        const innerW = width - margin.left - margin.right;
        const innerH = height - margin.top - margin.bottom;
        const x = (value) => margin.left + ((value - xMin) / (xMax - xMin)) * innerW;
        const y = (value) => margin.top + (1 - (value - yMin) / (yMax - yMin)) * innerH;
        const yTicks = [0, 0.25, 0.5, 0.75, 1].map((t) => yMin + t * (yMax - yMin));
        const xTicks = Array.from(new Set(allX)).sort((a, b) => a - b);
        const xStep = Math.max(1, Math.ceil(xTicks.length / 6));
        const grid = yTicks.map((tick) => {
          const yy = y(tick);
          return `<line class="chart-grid" x1="${margin.left}" y1="${yy}" x2="${width - margin.right}" y2="${yy}"></line>` +
            `<text class="chart-label" x="${margin.left - 8}" y="${yy + 4}" text-anchor="end">${fmt(tick)}</text>`;
        }).join("");
        const xLabels = xTicks.map((tick, i) => {
          if (i % xStep !== 0 && i !== xTicks.length - 1) return "";
          return `<text class="chart-label" x="${x(tick)}" y="${height - 14}" text-anchor="middle">${fmt(tick, 2)}</text>`;
        }).join("");
        const paths = rowsBySeries.map((item) => {
          const coords = item.points.map((point) => `${x(point.x)},${y(point.y)}`).join(" ");
          const circles = item.points.map((point) => (
            `<circle class="grouped-point chart-point" data-point-index="${point.index}" data-result="${esc(item.name)}" data-label="${esc(point.row.label || axis)}" data-value="${esc(point.y)}" cx="${x(point.x)}" cy="${y(point.y)}" r="4" fill="${item.color}">` +
            `<title>${esc(item.name)} | ${esc(point.row.label || axis)} | ${esc(metric.label)}: ${fmt(point.y)}</title></circle>`
          )).join("");
          return `<polyline class="chart-line" stroke="${item.color}" points="${coords}"></polyline>${circles}`;
        }).join("");
        return `<div class="grouped-panel" data-axis="${esc(axis)}"><h3>${esc(title)}</h3>` +
          `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(title)} ${esc(metric.label)}" data-inner-top="${margin.top}" data-inner-bottom="${margin.top + innerH}">` +
          grid +
          `<line class="chart-axis" x1="${margin.left}" y1="${margin.top + innerH}" x2="${width - margin.right}" y2="${margin.top + innerH}"></line>` +
          `<line class="chart-axis" x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${margin.top + innerH}"></line>` +
          paths + xLabels + `</svg></div>`;
      }

      function renderPoseSummaryHeatmap(scenario, metric, series) {
        const groups = Array.from(new Set(series.flatMap((item) => item.rows.map((row) => row.velocity_group).filter(Boolean))));
        const axes = ["pitch", "roll", "height"].filter((axis) => series.some((item) => item.rows.some((row) => row.pose_axis === axis || row.sweep_axis === axis)));
        if (!groups.length || !axes.length) return "";
        const cells = [];
        groups.forEach((group) => {
          axes.forEach((axis) => {
            const axisMetric = poseAxisMetric(axis);
            const values = series.flatMap((item) => item.rows
              .filter((row) => row.velocity_group === group && (row.pose_axis === axis || row.sweep_axis === axis))
              .map((row) => row[axisMetric])
              .filter(isNumber));
            if (values.length) cells.push({ group, axis, mean: values.reduce((s, v) => s + v, 0) / values.length, worst: Math.max(...values) });
          });
        });
        if (!cells.length) return "";
        const minV = Math.min(...cells.map((cell) => cell.mean));
        const maxV = Math.max(...cells.map((cell) => cell.mean));
        const heat = (value) => {
          if (!isNumber(value) || maxV <= minV) return "#f2f4f7";
          const t = (value - minV) / (maxV - minV);
          const hue = 145 - Math.floor(145 * t);
          return `hsl(${hue} 66% 88%)`;
        };
        const rows = groups.map((group) => `<div class="pose-summary-row"><strong>${esc(group)}</strong>` + axes.map((axis) => {
          const cell = cells.find((item) => item.group === group && item.axis === axis);
          if (!cell) return `<span class="pose-summary-cell pose-summary-empty">-</span>`;
          return `<span class="pose-summary-cell" style="background:${heat(cell.mean)}" data-mean="${cell.mean}" data-worst="${cell.worst}" title="${esc(group)} | ${esc(axis)} mean ${fmt(cell.mean)} worst ${fmt(cell.worst)}">${fmt(cell.mean, 3)}</span>`;
        }).join("") + "</div>").join("");
        return `<div class="pose-summary"><div class="pose-summary-controls"><span>pose summary</span><button type="button" class="pose-agg-tab active" data-agg="mean">mean</button><button type="button" class="pose-agg-tab" data-agg="worst">worst</button></div><div class="pose-summary-head"><span></span>${axes.map((axis) => `<strong>${esc(axis)}</strong>`).join("")}</div>${rows}</div>`;
      }

      function renderSharedLegend(series) {
        if (series.length <= 1) return "";
        return `<div class="shared-legend">${series.map((item) => `<span><i style="background:${esc(item.color)}"></i>${esc(item.name)}</span>`).join("")}</div>`;
      }

      function renderGroupedScenarioPlot(plot, metricKey, metric) {
        const scenario = plot.dataset.scenario || "";
        if (scenario !== "body_pose" && scenario !== "gait") return false;
        const data = JSON.parse(plot.dataset.plot || "{}");
        const chart = plot.querySelector(".metric-chart");
        const series = selectedPlotSeries(plot, data, metricKey).map((item, i) => ({
          name: item.name,
          color: item.color || resultColor(item.name, i),
          rows: item.rows || [],
        })).filter((item) => item.rows.length);
        const summarySeries = selectedPlotSeries(plot, data, metricKey, { ignoreVelocityGroup: true }).map((item, i) => ({
          name: item.name,
          color: item.color || resultColor(item.name, i),
          rows: item.rows || [],
        })).filter((item) => item.rows.length);
        if (!series.length) return false;
        const hasStructuredShape = scenario === "body_pose"
          ? series.some((item) => item.rows.some((row) => row.pose_axis || row.sweep_axis || row.velocity_group))
          : series.some((item) => item.rows.some((row) => row.sweep_axis));
        if (!hasStructuredShape) return false;
        const axes = groupRowsByAxis(series, scenario);
        const titles = {
          pitch: "Pitch sweep",
          roll: "Roll sweep",
          height: "Height sweep",
          gait_freq: "Gait frequency sweep",
          stance_width: "Stance width sweep",
          stance_length: "Stance length sweep",
        };
        const panels = axes.map((axis) => renderGroupedLinePanel(scenario, metric, axis, series, titles[axis] || axis)).filter(Boolean).join("");
        if (!panels) return false;
        const summary = scenario === "body_pose" ? renderPoseSummaryHeatmap(scenario, metric, summarySeries) : "";
        chart.innerHTML = `<div class="grouped-chart">${summary}${renderSharedLegend(series)}<div class="grouped-panels">${panels}</div></div>`;
        const state = { pinnedIndex: null };
        chart.querySelectorAll(".pose-agg-tab").forEach((button) => {
          button.addEventListener("click", () => {
            const agg = button.dataset.agg || "mean";
            chart.querySelectorAll(".pose-agg-tab").forEach((item) => item.classList.remove("active"));
            button.classList.add("active");
            const values = Array.from(chart.querySelectorAll(".pose-summary-cell[data-mean]"))
              .map((cell) => Number(cell.dataset[agg]))
              .filter(Number.isFinite);
            const minV = values.length ? Math.min(...values) : 0;
            const maxV = values.length ? Math.max(...values) : 0;
            chart.querySelectorAll(".pose-summary-cell[data-mean]").forEach((cell) => {
              const value = Number(cell.dataset[agg]);
              cell.textContent = fmt(value, 3);
              if (!Number.isFinite(value) || maxV <= minV) {
                cell.style.background = "#f2f4f7";
                return;
              }
              const t = (value - minV) / (maxV - minV);
              const hue = 145 - Math.floor(145 * t);
              cell.style.background = `hsl(${hue} 66% 88%)`;
            });
          });
        });
        function renderGroupedInteraction(pointIndex) {
          chart.querySelectorAll(".grouped-panel svg").forEach((svg) => {
            const points = Array.from(svg.querySelectorAll(`.grouped-point[data-point-index="${pointIndex}"]`));
            if (!points.length) return;
            const width = Number(svg.viewBox.baseVal.width || 620);
            const height = Number(svg.viewBox.baseVal.height || 250);
            const innerTop = Number(svg.dataset.innerTop || 18);
            const innerBottom = Number(svg.dataset.innerBottom || 208);
            const xx = Number(points[0].getAttribute("cx"));
            const label = points[0].dataset.label || String(pointIndex);
            const titleText = `${metric.label} | ${label}`;
            const values = sortTooltipValues(points.map((point) => ({
              name: point.dataset.result || "",
              color: point.getAttribute("fill") || "#344054",
              value: Number(point.dataset.value),
            })).filter((item) => Number.isFinite(item.value)));
            const tooltipW = Math.max(250, Math.min(380, 44 + titleText.length * 6.2));
            const rowH = 17;
            const tooltipH = 30 + values.length * rowH;
            const tx = xx > width - 22 - tooltipW ? Math.max(58, xx - tooltipW - 14) : xx + 14;
            const ty = Math.max(22, Math.min(height - tooltipH - 8, innerTop + 8));
            const valueRows = values.map((item, i) => {
              const yy = ty + 32 + i * rowH;
              return `<circle cx="${tx + 12}" cy="${yy - 4}" r="4" fill="${item.color}"></circle>` +
                `<text class="chart-tooltip-text" x="${tx + 22}" y="${yy}">${esc(item.name)}: ${fmt(item.value)}</text>`;
            }).join("");
            const tooltip = `<g class="chart-tooltip-group">` +
              `<line class="chart-crosshair" x1="${xx}" y1="${innerTop}" x2="${xx}" y2="${innerBottom}"></line>` +
              `<rect class="chart-tooltip" fill-opacity="0.82" x="${tx}" y="${ty}" width="${tooltipW}" height="${tooltipH}" rx="6"></rect>` +
              `<text class="chart-tooltip-title" x="${tx + 10}" y="${ty + 18}">${esc(titleText)}</text>` +
              valueRows +
              `</g>`;
            svg.insertAdjacentHTML("beforeend", tooltip);
          });
        }
        function activate(pointIndex, pinned = false, options = {}) {
          clearChartHighlights(plot);
          setLinkedHighlight(scenario, metric.key, pointIndex, pinned, options);
          chart.querySelectorAll(`.grouped-point[data-point-index="${pointIndex}"]`).forEach((point) => point.classList.add("is-active"));
          renderGroupedInteraction(pointIndex);
        }

        function groupedEventToPointIndex(svg, event) {
          const points = Array.from(svg.querySelectorAll(".grouped-point[data-point-index]"));
          if (!points.length) return null;
          const rect = svg.getBoundingClientRect();
          const viewBox = svg.viewBox.baseVal;
          const width = Number(viewBox.width || 620);
          const height = Number(viewBox.height || 250);
          const viewX = ((event.clientX - rect.left) / rect.width) * width;
          const viewY = ((event.clientY - rect.top) / rect.height) * height;
          let best = null;
          points.forEach((point) => {
            const cx = Number(point.getAttribute("cx"));
            const cy = Number(point.getAttribute("cy"));
            const pointIndex = Number(point.dataset.pointIndex);
            if (!Number.isFinite(cx) || !Number.isFinite(cy) || !Number.isFinite(pointIndex)) return;
            const dx = Math.abs(cx - viewX);
            const dy = Math.abs(cy - viewY);
            const distance = dx * dx + dy * dy * 0.18;
            if (!best || distance < best.distance || (distance === best.distance && pointIndex < best.pointIndex)) {
              best = { pointIndex, distance };
            }
          });
          return best ? best.pointIndex : null;
        }

        chart.querySelectorAll(".grouped-panel svg").forEach((svg) => {
          svg.addEventListener("mousemove", (event) => {
            if (state.pinnedIndex !== null) return;
            const pointIndex = groupedEventToPointIndex(svg, event);
            if (pointIndex === null) return;
            activate(pointIndex, false, { center: true });
          });
          svg.addEventListener("mouseleave", () => {
            if (state.pinnedIndex !== null) return;
            clearChartHighlights(plot);
            clearLinkedHighlights();
          });
          svg.addEventListener("click", (event) => {
            const pointIndex = groupedEventToPointIndex(svg, event);
            if (pointIndex === null) return;
            state.pinnedIndex = state.pinnedIndex === pointIndex ? null : pointIndex;
            if (state.pinnedIndex === null) {
              clearChartHighlights(plot);
              clearLinkedHighlights(true);
            } else {
              activate(state.pinnedIndex, true, { center: true });
            }
          });
        });
        plot.activatePoint = (pointIndex, pinned = false, options = {}) => {
          state.pinnedIndex = pinned ? pointIndex : null;
          activate(pointIndex, pinned, options);
        };
        plot.currentPointIndex = () => state.pinnedIndex;
        return true;
      }

      function renderMetricPlot(plot, metricKey) {
        const data = JSON.parse(plot.dataset.plot || "{}");
        const metric = (data.metrics || []).find((item) => item.key === metricKey) || (data.metrics || [])[0];
        const chart = plot.querySelector(".metric-chart");
        if (!metric || !chart) return;
        if (((plot.dataset.scenario || "") === "vel_grid" || (plot.dataset.scenario || "") === HOME_HEATMAP_SCENARIO) && renderVelocityHeatmap(plot, metricKey, metric)) return;
        if (renderGroupedScenarioPlot(plot, metricKey, metric)) return;

        const series = selectedPlotSeries(plot, data, metric.key).map((item, i) => ({
          name: item.name,
          rows: item.rows || [],
          labels: (item.rows || []).map((row) => String(row.label ?? "-")),
          axisLabels: (item.rows || []).map((row) => compactAxisLabel(plot.dataset.scenario || "", String(row.label ?? "-"))),
          pointIndices: (item.rows || []).map((row, index) => row.__pointIndex ?? index),
          values: (item.rows || []).map((row) => row[metric.key] === null ? NaN : Number(row[metric.key])),
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
        const pointIndices = series[0]?.pointIndices || [];
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
            const pointIndex = item.pointIndices[i] ?? i;
            return `<circle class="chart-point" data-point-index="${pointIndex}" data-result="${esc(item.name)}" cx="${x(i)}" cy="${y(value)}" r="4" fill="${item.color}">` +
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

        function displayIndexForPoint(pointIndex) {
          const index = pointIndices.indexOf(pointIndex);
          return index >= 0 ? index : pointIndex;
        }

        function renderInteraction(pointIndex, pinned = false, options = {}) {
          if (!Number.isFinite(pointIndex)) return;
          const displayIndex = displayIndexForPoint(pointIndex);
          if (displayIndex < 0 || displayIndex >= maxPoints) return;
          clearChartHighlights(plot);
          setLinkedHighlight(scenario, metric.key, pointIndex, pinned, options);
          const xx = x(displayIndex);
          svg.querySelectorAll(`.chart-point[data-point-index="${pointIndex}"]`).forEach((point) => {
            point.classList.add("is-active");
          });
          const values = sortTooltipValues(series.map((item) => ({
            name: item.name,
            color: item.color,
            value: item.values[item.pointIndices.indexOf(pointIndex)],
          })).filter((item) => Number.isFinite(item.value)));
          const titleText = `${metric.label} | ${labels[displayIndex] || pointIndex}`;
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
          const displayIndex = maxPoints <= 1 ? 0 : Math.round(((clamped - margin.left) / innerW) * (maxPoints - 1));
          return pointIndices[displayIndex] ?? displayIndex;
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
          syncPlotControlVisibility(plot);
          if (firstButton) renderMetricPlot(plot, firstButton.dataset.metric);
          plot.querySelectorAll(".metric-tab").forEach((button) => {
            button.addEventListener("click", () => {
              const pinnedIndex = typeof plot.currentPointIndex === "function" ? plot.currentPointIndex() : null;
              clearLinkedHighlights();
              setActiveMetricTab(plot, button.dataset.metric);
              if (pinnedIndex !== null) {
                plot.activatePoint?.(pinnedIndex, true, { center: true });
              } else {
                centerMetricColumn(plot.dataset.scenario || "", button.dataset.metric);
              }
            });
          });
          plot.querySelectorAll("[data-value-mode]").forEach((button) => {
            button.addEventListener("click", () => {
              plot.querySelectorAll("[data-value-mode]").forEach((item) => item.classList.remove("active"));
              button.classList.add("active");
              clearLinkedHighlights(true);
              renderMetricPlot(plot, activePlotMetric(plot));
            });
          });
          plot.querySelectorAll("[data-policy-mode]").forEach((button) => {
            button.addEventListener("click", () => {
              plot.querySelectorAll("[data-policy-mode]").forEach((item) => item.classList.remove("active"));
              button.classList.add("active");
              syncPlotControlVisibility(plot);
              clearLinkedHighlights(true);
              renderMetricPlot(plot, activePlotMetric(plot));
            });
          });
          plot.querySelectorAll(".baseline-select, .single-policy-select").forEach((select) => {
            select.addEventListener("change", () => {
              syncPlotControlVisibility(plot);
              clearLinkedHighlights(true);
              renderMetricPlot(plot, activePlotMetric(plot));
            });
          });
          plot.querySelectorAll(".velocity-group-select").forEach((select) => {
            select.addEventListener("change", () => {
              clearLinkedHighlights(true);
              renderMetricPlot(plot, activePlotMetric(plot));
            });
          });
        });
      }

      function bindTableChartLinks() {
        document.querySelectorAll("td[data-scenario][data-metric][data-point-index]").forEach((cell) => {
          const scenario = cell.dataset.scenario;
          const metric = cell.dataset.metric;
          const pointIndex = Number(cell.dataset.pointIndex);
          function preparePlot(plot) {
            if (!plot || scenario !== "body_pose" || !cell.dataset.velocityGroup) return;
            const select = plot.querySelector(".velocity-group-select");
            if (select && select.value !== cell.dataset.velocityGroup) {
              select.value = cell.dataset.velocityGroup;
              renderMetricPlot(plot, activePlotMetric(plot));
            }
          }
          function activateFromTable(pinned) {
            const plot = document.querySelector(`.metric-plot[data-scenario="${cssEscape(scenario)}"]`);
            preparePlot(plot);
            activatePlotMetricPoint(scenario, metric, pointIndex, pinned, { center: true });
          }
          cell.addEventListener("mouseenter", () => {
            if (pinnedInteraction) return;
            activateFromTable(false);
          });
          cell.addEventListener("mouseleave", () => {
            const plot = document.querySelector(`.metric-plot[data-scenario="${cssEscape(scenario)}"]`);
            if (pinnedInteraction) return;
            if (plot) clearChartHighlights(plot);
            clearLinkedHighlights(true);
          });
          cell.addEventListener("click", () => {
            activateFromTable(true);
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
            navLink("Chart + Table", "#home-chart"),
            navLink("Heatmap + Table", "#home-heatmap"),
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
        bindCollapsibleTables();
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
    .scenario-guide, .fall-diagnostics {{
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #ffffff;
      padding: 12px 14px;
      margin-bottom: 12px;
    }}
    .scenario-guide h3, .fall-diagnostics h3 {{
      margin: 0 0 6px;
      color: var(--header);
    }}
    .scenario-guide p, .fall-diagnostics p {{
      margin: 5px 0;
      color: var(--muted);
    }}
    .fall-cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
      gap: 8px;
      margin-top: 10px;
    }}
    .fall-card {{
      display: grid;
      gap: 3px;
      border: 1px solid #d9dee7;
      border-radius: 7px;
      background: var(--panel-2);
      padding: 9px 10px;
    }}
    .fall-card strong {{ color: var(--header); }}
    .fall-card span {{ color: var(--text); font-weight: 700; }}
    .fall-card small {{ color: var(--muted); line-height: 1.35; }}
    .table-group-controls {{
      display: flex;
      gap: 8px;
      justify-content: flex-end;
      margin: 4px 0 8px;
    }}
    .table-group-controls button, .table-group-toggle {{
      border: 1px solid var(--border);
      border-radius: 6px;
      background: #ffffff;
      color: var(--header);
      cursor: pointer;
      font: inherit;
      font-size: 12px;
      font-weight: 750;
      padding: 5px 8px;
    }}
    .table-group-controls button:hover, .table-group-toggle:hover {{
      background: var(--accent-soft);
      color: var(--accent);
    }}
    .table-group-row td {{
      text-align: left;
      background: #f8fafc;
      border-bottom: 1px solid var(--border);
    }}
    .table-group-cell {{
      position: sticky;
      left: 0;
      z-index: 5;
      min-width: 170px;
      max-width: 260px;
      box-shadow: 2px 0 0 var(--border);
    }}
    .table-group-fill {{
      min-width: 0;
    }}
    .table-group-toggle {{
      display: inline-flex;
      align-items: center;
      gap: 7px;
    }}
    .table-group-toggle small {{
      color: var(--muted);
      font-weight: 650;
    }}
    .table-group-caret {{
      display: inline-grid;
      place-items: center;
      width: 17px;
      height: 17px;
      border-radius: 4px;
      background: var(--accent-soft);
      color: var(--accent);
      font-weight: 900;
    }}
    tr.metric-row.is-collapsed {{ display: none; }}
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
    .plot-controls {{
      display: grid;
      gap: 8px;
      margin-bottom: 10px;
    }}
    .plot-control-row {{
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 8px 12px;
    }}
    .segmented {{
      display: inline-flex;
      border: 1px solid var(--border);
      border-radius: 6px;
      overflow: hidden;
      background: #ffffff;
    }}
    .segmented button {{
      border: 0;
      border-right: 1px solid var(--border);
      background: transparent;
      color: var(--text);
      cursor: pointer;
      font: inherit;
      font-size: 12px;
      font-weight: 750;
      padding: 5px 8px;
    }}
    .segmented button:last-child {{ border-right: 0; }}
    .segmented button:hover, .segmented button.active {{
      background: var(--accent-soft);
      color: var(--accent);
    }}
    .plot-select {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 750;
    }}
    .plot-select select {{
      max-width: 220px;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: #ffffff;
      color: var(--text);
      font: inherit;
      font-size: 12px;
      font-weight: 650;
      padding: 5px 7px;
    }}
    .is-hidden {{ display: none !important; }}
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
    .heatmap-chart {{
      display: grid;
      gap: 14px;
      min-width: 720px;
    }}
    .heat-axis-title {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }}
    .heat-policy {{
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #ffffff;
      padding: 10px;
    }}
    .heat-policy h3 {{
      margin: 0 0 8px;
      color: var(--header);
    }}
    .heat-facets {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
    }}
    .heat-facet h4 {{
      margin: 0 0 6px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
    }}
    .heat-body {{
      display: grid;
      grid-template-columns: 36px minmax(0, 1fr);
      grid-template-rows: auto 22px;
      gap: 4px;
      align-items: stretch;
    }}
    .heat-y {{
      display: grid;
      gap: 4px;
      color: var(--muted);
      font-size: 11px;
      text-align: right;
    }}
    .heat-y span {{
      min-height: 34px;
      line-height: 34px;
    }}
    .heat-grid {{
      display: grid;
      gap: 4px;
    }}
    .heat-x {{
      grid-column: 2;
      display: grid;
      gap: 4px;
      color: var(--muted);
      font-size: 11px;
      text-align: center;
    }}
    .heat-cell {{
      min-height: 34px;
      border: 1px solid #ffffff;
      border-radius: 5px;
      color: #1f2933;
      cursor: pointer;
      font: inherit;
      font-size: 11px;
      font-weight: 800;
      padding: 3px;
      text-align: center;
    }}
    .heat-cell span {{
      display: inline-block;
      padding: 1px 3px;
      border-radius: 4px;
      background: rgb(255 255 255 / 52%);
    }}
    .heat-cell:hover, .heat-cell.is-active {{
      outline: 2px solid var(--accent);
      outline-offset: -2px;
    }}
    .heat-empty {{
      background: repeating-linear-gradient(45deg, #f2f4f7, #f2f4f7 5px, #ffffff 5px, #ffffff 10px);
      cursor: default;
    }}
    .grouped-chart {{
      display: grid;
      gap: 14px;
      min-width: 720px;
    }}
    .shared-legend {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: center;
      gap: 8px 14px;
      color: var(--text);
      font-size: 12px;
      font-weight: 750;
    }}
    .shared-legend span {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      max-width: 260px;
    }}
    .shared-legend i {{
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 999px;
      box-shadow: 0 0 0 1px #ffffff, 0 0 0 2px var(--border);
    }}
    .grouped-panels {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 12px;
    }}
    .grouped-panel {{
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #ffffff;
      padding: 10px;
    }}
    .grouped-panel h3 {{
      margin: 0 0 6px;
      color: var(--header);
      font-size: 13px;
    }}
    .grouped-panel svg {{
      display: block;
      width: 100%;
      min-width: 300px;
      height: auto;
    }}
    .pose-summary {{
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #ffffff;
      padding: 10px;
      overflow-x: auto;
    }}
    .pose-summary-controls {{
      display: flex;
      align-items: center;
      gap: 6px;
      margin-bottom: 8px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
    }}
    .pose-agg-tab {{
      border: 1px solid var(--border);
      border-radius: 5px;
      background: #ffffff;
      color: var(--text);
      cursor: pointer;
      font: inherit;
      font-size: 11px;
      font-weight: 800;
      padding: 3px 7px;
    }}
    .pose-agg-tab.active, .pose-agg-tab:hover {{
      background: var(--accent-soft);
      color: var(--accent);
    }}
    .pose-summary-head, .pose-summary-row {{
      display: grid;
      grid-template-columns: 110px repeat(3, minmax(72px, 1fr));
      gap: 5px;
      align-items: center;
      min-width: 390px;
    }}
    .pose-summary-head {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
      margin-bottom: 5px;
      text-align: center;
    }}
    .pose-summary-row {{
      margin-top: 5px;
    }}
    .pose-summary-row strong {{
      color: var(--header);
      font-size: 12px;
    }}
    .pose-summary-cell {{
      border: 1px solid #ffffff;
      border-radius: 5px;
      color: #1f2933;
      display: block;
      font-size: 11px;
      font-weight: 800;
      min-height: 28px;
      line-height: 28px;
      text-align: center;
    }}
    .pose-summary-empty {{
      background: #f2f4f7;
      color: var(--muted);
    }}
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
