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

PRIMARY_METRICS = [
    ("lin_vel_x_rmse", "vx RMSE", "lower"),
    ("ang_vel_yaw_rmse", "yaw RMSE", "lower"),
    ("fall_rate", "fall rate", "lower"),
    ("tracking_lin_vel_reward", "lin reward", "higher"),
    ("tracking_ang_vel_reward", "yaw reward", "higher"),
    ("base_height_mean", "base height", "neutral"),
    ("max_torque_mean", "max torque", "lower"),
]


def _load_results(path: Path) -> Dict[str, Dict[str, List[dict]]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
    return sorted(names)


def _metric_average(rows: Iterable[dict], metric: str) -> Optional[float]:
    values = _finite_values(rows, metric)
    if not values:
        return None
    return mean(values)


def _summary_rows(results: Dict[str, Dict[str, List[dict]]]) -> List[dict]:
    rows = []
    for run_name, scenarios in results.items():
        all_rows = [row for scenario_rows in scenarios.values() for row in scenario_rows]
        row = {
            "run_name": run_name,
            "scenarios": len(scenarios),
            "points": len(all_rows),
        }
        for metric, _, _ in PRIMARY_METRICS:
            row[metric] = _metric_average(all_rows, metric)
        rows.append(row)
    return rows


def _scenario_metric_rows(results: Dict[str, Dict[str, List[dict]]], scenario: str) -> List[dict]:
    rows = []
    for run_name, scenarios in results.items():
        scenario_rows = scenarios.get(scenario, [])
        row = {"run_name": run_name, "points": len(scenario_rows)}
        for metric, _, _ in PRIMARY_METRICS:
            row[metric] = _metric_average(scenario_rows, metric)
        rows.append(row)
    return rows


def _rank_class(rows: List[dict], metric: str, direction: str, run_name: str) -> str:
    if direction == "neutral":
        return ""
    values = [(row["run_name"], row.get(metric)) for row in rows if _is_number(row.get(metric))]
    if len(values) < 2:
        return ""
    reverse = direction == "higher"
    values.sort(key=lambda item: item[1], reverse=reverse)
    if values[0][0] == run_name:
        return "best"
    if values[-1][0] == run_name:
        return "worst"
    return ""


def _table(headers: List[str], rows: List[List[Tuple[str, str]]]) -> str:
    head = "".join(f"<th>{escape(label)}</th>" for label in headers)
    body = []
    for row in rows:
        cells = "".join(f'<td class="{klass}">{value}</td>' for value, klass in row)
        body.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _summary_table(rows: List[dict]) -> str:
    headers = ["candidate", "scenarios", "points"] + [label for _, label, _ in PRIMARY_METRICS]
    body = []
    for row in rows:
        cells = [
            (f"<strong>{escape(row['run_name'])}</strong>", ""),
            (_fmt(row["scenarios"], 0), ""),
            (_fmt(row["points"], 0), ""),
        ]
        for metric, _, direction in PRIMARY_METRICS:
            cells.append((_fmt(row.get(metric)), _rank_class(rows, metric, direction, row["run_name"])))
        body.append(cells)
    return _table(headers, body)


def _scenario_table(results: Dict[str, Dict[str, List[dict]]], scenario: str) -> str:
    rows = _scenario_metric_rows(results, scenario)
    headers = ["candidate", "points"] + [label for _, label, _ in PRIMARY_METRICS]
    body = []
    for row in rows:
        cells = [(f"<strong>{escape(row['run_name'])}</strong>", ""), (_fmt(row["points"], 0), "")]
        for metric, _, direction in PRIMARY_METRICS:
            cells.append((_fmt(row.get(metric)), _rank_class(rows, metric, direction, row["run_name"])))
        body.append(cells)
    return _table(headers, body)


def _heat_value(value: Any, min_value: float, max_value: float, invert: bool = False) -> str:
    if not _is_number(value) or max_value <= min_value:
        return ""
    t = (float(value) - min_value) / (max_value - min_value)
    if invert:
        t = 1.0 - t
    hue = 145 - int(145 * t)
    return f' style="background:hsl({hue} 70% 90%)"'


def _velocity_grid(results: Dict[str, Dict[str, List[dict]]]) -> str:
    sections = []
    for run_name, scenarios in results.items():
        rows = scenarios.get("vel_grid", [])
        if not rows:
            continue
        metrics = ["lin_vel_x_rmse", "ang_vel_yaw_rmse", "fall_rate"]
        mins = {m: min(_finite_values(rows, m), default=0.0) for m in metrics}
        maxs = {m: max(_finite_values(rows, m), default=0.0) for m in metrics}
        headers = ["command", "vx RMSE", "yaw RMSE", "fall rate", "lin reward", "yaw reward"]
        body = []
        for row in rows:
            cells = [(escape(str(row.get("label", "-"))), "")]
            for metric in metrics:
                style = _heat_value(row.get(metric), mins[metric], maxs[metric])
                cells.append((f"<span{style}>{_fmt(row.get(metric))}</span>", "metric-cell"))
            cells.append((_fmt(row.get("tracking_lin_vel_reward")), ""))
            cells.append((_fmt(row.get("tracking_ang_vel_reward")), ""))
            body.append(cells)
        sections.append(f"<h3>{escape(run_name)}</h3>{_table(headers, body)}")
    if not sections:
        return ""
    return '<section class="panel"><h2>Velocity Grid Detail</h2>' + "".join(sections) + "</section>"


def _candidate_cards(results: Dict[str, Dict[str, List[dict]]]) -> str:
    cards = []
    for run_name, scenarios in results.items():
        points = sum(len(rows) for rows in scenarios.values())
        scenario_text = ", ".join(SCENARIO_TITLES.get(name, name) for name in sorted(scenarios))
        cards.append(
            '<div class="card">'
            f"<h3>{escape(run_name)}</h3>"
            f"<p>{escape(scenario_text)}</p>"
            f'<div class="stat"><span>{points}</span><label>points</label></div>'
            "</div>"
        )
    return "".join(cards)


def _rel_link(target: Path, base_dir: Path) -> str:
    return escape(os.path.relpath(target, base_dir))


def _artifact_nav(source: Path, output_path: Path) -> str:
    base_dir = output_path.parent
    links = []
    results_root = _find_results_root(source)
    index_path = results_root / "index.html"
    if index_path.exists() or results_root.exists():
        links.append(("All results", index_path))
    links.append(("results.json", source))
    report_md = source.with_name("report.md")
    if report_md.exists():
        links.append(("report.md", report_md))
    plots_dir = source.with_name("plots")
    if plots_dir.exists():
        links.append(("plots", plots_dir))
    return "".join(
        f'<a href="{_rel_link(target, base_dir)}">{escape(label)}</a>'
        for label, target in links
    )


def _plot_gallery(source: Path, output_path: Path) -> str:
    plots_dir = source.with_name("plots")
    if not plots_dir.is_dir():
        return ""

    images = sorted(plots_dir.glob("*.png"))
    if not images:
        return ""

    base_dir = output_path.parent
    figures = []
    for image in images:
        title = image.stem.replace("_", " ")
        src = _rel_link(image, base_dir)
        figures.append(
            "<figure>"
            f'<a href="{src}" class="plot-link" data-title="{escape(title)}">'
            f'<img src="{src}" alt="{escape(title)}">'
            "</a>"
            f"<figcaption>{escape(title)}</figcaption>"
            "</figure>"
        )
    return '<section class="panel"><h2>Plots</h2><div class="plot-grid">' + "".join(figures) + "</div></section>"


def _render_html(results: Dict[str, Dict[str, List[dict]]], source: Path, output_path: Path) -> str:
    scenarios = _scenario_names(results)
    summary = _summary_rows(results)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    scenario_sections = []
    for scenario in scenarios:
        title = SCENARIO_TITLES.get(scenario, scenario)
        scenario_sections.append(
            f'<section class="panel"><h2>{escape(title)}</h2>{_scenario_table(results, scenario)}</section>'
        )

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
      padding: 28px 32px 18px;
      border-bottom: 1px solid var(--border);
      background: #ffffff;
    }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    h2 {{ margin: 0 0 16px; font-size: 18px; }}
    h3 {{ margin: 16px 0 10px; font-size: 15px; }}
    .meta {{ color: var(--muted); display: flex; flex-wrap: wrap; gap: 12px 22px; }}
    nav {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 16px; }}
    nav a, footer a {{
      color: var(--accent);
      text-decoration: none;
      font-weight: 650;
    }}
    nav a {{
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 6px 10px;
      background: #fbfcfe;
    }}
    nav a:hover, footer a:hover {{ text-decoration: underline; }}
    main {{ padding: 24px 32px 40px; max-width: 1440px; margin: 0 auto; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; margin-bottom: 18px; }}
    .card, .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
    }}
    .card {{ padding: 16px; }}
    .card h3 {{ margin-top: 0; }}
    .card p {{ min-height: 36px; color: var(--muted); }}
    .stat span {{ font-size: 28px; font-weight: 700; color: var(--accent); }}
    .stat label {{ display: block; color: var(--muted); }}
    .panel {{ padding: 18px; margin-bottom: 18px; overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; min-width: 780px; }}
    th, td {{ padding: 9px 10px; border-bottom: 1px solid var(--border); text-align: right; white-space: nowrap; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ color: var(--muted); font-weight: 600; background: #fbfcfe; }}
    td.best {{ background: var(--best); color: #047857; font-weight: 650; }}
    td.worst {{ background: var(--worst); color: #b42318; }}
    td.metric-cell span {{ display: block; margin: -9px -10px; padding: 9px 10px; }}
    .plot-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 14px;
    }}
    figure {{
      margin: 0;
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
      background: #ffffff;
    }}
    figure img {{ display: block; width: 100%; height: auto; }}
    .plot-link {{ display: block; cursor: zoom-in; }}
    figcaption {{ padding: 9px 10px; color: var(--muted); border-top: 1px solid var(--border); }}
    .lightbox {{
      position: fixed;
      inset: 0;
      z-index: 1000;
      display: none;
      align-items: center;
      justify-content: center;
      background: rgba(17, 24, 39, 0.88);
      padding: 28px;
    }}
    .lightbox.open {{ display: flex; }}
    .lightbox-inner {{
      position: relative;
      display: grid;
      grid-template-rows: auto 1fr auto;
      gap: 10px;
      width: min(1120px, 96vw);
      height: min(820px, 94vh);
    }}
    .lightbox-title {{ color: #ffffff; font-weight: 650; }}
    .lightbox img {{
      align-self: center;
      justify-self: center;
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      background: #ffffff;
      border-radius: 8px;
    }}
    .lightbox-controls {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }}
    .lightbox button {{
      border: 1px solid rgba(255,255,255,0.28);
      border-radius: 6px;
      background: rgba(255,255,255,0.12);
      color: #ffffff;
      font: inherit;
      font-weight: 650;
      padding: 8px 12px;
      cursor: pointer;
    }}
    .lightbox button:hover {{ background: rgba(255,255,255,0.22); }}
    .lightbox-close {{
      position: absolute;
      top: 0;
      right: 0;
    }}
    .lightbox-count {{ color: rgba(255,255,255,0.78); }}
    footer {{ color: var(--muted); padding: 0 32px 30px; max-width: 1440px; margin: 0 auto; }}
  </style>
</head>
<body>
  <header>
    <h1>RoboDuet Benchmark Report</h1>
    <div class="meta">
      <span>source: {escape(str(source))}</span>
      <span>generated: {escape(generated)}</span>
      <span>candidates: {len(results)}</span>
      <span>scenarios: {len(scenarios)}</span>
    </div>
    <nav>{_artifact_nav(source, output_path)}</nav>
  </header>
  <main>
    <section class="cards">{_candidate_cards(results)}</section>
    <section class="panel"><h2>Summary</h2>{_summary_table(summary)}</section>
    {''.join(scenario_sections)}
    {_velocity_grid(results)}
    {_plot_gallery(source, output_path)}
  </main>
  <div class="lightbox" id="plot-lightbox" aria-hidden="true">
    <div class="lightbox-inner">
      <button class="lightbox-close" type="button" data-lightbox-close>Close</button>
      <div class="lightbox-title" id="lightbox-title"></div>
      <img id="lightbox-image" src="" alt="">
      <div class="lightbox-controls">
        <button type="button" data-lightbox-prev>Previous</button>
        <span class="lightbox-count" id="lightbox-count"></span>
        <button type="button" data-lightbox-next>Next</button>
      </div>
    </div>
  </div>
  <footer>Green cells mark best values within the current table; red cells mark worst values.</footer>
  <script>
    (() => {{
      const links = Array.from(document.querySelectorAll(".plot-link"));
      const lightbox = document.getElementById("plot-lightbox");
      const image = document.getElementById("lightbox-image");
      const title = document.getElementById("lightbox-title");
      const count = document.getElementById("lightbox-count");
      let index = 0;

      function render() {{
        const link = links[index];
        if (!link) return;
        const label = link.dataset.title || link.querySelector("img")?.alt || "plot";
        image.src = link.getAttribute("href");
        image.alt = label;
        title.textContent = label;
        count.textContent = `${{index + 1}} / ${{links.length}}`;
      }}

      function openAt(nextIndex) {{
        index = nextIndex;
        render();
        lightbox.classList.add("open");
        lightbox.setAttribute("aria-hidden", "false");
        document.body.style.overflow = "hidden";
      }}

      function close() {{
        lightbox.classList.remove("open");
        lightbox.setAttribute("aria-hidden", "true");
        image.src = "";
        document.body.style.overflow = "";
      }}

      function move(delta) {{
        if (!links.length) return;
        index = (index + delta + links.length) % links.length;
        render();
      }}

      links.forEach((link, i) => {{
        link.addEventListener("click", (event) => {{
          event.preventDefault();
          openAt(i);
        }});
      }});

      document.querySelector("[data-lightbox-close]")?.addEventListener("click", close);
      document.querySelector("[data-lightbox-prev]")?.addEventListener("click", () => move(-1));
      document.querySelector("[data-lightbox-next]")?.addEventListener("click", () => move(1));
      lightbox.addEventListener("click", (event) => {{
        if (event.target === lightbox) close();
      }});
      document.addEventListener("keydown", (event) => {{
        if (!lightbox.classList.contains("open")) return;
        if (event.key === "Escape") close();
        if (event.key === "ArrowLeft") move(-1);
        if (event.key === "ArrowRight") move(1);
      }});
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
    return {
        "name": os.path.relpath(run_dir, root),
        "results": results_path,
        "report": run_dir / "report.html",
        "markdown": run_dir / "report.md",
        "candidates": candidates,
        "scenarios": scenarios,
        "points": points,
    }


def _render_index(entries: List[dict], root: Path) -> str:
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for entry in entries:
        report = entry["report"]
        report_link = report if report.exists() else entry["results"]
        markdown = entry["markdown"]
        artifacts = [
            f'<a href="{_rel_link(report_link, root)}">open</a>',
            f'<a href="{_rel_link(entry["results"], root)}">json</a>',
        ]
        if markdown.exists():
            artifacts.append(f'<a href="{_rel_link(markdown, root)}">md</a>')
        rows.append(
            "<tr>"
            f'<td><a href="{_rel_link(report_link, root)}"><strong>{escape(entry["name"])}</strong></a></td>'
            f"<td>{escape(', '.join(entry['candidates']))}</td>"
            f"<td>{escape(', '.join(SCENARIO_TITLES.get(s, s) for s in entry['scenarios']))}</td>"
            f"<td>{entry['points']}</td>"
            f"<td>{' '.join(artifacts)}</td>"
            "</tr>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RoboDuet Benchmark Results</title>
  <style>
    body {{ margin: 0; font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f6f7f9; color: #1f2933; }}
    header {{ padding: 28px 32px 18px; background: #fff; border-bottom: 1px solid #d9dee7; }}
    main {{ padding: 24px 32px 40px; max-width: 1440px; margin: 0 auto; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    .meta {{ color: #667085; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #d9dee7; border-radius: 8px; overflow: hidden; }}
    th, td {{ padding: 11px 12px; border-bottom: 1px solid #d9dee7; text-align: left; vertical-align: top; }}
    th {{ color: #667085; background: #fbfcfe; font-weight: 600; }}
    a {{ color: #0f766e; text-decoration: none; font-weight: 600; margin-right: 10px; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <header>
    <h1>RoboDuet Benchmark Results</h1>
    <div class="meta">generated: {escape(generated)} | result sets: {len(entries)}</div>
  </header>
  <main>
    <table>
      <thead><tr><th>result</th><th>candidates</th><th>scenarios</th><th>points</th><th>artifacts</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </main>
</body>
</html>
"""


def _write_index_for_root(root: Path):
    entries = []
    for path in sorted(root.rglob("results.json"), reverse=True):
        entry = _read_index_entry(path, root)
        if entry is not None:
            entries.append(entry)
    index_path = root / "index.html"
    index_path.write_text(_render_index(entries, root), encoding="utf-8")
    print(f"[Benchmark] Results index saved -> {index_path}")


def _write_index(results_path: Path):
    _write_index_for_root(_find_results_root(results_path))


def _write_report(results_path: Path, output_path: Path):
    results = _load_results(results_path)
    html = _render_html(results, results_path, output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"[Benchmark] HTML report saved -> {output_path}")


def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Generate standalone HTML report from benchmark results.json")
    parser.add_argument("--results", default=None, help="Path to benchmark results.json")
    parser.add_argument(
        "--results_root",
        default=None,
        help="Generate report.html for every results.json under this benchmark results root",
    )
    parser.add_argument("--output", default=None, help="Output HTML path. Defaults to report.html next to results.json")
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
            _write_report(results_path, results_path.with_name("report.html"))
        if not args.no_index:
            _write_index_for_root(root)
        return

    if not args.results:
        raise ValueError("Provide either --results or --results_root")

    results_path = Path(args.results)
    output_path = Path(args.output) if args.output else results_path.with_name("report.html")
    _write_report(results_path, output_path)
    if not args.no_index:
        _write_index(results_path)


if __name__ == "__main__":
    main()
