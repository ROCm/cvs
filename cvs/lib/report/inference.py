'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Generic inference suite reports for DTNI.

Sits on top of the tabular ``print_results_table`` output: same ``inf_res_dict``
keys and column presets, plus optional HTML/JSON dashboard when pytest ``--html`` is set.

**Suite conftest** — wire via ``inference_wiring`` (see ``ADDING_A_SUITE.md``)::

    from cvs.lib.report.inference_wiring import (
        bind_inference_suite_report_session,
        configure_inference_suite_report,
    )
    from cvs.lib.report.presets.my_suite import MY_SUITE_REPORT_CONFIG

    def pytest_configure(config):
        configure_inference_suite_report(config, MY_SUITE_REPORT_CONFIG)

Demo preset: ``cvs.lib.report.presets.inferencex_atom.INFERENCEX_ATOM_REPORT_CONFIG``.
Session-end generation is automatic when ``--html`` is set; no lifecycle report test.
'''

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from cvs.lib import globals
from cvs.lib.report.cell_build import build_all_cells, select_summary_cells
from cvs.lib.report.formatting import fmt_num as _fmt_num
from cvs.lib.report.formatting import link_or_text_html as _link_or_text
from cvs.lib.report.formatting import status_badge_html as _status_badge
from cvs.lib.report.provenance import build_inference_report_provenance
from cvs.lib.report.render.cell_card import render_cell_card_html
from cvs.lib.report.viewer.scaffold import viewer_basename_for, write_interactive_viewer
from cvs.lib.report.types import InferenceReportConfig, ReportChartSeries

log = globals.log


def _aggregate_lifecycle(
    lifecycle_report: Mapping[str, list],
    labels: tuple[str, ...],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for rows in lifecycle_report.values():
        for label, value, unit in rows:
            if unit != "s":
                continue
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            if label in labels:
                out[label] = max(out.get(label, 0.0), v)
    return out


def _overall_status(config: InferenceReportConfig, cells: List[dict], enforce: bool) -> str:
    if not cells:
        return "na"
    if not enforce:
        return "record"
    for cell in cells:
        for tier in config.gated_tiers:
            if cell["tiers"].get(tier) == "fail":
                return "fail"
    return "pass"


def _build_chart_series(config: InferenceReportConfig, cells: List[dict]) -> Dict[str, List[Tuple[int, float]]]:
    series: Dict[str, List[Tuple[int, float]]] = {}
    for chart in config.chart_series:
        points: List[Tuple[int, float]] = []
        full = config.full_metric(chart.metric_suffix)
        for cell in cells:
            val = cell["actuals"].get(full)
            if val is None:
                continue
            try:
                points.append((int(cell["concurrency"]), float(val)))
            except (TypeError, ValueError):
                continue
        if points:
            series[chart.metric_suffix] = sorted(points, key=lambda p: p[0])
    return series


def _build_sweep_summaries(config: InferenceReportConfig, cells: List[dict]) -> List[dict]:
    groups: Dict[Tuple[str, str], List[dict]] = {}
    for cell in cells:
        groups.setdefault((str(cell["isl"]), str(cell["osl"])), []).append(cell)

    summaries: List[dict] = []
    for (isl, osl), group in sorted(groups.items()):
        points = []
        for cell in group:
            tput = cell["actuals"].get(config.sweep_throughput_metric)
            if tput is None:
                continue
            try:
                points.append((int(cell["concurrency"]), float(tput), cell))
            except (TypeError, ValueError):
                continue
        if not points:
            continue

        best_conc, best_tput, best_cell = max(points, key=lambda p: p[1])
        ttft_at_max = best_cell["actuals"].get(config.sweep_ttft_metric)
        sorted_points = sorted(points, key=lambda p: p[0])
        saturated = False
        if len(sorted_points) >= 2:
            last_conc, last_tput, _ = sorted_points[-1]
            prev_tput = sorted_points[-2][1]
            saturated = last_conc == best_conc and last_tput <= prev_tput * 1.01

        summaries.append(
            {
                "isl": isl,
                "osl": osl,
                "max_output_throughput": best_tput,
                "conc_at_max_tput": best_conc,
                "ttft_at_max_tput": ttft_at_max,
                "saturated": saturated,
                "cell_count": len(group),
            }
        )
    return summaries


def _build_gate_matrix(cells: List[dict]) -> List[dict]:
    return [
        {
            "label": _gate_cell_label(cell),
            "cell_id": cell["cell_id"],
            "concurrency": cell["concurrency"],
            "tiers": cell["tiers"],
        }
        for cell in cells
    ]


def _gate_cell_label(cell: dict) -> str:
    base = f"{cell['policy']} \u00b7 C={cell['concurrency']}"
    if cell.get("show_host_in_label"):
        return f"{base} \u00b7 {cell['host']}"
    return base


def _build_results_table(config: InferenceReportConfig, inf_res_dict: Mapping[tuple, Any]) -> dict:
    headers = [label for label, _key in config.results_columns]
    metric_keys = [key for _label, key in config.results_columns]
    rows: List[List[Any]] = []
    for key, host_dict in sorted(inf_res_dict.items(), key=lambda kv: (kv[0][4], kv[0][5])):
        model, gpu, isl, osl, policy, conc = key
        if not isinstance(host_dict, dict):
            continue
        fixed = [model, gpu, isl, osl, policy, conc]
        for host, metrics in host_dict.items():
            row = list(fixed)
            row.append(host)
            for mk in metric_keys[7:]:
                if mk is None:
                    row.append("\u2014")
                else:
                    v = metrics.get(mk)
                    row.append(v if v is not None else "\u2014")
            rows.append(row)
    return {"headers": headers, "rows": rows}


def build_inference_report_payload(
    *,
    config: InferenceReportConfig,
    variant_config,
    inf_res_dict: Mapping[tuple, Any],
    lifecycle_report: Mapping[str, list],
    cvs_version: str = "unknown",
    pytest_html_path: str = "",
    log_file_path: str = "",
    provenance: Optional[Mapping[str, str]] = None,
) -> dict:
    """Structured payload for HTML render, JSON export, and unit tests."""
    enforce = bool(getattr(variant_config, "enforce_thresholds", False))
    cells = build_all_cells(
        config,
        variant_config=variant_config,
        inf_res_dict=inf_res_dict,
        lifecycle_report=lifecycle_report,
    )

    prov = dict(provenance or {})
    if pytest_html_path:
        prov.setdefault("pytest_html_path", pytest_html_path)
    if log_file_path:
        prov.setdefault("log_file_path", log_file_path)
    if cvs_version:
        prov.setdefault("cvs_version", cvs_version)

    from cvs.lib.report.provenance import extend_run_card_display

    run_card_display = extend_run_card_display(
        config.run_card_display_builder(variant_config, prov),
        prov,
    )
    chart_series = _build_chart_series(config, cells)
    legacy_suffix = config.chart_series[0].metric_suffix if config.chart_series else ""
    legacy_chart = chart_series.get(legacy_suffix, [])
    chart_config = [
        {
            "suffix": ch.metric_suffix,
            "title": ch.title,
            "unit": ch.unit,
            "metric": config.full_metric(ch.metric_suffix),
            "invert": ch.invert,
        }
        for ch in config.chart_series
    ]

    from cvs.lib.report.panels.scaling import build_scaling_panel

    nnodes = int(getattr(getattr(variant_config, "params", None), "nnodes", 1) or 1)
    scaling_panel = build_scaling_panel(
        cells=cells,
        nnodes=nnodes,
        baseline_json_path=config.scaling_baseline_json or None,
        headline_metric=config.headline_metric,
    )
    panels = {}
    if scaling_panel:
        panels["scaling"] = scaling_panel

    return {
        "schema_version": 1,
        "suite_id": config.suite_id,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "cvs_version": cvs_version,
        "overall_status": _overall_status(config, cells, enforce),
        "report": {
            "title": config.title,
            "subtitle": config.subtitle,
            "footer": config.footer,
            "metric_tier_order": config.metric_tier_order,
            "headline_metric": config.headline_metric,
            "embed_summary": config.embed_summary,
        },
        "run_card_display": run_card_display,
        "provenance": prov,
        "lifecycle": _aggregate_lifecycle(lifecycle_report, config.session_lifecycle_labels),
        "cells": cells,
        "chart_points": legacy_chart,
        "chart_series": chart_series,
        "chart_config": chart_config,
        "sweep_summaries": _build_sweep_summaries(config, cells),
        "gate_matrix": _build_gate_matrix(cells),
        "results_table": _build_results_table(config, inf_res_dict),
        "panels": panels,
    }


def _render_bar_chart(
    title: str,
    points: List[Tuple[int, float]],
    unit: str,
    *,
    invert: bool = False,
    accent: str = "accent",
) -> str:
    if len(points) < 2:
        return ""
    values = [p[1] for p in points]
    max_val = max(values) or 1.0
    min_val = min(values) or 0.0
    bars = []
    for conc, val in points:
        if invert:
            span = max_val - min_val or max_val or 1.0
            norm = (max_val - val) / span if span else 1.0
            h = max(12.0, 20.0 + 80.0 * norm)
        else:
            h = max(12.0, 100.0 * val / max_val)
        bars.append(
            f"<div class='chart-col'><div class='chart-bar chart-bar-{accent}' "
            f"style='height:{h:.0f}%'></div>"
            f"<div class='chart-x'>C={conc}</div>"
            f"<div class='chart-y'>{_fmt_num(val)}</div></div>"
        )
    return (
        f"<div class='chart-panel'><h3>{html.escape(title)}</h3>"
        f"<div class='chart'>{''.join(bars)}</div>"
        f"<div class='chart-unit'>{html.escape(unit)}</div></div>"
    )


def _report_css() -> str:
    return """
:root {
  --bg: #0f1117; --panel: #1a1d27; --border: #2a2f3d; --text: #e8eaef; --muted: #9aa3b5;
  --accent: #ff6b35; --accent2: #6b9fff; --accent3: #c77dff;
  --pass: #3dd68c; --fail: #ff5c6a; --record: #6b9fff; --na: #5c6370;
}
* { box-sizing: border-box; }
body { margin: 0; font-family: "Segoe UI", system-ui, sans-serif;
  background: linear-gradient(160deg, #0a0c12 0%, #12151f 40%, #0f1117 100%);
  color: var(--text); line-height: 1.45; padding: 1.5rem; }
.wrap { max-width: 1140px; margin: 0 auto; }
.hero-head { display: flex; flex-wrap: wrap; align-items: flex-start;
  justify-content: space-between; gap: 1rem; margin-bottom: 1.5rem; }
h1 { font-size: 1.75rem; font-weight: 600; margin: 0 0 0.25rem; }
.subtitle { color: var(--muted); margin: 0; font-size: 0.95rem; }
.status-badge { display: inline-block; font-size: 0.85rem; font-weight: 700;
  letter-spacing: 0.06em; padding: 0.45rem 0.9rem; border-radius: 999px; border: 1px solid var(--border); }
.status-pass { background: rgba(61,214,140,0.15); color: var(--pass); }
.status-fail { background: rgba(255,92,106,0.15); color: var(--fail); }
.status-record { background: rgba(107,159,255,0.12); color: var(--record); }
.status-na { background: rgba(92,99,112,0.2); color: var(--na); }
.panel { background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
  padding: 1.25rem 1.5rem; margin-bottom: 1.25rem; box-shadow: 0 8px 32px rgba(0,0,0,0.35); }
.panel h2 { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em;
  color: var(--muted); margin: 0 0 1rem; font-weight: 600; }
.meta-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 0.75rem 1rem; }
.meta-item { display: flex; flex-direction: column; gap: 0.15rem; }
.meta-k { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }
.meta-v { font-size: 0.9rem; font-weight: 500; word-break: break-word; }
.meta-v a { color: var(--accent2); text-decoration: none; }
.tl-row { display: flex; gap: 3px; min-height: 52px; border-radius: 8px; overflow: hidden; }
.tl-seg { background: linear-gradient(180deg, #2d3548 0%, #232836 100%);
  display: flex; flex-direction: column; justify-content: center; align-items: center;
  padding: 0.35rem; min-width: 48px; border-right: 1px solid var(--border); }
.tl-lbl { font-size: 0.65rem; color: var(--muted); text-align: center; }
.tl-val { font-size: 0.8rem; font-weight: 600; color: var(--accent); }
.summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; margin-bottom: 1rem; }
.summary-card { background: rgba(255,255,255,0.03); border: 1px solid var(--border); border-radius: 10px; padding: 1rem; }
.summary-stat { font-size: 1.5rem; font-weight: 700; color: var(--accent); }
.summary-meta { font-size: 0.8rem; color: var(--muted); margin-top: 0.35rem; }
.chart-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1rem; }
.chart-panel { background: rgba(255,255,255,0.02); border: 1px solid var(--border); border-radius: 10px; padding: 1rem; }
.chart-panel h3 { margin: 0 0 0.75rem; font-size: 0.75rem; text-transform: uppercase; color: var(--muted); }
.chart { display: flex; align-items: flex-end; justify-content: center; gap: 1.25rem; height: 160px; }
.chart-col { display: flex; flex-direction: column; align-items: center; width: 64px; height: 100%; }
.chart-bar { width: 40px; border-radius: 6px 6px 2px 2px; margin-top: auto; }
.chart-bar-accent { background: linear-gradient(180deg, var(--accent) 0%, #c44d28 100%); }
.chart-bar-accent2 { background: linear-gradient(180deg, var(--accent2) 0%, #3d5a99 100%); }
.chart-bar-accent3 { background: linear-gradient(180deg, var(--accent3) 0%, #7a3db8 100%); }
.chart-x { font-size: 0.75rem; margin-top: 0.5rem; color: var(--muted); }
.chart-y { font-size: 0.7rem; font-weight: 600; margin-top: 0.2rem; }
.chart-unit { text-align: center; font-size: 0.7rem; color: var(--muted); margin-top: 0.35rem; }
.matrix-wrap, .results-wrap { overflow-x: auto; }
.matrix, .results-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
.matrix th, .matrix td, .results-table th, .results-table td {
  border: 1px solid var(--border); padding: 0.5rem 0.65rem; text-align: center; }
.matrix th, .results-table th { background: rgba(255,255,255,0.04); color: var(--muted);
  font-size: 0.7rem; text-transform: uppercase; }
.matrix td:first-child { text-align: left; font-weight: 500; }
.matrix-pass { background: rgba(61,214,140,0.12); color: var(--pass); }
.matrix-fail { background: rgba(255,92,106,0.12); color: var(--fail); }
.matrix-record { background: rgba(107,159,255,0.08); color: var(--record); }
.results-table { font-size: 0.8rem; }
.results-table td { text-align: right; white-space: nowrap; }
.results-table td:nth-child(-n+7) { text-align: left; }
.cells { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; }
.cell-card { background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
  padding: 1.25rem; display: flex; flex-direction: column; gap: 0.75rem; }
.headline { font-size: 2.25rem; font-weight: 700; color: var(--accent); line-height: 1; }
.headline-unit { font-size: 0.9rem; color: var(--muted); margin-left: 0.35rem; }
.cell-sub { font-size: 0.8rem; color: var(--muted); }
.cell-mini-tl { display: flex; gap: 4px; min-height: 36px; border-radius: 6px; overflow: hidden; font-size: 0.65rem; }
.cell-mini-seg { display: flex; flex-direction: column; justify-content: center; align-items: center;
  padding: 0.25rem; background: rgba(255,255,255,0.05); min-width: 40px; }
.chip { font-size: 0.7rem; font-weight: 600; text-transform: uppercase; padding: 0.2rem 0.5rem; border-radius: 999px; }
.chip-pass { background: rgba(61,214,140,0.15); color: var(--pass); }
.chip-fail { background: rgba(255,92,106,0.15); color: var(--fail); }
.chip-record { background: rgba(107,159,255,0.12); color: var(--record); }
.chip-na { background: rgba(92,99,112,0.2); color: var(--na); }
.bar-track { height: 4px; background: var(--border); border-radius: 2px; margin-top: 0.35rem; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 2px; }
.bar-pass { background: var(--pass); } .bar-fail { background: var(--fail); } .bar-record { background: var(--record); }
.target, .metric-label, .cell-foot { font-size: 0.7rem; color: var(--muted); }
.margin { font-size: 0.7rem; color: var(--pass); display: block; } .margin-fail { color: var(--fail); }
.muted { color: var(--muted); }
footer.page-foot { text-align: center; color: var(--muted); font-size: 0.75rem; margin-top: 2rem; }
.report-nav { display: flex; flex-wrap: wrap; gap: 0.5rem 1rem; margin-bottom: 1.25rem; padding: 0.75rem 1rem;
  background: var(--panel); border: 1px solid var(--border); border-radius: 10px; font-size: 0.8rem; }
.report-nav a { color: var(--accent2); text-decoration: none; }
.report-nav a:hover { text-decoration: underline; }
.viewer-banner { margin-bottom: 1rem; padding: 0.75rem 1rem; border-radius: 8px;
  background: rgba(107,159,255,0.08); border: 1px solid var(--border); font-size: 0.9rem; }
.viewer-banner a { color: var(--accent2); }
@media (max-width: 640px) { body { padding: 1rem; } .headline { font-size: 1.75rem; } }
@media print {
  body { background: #fff; color: #111; }
  .panel, .cell-card, .summary-card, .chart-panel { box-shadow: none; break-inside: avoid; }
}
"""


def render_report_html(payload: dict) -> str:
    report = payload["report"]
    tier_order = report["metric_tier_order"]
    lifecycle = payload.get("lifecycle") or {}
    all_cells = payload.get("cells") or []
    summary = payload.get("summary") or {}
    cells = all_cells
    if summary.get("mode") == "truncated":
        cells = select_summary_cells(
            all_cells,
            int(summary.get("inline_limit", len(all_cells))),
            gated_tiers=tuple(summary.get("gated_tiers") or ()),
        )
    chart_series = payload.get("chart_series") or {}
    summaries = payload.get("sweep_summaries") or []
    gate_matrix = payload.get("gate_matrix") or []
    results_table = payload.get("results_table") or {}
    overall = payload.get("overall_status", "na")
    enforce = any(
        row[1] == "enforced"
        for row in payload.get("run_card_display", [])
        if row[0] == "Thresholds"
    )

    hero_html = "".join(
        f"<div class='meta-item'><span class='meta-k'>{html.escape(label)}</span>"
        f"<span class='meta-v'>"
        f"{_link_or_text(value, label) if is_link else html.escape(str(value))}"
        f"</span></div>"
        for label, value, is_link in payload.get("run_card_display", [])
    )

    timeline_total = sum(lifecycle.values()) or 1.0
    timeline_parts = []
    for lbl in report.get("session_lifecycle_labels", ()) or _SESSION_FALLBACK:
        sec = lifecycle.get(lbl, 0.0)
        if sec <= 0:
            continue
        pct = 100.0 * sec / timeline_total
        timeline_parts.append(
            f"<div class='tl-seg' style='flex-grow:{pct:.2f}'>"
            f"<span class='tl-lbl'>{html.escape(lbl.replace('_', ' '))}</span>"
            f"<span class='tl-val'>{sec:.1f}s</span></div>"
        )
    timeline_html = "".join(timeline_parts) or "<p class='muted'>No lifecycle timings recorded.</p>"

    summary_html = "".join(
        f"<article class='summary-card'><h3>ISL={html.escape(str(s['isl']))} "
        f"OSL={html.escape(str(s['osl']))}</h3>"
        f"<div class='summary-stat'>{_fmt_num(s['max_output_throughput'])} "
        f"<span class='headline-unit'>tok/s</span></div>"
        f"<div class='summary-meta'>Peak at C={s['conc_at_max_tput']}"
        f" &middot; TTFT {_fmt_num(s.get('ttft_at_max_tput'))} ms"
        f"{' &middot; saturated at max C' if s.get('saturated') else ''}</div></article>"
        for s in summaries
    ) or "<p class='muted'>No sweep summary (no throughput data).</p>"

    chart_cfg = payload.get("chart_config") or []
    chart_accent = ("accent", "accent2", "accent3")
    chart_parts = []
    for idx, chart in enumerate(chart_cfg):
        points = chart_series.get(chart["suffix"], [])
        part = _render_bar_chart(
            chart["title"],
            points,
            chart["unit"],
            invert=bool(chart.get("invert")),
            accent=chart_accent[idx % 3],
        )
        if part:
            chart_parts.append(part)
    charts_html = (
        f"<div class='chart-grid'>{''.join(chart_parts)}</div>"
        if chart_parts
        else "<p class='muted'>Concurrency charts need two or more sweep cells.</p>"
    )

    matrix_header = "<tr><th>Cell</th>" + "".join(
        f"<th>{html.escape(t)}</th>" for t in tier_order
    ) + "</tr>"
    matrix_rows = "".join(
        f"<tr><td>{html.escape(row['label'])}</td>"
        + "".join(
            f"<td class='matrix-{html.escape(row['tiers'].get(t, 'na'))}'>"
            f"{html.escape(row['tiers'].get(t, 'na'))}</td>"
            for t in tier_order
        )
        + "</tr>"
        for row in gate_matrix
    )
    matrix_html = (
        f"<table class='matrix'>{matrix_header}{matrix_rows}</table>"
        if matrix_rows
        else "<p class='muted'>No gate matrix (no cells recorded).</p>"
    )

    rt_headers = results_table.get("headers") or []
    rt_rows = results_table.get("rows") or []
    results_html = (
        "<table class='results-table'><tr>"
        + "".join(f"<th>{html.escape(h)}</th>" for h in rt_headers)
        + "</tr>"
        + "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>"
            for row in rt_rows
        )
        + "</table>"
        if rt_rows
        else "<p class='muted'>No results table rows.</p>"
    )

    cell_lifecycle_labels = tuple(
        report.get("cell_lifecycle_labels") or ("server_ready", "client_complete")
    )
    cell_cards = [
        render_cell_card_html(
            c,
            tier_order=tuple(tier_order),
            headline_metric=report["headline_metric"],
            enforce=enforce,
            cell_lifecycle_labels=cell_lifecycle_labels,
        )
        for c in cells
    ]
    cells_banner = ""
    viewer_name = summary.get("viewer_html")
    if summary.get("mode") == "truncated" and viewer_name:
        cells_banner = (
            f"<div class='viewer-banner'>Showing {len(cells)} of {summary.get('total_cells', len(all_cells))} "
            f"cells in this summary. <a href='{html.escape(viewer_name)}'>Open interactive viewer</a> "
            f"for filter and search across all cells.</div>"
        )

    model_label = next((v for lbl, v, _ in payload.get("run_card_display", []) if lbl == "Model"), "run")

    from cvs.lib.report.panels.scaling import render_scaling_panel_html

    scaling_panel = (payload.get("panels") or {}).get("scaling")
    scaling_html = ""
    scaling_nav = ""
    if scaling_panel:
        scaling_html = (
            f"<section class='panel' id='scaling'><h2>Multi-node scaling</h2>"
            f"{render_scaling_panel_html(scaling_panel)}</section>"
        )
        scaling_nav = "<a href='#scaling'>Scaling</a>"

    viewer_nav = ""
    if viewer_name:
        viewer_nav = f"<a href='{html.escape(viewer_name)}'>Viewer</a>"

    nav = (
        "<nav class='report-nav'>"
        "<a href='#run-card'>Run card</a>"
        "<a href='#lifecycle'>Lifecycle</a>"
        "<a href='#sweep'>Sweep</a>"
        "<a href='#gates'>Gates</a>"
        f"{scaling_nav}"
        "<a href='#cells'>Cells</a>"
        "<a href='#results'>Results</a>"
        f"{viewer_nav}"
        "</nav>"
    )
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{html.escape(report['title'])} &mdash; {html.escape(str(model_label))}</title>
<style>{_report_css()}</style></head><body><div class="wrap">
<div class="hero-head"><div><h1>{html.escape(report['title'])}</h1>
<p class="subtitle">{html.escape(report['subtitle'])}</p></div>{_status_badge(overall)}</div>
{nav}
<section class="panel" id="run-card"><h2>Run card</h2><div class="meta-grid">{hero_html}</div></section>
<section class="panel" id="lifecycle"><h2>Lifecycle timeline</h2><div class="tl-row">{timeline_html}</div></section>
<section class="panel" id="sweep"><h2>Sweep analytics</h2><div class="summary-grid">{summary_html}</div>{charts_html}</section>
<section class="panel" id="gates"><h2>Gate matrix</h2><div class="matrix-wrap">{matrix_html}</div></section>
{scaling_html}
<section class="panel" id="cells"><h2>Sweep cells</h2>{cells_banner}<div class="cells">{''.join(cell_cards) or '<p class="muted">No cells.</p>'}</div></section>
<section class="panel" id="results"><h2>Full results</h2><div class="results-wrap">{results_html}</div></section>
<footer class="page-foot">{html.escape(report['footer'])}</footer></div></body></html>"""


_SESSION_FALLBACK = (
    "container_launch", "sshd_setup", "model_fetch", "server_ready", "client_complete", "teardown"
)


def render_report_embed_html(payload: dict, *, height: str = "900px") -> str:
    report = payload["report"]
    doc = render_report_html(payload)
    return (
        f'<details class="report-embed" open><summary><strong>{html.escape(report["embed_summary"])}</strong> '
        f"(embedded)</summary>"
        f'<iframe title="{html.escape(report["title"])}" srcdoc="{html.escape(doc, quote=True)}" '
        f'style="width:100%;height:{html.escape(height)};border:1px solid #ccc;border-radius:8px;"></iframe>'
        "</details>"
    )


def write_report(
    path: Path,
    *,
    config: InferenceReportConfig,
    variant_config,
    inf_res_dict: Mapping[tuple, Any],
    lifecycle_report: Mapping[str, list],
    cvs_version: str = "unknown",
    pytest_html_path: str = "",
    log_file_path: str = "",
    provenance: Optional[Mapping[str, str]] = None,
) -> dict:
    """Build payload, render HTML + JSON sidecar. Returns artifact paths and payload."""
    payload = build_inference_report_payload(
        config=config,
        variant_config=variant_config,
        inf_res_dict=inf_res_dict,
        lifecycle_report=lifecycle_report,
        cvs_version=cvs_version,
        pytest_html_path=pytest_html_path,
        log_file_path=log_file_path,
        provenance=provenance,
    )
    payload["report"]["session_lifecycle_labels"] = config.session_lifecycle_labels
    payload["report"]["cell_lifecycle_labels"] = config.cell_lifecycle_labels

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    total_cells = len(payload["cells"])
    viewer_path = None
    if config.interactive_viewer:
        viewer_name = viewer_basename_for(config.report_basename)
        viewer_path = path.parent / viewer_name
        write_interactive_viewer(
            viewer_path,
            json_basename=f"{config.report_basename}.json",
            title=config.title,
            subtitle=config.subtitle,
            tier_order=config.metric_tier_order,
        )
    summary_mode = config.interactive_viewer and total_cells > config.viewer_cell_threshold
    if summary_mode:
        payload["summary"] = {
            "mode": "truncated",
            "total_cells": total_cells,
            "inline_limit": config.viewer_cell_threshold,
            "viewer_html": viewer_basename_for(config.report_basename),
            "gated_tiers": list(config.gated_tiers),
        }
    else:
        payload["summary"] = {"mode": "full", "total_cells": total_cells}
        if viewer_path is not None:
            payload["summary"]["viewer_html"] = viewer_basename_for(config.report_basename)

    path.write_text(render_report_html(payload), encoding="utf-8")
    json_path = path.with_suffix(".json")
    export = {k: v for k, v in payload.items() if not k.startswith("_")}
    json_path.write_text(json.dumps(export, indent=2, default=str), encoding="utf-8")
    result = {"html": path, "json": json_path, "payload": payload}
    if viewer_path is not None:
        result["viewer"] = viewer_path
    return result


def publish_inference_suite_report(
    config: InferenceReportConfig,
    *,
    variant_config,
    inf_res_dict: Mapping[tuple, Any],
    lifecycle_report: Mapping[str, list],
    report_manager,
    pytest_config,
) -> Optional[dict]:
    """Write suite report artifacts and register them with the pytest HTML bundle."""
    if variant_config is None:
        log.warning("Skipping suite report generation: variant_config not in session store")
        return None

    import importlib.metadata

    try:
        cvs_version = importlib.metadata.version("cvs")
    except importlib.metadata.PackageNotFoundError:
        cvs_version = "dev"

    htmlpath = getattr(pytest_config.option, "htmlpath", None)
    if not htmlpath:
        return None

    html_path = Path(htmlpath).resolve()
    log_file = getattr(pytest_config.option, "log_file", None)
    provenance = build_inference_report_provenance(
        pytest_config,
        cvs_version=cvs_version,
        pytest_html_path=str(html_path),
        log_file_path=str(Path(log_file).resolve()) if log_file else "",
    )
    artifacts = write_report(
        html_path.parent / f"{config.report_basename}.html",
        config=config,
        variant_config=variant_config,
        inf_res_dict=inf_res_dict,
        lifecycle_report=lifecycle_report,
        cvs_version=cvs_version,
        pytest_html_path=str(html_path),
        log_file_path=str(Path(log_file).resolve()) if log_file else "",
        provenance=provenance,
    )
    log.info(
        "Suite report written (%s): %s (json: %s)",
        config.suite_id,
        artifacts["html"],
        artifacts["json"],
    )

    if report_manager and report_manager.is_enabled:
        report_manager.add_html_to_report(artifacts["html"], link_name=config.link_name)
        report_manager.add_html_to_report(
            artifacts["json"], link_name=f"{config.link_name} JSON"
        )
        viewer = artifacts.get("viewer")
        if viewer is not None:
            report_manager.add_html_to_report(
                viewer, link_name=f"{config.link_name} viewer"
            )

    return artifacts


def make_report_test(config: InferenceReportConfig):
    """Return a lifecycle stage test that writes the suite report when pytest-html is enabled."""

    def test_report(inf_res_dict, variant_config, lifecycle, request):
        htmlpath = getattr(request.config.option, "htmlpath", None)
        if not htmlpath:
            return

        import pytest_html

        mgr = getattr(request.config, "_html_report_manager", None)
        artifacts = publish_inference_suite_report(
            config,
            variant_config=variant_config,
            inf_res_dict=inf_res_dict,
            lifecycle_report=lifecycle.report,
            report_manager=mgr,
            pytest_config=request.config,
        )
        if artifacts and mgr and getattr(request.config.option, "self_contained_html", False):
            embed = render_report_embed_html(artifacts["payload"])
            request.node.user_properties.append(
                ("pytest_html_extra", pytest_html.extras.html(embed))
            )

    test_report.__doc__ = (
        f"Write {config.title} HTML/JSON when pytest-html is enabled (render-only)."
    )
    return test_report
