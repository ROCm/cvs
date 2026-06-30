'''Inference suite report payload builders.'''

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from cvs.lib.report.cell_build import build_all_cells
from cvs.lib.report.render.gate_matrix import build_gate_matrix_rows
from cvs.lib.report.types import InferenceReportConfig


def aggregate_lifecycle(
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


def overall_status(config: InferenceReportConfig, cells: List[dict], enforce: bool) -> str:
    if not cells:
        return "na"
    if not enforce:
        return "record"
    for cell in cells:
        for tier in config.gated_tiers:
            if cell["tiers"].get(tier) == "fail":
                return "fail"
    return "pass"


def build_chart_series(
    config: InferenceReportConfig, cells: List[dict]
) -> Dict[str, List[Tuple[int, float]]]:
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


def build_sweep_summaries(config: InferenceReportConfig, cells: List[dict]) -> List[dict]:
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


def build_results_table(config: InferenceReportConfig, inf_res_dict: Mapping[tuple, Any]) -> dict:
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
    report_dir: Optional[Path] = None,
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

    raw_run_card = config.run_card_display_builder(variant_config, prov)
    run_card_notes = ""
    run_card_rows: List[Tuple[str, str, bool]] = []
    for label, value, is_link in raw_run_card:
        if label == "Notes":
            run_card_notes = str(value)
            continue
        run_card_rows.append((label, value, is_link))

    run_card_display = extend_run_card_display(run_card_rows, prov)
    run_card_display = [
        (label, value, is_link)
        for label, value, is_link in run_card_display
        if label != "CVS version"
    ]
    display_labels = {label for label, _value, _link in run_card_display}
    if prov.get("cvs_version") and "CVS" not in display_labels:
        run_card_display.append(("CVS", str(prov["cvs_version"]), False))
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if "Generated" not in display_labels:
        run_card_display.append(("Generated", generated_at, False))
    if not run_card_notes:
        rc = getattr(variant_config, "run_card", None)
        if rc and getattr(rc, "notes", None):
            run_card_notes = str(rc.notes)

    chart_series = build_chart_series(config, cells)
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

    from cvs.lib.report.panels.prev_run import build_prev_run_panel, resolve_prev_run_json_path

    panels = {}
    prev_run_path = resolve_prev_run_json_path(
        config.prev_run_json,
        report_basename=config.report_basename,
        report_dir=report_dir,
    )
    if prev_run_path:
        prev_run_panel = build_prev_run_panel(cells, Path(prev_run_path), headline_metric=config.headline_metric)
        if prev_run_panel:
            panels["prev_run"] = prev_run_panel

    return {
        "schema_version": 1,
        "suite_id": config.suite_id,
        "generated_at": generated_at,
        "cvs_version": cvs_version,
        "overall_status": overall_status(config, cells, enforce),
        "report": {
            "title": config.title,
            "subtitle": config.subtitle,
            "footer": config.footer,
            "metric_tier_order": config.metric_tier_order,
            "headline_metric": config.headline_metric,
        },
        "run_card_display": run_card_display,
        "run_card_notes": run_card_notes,
        "provenance": prov,
        "lifecycle": aggregate_lifecycle(lifecycle_report, config.session_lifecycle_labels),
        "cells": cells,
        "chart_series": chart_series,
        "chart_config": chart_config,
        "sweep_summaries": build_sweep_summaries(config, cells),
        "gate_matrix": build_gate_matrix_rows(cells),
        "results_table": build_results_table(config, inf_res_dict),
        "panels": panels,
    }
