'''Inference suite report payload builders.'''

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from cvs.lib.report.cell_build import build_all_cells
from cvs.lib.report.panels.prev_run import build_prev_run_panel, resolve_prev_run_json_path
from cvs.lib.report.provenance import extend_run_card_display
from cvs.lib.report.render.gate_matrix import build_gate_matrix_rows
from cvs.lib.report.sweep_shape import (
    group_cells_by_shape,
    metric_values_by_concurrency,
    shape_label,
)
from cvs.lib.report.types import InferenceReportConfig


def _inf_res_sort_key(kv: tuple) -> tuple:
    key = kv[0]
    if isinstance(key, tuple) and len(key) >= 6:
        return (key[4], key[5])
    return (0, 0)


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


def sweep_has_multi_shape_comparison(cells: List[dict]) -> bool:
    """True when the sweep spans multiple ISL/OSL shapes at two or more concurrencies."""
    shapes: set[Tuple[str, str]] = set()
    concurrencies: set[int] = set()
    for cell in cells:
        shapes.add((str(cell.get("isl", "")), str(cell.get("osl", ""))))
        try:
            concurrencies.add(int(cell["concurrency"]))
        except (TypeError, ValueError, KeyError):
            continue
    return len(shapes) >= 2 and len(concurrencies) >= 2


def build_chart_series(config: InferenceReportConfig, cells: List[dict]) -> Dict[str, List[dict]]:
    """Per-metric sweep charts grouped by ISL/OSL shape.

    Each metric maps to a list of ``{isl, osl, label, points}`` entries so
    multi-shape sweeps do not collapse distinct cells onto the same concurrency
    axis.
    """
    groups = group_cells_by_shape(cells)
    series: Dict[str, List[dict]] = {}
    for chart in config.chart_series:
        full = chart.metric_key or config.full_metric(chart.metric_suffix)
        group_entries: List[dict] = []
        for (isl, osl), group_cells in sorted(groups.items()):
            values_by_conc = metric_values_by_concurrency(group_cells, full)
            points = sorted(values_by_conc.items())
            if len(points) >= 2:
                group_entries.append(
                    {
                        "isl": isl,
                        "osl": osl,
                        "label": shape_label(isl, osl),
                        "points": points,
                    }
                )
        if group_entries:
            series[chart.metric_suffix] = group_entries
    return series


def build_sweep_summaries(config: InferenceReportConfig, cells: List[dict]) -> List[dict]:
    groups = group_cells_by_shape(cells)

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
        # ``saturated``: throughput at the highest concurrency did not grow meaningfully
        # (>1%) vs the previous step, and that highest concurrency is where peak
        # throughput was observed (plateau at the sweep top end).
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
    n_fixed = sum(1 for _label, key in config.results_columns if key is None)
    rows: List[List[Any]] = []
    for key, host_dict in sorted(inf_res_dict.items(), key=_inf_res_sort_key):
        model, gpu, isl, osl, policy, conc = key
        if not isinstance(host_dict, dict):
            continue
        fixed = [model, gpu, isl, osl, policy, conc]
        for host, metrics in host_dict.items():
            row = list(fixed)
            row.append(host)
            for mk in metric_keys[n_fixed:]:
                if mk is None:
                    row.append("\u2014")
                else:
                    v = metrics.get(mk)
                    row.append(v if v is not None else "\u2014")
            rows.append(row)
    return {"headers": headers, "rows": rows}


def _build_run_card_display(
    config: InferenceReportConfig,
    variant_config,
    prov: dict[str, str],
) -> tuple[list[tuple[str, str, bool]], str, str]:
    raw_run_card = config.run_card_display_builder(variant_config, prov)
    run_card_notes = ""
    run_card_rows: List[Tuple[str, str, bool]] = []
    for label, value, is_link in raw_run_card:
        if label == "Notes":
            run_card_notes = str(value)
            continue
        run_card_rows.append((label, value, is_link))

    run_card_display = extend_run_card_display(run_card_rows, prov)
    run_card_display = [(label, value, is_link) for label, value, is_link in run_card_display if label != "CVS version"]
    display_labels = {label for label, _value, _link in run_card_display}
    if prov.get("cvs_version") and "CVS" not in display_labels:
        run_card_display.append(("CVS", str(prov["cvs_version"]), False))
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if "Generated" not in display_labels:
        run_card_display.append(("Generated", generated_at, False))
    return run_card_display, run_card_notes, generated_at


def _build_panels(
    config: InferenceReportConfig,
    cells: List[dict],
    report_dir: Optional[Path],
    provenance: Optional[Mapping[str, str]] = None,
) -> dict:
    panels: dict = {}
    prov = provenance or {}
    if prov.get("launch_server_cmd"):
        panels["launch"] = {
            "example_cell": prov.get("launch_example_cell", ""),
            "server_cmd": prov.get("launch_server_cmd", ""),
            "bench_cmd": prov.get("launch_bench_cmd", ""),
        }
    prev_run_path = resolve_prev_run_json_path(
        config.prev_run_json,
        report_basename=config.report_basename,
        report_dir=report_dir,
    )
    if not prev_run_path:
        return panels
    prev_run_panel = build_prev_run_panel(
        cells,
        Path(prev_run_path),
        headline_metric=config.headline_metric,
    )
    if prev_run_panel:
        panels["prev_run"] = prev_run_panel
    return panels


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

    run_card_display, run_card_notes, generated_at = _build_run_card_display(config, variant_config, prov)

    chart_series = build_chart_series(config, cells)
    panels = _build_panels(config, cells, report_dir, prov)

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
            "sweep_ttft_metric": config.sweep_ttft_metric,
            "session_lifecycle_labels": config.session_lifecycle_labels,
            "cell_lifecycle_labels": config.cell_lifecycle_labels,
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
