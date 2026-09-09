'''Inference suite report payload builders.'''

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from cvs.lib.report.accuracy_lifecycle import extract_accuracy_from_lifecycle
from cvs.lib.report.cell_build import CellRecordBuilder
from cvs.lib.report.panels.panel_builder import ComparisonPanelBuilder
from cvs.lib.report.provenance import ProvenanceCollector
from cvs.lib.report.render.gate_matrix import GateMatrixRenderer
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


class LifecycleAggregator:
    """Aggregate session-level lifecycle timings across pytest nodeids."""

    @staticmethod
    def aggregate(
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


class SweepAnalyticsBuilder:
    """Build chart series and sweep summary blocks from cell records."""

    def __init__(self, config: InferenceReportConfig):
        self.config = config

    def chart_series(self, cells: List[dict]) -> Dict[str, List[dict]]:
        groups = group_cells_by_shape(cells)
        series: Dict[str, List[dict]] = {}
        for chart in self.config.chart_series:
            full = self.config.full_metric(chart.metric_suffix)
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

    def summaries(self, cells: List[dict]) -> List[dict]:
        groups = group_cells_by_shape(cells)
        summaries: List[dict] = []
        for (isl, osl), group in sorted(groups.items()):
            points = []
            for cell in group:
                tput = cell["actuals"].get(self.config.sweep_throughput_metric)
                if tput is None:
                    continue
                try:
                    points.append((int(cell["concurrency"]), float(tput), cell))
                except (TypeError, ValueError):
                    continue
            if not points:
                continue

            best_conc, best_tput, best_cell = max(points, key=lambda p: p[1])
            ttft_at_max = best_cell["actuals"].get(self.config.sweep_ttft_metric)
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


class ResultsTableBuilder:
    """Build tabular sweep results from raw ``inf_res_dict``."""

    def __init__(self, config: InferenceReportConfig):
        self.config = config

    def build(self, inf_res_dict: Mapping[tuple, Any]) -> dict:
        headers = [label for label, _key in self.config.results_columns]
        metric_keys = [key for _label, key in self.config.results_columns]
        n_fixed = sum(1 for _label, key in self.config.results_columns if key is None)
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


@dataclass
class InferencePayloadContext:
    """Inputs for building a full inference report payload."""

    config: InferenceReportConfig
    variant_config: Any
    inf_res_dict: Mapping[tuple, Any]
    lifecycle_report: Mapping[str, list]
    cvs_version: str = "unknown"
    pytest_html_path: str = ""
    log_file_path: str = ""
    provenance: Optional[Mapping[str, str]] = None
    report_dir: Optional[Path] = None


class InferencePayloadBuilder:
    """Assemble structured inference report payloads for HTML, JSON, and tests."""

    def __init__(self, ctx: InferencePayloadContext):
        self.ctx = ctx
        self.config = ctx.config
        self._cell_builder = CellRecordBuilder(ctx.config)
        self._sweep_builder = SweepAnalyticsBuilder(ctx.config)
        self._results_builder = ResultsTableBuilder(ctx.config)
        self._gate_renderer = GateMatrixRenderer()

    def overall_status(self, cells: List[dict], enforce: bool) -> str:
        if not cells:
            return "na"
        if not enforce:
            return "record"
        evaluated = False
        for cell in cells:
            for tier in self.config.gated_tiers:
                status = cell["tiers"].get(tier)
                if status == "fail":
                    return "fail"
                if status == "pass":
                    evaluated = True
        return "pass" if evaluated else "na"

    def _build_run_card_display(
        self,
        variant_config,
        prov: dict[str, str],
    ) -> tuple[list[tuple[str, str, bool]], str, str]:
        raw_run_card = self.config.run_card_display_builder(variant_config, prov)
        run_card_notes = ""
        run_card_rows: List[Tuple[str, str, bool]] = []
        for label, value, is_link in raw_run_card:
            if label == "Notes":
                run_card_notes = str(value)
                continue
            run_card_rows.append((label, value, is_link))

        run_card_display = ProvenanceCollector.extend_run_card_display(run_card_rows, prov)
        run_card_display = [
            (label, value, is_link) for label, value, is_link in run_card_display if label != "CVS version"
        ]
        display_labels = {label for label, _value, _link in run_card_display}
        if prov.get("cvs_version") and "CVS" not in display_labels:
            run_card_display.append(("CVS", str(prov["cvs_version"]), False))
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        if "Generated" not in display_labels:
            run_card_display.append(("Generated", generated_at, False))
        return run_card_display, run_card_notes, generated_at

    def build(self) -> dict:
        enforce = bool(getattr(self.ctx.variant_config, "enforce_thresholds", False))
        cells = self._cell_builder.build_all(
            variant_config=self.ctx.variant_config,
            inf_res_dict=self.ctx.inf_res_dict,
            lifecycle_report=self.ctx.lifecycle_report,
        )

        prov = dict(self.ctx.provenance or {})
        if self.ctx.pytest_html_path:
            prov.setdefault("pytest_html_path", self.ctx.pytest_html_path)
        if self.ctx.log_file_path:
            prov.setdefault("log_file_path", self.ctx.log_file_path)
        if self.ctx.cvs_version:
            prov.setdefault("cvs_version", self.ctx.cvs_version)

        run_card_display, run_card_notes, generated_at = self._build_run_card_display(
            self.ctx.variant_config,
            prov,
        )

        chart_series = self._sweep_builder.chart_series(cells)
        accuracy_metrics = extract_accuracy_from_lifecycle(self.ctx.lifecycle_report)
        panels = ComparisonPanelBuilder(
            self.config,
            self.ctx.report_dir,
            provenance=prov,
        ).build(
            cells,
            lifecycle_report=self.ctx.lifecycle_report,
            variant_config=self.ctx.variant_config,
        )

        chart_config = [
            {
                "suffix": ch.metric_suffix,
                "title": ch.title,
                "unit": ch.unit,
                "metric": self.config.full_metric(ch.metric_suffix),
                "invert": ch.invert,
            }
            for ch in self.config.chart_series
        ]

        from cvs.lib.report.rundeck.viewer_config import build_viewer_config

        payload = {
            "schema_version": 1,
            "suite_id": self.config.suite_id,
            "generated_at": generated_at,
            "cvs_version": self.ctx.cvs_version,
            "overall_status": self.overall_status(cells, enforce),
            "report": {
                "title": self.config.title,
                "subtitle": self.config.subtitle,
                "footer": self.config.footer,
                "metric_tier_order": self.config.metric_tier_order,
                "headline_metric": self.config.headline_metric,
                "sweep_ttft_metric": self.config.sweep_ttft_metric,
                "session_lifecycle_labels": self.config.session_lifecycle_labels,
                "cell_lifecycle_labels": self.config.cell_lifecycle_labels,
            },
            "run_card_display": run_card_display,
            "run_card_notes": run_card_notes,
            "provenance": prov,
            "lifecycle": LifecycleAggregator.aggregate(
                self.ctx.lifecycle_report,
                self.config.session_lifecycle_labels,
            ),
            "accuracy": accuracy_metrics,
            "cells": cells,
            "chart_series": chart_series,
            "chart_config": chart_config,
            "sweep_summaries": self._sweep_builder.summaries(cells),
            "gate_matrix": self._gate_renderer.build_rows(cells),
            "results_table": self._results_builder.build(self.ctx.inf_res_dict),
            "panels": panels,
        }
        payload["viewer_config"] = build_viewer_config({}, self.config)
        return payload


def aggregate_lifecycle(
    lifecycle_report: Mapping[str, list],
    labels: tuple[str, ...],
) -> Dict[str, float]:
    return LifecycleAggregator.aggregate(lifecycle_report, labels)


def overall_status(config: InferenceReportConfig, cells: List[dict], enforce: bool) -> str:
    return InferencePayloadBuilder(
        InferencePayloadContext(
            config=config,
            variant_config=None,
            inf_res_dict={},
            lifecycle_report={},
        )
    ).overall_status(cells, enforce)


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
    return SweepAnalyticsBuilder(config).chart_series(cells)


def build_sweep_summaries(config: InferenceReportConfig, cells: List[dict]) -> List[dict]:
    return SweepAnalyticsBuilder(config).summaries(cells)


def build_results_table(config: InferenceReportConfig, inf_res_dict: Mapping[tuple, Any]) -> dict:
    return ResultsTableBuilder(config).build(inf_res_dict)


def build_run_card_display(
    config: InferenceReportConfig,
    variant_config,
    prov: dict[str, str],
) -> tuple[list[tuple[str, str, bool]], str, str]:
    return InferencePayloadBuilder(
        InferencePayloadContext(
            config=config,
            variant_config=variant_config,
            inf_res_dict={},
            lifecycle_report={},
        )
    )._build_run_card_display(variant_config, prov)


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
    ctx = InferencePayloadContext(
        config=config,
        variant_config=variant_config,
        inf_res_dict=inf_res_dict,
        lifecycle_report=lifecycle_report,
        cvs_version=cvs_version,
        pytest_html_path=pytest_html_path,
        log_file_path=log_file_path,
        provenance=provenance,
        report_dir=report_dir,
    )
    return InferencePayloadBuilder(ctx).build()
