'''Pytest-html row extras for inference suite reports (Phase A).'''

from __future__ import annotations

from pathlib import Path
from typing import Optional

from cvs.lib.inference.utils.inference_suite_lifecycle import sweep_cell_result_key
from cvs.lib.report.cell_build import build_cell_record
from cvs.lib.report.registry import get_suite_report_config
from cvs.lib.report.render.cell_card import cell_card_css, render_cell_card_html
from cvs.lib.report.types import InferenceReportConfig


def _highlight_metric(item, report_config: InferenceReportConfig, cell: dict) -> Optional[str]:
    funcargs = getattr(item, "funcargs", {}) or {}
    metric = funcargs.get("metric")
    if metric:
        return report_config.full_metric(str(metric))
    tier = funcargs.get("metric_tier")
    if tier:
        variant_config = funcargs.get("variant_config")
        thresholds = getattr(variant_config, "thresholds", {}) or {}
        thresholds_cell = thresholds.get(cell["cell_id"]) or {}
        specs = report_config.tier_metric_specs(thresholds_cell, str(tier))
        if specs:
            return next(iter(specs))
    return None


def attach_inference_cell_row_extra(item, report) -> None:
    """Attach a compact cell card to pytest-html rows for metric tests."""
    if report.when != "call":
        return

    report_config = get_suite_report_config(item.config)
    if not isinstance(report_config, InferenceReportConfig):
        return
    if not report_config.row_card_extras:
        return

    test_name = item.originalname or item.name.split("[")[0]
    if test_name not in report_config.row_card_test_names:
        return

    funcargs = getattr(item, "funcargs", {}) or {}
    inf_res_dict = funcargs.get("inf_res_dict")
    variant_config = funcargs.get("variant_config")
    lifecycle = funcargs.get("lifecycle")
    seq_combo = funcargs.get("seq_combo")
    concurrency = funcargs.get("concurrency")
    if not all((inf_res_dict is not None, variant_config, lifecycle, seq_combo is not None, concurrency is not None)):
        return

    isl = seq_combo["isl"]
    osl = seq_combo["osl"]
    key = sweep_cell_result_key(variant_config, seq_combo, isl, osl, concurrency)
    host_dict = inf_res_dict.get(key)
    if not isinstance(host_dict, dict) or not host_dict:
        return

    host, actuals = next(iter(sorted(host_dict.items())))
    cell = build_cell_record(
        report_config,
        key=key,
        host=host,
        actuals=actuals,
        variant_config=variant_config,
        lifecycle_report=getattr(lifecycle, "report", {}),
        multi_host=len(host_dict) > 1,
    )
    cell["pytest_metrics_nodeid"] = item.nodeid
    highlight = _highlight_metric(item, report_config, cell)
    enforce = bool(getattr(variant_config, "enforce_thresholds", False))
    htmlpath = getattr(item.config.option, "htmlpath", None)
    pytest_basename = Path(htmlpath).name if htmlpath else ""
    card = render_cell_card_html(
        cell,
        tier_order=report_config.metric_tier_order,
        headline_metric=report_config.headline_metric,
        enforce=enforce,
        cell_lifecycle_labels=report_config.cell_lifecycle_labels,
        compact=True,
        highlight_metric=highlight,
        pytest_html_basename=pytest_basename or None,
    )
    snippet = f"<style>{cell_card_css(compact=True)}</style><div class='cvs-cell-row-extra'>{card}</div>"

    try:
        import pytest_html
    except ImportError:
        return

    extras = getattr(report, "extras", [])
    extras.append(pytest_html.extras.html(snippet))
    report.extras = extras
