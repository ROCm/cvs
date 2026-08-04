'''
Copyright 2025 Advanced Micro Devices Inc.
All rights reserved.

Assemble Run Deck payload from profile, session sources, and dataset builders.
'''

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from cvs.lib.report.inference_payload import _build_panels, _build_run_card_display
from cvs.lib.report.profile import DeckProfile
from cvs.lib.report.rundeck.config_adapter import resolve_report_config
from cvs.lib.report.rundeck.dataset_builders.registry import build_datasets
from cvs.lib.report.rundeck.dataset_builders.sweep import select_inline_cells
from cvs.lib.report.types import InferenceReportConfig


def _normalize_sources(store: Mapping[str, Any]) -> dict[str, Any]:
    lifecycle = store.get("lifecycle_report") or {}
    return {
        "results": store.get("cvs_results_dict") or store.get("inf_res_dict") or {},
        "cvs_results_dict": store.get("cvs_results_dict") or store.get("inf_res_dict") or {},
        "inf_res_dict": store.get("inf_res_dict") or store.get("cvs_results_dict") or {},
        "variant": store.get("variant_config"),
        "variant_config": store.get("variant_config"),
        "lifecycle_report": lifecycle,
        "lifecycle": lifecycle,
        "reference": store.get("reference_results") or store.get("golden_results"),
        "golden": store.get("golden_results") or store.get("reference_results"),
    }


def build_rundeck_payload(
    *,
    profile: DeckProfile,
    store: Mapping[str, Any],
    provenance: Optional[Mapping[str, str]] = None,
    cvs_version: str = "unknown",
    pytest_html_path: str = "",
    log_file_path: str = "",
    report_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Build the full publish context and legacy-compatible payload."""
    sources = _normalize_sources(store)
    profile_dict = profile if isinstance(profile, dict) else {}
    builder_id = profile_dict.get("dataset_builder") or (
        "sweep" if isinstance(profile, InferenceReportConfig) else "sweep"
    )

    config = resolve_report_config(profile)
    datasets: dict[str, Any] = {}
    if builder_id:
        datasets[builder_id] = build_datasets(builder_id, sources, profile)

    prov = dict(provenance or {})
    if pytest_html_path:
        prov.setdefault("pytest_html_path", pytest_html_path)
    if log_file_path:
        prov.setdefault("log_file_path", log_file_path)
    if cvs_version:
        prov.setdefault("cvs_version", cvs_version)

    variant_config = sources.get("variant")
    lifecycle_report = sources.get("lifecycle_report") or {}
    run_card_display, run_card_notes, generated_at = _build_run_card_display(config, variant_config, prov)

    sweep_data = datasets.get("sweep") or {}
    cells = sweep_data.get("all_cells") or sweep_data.get("cells") or []
    panels = _build_panels(config, cells, report_dir, prov)

    from cvs.lib.report.inference_payload import aggregate_lifecycle

    lifecycle = aggregate_lifecycle(lifecycle_report, config.session_lifecycle_labels)

    payload = {
        "schema_version": 1,
        "suite_id": config.suite_id,
        "generated_at": generated_at,
        "cvs_version": cvs_version,
        "overall_status": sweep_data.get("overall_status") or ("record" if builder_id != "sweep" else "na"),
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
        "lifecycle": lifecycle,
        "cells": cells,
        "chart_series": sweep_data.get("chart_series") or {},
        "chart_config": sweep_data.get("chart_config") or [],
        "sweep_summaries": sweep_data.get("sweep_summaries") or [],
        "gate_matrix": sweep_data.get("gate_matrix") or [],
        "results_table": sweep_data.get("results_table") or datasets.get("series", {}).get("results_table") or {},
        "panels": panels,
        "datasets": datasets,
        "deck_profile": profile_dict or {"cards": default_inference_cards()},
    }

    if isinstance(profile, dict) and profile.get("cards"):
        payload["deck_profile"] = profile

    return payload


def apply_summary_meta(payload: dict, config: InferenceReportConfig) -> dict:
    from cvs.lib.report.viewer.scaffold import viewer_basename_for

    total_cells = len(payload.get("cells") or [])
    if config.interactive_viewer and total_cells > config.viewer_cell_threshold:
        payload["summary"] = {
            "mode": "truncated",
            "total_cells": total_cells,
            "inline_limit": config.viewer_cell_threshold,
            "viewer_html": viewer_basename_for(config.report_basename),
            "gated_tiers": list(config.gated_tiers),
        }
        payload["cells"] = select_inline_cells(
            config,
            payload["cells"],
            mode="truncated",
            inline_limit=config.viewer_cell_threshold,
        )
    else:
        payload["summary"] = {"mode": "full", "total_cells": total_cells}
        if config.interactive_viewer:
            payload["summary"]["viewer_html"] = viewer_basename_for(config.report_basename)
    return payload


def _profile_dict_from_config(config: InferenceReportConfig) -> dict[str, Any]:
    return {
        "dataset_builder": "sweep",
        "suite_id": config.suite_id,
        "report_basename": config.report_basename,
        "title": config.title,
        "cards": default_inference_cards(),
    }


def default_inference_cards() -> list[dict[str, Any]]:
    return [
        {"type": "run_card", "id": "run-card", "title": "Run card", "bind": "run_card_display"},
        {"type": "launch_panel", "bind": "panels.launch", "when_empty": "hide"},
        {"type": "lifecycle_timeline", "id": "lifecycle", "title": "Lifecycle timeline", "bind": "lifecycle"},
        {"type": "sweep_analytics", "id": "sweep", "title": "Sweep analytics", "bind": "datasets.sweep"},
        {"type": "gate_matrix", "id": "gates", "title": "Gate matrix", "bind": "gate_matrix"},
        {"type": "gate_heatmap", "bind": "gate_matrix", "when_empty": "hide"},
        {"type": "sweep_cell_cards", "id": "cells", "title": "Sweep cells", "bind": "cells"},
        {"type": "table", "id": "results", "title": "Full results", "bind": "results_table"},
    ]
