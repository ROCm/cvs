'''
Copyright 2025 Advanced Micro Devices Inc.
All rights reserved.

Assemble Run Deck payload from profile, session sources, and dataset builders.
'''

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from cvs.lib.report.inference_payload import _build_panels, _build_run_card_display
from cvs.lib.report.profile import DeckProfile
from cvs.lib.report.rundeck.config_adapter import resolve_report_config
from cvs.lib.report.rundeck.dataset_builders.registry import build_datasets
from cvs.lib.report.rundeck.dataset_builders.sweep import select_inline_cells
from cvs.lib.report.types import InferenceReportConfig


@dataclass
class RundeckPayloadContext:
    """Inputs required to assemble a Run Deck publish payload."""

    profile: DeckProfile
    store: Mapping[str, Any]
    provenance: Optional[Mapping[str, str]] = None
    cvs_version: str = "unknown"
    pytest_html_path: str = ""
    log_file_path: str = ""
    report_dir: Optional[Path] = None


class RundeckPayloadBuilder:
    """Build the full publish context and legacy-compatible Run Deck payload."""

    def __init__(self, ctx: RundeckPayloadContext):
        self.ctx = ctx
        self.profile = ctx.profile
        self.profile_dict = ctx.profile if isinstance(ctx.profile, dict) else {}
        self.sources = self._normalize_sources(ctx.store)
        self.builder_id = self.profile_dict.get("dataset_builder") or (
            "sweep" if isinstance(ctx.profile, InferenceReportConfig) else "sweep"
        )
        self.config = resolve_report_config(ctx.profile)

    def build(self) -> dict[str, Any]:
        datasets = self._build_datasets()
        prov = self._provenance()
        variant_config = self.sources.get("variant")
        lifecycle_report = self.sources.get("lifecycle_report") or {}
        run_card_display, run_card_notes, generated_at = _build_run_card_display(self.config, variant_config, prov)

        sweep_data = datasets.get("sweep") or {}
        cells = sweep_data.get("all_cells") or sweep_data.get("cells") or []
        panels = _build_panels(
            self.config,
            cells,
            self.ctx.report_dir,
            provenance=prov,
            lifecycle_report=lifecycle_report,
            variant_config=variant_config,
        )

        from cvs.lib.report.accuracy_lifecycle import extract_accuracy_from_lifecycle
        from cvs.lib.report.inference_payload import aggregate_lifecycle

        lifecycle = aggregate_lifecycle(lifecycle_report, self.config.session_lifecycle_labels)
        accuracy_metrics = extract_accuracy_from_lifecycle(lifecycle_report)

        payload = {
            "schema_version": 1,
            "suite_id": self.config.suite_id,
            "generated_at": generated_at,
            "cvs_version": self.ctx.cvs_version,
            "overall_status": sweep_data.get("overall_status") or ("record" if self.builder_id != "sweep" else "na"),
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
            "lifecycle": lifecycle,
            "accuracy": accuracy_metrics,
            "cells": cells,
            "chart_series": sweep_data.get("chart_series") or {},
            "chart_config": sweep_data.get("chart_config") or [],
            "sweep_summaries": sweep_data.get("sweep_summaries") or [],
            "gate_matrix": sweep_data.get("gate_matrix") or [],
            "results_table": sweep_data.get("results_table") or datasets.get("series", {}).get("results_table") or {},
            "panels": panels,
            "datasets": datasets,
            "deck_profile": self.profile_dict or {"cards": default_deck_cards()},
        }

        if isinstance(self.profile, dict) and self.profile.get("cards"):
            payload["deck_profile"] = self.profile

        if self.builder_id == "sweep":
            from cvs.lib.report.rundeck.viewer_config import ViewerConfigBuilder

            profile_for_viewer = self.profile_dict if isinstance(self.profile, dict) else {}
            payload["viewer_config"] = ViewerConfigBuilder(profile_for_viewer, self.config).build()

        return payload

    def _build_datasets(self) -> dict[str, Any]:
        datasets: dict[str, Any] = {}
        if self.builder_id:
            datasets[self.builder_id] = build_datasets(self.builder_id, self.sources, self.profile)
        return datasets

    def _provenance(self) -> dict[str, str]:
        prov = dict(self.ctx.provenance or {})
        if self.ctx.pytest_html_path:
            prov.setdefault("pytest_html_path", self.ctx.pytest_html_path)
        if self.ctx.log_file_path:
            prov.setdefault("log_file_path", self.ctx.log_file_path)
        if self.ctx.cvs_version:
            prov.setdefault("cvs_version", self.ctx.cvs_version)
        return prov

    @staticmethod
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


class SummaryMetaApplier:
    """Apply viewer truncation metadata to a Run Deck payload."""

    def __init__(self, config: InferenceReportConfig):
        self.config = config

    def apply(self, payload: dict) -> dict:
        from cvs.lib.report.viewer.scaffold import viewer_basename_for

        total_cells = len(payload.get("cells") or [])
        if self.config.interactive_viewer and total_cells > self.config.viewer_cell_threshold:
            payload["summary"] = {
                "mode": "truncated",
                "total_cells": total_cells,
                "inline_limit": self.config.viewer_cell_threshold,
                "viewer_html": viewer_basename_for(self.config.report_basename),
                "gated_tiers": list(self.config.gated_tiers),
            }
            payload["cells"] = select_inline_cells(
                self.config,
                payload["cells"],
                mode="truncated",
                inline_limit=self.config.viewer_cell_threshold,
            )
        else:
            payload["summary"] = {"mode": "full", "total_cells": total_cells}
            if self.config.interactive_viewer:
                payload["summary"]["viewer_html"] = viewer_basename_for(self.config.report_basename)
        return payload


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
    ctx = RundeckPayloadContext(
        profile=profile,
        store=store,
        provenance=provenance,
        cvs_version=cvs_version,
        pytest_html_path=pytest_html_path,
        log_file_path=log_file_path,
        report_dir=report_dir,
    )
    return RundeckPayloadBuilder(ctx).build()


def apply_summary_meta(payload: dict, config: InferenceReportConfig) -> dict:
    return SummaryMetaApplier(config).apply(payload)


def default_deck_cards() -> list[dict[str, Any]]:
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
