'''
Copyright 2025 Advanced Micro Devices Inc.
All rights reserved.

Build profile-driven interactive viewer configuration for sweep Run Deck payloads.
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from cvs.lib.report.types import InferenceReportConfig

# Results-table column labels → cell record fields (sweep cells).
LABEL_TO_FIELD: dict[str, str] = {
    "Model": "model",
    "GPU": "gpu",
    "ISL": "isl",
    "OSL": "osl",
    "Policy": "policy",
    "Conc": "concurrency",
    "Host": "host",
    "Cell": "cell_id",
}

FIELD_LABELS: dict[str, str] = {
    "cell_id": "Cell",
    "model": "Model",
    "gpu": "GPU",
    "isl": "ISL",
    "osl": "OSL",
    "policy": "Policy",
    "concurrency": "C",
    "host": "Host",
}


@dataclass(frozen=True)
class ViewerConfigContext:
    """Immutable inputs for interactive viewer configuration."""

    profile: dict[str, Any]
    config: InferenceReportConfig


class ViewerConfigBuilder:
    """Build ``viewer_config`` for the interactive sweep viewer."""

    def __init__(self, profile: Optional[dict[str, Any]], config: InferenceReportConfig):
        self.profile = profile or {}
        self.config = config
        self.viewer = self.profile.get("viewer") or {}
        self.sweep = self.profile.get("sweep") or {}

    def build(self) -> dict[str, Any]:
        group_by = self._group_by()
        filter_fields = self._filter_fields(group_by)
        metrics = self._metrics()
        heatmap_metrics = self._heatmap_metrics(metrics)
        margin_metrics = self._margin_metrics(metrics)
        table_columns = self._table_columns()
        interactivity = self._interactivity()
        dim_labels = {f: self._field_label(f) for f in group_by}
        comparison_hint = self.viewer.get("comparison_hint") or (
            f"Compare {' / '.join(dim_labels.get(f, f.upper()) for f in group_by)} shapes at each concurrency"
        )
        return {
            "group_by": group_by,
            "group_labels": dim_labels,
            "filters": [{"field": f, "label": self._field_label(f)} for f in filter_fields],
            "concurrency_field": self.viewer.get("concurrency_field", "concurrency"),
            "metrics": metrics,
            "heatmap_metrics": heatmap_metrics,
            "margin_metrics": margin_metrics,
            "default_heatmap_metric": self.viewer.get("default_heatmap_metric") or self.config.headline_metric,
            "default_margin_metric": self.viewer.get("default_margin_metric") or self.config.headline_metric,
            "table_columns": table_columns,
            "heatmap_row_fields": list(self.viewer.get("heatmap_row_fields") or group_by + ["policy"]),
            "comparison_hint": comparison_hint,
            "interactivity": interactivity,
        }

    def _group_by(self) -> list[str]:
        return list(self.viewer.get("group_by") or self.sweep.get("group_by") or ("isl", "osl"))

    def _filter_fields(self, group_by: list[str]) -> list[str]:
        return list(self.viewer.get("filters") or group_by + ["policy", "host"])

    def _metrics(self) -> dict[str, dict[str, Any]]:
        metrics = self._metrics_from_config()
        extra_metrics = self.viewer.get("metrics") or {}
        for key, meta in extra_metrics.items():
            if isinstance(meta, dict):
                metrics[str(key)] = {
                    "label": meta.get("label", key),
                    "unit": meta.get("unit", ""),
                    "higher_better": bool(meta.get("higher_better", True)),
                }
        return metrics

    def _heatmap_metrics(self, metrics: dict[str, dict[str, Any]]) -> list[str]:
        heatmap_metrics = list(
            self.viewer.get("heatmap_metrics")
            or [self.config.full_metric(ch.metric_suffix) for ch in (self.config.chart_series or ())]
        )
        return [m for m in heatmap_metrics if m in metrics]

    def _margin_metrics(self, metrics: dict[str, dict[str, Any]]) -> list[str]:
        margin_metrics = list(self.viewer.get("margin_metrics") or [])
        if margin_metrics:
            return margin_metrics
        candidates = [
            self.config.headline_metric,
            self.config.sweep_ttft_metric,
            self.config.full_metric("mean_tpot_ms"),
        ]
        return [m for m in candidates if m and m in metrics]

    def _table_columns(self) -> list[dict[str, Any]]:
        table_columns = self.viewer.get("table_columns")
        if table_columns:
            parsed = self._parse_profile_columns(table_columns) if isinstance(table_columns, list) else []
            return parsed or self._table_columns_from_config()
        raw_cols = self.sweep.get("results_table_columns") or self.profile.get("results_columns")
        if raw_cols:
            parsed = self._parse_profile_columns(raw_cols)
            parsed.append({"computed": "status", "label": "Status"})
            return parsed
        return self._table_columns_from_config()

    def _interactivity(self) -> dict[str, Any]:
        interactivity_raw = (
            self.viewer.get("interactivity") if isinstance(self.viewer.get("interactivity"), dict) else {}
        )
        tpot_metric = interactivity_raw.get("tpot_metric") or self.config.full_metric("mean_tpot_ms")
        output_metric = interactivity_raw.get("output_throughput_metric") or self.config.headline_metric
        total_metric = interactivity_raw.get("total_throughput_metric") or self.config.full_metric(
            "total_token_throughput"
        )
        return {
            "enabled": bool(interactivity_raw.get("enabled", self.config.interactive_viewer)),
            "tpot_metric": tpot_metric,
            "output_throughput_metric": output_metric,
            "total_throughput_metric": total_metric,
            "title": interactivity_raw.get("title") or "Token Throughput per GPU vs. Interactivity",
            "hint": interactivity_raw.get(
                "hint",
                "Interactivity = 1000 / mean TPOT (ms) (tok/s/user), matching InferenceX · "
                "Y = total token throughput per GPU · scroll to zoom · shift+drag to pan · "
                "drag to box-zoom · click a point to pin the detail card",
            ),
        }

    @staticmethod
    def _field_label(field: str) -> str:
        return FIELD_LABELS.get(field, field.replace("_", " ").title())

    def _metric_meta(self, full_key: str, label: str, *, invert: bool = False) -> dict[str, Any]:
        short = full_key.split(".", 1)[-1] if "." in full_key else full_key
        unit = self.config.metric_units.get(short, "")
        higher = not invert
        if "ms" in label.lower() or "latency" in label.lower() or invert:
            higher = False
        if "throughput" in label.lower() or "tok/s" in unit or "req/s" in unit:
            higher = True
        return {"label": label, "unit": unit, "higher_better": higher}

    def _metrics_from_config(self) -> dict[str, dict[str, Any]]:
        metrics: dict[str, dict[str, Any]] = {}
        for ch in self.config.chart_series or ():
            full = self.config.full_metric(ch.metric_suffix)
            metrics[full] = self._metric_meta(full, ch.title, invert=ch.invert)
        for label, key in self.config.results_columns:
            if not key:
                continue
            key_str = str(key)
            if key_str not in metrics:
                metrics[key_str] = self._metric_meta(key_str, str(label))
        return metrics

    def _table_columns_from_config(self) -> list[dict[str, Any]]:
        columns: list[dict[str, Any]] = [{"field": "cell_id", "label": "Cell"}]
        for label, key in self.config.results_columns:
            if key is None:
                field = LABEL_TO_FIELD.get(str(label))
                if field and field != "cell_id":
                    columns.append({"field": field, "label": str(label)})
            elif str(key).startswith("client."):
                columns.append({"metric": str(key), "label": str(label)})
        if len(columns) <= 1:
            columns.extend(
                [
                    {"field": "isl", "label": "ISL"},
                    {"field": "osl", "label": "OSL"},
                    {"field": "policy", "label": "Policy"},
                    {"field": "host", "label": "Host"},
                    {"field": "concurrency", "label": "C"},
                ]
            )
            for suffix, hl_label in (self.config.cell_highlights or ())[:3]:
                columns.append({"metric": self.config.full_metric(suffix), "label": str(hl_label)})
        columns.append({"computed": "status", "label": "Status"})
        return columns

    @staticmethod
    def _parse_profile_columns(raw: list) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for entry in raw:
            if isinstance(entry, dict):
                if entry.get("field") or entry.get("metric") or entry.get("computed"):
                    out.append(dict(entry))
                continue
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                label, key = entry
                if key is None:
                    field = LABEL_TO_FIELD.get(str(label))
                    if field:
                        out.append({"field": field, "label": str(label)})
                elif key is not None:
                    out.append({"metric": str(key), "label": str(label)})
        return out


def build_viewer_config(
    profile: Optional[dict[str, Any]],
    config: InferenceReportConfig,
) -> dict[str, Any]:
    """Materialize ``viewer_config`` for the interactive sweep viewer."""
    return ViewerConfigBuilder(profile, config).build()
