'''Build structured cell records from ``inf_res_dict`` for reports and row extras.'''

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from cvs.lib.report.formatting import fmt_num
from cvs.lib.report.types import InferenceReportConfig
from cvs.lib.report.verdict import _check_one


def metric_pass(metric: str, actual: Any, spec: Optional[dict]) -> str:
    if spec is None or actual is None:
        return "na"
    err = _check_one(metric, actual, spec)
    return "fail" if err else "pass"


def bar_pct(actual: float, spec: dict) -> float:
    kind = spec.get("kind", "")
    target = float(spec["value"])
    if target <= 0:
        return 0.0
    if kind in ("min", "min_tok_s", "min_ratio"):
        return min(100.0, 100.0 * float(actual) / target)
    if kind in ("max", "max_ms"):
        if float(actual) <= 0:
            return 100.0
        return min(100.0, 100.0 * target / float(actual))
    if kind == "within":
        return 100.0
    return 50.0


def margin_text(actual: Any, spec: Optional[dict]) -> Optional[str]:
    if spec is None or actual is None:
        return None
    try:
        act = float(actual)
        target = float(spec["value"])
    except (TypeError, ValueError):
        return None
    kind = spec.get("kind", "")
    if kind in ("min", "min_tok_s", "min_ratio"):
        diff = act - target
        if target:
            return f"+{fmt_num(diff)} ({100.0 * diff / target:.0f}% above gate)"
        return f"+{fmt_num(diff)} above gate"
    if kind in ("max", "max_ms"):
        diff = target - act
        return f"{fmt_num(diff)} under gate"
    return None


class CellRecordBuilder:
    """Build cell dicts for payload, row extras, and CI summaries."""

    def __init__(self, config: InferenceReportConfig):
        self.config = config

    def tier_status(
        self,
        actuals: Mapping[str, Any],
        thresholds_cell: dict,
        tier: str,
        enforce: bool,
    ) -> str:
        if not enforce or tier == self.config.record_tier:
            return "record"
        specs = self.config.tier_metric_specs(thresholds_cell, tier)
        if not specs:
            return "na"
        for metric, spec in specs.items():
            if metric_pass(metric, actuals.get(metric), spec) == "fail":
                return "fail"
        return "pass"

    def resolve_pytest_nodeids(self, concurrency: Any) -> dict[str, str]:
        conc_suffix = f"-{concurrency}]"
        inference_nid = ""
        metrics_nid = ""
        for nodeid in self._lifecycle_report():
            if conc_suffix not in nodeid:
                continue
            if self.config.inference_test_substring in nodeid:
                inference_nid = inference_nid or nodeid
            if "test_cell_metrics" in nodeid or "test_metric" in nodeid:
                if not metrics_nid or "test_cell_metrics" in nodeid:
                    metrics_nid = nodeid
        return {
            "pytest_inference_nodeid": inference_nid,
            "pytest_metrics_nodeid": metrics_nid,
        }

    def _lifecycle_report(self) -> Mapping[str, list]:
        return getattr(self, "_ctx_lifecycle", {})

    def lifecycle_for_cell(self, lifecycle_report: Mapping[str, list], concurrency: Any) -> Dict[str, float]:
        conc = str(concurrency)
        suffix = f"-{conc}]"
        out: Dict[str, float] = {}
        for nodeid, rows in lifecycle_report.items():
            if self.config.inference_test_substring not in nodeid or suffix not in nodeid:
                continue
            for label, value, unit in rows:
                if unit != "s" or label not in self.config.cell_lifecycle_labels:
                    continue
                try:
                    out[label] = float(value)
                except (TypeError, ValueError):
                    continue
        return out

    def build_one(
        self,
        *,
        key: tuple,
        host: str,
        actuals: Mapping[str, Any],
        variant_config,
        lifecycle_report: Mapping[str, list],
        multi_host: bool,
    ) -> dict:
        model, gpu, isl, osl, policy, conc = key
        cell_id = variant_config.cell_key(isl, osl, conc)
        thresholds_map = getattr(variant_config, "thresholds", {}) or {}
        thresholds_cell = thresholds_map.get(cell_id) or {}
        enforce = bool(getattr(variant_config, "enforce_thresholds", False))

        metrics = []
        for short, label in self.config.cell_highlights:
            full = self.config.full_metric(short)
            spec = thresholds_cell.get(full)
            actual = actuals.get(full)
            metrics.append(
                {
                    "label": label,
                    "metric": full,
                    "actual": actual,
                    "unit": self.config.metric_units.get(short, ""),
                    "spec": spec,
                    "status": metric_pass(full, actual, spec) if enforce and spec else "record",
                    "bar_pct": bar_pct(float(actual), spec) if spec is not None and actual is not None else None,
                    "margin": margin_text(actual, spec) if spec else None,
                }
            )

        tiers = {
            tier: self.tier_status(actuals, thresholds_cell, tier, enforce) for tier in self.config.metric_tier_order
        }
        self._ctx_lifecycle = lifecycle_report
        pytest_links = self.resolve_pytest_nodeids(conc)

        return {
            "model": model,
            "gpu": gpu,
            "isl": isl,
            "osl": osl,
            "policy": policy,
            "concurrency": conc,
            "host": host,
            "show_host_in_label": multi_host,
            "cell_id": cell_id,
            "metrics": metrics,
            "tiers": tiers,
            "actuals": dict(actuals),
            "cell_lifecycle": self.lifecycle_for_cell(lifecycle_report, conc),
            **pytest_links,
        }

    def build_all(
        self,
        *,
        variant_config,
        inf_res_dict: Mapping[tuple, Any],
        lifecycle_report: Mapping[str, list],
    ) -> List[dict]:
        cells: List[dict] = []
        for key, host_dict in sorted(
            inf_res_dict.items(),
            key=lambda kv: (kv[0][4], kv[0][5]) if isinstance(kv[0], tuple) and len(kv[0]) >= 6 else (0, 0),
        ):
            if not isinstance(key, tuple) or len(key) != 6:
                continue
            if not isinstance(host_dict, dict) or not host_dict:
                continue
            multi_host = len(host_dict) > 1
            for host, actuals in sorted(host_dict.items()):
                cells.append(
                    self.build_one(
                        key=key,
                        host=host,
                        actuals=actuals,
                        variant_config=variant_config,
                        lifecycle_report=lifecycle_report,
                        multi_host=multi_host,
                    )
                )
        return cells

    @staticmethod
    def select_summary(cells: List[dict], limit: int, *, gated_tiers: tuple[str, ...]) -> List[dict]:
        if limit <= 0 or len(cells) <= limit:
            return cells
        fails = [c for c in cells if cell_has_gated_failure(c, gated_tiers)]
        rest = [c for c in cells if not cell_has_gated_failure(c, gated_tiers)]
        selected = list(fails[:limit])
        if len(selected) < limit:
            selected.extend(rest[: limit - len(selected)])
        return selected


def tier_status(
    config: InferenceReportConfig,
    actuals: Mapping[str, Any],
    thresholds_cell: dict,
    tier: str,
    enforce: bool,
) -> str:
    return CellRecordBuilder(config).tier_status(actuals, thresholds_cell, tier, enforce)


def resolve_pytest_nodeids_for_cell(
    config: InferenceReportConfig,
    lifecycle_report: Mapping[str, list],
    concurrency: Any,
) -> dict[str, str]:
    builder = CellRecordBuilder(config)
    builder._ctx_lifecycle = lifecycle_report
    return builder.resolve_pytest_nodeids(concurrency)


def lifecycle_for_cell(
    config: InferenceReportConfig,
    lifecycle_report: Mapping[str, list],
    concurrency: Any,
) -> Dict[str, float]:
    return CellRecordBuilder(config).lifecycle_for_cell(lifecycle_report, concurrency)


def build_cell_record(
    config: InferenceReportConfig,
    *,
    key: tuple,
    host: str,
    actuals: Mapping[str, Any],
    variant_config,
    lifecycle_report: Mapping[str, list],
    multi_host: bool,
) -> dict:
    return CellRecordBuilder(config).build_one(
        key=key,
        host=host,
        actuals=actuals,
        variant_config=variant_config,
        lifecycle_report=lifecycle_report,
        multi_host=multi_host,
    )


def build_all_cells(
    config: InferenceReportConfig,
    *,
    variant_config,
    inf_res_dict: Mapping[tuple, Any],
    lifecycle_report: Mapping[str, list],
) -> List[dict]:
    return CellRecordBuilder(config).build_all(
        variant_config=variant_config,
        inf_res_dict=inf_res_dict,
        lifecycle_report=lifecycle_report,
    )


def cell_has_gated_failure(cell: dict, gated_tiers: tuple[str, ...]) -> bool:
    tiers = cell.get("tiers") or {}
    return any(tiers.get(t) == "fail" for t in gated_tiers)


def cell_has_any_tier_failure(cell: dict) -> bool:
    return any(t == "fail" for t in (cell.get("tiers") or {}).values())


def select_summary_cells(
    cells: List[dict],
    limit: int,
    *,
    gated_tiers: tuple[str, ...],
) -> List[dict]:
    return CellRecordBuilder.select_summary(cells, limit, gated_tiers=gated_tiers)
