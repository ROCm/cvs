"""Deck-facing adapter that feeds an Aorta run to the shared training Run Deck.

Copyright 2026 Advanced Micro Devices, Inc. All rights reserved.
"""

import statistics

import pytest

from cvs.lib.benchmark.aorta.aorta_thresholds import (
    THRESHOLD_METRICS,
    active_thresholds,
    threshold_actual,
    threshold_verdicts,
)
from cvs.parsers.aorta_report import AortaReportParser
from cvs.parsers.tracelens import TraceLensParser

# The shared training cell flattener emits actuals under this prefix.
METRIC_PREFIX = "training."

METRIC_UNITS = {
    "avg_iteration_time_ms": "ms",
    "std_iteration_time_ms": "ms",
    "min_iteration_time_ms": "ms",
    "max_iteration_time_ms": "ms",
    "time_variance_ratio": "ratio",
    "avg_compute_time_ms": "ms",
    "avg_comm_time_ms": "ms",
    "avg_compute_ratio": "ratio",
    "avg_comm_ratio": "ratio",
    "avg_overlap_ratio": "ratio",
    "ranks": "ranks",
}

METRIC_TIERS = {
    "iteration_time": ("training.avg_iteration_time_ms",),
    "compute_ratio": ("training.avg_compute_ratio",),
    "overlap_ratio": ("training.avg_overlap_ratio",),
    "rank_balance": ("training.time_variance_ratio",),
    "record": tuple(
        METRIC_PREFIX + name
        for name in METRIC_UNITS
        if name not in ("avg_iteration_time_ms", "avg_compute_ratio", "avg_overlap_ratio", "time_variance_ratio")
    ),
}

STAGE_LABELS = {
    "test_launch_container": "container_launch",
    "test_clone_aorta": "clone",
    "test_setup_rdma": "rdma_setup",
    "test_build_rccl": "rccl_build",
    "test_run_benchmark": "benchmark",
    "test_collect_traces": "collect_traces",
    "test_analyze": "analysis",
    "test_parse_results": "parse",
    "test_validate_thresholds": "thresholds",
    "test_generate_report": "report",
    "test_teardown": "teardown",
}


def tier_metric_specs(thresholds_cell, tier):
    """Return configured metric specs in one deck tier."""
    return {
        name: thresholds_cell[name]
        for name in METRIC_TIERS.get(tier, ())
        if thresholds_cell and thresholds_cell.get(name) is not None
    }


def threshold_specs(expected):
    """Map Aorta threshold keys to training cell metric specs."""
    return {
        METRIC_PREFIX + metric: {"kind": kind, "value": expected[key]}
        for key, metric, kind in THRESHOLD_METRICS
        if expected.get(key) is not None
    }


def deck_cell_id(variant_config):
    """Identify the run's single deck cell."""
    return str(getattr(getattr(variant_config, "model", None), "id", "") or "default")


def deck_results(result, cell_id):
    """Convert parsed Aorta metrics into the shared training cell shape."""
    ranks = sorted(result.per_rank_metrics, key=lambda metric: metric.rank)
    entry = {
        "avg_iteration_time_ms": [result.avg_iteration_time_ms],
        "std_iteration_time_ms": [result.std_iteration_time_us / 1000.0],
        "min_iteration_time_ms": [result.min_iteration_time_us / 1000.0],
        "max_iteration_time_ms": [result.max_iteration_time_us / 1000.0],
        "avg_compute_time_ms": [statistics.mean(rank.compute_time_us for rank in ranks) / 1000.0],
        "avg_comm_time_ms": [statistics.mean(rank.communication_time_us for rank in ranks) / 1000.0],
        "avg_compute_ratio": [result.avg_compute_ratio],
        "avg_comm_ratio": [result.avg_comm_ratio],
        "avg_overlap_ratio": [result.avg_overlap_ratio],
        "ranks": [len(ranks)],
        "_dimensions": {
            "nodes": str(result.num_nodes),
            "gpus_per_node": str(result.gpus_per_node),
            "nccl_channels": str(result.nccl_channels or ""),
            "rccl_branch": result.rccl_branch or "",
        },
    }
    variance = threshold_actual(result, "max_time_variance_ratio")
    if variance is not None:
        entry["time_variance_ratio"] = [variance]
    if len(ranks) >= 2:
        definitions = (
            ("Time", "ms", "iteration", "total_time_us", 0.001),
            ("Time", "ms", "compute", "compute_time_us", 0.001),
            ("Time", "ms", "comm", "communication_time_us", 0.001),
            ("Share of iteration", "%", "compute", "compute_ratio", 100),
            ("Share of iteration", "%", "comm", "comm_ratio", 100),
            ("Comm overlap", "%", "overlap", "compute_comm_overlap", 100),
        )
        entry["_metric_charts"] = {
            "metrics": [],
            "heatmaps": [],
            "series": [
                {
                    "group": "Per rank",
                    "name": name,
                    "unit": unit,
                    "node": node,
                    "x_label": "rank",
                    "points": [[rank.rank, getattr(rank, field) * scale] for rank in ranks],
                }
                for name, unit, node, field, scale in definitions
            ],
        }
    return {cell_id: entry}


def metrics_source(parser):
    """Name the parser path without inferring a path for each rank."""
    if parser is None:
        return "—"
    if isinstance(parser, AortaReportParser):
        return "Aorta TraceLens Excel reports"
    if isinstance(parser, TraceLensParser):
        if parser.use_tracelens:
            return "TraceLens on raw traces (basic-scan fallback possible per rank)"
        return "Basic raw-trace scan"
    return type(parser).__name__


def record_stage_report(lifecycle, test_name, nodeid, report, call):
    """Record call-phase stage timing and whether the threshold gate evaluated."""
    if report.when != "call" or report.skipped or test_name not in STAGE_LABELS:
        return
    label = STAGE_LABELS[test_name]
    stage_rows = getattr(lifecycle, "report", None)
    # This runs inside a pytest hook: a lifecycle without a timing dict must not become an internal error.
    if isinstance(stage_rows, dict):
        rows = stage_rows.setdefault(nodeid, [])
        # Sub-test reports use the same node ID; the parent's final duration wins.
        rows[:] = [row for row in rows if row[0] != label]
        rows.append((label, float(report.duration), "s"))
    if test_name == "test_validate_thresholds" and (
        report.passed or (call.excinfo is not None and call.excinfo.errisinstance(pytest.fail.Exception))
    ):
        lifecycle.thresholds_checked = True


def aorta_cell_dimensions(variant, sweep_name):
    """Return stored cluster dimensions for a deck cell."""
    return dict(((getattr(variant, "results", None) or {}).get(sweep_name) or {}).get("_dimensions") or {})


def aorta_metric_verdict(metric, actual, spec):
    """Use the parser gate's verdict when it evaluated this metric."""
    if spec.get("verdict") in ("pass", "fail"):
        # The parser's verdict is authoritative for an evaluated gate.
        return spec["verdict"], spec.get("reason") or ""
    if actual is None:
        return "na", f"{metric}: no value"
    kind = spec.get("kind")
    value = spec.get("value")
    if kind == "max" and float(actual) > float(value):
        return "fail", f"{metric}: actual {actual} > max {value}"
    if kind == "min" and float(actual) < float(value):
        return "fail", f"{metric}: actual {actual} < min {value}"
    if kind not in ("max", "min"):
        return "fail", f"{metric}: unknown threshold kind {kind!r}"
    return "pass", ""


class AortaDeckVariant:
    """Deck-facing view of an Aorta variant and the run's lifecycle state."""

    def __init__(self, variant_config, lifecycle):
        self.model = variant_config.model
        self.container = variant_config.container
        self.cell_id = deck_cell_id(variant_config)
        self.expected = active_thresholds(variant_config.thresholds)
        self._configured = bool(variant_config.enforce_thresholds)
        self._lifecycle = lifecycle

    @property
    def results(self):
        return self._lifecycle.deck_results

    @property
    def gate_ran(self):
        return bool(
            self._configured
            and self._lifecycle.thresholds_checked
            and self._lifecycle.benchmark_result is not None
            and self._lifecycle.parser is not None
        )

    @property
    def enforce_thresholds(self):
        return self.gate_ran

    @property
    def threshold_state(self):
        if self.gate_ran:
            return "enforced"
        return "not evaluated" if self._configured else "record-only"

    @property
    def suite_failed(self):
        return bool(self._lifecycle.failed)

    @property
    def metrics_source(self):
        return metrics_source(self._lifecycle.parser)

    def threshold_rows(self):
        """Return one row per configured threshold in gate order."""
        if self.gate_ran:
            return threshold_verdicts(self._lifecycle.parser, self._lifecycle.benchmark_result, self.expected)
        rows = []
        for key, metric, kind in THRESHOLD_METRICS:
            if self.expected.get(key) is None:
                continue
            result = self._lifecycle.benchmark_result
            rows.append(
                {
                    "threshold": key,
                    "metric": metric,
                    "kind": kind,
                    "limit": self.expected[key],
                    "actual": threshold_actual(result, key) if result is not None else None,
                    "status": "not evaluated" if self._configured else "record",
                    "message": "",
                }
            )
        return rows

    @property
    def thresholds(self):
        specs = threshold_specs(self.expected)
        if self.gate_ran:
            for verdict in self.threshold_rows():
                spec = specs[METRIC_PREFIX + verdict["metric"]]
                spec["verdict"] = verdict["status"]
                spec["reason"] = verdict["message"]
        return {self.cell_id: specs}
