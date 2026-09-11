'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

TorchTitan adapters for the shared sweep Run Deck.
'''

from cvs.lib.report.rundeck.config_builder import provenance_link_rows, thresholds_run_card_row
from cvs.lib.training.torchtitan.utils.training_config_loader import parse_sweep_cell_key
from cvs.lib.utils.verdict import _check_one


METRIC_UNITS = {
    "tokens_per_sec": "tok/s",
    "throughput_per_gpu": "tok/s/GPU",
    "loss": "",
    "mem_usage_gb": "GiB",
    "scaling_efficiency_pct": "%",
    "steps_to_target": "steps",
    "time_to_target_seconds": "s",
}

_PARALLELISM_FIELDS = (
    ("TP", "tensor_parallel_degree"),
    ("PP", "pipeline_parallel_degree"),
    ("CP", "context_parallel_degree"),
    ("EP", "expert_parallel_degree"),
    ("DP shard", "data_parallel_shard_degree"),
)


class TorchTitanRundeckVariant:
    def __init__(self, variant, nnodes):
        self._variant = variant
        self.nnodes = nnodes

    def __getattr__(self, name):
        return getattr(self._variant, name)


def _model_name(variant):
    params = variant.model_params
    return params.get("hf_model_name") or params.get("model_name") or "\u2014"


def _parallelism(variant):
    params = variant.model_params
    return ", ".join(f"{label}={params.get(field, '-')}" for label, field in _PARALLELISM_FIELDS)


def torchtitan_run_card_display(variant, provenance):
    rows = [
        ("Model", _model_name(variant), False),
        ("GPU", variant.gpu_arch, False),
        ("nnodes", str(variant.nnodes), False),
        ("Parallelism", _parallelism(variant), False),
        thresholds_run_card_row(variant),
    ]
    rows.extend(provenance_link_rows(provenance))
    return rows


def torchtitan_tier_metric_specs(thresholds_cell, tier):
    if tier != "gates":
        return {}
    return {metric: spec for metric, spec in thresholds_cell.items() if spec.get("kind") != "info"}


def torchtitan_metric_verdict(metric, actual, spec):
    if actual is None:
        return "fail", f"{metric}: value is unavailable"
    error = _check_one(metric, actual, spec)
    return ("fail", error) if error else ("pass", "")


def _latest_value(value):
    if isinstance(value, (list, tuple)):
        return value[-1] if value else None
    return value


def update_torchtitan_rundeck_results(target, results, variant):
    for cell_id, raw_metrics in results.items():
        combo = variant.sweep.combinations[cell_id]
        parsed = parse_sweep_cell_key(cell_id)
        actuals = {
            f"training.{metric}": _latest_value(value)
            for metric, value in raw_metrics.items()
            if not metric.startswith("_")
        }
        actuals.update(
            {
                "_rundeck_cell_id": cell_id,
                "nnodes": variant.nnodes,
                "parallelism": _parallelism(variant),
            }
        )
        key = (
            _model_name(variant),
            variant.gpu_arch,
            parsed["micro_batch_size"],
            parsed["global_batch_size"],
            cell_id,
            combo.precision,
        )
        target[key] = {"all nodes": actuals}
    return target


__all__ = [
    "METRIC_UNITS",
    "TorchTitanRundeckVariant",
    "torchtitan_metric_verdict",
    "torchtitan_run_card_display",
    "torchtitan_tier_metric_specs",
    "update_torchtitan_rundeck_results",
]
