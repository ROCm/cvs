'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Megatron Run Deck hooks.
'''

from cvs.lib.report.rundeck.config_builder import provenance_link_rows, thresholds_run_card_row
from cvs.lib.utils.verdict import _check_one


MEGATRON_METRIC_UNITS = {
    "throughput_per_gpu": "TFLOP/s/GPU",
    "tokens_per_gpu": "tok/s/GPU",
    "elapsed_time_per_iteration": "ms",
    "mem_usage": "",
    "scaling_efficiency_pct": "%",
    "lm_loss": "",
    "grad_norm": "",
    "hip_mem_usage_ratio": "%",
    "rocm_mem_usage_ratio": "%",
    "steps_to_target": "steps",
    "time_to_target_seconds": "s",
}


def megatron_tier_metric_specs(thresholds_cell, tier):
    if tier == "thresholds":
        return dict(thresholds_cell or {})
    return {}


def megatron_metric_verdict(metric, actual, spec):
    if actual is None:
        if spec.get("optional"):
            return "pass", f"{metric} is optional and was not reported"
        return "fail", f"{metric} was not reported"
    error = _check_one(metric, actual, spec)
    return ("fail", error) if error else ("pass", "")


def megatron_run_card_display(variant, provenance):
    train_params = getattr(variant, "train_params", {}) or {}
    env = getattr(getattr(variant, "container", None), "env", {}) or {}
    parallelism = ", ".join(
        (
            f"TP={train_params.get('tensor_parallelism', '—')}",
            f"PP={train_params.get('pipeline_parallelism', '—')}",
            f"FSDP={train_params.get('fsdp', '—')}",
        )
    )
    rows = [
        ("Model", train_params.get("model_name") or train_params.get("model") or "—", False),
        ("GPU", getattr(variant, "gpu_arch", "—"), False),
        ("Nodes", str(env.get("NNODES") or train_params.get("nnodes") or "—"), False),
        ("Parallelism", parallelism, False),
        thresholds_run_card_row(variant),
    ]
    rows.extend(provenance_link_rows(provenance))
    return rows


__all__ = [
    "MEGATRON_METRIC_UNITS",
    "megatron_metric_verdict",
    "megatron_run_card_display",
    "megatron_tier_metric_specs",
]
