'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Megatron Run Deck hooks.
'''

from types import SimpleNamespace

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


def build_legacy_rundeck_variant(raw, stem, training_dict, model_params, gpu_type, variant_config):
    """Map a legacy Llama JSON config onto the named-cell variant, or None on mismatch."""
    if not isinstance(raw, dict) or "config" not in raw:
        return variant_config
    try:
        distributed = str(stem or "").endswith("_distributed")
        model_name = "llama3_1_70b" if "70b" in str(stem or "") else "llama3_1_8b"
        topology = "multi_node" if distributed else "single_node"
        params = dict(model_params[topology][model_name][gpu_type])
        cell_id = f"MBS={params['micro_batch_size']},GBS={params['batch_size']},PRECISION={params['precision']}"
        threshold_specs = {
            f"training.{metric}": {"kind": "min", "value": value}
            for metric, value in (params.get("result_dict") or {}).items()
            if not str(metric).startswith("_")
        }
        params["model_name"] = model_name
        params["nnodes"] = str((training_dict or {}).get("nnodes") or 1)
        combo = SimpleNamespace(
            micro_batch_size=str(params["micro_batch_size"]),
            global_batch_size=str(params["batch_size"]),
            precision=str(params["precision"]),
        )
        return SimpleNamespace(
            train_params=params,
            gpu_arch=str(gpu_type).upper(),
            enforce_thresholds=True,
            thresholds={cell_id: threshold_specs},
            sweep=SimpleNamespace(combinations={cell_id: combo}, runs=[cell_id]),
            container=SimpleNamespace(env={"NNODES": params["nnodes"]}),
        )
    except (KeyError, TypeError, AttributeError):
        return None


def megatron_run_card_display(variant, provenance):
    if variant is None:
        variant = SimpleNamespace()
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
    "build_legacy_rundeck_variant",
    "megatron_metric_verdict",
    "megatron_run_card_display",
    "megatron_tier_metric_specs",
]
