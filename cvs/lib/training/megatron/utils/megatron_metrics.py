'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Metric vocabulary for Megatron Run Deck profiles.
'''

METRIC_UNITS = {
    "throughput_per_gpu": "TFLOP/s/GPU",
    "tokens_per_gpu": "tok/s/GPU",
    "elapsed_time_per_iteration": "s",
    "step_time_p50_ms": "ms",
    "step_time_p95_ms": "ms",
    "mem_usage": "ratio",
    "scaling_efficiency_pct": "%",
    "steps_to_target": "steps",
    "time_to_target_seconds": "s",
}

METRIC_TIERS = {
    "throughput": ("training.throughput_per_gpu", "training.tokens_per_gpu"),
    "latency": ("training.elapsed_time_per_iteration",),
    "health": ("training.mem_usage",),
    "record": (
        "training.scaling_efficiency_pct",
        "training.step_time_p50_ms",
        "training.step_time_p95_ms",
        "training.steps_to_target",
        "training.time_to_target_seconds",
    ),
}


def tier_metric_specs(thresholds_cell, tier):
    names = METRIC_TIERS.get(tier, ())
    specs = {}
    for name in names:
        spec = (thresholds_cell or {}).get(name)
        if spec is not None:
            specs[name] = spec
    return specs
