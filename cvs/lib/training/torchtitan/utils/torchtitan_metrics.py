'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Metric vocabulary for TorchTitan Run Deck profiles.

Names match the keys ``summarize_step_metrics`` and the training tests store
under ``training.<name>``. Throughput is tokens/s/device (``tps``), not TFLOP/s.
'''

METRIC_UNITS = {
    "tokens_per_sec": "tok/s/device",
    "tflops": "TFLOP/s/device",
    "loss": "nats",
    "mem_usage_gb": "GiB",
    "grad_norm": "L2",
    "step_time_p50_ms": "ms",
    "step_time_p95_ms": "ms",
    "scaling_efficiency_pct": "%",
    "steps_to_target": "steps",
    "time_to_target_seconds": "s",
}

METRIC_TIERS = {
    "throughput": ("training.tokens_per_sec", "training.tflops"),
    "latency": ("training.step_time_p50_ms", "training.step_time_p95_ms"),
    "health": ("training.loss", "training.mem_usage_gb", "training.grad_norm"),
    "record": (
        "training.scaling_efficiency_pct",
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
