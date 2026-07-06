'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Metric vocabulary for the pytorch_vision suite.

Pure-transform layer (no I/O): maps the JSON artifact the in-container benchmark
script writes into a namespaced `{"vision.<name>": value}` dict, exactly the way
`cvs.lib.inference.utils.vllm_parsing.to_client_metrics` does for vLLM. The
fetch (reading the artifact off the container) lives in the job class; only the
transform lives here so any future vision suite variant can reuse it.

`VISION_METRICS` is the ordered display surface (one HTML row per metric per
cell). `GATED_METRICS` is the asserted subset -- a metric is record-only until
its short name is added here, at which point the loader's coverage check forces
a threshold spec for it in every cell before the suite can go green.
'''

from __future__ import annotations

from typing import Any, Dict, Optional


# Ordered (short_name, unit) display surface. pytest_generate_tests iterates this
# to emit one test_metric row per metric per cell.
VISION_METRICS = [
    ("throughput_img_s", "img/s"),
    ("latency_ms_mean", "ms"),
    ("latency_ms_p50", "ms"),
    ("latency_ms_p90", "ms"),
    ("latency_ms_p99", "ms"),
    ("images", "count"),
]

VISION_METRIC_UNITS = dict(VISION_METRICS)

# The asserted subset: out-of-range == FAILURE. Closed-world -- everything else
# in VISION_METRICS is record-only (captured + displayed, never fails). Keep this
# small until baselines are calibrated; a new gated metric forces a threshold
# spec in every cell (see VariantConfig._check_thresholds_cover_sweep, axis 2).
GATED_METRICS = {"throughput_img_s"}


def _num(value: Any) -> Optional[float]:
    """Coerce a raw value to float, degrading to None (never raising).

    A None/missing/non-numeric field renders as '-' in the results table and, if
    the metric is gated, surfaces as a loud violation in evaluate_all rather than
    a TypeError.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_vision_metrics(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Namespace the benchmark artifact's scalars 1:1 as `vision.<name>`.

    `raw` is the already-parsed JSON the in-container benchmark script emits (the
    caller reads + json.loads the artifact). Only the keys in VISION_METRICS are
    surfaced; anything else in the artifact (model, precision, batch_size) is
    passed through under its own `vision.<key>` so the results table can show it.
    """
    metrics: Dict[str, Any] = {}
    for short, _unit in VISION_METRICS:
        metrics[f"vision.{short}"] = _num(raw.get(short))
    # Pass-through descriptive fields (not asserted, handy in the table/logs).
    for key in ("model", "precision", "input_size", "batch_size", "device"):
        if key in raw:
            metrics[f"vision.{key}"] = raw[key]
    return metrics
