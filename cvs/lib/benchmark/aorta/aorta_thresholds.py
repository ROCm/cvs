"""Aorta threshold table and per-threshold verdicts from the parser gate.

Copyright 2026 Advanced Micro Devices, Inc. All rights reserved.
"""

THRESHOLD_METRICS = (
    ("max_avg_iteration_ms", "avg_iteration_time_ms", "max"),
    ("min_compute_ratio", "avg_compute_ratio", "min"),
    ("min_overlap_ratio", "avg_overlap_ratio", "min"),
    ("max_time_variance_ratio", "time_variance_ratio", "max"),
)


def active_thresholds(thresholds):
    """Return configured thresholds with null values dropped."""
    return {
        key: value for key, value in ((thresholds or {}).get("expected_results") or {}).items() if value is not None
    }


def threshold_actual(result, key):
    """Return the value used by the parser gate for a threshold."""
    fields = {
        "max_avg_iteration_ms": result.avg_iteration_time_ms,
        "min_compute_ratio": result.avg_compute_ratio,
        "min_overlap_ratio": result.avg_overlap_ratio,
        "max_time_variance_ratio": (
            result.std_iteration_time_us / result.avg_iteration_time_us if result.avg_iteration_time_us > 0 else None
        ),
    }
    return fields[key]


def threshold_verdicts(parser, result, expected):
    """Return ordered verdicts from the parser's per-threshold gate."""
    verdicts = []
    for key, metric, kind in THRESHOLD_METRICS:
        if expected.get(key) is None:
            continue
        limit = expected[key]
        failures = parser.validate_thresholds(result, {key: limit})
        verdicts.append(
            {
                "threshold": key,
                "metric": metric,
                "kind": kind,
                "limit": limit,
                "actual": threshold_actual(result, key),
                "status": "fail" if failures else "pass",
                "message": "; ".join(failures),
            }
        )
    return verdicts
