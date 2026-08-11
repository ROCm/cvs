'''Extract accuracy metrics recorded in pytest lifecycle rows.'''

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


def extract_accuracy_from_lifecycle(lifecycle_report: Mapping[str, list]) -> Dict[str, float]:
    """Flatten ``test_accuracy_eval`` lifecycle records into metric keys."""
    out: Dict[str, float] = {}
    for nodeid, rows in lifecycle_report.items():
        if "test_accuracy_eval" not in nodeid:
            continue
        for label, value, unit in rows:
            if unit == "s":
                continue
            if "." not in str(label):
                continue
            try:
                out[str(label)] = float(value)
            except (TypeError, ValueError):
                continue
    return out


def build_accuracy_prev_run_panel(
    current: Mapping[str, float],
    baseline_payload: Mapping[str, Any],
    *,
    metric_key: str = "gsm8k_flex.gsm8k.exact_match__flexible-extract",
    max_drop: float = 0.01,
) -> Optional[dict]:
    baseline = baseline_payload.get("accuracy") or {}
    if not isinstance(baseline, dict):
        return None
    current_val = current.get(metric_key)
    baseline_val = baseline.get(metric_key)
    if current_val is None or baseline_val is None:
        return None
    try:
        delta = float(current_val) - float(baseline_val)
    except (TypeError, ValueError):
        return None
    regression = delta < -max_drop
    return {
        "metric_key": metric_key,
        "max_drop": max_drop,
        "current": current_val,
        "baseline": baseline_val,
        "compare.prev_run.gsm8k_delta": delta,
        "regression": regression,
    }
