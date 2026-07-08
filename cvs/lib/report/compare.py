'''Numeric comparison helpers for panels and parity reports.'''

from __future__ import annotations

from typing import Any, Mapping, Optional


def metric_ratio(candidate: Optional[float], reference: Optional[float]) -> Optional[float]:
    if candidate is None or reference is None or reference == 0:
        return None
    return candidate / reference


def metric_delta_pct(current: float, previous: float) -> Optional[float]:
    if previous == 0:
        return None
    return 100.0 * (current - previous) / previous


def prev_run_change_flags(delta_pct: Optional[float], threshold_pct: float) -> tuple[bool, bool]:
    if delta_pct is None:
        return False, False
    changed = abs(delta_pct) > threshold_pct
    regression = changed and delta_pct < 0
    return changed, regression


def build_prev_run_compare_row(
    cell: Mapping[str, Any],
    prev_cell: Optional[Mapping[str, Any]],
    *,
    headline_metric: str,
    threshold_pct: float,
) -> dict:
    actuals = cell.get("actuals") or {}
    prev_actuals = (prev_cell or {}).get("actuals") or {}
    current = actuals.get(headline_metric)
    previous = prev_actuals.get(headline_metric)
    delta_pct = None
    changed = False
    regression = False
    if current is not None and previous is not None:
        try:
            delta_pct = metric_delta_pct(float(current), float(previous))
            changed, regression = prev_run_change_flags(delta_pct, threshold_pct)
        except (TypeError, ValueError):
            pass
    return {
        "cell_id": cell.get("cell_id"),
        "host": cell.get("host"),
        "concurrency": cell.get("concurrency"),
        "current_throughput": current,
        "previous_throughput": previous,
        "compare.prev_run.throughput_delta_pct": delta_pct,
        "changed": changed,
        "regression": regression,
    }
