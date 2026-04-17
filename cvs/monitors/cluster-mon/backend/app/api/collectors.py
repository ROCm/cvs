"""
Collectors status API endpoint.
Returns per-collector state and aggregate overall_status.
"""

from fastapi import APIRouter
from typing import Any

router = APIRouter()


def _compute_overall_status(collector_results: dict, collectors_meta: dict) -> str:
    """
    Compute aggregate status from per-collector results.

    - "healthy"  : all collectors in OK or NO_SERVICE state
    - "degraded" : some collectors erroring, but no critical ones
    - "critical" : any collector marked critical=True is in ERROR or UNREACHABLE
    """
    if not collector_results:
        return "healthy"

    for name, result in collector_results.items():
        state = result.state if hasattr(result, 'state') else result.get('state', 'ok')
        state_str = state.value if hasattr(state, 'value') else str(state)
        is_error = state_str in ("error", "unreachable")
        is_critical = collectors_meta.get(name, {}).get('critical', False)
        if is_error and is_critical:
            return "critical"

    any_error = any(
        (r.state.value if hasattr(r.state, 'value') else str(r.state)) in ("error", "unreachable")
        for r in collector_results.values()
        if hasattr(r, 'state')
    )
    return "degraded" if any_error else "healthy"


@router.get("/status")
async def get_collectors_status() -> dict[str, Any]:
    """
    Return per-collector state and aggregate overall_status.

    Response shape:
    {
      "gpu":  {"state": "ok", "timestamp": "...", "error": null},
      "nic":  {"state": "ok", "timestamp": "...", "error": null},
      "rccl": {"state": "no_service", "timestamp": "...", "error": "..."},
      "overall_status": "healthy"
    }
    """
    from app.main import app_state, REGISTERED_COLLECTORS

    # Build collectors metadata (critical flag) from REGISTERED_COLLECTORS
    collectors_meta = {
        cls.name: {"critical": getattr(cls, "critical", False)}
        for cls in REGISTERED_COLLECTORS
    }

    result: dict[str, Any] = {}
    for name, collector_result in app_state.collector_results.items():
        state_val = (
            collector_result.state.value
            if hasattr(collector_result.state, 'value')
            else str(collector_result.state)
        )
        result[name] = {
            "state": state_val,
            "timestamp": collector_result.timestamp,
            "error": collector_result.error,
        }

    result["overall_status"] = _compute_overall_status(
        app_state.collector_results, collectors_meta
    )
    return result
