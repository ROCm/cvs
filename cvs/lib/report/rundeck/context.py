'''Resolve dot-path bindings against the Run Deck publish context.'''

from __future__ import annotations

from typing import Any


def resolve_bind(context: dict[str, Any], bind: str) -> Any:
    """Walk ``bind`` (for example ``datasets.sweep.results_table``) through *context*."""
    cur: Any = context
    for part in bind.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (str, bytes)) and not value:
        return True
    if isinstance(value, (list, tuple, dict, set)) and len(value) == 0:
        return True
    return False
