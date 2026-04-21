"""Filename-safe ``case_id`` helpers (no config / I/O imports)."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", value)


def _no_matrix_case_id(collective_index: int, collective: str) -> str:
    return f"c{collective_index}_{_slug(collective)}"


def _canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _ensure_unique_case_id(base: str, used: set[str], resolved: dict[str, Any]) -> str:
    if base not in used:
        used.add(base)
        return base
    digest = hashlib.sha256(_canonical_json_bytes(resolved)).hexdigest()[:8]
    n = 0
    while True:
        suffix = f"__dup{digest}" + (f"_{n}" if n else "")
        candidate = f"{base}{suffix}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        n += 1
