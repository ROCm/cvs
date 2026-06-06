"""Workload hash: sha256(image_digest || canonical(workload) || canonical(thresholds))[:12].

Canonical JSON form is pinned so whitespace edits don't change the hash.
image_digest is required (fail-loud if missing) so a hash collision can't
happen on identical configs with different images.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

HASH_LEN = 12


def canonical_json(obj: Any) -> bytes:
    """RFC-8785-ish: sorted keys, compact separators, ASCII-safe."""
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def workload_hash(*, image_digest: str, workload: dict, thresholds: dict) -> str:
    if not image_digest:
        raise ValueError("image_digest is required for workload_hash (fail-loud)")
    h = hashlib.sha256()
    h.update(image_digest.encode("ascii"))
    h.update(b"\x00")
    h.update(canonical_json(workload))
    h.update(b"\x00")
    h.update(canonical_json(thresholds))
    return h.hexdigest()[:HASH_LEN]
