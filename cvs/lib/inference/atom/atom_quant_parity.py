'''
Copyright 2025 Advanced Micro Devices Inc.
All rights reserved.

ACC-7 quant parity probe — fixed-prompt completion fingerprint (lab pairing TBD).
'''

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


def extract_completion_text(response_text: str) -> str:
    try:
        payload = json.loads(response_text)
    except (json.JSONDecodeError, TypeError):
        return (response_text or "").strip()
    choices = payload.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
    message = choice.get("message") or {}
    content = message.get("content")
    if content:
        return str(content).strip()
    return str(choice.get("text") or "").strip()


def completion_fingerprint(text: str) -> str:
    normalized = (text or "").strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def run_quant_parity_probe(*, probe_text: str) -> dict[str, Any]:
    return {
        "quant_parity.probe_sha256": completion_fingerprint(probe_text),
        "quant_parity.probe_chars": len((probe_text or "").strip()),
    }


def compare_quant_fingerprints(
    current: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    cur = current.get("quant_parity.probe_sha256")
    ref = reference.get("quant_parity.probe_sha256")
    if not cur or not ref:
        return {}
    match = cur == ref
    return {
        "quant_parity.probe_match": 1.0 if match else 0.0,
        "quant_parity.probe_sha256_current": cur,
        "quant_parity.probe_sha256_reference": ref,
    }
