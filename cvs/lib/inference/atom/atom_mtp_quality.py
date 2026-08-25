'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Pure helpers for MTP speculative-decode quality metrics (ACC-4/5/13).
'''

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Optional

_ACCEPTANCE_RATE_RE = re.compile(
    r"(?:acceptance[_\s-]?rate|draft[_\s-]?acceptance)[=:\s]+([0-9]*\.?[0-9]+)",
    re.I,
)
_SPEC_TOKENS_RE = re.compile(
    r"(?:speculative[_\s-]?tokens|avg[_\s-]?spec[_\s-]?tokens)[=:\s]+([0-9]*\.?[0-9]+)",
    re.I,
)
_REPEAT_RUN_RE = re.compile(r"(.{1,32})\1{4,}", re.S)


def parse_mtp_log_metrics(log_text: str) -> Dict[str, float]:
    """Scrape MTP telemetry lines from a server or bench log tail."""
    out: Dict[str, float] = {}
    if not log_text:
        return out
    m = _ACCEPTANCE_RATE_RE.search(log_text)
    if m:
        out["mtp.acceptance_rate"] = float(m.group(1))
    m = _SPEC_TOKENS_RE.search(log_text)
    if m:
        out["mtp.speculative_tokens_avg"] = float(m.group(1))
    return out


def degenerate_decode_ratio(text: str) -> float:
    """Fraction of responses that are empty or dominated by short repeats."""
    if not text or not text.strip():
        return 1.0
    stripped = text.strip()
    if len(stripped) < 8:
        return 1.0
    if _REPEAT_RUN_RE.search(stripped):
        return 1.0
    return 0.0


def chat_template_sha256(completion_text: str) -> str:
    normalized = (completion_text or "").strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def chat_template_ok(completion_text: str, expected_sha256: str) -> Optional[float]:
    """Return 1.0 when hash matches expected; None when no expected hash configured."""
    expected = (expected_sha256 or "").strip().lower()
    if not expected:
        return None
    actual = chat_template_sha256(completion_text)
    return 1.0 if actual == expected else 0.0


def extract_completion_text(payload: Any) -> str:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return payload
    if not isinstance(payload, dict):
        return str(payload or "")
    choices = payload.get("choices") or []
    if not choices:
        return ""
    first = choices[0] or {}
    message = first.get("message") or {}
    if message.get("content"):
        return str(message["content"])
    return str(first.get("text") or "")
