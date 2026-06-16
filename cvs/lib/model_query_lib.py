'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. ...
'''

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Union


JsonDict = dict[str, Any]

# OpenAI-compatible HTTP smoke-verify 
OPENAI_COMPATIBLE_VERIFY_TIMEOUT_S = 120.0
OPENAI_COMPATIBLE_VERIFY_CHAT_MAX_TOKENS = 8
OPENAI_COMPATIBLE_VERIFY_COMPLETION_MAX_TOKENS = 50
OPENAI_COMPATIBLE_VERIFY_STRUCTURED_BOOK_MAX_TOKENS = 256
OPENAI_COMPATIBLE_VERIFY_CHAT_USER = "Reply with exactly one word: OK."
OPENAI_COMPATIBLE_VERIFY_COMPLETION_PROMPT = "The capital of France is"
OPENAI_COMPATIBLE_VERIFY_STRUCTURED_BOOK_SYSTEM = (
    "Respond with a single JSON object only. "
    "No markdown or text outside the JSON."
)
OPENAI_COMPATIBLE_VERIFY_STRUCTURED_BOOK_USER = (
    "Return one book as a JSON object with keys: "
    "title (string), author (string), year (integer), genre (string)."
)

OPENAI_PROBE_STEP_TITLES: dict[str, str] = {
    "model_endpoint": "Model endpoint — GET /v1/models",
    "chat_completion_endpoint": "Chat completion endpoint — POST /v1/chat/completions",
    "completion_endpoint": "Completion endpoint — POST /v1/completions",
    "structured_output_book": (
        "Structured output (book) — POST /v1/chat/completions "
        "(response_format: json_object)"
    ),
}

# OpenAI probe script
def openai_probe_script(
        port: int,
        model: str,
        *,
        timeout_s: float = OPENAI_COMPATIBLE_VERIFY_TIMEOUT_S,
        chat_max_tokens: int = OPENAI_COMPATIBLE_VERIFY_CHAT_MAX_TOKENS,
        completion_max_tokens: int = OPENAI_COMPATIBLE_VERIFY_COMPLETION_MAX_TOKENS,
        structured_book_max_tokens: int = OPENAI_COMPATIBLE_VERIFY_STRUCTURED_BOOK_MAX_TOKENS,
    ) -> str:
    """Stdlib-only script for docker exec; bodies built from shared verify strings."""
    chat_messages = [{"role": "user", "content": OPENAI_COMPATIBLE_VERIFY_CHAT_USER}]
    book_messages = [
        {"role": "system", "content": OPENAI_COMPATIBLE_VERIFY_STRUCTURED_BOOK_SYSTEM},
        {"role": "user", "content": OPENAI_COMPATIBLE_VERIFY_STRUCTURED_BOOK_USER},
    ]
    return "\n".join(
        [
            "import json, urllib.request, urllib.error",
            f"PORT = {int(port)}",
            f"TIMEOUT = {float(timeout_s)}",
            f"CHAT_MAX = {int(chat_max_tokens)}",
            f"COMP_MAX = {int(completion_max_tokens)}",
            f"BOOK_MAX = {int(structured_book_max_tokens)}",
            f"MODEL = {json.dumps(model)}",
            'BASE = "http://0.0.0.0:%d" % PORT',
            "",
            "def req(method, path, body=None):",
            "    url = BASE + path",
            "    data = None if body is None else json.dumps(body).encode('utf-8')",
            "    hdrs = {}",
            "    if body is not None:",
            '        hdrs["Content-Type"] = "application/json"',
            "    r = urllib.request.Request(url, data=data, headers=hdrs, method=method)",
            "    try:",
            "        with urllib.request.urlopen(r, timeout=TIMEOUT) as resp:",
            '            raw = resp.read().decode("utf-8", errors="replace")',
            "            code = int(getattr(resp, 'status', 200))",
            "            try:",
            "                return code, json.loads(raw) if raw else {}",
            "            except json.JSONDecodeError:",
            "                return code, raw",
            "    except urllib.error.HTTPError as e:",
            '        raw = e.read().decode("utf-8", errors="replace")',
            "        try:",
            "            return int(e.code), json.loads(raw) if raw else {}",
            "        except json.JSONDecodeError:",
            "            return int(e.code), raw",
            "",
            "out = {}",
            'out["model_endpoint"] = req("GET", "/v1/models")',
            'out["chat_completion_endpoint"] = req("POST", "/v1/chat/completions", {',
            '    "model": MODEL,',
            f'    "messages": {json.dumps(chat_messages)},',
            '    "max_tokens": CHAT_MAX,',
            '    "temperature": 0.0,',
            "})",
            'out["completion_endpoint"] = req("POST", "/v1/completions", {',
            '    "model": MODEL,',
            f'    "prompt": {json.dumps(OPENAI_COMPATIBLE_VERIFY_COMPLETION_PROMPT)},',
            '    "max_tokens": COMP_MAX,',
            '    "temperature": 0.0,',
            "})",
            'out["structured_output_book"] = req("POST", "/v1/chat/completions", {',
            '    "model": MODEL,',
            f'    "messages": {json.dumps(book_messages)},',
            '    "max_tokens": BOOK_MAX,',
            '    "temperature": 0.0,',
            '    "response_format": {"type": "json_object"},',
            "})",
            "print(json.dumps(out, default=str))",
        ]
    )

# Log OpenAI probe results
def log_openai_probe_results(results: dict[str, tuple[int, Any]], logger: Any) -> None:
    """One headed block per endpoint for logs / HTML capture."""
    for step, (code, body) in results.items():
        title = OPENAI_PROBE_STEP_TITLES.get(step, step)
        logger.info("#================ %s ================#", title)
        logger.info("HTTP status: %s", code)
        if isinstance(body, (dict, list)):
            try:
                logger.info("Body:\n%s", json.dumps(body, indent=2, default=str))
            except (TypeError, ValueError):
                logger.info("Body: %s", body)
        else:
            logger.info("Body: %s", body)