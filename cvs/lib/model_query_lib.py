'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. ...
'''

from __future__ import annotations

import json
import re
import shlex
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

# Verify OpenAI compatible probe results
def check_openai_compatible_probe_results(
    results: dict[str, tuple[int, Any]],
    *,
    port: Any,
    logger: Optional[Any] = None,
) -> tuple[bool, Optional[str]]:
    """
    Each step: HTTP 200, non-null JSON object body, and minimal useful content —
    models list for ``model_endpoint``; for chat/completion/structured, top-level
    ``model``, ``choices``, and non-empty text or assistant content (structured:
    JSON object with title/author/year/genre).
    """
    failures: list[str] = []

    def _fail(detail: str) -> None:
        failures.append(detail)
        if logger is not None:
            logger.info(
                "OpenAI-compatible probe check FAILED: %s port=%r",
                detail,
                port,
            )

    for step, (code, body) in results.items():
        title = OPENAI_PROBE_STEP_TITLES.get(step, step)
        if code != 200:
            _fail(f"{title} (step={step!r}): HTTP status={code} body={body!r}")
            continue
        if body is None:
            _fail(f"{title}: response body is None")
            continue
        if not isinstance(body, dict):
            _fail(f"{title}: body is not a JSON object ({type(body).__name__})")
            continue

        if step == "model_endpoint":
            data = body.get("data")
            if not isinstance(data, list) or not data:
                _fail(f"{title}: missing or empty models list")
                continue
            first = data[0]
            if not isinstance(first, dict) or not str(
                first.get("id") or first.get("model") or ""
            ).strip():
                _fail(f"{title}: no model id in models response")
            continue

        if not str(body.get("model") or "").strip():
            _fail(f"{title}: response has no model field")
            continue

        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            _fail(f"{title}: missing or empty choices")
            continue
        ch0 = choices[0]
        if not isinstance(ch0, dict):
            _fail(f"{title}: choices[0] is not an object")
            continue

        if step == "completion_endpoint":
            if not str(ch0.get("text") or "").strip():
                _fail(f"{title}: empty completion text")
            continue

        if step in ("chat_completion_endpoint", "structured_output_book"):
            msg = ch0.get("message")
            if not isinstance(msg, dict):
                _fail(f"{title}: choices[0].message is not an object")
                continue
            content = msg.get("content")
            if content is None or not str(content).strip():
                _fail(f"{title}: empty assistant content")
                continue
            if step == "structured_output_book":
                try:
                    obj = json.loads(str(content))
                except json.JSONDecodeError as e:
                    _fail(f"{title}: assistant content is not JSON ({e!r})")
                    continue
                if not isinstance(obj, dict):
                    _fail(f"{title}: parsed content is not a JSON object")
                    continue
                for key in ("title", "author", "year", "genre"):
                    if key not in obj:
                        _fail(f"{title}: JSON missing {key!r}")
                        break
                    v = obj[key]
                    if key == "year":
                        if isinstance(v, (int, float)):
                            continue
                        if not str(v).strip():
                            _fail(f"{title}: JSON year empty")
                            break
                    elif not str(v).strip():
                        _fail(f"{title}: JSON field {key!r} empty")
                        break
            continue

    if failures:
        return False, (
            f"OpenAI-compatible probe failed port={port!r}: " + " | ".join(failures)
        )
    return True, None


# lm-eval functions
def lm_eval_metric_value(text: str, task: str, metric: str) -> float | None:
    """Parse a metric value from lm-eval's ASCII results table."""
    m = re.search(
        rf"{re.escape(metric)}\s*\|\s*[^|\n]+\|\s*([0-9]+(?:\.[0-9]+)?)",
        text,
        flags=re.I,
    )
    return float(m.group(1)) if m else None
def lm_eval_openai_base_url(port: int, lm_eval_model: str) -> str:
    """Build the OpenAI-compatible base_url for lm-eval's local-* adapters."""
    path_suffix = (
        "/v1/chat/completions"
        if "chat" in lm_eval_model.lower()
        else "/v1/completions"
    )
    return f"http://0.0.0.0:{int(port)}{path_suffix}"
def build_lm_eval_model_args(
        *,
        model_id: str,
        base_url: str,
        num_concurrent: str,
        extra_model_args: str = "",
    ) -> str:
    """Comma-separated ``--model_args`` string for lm-eval."""
    model_args = (
        f"model={model_id},base_url={base_url},num_concurrent={num_concurrent},"
        f"tokenized_requests=False"
    )
    extra = str(extra_model_args or "").strip()
    if extra:
        model_args = f"{model_args},{extra}"
    return model_args
def build_lm_eval_command(
        *,
        lm_eval_model: str,
        model_id: str,
        base_url: str,
        tasks: str,
        num_fewshot: str,
        batch_size: str,
        num_concurrent: str,
        limit: str = "",
        extra_model_args: str = "",
        log_path: str,
        pip_install: bool = True,
    ) -> str:
    """
    Inner shell fragment to run lm-eval (no docker/ssh).
    Caller is responsible for any leading ``mkdir``, ``source ...``, etc.
    Output is tee'd to ``log_path``.
    """
    limit_s = str(limit or "").strip()
    limit_arg = f" --limit {limit_s}" if limit_s else ""
    model_args_q = shlex.quote(
        build_lm_eval_model_args(
            model_id=model_id,
            base_url=base_url,
            num_concurrent=num_concurrent,
            extra_model_args=extra_model_args,
        )
    )
    parts: list[str] = []
    if pip_install:
        parts.append("pip install -q 'lm-eval[api]'")
    parts.append(
        "python3 -m lm_eval "
        f"--model {lm_eval_model} "
        f"--model_args {model_args_q} "
        f"--tasks {tasks} "
        f"--num_fewshot {num_fewshot} "
        f"--batch_size {batch_size}"
        f"{limit_arg} "
        f"2>&1 | tee {shlex.quote(log_path)}"
    )
    return " && ".join(parts)

def check_lm_eval_results(
        text: str,
        *,
        task_name: str,
        parse_metric: str,
        expected: float,
        metric_key: str,
        tasks: str = "",
        tolerance_frac: float = 0.05,
        log_path: str = "",
        label: str = "",
    ) -> tuple[bool, dict[str, Any] | None, str | None]:
    """
    Validate lm-eval stdout after exec.
    Returns ``(ok, summary, err)`` where ``summary`` is::
        {"task": ..., "metric_key": ..., "actual": ..., "expected": ...}
  on success.
    """
    display = label or task_name
    log_hint = f" (see {log_path})" if log_path else ""
    if not text or not str(text).strip():
        return False, None, f"lm-eval {display} produced no output"
    if re.search(r"Traceback \(most recent call last\)", text):
        return (
            False,
            None,
            f"lm-eval {display} failed: Python traceback in output",
        )
    if not re.search(re.escape(task_name), text, re.I):
        return (
            False,
            None,
            f"lm-eval {display} output missing {task_name!r} results{log_hint}",
        )
    actual = lm_eval_metric_value(text, task_name, parse_metric)
    if actual is None:
        return (
            False,
            None,
            f"could not parse lm-eval table score for {task_name} / {parse_metric}",
        )
    expected_f = float(expected)
    if abs(actual - expected_f) > tolerance_frac * abs(expected_f):
        short_metric = parse_metric
        if "flexible" in metric_key.lower():
            short_metric = "flexible-extract"
        return (
            False,
            None,
            (
                f"{task_name} {short_metric} {actual:.4f} not within "
                f"{tolerance_frac * 100:.0f}% of expected {expected_f:.4f}"
            ),
        )
    summary = {
        "task": str(tasks or task_name),
        "metric_key": metric_key,
        "actual": float(actual),
        "expected": expected_f,
    }
    return True, summary, None

def run_lm_eval_openai_benchmark(
        i_dict: Mapping[str, Any],
        *,
        port: int,
        model_id: str,
        task_name: str,
        default_tasks: str,
        default_metric: str,
        default_metric_key: str,
        log_dir: str,
        log_basename: str,
        default_num_concurrent: str = "4",
        host: str = "0.0.0.0",
    ) -> tuple[str, dict[str, Any]]:
    """
    Resolve config + build the inner lm-eval shell command.
    Returns ``(inner_cmd, scoring_config)``. The caller runs ``inner_cmd``
    (optionally wrapped in docker/ssh) and passes stdout to
    :func:`check_lm_eval_results` with ``scoring_config``.
    ``host`` is reserved for future use; base URL currently uses
    ``0.0.0.0:{port}`` to match existing SGLang bench containers.
    """
    del host  # unused for now; keeps signature stable for vLLM/orch callers
    lm_eval_model = str(i_dict.get("lm_eval_model", "local-completions"))
    base_url = lm_eval_openai_base_url(port, lm_eval_model)
    num_concurrent = str(i_dict.get("num_concurrent", default_num_concurrent))
    tasks = str(i_dict.get("tasks", default_tasks))
    num_fewshot = str(i_dict.get("num_fewshot", "5"))
    batch_size = str(i_dict.get("batch_size", "auto"))
    limit = str(i_dict.get("limit", "")).strip()
    extra_model_args = str(i_dict.get("extra_model_args", "")).strip()
    exec_timeout_sec = int(i_dict.get("exec_timeout_sec", 7200))
    tolerance_frac = float(i_dict.get("tolerance_frac", 0.05))
    log_path = f"{log_dir.rstrip('/')}/benchmark_node/{log_basename}"
    expected_block = i_dict.get("expected_results") or {}
    if not isinstance(expected_block, Mapping):
        raise ValueError("expected_results must be a mapping")
    task_expected = expected_block.get(task_name) or {}
    if not isinstance(task_expected, Mapping):
        raise ValueError(f"expected_results[{task_name!r}] must be a mapping")
    if default_metric_key not in task_expected:
        raise KeyError(
            f"expected_results[{task_name!r}][{default_metric_key!r}] missing"
        )
    expected = float(task_expected[default_metric_key])
    inner_cmd = build_lm_eval_command(
        lm_eval_model=lm_eval_model,
        model_id=model_id,
        base_url=base_url,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        num_concurrent=num_concurrent,
        limit=limit,
        extra_model_args=extra_model_args,
        log_path=log_path,
        pip_install=bool(i_dict.get("pip_install", True)),
    )
    scoring_config: dict[str, Any] = {
        "task_name": task_name,
        "parse_metric": default_metric,
        "metric_key": default_metric_key,
        "expected": expected,
        "tasks": tasks,
        "tolerance_frac": tolerance_frac,
        "log_path": log_path,
        "exec_timeout_sec": exec_timeout_sec,
        "label": str(i_dict.get("label", task_name)),
        "lm_eval_model": lm_eval_model,
        "base_url": base_url,
    }
    return inner_cmd, scoring_config