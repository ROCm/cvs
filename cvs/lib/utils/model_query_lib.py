'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
'''

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import Any, Mapping, Optional

JsonDict = dict[str, Any]

# Keys passed from scoring_config into LmEvalBenchmark.check_results().
LM_EVAL_CHECK_RESULT_KEYS = (
    "task_name",
    "parse_metric",
    "expected",
    "metric_key",
    "tasks",
    "tolerance_frac",
    "log_path",
    "label",
)


class OpenAIProbe:
    """OpenAI-compatible HTTP smoke tests (stdlib probe script + validation)."""

    TIMEOUT_S = 120.0
    CHAT_MAX_TOKENS = 8
    COMPLETION_MAX_TOKENS = 50
    STRUCTURED_BOOK_MAX_TOKENS = 256

    CHAT_USER = "Reply with exactly one word: OK."
    COMPLETION_PROMPT = "The capital of France is"
    STRUCTURED_BOOK_SYSTEM = (
        "Respond with a single JSON object only. "
        "No markdown or text outside the JSON."
    )
    STRUCTURED_BOOK_USER = (
        "Return one book as a JSON object with keys: "
        "title (string), author (string), year (integer), genre (string)."
    )

    STEP_TITLES: dict[str, str] = {
        "model_endpoint": "Model endpoint — GET /v1/models",
        "chat_completion_endpoint": "Chat completion endpoint — POST /v1/chat/completions",
        "completion_endpoint": "Completion endpoint — POST /v1/completions",
        "structured_output_book": (
            "Structured output (book) — POST /v1/chat/completions "
            "(response_format: json_object)"
        ),
    }

    _FAILURE_MARKER = "OpenAI-compatible probe failed port="

    @classmethod
    def resolve_local_model_identity(cls, model_dir: str) -> str:
        """Config grep highlights + first 20 README lines for a local model dir."""
        root = Path(model_dir)
        parts: list[str] = [f"path={root}"]
        key_re = re.compile(
            r"model_type|quantization|quant|dtype|torch_dtype|architectures", re.I
        )

        cfg_path = root / "config.json"
        if cfg_path.is_file():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                hits = [
                    ln
                    for ln in json.dumps(cfg, indent=2).splitlines()
                    if key_re.search(ln)
                ]
                parts.append("config.json:\n" + ("\n".join(hits) if hits else "(no matches)"))
            except (OSError, json.JSONDecodeError) as e:
                parts.append(f"config.json: error ({e})")

        readme_path = root / "README.md"
        if readme_path.is_file():
            try:
                head = readme_path.read_text(encoding="utf-8").splitlines()[:20]
                parts.append("README.md (first 20 lines):\n" + "\n".join(head))
            except OSError as e:
                parts.append(f"README.md: error ({e})")

        return "\n\n".join(parts)

    @classmethod
    def probe_script(
        cls,
        port: int,
        model: str,
        *,
        timeout_s: float = TIMEOUT_S,
        chat_max_tokens: int = CHAT_MAX_TOKENS,
        completion_max_tokens: int = COMPLETION_MAX_TOKENS,
        structured_book_max_tokens: int = STRUCTURED_BOOK_MAX_TOKENS,
    ) -> str:
        """Stdlib-only Python source for docker exec / orch.exec."""
        chat_messages = [{"role": "user", "content": cls.CHAT_USER}]
        book_messages = [
            {"role": "system", "content": cls.STRUCTURED_BOOK_SYSTEM},
            {"role": "user", "content": cls.STRUCTURED_BOOK_USER},
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
                f'    "prompt": {json.dumps(cls.COMPLETION_PROMPT)},',
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

    @classmethod
    def log_results(cls, results: dict[str, tuple[int, Any]], logger: Any) -> None:
        """One headed block per endpoint for logs / HTML capture."""
        for step, (code, body) in results.items():
            title = cls.STEP_TITLES.get(step, step)
            logger.info("#================ %s ================#", title)
            logger.info("HTTP status: %s", code)
            if isinstance(body, (dict, list)):
                try:
                    logger.info("Body:\n%s", json.dumps(body, indent=2, default=str))
                except (TypeError, ValueError):
                    logger.info("Body: %s", body)
            else:
                logger.info("Body: %s", body)

    @classmethod
    def check_results(
        cls,
        results: dict[str, tuple[int, Any]],
        *,
        port: Any,
        logger: Optional[Any] = None,
    ) -> tuple[bool, Optional[str]]:
        """
        Each step: HTTP 200, JSON object body, and minimal useful content.
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
            title = cls.STEP_TITLES.get(step, step)
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
            return False, f"{cls._FAILURE_MARKER}{port!r}: " + " | ".join(failures)
        return True, None

    @classmethod
    def summarize_results(
        cls,
        results: dict[str, tuple[int, Any]],
        ok: bool,
        err: Optional[str],
    ) -> list[str]:
        """Human-readable per-step pass/fail lines for logs and inf_res_dict."""
        failure_parts: list[str] = []
        if not ok and err and err.startswith(cls._FAILURE_MARKER):
            rest = err[len(cls._FAILURE_MARKER) :]
            colon_idx = rest.find(": ")
            if colon_idx != -1:
                failure_parts = [
                    p.strip() for p in rest[colon_idx + 2 :].split("|")
                ]

        summary: list[str] = []
        for step, (status, _content) in results.items():
            title = cls.STEP_TITLES.get(step, step)
            if ok:
                outcome = "Pass" if status == 200 else "Fail"
            elif status != 200:
                outcome = "Fail"
            elif any(
                p.startswith(title) or p.startswith(f"{title} (step=")
                for p in failure_parts
            ):
                outcome = "Fail"
            else:
                outcome = "Pass"
            summary.append(f"{title} -> {outcome} ({status})")
        return summary


class LmEvalBenchmark:
    """lm-eval helpers for OpenAI-compatible local-* adapters."""

    DEFAULT_NUM_CONCURRENT = "4"
    DEFAULT_NUM_FEWSHOT = "5"
    DEFAULT_BATCH_SIZE = "auto"
    DEFAULT_LM_EVAL_MODEL = "local-completions"
    DEFAULT_EXEC_TIMEOUT_SEC = 7200
    DEFAULT_TOLERANCE_FRAC = 0.05

    @staticmethod
    def parse_metric_value(text: str, task: str, metric: str) -> float | None:
        """Parse a metric value from lm-eval's ASCII results table."""
        m = re.search(
            rf"{re.escape(metric)}\s*\|\s*[^|\n]+\|\s*([0-9]+(?:\.[0-9]+)?)",
            text,
            flags=re.I,
        )
        return float(m.group(1)) if m else None

    @staticmethod
    def openai_base_url(port: int, lm_eval_model: str) -> str:
        """Build base_url for lm-eval's local-completions / local-chat-completions."""
        path = (
            "/v1/chat/completions"
            if "chat" in lm_eval_model.lower()
            else "/v1/completions"
        )
        return f"http://0.0.0.0:{int(port)}{path}"

    @classmethod
    def build_model_args(
        cls,
        *,
        model_id: str,
        base_url: str,
        num_concurrent: str,
        extra_model_args: str = "",
    ) -> str:
        model_args = (
            f"model={model_id},base_url={base_url},num_concurrent={num_concurrent},"
            f"tokenized_requests=False"
        )
        extra = str(extra_model_args or "").strip()
        if extra:
            model_args = f"{model_args},{extra}"
        return model_args

    @classmethod
    def build_command(
        cls,
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
        Caller handles mkdir, source env, docker exec, etc.
        """
        limit_s = str(limit or "").strip()
        limit_arg = f" --limit {limit_s}" if limit_s else ""
        model_args_q = shlex.quote(
            cls.build_model_args(
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

    @classmethod
    def check_results(
        cls,
        text: str,
        *,
        task_name: str,
        parse_metric: str,
        expected: float,
        metric_key: str,
        tasks: str = "",
        tolerance_frac: float = DEFAULT_TOLERANCE_FRAC,
        log_path: str = "",
        label: str = "",
    ) -> tuple[bool, dict[str, Any] | None, str | None]:
        """
        Validate lm-eval stdout after exec.

        Returns ``(ok, summary, err)`` where summary is::

            {"task", "metric_key", "actual", "expected"}
        """
        display = label or task_name
        log_hint = f" (see {log_path})" if log_path else ""

        if not text or not str(text).strip():
            return False, None, f"lm-eval {display} produced no output"
        if re.search(r"Traceback \(most recent call last\)", text):
            return False, None, f"lm-eval {display} failed: Python traceback in output"
        if not re.search(re.escape(task_name), text, re.I):
            return (
                False,
                None,
                f"lm-eval {display} output missing {task_name!r} results{log_hint}",
            )

        actual = cls.parse_metric_value(text, task_name, parse_metric)
        if actual is None:
            return (
                False,
                None,
                f"could not parse lm-eval table score for {task_name} / {parse_metric}",
            )

        expected_f = float(expected)
        if abs(actual - expected_f) > tolerance_frac * abs(expected_f):
            short_metric = (
                "flexible-extract"
                if "flexible" in metric_key.lower()
                else parse_metric
            )
            err = (
                f"{task_name} {short_metric} {actual:.4f} not within "
                f"{tolerance_frac * 100:.0f}% of expected {expected_f:.4f}"
            )
            summary = {
                "task": str(tasks or task_name),
                "metric_key": metric_key,
                "actual": float(actual),
                "expected": expected_f,
                "passed": False,
            }
            return False, summary, err
        summary = {
            "task": str(tasks or task_name),
            "metric_key": metric_key,
            "actual": float(actual),
            "expected": expected_f,
            "passed": True,
        }
        return True, summary, None

    @classmethod
    def prepare(
        cls,
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
        default_num_concurrent: str = DEFAULT_NUM_CONCURRENT,
    ) -> tuple[str, dict[str, Any]]:
        """
        Resolve config and build the inner lm-eval shell command.

        Returns ``(inner_cmd, scoring_config)``.
        """
        lm_eval_model = str(i_dict.get("lm_eval_model", cls.DEFAULT_LM_EVAL_MODEL))
        base_url = cls.openai_base_url(port, lm_eval_model)
        num_concurrent = str(i_dict.get("num_concurrent", default_num_concurrent))
        tasks = str(i_dict.get("tasks", default_tasks))
        num_fewshot = str(i_dict.get("num_fewshot", cls.DEFAULT_NUM_FEWSHOT))
        batch_size = str(i_dict.get("batch_size", cls.DEFAULT_BATCH_SIZE))
        limit = str(i_dict.get("limit", "")).strip()
        extra_model_args = str(i_dict.get("extra_model_args", "")).strip()
        exec_timeout_sec = int(i_dict.get("exec_timeout_sec", cls.DEFAULT_EXEC_TIMEOUT_SEC))
        tolerance_frac = float(i_dict.get("tolerance_frac", cls.DEFAULT_TOLERANCE_FRAC))
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

        inner_cmd = cls.build_command(
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

    @classmethod
    def check_kwargs_from_scoring(cls, scoring: Mapping[str, Any]) -> dict[str, Any]:
        """Extract kwargs for check_results() from a scoring_config dict."""
        return {k: scoring[k] for k in LM_EVAL_CHECK_RESULT_KEYS}

    @classmethod
    def fallback_summary(
        cls,
        scoring: Mapping[str, Any],
        *,
        actual: float | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        return {
            "task": str(scoring.get("tasks") or scoring["task_name"]),
            "metric_key": scoring["metric_key"],
            "actual": actual,
            "expected": float(scoring["expected"]),
            "passed": False,
            "error": error,
        }