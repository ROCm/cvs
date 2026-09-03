'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

lm-eval-harness command construction and execution against an already-live
inference server (see cvs/lib/inference/utils/AGENTS.md for the broader
accuracy-evaluation design). Routes through `orch.exec_on_head` rather than a
raw `docker exec`, matching current suite conventions (mirrors
`VllmJob.run_client`'s head-only execution rationale).
'''

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from typing import Any, Dict, List

from cvs.lib.inference.utils.accuracy_config import AccuracyTask
from cvs.lib.inference.utils.lm_eval_parsing import project

LM_EVAL_VERSION = "0.4.12"
LM_EVAL_INSTALL_CHECK_CMD = (
    "python -c \"import importlib.metadata as m; import lm_eval, math_verify; "
    f"assert m.version('lm-eval') == '{LM_EVAL_VERSION}'\" 2>/dev/null || "
    f"pip install -q 'lm-eval[api,math]=={LM_EVAL_VERSION}'"
)


@dataclass
class LmEvalCtx:
    base_url: str
    model_id: str
    model_path: str
    output_dir: str


def normalize_client_base_url(base_url: str) -> str:
    """Map bind-all addresses to localhost for outbound HTTP clients."""
    return base_url.replace("0.0.0.0", "127.0.0.1")


def _split_model_args(value: str) -> Dict[str, str]:
    """Parse comma-separated model args without splitting quoted JSON values."""
    if not value:
        return {}

    fragments = []
    start = 0
    depth = 0
    quote = None
    escaped = False
    for index, char in enumerate(value):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in "'\"":
            quote = char
        elif char in "[{(":
            depth += 1
        elif char in "]})":
            depth -= 1
            if depth < 0:
                raise ValueError(f"invalid extra_model_args: {value!r}")
        elif char == "," and depth == 0:
            fragments.append(value[start:index])
            start = index + 1
    if quote or depth:
        raise ValueError(f"invalid extra_model_args: {value!r}")
    fragments.append(value[start:])

    parsed = {}
    for fragment in fragments:
        key, separator, argument = fragment.strip().partition("=")
        if not separator or not key:
            raise ValueError(f"extra_model_args entries must be key=value, got: {fragment!r}")
        if key in parsed:
            raise ValueError(f"duplicate extra_model_args key: {key!r}")
        parsed[key] = argument
    return parsed


def _optional_arg(args: List[str], name: str, value) -> None:
    if value is not None:
        args += [name, str(value)]


def build_lm_eval_cmd(task: AccuracyTask, ctx: LmEvalCtx) -> str:
    model_flag = task.resolved_lm_eval_model
    endpoint_path = "/v1/chat/completions" if model_flag == "local-chat-completions" else "/v1/completions"
    model_args = {
        "base_url": f"{ctx.base_url}{endpoint_path}",
        "model": ctx.model_id,
        "tokenizer": ctx.model_path,
        "tokenizer_backend": "huggingface",
        "num_concurrent": str(task.num_concurrent),
        "max_retries": "3",
        "tokenized_requests": "False",
        "trust_remote_code": "True",
    }
    extras = _split_model_args(task.extra_model_args)
    collisions = sorted(key for key, value in extras.items() if key in model_args and model_args[key] != value)
    if collisions:
        raise ValueError(f"extra_model_args cannot override CVS-owned model args: {collisions}")
    model_args.update({key: value for key, value in extras.items() if key not in model_args})
    rendered_model_args = ",".join(f"{key}={value}" for key, value in model_args.items())

    args = [
        "lm_eval",
        "--model",
        model_flag,
        "--model_args",
        rendered_model_args,
        "--tasks",
        *task.task_names(),
        "--output_path",
        f"{ctx.output_dir}/{task.id}",
        "--log_samples",
    ]

    _optional_arg(args, "--num_fewshot", task.num_fewshot)
    _optional_arg(args, "--batch_size", task.batch_size)
    _optional_arg(args, "--max_batch_size", task.max_batch_size)
    _optional_arg(args, "--device", task.device)
    _optional_arg(args, "--limit", task.limit)
    _optional_arg(args, "--use_cache", task.use_cache)
    if task.samples is not None:
        _optional_arg(args, "--samples", json.dumps(task.samples) if isinstance(task.samples, dict) else task.samples)
    _optional_arg(args, "--cache_requests", task.cache_requests)

    if task.check_integrity:
        args.append("--check_integrity")
    _optional_arg(args, "--system_instruction", task.system_instruction)
    if task.apply_chat_template:
        args.append("--apply_chat_template")
        if isinstance(task.apply_chat_template, str):
            args.append(task.apply_chat_template)
    if task.fewshot_as_multiturn is not None:
        args.append("--fewshot_as_multiturn")
        if task.fewshot_as_multiturn is False:
            args.append("false")

    if task.metadata:
        args += ["--metadata", json.dumps(task.metadata)]

    if task.include_path:
        args += ["--include_path", task.include_path]

    if task.gen_kwargs:
        gen_kwargs = ",".join(f"{k}={v}" for k, v in task.gen_kwargs.items())
        args += ["--gen_kwargs", gen_kwargs]
    if task.predict_only:
        args.append("--predict_only")
    _optional_arg(args, "--seed", ",".join("None" if value is None else str(value) for value in task.seed))
    if task.trust_remote_code:
        args.append("--trust_remote_code")
    if task.confirm_run_unsafe_code:
        args.append("--confirm_run_unsafe_code")

    lm_eval_cmd = " ".join(shlex.quote(str(a)) for a in args)
    return f"source /tmp/server_env_script.sh && {LM_EVAL_INSTALL_CHECK_CMD} && {lm_eval_cmd}"


def run_accuracy_tasks(
    *,
    orch: Any,
    tasks: List[AccuracyTask],
    base_url: str,
    model_id: str,
    model_path: str,
    output_dir: str,
) -> Dict[str, Dict[str, float]]:
    ctx = LmEvalCtx(
        base_url=normalize_client_base_url(base_url),
        model_id=model_id,
        model_path=model_path,
        output_dir=output_dir,
    )
    results: Dict[str, Dict[str, float]] = {}

    for task in tasks:
        cmd = build_lm_eval_cmd(task, ctx)
        out = orch.exec_on_head(cmd, timeout=task.exec_timeout_sec, detailed=True)
        try:
            (run_result,) = out.values()
        except ValueError as e:
            raise RuntimeError(
                f"lm_eval task {task.id!r}: expected exactly one exec_on_head result, got {len(out)}: {e}"
            ) from e
        run_output = (run_result or {}).get("output") or ""
        exit_code = (run_result or {}).get("exit_code", -1)
        if exit_code != 0:
            raise RuntimeError(
                f"lm_eval task {task.id!r} exited with code {exit_code} "
                f"-- treating as a run failure. Command output tail: {run_output[-2000:]!r}"
            )

        task_out_dir = f"{output_dir}/{task.id}"
        find_cmd = f"find {shlex.quote(task_out_dir)} -name 'results*.json' -printf '%T@ %p\\n' | sort -rn"
        find_out = orch.exec_on_head(find_cmd)
        try:
            (find_output,) = find_out.values()
        except ValueError as e:
            raise RuntimeError(
                f"lm_eval task {task.id!r}: expected exactly one exec_on_head result for find, got {len(find_out)}: {e}"
            ) from e
        lines = (find_output or "").strip().splitlines()
        result_path = lines[0].split(" ", 1)[1] if lines else ""

        if not result_path:
            raise RuntimeError(
                f"lm_eval task {task.id!r} produced no results*.json under {task_out_dir} "
                f"-- treating as a run failure (install or execution error). "
                f"Command output tail: {run_output[-2000:]!r}"
            )

        cat_out = orch.exec_on_head(f"cat {shlex.quote(result_path)}")
        try:
            (payload_text,) = cat_out.values()
        except ValueError as e:
            raise RuntimeError(
                f"lm_eval task {task.id!r}: expected exactly one exec_on_head result for cat, got {len(cat_out)}: {e}"
            ) from e
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"lm_eval task {task.id!r} produced unparseable results at {result_path}: {e}") from e
        results[task.id] = project(payload)

    return results
