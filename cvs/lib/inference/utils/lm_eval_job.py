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

LM_EVAL_INSTALL_CHECK_CMD = "pip list 2>/dev/null | grep -q '^lm[_-]eval ' || pip install -q 'lm-eval[api]>=0.4.4'"
PER_TASK_TIMEOUT_S = 4 * 60 * 60


@dataclass
class LmEvalCtx:
    base_url: str
    model_id: str
    model_path: str
    output_dir: str


def build_lm_eval_cmd(task: AccuracyTask, ctx: LmEvalCtx) -> str:
    model_flag = "local-chat-completions" if task.apply_chat_template else "local-completions"
    endpoint_path = "/v1/chat/completions" if task.apply_chat_template else "/v1/completions"

    model_args = ",".join(
        [
            f"base_url={ctx.base_url}{endpoint_path}",
            f"model={ctx.model_id}",
            f"tokenizer={ctx.model_path}",
            "tokenizer_backend=huggingface",
            f"num_concurrent={task.num_concurrent}",
            "max_retries=3",
        ]
    )

    args = [
        "lm_eval",
        "--model",
        model_flag,
        "--model_args",
        model_args,
        "--tasks",
        task.task,
        "--num_fewshot",
        str(task.num_fewshot),
        "--output_path",
        f"{ctx.output_dir}/{task.id}",
        "--log_samples",
    ]

    if task.apply_chat_template:
        args.append("--apply_chat_template")

    if task.metadata:
        args += ["--metadata", json.dumps(task.metadata)]

    if task.include_path:
        args += ["--include_path", task.include_path]

    if task.gen_kwargs:
        gen_kwargs = ",".join(f"{k}={v}" for k, v in task.gen_kwargs.items())
        args += ["--gen_kwargs", gen_kwargs]

    lm_eval_cmd = " ".join(shlex.quote(str(a)) for a in args)
    return f"{LM_EVAL_INSTALL_CHECK_CMD} && {lm_eval_cmd}"


def run_accuracy_tasks(
    *,
    orch: Any,
    tasks: List[AccuracyTask],
    base_url: str,
    model_id: str,
    model_path: str,
    output_dir: str,
) -> Dict[str, Dict[str, float]]:
    ctx = LmEvalCtx(base_url=base_url, model_id=model_id, model_path=model_path, output_dir=output_dir)
    results: Dict[str, Dict[str, float]] = {}

    for task in tasks:
        cmd = build_lm_eval_cmd(task, ctx)
        out = orch.exec_on_head(cmd, timeout=PER_TASK_TIMEOUT_S)
        try:
            (run_output,) = out.values()
        except ValueError as e:
            raise RuntimeError(
                f"lm_eval task {task.id!r}: expected exactly one exec_on_head result, got {len(out)}: {e}"
            ) from e
        run_output = run_output or ""

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
