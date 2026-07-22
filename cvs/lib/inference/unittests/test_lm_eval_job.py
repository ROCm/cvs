'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs.lib.inference.utils.lm_eval_job (build_lm_eval_cmd, run_accuracy_tasks).

build_lm_eval_cmd is a PURE function (no I/O); run_accuracy_tasks is
orchestration-dependent and is tested via a FakeOrch test double, following the
FakeOrch pattern established in
cvs/lib/inference/unittests/test_vllm_job_server_reuse.py.
'''

import copy
import json
import unittest

from cvs.lib.inference.utils.accuracy_config import AccuracyTask
from cvs.lib.inference.utils.lm_eval_job import (
    LM_EVAL_INSTALL_CHECK_CMD,
    LmEvalCtx,
    build_lm_eval_cmd,
    run_accuracy_tasks,
)


def _task(**overrides):
    defaults = dict(id="mmlu", task="mmlu")
    defaults.update(overrides)
    return AccuracyTask(**defaults)


def _ctx(**overrides):
    defaults = dict(
        base_url="http://127.0.0.1:8000",
        model_id="meta-llama/Llama-3-8b",
        model_path="/data/models/Llama-3-8b",
        output_dir="/tmp/accuracy-out",
    )
    defaults.update(overrides)
    return LmEvalCtx(**defaults)


class TestBuildLmEvalCmd(unittest.TestCase):
    def test_install_guard_prefix_always_present(self):
        cmd = build_lm_eval_cmd(_task(), _ctx())
        self.assertTrue(cmd.startswith(LM_EVAL_INSTALL_CHECK_CMD + " && "))

    def test_default_single_task_shape(self):
        cmd = build_lm_eval_cmd(_task(), _ctx())
        self.assertIn("lm_eval", cmd)
        self.assertIn("--model local-completions", cmd)
        self.assertIn(
            "--model_args base_url=http://127.0.0.1:8000/v1/completions,"
            "model=meta-llama/Llama-3-8b,tokenizer=/data/models/Llama-3-8b,"
            "tokenizer_backend=huggingface,num_concurrent=8,max_retries=3",
            cmd,
        )
        self.assertIn("--tasks mmlu", cmd)
        self.assertIn("--num_fewshot 0", cmd)
        self.assertIn("--output_path /tmp/accuracy-out/mmlu", cmd)
        self.assertIn("--log_samples", cmd)

    def test_apply_chat_template_switches_model_flag(self):
        cmd = build_lm_eval_cmd(_task(apply_chat_template=True), _ctx())
        self.assertIn("--model local-chat-completions", cmd)
        self.assertNotIn("--model local-completions", cmd)
        # base_url / model_args shape is unchanged, only --model flag value differs.
        self.assertIn("base_url=http://127.0.0.1:8000/v1/completions", cmd)

    def test_num_fewshot_and_num_concurrent_reflected(self):
        cmd = build_lm_eval_cmd(_task(num_fewshot=5, num_concurrent=32), _ctx())
        self.assertIn("--num_fewshot 5", cmd)
        self.assertIn("num_concurrent=32", cmd)

    def test_metadata_json_encoded_when_present(self):
        cmd = build_lm_eval_cmd(_task(metadata={"seq_len": 4096}), _ctx())
        self.assertIn("--metadata", cmd)
        self.assertIn(json.dumps({"seq_len": 4096}), cmd)

    def test_metadata_omitted_when_empty(self):
        cmd = build_lm_eval_cmd(_task(metadata={}), _ctx())
        self.assertNotIn("--metadata", cmd)

    def test_include_path_included_when_nonempty(self):
        cmd = build_lm_eval_cmd(_task(include_path="/opt/custom_tasks"), _ctx())
        self.assertIn("--include_path /opt/custom_tasks", cmd)

    def test_include_path_omitted_when_empty_string(self):
        cmd = build_lm_eval_cmd(_task(include_path=""), _ctx())
        self.assertNotIn("--include_path", cmd)

    def test_gen_kwargs_comma_joined_and_insertion_order_preserved(self):
        cmd = build_lm_eval_cmd(
            _task(gen_kwargs={"temperature": 0, "max_gen_toks": 128, "top_p": 0.9}),
            _ctx(),
        )
        self.assertIn("--gen_kwargs temperature=0,max_gen_toks=128,top_p=0.9", cmd)

    def test_gen_kwargs_omitted_when_empty(self):
        cmd = build_lm_eval_cmd(_task(gen_kwargs={}), _ctx())
        self.assertNotIn("--gen_kwargs", cmd)

    def test_shell_quoting_safety_for_special_characters(self):
        task = _task(
            id="weird id",
            task="weird task",
            include_path="/path with spaces/tasks",
            gen_kwargs={"stop": "a b\"c"},
        )
        ctx = _ctx(output_dir="/tmp/out dir")
        cmd = build_lm_eval_cmd(task, ctx)
        # Command must be shell-parseable without raising, and round-trip the
        # exact values through shlex (proves quoting, not just substring presence).
        import shlex as _shlex

        parts = _shlex.split(cmd)
        self.assertIn("weird task", parts)
        self.assertIn("/tmp/out dir/weird id", parts)
        self.assertIn("/path with spaces/tasks", parts)
        self.assertIn('stop=a b"c', parts)

    def test_does_not_mutate_task_or_ctx(self):
        task = _task(metadata={"a": 1}, gen_kwargs={"b": 2})
        ctx = _ctx()
        task_before = copy.deepcopy(task)
        ctx_before = copy.deepcopy(ctx)
        build_lm_eval_cmd(task, ctx)
        self.assertEqual(task, task_before)
        self.assertEqual(ctx, ctx_before)

    def test_returns_single_string_not_list(self):
        cmd = build_lm_eval_cmd(_task(), _ctx())
        self.assertIsInstance(cmd, str)


class FakeOrch:
    """Head-only orch test double: records commands, returns queued responses."""

    def __init__(self, responses=None):
        self.head_cmds = []
        self._responses = list(responses or [])

    def exec_on_head(self, cmd, *a, **k):
        self.head_cmds.append(cmd)
        if self._responses:
            return {"10.0.0.1": self._responses.pop(0)}
        return {"10.0.0.1": ""}


class TestRunAccuracyTasks(unittest.TestCase):
    def _run_kwargs(self, orch, tasks):
        return dict(
            orch=orch,
            tasks=tasks,
            base_url="http://127.0.0.1:8000",
            model_id="meta-llama/Llama-3-8b",
            model_path="/data/models/Llama-3-8b",
            output_dir="/tmp/accuracy-out",
        )

    def test_single_task_success_id_keyed_and_projected(self):
        payload = {"results": {"mmlu": {"acc,none": 0.5, "alias": "mmlu"}}}
        orch = FakeOrch(
            responses=[
                "",  # lm_eval run output
                "/tmp/accuracy-out/mmlu/model/results_2025.json",  # find
                json.dumps(payload),  # cat
            ]
        )
        out = run_accuracy_tasks(**self._run_kwargs(orch, [_task()]))
        self.assertEqual(out, {"mmlu": {"mmlu.acc__none": 0.5}})

    def test_multiple_tasks_each_contribute_own_dict(self):
        orch = FakeOrch(
            responses=[
                "",
                "/out/mmlu/model/results.json",
                json.dumps({"results": {"mmlu": {"acc,none": 0.5}}}),
                "",
                "/out/gsm8k/model/results.json",
                json.dumps({"results": {"gsm8k": {"acc,none": 0.7}}}),
            ]
        )
        tasks = [_task(id="mmlu", task="mmlu"), _task(id="gsm8k", task="gsm8k")]
        out = run_accuracy_tasks(**self._run_kwargs(orch, tasks))
        self.assertEqual(
            out,
            {"mmlu": {"mmlu.acc__none": 0.5}, "gsm8k": {"gsm8k.acc__none": 0.7}},
        )

    def test_missing_results_file_raises_runtime_error(self):
        orch = FakeOrch(responses=["", ""])  # run output, then empty find output
        with self.assertRaises(RuntimeError):
            run_accuracy_tasks(**self._run_kwargs(orch, [_task()]))

    def test_exec_on_head_called_head_only_not_broadcast(self):
        payload = {"results": {"mmlu": {"acc,none": 0.5}}}
        orch = FakeOrch(
            responses=["", "/out/mmlu/model/results.json", json.dumps(payload)]
        )
        run_accuracy_tasks(**self._run_kwargs(orch, [_task()]))
        # exactly 3 exec_on_head calls for a single task: run, find, cat.
        self.assertEqual(len(orch.head_cmds), 3)
        self.assertFalse(hasattr(orch, "exec"))

    def test_install_guard_present_in_executed_command(self):
        payload = {"results": {"mmlu": {"acc,none": 0.5}}}
        orch = FakeOrch(
            responses=["", "/out/mmlu/model/results.json", json.dumps(payload)]
        )
        run_accuracy_tasks(**self._run_kwargs(orch, [_task()]))
        self.assertIn(LM_EVAL_INSTALL_CHECK_CMD, orch.head_cmds[0])


if __name__ == "__main__":
    unittest.main()
