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
    LM_EVAL_VERSION,
    LmEvalCtx,
    build_lm_eval_cmd,
    normalize_client_base_url,
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
    def test_normalize_client_base_url_rewrites_bind_all(self):
        self.assertEqual(normalize_client_base_url("http://0.0.0.0:8000"), "http://127.0.0.1:8000")
        self.assertEqual(normalize_client_base_url("http://127.0.0.1:8000"), "http://127.0.0.1:8000")

    def test_env_script_sourced_before_install_guard(self):
        # HF_HUB_CACHE/HF_TOKEN are written to /tmp/server_env_script.sh by
        # server setup (see vllm_job.build_server_cmd) and are NOT inherited
        # across separate exec_on_head invocations -- lm_eval must source
        # that script itself, matching VllmJob's client-launch convention.
        cmd = build_lm_eval_cmd(_task(), _ctx())
        self.assertTrue(cmd.startswith("source /tmp/server_env_script.sh && " + LM_EVAL_INSTALL_CHECK_CMD + " && "))

    def test_default_single_task_shape(self):
        cmd = build_lm_eval_cmd(_task(), _ctx())
        self.assertIn("lm_eval", cmd)
        self.assertIn("--model local-completions", cmd)
        self.assertIn(
            "--model_args base_url=http://127.0.0.1:8000/v1/completions,"
            "model=meta-llama/Llama-3-8b,tokenizer=/data/models/Llama-3-8b,"
            "tokenizer_backend=huggingface,num_concurrent=8,max_retries=3,"
            "tokenized_requests=False,trust_remote_code=True",
            cmd,
        )
        self.assertIn("--tasks mmlu", cmd)
        self.assertNotIn("--num_fewshot", cmd)
        self.assertIn("--batch_size 1", cmd)
        self.assertIn("--device cuda:0", cmd)
        self.assertIn("--seed 0,1234,1234,1234", cmd)
        self.assertIn("--output_path /tmp/accuracy-out/mmlu", cmd)
        self.assertIn("--log_samples", cmd)

    def test_trust_remote_code_always_present(self):
        # Models with custom tokenizer code (Qwen, ChatGLM, Phi, MPT, ...)
        # fail to load without this -- unconditional since it's a no-op for
        # models that don't need it.
        cmd = build_lm_eval_cmd(_task(), _ctx())
        self.assertIn("trust_remote_code=True", cmd)

    def test_tokenized_requests_disabled_for_string_prompts(self):
        # lm-eval defaults tokenized_requests=True, which POSTs token-id arrays
        # to /v1/completions; vLLM/ATOM expect a string prompt and return 422.
        cmd = build_lm_eval_cmd(_task(), _ctx())
        self.assertIn("tokenized_requests=False", cmd)

    def test_apply_chat_template_switches_model_flag(self):
        cmd = build_lm_eval_cmd(_task(apply_chat_template=True), _ctx())
        self.assertIn("--model local-chat-completions", cmd)
        self.assertNotIn("--model local-completions", cmd)
        # chat-template tasks must hit the chat endpoint, not /v1/completions.
        self.assertIn("base_url=http://127.0.0.1:8000/v1/chat/completions", cmd)
        self.assertNotIn("base_url=http://127.0.0.1:8000/v1/completions,", cmd)

    def test_apply_chat_template_passes_lm_eval_flag(self):
        # local-chat-completions requires lm-eval's own --apply_chat_template
        # flag to format prompts as messages=list[dict]; without it lm-eval
        # sends a plain string and the chat-completions client asserts.
        cmd = build_lm_eval_cmd(_task(apply_chat_template=True), _ctx())
        self.assertIn("--apply_chat_template", cmd)

    def test_default_task_uses_completions_endpoint(self):
        cmd = build_lm_eval_cmd(_task(apply_chat_template=False), _ctx())
        self.assertIn("base_url=http://127.0.0.1:8000/v1/completions", cmd)
        self.assertNotIn("--apply_chat_template", cmd)

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

    def test_complete_lm_eval_surface_is_forwarded(self):
        cmd = build_lm_eval_cmd(
            _task(
                tasks=["hellaswag", "gsm8k"],
                task="hellaswag,gsm8k",
                lm_eval_model="local-chat-completions",
                extra_model_args="retry=5",
                num_fewshot=5,
                batch_size="auto:4",
                max_batch_size=8,
                device="cpu",
                limit=100,
                samples={"gsm8k": [0]},
                use_cache="/tmp/cache.db",
                cache_requests="refresh",
                check_integrity=True,
                system_instruction="Be concise",
                apply_chat_template="chatml",
                fewshot_as_multiturn=False,
                include_path="/tmp/tasks",
                predict_only=True,
                seed=[1, None, 3, 4],
                trust_remote_code=True,
                confirm_run_unsafe_code=True,
                metadata={"source": "test"},
            ),
            _ctx(),
        )
        for part in (
            "--tasks hellaswag gsm8k",
            "--batch_size auto:4",
            "--max_batch_size 8",
            "--device cpu",
            "--limit 100.0",
            "--use_cache /tmp/cache.db",
            "--cache_requests refresh",
            "--check_integrity",
            "--system_instruction 'Be concise'",
            "--apply_chat_template chatml",
            "--fewshot_as_multiturn false",
            "--include_path /tmp/tasks",
            "--predict_only",
            "--seed 1,None,3,4",
            "--trust_remote_code",
            "--confirm_run_unsafe_code",
        ):
            self.assertIn(part, cmd)
        self.assertIn("retry=5", cmd)

    def test_rejects_extra_model_arg_collision(self):
        with self.assertRaisesRegex(ValueError, "CVS-owned"):
            build_lm_eval_cmd(_task(extra_model_args="model=other"), _ctx())

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


class TestInstallGuard(unittest.TestCase):
    '''The install guard must cover the `math` extra and detect it by capability.

    leaderboard_math_hard imports math_verify at task-build time; the `api`
    extra alone omits it, so the task dies with ModuleNotFoundError after the
    server is already up (observed on GLM-5.2, 2026-07-31).
    '''

    def test_installs_math_extra(self):
        self.assertIn("lm-eval[api,math]", LM_EVAL_INSTALL_CHECK_CMD)

    def test_pins_lm_eval_version(self):
        self.assertIn(f"=={LM_EVAL_VERSION}", LM_EVAL_INSTALL_CHECK_CMD)

    def test_guard_probes_math_verify_import_not_just_lm_eval_presence(self):
        # A `pip list | grep lm_eval` guard short-circuits on an image that
        # preinstalls bare lm-eval, silently skipping the math extra. Probing
        # the import instead fails closed.
        self.assertIn("import lm_eval, math_verify", LM_EVAL_INSTALL_CHECK_CMD)
        self.assertNotIn("pip list", LM_EVAL_INSTALL_CHECK_CMD)

    def test_install_still_runs_only_when_probe_fails(self):
        self.assertIn("||", LM_EVAL_INSTALL_CHECK_CMD)
        self.assertIn("pip install", LM_EVAL_INSTALL_CHECK_CMD)


class FakeOrch:
    """Head-only orch test double: records commands, returns queued responses.

    The first exec_on_head call (the lm_eval run itself) is made with
    detailed=True and expects a {'output': ..., 'exit_code': ...} response;
    responses for that call may be given as a bare string (wrapped here with
    exit_code=0) or as an explicit dict to simulate a non-zero exit.
    """

    def __init__(self, responses=None):
        self.head_cmds = []
        self.head_kwargs = []
        self._responses = list(responses or [])

    def exec_on_head(self, cmd, *a, **k):
        self.head_cmds.append(cmd)
        self.head_kwargs.append(k)
        if self._responses:
            response = self._responses.pop(0)
        else:
            response = "" if not k.get("detailed") else {"output": "", "exit_code": 0}
        if k.get("detailed") and not isinstance(response, dict):
            response = {"output": response, "exit_code": 0}
        return {"10.0.0.1": response}


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
                "1700000000.123456 /tmp/accuracy-out/mmlu/model/results_2025.json",  # find
                json.dumps(payload),  # cat
            ]
        )
        out = run_accuracy_tasks(**self._run_kwargs(orch, [_task()]))
        self.assertEqual(out, {"mmlu": {"mmlu.acc__none": 0.5}})

    def test_multiple_tasks_each_contribute_own_dict(self):
        orch = FakeOrch(
            responses=[
                "",
                "1700000000.0 /out/mmlu/model/results.json",
                json.dumps({"results": {"mmlu": {"acc,none": 0.5}}}),
                "",
                "1700000001.0 /out/gsm8k/model/results.json",
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
        orch = FakeOrch(responses=["", "1700000000.0 /out/mmlu/model/results.json", json.dumps(payload)])
        run_accuracy_tasks(**self._run_kwargs(orch, [_task()]))
        # exactly 3 exec_on_head calls for a single task: run, find, cat.
        self.assertEqual(len(orch.head_cmds), 3)
        self.assertFalse(hasattr(orch, "exec"))

    def test_install_guard_present_in_executed_command(self):
        payload = {"results": {"mmlu": {"acc,none": 0.5}}}
        orch = FakeOrch(responses=["", "1700000000.0 /out/mmlu/model/results.json", json.dumps(payload)])
        run_accuracy_tasks(**self._run_kwargs(orch, [_task()]))
        self.assertIn(LM_EVAL_INSTALL_CHECK_CMD, orch.head_cmds[0])

    def test_picks_newest_result_when_multiple_present(self):
        payload = {"results": {"mmlu": {"acc,none": 0.5}}}
        orch = FakeOrch(
            responses=[
                "",
                # `find ... -printf '%T@ %p\n' | sort -rn` sorts newest-first on
                # the real host -- FakeOrch can't execute the shell pipeline
                # itself, so this fixture simulates the already-sorted output a
                # real run would produce.
                "1700000999.0 /out/mmlu/model/results_new.json\n1700000000.0 /out/mmlu/model/results_old.json",
                json.dumps(payload),
            ]
        )
        run_accuracy_tasks(**self._run_kwargs(orch, [_task()]))
        # find step's own command text must actually sort by mtime descending --
        # FakeOrch ignores command text when producing its response, so this
        # assertion is the only thing that would catch a shell-logic regression
        # (e.g. `sort -n` instead of `sort -rn`); the fixture above only proves
        # the Python-side parsing of already-sorted output picks line 0.
        find_cmd = orch.head_cmds[1]
        self.assertIn("-printf", find_cmd)
        self.assertIn("sort -rn", find_cmd)
        # cat must be issued against the first (newest) line's path.
        self.assertIn("results_new.json", orch.head_cmds[2])
        self.assertNotIn("results_old.json", orch.head_cmds[2])

    def test_malformed_json_result_raises_runtime_error(self):
        orch = FakeOrch(responses=["", "1700000000.0 /out/mmlu/model/results.json", "{not valid json"])
        with self.assertRaises(RuntimeError):
            run_accuracy_tasks(**self._run_kwargs(orch, [_task()]))

    def test_none_run_output_does_not_crash_on_missing_result(self):
        orch = FakeOrch(responses=[None, ""])  # run output is None, find is empty
        with self.assertRaises(RuntimeError):
            run_accuracy_tasks(**self._run_kwargs(orch, [_task()]))

    def test_nonzero_exit_code_raises_before_checking_for_results(self):
        # A results*.json can exist on disk from a prior run even though this
        # invocation of lm_eval itself failed -- exit_code must be checked
        # before treating the run as successful, independent of file presence.
        orch = FakeOrch(responses=[{"output": "traceback...", "exit_code": 1}])
        with self.assertRaises(RuntimeError) as ctx:
            run_accuracy_tasks(**self._run_kwargs(orch, [_task()]))
        self.assertIn("exited with code 1", str(ctx.exception))
        # must fail fast: no find/cat calls issued after a nonzero exit.
        self.assertEqual(len(orch.head_cmds), 1)

    def test_run_invocation_requests_detailed_exit_code(self):
        payload = {"results": {"mmlu": {"acc,none": 0.5}}}
        orch = FakeOrch(responses=["", "1700000000.0 /out/mmlu/model/results.json", json.dumps(payload)])
        run_accuracy_tasks(**self._run_kwargs(orch, [_task()]))
        self.assertTrue(orch.head_kwargs[0].get("detailed"))

    def test_uses_task_specific_timeout(self):
        payload = {"results": {"mmlu": {"acc,none": 0.5}}}
        orch = FakeOrch(responses=["", "1700000000.0 /out/mmlu/model/results.json", json.dumps(payload)])
        run_accuracy_tasks(**self._run_kwargs(orch, [_task(exec_timeout_sec=7200)]))
        self.assertEqual(orch.head_kwargs[0]["timeout"], 7200)

    def test_zero_exit_code_with_dict_response_succeeds(self):
        payload = {"results": {"mmlu": {"acc,none": 0.5}}}
        orch = FakeOrch(
            responses=[
                {"output": "", "exit_code": 0},
                "1700000000.0 /out/mmlu/model/results.json",
                json.dumps(payload),
            ]
        )
        out = run_accuracy_tasks(**self._run_kwargs(orch, [_task()]))
        self.assertEqual(out, {"mmlu": {"mmlu.acc__none": 0.5}})

    def test_run_accuracy_tasks_rewrites_bind_all_base_url(self):
        payload = {"results": {"mmlu": {"acc,none": 0.5}}}
        orch = FakeOrch(responses=["", "1700000000.0 /out/mmlu/model/results.json", json.dumps(payload)])
        run_accuracy_tasks(
            orch=orch,
            tasks=[_task()],
            base_url="http://0.0.0.0:8000",
            model_id="meta-llama/Llama-3-8b",
            model_path="/data/models/Llama-3-8b",
            output_dir="/tmp/accuracy-out",
        )
        self.assertIn("base_url=http://127.0.0.1:8000/v1/completions", orch.head_cmds[0])


if __name__ == "__main__":
    unittest.main()
