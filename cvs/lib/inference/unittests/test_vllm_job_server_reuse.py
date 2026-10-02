'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs.lib.inference.vllm_job.VllmJob server-command construction:
  - per-cell --max-model-len fallback and explicit override precedence
  - server_signature(), which gates cross-cell server reuse
  - _flatten_serve_args boolean handling and log-level pass-through
  - _check_early_failure tail emission and CLI parse error detection
  - RoleServer.serve_args log-level validator
  - probe_openai_endpoints(), the OpenAI-compatible HTTP smoke probe
'''

import json
import unittest
import unittest.mock as mock
from cvs.lib.inference.utils.vllm_config_loader import VariantConfig, serialize_cli_options
from cvs.lib.inference.vllm_job import VllmJob

_TP = 8
_PP = 2
_NNODES = 2


class FakeOrch:
    hosts = ["10.0.0.1", "10.0.0.2"]

    def __init__(self):
        self.head_cmds = []

    def exec(self, *a, **k):
        return {}

    def exec_on_head(self, cmd, *a, **k):
        self.head_cmds.append(cmd)
        return {}


class FakeOrchWithOutput:
    """Single-rank fake orch that returns controllable tail/grep output."""

    hosts = ["10.0.0.1"]

    def __init__(self, tail_output="", grep_exit=1):
        self.head_cmds = []
        self._tail_output = tail_output
        self._grep_exit = grep_exit  # 1 = no match (safe), 0 = match found

    def exec(self, cmd, hosts=None, detailed=False, print_console=True):
        if detailed:
            return {"10.0.0.1": {"exit_code": self._grep_exit, "stdout": ""}}
        return {"10.0.0.1": self._tail_output}

    def exec_on_head(self, cmd, *a, **k):
        self.head_cmds.append(cmd)
        return {}


class FakeOrchMultiHost:
    """Two-host fake orch that records which hosts each exec() call targeted."""

    hosts = ["10.0.0.1", "10.0.0.2"]

    def __init__(self):
        self.exec_calls = []  # list of (cmd, hosts) actually issued

    def exec(self, cmd, hosts=None, detailed=False, print_console=True):
        self.exec_calls.append((cmd, hosts))
        host = hosts[0]
        return {host: f"content for {host}"}

    def exec_on_head(self, cmd, *a, **k):
        return {}


def _make_job_for_check(tail_output="", grep_exit=1):
    """Construct a VllmJob suitable for testing _check_early_failure."""
    variant = _variant()
    variant.paths.log_dir = "/tmp/test_logs"
    orch = FakeOrchWithOutput(tail_output=tail_output, grep_exit=grep_exit)
    return VllmJob(
        orch=orch,
        variant=variant,
        hf_token="tok",
        isl="1024",
        osl="1024",
        concurrency="8",
        num_prompts="100",
    )


def _variant(serve_args=None, benchmark_params=None, sweep_options=None):
    options = {key.replace("-", "_"): value for key, value in (serve_args or {}).items()}
    distributed_executor_backend = options.pop("distributed_executor_backend", "mp")
    cell = f"ISL=1024,OSL=1024,TP={_TP},PP={_PP},CONC=16"
    return VariantConfig(
        enforce_thresholds=False,
        threshold_json="threshold.json",
        ib_netdev="enp159s0np0",
        paths={"shared_fs": "/logs", "models_dir": "/models", "log_dir": "/logs", "hf_token_file": "/logs/.hf"},
        container={"name": "test", "image": "test", "env": {}, "runtime": {"name": "docker", "args": {}}},
        server_params={
            "model": "/models/Kimi-K2.5-W4A8",
            "tensor_parallel_size": _TP,
            "pipeline_parallel_size": _PP,
            "port": 8000,
            "distributed_executor_backend": distributed_executor_backend,
            **options,
        },
        benchmark_params=benchmark_params or {},
        sweeps={cell: sweep_options or {}},
        runs=[cell],
    )


def _job(isl, osl, conc, serve_args=None, benchmark_params=None):
    return VllmJob(
        orch=FakeOrch(),
        variant=_variant(serve_args, benchmark_params),
        hf_token="tok",
        isl=isl,
        osl=osl,
        concurrency=conc,
        num_prompts="640",
    )


class TestMaxModelLenFallback(unittest.TestCase):
    def test_config_pin_wins_and_no_duplicate(self):
        argv = _job("1024", "1024", 16, serve_args={"max_model_len": "16384"})._server_argv(0)
        idxs = [i for i, a in enumerate(argv) if a == "--max-model-len"]
        self.assertEqual(len(idxs), 1, "config-pinned max-model-len must appear exactly once")
        self.assertEqual(argv[idxs[0] + 1], "16384", "config value must win")

    def test_explicit_null_suppresses_fallback_and_emits_no_flag(self):
        argv = _job("1024", "1024", 16, serve_args={"max_model_len": None})._server_argv(0)
        self.assertNotIn("--max-model-len", argv)

    def test_derives_expected_catalog_lengths_when_omitted(self):
        for isl, osl, expected in (
            ("1024", "1024", "2056"),
            ("1024", "8192", "9224"),
            ("8192", "1024", "9224"),
        ):
            with self.subTest(isl=isl, osl=osl):
                argv = _job(isl, osl, 16, serve_args={})._server_argv(0)
                idx = argv.index("--max-model-len")
                self.assertEqual(argv.count("--max-model-len"), 1)
                self.assertEqual(argv[idx + 1], expected)

    def test_uses_effective_per_cell_range_and_prefix_overrides(self):
        variant = _variant(
            benchmark_params={"random_range_ratio": 0.8, "random_prefix_len": 64},
            sweep_options={"random_range_ratio": 0.25, "random_prefix_len": 7},
        )
        (run,) = variant.resolved_runs()
        job = VllmJob(
            orch=FakeOrch(),
            variant=variant,
            hf_token="tok",
            isl=run.cell.isl,
            osl=run.cell.osl,
            concurrency=run.cell.concurrency,
            num_prompts=run.benchmark_params["num_prompts"],
            benchmark_params=run.benchmark_params,
        )
        argv = job._server_argv(0)
        idx = argv.index("--max-model-len")
        self.assertEqual(argv[idx + 1], "2575")


class TestEffectiveHosts(unittest.TestCase):
    def test_node_count_comes_from_orchestrator_hosts(self):
        job = _job("1024", "1024", 16)
        self.assertEqual(job.nnodes, "2")
        self.assertEqual(job.hosts, ("10.0.0.1", "10.0.0.2"))


class TestServerSignatureReuse(unittest.TestCase):
    def test_invariant_to_concurrency(self):
        # Pinned max-model-len: cells differing only in concurrency share a server.
        sa = {"max-model-len": "16384"}
        self.assertEqual(
            _job("1024", "1024", 4, sa).server_signature(),
            _job("1024", "1024", 64, sa).server_signature(),
        )

    def test_pinned_mml_shares_across_isl_osl(self):
        # With a fixed max-model-len, ISL/OSL never reach the server argv, so all
        # cells legitimately share one server (ISL/OSL are client-only knobs).
        sa = {"max-model-len": "16384"}
        self.assertEqual(
            _job("1024", "1024", 16, sa).server_signature(),
            _job("8192", "1024", 16, sa).server_signature(),
        )

    def test_different_derived_mml_restarts_across_cells(self):
        self.assertNotEqual(
            _job("1024", "1024", 16, serve_args={}).server_signature(),
            _job("1024", "8192", 16, serve_args={}).server_signature(),
        )

    def test_equal_derived_mml_reuses_across_cells(self):
        self.assertEqual(
            _job("1024", "8192", 16, serve_args={}).server_signature(),
            _job("8192", "1024", 16, serve_args={}).server_signature(),
        )

    def test_explicit_null_mml_reuses_across_isl_osl(self):
        serve_args = {"max_model_len": None}
        self.assertEqual(
            _job("1024", "1024", 16, serve_args).server_signature(),
            _job("8192", "1024", 16, serve_args).server_signature(),
        )

    def test_signature_strips_node_rank_and_is_hashable(self):
        job = _job("1024", "1024", 16, serve_args={"max-model-len": "16384"})
        self.assertIn("--node-rank", job._server_argv(0))
        sig = job.server_signature()
        self.assertNotIn("--node-rank", sig[0])
        # hashable + stable
        self.assertEqual(hash(sig), hash(job.server_signature()))


class TestRunClientEnsuresOutDir(unittest.TestCase):
    """The server-reuse path skips build_server_cmd (which creates the per-cell
    out_dir), so run_client must create its own out_dir or the client's
    client.log/results writes fail with 'No such file or directory'."""

    def test_run_client_mkdirs_out_dir(self):
        job = _job("1024", "1024", 8, serve_args={"max-model-len": "16384"})
        job.run_client()
        mkdir_cmds = [c for c in job.orch.head_cmds if "mkdir -p" in c and job.out_dir in c]
        self.assertTrue(
            mkdir_cmds,
            f"run_client must mkdir -p its out_dir ({job.out_dir}) so the reuse path "
            f"(which skips build_server_cmd) can still write client.log; head cmds: {job.orch.head_cmds}",
        )

    def test_run_client_uses_harness_owned_percentiles(self):
        job = _job("1024", "1024", 8)
        job.run_client()
        bench = next(command for command in job.orch.head_cmds if "vllm" in command and "bench" in command)
        self.assertIn("--percentile-metrics ttft,tpot,itl,e2el", bench)
        self.assertIn("--metric-percentiles 50,90,95,99", bench)


class TestRunClientTrustRemoteCode(unittest.TestCase):
    """Models with a custom tokenizer (e.g. Kimi-K2.6's auto_map) need the bench
    client to pass --trust-remote-code, mirroring the server's serve_args, or the
    client's tokenizer load raises ValueError before any request is sent."""

    def _bench_cmd(self, job):
        job.run_client()
        bench = [c for c in job.orch.head_cmds if "vllm" in c and "bench" in c]
        self.assertTrue(bench, f"no bench client command issued; head cmds: {job.orch.head_cmds}")
        return bench[-1]

    def test_trust_remote_code_passed_when_benchmark_enables_it(self):
        job = _job("1024", "1024", 8, benchmark_params={"trust_remote_code": True})
        self.assertIn("--trust-remote-code", self._bench_cmd(job))

    def test_trust_remote_code_absent_when_benchmark_omits_it(self):
        job = _job("1024", "1024", 8)
        self.assertNotIn("--trust-remote-code", self._bench_cmd(job))

    def test_resolved_cell_flags_override_benchmark_defaults(self):
        for base, override, present in ((False, True, True), (True, False, False)):
            with self.subTest(base=base, override=override):
                variant = _variant(
                    benchmark_params={"ignore_eos": base, "trust_remote_code": base},
                    sweep_options={"ignore_eos": override, "trust_remote_code": override},
                )
                (run,) = variant.resolved_runs()
                job = VllmJob(
                    orch=FakeOrch(),
                    variant=variant,
                    hf_token="tok",
                    isl=run.cell.isl,
                    osl=run.cell.osl,
                    concurrency=run.cell.concurrency,
                    num_prompts=run.benchmark_params["num_prompts"],
                    benchmark_params=run.benchmark_params,
                )
                command = self._bench_cmd(job)
                self.assertEqual("--ignore-eos" in command, present)
                self.assertEqual("--trust-remote-code" in command, present)


class TestSerializeServeArgs(unittest.TestCase):
    def test_false_value_is_rejected(self):
        with self.assertRaises(ValueError):
            serialize_cli_options({"enable_prefix_caching": False})

    def test_true_value_emits_flag_only(self):
        result = serialize_cli_options({"enforce_eager": True})
        self.assertEqual(result, ["--enforce-eager"])

    def test_log_level_passed_through(self):
        result = serialize_cli_options({"log_level": "debug"})
        self.assertEqual(result, ["--log-level", "debug"])


class TestCheckEarlyFailureEmitTail(unittest.TestCase):
    def test_emit_tail_true_logs_content(self):
        job = _make_job_for_check(tail_output="INFO engine loading\nINFO weights done")
        with mock.patch("cvs.lib.inference.vllm_job.log") as mock_log:
            job._check_early_failure(emit_tail=True)
        logged_lines = [call.args[3] for call in mock_log.info.call_args_list if len(call.args) >= 4]
        self.assertIn("INFO engine loading", logged_lines)
        self.assertIn("INFO weights done", logged_lines)

    def test_raises_on_cli_parse_error(self):
        job = _make_job_for_check(tail_output="vllm: error: unrecognized arguments: False")
        with self.assertRaises(RuntimeError):
            job._check_early_failure()


class TestDumpServerLog(unittest.TestCase):
    def test_logs_full_content_per_rank(self):
        job = _make_job_for_check(tail_output="line one\nline two")
        with mock.patch("cvs.lib.inference.vllm_job.log") as mock_log:
            job.dump_server_log()
        logged_lines = [call.args[3] for call in mock_log.info.call_args_list if len(call.args) >= 4]
        self.assertIn("line one", logged_lines)
        self.assertIn("line two", logged_lines)

    def test_mp_multinode_dumps_every_rank(self):
        """mp backend: every rank runs its own vllm serve, so every rank is dumped."""
        orch = FakeOrchMultiHost()
        job = VllmJob(
            orch=orch,
            variant=_variant({"distributed-executor-backend": "mp"}),
            hf_token="tok",
            isl="1024",
            osl="1024",
            concurrency=8,
            num_prompts="640",
        )
        with mock.patch("cvs.lib.inference.vllm_job.log") as mock_log:
            job.dump_server_log()
        ranks_dumped = {call.args[2] for call in mock_log.info.call_args_list if len(call.args) >= 4}
        self.assertEqual(ranks_dumped, {0, 1})
        self.assertEqual(len(orch.exec_calls), 2, "one cat per rank")

    def test_ray_multinode_skips_worker_ranks(self):
        """Ray multinode: only rank 0 runs vllm serve, so only rank 0 is dumped."""
        orch = FakeOrchMultiHost()
        job = VllmJob(
            orch=orch,
            variant=_variant({"distributed-executor-backend": "ray"}),
            hf_token="tok",
            isl="1024",
            osl="1024",
            concurrency=8,
            num_prompts="640",
        )
        with mock.patch("cvs.lib.inference.vllm_job.log") as mock_log:
            job.dump_server_log()
        ranks_dumped = {call.args[2] for call in mock_log.info.call_args_list if len(call.args) >= 4}
        self.assertEqual(ranks_dumped, {0}, "worker rank 1 has no server log under ray and must be skipped")
        self.assertEqual(len(orch.exec_calls), 1, "only rank 0's cat should be issued")


class FakeOrchWithHeadOutput:
    """Single-rank fake orch whose exec_on_head returns a controllable string,
    mirroring what `orch.exec_on_head` would ship back from the container."""

    hosts = ["10.0.0.1"]

    def __init__(self, head_output=""):
        self.head_cmds = []
        self._head_output = head_output

    def exec(self, *a, **k):
        return {}

    def exec_on_head(self, cmd, *a, **k):
        self.head_cmds.append(cmd)
        return {"10.0.0.1": self._head_output}


class TestProbeOpenAIEndpoints(unittest.TestCase):
    """Unit tests for VllmJob.probe_openai_endpoints. No hardware: FakeOrchWithHeadOutput
    returns a canned base64-decoded-script's stdout line (the JSON dict the
    stdlib probe script prints), mirroring what `orch.exec_on_head` would ship back
    from the container."""

    _GOOD_BODY = {
        "model": "amd/Llama-3.1-70B-Instruct-FP8-KV",
        "choices": [{"message": {"content": "OK"}, "text": "Paris"}],
    }
    _BOOK_CONTENT = json.dumps({"title": "T", "author": "A", "year": 2000, "genre": "G"})

    def _raw(self, results):
        return json.dumps(results)

    def _all_pass_results(self):
        return {
            "model_endpoint": [200, {"data": [{"id": "amd/Llama-3.1-70B-Instruct-FP8-KV"}]}],
            "chat_completion_endpoint": [200, {**self._GOOD_BODY, "choices": [{"message": {"content": "OK"}}]}],
            "completion_endpoint": [200, {**self._GOOD_BODY, "choices": [{"text": "Paris"}]}],
            "structured_output_book": [
                200,
                {**self._GOOD_BODY, "choices": [{"message": {"content": self._BOOK_CONTENT}}]},
            ],
        }

    def test_issues_single_head_exec_with_port_and_model(self):
        orch = FakeOrchWithHeadOutput(head_output=self._raw(self._all_pass_results()))
        job = _job("1024", "1024", 1, serve_args={"max-model-len": "16384"})
        job.orch = orch
        job.probe_openai_endpoints()
        self.assertEqual(len(orch.head_cmds), 1)
        cmd = orch.head_cmds[0]
        self.assertIn("base64 -d", cmd)
        self.assertIn("python3", cmd)

    def test_all_pass_returns_summary_lines(self):
        orch = FakeOrchWithHeadOutput(head_output=self._raw(self._all_pass_results()))
        job = _job("1024", "1024", 1, serve_args={"max-model-len": "16384"})
        job.orch = orch
        summary = job.probe_openai_endpoints()
        self.assertEqual(len(summary), 4)
        for line in summary:
            self.assertIn("-> Pass (200)", line)

    def test_http_failure_raises(self):
        results = self._all_pass_results()
        results["model_endpoint"] = [500, {"error": "boom"}]
        orch = FakeOrchWithHeadOutput(head_output=self._raw(results))
        job = _job("1024", "1024", 1, serve_args={"max-model-len": "16384"})
        job.orch = orch
        with self.assertRaises(RuntimeError):
            job.probe_openai_endpoints()

    def test_empty_content_raises(self):
        results = self._all_pass_results()
        results["chat_completion_endpoint"][1]["choices"] = [{"message": {"content": ""}}]
        orch = FakeOrchWithHeadOutput(head_output=self._raw(results))
        job = _job("1024", "1024", 1, serve_args={"max-model-len": "16384"})
        job.orch = orch
        with self.assertRaises(RuntimeError):
            job.probe_openai_endpoints()

    def test_no_output_raises(self):
        orch = FakeOrchWithHeadOutput(head_output="")
        job = _job("1024", "1024", 1, serve_args={"max-model-len": "16384"})
        job.orch = orch
        with self.assertRaises(RuntimeError):
            job.probe_openai_endpoints()

    def test_unparseable_output_raises(self):
        orch = FakeOrchWithHeadOutput(head_output="not json {{{")
        job = _job("1024", "1024", 1, serve_args={"max-model-len": "16384"})
        job.orch = orch
        with self.assertRaises(RuntimeError):
            job.probe_openai_endpoints()

    def test_bad_shape_raises(self):
        orch = FakeOrchWithHeadOutput(head_output=json.dumps({"model_endpoint": "not-a-pair"}))
        job = _job("1024", "1024", 1, serve_args={"max-model-len": "16384"})
        job.orch = orch
        with self.assertRaises(RuntimeError):
            job.probe_openai_endpoints()


if __name__ == "__main__":
    unittest.main()
