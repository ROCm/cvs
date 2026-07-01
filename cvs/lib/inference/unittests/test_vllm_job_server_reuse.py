'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs.lib.inference.vllm_job.VllmJob server-command construction:
  - the duplicate --max-model-len fix (config-pin suppresses the derived value)
  - server_signature(), which gates cross-cell server reuse
  - _flatten_serve_args boolean handling and log-level pass-through
  - _check_early_failure tail emission and CLI parse error detection
  - RoleServer.serve_args log-level validator
'''

import unittest
import unittest.mock as mock
from types import SimpleNamespace

import pydantic

from cvs.lib.inference.utils.vllm_config_loader import RoleServer
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

    def exec(self, cmd, hosts=None, detailed=False):
        if detailed:
            return {"10.0.0.1": {"exit_code": self._grep_exit, "stdout": ""}}
        return {"10.0.0.1": self._tail_output}

    def exec_on_head(self, cmd, *a, **k):
        self.head_cmds.append(cmd)
        return {}


def _make_job_for_check(tail_output="", grep_exit=1):
    """Construct a VllmJob suitable for testing _check_early_failure."""
    variant = mock.MagicMock()
    variant.params.tensor_parallelism = "8"
    variant.params.pipeline_parallel_size = "1"
    variant.params.master_addr = "localhost"
    variant.params.master_port = "29501"
    variant.params.nnodes = "1"
    variant.params.port_no = "8000"
    variant.params.random_range_ratio = "0.0"
    variant.params.random_prefix_len = "0"
    variant.params.burstiness = "1.0"
    variant.params.seed = "0"
    variant.params.request_rate = "inf"
    variant.params.tokenizer_mode = "auto"
    variant.params.percentile_metrics = "ttft,tpot,itl,e2el"
    variant.params.metric_percentiles = "50,90,95,99"
    variant.params.base_url = "http://0.0.0.0"
    variant.params.dataset_name = "random"
    variant.params.backend = "vllm"
    variant.model.id = "/models/test-model"
    variant.paths.log_dir = "/tmp/test_logs"
    variant.paths.models_dir = "/tmp/models"
    variant.roles.server.serve_args = {}
    variant.roles.server.env = {}
    variant.roles.server.ib_netdev = None
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


def _variant(serve_args=None):
    params = SimpleNamespace(
        tensor_parallelism=str(_TP),
        pipeline_parallel_size=str(_PP),
        master_addr="10.0.0.1",
        master_port="29501",
        nnodes=str(_NNODES),
        port_no="8000",
        random_range_ratio="0.8",
        random_prefix_len="0",
        burstiness="1.0",
        seed="0",
        request_rate="inf",
        tokenizer_mode="auto",
        percentile_metrics="ttft,tpot,itl,e2el",
        metric_percentiles="50,90,95,99",
        base_url="http://0.0.0.0",
        dataset_name="random",
        backend="vllm",
    )
    return SimpleNamespace(
        params=params,
        model=SimpleNamespace(id="/models/Kimi-K2.5-W4A8"),
        paths=SimpleNamespace(log_dir="/logs", models_dir="/models"),
        roles=SimpleNamespace(
            server=SimpleNamespace(
                serve_args=dict(serve_args or {}),
                env={"VLLM_ROCM_USE_AITER": "1"},
                ib_netdev="enp159s0np0",
            )
        ),
    )


def _job(isl, osl, conc, serve_args=None):
    return VllmJob(
        orch=FakeOrch(),
        variant=_variant(serve_args),
        hf_token="tok",
        isl=isl,
        osl=osl,
        concurrency=conc,
        num_prompts="640",
    )


class TestMaxModelLenNoDuplicate(unittest.TestCase):
    def test_config_pin_wins_and_no_duplicate(self):
        argv = _job("1024", "1024", 16, serve_args={"max-model-len": "16384"})._server_argv(0)
        idxs = [i for i, a in enumerate(argv) if a == "--max-model-len"]
        self.assertEqual(len(idxs), 1, "config-pinned max-model-len must appear exactly once")
        self.assertEqual(argv[idxs[0] + 1], "16384", "config value must win")

    def test_derived_emitted_when_not_pinned(self):
        argv = _job("1024", "1024", 16, serve_args={})._server_argv(0)
        idxs = [i for i, a in enumerate(argv) if a == "--max-model-len"]
        self.assertEqual(len(idxs), 1, "derived max-model-len must still be emitted when unpinned")
        # 1024+1024 worst-case derived value, definitely not the 16384 config value
        self.assertNotEqual(argv[idxs[0] + 1], "16384")


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

    def test_derived_mml_distinguishes_osl(self):
        # Without a pin, max-model-len is derived per (isl+osl); different OSL must
        # change the signature so a real restart happens.
        self.assertNotEqual(
            _job("1024", "1024", 16, serve_args={}).server_signature(),
            _job("1024", "8192", 16, serve_args={}).server_signature(),
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


class TestRunClientTrustRemoteCode(unittest.TestCase):
    """Models with a custom tokenizer (e.g. Kimi-K2.6's auto_map) need the bench
    client to pass --trust-remote-code, mirroring the server's serve_args, or the
    client's tokenizer load raises ValueError before any request is sent."""

    def _bench_cmd(self, job):
        job.run_client()
        bench = [c for c in job.orch.head_cmds if "vllm" in c and "bench" in c]
        self.assertTrue(bench, f"no bench client command issued; head cmds: {job.orch.head_cmds}")
        return bench[-1]

    def test_trust_remote_code_passed_when_server_enables_it(self):
        job = _job("1024", "1024", 8, serve_args={"max-model-len": "16384", "trust-remote-code": True})
        self.assertIn("--trust-remote-code", self._bench_cmd(job))

    def test_trust_remote_code_absent_when_server_omits_it(self):
        job = _job("1024", "1024", 8, serve_args={"max-model-len": "16384"})
        self.assertNotIn("--trust-remote-code", self._bench_cmd(job))


class TestFlattenServeArgsFalse(unittest.TestCase):
    def test_false_value_omitted(self):
        result = VllmJob._flatten_serve_args({"enable-prefix-caching": False, "tensor-parallel-size": "8"})
        self.assertNotIn("--enable-prefix-caching", result)
        self.assertNotIn("False", result)
        self.assertEqual(result, ["--tensor-parallel-size", "8"])

    def test_true_value_emits_flag_only(self):
        result = VllmJob._flatten_serve_args({"enforce-eager": True})
        self.assertEqual(result, ["--enforce-eager"])

    def test_log_level_passed_through(self):
        result = VllmJob._flatten_serve_args({"log-level": "debug"})
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


class TestRoleServerLogLevelValidator(unittest.TestCase):
    def test_invalid_log_level_rejected(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            RoleServer(serve_args={"log-level": "verbose"})
        msg = str(ctx.exception)
        self.assertIn("log-level", msg)
        self.assertIn("verbose", msg)

    def test_valid_log_level_accepted(self):
        rs = RoleServer(serve_args={"log-level": "debug"})
        self.assertEqual(rs.serve_args["log-level"], "debug")


if __name__ == "__main__":
    unittest.main()
