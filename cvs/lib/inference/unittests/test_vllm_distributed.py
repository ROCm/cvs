'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for VllmDistributedJob -- the multinode vLLM job driven by a
ContainerOrchestrator. No hardware: a FakeOrch records the commands the job
issues and returns canned per-host output.

These tests are committed RED (greenfield): the implementation in
cvs/lib/inference/vllm_distributed.py is a placeholder and is filled in by a
separate agent who cannot edit this file. The contract under test comes from the
vllm_distributed spec, NOT from any implementation source.

Key distributed-vs-single contract (spec File 2):
  - server env script + out-dir mkdir broadcast to ALL nodes via orch.exec()
    (no hosts kwarg / hosts=None)
  - server launched per-host via orch.exec(..., hosts=[host]) with per-rank
    --node-rank, --master-addr, --pipeline-parallel-size, --distributed-
    executor-backend mp
  - client launched + polled + results-fetched via orch.exec_on_head()
    (single-entry {head_host: text} dict), never the broadcast orch.exec()
'''

import json
import re
import unittest
import unittest.mock
from pathlib import Path
from types import SimpleNamespace

from cvs.lib.inference.vllm_distributed import VllmDistributedJob

_HERE = Path(__file__).parent
_FIXTURES = _HERE / "fixtures"

# isl/tp must match the artifact fixture for the derived-math assertions to be
# meaningful (real artifact: isl=128, tp=8).
_ISL = 128
_TP = 8
_PP = 2
_NNODES = 2
_MASTER_ADDR = "node-head"  # symbolic, not real infra (anti-pattern: hardcoded IPs)
_MASTER_PORT = "29501"

# Symbolic host names -- NOT real infra IPs (Hardcoded-Infrastructure-IDs anti-pattern).
_HOSTS = ["node-a", "node-b"]
_HEAD = _HOSTS[0]


class FakeOrch:
    """Stand-in for ContainerOrchestrator.

    Records every exec / exec_on_head call (command + kwargs) so tests can assert
    on the broadcast-vs-head routing and per-host rank dispatch. `hosts` mirrors a
    real multinode orchestrator so enumerate(orch.hosts) yields (rank, host).
    """

    def __init__(self, hosts=None, exec_return=None, head_return=None):
        self.hosts = list(hosts) if hosts is not None else list(_HOSTS)
        self._exec_return = exec_return
        self._head_return = head_return
        self.exec_calls = []          # list of (cmd, kwargs)
        self.head_calls = []          # list of (cmd, kwargs)

    def exec(self, cmd, **kwargs):
        self.exec_calls.append((cmd, kwargs))
        ret = self._exec_return
        if callable(ret):
            ret = ret(cmd, kwargs)
        if ret is not None:
            return ret
        # Default: broadcast result keyed by the targeted hosts (or all hosts).
        targets = kwargs.get("hosts") or self.hosts
        return {h: "" for h in targets}

    def exec_on_head(self, cmd, **kwargs):
        self.head_calls.append((cmd, kwargs))
        ret = self._head_return
        if callable(ret):
            ret = ret(cmd, kwargs)
        if ret is not None:
            return ret
        return {_HEAD: ""}


def _fake_variant():
    """SimpleNamespace tree carrying exactly the attrs VllmDistributedJob reads.

    Mirrors vllm_single's _fake_variant plus the distributed params
    (pipeline_parallel_size, master_addr, master_port, nnodes).
    """
    params = SimpleNamespace(
        tensor_parallelism=str(_TP),
        pipeline_parallel_size=str(_PP),
        master_addr=_MASTER_ADDR,
        master_port=_MASTER_PORT,
        nnodes=str(_NNODES),
        port_no="8888",
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
        model=SimpleNamespace(id="amd/Llama-3.1-70B-Instruct-FP8-KV"),
        roles=SimpleNamespace(server=SimpleNamespace(serve_args={}, env={})),
        paths=SimpleNamespace(log_dir="/tmp/logs", models_dir="/tmp/models"),
    )


def _make_job(orch, goodput_slo=None, **kwargs):
    return VllmDistributedJob(
        orch=orch,
        variant=_fake_variant(),
        hf_token="tok",
        isl=_ISL,
        osl=2048,
        concurrency=256,
        num_prompts=12800,
        goodput_slo=goodput_slo,
        **kwargs,
    )


def _load_fixture(name):
    return (_FIXTURES / name).read_text()


def _argv_after(argv, flag):
    """Return the token following `flag` in argv, or None if flag absent/last."""
    if flag not in argv:
        return None
    i = argv.index(flag)
    return argv[i + 1] if i + 1 < len(argv) else None


# ---------------------------------------------------------------------------
# Pure / derived builders: _server_argv, _derive_max_model_len
# Classification: pure (output depends only on self fields set at construction).
# ---------------------------------------------------------------------------


class TestServerArgv(unittest.TestCase):
    """`_server_argv(rank)` builds the per-rank `vllm serve` arg list (spec File 2
    `_server_argv`). Asserted on the parsed argv list, not a rendered blob."""

    def setUp(self):
        self.job = _make_job(FakeOrch())

    def test_server_argv_rank0(self):
        argv = self.job._server_argv(0)
        # Distributed flags present with correct values for the head rank.
        self.assertEqual(_argv_after(argv, "--node-rank"), "0")
        self.assertEqual(_argv_after(argv, "--master-addr"), _MASTER_ADDR)
        self.assertEqual(_argv_after(argv, "--master-port"), _MASTER_PORT)
        self.assertEqual(_argv_after(argv, "--pipeline-parallel-size"), str(_PP))
        self.assertEqual(_argv_after(argv, "--tensor-parallel-size"), str(_TP))
        self.assertEqual(_argv_after(argv, "--nnodes"), str(_NNODES))
        self.assertEqual(_argv_after(argv, "--distributed-executor-backend"), "mp")
        # It still serves the model and carries the per-cell derived max-model-len.
        self.assertIn("serve", argv)
        self.assertIn(self.job.model_id, argv)
        self.assertEqual(_argv_after(argv, "--max-model-len"), self.job._derive_max_model_len())
        self.assertEqual(_argv_after(argv, "--port"), "8888")

    def test_server_argv_rank1(self):
        argv = self.job._server_argv(1)
        # Only the rank differs between hosts; rendezvous coords stay identical.
        self.assertEqual(_argv_after(argv, "--node-rank"), "1")
        self.assertEqual(_argv_after(argv, "--master-addr"), _MASTER_ADDR)
        self.assertEqual(_argv_after(argv, "--master-port"), _MASTER_PORT)

    def test_node_rank_is_the_only_per_rank_difference(self):
        """Invariant: argv(rank0) and argv(rank1) differ ONLY in the --node-rank
        value -- every other rendezvous/parallelism flag is rank-invariant. A
        smeared rank or a per-rank master-addr drift is caught here."""
        a0 = self.job._server_argv(0)
        # Verify invariant for ranks 1, 2, 3 — not just rank=1.
        for rank in range(1, 4):
            with self.subTest(rank=rank):
                ar = self.job._server_argv(rank)
                self.assertEqual(len(a0), len(ar),
                                  "argv length must be identical for all ranks")
                diffs = [(x, y) for x, y in zip(a0, ar) if x != y]
                self.assertEqual(len(diffs), 1,
                                  f"only --node-rank value should differ for rank {rank}")
                self.assertEqual(diffs[0], ("0", str(rank)))

    def test_serve_args_appended(self):
        # Per-model serve_args from roles.server.serve_args flow through.
        v = _fake_variant()
        v.roles.server.serve_args = {"kv-cache-dtype": "fp8"}
        job = VllmDistributedJob(
            orch=FakeOrch(), variant=v, hf_token="tok",
            isl=_ISL, osl=2048, concurrency=256, num_prompts=12800,
        )
        argv = job._server_argv(0)
        self.assertEqual(_argv_after(argv, "--kv-cache-dtype"), "fp8")

    def test_serve_args_bare_flag(self):
        v = _fake_variant()
        v.roles.server.serve_args = {"trust-remote-code": True}
        job = VllmDistributedJob(orch=FakeOrch(), variant=v, hf_token="tok",
                                 isl=_ISL, osl=2048, concurrency=256, num_prompts=12800)
        argv = job._server_argv(0)
        self.assertIn("--trust-remote-code", argv)
        idx = argv.index("--trust-remote-code")
        next_is_flag_or_end = (idx + 1 >= len(argv)) or argv[idx + 1].startswith("--")
        self.assertTrue(
            next_is_flag_or_end,
            f"bare flag --trust-remote-code must not be followed by a value token, "
            f"got: {argv[idx + 1] if idx + 1 < len(argv) else '<end>'}",
        )

    def test_serve_args_list_value(self):
        v = _fake_variant()
        v.roles.server.serve_args = {"lora-modules": ["mod-a", "mod-b"]}
        job = VllmDistributedJob(orch=FakeOrch(), variant=v, hf_token="tok",
                                 isl=_ISL, osl=2048, concurrency=256, num_prompts=12800)
        argv = job._server_argv(0)
        # list renders as --flag v1 --flag v2
        indices = [i for i, a in enumerate(argv) if a == "--lora-modules"]
        self.assertEqual(len(indices), 2)
        self.assertEqual(argv[indices[0] + 1], "mod-a")
        self.assertEqual(argv[indices[1] + 1], "mod-b")

    def test_server_argv_tracks_pp_and_nnodes(self):
        """F2: _server_argv must reflect non-default pipeline_parallel_size and
        nnodes values. Verifies two points: (pp=4, nnodes=4) and (pp=1, nnodes=1)."""
        # Point 1: pp=4, nnodes=4
        v = _fake_variant()
        v.params.pipeline_parallel_size = "4"
        v.params.nnodes = "4"
        job = VllmDistributedJob(orch=FakeOrch(), variant=v, hf_token="tok",
                                 isl=_ISL, osl=2048, concurrency=256, num_prompts=12800)
        argv = job._server_argv(0)
        self.assertEqual(_argv_after(argv, "--pipeline-parallel-size"), "4")
        self.assertEqual(_argv_after(argv, "--nnodes"), "4")

        # Point 2: pp=1, nnodes=1 — must carry these values and NOT the prior "4" defaults
        v2 = _fake_variant()
        v2.params.pipeline_parallel_size = "1"
        v2.params.nnodes = "1"
        job2 = VllmDistributedJob(orch=FakeOrch(), variant=v2, hf_token="tok",
                                  isl=_ISL, osl=2048, concurrency=256, num_prompts=12800)
        argv2 = job2._server_argv(0)
        self.assertEqual(_argv_after(argv2, "--pipeline-parallel-size"), "1")
        self.assertEqual(_argv_after(argv2, "--nnodes"), "1")
        self.assertNotEqual(_argv_after(argv2, "--pipeline-parallel-size"), "2",
                            "pp=1 variant must not emit the default '2'")

    def test_serve_args_tuple_value(self):
        """F3: tuple values in serve_args must render the same as list values —
        --flag v1 --flag v2."""
        v = _fake_variant()
        v.roles.server.serve_args = {"lora-modules": ("mod-a", "mod-b")}
        job = VllmDistributedJob(orch=FakeOrch(), variant=v, hf_token="tok",
                                 isl=_ISL, osl=2048, concurrency=256, num_prompts=12800)
        argv = job._server_argv(0)
        indices = [i for i, a in enumerate(argv) if a == "--lora-modules"]
        self.assertEqual(len(indices), 2,
                         "--lora-modules must appear twice for a tuple with two values")
        self.assertEqual(argv[indices[0] + 1], "mod-a")
        self.assertEqual(argv[indices[1] + 1], "mod-b")


class TestDerivedMaxModelLen(unittest.TestCase):
    """MAX_MODEL_LEN derived per cell: worst = (isl+osl)*(1+r); + prefix + pad.
    Identical contract to VllmJob. Range table via subTest + boundary cases."""

    def test_derive_max_model_len_ranges(self):
        # (random_range_ratio, random_prefix_len) -> expected string.
        # base cell: isl=128, osl=2048 => isl+osl = 2176.
        cases = [
            ("0.8", "0",  "3925"),   # ceil(2176*1.8)=3917 +0  +8
            ("0.1", "0",  "2402"),   # ceil(2176*1.1)=2394 +0  +8
            ("0.0", "0",  "2184"),   # 2176              +0  +8 (boundary: no jitter)
            ("0.0", "64", "2248"),   # 2176              +64 +8 (prefix only)
            ("0.8", "64", "3989"),   # ceil(2176*1.8)=3917 +64 +8 (ratio AND prefix combined)
            ("1.0", "0",  "4360"),   # ceil(2176*2.0)=4352 +0 +8 (boundary: full-width doubling)
        ]
        for ratio, prefix, expected in cases:
            with self.subTest(ratio=ratio, prefix=prefix):
                job = _make_job(FakeOrch())
                job.random_range_ratio = ratio
                job.random_prefix_len = prefix
                self.assertEqual(job._derive_max_model_len(), expected)

    def test_monotonic_in_ratio(self):
        """Invariant: the derived window is non-decreasing as the jitter ratio
        grows (a wider sampling band can only need a longer max-model-len)."""
        job = _make_job(FakeOrch())
        prev = None
        for ratio in ("0.0", "0.1", "0.4", "0.8", "1.0"):
            job.random_range_ratio = ratio
            cur = int(job._derive_max_model_len())
            if prev is not None:
                self.assertGreaterEqual(cur, prev)
            prev = cur

    def test_derive_max_model_len_varies_isl(self):
        # isl=512, osl=1024, ratio=0.8: ceil((512+1024)*1.8) + 0 + 8 = ceil(2764.8)+8 = 2773
        job = VllmDistributedJob(
            orch=FakeOrch(), variant=_fake_variant(), hf_token="tok",
            isl=512, osl=1024, concurrency=16, num_prompts=800,
        )
        job.random_range_ratio = "0.8"
        job.random_prefix_len = "0"
        import math
        expected = str(math.ceil((512 + 1024) * 1.8) + 8)
        self.assertEqual(job._derive_max_model_len(), expected)


# ---------------------------------------------------------------------------
# Broadcast vs head routing: build_server_cmd, start_server, run_client,
# wait_client_complete, parse_results.
# Classification: subsystem -- mocked at the orch seam. The load-bearing
# contract is WHICH orch entrypoint (broadcast exec vs exec_on_head) each step
# uses and, for start_server, the per-host rank dispatch.
# ---------------------------------------------------------------------------


class TestBuildServerCmdBroadcast(unittest.TestCase):
    def test_build_server_cmd_broadcasts_to_all_nodes(self):
        """Env-script write and out-dir mkdir must go to ALL nodes: broadcast
        orch.exec() with no per-host targeting (no hosts kwarg, or hosts=None).
        A hosts=[one] here would leave the other node without the env/out-dir."""
        orch = FakeOrch()
        job = _make_job(orch)
        job.build_server_cmd()

        self.assertGreaterEqual(len(orch.exec_calls), 2,
                                "expected at least env-script write + mkdir")
        # Every exec issued by build_server_cmd is a broadcast (hosts unset/None).
        for cmd, kwargs in orch.exec_calls:
            self.assertIsNone(kwargs.get("hosts"),
                              f"build_server_cmd must broadcast, got hosts={kwargs.get('hosts')!r} for {cmd!r}")
        # It does not touch the head-only channel.
        self.assertEqual(orch.head_calls, [])
        joined = " ".join(c for c, _ in orch.exec_calls)
        self.assertIn("/tmp/server_env_script.sh", joined)
        self.assertIn("mkdir -p", joined)
        mkdir_cmd = next((c for c, _ in orch.exec_calls if "mkdir" in c), None)
        self.assertIsNotNone(mkdir_cmd, "no mkdir command found in exec_calls")
        self.assertIn(job.out_dir, mkdir_cmd, "mkdir must target job.out_dir")

    def test_env_script_carries_required_exports(self):
        orch = FakeOrch()
        job = _make_job(orch)
        job.build_server_cmd()
        env_cmd = next((c for c, _ in orch.exec_calls if "/tmp/server_env_script.sh" in c), None)
        self.assertIsNotNone(env_cmd, "no env-script write command found")
        for token in (
            "HF_TOKEN=tok",
            f"HF_HUB_CACHE={job.models_dir}",
            "VLLM_USE_AITER_UNIFIED_ATTENTION=1",
            "VLLM_ROCM_USE_AITER_MHA=0",
            "VLLM_ROCM_USE_AITER_FUSED_MOE_A16W4=1",
        ):
            self.assertRegex(env_cmd, r'\bexport\s+' + re.escape(token),
                             f"env script must export '{token}'")

    def test_out_dir_encodes_cell_parameters(self):
        """F8: out_dir must encode (isl, osl, concurrency) so that distinct cells
        write to distinct directories. Verifies both uniqueness and substring presence."""
        orch_a = FakeOrch()
        job_a = VllmDistributedJob(
            orch=orch_a, variant=_fake_variant(), hf_token="tok",
            isl=128, osl=2048, concurrency=16, num_prompts=800,
        )
        orch_b = FakeOrch()
        job_b = VllmDistributedJob(
            orch=orch_b, variant=_fake_variant(), hf_token="tok",
            isl=512, osl=1024, concurrency=32, num_prompts=800,
        )
        self.assertNotEqual(job_a.out_dir, job_b.out_dir,
                            "distinct (isl, osl, concurrency) must produce distinct out_dir values")
        # Each out_dir must contain its cell parameters as substrings.
        for val in ("128", "2048", "16"):
            self.assertIn(val, job_a.out_dir,
                          f"job_a.out_dir must contain '{val}' (isl/osl/concurrency)")
        for val in ("512", "1024", "32"):
            self.assertIn(val, job_b.out_dir,
                          f"job_b.out_dir must contain '{val}' (isl/osl/concurrency)")

    def test_env_script_carries_server_env_overrides(self):
        v = _fake_variant()
        v.roles.server.env = {"CUSTOM_VAR": "custom_val"}
        orch = FakeOrch()
        job = VllmDistributedJob(orch=orch, variant=v, hf_token="tok",
                                 isl=_ISL, osl=2048, concurrency=256, num_prompts=12800)
        job.build_server_cmd()
        # Find the env-script write among the broadcast calls (order-independent).
        env_cmd = next((c for c, _ in orch.exec_calls if "/tmp/server_env_script.sh" in c), None)
        self.assertIsNotNone(env_cmd, "no env-script write found in exec_calls")
        self.assertRegex(env_cmd, r'\bexport\s+CUSTOM_VAR=custom_val',
                         "env script must export CUSTOM_VAR=custom_val")


class TestStartServerPerHost(unittest.TestCase):
    def test_start_server_calls_per_host_with_correct_rank(self):
        """start_server iterates enumerate(orch.hosts): exactly one
        orch.exec(..., hosts=[host]) per host, and host i carries --node-rank i.
        Verified with N=2 and N=4 to rule out special-casing the first two hosts."""
        for n_hosts in (2, 4):
            hosts = [f"node-{i}" for i in range(n_hosts)]
            with self.subTest(n_hosts=n_hosts):
                orch = FakeOrch(hosts=hosts)
                job = _make_job(orch)
                job.start_server()

                # One targeted exec per host; never a broadcast launch and never head-only.
                self.assertEqual(len(orch.exec_calls), n_hosts)
                self.assertEqual(orch.head_calls, [])

                seen = {}  # host -> rank parsed from the launched command
                for cmd, kwargs in orch.exec_calls:
                    h_list = kwargs.get("hosts")
                    self.assertIsInstance(h_list, list)
                    self.assertEqual(len(h_list), 1, "each launch targets exactly one host")
                    host = h_list[0]
                    m = re.search(r"--node-rank\s+(\d+)", cmd)
                    self.assertIsNotNone(m, f"launch for {host} has no --node-rank: {cmd!r}")
                    seen[host] = int(m.group(1))
                    # Per-host command must source env script and carry distributed flags.
                    self.assertIn("source /tmp/server_env_script.sh", cmd,
                                  "per-host launch must source the env script")
                    self.assertIn("--distributed-executor-backend", cmd)
                    self.assertIn("--master-addr", cmd)
                    self.assertIn("--pipeline-parallel-size", cmd)
                    self.assertIn("--nnodes", cmd)
                    self.assertIn("nohup", cmd, "per-host launch must use nohup to background")
                    self.assertRegex(cmd, r'2>&1\s*&', "per-host launch must be backgrounded with 2>&1 &")
                    self.assertIn("vllm serve", cmd, "per-host launch must use 'vllm serve' subcommand")

                self.assertEqual(set(seen), set(hosts), "every host got launched exactly once")
                # host at orch.hosts[i] must receive rank i.
                for i, host in enumerate(hosts):
                    self.assertEqual(seen[host], i, f"{host} (index {i}) got rank {seen[host]}")

    def test_start_server_raises_on_early_failure(self):
        """If a host's launch output matches EARLY_FAILURE_RE, start_server must
        raise RuntimeError. Covers all four EARLY_FAILURE_RE arms via subTest."""
        early_failure_strings = [
            "command not found: vllm",
            "no such file or directory: vllm",
            "cannot access /usr/bin/vllm",
            "failed to start vllm server",
        ]
        for failure_text in early_failure_strings:
            with self.subTest(failure=failure_text):
                def make_fail(ft=failure_text):
                    def fail(cmd, kwargs):
                        host = kwargs.get("hosts", [None])[0]
                        return {host: (ft if host == _HOSTS[1] else "")}
                    return fail
                orch = FakeOrch(hosts=_HOSTS, exec_return=make_fail())
                job = _make_job(orch)
                with self.assertRaises(RuntimeError):
                    job.start_server()

    def test_start_server_raises_on_head_failure(self):
        """Rank-0 (head) failure must also raise — not silently ignored."""
        def head_fails(cmd, kwargs):
            host = kwargs.get("hosts", [None])[0]
            return {host: ("command not found: vllm" if host == _HOSTS[0] else "")}
        orch = FakeOrch(hosts=_HOSTS, exec_return=head_fails)
        job = _make_job(orch)
        with self.assertRaises(RuntimeError):
            job.start_server()

    def test_start_server_clean_launch_does_not_raise(self):
        orch = FakeOrch(hosts=_HOSTS, exec_return=lambda c, k: {k.get("hosts", [None])[0]: ""})
        job = _make_job(orch)
        job.start_server()  # no raise on empty/benign output

    def test_start_server_clean_launch_with_benign_output(self):
        """Non-empty output that does NOT match EARLY_FAILURE_RE must not raise.
        Distinguishes EARLY_FAILURE_RE gating from 'any non-empty raises'."""
        benign = "nohup: ignoring input\nINFO: starting vllm..."

        def benign_return(cmd, kwargs):
            host = kwargs.get("hosts", [None])[0]
            return {host: benign}

        orch = FakeOrch(hosts=_HOSTS, exec_return=benign_return)
        job = _make_job(orch)
        job.start_server()  # must NOT raise despite non-empty output


class TestRunClientHeadOnly(unittest.TestCase):
    def test_run_client_uses_exec_on_head(self):
        """The bench client runs ONLY on the head node: via exec_on_head, never
        the broadcast orch.exec (which would start N redundant clients)."""
        orch = FakeOrch()
        job = _make_job(orch)
        job.run_client()
        self.assertEqual(len(orch.head_calls), 1)
        self.assertEqual(orch.exec_calls, [])
        client_cmd = orch.head_calls[0][0]
        self.assertIn("bench serve", client_cmd)
        self.assertIn("source /tmp/server_env_script.sh", client_cmd,
                      "run_client must source the env script")
        self.assertIn(job.client_log, client_cmd)
        self.assertIn("2>&1 &", client_cmd, "run_client must background the bench client with 2>&1 &")
        self.assertIn("--result-dir", client_cmd)
        self.assertIn(job.out_dir, client_cmd)
        self.assertIn("--result-filename", client_cmd)
        self.assertIn("--save-result", client_cmd)
        self.assertIn("--ignore-eos", client_cmd)
        self.assertIn("--model", client_cmd)
        self.assertIn(job.model_id, client_cmd)
        self.assertIn("--base-url", client_cmd)
        self.assertIn(f"{job.base_url}:{job.port_no}", client_cmd)
        self.assertIn("--num-prompts", client_cmd)
        # Use regex to verify each flag is paired with its correct value,
        # preventing false passes from substring collisions (e.g. 128 in 12800).
        def _flag_val(flag):
            m = re.search(re.escape(flag) + r'\s+(\S+)', client_cmd)
            return m.group(1) if m else None
        self.assertEqual(_flag_val("--result-filename"), "results")
        self.assertEqual(_flag_val("--random-input-len"), str(_ISL))
        self.assertEqual(_flag_val("--random-output-len"), str(2048))
        self.assertEqual(_flag_val("--max-concurrency"), str(256))
        self.assertEqual(_flag_val("--num-prompts"), str(12800))
        self.assertEqual(_flag_val("--percentile-metrics"), "ttft,tpot,itl,e2el")
        self.assertEqual(_flag_val("--metric-percentiles"), "50,90,95,99")
        self.assertEqual(_flag_val("--random-range-ratio"), "0.8")
        self.assertEqual(_flag_val("--random-prefix-len"), "0")
        self.assertEqual(_flag_val("--request-rate"), "inf")
        self.assertEqual(_flag_val("--burstiness"), "1.0")
        self.assertEqual(_flag_val("--seed"), "0")
        self.assertEqual(_flag_val("--tokenizer-mode"), "auto")

    def test_run_client_goodput_flag_built_from_slo(self):
        orch = FakeOrch()
        job = _make_job(orch, goodput_slo={"ttft_ms": 500.0, "tpot_ms": 50.0, "e2el_ms": 60000.0})
        job.run_client()
        self.assertEqual(orch.exec_calls, [], "run_client must not broadcast even with goodput_slo set")
        cmd = orch.head_calls[0][0]
        self.assertIn("--goodput", cmd)
        for tok in ("ttft:500.0", "tpot:50.0", "e2el:60000.0"):
            self.assertIn(tok, cmd)

    def test_run_client_goodput_omitted_when_none(self):
        orch = FakeOrch()
        job = _make_job(orch, goodput_slo=None)
        job.run_client()
        self.assertEqual(orch.exec_calls, [], "run_client must not broadcast even with goodput_slo=None")
        self.assertNotIn("--goodput", orch.head_calls[0][0])

    def test_run_client_goodput_sparse_dict_skips_none_keys(self):
        # Only ttft_ms present; tpot and e2el absent — must NOT emit tpot:None or e2el:None.
        orch = FakeOrch()
        job = _make_job(orch, goodput_slo={"ttft_ms": 500.0})
        job.run_client()
        self.assertEqual(orch.exec_calls, [], "run_client must not broadcast even with sparse goodput_slo")
        cmd = orch.head_calls[0][0]
        self.assertIn("--goodput", cmd)
        self.assertIn("ttft:500.0", cmd)
        self.assertNotIn("tpot:", cmd)
        self.assertNotIn("e2el:", cmd)


class TestWaitClientCompleteHeadOnly(unittest.TestCase):
    """wait_client_complete polls the client log on the head node only
    (exec_on_head returns {head: text}). Failure-before-completion semantics
    mirror VllmJob but iterate the single-entry dict."""

    def _job_with_head(self, head_text):
        orch = FakeOrch(head_return={_HEAD: head_text})
        # Pass timing overrides via constructor kwargs (F4: constructor path).
        job = _make_job(orch,
                        client_initial_wait_s=0,
                        client_poll_wait_s=0,
                        client_poll_count=1)
        # Belt-and-suspenders: also set private attrs so the test works regardless
        # of whether the impl reads constructor kwargs or private attrs.
        job._client_initial_wait = 0
        job._client_poll_wait = 0
        job._client_poll_count = 1
        return orch, job

    def test_uses_exec_on_head(self):
        orch, job = self._job_with_head("Serving Benchmark Result\nFailed requests: 0\n")
        job.wait_client_complete()
        self.assertGreaterEqual(len(orch.head_calls), 1)
        self.assertEqual(orch.exec_calls, [], "client poll must not broadcast")
        polled_cmd = orch.head_calls[-1][0]
        self.assertIn(job.client_log, polled_cmd, "wait_client_complete must poll job.client_log")

    def test_completion_marker_alone_is_sufficient(self):
        # Completion banner without a "Failed requests:" summary line must still return.
        orch, job = self._job_with_head("Serving Benchmark Result\n")
        job.wait_client_complete()  # must not raise
        self.assertGreaterEqual(len(orch.head_calls), 1)

    def test_nonzero_failed_requests_raises(self):
        orch, job = self._job_with_head("Serving Benchmark Result\nFailed requests: 7\n")
        with self.assertRaises(RuntimeError):
            job.wait_client_complete()
        self.assertEqual(orch.exec_calls, [], "failure path must not broadcast")

    def test_zero_failed_requests_is_not_a_failure(self):
        # "Failed requests: 0" is always printed — it must NOT raise.
        # Distinguish from test_completion_marker_alone_is_sufficient by asserting
        # the head channel was actually called (not a no-op return).
        orch, job = self._job_with_head("Serving Benchmark Result\nFailed requests: 0\n")
        job.wait_client_complete()
        self.assertGreaterEqual(len(orch.head_calls), 1, "poll must have been issued")
        self.assertEqual(orch.exec_calls, [], "must not broadcast")

    def test_client_crash_raises(self):
        orch, job = self._job_with_head("Traceback (most recent call last):\n  ...\n")
        with self.assertRaises(RuntimeError):
            job.wait_client_complete()
        self.assertEqual(orch.exec_calls, [], "failure path must not broadcast")

    def test_launch_failure_raises(self):
        arms = [
            "error: argument --bogus: unrecognized arguments\n",
            "invalid choice: bench\n",
            "command not found: vllm\n",
            "/usr/bin/vllm: No such file or directory\n",
        ]
        for text in arms:
            with self.subTest(text=text.strip()):
                orch, job = self._job_with_head(text)
                with self.assertRaises(RuntimeError):
                    job.wait_client_complete()
                self.assertEqual(orch.exec_calls, [], "failure path must not broadcast")

    def test_no_completion_within_cap_raises(self):
        # Log never shows the summary -> poll cap exhausted -> RuntimeError.
        orch, job = self._job_with_head("still warming up...\n")
        with self.assertRaises(RuntimeError):
            job.wait_client_complete()
        self.assertEqual(orch.exec_calls, [], "timeout path must not broadcast")

    def test_completion_on_second_poll(self):
        """Deferred-completion path: banner absent on poll 1, present on poll 2.
        An implementation that exits after exactly one poll (ignoring the loop
        cap) would raise RuntimeError here instead of returning normally."""
        call_count = [0]

        def deferred(cmd, kwargs):
            call_count[0] += 1
            if call_count[0] >= 2:
                return {_HEAD: "Serving Benchmark Result\nFailed requests: 0\n"}
            return {_HEAD: "still warming up...\n"}

        orch = FakeOrch(head_return=deferred)
        # Pass timing overrides via constructor kwargs (F4: constructor path).
        job = _make_job(orch,
                        client_initial_wait_s=0,
                        client_poll_wait_s=0,
                        client_poll_count=3)
        # Belt-and-suspenders: also set private attrs.
        job._client_initial_wait = 0
        job._client_poll_wait = 0
        job._client_poll_count = 3
        job.wait_client_complete()  # must return without raising
        self.assertGreaterEqual(len(orch.head_calls), 2, "must have polled at least twice")


class TestWaitReadyBroadcast(unittest.TestCase):
    """wait_ready / is_ready poll the server log on ALL nodes via broadcast
    orch.exec (not exec_on_head). For distributed runs this is load-bearing:
    a crashed rank-1 shard must be detected, not silently skipped."""

    def _job_ready(self, exec_return):
        orch = FakeOrch(hosts=_HOSTS, exec_return=exec_return)
        # Pass timing overrides via constructor kwargs (F4: constructor path).
        job = _make_job(orch,
                        server_precheck_wait_s=0,
                        server_warmup_wait_s=0,
                        server_poll_count=3,
                        server_poll_wait_s=0)
        # Belt-and-suspenders: also set private attrs.
        job._precheck_wait = 0
        job._warmup_wait = 0
        job._server_poll_count = 3
        job._server_poll_wait = 0
        return orch, job

    # Command-discriminating dispatcher: wait_ready calls orch.exec for two purposes:
    # (1) tail -30 <server_log>  — early-failure pre-check; expects string per-host values
    # (2) grep -qiE <pattern> <server_log>  — is_ready poll; expects {exit_code: N} dicts
    # The mock must return the correct shape for each call type to avoid TypeErrors.
    @staticmethod
    def _ready_dispatcher(grep_exit_code_fn):
        """Return an exec_return callable that dispatches on command type.

        grep_exit_code_fn(call_index) -> int: exit code for is_ready grep calls.
        Tail/precheck calls always return benign empty strings.
        """
        grep_calls = [0]

        def dispatch(cmd, kwargs):
            if "grep" in cmd:
                grep_calls[0] += 1
                code = grep_exit_code_fn(grep_calls[0])
                return {h: {"exit_code": code} for h in _HOSTS}
            # tail / other precheck: return benign string output
            return {h: "" for h in _HOSTS}
        return dispatch

    def test_is_ready_uses_broadcast_exec(self):
        """is_ready must poll via broadcast orch.exec (all nodes), not exec_on_head.
        Also asserts the command targets job.server_log (not a hardcoded/wrong path)."""
        orch = FakeOrch(hosts=_HOSTS, exec_return=self._ready_dispatcher(lambda _: 0))
        job = _make_job(orch)
        job.is_ready()
        self.assertTrue(orch.exec_calls, "is_ready must issue at least one broadcast exec")
        self.assertEqual(orch.head_calls, [], "is_ready must not use exec_on_head")
        grep_cmd = next((c for c, _ in orch.exec_calls if "grep" in c), None)
        self.assertIsNotNone(grep_cmd, "is_ready must issue a grep command")
        self.assertIn(job.server_log, grep_cmd, "is_ready must grep job.server_log")
        for arm in ("Application startup complete", "Uvicorn running", "Started server"):
            self.assertIn(arm.lower(), grep_cmd.lower(),
                          f"READINESS_RE arm '{arm}' missing from grep command")

    def test_wait_ready_returns_when_all_nodes_ready(self):
        """wait_ready returns normally when is_ready() reports all nodes up."""
        orch, job = self._job_ready(self._ready_dispatcher(lambda _: 0))
        job.wait_ready()  # must not raise
        self.assertTrue(orch.exec_calls, "wait_ready must have issued at least one exec")
        tail_cmd = next((c for c, _ in orch.exec_calls if "tail" in c), None)
        if tail_cmd is not None:
            self.assertIn(job.server_log, tail_cmd,
                          "wait_ready precheck must tail job.server_log")

    def test_wait_ready_raises_on_timeout(self):
        """wait_ready raises RuntimeError when poll cap is exhausted without readiness."""
        orch, job = self._job_ready(self._ready_dispatcher(lambda _: 1))
        with self.assertRaises(RuntimeError):
            job.wait_ready()

    def test_wait_ready_raises_when_one_node_not_ready(self):
        """Distributed-specific: a single not-ready node must hold the entire
        cluster in the not-ready state. Distinguishes all() from any()."""
        def one_lagging(cmd, kwargs):
            if "grep" in cmd:
                # node-a ready, node-b not — partial cluster case
                return {"node-a": {"exit_code": 0}, "node-b": {"exit_code": 1}}
            return {h: "" for h in _HOSTS}

        orch, job = self._job_ready(one_lagging)
        with self.assertRaises(RuntimeError):
            job.wait_ready()

    def test_wait_ready_returns_on_second_poll(self):
        """Deferred-readiness: not ready on poll 1, ready on poll 2.
        An implementation that calls is_ready() only once raises instead of returning."""
        grep_calls = [0]

        def deferred_ready(cmd, kwargs):
            if "grep" in cmd:
                grep_calls[0] += 1
                code = 0 if grep_calls[0] >= 2 else 1
                return {h: {"exit_code": code} for h in _HOSTS}
            return {h: "" for h in _HOSTS}

        orch, job = self._job_ready(deferred_ready)
        job.wait_ready()  # must return without raising
        grep_count = sum(1 for c, _ in orch.exec_calls if "grep" in c)
        self.assertGreaterEqual(grep_count, 2, "must have polled is_ready at least twice")

    def test_wait_ready_raises_on_non_head_early_failure(self):
        """Non-head node (rank>0) early-failure must also raise — not silently ignored.
        Pins the distributed-specific contract: every shard is inspected."""
        for failure_text in ("command not found: vllm", "no such file or directory: vllm"):
            with self.subTest(failure=failure_text):
                def non_head_fail(cmd, kwargs, _ft=failure_text):
                    if "tail" in cmd:
                        # node-a (head) is clean; node-b (rank-1) reports the failure
                        return {"node-a": "", "node-b": _ft}
                    return {h: {"exit_code": 0} for h in _HOSTS}

                orch = FakeOrch(hosts=_HOSTS, exec_return=non_head_fail)
                # Pass timing overrides via constructor kwargs (F4: constructor path).
                job = _make_job(orch,
                                server_precheck_wait_s=0,
                                server_warmup_wait_s=0,
                                server_poll_count=1,
                                server_poll_wait_s=0)
                # Belt-and-suspenders: also set private attrs.
                job._precheck_wait = 0
                job._warmup_wait = 0
                job._server_poll_count = 1
                job._server_poll_wait = 0
                with self.assertRaises(RuntimeError):
                    job.wait_ready()

    def test_wait_ready_raises_on_early_failure(self):
        """EARLY_FAILURE_RE output during the readiness wait raises immediately.
        Covers all four EARLY_FAILURE_RE arms via subTest."""
        early_failure_strings = [
            "command not found: vllm",
            "no such file or directory: vllm",
            "cannot access /usr/bin/vllm",
            "failed to start vllm server",
        ]
        for failure_text in early_failure_strings:
            with self.subTest(failure=failure_text):
                def early_fail(cmd, kwargs, _ft=failure_text):
                    if "tail" in cmd:
                        return {"node-a": _ft, "node-b": ""}
                    return {h: {"exit_code": 0} for h in _HOSTS}

                orch = FakeOrch(hosts=_HOSTS, exec_return=early_fail)
                # Pass timing overrides via constructor kwargs (F4: constructor path).
                job = _make_job(orch,
                                server_precheck_wait_s=0,
                                server_warmup_wait_s=0,
                                server_poll_count=1,
                                server_poll_wait_s=0)
                # Belt-and-suspenders: also set private attrs.
                job._precheck_wait = 0
                job._warmup_wait = 0
                job._server_poll_count = 1
                job._server_poll_wait = 0
                with self.assertRaises(RuntimeError):
                    job.wait_ready()


class TestParseResultsHeadOnly(unittest.TestCase):
    """parse_results cats the `results` artifact on the head node via
    exec_on_head and delegates to to_client_metrics. Raises on
    empty/missing/unparseable."""

    def _parse(self, head_text):
        orch = FakeOrch(head_return={_HEAD: head_text})
        job = _make_job(orch)
        return orch, job

    def test_parse_results_uses_exec_on_head(self):
        orch, job = self._parse(_load_fixture("vllm_results_widened.json"))
        out = job.parse_results()
        self.assertGreaterEqual(len(orch.head_calls), 1)
        self.assertEqual(orch.exec_calls, [], "results fetch must not broadcast")
        # Delegation produced namespaced client.* metrics for the head host.
        self.assertIn(_HEAD, out)
        self.assertIn("client.total_token_throughput", out[_HEAD])
        fetch_cmd = orch.head_calls[-1][0]
        self.assertIn(job.out_dir, fetch_cmd, "parse_results must cat from out_dir")
        # Verify exact artifact name 'results' (not a superstring like 'results.json').
        self.assertRegex(fetch_cmd, r'(?<!\w)results(?![\w.])',
                         "parse_results must cat the bare 'results' artifact")

    def test_parse_results_derived_metrics_exact(self):
        orch, job = self._parse(_load_fixture("vllm_results_widened.json"))
        w = json.loads(_load_fixture("vllm_results_widened.json"))
        m = job.parse_results()[_HEAD]
        self.assertAlmostEqual(m["client.total_token_throughput"], w["total_token_throughput"])
        self.assertAlmostEqual(m["client.per_gpu_throughput"], w["total_token_throughput"] / _TP)
        self.assertAlmostEqual(m["client.normalized_ttft_ms_per_tok"], w["mean_ttft_ms"] / _ISL)
        self.assertAlmostEqual(
            m["client.decode_latency_ratio"],
            w["p99_itl_ms"] / w["p50_itl_ms"],
        )
        self.assertAlmostEqual(
            m["client.decode_throughput_p50"],
            1000 / w["median_tpot_ms"],
        )
        # success_rate = completed/(completed+failed). The fixture has completed=1791,
        # failed=1409, sum=3200 == num_prompts — so we verify with synthetic data where
        # completed+failed != num_prompts to distinguish from a /num_prompts bug.
        synthetic = dict(w, completed=1500, failed=200)  # sum=1700 != 3200=num_prompts
        synthetic_text = json.dumps(synthetic)
        orch2 = FakeOrch(head_return={_HEAD: synthetic_text})
        job2 = _make_job(orch2)
        m2 = job2.parse_results()[_HEAD]
        self.assertAlmostEqual(m2["client.success_rate"], 1500 / 1700)
        # client.goodput alias contract: must equal request_goodput, not be absent or renamed.
        self.assertAlmostEqual(m["client.goodput"], w["request_goodput"])
        self.assertAlmostEqual(m["client.request_goodput"], w["request_goodput"])

    def test_parse_results_derived_metrics_scale_with_tp_and_isl(self):
        """per_gpu_throughput and normalized_ttft must track job.tp and job.isl,
        not hardcoded constants. Verified with two distinct (tp, isl) points."""
        fixture_text = _load_fixture("vllm_results_widened.json")
        w = json.loads(fixture_text)
        cases = [
            (4, 256),
            (16, 64),
        ]
        for tp, isl in cases:
            with self.subTest(tp=tp, isl=isl):
                v = _fake_variant()
                v.params.tensor_parallelism = str(tp)
                orch = FakeOrch(head_return={_HEAD: fixture_text})
                job = VllmDistributedJob(
                    orch=orch, variant=v, hf_token="tok",
                    isl=isl, osl=1024, concurrency=16, num_prompts=800,
                )
                m = job.parse_results()[_HEAD]
                self.assertAlmostEqual(
                    m["client.per_gpu_throughput"],
                    w["total_token_throughput"] / tp,
                )
                self.assertAlmostEqual(
                    m["client.normalized_ttft_ms_per_tok"],
                    w["mean_ttft_ms"] / isl,
                )

    def test_parse_results_decode_latency_ratio_none_when_p50_itl_absent(self):
        """On real vllm bench serve artifacts p50_itl_ms is absent (only median_itl_ms
        is emitted). decode_latency_ratio must be None in that case, not a fallback value."""
        w = json.loads(_load_fixture("vllm_results_widened.json"))
        w.pop("p50_itl_ms", None)   # simulate a real artifact
        orch = FakeOrch(head_return={_HEAD: json.dumps(w)})
        job = _make_job(orch)
        m = job.parse_results()[_HEAD]
        self.assertIsNone(m.get("client.decode_latency_ratio"),
                          "decode_latency_ratio must be None when p50_itl_ms is absent")

    def test_parse_results_empty_raises(self):
        orch, job = self._parse("")
        with self.assertRaises(RuntimeError):
            job.parse_results()

    def test_parse_results_unparseable_raises(self):
        orch, job = self._parse("not json {{{")
        with self.assertRaises(RuntimeError):
            job.parse_results()


# ---------------------------------------------------------------------------
# Lifecycle: VllmDistributedJob is a stateful subsystem (server up -> client ->
# stopped). The value-add rows are the routing-illegal cases (a step that must
# NOT broadcast, a step that must NOT go head-only) and idempotent re-entry.
# ---------------------------------------------------------------------------


class TestVllmDistributedJobLifecycle(unittest.TestCase):
    """Transition table (routing is the observable state of each phase):

    | from        | event                  | to / effect                                   |
    |-------------|------------------------|-----------------------------------------------|
    | constructed | build_server_cmd()     | env+mkdir BROADCAST to all nodes              |
    | env-ready   | start_server()         | one TARGETED exec per host, rank i -> host i  |
    | started     | run_client()           | HEAD-ONLY exec_on_head, no broadcast          |
    | client-up   | wait_client_complete() | HEAD-ONLY poll                                |
    | complete    | parse_results()        | HEAD-ONLY cat + parse                         |
    | any         | stop_server()          | pkill BROADCAST to all nodes                  |

    Illegal/guard rows: run_client must never broadcast; build_server_cmd must
    never go head-only; start_server on a single host yields a single rank-0.
    """

    def test_legal_sequence_routes_correctly(self):
        # exec_return: dispatches by command type so each method gets the right shape.
        # is_ready uses grep with detailed=True → {exit_code: 0} dicts.
        # start_server / build_server_cmd expect string output for EARLY_FAILURE_RE.
        def broadcast_return(cmd, kwargs):
            targets = kwargs.get("hosts") or _HOSTS
            if kwargs.get("detailed") or "grep" in cmd:
                return {h: {"exit_code": 0} for h in targets}
            return {h: "" for h in targets}

        # head_return: dispatch on command content so the mock is deterministic
        # regardless of how many exec_on_head calls run_client issues internally.
        # parse_results cats job.out_dir/.../results; wait_client_complete tails/greps
        # job.client_log. We detect the parse_results call by the presence of "results"
        # and job.out_dir (the artifact path), and return the JSON only for that call.
        results_text = _load_fixture("vllm_results_widened.json")
        # Build a temp job to get out_dir for the dispatch predicate.
        _tmp_job = _make_job(FakeOrch(hosts=_HOSTS))
        _out_dir = _tmp_job.out_dir
        del _tmp_job

        def stateful_head(cmd, kwargs):
            # parse_results calls: cat <out_dir>/results
            if _out_dir in cmd and "results" in cmd:
                return {_HEAD: results_text}
            # wait_client_complete and run_client calls: return the completion banner
            return {_HEAD: "Serving Benchmark Result\nFailed requests: 0\n"}

        orch = FakeOrch(
            hosts=_HOSTS,
            exec_return=broadcast_return,
            head_return=stateful_head,
        )
        # Pass all timing overrides via constructor kwargs (F4: constructor path).
        job = _make_job(orch,
                        client_initial_wait_s=0,
                        client_poll_wait_s=0,
                        client_poll_count=1,
                        server_precheck_wait_s=0,
                        server_warmup_wait_s=0,
                        server_poll_count=1,
                        server_poll_wait_s=0)
        # Belt-and-suspenders: also set private attrs.
        job._client_initial_wait = 0
        job._client_poll_wait = 0
        job._client_poll_count = 1
        job._precheck_wait = 0
        job._warmup_wait = 0
        job._server_poll_count = 1
        job._server_poll_wait = 0

        job.build_server_cmd()
        broadcast_after_build = len(orch.exec_calls)
        self.assertTrue(all(k.get("hosts") is None for _, k in orch.exec_calls))

        job.start_server()
        # start_server added exactly one targeted exec per host.
        per_host = orch.exec_calls[broadcast_after_build:]
        self.assertEqual(len(per_host), len(_HOSTS))
        self.assertTrue(all(isinstance(k.get("hosts"), list) for _, k in per_host))

        exec_before_ready = len(orch.exec_calls)
        job.wait_ready()
        # wait_ready must have issued at least one broadcast exec (no hosts kwarg).
        ready_execs = orch.exec_calls[exec_before_ready:]
        self.assertTrue(ready_execs, "wait_ready issued no broadcast exec")
        self.assertTrue(all(k.get("hosts") is None for _, k in ready_execs))

        head_before_client = len(orch.head_calls)
        job.run_client()
        job.wait_client_complete()
        out = job.parse_results()
        # Every client-side step used the head channel, none broadcast.
        self.assertGreater(len(orch.head_calls), head_before_client)
        self.assertIn(_HEAD, out)

    def test_stop_server_broadcasts(self):
        """Illegal-if-head-only: teardown must pkill on ALL nodes, else a stray
        server lingers on the non-head node. Broadcast exec, no head-only call."""
        orch = FakeOrch(hosts=_HOSTS)
        job = _make_job(orch)
        with unittest.mock.patch("time.sleep"):
            job.stop_server()
        self.assertTrue(orch.exec_calls, "stop_server issued no command")
        self.assertEqual(orch.head_calls, [], "stop_server must not use exec_on_head")
        for cmd, kwargs in orch.exec_calls:
            self.assertIsNone(kwargs.get("hosts"),
                              "stop_server must broadcast to all nodes")
            self.assertIn("pkill", cmd)
            self.assertIn("-f", cmd, "pkill must use -f for full-cmdline match")
            self.assertIn("vllm serve", cmd, "pkill must target 'vllm serve' process")

    def test_stop_server_is_idempotent(self):
        """Idempotent re-entry: calling stop_server twice must not raise (pkill
        on an already-dead server is a no-op). Both calls must still broadcast."""
        orch = FakeOrch(hosts=_HOSTS)
        job = _make_job(orch)
        with unittest.mock.patch("time.sleep"):
            job.stop_server()
            after_first = len(orch.exec_calls)
            self.assertGreater(after_first, 0, "first stop_server issued no exec")
            job.stop_server()
        self.assertGreater(len(orch.exec_calls), after_first,
                           "second stop_server must still broadcast pkill")
        self.assertEqual(orch.head_calls, [], "stop_server must not use exec_on_head")
        for cmd, kwargs in orch.exec_calls:
            self.assertIn("pkill", cmd)
            self.assertIn("vllm serve", cmd, "pkill must target 'vllm serve' process")

    def test_single_host_cluster_yields_single_rank0(self):
        """Boundary (nnodes->1 host in orch.hosts): start_server dispatches a
        single rank-0 launch, never a phantom rank-1."""
        orch = FakeOrch(hosts=["solo-node"])
        job = _make_job(orch)
        job.start_server()
        self.assertEqual(len(orch.exec_calls), 1)
        cmd, kwargs = orch.exec_calls[0]
        self.assertEqual(kwargs.get("hosts"), ["solo-node"])
        self.assertEqual(re.search(r"--node-rank\s+(\d+)", cmd).group(1), "0")


# ---------------------------------------------------------------------------
# VariantConfig.cell_key format (spec File 1). Pure function over its three
# args + the configured TP/PP. Imported lazily so a not-yet-existent loader
# fails THIS test cleanly (RED) rather than erroring collection of the module.
# ---------------------------------------------------------------------------


class TestCellKeyFormat(unittest.TestCase):
    def _variant_config(self):
        from cvs.lib.inference.utils.vllm_distributed_config_loader import VariantConfig
        return VariantConfig(**self._raw_config())

    @staticmethod
    def _raw_config():
        # Minimal raw config kwargs sufficient to construct VariantConfig with the
        # spec's default params (tensor_parallelism=8, pipeline_parallel_size=2).
        return {
            "schema_version": 1,
            "framework": "vllm_distributed",
            "gpu_arch": "mi300x",
            "enforce_thresholds": False,
            "paths": {"shared_fs": "/tmp", "models_dir": "/tmp/models",
                      "log_dir": "/tmp/LOGS", "hf_token_file": "/tmp/tok"},
            "model": {"id": "amd/Llama-3.1-70B-Instruct-FP8-KV", "remote": 0},
            "roles": {"server": {"serve_args": {}}},
            "params": {"master_addr": "node-head"},
            "sweep": {"sequence_combinations": [], "runs": []},
            "thresholds": {},
        }

    def test_cell_key_format(self):
        """cell_key(isl, osl, conc) == 'ISL=<isl>,OSL=<osl>,TP=<tp>,PP=<pp>,CONC=<c>'
        with TP/PP pulled from params (default 8 / 2). The PP segment is the
        distributed-specific addition over vllm_single's key."""
        vc = self._variant_config()
        self.assertEqual(
            vc.cell_key("1000", "1000", 16),
            "ISL=1000,OSL=1000,TP=8,PP=2,CONC=16",
        )

    def test_cell_key_reflects_param_values(self):
        # TP/PP in the key track the configured params, not hardcoded literals.
        # Use non-default values (TP=2, PP=1) so the test independently falsifies
        # any implementation that hardcodes TP=8/PP=2.
        raw = self._raw_config()
        raw["params"]["tensor_parallelism"] = "2"
        raw["params"]["pipeline_parallel_size"] = "1"
        from cvs.lib.inference.utils.vllm_distributed_config_loader import VariantConfig
        vc = VariantConfig(**raw)
        key = vc.cell_key("8000", "1024", 16)
        self.assertIn("TP=2", key)
        self.assertIn("PP=1", key)
        self.assertNotIn("TP=8", key)
        self.assertNotIn("PP=2", key)
        self.assertIn("ISL=8000", key)
        self.assertIn("OSL=1024", key)
        self.assertIn("CONC=16", key)

    def test_cell_key_uses_non_default_tp_pp(self):
        # If cell_key hardcodes TP=8/PP=2 rather than reading params, this fails.
        raw = self._raw_config()
        raw["params"]["tensor_parallelism"] = "4"
        raw["params"]["pipeline_parallel_size"] = "4"
        from cvs.lib.inference.utils.vllm_distributed_config_loader import VariantConfig
        vc = VariantConfig(**raw)
        key = vc.cell_key("1000", "1000", 16)
        self.assertIn("TP=4", key)
        self.assertIn("PP=4", key)
        self.assertNotIn("TP=8", key)
        self.assertNotIn("PP=2", key)


if __name__ == "__main__":
    unittest.main()
