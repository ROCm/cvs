'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for the Ray distributed-executor-backend support added to
cvs.lib.inference.vllm_job.VllmJob and
cvs.lib.inference.utils.vllm_config_loader.VariantConfig.

Impl-blind / spec-derived (greenfield): these tests are written from the
behavioral spec before the implementation exists and are committed RED. A
different agent makes them green and may NOT edit this file.

Coverage map (spec AC -> test):
  Config validator relaxation ...... AC1-6   -> TestVariantConfigRayConsistency
  cell_key ray multi-node .......... AC7     -> TestCellKeyRayMultiNode
  _is_ray_backend .................. AC8     -> TestIsRayBackend
  _server_argv ray vs mp ........... AC16,17 + RC1,RC3 -> TestServerArgvRayVsMp
  start_server bootstrap/order ..... AC9-15,26,27 -> TestStartServerRayBootstrap
  stop_server teardown ............. AC18-21 -> TestStopServerRayTeardown
  _check_early_failure ray skip .... AC22,23 -> TestCheckEarlyFailureRayWorkerSkip
  server_signature ................. AC24,25 -> TestServerSignatureRay
  lifecycle (transition table) ..... regr    -> TestVllmJobRayLifecycle

Coverage-gap additions (post-review, impl-blind against the same spec):
  ray pp>1 keeps --pipeline-parallel-size ......... TestServerArgvRayVsMp
  serve-launch EARLY_FAILURE (ray head, mp worker)  TestStartServerRayBootstrap
  bootstrap OR-branch matrix (bad-out head, exit!=0 worker) TestStartServerRayBootstrap
  re-entrant start_server ......................... TestVllmJobRayLifecycle
  node-rank strip strengthened (mp, non-vacuous) .. TestServerSignatureRay
  worker master_addr regression (distinct from hosts[0]) TestStartServerRayBootstrap

Round-2 coverage-gap additions (impl-blind against the same spec):
  6  empty/None exec silent-success guard ......... TestStartServerRayBootstrap
  7  nnodes=3 worker loop (all bootstrap; fail last) TestStartServerRayBootstrap
  8  ray+pp>1 through the REAL VariantConfig ....... TestVariantConfigRayConsistency
  9  stop-after-failed-start asserts teardown calls  TestVllmJobRayLifecycle
  10 FATAL_LOG_RE grep -> "vllm server fatal error"  TestCheckEarlyFailureRayWorkerSkip

Post-review round additions (impl-blind against the same spec):
  R1.1 bootstrap-fail return omitting 'output' key . TestStartServerRayBootstrap
  R1.2 server_signature env tuple independent oracle TestServerSignatureRay
  R2.1 mp dist-block flag VALUES (not just presence)  TestServerArgvRayVsMp
  R2.2 is_ready True/False/empty + multinode skip .. TestVllmJobIsReady
  R2.3 parse_results empty/unparseable/delegation .. TestVllmJobParseResults

Round-3 coverage-gap additions (impl-blind against the same spec):
  R3.1 parse_results asserts RETURN value (not just delegation) TestVllmJobParseResults
  R3.2 wait_ready poll state machine (return/timeout/order) .. TestVllmJobWaitReady
  R3.3 build_server_cmd env-script + per-rank mkdir branches .. TestVllmJobBuildServerCmd

Round-4 coverage-gap additions (impl-blind against the same spec):
  R4.1 server_env pass-through pins KEY=VALUE (non-colliding) TestVllmJobBuildServerCmd
  R4.2 ray server_signature invariant to nnodes (2 vs 3) .... TestServerSignatureRay
  R4.3 cell_key pp>1 branch (PP= segment) + pp==1, subTest ... TestCellKeyRayMultiNode
  R4.4 _flatten_serve_args list/tuple repeat branch ......... TestFlattenServeArgsBranches
'''

import unittest
import unittest.mock as mock
from types import SimpleNamespace

from pydantic import ValidationError

from cvs.lib.inference.utils.vllm_config_loader import VariantConfig
from cvs.lib.inference.vllm_job import VllmJob

RAY = {"distributed-executor-backend": "ray"}

# EARLY_FAILURE_RE-matching / non-matching bootstrap outputs (spec Failure Modes).
_CLEAN = "Local node IP: 10.0.0.1"  # confirmed NON-matching
_BAD = "command not found"  # confirmed matching


# --------------------------------------------------------------------------- #
# Fakes / fixtures (per spec "Test fake orchestrator contract")
# --------------------------------------------------------------------------- #
class RecordingOrch:
    """Records (cmd, hosts) per exec call so host-targeting and ordering ACs
    are checkable. `responder(cmd, hosts, detailed) -> dict` controls returns."""

    hosts = ["10.0.0.1", "10.0.0.2"]  # index 0 = head/rank0, 1 = worker/rank1

    def __init__(self, responder=None, hosts=None, head_responder=None):
        self.calls = []  # list of (cmd, hosts) in call order
        self.head_cmds = []
        self._responder = responder
        # head_responder(cmd) -> return value for exec_on_head; None preserves the
        # legacy {} return so existing tests that never inspect exec_on_head output
        # are unaffected. Only parse_results (which fetches via exec_on_head) needs it.
        self._head_responder = head_responder
        if hosts is not None:
            self.hosts = list(hosts)

    def exec(self, cmd, hosts=None, detailed=False, **k):
        self.calls.append((cmd, hosts))
        if self._responder is not None:
            return self._responder(cmd, hosts, detailed)
        return {}

    def exec_on_head(self, cmd, *a, **k):
        self.head_cmds.append(cmd)
        if self._head_responder is not None:
            return self._head_responder(cmd)
        return {}


HEAD = RecordingOrch.hosts[0]
WORKER = RecordingOrch.hosts[1]
HOST2 = "10.0.0.3"  # third host for nnodes=3 worker-loop coverage (not in default hosts)


def _responder_ok():
    """Every bootstrap succeeds (exit 0, clean output); serve launch is clean.

    A detailed `grep` (the _check_early_failure FATAL scan) returns exit_code 1
    = "no fatal pattern found"; every other detailed call (ray bootstrap) returns
    exit_code 0 = success. Non-detailed calls (serve launch / tail) return clean
    text that does not match EARLY_FAILURE_RE.
    """

    def r(cmd, hosts, detailed):
        host = hosts[0] if hosts else HEAD
        if detailed:
            exit_code = 1 if "grep" in cmd else 0
            return {host: {"exit_code": exit_code, "output": _CLEAN, "stdout": ""}}
        return {host: ""}

    return r


def _responder_bootstrap_fail(fail_map):
    """fail_map: host -> detailed return dict for that host's bootstrap; all
    other hosts succeed, serve launches are clean."""

    def r(cmd, hosts, detailed):
        host = hosts[0] if hosts else HEAD
        if detailed:
            return {host: fail_map.get(host, {"exit_code": 0, "output": _CLEAN, "stdout": _CLEAN})}
        return {host: ""}

    return r


def _responder_serve_fail(bad_serve_hosts):
    """Ray/mp bootstrap detailed calls all succeed (exit 0, clean output); the
    NON-detailed `vllm serve` launch returns EARLY_FAILURE_RE-matching output for
    hosts in `bad_serve_hosts`, clean otherwise.

    Exercises the post-bootstrap serve-launch EARLY_FAILURE check (the
    "vllm server failed to launch on ... (rank N)" RuntimeError site), which is
    distinct from the bootstrap failure sites covered by _responder_bootstrap_fail.
    """

    def r(cmd, hosts, detailed):
        host = hosts[0] if hosts else HEAD
        if detailed:
            # No grep is issued by start_server; bootstrap detailed calls succeed.
            exit_code = 1 if "grep" in cmd else 0
            return {host: {"exit_code": exit_code, "output": _CLEAN, "stdout": ""}}
        if "vllm serve" in cmd and host in bad_serve_hosts:
            return {host: _BAD}
        return {host: ""}

    return r


def _responder_const(value):
    """Every orch.exec call (bootstrap detailed AND serve non-detailed) returns
    the SAME constant `value` -- used to drive the empty/None silent-success guard
    (`(out or {}).items()`) in _bootstrap_ray_cluster and start_server. With
    value={} or value=None the guard's iterable is empty, so no per-host failure
    check runs and no false positive is raised."""

    def r(cmd, hosts, detailed):
        return value

    return r


# FATAL_LOG_RE-matching text confirmed by the class regex (see stub FATAL_LOG_RE:
# "...|Engine core initialization failed|..."). Distinct from EARLY_FAILURE_RE.
_FATAL = "Engine core initialization failed"


def _responder_fatal_grep(fatal_hosts, fatal_text=_FATAL):
    """Detailed `grep` (the _check_early_failure FATAL_LOG_RE scan) returns
    exit_code 0 (match found) with FATAL-matching `stdout` for hosts in
    `fatal_hosts`; every other detailed call (and every host's grep otherwise)
    returns exit_code 1 = no match. Non-detailed `tail` returns clean text that
    does NOT match EARLY_FAILURE_RE, so the FATAL_LOG_RE branch -- not the tail
    EARLY_FAILURE branch -- is the one that fires."""

    def r(cmd, hosts, detailed):
        host = hosts[0] if hosts else HEAD
        if detailed:
            if "grep" in cmd and host in fatal_hosts:
                return {host: {"exit_code": 0, "stdout": fatal_text, "output": fatal_text}}
            return {host: {"exit_code": 1, "stdout": "", "output": _CLEAN}}
        return {host: ""}

    return r


def _responder_readiness(exit_code=0, empty=False):
    """is_ready() greps each non-skipped rank's readiness log via
    orch.exec(detailed=True) and returns {host: {"exit_code": ...}}; exit_code 0
    means the readiness pattern was found (server ready). empty=True returns {}
    to exercise the `not out` (empty result) False path. Non-detailed calls
    return clean text (unused by is_ready)."""

    def r(cmd, hosts, detailed):
        if empty:
            return {}
        host = hosts[0] if hosts else HEAD
        if detailed:
            return {host: {"exit_code": exit_code, "output": "", "stdout": ""}}
        return {host: ""}

    return r


def _variant(serve_args=None, nnodes="2", pp="2", ib_netdev="enp159s0np0", tp="8", master_addr="10.0.0.1", env=None):
    """Minimal SimpleNamespace variant mirroring _variant() in the reuse suite."""
    params = SimpleNamespace(
        tensor_parallelism=tp,
        pipeline_parallel_size=pp,
        master_addr=master_addr,
        master_port="29501",
        nnodes=nnodes,
        port_no="8000",
        random_range_ratio="0.0",
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
        model=SimpleNamespace(id="/models/test-model"),
        paths=SimpleNamespace(log_dir="/logs", models_dir="/models"),
        roles=SimpleNamespace(
            server=SimpleNamespace(serve_args=dict(serve_args or {}), env=dict(env or {}), ib_netdev=ib_netdev)
        ),
    )


def _job(
    orch=None,
    serve_args=None,
    nnodes="2",
    pp="2",
    ib_netdev="enp159s0np0",
    concurrency=16,
    isl="1024",
    osl="1024",
    tp="8",
    master_addr="10.0.0.1",
    env=None,
    ib_hcas=None,
):
    orch = RecordingOrch() if orch is None else orch
    return VllmJob(
        orch=orch,
        variant=_variant(serve_args, nnodes, pp, ib_netdev, tp, master_addr, env),
        hf_token="tok",
        isl=isl,
        osl=osl,
        concurrency=concurrency,
        num_prompts="640",
        ib_hcas=ib_hcas,
    )


def _vc(nnodes="2", pp="1", serve_args=None, ib_netdev="eth0", tp="8"):
    """A real pydantic VariantConfig exercising _check_distributed_consistency.

    enforce_thresholds=False so the (independent) threshold-coverage validator
    only warns and never masks the distributed-consistency error under test.
    """
    return VariantConfig(
        schema_version=1,
        framework="vllm",
        gpu_arch="mi300x",
        enforce_thresholds=False,
        paths={
            "shared_fs": "/home/x",
            "models_dir": "/home/x/models",
            "log_dir": "/home/x/LOGS",
            "hf_token_file": "/home/x/.hf",
        },
        model={"id": "/models/test-model", "remote": 0},
        params={"tensor_parallelism": tp, "pipeline_parallel_size": pp, "nnodes": nnodes},
        roles={"server": {"serve_args": dict(serve_args or {}), "env": {}, "ib_netdev": ib_netdev}},
        sweep={
            "sequence_combinations": [{"name": "a", "isl": "1024", "osl": "1024"}],
            "runs": [{"combo": "a", "concurrency": 16}],
        },
        thresholds={},
    )


# --------------------------------------------------------------------------- #
# helpers for argv / call inspection
# --------------------------------------------------------------------------- #
def _value_after(argv, flag):
    """Return the element following `flag` in argv, or None if flag absent."""
    for i, a in enumerate(argv):
        if a == flag:
            return argv[i + 1] if i + 1 < len(argv) else None
    return None


def _calls_to(orch, host):
    return [cmd for cmd, hosts in orch.calls if hosts == [host]]


def _first_index(orch, predicate):
    for i, (cmd, hosts) in enumerate(orch.calls):
        if predicate(cmd, hosts):
            return i
    return -1


def _all_cmds(orch):
    return [c for c, _ in orch.calls] + list(orch.head_cmds)


# --------------------------------------------------------------------------- #
# Config validator: VariantConfig._check_distributed_consistency  (AC1-6)
# --------------------------------------------------------------------------- #
class TestVariantConfigRayConsistency(unittest.TestCase):
    """The ray relaxation applies ONLY to the (nn>1 & pp==1) rule and ONLY for
    the exact string 'ray'. ib_netdev and the (pp>1 & nn==1) rule are untouched."""

    def test_accepts_valid(self):
        # (nnodes, pp, serve_args, ib_netdev) that must construct without error.
        cases = [
            ("1", "1", {}, None),  # baseline: unrelaxed single-node default path
            ("2", "1", RAY, "eth0"),  # AC1: ray relaxation permits nn>1 & pp==1
            ("2", "2", {}, "eth0"),  # AC3: mp multi-node path unchanged
            ("2", "2", RAY, "eth0"),  # finding 8: ray + pp>1 is legal (nn>1 & pp>1
            # is valid for ANY backend; the ray relaxation only special-cases pp==1,
            # it never REJECTS ray+pp>1). Validated through the REAL VariantConfig
            # validator, not just the SimpleNamespace fake used by _server_argv tests.
        ]
        for nn, pp, sa, ib in cases:
            with self.subTest(nnodes=nn, pp=pp, serve_args=sa):
                try:
                    _vc(nnodes=nn, pp=pp, serve_args=sa, ib_netdev=ib)
                except ValidationError as e:  # pragma: no cover - failure path
                    self.fail(f"unexpected ValidationError: {e}")

    def test_rejects_invalid(self):
        # (nnodes, pp, serve_args, ib_netdev, field-token-in-message)
        cases = [
            ("2", "1", {}, "eth0", "pipeline_parallel_size"),  # AC2 no ray key
            ("1", "2", RAY, "eth0", "pipeline_parallel_size"),  # AC4 pp>1 & nn==1 never relaxed
            ("2", "1", RAY, None, "ib_netdev"),  # AC5 ib_netdev not relaxed by ray
            ("2", "1", {"distributed-executor-backend": "RAY"}, "eth0", "pipeline_parallel_size"),  # AC6 case-sensitive
            ("2", "1", {"distributed-executor-backend": "Ray"}, "eth0", "pipeline_parallel_size"),  # AC6 case-sensitive
        ]
        for nn, pp, sa, ib, token in cases:
            with self.subTest(nnodes=nn, pp=pp, serve_args=sa, ib_netdev=ib):
                with self.assertRaises(ValidationError) as ctx:
                    _vc(nnodes=nn, pp=pp, serve_args=sa, ib_netdev=ib)
                self.assertIn(token, str(ctx.exception))


class TestCellKeyRayMultiNode(unittest.TestCase):
    """AC7: ray multi-node has pp=1, so cell_key uses the single-node format
    (no PP= segment), identical to a genuine single-node cell.

    Round-4 finding 3: cell_key has two branches -- pp==1 (no PP= segment) and
    pp>1 (a "PP=<pp>," segment inserted before CONC). The pp>1 branch had zero
    coverage anywhere for THIS VariantConfig class, so both branches are now
    pinned together in one subTest table (discipline rule B), asserting the exact
    segment position/value/comma placement, not just presence."""

    def test_cell_key_format_both_pp_branches(self):
        # (nnodes, pp, serve_args, ib_netdev, expected_key)
        cases = [
            # AC7: ray multi-node, pp==1 -> single-node format, NO PP= segment.
            ("2", "1", RAY, "eth0", "ISL=1024,OSL=1024,TP=8,CONC=16"),
            # pp>1 branch -> "PP=2," inserted immediately before CONC. pp>1 requires
            # nnodes>1 (the pp>1 & nn==1 rule always fires), so this is a valid mp
            # multi-node config; the PP segment is what distinguishes it from the
            # pp==1 key above.
            ("2", "2", {}, "eth0", "ISL=1024,OSL=1024,TP=8,PP=2,CONC=16"),
        ]
        for nn, pp, sa, ib, expected in cases:
            with self.subTest(nnodes=nn, pp=pp):
                vc = _vc(nnodes=nn, pp=pp, serve_args=sa, ib_netdev=ib, tp="8")
                self.assertEqual(vc.cell_key(isl="1024", osl="1024", concurrency="16"), expected)


# --------------------------------------------------------------------------- #
# _is_ray_backend  (AC8)
# --------------------------------------------------------------------------- #
class TestIsRayBackend(unittest.TestCase):
    def test_backend_detection_is_exact_string(self):
        # (serve_args, expected)
        cases = [
            ({"distributed-executor-backend": "ray"}, True),
            ({}, False),
            ({"distributed-executor-backend": "mp"}, False),
            ({"distributed-executor-backend": "RAY"}, False),
            ({"distributed-executor-backend": "Ray"}, False),
        ]
        for sa, expected in cases:
            with self.subTest(serve_args=sa):
                job = _job(serve_args=sa, nnodes="1", pp="1")
                self.assertIs(job._is_ray_backend, expected)

    def test_property_reflects_live_serve_args_not_a_cached_snapshot(self):
        job = _job(serve_args={}, nnodes="1", pp="1")
        self.assertIs(job._is_ray_backend, False)
        job.serve_args["distributed-executor-backend"] = "ray"
        self.assertIs(job._is_ray_backend, True)


# --------------------------------------------------------------------------- #
# _server_argv  (AC16, AC17, RC1, RC3)
# --------------------------------------------------------------------------- #
class TestServerArgvRayVsMp(unittest.TestCase):
    _DRIVER_DIST_FLAGS = [
        "--node-rank",
        "--headless",
        "--pipeline-parallel-size",
        "--master-addr",
        "--master-port",
        "--nnodes",
    ]

    def test_ray_multinode_omits_all_driver_dist_flags(self):
        # AC16: the mp block is skipped entirely under ray.
        argv = _job(serve_args=RAY, nnodes="2", pp="1")._server_argv(0)
        for flag in self._DRIVER_DIST_FLAGS:
            with self.subTest(flag=flag):
                self.assertNotIn(flag, argv)

    def test_ray_multinode_backend_arrives_via_serve_args(self):
        # AC17: --distributed-executor-backend ray comes from _flatten_serve_args.
        argv = _job(serve_args=RAY, nnodes="2", pp="1")._server_argv(0)
        self.assertIn("--distributed-executor-backend", argv)
        self.assertEqual(_value_after(argv, "--distributed-executor-backend"), "ray")

    def test_mp_multinode_injects_full_dist_block(self):
        # RC1: mp multi-node keeps the driver-injected block + hardcoded mp backend.
        # Round-2 finding 1: assert the VALUE after each flag, not just presence. A
        # mutant that emits the right flag names but wrong/hardcoded values -- e.g.
        # swapping master_addr/master_port, hardcoding --nnodes 1, or dropping the
        # pipeline width -- would pass a presence-only check while breaking the
        # launch. Each expected value is pinned to the job's real attribute (read
        # from variant.params, not re-read from the produced argv), so the check is
        # independent of the argv it is validating.
        job = _job(serve_args={}, nnodes="2", pp="2")
        argv = job._server_argv(0)
        expected = [
            ("--node-rank", "0"),  # the rank argument passed to _server_argv(0)
            ("--master-addr", job.master_addr),
            ("--master-port", job.master_port),
            ("--nnodes", job.nnodes),
            ("--pipeline-parallel-size", job.pp),
            ("--distributed-executor-backend", "mp"),  # hardcoded on the mp path
        ]
        for flag, val in expected:
            with self.subTest(flag=flag):
                self.assertIn(flag, argv)
                self.assertEqual(_value_after(argv, flag), val)

    def test_mp_worker_rank_is_headless(self):
        # RC1: rank>0 mp worker additionally carries --headless; rank 0 does not.
        # Round-2 finding 1: also pin --node-rank's VALUE to the actual rank arg, so
        # a mutant that always emits "--node-rank 0" regardless of rank (breaking
        # multi-node distribution) is caught -- not merely flag/--headless presence.
        job = _job(serve_args={}, nnodes="2", pp="2")
        argv0 = job._server_argv(0)
        argv1 = job._server_argv(1)
        self.assertNotIn("--headless", argv0)
        self.assertIn("--headless", argv1)
        self.assertEqual(_value_after(argv0, "--node-rank"), "0")
        self.assertEqual(_value_after(argv1, "--node-rank"), "1")

    def test_single_node_ray_passthrough_no_driver_flags(self):
        # RC3 / Edge: single-node omits all driver-injected dist flags, but the
        # user's serve_args backend still passes through verbatim.
        argv = _job(serve_args=RAY, nnodes="1", pp="1")._server_argv(0)
        for flag in self._DRIVER_DIST_FLAGS:
            with self.subTest(flag=flag):
                self.assertNotIn(flag, argv)
        self.assertEqual(_value_after(argv, "--distributed-executor-backend"), "ray")

    def test_ray_multinode_pp_gt_1_keeps_pipeline_parallel_size(self):
        # Coverage-gap (finding 1): VariantConfig permits a ray backend with
        # nnodes>1 AND pp>1 (the ray relaxation only special-cases pp==1; the
        # nn>1 & pp>1 combo is legal for any backend). The mp block that normally
        # carries "--pipeline-parallel-size" is skipped for every ray job, so a
        # ray+pp=2 config must NOT silently drop the pipeline-parallel width: the
        # head's single `vllm serve` still has to be told pp=2 (via the flag with
        # value self.pp) or the cluster silently runs at pp=1. A mutant that drops
        # the flag under ray (the current guard `nnodes>1 and not _is_ray_backend`)
        # is caught here.
        argv = _job(serve_args=RAY, nnodes="2", pp="2")._server_argv(0)
        self.assertIn("--pipeline-parallel-size", argv)
        self.assertEqual(_value_after(argv, "--pipeline-parallel-size"), "2")
        # Backend is still ray (contributed by serve_args passthrough), not mp.
        self.assertEqual(_value_after(argv, "--distributed-executor-backend"), "ray")
        # Ray still manages rendezvous, so the torchrun-style mp flags stay absent
        # even though pp>1 (ray does not use --node-rank/--master-*/--nnodes/--headless).
        for flag in ("--node-rank", "--headless", "--master-addr", "--master-port", "--nnodes"):
            with self.subTest(flag=flag):
                self.assertNotIn(flag, argv)


# --------------------------------------------------------------------------- #
# _flatten_serve_args: the four equivalence classes (RC8)
# --------------------------------------------------------------------------- #
class TestFlattenServeArgsBranches(unittest.TestCase):
    """RC8: _flatten_serve_args has four branches -- True (bare flag), False
    (omitted), list/tuple (flag repeated per element), scalar (flag + str(value)).
    True/False/scalar are covered in test_vllm_job_server_reuse.py; the list/tuple
    repeat branch (Round-4 finding 4) is covered here so all four cells of this
    pure function's table are pinned (discipline rule B). A bug in the repeat branch
    (wrong flag repeated, values not str()-cast, wrong order) is caught."""

    def test_list_and_tuple_values_repeat_the_flag_per_element(self):
        # (value, expected) -- list and tuple both repeat "--<key>" before each
        # element, in order; non-string elements are str()-cast.
        cases = [
            (["a.b.C", "d.e.F"], ["--middleware", "a.b.C", "--middleware", "d.e.F"]),
            (("a.b.C", "d.e.F"), ["--middleware", "a.b.C", "--middleware", "d.e.F"]),
            ([1, 2], ["--middleware", "1", "--middleware", "2"]),  # str()-cast
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(VllmJob._flatten_serve_args({"middleware": value}), expected)


# --------------------------------------------------------------------------- #
# start_server: ray bootstrap ordering, host targeting, failure  (AC9-15,26,27)
# --------------------------------------------------------------------------- #
class TestStartServerRayBootstrap(unittest.TestCase):
    def test_head_bootstrap_command_and_target(self):
        # AC9
        orch = RecordingOrch(responder=_responder_ok())
        _job(orch=orch, serve_args=RAY, nnodes="2", pp="1").start_server()
        head_ray = [c for c in _calls_to(orch, HEAD) if "ray start" in c]
        self.assertTrue(head_ray, "expected a ray start command targeting the head")
        cmd = head_ray[0]
        for token in ("ray start", "--head", "--port=29501"):
            with self.subTest(token=token):
                self.assertIn(token, cmd)

    def test_worker_bootstrap_command_and_target(self):
        # AC10
        orch = RecordingOrch(responder=_responder_ok())
        _job(orch=orch, serve_args=RAY, nnodes="2", pp="1").start_server()
        worker_ray = [c for c in _calls_to(orch, WORKER) if "ray start" in c]
        self.assertTrue(worker_ray, "expected a ray start command targeting the worker")
        cmd = worker_ray[0]
        self.assertIn("ray start", cmd)
        self.assertIn("--address=10.0.0.1:29501", cmd)

    def test_worker_bootstrap_targets_master_addr_not_head_host(self):
        # AC10 disambiguation / REGRESSION: the worker's Ray rendezvous --address
        # must be self.master_addr (the data-plane IP the head actually started
        # with via `ray start --head --port=...`), NOT self.orch.hosts[0] (the
        # SSH/management host). The default fixture sets master_addr == hosts[0]
        # ("10.0.0.1"), so the plain AC10 test above passes regardless of which
        # field the impl uses. Here master_addr is DISTINCT from hosts[0]
        # (hosts=["10.0.0.1","10.0.0.2"], master_addr="172.16.0.1"), so only an
        # impl that targets master_addr passes; one that targets hosts[0] fails.
        orch = RecordingOrch(responder=_responder_ok())  # hosts[0]=HEAD=10.0.0.1
        _job(orch=orch, serve_args=RAY, nnodes="2", pp="1", master_addr="172.16.0.1").start_server()
        worker_ray = [c for c in _calls_to(orch, WORKER) if "ray start" in c]
        self.assertTrue(worker_ray, "expected a ray start command targeting the worker")
        cmd = worker_ray[0]
        self.assertIn(
            "--address=172.16.0.1:29501",
            cmd,
            "worker rendezvous must target master_addr (data-plane IP), not hosts[0]",
        )
        self.assertNotIn(
            "--address=10.0.0.1:29501",
            cmd,
            "worker must NOT rendezvous against the SSH/management host hosts[0]",
        )

    def test_bootstrap_precedes_serve_launch(self):
        # AC11: every ray start bootstrap (head AND worker) precedes the vllm serve launch.
        orch = RecordingOrch(responder=_responder_ok())
        _job(orch=orch, serve_args=RAY, nnodes="2", pp="1").start_server()
        first_head_bootstrap = _first_index(orch, lambda c, h: "ray start" in c and h == [HEAD])
        first_worker_bootstrap = _first_index(orch, lambda c, h: "ray start" in c and h == [WORKER])
        first_serve = _first_index(orch, lambda c, h: "vllm serve" in c)
        self.assertNotEqual(first_head_bootstrap, -1, "no head ray start call recorded")
        self.assertNotEqual(first_worker_bootstrap, -1, "no worker ray start call recorded")
        self.assertNotEqual(first_serve, -1, "no vllm serve call recorded")
        self.assertLess(first_head_bootstrap, first_serve)
        self.assertLess(first_worker_bootstrap, first_serve)

    def test_no_serve_on_worker_under_ray(self):
        # AC12: vllm serve runs only on the head under ray.
        orch = RecordingOrch(responder=_responder_ok())
        _job(orch=orch, serve_args=RAY, nnodes="2", pp="1").start_server()
        self.assertEqual([c for c in _calls_to(orch, WORKER) if "vllm serve" in c], [])
        self.assertTrue([c for c in _calls_to(orch, HEAD) if "vllm serve" in c])

    def test_mp_serves_every_host_no_ray_start(self):
        # AC13: mp multi-node serves on every host (incl. worker), no ray start.
        orch = RecordingOrch(responder=_responder_ok())
        _job(orch=orch, serve_args={}, nnodes="2", pp="2").start_server()
        self.assertTrue([c for c in _calls_to(orch, HEAD) if "vllm serve" in c])
        self.assertTrue([c for c in _calls_to(orch, WORKER) if "vllm serve" in c])
        self.assertEqual([c for c, _ in orch.calls if "ray start" in c], [])

    def test_single_node_ray_no_bootstrap(self):
        # AC14: 1-node ray issues no ray start and exactly one vllm serve launch.
        orch = RecordingOrch(responder=_responder_ok(), hosts=[HEAD])
        _job(orch=orch, serve_args=RAY, nnodes="1", pp="1", ib_netdev=None).start_server()
        self.assertEqual([c for c in _all_cmds(orch) if "ray start" in c], [])
        serves = [c for c in _all_cmds(orch) if "vllm serve" in c]
        self.assertEqual(len(serves), 1, f"expected exactly one serve launch, got {serves}")

    def test_happy_path_launches_serve_on_head(self):
        # AC15: clean bootstrap -> no exception + serve on head via exec(hosts=[head]).
        orch = RecordingOrch(responder=_responder_ok())
        _job(orch=orch, serve_args=RAY, nnodes="2", pp="1").start_server()
        self.assertTrue([c for c in _calls_to(orch, HEAD) if "vllm serve" in c])

    def test_head_bootstrap_failure_aborts_before_serve(self):
        # AC26: head exit_code!=0 -> RuntimeError(rank 0), no serve, no worker bootstrap.
        orch = RecordingOrch(
            responder=_responder_bootstrap_fail({HEAD: {"exit_code": 1, "output": "something went wrong"}})
        )
        with self.assertRaises(RuntimeError) as ctx:
            _job(orch=orch, serve_args=RAY, nnodes="2", pp="1").start_server()
        msg = str(ctx.exception)
        self.assertIn("ray bootstrap failed on", msg)
        self.assertIn("rank 0", msg)
        self.assertEqual([c for c, _ in orch.calls if "vllm serve" in c], [])
        self.assertEqual([c for c in _calls_to(orch, WORKER) if "ray start" in c], [])

    def test_worker_bootstrap_failure_bad_output(self):
        # AC27: head ok, worker exit 0 but output matches EARLY_FAILURE_RE -> rank 1.
        orch = RecordingOrch(responder=_responder_bootstrap_fail({WORKER: {"exit_code": 0, "output": _BAD}}))
        with self.assertRaises(RuntimeError) as ctx:
            _job(orch=orch, serve_args=RAY, nnodes="2", pp="1").start_server()
        msg = str(ctx.exception)
        self.assertIn("ray bootstrap failed on", msg)
        self.assertIn("rank 1", msg)
        self.assertEqual([c for c, _ in orch.calls if "vllm serve" in c], [])

    # ---- Coverage-gap (finding 5): the bootstrap failure check is an OR of
    # (exit_code != 0) OR (EARLY_FAILURE_RE matches output), applied to BOTH head
    # and worker. Existing tests cover only exit!=0-on-head and bad-output-on-worker;
    # the two mirror combinations below (bad-output-on-head, exit!=0-on-worker) close
    # the OR-branch matrix so a mutant dropping either half on either host is killed.
    def test_head_bootstrap_failure_bad_output_exit0(self):
        # head: exit_code 0 but output matches EARLY_FAILURE_RE -> RuntimeError rank 0,
        # aborting before any worker bootstrap and before serve launch.
        orch = RecordingOrch(responder=_responder_bootstrap_fail({HEAD: {"exit_code": 0, "output": _BAD}}))
        with self.assertRaises(RuntimeError) as ctx:
            _job(orch=orch, serve_args=RAY, nnodes="2", pp="1").start_server()
        msg = str(ctx.exception)
        self.assertIn("ray bootstrap failed on", msg)
        self.assertIn("rank 0", msg)
        self.assertEqual([c for c in _calls_to(orch, WORKER) if "ray start" in c], [])
        self.assertEqual([c for c, _ in orch.calls if "vllm serve" in c], [])

    def test_worker_bootstrap_failure_nonzero_exit_clean_output(self):
        # worker: exit_code != 0 with CLEAN (non-EARLY_FAILURE) output -> RuntimeError
        # rank 1. The head bootstrap succeeded, so the failure is attributed to rank 1.
        orch = RecordingOrch(responder=_responder_bootstrap_fail({WORKER: {"exit_code": 1, "output": _CLEAN}}))
        with self.assertRaises(RuntimeError) as ctx:
            _job(orch=orch, serve_args=RAY, nnodes="2", pp="1").start_server()
        msg = str(ctx.exception)
        self.assertIn("ray bootstrap failed on", msg)
        self.assertIn("rank 1", msg)

    # ---- Coverage-gap (Round-1 finding 1): the bootstrap-failure detailed return
    # dict may OMIT the content key entirely. The spec's Failure Modes section is
    # explicit: the check reads r.get("output", ""), so a return dict missing the
    # content key is treated as empty string -- "no KeyError, no false positive".
    # Every other responder in this file always supplies an "output" key, so a
    # regression that read r["output"] (KeyError on a real orchestrator response
    # missing that key) would slip through. Here the failing head returns a dict
    # with exit_code=1 and NO "output" key at all: start_server() must still raise
    # the normal RuntimeError naming rank 0 (the empty-output path), NOT a KeyError.
    # assertRaises(RuntimeError) does not catch KeyError, so an r["output"] mutant
    # surfaces as a test error/failure rather than a false pass.
    def test_head_bootstrap_failure_output_key_absent_no_keyerror(self):
        orch = RecordingOrch(responder=_responder_bootstrap_fail({HEAD: {"exit_code": 1}}))
        try:
            with self.assertRaises(RuntimeError) as ctx:
                _job(orch=orch, serve_args=RAY, nnodes="2", pp="1").start_server()
        except KeyError as e:  # pragma: no cover - regression guard
            self.fail(f"missing 'output' key must be treated as empty string, not raise KeyError: {e!r}")
        msg = str(ctx.exception)
        self.assertIn("ray bootstrap failed on", msg)
        self.assertIn("rank 0", msg)
        # Aborts before any serve launch, exactly like the with-output failure path.
        self.assertEqual([c for c, _ in orch.calls if "vllm serve" in c], [])
        self.assertEqual([c for c, _ in orch.calls if "vllm serve" in c], [])

    # ---- Coverage-gap (finding 4): the post-bootstrap `vllm serve` launch has its
    # own EARLY_FAILURE_RE check (RuntimeError "vllm server failed to launch on ...
    # (rank N)"), distinct from the bootstrap checks above. No existing test returns
    # EARLY_FAILURE output for the non-detailed serve launch, so these two sites --
    # the ray head launch and the mp non-head-rank launch -- were never exercised.
    def test_ray_head_serve_launch_failure_raises_rank0(self):
        # ray path: bootstrap (head + worker) succeeds, but the head's post-bootstrap
        # vllm serve launch output matches EARLY_FAILURE_RE -> RuntimeError rank 0.
        orch = RecordingOrch(responder=_responder_serve_fail({HEAD}))
        with self.assertRaises(RuntimeError) as ctx:
            _job(orch=orch, serve_args=RAY, nnodes="2", pp="1").start_server()
        msg = str(ctx.exception)
        self.assertIn("vllm server failed to launch on", msg)
        self.assertIn("rank 0", msg)

    def test_mp_worker_serve_launch_failure_raises_rank1(self):
        # mp path (else branch): the non-head-rank (rank 1) vllm serve launch output
        # matches EARLY_FAILURE_RE -> RuntimeError rank 1. Confirms the serve-launch
        # failure check fires for a worker on the mp path, not just the head.
        orch = RecordingOrch(responder=_responder_serve_fail({WORKER}))
        with self.assertRaises(RuntimeError) as ctx:
            _job(orch=orch, serve_args={}, nnodes="2", pp="2").start_server()
        msg = str(ctx.exception)
        self.assertIn("vllm server failed to launch on", msg)
        self.assertIn("rank 1", msg)

    # ---- Coverage-gap (finding 6): the per-host failure check iterates
    # `(out or {}).items()` in BOTH _bootstrap_ray_cluster and the serve-launch
    # scan in start_server. When orch.exec returns {} or None (an empty/omitted
    # result -- which the real orchestrator can produce, and which the spec says
    # must be treated as empty output "no KeyError, no false positive"), the
    # iterable is empty, no host entry is examined, and the code proceeds as a
    # silent success. No prior responder ever returned {}/None, so this guard was
    # never exercised. Intended behavior: start_server does NOT raise and the ray
    # start + vllm serve calls are still issued (returns are recorded regardless).
    def test_empty_or_none_bootstrap_result_is_silent_success(self):
        for value in ({}, None):
            with self.subTest(exec_return=value):
                orch = RecordingOrch(responder=_responder_const(value))
                try:
                    _job(orch=orch, serve_args=RAY, nnodes="2", pp="1").start_server()
                except Exception as e:  # pragma: no cover - failure path
                    self.fail(f"empty/None exec return must be silent-success, raised: {e!r}")
                # The calls were still dispatched (their empty returns just yield no
                # failure to detect): head+worker ray start and a head vllm serve.
                self.assertTrue([c for c in _calls_to(orch, HEAD) if "ray start" in c])
                self.assertTrue([c for c in _calls_to(orch, WORKER) if "ray start" in c])
                self.assertTrue([c for c in _calls_to(orch, HEAD) if "vllm serve" in c])

    # ---- Coverage-gap (finding 7): the worker bootstrap loop was only ever
    # exercised with exactly ONE worker (nnodes=2). A loop that bootstraps only
    # hosts[1] (off-by-one / no loop) would pass every nnodes=2 test. These two
    # nnodes=3 cases pin the loop: (a) EVERY worker is bootstrapped on the happy
    # path; (b) a failure injected on the LAST worker still aborts with the
    # correct rank, proving the loop reaches it (no short-circuit after rank 1).
    def test_three_node_ray_bootstraps_every_worker(self):
        orch = RecordingOrch(responder=_responder_ok(), hosts=[HEAD, WORKER, HOST2])
        _job(orch=orch, serve_args=RAY, nnodes="3", pp="1").start_server()
        # Both workers (rank 1 and rank 2) get a ray start; the head gets one too.
        self.assertTrue([c for c in _calls_to(orch, HEAD) if "ray start" in c], "head ray start missing")
        self.assertTrue([c for c in _calls_to(orch, WORKER) if "ray start" in c], "rank-1 worker ray start missing")
        self.assertTrue([c for c in _calls_to(orch, HOST2) if "ray start" in c], "rank-2 worker ray start missing")
        # Ray still serves only on the head; no serve on either worker.
        self.assertTrue([c for c in _calls_to(orch, HEAD) if "vllm serve" in c])
        self.assertEqual([c for c in _calls_to(orch, WORKER) if "vllm serve" in c], [])
        self.assertEqual([c for c in _calls_to(orch, HOST2) if "vllm serve" in c], [])

    def test_three_node_ray_failure_on_last_worker_aborts_rank2(self):
        # Failure on the LAST worker (rank 2), not the first -- confirms the loop
        # does not short-circuit at rank 1. The head and rank-1 worker bootstrap
        # cleanly; rank-2 fails, so the RuntimeError names rank 2 and no serve runs.
        orch = RecordingOrch(
            responder=_responder_bootstrap_fail({HOST2: {"exit_code": 1, "output": _BAD}}),
            hosts=[HEAD, WORKER, HOST2],
        )
        with self.assertRaises(RuntimeError) as ctx:
            _job(orch=orch, serve_args=RAY, nnodes="3", pp="1").start_server()
        msg = str(ctx.exception)
        self.assertIn("ray bootstrap failed on", msg)
        self.assertIn("rank 2", msg)
        # The loop DID reach the earlier ranks before failing at the last worker.
        self.assertTrue([c for c in _calls_to(orch, HEAD) if "ray start" in c], "head must have bootstrapped")
        self.assertTrue(
            [c for c in _calls_to(orch, WORKER) if "ray start" in c], "rank-1 worker must have bootstrapped"
        )
        # No serve launched anywhere after the abort.
        self.assertEqual([c for c, _ in orch.calls if "vllm serve" in c], [])


# --------------------------------------------------------------------------- #
# stop_server: ray teardown  (AC18-21)
# --------------------------------------------------------------------------- #
@mock.patch("cvs.lib.inference.vllm_job.time.sleep")
class TestStopServerRayTeardown(unittest.TestCase):
    def test_ray_multinode_broadcasts_single_ray_stop(self, mock_sleep):
        # AC18: exactly one broadcast (hosts=None) ray stop.
        orch = RecordingOrch()
        _job(orch=orch, serve_args=RAY, nnodes="2", pp="1").stop_server()
        ray_stops = [(c, h) for c, h in orch.calls if "ray stop" in c]
        self.assertEqual(len(ray_stops), 1, f"expected one ray stop, got {ray_stops}")
        self.assertIsNone(ray_stops[0][1], "ray stop must be broadcast (hosts=None)")

    def test_pkill_precedes_ray_stop(self, mock_sleep):
        # AC19: pkill vllm serve broadcast comes before ray stop.
        orch = RecordingOrch()
        _job(orch=orch, serve_args=RAY, nnodes="2", pp="1").stop_server()
        pkill_idx = _first_index(orch, lambda c, h: "pkill" in c and "vllm serve" in c)
        ray_idx = _first_index(orch, lambda c, h: "ray stop" in c)
        self.assertNotEqual(pkill_idx, -1, "no pkill vllm serve call recorded")
        self.assertNotEqual(ray_idx, -1, "no ray stop call recorded")
        self.assertLess(pkill_idx, ray_idx)

    def test_mp_multinode_no_ray_stop(self, mock_sleep):
        # AC20 / RC4
        orch = RecordingOrch()
        _job(orch=orch, serve_args={}, nnodes="2", pp="2").stop_server()
        self.assertEqual([c for c, _ in orch.calls if "ray stop" in c], [])

    def test_single_node_ray_no_ray_stop(self, mock_sleep):
        # AC21
        orch = RecordingOrch(hosts=[HEAD])
        _job(orch=orch, serve_args=RAY, nnodes="1", pp="1", ib_netdev=None).stop_server()
        self.assertEqual([c for c, _ in orch.calls if "ray stop" in c], [])


# --------------------------------------------------------------------------- #
# _check_early_failure: Ray worker skip  (AC22, AC23)
# --------------------------------------------------------------------------- #
class TestCheckEarlyFailureRayWorkerSkip(unittest.TestCase):
    def test_ray_worker_is_skipped(self):
        # AC22: ray workers have no per-rank server log -> no tail/grep on worker.
        orch = RecordingOrch(responder=_responder_ok())
        _job(orch=orch, serve_args=RAY, nnodes="2", pp="1")._check_early_failure()
        self.assertEqual(_calls_to(orch, WORKER), [], "ray worker must not be tailed/grepped")
        self.assertTrue(_calls_to(orch, HEAD), "rank 0 head must still be checked")

    def test_mp_worker_is_checked(self):
        # AC23: mp workers DO produce a per-rank log -> rank-1 worker is checked.
        orch = RecordingOrch(responder=_responder_ok())
        _job(orch=orch, serve_args={}, nnodes="2", pp="2")._check_early_failure()
        self.assertTrue(_calls_to(orch, WORKER), "mp rank-1 worker must be tailed/grepped")

    def test_fatal_log_match_raises_with_rank(self):
        # Coverage-gap (finding 10): the FATAL_LOG_RE grep branch (detailed grep
        # returns exit_code 0 with stdout matching FATAL_LOG_RE) raises a
        # RuntimeError "vllm server fatal error". Every prior fixture returned
        # exit_code 1 / no-match for the grep, so this RuntimeError site was never
        # exercised. Single-host job so exactly rank 0 is inspected; the tail
        # returns clean text so the EARLY_FAILURE_RE branch does NOT pre-empt the
        # FATAL_LOG_RE branch under test.
        orch = RecordingOrch(responder=_responder_fatal_grep({HEAD}), hosts=[HEAD])
        job = _job(orch=orch, serve_args={}, nnodes="1", pp="1", ib_netdev=None)
        with self.assertRaises(RuntimeError) as ctx:
            job._check_early_failure()
        msg = str(ctx.exception)
        self.assertIn("vllm server fatal error", msg)
        self.assertIn("rank 0", msg)

    def test_fatal_log_match_on_mp_worker_reports_rank1(self):
        # Companion to finding 10: the FATAL_LOG_RE match on an mp rank-1 worker
        # (which IS inspected, unlike a ray worker) must attribute the fatal error
        # to rank 1 -- confirming the rank is threaded into the message, not
        # hard-coded to 0. Head grep is clean; only the worker's grep matches.
        orch = RecordingOrch(responder=_responder_fatal_grep({WORKER}))
        job = _job(orch=orch, serve_args={}, nnodes="2", pp="2")
        with self.assertRaises(RuntimeError) as ctx:
            job._check_early_failure()
        msg = str(ctx.exception)
        self.assertIn("vllm server fatal error", msg)
        self.assertIn("rank 1", msg)


# --------------------------------------------------------------------------- #
# server_signature  (AC24, AC25)
# --------------------------------------------------------------------------- #
class TestServerSignatureRay(unittest.TestCase):
    def test_hashable_and_stable(self):
        # AC24
        job = _job(serve_args=RAY, nnodes="2", pp="1")
        sig = job.server_signature()
        self.assertEqual(hash(sig), hash(job.server_signature()))
        self.assertEqual(sig, job.server_signature())

    def test_invariant_to_concurrency(self):
        # AC25: concurrency is client-only; two ray jobs differing only in it match.
        self.assertEqual(
            _job(serve_args=RAY, nnodes="2", pp="1", concurrency=4).server_signature(),
            _job(serve_args=RAY, nnodes="2", pp="1", concurrency=64).server_signature(),
        )

    def test_ray_signature_invariant_to_nnodes(self):
        # Round-4 finding 2 / spec Edge Cases (line 299, called out as "intentional"):
        # two ray jobs differing ONLY in nnodes (2 vs 3) must produce EQUAL
        # server_signature() values, because ray's _server_argv(0) never emits
        # --nnodes (ray manages cluster size, not the `vllm serve` command). This
        # is what lets a reused server span differently-sized ray clusters. A
        # regression that leaks nnodes / worker count into the ray argv (e.g. a
        # future _bootstrap change reusing _server_argv) would break server reuse
        # and is caught here -- the mirror of the concurrency-invariance test above.
        self.assertEqual(
            _job(serve_args=RAY, nnodes="2", pp="1").server_signature(),
            _job(serve_args=RAY, nnodes="3", pp="1").server_signature(),
        )

    def test_node_rank_strip_removes_flag_and_value(self):
        # Strengthened (finding 3): the old test only asserted --node-rank absent
        # from a ray signature, which is vacuous (ray argv never contains it, so no
        # implementation could fail it -- redundant with AC16). Here we exercise the
        # strip loop where it CAN fail: an mp multi-node job's _server_argv(0) DOES
        # contain "--node-rank <n>", and server_signature() must remove exactly that
        # flag+value pair (two elements) while leaving every other token intact. A
        # mutant that strips nothing, strips only the flag, or strips extra tokens
        # is caught. The ray no-op (no --node-rank to strip) is also re-asserted.
        mp_job = _job(serve_args={}, nnodes="2", pp="2")
        argv = list(mp_job._server_argv(0))
        self.assertIn("--node-rank", argv)  # precondition: mp argv has it
        i = argv.index("--node-rank")
        expected = argv[:i] + argv[i + 2 :]  # argv minus the flag+value pair
        sig_argv = list(mp_job.server_signature()[0])
        self.assertNotIn("--node-rank", sig_argv)
        self.assertEqual(sig_argv, expected)
        self.assertEqual(len(sig_argv), len(argv) - 2)
        # Ray path: no --node-rank is ever present, so the strip is a documented no-op.
        ray_sig = _job(serve_args=RAY, nnodes="2", pp="1").server_signature()
        self.assertNotIn("--node-rank", ray_sig[0])

    def test_signature_pins_actual_argv_content_not_a_constant(self):
        # Rejects a hard-coded/degenerate server_signature(): the signature must
        # actually contain the job's real server argv (rank-0, --node-rank
        # stripped) and the real env map, not an opaque constant.
        job = _job(serve_args=RAY, nnodes="2", pp="1")
        expected_argv = list(job._server_argv(0))
        if "--node-rank" in expected_argv:
            i = expected_argv.index("--node-rank")
            del expected_argv[i : i + 2]
        expected_env = tuple(sorted((str(k), str(v)) for k, v in job.server_env.items()))
        sig = job.server_signature()
        self.assertEqual(sig, (tuple(expected_argv), expected_env))
        self.assertIn("--tensor-parallel-size", sig[0])
        self.assertIn("--distributed-executor-backend", sig[0])

    def test_signature_env_is_independently_sorted_and_str_cast(self):
        # Round-1 finding 2: the pin test above derives its expected env tuple with
        # the SAME sorted((str(k),str(v)) ...) expression as production, so the env
        # half of that assertion is tautological -- a bug in that exact transform
        # (wrong sort key, missing str() cast, unsorted output) would reproduce in
        # both sides and still pass. Here the expected env is an INDEPENDENTLY
        # hard-coded literal, and server_env is populated with multiple out-of-order
        # keys plus a non-string value, so the assertion actually verifies: (a) keys
        # are sorted, (b) both key and value are str()-cast (the int 3 -> "3"), (c)
        # the result is a tuple of (str, str) pairs. No other test in the suite
        # exercises server_env with >1 entry, so this is the sole real coverage of
        # that transform.
        job = _job(serve_args=RAY, nnodes="2", pp="1")
        # Deliberately out-of-order insertion order; "MID" maps to an int to force str().
        job.server_env = {"ZEBRA": "z1", "ALPHA": "a1", "MID": 3}
        # Independently-constructed literal oracle (not re-derived from server_env).
        expected_env = (("ALPHA", "a1"), ("MID", "3"), ("ZEBRA", "z1"))
        sig = job.server_signature()
        self.assertEqual(sig[1], expected_env)

    def test_differing_tensor_parallelism_yields_different_ray_signature(self):
        # A ray job differing in a server-affecting field (tp) must NOT share a
        # signature with another ray job — otherwise an incompatible server
        # would be wrongly reused across cells.
        self.assertNotEqual(
            _job(serve_args=RAY, nnodes="2", pp="1", tp="4").server_signature(),
            _job(serve_args=RAY, nnodes="2", pp="1", tp="8").server_signature(),
        )

    def test_differing_model_id_yields_different_ray_signature(self):
        # model_id is always in argv regardless of backend (unlike master_addr,
        # which ray legitimately omits per AC16 / _DRIVER_DIST_FLAGS above).
        job_a = _job(serve_args=RAY, nnodes="2", pp="1")
        job_b = _job(serve_args=RAY, nnodes="2", pp="1")
        job_b.model_id = "/models/a-different-model"
        self.assertNotEqual(job_a.server_signature(), job_b.server_signature())


# --------------------------------------------------------------------------- #
# is_ready  (Round-2 finding 2: previously zero direct coverage)
# --------------------------------------------------------------------------- #
class TestVllmJobIsReady(unittest.TestCase):
    """is_ready() greps rank-0's readiness log via orch.exec(detailed=True) and
    returns True iff the collected result is non-empty AND every grepped rank's
    exit_code == 0 (exit 0 = readiness pattern found). rank>0 workers are skipped
    when int(nnodes) > 1 (the pre-existing guard, NOT ray-gated -- spec RC9).
    These tests pin the True path, both False paths (non-zero exit, empty result),
    and the multi-node worker-skip; none of it was exercised before (is_ready is
    never called by the start_server tests)."""

    def test_true_when_readiness_found(self):
        orch = RecordingOrch(responder=_responder_readiness(exit_code=0), hosts=[HEAD])
        job = _job(orch=orch, serve_args={}, nnodes="1", pp="1", ib_netdev=None)
        self.assertTrue(job.is_ready())

    def test_false_when_readiness_absent(self):
        orch = RecordingOrch(responder=_responder_readiness(exit_code=1), hosts=[HEAD])
        job = _job(orch=orch, serve_args={}, nnodes="1", pp="1", ib_netdev=None)
        self.assertFalse(job.is_ready())

    def test_false_when_result_empty(self):
        # `not out` branch: an empty/None exec result must read as NOT ready.
        orch = RecordingOrch(responder=_responder_readiness(empty=True), hosts=[HEAD])
        job = _job(orch=orch, serve_args={}, nnodes="1", pp="1", ib_netdev=None)
        self.assertFalse(job.is_ready())

    def test_multinode_skips_workers_and_only_checks_rank0(self):
        # nnodes=2: only rank 0 (head) is grepped; the rank-1 worker is skipped
        # (rank>0 & nnodes>1 guard), so readiness is decided from rank 0 alone.
        orch = RecordingOrch(responder=_responder_readiness(exit_code=0))
        job = _job(orch=orch, serve_args={}, nnodes="2", pp="2")
        self.assertTrue(job.is_ready())
        self.assertEqual(_calls_to(orch, WORKER), [], "rank>0 worker must be skipped by is_ready under nnodes>1")
        self.assertTrue(_calls_to(orch, HEAD), "rank 0 head readiness log must be grepped")


# --------------------------------------------------------------------------- #
# parse_results  (Round-2 finding 3: previously zero coverage)
# --------------------------------------------------------------------------- #
class TestVllmJobParseResults(unittest.TestCase):
    """parse_results() fetches the client results artifact via orch.exec_on_head
    (which returns {host: content}), json-loads it, and returns
    to_client_metrics(raw, tp=self.tp, isl=self.isl, pp=self.pp) per host. Two
    documented exception modes: empty/missing artifact -> RuntimeError;
    unparseable JSON -> RuntimeError. Exception assertions pin the TYPE only
    (message text is an implementation detail per the authoring anti-patterns).
    The happy path pins the delegation to to_client_metrics with the correct
    keyword-only tp/isl/pp."""

    def test_empty_artifact_raises_runtimeerror(self):
        orch = RecordingOrch(head_responder=lambda cmd: {HEAD: ""}, hosts=[HEAD])
        job = _job(orch=orch, serve_args={}, nnodes="1", pp="1", ib_netdev=None)
        with self.assertRaises(RuntimeError):
            job.parse_results()

    def test_unparseable_json_raises_runtimeerror(self):
        orch = RecordingOrch(head_responder=lambda cmd: {HEAD: "not-json{"}, hosts=[HEAD])
        job = _job(orch=orch, serve_args={}, nnodes="1", pp="1", ib_netdev=None)
        with self.assertRaises(RuntimeError):
            job.parse_results()

    def test_valid_artifact_delegates_to_to_client_metrics_with_tp_isl_pp(self):
        # tp, isl, and pp are keyword-only in to_client_metrics, so they MUST
        # arrive as kwargs; raw (the json-loaded artifact) arrives positionally.
        # Patching the symbol as imported into vllm_job keeps this impl-blind on
        # the metric math.
        #
        # Round-3 finding 1: capture and assert the RETURN VALUE, not just that the
        # mock was called with the right args. Production threads the metric result
        # back out as {host: to_client_metrics(...)}; a mutant that calls
        # to_client_metrics for its side effect but then stores `raw` (or the wrong
        # host key, or returns early) would satisfy a call-args-only check while
        # breaking the actual output. The mock's return_value is the independent
        # oracle for what must appear under the head host key.
        #
        # Post-mortem finding (Spec A1, loop 1): the prior version of this test
        # asserted tp/isl only, so a broken pp passthrough at this call site
        # (e.g. AC6's pp=self.pp regressing to a hardcoded value) would slip
        # through silently. Assert pp explicitly, and vary it across a subTest
        # so a mutant that ignores job.pp entirely is also caught.
        import json as _json

        raw = {"output_throughput": 1234.0, "request_goodput": 10.0}
        sentinel = {"client.sentinel": 1}
        for pp in ("1", "2"):
            with self.subTest(pp=pp):
                orch = RecordingOrch(head_responder=lambda cmd: {HEAD: _json.dumps(raw)}, hosts=[HEAD])
                job = _job(orch=orch, serve_args={}, nnodes="1", pp=pp, ib_netdev=None, isl="1024")
                with mock.patch("cvs.lib.inference.vllm_job.to_client_metrics") as m_tcm:
                    m_tcm.return_value = sentinel
                    result = job.parse_results()
                self.assertTrue(m_tcm.called, "parse_results must delegate to to_client_metrics")
                args, kwargs = m_tcm.call_args
                self.assertEqual(kwargs.get("tp"), job.tp)
                self.assertEqual(kwargs.get("isl"), job.isl)
                self.assertEqual(kwargs.get("pp"), job.pp)
                self.assertEqual(args[0], raw, "raw must be the json-loaded artifact passed positionally")
                # The metric result must be threaded back out under the head host key --
                # NOT the raw artifact, and NOT dropped/re-keyed.
                self.assertEqual(result, {HEAD: sentinel})


# --------------------------------------------------------------------------- #
# wait_ready  (Round-3 finding 2: the readiness polling state machine had zero
# coverage -- it is the real caller of is_ready() and _check_early_failure())
# --------------------------------------------------------------------------- #
@mock.patch("cvs.lib.inference.vllm_job.time.sleep")
class TestVllmJobWaitReady(unittest.TestCase):
    """wait_ready() drives the sequence: precheck-wait -> early-failure-check ->
    warmup-wait -> early-failure-check -> poll-loop(is_ready) -> RuntimeError on
    timeout. is_ready() and _check_early_failure() are unit-tested in isolation
    elsewhere; here they are mocked on the instance so the ORCHESTRATION itself is
    what is exercised: that is_ready is actually polled, that an exhausted poll
    budget raises (not swallowed), that the early-failure check runs before the
    poll loop, and that a failure surfaced during warmup aborts before polling.
    time.sleep is patched at the module seam so no real waiting occurs."""

    def test_returns_when_ready_and_stops_polling(self, mock_sleep):
        # Happy path: is_ready flips True on the 3rd poll; wait_ready must return
        # (no raise) and must stop polling immediately once ready (the side_effect
        # list has no 4th element, so a spurious extra poll raises StopIteration).
        job = _job(serve_args={}, nnodes="1", pp="1", ib_netdev=None)
        job._check_early_failure = mock.Mock()
        job.is_ready = mock.Mock(side_effect=[False, False, True])
        try:
            job.wait_ready()
        except Exception as e:  # pragma: no cover - failure path
            self.fail(f"wait_ready must return once is_ready() is True, raised: {e!r}")
        self.assertEqual(job.is_ready.call_count, 3, "wait_ready must poll is_ready until it returns True, then stop")
        self.assertTrue(job._check_early_failure.called, "wait_ready must run the early-failure check")

    def test_timeout_raises_after_exhausting_poll_budget(self, mock_sleep):
        # Liveness/termination: is_ready never becomes True. wait_ready must NOT
        # spin forever and must NOT swallow the failure -- it raises RuntimeError
        # once the poll budget (server_poll_count) is exhausted, having polled
        # is_ready exactly server_poll_count times.
        # server_poll_count is bound at construction, so set it via the documented
        # constructor parameter (a small budget keeps the test fast and pins the
        # expected poll count without depending on the internal attribute name).
        poll_count = 3
        job = VllmJob(
            orch=RecordingOrch(responder=_responder_ok(), hosts=[HEAD]),
            variant=_variant(serve_args={}, nnodes="1", pp="1", ib_netdev=None),
            hf_token="tok",
            isl="1024",
            osl="1024",
            concurrency=16,
            num_prompts="640",
            server_poll_count=poll_count,
        )
        job._check_early_failure = mock.Mock()
        job.is_ready = mock.Mock(return_value=False)
        with self.assertRaises(RuntimeError):
            job.wait_ready()
        self.assertEqual(
            job.is_ready.call_count,
            poll_count,
            "on timeout wait_ready must have polled is_ready exactly server_poll_count times",
        )

    def test_early_failure_check_runs_before_polling(self, mock_sleep):
        # Ordering: the early-failure check must precede the is_ready poll loop, so a
        # crash detectable in the log is surfaced before spending the poll budget.
        job = _job(serve_args={}, nnodes="1", pp="1", ib_netdev=None)
        order = []
        job._check_early_failure = mock.Mock(side_effect=lambda *a, **k: order.append("check"))
        job.is_ready = mock.Mock(side_effect=lambda: (order.append("ready"), True)[1])
        job.wait_ready()
        self.assertIn("check", order, "the early-failure check must be invoked")
        self.assertIn("ready", order, "is_ready must be polled")
        self.assertEqual(order[0], "check", "early-failure check must run before the first is_ready poll")

    def test_failure_detected_during_warmup_aborts_before_polling(self, mock_sleep):
        # If _check_early_failure raises (a fatal log line found during precheck/
        # warmup), wait_ready must propagate it and NOT proceed to poll is_ready --
        # the server is already known dead.
        job = _job(serve_args={}, nnodes="1", pp="1", ib_netdev=None)
        job._check_early_failure = mock.Mock(side_effect=RuntimeError("vllm server fatal error (rank 0)"))
        job.is_ready = mock.Mock(return_value=True)
        with self.assertRaises(RuntimeError):
            job.wait_ready()
        self.assertFalse(
            job.is_ready.called,
            "a failure surfaced by _check_early_failure must abort wait_ready before the poll loop",
        )


# --------------------------------------------------------------------------- #
# build_server_cmd  (Round-3 finding 3: env-script construction had zero direct
# coverage -- multiple conditional branches forming clear equivalence classes)
# --------------------------------------------------------------------------- #
class TestVllmJobBuildServerCmd(unittest.TestCase):
    """build_server_cmd() writes an env-script (broadcast to all nodes) and issues
    per-rank mkdir commands. Its documented equivalence classes (per discipline
    rule B, driven by subTest tables): ib_hcas present vs empty/None (the
    NCCL_IB_HCA line is emitted or not), ib_netdev present vs None (the socket
    interface exports are emitted or not), the server_env pass-through loop, and the
    per-rank mkdir loop bounded by nnodes. Assertions target stable, named tokens
    (NCCL_IB_HCA, SOCKET_IFNAME, the env keys, mkdir) rather than exact shell
    formatting, and are collected from every command the method emits (whether
    issued via orch.exec or returned) so the test does not couple to the transport
    detail of how the script is delivered."""

    @staticmethod
    def _script(orch, ret):
        parts = list(_all_cmds(orch))
        if isinstance(ret, str):
            parts.append(ret)
        elif isinstance(ret, (list, tuple)):
            parts.extend(str(x) for x in ret)
        return "\n".join(parts)

    def test_nccl_ib_hca_line_present_only_when_ib_hcas_supplied(self):
        # (ib_hcas, hca_present) -- the NCCL_IB_HCA export is gated on a non-empty
        # ib_hcas list; empty list and None must NOT emit it.
        cases = [
            (["mlx5_0", "mlx5_1"], True),
            ([], False),
            (None, False),
        ]
        for ib_hcas, present in cases:
            with self.subTest(ib_hcas=ib_hcas):
                orch = RecordingOrch()
                job = _job(orch=orch, serve_args={}, nnodes="2", pp="2", ib_hcas=ib_hcas)
                ret = job.build_server_cmd()
                script = self._script(orch, ret)
                if present:
                    self.assertIn("NCCL_IB_HCA", script)
                    # The supplied HCA name must actually reach the export value.
                    self.assertIn("mlx5_0", script)
                else:
                    self.assertNotIn("NCCL_IB_HCA", script)

    def test_socket_ifname_exports_present_only_when_ib_netdev_set(self):
        # ib_netdev set -> the socket-interface exports are emitted (all three name
        # the device); ib_netdev None -> none are emitted.
        orch_set = RecordingOrch()
        _job(orch=orch_set, serve_args={}, nnodes="2", pp="2", ib_netdev="eth0").build_server_cmd()
        script_set = self._script(orch_set, None)
        self.assertEqual(
            script_set.count("SOCKET_IFNAME"),
            3,
            "ib_netdev must emit exactly the three socket-ifname exports",
        )
        self.assertIn("eth0", script_set, "the configured ib_netdev must reach the export value")

        orch_none = RecordingOrch()
        _job(orch=orch_none, serve_args={}, nnodes="2", pp="2", ib_netdev=None).build_server_cmd()
        script_none = self._script(orch_none, None)
        self.assertNotIn("SOCKET_IFNAME", script_none, "no ib_netdev -> no socket-ifname exports")

    def test_server_env_entries_passed_through(self):
        # Every server_env key/value must appear in the emitted env-script (the
        # pass-through loop). Two entries so a single-entry short-circuit is caught.
        #
        # Round-4 finding 1: the values MUST be distinctive strings that cannot
        # collide with any boilerplate line the env-script also emits. A bare value
        # like "1" trivially matches elsewhere (e.g. "...AITER_UNIFIED_ATTENTION=1"),
        # so assertIn("1", script) is vacuous -- a mutant that hard-codes a wrong
        # value or drops the CUSTOM_A line entirely still passes. Using unique
        # values AND asserting the "KEY=VALUE" pairing (not the bare value) pins
        # both the presence and the key/value association without coupling to the
        # exact "export " prefix formatting.
        orch = RecordingOrch()
        job = _job(
            orch=orch,
            serve_args={},
            nnodes="2",
            pp="2",
            env={"CUSTOM_A": "CUSTOM_A_VALUE_XYZ", "CUSTOM_B": "CUSTOM_B_VALUE_QRS"},
        )
        ret = job.build_server_cmd()
        script = self._script(orch, ret)
        for key, val in (("CUSTOM_A", "CUSTOM_A_VALUE_XYZ"), ("CUSTOM_B", "CUSTOM_B_VALUE_QRS")):
            with self.subTest(key=key):
                self.assertIn(f"{key}={val}", script, f"server_env {key} must be exported paired with its value")

    def test_mkdir_count_scales_with_nnodes(self):
        # The per-rank mkdir loop is bounded by nnodes: a 3-node job must issue
        # strictly more mkdir commands than a single-node job (all else equal).
        orch1 = RecordingOrch(hosts=[HEAD])
        _job(orch=orch1, serve_args={}, nnodes="1", pp="1", ib_netdev=None).build_server_cmd()
        orch3 = RecordingOrch(hosts=[HEAD, WORKER, HOST2])
        _job(orch=orch3, serve_args={}, nnodes="3", pp="1", ib_netdev="eth0").build_server_cmd()
        mk1 = self._script(orch1, None).count("mkdir")
        mk3 = self._script(orch3, None).count("mkdir")
        self.assertGreater(mk1, 0, "build_server_cmd must create at least the rank-0 log dir")
        self.assertGreater(mk3, mk1, "per-rank mkdir loop must scale with nnodes")


# --------------------------------------------------------------------------- #
# Lifecycle  (transition table)
# --------------------------------------------------------------------------- #
# | from state           | event                     | to state / effect                     |
# |----------------------|---------------------------|---------------------------------------|
# | constructed          | start_server() [ok ray]   | bootstrap head+worker, serve on head  |
# | started              | start_server() again      | re-entrant: no raise, re-launch serve |
# | started              | stop_server()             | pkill + ray stop broadcast -> down    |
# | bootstrap-failed     | start_server() [head bad] | RuntimeError, no serve (illegal txn)  |
# | bootstrap-failed     | stop_server()             | must NOT raise (partial cleanup)      |
# | down                 | stop_server() again       | idempotent no-op, no raise            |
@mock.patch("cvs.lib.inference.vllm_job.time.sleep")
class TestVllmJobRayLifecycle(unittest.TestCase):
    def test_legal_start_then_stop(self, mock_sleep):
        orch = RecordingOrch(responder=_responder_ok())
        job = _job(orch=orch, serve_args=RAY, nnodes="2", pp="1")
        job.start_server()  # constructed -> started
        job.stop_server()  # started -> down
        self.assertTrue([c for c in _calls_to(orch, HEAD) if "vllm serve" in c])
        self.assertTrue([c for c, _ in orch.calls if "ray stop" in c])

    def test_illegal_start_on_bad_bootstrap_is_rejected(self, mock_sleep):
        orch = RecordingOrch(responder=_responder_bootstrap_fail({HEAD: {"exit_code": 1, "output": _BAD}}))
        job = _job(orch=orch, serve_args=RAY, nnodes="2", pp="1")
        with self.assertRaises(RuntimeError):
            job.start_server()

    def test_stop_after_failed_start_does_not_raise(self, mock_sleep):
        # Regression: partial bootstrap -> caller must be able to stop_server safely.
        orch = RecordingOrch(responder=_responder_bootstrap_fail({HEAD: {"exit_code": 1, "output": _BAD}}))
        job = _job(orch=orch, serve_args=RAY, nnodes="2", pp="1")
        with self.assertRaises(RuntimeError):
            job.start_server()
        calls_before_stop = len(orch.calls)
        try:
            job.stop_server()  # must be robust after a failed/partial start
        except Exception as e:  # pragma: no cover - failure path
            self.fail(f"stop_server after failed start raised: {e!r}")
        # Finding 9: not merely "no exception" -- assert stop_server actually did
        # the full teardown after the partial start. Both broadcasts (hosts=None)
        # must be issued: the pkill vllm serve, then (nnodes>1 & ray) ray stop.
        stop_calls = orch.calls[calls_before_stop:]
        pkill = [(c, h) for c, h in stop_calls if "pkill" in c and "vllm serve" in c]
        ray_stop = [(c, h) for c, h in stop_calls if "ray stop" in c]
        self.assertEqual(len(pkill), 1, f"expected one pkill broadcast on stop, got {pkill}")
        self.assertIsNone(pkill[0][1], "pkill must be a broadcast (hosts=None)")
        self.assertEqual(len(ray_stop), 1, f"expected one ray stop broadcast on stop, got {ray_stop}")
        self.assertIsNone(ray_stop[0][1], "ray stop must be a broadcast (hosts=None)")

    def test_reentrant_start_is_not_rejected(self, mock_sleep):
        # Coverage-gap (finding 2): the transition table had a legal start and an
        # idempotent stop-reentry, but no started -> start_server()-again case. The
        # spec documents no idempotency guard / no documented error on re-entrant
        # start, so a second start_server() must not raise and re-runs the bootstrap
        # + serve sequence (mirroring the re-run semantics of the stop reentry test:
        # each teardown issues its own ray stop, so each start issues its own serve).
        orch = RecordingOrch(responder=_responder_ok())
        job = _job(orch=orch, serve_args=RAY, nnodes="2", pp="1")
        job.start_server()  # constructed -> started
        try:
            job.start_server()  # started -> start again (re-entrant)
        except Exception as e:  # pragma: no cover - failure path
            self.fail(f"re-entrant start_server raised: {e!r}")
        head_serves = [c for c in _calls_to(orch, HEAD) if "vllm serve" in c]
        self.assertEqual(
            len(head_serves), 2, f"each start_server must (re-)launch vllm serve on the head; got {head_serves}"
        )

    def test_idempotent_stop_reentry(self, mock_sleep):
        orch = RecordingOrch(responder=_responder_ok())
        job = _job(orch=orch, serve_args=RAY, nnodes="2", pp="1")
        job.start_server()
        job.stop_server()
        try:
            job.stop_server()  # cleanup twice must not raise
        except Exception as e:  # pragma: no cover - failure path
            self.fail(f"second stop_server raised: {e!r}")
        # Each teardown issues its own ray stop broadcast.
        self.assertEqual(len([c for c, _ in orch.calls if "ray stop" in c]), 2)


if __name__ == "__main__":
    unittest.main()
