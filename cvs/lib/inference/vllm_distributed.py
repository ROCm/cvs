'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Multinode vLLM distributed job driven by a ContainerOrchestrator.

Runs vLLM with --distributed-executor-backend mp across N nodes.
Server launches on every node with per-rank --node-rank flag.
Client runs exclusively on the head node (rank 0).

Key distributed-vs-single routing contract:
  - build_server_cmd: env-script write + mkdir BROADCAST to ALL nodes
    (orch.exec with no hosts kwarg / hosts=None)
  - start_server: one targeted orch.exec(..., hosts=[host]) per host,
    with per-rank --node-rank
  - run_client / wait_client_complete / parse_results: head-only via
    orch.exec_on_head (never broadcast)
  - wait_ready / is_ready: BROADCAST orch.exec so every shard is checked
  - stop_server: pkill BROADCAST to all nodes
'''

from __future__ import annotations

import json
import math
import re
import shlex
import time

from cvs.lib import globals
from cvs.lib.inference.utils.vllm_parsing import to_client_metrics

log = globals.log


class VllmDistributedJob:
    """Multinode vLLM benchmark job driven by an injected ContainerOrchestrator.

    All container/SSH plumbing belongs to `orch`. This class composes the
    server-env script, launches the server on EVERY node in the background
    (each with its per-rank --node-rank), polls ALL nodes until ready, runs
    the bench_serving client on the HEAD node only, and parses the resulting
    log (head-only).

    The `orch.hosts` list provides the enumeration: `enumerate(orch.hosts)`
    yields (rank, host) pairs for per-host rank dispatch.
    """

    READINESS_RE = re.compile(r"Application startup complete|Uvicorn running|Started server", re.I)
    COMPLETION_RE = re.compile(r"Serving Benchmark Result", re.I)
    FAILED_REQUESTS_RE = re.compile(r"Failed requests:\s+([0-9]+)", re.I)
    CLIENT_CRASH_RE = re.compile(r"Traceback \(most recent call last\)", re.I)
    CLIENT_LAUNCH_FAIL_RE = re.compile(
        r"unrecognized arguments|invalid choice|error: argument |command not found|: No such file or directory",
        re.I,
    )
    EARLY_FAILURE_RE = re.compile(
        r"no such file or directory|command not found|cannot access|failed to start"
        r"|Free memory on device.*less than desired"
        r"|Engine core initialization failed"
        r"|WorkerProc failed to start",
        re.I,
    )
    # Pattern checked against full NFS log after warmup to catch hard failures.
    FATAL_LOG_RE = re.compile(
        r"Free memory on device.{0,80}less than desired"
        r"|Engine core initialization failed"
        r"|RuntimeError:.*[Ee]ngine",
        re.I,
    )

    def __init__(
        self,
        orch,
        variant,
        hf_token,
        isl,
        osl,
        concurrency,
        num_prompts,
        goodput_slo=None,
        log_subdir="vllm",
        server_precheck_wait_s=30,
        server_warmup_wait_s=330,
        server_poll_count=60,
        server_poll_wait_s=60,
        client_initial_wait_s=120,
        client_poll_count=20,
        client_poll_wait_s=60,
    ):
        self.orch = orch
        self.variant = variant
        self.hf_token = hf_token
        self.isl = str(isl)
        self.osl = str(osl)
        self.concurrency = str(concurrency)
        self.num_prompts = str(num_prompts)
        self.goodput_slo = goodput_slo
        self.log_subdir = log_subdir

        p = variant.params
        self.tp = p.tensor_parallelism
        self.pp = p.pipeline_parallel_size
        self.master_addr = p.master_addr
        self.master_port = p.master_port
        self.nnodes = p.nnodes
        self.port_no = p.port_no
        self.random_range_ratio = p.random_range_ratio
        self.random_prefix_len = p.random_prefix_len
        self.burstiness = p.burstiness
        self.seed = p.seed
        self.request_rate = p.request_rate
        self.tokenizer_mode = p.tokenizer_mode
        self.percentile_metrics = p.percentile_metrics
        self.metric_percentiles = p.metric_percentiles
        self.base_url = p.base_url
        self.dataset_name = p.dataset_name
        self.backend = p.backend

        self.model_id = variant.model.id
        self.log_dir = variant.paths.log_dir
        self.serve_args = dict(variant.roles.server.serve_args)
        self.server_env = dict(variant.roles.server.env)
        self.models_dir = variant.paths.models_dir

        # Per-cell output directory, keyed by (isl/osl/conc).
        self.out_dir = (
            f"{self.log_dir}/{self.log_subdir}/out-node0"
            f"/isl{self.isl}_osl{self.osl}_conc{self.concurrency}"
        )
        self.server_log = f"{self.out_dir}/vllm_serve_server.log"
        self.client_log = f"{self.out_dir}/client.log"

        self._precheck_wait = server_precheck_wait_s
        self._warmup_wait = server_warmup_wait_s
        self._server_poll_count = server_poll_count
        self._server_poll_wait = server_poll_wait_s
        self._client_initial_wait = client_initial_wait_s
        self._client_poll_count = client_poll_count
        self._client_poll_wait = client_poll_wait_s

    # ---------- derived builders ----------

    _MML_PAD = 8

    def _derive_max_model_len(self):
        r = float(self.random_range_ratio)
        worst = (int(self.isl) + int(self.osl)) * (1.0 + r)
        return str(math.ceil(worst) + int(self.random_prefix_len) + self._MML_PAD)

    @staticmethod
    def _flatten_serve_args(mapping):
        """A {flag: value} serve-args map -> a flat `vllm serve` arg list."""
        argv = []
        for flag, value in mapping.items():
            opt = f"--{flag}"
            if value is True:
                argv.append(opt)
            elif isinstance(value, (list, tuple)):
                for v in value:
                    argv.extend([opt, str(v)])
            else:
                argv.extend([opt, str(value)])
        return argv

    def _server_argv(self, rank):
        """The `vllm serve` arg list for a specific node rank.

        All distributed flags are added: --node-rank, --master-addr,
        --master-port, --pipeline-parallel-size, --nnodes, and
        --distributed-executor-backend mp.  Only --node-rank varies per rank;
        all other flags are rank-invariant.
        """
        argv = [
            "vllm",
            "serve",
            self.model_id,
            "--tensor-parallel-size",
            str(self.tp),
            "--pipeline-parallel-size",
            str(self.pp),
            "--max-model-len",
            self._derive_max_model_len(),
            "--port",
            str(self.port_no),
            "--node-rank",
            str(rank),
            "--master-addr",
            str(self.master_addr),
            "--master-port",
            str(self.master_port),
            "--nnodes",
            str(self.nnodes),
            "--distributed-executor-backend",
            "mp",
        ]
        argv.extend(self._flatten_serve_args(self.serve_args))
        return argv

    # ---------- server side ----------

    def _rank_log(self, rank):
        """Return the server log path for a specific rank (avoids shared-file clobber)."""
        return self.server_log.replace("out-node0", f"out-node{rank}")

    def build_server_cmd(self):
        """Write the server-env script and create the per-node out-dirs.

        Both are BROADCAST to ALL nodes (no hosts kwarg) so every rank has the
        env script and out-dir before any per-host server launch.
        """
        env_lines = [
            f"export HF_TOKEN={shlex.quote(self.hf_token)}",
            f"export HF_HUB_CACHE={shlex.quote(self.models_dir)}",
            "export VLLM_USE_AITER_UNIFIED_ATTENTION=1",
            "export VLLM_ROCM_USE_AITER_MHA=0",
            "export VLLM_ROCM_USE_AITER_FUSED_MOE_A16W4=1",
        ]
        for k, v in self.server_env.items():
            env_lines.append(f"export {k}={shlex.quote(str(v))}")
        env_script = "\n".join(env_lines) + "\n"
        # Broadcast: no hosts kwarg -> all nodes get the env script.
        self.orch.exec(
            "bash -c " + shlex.quote(
                f"printf '%s' {shlex.quote(env_script)} > /tmp/server_env_script.sh"
            )
        )
        # Create a per-rank out-dir for every node so logs never share a path.
        for rank in range(int(self.nnodes)):
            rank_dir = self.out_dir.replace("out-node0", f"out-node{rank}")
            self.orch.exec(f"mkdir -p {shlex.quote(rank_dir)}")

        # Patch 0: Drop stale .pyc files so Python recompiles from the .py source
        # that is already present in the Docker image. Two known-stale files:
        #   multiproc_executor.pyc — compiled before the fix that returns [] instead
        #     of asserting rpc_broadcast_mq is not None on follower nodes.
        #   core.pyc — compiled before the follower guards added in Patches 1-3.
        # Python skips recompilation when the .pyc mtime/hash matches the .py,
        # so an in-place image update leaves stale .pyc files silently in use.
        _stale_pycs = [
            (
                "/opt/python/lib/python3.12/site-packages/vllm/v1/executor/"
                "__pycache__/multiproc_executor.cpython-312.pyc"
            ),
            (
                "/opt/python/lib/python3.12/site-packages/vllm/v1/engine/"
                "__pycache__/core.cpython-312.pyc"
            ),
        ]
        for _pyc in _stale_pycs:
            self.orch.exec(f"rm -f {shlex.quote(_pyc)}")
        log.info("vllm stale pycs removed (multiproc_executor, core)")

        # Patch 0b: Fix collective_rpc assert in multiproc_executor.py.
        # The image's .py has `assert self.rpc_broadcast_mq is not None` which
        # fires on follower nodes. Replace with a guard that returns [] (or None
        # for single-return calls) so followers silently skip the broadcast.
        _mpexec_script = "\n".join([
            "import pathlib",
            "p = pathlib.Path('/opt/python/lib/python3.12/site-packages/vllm/v1/executor/multiproc_executor.py')",
            "src = p.read_text()",
            "if 'if self.rpc_broadcast_mq is None' in src:",
            "    print('ALREADY_PATCHED')",
            "else:",
            "    old = ('        assert self.rpc_broadcast_mq is not None, (\\n'",
            "           '            \"collective_rpc should not be called on follower node\"\\n'",
            "           '        )')",
            "    new = ('        if self.rpc_broadcast_mq is None:\\n'",
            "           '            return None if (unique_reply_rank is not None or kv_output_aggregator is not None) else []')",
            "    if old in src:",
            "        p.write_text(src.replace(old, new, 1))",
            "        print('PATCHED')",
            "    else:",
            "        print('NOT_FOUND')",
        ]) + "\n"
        self.orch.exec(
            "bash -c " + shlex.quote(
                f"printf '%s' {shlex.quote(_mpexec_script)} > /tmp/vllm_patch0b.py"
            )
        )
        patch_out0b = self.orch.exec("python3 /tmp/vllm_patch0b.py")
        for host, out in (patch_out0b or {}).items():
            log.info("vllm multiproc_executor.py patch0b on %s: %s", host, (out or "").strip())

        # Patch 1: Guard _initialize_kv_caches for follower nodes (node_rank > 0).
        # collective_rpc requires rpc_broadcast_mq which is None on followers;
        # skip to a dummy KVCacheConfig so init can proceed.
        _patch1_script = "\n".join([
            "import pathlib",
            "p = pathlib.Path('/opt/python/lib/python3.12/site-packages/vllm/v1/engine/core.py')",
            "src = p.read_text()",
            "old = '        kv_cache_config = self._initialize_kv_caches(vllm_config)\\n'",
            "new = (",
            "    '        if vllm_config.parallel_config.node_rank_within_dp == 0:\\n'",
            "    '            kv_cache_config = self._initialize_kv_caches(vllm_config)\\n'",
            "    '        else:\\n'",
            "    '            vllm_config.cache_config.num_gpu_blocks = 1\\n'",
            "    '            kv_cache_config = KVCacheConfig(num_blocks=1, kv_cache_tensors=[], kv_cache_groups=[])\\n'",
            ")",
            "already = 'vllm_config.cache_config.num_gpu_blocks = 1'",
            "if already in src:",
            "    print('ALREADY_PATCHED')",
            "elif old in src:",
            "    p.write_text(src.replace(old, new, 1))",
            "    print('PATCHED')",
            "else:",
            "    print('NOT_FOUND')",
        ]) + "\n"
        self.orch.exec("bash -c " + shlex.quote(
            f"printf '%s' {shlex.quote(_patch1_script)} > /tmp/vllm_patch1.py"
        ))
        patch_out = self.orch.exec("python3 /tmp/vllm_patch1.py")
        for host, out in (patch_out or {}).items():
            log.info("vllm core.py patch1 on %s: %s", host, (out or "").strip())

        # Patch 2: Guard Scheduler() creation for follower nodes.
        # Scheduler.__init__ → KVCacheManager → HybridKVCacheCoordinator asserts
        # len(attention_groups) > 1, but followers have kv_cache_groups=[].
        # Stub it out with _F: follower EngineCore only needs workers running.
        _patch2_script = "\n".join([
            "import pathlib",
            "p = pathlib.Path('/opt/python/lib/python3.12/site-packages/vllm/v1/engine/core.py')",
            "src = p.read_text()",
            "old = '        self.scheduler: SchedulerInterface = Scheduler(\\n'",
            "new = (",
            "    '        if vllm_config.parallel_config.node_rank_within_dp != 0:\\n'",
            "    '            class _F:\\n'",
            "    '                connector = None\\n'",
            "    '                def get_kv_connector(self): return None\\n'",
            "    '                def __getattr__(self, n): return lambda *a, **k: None\\n'",
            "    '            self.scheduler = _F()\\n'",
            "    '        else:\\n'",
            "    '            self.scheduler: SchedulerInterface = Scheduler(\\n'",
            ")",
            "already = 'class _F:'",
            "if already in src:",
            "    print('ALREADY_PATCHED')",
            "elif old in src:",
            "    p.write_text(src.replace(old, new, 1))",
            "    print('PATCHED')",
            "else:",
            "    print('NOT_FOUND')",
        ]) + "\n"
        self.orch.exec("bash -c " + shlex.quote(
            f"printf '%s' {shlex.quote(_patch2_script)} > /tmp/vllm_patch2.py"
        ))
        patch_out2 = self.orch.exec("python3 /tmp/vllm_patch2.py")
        for host, out in (patch_out2 or {}).items():
            log.info("vllm core.py patch2 on %s: %s", host, (out or "").strip())

        # Patch 3: Ensure get_supported_tasks returns ("generate",) for follower
        # nodes instead of calling collective_rpc which returns [] on followers
        # (causing IndexError in abstract.py:supported_tasks → output[0]).
        # Two image variants are handled:
        #   A) Image has guard but uses SupportedTask.GENERATE (Literal, not Enum)
        #      → replace the bad return value.
        #   B) Image has bare form (no guard at all) → insert the guard block.
        _patch3_script = "\n".join([
            "import pathlib",
            "p = pathlib.Path('/opt/python/lib/python3.12/site-packages/vllm/v1/engine/core.py')",
            "src = p.read_text()",
            "case_a_old = '            return (SupportedTask.GENERATE,)'",
            "case_b_old = '    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:\\n        return self.model_executor.supported_tasks\\n'",
            "case_b_new = ('    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:\\n'",
            "              '        if self.vllm_config.parallel_config.node_rank_within_dp != 0:\\n'",
            "              '            return (\"generate\",)\\n'",
            "              '        return self.model_executor.supported_tasks\\n')",
            "if '\"generate\"' in src and 'node_rank_within_dp != 0' in src:",
            "    print('ALREADY_PATCHED')",
            "elif case_a_old in src:",
            "    p.write_text(src.replace(case_a_old, '            return (\"generate\",)', 1))",
            "    print('PATCHED_A')",
            "elif case_b_old in src:",
            "    p.write_text(src.replace(case_b_old, case_b_new, 1))",
            "    print('PATCHED_B')",
            "else:",
            "    idx = src.find('def get_supported_tasks')",
            "    print('NOT_FOUND ctx:', src[idx:idx+150] if idx != -1 else 'fn absent')",
        ]) + "\n"
        self.orch.exec(
            "bash -c " + shlex.quote(
                f"printf '%s' {shlex.quote(_patch3_script)} > /tmp/vllm_patch3.py"
            )
        )
        patch_out3 = self.orch.exec("python3 /tmp/vllm_patch3.py")
        for host, out in (patch_out3 or {}).items():
            log.info("vllm core.py patch3 on %s: %s", host, (out or "").strip())

    def start_server(self):
        """Launch vllm serve on each host with the correct --node-rank.

        Iterates enumerate(orch.hosts) -> (rank, host), issues one targeted
        orch.exec(..., hosts=[host]) per host. Raises on early failure.
        """
        for rank, host in enumerate(self.orch.hosts):
            serve_cmd = " ".join(shlex.quote(str(a)) for a in self._server_argv(rank))
            rank_log = self._rank_log(rank)
            inner = (
                f"source /tmp/server_env_script.sh && "
                f"nohup {serve_cmd} > {shlex.quote(rank_log)} 2>&1 &"
            )
            out = self.orch.exec("bash -c " + shlex.quote(inner), hosts=[host])
            for h, output in out.items():
                if self.EARLY_FAILURE_RE.search(output or ""):
                    raise RuntimeError(
                        f"vllm server failed to launch on {h} (rank {rank}): {output[-500:]}"
                    )

    def is_ready(self):
        """Broadcast grep to ALL nodes; ready only when every node's exit_code == 0."""
        pattern = self.READINESS_RE.pattern
        out = self.orch.exec(
            f"grep -qiE {shlex.quote(pattern)} {shlex.quote(self.server_log)}",
            detailed=True,
        )
        return bool(out) and all(r["exit_code"] == 0 for r in out.values())

    def wait_ready(self):
        log.info("waiting %ds for server log to materialise", self._precheck_wait)
        time.sleep(self._precheck_wait)

        # Broadcast tail to ALL nodes to detect early failures on any shard.
        out = self.orch.exec(f"tail -30 {shlex.quote(self.server_log)}")
        for host, output in out.items():
            if self.EARLY_FAILURE_RE.search(output or ""):
                raise RuntimeError(f"vllm server early failure on {host}: {output[-500:]}")

        log.info("warmup wait %ds", self._warmup_wait)
        time.sleep(self._warmup_wait)

        # After warmup, check for hard fatal signatures before polling.
        out = self.orch.exec(
            f"grep -m1 -iE {shlex.quote(self.FATAL_LOG_RE.pattern)} {shlex.quote(self.server_log)}",
            detailed=True,
        )
        for host, r in (out or {}).items():
            if r.get("exit_code") == 0 and r.get("stdout", "").strip():
                raise RuntimeError(
                    f"vllm server fatal error on {host}: {r['stdout'].strip()[-500:]}"
                )

        for it in range(self._server_poll_count):
            if self.is_ready():
                log.info("server ready (iter=%d)", it)
                return
            time.sleep(self._server_poll_wait)
        raise RuntimeError("vllm server did not become ready before timeout")

    def stop_server(self):
        """Broadcast pkill to ALL nodes so no stray shard lingers."""
        log.info("stopping vllm server")
        self.orch.exec("bash -c 'pkill -f \"vllm serve\" || true'")
        time.sleep(5)

    # ---------- client side (head-only) ----------

    def run_client(self):
        """Launch bench serve on the HEAD node only via exec_on_head."""
        args = [
            "vllm",
            "bench",
            "serve",
            "--model",
            self.model_id,
            "--backend",
            self.backend,
            "--base-url",
            f"{self.base_url}:{self.port_no}",
            "--dataset-name",
            self.dataset_name,
            "--num-prompts",
            self.num_prompts,
            "--random-input-len",
            self.isl,
            "--random-output-len",
            self.osl,
            "--max-concurrency",
            self.concurrency,
            "--request-rate",
            self.request_rate,
            "--burstiness",
            self.burstiness,
            "--tokenizer-mode",
            self.tokenizer_mode,
            "--seed",
            self.seed,
            "--random-range-ratio",
            self.random_range_ratio,
            "--random-prefix-len",
            self.random_prefix_len,
            "--percentile-metrics",
            self.percentile_metrics,
            "--metric-percentiles",
            self.metric_percentiles,
            "--ignore-eos",
            "--save-result",
            "--result-dir",
            self.out_dir,
            "--result-filename",
            "results",
        ]
        if self.goodput_slo:
            args.append("--goodput")
            for metric, key in (("ttft", "ttft_ms"), ("tpot", "tpot_ms"), ("e2el", "e2el_ms")):
                val = self.goodput_slo.get(key)
                if val is not None:
                    args.append(f"{metric}:{val}")
        bench_cmd = " ".join(shlex.quote(str(a)) for a in args)
        client_cmd = (
            f"source /tmp/server_env_script.sh && "
            f"{bench_cmd} > {shlex.quote(self.client_log)} 2>&1 &"
        )
        self.orch.exec_on_head("bash -c " + shlex.quote(client_cmd))

    def wait_client_complete(self):
        """Poll the client log on the HEAD node only via exec_on_head."""
        log.info("client initial wait %ds", self._client_initial_wait)
        time.sleep(self._client_initial_wait)
        for it in range(self._client_poll_count):
            out = self.orch.exec_on_head(f"tail -2000 {shlex.quote(self.client_log)}")
            failed = []
            done = []
            for host, output in out.items():
                txt = output or ""
                done.append(bool(self.COMPLETION_RE.search(txt)))
                if self.CLIENT_CRASH_RE.search(txt) or self.CLIENT_LAUNCH_FAIL_RE.search(txt):
                    failed.append((host, txt[-500:]))
                else:
                    fm = self.FAILED_REQUESTS_RE.search(txt)
                    if fm and int(fm.group(1)) > 0:
                        failed.append((host, f"Failed requests: {fm.group(1)} -- {txt[-500:]}"))
            if failed:
                raise RuntimeError("client failed: " + "; ".join(f"{h}: {m}" for h, m in failed))
            if done and all(done):
                log.info("client complete (iter=%d)", it)
                return
            time.sleep(self._client_poll_wait)
        raise RuntimeError("client did not complete before poll cap")

    def parse_results(self):
        """Fetch and parse the results artifact from the HEAD node via exec_on_head."""
        artifact = f"{self.out_dir}/results"
        out = self.orch.exec_on_head(f"cat {shlex.quote(artifact)}")
        results = {}
        for host, text in out.items():
            text = (text or "").strip()
            if not text:
                raise RuntimeError(f"empty/missing results artifact on {host}: {artifact}")
            try:
                raw = json.loads(text)
            except (json.JSONDecodeError, ValueError) as e:
                raise RuntimeError(
                    f"unparseable results artifact on {host}: {artifact}: {e}"
                ) from e
            results[host] = to_client_metrics(raw, tp=self.tp, isl=self.isl)
        return results
