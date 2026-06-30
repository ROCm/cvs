'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unified vLLM benchmark job for single-node and multinode distributed runs.

Routing contract:
  - build_server_cmd: BROADCAST env-script write + mkdir to ALL nodes.
    On single-node (nnodes=1) broadcast and targeted exec are equivalent.
  - start_server: one targeted orch.exec(..., hosts=[host]) per host, with
    per-rank --node-rank. On single-node this yields one (0, head) iteration.
  - run_client / wait_client_complete / parse_results: HEAD-ONLY via
    orch.exec_on_head. Semantically correct for both topologies: the client
    always connects to http://head:port. Using broadcast exec here would
    launch N competing clients on multinode.
  - wait_ready / is_ready: BROADCAST so every shard is checked.
  - stop_server: BROADCAST pkill to all nodes.

Distributed vs single-node branching is localised to _server_argv only:
distributed flags (--node-rank, --master-addr, --master-port, --nnodes,
--pipeline-parallel-size, --distributed-executor-backend) are added iff
int(nnodes) > 1. Everything else is topology-blind.

IB device config (distributed only):
  ib_hcas: discovered HCA names for NCCL_IB_HCA, passed in from the
      test_discover_topology lifecycle step. Written into the per-node env
      script.
  ib_netdev: explicit Linux netdev name for NCCL_SOCKET_IFNAME /
      GLOO_SOCKET_IFNAME. Read directly from variant.roles.server.ib_netdev.
      Required when nnodes > 1 (enforced by VariantConfig validator).
'''

from __future__ import annotations

import json
import math
import re
import shlex
import time
from typing import Optional

from cvs.lib import globals
from cvs.lib.inference.utils.vllm_parsing import to_client_metrics

log = globals.log


class VllmJob:
    """Unified vLLM benchmark job for single-node and multinode distributed runs.

    Construct with the result of test_discover_topology (ib_hcas) for distributed
    runs; pass None (or omit) for single-node.

    The ``orch`` instance is expected to already have ``setup_containers()`` and
    (for multinode) ``setup_sshd()`` called against it by the test lifecycle.
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
        ib_hcas: Optional[list] = None,
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
        # Discovered HCA names for NCCL_IB_HCA (multinode only). Passed in from
        # test_discover_topology so discovery runs once per lifecycle, not per cell.
        self.ib_hcas = ib_hcas or []
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
        self.ib_netdev = variant.roles.server.ib_netdev

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
        """Convert {flag: value} serve-args map to a flat vllm serve arg list."""
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

    def _server_argv(self, rank: int) -> list:
        """vllm serve arg list for a specific node rank.

        Distributed flags added iff nnodes > 1. On single-node (nnodes=1)
        this yields a plain single-node vllm serve command.
        """
        argv = [
            "vllm",
            "serve",
            self.model_id,
            "--tensor-parallel-size",
            str(self.tp),
            "--max-model-len",
            self._derive_max_model_len(),
            "--port",
            str(self.port_no),
        ]
        if int(self.nnodes) > 1:
            argv += [
                "--node-rank",
                str(rank),
                "--master-addr",
                str(self.master_addr),
                "--master-port",
                str(self.master_port),
                "--nnodes",
                str(self.nnodes),
                "--pipeline-parallel-size",
                str(self.pp),
                "--distributed-executor-backend",
                "mp",
            ]
            if rank > 0:
                argv.append("--headless")
        argv.extend(self._flatten_serve_args(self.serve_args))
        return argv

    def _rank_log(self, rank: int) -> str:
        return self.server_log.replace("out-node0", f"out-node{rank}")

    # ---------- server side ----------

    def build_server_cmd(self):
        """Write per-node env scripts and create per-rank output directories.

        Broadcast to ALL nodes so every rank has its env script and out-dir
        before start_server launches the per-host processes. On single-node
        the broadcast and targeted exec are equivalent.

        IB devices (ib_hcas, ib_netdev) are written into the env script only
        when present — they come from test_discover_topology (ib_hcas) and
        directly from the config (ib_netdev). No runtime patches, no probing.
        """
        env_lines = [
            f"export HF_TOKEN={shlex.quote(self.hf_token)}",
            f"export HF_HUB_CACHE={shlex.quote(self.models_dir)}",
            "export VLLM_USE_AITER_UNIFIED_ATTENTION=1",
            "export VLLM_ROCM_USE_AITER_MHA=0",
            "export VLLM_ROCM_USE_AITER_FUSED_MOE_A16W4=1",
        ]
        if self.ib_hcas:
            env_lines.append(f"export NCCL_IB_HCA={shlex.quote(','.join(self.ib_hcas))}")
        if self.ib_netdev:
            env_lines.append(f"export NCCL_SOCKET_IFNAME={shlex.quote(self.ib_netdev)}")
            env_lines.append(f"export GLOO_SOCKET_IFNAME={shlex.quote(self.ib_netdev)}")
            env_lines.append(f"export TP_SOCKET_IFNAME={shlex.quote(self.ib_netdev)}")
        for k, v in self.server_env.items():
            env_lines.append(f"export {k}={shlex.quote(str(v))}")
        env_script = "\n".join(env_lines) + "\n"
        self.orch.exec(
            "bash -c " + shlex.quote(f"printf '%s' {shlex.quote(env_script)} > /tmp/server_env_script.sh")
        )
        for rank in range(int(self.nnodes)):
            rank_dir = self.out_dir.replace("out-node0", f"out-node{rank}")
            self.orch.exec(f"mkdir -p {shlex.quote(rank_dir)}")

    def start_server(self):
        """Launch vllm serve on each host with the correct --node-rank.

        enumerate(orch.hosts) yields (rank, host). On single-node this is one
        (0, head) iteration. On multinode each host gets its own targeted exec.
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
        """Check readiness on each node using its own per-rank log path.

        Headless worker nodes (rank > 0) never log 'Application startup
        complete' — they have no API server. Only check the head (rank 0)
        for the startup pattern; worker ranks are considered ready implicitly
        once the head is up.
        """
        pattern = self.READINESS_RE.pattern
        for rank, host in enumerate(self.orch.hosts):
            if rank > 0 and int(self.nnodes) > 1:
                continue
            rank_log = self._rank_log(rank)
            out = self.orch.exec(
                f"grep -qiE {shlex.quote(pattern)} {shlex.quote(rank_log)}",
                detailed=True,
                hosts=[host],
            )
            if not out or not all(r["exit_code"] == 0 for r in out.values()):
                return False
        return True

    def _check_early_failure(self):
        """Check per-rank logs on each host for early failure / fatal patterns."""
        for rank, host in enumerate(self.orch.hosts):
            rank_log = self._rank_log(rank)
            out = self.orch.exec(f"tail -30 {shlex.quote(rank_log)}", hosts=[host])
            for h, output in (out or {}).items():
                if self.EARLY_FAILURE_RE.search(output or ""):
                    raise RuntimeError(f"vllm server early failure on {h} (rank {rank}): {(output or '')[-500:]}")
            out = self.orch.exec(
                f"grep -m1 -iE {shlex.quote(self.FATAL_LOG_RE.pattern)} {shlex.quote(rank_log)}",
                detailed=True,
                hosts=[host],
            )
            for h, r in (out or {}).items():
                if r.get("exit_code") == 0 and r.get("stdout", "").strip():
                    raise RuntimeError(
                        f"vllm server fatal error on {h} (rank {rank}): {r['stdout'].strip()[-500:]}"
                    )

    def wait_ready(self):
        log.info("waiting %ds for server log to materialise", self._precheck_wait)
        time.sleep(self._precheck_wait)

        self._check_early_failure()

        log.info("warmup wait %ds", self._warmup_wait)
        time.sleep(self._warmup_wait)

        self._check_early_failure()

        for it in range(self._server_poll_count):
            if self.is_ready():
                log.info("server ready (iter=%d)", it)
                return
            self._check_early_failure()
            time.sleep(self._server_poll_wait)
        raise RuntimeError("vllm server did not become ready before timeout")

    def stop_server(self):
        """Broadcast pkill to ALL nodes so no stray shard lingers."""
        log.info("stopping vllm server")
        self.orch.exec("bash -c 'pkill -f \"vllm serve\" || true'")
        time.sleep(5)

    # ---------- client side (head-only) ----------

    def run_client(self):
        """Launch bench serve on the HEAD node only via exec_on_head.

        exec_on_head is required (not orch.exec broadcast): on multinode,
        broadcast would launch N competing clients, each connecting to the same
        server endpoint and inflating load.
        """
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
        """Poll the client log on the HEAD node only via exec_on_head.

        Polls silently (no per-iteration log dump) to keep the captured section
        clean. After completion, dump_client_log() emits the full log once.
        """
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
                self.dump_client_log()
                raise RuntimeError("client failed: " + "; ".join(f"{h}: {m}" for h, m in failed))
            if done and all(done):
                log.info("client complete (iter=%d)", it)
                self.dump_client_log()
                return
            time.sleep(self._client_poll_wait)
        self.dump_client_log()
        raise RuntimeError("client did not complete before poll cap")

    def dump_client_log(self):
        """Emit the full client log to the captured section once after completion."""
        out = self.orch.exec_on_head(f"cat {shlex.quote(self.client_log)}")
        for host, text in (out or {}).items():
            for line in (text or "").splitlines():
                log.info("[%s client.log] %s", host, line)

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
