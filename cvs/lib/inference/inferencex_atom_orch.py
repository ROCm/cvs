'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

InferenceX ATOM job driven by a ContainerOrchestrator (single- or multi-node).

``params.driver=atom`` (target): ``atom.entrypoints.openai_server`` +
``atom.benchmarks.benchmark_serving`` with ATOM JSON artifacts.

``params.driver=vllm`` (interim uplift): ``vllm serve`` + ``vllm bench serve``.

When ``params.nnodes`` > 1 the job launches one ATOM server per cluster host.
For ``driver=atom``, multinode uses ATOM SPMD data-parallel env vars
(``ATOM_DP_*``) plus ``-dp N`` when each node has enough GPUs for local DP
replicas (``tensor_parallelism * nnodes <= 8``). With TP8 across two 8-GPU
nodes, each host runs an independent ``-tp 8`` replica (no ``-dp``) because
ATOM cannot place two local DP ranks when TP already consumes all eight GPUs;
the benchmark client still runs on the head node only. For ``driver=vllm``,
multinode keeps vLLM's ``--node-rank`` / ``--master-addr`` flags.

Does NOT subclass :class:`cvs.lib.inference.base.InferenceBaseJob`.
'''

from __future__ import annotations

import json
import re
import shlex
import time

from cvs.lib import globals
from cvs.lib.inference.utils.inferencex_atom_parsing import to_client_metrics

log = globals.log


class InferenceXAtomJob:
    """InferenceX ATOM benchmark job driven by an injected ContainerOrchestrator."""

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

    _DEFAULT_SERVE_ARGS = {
        "block-size": 64,
        "no-enable-prefix-caching": True,
    }

    def __init__(
        self,
        orch,
        variant,
        hf_token,
        isl,
        osl,
        concurrency,
        num_prompts,
        log_subdir="inferencex-atom",
        server_precheck_wait_s=30,
        server_warmup_wait_s=330,
        server_poll_count=60,
        server_poll_wait_s=60,
        client_initial_wait_s=120,
        client_poll_count=50,
        client_poll_wait_s=60,
        bench_max_failed_requests=0,
    ):
        self.orch = orch
        self.variant = variant
        self.hf_token = hf_token
        self.isl = str(isl)
        self.osl = str(osl)
        self.concurrency = str(concurrency)
        self.num_prompts = str(num_prompts)
        self.log_subdir = log_subdir

        p = variant.params
        self.driver = p.driver
        self.tp = p.tensor_parallelism
        self.pp = p.pipeline_parallel_size
        self.nnodes = int(p.nnodes)
        self.distributed = self.nnodes > 1
        raw_master = (p.master_addr or "").strip()
        self.master_addr = raw_master or (orch.hosts[0] if getattr(orch, "hosts", None) else "localhost")
        self.master_port = p.master_port
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
        self.max_model_length = str(p.max_model_length)
        self.bench_extra_args = (p.bench_extra_args or "").strip()
        self.result_stem = (p.result_filename or "results").removesuffix(".json")
        raw_baseline = (p.scaling_baseline_output_throughput or "").strip()
        self._scaling_baseline = float(raw_baseline) if raw_baseline else None

        self.model_id = variant.model.id
        self.log_dir = variant.paths.log_dir
        self.models_dir = variant.paths.models_dir
        self.serve_args = self._merged_serve_args(variant)
        self.atom_server_args = list(variant.roles.server.atom_args)
        self.server_env = dict(variant.roles.server.env)

        self.out_dir = self._node_out_dir(0)
        self.server_log = self._rank_server_log(0)
        self.client_log = f"{self.out_dir}/client.log"
        self._result_artifact = (
            f"{self.out_dir}/{self.result_stem}.json"
            if self.driver == "atom"
            else f"{self.out_dir}/{self.result_stem}"
        )

        self._precheck_wait = server_precheck_wait_s
        self._warmup_wait = server_warmup_wait_s
        self._server_poll_count = server_poll_count
        self._server_poll_wait = server_poll_wait_s
        self._client_initial_wait = client_initial_wait_s
        self._client_poll_count = client_poll_count
        self._client_poll_wait = client_poll_wait_s
        self._bench_max_failed_requests = int(bench_max_failed_requests)

    @classmethod
    def from_variant(cls, orch, variant, hf_token, isl, osl, concurrency, **overrides):
        """Construct a job with server/client timing from ``variant.params``."""
        p = variant.params

        def _int_attr(name, default):
            raw = getattr(p, name, None)
            if raw is None or str(raw).strip() == "":
                return default
            try:
                return int(raw)
            except (TypeError, ValueError):
                return default

        kw = dict(
            orch=orch,
            variant=variant,
            hf_token=hf_token,
            isl=isl,
            osl=osl,
            concurrency=concurrency,
            num_prompts=p.num_prompts,
            server_precheck_wait_s=_int_attr("server_precheck_wait_s", 30),
            server_warmup_wait_s=_int_attr("server_warmup_wait_s", 330),
            server_poll_count=_int_attr("server_poll_count", 60),
            server_poll_wait_s=_int_attr("server_poll_wait_time", 60),
            client_initial_wait_s=_int_attr("client_initial_wait_s", 120),
            client_poll_count=_int_attr("client_poll_count", 50),
            client_poll_wait_s=_int_attr("client_poll_wait_time", 60),
            bench_max_failed_requests=_int_attr("bench_max_failed_requests", 0),
        )
        kw.update(overrides)
        return cls(**kw)

    def _node_out_dir(self, rank):
        return (
            f"{self.log_dir}/{self.log_subdir}/out-node{rank}/"
            f"isl{self.isl}_osl{self.osl}_conc{self.concurrency}"
        )

    def _rank_server_log(self, rank):
        base = self._node_out_dir(rank)
        name = "atom_server.log" if self.driver == "atom" else "vllm_serve_server.log"
        return f"{base}/{name}"

    def _exec_all(self, cmd, **kwargs):
        return self.orch.exec(cmd, **kwargs)

    def _exec_head(self, cmd, **kwargs):
        if self.distributed:
            return self.orch.exec_on_head(cmd, **kwargs)
        return self.orch.exec(cmd, **kwargs)

    def prepare_cell_out_dir(self):
        """Create per-cell output directory without touching server env or cache."""
        if self.distributed:
            for rank in range(self.nnodes):
                self._exec_all(f"mkdir -p {shlex.quote(self._node_out_dir(rank))}")
        else:
            self._exec_all(f"mkdir -p {shlex.quote(self.out_dir)}")

    @classmethod
    def _merged_serve_args(cls, variant):
        merged = dict(cls._DEFAULT_SERVE_ARGS)
        merged.update(variant.roles.server.serve_args)
        env = variant.roles.server.env
        gpu_mem = env.get("CVS_GPU_MEMORY_UTIL") or env.get("VLLM_GPU_MEMORY_UTIL")
        if gpu_mem is not None and "gpu-memory-utilization" not in merged:
            merged["gpu-memory-utilization"] = str(gpu_mem)
        if "enforce-eager" not in merged:
            merged["enforce-eager"] = True
        return merged

    @staticmethod
    def _flatten_serve_args(mapping):
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

    def _atom_spmd_dp_cli(self):
        """Return ``-dp N`` argv when local GPUs can host *N* TP shards each."""
        if not self.distributed:
            return []
        tp = int(self.tp)
        if tp * self.nnodes > 8:
            return []
        if self._argv_has_flag(self.atom_server_args, "-dp", "--data-parallel-size"):
            return []
        return ["-dp", str(self.nnodes)]

    @staticmethod
    def _argv_has_flag(argv, *names):
        for tok in argv:
            if tok in names:
                return True
        return False

    def _atom_spmd_env_exports(self, rank):
        if not self.distributed or not self._atom_spmd_dp_cli():
            return []
        return [
            f"export ATOM_DP_RANK={rank}",
            f"export ATOM_DP_SIZE={self.nnodes}",
            "export ATOM_DP_RANK_LOCAL=0",
            f"export ATOM_DP_MASTER_IP={shlex.quote(self.master_addr)}",
            f"export ATOM_DP_MASTER_PORT={self.master_port}",
        ]

    def _distributed_argv(self, rank):
        if not self.distributed:
            return []
        if self.driver == "atom":
            return self._atom_spmd_dp_cli()
        return [
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

    def build_server_cmd(self, *, clear_atom_cache=True):
        env_lines = [
            f"export HF_TOKEN={shlex.quote(self.hf_token)}",
            f"export HF_HUB_CACHE={shlex.quote(self.models_dir)}",
        ]
        if self.driver == "vllm":
            env_lines.extend(
                [
                    "export VLLM_USE_AITER_UNIFIED_ATTENTION=1",
                    "export VLLM_ROCM_USE_AITER_MHA=0",
                    "export VLLM_ROCM_USE_AITER_FUSED_MOE_A16W4=1",
                ]
            )
        for k, v in self.server_env.items():
            if k in ("CVS_GPU_MEMORY_UTIL", "VLLM_GPU_MEMORY_UTIL", "VLLM_ENFORCE_EAGER"):
                continue
            env_lines.append(f"export {k}={shlex.quote(str(v))}")
        env_script = "\n".join(env_lines) + "\n"
        self._exec_all(
            "bash -c " + shlex.quote(f"printf '%s' {shlex.quote(env_script)} > /tmp/server_env_script.sh")
        )
        if self.distributed:
            for rank in range(self.nnodes):
                self._exec_all(f"mkdir -p {shlex.quote(self._node_out_dir(rank))}")
        else:
            self._exec_all(f"mkdir -p {shlex.quote(self.out_dir)}")
        if self.driver == "atom" and clear_atom_cache:
            self._exec_all("bash -c 'rm -rf ~/.cache/atom/* 2>/dev/null || true'")

    def _server_argv(self, rank=0):
        argv = [
            "vllm",
            "serve",
            self.model_id,
            "--host",
            "0.0.0.0",
            "--tensor-parallel-size",
            str(self.tp),
            "--max-model-len",
            self.max_model_length,
            "--port",
            str(self.port_no),
        ]
        argv.extend(self._distributed_argv(rank))
        argv.extend(self._flatten_serve_args(self.serve_args))
        return argv

    def _atom_server_argv(self, rank=0):
        argv = [
            "python",
            "-m",
            "atom.entrypoints.openai_server",
            "--model",
            self.model_id,
            "--server-port",
            str(self.port_no),
        ]
        argv.extend(self.atom_server_args)
        argv.extend(self._distributed_argv(rank))
        return argv

    def start_server(self):
        hosts = list(getattr(self.orch, "hosts", []) or ["node0"])
        if self.distributed and len(hosts) != self.nnodes:
            raise RuntimeError(
                f"params.nnodes={self.nnodes} but cluster has {len(hosts)} host(s); "
                "align cluster node_dict with params.nnodes"
            )
        label = "atom" if self.driver == "atom" else "vllm"
        launch_hosts = enumerate(hosts) if self.distributed else [(0, hosts[0])]
        for rank, host in launch_hosts:
            argv = self._atom_server_argv(rank) if self.driver == "atom" else self._server_argv(rank)
            serve_cmd = " ".join(shlex.quote(str(a)) for a in argv)
            rank_log = self._rank_server_log(rank)
            rank_env = " && ".join(self._atom_spmd_env_exports(rank))
            env_prefix = f"{rank_env} && " if rank_env else ""
            inner = (
                f"source /tmp/server_env_script.sh && "
                f"{env_prefix}"
                f"nohup {serve_cmd} > {shlex.quote(rank_log)} 2>&1 &"
            )
            if self.distributed:
                out = self._exec_all("bash -c " + shlex.quote(inner), hosts=[host])
            else:
                out = self._exec_all("bash -c " + shlex.quote(inner))
            for h, output in out.items():
                if self.EARLY_FAILURE_RE.search(output or ""):
                    raise RuntimeError(f"{label} server failed to launch on {h} (rank {rank}): {output[-500:]}")

    def _atom_health_ok(self):
        url = f"http://localhost:{self.port_no}/health"
        probe = f"curl -sf {shlex.quote(url)} -o /dev/null && echo OK || echo NO"
        if self.distributed:
            out = self._exec_all("bash -c " + shlex.quote(probe))
        else:
            out = self._exec_head("bash -c " + shlex.quote(probe))
        return bool(out) and all("OK" in (v or "") for v in out.values())

    def _atom_warmup_ok(self):
        payload = json.dumps(
            {"model": self.model_id, "prompt": "hi", "max_tokens": 1},
            separators=(",", ":"),
        )
        url = f"http://localhost:{self.port_no}/v1/completions"
        inner = (
            f"curl -sf {shlex.quote(url)} -H 'Content-Type: application/json' "
            f"-d {shlex.quote(payload)} -o /dev/null --max-time 120 && echo OK || echo NO"
        )
        out = self._exec_head("bash -c " + shlex.quote(inner))
        return bool(out) and all("OK" in (v or "") for v in out.values())

    def is_ready(self):
        if self.driver == "atom":
            return self._atom_health_ok()
        if self.distributed:
            pattern = self.READINESS_RE.pattern
            ready = True
            for rank in range(self.nnodes):
                log_path = self._rank_server_log(rank)
                out = self._exec_all(
                    f"grep -qiE {shlex.quote(pattern)} {shlex.quote(log_path)}",
                    detailed=True,
                )
                if not out or not all(r["exit_code"] == 0 for r in out.values()):
                    ready = False
                    break
            return ready
        pattern = self.READINESS_RE.pattern
        out = self._exec_all(
            f"grep -qiE {shlex.quote(pattern)} {shlex.quote(self.server_log)}",
            detailed=True,
        )
        return bool(out) and all(r["exit_code"] == 0 for r in out.values())

    def _tail_server_logs(self, lines=30):
        if self.distributed:
            out = {}
            for rank in range(self.nnodes):
                chunk = self._exec_all(f"tail -{lines} {shlex.quote(self._rank_server_log(rank))}")
                out.update(chunk or {})
            return out
        return self._exec_all(f"tail -{lines} {shlex.quote(self.server_log)}")

    def wait_ready(self):
        log.info("waiting %ds for server log to materialise", self._precheck_wait)
        time.sleep(self._precheck_wait)

        out = self._tail_server_logs(30)
        for host, output in out.items():
            if self.EARLY_FAILURE_RE.search(output or ""):
                label = "atom" if self.driver == "atom" else "vllm"
                raise RuntimeError(f"{label} server early failure on {host}: {output[-500:]}")

        log.info("warmup wait %ds", self._warmup_wait)
        time.sleep(self._warmup_wait)

        for it in range(self._server_poll_count):
            if not self.is_ready():
                if self.driver == "atom":
                    poll_out = self._tail_server_logs(30)
                    for host, output in poll_out.items():
                        if self.EARLY_FAILURE_RE.search(output or ""):
                            raise RuntimeError(
                                f"atom server early failure on {host}: {output[-500:]}"
                            )
                time.sleep(self._server_poll_wait)
                continue
            log.info("server health ready (iter=%d)", it)
            break
        else:
            raise RuntimeError("server did not become ready before timeout")

        if self.driver == "atom":
            for it in range(10):
                if self._atom_warmup_ok():
                    log.info("server warmup complete (iter=%d)", it)
                    return
                time.sleep(30)
            raise RuntimeError("atom server warmup did not complete before timeout")

    def stop_server(self):
        if self.driver == "atom":
            log.info("stopping atom server")
            self._exec_all(
                "bash -c "
                + shlex.quote(
                    "pkill -f 'atom.entrypoints.openai_server' || "
                    "pkill -f 'openai_server' || true"
                )
            )
        else:
            log.info("stopping vllm server")
            self._exec_all("bash -c 'pkill -f \"vllm serve\" || true'")
        time.sleep(5)

    def _atom_client_argv(self):
        warmups = int(self.concurrency) * 2
        argv = [
            "python",
            "-m",
            "atom.benchmarks.benchmark_serving",
            "--model",
            self.model_id,
            "--backend",
            "vllm",
            "--base-url",
            f"http://localhost:{self.port_no}",
            "--dataset-name",
            self.dataset_name,
            "--random-input-len",
            self.isl,
            "--random-output-len",
            self.osl,
            "--random-range-ratio",
            self.random_range_ratio,
            "--max-concurrency",
            self.concurrency,
            "--num-prompts",
            self.num_prompts,
            "--trust-remote-code",
            "--num-warmups",
            str(warmups),
            "--request-rate",
            self.request_rate,
            "--ignore-eos",
            "--save-result",
            "--percentile-metrics",
            self.percentile_metrics,
            "--result-dir",
            self.out_dir,
            "--result-filename",
            f"{self.result_stem}.json",
        ]
        if self.bench_extra_args:
            argv.extend(shlex.split(self.bench_extra_args))
        return argv

    def _vllm_client_argv(self):
        return [
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
            self.result_stem,
        ]

    def _clear_stale_result_artifact(self):
        """Remove a prior run's result file so poll logic cannot treat it as complete."""
        artifact = shlex.quote(self._result_artifact)
        self._exec_head(f"rm -f {artifact}")

    def run_client(self):
        self._clear_stale_result_artifact()
        args = self._atom_client_argv() if self.driver == "atom" else self._vllm_client_argv()
        bench_cmd = " ".join(shlex.quote(str(a)) for a in args)
        client_cmd = f"source /tmp/server_env_script.sh && {bench_cmd} > {shlex.quote(self.client_log)} 2>&1 &"
        self._exec_head("bash -c " + shlex.quote(client_cmd))

    def _atom_result_ready(self):
        out = self._exec_head(
            f"test -s {shlex.quote(self._result_artifact)} && echo OK || echo NO"
        )
        return bool(out) and all("OK" in (v or "") for v in out.values())

    def _client_log_failures(self, tail_lines=2000):
        out = self._exec_head(f"tail -{tail_lines} {shlex.quote(self.client_log)}")
        failed = []
        for host, output in out.items():
            txt = output or ""
            if self.CLIENT_CRASH_RE.search(txt) or self.CLIENT_LAUNCH_FAIL_RE.search(txt):
                failed.append((host, txt[-500:]))
                continue
            fm = self.FAILED_REQUESTS_RE.search(txt)
            if fm:
                fc = int(fm.group(1))
                cap = self._bench_max_failed_requests
                if fc > cap:
                    failed.append((host, f"Failed requests: {fc} (cap {cap}) -- {txt[-500:]}"))
                elif fc > 0:
                    log.warning(
                        "client on %s completed with %d failed requests (allowed up to %d)",
                        host,
                        fc,
                        cap,
                    )
        return failed

    def wait_client_complete(self):
        if self.driver == "atom":
            log.info(
                "client initial wait (atom: polling for result artifact, up to %ds)",
                self._client_initial_wait,
            )
            deadline = time.monotonic() + self._client_initial_wait
            poll_s = 15
            while time.monotonic() < deadline:
                failed = self._client_log_failures(tail_lines=500)
                if failed:
                    raise RuntimeError("client failed: " + "; ".join(f"{h}: {m}" for h, m in failed))
                if self._atom_result_ready():
                    log.info("client result artifact ready during initial wait")
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(poll_s, remaining))
        else:
            log.info("client initial wait %ds", self._client_initial_wait)
            time.sleep(self._client_initial_wait)

        for it in range(self._client_poll_count):
            failed = self._client_log_failures()
            if failed:
                raise RuntimeError("client failed: " + "; ".join(f"{h}: {m}" for h, m in failed))
            if self.driver == "atom":
                if self._atom_result_ready():
                    log.info("client complete (iter=%d)", it)
                    return
            else:
                out = self._exec_head(f"tail -2000 {shlex.quote(self.client_log)}")
                done = [bool(self.COMPLETION_RE.search(txt or "")) for txt in out.values()]
                if done and all(done):
                    log.info("client complete (iter=%d)", it)
                    return
            time.sleep(self._client_poll_wait)
        raise RuntimeError("client did not complete before poll cap")

    def parse_results(self):
        out = self._exec_head(f"cat {shlex.quote(self._result_artifact)}")
        results = {}
        for host, text in out.items():
            text = (text or "").strip()
            if not text:
                raise RuntimeError(f"empty/missing results artifact on {host}: {self._result_artifact}")
            try:
                raw = json.loads(text)
            except (json.JSONDecodeError, ValueError) as e:
                raise RuntimeError(
                    f"unparseable results artifact on {host}: {self._result_artifact}: {e}"
                ) from e
            if self.driver == "atom":
                raw.setdefault("random_input_len", int(self.isl))
                raw.setdefault("random_output_len", int(self.osl))
            results[host] = to_client_metrics(
                raw,
                tp=self.tp,
                isl=self.isl,
                scaling_baseline_output_throughput=self._scaling_baseline,
                nnodes=self.nnodes,
            )
        return results
