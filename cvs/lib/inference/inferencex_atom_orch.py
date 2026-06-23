'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Standalone InferenceX ATOM single-node job driven by a ContainerOrchestrator.

Mirrors :class:`cvs.lib.inference.vllm_orch.VllmJob`: Python-built ``vllm serve``,
``vllm bench serve``, and artifact parsing via ``to_client_metrics``. Does NOT
subclass :class:`cvs.lib.inference.base.InferenceBaseJob`.
'''

from __future__ import annotations

import json
import re
import shlex
import time

from cvs.lib import globals
from cvs.lib.inference.utils.vllm_parsing import to_client_metrics

log = globals.log


class InferenceXAtomJob:
    """Single-node InferenceX ATOM benchmark job driven by an injected ContainerOrchestrator."""

    READINESS_RE = re.compile(r"Application startup complete|Uvicorn running|Started server", re.I)
    COMPLETION_RE = re.compile(r"Serving Benchmark Result", re.I)
    FAILED_REQUESTS_RE = re.compile(r"Failed requests:\s+([0-9]+)", re.I)
    CLIENT_CRASH_RE = re.compile(r"Traceback \(most recent call last\)", re.I)
    CLIENT_LAUNCH_FAIL_RE = re.compile(
        r"unrecognized arguments|invalid choice|error: argument |command not found|: No such file or directory",
        re.I,
    )
    EARLY_FAILURE_RE = re.compile(
        r"no such file or directory|command not found|cannot access|failed to start",
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
        self.tp = p.tensor_parallelism
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

        self.model_id = variant.model.id
        self.log_dir = variant.paths.log_dir
        self.models_dir = variant.paths.models_dir
        self.serve_args = self._merged_serve_args(variant)
        self.server_env = dict(variant.roles.server.env)

        self.out_dir = (
            f"{self.log_dir}/{self.log_subdir}/out-node0/"
            f"isl{self.isl}_osl{self.osl}_conc{self.concurrency}"
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
        self._bench_max_failed_requests = int(bench_max_failed_requests)

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

    def build_server_cmd(self):
        env_lines = [
            f"export HF_TOKEN={shlex.quote(self.hf_token)}",
            f"export HF_HUB_CACHE={shlex.quote(self.models_dir)}",
            "export VLLM_USE_AITER_UNIFIED_ATTENTION=1",
            "export VLLM_ROCM_USE_AITER_MHA=0",
            "export VLLM_ROCM_USE_AITER_FUSED_MOE_A16W4=1",
        ]
        for k, v in self.server_env.items():
            if k in ("CVS_GPU_MEMORY_UTIL", "VLLM_GPU_MEMORY_UTIL", "VLLM_ENFORCE_EAGER"):
                continue
            env_lines.append(f"export {k}={shlex.quote(str(v))}")
        env_script = "\n".join(env_lines) + "\n"
        self.orch.exec("bash -c " + shlex.quote(f"printf '%s' {shlex.quote(env_script)} > /tmp/server_env_script.sh"))
        self.orch.exec(f"mkdir -p {shlex.quote(self.out_dir)}")

    def _server_argv(self):
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
        argv.extend(self._flatten_serve_args(self.serve_args))
        return argv

    def start_server(self):
        serve_cmd = " ".join(shlex.quote(str(a)) for a in self._server_argv())
        inner = (
            f"source /tmp/server_env_script.sh && "
            f"nohup {serve_cmd} > {shlex.quote(self.server_log)} 2>&1 &"
        )
        out = self.orch.exec("bash -c " + shlex.quote(inner))
        for host, output in out.items():
            if self.EARLY_FAILURE_RE.search(output or ""):
                raise RuntimeError(f"vllm server failed to launch on {host}: {output[-500:]}")

    def is_ready(self):
        pattern = self.READINESS_RE.pattern
        out = self.orch.exec(
            f"grep -qiE {shlex.quote(pattern)} {shlex.quote(self.server_log)}",
            detailed=True,
        )
        return bool(out) and all(r["exit_code"] == 0 for r in out.values())

    def wait_ready(self):
        log.info("waiting %ds for server log to materialise", self._precheck_wait)
        time.sleep(self._precheck_wait)

        out = self.orch.exec(f"tail -30 {shlex.quote(self.server_log)}")
        for host, output in out.items():
            if self.EARLY_FAILURE_RE.search(output or ""):
                raise RuntimeError(f"vllm server early failure on {host}: {output[-500:]}")

        log.info("warmup wait %ds", self._warmup_wait)
        time.sleep(self._warmup_wait)

        for it in range(self._server_poll_count):
            if self.is_ready():
                log.info("server ready (iter=%d)", it)
                return
            time.sleep(self._server_poll_wait)
        raise RuntimeError("vllm server did not become ready before timeout")

    def stop_server(self):
        log.info("stopping vllm server")
        self.orch.exec("bash -c 'pkill -f \"vllm serve\" || true'")
        time.sleep(5)

    def run_client(self):
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
        bench_cmd = " ".join(shlex.quote(str(a)) for a in args)
        client_cmd = f"source /tmp/server_env_script.sh && {bench_cmd} > {shlex.quote(self.client_log)} 2>&1 &"
        self.orch.exec("bash -c " + shlex.quote(client_cmd))

    def wait_client_complete(self):
        log.info("client initial wait %ds", self._client_initial_wait)
        time.sleep(self._client_initial_wait)
        cap = self._bench_max_failed_requests
        for it in range(self._client_poll_count):
            out = self.orch.exec(f"tail -2000 {shlex.quote(self.client_log)}")
            failed = []
            done = []
            for host, output in out.items():
                txt = output or ""
                done.append(bool(self.COMPLETION_RE.search(txt)))
                if self.CLIENT_CRASH_RE.search(txt) or self.CLIENT_LAUNCH_FAIL_RE.search(txt):
                    failed.append((host, txt[-500:]))
                else:
                    fm = self.FAILED_REQUESTS_RE.search(txt)
                    if fm:
                        fc = int(fm.group(1))
                        if fc > cap:
                            failed.append((host, f"Failed requests: {fc} (cap {cap}) -- {txt[-500:]}"))
                        elif fc > 0:
                            log.warning(
                                "client on %s completed with %d failed requests (allowed up to %d)",
                                host,
                                fc,
                                cap,
                            )
            if failed:
                raise RuntimeError("client failed: " + "; ".join(f"{h}: {m}" for h, m in failed))
            if done and all(done):
                log.info("client complete (iter=%d)", it)
                return
            time.sleep(self._client_poll_wait)
        raise RuntimeError("client did not complete before poll cap")

    def parse_results(self):
        artifact = f"{self.out_dir}/results"
        out = self.orch.exec(f"cat {shlex.quote(artifact)}")
        results = {}
        for host, text in out.items():
            text = (text or "").strip()
            if not text:
                raise RuntimeError(f"empty/missing results artifact on {host}: {artifact}")
            try:
                raw = json.loads(text)
            except (json.JSONDecodeError, ValueError) as e:
                raise RuntimeError(f"unparseable results artifact on {host}: {artifact}: {e}") from e
            results[host] = to_client_metrics(raw, tp=self.tp, isl=self.isl)
        return results
