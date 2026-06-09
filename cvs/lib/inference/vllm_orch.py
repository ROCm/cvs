'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Standalone vLLM single-node job driven by a ContainerOrchestrator.

Lives next to (not subclasses) the legacy `cvs.lib.inference.vllm.VllmJob`
because the legacy class is still consumed by other suites (sglang,
inferencemax). This class talks only to `orch.exec` / `orch.exec_on_head`,
which already route into the running container, and to a `VariantConfig`
(see `cvs.lib.dtni.config_loader`).

Drops, vs. the legacy class:
  - the dead `self.port_no` distributed branch
  - the `random_range_ration` typo
  - the `globals.error_list` indirection
  - the silent-skip in `verify_inference_results` (callers run `evaluate_all`
    against the parsed actuals instead)
'''

from __future__ import annotations

import re
import time
from typing import Dict

from cvs.lib import globals

log = globals.log


_METRIC_RES = [
    ("successful_requests", re.compile(r"Successful requests:\s+([0-9]+)", re.I)),
    ("benchmark_duration", re.compile(r"Benchmark duration\s+\(s\):\s+([0-9\.]+)", re.I)),
    ("total_input_tokens", re.compile(r"Total input tokens:\s+([0-9\.]+)", re.I)),
    ("total_generated_tokens", re.compile(r"Total generated tokens:\s+([0-9\.]+)", re.I)),
    ("request_throughput_per_sec", re.compile(r"Request throughput \(req/s\):\s+([0-9\.]+)", re.I)),
    ("output_throughput_per_sec", re.compile(r"Output token throughput \(tok/s\):\s+([0-9\.]+)", re.I)),
    ("total_throughput_per_sec", re.compile(r"Total Token throughput \(tok/s\):\s+([0-9\.]+)", re.I)),
    ("mean_ttft_ms", re.compile(r"Mean TTFT \(ms\):\s+([0-9\.]+)", re.I)),
    ("median_ttft_ms", re.compile(r"Median TTFT \(ms\):\s+([0-9\.]+)", re.I)),
    ("p99_ttft_ms", re.compile(r"P99 TTFT \(ms\):\s+([0-9\.]+)", re.I)),
    ("mean_tpot_ms", re.compile(r"Mean TPOT \(ms\):\s+([0-9\.]+)", re.I)),
    ("median_tpot_ms", re.compile(r"Median TPOT \(ms\):\s+([0-9\.]+)", re.I)),
    ("p99_tpot_ms", re.compile(r"P99 TPOT \(ms\):\s+([0-9\.]+)", re.I)),
    ("mean_itl_ms", re.compile(r"Mean ITL \(ms\):\s+([0-9\.]+)", re.I)),
    ("median_itl_ms", re.compile(r"Median ITL \(ms\):\s+([0-9\.]+)", re.I)),
    ("p99_itl_ms", re.compile(r"P99 ITL \(ms\):\s+([0-9\.]+)", re.I)),
    ("mean_e2el_ms", re.compile(r"Mean E2EL \(ms\):\s+([0-9\.]+)", re.I)),
    ("median_e2el_ms", re.compile(r"Median E2EL \(ms\):\s+([0-9\.]+)", re.I)),
    ("p99_e2el_ms", re.compile(r"P99 E2EL \(ms\):\s+([0-9\.]+)", re.I)),
]


def _shquote(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


class VllmJob:
    """Single-node vLLM benchmark job driven by an injected ContainerOrchestrator.

    All container/SSH plumbing belongs to `orch`. This class composes the
    server-env script, launches the server in the background inside the
    container, polls until ready, runs the bench_serving client, and parses
    the resulting log.

    The `orch` instance is expected to already have `setup_containers()` and
    `setup_sshd()` called against it (by the test fixture); lifecycle is
    explicitly NOT owned here.
    """

    READINESS_RE = re.compile(r"Application startup complete|Uvicorn running|Started server", re.I)
    COMPLETION_RE = re.compile(r"End-to-end Latency", re.I)
    EARLY_FAILURE_RE = re.compile(
        r"no such file or directory|command not found|cannot access|permission denied|"
        r"error:|exception:|traceback|failed to start",
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
        log_subdir="vllm",
        server_precheck_wait_s=30,
        server_warmup_wait_s=330,
        server_poll_count=30,
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
        self.log_subdir = log_subdir

        p = variant.params
        self.tp = p.tensor_parallelism
        self.port_no = p.port_no
        self.max_model_length = p.max_model_length
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
        self.bench_serv_script = p.bench_serv_script

        self.model_id = variant.model.id
        self.server_script = variant.roles.server.server_script
        self.log_dir = variant.paths.log_dir
        self.scripts_dir = variant.paths.benchmark_scripts_dir

        # Single-node: one output directory.
        self.out_dir = f"{self.log_dir}/{self.log_subdir}/out-node0"
        self.server_log = f"{self.out_dir}/{self.server_script}_server.log"
        self.client_log = f"{self.out_dir}/bench_serv_script.log"

        self._precheck_wait = server_precheck_wait_s
        self._warmup_wait = server_warmup_wait_s
        self._server_poll_count = server_poll_count
        self._server_poll_wait = server_poll_wait_s
        self._client_initial_wait = client_initial_wait_s
        self._client_poll_count = client_poll_count
        self._client_poll_wait = client_poll_wait_s

    # ---------- server side ----------

    def build_server_cmd(self):
        """Write the server-env script and create the per-node out-dir inside the container."""
        env_lines = [
            f"export MODEL={self.model_id}",
            f"export ISL={self.isl}",
            f"export OSL={self.osl}",
            f"export MAX_MODEL_LEN={self.max_model_length}",
            f"export RANDOM_RANGE_RATIO={self.random_range_ratio}",
            f"export TP={self.tp}",
            f"export CONC={self.concurrency}",
            f"export HF_TOKEN={self.hf_token}",
            "export VLLM_USE_AITER_UNIFIED_ATTENTION=1",
            "export VLLM_ROCM_USE_AITER_MHA=0",
            "export VLLM_ROCM_USE_AITER_FUSED_MOE_A16W4=1",
            "export RESULT_FILENAME=results",
            f"export PORT={self.port_no}",
        ]
        env_script = "\n".join(env_lines) + "\n"
        # Use printf with quoted body to avoid heredoc inside orch.exec layers.
        self.orch.exec("bash -c " + _shquote(f"printf '%s' {_shquote(env_script)} > /tmp/server_env_script.sh"))
        self.orch.exec(f"mkdir -p {self.out_dir}")

    def start_server(self):
        cmd = (
            f"bash -c 'cd {self.scripts_dir} && source /tmp/server_env_script.sh && "
            f"nohup /bin/bash {self.server_script} > {self.server_log} 2>&1 &'"
        )
        out = self.orch.exec(cmd)
        for host, output in out.items():
            if self.EARLY_FAILURE_RE.search(output or ""):
                raise RuntimeError(f"vllm server failed to launch on {host}: {output[-500:]}")

    def is_ready(self):
        out = self.orch.exec(f"tail -30 {self.server_log}")
        return bool(out) and all(bool(self.READINESS_RE.search(output or "")) for output in out.values())

    def wait_ready(self):
        log.info("waiting %ds for server log to materialise", self._precheck_wait)
        time.sleep(self._precheck_wait)

        out = self.orch.exec(f"tail -30 {self.server_log}")
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

    # ---------- client side ----------

    def _clone_bench_serving(self, clone_dir="/app"):
        cmd = (
            f"bash -c 'mkdir -p {clone_dir} && cd {clone_dir} && "
            f"(test -d bench_serving || git clone https://github.com/kimbochen/bench_serving.git)'"
        )
        out = self.orch.exec(cmd)
        for host, output in out.items():
            if re.search(r"(error|fatal):", output or "", re.I) and not re.search(r"already exists", output or "", re.I):
                raise RuntimeError(f"bench_serving clone failed on {host}: {output[-500:]}")

    def run_client(self):
        self._clone_bench_serving("/app")
        client_cmd = (
            "source /tmp/server_env_script.sh && cd /app && "
            f"python3 bench_serving/{self.bench_serv_script} "
            f"--model {self.model_id} "
            f"--backend {self.backend} "
            f"--base-url {self.base_url}:{self.port_no} "
            f"--dataset-name {self.dataset_name} "
            f"--num-prompts {self.num_prompts} "
            f"--random-input-len {self.isl} "
            f"--random-output-len {self.osl} "
            f"--max-concurrency {self.concurrency} "
            f"--request-rate {self.request_rate} "
            f"--burstiness {self.burstiness} "
            f"--tokenizer-mode {self.tokenizer_mode} "
            f"--seed {self.seed} "
            f"--random-range-ratio {self.random_range_ratio} "
            f"--random-prefix-len {self.random_prefix_len} "
            f"--percentile-metrics {self.percentile_metrics} "
            "--ignore-eos --save-result "
            f"--result-dir {self.out_dir} --result-filename results "
            f"> {self.client_log} 2>&1 &"
        )
        self.orch.exec("bash -c " + _shquote(client_cmd))

    def wait_client_complete(self):
        log.info("client initial wait %ds", self._client_initial_wait)
        time.sleep(self._client_initial_wait)
        for it in range(self._client_poll_count):
            out = self.orch.exec(f"tail -2000 {self.client_log}")
            failed = []
            done = []
            for host, output in out.items():
                txt = output or ""
                if re.search(r"Failed", txt, re.I):
                    failed.append((host, txt[-500:]))
                done.append(bool(self.COMPLETION_RE.search(txt)))
            if failed:
                raise RuntimeError("client failed: " + "; ".join(f"{h}: {m}" for h, m in failed))
            if done and all(done):
                log.info("client complete (iter=%d)", it)
                return
            time.sleep(self._client_poll_wait)
        raise RuntimeError("client did not complete before poll cap")

    def parse_results(self):
        """Return {host: {metric: str_value}} parsed from the client log."""
        out = self.orch.exec(f"tail -2000 {self.client_log}")
        results = {}
        for host, text in out.items():
            text = text or ""
            m = {}
            for key, pat in _METRIC_RES:
                hit = pat.search(text)
                if hit:
                    m[key] = hit.group(1)
            results[host] = m
        return results
