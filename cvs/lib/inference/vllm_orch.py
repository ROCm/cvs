'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Standalone vLLM single-node job driven by a ContainerOrchestrator.

This class talks only to `orch.exec`, which already routes into the running
container, and to a typed `VariantConfig` (see `cvs.lib.dtni.config_loader`).
It is deliberately single-node and free of the `c_phdl`/`s_phdl` + manual
`docker exec` plumbing that `cvs.lib.inference.base.InferenceBaseJob` carries.

It does NOT subclass `InferenceBaseJob`: the base runs against raw
`c_phdl`/`s_phdl` handles and untyped `if_dict`/`bp_dict` config, while this
job runs against an `orch` and a pydantic `VariantConfig`. Bridging the two
is a base-layer refactor (out of scope for this PoC); see
`plans/vllm-single-orch-poc.md`. The legacy `cvs.lib.inference.vllm.VllmJob`
has no remaining importers and can be removed in that follow-up.

Behavioural improvements over the base-class lifecycle it mirrors:
  - no dead distributed/`nnodes` branch
  - readiness is detected by scanning the whole server log, not `tail -30`
    (the startup banner scrolls out of a fixed tail once vLLM gets chatty)
  - completion is checked before failure, and only a nonzero failed-request
    count is treated as a client failure (the summary always prints
    "Failed requests: N")
'''

from __future__ import annotations

import re
import shlex
import time

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
    # bench_serving ALWAYS prints "Failed requests: N" in its summary, so a bare
    # "Failed" match is a false positive on every successful run. Only a NONZERO
    # count is a real failure.
    FAILED_REQUESTS_RE = re.compile(r"Failed requests:\s+([0-9]+)", re.I)
    # A client-side crash (no summary at all) shows up as a Python traceback.
    CLIENT_CRASH_RE = re.compile(r"Traceback \(most recent call last\)", re.I)
    # Narrow launch-failure markers only. Bare "error:"/"exception:"/"traceback" are
    # NOT included: vLLM/ROCm startup routinely logs benign lines containing them
    # (deprecation notes, ignored-exception handlers, optional-probe failures), and
    # matching those aborts a server that would have come up fine.
    EARLY_FAILURE_RE = re.compile(
        r"no such file or directory|command not found|cannot access|failed to start",
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
        # 60*60s = 60min readiness budget. A remote (online) model pull on cell 1
        # downloads ~152GB into the HF cache before the server reports ready; the
        # old 30*60s=30min cap raced that download. Free on the happy path: the
        # loop returns as soon as is_ready(), so a bigger cap only lengthens the
        # FAILURE path (how long a genuinely-stuck server waits before raising).
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
        # Pin the HF cache onto the mounted models dir. The container binds
        # models_dir both at /models and (via the home bind mount) at its own
        # host path, so this path is valid inside the container and the bytes
        # survive teardown. Without it HF defaults to container-internal
        # ~/.cache/huggingface, which is invisible to the host and re-downloads
        # every run. Same value the model-fetch test polls with `du`.
        self.models_dir = variant.paths.models_dir

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
            f"export MODEL={shlex.quote(self.model_id)}",
            f"export ISL={shlex.quote(self.isl)}",
            f"export OSL={shlex.quote(self.osl)}",
            f"export MAX_MODEL_LEN={shlex.quote(self.max_model_length)}",
            f"export RANDOM_RANGE_RATIO={shlex.quote(self.random_range_ratio)}",
            f"export TP={shlex.quote(str(self.tp))}",
            f"export CONC={shlex.quote(self.concurrency)}",
            f"export HF_TOKEN={shlex.quote(self.hf_token)}",
            f"export HF_HUB_CACHE={shlex.quote(self.models_dir)}",
            "export VLLM_USE_AITER_UNIFIED_ATTENTION=1",
            "export VLLM_ROCM_USE_AITER_MHA=0",
            "export VLLM_ROCM_USE_AITER_FUSED_MOE_A16W4=1",
            "export RESULT_FILENAME=results",
            f"export PORT={shlex.quote(str(self.port_no))}",
        ]
        env_script = "\n".join(env_lines) + "\n"
        # printf the script body verbatim; shlex.quote protects the outer bash layer.
        self.orch.exec("bash -c " + shlex.quote(f"printf '%s' {shlex.quote(env_script)} > /tmp/server_env_script.sh"))
        self.orch.exec(f"mkdir -p {shlex.quote(self.out_dir)}")

    def start_server(self):
        inner = (
            f"cd {shlex.quote(self.scripts_dir)} && source /tmp/server_env_script.sh && "
            f"nohup /bin/bash {shlex.quote(self.server_script)} > {shlex.quote(self.server_log)} 2>&1 &"
        )
        out = self.orch.exec("bash -c " + shlex.quote(inner))
        for host, output in out.items():
            if self.EARLY_FAILURE_RE.search(output or ""):
                raise RuntimeError(f"vllm server failed to launch on {host}: {output[-500:]}")

    def is_ready(self):
        # Evaluate readiness IN the container and ship back only an exit code.
        # grep scans the whole log (the one-shot startup banner scrolls out of any
        # tail once vLLM gets chatty) but `-q` stops at the first match and prints
        # nothing -- no cat, no megabytes of log over the wire. Derive the pattern
        # from the one regex so the two cannot drift.
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

    # ---------- client side ----------

    def _clone_bench_serving(self, clone_dir="/app"):
        # bench_serving is a calibration-bearing fork (kimbochen), NOT stock vLLM:
        # it carries a warmup phase + redefined random range-ratio/seq-length that
        # our thresholds are tuned against, so the in-image vLLM scripts won't do.
        # Hardcoded for parity with the legacy path (base.py's benchmark_script_repo
        # default); cloned at HEAD (unpinned) -- pin if upstream drift ever bites.
        cmd = (
            f"bash -c 'mkdir -p {clone_dir} && cd {clone_dir} && "
            f"(test -d bench_serving || git clone https://github.com/kimbochen/bench_serving.git)'"
        )
        out = self.orch.exec(cmd)
        for host, output in out.items():
            if re.search(r"(error|fatal):", output or "", re.I) and not re.search(
                r"already exists", output or "", re.I
            ):
                raise RuntimeError(f"bench_serving clone failed on {host}: {output[-500:]}")

    def run_client(self):
        self._clone_bench_serving("/app")
        # Build as an arg list and shlex.quote each token: a model id or path
        # containing a space or $ would otherwise break the inner bash layer
        # silently. Mirrors the per-field quoting on the server side.
        args = [
            "python3",
            f"bench_serving/{self.bench_serv_script}",
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
            "--ignore-eos",
            "--save-result",
            "--result-dir",
            self.out_dir,
            "--result-filename",
            "results",
        ]
        bench_cmd = " ".join(shlex.quote(str(a)) for a in args)
        client_cmd = (
            f"source /tmp/server_env_script.sh && cd /app && {bench_cmd} > {shlex.quote(self.client_log)} 2>&1 &"
        )
        self.orch.exec("bash -c " + shlex.quote(client_cmd))

    def wait_client_complete(self):
        log.info("client initial wait %ds", self._client_initial_wait)
        time.sleep(self._client_initial_wait)
        for it in range(self._client_poll_count):
            out = self.orch.exec(f"tail -2000 {shlex.quote(self.client_log)}")
            failed = []
            done = []
            for host, output in out.items():
                txt = output or ""
                done.append(bool(self.COMPLETION_RE.search(txt)))
                # A crash before the summary -> hard failure now.
                if self.CLIENT_CRASH_RE.search(txt):
                    failed.append((host, txt[-500:]))
                else:
                    # The summary always reports a failed-request count; only a
                    # nonzero count is a real failure (NOT the literal word "Failed").
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
        """Return {host: {metric: str_value}} parsed from the client log."""
        out = self.orch.exec(f"tail -2000 {shlex.quote(self.client_log)}")
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
