'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Single-node InferenceX SGLang benchmark job driven by a ContainerOrchestrator.

``python -m sglang.launch_server`` + ``python -m sglang.bench_serving`` with a
JSON results artifact parsed through :func:`to_client_metrics`.
'''

from __future__ import annotations

import json
import re
import shlex
import time

from cvs.lib import globals
from cvs.lib.inference.utils.vllm_parsing import to_client_metrics

log = globals.log


class InferenceXAtomSglangJob:
    """Single-node SGLang parity job driven by an injected ContainerOrchestrator."""

    READINESS_RE = re.compile(r"Uvicorn running|Application startup complete|Started server", re.I)
    COMPLETION_RE = re.compile(r"Serving Benchmark Result|Benchmark Result", re.I)
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

    def __init__(
        self,
        orch,
        variant,
        hf_token,
        isl,
        osl,
        concurrency,
        num_prompts,
        log_subdir="inferencex-atom-sglang",
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
        self.bench_extra_args = (p.bench_extra_args or "").strip()
        self.result_stem = (p.result_filename or "results").removesuffix(".json")

        self.model_id = variant.model.id
        self.log_dir = variant.paths.log_dir
        self.models_dir = variant.paths.models_dir
        self.sglang_server_args = list(variant.roles.server.sglang_args)
        self.server_env = dict(variant.roles.server.env)

        self.out_dir = (
            f"{self.log_dir}/{self.log_subdir}/out-node0/"
            f"isl{self.isl}_osl{self.osl}_conc{self.concurrency}"
        )
        self.server_log = f"{self.out_dir}/sglang_server.log"
        self.client_log = f"{self.out_dir}/client.log"
        self._result_artifact = f"{self.out_dir}/{self.result_stem}.json"

        self._precheck_wait = server_precheck_wait_s
        self._warmup_wait = server_warmup_wait_s
        self._server_poll_count = server_poll_count
        self._server_poll_wait = server_poll_wait_s
        self._client_initial_wait = client_initial_wait_s
        self._client_poll_count = client_poll_count
        self._client_poll_wait = client_poll_wait_s
        self._bench_max_failed_requests = int(bench_max_failed_requests)

    def build_server_cmd(self):
        env_lines = [
            f"export HF_TOKEN={shlex.quote(self.hf_token)}",
            f"export HF_HUB_CACHE={shlex.quote(self.models_dir)}",
            "export SGLANG_USE_AITER=1",
        ]
        for k, v in self.server_env.items():
            env_lines.append(f"export {k}={shlex.quote(str(v))}")
        env_script = "\n".join(env_lines) + "\n"
        self.orch.exec("bash -c " + shlex.quote(f"printf '%s' {shlex.quote(env_script)} > /tmp/server_env_script.sh"))
        self.orch.exec(f"mkdir -p {shlex.quote(self.out_dir)}")

    def _server_argv(self):
        argv = [
            "python",
            "-m",
            "sglang.launch_server",
            "--model",
            self.model_id,
            "--host",
            "0.0.0.0",
            "--port",
            str(self.port_no),
            "--tp",
            str(self.tp),
            "--trust-remote-code",
        ]
        argv.extend(self.sglang_server_args)
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
                raise RuntimeError(f"sglang server failed to launch on {host}: {output[-500:]}")

    def _health_ok(self):
        url = f"http://localhost:{self.port_no}/health"
        out = self.orch.exec(
            f"curl -sf {shlex.quote(url)} -o /dev/null && echo OK || echo NO"
        )
        return bool(out) and all("OK" in (v or "") for v in out.values())

    def is_ready(self):
        if self._health_ok():
            return True
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
                raise RuntimeError(f"sglang server early failure on {host}: {output[-500:]}")

        log.info("warmup wait %ds", self._warmup_wait)
        time.sleep(self._warmup_wait)

        for it in range(self._server_poll_count):
            if self.is_ready():
                log.info("sglang server ready (iter=%d)", it)
                return
            time.sleep(self._server_poll_wait)
        raise RuntimeError("sglang server did not become ready before timeout")

    def stop_server(self):
        log.info("stopping sglang server")
        self.orch.exec(
            "bash -c "
            + shlex.quote("pkill -f 'sglang.launch_server' || pkill -f 'launch_server' || true")
        )
        time.sleep(5)

    def _client_argv(self):
        argv = [
            "python",
            "-m",
            "sglang.bench_serving",
            "--backend",
            self.backend,
            "--model",
            self.model_id,
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
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port_no),
            "--ignore-eos",
            "--save-result",
            "--result-dir",
            self.out_dir,
            "--result-filename",
            f"{self.result_stem}.json",
        ]
        if self.bench_extra_args:
            argv.extend(shlex.split(self.bench_extra_args))
        return argv

    def _clear_stale_result_artifact(self):
        artifact = shlex.quote(self._result_artifact)
        self.orch.exec(f"rm -f {artifact}")

    def run_client(self):
        self._clear_stale_result_artifact()
        bench_cmd = " ".join(shlex.quote(str(a)) for a in self._client_argv())
        client_cmd = f"source /tmp/server_env_script.sh && {bench_cmd} > {shlex.quote(self.client_log)} 2>&1 &"
        self.orch.exec("bash -c " + shlex.quote(client_cmd))

    def _result_ready(self):
        out = self.orch.exec(f"test -s {shlex.quote(self._result_artifact)} && echo OK || echo NO")
        return bool(out) and all("OK" in (v or "") for v in out.values())

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
                done.append(self._result_ready())
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
                log.info("sglang client complete (iter=%d)", it)
                return
            txt_all = "".join((output or "") for output in out.values())
            if self.COMPLETION_RE.search(txt_all) and self._result_ready():
                log.info("sglang client complete via log marker (iter=%d)", it)
                return
            time.sleep(self._client_poll_wait)
        raise RuntimeError("sglang client did not complete before poll cap")

    def parse_results(self):
        out = self.orch.exec(f"cat {shlex.quote(self._result_artifact)}")
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
            raw.setdefault("random_input_len", int(self.isl))
            raw.setdefault("random_output_len", int(self.osl))
            results[host] = to_client_metrics(raw, tp=self.tp, isl=self.isl)
        return results
