'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Standalone vLLM single-node job driven by a ContainerOrchestrator.

This class talks only to `orch.exec`, which already routes into the running
container, and to a typed `VariantConfig` (see
`cvs.lib.inference.utils.inferencing_config_loader`).
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

import json
import math
import re
import shlex
import time

from cvs.lib import globals
from cvs.lib.inference.utils.vllm_parsing import to_client_metrics

log = globals.log


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
    # The "Serving Benchmark Result" banner is printed unconditionally at the end
    # of every completed `vllm bench serve` run. Do NOT key off a metric header
    # like "End-to-end Latency": stock prints those only when the metric is in
    # --percentile-metrics, so a config omitting e2el would never complete.
    COMPLETION_RE = re.compile(r"Serving Benchmark Result", re.I)
    # bench_serving ALWAYS prints "Failed requests: N" in its summary, so a bare
    # "Failed" match is a false positive on every successful run. Only a NONZERO
    # count is a real failure.
    FAILED_REQUESTS_RE = re.compile(r"Failed requests:\s+([0-9]+)", re.I)
    # A client-side crash (no summary at all) shows up as a Python traceback.
    CLIENT_CRASH_RE = re.compile(r"Traceback \(most recent call last\)", re.I)
    # A launch failure (bad/renamed flag, missing `bench` subcommand, vllm not on
    # PATH) makes the CLI exit before any summary. argparse errors are NOT Python
    # tracebacks and carry no 'Failed requests:' line, so without this the poll
    # loop would spin to its cap (~90 min) before failing. Patterns are narrow
    # CLI-failure markers, not bare 'error:'.
    CLIENT_LAUNCH_FAIL_RE = re.compile(
        r"unrecognized arguments|invalid choice|error: argument |command not found|: No such file or directory",
        re.I,
    )
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
        goodput_slo=None,
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
        # Per-cell SLO dict {ttft_ms, tpot_ms, e2el_ms} or None. An INPUT to the
        # run (passed to `vllm bench serve --goodput`), threaded per-cell like isl
        # because e2el scales with osl. None -> the --goodput flag is omitted and
        # stock leaves request_goodput null.
        self.goodput_slo = goodput_slo
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

        self.model_id = variant.model.id
        self.log_dir = variant.paths.log_dir
        # Per-model server quirks from config (both default empty): extra
        # `vllm serve` flags and extra env vars merged over the orchestrator's
        # defaults. The server command itself is Python-built (no .sh script).
        self.serve_args = dict(variant.roles.server.serve_args)
        self.server_env = dict(variant.roles.server.env)
        # Pin the HF cache onto the mounted models dir. The container binds
        # models_dir both at /models and (via the home bind mount) at its own
        # host path, so this path is valid inside the container and the bytes
        # survive teardown. Without it HF defaults to container-internal
        # ~/.cache/huggingface, which is invisible to the host and re-downloads
        # every run. Same value the model-fetch test polls with `du`.
        self.models_dir = variant.paths.models_dir

        # Single-node, per-cell output directory. Keyed by the cell (isl/osl/
        # conc) so a multi-cell sweep does not overwrite an earlier cell's
        # artifacts -- and so parse_results can never cat a stale `results` from
        # a prior cell when the current cell's client failed to write one.
        self.out_dir = f"{self.log_dir}/{self.log_subdir}/out-node0/isl{self.isl}_osl{self.osl}_conc{self.concurrency}"
        self.server_log = f"{self.out_dir}/vllm_serve_server.log"
        self.client_log = f"{self.out_dir}/client.log"

        self._precheck_wait = server_precheck_wait_s
        self._warmup_wait = server_warmup_wait_s
        self._server_poll_count = server_poll_count
        self._server_poll_wait = server_poll_wait_s
        self._client_initial_wait = client_initial_wait_s
        self._client_poll_count = client_poll_count
        self._client_poll_wait = client_poll_wait_s

    # ---------- server side ----------

    # vLLM's RandomDataset samples input in [isl*(1-r), isl*(1+r)] and output in
    # [osl*(1-r), osl*(1+r)] (r = random_range_ratio), then prepends random_prefix_len
    # fixed tokens. --max-model-len must cover the worst-case input+output+prefix or
    # vLLM 400s every over-length request. Derive it per cell so any sweep change
    # (isl/osl/ratio) stays self-consistent; +8 absorbs the sampler's integer rounding.
    _MML_PAD = 8

    def _derive_max_model_len(self):
        r = float(self.random_range_ratio)
        worst = (int(self.isl) + int(self.osl)) * (1.0 + r)
        return str(math.ceil(worst) + int(self.random_prefix_len) + self._MML_PAD)

    def build_server_cmd(self):
        """Write the server-env script (sourced by both server and client)
        and create the per-node out-dir inside the container."""
        # Only the HF cache pin + token and the AITER tuning flags are read by
        # the vllm process. The server and client commands are Python-built and
        # pass every other value (model, isl/osl, tp, port, max-model-len) as an
        # explicit flag, so exporting them here would be dead.
        env_lines = [
            f"export HF_TOKEN={shlex.quote(self.hf_token)}",
            f"export HF_HUB_CACHE={shlex.quote(self.models_dir)}",
            "export VLLM_USE_AITER_UNIFIED_ATTENTION=1",
            "export VLLM_ROCM_USE_AITER_MHA=0",
            "export VLLM_ROCM_USE_AITER_FUSED_MOE_A16W4=1",
        ]
        # Per-model env overrides win over the defaults above (appended last).
        for k, v in self.server_env.items():
            env_lines.append(f"export {k}={shlex.quote(str(v))}")
        env_script = "\n".join(env_lines) + "\n"
        # printf the script body verbatim; shlex.quote protects the outer bash layer.
        self.orch.exec("bash -c " + shlex.quote(f"printf '%s' {shlex.quote(env_script)} > /tmp/server_env_script.sh"))
        self.orch.exec(f"mkdir -p {shlex.quote(self.out_dir)}")

    @staticmethod
    def _flatten_serve_args(mapping):
        """A {flag: value} serve-args map -> a flat `vllm serve` arg list.

        Flags are given without the leading `--`. A scalar renders
        `--flag <value>`; True renders a bare `--flag` (e.g. --trust-remote-code);
        a list renders the flag once per element (a repeatable flag). This keeps
        config readable ({"kv-cache-dtype": "fp8"}) while still covering vllm's
        bare and repeatable flags, which a flat list could express but not read.
        """
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

    def _server_argv(self):
        """The `vllm serve` arg list for this cell.

        Built in Python (mirrors run_client) so a run is self-contained -- no
        external `.sh` to clone/stage. Only the derived, framework-generic flags
        (tp/max-model-len/port, computed per cell) are set here; per-model knobs
        (e.g. --kv-cache-dtype for an FP8-KV model) come from roles.server.serve_args
        so this driver stays model-agnostic.
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
        argv.extend(self._flatten_serve_args(self.serve_args))
        return argv

    def start_server(self):
        # Each token shlex.quoted: a model id/path with a space or $ would
        # otherwise break the inner bash layer silently (same quoting as the
        # client). The env script (HF token, AITER flags, cache pin) is sourced
        # first; nohup backgrounds the server into its fixed log.
        serve_cmd = " ".join(shlex.quote(str(a)) for a in self._server_argv())
        inner = f"source /tmp/server_env_script.sh && nohup {serve_cmd} > {shlex.quote(self.server_log)} 2>&1 &"
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

    def run_client(self):
        # Build as an arg list and shlex.quote each token: a model id or path
        # containing a space or $ would otherwise break the inner bash layer
        # silently. Mirrors the per-field quoting on the server side.
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
        # Goodput SLO gate (optional). Stock computes request_goodput (good-req/s)
        # only when --goodput is passed; a request is good iff it meets EVERY named
        # SLO. Omit the flag entirely when no per-cell SLO is set (passing
        # ttft:None would be a launch failure).
        if self.goodput_slo:
            args.append("--goodput")
            for metric, key in (("ttft", "ttft_ms"), ("tpot", "tpot_ms"), ("e2el", "e2el_ms")):
                val = self.goodput_slo.get(key)
                if val is not None:
                    args.append(f"{metric}:{val}")
        bench_cmd = " ".join(shlex.quote(str(a)) for a in args)
        client_cmd = f"source /tmp/server_env_script.sh && {bench_cmd} > {shlex.quote(self.client_log)} 2>&1 &"
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
                # A crash or launch failure before the summary -> hard failure now.
                if self.CLIENT_CRASH_RE.search(txt) or self.CLIENT_LAUNCH_FAIL_RE.search(txt):
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
        """Return {host: {client.METRIC: value}} parsed from the stock `results` artifact.

        Fetches the extensionless JSON `results` file `vllm bench serve` writes to
        `--result-dir` (NOT the console log; NOT `results.json`) and delegates the
        namespacing + derived-metric math to the pure
        `cvs.lib.inference.utils.vllm_parsing.to_client_metrics`. Raises if the artifact is
        missing/empty/unparseable -- the test wraps the job in try/except ... raise,
        so this hard-fails the cell rather than recording an empty (silently-green)
        row. The fetch lives here because artifact layout is job-specific; the
        transform lives in inference.utils so distributed/disagg/InferenceX ATOM can reuse it.
        """
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
