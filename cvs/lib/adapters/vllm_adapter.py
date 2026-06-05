"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

VllmAdapter -- the v1 reference adapter and the canonical "how to add a
framework" example. Replaces the v1 inference base + per-model wrappers.

Bake-ins (addendum §5, "Integration Milestone" decisions):

- C1: readiness asserts HTTP 200 via ``curl -s -o /dev/null -w '%{http_code}'``
  against ``params.base_url:params.port_no/health`` -- no hardcoded
  ``:8888``, no "any non-empty output" heuristic. Probe runs in ``launch``
  with a bounded poll budget BEFORE the bench dispatches; a never-ready
  server raises ``LivenessFailure`` at the correct boundary.
- C2: benchmark runs inside the launched container via ``docker exec -d``
  (DETACHED) so ``launch`` returns promptly and
  ``await_completion``/``progress_predicate`` actually observe the
  bench finishing via the result file. Result delivery works under
  either site shape: on a shared FS (Weka/NFS at the same path on
  devbox + nodes) the local-existence check in ``progress_predicate``
  succeeds with no round-trip; otherwise ``progress_predicate`` falls
  back to ``ssh test -f`` and ``parse`` SFTP-fetches via
  ``ctx.executor.download``.
- C3: the adapter reads ``params.tensor_parallelism`` /
  ``params.concurrency`` / ``params.isl`` / ``params.osl`` as
  scalars (single-cell shape). The multi-cell sweep machinery in
  ``cvs/lib/config/sweep.py`` exists but is not wired into the
  conftest's cell-parametrize hook today; PR-Z lifts these scalars to
  swept axes.
- C4: descoped (security removed). HF token rides in
  ``cfg.container.env['HF_TOKEN']`` and is passed to the container via
  ``-e HF_TOKEN=<value>`` -- recorded verbatim in pssh logs / manifest
  commands / container.log / docker inspect.
- G3.2 sidecar: per-request columns ``ttft_ms`` / ``tpot_ms`` / ``itl_ms``
  / ``e2el_ms`` / ``output_tokens`` are adapter-private; documented in
  ``cvs/lib/adapters/AGENTS.md``.

``read_bench_result`` is the only vLLM-specific shape coupling and is
static + pure so it is unit-testable against a fixture file without any
cluster.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from cvs.lib.adapter_protocol import Progress
from cvs.lib.base_adapter import BaseWorkloadAdapter
from cvs.lib.config.thresholds import PercentileThreshold
from cvs.lib.config.thresholds import ResultView
from cvs.lib.failure_taxonomy import SetupFailure
from cvs.lib.registry import register_adapter

# Maps the typed top-level ``knobs`` (in the YAML config) to vLLM/AITER env
# vars. Lifting these out of the shared v1 inference base (where they were
# hardcoded for every framework regardless of relevance) is a core point of
# the refactor: framework-specific env belongs to the adapter that needs it.
_AITER_ENV = {
    "aiter_unified": {"VLLM_USE_AITER_UNIFIED_ATTENTION": "1"},
    "aiter_mha": {"VLLM_ROCM_USE_AITER_MHA": "1"},
}
_FUSED_MOE_ENV = {
    "a16w4": {"VLLM_ROCM_USE_AITER_FUSED_MOE_A16W4": "1"},
}

_BENCH_RESULT_FILENAME = "bench_result.json"


@register_adapter("vllm", kind="inference")
class VllmAdapter(BaseWorkloadAdapter):
    """Single-node vLLM online-serving benchmark adapter."""

    framework = "vllm"
    # Single-role single-node inference: one ``server`` host. Matches the
    # ``Topology.nnodes: N`` shorthand which expands to {server: {count: N}}.
    # A future multi-server variant (data-parallel servers behind a router)
    # would override this.
    required_roles = ("server",)
    completion_timeout_s = 5400.0
    poll_interval_s = 15.0
    # Server-readiness budget inside launch(): bounded poll of HTTP /health
    # before the bench fires. Default 10 minutes is enough to cold-load a
    # 70B FP8 checkpoint on MI300; tunable for larger models.
    server_ready_timeout_s: float = 600.0
    server_ready_interval_s: float = 5.0

    # ---- helpers ---------------------------------------------------------

    def _knob_env(self, ctx) -> Dict[str, str]:
        knobs = getattr(ctx.config, "knobs", None) or {}
        env: Dict[str, str] = {}
        env.update(_AITER_ENV.get(knobs.get("attention", ""), {}))
        env.update(_FUSED_MOE_ENV.get(knobs.get("fused_moe", ""), {}))
        return env

    def _bench_results_local_path(self, ctx) -> Path:
        return Path(ctx.layout.logs_dir) / _BENCH_RESULT_FILENAME

    # ---- argv composition ------------------------------------------------

    def _server_command(self, ctx, tp: int) -> str:
        """Compose ``vllm serve`` argv from typed params (default mode).

        The adapter owns the CLI shape so a vLLM CLI rename / flag change
        is a one-line patch here, not a tree-wide YAML migration. The
        legacy escape hatch (``params.server_script``) wins when set --
        operators with a custom in-image launcher still work.
        """
        p = ctx.config.params
        parts = [
            "vllm",
            "serve",
            ctx.config.model,
            "--tensor-parallel-size",
            str(tp),
            "--host",
            "0.0.0.0",
            "--port",
            str(p.port_no),
            "--max-model-len",
            str(p.max_model_len),
            "--gpu-memory-utilization",
            str(p.gpu_memory_utilization),
        ]
        if p.quantization:
            parts += ["--quantization", p.quantization]
        if p.dtype:
            parts += ["--dtype", p.dtype]
        if p.trust_remote_code:
            parts.append("--trust-remote-code")
        if p.download_dir:
            parts += ["--download-dir", p.download_dir]
        parts += list(p.server_extra_args)
        return shlex.join(parts)

    def _bench_argv(
        self,
        ctx,
        results_path: Path,
        percentile_metrics: str,
        metric_percentiles: str,
        concurrency: int,
        isl: int,
        osl: int,
    ) -> List[str]:
        """Compose ``vllm bench serve`` argv from typed params (default mode).

        Returns a list of argv tokens; the caller wraps the argv into a
        ``docker exec -d <name> sh -c "python -m vllm.entrypoints.cli.main
        bench serve <argv> > <log> 2>&1"`` invocation. ``--base-url``
        carries the port (the new CLI shape -- the old ``--port`` flag is
        silently ignored).
        """
        p = ctx.config.params
        argv = [
            "--backend",
            p.backend,
            "--model",
            ctx.config.model,
            "--base-url",
            f"{p.base_url}:{p.port_no}",
            "--dataset-name",
            p.dataset_name,
            "--num-prompts",
            str(p.num_prompts),
            "--max-concurrency",
            str(concurrency),
        ]
        # ``--random-input-len`` / ``--random-output-len`` are only
        # meaningful for the synthetic ``random`` dataset; emitting them
        # for ``sharegpt`` / ``hf`` / ``sonnet`` would shadow the real
        # prompts. Gate the emission on dataset_name.
        if p.dataset_name == "random":
            argv += [
                "--random-input-len",
                str(isl),
                "--random-output-len",
                str(osl),
            ]
        argv += [
            "--request-rate",
            str(p.request_rate),
            "--burstiness",
            str(p.burstiness),
            "--seed",
            str(ctx.config.seed),
            "--tokenizer-mode",
            p.tokenizer_mode,
            "--percentile-metrics",
            percentile_metrics,
            "--metric-percentiles",
            metric_percentiles,
            "--save-result",
            "--result-filename",
            str(results_path),
        ]
        argv += list(p.bench_extra_args)
        return argv

    # ---- lifecycle -------------------------------------------------------

    def launch(self, ctx) -> None:
        """Start the server container, gate on HTTP 200, then dispatch the
        bench client DETACHED so ``progress_predicate`` can observe the
        result file landing.

        This is the load-bearing C1 + C2 wiring. Order matters:

        1. Build a ContainerHandle from ContainerSpec (A3); merge AITER
           knob-env on top of container.env (workload env wins on conflict).
        2. Bind-mount logs_dir into the container at the same path so the
           bench writes its --save-result file at a shared-FS path devbox
           also reads (A1 staging deferred via shared-FS path).
        3. ``handle.__enter__()`` starts the container (returns when
           ``docker run -d`` returns the ID, not when the server is ready).
        4. Bounded poll on HTTP /health -> 200 (C1); raise
           ``LivenessFailure`` on timeout at the correct boundary so
           classification names the actual condition (server never came
           up), not the wrong phase.
        5. Emit ``launch.role_ready`` only after the probe succeeds --
           the event must reflect an observation, not a hope.
        6. Dispatch the bench via ``docker exec -d`` (DETACHED). The
           bench writes ``bench_result.json`` when done; the
           ``progress_predicate`` polls its existence and the
           ``await_completion`` budget (5400s) bounds the run.

        If the bench were dispatched synchronously, (a) the await phase
        would collapse to ~0 (predicate is dead code in the happy path),
        (b) any bench failure would surface as ``SetupFailure`` (wrong
        category -- the failure is in the workload phase), and (c) the
        manifest's phase timings would attribute the bench duration to
        launch. The current order keeps the spine semantics live.
        """
        if ctx.executor is None:
            raise SetupFailure("vLLM launch requires an executor (Pssh) bound to the server node")
        params = ctx.config.params

        # ContainerSpec -> handle kwargs (A3). Merge order, last write wins:
        #   1. fabric.to_env()    -- NCCL/UCX/Gloo site-fabric defaults
        #   2. _knob_env(ctx)     -- AITER / vLLM tunables from cfg.knobs
        #   3. container.env      -- workload-authored env, including secrets
        # ``setdefault`` from least-to-most-authoritative means container.env
        # always wins on conflict, which is the documented precedence.
        spec_kwargs = ctx.config.container.to_handle_kwargs()
        workload_env = dict(spec_kwargs.get("env", {}))
        env: dict = {}
        fabric = getattr(ctx.config, "fabric", None)
        if fabric is not None:
            env.update(fabric.to_env())
        for key, value in self._knob_env(ctx).items():
            env.setdefault(key, value)
        for key, value in workload_env.items():
            env[key] = value
        spec_kwargs["env"] = env

        # Shared-FS bind mount: logs_dir at the same path on both sides so
        # the bench's --save-result lands where devbox reads from (C2; A1
        # staging deferred).
        ctx.layout.ensure()
        logs_dir = str(Path(ctx.layout.logs_dir))
        volumes = dict(spec_kwargs.get("volumes", {}))
        volumes.setdefault(logs_dir, logs_dir)
        spec_kwargs["volumes"] = volumes

        # Port mapping for the server endpoint. Docker emits a warning and
        # silently ignores -p when --network host is set (the port is
        # already exposed via the host's network namespace), so skip the
        # injection in that case to keep the recorded command clean.
        if spec_kwargs.get("network") != "host":
            ports = dict(spec_kwargs.get("ports", {}))
            port = str(params.port_no)
            ports.setdefault(port, port)
            spec_kwargs["ports"] = ports

        tp = int(params.tensor_parallelism)
        # The in-container launcher (used only when params.server_script is
        # set; otherwise the adapter composes vllm-serve argv directly) can
        # read CVS_TP to pass --tensor-parallel-size.
        spec_kwargs["env"]["CVS_TP"] = str(tp)

        image = ctx.config.container.image
        if not image:
            raise SetupFailure("no container image: set container.image on the workload config")

        server_command = params.server_script or self._server_command(ctx, tp)
        # bindings["server"] is single-host today; the loop in _launch_role
        # runs once. Future data-parallel / disagg adapters that need >1
        # server reuse the same helper without forking the launch path.
        handles = self._launch_role(
            ctx,
            "server",
            image=image,
            command=server_command,
            **spec_kwargs,
        )
        handle = handles[0]

        # C1 readiness gate -- bounded poll of HTTP /health on every
        # registered server handle (pool-of-one today). Hits localhost
        # inside the per-host runner, not params.base_url -- the runner
        # is already scoped to the right host, so a node-local probe is
        # the correct shape for the multi-host generalization.
        self._wait_http_pool("server", "/health", int(params.port_no), self.server_ready_timeout_s)
        ctx.events.emit("launch.role_ready", role="server")

        # C2: dispatch the bench DETACHED inside the container so launch()
        # returns promptly. The shared-FS result file is the DONE signal
        # observed by progress_predicate.
        results_path = self._bench_results_local_path(ctx)
        ctx.scratch["results_path"] = str(results_path)
        bench_cmd = self._bench_command(ctx, results_path, handle.name)
        ctx.scratch.setdefault("commands", []).append(bench_cmd)
        ctx.executor.exec(bench_cmd)

    def _bench_command(self, ctx, results_path: Path, container_name: str) -> str:
        """C2: full bench-serve flag set, run detached inside the container.

        Reads ``params.tensor_parallelism`` / ``concurrency`` / ``isl`` /
        ``osl`` as scalars (single-cell shape). PR-Z lifts these to sweep
        axes; until then the YAML declares one value per knob and the
        adapter consumes them directly.
        """
        params = ctx.config.params
        concurrency = int(params.concurrency)
        isl = int(params.isl)
        osl = int(params.osl)
        # Derive the bench flags from thresholds (single source of truth),
        # union'd with any measure-only ``extra_percentile_metrics``. The
        # metric-root is the threshold ``metric`` with the trailing ``_ms``
        # stripped -- bench flags speak in roots (``ttft``), threshold
        # metrics in units (``ttft_ms``). A no-percentile-threshold config
        # falls back to the legacy default set so single-cell barebones
        # without explicit thresholds still emits sensible flags.
        pct_thresholds = [t for t in ctx.config.thresholds if isinstance(t, PercentileThreshold)]
        metric_roots = []
        seen = set()
        for t in pct_thresholds:
            root = t.metric[:-3] if t.metric.endswith("_ms") else t.metric
            if root not in seen:
                seen.add(root)
                metric_roots.append(root)
        for extra in params.extra_percentile_metrics:
            if extra not in seen:
                seen.add(extra)
                metric_roots.append(extra)
        if not metric_roots:
            metric_roots = ["ttft", "tpot", "itl", "e2el"]
        percentile_metrics = ",".join(metric_roots)
        # ``--metric-percentiles`` accepts CSV; emit the unique set across
        # configured percentile thresholds (almost always one value, e.g. 99).
        metric_percentiles_values = sorted({int(t.percentile) for t in pct_thresholds}) or [99]
        metric_percentiles = ",".join(str(v) for v in metric_percentiles_values)
        # ``docker exec -d`` returns immediately; the bench runs in the
        # container's background and writes the result file when done.
        # Redirect stdout/stderr inside the container so the bench log is
        # captured under the shared-FS logs_dir for post-mortem.
        bench_log = Path(ctx.layout.logs_dir) / "bench.log"
        if params.bench_serv_script:
            # Legacy escape hatch: pass the operator string through as-is to
            # the container shell. No ``python`` prefix, no flag injection --
            # the operator owns the full command shape.
            random_flags = (
                f"--random-input-len {isl} --random-output-len {osl} " if params.dataset_name == "random" else ""
            )
            inner = (
                f"{params.bench_serv_script} "
                f"--backend {params.backend} "
                f"--model {ctx.config.model} "
                f"--base-url {params.base_url}:{params.port_no} "
                f"--dataset-name {params.dataset_name} "
                f"--num-prompts {params.num_prompts} "
                f"--max-concurrency {concurrency} "
                f"{random_flags}"
                f"--request-rate {params.request_rate} "
                f"--burstiness {params.burstiness} "
                f"--seed {ctx.config.seed} "
                f"--tokenizer-mode {params.tokenizer_mode} "
                f"--percentile-metrics {percentile_metrics} "
                f"--metric-percentiles {metric_percentiles} "
                f"--save-result --result-filename {results_path}"
            )
        else:
            argv = self._bench_argv(
                ctx,
                results_path,
                percentile_metrics,
                metric_percentiles,
                concurrency,
                isl,
                osl,
            )
            inner = "python -m vllm.entrypoints.cli.main bench serve " + shlex.join(argv)
        # The shell wrapper redirects to the log file; the inner command is
        # carefully composed via shlex.join above so embedded spaces or
        # quotes in argv tokens (e.g. extra-args) survive intact.
        return f'docker exec -d {container_name} sh -c "{inner} > {bench_log} 2>&1"'

    def progress_predicate(self, ctx) -> Progress:
        """RUNNING / DONE, observed via the bench's result file on the bound node.

        Polls the local path first (shared-FS case: zero round-trip). When
        the run is on a no-shared-FS cluster the result lives on the node
        only, so we fall back to ``ssh test -f`` via the executor. ``parse``
        is what actually SFTP-fetches the bytes once we know the file
        exists -- the predicate stays cheap.

        Note: this adapter never returns ``Progress.BROKEN`` today --
        single-cell vLLM relies on ``await_completion`` 's 5400s timeout
        to surface a dead/hung server as ``LivenessFailure``. The bench's
        own exit-code path is the canonical "dead-server" signal in C2
        docker-exec mode (the result file never appears when the bench
        client crashes). A future enhancement could probe
        ``docker inspect -f '{{.State.Running}}'`` with a grace counter to
        return ``Progress.BROKEN`` directly (= ``SafetyViolation`` -- the
        right category for a server that died mid-run); deferred until
        the first adapter that genuinely needs it.
        """
        results_path = ctx.scratch.get("results_path")
        if not results_path:
            return Progress.RUNNING
        if Path(results_path).exists():
            return Progress.DONE
        if ctx.executor is not None:
            try:
                out = ctx.executor.exec(f"test -f {results_path} && echo Y || true")
            except Exception:  # noqa: BLE001 - transient probe miss
                return Progress.RUNNING
            if "Y" in (out or ""):
                return Progress.DONE
        return Progress.RUNNING

    def parse(self, ctx) -> None:
        """Populate scalars + long-format samples from bench_serving JSON.

        When the result file is not already on the local FS (no shared FS
        between devbox and the bound node), SFTP-fetch it via the
        executor's ``download`` passthrough to ``Pssh.download_file``.
        ``ctx.executor`` is the single-host executor the conftest builds;
        adapters never construct their own Pssh.

        A1 staging seam DEFERRED: when a *second* adapter needs the same
        fetch shape, promote this two-line call into a
        ``RunContext.fetch(remote) -> local`` helper. Until then, keeping
        it inline avoids adding generic plumbing for one consumer.
        """
        results_path = ctx.scratch.get("results_path")
        if results_path and not Path(results_path).exists() and ctx.executor is not None:
            try:
                ctx.executor.download(results_path, results_path)
            except Exception as exc:  # noqa: BLE001 - parser tolerates missing
                ctx.events.emit("parse.fetch_failed", path=results_path, error=str(exc))
        scalars, samples = self.read_bench_result(results_path)
        ctx.result = ResultView(scalars=scalars, samples=samples)
        self._write_sidecars(ctx, samples)

    @staticmethod
    def read_bench_result(
        results_path: Optional[str],
    ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """Parse vLLM ``benchmark_serving --save-result`` JSON -> (scalars, samples).

        Pure and static so it is unit-testable against a fixture file. Tolerant
        of:
        - missing/None ``results_path`` -> ``({}, [])`` (parse may run before
          the bench finished in some failure paths; ``progress_predicate`` is
          the existence oracle).
        - missing individual keys in the JSON (skipped, not zero-defaulted).

        Per-request lists (``ttfts``/``tpots``/``itls``/``e2els``/``output_lens``)
        zip into one ``samples`` row per request (G3.2 sidecar policy).
        """
        scalars: Dict[str, float] = {}
        samples: List[Dict[str, float]] = []
        if not results_path or not Path(results_path).exists():
            return scalars, samples
        data = json.loads(Path(results_path).read_text())

        def _num(key: str) -> Optional[float]:
            value = data.get(key)
            return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

        # input_key -> output_key (CVS-internal scalar name).
        mapping = {
            "duration": "elapsed_s",
            "request_throughput": "request_throughput",
            "output_throughput": "output_throughput",
            "total_token_throughput": "total_throughput",
            "mean_ttft_ms": "mean_ttft_ms",
            "p99_ttft_ms": "p99_ttft_ms",
            "mean_tpot_ms": "mean_tpot_ms",
            "p99_tpot_ms": "p99_tpot_ms",
        }
        for in_key, out_key in mapping.items():
            value = _num(in_key)
            if value is not None:
                scalars[out_key] = value

        ttfts = data.get("ttfts") or []
        tpots = data.get("tpots") or []
        itls = data.get("itls") or []
        e2els = data.get("e2els") or []
        out_lens = data.get("output_lens") or []
        n = max(len(ttfts), len(tpots), len(itls), len(e2els), len(out_lens))
        for idx in range(n):
            row: Dict[str, float] = {"request_id": float(idx)}
            if idx < len(ttfts):
                row["ttft_ms"] = float(ttfts[idx])
            if idx < len(tpots):
                row["tpot_ms"] = float(tpots[idx])
            if idx < len(itls):
                row["itl_ms"] = float(itls[idx])
            if idx < len(e2els):
                row["e2el_ms"] = float(e2els[idx])
            if idx < len(out_lens):
                row["output_tokens"] = float(out_lens[idx])
            row["role"] = "server"
            samples.append(row)
        return scalars, samples

    def _write_sidecars(self, ctx, samples: List[Dict[str, float]]) -> None:
        if not samples:
            return
        # Lazy import: keeps the adapter importable for registry smoke tests
        # without pyarrow on the path.
        from cvs.lib.manifest.sidecars import write_samples

        write_samples(ctx.layout.samples_path, samples)
