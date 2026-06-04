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
  bench finishing via the shared-FS result file. Full
  ``benchmark_serving`` flag set. Result delivery uses the SHARED-FS
  path: the run's ``logs_dir`` is bind-mounted into the container at
  the same path it has on devbox, so the bench writes
  ``bench_result.json`` once and devbox reads it with no SFTP fetch.
  (A1 staging was deferred in G5b; see addendum §9 Integration Milestone
  for the chosen path.)
- C3-barebones: consume the typed ``sweep.tensor_parallelism`` /
  ``sweep.concurrency`` / ``sweep.sequence_combinations`` values
  directly (fail-closed when absent or multi-valued). Single-cell
  barebones -- PR-Z lowers swept values per cell when the matrix
  layer ships.
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
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from cvs.lib.adapter_protocol import Progress
from cvs.lib.base_adapter import BaseWorkloadAdapter
from cvs.lib.config.thresholds import ResultView
from cvs.lib.failure_taxonomy import LivenessFailure, SetupFailure
from cvs.lib.registry import register_adapter
from cvs.lib.runtime.container_handle import ContainerHandle

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

    def _single_axis(self, ctx, attr: str):
        """Pull a barebones single-cell value off ``sweep.<attr>``.

        PR-Y ships a single-cell sweep: every axis is a one-element list.
        We read it directly off the typed sweep so the YAML is honored
        even when ``cell.params`` is empty (the conftest passes
        ``cell=None`` in barebones). Fail-closed on absent or multi-valued
        to match ``_tensor_parallelism`` and to keep the silent-coincidence
        bug (axis declared but ignored) impossible. PR-Z lowers
        ``cell.params`` per cell; this helper is only the single-cell
        default, not the production sweep machinery.
        """
        sweep = getattr(ctx.config, "sweep", None)
        values = getattr(sweep, attr, None) if sweep is not None else None
        if not values:
            raise SetupFailure(f"vLLM: sweep.{attr} must declare exactly one value in barebones mode")
        if len(values) != 1:
            raise SetupFailure(
                f"vLLM barebones expects a single sweep.{attr} value, got {values!r}; the sweep layer lands in PR-Z"
            )
        return values[0]

    def _tensor_parallelism(self, ctx) -> int:
        """C3-barebones: pick the single configured TP value."""
        return int(self._single_axis(ctx, "tensor_parallelism"))

    def _bench_results_local_path(self, ctx) -> Path:
        return Path(ctx.layout.logs_dir) / _BENCH_RESULT_FILENAME

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

        # ContainerSpec -> handle kwargs (A3); merge AITER knob-env on top.
        # Workload env (container.env, including HF_TOKEN) wins over knob env
        # on conflict -- this is the standard precedence already documented in
        # G4 NodeNetwork's merge-order contract.
        spec_kwargs = ctx.config.container.to_handle_kwargs()
        env = dict(spec_kwargs.get("env", {}))
        for key, value in self._knob_env(ctx).items():
            env.setdefault(key, value)
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

        tp = self._tensor_parallelism(ctx)
        # C3-barebones: the server script receives the TP as an env var so the
        # in-container launcher can pass `--tensor-parallel-size $TP`. PR-Z
        # lowers the swept value as an argv when the matrix layer ships.
        spec_kwargs["env"]["CVS_TP"] = str(tp)

        image = getattr(params, "container_image", None)
        if not image:
            raise SetupFailure("no container image: set params.container_image")

        handle = ContainerHandle(
            image=image,
            run_id=ctx.run_id,
            runner=ctx.executor,
            name=f"vllm_{ctx.run_id}",
            command=params.server_script,
            **spec_kwargs,
        )
        ctx.containers.append(handle)
        handle.__enter__()
        ctx.events.emit("launch.container_up", run_id=ctx.run_id, name=handle.name)

        # C1 readiness gate -- bounded poll of HTTP /health before bench.
        self._wait_for_server_ready(ctx)
        ctx.events.emit("launch.role_ready", role="server")

        # C2: dispatch the bench DETACHED inside the container so launch()
        # returns promptly. The shared-FS result file is the DONE signal
        # observed by progress_predicate.
        results_path = self._bench_results_local_path(ctx)
        ctx.scratch["results_path"] = str(results_path)
        bench_cmd = self._bench_command(ctx, results_path, handle.name)
        ctx.scratch.setdefault("commands", []).append(bench_cmd)
        ctx.executor.exec(bench_cmd)

    def _readiness_curl(self, ctx) -> str:
        """C1 probe command. Hits ``/health`` -- vLLM's readiness endpoint.

        Frameworks with a different readiness endpoint should subclass
        and override; see ``cvs/lib/adapters/AGENTS.md`` Standing
        bake-ins C1 for the convention.
        """
        params = ctx.config.params
        base = params.base_url.rstrip("/")
        return f"curl -s -o /dev/null -w '%{{http_code}}' {base}:{params.port_no}/health"

    def _server_ready(self, ctx) -> bool:
        try:
            out = ctx.executor.exec(self._readiness_curl(ctx))
        except Exception:  # noqa: BLE001 - probe failure handled by the poller
            return False
        text = out if isinstance(out, str) else "\n".join(str(v) for v in out.values())
        return "200" in text

    def _wait_for_server_ready(self, ctx) -> None:
        """Bounded poll for HTTP 200 -- raises LivenessFailure on timeout."""
        deadline = time.monotonic() + self.server_ready_timeout_s
        while True:
            if self._server_ready(ctx):
                return
            if time.monotonic() >= deadline:
                raise LivenessFailure(
                    f"vLLM server did not return HTTP 200 from /health within {self.server_ready_timeout_s}s"
                )
            time.sleep(self.server_ready_interval_s)

    def _bench_command(self, ctx, results_path: Path, container_name: str) -> str:
        """C2: full benchmark_serving flag set, run detached inside the container.

        Single-cell barebones reads concurrency / isl / osl from the
        typed ``sweep`` block (fail-closed via ``_single_axis``);
        ``ctx.param`` still wins when PR-Z lowers ``cell.params``.
        """
        params = ctx.config.params
        seq = self._single_axis(ctx, "sequence_combinations")
        # ``seq`` is a SeqCombo pydantic model in the typed path.
        seq_isl = getattr(seq, "isl", None) if not isinstance(seq, dict) else seq.get("isl")
        seq_osl = getattr(seq, "osl", None) if not isinstance(seq, dict) else seq.get("osl")
        if seq_isl is None or seq_osl is None:
            raise SetupFailure(f"vLLM: sweep.sequence_combinations[0] missing isl/osl: {seq!r}")
        default_concurrency = int(self._single_axis(ctx, "concurrency"))
        concurrency = ctx.param("concurrency", default_concurrency)
        isl = ctx.param("isl", int(seq_isl))
        osl = ctx.param("osl", int(seq_osl))
        percentile_metrics = ",".join(params.percentile_metrics)
        # ``docker exec -d`` returns immediately; the bench runs in the
        # container's background and writes the result file when done.
        # Redirect stdout/stderr inside the container so the bench log is
        # captured under the shared-FS logs_dir for post-mortem.
        bench_log = Path(ctx.layout.logs_dir) / "bench.log"
        return (
            f"docker exec -d {container_name} sh -c "
            f"\"python {params.bench_serv_script} "
            f"--backend {params.backend} "
            f"--model {ctx.config.model} "
            f"--base-url {params.base_url} --port {params.port_no} "
            f"--dataset-name {params.dataset_name} "
            f"--num-prompts {params.num_prompts} "
            f"--max-concurrency {concurrency} "
            f"--random-input-len {isl} --random-output-len {osl} "
            f"--request-rate {params.request_rate} "
            f"--burstiness {params.burstiness} "
            f"--seed {ctx.config.seed} "
            f"--tokenizer-mode {params.tokenizer_mode} "
            f"--percentile-metrics {percentile_metrics} "
            f"--metric-percentiles {params.metric_percentiles} "
            f"--save-result --result-filename {results_path} "
            f"> {bench_log} 2>&1\""
        )

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
