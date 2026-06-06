"""Run accuracy benchmarks against a healthy server, project scores to scalars.

Called from the adapter's ``parse()`` once the server is up. For each
benchmark, the harness runs inside the server container (which has the host
artifacts_dir bind-mounted at ``OUTPUT_DIR_IN_CONTAINER`` by the adapter at
launch time) and drops a ``results_<ts>.json`` file in a per-benchmark
subdir. We glob the host side of that mount, project the configured metric
to ``ctx.result.scalars`` under the benchmark id so ``threshold.json`` can
name it directly:

    scalars["mmlu"]  = 0.732   ← matches `"mmlu": {"kind": "min", "value": 0.7}`
    scalars["gsm8k"] = 0.815
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

from cvs.lib.benchmarks.harness_invokers import HarnessCtx, build_command
from cvs.lib.benchmarks.registry import BenchmarkSpec, lookup
from cvs.lib.errors import WorkloadError


# Per-benchmark wall-clock ceiling. MMLU on a smaller model can take ~2h;
# this gives headroom. Total parse-phase budget is N benchmarks * this.
PER_BENCHMARK_TIMEOUT_S: int = 4 * 3600


def run_benchmarks(
    *,
    benchmarks: list[str],
    server_handle,
    base_url: str,
    model_id: str,
    output_dir_in_container: str,
    output_dir_on_host: Path,
) -> dict[str, float]:
    """Run each benchmark id, return flat scalars dict for ctx.result.scalars.

    ``server_handle`` is the ContainerHandle whose runner.exec we use to
    ``docker exec`` the harness inside the same container as the server (the
    rocm/vllm image ships ``lm_eval``).
    """

    scalars: dict[str, float] = {}
    for bid in benchmarks:
        spec = lookup(bid)
        hctx = HarnessCtx(
            base_url=base_url,
            model_id=model_id,
            output_dir=output_dir_in_container,
        )
        harness_cmd = build_command(spec, hctx)
        docker_cmd = (
            f"docker exec {shlex.quote(server_handle.name)} "
            f"bash -c {shlex.quote(harness_cmd)}"
        )
        try:
            server_handle.runner.exec(docker_cmd, timeout=PER_BENCHMARK_TIMEOUT_S)
        except Exception as exc:
            raise WorkloadError(f"benchmark {bid!r} harness failed: {exc}") from exc

        result_json = _find_lm_eval_result(output_dir_on_host / bid)
        if result_json is None:
            raise WorkloadError(
                f"benchmark {bid!r}: harness ran but produced no results JSON under "
                f"{output_dir_on_host / bid}"
            )
        try:
            payload = json.loads(result_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise WorkloadError(
                f"benchmark {bid!r}: harness output {result_json} is not valid JSON: {exc}"
            ) from exc

        scalars.update(_project_scalars(spec, payload))

    return scalars


def _find_lm_eval_result(subdir: Path) -> Path | None:
    """Return the newest ``results*.json`` under ``subdir`` (recursive), or None.

    lm-eval-harness writes ``<output_path>/results_<timestamp>.json`` (and
    sometimes nests under a model-name dir). We pick the most recently
    modified match to be robust to either layout.
    """
    if not subdir.exists():
        return None
    matches = sorted(
        subdir.rglob("results*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def _project_scalars(spec: BenchmarkSpec, payload: dict[str, Any]) -> dict[str, float]:
    """Extract spec.score_metric from lm-eval JSON under the benchmark id key.

    lm-eval writes ``{"results": {task: {metric: value, metric_stderr: ...}}}``.
    Modern lm-eval suffixes the metric name with ``,<filter>`` (e.g.
    ``acc,none`` or ``exact_match,strict-match``). We try, in order:
      1) the spec's preferred filter (``score_filter``)
      2) the bare metric key (legacy)
      3) ``,none`` (default filter slug)
    Also surfaces ``<id>_stderr`` for diagnostics using the same lookup order.
    """

    out: dict[str, float] = {}
    results = payload.get("results") or {}
    task = spec.extra.get("task", spec.id)
    task_results = results.get(task, {})

    score = _pick_metric(task_results, spec.score_metric, spec.score_filter)
    if isinstance(score, (int, float)):
        out[spec.id] = float(score)

    stderr = _pick_metric(task_results, f"{spec.score_metric}_stderr", spec.score_filter)
    if isinstance(stderr, (int, float)):
        out[f"{spec.id}_stderr"] = float(stderr)

    return out


def _pick_metric(task_results: dict[str, Any], metric: str, preferred_filter: str | None) -> Any:
    """Walk filter-suffix variants for ``metric`` in priority order."""
    candidates: list[str] = []
    if preferred_filter:
        candidates.append(f"{metric},{preferred_filter}")
    candidates.append(metric)
    candidates.append(f"{metric},none")
    for key in candidates:
        if key in task_results:
            return task_results[key]
    return None
