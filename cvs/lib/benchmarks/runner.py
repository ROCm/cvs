"""Run benchmarks against a healthy server, project scores to scalars.

Called from the adapter's ``parse()`` once the server is up. For each
benchmark, the harness runs inside the server container (which has the host
artifacts_dir bind-mounted at ``OUTPUT_DIR_IN_CONTAINER`` by the adapter at
launch time) and drops its result JSON in a per-benchmark subdir. The
runner finds it, calls the per-harness projector, and merges the resulting
scalars into ``ctx.result.scalars`` so ``threshold.json`` can name them
directly. Examples:

    scalars["mmlu"]                              = 0.732  # lm-eval
    scalars["serve_synth_short.ttft_p95_ms"]     = 84.1   # vllm-bench-serve
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path

from cvs.lib.benchmarks.harness_invokers import HarnessCtx, build_command
from cvs.lib.benchmarks.projectors import project
from cvs.lib.benchmarks.registry import lookup
from cvs.lib.errors import WorkloadError


# Per-benchmark wall-clock ceiling. MMLU on a smaller model can take ~2h;
# this gives headroom. Total parse-phase budget is N benchmarks * this.
PER_BENCHMARK_TIMEOUT_S: int = 4 * 3600

# Result-file glob per harness. lm-eval writes ``results_<ts>.json`` (and
# sometimes nests under a model dir); vllm bench serve writes the exact
# ``result.json`` we asked for via --result-filename.
_RESULT_GLOBS: dict[str, str] = {
    "lm-eval-harness":  "results*.json",
    "vllm-bench-serve": "result.json",
}


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
    ``docker exec`` the harness inside the same container as the server
    (the rocm/vllm image ships both ``lm_eval`` and ``vllm bench serve``).
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

        glob_pat = _RESULT_GLOBS.get(spec.harness, "*.json")
        result_json = _find_result(output_dir_on_host / bid, glob_pat)
        if result_json is None:
            raise WorkloadError(
                f"benchmark {bid!r}: harness ran but produced no result JSON "
                f"({glob_pat}) under {output_dir_on_host / bid}"
            )
        try:
            payload = json.loads(result_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise WorkloadError(
                f"benchmark {bid!r}: harness output {result_json} is not "
                f"valid JSON: {exc}"
            ) from exc

        scalars.update(project(spec, payload))

    return scalars


def _find_result(subdir: Path, glob_pat: str) -> Path | None:
    """Return the newest matching JSON under ``subdir`` (recursive), or None."""
    if not subdir.exists():
        return None
    matches = sorted(
        subdir.rglob(glob_pat),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None
