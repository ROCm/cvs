"""Build the in-container command for a benchmark harness.

One invoker in v1: ``lm-eval-harness`` against an OpenAI-compatible endpoint.
Pure (no I/O) so unit-testable.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Callable

from cvs.lib.benchmarks.registry import BenchmarkSpec


# Well-known in-container path where the adapter bind-mounts the run's host
# artifacts_dir. The harness writes results under this; runner reads them
# from the host via the same mount.
OUTPUT_DIR_IN_CONTAINER: str = "/cvs_artifacts"


@dataclass(frozen=True)
class HarnessCtx:
    """Inputs the harness needs to construct its command line.

    - ``base_url``: where the server is reachable from inside the harness's
      container. For v1 single-node, harness shares ``network=host`` with the
      server, so ``http://localhost:<port>`` is correct.
    - ``model_id``: vLLM's ``--served-model-name`` — what we POST as ``model``.
    - ``output_dir``: in-container path where the harness drops its JSON
      (bind-mounted to the host artifacts dir). Each benchmark writes into a
      ``<output_dir>/<benchmark_id>/`` subdir to keep results isolated.
    """

    base_url: str
    model_id: str
    output_dir: str


def _lm_eval(spec: BenchmarkSpec, hctx: HarnessCtx) -> str:
    """Build the lm_eval command line.

    ``--output_path`` is a DIRECTORY for lm-eval-harness >=0.4: it writes
    ``results_<timestamp>.json`` inside it. We give each benchmark its own
    subdir so the runner can find that file unambiguously.
    """
    task = spec.extra.get("task", spec.id)
    out_path = f"{hctx.output_dir}/{spec.id}"
    parts = [
        "lm_eval",
        "--model", "local-completions",
        "--model_args",
        f"base_url={hctx.base_url}/v1/completions,model={hctx.model_id},num_concurrent=8,max_retries=3",
        "--tasks", task,
        "--num_fewshot", str(spec.shots),
        "--output_path", out_path,
        "--log_samples",
    ]
    return " ".join(shlex.quote(p) for p in parts)


HARNESS_INVOKERS: dict[str, Callable[[BenchmarkSpec, HarnessCtx], str]] = {
    "lm-eval-harness": _lm_eval,
}


def build_command(spec: BenchmarkSpec, hctx: HarnessCtx) -> str:
    if spec.harness not in HARNESS_INVOKERS:
        raise KeyError(
            f"benchmark {spec.id!r}: unknown harness {spec.harness!r}; "
            f"known: {sorted(HARNESS_INVOKERS)}"
        )
    return HARNESS_INVOKERS[spec.harness](spec, hctx)
