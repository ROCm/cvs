"""Build the in-container command for each benchmark harness.

Two invokers in v1:
- ``lm-eval-harness``: accuracy (MMLU, GSM8K).
- ``vllm-bench-serve``: the perf families from the DTNI Validation Tracker
  (TTFT/TPOT/ITL/E2EL percentiles, throughput, goodput, ShareGPT trace).

Pure (no I/O) so unit-testable.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Callable

from cvs.lib.benchmarks.registry import BenchmarkSpec
from cvs.lib.errors import ConfigError


# Well-known in-container path where the adapter bind-mounts the run's host
# artifacts_dir. The harness writes results under this; runner reads them
# from the host via the same mount.
OUTPUT_DIR_IN_CONTAINER: str = "/cvs_artifacts"


@dataclass(frozen=True)
class HarnessCtx:
    """Inputs the harness needs to construct its command line.

    - ``base_url``: where the server is reachable from inside the harness's
      container. For v1 single-node, harness shares ``network=host`` with
      the server, so ``http://localhost:<port>`` is correct.
    - ``model_id``: vLLM's ``--served-model-name`` — what we POST as
      ``model``.
    - ``model_path``: in-container filesystem path to the model directory.
      Used as the tokenizer source so the harness does not try to fetch
      ``<model_id>`` from the HF hub (the id is just the API alias).
    - ``output_dir``: in-container path where the harness drops its JSON
      (bind-mounted to the host artifacts dir). Each benchmark writes into
      a ``<output_dir>/<benchmark_id>/`` subdir to keep results isolated.
    """

    base_url: str
    model_id: str
    model_path: str
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
        "pip", "install", "-q", "lm-eval[api]>=0.4.4", "&&",
        "lm_eval",
        "--model", "local-completions",
        "--model_args",
        f"base_url={hctx.base_url}/v1/completions,model={hctx.model_id},tokenizer={hctx.model_path},tokenizer_backend=huggingface,num_concurrent=8,max_retries=3",
        "--tasks", task,
        "--num_fewshot", str(spec.shots),
        "--output_path", out_path,
        "--log_samples",
    ]
    quoted = []
    for tok in parts:
        if tok == "&&":
            quoted.append(tok)
        else:
            quoted.append(shlex.quote(tok))
    return " ".join(quoted)


def _vllm_bench_serve(spec: BenchmarkSpec, hctx: HarnessCtx) -> str:
    """Build the ``vllm bench serve`` command line.

    Writes a single ``result.json`` under ``<output_dir>/<spec.id>/`` so the
    runner can locate it deterministically. The required extras are:

    - ``dataset_name``: ``random`` | ``sharegpt`` | etc.
    - ``num_prompts``: int.
    For ``random``: ``random_input_len``, ``random_output_len``.
    For ``sharegpt``: ``dataset_path`` (in-container path to the JSON).

    Optional extras:
    - ``max_concurrency``, ``request_rate``, ``percentiles``,
      ``goodput_slo`` (passed verbatim to ``--goodput`` — space-separated
      ``KEY:VALUE_ms`` pairs).
    """
    e = spec.extra
    if "dataset_name" not in e or "num_prompts" not in e:
        raise ConfigError(
            f"benchmark {spec.id!r}: vllm-bench-serve needs "
            f"extra.dataset_name and extra.num_prompts"
        )
    out_dir = f"{hctx.output_dir}/{spec.id}"
    parts: list[str] = [
        "mkdir", "-p", out_dir, "&&",
        "vllm", "bench", "serve",
        "--backend", "vllm",
        "--base-url", hctx.base_url,
        "--model", hctx.model_id,
        "--tokenizer", hctx.model_path,
        "--dataset-name", str(e["dataset_name"]),
        "--num-prompts", str(int(e["num_prompts"])),
        "--save-result",
        "--result-dir", out_dir,
        "--result-filename", "result.json",
    ]

    if e["dataset_name"] == "random":
        if "random_input_len" not in e or "random_output_len" not in e:
            raise ConfigError(
                f"benchmark {spec.id!r}: dataset_name=random needs "
                f"random_input_len and random_output_len in extra"
            )
        parts += [
            "--random-input-len", str(int(e["random_input_len"])),
            "--random-output-len", str(int(e["random_output_len"])),
        ]
    elif e["dataset_name"] == "sharegpt":
        if "dataset_path" not in e:
            raise ConfigError(
                f"benchmark {spec.id!r}: dataset_name=sharegpt needs "
                f"extra.dataset_path (in-container path to the JSON)"
            )
        parts += ["--dataset-path", str(e["dataset_path"])]

    if "max_concurrency" in e:
        parts += ["--max-concurrency", str(int(e["max_concurrency"]))]
    if "request_rate" in e:
        parts += ["--request-rate", str(e["request_rate"])]
    if "percentiles" in e:
        parts += ["--metric-percentiles", str(e["percentiles"])]
    if "goodput_slo" in e:
        # vllm splits goodput SLO strings on whitespace itself; pass each
        # KEY:VALUE pair as a separate token after --goodput per its CLI.
        slo_parts = str(e["goodput_slo"]).split()
        if slo_parts:
            parts += ["--goodput", *slo_parts]

    # quote argv tokens, leave the shell operators (`&&`, `mkdir -p ...`)
    # intact — the docker exec wrapper runs this under `bash -c`.
    quoted: list[str] = []
    for p in parts:
        if p in ("&&",):
            quoted.append(p)
        else:
            quoted.append(shlex.quote(p))
    return " ".join(quoted)


HARNESS_INVOKERS: dict[str, Callable[[BenchmarkSpec, HarnessCtx], str]] = {
    "lm-eval-harness": _lm_eval,
    "vllm-bench-serve": _vllm_bench_serve,
}


def build_command(spec: BenchmarkSpec, hctx: HarnessCtx) -> str:
    if spec.harness not in HARNESS_INVOKERS:
        raise KeyError(
            f"benchmark {spec.id!r}: unknown harness {spec.harness!r}; "
            f"known: {sorted(HARNESS_INVOKERS)}"
        )
    return HARNESS_INVOKERS[spec.harness](spec, hctx)
