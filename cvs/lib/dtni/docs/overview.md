# DTNI v1 Overview

DTNI (DeTermINistic Inference) is the CVS workload-runner framework for
validating LLM inference across serving frameworks (vllm, sglang, ...). It
lives under `cvs/lib/dtni/` and is invoked by the standard `cvs run <suite>`
entrypoint; each suite resolves to a pytest test that instantiates a `Job` and
drives a fixed phase pipeline against a framework adapter.

This document is the top-of-funnel pointer. Read it cold, then jump to the
extension doc that matches what you are trying to do.

## Lifecycle

```
cvs run <suite>
   |
   v
pytest collection (cvs/lib/dtni/unittests/conftest.py + suite test file)
   |
   v
RunContext built  (executor, bindings, thresholds, config)
   |
   v
Adapter resolved via FRAMEWORK_REGISTRY[framework_name]
   |
   v
Job(adapter, ctx).run()
   |
   +--> prepare        (adapter: pull image, render configs, validate roles)
   +--> launch         (adapter: start container(s), wait for readiness)
   +--> await          (adapter: poll progress_predicate until done)
   +--> parse          (adapter: read result files -> ctx.result.scalars)
   +--> verify         (Job inline: verdict.evaluate_all vs thresholds)
   +--> teardown       (adapter: stop/cleanup; errors do not mask earlier fail)
   |
   v
JobResult -> verdict dict (PASS / FAIL / SKIP) -> artifact writer
```

`verify` is intentionally not an adapter method in v1; the Job calls
`cvs.lib.dtni.verdict.evaluate_all(ctx.thresholds, ctx.result.scalars)`
directly. Adapters only need to populate `ctx.result.scalars` during `parse`.

## File map

`cvs/lib/dtni/` (top level):

| File | Purpose |
|---|---|
| `__init__.py` | Package marker. |
| `runner.py` | Entrypoint wiring `cvs run` -> pytest -> Job. |
| `job.py` | `Job` class; the 6-phase pipeline driver. |
| `base_adapter.py` | `BaseWorkloadAdapter` ABC; role contract, container launch helpers, HTTP readiness pool. |
| `adapter_protocol.py` | Structural typing protocol adapters must satisfy. |
| `run_context.py` | `RunContext` dataclass (executor, bindings, thresholds, result, etc.). |
| `config_loader.py` | Loads + merges `config.json` / `threshold.json` from `cvs/input/dtni/...`. |
| `substitution.py` | `${VAR}` substitution for config templates. |
| `verdict.py` | `evaluate_all` / `all_passed`; threshold comparison. |
| `executor.py` | `executor_for(host).exec()` seam; LocalExecutor for unittests, SSH/cluster executor in prod. |
| `topology.py` | Role -> host binding resolution. |
| `resource_resolver.py` | GPU / port / mount resolution. |
| `container_handle.py` | Container lifecycle wrapper used by adapters. |
| `arch_detect.py` | Detect GPU arch (MI300X, MI325X, ...) for arch-gated thresholds. |
| `hashing.py` | `workload_hash` for receipts / dedup. |
| `catalog.py` | Suite/workload discovery for `cvs run`. |
| `artifact_writer.py` | Writes verdict + per-phase receipts to the run dir. |
| `errors.py` | `WorkloadError(phase=...)` and friends. |

`cvs/lib/dtni/frameworks/`:

| File | Purpose |
|---|---|
| `registry.py` | `FRAMEWORK_REGISTRY` mapping; adapters self-register on import. |
| `vllm_single_adapter.py` | Reference adapter: single-node vLLM serving. |

`cvs/lib/dtni/benchmarks/`:

| File | Purpose |
|---|---|
| `registry.py` | `BENCHMARK_REGISTRY`: dict of `BenchmarkSpec` entries (one per benchmark). |
| `harness_invokers.py` | Pure functions that build the harness command line. |
| `projectors.py` | Pure functions that map raw harness output -> scalar metrics. |
| `runner.py` | Drives a benchmark inside an adapter's launched container. |

`cvs/lib/dtni/unittests/`:

| File | Purpose |
|---|---|
| `conftest.py` | Pytest fixtures: `LocalExecutor`, fake bindings, tmp run dirs. |
| `test_*.py` | Per-module tests (topology, projectors, registry, verdict, ...). |

## Extension points

| I want to... | Read this | Files touched |
|---|---|---|
| Add a workload variant (new threshold/config under an existing framework) | `add_workload.md` | `cvs/input/dtni/<fw>/<model>/<variant>/{config,threshold}.json` |
| Add a benchmark to an existing harness | `add_benchmark.md` | `benchmarks/<harness>_specs.py` (new spec entry) |
| Add a new harness (lm_eval-style tool) | `add_harness.md` | `benchmarks/harness_invokers.py`, `benchmarks/registry.py`, `benchmarks/projectors.py` |
| Add a new framework from scratch | `framework/RUNBOOK.md` | `frameworks/<fw>_adapter.py`, `frameworks/registry.py`, new input tree |
| Distributed / disaggregated patterns | `patterns/distributed.md`, `patterns/disagg.md` | adapter + `ctx.bindings` multi-role wiring |

## Key seams

- `ctx.executor.executor_for(host).exec(cmd, ...)` is the duck-typed shell
  seam. Unittests inject `LocalExecutor`; prod uses an SSH/cluster executor.
  Adapters MUST go through this, never `subprocess` directly.
- `FRAMEWORK_REGISTRY` in `frameworks/registry.py`: adapters self-register at
  import time. A new framework is a new module + a registry entry.
- Benchmark registries (split across four files, all keyed by harness name):
  - `HARNESS_INVOKERS` in `cvs/lib/dtni/benchmarks/harness_invokers.py`
  - `_RESULT_GLOBS` in `cvs/lib/dtni/benchmarks/runner.py`
  - `PROJECTORS` in `cvs/lib/dtni/benchmarks/projectors.py`
  - `BENCHMARK_REGISTRY` (dict of `BenchmarkSpec`) in `cvs/lib/dtni/benchmarks/registry.py`
  Adding a harness means adding one entry to each of the first three; adding
  a benchmark in an existing harness means adding one `BenchmarkSpec` to the
  fourth.
- `ctx.bindings[role] -> [hosts]` is the multi-role topology view. Single-node
  adapters use one role (`"server"`); disagg/distributed adapters use multiple
  (`"prefill"`, `"decode"`, ...).
- Mandatory smoke metrics: every adapter MUST emit
  `smoke_request_latency_ms` and `smoke_completion_tokens` into
  `ctx.result.scalars` during `parse`. These guarantee the verify phase has
  something to check even when a benchmark is skipped.

## Configs

Workload configs live under:

```
cvs/input/dtni/<framework>/<model>/<variant>/
    config.json       # framework + benchmark config (templated, ${VAR}-substituted)
    threshold.json    # per-metric pass/fail thresholds, arch-gated
```

Variant naming convention is a hard rule: every variant directory MUST end in
`_perf` or `_accuracy`. Unsuffixed directories are leftover stragglers and
should be migrated, not copied. Example:

```
cvs/input/dtni/vllm_single/llama_3_1_8b/
    llama_3_1_8b_bf16_single_1_perf/
    llama_3_1_8b_bf16_single_1_accuracy/
```

## End-user documentation

User-facing docs (published to rocm.docs.amd.com) live under
`docs/reference/configuration-files/*.rst`. See `sphinx_rst.md` in this docs
directory for the RST authoring conventions and the build/preview commands.
Developer docs (this file, the extension guides, runbooks) stay in
`cvs/lib/dtni/docs/` as Markdown.
