# DTNI Suite Developer Guide (draft)

Status: draft, written against the `vllm_single` PoC. Conventions here are not yet enforced — they will harden as more suites port over.

This guide covers how to **port** an existing inference/training suite into the DTNI layout, or **author** a new one. Audience: CVS developers who can already run `cvs` and have touched a test or lib function, but who haven't dug deep into the inference test internals. The PoC reference is `cvs/tests/inference/vllm/vllm_single.py` plus `cvs/input/dtni/vllm_single/`.

## Background — what's familiar, what's new

**Familiar:** `cvs run <suite> --cluster_file=... --config_file=...` still works the same. `cvs list <suite>` still enumerates. Cluster files are unchanged. Pytest collection, HTML reports, `cvs exec`, all unchanged. You still write a test, point a config at it, and run it.

**New, two things:**

1. **Orchestrator handle (`orch`)** — `cvs/core/orchestrators/{baremetal,container}.py`. Replaces the ad-hoc mix of `docker_lib`, `parallel_ssh_lib`, and inline `ssh` calls that the old tests use. One handle, two methods you'll touch (`orch.exec`, `orch.exec_on_head`), and two lifecycle hooks (`orch.setup_containers`, `orch.teardown_containers`). When `container.enabled=true`, exec automatically routes inside the container — no more "did this run on the host or in the container?" guessing.
2. **Job class** — one class per framework that owns the framework-specific verbs. For vllm: `build_server_cmd`, `start_server`, `wait_ready`, `run_client`, `parse_results`, `stop_server`. Today these are spread across the test wrapper, `vllm_lib`, `docker_lib`, and `parallel_ssh_lib`. The Job class collects them into one file with named methods. **It's not magic** — it's the same code, relocated.

### Before vs after (the shape change)

**Today** — `vllm_qwen3_80b_single.py` (480 lines), with separate calls into multiple helpers:

```python
def test_cleanup_stale_containers(...): docker_lib.cleanup(...)
def test_launch_inference_containers(...): docker_lib.run(..., image=..., env=...)
def test_vllm_inference(seq_combo, concurrency, ...):
    cmd = vllm_lib.build_serve_cmd(model, params)
    parallel_ssh_lib.run(node, cmd, ...)  # start server
    vllm_lib.wait_for_health(...)
    client_cmd = vllm_lib.build_client_cmd(seq_combo, concurrency)
    parallel_ssh_lib.run(node, client_cmd, ...)
    results = vllm_lib.parse_results(...)
    inf_res_dict[(seq_combo, concurrency)] = results
    # threshold check inline, often missing
```

**After** — `vllm_single.py` (~60 lines), one test:

```python
def test_vllm_inference(orch, variant_config, hf_token, seq_combo, concurrency, inf_res_dict):
    job = VllmJob(orch=orch, model=variant_config.model, params=variant_config.params, ...)
    job.stop_server()                       # idempotent cleanup
    job.start_server(job.build_server_cmd())
    job.wait_ready()
    job.run_client(seq_combo, concurrency)
    job.wait_client_complete()
    results = job.parse_results()
    inf_res_dict[(seq_combo, concurrency)] = results
    evaluate_all(results, variant_config.thresholds)  # raises on regression
```

Container lifecycle (the old `test_cleanup_stale_containers` + `test_launch_inference_containers`) moves into the `orch` fixture — pytest setup/teardown — so it's invisible in the test body.

### A note on `InferenceBaseJob`

There's already an `InferenceBaseJob` in `cvs/lib/inference/base.py`. **New DTNI Jobs do not inherit from it.** It's an informal ABC with a few bugs (vllm-shaped env vars leaked into the base, silent-skip in `verify_inference_results`, a dead distributed branch) and other suites still depend on it, so we leave it alone. If you grep and find it, ignore for new ports — write a fresh standalone Job class.

## Mental model

A DTNI suite is three things, separated on purpose:

1. **Test file** (`cvs/tests/<domain>/<framework>/<suite>.py`) — pytest entry. Control flow only: build Job → run verbs → assert thresholds. No hardcoded paths, models, or knobs.
2. **Job class** (`cvs/lib/<domain>/<framework>_orch.py`) — framework-specific verbs. Takes an `orch` handle. No pytest, no config parsing, no filesystem layout assumptions.
3. **Variant dirs** (`cvs/input/dtni/<suite>/<variant>/{config.json, threshold.json}`) — one dir per (model × precision × purpose) tuple. Fully self-describing.

If you find yourself reaching across these layers (e.g. test reads `os.environ`, Job opens a JSON, config knows a pytest fixture name), stop and re-split.

## Step 0 — Read first

Before porting, read the source suite end-to-end and answer:
- What containers does it launch? Who launches them today (test, lib, manual)?
- What gets parametrized today? (Often: model name in the wrapper filename + sequence/concurrency in a JSON.)
- Which numbers are perf gates vs. logged-only?
- Which env vars are framework-required vs. accidental carry-over?

Write findings into a one-page port note before touching code. The vllm port surfaced 4 wrappers that differed only by model and a dead distributed branch — that observation drove the PoC shape.

## Step 1 — Move framework work into the Job class

Pull every framework-specific verb out of the test wrapper and helpers. Verbs in scope for an inference Job:

| Verb | What it owns |
|---|---|
| `build_server_cmd()` | The exact `vllm serve …` string. Reads `model`, `params`. |
| `start_server(cmd)` | Launches via `orch.exec_on_head(cmd, detach=True)`. |
| `wait_ready()` | Polls `/health`. Timeout/backoff lives here, not in the test. |
| `run_client(seq_combo, conc)` | Builds and launches `benchmark_serving.py …` |
| `wait_client_complete()` | Blocks until client returns; collects stdout/stderr. |
| `parse_results()` | Turns benchmark output into a flat dict `{ttft_ms, tpot_ms, …}`. |
| `stop_server()` | Idempotent. Used both for pre-test cleanup and teardown. |

**Out of scope for the Job:**
- Container lifecycle (orch's job).
- Threshold evaluation (test's job, via `evaluate_all`).
- Result table formatting (a shared test in `_shared.py`).
- Reading config files (the fixture's job).

## Step 2 — Use `conftest.py` and `_shared.py`

Duplication across suites in the same framework family (e.g. `vllm_single` and a future `vllm_distributed`) goes in two places:

- `cvs/tests/<domain>/<framework>/conftest.py` — **fixtures only**. The PoC has: `cluster_dict`, `variant_config`, `orch` (constructs `OrchestratorConfig`, calls `setup_containers()`, yields, calls `teardown_containers()`), `hf_token`, `inf_res_dict`, plus `pytest_generate_tests` for variant parametrization. Pure plumbing — no schema interpretation, no test logic.
- `cvs/tests/<domain>/<framework>/_shared.py` — **tests** that every suite in the family inherits via `from ._shared import *` (e.g. `test_print_results_table`). Keep small; if a "shared" test grows a conditional on suite name, it isn't shared.

**Anti-pattern:** schema knowledge in `conftest.py`. Different suites may want different slices of the config (a hardware-only test reads `paths` but not `benchmark_params`). The fixture loads and validates; each test pulls what it needs.

## Step 3 — Config vs threshold: the philosophy

This split is the load-bearing convention of the DTNI layout. Keep it strict.

**`config.json`** answers *"what are we running?"* — identity, inputs, knobs, infrastructure.
**`threshold.json`** answers *"did it pass?"* — pass/fail predicates per metric.

**Why separate files, not one:**
- **Different churn rates.** Tuning a perf gate after a regression is a different change than swapping a model or bumping an image tag. Separate files = separate diffs = easier review.
- **Different ownership.** A perf engineer owns thresholds; an integrator owns the config. Different reviewers, different cadence.
- **Different lifecycle.** A config without thresholds is meaningful (smoke runs, debug, "just produce numbers"). A threshold file without a config is not.
- **`cvs list` granularity.** One variant = one directory = one row in enumeration.
- **Rejects v1's worst anti-pattern.** v1 had non-metric checks encoded as numeric thresholds (e.g. "did the container start" as `min: 1`). With the split, the threshold file is by definition only about measured metrics; non-metric checks belong in the test as `assert`.

### Concrete shape: `config.json`

Real example (`cvs/input/dtni/vllm_single/Qwen3-Next-80B-A3B-Instruct_perf/config.json`):

```json
{
  "schema_version": 1,
  "framework": "vllm_single",
  "gpu_arch": "mi355x",
  "paths": {
    "shared_fs":     "/mnt/dtni/{user-id}/cvs",
    "models_dir":    "/mnt/dtni/{user-id}/models",
    "datasets_dir":  "{shared_fs}/datasets",
    "artifacts_dir": "{shared_fs}/artifacts"
  },
  "model": {
    "id": "Qwen3-Next-80B-A3B-Instruct",
    "remote": 0,
    "precision": "bf16"
  },
  "image": {
    "tag": "rocm/vllm:latest",
    "remote": 1
  },
  "container": {
    "enabled": true,
    "launch": true,
    "name": "vllm_inference_rocm",
    "runtime": {
      "name": "docker",
      "args": {
        "volumes": {"{paths.models_dir}": "/models"},
        "env": {"VLLM_USE_TRITON_FLASH_ATTN": "0"},
        "shm_size": "64G"
      }
    }
  },
  "params": {
    "tensor_parallelism": 8,
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.85
  },
  "benchmark_params": {
    "concurrency_levels": [16],
    "sequence_combinations": [
      {"name": "balanced", "isl": 1024, "osl": 1024}
    ]
  }
}
```

Blocks, in order:
- **Identity** (`schema_version`, `framework`, `gpu_arch`) — fixed strings; loader checks them.
- **Paths** — file locations with placeholder substitution. `{user-id}` resolves from cluster_dict; `{shared_fs}` is a self-reference resolved in a second pass.
- **Model / image** — what to serve, where it comes from. `remote: 0` = pre-staged in `models_dir`, `remote: 1` = HF download (schema accepts, not yet implemented).
- **Container** — passed through to `OrchestratorConfig`. `launch: true` means orch owns the container. `runtime.args` is the only place `extra="allow"` applies (runtime-specific args).
- **Params** — framework flags (`vllm serve …`).
- **Benchmark_params** — client flags (sweep dimensions). The PoC parametrizes `pytest_generate_tests` over the Cartesian product of `concurrency_levels × sequence_combinations`.

### Concrete shape: `threshold.json`

Real example, same variant:

```json
{
  "smoke_request_latency_ms":          {"kind": "max_ms", "value": 600000},
  "smoke_completion_tokens":           {"kind": "min",    "value": 1},

  "balanced_conc16.request_throughput":  {"kind": "min",    "value": 0.5},
  "balanced_conc16.output_throughput":   {"kind": "min_tok_s", "value": 50.0},
  "balanced_conc16.ttft_p95_ms":         {"kind": "max_ms", "value": 60000},
  "balanced_conc16.tpot_p95_ms":         {"kind": "max_ms", "value": 5000}
}
```

- **Flat namespace.** Keys are `<variant-cell>.<metric>` for sweep cells, bare for one-off smoke checks.
- **Five predicate kinds:** `min`, `max_ms`, `within` (value ± tolerance), `min_tok_s`, `min_ratio`. Each entry is `{kind, value, tolerance?}` — explicit, not a magic number.
- **No identity, no paths, no config knobs.** If you find them creeping in, you're recreating v1's mistakes.
- **Missing entries are allowed.** A metric without a threshold is logged but not gated. Useful for "watch this number, don't fail on it yet."

### Anti-patterns to reject in review

- A `config.json` key whose value is a pass/fail threshold ("max_latency_ms": 5000). Move it to `threshold.json`.
- A `threshold.json` entry that's actually a config flag ("tensor_parallelism": 8). Move it to `config.json`.
- A threshold that branches on hardware inside the file ("if gpu_arch == mi300x then 50 else 80"). Split into separate variant dirs (`<model>_mi300x_perf`, `<model>_mi355x_perf`).
- Substituting one file's values into the other with placeholders. They are different lifecycle, do not couple them.
- A "smoke" threshold of `min: 1` standing in for "did the thing start." If it's binary, assert in the test.

## Step 4 — Variant directories and naming

`cvs/input/dtni/<suite>/<variant-name>/{config.json, threshold.json}`

Naming convention used in the PoC: `<full-model-id>_<purpose>` where purpose is `perf` or `accuracy`. Full model ID (e.g. `Qwen3-Next-80B-A3B-Instruct_perf`) so that `cvs list` output is self-describing. No abbreviations — `qwen3_80b` collides with future Qwen 3.x 80B variants.

One variant per directory. Resist the urge to glob multiple models into one config with a `models: [...]` array — `cvs list` granularity drops and per-model thresholds become a switch statement.

## Step 5 — Typed config loading

Use Pydantic models with `extra="forbid"` at every level except the orchestrator passthrough (`container.runtime.args` uses `extra="allow"` because runtime args are runtime-specific). `extra="forbid"` catches typos at load time, not deep in the run. The v1 spec called out `percentiles_metrics` vs `percentile_metrics` as a class of bug this prevents.

Placeholder substitution happens in the loader, in a fixed order:
1. Cluster-derived (`{user-id}`, `{home-mount-dir}`) from `cluster_dict`.
2. Self-reference (`{shared_fs}` inside `paths.*`) from already-resolved keys.
3. Cross-block (`{paths.models_dir}` inside `container.runtime.args.volumes`).

Document the substitution order in the loader's docstring. Out-of-order references are a load-time error, not a runtime surprise.

## Step 6 — Verification before merge

Per the planning discipline, the PR plan must include concrete runnable checks. For a suite port, minimum set:

1. `pytest --collect-only` resolves to the expected parametrized IDs for one variant.
2. `cvs list <suite>` enumerates all variants.
3. One end-to-end run on real hardware producing numbers within plus/minus 10% of the pre-port baseline (cite the artifact zip).
4. Hardware-side check that the container appears at `setup_containers()` and disappears at `teardown_containers()` — confirms lifecycle moved to orch.
5. A negative test: missing model path / typo'd config key — expect a clean validation error, not a deep crash.

If you cannot point at a pre-port baseline, say so before merging — coverage regression in CVS is hard to spot because pass/fail is not always trustworthy (see `cvs-runs.md`).

## Step 7 — What stays out

Apply small-PR discipline. One suite per PR. The Out-of-scope section is where "while we're here" work goes — common temptations:

- Refactor `InferenceBaseJob` / delete old wrappers' shared lib. Do not. Other suites still use it.
- Add a `cvs migrate-config` tool. Hand-write variants for the first 2–3 ports — the tool's contract is unclear until you've felt the friction.
- `model.remote=1` (HF auto-download). Schema accepts it but raises `NotImplementedError`. Port from cvs-dtni-v1's `resource_resolver.py` when a suite actually needs it.
- Accuracy variants. Land perf first, then accuracy as a separate variant directory + test function.
- Sweep semantics rework. The PoC keeps `(seq_combo × concurrency)` from the old shape; a richer sweep DSL can come later.

## Open conventions (not yet decided)

- New Job class naming — `VllmJob` in a new module vs `VllmOrchJob` during transition. Pick one before the second suite ports.
- Where the per-suite port note lives — `plans/` (informal) vs `docs/dev/` (published). Defer until 2nd port.
- Stub tests for fixture-replaced lifecycle steps (e.g. `test_launch_inference_containers`) — keep for report shape, or accept they vanish. PoC open question.

## Reference

- PoC plan: `plans/vllm-single-orch-poc.md`
- Orchestrator surface: `cvs/core/orchestrators/{base,baremetal,container}.py`, `cvs/core/orchestrators/factory.py`
- Original v1 spec (rejected as "too many things at once" — kept as reference for typed configs and threshold predicates): on the `dev/dtni-v1` worktree at `docs/prd/cvs-dtni-v1-spec.md`
