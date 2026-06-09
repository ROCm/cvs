# DTNI Suite Developer Guide (draft)

Status: draft, written against the `vllm_single` PoC. Conventions here are not yet enforced — they will harden as more suites port over.

This guide covers how to **port** an existing inference/training suite into the DTNI layout, or **author** a new one. The PoC reference is `cvs/tests/inference/vllm/vllm_single.py` plus `cvs/input/dtni/vllm_single/`.

## Mental model

A DTNI suite is three things, separated on purpose:

1. **Test file** (`cvs/tests/<domain>/<framework>/<suite>.py`) — pytest entry. Owns control flow only: build job → run → assert thresholds. No hardcoded paths, models, or knobs.
2. **Job class** (`cvs/lib/<domain>/<framework>_orch.py`) — framework-specific verbs (`build_server_cmd`, `start_server`, `wait_ready`, `run_client`, `parse_results`). Takes an `orch` handle. No pytest, no config parsing, no filesystem layout assumptions.
3. **Variant dirs** (`cvs/input/dtni/<suite>/<variant>/{config.json, threshold.json}`) — one dir per (model × precision × purpose) tuple. Fully self-describing.

If you find yourself reaching across these layers (e.g. test reads `os.environ`, Job opens a JSON, config knows a pytest fixture name), stop and re-split.

## Step 0 — Read first

Before porting, read the source suite end-to-end and answer:
- What containers does it launch? Who launches them today (test, lib, manual)?
- What gets parametrized today? (Often: model name in the wrapper filename + sequence/concurrency in a JSON.)
- Which numbers are perf gates vs. logged-only?
- Which env vars are framework-required vs. accidental carry-over?

Write findings into a one-page port note before touching code. The vllm port surfaced 4 wrappers that differed only by model and a dead distributed branch — that observation drove the PoC shape.

## Step 1 — Use the orchestrator

The orchestrator (`cvs/core/orchestrators/{baremetal,container}.py`) is new. If you have not used it before, the contract is:

- `orch.exec(cmd, nodes=...)` and `orch.exec_on_head(cmd)` are how you run shell. They route into the container automatically when `container.enabled=true`.
- `orch.setup_containers()` / `orch.teardown_containers()` own the full container lifecycle. **Do not** call `docker run`, `docker_lib`, or `parallel_ssh_lib` from your Job class.
- `OrchestratorConfig` (see `cvs/core/orchestrators/factory.py`) is built from cluster identity + a `container` block. The PoC builds it inside the `orch` fixture from `variant_config.container.dict()`.
- Set `container.launch=true` to have the orchestrator manage the container. The old `launch=false` pattern (Job launches its own container) is being phased out for DTNI suites.

If your Job inherits `InferenceBaseJob`, you are on the old path. New DTNI Jobs **do not inherit** `InferenceBaseJob` — they take `orch` and call it directly. `InferenceBaseJob` stays for non-ported suites until those port too.

## Step 2 — Use `conftest.py` and a `_shared.py`

Duplication across suites in the same framework family (e.g. `vllm_single` and a future `vllm_distributed`) goes in two places:

- `cvs/tests/<domain>/<framework>/conftest.py` — fixtures only. The PoC has: `cluster_dict`, `variant_config`, `orch`, `hf_token`, `inf_res_dict`, plus `pytest_generate_tests` for variant parametrization. Pure plumbing — no schema interpretation, no test logic.
- `cvs/tests/<domain>/<framework>/_shared.py` — tests that every suite in the family inherits via `from ._shared import *` (e.g. `test_print_results_table`). Keep this small; if a "shared" test grows a conditional on suite name, it isn't shared.

**Anti-pattern:** putting schema knowledge in `conftest.py`. Different suites may want different slices of the config (a hardware-only test reads `paths` but not `benchmark_params`). Let each test pull what it needs from `variant_config` — the fixture only loads and validates.

## Step 3 — Split config from thresholds

This split is the load-bearing convention of the DTNI layout. Keep it strict.

**`config.json`** answers *"what are we running?"*
- Identity: `framework`, `gpu_arch`, `schema_version`.
- Inputs: `model {id, remote, precision}`, `image {tag, remote}`, `paths`.
- Knobs: `params` (framework flags), `benchmark_params` (client flags), `sweep` (which combos to run).
- Infrastructure: `container` block (passed through to `OrchestratorConfig`).

**`threshold.json`** answers *"did it pass?"*
- A flat list of predicates keyed by metric name. Five kinds: `min`, `max_ms`, `within`, `min_tok_s`, `min_ratio`.
- Each entry is `{kind, value, tolerance?}` — explicit, not a magic-encoded number.
- Lives next to `config.json` so a variant is one directory.

**Why separate?**
- Thresholds churn far more than configs (tuning a perf gate is not the same as changing the run). Separate files mean separate review and diff noise.
- A config without thresholds is still meaningful (smoke runs, debug). A threshold file without a config is not.
- One variant = one directory keeps `cvs list <suite>` enumeration trivial (walk the dir).
- Forbids the v1 anti-pattern of encoding non-metric checks as numeric thresholds (e.g. "did the container start" as `min: 1`).

**Anti-patterns to reject in review:**
- A "config" key whose value is a pass/fail threshold (move it).
- A threshold that branches on hardware (split the variant dir instead).
- Substituting one with placeholders from the other (different lifecycle, do not couple them).

## Step 4 — Variant directories and naming

`cvs/input/dtni/<suite>/<variant-name>/{config.json, threshold.json}`

Naming convention used in the PoC: `<full-model-id>_<purpose>` where purpose is `perf` or `accuracy`. Full model ID (e.g. `Qwen3-Next-80B-A3B-Instruct_perf`) so that `cvs list` output is self-describing. No abbreviations — `qwen3_80b` collides with future Qwen 3.x 80B variants.

One variant per directory. Resist the urge to glob multiple models into one config file with a `models: [...]` array — `cvs list` granularity drops, and per-model thresholds become a switch statement.

## Step 5 — Typed config loading

Use Pydantic models with `extra="forbid"` at every level except the orchestrator passthrough (`container.runtime.args` uses `extra="allow"` because runtime args are runtime-specific).

`extra="forbid"` catches typos at load time, not deep in the run. The v1 spec called out `percentiles_metrics` vs `percentile_metrics` as a class of bug this prevents.

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
- Original v1 spec (rejected as "too many things at once" — kept as reference for typed configs and threshold predicates): `docs/prd/cvs-dtni-v1-spec.md` (on a separate branch)
