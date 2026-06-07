# config.json and threshold.json — schema reference

## 1. Scope

This document defines the on-disk schemas for a DTNI workload variant directory:
`config.json` (workload spec), `threshold.json` (per-metric pass/fail), the
substitution-token grammar, the mandatory smoke metrics, and the verdict kinds.
It does not cover authoring recipes (see `add_workload.md`) or end-to-end
operation (see `framework/RUNBOOK.md`). Source of truth: `cvs/lib/dtni/config_loader.py`,
`cvs/lib/dtni/substitution.py`, `cvs/lib/dtni/verdict.py`.

## 2. `config.json` schema

Validated by Pydantic v2 (`WorkloadConfig` in `cvs/lib/dtni/config_loader.py`).
Unknown fields are rejected; missing required fields raise `ConfigError`.

| Field | Type | Required | Semantics |
|---|---|---|---|
| `schema_version` | `Literal[1]` | yes | Schema version; only `1` is accepted. |
| `framework` | `str` | yes | Adapter family (e.g. `vllm_single`, `sglang_disagg`). |
| `gpu_arch` | `str` | yes | One of `cvs.lib.dtni.arch_detect.VALID_ARCHES` (e.g. `mi300x`). |
| `paths` | `PathsBlock` | yes | Boilerplate; see below. |
| `model` | `ModelBlock` | yes | `{id, remote, precision?, hf_token_env?}`; `id` must exist in catalog. |
| `dataset` | `DatasetBlock` | no | `{id, remote, hf_token_env?}`; `id` must exist in catalog. |
| `benchmarks` | `list[str]` | no | Each id must exist in `BENCHMARK_REGISTRY` (catalog cross-check). |
| `benchmark_datasets_remote` | `0`/`1` | no | Whether benchmark datasets are fetched at run time. |
| `image` | `ImageBlock` | conditional | Top-level container image; required unless every role declares its own `image`. |
| `topology` | `TopologyBlock` | yes | `{roles: {<name>: {count, gpus_per_node}}}`. |
| `roles` | `dict[str, RoleSpec]` | yes | Keys MUST match `topology.roles` exactly. |
| `params` | `dict[str, Any]` | no | Free-form scalars exposed as `{params.<key>}` tokens. |

### 2.1 `paths` block (boilerplate)

Five keys; identical across every variant. Copy verbatim from any sibling
variant when authoring a new one:

```json
"paths": {
  "shared_fs":    "/mnt/dtni/{user-id}/cvs",
  "models_dir":   "/mnt/dtni/{user-id}/models",
  "datasets_dir": "{shared_fs}/datasets",
  "artifacts_dir":"{shared_fs}/artifacts",
  "images_dir":   "{shared_fs}/images"
}
```

Tech debt: these should become Pydantic defaults on `PathsBlock` so variants
can omit the block entirely. Until then, the five keys are mandatory.

### 2.2 Role spec

`RoleSpec` fields (all optional except `command`): `shm_size`, `devices`,
`volumes`, `env`, `port`, `health_path`, `command`, `seccomp_unconfined`,
`extra_args`, `image` (per-role override for disaggregated topologies).
`command` is the single shell string executed inside the container after
token substitution.

## 3. Substitution tokens

Implementation: `cvs/lib/dtni/substitution.py`. Tokens use `{name}` syntax;
unknown tokens raise `ValueError` (fail-loud). Resolution is recursive up to
depth 8.

| Token | Resolves to | Resolved at |
|---|---|---|
| `{shared_fs}` | `paths.shared_fs` after fixed-point | config-load (paths block) |
| `{models_dir}` | `paths.models_dir` | config-load |
| `{datasets_dir}` | `paths.datasets_dir` | config-load |
| `{artifacts_dir}` | `paths.artifacts_dir` | config-load |
| `{images_dir}` | `paths.images_dir` | config-load |
| `{user-id}` | Cluster `username` (or `getpass.getuser()` fallback) | config-load (paths) + job-runtime (context) |
| `{run_id}` | The job's run identifier | job-runtime |
| `{model.id}` | `model.id` from the model block | job-runtime |
| `{model.path}` | Resolved local model directory | job-runtime |
| `{model.precision}` | `model.precision` from the model block | job-runtime |
| `{params.<key>}` | Any scalar key from `params` (arbitrary) | job-runtime |

Two-phase resolution:

1. **Config-load**: `resolve_paths_block` runs a fixed-point pass over the
   five `paths` keys, seeded with `{user-id}` from the cluster file. Output is
   the fully-resolved `paths` dict.
2. **Job-runtime**: `build_context` assembles the per-job substitution ctx
   from resolved paths, `run_id`, `user-id`, the `model` block, and every
   `params.<key>`. Token values must be scalar (`str`/`int`/`float`); resolving
   to a non-scalar raises.

## 4. Perf vs accuracy variants — HARD convention

Variant directory names MUST end in `_perf` or `_accuracy`. The suffix is a
contract observed by the harness and by downstream report tooling.

| Suffix | Threshold focus | Typical benchmarks | Run length |
|---|---|---|---|
| `_perf` | Latency ceilings (`max_ms`), throughput floors (`min`, `min_tok_s`) | `serve_synth_short`, `serve_synth_long` | Short |
| `_accuracy` | Benchmark scores (`min` on `mmlu`, `gsm8k`, etc.) | `mmlu`, `gsm8k`, ... | Longer |

Unsuffixed directories (e.g. `qwen3_next_80b_bf16_single_8/` with no
`_perf`/`_accuracy` tail) are stragglers from earlier authoring and should be
renamed or removed; do not pattern-match them as a template.

## 5. `threshold.json` schema

Top-level object: `{<metric_name>: {kind, ...}}`. Loaded by
`load_thresholds`; each entry is checked for required keys at load time.
Evaluation is in `cvs/lib/dtni/verdict.py`; `VALID_KINDS` is the authoritative
list.

| `kind` | Comparison | Extra keys | Used for |
|---|---|---|---|
| `min` | `actual >= value` | `value` | Accuracy scores, throughput floors, completion counts |
| `max_ms` | `actual <= value` | `value` | Latency ceilings (request latency, TTFT, TPOT) |
| `within` | `lo <= actual <= hi` | `lo`, `hi` | Exact-ish numerics; reject drift in either direction |
| `min_tok_s` | `actual >= value` | `value` | Token-throughput floors (semantic alias of `min`) |
| `min_ratio` | `actual >= value` | `value` | Ratio metrics (e.g. goodput / submitted) |

A metric not produced by the adapter yields a failing `Verdict` with
`note="metric not produced by adapter"` — there is no "soft-miss" mode.

## 6. Mandatory smoke metrics

> **Every adapter MUST emit both, and `threshold.json` MUST score both:**
>
> - **`smoke_request_latency_ms`** — `max_ms` threshold. End-to-end latency
>   of the smoke request the adapter sends after launch.
> - **`smoke_completion_tokens`** — `min` threshold. Token count returned by
>   the smoke request.
>
> Rationale: the smoke guard exists to catch "container came up, served HTTP,
> produced empty output" failures that pass every benchmark-level threshold
> by vacuously emitting nothing. If you skip these, a silently-broken role
> ships green.

Reference values from a healthy `vllm_single` variant: `max_ms=600000`,
`min=1`.

## 7. Catalog coupling

Every string in `benchmarks: [...]` must be a key in `BENCHMARK_REGISTRY`
(see `cvs/lib/dtni/benchmarks/registry.py`). Unknown ids are rejected at
config-load with a `difflib` "did you mean ...?" hint. To enumerate the
currently-registered ids:

```bash
grep 'BenchmarkSpec(' cvs/lib/dtni/benchmarks/registry.py
```

Adding a new benchmark id to a `config.json` without registering it in
`BENCHMARK_REGISTRY` will fail load with `CatalogError`. The same rule
applies to `model.id` (catalog `models`) and `dataset.id` (catalog `datasets`).

## 8. Validation flow

`load_workload` calls `WorkloadConfig.model_validate(raw)` (Pydantic v2) then
runs three cross-checks: `model.id` in catalog, `dataset.id` in catalog (if
present), every `benchmarks[i]` in `BENCHMARK_REGISTRY`. Pydantic errors are
re-raised as `ConfigError(f"{path}: {exc}")`; catalog misses raise
`CatalogError` with a suggestion. Typical messages: `field required`
(missing key), `Input should be a valid integer` (wrong type),
`unknown token: {foo}` (raised later from `substitution.py` when the bad
token is referenced).
