# Add a workload variant

## Goal

Add a new variant (e.g. `_perf` or `_accuracy`) under an existing framework + model in DTNI. This recipe does NOT cover new frameworks (see `RUNBOOK.md`), new benchmarks (see `add_benchmark.md`), or new harnesses (see `add_harness.md`).

## Inputs (decide before starting)

1. **Framework** — must already exist in `cvs/lib/dtni/frameworks/` (e.g. `vllm_single`).
2. **Model** — must already have a directory under `cvs/input/dtni/<framework>/` (e.g. `llama_3_1_8b`).
3. **Variant kind** — `_perf` (latency/throughput thresholds) OR `_accuracy` (benchmark-score thresholds). Hard convention; the suffix must end in one of these two tokens.
4. **Variant suffix** — descriptive token before the kind, e.g. `_mmlu_only_accuracy`, `_bf16_single_8_perf`.

## Files to create

```
cvs/input/dtni/<framework>/<model>/<model>_<variant>/
  config.json
  threshold.json
```

Copy from the nearest sibling variant directory; do not start from scratch.

## `config.json` template

```json
{
  "schema_version": 1,
  "framework": "<framework>",
  "gpu_arch": "mi300x",
  "paths": {
    "shared_fs":     "/mnt/dtni/{user-id}/cvs",
    "models_dir":    "/mnt/dtni/{user-id}/models",
    "datasets_dir":  "{shared_fs}/datasets",
    "artifacts_dir": "{shared_fs}/artifacts",
    "images_dir":    "{shared_fs}/images"
  },
  "model":   { "id": "<hf-id>", "remote": 0, "precision": "bf16" },
  "benchmarks": ["<bench_id>", "..."],
  "benchmark_datasets_remote": 1,
  "image":   { "tag": "rocm/vllm:latest", "remote": 1 },
  "topology": { "roles": { "server": { "count": 1, "gpus_per_node": 1 } } },
  "roles":   { "server": { "...": "see sibling" } },
  "params":  { "tensor_parallelism": 1, "input_len": 128, "output_len": 256, "concurrency": 32 }
}
```

### Section notes

- **`paths`** — boilerplate. Copy the 5 lines verbatim from any sibling config. (Duplication across every variant is known tech debt; do not edit values.)
- **`model.id`** — HuggingFace id (used as `{model.id}` in the role command).
- **`model.remote`** — `0` means already present in `models_dir`; `1` means fetch.
- **`benchmarks`** — list of ids that MUST exist in the benchmark registry (see Catalog coupling).
- **`topology`** — single-role example shown; for multi-host see `patterns/distributed.md`.
- **`roles.<role>.command`** — uses substitution tokens; reference sibling perf variant for the canonical vllm serve line.
- **`params`** — flat scalar map; referenced as `{params.<key>}` from `command` or harness invokers.

### Substitution tokens

`{shared_fs}`, `{models_dir}`, `{datasets_dir}`, `{artifacts_dir}`, `{images_dir}`, `{run_id}`, `{user-id}`, `{model.path}`, `{model.id}`, `{params.*}`

## `threshold.json` template

Threshold kinds:

- `min` — value must be >= bound (scores, counts, throughputs).
- `max_ms` — value must be <= bound (latencies in ms).
- `within` — value must lie in `[lo, hi]`.
- `min_tok_s` — throughput floor in tokens/sec.
- `min_ratio` — ratio floor in `[0, 1]`.

**Mandatory smoke thresholds (every variant):**

```json
{
  "smoke_request_latency_ms": { "kind": "max_ms", "value": 600000 },
  "smoke_completion_tokens":  { "kind": "min",    "value": 1 }
}
```

Omitting either fails the smoke guard before any benchmark runs.

**Perf variant** — one entry per scalar emitted by each listed benchmark, e.g.:

```json
"serve_synth_short.request_throughput": { "kind": "min",    "value": 0.5 },
"serve_synth_short.ttft_p95_ms":        { "kind": "max_ms", "value": 60000 }
```

**Accuracy variant** — one entry per task score, e.g.:

```json
"mmlu":  { "kind": "min", "value": 0.65 },
"gsm8k": { "kind": "min", "value": 0.55 }
```

## Catalog coupling

Every id in `config.json.benchmarks` must be registered in `cvs/lib/dtni/benchmarks/registry.py` (BENCHMARK_REGISTRY). To list known ids:

```bash
grep -n "BenchmarkSpec(" cvs/lib/dtni/benchmarks/registry.py
```

If your id is missing, stop and follow `add_benchmark.md` first.

## Verification

```bash
# 1. Config parses and resolves substitutions
cd cvs && python -m cvs.lib.dtni.config_loader \
  cvs/input/dtni/<fw>/<model>/<model>_<variant>/config.json

# 2. cvs sees the new variant
cvs list <framework> | grep <variant>

# 3. pytest collects it
cvs run <suite_id> --collect-only

# 4. No unittest regressions
pytest cvs/tests/dtni/ -x
```

## Worked example: `llama_3_1_8b_mmlu_only_accuracy`

```
cvs/input/dtni/vllm_single/llama_3_1_8b/llama_3_1_8b_mmlu_only_accuracy/
  config.json
  threshold.json
```

`config.json` (delta from sibling `_bf16_single_1_accuracy`):

```json
"benchmarks": ["mmlu"],
"params": { "tensor_parallelism": 1, "input_len": 16, "output_len": 32, "concurrency": 1 }
```

`threshold.json`:

```json
{
  "smoke_request_latency_ms": { "kind": "max_ms", "value": 600000 },
  "smoke_completion_tokens":  { "kind": "min",    "value": 1 },
  "mmlu":                     { "kind": "min",    "value": 0.65 }
}
```

---

> End-user docs: if this is the first variant exposed in this suite to users, also update `docs/reference/configuration-files/*.rst` per `docs/sphinx_rst.md`.
