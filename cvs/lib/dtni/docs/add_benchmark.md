# Add a benchmark

## Goal

Register a new `BenchmarkSpec` against an EXISTING harness (e.g. another `lm-eval-harness` task or another `vllm-bench-serve` shape). This recipe does NOT cover wiring a brand-new harness — see `add_harness.md`. It also does NOT cover wiring a workload variant — see `add_workload.md`.

## Inputs (decide before starting)

1. **Harness** — must already be a key in `HARNESS_INVOKERS` (see `cvs/lib/dtni/benchmarks/harness_invokers.py`). Today that means `lm-eval-harness` or `vllm-bench-serve`.
2. **Benchmark id** — short slug, used as
   - the string in a workload's `config.json` `"benchmarks": [...]`, AND
   - the scalar key (lm-eval) or scalar-key prefix (perf) in `threshold.json`.
   Must be globally unique in `BENCHMARK_REGISTRY`.
3. **Harness-specific extras** — task name + few-shot count for `lm-eval-harness`; dataset shape + traffic params for `vllm-bench-serve`.

## File to edit

`cvs/lib/dtni/benchmarks/registry.py` — append one `BenchmarkSpec(...)` to the `_ACCURACY` or `_PERF` tuple. `BENCHMARK_REGISTRY` is rebuilt from these tuples at import time.

## `BenchmarkSpec` signature

From `registry.py`:

```python
@dataclass(frozen=True)
class BenchmarkSpec:
    id: str
    harness: str                       # key in HARNESS_INVOKERS / PROJECTORS
    dataset_id: str = ""               # catalog dataset key; "" for synthetic
    score_metric: str | None = None    # lm-eval only ("acc", "exact_match", ...)
    score_filter: str | None = None    # lm-eval only ("none", "strict-match", ...)
    shots: int = 0                     # lm-eval only
    extra: dict[str, Any] = field(default_factory=dict)
```

## Template — new `lm-eval-harness` task

Add to `_ACCURACY`:

```python
BenchmarkSpec(
    id="hellaswag",
    harness="lm-eval-harness",
    dataset_id="hellaswag",
    score_metric="acc",
    score_filter="none",
    shots=0,
    extra={"task": "hellaswag"},
),
```

Notes:
- `extra["task"]` is the lm-eval task name passed to `--tasks`. Defaults to `spec.id` if omitted, but set it explicitly when the two differ.
- `score_filter` is the lm-eval >=0.4 filter slug (results land at `acc,none` or `exact_match,strict-match`). The projector tries `<metric>,<filter>` → bare `<metric>` → `<metric>,none` in that order.
- The scalar produced is `scalars["<id>"]` (and `scalars["<id>_stderr"]` if present), which is what `threshold.json` should name.

## Template — new `vllm-bench-serve` shape

Add to `_PERF`:

```python
BenchmarkSpec(
    id="serve_synth_medium",
    harness="vllm-bench-serve",
    extra={
        "dataset_name":      "random",   # required
        "num_prompts":       128,        # required
        "random_input_len":  512,        # required when dataset_name=="random"
        "random_output_len": 256,        # required when dataset_name=="random"
        "max_concurrency":   16,         # optional
        "request_rate":      "inf",      # optional
        "percentiles":       "50,90,95,99",  # optional
        "goodput_slo":       "ttft:2000 tpot:200",  # optional, space-separated KEY:VALUE_ms
    },
),
```

Notes:
- `dataset_name="sharegpt"` requires `extra["dataset_path"]` (in-container path to the JSON).
- Perf scalars land at `scalars["<id>.<field>"]` (e.g. `serve_synth_medium.ttft_p95_ms`, `serve_synth_medium.request_throughput`). The `<id>` prefix prevents collisions across invocations of the same harness.
- See `_VLLM_SERVE_SCALAR_FIELDS` and `_VLLM_SERVE_PCTL_METRICS` in `cvs/lib/dtni/benchmarks/projectors.py` for the exact set of recognized field names.

## Wiring checks (before you ship)

- `spec.harness` MUST be a key in `HARNESS_INVOKERS` (`harness_invokers.py`) AND in `PROJECTORS` (`projectors.py`). If it is not, you need `add_harness.md`, not this recipe.
- `spec.id` MUST be unique across `_ACCURACY + _PERF` — `BENCHMARK_REGISTRY = {b.id: b for b in ...}` will silently last-write-wins on a collision.
- Any `threshold.json` that names this benchmark must use the projected scalar key exactly (`"<id>"` for lm-eval, `"<id>.<field>"` for perf).

## Verification

```bash
# 1. Spec parses and is discoverable.
python -c "from cvs.lib.dtni.benchmarks.registry import BENCHMARK_REGISTRY, lookup; print(lookup('<id>'))"

# 2. Registry unittest still green (catches duplicate-id and missing-harness regressions).
pytest cvs/lib/dtni/unittests/test_benchmark_registry.py -x

# 3. Reference the new id from a workload variant's config.json:
#       "benchmarks": ["<id>", ...]
#    and add the projected scalar keys to that variant's threshold.json.

# 4. Full benchmarks unittests (covers invoker + projector for the harness).
pytest cvs/lib/dtni/unittests/ -x
```

## End-user docs

If this benchmark is user-visible (named in a public workload variant or in a threshold a user authors), add a row to the benchmark table in `docs/reference/configuration-files/*.rst` per the conventions in `sphinx_rst.md`.
