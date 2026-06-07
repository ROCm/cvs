# Add a benchmark harness

## Goal

Wire a NEW benchmark harness (a different CLI tool — e.g. `bigcode-eval`, `lighteval`) into DTNI's three-tuple harness pattern. This recipe does NOT cover adding another task to an existing harness — see `add_benchmark.md`.

## When you need a new harness vs a new benchmark

Decision tree:

- Same CLI tool, new task / new traffic shape, same result-JSON schema → **just a `BenchmarkSpec`**. Use `add_benchmark.md`.
- Different CLI binary, OR different result-file name/format, OR different scalar projection logic → **new harness**. Continue with this recipe.

## The three-tuple pattern

Every harness has exactly three registrations, keyed by the same harness name string:

1. **Invoker** — `HARNESS_INVOKERS[name]` in `cvs/lib/dtni/benchmarks/harness_invokers.py`. Pure function `(BenchmarkSpec, HarnessCtx) -> str` that returns a `bash -c`-runnable command line. The command runs inside the server container via `docker exec`; it MUST write its output under `<hctx.output_dir>/<spec.id>/` (a per-benchmark subdir) so `runner.py` can locate the result deterministically.
2. **Result glob** — `_RESULT_GLOBS[name]` in `cvs/lib/dtni/benchmarks/runner.py`. Filename pattern (e.g. `"result.json"`, `"results*.json"`) the runner globs for under `<host_artifacts_dir>/<spec.id>/`. The newest match wins.
3. **Projector** — `PROJECTORS[name]` in `cvs/lib/dtni/benchmarks/projectors.py`. Pure function `(BenchmarkSpec, dict) -> dict[str, float]` mapping the parsed result JSON into the flat `{scalar_key: float}` dict that lands in `ctx.result.scalars`.

All three MUST use the same name string. The `BenchmarkSpec.harness` field selects which trio runs.

## Files you will edit (four)

1. `cvs/lib/dtni/benchmarks/harness_invokers.py` — add invoker function + `HARNESS_INVOKERS[name]` entry.
2. `cvs/lib/dtni/benchmarks/runner.py` — add `_RESULT_GLOBS[name]` entry.
3. `cvs/lib/dtni/benchmarks/projectors.py` — add projector function + `PROJECTORS[name]` entry.
4. `cvs/lib/dtni/benchmarks/registry.py` — add at least one `BenchmarkSpec(harness=<name>, ...)` so the wiring is reachable.

## Invoker contract

```python
def _bigcode_eval(spec: BenchmarkSpec, hctx: HarnessCtx) -> str: ...
```

- Pure (no I/O, no env reads, no clocks) — unit-testable by string compare.
- Builds an in-container shell line. Use `shlex.quote` on every argv token; leave `&&`, `mkdir -p`, etc. unquoted so the docker-exec wrapper (`bash -c`) parses them.
- Writes output under `f"{hctx.output_dir}/{spec.id}"`. Create the dir (`mkdir -p`) at the start of the command if the tool does not.
- The well-known in-container artifact mount path is `OUTPUT_DIR_IN_CONTAINER = "/cvs_artifacts"`; `hctx.output_dir` is normally this string. The host side of the same mount is `output_dir_on_host`, which the runner uses to read results.
- `HarnessCtx` exposes: `base_url`, `model_id`, `model_path`, `output_dir`. If your tool needs more (e.g. dataset path), thread it via `spec.extra`, not via new ctx fields.
- Raise `ConfigError` on missing required `spec.extra` keys — never silently substitute defaults for required inputs.

## Projector contract

```python
def _project_bigcode_eval(spec: BenchmarkSpec, payload: dict[str, Any]) -> dict[str, float]: ...
```

- Pure: no I/O, no globals, no randomness. Same `(spec, payload)` MUST always produce the same dict.
- Returns ONLY real numbers. Use the `_is_real_number` helper from `projectors.py` (rejects `bool`, which is an `int` subclass) before coercing.
- Scalar key convention:
  - **Accuracy-style** (one headline number per task): key = bare `spec.id`, plus optional `<spec.id>_stderr`.
  - **Perf-style** (a whole family of scalars per run): key = `f"{spec.id}.{field}"` so multiple invocations of the same harness do not collide and `threshold.json` can name individual scalars unambiguously.
- The returned keys ARE the names `threshold.json` matches against. Pick them deliberately and document them next to the projector.

## Skeleton — end-to-end new harness `bigcode-eval`

```python
# --- cvs/lib/dtni/benchmarks/harness_invokers.py ---
def _bigcode_eval(spec: BenchmarkSpec, hctx: HarnessCtx) -> str:
    task = spec.extra.get("task")
    if not task:
        raise ConfigError(f"benchmark {spec.id!r}: bigcode-eval needs extra.task")
    out_dir = f"{hctx.output_dir}/{spec.id}"
    parts = [
        "mkdir", "-p", out_dir, "&&",
        "bigcode-eval",
        "--model", hctx.model_id,
        "--tokenizer", hctx.model_path,
        "--base-url", hctx.base_url,
        "--tasks", str(task),
        "--n_samples", str(int(spec.extra.get("n_samples", 1))),
        "--save_generations_path", f"{out_dir}/result.json",
    ]
    return " ".join(p if p == "&&" else shlex.quote(p) for p in parts)

HARNESS_INVOKERS["bigcode-eval"] = _bigcode_eval

# --- cvs/lib/dtni/benchmarks/runner.py ---
_RESULT_GLOBS["bigcode-eval"] = "result.json"

# --- cvs/lib/dtni/benchmarks/projectors.py ---
def _project_bigcode_eval(spec: BenchmarkSpec, payload: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    score = (payload.get("results") or {}).get("pass@1")
    if _is_real_number(score):
        out[spec.id] = float(score)
    return out

PROJECTORS["bigcode-eval"] = _project_bigcode_eval

# --- cvs/lib/dtni/benchmarks/registry.py ---
BenchmarkSpec(
    id="humaneval",
    harness="bigcode-eval",
    dataset_id="humaneval",
    extra={"task": "humaneval", "n_samples": 1},
),
```

## Wiring checks (before you ship)

- The harness name string MUST be identical across `HARNESS_INVOKERS`, `_RESULT_GLOBS`, `PROJECTORS`, and `BenchmarkSpec.harness`. A typo in any one of the four means `build_command`, `runner.run_benchmarks`, or `project` will `KeyError` at run time.
- The server-container image MUST contain the harness binary, or the invoker MUST install it (see `_lm_eval` in `harness_invokers.py` for the `pip install -q ...` precedent).
- Result glob MUST match what the invoker actually writes. Run the invoker output through `bash -n` mentally — does the file land where you said?

## Verification

```bash
# 1. The three-tuple imports and is consistent.
python -c "
from cvs.lib.dtni.benchmarks.harness_invokers import HARNESS_INVOKERS
from cvs.lib.dtni.benchmarks.runner          import _RESULT_GLOBS
from cvs.lib.dtni.benchmarks.projectors      import PROJECTORS
name = '<harness-name>'
assert name in HARNESS_INVOKERS, 'missing invoker'
assert name in _RESULT_GLOBS,    'missing result glob'
assert name in PROJECTORS,       'missing projector'
print('ok')
"

# 2. Invoker is deterministic + quoted (add a case to test_harness_invokers.py).
pytest cvs/lib/dtni/unittests/test_harness_invokers.py -x

# 3. Projector is pure + deterministic (add a case to test_projectors.py).
pytest cvs/lib/dtni/unittests/test_projectors.py -x

# 4. At least one BenchmarkSpec using the new harness resolves.
python -c "from cvs.lib.dtni.benchmarks.registry import lookup; print(lookup('<bench-id>'))"

# 5. Full benchmark unittests + runner tests.
pytest cvs/lib/dtni/unittests/ -x
```

## End-user docs

If the new harness produces user-visible scalars (named in a public workload's `threshold.json`), document the scalar names and units in `docs/reference/configuration-files/*.rst` per `sphinx_rst.md`.
