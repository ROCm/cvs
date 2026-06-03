# DTNI pytest layer — AGENTS.md

> Scope: `cvs/tests/dtni/` — the pytest tier-tree root and its conftest.
> Companion: `cvs/lib/AGENTS.md` (Job driver, RunContext, failure taxonomy).

## What this layer is for

A DTNI workload run (vLLM, sglang, megatron, …) does ONE expensive thing
on real hardware: launch a container, run a benchmark, capture logs.
Many independent assertions then want to inspect the same run — "did the
container come up?", "did the kernel log an OOM?", "did the framework
boot?", "did the bench complete?", "did p99 latency meet the threshold?".

This layer makes those assertions look like ordinary pytest tests. Each
test function takes one `workload_run` fixture, gets back the workload's
`Manifest`, and asserts against it. The expensive run happens once and
is cached across every test in the session that asks for the same cell.

## How a suite is organised

A "suite" is a **directory** under `cvs/tests/dtni/<suite_name>/`. The
directory name is the name passed to `cvs run`:

```
cvs/tests/dtni/
├── conftest.py              # workload_run fixture; loaded for every suite
├── __init__.py
├── <suite_name>/            # one suite, e.g. vllm_llama3_1_70b_single/
│   ├── __init__.py
│   ├── test_*.py            # tier tests (inspecting Manifest only)
│   └── unittests/           # offline tests for this suite; NOT runnable via `cvs run`
└── unittests/               # offline tests for this conftest itself
```

`cvs run <suite_name>` resolves the suite name to its directory and hands
the whole tree to pytest, which collects every `test_*.py` inside. An
unknown suite name fails closed — it does not silently run zero tests.

### Reserved directory names (not surfaced as runnable suites)

- `unittests/` — suite-private offline gate tests. Run with
  `python -m unittest cvs.tests.dtni.<suite>.unittests.test_<name>`.
  Excluded both at the suite root and inside each suite.
- Names beginning with `_` or `.`.
- Any name that collides with a flat `test_<name>.py` stem already
  discovered elsewhere (reserved at discover time; do not create
  such collisions).

## The `workload_run` fixture

A tier test should be one short assertion:

```python
def test_dmesg_clean(workload_run):
    assert "Out of memory" not in workload_run.logs.dmesg
```

The fixture does everything else:

1. Loads the typed config (`--config_file`) and cluster pool (`--cluster_file`).
2. Calls the binder to map workload roles to concrete hosts.
3. Builds a `RunContext` (artifact layout, event writer, executor, run id).
4. Looks up the adapter for `cfg.framework` from the registry.
5. Constructs the `Job` driver — **with the failure-pattern scanner
   wired in (see "Scanner wiring" below)**.
6. Calls `Job.run()`, which executes the six-phase lifecycle on real
   hardware and writes the manifest + sidecars to disk.
7. Returns the `Manifest`.

### Caching

The fixture is session-scoped and cached on `pytest.config._dtni_runs`
keyed by cell id. Every tier test in one pytest session that reads the
same cell sees the same `Manifest` object — the workload is launched
once per cell, not once per test.

### Skipped cells

If the binder cannot place the workload (insufficient nodes, no host
matches the selector), the fixture writes a manifest with
`overall_status="skipped"` and then `pytest.skip()`s every tier test
for that cell, using the binder's reason as the skip message.

## Scanner wiring (the load-bearing invariant)

The conftest exposes a tiny helper:

```python
def _build_job(adapter, ctx) -> Job:
    return Job(adapter, ctx, scanner=FailurePatternScanner())
```

This is the **only** place in the production tier-test path where `Job`
is constructed. The `scanner=` argument is load-bearing: without it,
captured `dmesg` and `container.log` streams are written to disk but
never matched against the failure-pattern catalog. A real OOM ends up
in a log file unnoticed and `test_dmesg_clean` passes.

**Hard rule:** never instantiate `Job(adapter, ctx)` in the tier-test
path without going through `_build_job` (or threading `scanner=`
yourself). The helper exists so the wiring is unit-testable —
`cvs/tests/dtni/unittests/test_conftest_wiring.py` guards it. If you
refactor the fixture, that test must keep passing; do not delete it to
make a refactor green.

## Anti-patterns

- **Putting `load_config_file` / `bind` / `Job(...)` in test files.** Test
  functions stay one assertion long. If you are reaching for any of those
  imports inside a `test_*.py`, you are on the wrong abstraction layer.
- **Constructing `Job` without `scanner=`** in any production path under
  `cvs/tests/dtni/`. The wiring test will fail.
- **Function-scoped fixtures over session-scoped ones** for anything that
  drives a real workload run. A function-scoped `workload_run` would
  defeat the session cache and relaunch the workload per test.
- **Adding flat `test_*.py` files directly under `cvs/tests/dtni/`** (not
  under a suite subdir). They will not be reachable as a runnable suite
  via `cvs run`.

## Current limitations

- **Single cell per suite.** The fixture currently runs each suite as one
  cell with a fixed id of `"single"`. Sweep expansion (one cell per
  combination of swept knobs) is not yet wired; when it is, the cache
  key will change from the literal `"single"` to the per-cell id, and
  `pytest_generate_tests` will parametrize a `dtni_cell` fixture.
- **No marker auto-derivation.** Tests are collected and run flat. There
  are no auto-applied `framework_*` / `model_*` / `tier_*` / `knob_*`
  markers and no tier-based deselection yet, so `pytest -m
  "framework_vllm and tier_1"` will not narrow collection.
- **Flat suites, no tier directory layout yet.** Suites today have their
  test files directly under `<suite_name>/`. A 6-tier layout
  (`logistics/`, `inference/`, `frameworks/`, `benchmarks/`, `models/`)
  is planned; the discover walk already recurses, so the layout will
  work without code changes here when adopted.

These are additive: when they land they extend this conftest in place;
the `workload_run` / `_build_job` contracts are stable.
