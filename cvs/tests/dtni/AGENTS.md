# DTNI pytest layer — AGENTS.md

> Scope: `cvs/tests/dtni/` (the pytest tier-tree root + its conftest).
> Workstream: W6 (matrix/markers/tiers). Companion to `cvs/lib/AGENTS.md`
> (Job/RunContext) and the addendum §9 "G6 re-split — barebones-first"
> entry that records what landed when.
>
> **Current state: BAREBONES (PR-X).** The full matrix surface
> (`pytest_generate_tests` sweep expansion, `cvs/lib/markers.py`,
> `cvs/lib/tiers.py`, `pytest_collection_modifyitems` tier deselection,
> `@requires_benchmark`, the 6-tier directory layout) lands in **PR-Z**.
> Until then, every collected test in every suite runs as a single cell.

## Suite layout (A4)

A "DTNI suite" is a **directory** under `cvs/tests/dtni/<suite_name>/`
containing test files. The directory name IS the suite name passed to
`cvs run` / `cvs list`:

```
cvs/tests/dtni/
├── conftest.py              # the workload_run fixture; loaded for every suite
├── __init__.py
├── <suite_name>/            # ONE suite (e.g. vllm_llama3_1_70b_single/)
│   ├── __init__.py
│   ├── test_*.py            # tier tests; flat in barebones, nested in PR-Z
│   └── unittests/           # offline gate tests; EXCLUDED from `cvs run` discovery
└── unittests/               # this conftest's own offline gate tests
```

`cvs run <suite_name>` calls `ListPlugin._find_test(<suite_name>)`, which
returns the module path of the suite **package**. `get_test_file()`
detects that the path resolves to a directory and hands the whole tree
to `pytest.main`, which collects every `test_*.py` under it. An unknown
suite name returns `None` -- fail-closed, never silently runs zero tests.

### Reserved directory names (not surfaced as suites)

- `unittests/` -- suite-private offline gate tests. Run via
  `python -m unittest cvs.tests.dtni.<suite>.unittests.test_<name>`, not via
  `cvs run`. The same exclusion applies to the conftest's own
  `cvs/tests/dtni/unittests/` tree.
- Names beginning with `_` or `.`.
- Any name that already exists as a flat `test_<name>.py` stem (discover
  time reservation; do not create such collisions).

## The `workload_run` fixture (the entry point for tier tests)

A test file under `cvs/tests/dtni/<suite>/test_*.py` is **tiny by design** --
it asks pytest for `workload_run` and inspects the resulting `Manifest`:

```python
def test_dmesg_clean(workload_run):
    assert "Out of memory" not in workload_run.logs.dmesg
```

The fixture does the rest: loads the config + cluster file, calls G4 `bind`,
builds a `RunContext`, looks up the adapter via G5a `get_adapter`,
constructs a `Job` (with the scanner wired -- see B5 below), and calls
`Job.run()`. It returns the `Manifest`.

### Caching

The fixture is **session-scoped** and cached on `pytest.config._dtni_runs`
keyed by cell id (currently the literal `"single"`; PR-Z keys by
`SweepCell.id`). All tier tests in one pytest session that read the same
cell share **one** real workload run -- vLLM is launched once, not once per
tier test.

### Skipped cells

If `bind` returns `status: skipped` (insufficient nodes, selector mismatch),
the fixture writes a `Manifest(verdicts.overall_status="skipped")` and then
`pytest.skip()`s every test that requests `workload_run` for that cell --
with the binder's reason as the skip message.

## The B5 contract (`_build_job`)

```python
def _build_job(adapter, ctx) -> Job:
    return Job(adapter, ctx, scanner=FailurePatternScanner())
```

This is the **one place** in the production tier-test path where `Job` is
constructed. The `scanner=` argument is load-bearing: without it,
`Job._collect_pattern_hits` is a no-op and captured `dmesg` / `container.log`
streams are never matched against the failure-pattern catalog. A real OOM
gets logged to disk and `test_dmesg_clean` reports "passed."

**Hard invariant:** never instantiate `Job(adapter, ctx)` in the tier-test
path without going through `_build_job` (or threading `scanner=` yourself).
The helper exists specifically so the contract is unit-testable
(`cvs/tests/dtni/unittests/test_conftest_wiring.py`) -- if you refactor the
fixture, that test guards the wiring.

## Anti-patterns

- **Putting `load_config_file` / `bind` / `Job(...)` in test files.** The
  fixture exists so test functions stay one assertion long. If you find
  yourself reaching for any of those imports inside `test_*.py`, you are
  on the wrong abstraction layer.
- **Instantiating `Job` without `scanner=`** in any production path under
  `cvs/tests/dtni/`. The test in
  `cvs/tests/dtni/unittests/test_conftest_wiring.py` will fail; do not
  delete the test to make a refactor pass.
- **Function-scoped fixtures over session-scoped ones** for anything that
  drives a real workload run. A function-scoped `workload_run` would
  invalidate the cache assumption and silently relaunch vLLM per test.
- **Adding flat `test_*.py` files directly under `cvs/tests/dtni/`** (not
  under a suite subdir). The discover-time reservation will block the suite
  from registering and the file will only be reachable via the legacy flat
  lookup, defeating the suite-as-directory contract.
- **Reusing the cell key `"single"` once PR-Z lands.** PR-Z keys the
  fixture cache by `SweepCell.id`; the literal `"single"` is a barebones
  placeholder. When you delete it, also delete this paragraph.

## Forward pointers (what changes when PR-Z lands)

PR-Z (the matrix layer) extends this conftest in place; the
`workload_run` fixture body and `_build_job` keep their current shape. PR-Z
adds:

- `pytest_generate_tests` -- calls `expand_sweep(cfg.sweep)` and parametrizes
  `dtni_cell` (one cell per sweep value). `_execute_single_cell` becomes
  `_execute_cell(config, dtni_cell)` and the cache key becomes
  `dtni_cell.id`.
- `pytest_collection_modifyitems` -- calls
  `derive_markers(cfg)` / `tier_matches` / `benchmark_matches` to deselect
  tests whose tier / framework / benchmark doesn't match the config.
- Auto-applied markers (`framework_*`, `model_*`, `topology_*`, `gpu_*`,
  `knob_*_*`, `tier_N`, `skipped_*`) so `-m "framework_vllm and tier_1"`
  works.
- The 6-tier directory layout under each suite
  (`logistics/`, `inference/`, `frameworks/`, `benchmarks/`, `models/`).
  The A4 discover walk already recurses, so the layout works without
  list_plugin changes.

PR-Y (the vLLM Integration Milestone) ships **before** PR-Z and authors
flat test files under `cvs/tests/dtni/vllm_llama3_1_70b_single/test_*.py`.
PR-Z reorganizes those into the 6-tier layout when it lands.
