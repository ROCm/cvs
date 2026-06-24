# Adding a new DTNI suite

Step-by-step guide for adding a new Distributed Training aNd Inference suite.
The `vllm_single` suite is the reference implementation throughout.
Follow this top to bottom; each step links to the authoritative contract at the
moment you need it.

---

## The layer map (read this first)

Every concern has exactly one home. Before writing any code, locate your work on
this table:

| Layer | Directory | What belongs here |
|---|---|---|
| Framework-agnostic | `cvs/lib/utils/` | `BaseVariantConfig`, `substitute_config`, `Paths`, `ContainerSpec`, `evaluate_all` |
| Serving-generic | `cvs/lib/inference/utils/` | `Sweep`, `SeqCombo`, `GoodputSlo`, `Roles`, `validate_sweep_selector` |
| Framework-specific | `cvs/lib/<framework>/utils/` | `VariantConfig` subclass, `Params`, `load_variant`, metric vocabulary |
| Test suite | `cvs/tests/inference/<framework>/` | `conftest.py`, test module(s) |
| Input configs | `cvs/input/config_file/inference/<framework>/` | `_config.json`, `_threshold.json` |

Decision rule at every layer boundary:

- "Does any other suite (now or plausibly soon) need this?" → move it up one layer.
- "Is this specific to my framework's CLI flags or artifact format?" → keep it here.

When in doubt, push up. Code stranded too low gets copy-pasted into the next
suite; code pushed too high creates invisible coupling. Neither is free.

---

## Step 1: Decide what is generic vs framework-specific

Before writing a single class, answer these questions:

**Is this a serving/inference suite?**

Serving suites sweep sequence lengths (`isl`/`osl`) at concurrency levels.
The sweep machinery — `Sweep`, `SeqCombo`, `GoodputSlo`, `Roles`,
`validate_sweep_selector` — already lives in `cvs/lib/inference/utils/` and is
reusable unchanged. The only framework-specific piece is `Params`: the CLI flags
your benchmark tool accepts.

See [The serving-generic / vllm-specific seam](utils/AGENTS.md#the-serving-generic--vllm-specific-seam)
for the second-framework checklist.

**Is this a training suite?**

Training suites typically sweep different dimensions: `batch_size`, `seq_len`,
`num_gpus`, or similar. Write your own sweep schema. Still subclass
`BaseVariantConfig` (the framework-agnostic skeleton is always your base).
Your `cell_key` format is your choice — it just has to match your
`threshold.json` top-level keys exactly.

---

## Step 2: Subclass `BaseVariantConfig`

Create `cvs/lib/<your_framework>/utils/<your_framework>_config_loader.py`.

**Minimal skeleton for a serving suite:**

```python
from pydantic import model_validator
from typing_extensions import Literal

from cvs.lib.utils.config_loader import BaseVariantConfig, _Forbid, substitute_config
from cvs.lib.inference.utils.inferencing_config_loader import (
    GoodputSlo, Roles, Run, Sweep, SeqCombo, validate_sweep_selector,
)
from cvs.lib.<your_framework>.utils.<your_framework>_parsing import GATED_METRICS


class Params(_Forbid):
    # Your framework's CLI flags. All fields str (passed as CLI arguments).
    tensor_parallelism: str = "1"
    port_no: str = "8888"
    # ... add your flags here


class VariantConfig(BaseVariantConfig):
    framework: Literal["your_framework"]
    gpu_arch: str
    roles: Roles = Roles()
    params: Params
    sweep: Sweep

    def cell_key(self, isl, osl, concurrency) -> str:
        """Single source of truth for the threshold key for one sweep cell."""
        return f"ISL={isl},OSL={osl},TP={self.params.tensor_parallelism},CONC={concurrency}"

    def expected_cells(self) -> list:
        """Every cell key the sweep's runs selector picks."""
        by_name = {c.name: c for c in self.sweep.sequence_combinations}
        return [
            self.cell_key(by_name[r.combo].isl, by_name[r.combo].osl, r.concurrency)
            for r in self.sweep.runs
        ]

    @model_validator(mode="after")
    def _check_thresholds_cover_sweep(self):
        # Copy the two-axis check from inferencing_config_loader.py:
        # Axis 1: every sweep cell has a threshold entry; no key names a phantom cell.
        # Axis 2: every present cell has a spec for every GATED_METRICS member.
        # When enforce_thresholds=False: warn instead of raise.
        ...
        return self


def load_variant(config_path, cluster_dict) -> VariantConfig:
    raw, thresholds = substitute_config(config_path, cluster_dict)
    raw["thresholds"] = thresholds
    return VariantConfig(**raw)
```

See [Subclassing BaseVariantConfig](../utils/AGENTS.md#subclassing-basevariantconfig)
for the full contract: what fields you must add, what methods you must implement,
and the validator ordering rules.

Key points:

- `cell_key` is the **single source of truth**. The loader's coverage check
  (`_check_thresholds_cover_sweep`) calls it to build expected keys; `test_metric`
  calls it to look up the threshold spec. Change the format in one place and both
  paths move together. A space, field-order change, or separator difference silently
  drops the cell (no threshold match, no verdict).
- `_check_thresholds_cover_sweep` must check **both axes**. Without axis 2, a gated
  metric with no threshold spec falls through the record-only branch of `test_metric`
  and reports a green PASS with zero assertions even when `enforce_thresholds=True`.
- `load_variant` must call `substitute_config` — never reimplement file-read or
  placeholder substitution. See [substitute_config contract](../utils/AGENTS.md#config_loaderpy)
  for what it returns and what it does not do (it does not validate or type-coerce).

---

## Step 2b: Write the metric vocabulary module

Create `cvs/lib/<framework>/utils/<framework>_parsing.py`.

Reference: `cvs/lib/inference/utils/vllm_parsing.py`.

This module is a pure-transform layer with no I/O. It contains:

1. **A pure transform function** that maps a raw benchmark artifact dict to a
   namespaced `{"client.<name>": value}` dict. This function accepts only
   data structures (no `orch`, no file paths) so it can be unit-tested without
   a running container.

2. **`YOUR_METRICS: list[tuple[str, str]]`** — the display surface: a list of
   `(short_name, unit)` pairs for every metric the suite surfaces. This list is
   iterated by `pytest_generate_tests` to emit one `test_metric` row per metric
   per cell.

3. **`GATED_METRICS: set[str]`** — the asserted subset: the short names whose
   threshold specs are required in `threshold.json` for every sweep cell when
   `enforce_thresholds=True`. This set is imported by `VariantConfig`'s
   `_check_thresholds_cover_sweep` to run the axis-2 coverage check at load time.

4. **The gated-vs-record-only decision rule:** gate a metric (add it to
   `GATED_METRICS`) when you have a calibrated baseline and a regression means a
   real performance failure. Keep a metric record-only (in `YOUR_METRICS` but not
   in `GATED_METRICS`) for diagnostic or informational metrics (e.g. percentiles
   useful for debugging but not yet part of the SLO contract) or for metrics
   whose baselines are not yet calibrated. Record-only metrics still appear in
   the HTML results table; they simply do not trigger a FAIL.

---

## Step 3: Write the job class

Create `cvs/lib/inference/<your_framework>_job.py` (or a similarly named module).

Reference: `cvs/lib/inference/vllm_single.py` (`VllmJob`).

The job class owns the benchmark lifecycle for a single cell. It is deliberately
I/O-agnostic above the `orch.exec` boundary — all container/SSH plumbing belongs
to `orch`, which is injected.

**Constructor:** accept `orch`, `variant` (your `VariantConfig`), and every
per-cell parameter (`isl`, `osl`, `concurrency`, etc.) as explicit arguments.
Pull all config values from `variant.params` and `variant.paths` here, not in
the methods, so the methods stay stateless and testable.

**Required methods:**

```python
class YourJob:
    def build_server_cmd(self):
        """Write the env script and create per-cell output directories inside the container."""

    def start_server(self):
        """Launch the server in the background inside the container."""

    def is_ready(self) -> bool:
        """Check readiness by scanning the server log (not a fixed tail)."""

    def wait_ready(self):
        """Poll until is_ready() or raise on timeout."""

    def stop_server(self):
        """Kill the server process."""

    def run_client(self):
        """Launch the benchmark client in the background inside the container."""

    def wait_client_complete(self):
        """Poll the client log until completion, crash, or timeout."""

    def parse_results(self) -> dict:
        """Fetches the results artifact (the only method that reads output data); the metric
        transform is delegated to the pure function in <your_framework>_parsing.py."""
```

**Key patterns from `VllmJob` to carry forward:**

- **Scan the whole server log for readiness**, not `tail -N`. The startup banner
  scrolls out of a fixed tail once the server gets chatty.
- **Accumulate completion and failure states independently in each poll iteration,
  then raise on failure before returning on completion.** The benchmark tool
  always prints an explicit completion marker (`COMPLETION_RE`) — key off that
  positive signal rather than the absence of an error line.
- **Per-cell output directories** keyed by `isl/osl/concurrency`. A multi-cell
  sweep must not overwrite an earlier cell's artifact; without this,
  `parse_results` may silently read stale data from a prior cell.
- **`parse_results` raises on empty/missing/unparseable artifacts.** The test
  wraps it in `try/except ... raise`, so a hard failure here is the correct
  behavior — it breaks the cell cleanly rather than recording a silently-green row.
- **`parse_results` delegates the transform** to the pure function in
  `<your_framework>_parsing.py`. The fetch (I/O) lives in the job class because
  artifact layout is job-specific; the metric math lives in `_parsing.py` so
  other suite variants can reuse it.

---

## Step 4: Write the conftest fixtures

Create `cvs/tests/inference/<your_framework>/conftest.py`.

Reference: `cvs/tests/inference/vllm/conftest.py`.

See [conftest fixtures](utils/AGENTS.md#conftest-fixtures) for the fixture
ownership table.

**All fixtures must be `scope="module"`** so they are shared across the entire
parametrized test run. A `scope="function"` fixture would re-launch the container
for every single metric row.

**Required fixtures:**

```python
@pytest.fixture(scope="module")
def cluster_dict(pytestconfig):
    cluster_file = pytestconfig.getoption("cluster_file")
    if not cluster_file:
        pytest.fail("--cluster_file is required")
    with open(cluster_file) as fp:
        d = json.load(fp)
    return resolve_cluster_config_placeholders(d)


@pytest.fixture(scope="module")
def variant_config(pytestconfig, cluster_dict):
    config_file = pytestconfig.getoption("config_file")
    if not config_file:
        pytest.fail("--config_file is required")
    return load_variant(config_file, cluster_dict)


@pytest.fixture(scope="module")
def lifecycle():
    return _Lifecycle()   # see _Lifecycle class below


@pytest.fixture(scope="module")
def orch(cluster_dict, variant_config, lifecycle):
    container_block = _deep_merge(
        cluster_dict.get("container", {}),
        variant_config.container.model_dump(),
    )
    testsuite_config = {"orchestrator": "container", "container": container_block}
    cfg = OrchestratorConfig.from_configs(cluster_dict, testsuite_config)
    o = OrchestratorFactory.create_orchestrator(log, cfg)
    yield o
    if not lifecycle.torn_down:
        log.info("orch fixture leak-guard: tearing down container")
        o.teardown_containers()


@pytest.fixture(scope="module")
def hf_token(variant_config):
    path = variant_config.paths.hf_token_file
    if not os.path.isfile(path):
        pytest.skip(f"hf_token file missing: {path}")
    with open(path) as fp:
        return fp.read().strip()


@pytest.fixture(scope="module")
def inf_res_dict():   # or train_res_dict for a training suite
    return {}
```

The `hf_token` fixture reads `variant_config.paths.hf_token_file` and calls
`pytest.skip` (not `pytest.fail`) when the file is missing. Using `skip` rather
than `fail` means suites that do not require an HF token can simply omit this
fixture from their conftest without breaking collection; the inference test that
accepts it as an argument will be skipped rather than erroring at fixture setup.

**`_Lifecycle`** — cross-test state for the lifecycle-as-tests model. Copy from
`cvs/tests/inference/vllm/conftest.py`. It carries three fields:
- `failed: bool` — set when any stage fails; causes remaining stages to skip
- `torn_down: bool` — set when `test_teardown` succeeds; suppresses the
  `orch` leak-guard finalizer so teardown never runs twice
- `report: dict` — maps `nodeid → [(label, value, unit)]`; populated by
  `lifecycle.record(...)` and rendered by `pytest_runtest_makereport`

**The `_deep_merge` pattern:**

`OrchestratorConfig.from_configs` does a top-level `dict.update`, so a bare
variant container block wipes all cluster-set container settings. Deep-merge the
variant ONTO the cluster block so cluster-set keys (e.g. `shm_size`, env maps)
survive, with the variant winning on conflicts. Copy `_deep_merge` verbatim from
the vllm conftest. If your suite is the second to need it, extract it to
`cvs/lib/utils/` instead of copy-pasting again.

**Required pytest hooks:**

```python
def pytest_collection_modifyitems(items):
    """Pin lifecycle order explicitly — never rely on definition order."""
    rank = {
        "test_launch_container": 0,
        "test_setup_sshd": 1,
        "test_model_fetch": 2,
        "test_<your_workload>": 3,
        "test_metric": 4,
        "test_print_results_table": 5,
        "test_teardown": 6,
    }
    items.sort(key=lambda it: rank.get(it.originalname or it.name.split("[")[0], 99))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Attach this test's recorded rows to its HTML detail panel."""
    ...  # copy from vllm conftest


def pytest_html_results_table_header(cells):
    """Add Value + Unit columns."""
    cells.insert(-1, "<th>Value</th>")
    cells.insert(-1, "<th>Unit</th>")


def pytest_html_results_table_row(report, cells):
    """Populate Value + Unit from metric_value / metric_unit user properties."""
    ...  # copy from vllm conftest
```

`pytest_collection_modifyitems` is not optional. `test_print_results_table` is
typically an imported function whose source line points into a shared module, so
default pytest ordering collects it first — which logs an empty table before any
cell ran. Explicit ranking fixes this.

---

## Step 5: Wire up `pytest_generate_tests`

Add `pytest_generate_tests` to your **test module** (not conftest), immediately
above the test functions.

Reference: `cvs/tests/inference/vllm/vllm_single.py`.

See [pytest_generate_tests mirror rule](utils/AGENTS.md#pytest_generate_tests-mirror-rule)
for why this must call `validate_sweep_selector`.

```python
def pytest_generate_tests(metafunc):
    """Parametrize the workload test and test_metric from the raw config sweep.

    Runs at collection time before fixtures exist — reads raw JSON directly.
    """
    config_file = metafunc.config.getoption("config_file")
    if not config_file or not os.path.isfile(config_file):
        return
    with open(config_file) as fp:
        raw = json.load(fp)
    sweep = raw.get("sweep", {})
    combos = sweep.get("sequence_combinations", [])
    runs = sweep.get("runs", [])

    # Validate GoodputSlo dicts through the _Forbid model so a typo'd SLO key
    # fails collection, not silently drops the gate on hardware.
    for combo in combos:
        if combo.get("goodput_slo") is not None:
            GoodputSlo(**combo["goodput_slo"])

    # Mirror the typed Sweep validator via the shared rule.
    # If you add a check to Sweep, add it to validate_sweep_selector so both paths enforce it.
    validate_sweep_selector([c["name"] for c in combos], [r["combo"] for r in runs])

    by_name = {c["name"]: c for c in combos}
    cases = [(by_name[r["combo"]], r["concurrency"]) for r in runs]
    ids = [r["combo"] + "-conc" + str(r["concurrency"]) for r in runs]

    if "metric" in metafunc.fixturenames:
        # test_metric: one case per (cell, metric)
        metric_cases = []
        metric_ids = []
        for (combo, c), cid in zip(cases, ids):
            for short, _unit in YOUR_METRICS:
                metric_cases.append((combo, c, short))
                metric_ids.append(cid + "-" + short)
        metafunc.parametrize("seq_combo,concurrency,metric", metric_cases, ids=metric_ids)
    elif "seq_combo" in metafunc.fixturenames and "concurrency" in metafunc.fixturenames and cases:
        # workload test: one case per cell
        metafunc.parametrize("seq_combo,concurrency", cases, ids=ids)
```

**Why `validate_sweep_selector` is mandatory here:**
`pytest_generate_tests` reads raw JSON at collection time, before `load_variant`
and the typed `Sweep` validator have run. Without calling `validate_sweep_selector`
here, a duplicate combo name or a `run.combo` typo is a silently-dropped cell at
collection time — the sweep runs a different matrix than the config reads.

---

## Step 6: Write the test module

Create `cvs/tests/inference/<your_framework>/<suite_name>.py`.

Reference: `cvs/tests/inference/vllm/vllm_single.py`.

**The lifecycle-as-tests model:** each stage is an independent pytest test, not
fixture body code. Each appears as a timed, independently pass/fail HTML row.

**Standard lifecycle order** (must match the rank dict in `pytest_collection_modifyitems`):

1. `test_launch_container` — calls `orch.setup_containers()`; asserts container is running
2. `test_setup_sshd` — calls `orch.setup_sshd()`; probes `:2224` for multinode
3. `test_model_fetch` — ensures model bytes present; polls/downloads if remote
4. `test_<your_workload>` — benchmark loop; stores results in `res_dict`
5. `test_metric` — one test per metric per cell; reads `res_dict`; asserts verdict
6. `test_print_results_table` — summary log; must run after all cells
7. `test_teardown` — calls `orch.teardown_containers()`; sets `lifecycle.torn_down`

**Invariants — every suite must enforce these:**

- Every test except `test_launch_container`, `test_print_results_table`, and
  `test_teardown` checks `lifecycle.failed` and calls `pytest.skip(...)` if
  `True`. This prevents cascading failures where a broken launch causes every
  subsequent cell to re-fail instead of skipping cleanly.
  `test_print_results_table` does not guard on `lifecycle.failed`; instead it
  checks whether `inf_res_dict` is empty and logs whatever results were
  recorded (even a partial sweep produces a useful table). This behavior is
  implemented in `cvs/tests/inference/vllm/_shared.py`; verify the check there
  if you are adapting the pattern for a new suite.
- `test_<workload>` wraps its entire body in `try/except`: on any exception, set
  `lifecycle.failed = True` then re-raise.
- `test_teardown` **never skips** — it must run even when `lifecycle.failed` is
  `True`. The container must be torn down regardless of what happened in the sweep.
- `test_teardown` sets `lifecycle.torn_down = True` after a successful teardown.
  This suppresses the `orch` fixture's leak-guard finalizer so the container is
  not torn down twice.

**`test_metric` pattern:**

```python
def test_metric(seq_combo, concurrency, metric, inf_res_dict, variant_config, lifecycle, request):
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")

    # Build the lookup key (must match what test_<workload> stored)
    isl, osl = seq_combo["isl"], seq_combo["osl"]
    key = (variant_config.model.id, variant_config.gpu_arch, isl, osl,
           seq_combo.get("name", "default"), concurrency)
    if key not in inf_res_dict:
        pytest.skip(f"no recorded results for cell {key!r}")

    host_dict = inf_res_dict[key]
    _host, actuals = next(iter(host_dict.items()))
    full = "client." + metric
    value = actuals.get(full)
    unit = YOUR_METRIC_UNITS.get(metric, "-")

    # Attach for HTML rendering (Value/Unit columns)
    request.node.user_properties.append(("metric_value", value))
    request.node.user_properties.append(("metric_unit", unit))

    if not variant_config.enforce_thresholds:
        return  # record-only

    cell = variant_config.cell_key(isl, osl, concurrency)
    spec = (variant_config.thresholds.get(cell) or {}).get(full)
    if spec is None:
        return  # record-only (no spec for this metric)

    # Pass the FULL per-cell actuals dict, not just this one metric's value.
    # evaluate_all needs the full dict so a min_ratio spec can resolve its
    # reference metric from the same actuals.
    evaluate_all(actuals, {full: spec})
```

See [evaluate_all contract](../utils/AGENTS.md#verdictpy) for why full cell
actuals are passed (needed for `min_ratio` reference resolution), and for the
behavior on `None` values and missing metrics.

---

## Step 7: Write the config and threshold files

**Config JSON** (`cvs/input/config_file/inference/<framework>/<variant>_config.json`):

```json
{
  "schema_version": 1,
  "framework": "your_framework",
  "gpu_arch": "mi300x",
  "enforce_thresholds": false,
  "threshold_json": "/absolute/path/to/your_threshold.json",
  "paths": {
    "shared_fs": "/mnt/data/{user-id}",
    "models_dir": "{shared_fs}/models",
    "log_dir": "{shared_fs}/logs",
    "hf_token_file": "/home/{user-id}/.hf_token"
  },
  "model": {"id": "meta-llama/Llama-3.1-70B-Instruct", "remote": 0},
  "container": {
    "lifetime": "per_run",
    "name": "your_suite_container",
    "image": "your.registry/image:tag",
    "runtime": {"name": "docker", "args": {}}
  },
  "params": {"tensor_parallelism": "8"},
  "sweep": {
    "sequence_combinations": [
      {"name": "isl1000_osl1000", "isl": "1000", "osl": "1000"}
    ],
    "runs": [
      {"combo": "isl1000_osl1000", "concurrency": 16}
    ]
  }
}
```

Start with `"enforce_thresholds": false` until you have calibrated baselines.
Flip to `true` once threshold values are established.

`threshold_json` is a **literal absolute path** — not relative to the config
file, not a glob. No placeholder substitution of any kind is applied to
`threshold_json` — not cluster placeholders, not `{paths.*}`. It is read
verbatim before any substitution pass runs. If the threshold path must vary by
user, it must be pre-resolved before being written into the config file. See
[placeholder-substitution.md](../utils/docs/placeholder-substitution.md) for a
worked example.

**Threshold JSON** (`<variant>_threshold.json`):

```json
{
  "_comment": "keys starting with _ are stripped before the coverage check",
  "ISL=1000,OSL=1000,TP=8,CONC=16": {
    "client.total_token_throughput": {"kind": "min_tok_s", "value": 12000},
    "client.output_throughput":      {"kind": "min_tok_s", "value": 1500},
    "client.mean_ttft_ms":           {"kind": "max_ms",   "value": 200},
    "client.success_rate":           {"kind": "min",       "value": 0.99},
    "client.failed":                 {"kind": "max",       "value": 0}
  }
}
```

Every top-level key must match `cell_key(...)` output exactly — same field
order, same separators, no spaces. See
[cell-key-format.md](utils/docs/cell-key-format.md) for the exact format spec
and common mistake patterns.

Every `GATED_METRICS` member must have a spec for every present cell. Missing
specs are caught at load time by `_check_thresholds_cover_sweep` (axis 2) when
`enforce_thresholds=True`.

See [threshold-kinds.md](../utils/docs/threshold-kinds.md) for the full kind
reference (`min`, `max`, `max_ms`, `within`, `min_tok_s`, `min_ratio`).

---

## Pre-PR checklist

Walk this list against your suite before opening a PR. Each item is verifiable
in the existing code.

**Config machinery**

- [ ] `load_variant` calls `substitute_config` — does not reimplement file-read or substitution
- [ ] `VariantConfig` subclasses `BaseVariantConfig`
- [ ] `VariantConfig` declares `framework: Literal["your_framework"]`, `params`, and `sweep`
- [ ] `cell_key` is implemented and is the single source of truth used by both
      `_check_thresholds_cover_sweep` and `test_metric`
- [ ] `expected_cells` is implemented and returns the full list of cell keys
- [ ] `_check_thresholds_cover_sweep` is present as a `@model_validator(mode="after")`
      and checks both axes (cell coverage + gated-metric coverage)

**Test fixtures**

- [ ] All fixtures are `scope="module"`
- [ ] `orch` fixture has a leak-guard finalizer that calls `teardown_containers()` when
      `lifecycle.torn_down` is `False`
- [ ] `_deep_merge` is used when building the container block (not bare `dict.update`)

**Test lifecycle**

- [ ] Lifecycle order is pinned explicitly in `pytest_collection_modifyitems`
- [ ] Every test except `test_launch_container`, `test_print_results_table`, and `test_teardown` checks `lifecycle.failed` and skips if `True` (`test_print_results_table` instead checks whether `inf_res_dict` is empty — see `_shared.py`)
- [ ] `test_<workload>` catches all exceptions, sets `lifecycle.failed = True`, re-raises
- [ ] `test_teardown` does NOT skip on `lifecycle.failed`
- [ ] `test_teardown` sets `lifecycle.torn_down = True` after successful teardown

**`pytest_generate_tests`**

- [ ] Calls `validate_sweep_selector` to mirror the typed `Sweep` validator — collection-time
      and load-time paths enforce the same rules
- [ ] Validates `GoodputSlo` dicts through the `_Forbid` model if your sweep uses goodput SLOs

**`test_metric`**

- [ ] Passes the full per-cell actuals dict to `evaluate_all`, not just the single metric value
- [ ] Returns (record-only) when `enforce_thresholds` is `False`
- [ ] Returns (record-only) when `spec is None` for this cell+metric

**Layer placement**

- [ ] Things only this suite needs → `cvs/lib/<framework>/utils/` (not pushed up)
- [ ] Things any serving suite needs → `cvs/lib/inference/utils/` (not kept local)
- [ ] Things any CVS suite needs → `cvs/lib/utils/` (not kept at the inference layer)

**Config files**

- [ ] `threshold_json` is a literal absolute path (not relative, not a glob)
- [ ] Every threshold key matches `cell_key(...)` output exactly
- [ ] Every `GATED_METRICS` member has a spec for every present cell
- [ ] `enforce_thresholds` starts as `false` until baselines are calibrated
- [ ] New gated metric → every existing `threshold.json` that covers cells where the
      metric is defined has a spec for it
