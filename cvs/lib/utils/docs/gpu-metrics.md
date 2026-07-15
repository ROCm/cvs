# GPU Metrics Polling — Integration Guide

`cvs/lib/utils/gpu.py` is a shared library that any CVS suite — inference or training —
can use to collect GPU utilisation data during a run and surface it as rows in the
HTML report. It has no suite-specific logic: it shells out to `amd-smi metric --json`
via an `Orchestrator` and parses/aggregates the result. This document explains what the
library measures and how a suite can wire it in; the exact fixture/parametrize/threshold
plumbing shown below is illustrative reference pseudocode drawn from an inference suite —
adapt it to your suite's own lifecycle-as-tests structure.

> **Prerequisite**: this guide assumes you have completed (or are familiar with)
> the steps in `cvs/lib/inference/ADDING_A_SUITE.md`. Concepts like `cell_key`,
> `GATED_METRICS`, and `inf_res_dict` structure are defined there.

---

## What it measures

Five derived metrics are produced per run:

| Metric key | Unit | Description |
|---|---|---|
| `gpu.peak_gpu_memory_mb` | MB | Highest VRAM used across all GPUs at any single poll during inference. Each poll sums VRAM across all GPUs on the node; this value is the max of those sums. |
| `gpu.model_load_memory_mb` | MB | VRAM delta between a snapshot taken before model load and one taken after. Represents the memory cost of loading the model weights. |
| `gpu.model_load_s` | s | Wall-clock time from server start to the post-load snapshot. |
| `gpu.gpu_bandwidth_util_pct` | % | Mean UMC (unified memory controller) activity across all GPUs, averaged over all polls taken during inference. |
| `gpu.gpu_compute_util_pct` | % | Mean GFX (shader/compute) activity across all GPUs, averaged over all polls taken during inference. |

Each metric appears as its own row in the HTML report, with value, unit, and a
pass/fail result if a threshold is configured.

---

## How polling works

1. **Pre-load snapshot** — `capture_gpu_metrics(orch)` is called before the server
   starts. Records baseline VRAM.
2. **Server start + post-load snapshot** — after the server is ready,
   `capture_gpu_metrics(orch)` is called again. The VRAM delta and elapsed time give
   `model_load_memory_mb` and `model_load_s`.
3. **Client phase polling** — `poll_gpu_metrics(...)` is called (either synchronously
   with a backgrounded client, or from a thread with a synchronous client) and calls
   `amd-smi metric --json` on the head node every `poll_interval_s` seconds
   (default 15 s) until `is_done_fn()` returns `True`.
4. **Aggregation** — after the client completes, `agg_readings(readings)` reduces the
   poll list to `peak_gpu_memory_mb`, `gpu_compute_util_pct`, and
   `gpu_bandwidth_util_pct`.
5. **Results stored** — all five derived metrics are written into `inf_res_dict` under
   `gpu.<key>` so `test_gpu_metric` can read them.

`amd-smi` runs on the host node, not inside the container. Single-node suites use
`orch.exec_on_head("amd-smi metric --json")`; multi-node suites pass a `nodes` list and
`gpu.py` calls `orch.exec("amd-smi metric --json", hosts=hosts)` per node instead. This
is intentional — `amd-smi` is a host-side tool and is not available inside the benchmark
container.

---

## Multi-node polling

Both `capture_gpu_metrics` and `poll_gpu_metrics` accept an optional `nodes` parameter:
a `list[(label, hosts)]`, where `hosts` is a list of hostnames. When provided, `gpu.py`
calls `orch.exec("amd-smi metric --json", hosts=hosts)` once per `(label, hosts)` pair
per poll, merges every node's GPU entries into a single aggregated snapshot (same shape
as the single-node case), and separately tracks the last successful per-node VRAM
reading for the summary block.

```python
nodes = [
    ("prefill-0", prefill_node_list),
    ("decode-0", decode_node_list),
]
poll_readings = poll_gpu_metrics(
    orch,
    is_done_fn=<your done predicate>,
    log_path=str(_gpu_log) if _gpu_log else None,
    model_load_s=load_s,
    model_load_memory_mb=load_mb,
    nodes=nodes,
)
```

Any `Orchestrator` subclass works here since `.exec(cmd, hosts=...)` is part of the base
`Orchestrator` contract — no need to construct raw `Pssh`/ssh handles per role. See
`cvs/lib/inference/sglang_disagg_lib.py::sglang_disagg_gpu_counts` for a disaggregated
prefill/decode suite that groups nodes by role this way.

When `nodes` is provided, log lines are tagged with `[label1+label2]` and the summary
block gains a `--- per-node vram (last reading) ---` section listing each label's most
recent successful VRAM reading (or `-` if every poll failed for that node).

---

## Integrating into a suite

### 1. Add the GPU polling block to `test_<framework>_inference`

The function signature must include `gpu_metrics_snap` (see Step 3). Wrap
`capture_gpu_metrics` in a helper that degrades gracefully if `amd-smi` is unavailable
at snapshot time — unlike `poll_gpu_metrics`, it can raise.

**Pattern A — client is backgrounded by the framework (synchronous poll):**

```python
import pathlib
import time
from cvs.lib.utils.gpu import GPU_METRICS, GPU_METRIC_UNITS, agg_readings, capture_gpu_metrics, poll_gpu_metrics

def test_<framework>_inference(orch, variant_config, inf_res_dict, gpu_metrics_snap, request, ...):

    def _snap():
        try:
            return capture_gpu_metrics(orch)
        except Exception:
            return {}

    pre_snap = _snap()
    t0 = time.monotonic()
    # ... start server (returns immediately; framework backgrounds the client) ...
    post_snap = _snap()
    load_s = time.monotonic() - t0
    load_mb = ((post_snap.get("gpu.used_vram") or 0) - (pre_snap.get("gpu.used_vram") or 0)) or None

    # Write the log into the local report dir so it lands in the zip bundle.
    _htmlpath = getattr(request.config.option, "htmlpath", None)
    _html_dir = getattr(request.config, "_test_html_dir", "test_html")
    _gpu_log = (
        pathlib.Path(_htmlpath).parent / _html_dir / "gpu_poll.log"
        if _htmlpath else None
    )

    poll_readings = poll_gpu_metrics(
        orch,
        is_done_fn=<your done predicate>,  # e.g. job.is_client_done
        log_path=str(_gpu_log) if _gpu_log else None,
        model_load_s=load_s,
        model_load_memory_mb=load_mb,
    )

    agg = agg_readings(poll_readings)
    inf_res_dict["gpu.peak_gpu_memory_mb"]     = agg.get("peak_gpu_memory_mb")
    inf_res_dict["gpu.model_load_memory_mb"]   = load_mb
    inf_res_dict["gpu.model_load_s"]           = load_s
    inf_res_dict["gpu.gpu_bandwidth_util_pct"] = agg.get("gpu_bandwidth_util_pct")
    inf_res_dict["gpu.gpu_compute_util_pct"]   = agg.get("gpu_compute_util_pct")
```

**Pattern B — client runs synchronously in the main thread (thread the poll):**

```python
import threading

    done_flag = threading.Event()
    poll_readings = []
    def _poll():
        poll_readings.extend(poll_gpu_metrics(
            orch, done_flag.is_set,
            log_path=f"{variant_config.paths.log_dir}/gpu_poll.log",
            model_load_s=load_s,
            model_load_memory_mb=load_mb,
        ))
    poll_thread = threading.Thread(target=_poll, daemon=True)
    poll_thread.start()
    # ... run client synchronously ...
    done_flag.set()
    poll_thread.join()
    # then aggregate as in Pattern A
```

### 2. Add `test_gpu_metric`

`test_gpu_metric` is parametrized via `pytest_generate_tests` (see Step 4), not via a
`@pytest.mark.parametrize` decorator. The fixture parameter name is `gpu_metric`
(singular, matching the `pytest_generate_tests` branch).

Pass the **full** per-cell actuals dict to `evaluate_all` — not just the single metric
— so that `min_ratio` threshold specs can resolve their reference metric:

```python
from cvs.lib.utils.gpu import GPU_METRIC_UNITS
from cvs.lib.utils.verdict import ThresholdViolation, evaluate_all

def test_gpu_metric(gpu_metric, inf_res_dict, variant_config, request):
    val  = inf_res_dict.get(gpu_metric)
    unit = GPU_METRIC_UNITS.get(gpu_metric, "")

    request.node.user_properties.append(("metric_value", val))
    request.node.user_properties.append(("metric_unit", unit))

    if val is None:
        pytest.skip(f"{gpu_metric}: no value recorded (amd-smi unavailable or polling failed)")

    if not variant_config.enforce_thresholds:
        return

    cell = variant_config.cell_key(isl, osl, concurrency)  # same key used for test_metric
    spec = (variant_config.thresholds.get(cell) or {}).get(gpu_metric)
    if spec is None:
        return  # no spec → record-only

    # Pass full cell actuals so min_ratio specs can resolve their reference metric
    cell_actuals = {k: inf_res_dict.get(k) for k in inf_res_dict}
    try:
        evaluate_all(cell_actuals, {gpu_metric: spec})
    except ThresholdViolation as exc:
        pytest.fail(str(exc))
```

### 3. Add `gpu_metrics_snap` fixture to `conftest.py`

```python
@pytest.fixture(scope="module")
def gpu_metrics_snap():
    return {}
```

This fixture is a forward-declaration that lets `test_gpu_metric` be collected without
errors even if a future version stores intermediate state in it.

### 4. Register `test_gpu_metric` in `pytest_collection_modifyitems` and `pytest_generate_tests`

**Collection sort** — add `test_gpu_metric` at rank 4 alongside `test_metric`:

```python
rank = {
    "test_launch_container":    0,
    "test_setup_sshd":          1,
    "test_model_fetch":         2,
    "test_<framework>_inference": 3,
    "test_metric":              4,
    "test_gpu_metric":          4,   # must be present; omitting → rank 99 → runs after teardown
    "test_print_results_table": 5,
    "test_teardown":            6,
}
```

**Parametrize** — add an `elif` branch to `pytest_generate_tests` in the test module.
The fixture name is `gpu_metric` (singular):

```python
from cvs.lib.utils.gpu import GPU_METRICS

def pytest_generate_tests(metafunc):
    if "metric" in metafunc.fixturenames:
        # ... your existing metric parametrize branch ...
    elif "gpu_metric" in metafunc.fixturenames:
        metafunc.parametrize(
            "gpu_metric",
            [k for k, _ in GPU_METRICS],
            ids=[k for k, _ in GPU_METRICS],
        )
```

Without this branch `test_gpu_metric` collects zero instances and produces no HTML rows.

### 5. Add threshold entries and update `GATED_METRICS`

**Threshold JSON** — threshold keys use the `gpu.` prefix. For each sweep cell:

```json
"isl1000_osl1000_conc16": {
  "client.total_token_throughput": { "kind": "min_tok_s", "value": 1000 },
  "gpu.peak_gpu_memory_mb":        { "kind": "max",       "value": 200000 },
  "gpu.model_load_memory_mb":      { "kind": "max",       "value": 150000 },
  "gpu.model_load_s":              { "kind": "max",       "value": 300 },
  "gpu.gpu_bandwidth_util_pct":    { "kind": "min",       "value": 10 },
  "gpu.gpu_compute_util_pct":      { "kind": "min",       "value": 5 }
}
```

**`GATED_METRICS`** — if your `VariantConfig` subclass validates that every gated
metric has a threshold entry (the two-axis coverage check in `ADDING_A_SUITE.md`
Step 2), add all five `gpu.*` keys to your `GATED_METRICS` set:

```python
GATED_METRICS = {
    "client.total_token_throughput",
    ...
    "gpu.peak_gpu_memory_mb",
    "gpu.model_load_memory_mb",
    "gpu.model_load_s",
    "gpu.gpu_bandwidth_util_pct",
    "gpu.gpu_compute_util_pct",
}
```

Omitting them causes a silent green PASS with no assertions when `enforce_thresholds=True`
and the spec is missing.

**First run / characterisation** — set `enforce_thresholds: false` in the suite config.
All five metrics will be collected and surfaced as HTML rows but will never cause a
test failure. Use the reported values to populate your threshold JSON, then flip
`enforce_thresholds` to `true`.

See `docs/threshold-kinds.md` for the full threshold kind reference (`min`, `max`,
`max_ms`, `within`, `min_tok_s`, `min_ratio`).

---

## The `gpu_poll.log` file

Every run writes `gpu_poll.log` into the local HTML report directory (the same folder
as the per-test HTML files, e.g. `<suite>_html/`). Because the zip bundle includes
that directory, the log is always available in the run archive. It is also copied to
the suite's NFS `out_dir` on the head node for cluster-side inspection.

The file contains one line per poll and a summary block:

```
[gpu poll 1/?] used_vram=131072 MB  gfx=87%  umc=74%  mm=0%
[gpu poll 2/?] used_vram=132864 MB  gfx=91%  umc=78%  mm=0%
...
[gpu poll 12/?] used_vram=132480 MB  gfx=89%  umc=76%  mm=0%  [done]

--- summary ---
samples:              12
peak_gpu_memory_mb:   132864 MB
model_load_memory_mb: 127418 MB
model_load_s:         148.3 s
gpu_compute_util_pct:  89.2 %
gpu_bandwidth_util_pct: 76.1 %
```

A poll that fails (e.g. `amd-smi` exits non-zero or returns unparseable JSON) is
logged with a `FAILED [N/max consecutive]` tag and excluded from aggregation. After
`max_consecutive_failures` (default 3) consecutive failures the loop stops early and
logs a warning.

---

## Failure handling and None values

The library never raises from `poll_gpu_metrics`. Every metric can be `None`:

| Situation | Result |
|---|---|
| `amd-smi` fails or returns unparseable JSON | snapshot excluded from aggregation; metric may be `None` if all polls fail |
| GPU reports `"N/A"` for a field | that field is `None` in the snapshot |
| Zero valid polls | all three `agg_readings` outputs are `None` |
| Caller passes `model_load_memory_mb=None` | stored as `None`; `test_gpu_metric` should `pytest.skip` rather than fail |

`test_gpu_metric` should always check for `None` before evaluating thresholds.
`pytest.skip` (not `pytest.fail`) is the correct response when a metric is `None` —
the metric was unavailable for this run, not a regression.

---

## Gotchas

- **`model_load_memory_mb` should be `None` when VRAM data is unavailable, not `0`.**
  Use `... or None` after the subtraction (as shown in Step 1). A zero stored as `0`
  gets gated against thresholds and displayed as `"0"` in the report; `None` causes
  `test_gpu_metric` to skip instead.
- **`capture_gpu_metrics` can raise; `poll_gpu_metrics` never does.** Always wrap
  one-shot snapshot calls in a `try/except` that returns `{}` on failure.
- **`agg_readings` returns 3 keys, not 5.** `model_load_memory_mb` and `model_load_s`
  are measured by the caller and stored separately. Do not look for them in
  `agg_readings` output.
- **Raw snapshot keys differ from derived metric keys.** The poll loop returns dicts
  with keys like `gpu.used_vram`; the stored/threshold-gated keys use names like
  `gpu.peak_gpu_memory_mb`. Do not pass raw snapshots to `evaluate_all`.
- **Threshold JSON keys use the `gpu.` prefix** (`"gpu.peak_gpu_memory_mb"`, not
  `"peak_gpu_memory_mb"`). A missing prefix means the spec is never found and the
  metric silently operates as record-only even when `enforce_thresholds=True`.
- **`amd-smi` runs on the host, not in the container.** Single-node polling requires
  `orch.exec_on_head`; multi-node polling (via `nodes=`) requires `orch.exec(cmd,
  hosts=...)` instead. Every `Orchestrator` subclass supports both — if yours doesn't,
  GPU polling is not available for your suite.
- **Multi-node degrades per label, not globally.** If `orch.exec` raises for one node in
  `nodes`, that node's entries are excluded from the merged snapshot and its per-node
  VRAM is `None` for that poll; other nodes' data is unaffected.
- **Pass the full cell actuals dict to `evaluate_all`.** `min_ratio` threshold specs
  need to resolve a reference metric from `actuals`. Passing only the single metric's
  value causes a reference-resolution failure.
