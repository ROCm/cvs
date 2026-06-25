# Suite reports (`cvs.lib.report`)

CVS can attach an HTML/JSON **suite report** to pytest runs that use `--html`. Reports are
generated at **session end** (not as a lifecycle test) and bundled into the same results zip
as the pytest HTML report.

Reports are **render-only** — they do not change pass/fail or threshold enforcement.

## Report types

When `--html` is set, CVS may produce the following outputs. All suite-specific files use
the preset `report_basename` (e.g. `inferencex_atom_report`).

| Output | When | Format | Contents |
|--------|------|--------|----------|
| **Pytest HTML report** | Always with `--html` | `.html` | Standard pytest-html test log, lifecycle tables, and links to bundled files |
| **Inference suite report** | Inference suite registered + `inf_res_dict` populated | `.html` + `.json` | Run card, session lifecycle timeline, sweep summaries, concurrency charts, gate matrix, full results table, per-cell cards (inline or summary mode for large sweeps). JSON holds the full payload for tooling and the viewer |
| **Interactive viewer** | `interactive_viewer=True` on inference preset (default) | `{basename}_viewer.html` | Filterable cell browser, Chart.js concurrency charts, throughput heatmap, gate matrix, CSV export. Reads the JSON sidecar |
| **Pytest row cell cards** | Inference preset with `row_card_extras` | Embedded in pytest HTML | Compact per-cell metric cards on `test_cell_metrics` / `test_metric` rows |
| **Embedded suite dashboard** | `--self-contained-html` | Embedded in pytest HTML | Static inference suite report iframe/summary inside the pytest report |
| **Inference parity report** | `parity_compare_jsons` or `CVS_INFERENCE_PARITY_COMPARE` and all JSON paths exist | `inference_parity_report.html` + `.json` | Side-by-side comparison of aligned sweep cells across framework runs (e.g. ATOM vs vLLM vs SGLang) |
| **Training suite report** | Training suite registered + `training_res_dict` populated | `.html` + `.json` | Per-node metrics table (throughput, tokens, iteration time, NaN count, memory). Optional **training parity** panel when a baseline JSON is configured |
| **Scaling panel** | Multi-node run (`nnodes > 1`) or multiple hosts per cell | Section inside inference `.html` / `.json` `panels.scaling` | Cluster throughput breakdown and optional efficiency vs a single-node baseline JSON |

**Results zip** (created at session end): pytest HTML, `{suite}_html/` attachments (suite reports, logs, config copies), and optional `assets/`.

**Training example basename:** `megatron_training_report` (Megatron 8B single pilot).

## Package layout

```text
cvs/lib/report/
├── inference.py, training.py     # build payload, write HTML/JSON
├── inference_wiring.py           # inference conftest helpers
├── training_wiring.py            # training conftest helpers
├── registry.py                   # preset registration + session store
├── types.py                      # InferenceReportConfig, TrainingReportConfig, …
├── presets/<suite>.py            # columns, tiers, chart series, basenames
├── parity/                       # cross-run inference JSON merge
├── panels/                       # optional sections (scaling, training parity)
├── viewer/                       # interactive *_report_viewer.html
└── unittests/
```

**Generic code** stays in this package. **Suite-specific** pieces are only the preset and
conftest wiring. Reference: `cvs/tests/inference/inferencex_atom/conftest.py`.

## Inference suites

### 1. Add a preset

Create `cvs/lib/report/presets/<your_suite>.py` with an `InferenceReportConfig`. Typical
fields:

| Field | Purpose |
|-------|---------|
| `suite_id` | Registry key |
| `report_basename` | Output stem, e.g. `my_suite_report` → `.html` / `.json` |
| `columns` | Results table columns (same keys as `inf_res_dict`) |
| `tier_metric_specs` | Metric tiers for gate matrix / cell cards |
| `chart_series` | Concurrency chart series |
| `inference_test_substring` | Which test nodeids count as inference rows |
| `interactive_viewer` | Write `*_report_viewer.html` (default `True`) |
| `viewer_cell_threshold` | Truncate inline cell cards in static HTML above this count |
| `parity_compare_jsons` | Optional `(framework_id, json_path)` tuples for parity |

Copy an existing preset (`inferencex_atom.py`) and adjust columns/tiers.

### 2. Wire the suite conftest

```python
from cvs.lib.report.inference_wiring import (
    attach_inference_suite_report_row_extra,
    bind_inference_suite_report_session,
    configure_inference_suite_report,
)
from cvs.lib.report.presets.my_suite import MY_SUITE_REPORT_CONFIG

def pytest_configure(config):
    configure_inference_suite_report(config, MY_SUITE_REPORT_CONFIG)

@pytest.fixture(scope="module", autouse=True)
def _suite_report_session(inf_res_dict, variant_config, lifecycle):
    bind_inference_suite_report_session(
        inf_res_dict=inf_res_dict,
        variant_config=variant_config,
        lifecycle=lifecycle,
    )
    yield

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    attach_lifecycle_html_table(item, report)  # if using inference lifecycle
    attach_inference_suite_report_row_extra(item, report)
```

### 3. Populate data during tests

- **`inf_res_dict`**: `cell_key → {host → {metric: value}}` (same contract as
  `print_results_table`)
- **`variant_config`**: sweep, thresholds, `enforce_thresholds`
- **`lifecycle`**: stage timings via `lifecycle.record(...)`

No `test_*_report` test is required. Root `cvs/conftest.py` calls
`HtmlReportManager.generate_suite_reports()` before the zip is created.

### 4. Run and verify

```bash
cvs run my_suite --cluster_file ... --config_file ... --html=~/cvs_results/run.html --self-contained-html
```

In `{run}.zip` expect:

- `{report_basename}.html` — static dashboard
- `{report_basename}.json` — full payload (viewer + parity input)
- `{report_basename}_viewer.html` — filters, charts, heatmap, CSV export (when enabled)

The pytest HTML **Reports** section links these files. With `--self-contained-html`, the
static dashboard can also be embedded in the pytest report.

### Optional: inference parity

Compare the reference suite JSON to other framework runs (same sweep cells):

```bash
export CVS_INFERENCE_PARITY_COMPARE=vllm=/path/vllm_report.json,sglang=/path/sglang_report.json
```

Or set `parity_compare_jsons` on the preset. When all paths exist, the zip also contains
`inference_parity_report.html` and `.json`.

Standalone merge (no pytest):

```bash
python -m cvs.lib.report.parity.inference \
  --reference ref_report.json --reference-id atom \
  --compare vllm=/path/vllm_report.json \
  --out inference_parity_report.html
```

## Training suites

### 1. Preset + conftest

Preset example: `presets/megatron_8b_single.py`. Reference conftest:
`cvs/tests/training/megatron/conftest.py`.

```python
from cvs.lib.report.training_wiring import (
    bind_training_suite_report_session,
    configure_training_suite_report,
)
from cvs.lib.report.presets.megatron_8b_single import MEGATRON_LLAMA3_8B_SINGLE_REPORT_CONFIG

def pytest_configure(config):
    configure_training_suite_report(config, MEGATRON_LLAMA3_8B_SINGLE_REPORT_CONFIG)

@pytest.fixture(scope="module", autouse=True)
def _training_report_session(training_res_dict, variant_config):
    bind_training_suite_report_session(
        training_res_dict=training_res_dict,
        variant_config=variant_config,
    )
    yield
```

### 2. Collect results

Use a module-scoped `training_res_dict` and record after training, e.g.
`record_megatron_training_results(training_res_dict, mt_obj)` in the Megatron pilot.

### 3. Optional training parity

Set `parity_baseline_json` on `TrainingReportConfig`, or:

```bash
export CVS_TRAINING_PARITY_BASELINE_JSON=/path/prior_megatron_training_report.json
```

## Unit tests

```bash
python -m pytest cvs/lib/report/unittests/ -q
```

## Related docs

- `cvs/lib/inference/ADDING_A_SUITE.md` — checklist entry for new inference suites
- `plans/suite-reporting-library-pitch.md` — short scope / follow-up list
