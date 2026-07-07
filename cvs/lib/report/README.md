# Suite reports (`cvs.lib.report`)

CVS can attach an HTML/JSON **suite report** to pytest runs that use `--html`. Reports are
generated at **session end** (not as a lifecycle test) and bundled into the same results zip
as the pytest HTML report.

Reports are **render-only** — they do not change pass/fail or threshold enforcement.

---

## Reference example: InferenceX ATOM (`inferencex_atom_single`)

This is the working pilot. Other inference suite owners copy the same pattern with minor
preset tweaks — no report logic in test files.

### End-to-end flow

```text
cvs run inferencex_atom_single --html=run.html
        │
        ▼
Tests collect data (existing suite code)
  • inf_res_dict[cell_key] = {host → metrics}   in test_inferencex_atom_inference
  • lifecycle.record(nodeid, stage, seconds)      on server/client stages
  • variant_config + thresholds                   from conftest fixtures
        │
        ▼
Root cvs/conftest.py (automatic when preset exists)
  • auto_register: loads presets/inferencex_atom_single.py (stem match)
  • binds inf_res_dict / variant_config / lifecycle for session end
  • attaches lifecycle tables + cell cards to pytest-html rows
        │
        ▼
pytest_sessionfinish → generate_suite_reports()
  • inferencex_atom_run_deck.html + .json
  • inferencex_atom_run_deck_viewer.html
        │
        ▼
Results zip bundles pytest HTML + all report files
```

### Files to read (IX-atom)

| File | Role |
|------|------|
| `cvs/lib/report/presets/inferencex_atom.py` | Full preset: columns, tiers, charts, run card |
| `cvs/lib/report/presets/inferencex_atom_single.py` | One-line shim so `cvs run inferencex_atom_single` auto-loads the preset |
| `cvs/tests/inference/inferencex_atom/inferencex_atom_single.py` | Data collection only (`inf_res_dict`, `lifecycle.record`) — no report imports |
| `cvs/tests/inference/inferencex_atom/conftest.py` | Standard fixtures (`inf_res_dict`, `lifecycle`, `variant_config`) — no report wiring |
| `cvs/lib/report/presets/_inference_suite_template.py` | Copy this to start a new suite preset |
| `cvs/lib/report/presets/builder.py` | `make_inference_report_config()` — fills title, basename, charts from columns |

### Verify IX-atom reports

```bash
cvs run inferencex_atom_single --cluster_file ... --config_file ... --html=~/cvs_results/run.html
```

Expect in the zip: `inferencex_atom_run_deck.html`, `.json`, `_viewer.html`.

---

## Quick start — add reports to your inference suite

Suite owners add **one preset file** named after the `cvs run` stem. Root `cvs/conftest.py`
handles pytest wiring and session-end generation when `--html` is set.

### What your suite must already collect

| Data | Contract |
|------|----------|
| `inf_res_dict` | Module-scoped: cell key → `{host → {metric: value}}`. Keys are 6-tuples `(model, gpu, isl, osl, policy, concurrency)` — same as `print_results_table`. |
| `variant_config` | `model`, `gpu_arch`, `enforce_thresholds`, `thresholds`, `cell_key(isl, osl, conc)`. |
| `lifecycle` | `.report` dict; `lifecycle.record(nodeid, label, value, unit="s")` on stages. |

See IX-atom: `inf_res_dict[...] = results` in the workload test and `lifecycle.record` in
`test_inferencex_atom_inference` (`cvs/tests/inference/inferencex_atom/inferencex_atom_single.py`).

### Step 1 — Add a preset (`presets/<cvs_run_stem>.py`)

**Minimal (builder defaults):** copy `_inference_suite_template.py` and fill TODOs.

**Full control:** copy `inferencex_atom.py` and adjust columns, `tier_metric_specs`, run card.

```bash
cp cvs/lib/report/presets/_inference_suite_template.py cvs/lib/report/presets/my_suite_single.py
```

The module name must match the test file stem: `cvs run my_suite_single` → `presets/my_suite_single.py`.

Example using the builder (most suites only need this much):

```python
from cvs.lib.inference.inference_suite_results_table import MY_RESULTS_COLUMNS
from cvs.lib.inference.utils.my_parsing import (
    CLIENT_METRIC_UNITS,
    METRIC_TIER_ORDER,
    tier_metric_specs,
)
from cvs.lib.report.presets.builder import make_inference_report_config

MY_SUITE_SINGLE_REPORT_CONFIG = make_inference_report_config(
    suite_id="my_suite",
    results_columns=MY_RESULTS_COLUMNS,
    metric_units=CLIENT_METRIC_UNITS,
    tier_metric_specs=tier_metric_specs,
    metric_tier_order=METRIC_TIER_ORDER,
    inference_test_substring="test_my_suite_inference",
    row_card_test_names=("test_metric",),  # or test_cell_metrics
)
```

No `conftest.py` changes required. Optional explicit registration:

```python
from cvs.lib.report.inference_wiring import configure_inference_suite_report
from cvs.lib.report.presets.my_suite_single import MY_SUITE_SINGLE_REPORT_CONFIG

def pytest_configure(config):
    configure_inference_suite_report(config, MY_SUITE_SINGLE_REPORT_CONFIG)
```

### Step 2 — Run with `--html`

```bash
cvs run my_suite_single --cluster_file ... --config_file ... --html=~/cvs_results/run.html
```

---

Inference reports automatically include **provenance** in the JSON sidecar and run card when
generated via pytest: CVS version, optional git commit, `--cluster_file`, and `--config_file`.

## Report types

When `--html` is set, CVS may produce the following outputs. All suite-specific files use
the preset `report_basename` (e.g. `inferencex_atom_report`).

| Output | When | Format | Contents |
|--------|------|--------|----------|
| **Pytest HTML report** | Always with `--html` | `.html` | Standard pytest-html test log, lifecycle tables, and links to bundled files |
| **Inference suite report** | Inference suite registered + `inf_res_dict` populated | `.html` + `.json` | Run card, session lifecycle timeline, sweep summaries, concurrency charts, gate matrix, full results table, per-cell cards (inline or summary mode for large sweeps). JSON holds the full payload for tooling and the viewer |
| **Interactive viewer** | `interactive_viewer=True` on inference preset (default) | `{basename}_viewer.html` | Filterable cells; cross-shape comparison; per-shape throughput/latency scaling; P90/P95/P99 fans; gate margin vs C; heatmap metric toggle; throughput-vs-latency tradeoff; gate matrix; CSV export |
| **Pytest row cell cards** | Inference preset with `row_card_extras` | Embedded in pytest HTML | Compact per-cell metric cards on `test_cell_metrics` / `test_metric` rows |

**Results zip** (created at session end): pytest HTML, `{suite}_html/` attachments (suite reports, logs, config copies), and optional `assets/`.

## Package layout

```text
cvs/lib/report/
├── inference.py                  # build payload, write HTML/JSON
├── auto_register.py              # load presets/<cvs_run_stem>.py at session start
├── inference_wiring.py           # optional explicit preset registration
├── registry.py                   # preset registration + session store
├── types.py                      # InferenceReportConfig
├── presets/
│   ├── inferencex_atom.py        # IX-atom full preset (reference)
│   ├── inferencex_atom_single.py # auto-load shim for cvs run stem
│   ├── builder.py                # make_inference_report_config()
│   ├── _inference_suite_template.py  # copy → presets/<your_stem>.py
│   └── <cvs_run_stem>.py         # one file per suite owner
├── viewer/                       # interactive *_report_viewer.html
└── unittests/
```

**Generic code** stays in this package. **Suite-specific** pieces are only the preset and
conftest wiring (see **Quick start** above).

| Suite | Preset file | Notes |
|-------|-------------|-------|
| InferenceX ATOM | `presets/inferencex_atom_single.py` | **Reference** — auto-loaded for `cvs run inferencex_atom_single` |
| vLLM single-node | `presets/vllm_single.py` | Auto-loaded for `cvs run vllm_single` |
| Your suite | `presets/<cvs_run_stem>.py` | Copy `_inference_suite_template.py` or `inferencex_atom.py` |

## Inference suites (field reference)

### Preset fields (`InferenceReportConfig`)

| Field | Purpose |
|-------|---------|
| `suite_id` | Registry key |
| `report_basename` | Output stem, e.g. `my_suite_report` → `.html` / `.json` |
| `results_columns` | Results table columns (same keys as `inf_res_dict`) |
| `tier_metric_specs` | Metric tiers for gate matrix / cell cards |
| `chart_series` | Concurrency chart series (grouped per ISL/OSL in JSON + static HTML) |
| `inference_test_substring` | Which test nodeids count as inference rows |
| `interactive_viewer` | Write `*_report_viewer.html` (default `True`) |
| `viewer_cell_threshold` | Truncate inline cell cards in static HTML above this count |

### Pytest deep links

Cell cards link to the matching pytest-html row when `--html` is set (same directory in the
zip). Metric test rows (`test_cell_metrics` / `test_metric`) link to the current test nodeid.

## JSON sidecar (`schema_version: 1`)

Written as `{report_basename}.json` next to the HTML dashboard. External tools should read
`schema_version` first and ignore unknown keys.

| Field | Description |
|-------|-------------|
| `suite_id`, `generated_at`, `cvs_version`, `overall_status` | Run identity and gate summary |
| `report` | Title, subtitle, tier order, headline metric |
| `run_card_display`, `provenance` | Run card rows and paths (pytest HTML, cluster/config, git) |
| `lifecycle` | Session stage → seconds |
| `cells` | Per sweep cell: `actuals`, `tiers`, `metrics`, optional pytest nodeids |
| `chart_series`, `chart_comparison`, `sweep_summaries`, `gate_matrix`, `results_table` | Charts and tabular export (`chart_comparison` for JSON tooling; viewer rebuilds comparison from filtered cells) |
| `summary` | Truncation mode and viewer basename when cell count is large |

## Unit tests

```bash
python -m pytest cvs/lib/report/unittests/ -q
```

## See also

- `cvs/lib/inference/ADDING_A_SUITE.md` — new-suite checklist (IX-atom report example)
- `cvs/lib/report/presets/_inference_suite_template.py` — minimal preset starter
- `cvs/lib/report/presets/inferencex_atom.py` — full preset reference
- `cvs/lib/report/presets/builder.py` — `make_inference_report_config()` helper
