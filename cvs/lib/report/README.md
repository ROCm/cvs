# Suite reports (`cvs.lib.report`)

CVS can attach an HTML/JSON **suite report** to pytest runs that use `--html`. Reports are
generated at **session end** and bundled into the same results zip as the pytest HTML report.
They are **render-only** — they do not change pass/fail or threshold enforcement.

**IX-atom reference:** `presets/inferencex_atom.py` + shim `presets/inferencex_atom_single.py`
(auto-loaded for `cvs run inferencex_atom_single`). Suite owners also see Step 8 in
`cvs/lib/inference/ADDING_A_SUITE.md`.

## Quick start

1. Copy `presets/_inference_suite_template.py` → `presets/<cvs_run_stem>.py` (stem must match
   `cvs run <stem>`).
2. Fill `make_inference_report_config(...)` with your `results_columns`, `tier_metric_specs`, and
   `metric_tier_order` (see `presets/inferencex_atom.py` for a full example).
3. Run with `--html`. Root `cvs/conftest.py` auto-loads the preset and writes reports at session
   end — no suite `conftest.py` wiring required.

Your suite must already collect:

| Data | Contract |
|------|----------|
| `inf_res_dict` | Module-scoped: cell key → `{host → {metric: value}}` |
| `variant_config` | Thresholds, `enforce_thresholds`, `cell_key(isl, osl, conc)` |
| `lifecycle` | `.record(nodeid, label, seconds)` on server/client stages |

```bash
cvs run inferencex_atom_single --cluster_file ... --config_file ... --html=~/cvs_results/run.html
python -m pytest cvs/lib/report/unittests/ -q
```

## Outputs (`report_basename` from preset)

| File | Contents |
|------|----------|
| `{basename}.html` + `.json` | Static run deck + full payload |
| `{basename}_viewer.html` | Interactive viewer (filters, charts, baseline upload, CSV) |
| `{basename}_summary.html` | CI one-pager |

Provenance (CVS version, git commit, cluster/config paths) is included when generated via pytest.

## Key preset fields

`suite_id`, `report_basename`, `results_columns`, `tier_metric_specs`, `chart_series`,
`inference_test_substring`, `interactive_viewer`, `viewer_cell_threshold`, `prev_run_json`.

**Baseline comparison:** resolves preset path → `CVS_INFERENCE_PREV_REPORT_JSON` → sibling
`{basename}_prev.json`. Payload includes `panels.prev_run`; the viewer can also upload any prior
JSON and flag delta % regressions.

## JSON sidecar

`{basename}.json` uses `schema_version: 1`. Main keys: `cells`, `chart_series`, `sweep_summaries`,
`gate_matrix`, `results_table`, `panels`, `overall_status`, `provenance`. Unknown keys should be
ignored by external tools.

## See also

- `presets/_inference_suite_template.py` — minimal starter
- `presets/inferencex_atom.py` — full reference preset
- `presets/builder.py` — `make_inference_report_config()`
