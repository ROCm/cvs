# Suite reporting library

## Goal

One shared library (`cvs/lib/report/`) produces HTML/JSON suite dashboards when tests run with
`--html`. Output is bundled into the existing pytest results zip alongside the pytest HTML
report.

## Implemented

- Generic inference and training report engines, registry, session store
- Session-end generation in `HtmlReportManager.generate_suite_reports()` (root `conftest.py`)
- Inference wiring: preset + conftest + row extras (`inference_wiring.py`)
- Training wiring: Megatron 8B single pilot (`training_wiring.py`)
- Reference suite: `inferencex_atom_single`

## Report types

See `cvs/lib/report/README.md` for wiring. Summary:

| Type | Files |
|------|-------|
| Pytest HTML | `{run}.html` |
| Inference suite | `{report_basename}.html`, `.json`, optional `{basename}_viewer.html` |
| Inference parity | `inference_parity_report.html`, `.json` (optional) |
| Training suite | `{report_basename}.html`, `.json` (e.g. `megatron_training_report.*`) |

Inference reports include run card, lifecycle, sweep summaries, charts, gate matrix, and
results table. Training reports include a per-node metrics table. Optional panels: scaling
(inference), training parity (training), inference parity (cross-run).

## Artifacts (typical inference run)

```text
{pytest_html_basename}.zip
├── report.html                    # pytest-html
├── {report_basename}.html         # static suite dashboard
├── {report_basename}.json
├── {report_basename}_viewer.html  # when interactive_viewer is enabled
└── inference_parity_report.*      # when parity JSONs are configured
```

## Onboarding a new suite

1. Add `presets/<suite>.py` (`InferenceReportConfig` or `TrainingReportConfig`)
2. Register preset and bind session data in suite `conftest.py` (see `cvs/lib/report/README.md`)
3. Run with `--html` and confirm zip contents

No per-suite lifecycle test for report generation.

## Future TODO

Prioritized ideas for later work. Not scheduled; pick up when a suite or lab need arises.

### High value

- [x] Wire `vllm_single` conftest (`VLLM_SINGLE_REPORT_CONFIG` preset already exists)
- [ ] Vendor Chart.js for offline viewer (today loads from CDN; air-gapped labs need local assets)
- [ ] Richer training report: run card, cluster aggregates (mean/worst throughput, NaN count), PASS badge styling aligned with inference
- [ ] Multi-host cell cards in static HTML and pytest row extras (all hosts per sweep cell, not first host only)
- [ ] `compare.prev_run` panel: generic run vs last green JSON (CI artifact path or env), for inference and training

### Sweeps and CI

- [ ] Viewer run-to-run diff: load two JSON sidecars, highlight cells/metrics beyond a threshold
- [x] Config and provenance block in run card: cluster/config paths, CVS version, git SHA when available
- [x] Threshold margin on cell cards in record-only mode (when threshold spec exists)
- [ ] Deep links from suite report cells to matching pytest `test_cell_metrics` rows
- [ ] Accuracy / gsm8k panel when accuracy metrics land in `inf_res_dict` (same optional panel pattern as scaling)

### Polish

- [ ] Gate matrix heatmap in static HTML (viewer already has heatmap)
- [ ] One-page CI summary HTML: overall status, worst cells, parity pass/fail, link to full report
- [ ] Short JSON schema reference for `schema_version` and payload fields (external tooling)

### Suite rollout

- [ ] Additional training suites: distributed Megatron, JAX
- [ ] Scaling panel validation on multi-node lab hardware

### Out of scope (for now)

- Per-suite visual themes
- Live reporting during the run (session-end generation stays the model)
- Replacing pytest-html

## References

- `cvs/lib/report/README.md`: how to wire and use the library
- `cvs/lib/inference/ADDING_A_SUITE.md`: inference suite checklist (reports section)
- `cvs/lib/report_plugins.py`: zip bundling and `generate_suite_reports`
