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
- Interactive viewer (`*_report_viewer.html`) driven by JSON sidecar
- Optional inference parity (env `CVS_INFERENCE_PARITY_COMPARE` or preset `parity_compare_jsons`)
- Optional training parity (`parity_baseline_json` or `CVS_TRAINING_PARITY_BASELINE_JSON`)
- Reference suite: `inferencex_atom_single`

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

## Follow-ups (optional)

- `vllm_single` preset and conftest wiring
- Additional training suites (distributed Megatron, JAX)
- Offline Chart.js in viewer (today loads from CDN)

## References

- `cvs/lib/report/README.md` — how to wire and use the library
- `cvs/lib/inference/ADDING_A_SUITE.md` — inference suite checklist (reports section)
- `cvs/lib/report_plugins.py` — zip bundling and `generate_suite_reports`
