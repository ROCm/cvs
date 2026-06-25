# Architecture

Developer guide: **[README.md](README.md)**.

## Layering

```text
Generic engines (cvs/lib/report/*.py, parity/, panels/, viewer/)
        ↑ configured by
Presets (cvs/lib/report/presets/<suite>.py)
        ↑ registered from
Suite conftest (configure_*_suite_report + session bind + row extras)
        ↑ triggered at session end by
cvs/conftest.py → HtmlReportManager.generate_suite_reports() → zip
```

Inference and training use separate config types (`InferenceReportConfig`,
`TrainingReportConfig`). Optional panels (scaling, parity) merge into the same HTML/JSON
when data is present; they are not separate zip roots.
