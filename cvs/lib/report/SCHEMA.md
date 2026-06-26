# Inference suite report JSON (`schema_version: 1`)

Machine-readable sidecar written as `{report_basename}.json` alongside the HTML dashboard.
External tools should read `schema_version` first and ignore unknown top-level keys.

## Top-level fields

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | `int` | Currently `1` |
| `suite_id` | `string` | Registry id from the suite preset |
| `generated_at` | `string` | UTC timestamp, e.g. `2026-06-25 12:34 UTC` |
| `cvs_version` | `string` | Installed CVS package version |
| `overall_status` | `string` | `pass`, `fail`, `record`, or `na` |
| `report` | `object` | Title, subtitle, footer, tier order, headline metric |
| `run_card_display` | `array` | `[label, value, is_link]` tuples for the run card |
| `provenance` | `object` | Paths, git commit, CVS version (see below) |
| `lifecycle` | `object` | Session stage label → seconds |
| `cells` | `array` | Per sweep cell records (primary data) |
| `chart_series` | `object` | Metric suffix → `[[concurrency, value], ...]` |
| `chart_config` | `array` | Chart metadata for the interactive viewer |
| `sweep_summaries` | `array` | Per ISL/OSL aggregate stats |
| `gate_matrix` | `array` | `{label, cell_id, concurrency, tiers}` rows |
| `results_table` | `object` | `{headers, rows}` full tabular export |
| `panels` | `object` | Optional sections: `scaling`, `prev_run`, … |
| `summary` | `object` | Truncation mode, viewer basename, cell counts |

## `provenance` (common keys)

| Key | Description |
|-----|-------------|
| `pytest_html_path` | Absolute path to pytest HTML on the launcher |
| `pytest_html_basename` | Basename for zip-relative links |
| `log_file_path` | Run log path |
| `cluster_file` | `--cluster_file` argument |
| `config_file` | `--config_file` argument |
| `git_commit` | Short git SHA when available |
| `cvs_version` | Duplicate of top-level version |

## `cells[]` entry

| Field | Type | Description |
|-------|------|-------------|
| `cell_id` | `string` | Threshold / gate key for the cell |
| `host` | `string` | Node or client host |
| `isl`, `osl`, `policy`, `concurrency` | | Sweep dimensions |
| `actuals` | `object` | Full metric dict, e.g. `client.output_throughput` |
| `tiers` | `object` | Tier name → `pass` / `fail` / `record` / `na` |
| `metrics` | `array` | Highlight metrics with specs, margins, bar_pct |
| `pytest_inference_nodeid` | `string` | Optional pytest nodeid for deep links |
| `pytest_metrics_nodeid` | `string` | Optional metric-test nodeid |

## `panels.prev_run` (when baseline JSON configured)

| Field | Description |
|-------|-------------|
| `baseline_json` | Path to the comparison JSON |
| `threshold_pct` | Flag threshold for throughput delta |
| `rows[]` | Per-cell `current_throughput`, `previous_throughput`, `compare.prev_run.throughput_delta_pct`, `regression` |

## `panels.scaling` (multi-node runs)

| Field | Description |
|-------|-------------|
| `nnodes` | Node count |
| `cluster_throughput` | Aggregated tok/s |
| `compare.scaling.efficiency_pct` | Optional vs single-node baseline |
| `per_node_rows` | Host-level breakdown |

## Training reports

Training JSON uses `report_kind: "training"`, `nodes`, `node_rows`, and the same `schema_version: 1`.
See `training.py` payload shape.

## Related artifacts (not in this JSON)

| File | Purpose |
|------|---------|
| `{basename}_viewer.html` | Interactive explorer (loads this JSON) |
| `{basename}_summary.html` | One-page CI summary |
| `inference_parity_report.json` | Cross-framework parity (separate schema) |
