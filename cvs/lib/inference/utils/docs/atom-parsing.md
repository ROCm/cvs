# ATOM parsing

`atom_parsing.py` owns the **W1 SLO contract** (`GATED_METRICS`) and
ATOM-specific derived metrics. It reuses `vllm_parsing.to_client_metrics` for the
stock `benchmark_serving` / `vllm bench serve` JSON scalars because ATOM and vLLM
bench artifacts share the same keys.

## Drivers and artifacts

| `params.driver` | Result artifact | Parser entry |
| --- | --- | --- |
| `atom` | `{result_stem}.json` from `benchmark_serving` | `to_client_metrics` |
| `vllm`, `vllm_atom` | vLLM bench `{result_stem}` (no `.json` suffix) | `to_client_metrics` |
| `sglang` | SGLang bench log / artifact (client poll on log today) | `to_client_metrics` when JSON present |

Multinode **scaling** adds `scaling.efficiency_pct` when
`params.scaling_baseline_output_throughput` is set (multinode PP configs).

## Why not `vllm_parsing` alone?

- `vllm_single` keeps its own `GATED_METRICS` (legacy suite).
- W1 gates (`per_gpu_throughput`, `output_tput_per_gpu`, tail percentiles) live here.
- Driver choice affects orchestration, not the `client.*` namespace once metrics are parsed.

## Derived metrics (display + gates)

| Metric | Formula |
| --- | --- |
| `client.per_gpu_throughput` | `total_token_throughput / tp` (from shared parser) |
| `client.output_tput_per_gpu` | `output_throughput / tp` (added in this module) |
| `scaling.efficiency_pct` | `output_throughput / (baseline × nnodes) × 100` when baseline set |

## Consumers

- `atom_orch.AtomJob.parse_results`
- `atom_config_loader` threshold coverage (`gated_metrics=GATED_METRICS`)
- `cvs.tests.inference.atom.atom` — `test_cell_metrics` tiers (throughput, ttft, tpot, health, record)
