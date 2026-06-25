# InferenceX ATOM parsing

`inferencex_atom_parsing.py` owns the **IX W1 SLO contract** (`GATED_METRICS`) and
ATOM-specific derived metrics. It reuses `vllm_parsing.to_client_metrics` for the
stock `benchmark_serving` / `vllm bench serve` JSON scalars because ATOM emits the
same artifact keys.

## Why not `vllm_parsing`?

- `vllm_single` keeps its own `GATED_METRICS` (vLLM parity is a separate milestone).
- W1 IX gates (`per_gpu_throughput`, `output_tput_per_gpu`, tail percentiles) are
  ATOM automation scope until vLLM parity lands.
- GPT-OSS uplift configs may still set `params.driver=vllm`; they use the same
  `InferenceXAtomJob` + this module for metric display and threshold coverage.

## Derived metrics (IX-only display + gates)

| Metric | Formula |
|---|---|
| `client.per_gpu_throughput` | `total_token_throughput / tp` (from shared parser) |
| `client.output_tput_per_gpu` | `output_throughput / tp` (added in this module) |

## Consumers

- `inferencex_atom_orch.InferenceXAtomJob.parse_results`
- `inferencex_atom_config_loader` threshold coverage (`gated_metrics=GATED_METRICS`)
- `cvs.tests.inference.inferencex_atom.inferencex_atom_single` HTML rows
