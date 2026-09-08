# Engine templates and key patterns

Resolve paths from the repo root. Prefer a same-topology sibling in the same
directory over a different engine.

## Inference

| Engine | Directory | Topologies |
|---|---|---|
| sglang | `cvs/input/config_file/inference/sglang/` | single, distributed, disaggregated |
| vllm (legacy) | `cvs/input/config_file/inference/vllm/` | single |
| vllm workloads | `cvs/input/config_file/inference/vllm_mi300x_workloads/` | `*_single_config.json`, `*_distributed_config.json` |
| atom | `cvs/input/config_file/inference/atom/` | single, distributed, accuracy, longctx |
| xdit | `cvs/input/config_file/inference/xdit/` | single, distributed |

### SGLang skeletons

| Topology | Config | Threshold key pattern |
|---|---|---|
| distributed | `mi3xx_sglang_deepseek_r1_0528_distributed.json` or `mi3xx_sglang_llama_70b_distributed.json` | `mi325_sglang_deepseek_r1_0528_threshold.json` |
| disaggregated | `mi3xx_sglang_*_disaggregated.json` | same threshold shape; PP in keys is often `1` |
| single | `mi3xx_sglang_*_single.json` | same |

SGLang distributed **config** fields to retarget: `server_params.model`,
`nnodes`, `tensor_parallelism`, `pipeline_parallelism`, `server_node_list`
(one `<changeme>` per node), `max_concurrency`, `sweeps` keys, `sweep.runs`,
`benchmark_params.model_num_params`, `threshold_json`.

SGLang **threshold** keys:

```
ISL=<isl>,OSL=<osl>,TP=<tp>,PP=<pp>,CONC=<conc>
ACC_ISL=<isl>,OSL=<osl>
BENCH=<accuracy.task.id>
```

Perf metrics (keep names): `output_throughput_per_sec`, `mean_ttft_ms`,
`mean_tpot_ms`, `mean_e2e_latency_ms`, `goodput`, `mfu`.

`sweeps` / `sweep.runs` combo strings must use the **actual** TP and PP from
the new config. Do not copy `PP=1` from a disagg-oriented threshold onto a
`PP=2` distributed workload.

### vLLM workloads

Threshold keys follow sweep combo names plus concurrency, with `client.*`,
`gpu.*`, `prom.*` metric objects. Record-only samples use `"value": 0` and
`enforce_thresholds: false`. See
`mi300x_vllm_kimi-k26_mxfp4_distributed_config.json`.

### Atom

See `cvs/input/config_file/inference/atom/README.md`. Pair
`{gpu}_atom_{model}_{precision}[_{mode}].json` with `*_threshold.json`.

## Training

| Engine | Directory |
|---|---|
| megatron | `cvs/input/config_file/training/megatron/` |
| torchtitan | `cvs/input/config_file/training/torchtitan/` |
| jaxmaxtext | `cvs/input/config_file/training/jaxmaxtext/` |

Megatron cell keys (example): `MBS=<mbs>,GBS=<gbs>,PRECISION=<precision>` with
`training.throughput_per_gpu`, `training.elapsed_time_per_iteration`,
`training.tokens_per_gpu`, `training.mem_usage`.

Copy `schema_version` / `framework` / `gpu_arch` when the template has them.
Older SGLang files omit those keys — do not add them unless siblings in that
folder already have them.

## GPU family tokens

| Requested GPU | Config prefix (SGLang) | Threshold prefix | `peak_gpu_tflops` (SGLang samples) | `GPU_ARCHS` |
|---|---|---|---|---|
| MI325X | `mi3xx` | `mi325` | `2615` | `gfx942` |
| MI300X | `mi3xx` or `mi300x` | `mi300x` | copy sibling | `gfx942` |
| MI355X | copy `mi355x` / `mi35x` sibling | `mi355x` | copy sibling | copy sibling |

## Record-only SGLang placeholder cell

Use one object per combo (adjust TP/PP/CONC):

```json
"ISL=1024,OSL=1024,TP=8,PP=2,CONC=4": {
  "output_throughput_per_sec": { "kind": "min_tok_s", "value": 1 },
  "mean_ttft_ms": { "kind": "max_ms", "value": 60000 },
  "mean_tpot_ms": { "kind": "max_ms", "value": 1000 },
  "mean_e2e_latency_ms": { "kind": "max_ms", "value": 120000 },
  "goodput": { "kind": "min", "value": 0 },
  "mfu": { "kind": "min", "value": 0 }
}
```

Accuracy placeholders: `{ "kind": "min", "value": 0 }` for `pass_rate` and
lm-eval metric keys from the template.
