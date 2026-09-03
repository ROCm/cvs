# vllm MI3XX workload configs

14 MI325X inference workloads for the `vllm_single` and `vllm_distributed`
suites, each shipped as a `single` / `distributed` pair — 28 configs and 28
sibling thresholds.

## Layout

Flat sibling pairs, same convention as `inferencex_atom_single`:

```text
mi3xx_vllm_{model}_{precision}_{topology}.json
mi325x_vllm_{model}_{precision}_{topology}_threshold.json
```

`topology` is `single` or `distributed`. Each config points `threshold_json` at
its sibling filename, so **copy one variant at a time into its own directory**
on the lab machine — `substitute_config` globs the config's parent for
`*threshold.json` and raises on more than one match only when `threshold_json`
is absent, but keeping one pair per directory avoids the trap entirely.

## Topology

| Mode | PP | Host behavior |
|---|---|---|
| `single` | 1 | first cluster host only |
| `distributed` | 2 | exactly two cluster hosts form one service |

TP is **per model**, following the source workload list: TP=4 for
`deepseek-v4-flash`, `kimi-k26`, `kimi-k25` and `gpt-oss-120b`; TP=8 for
everything else. A TP=4 distributed variant still spans 2 nodes via PP=2,
using 4 GPUs per node.

Distributed configs support one-host fallback or exactly two hosts. CVS rejects
larger clusters until a matching N-host recipe and threshold set are available.

## Sweeps

`sweeps` maps canonical run-cell keys to per-cell benchmark overrides.
`runs` is a required, ordered, explicit list of the cells to execute. To run
every available cell, list every key in `runs`; there is no implicit "all"
selection.

```text
ISL=1024,OSL=1024,TP=<tp>,PP=<pp>,CONC=<concurrency>
```

The TP and PP in every key must match `server_params.tensor_parallel_size` and
`server_params.pipeline_parallel_size`, including `PP=1` for single-node
configs. Cell values override `benchmark_params`.

`benchmark_params.random_range_ratio` is `0.0` so ISL/OSL are exact rather
than jittered ±80%.

`num_prompts` is **320**, not the `3200` schema default used by the configs in
`cvs/input/config_file/inference/vllm/`. That makes each cell a characterization
pass — enough to shake out topology, AITER and kv-cache settings on new
hardware, at roughly a tenth the wall-clock. Raise it to `3200` before quoting
numbers that need to line up with the shipped examples.

## Thresholds

`enforce_thresholds` is **false** on every config, so nothing gates and metrics
are only recorded. Each threshold file carries a placeholder for every metric
the suite can gate on — 32 per file. That full grid is a convenience for later
calibration, **not** a loader requirement: the vLLM loader checks cell coverage
only, and an absent metric spec means "don't gate this metric". A threshold file
may gate just the handful of metrics you care about.

| Family | Count | Source of the list |
|---|---|---|
| `client.*` | 23 | the suite's gated-metric set |
| `gpu.*` | 5 | `cvs.lib.utils.gpu.GPU_METRICS` |
| `prom.*` | 4 | `vllm_server_metrics.PROM_METRICS` |

**Every value is `0`**, meaning *not yet measured* — not a real bound. On a
`max`/`max_ms` kind, `0` is an impossible bound, so enabling enforcement before
calibrating fails loudly rather than passing silently. Replace them with
measured values from a calibration run before flipping `enforce_thresholds`.

### Accuracy

Accuracy is split across the two files, unlike the three families above:

- **`config.json` → `accuracy.tasks`** selects *which* lm-eval tasks run.
  Shipped empty, so no accuracy stage runs and the pytest node is auto-skipped.
- **`threshold.json` → `accuracy`** holds the gating values, keyed by task id
  then by lm-eval metric key. Shipped as `{}`.

Because the threshold keys are derived from the task ids you choose, they
cannot be pre-enumerated the way `client.*`/`gpu.*`/`prom.*` can — the two
blocks must be filled in together:

```jsonc
// config.json
"accuracy": {"tasks": [{"id": "gsm8k", "task": "gsm8k", "num_fewshot": 5}]}

// threshold.json
"accuracy": {"gsm8k": {"gsm8k.exact_match__strict-match": {"kind": "min", "value": 0}}}
```

The metric key is the lm-eval `results.json` key with commas replaced by `__`.
The `accuracy` block is exempt from the sweep-cell coverage check via
`NON_SWEEP_THRESHOLD_KEYS`, so it does not need a cell key.

## Before running — fill in the `<changeme>` fields

Every environment-specific value is redacted. Per config:

| Field | What to set |
|---|---|
| `server_params.model` | Local model path (e.g. `/models/GLM-5.1-FP8`) |
| `container.image` | The vLLM/ROCm image tag under test |
| `container.runtime.args.volumes[1]` | Replace `<changeme-models-mount>` with the host models directory |
| `ib_netdev` | *(distributed only)* socket interface name for `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` / `TP_SOCKET_IFNAME`. Must be **UP and hold a routable IPv4 reaching the other node** — check `ip -o -4 addr show`, not just `ip -o link show`. |

`paths.models_dir` is `/models`, the in-container mount point — it is exported
as `HF_HUB_CACHE`. When `model.id` is an absolute path under `/models`, vLLM
loads straight from the mount and no download occurs. CVS derives the
distributed rendezvous address from the cluster head.

## Workload set

| Config stem | Model | Notes |
|---|---|---|
| `llama33-70b_fp8` | Llama 3.3 70B FP8 | `kv-cache-dtype: fp8` |
| `glm-51_fp8` | GLM 5.1 FP8 | |
| `glm-52_fp8` | GLM 5.2 FP8 | |
| `deepseek-v4-pro_fp8` | DeepSeek V4 Pro FP8 | |
| `deepseek-v4-flash_fp8` | DeepSeek V4 Flash FP8 | |
| `kimi-k26_mxfp4` | Kimi K2.6 MXFP4 | |
| `kimi-k27-code_mxfp4` | Kimi K2.7 Code MXFP4 | |
| `kimi-k25_w4a8` | Kimi K2.5 W4A8 | |
| `qwen35-397b-a17b_bf16` | Qwen3.5 397B A17B BF16 | |
| `minimax-m3_bf16` | MiniMax M3 BF16 | |
| `mimo-v25-pro_fp8` | MiMo V2.5 Pro FP8 | |
| `mistral-large-3_bf16` | Mistral Large 3 BF16 | Mistral-native format: `tokenizer-mode`/`config-format`/`load-format` all `mistral` |
| `deepseek-r1-0528_fp8` | DeepSeek R1 0528 FP8 PTPC | |
| `gpt-oss-120b_mxfp4` | GPT-OSS 120B MXFP4 | |

Server options live directly under `server_params` as snake-case vLLM options:
`gpu_memory_utilization` becomes `--gpu-memory-utilization`. Benchmark options
live under `benchmark_params`; `trust_remote_code` is set independently there
when the bench client needs it.

## Running

```bash
cd ~/cvs && source .cvs_venv/bin/activate

VAR=mi3xx_vllm_glm-51_fp8_single
DIR=~/input/config_file/inference/vllm/$VAR
mkdir -p "$DIR"

THRESHOLD=mi325x_vllm_glm-51_fp8_single_threshold.json
cvs copy-config inference/vllm/${VAR}.json \
  --output "$DIR/${VAR}.json"
cvs copy-config "inference/vllm/${THRESHOLD}" \
  --output "$DIR/${THRESHOLD}"
# then edit "$DIR/${VAR}.json" and fill in every <changeme>

TS=$(date +%Y%m%d_%H%M%S)
cvs run vllm_single \
  --cluster_file ~/input/cluster_file/<your-cluster>.json \
  --config_file "$DIR/${VAR}.json" \
  --html=~/cvs_results/${TS}_${VAR}.html \
  --self-contained-html \
  --log-file=~/cvs_results/${TS}_${VAR}.log \
  -vvv -s
```

Do not pass pytest function names — let the suite's default tests run.
For a ``*_distributed.json`` file, use ``cvs run vllm_distributed`` instead.

## See also

This file covers only what is specific to this workload set. For the vLLM suite
itself — threshold kinds, cell-key format, multinode prerequisites, accuracy
metric keys — see the suite reference and how-to:

- `docs/reference/configuration-files/inference/vllm.rst`
- `docs/how-to/run-vllm-benchmarks.rst`
