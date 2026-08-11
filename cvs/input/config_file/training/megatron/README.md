# Megatron Training — Config and Threshold Files

This folder holds the input files for the `megatron_single` / `megatron_distributed` suites (see `cvs/tests/training/megatron/README.md` for how to run them). Each config file has a sibling threshold file referenced by its `threshold_json` field. One config = one GPU arch + mode (single or distributed).

Keys prefixed with `_` (e.g. `_scaling_baseline_comment`) are inline comments and are ignored by the loader.

## File Inventory

| Config | Threshold | Arch / mode |
|---|---|---|
| `mi325x_megatron_llama-3.1-8b_single.json` | `mi325x_megatron_llama-3.1-8b_single_threshold.json` | MI325X, single-node |
| `mi325x_megatron_llama-3.3-70b_single.json` | `mi325x_megatron_llama-3.3-70b_single_threshold.json` | MI325X, single-node |
| `mi325x_megatron_llama-3.3-70b_distributed.json` | `mi325x_megatron_llama-3.3-70b_distributed_threshold.json` | MI325X, distributed |
| `mi325x_megatron_deepseek-v2-lite_single.json` | `mi325x_megatron_deepseek-v2-lite_single_threshold.json` | MI325X, single-node |

Add analogous config + threshold pairs for other archs (e.g. MI355X) as needed.

## What You MUST Change for Your Cluster

Start from the config closest to your target arch/mode and edit these:

| Where | Field | Change to |
|---|---|---|
| `container.image` | Docker image | Your Megatron-LM ROCm image tag accessible on all nodes |
| `container.name` | Container name | Any unique name (optional) |
| `config.hf_token_file` | HF token path | Location of your Hugging Face token file on the nodes |
| `config.log_dir` / `scripts_dir` / `data_cache_dir` | Paths | Replace `{user-id}` with your actual username |
| `config.megatron_root` | Megatron path | In-container path to Megatron-LM (default `/workspace/Megatron-LM`) |
| `config.nnodes` | Node count | Number of nodes in your cluster (**distributed only**) |
| `config.master_address` | Head node IP | IP of the head node (**distributed only**) |
| `config.nic_type` | NIC family | `thor2` (Broadcom) or your NIC type (**distributed only**) |
| `config.nccl_ib_hca_list` / `nccl_ib_hca` | RDMA HCA devices | Your nodes' RDMA device names (e.g. `bnxt_re0,...,bnxt_re7`) (**distributed only**) |
| `config.nccl_socket_ifname` / `gloo_socket_ifname` | Control NIC | Your management interface name (e.g. `ensf1np1`) (**distributed only**) |
| `scaling_baseline.tokens_per_sec_total` | 1-node baseline | Your measured single-node total tok/s (`tok/s/GPU × 8`); `0.0` disables scaling efficiency (**distributed only**) |
| Threshold JSON gated values | Thresholds | Calibrated PASS/FAIL bounds for your hardware |
| Cluster file | Node IPs | Your node IPs (first entry is the coordinator node) |

Also set `sweep.runs` to the combo(s) you want to run, and `enforce_thresholds` to `true` for real PASS/FAIL or `false` for record-only.

## Placeholder Substitution

Configs use `{user-id}` in path fields, resolved at load time to the cluster username (or local OS user as fallback). Unresolved `<changeme>` placeholders cause a hard exit at startup.

## Config Structure

Top-level fields:

| Field | Meaning |
|---|---|
| `schema_version` | Always `1` |
| `framework` | `megatron_single` (single-node) or `megatron_distributed` (multi-node) |
| `gpu_arch` | `MI325X` / `MI300X` etc. — labels the run, informational |
| `enforce_thresholds` | `true` = metrics gate PASS/FAIL; `false` = record-only |
| `threshold_json` | Sibling threshold filename; resolved next to the config |
| `scaling_baseline` | 1-node baseline for scaling efficiency % (distributed only) |
| `config` | Runtime, paths, NCCL, and NIC settings |
| `model_params` | Model architecture and default hyperparameters |
| `container` | Docker container settings |
| `sweep` | Training combinations and the ordered run list |

### `config` block

| Field | Default | Description |
|---|---|---|
| `hf_token_file` | `/home/{user-id}/.hf_token` | Hugging Face access token file path |
| `log_dir` | `/home/{user-id}/LOGS/megatron` | Training log output directory |
| `scripts_dir` | `/home/{user-id}/SCRIPTS/megatron` | Generated per-node wrapper scripts directory |
| `data_cache_dir` | `/home/{user-id}/cache` | Tokenizer and dataset cache directory |
| `rocm_dir` | `""` | ROCm path; empty string triggers auto-detection |
| `megatron_root` | `/workspace/Megatron-LM` | Megatron-LM path inside the container |
| `training_iterations` | `"10"` | Training iterations per combo |
| `nnodes` | `"1"` / `<changeme>` | Node count; must match cluster file |
| `master_address` | `"127.0.0.1"` / `<changeme>` | Head-node IP for distributed coordination |
| `nic_type` | `"thor2"` | NIC family; `thor2` triggers Broadcom RDMA-lib copy |
| `nccl_ib_hca_list` / `nccl_ib_hca` | `<changeme>` | Comma-separated RDMA HCA list |
| `nccl_socket_ifname` / `gloo_socket_ifname` | `"ensf1np1"` | Control-plane interface name |
| `hca_id_pattern` | `"bnxt_\|rocep"` | `\|`-separated NIC prefixes for ibv_devinfo validation |
| `nccl_ib_gid_index` | `"3"` | GID index for RoCE (standard `"3"` for Broadcom) |
| `nccl_debug` | `"ERROR"` | NCCL log verbosity (`"ERROR"`, `"WARN"`, `"INFO"`, `"TRACE"`) |
| `verify_network_errors` | `"False"` | `"True"` to compare RDMA/ethtool error counters before and after training |

### `model_params` block

Defaults applied to every sweep combo; individual combos override them.

| Field | Description |
|---|---|
| `model_name` | Friendly name used in log paths and labels |
| `tokenizer_model` | Hugging Face repo ID (e.g. `"meta-llama/Llama-3.1-8B"`) |
| `model_size` | Parameter count in billions (e.g. `"8"`, `"70"`) |
| `sequence_length` | Sequence length in tokens |
| `micro_batch_size` | Default micro-batch size (overridden by sweep) |
| `global_batch_size` | Default global batch size (overridden by sweep) |
| `tensor_parallelism` | Tensor parallel degree (TP) |
| `pipeline_parallelism` | Pipeline parallel degree (PP) |
| `recompute` | Activation recompute (`"0"` off, `"1"` on) |
| `fsdp` | Fully Sharded Data Parallel (`"0"` off, `"1"` on) |
| `precision` | Default precision; overridden by sweep (`"FP8"`, `"BF16"`, `"MXFP4"`, `"MXFP8"`) |

### `container` block

| Field | Description |
|---|---|
| `lifetime` | `"per_run"` — launched once per session, torn down after |
| `name` | Docker container name |
| `image` | **Required** — Docker image URI; replace `<changeme>` |
| `runtime.args.volumes` | Host paths volume-mounted into the container |
| `runtime.args.devices` | Host devices exposed (`/dev/kfd`, `/dev/dri` for AMD GPUs) |

Distributed configs additionally mount the Broadcom RDMA library and expose `/dev/infiniband/rdma_cm`:

```json
"/usr/local/lib/libbnxt_re-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so.host",
"/lib/libibverbs.d:/lib/libibverbs.d"
```

### `scaling_baseline` block — distributed only

| Field | Description |
|---|---|
| `tokens_per_sec_total` | Single-node baseline total tok/s (`tok/s/GPU × 8`); `0.0` disables efficiency calculation |
| `num_nodes` | Number of nodes used for the baseline (typically `1`) |

## Sweeps

Each entry in `sweep.combinations` is one parametrized training run. `sweep.runs` is the ordered list of combo IDs to execute; omit it to run all combinations.

```json
"sweep": {
  "combinations": {
    "llama3_1_8b-mi325-bs128-mbs4-fp8": {
      "name": "llama3_1_8b_mbs4_gbs128_FP8",
      "global_batch_size": "128",
      "micro_batch_size": "4",
      "precision": "FP8"
    }
  },
  "runs": ["llama3_1_8b-mi325-bs128-mbs4-fp8"]
}
```

| Combo field | Description |
|---|---|
| `name` | Human-readable label (used in reports) |
| `global_batch_size` | Global batch size for this combo |
| `micro_batch_size` | Micro-batch size for this combo |
| `precision` | Precision override (`"FP8"`, `"BF16"`, `"MXFP4"`, `"MXFP8"`) |

Any key in a combo overrides the matching `model_params` field — adding a new sweep parameter (e.g. `tensor_parallelism`) requires only a config edit, no code change.

## Threshold Files

A threshold file maps each sweep combo (cell key) to per-metric pass/fail limits. A metric is gated only when `enforce_thresholds: true` and it has a numeric spec; otherwise it is recorded.

Cell keys must match the format `MBS=<mbs>,GBS=<gbs>,PRECISION=<precision>`.

Example cell:

```json
"MBS=4,GBS=128,PRECISION=FP8": {
  "training.throughput_per_gpu":        { "kind": "min", "value": 100 },
  "training.tokens_per_gpu":            { "kind": "min", "value": 1000 },
  "training.elapsed_time_per_iteration":{ "kind": "max", "value": 500 },
  "training.mem_usage":                 { "kind": "max", "value": 0 }
}
```

### Threshold kinds

| Kind | Passes when |
|---|---|
| `min` | actual ≥ value |
| `max` | actual ≤ value |
| `min_ratio` | actual / reference ≥ value (needs `reference` key) |

### Tracked metrics

| Metric | Description |
|---|---|
| `training.throughput_per_gpu` | TFLOP/s per GPU |
| `training.tokens_per_gpu` | Tokens per GPU per second |
| `training.elapsed_time_per_iteration` | Wall time per training step (ms) |
| `training.mem_usage` | GPU memory usage |

To start gating a metric: set a calibrated `value` and the appropriate `kind`. The cell key must match the combo's `MBS=`, `GBS=`, and `PRECISION=` values exactly, or the metric falls back to record-only.
