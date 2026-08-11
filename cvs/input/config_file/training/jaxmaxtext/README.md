# JAX MaxText Training - Config and Threshold Files

This folder holds the input files for the `jaxmaxtext_single` /
`jaxmaxtext_distributed` suites (see
`cvs/tests/training/jaxmaxtext/README.md` for how to run them). Each **config**
file has a sibling **threshold** file (referenced by its `threshold_json`
field). One config = one GPU arch + mode (single or distributed).

## File inventory

| Config | Threshold | Arch / mode |
|---|---|---|
| `mi300x_jaxmaxtext_llama-3.3-70b_single.json` | `mi300x_jaxmaxtext_llama-3.3-70b_single_threshold.json` | MI300X, single-node |
| `mi300x_jaxmaxtext_llama-3.3-70b_distributed.json` | `mi300x_jaxmaxtext_llama-3.3-70b_distributed_threshold.json` | MI300X, distributed |
| `mi325x_jaxmaxtext_llama-3.3-70b_distributed.json` | `mi325x_jaxmaxtext_llama-3.3-70b_distributed_threshold.json` | MI325X, distributed |

Add analogous config + threshold pairs for other archs (e.g. MI355X) as needed.

Keys prefixed with `_` (e.g. `_error_patterns_comment`) are inline comments and
are ignored by the loader.

## What you MUST change for your cluster / setup

Start from the config closest to your target arch/mode and edit these:

| Where | Variable | Change to |
|---|---|---|
| `container.image` | container image | Your MaxText/JAX ROCm image tag present on the nodes |
| `container.name` | container name | Any unique name (optional) |
| `paths.shared_fs` | base path (`/home/{user-id}`) | A path reachable from all nodes; `{user-id}` resolves to the cluster/OS user |
| `paths.hf_token_file` | HF token path | Location of your Hugging Face token file on the nodes |
| `training.tokenizer.hf_model_id` | tokenizer repo | The HF tokenizer to download (matches the model) |
| `training.tokenizer.tokenizer_path` | in-container tokenizer dir | Where the tokenizer is written (usually under `{paths.models_dir}`) |
| `training.gpus_per_node` | GPUs per node | Your node's GPU count (default 8); feeds total-throughput/scaling metrics - do not assume a fixed topology |
| `training.nic_type` | NIC type | `thor2` (Broadcom) etc. for distributed; `none` for single-node |
| `training.rdma_lib.*` | RDMA lib paths | Host/container paths for the NIC's `libibverbs` provider (distributed) |
| `training.nccl.ib_hca` / `ib_hca_list` | RDMA HCA devices | Your nodes' RDMA device names (e.g. `rdma0..rdma7`) |
| `training.nccl.socket_ifname` / `gloo_socket_ifname` | control NIC | Your management interface name (e.g. `eno0`) |
| `training.jax_distributed.coordinator_ip` | JAX coordinator | Keep `auto` (uses the first node in the cluster `node_dict`), or set a specific IP |
| `training.sweeps[].maxtext_overrides.quantization` | FP8 recipe | `nanoo_fp8` on MI300X/MI325X (CDNA3); `fp8` on MI355X/MI350X (CDNA4) |
| `training.scaling_baseline.tokens_per_sec_total` | 1-node baseline | Your measured single-node total tok/s (0.0 disables scaling efficiency) |
| `<threshold>.json` gated values | thresholds | Calibrated PASS/FAIL bounds for your hardware |
| cluster file `node_dict` | node IPs | Your node IPs (first entry is the coordinator when `coordinator_ip: auto`) |

Also set `training.enabled_sweep_list` to the sweep(s) you want to run (each is a
full training run), and `enforce_thresholds` to `true` for real PASS/FAIL or
`false` for record-only.

## Placeholder substitution

Configs use placeholders resolved at load time:

- `{user-id}` - the cluster username (or the local OS user as fallback).
- `{shared_fs}` - self-reference within the `paths` block.
- `{paths.models_dir}` (and other `{paths.*}`) - cross-referenced anywhere.

`threshold_json` is a literal filename resolved next to the config; no
placeholder substitution is applied to it.

## Config structure

Top-level (framework-agnostic) fields:

| Field | Meaning |
|---|---|
| `schema_version` | Always `1` |
| `framework` | `jaxmaxtext` |
| `gpu_arch` | `mi300x` / `mi325x` / `mi355x` (labels the run) |
| `enforce_thresholds` | `true` = metrics gate PASS/FAIL; `false` = record-only |
| `threshold_json` | Sibling threshold filename |
| `paths` | `shared_fs`, `models_dir`, `log_dir`, `hf_token_file` |
| `model` | `id`, `remote` (0 = already cached), `precision` (label) |
| `container` | `lifetime`, `name`, `image`, `runtime` (docker `args`: network/ipc/privileged/shm-size/ulimit/volumes) |

### `training` block

| Field | Meaning |
|---|---|
| `distributed` | `true` for multi-node (adds the RDMA setup stage), `false` for single-node |
| `gpus_per_node` | GPUs per node (default 8); `num_gpus = num_nodes x gpus_per_node` feeds `tokens_per_sec_total` and scaling efficiency |
| `steps` | Training steps; also drives completion detection and poll budget |
| `enable_checkpointing` | Whether MaxText writes checkpoints |
| `train_script_paths` | Candidate in-container paths to the MaxText train entrypoint; the job picks the first one that exists in the running container. List them newest-first (e.g. v26.4+ path before the v26.3 path) so a version bump only needs a new entry, not an edit. `train_script` (single path) is still accepted as a deprecated fallback. |
| `maxtext_config` | MaxText YAML params written verbatim (see below) |
| `tokenizer` | `hf_model_id` (download source), `tokenizer_path` (in-container dir) |
| `nic_type` | NIC family; `thor2` triggers the RDMA-lib copy, `none` skips it |
| `rdma_lib` | Host/container paths for the NIC's libibverbs provider (distributed) |
| `env_vars` | Environment exported before training (NCCL/NVTE/HIP/XLA client) |
| `xla_flags` | `XLA_FLAGS` passed to the run |
| `nccl` | RDMA HCA list + control interface names for distributed comms |
| `jax_distributed` | `coordinator_ip` (`auto` = first node), `coordinator_port`, init/heartbeat timeouts |
| `scaling_baseline` | 1-node `tokens_per_sec_total` + `num_nodes` for scaling-efficiency % |
| `convergence` | `target_metric` (`auto`/`train_loss`/`eval_loss`) + `target_value` for time-to-target |
| `loss_curve` | `sample_every`, `milestone_steps`, `max_slope`, `enforce` for the loss-curve check |
| `error_patterns` | `{name: regex}` scanned in the training log (see below) |
| `sweeps` | List of `{name, maxtext_overrides}`; `name` is the threshold cell key |
| `enabled_sweep_list` | Subset of sweep names to actually run |

### `maxtext_config` (selected keys)

Written straight into the MaxText YAML, so any valid MaxText param can be set
here. Common ones: `base_config`, `hardware`, `attention`, `dtype`,
`weight_dtype`, `quantization`, `dataset_type`, `per_device_batch_size`,
`max_target_length`, the `ici_*` / `dcn_*` parallelism dims, `remat_policy`,
`scan_layers`, and `eval_interval` / `eval_steps` (set `eval_interval > 0` with a
validation dataset to produce `eval_loss`). Note `steps`, `enable_checkpointing`,
`run_name`, `base_output_directory`, and `tokenizer_path` are injected by the
driver and should not be set here.

### Sweeps and FP8

Each sweep is a full training run; `maxtext_overrides` merges onto
`maxtext_config` for that run. The `name` encodes the cell as
`NNODES=..,STEPS=..,PRECISION=..,BATCH=..,GBS=..,SEQLEN=..` and must match the
key used in the threshold file. `NNODES` (cluster), `STEPS` (`training.steps`),
and `GBS` (derived = `per_device_batch_size x total GPUs`) are labels only - set
the real knobs (`per_device_batch_size`, `max_target_length`, precision) in
`maxtext_overrides`.

FP8 quantization value by arch:

| Arch | `quantization` for FP8 |
|---|---|
| MI300X, MI325X (CDNA3) | `nanoo_fp8` |
| MI355X, MI350X (CDNA4) | `fp8` |

BF16 sweeps use `quantization: ""` and keep `dtype`/`weight_dtype: bfloat16`.

### `error_patterns`

`{name: regex}` scanned in each node's `training.log` during polling; a match
fails that sweep's `test_training_run` with the matched name. Add/remove entries
as you find new signatures. Remove the whole block to fall back to the driver's
built-in defaults. Escape backslashes per JSON (e.g. two backslashes for `\d`).

## Threshold files

A threshold file maps each **sweep name** (cell key) to `{metric: spec}`. A
metric is gated only when `enforce_thresholds: true` and it has a numeric spec;
otherwise it is recorded. Metrics with no value this run report `N/A`.

Threshold kinds:

| kind | Passes when | Notes |
|---|---|---|
| `min` | `actual >= value` | lower bound |
| `max` | `actual <= value` | upper bound |
| `max_ms` | `actual <= value` | upper bound, `ms` in the message |
| `min_tok_s` | `actual >= value` | lower bound, `tok/s` in the message |
| `within` | `value +/- tolerance_pct%` | needs `tolerance_pct` |
| `min_ratio` | `actual / actuals[reference] >= value` | needs `reference` |
| `info` | always | record-only; retains a default `value` placeholder to calibrate later |

Example cell:

```json
"NNODES=2,STEPS=30,PRECISION=FP8,BATCH=3,GBS=48,SEQLEN=8192": {
  "training.tflops_per_sec_per_gpu": { "kind": "min", "value": 350.0 },
  "training.tokens_per_sec_per_gpu": { "kind": "min", "value": 700.0 },
  "training.final_loss":             { "kind": "max", "value": 15.0 },
  "training.loss_decreased":         { "kind": "min", "value": 1 },
  "training.step_time_p95_ms":       { "kind": "info", "value": 3600000.0 }
}
```

To start gating a metric currently marked `info`: replace `"kind": "info"` with
`min`/`max`/etc. and set a calibrated `value`. The threshold cell key must match
the sweep's `name` exactly (including `NNODES`), or the metric falls back to
`RECORD`.
