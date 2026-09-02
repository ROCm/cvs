# JAX MaxText Training - Config and Threshold Files

This folder holds the input files for the `jaxmaxtext_single` /
`jaxmaxtext_distributed` suites (see
`cvs/tests/training/jaxmaxtext/README.md` for how to run them). Each **config**
file has a sibling **threshold** file (referenced by its `threshold_json`
field). One config = one GPU arch + mode (single or distributed).

## File inventory

Each config has a sibling threshold file named by its `threshold_json` field
(same stem + `_threshold.json`).

### MI300X

| Config | Model | Mode |
|---|---|---|
| `mi300x_jaxmaxtext_llama-3.3-70b_single.json` | Llama-3.3-70B | single-node |
| `mi300x_jaxmaxtext_llama-3.3-70b_distributed.json` | Llama-3.3-70B | distributed |
| `mi300x_jaxmaxtext_llama-3.1-8b_distributed.json` | Llama-3.1-8B | distributed |
| `mi300x_jaxmaxtext_deepseek-v2-lite_distributed.json` | DeepSeek-V2-Lite | distributed |

### MI325X

| Config | Model | Mode |
|---|---|---|
| `mi325x_jaxmaxtext_llama-3.3-70b_distributed.json` | Llama-3.3-70B | distributed |
| `mi325x_jaxmaxtext_llama-3.1-8b_distributed.json` | Llama-3.1-8B | distributed |
| `mi325x_jaxmaxtext_llama-3.1-405b_distributed.json` | Llama-3.1-405B | distributed |
| `mi325x_jaxmaxtext_deepseek-v2-lite_distributed.json` | DeepSeek-V2-Lite | distributed |

### MI355X

| Config | Model | Mode |
|---|---|---|
| `mi355x_jaxmaxtext_llama-3.3-70b_distributed.json` | Llama-3.3-70B | distributed |
| `mi355x_jaxmaxtext_llama-3.1-8b_distributed.json` | Llama-3.1-8B | distributed |
| `mi355x_jaxmaxtext_llama-3.1-405b_distributed.json` | Llama-3.1-405B | distributed |
| `mi355x_jaxmaxtext_deepseek-v2-lite_distributed.json` | DeepSeek-V2-Lite | distributed |

One config = one GPU arch + mode. Add analogous config + threshold pairs for new
archs/models as needed.

Keys prefixed with `_` (e.g. `_error_patterns_comment`, `_example_ib_hca`) are
inline comments/examples and are ignored by the loader.

## What you MUST change for your cluster / setup

Start from the config closest to your target arch/mode and edit these:

| Where | Variable | Change to |
|---|---|---|
| `container.image` | container image | Your MaxText/JAX ROCm image tag present on the nodes |
| `container.name` | container name | Any unique name (optional) |
| `paths.shared_fs` | base path (`/home/{user-id}`) | A path reachable from all nodes; `{user-id}` resolves to the cluster/OS user |
| `paths.temp_dir` | in-container scratch (`/tmp/{user-id}/jaxmaxtext`) | Host-user-namespaced scratch for launcher scripts / MaxText YAML; keep `{user-id}` so shared nodes never collide on `/tmp/root` |
| `paths.hf_token_file` | HF token path | Location of your Hugging Face token file on the nodes |
| `training.tokenizer.hf_model_id` | tokenizer repo | The HF tokenizer to download (matches the model). Skipped automatically when every enabled run uses `dataset_type: synthetic` |
| `training.tokenizer.tokenizer_path` | in-container tokenizer dir | Where the tokenizer is written (usually under `{paths.models_dir}`) |
| `training.gpus_per_node` | GPUs per node | Your node's GPU count (default 8); feeds total-throughput/scaling metrics - do not assume a fixed topology |
| `training.nic_type` | NIC type | `thor2` (Broadcom) etc. for distributed; `none` for single-node |
| `training.rdma_lib.*` | RDMA lib paths | **Backup** 2-stage copy of the NIC's `libibverbs` provider into the container; used only when a direct read-only `.so` bind-mount is not allowed (distributed). Unused when the `.so` is mounted `:ro` directly (the default) |
| `training.nccl.ib_hca` / `ib_hca_list` | RDMA HCA devices | Your nodes' RDMA device names (e.g. `rdma0..rdma7`) |
| `training.nccl.socket_ifname` / `gloo_socket_ifname` | control NIC | Your management interface name (e.g. `eno0`) |
| `training.nccl.ib_gid_index` | RoCE GID index | Your cluster's GID index (commonly `3`); exported as `NCCL_IB_GID_INDEX`. Shipped as `<changeme>` (distributed hard-exits until set) |
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
| `paths` | `shared_fs`, `models_dir`, `log_dir`, `hf_token_file`, `temp_dir` (host-user scratch, `/tmp/{user-id}/jaxmaxtext`) |
| `model` | `id`, `remote` (0 = already cached), `precision` (label) |
| `container` | `lifetime`, `name`, `image`, `runtime` (docker `args`: network/ipc/privileged/shm-size/ulimit/volumes) |

### `training` block

| Field | Meaning |
|---|---|
| `distributed` | `true` for multi-node (adds the RDMA setup stage), `false` for single-node |
| `gpus_per_node` | GPUs per node (default 8); `num_gpus = num_nodes x gpus_per_node` feeds `tokens_per_sec_total` and scaling efficiency |
| `verify_dmesg` | Scan host `dmesg` on all nodes for GPU/HW/kernel faults over the training window (default `true`); set `false` on clusters without passwordless `sudo` for `dmesg` |
| `steps` | Training steps; also drives completion detection and poll budget |
| `enable_checkpointing` | Whether MaxText writes checkpoints |
| `train_script_paths` | Candidate in-container paths to the MaxText train entrypoint; the job picks the first one that exists in the running container. List them newest-first (e.g. v26.4+ path before the v26.3 path) so a version bump only needs a new entry, not an edit. `train_script` (single path) is still accepted as a deprecated fallback. |
| `maxtext_config` | MaxText YAML params written verbatim (see below) |
| `tokenizer` | `hf_model_id` (download source), `tokenizer_path` (in-container dir). Download is skipped when every enabled run is `dataset_type: synthetic` |
| `nic_type` | NIC family; `thor2` triggers the backup RDMA-lib copy, `none` skips it |
| `rdma_lib` | **Backup** host/container paths for the 2-stage libibverbs copy (distributed); unused when the `.so` is bind-mounted `:ro` directly |
| `env_vars` | Environment exported before training (NCCL/NVTE/HIP/XLA client). `NCCL_IB_TC`/`NCCL_IB_SL` live here; `NCCL_IB_GID_INDEX` is driven by `nccl.ib_gid_index` instead |
| `xla_flags` | `XLA_FLAGS` passed to the run |
| `nccl` | RDMA HCA list + control interface names + `ib_gid_index` (exported as `NCCL_IB_GID_INDEX`) for distributed comms |
| `jax_distributed` | `coordinator_ip` (`auto` = first node), `coordinator_port`, init/heartbeat timeouts |
| `scaling_baseline` | 1-node `tokens_per_sec_total` + `num_nodes` for scaling-efficiency % |
| `convergence` | `target_metric` (`auto`/`train_loss`/`eval_loss`) + `target_value` for time-to-target |
| `loss_curve` | `sample_every`, `milestone_steps`, `max_slope`, `enforce` for the loss-curve check |
| `smoke` | Smoke test (test_smoke), ENABLED by default; `enabled`, `steps`, `per_device_batch_size`, `max_target_length` (see below) |
| `checkpoint_resume` | Checkpoint save+resume + I/O-timing test (test_checkpoint_resume), opt-in (`enabled: false`); see below |
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

### `smoke` block

The smoke test (`test_smoke`) loads the model and runs a few steps at a small
fixed batch/seqlen in BF16, passing only if no `error_patterns`/NaN signature
fires (no metric/threshold checks). A failure gates the rest of the suite.

| Field | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Runs by default (opt-OUT). Set `false` to skip it, e.g. during iterative experiments |
| `steps` | `5` | Steps for the smoke run |
| `per_device_batch_size` | `1` | Small fixed batch for the smoke run |
| `max_target_length` | `2048` | Small fixed sequence length for the smoke run |

You can also skip it ad-hoc without editing the config: `cvs run … -k "not smoke"`.

### `checkpoint_resume` block

Opt-in (`enabled: false` by default). Runs ONE sweep twice: Phase 1 trains
`steps_before_ckpt` with checkpointing on (a checkpoint is written at
`checkpoint_period`); Phase 2 resumes and trains `steps_after_resume` more.
Passes when Phase 2 restarts at the checkpoint step AND the loss at the resume
boundary matches Phase 1 within `loss_tolerance`. Also benchmarks
`checkpoint_save_seconds` / `checkpoint_load_seconds`, gated against
`max_save_seconds` / `max_load_seconds` when those are `> 0` (else record-only).

| Field | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Opt-in switch |
| `sweep` | `""` | Which sweep to exercise (`""` = first enabled) |
| `steps_before_ckpt` | `6` | Phase-1 steps (checkpoint saved at `checkpoint_period`) |
| `steps_after_resume` | `6` | Phase-2 steps after resuming |
| `checkpoint_period` | `5` | Save frequency; must be `<= steps_before_ckpt` or Phase 1 saves nothing |
| `loss_tolerance` | `0.1` | Max loss delta at the resume boundary |
| `max_save_seconds` / `max_load_seconds` | `0.0` | I/O time gates; `0` = record-only |
| `delete_ckpt_dir` | `true` | Delete the checkpoint dir after the test (`false` keeps it for inspection) |
| `smoke_model_overrides` | `{}` | Optional shrink of the model (same tokenizer/vocab) for a fast I/O check |

## Datasets and the tokenizer

`dataset_type: synthetic` (random token ids in `[0, vocab_size)`) needs no
tokenizer, so `test_setup_tokenizer` is skipped automatically when every enabled
run is synthetic. Any other `dataset_type` (or an unset one, MaxText's tfds/C4
default) triggers the HF tokenizer download from `training.tokenizer.hf_model_id`.

Configs ship with `dataset_type: synthetic` by default. Synthetic is ideal for
throughput/functional runs, but the model overfits the fixed random tokens, so
the loss is **not** a real curve and eventually diverges to `NaN` on long runs -
do not read a synthetic loss curve as convergence.

### Using real data (HuggingFace C4)

For a genuine loss curve, switch to real data. C4 is large (~300&nbsp;GB for the
`en` split), but the HF pipeline **streams** it - nothing is fully downloaded.
Set these keys **inside `maxtext_config`**:

```json
"dataset_type": "hf",
"hf_path": "allenai/c4",
"hf_data_dir": "en",
"train_split": "train",
"tokenizer_type": "huggingface"
```

Key notes:

- `tokenizer_type: "huggingface"` is **required** when the tokenizer is a HF
  `tokenizer.json` (e.g. DeepSeek-V4-Flash, Llama-3). MaxText's default is
  `sentencepiece`, which expects a `.model` file and would mismatch a HF
  tokenizer. Match the type to whatever `training.tokenizer.hf_model_id` ships.
- Real data triggers the tokenizer download from `training.tokenizer.hf_model_id`
  into `training.tokenizer.tokenizer_path` (skipped for synthetic), so ensure
  those two fields point at the right repo/dir.
- `hf_path`/`hf_data_dir`/`train_split` select the dataset, its config/subdir,
  and the split. For a tiny alternative, point `hf_path` at a small public set
  (e.g. `stas/c4-en-10k`) and drop `hf_data_dir`.
- For a validation loss, also add `eval_interval > 0` (and `eval_steps`) plus a
  validation split (`hf_eval_split`/`eval_split`); then `convergence.target_metric:
  auto` reports `eval_loss`.

> **Do not put comment (`_`-prefixed) keys inside `maxtext_config`.** Every key in
> `maxtext_config` is written verbatim into the run YAML, and MaxText **rejects
> unknown keys** - a stray `_comment` there will fail the run at config parse.
> Keep notes at the `training` level (e.g. `_dataset_comment`) instead.

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
