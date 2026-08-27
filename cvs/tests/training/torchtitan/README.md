# TorchTitan Training Suite (single-node and distributed)

Cluster validation suite that runs TorchTitan pre-training on AMD Instinct GPUs (single-node or multi-node) and gates the run on performance and correctness metrics with a PASS/FAIL HTML report.

## Overview

The suite drives a TorchTitan training job inside a Docker container on one or more cluster nodes, then parses the training log to produce metrics and verdicts. It provides:

- **Two suites** — `torchtitan_single` (single-node) and `torchtitan_distributed` (multi-node, adds RDMA/NIC setup).
- **Parameter sweeps** — one full training run per enabled combo (e.g. FP8, BF16, MXFP8), each with its own result rows in the report.
- **Loss curve** — a per-combo decreasing-trend check at configurable milestone steps (default: 100 / 500 / 1k / 5k).
- **Checkpoint save/resume** — optional two-phase checkpoint testing with I/O timing and loss continuity validation.
- **Convergence tracking** — steps and wall-clock time to reach a target loss.
- **Training-log error scanning** — NCCL, GPU HW faults, OOM, and other signatures fail a run early with a clear reason.
- **HTML report** — per-test rows with linked logs and a consolidated metric results page.

The mode (single vs distributed) is determined by the config file's `framework` field (`torchtitan_single` or `torchtitan_distributed`).

## Quick Start

### Single-node

```bash
cvs run torchtitan_single \
  --cluster_file input/cluster_file/cluster.json \
  --config_file input/config_file/training/torchtitan/mi355x_torchtitan_llama-3.1-8b_single.json \
  --html ./logs/torchtitan_single.html --self-contained-html -vvv -s
```

### Distributed (multi-node)

```bash
cvs run torchtitan_distributed \
  --cluster_file input/cluster_file/cluster.json \
  --config_file input/config_file/training/torchtitan/mi355x_torchtitan_llama-3.3-70b_distributed.json \
  --html ./logs/torchtitan_distributed.html --self-contained-html -vvv -s
```

- `--cluster_file` — JSON describing the node(s); see `cvs/input/cluster_file/README.md`.
- `--config_file` — one of the config files in `cvs/input/config_file/training/torchtitan/`; see that folder's README for the full variable reference.
- `--html` / `--self-contained-html` — write the HTML report.

Use a single-node config with `torchtitan_single` and a distributed config with `torchtitan_distributed`. The config's `framework` field must match the suite.

### Run a specific stage

```bash
cvs run torchtitan_single test_smoke \
  --cluster_file input/cluster_file/cluster.json \
  --config_file input/config_file/training/torchtitan/mi355x_torchtitan_llama-3.1-8b_single.json
```

## The Two Suites

| Suite (`cvs run <name>`) | File | Distributed stages | Use with |
|---|---|---|---|
| `torchtitan_single` | `torchtitan_single.py` | none | single-node config (`framework: torchtitan_single`) |
| `torchtitan_distributed` | `torchtitan_distributed.py` | `test_setup_rdma` | multi-node config (`framework: torchtitan_distributed`) |

Both suites share fixtures and hooks from `conftest.py`.

## Test Lifecycle

Tests run in this pinned order. `[combo]` = one row per enabled sweep combo.

| Order | Test | Runs on | Purpose |
|---|---|---|---|
| 0 | `test_launch_container` | once | Launch and verify the container |
| 1 | `test_setup_rdma` | distributed only | Copy RDMA lib into container (thor2 NIC) and verify `ibv_devinfo` |
| 2 | `test_download_tokenizer` | once | Download HF tokenizer for models that require a local file (DeepSeek, Mixtral) |
| 3 | `test_smoke` | once | Fixed small run confirming the model loads and trains without error |
| 4 | `test_training[combo]` | per combo | Build cmd, train, poll logs, parse results; GPU memory freed between combos |
| 5 | `test_checkpoint[combo]` | per combo (optional) | Two-phase checkpoint save + resume test with step counter and loss continuity validation |
| 6 | `test_metric[combo]` | per combo | Threshold PASS/FAIL per metric |
| 7 | `test_loss_curve[combo]` | per combo | Gate on downward loss trend at milestone steps |
| 8 | `test_teardown` | once | Tear the container down |

A training failure is isolated to that combo's `test_training` row; other combos still run. When a combo's training does not complete, its downstream `test_checkpoint`, `test_metric`, and `test_loss_curve` rows are skipped. If an early lifecycle stage fails, all subsequent stages are skipped via `lifecycle.failed`.

On a training failure, lingering GPU processes are killed (`stop_training_processes`) so the next combo does not launch on top of them.

## Sweeps

A sweep combo is one full training run declared in `sweep.combinations`. `sweep.runs` is the ordered list of combo IDs to execute; set it to a subset to run only selected combos without editing `combinations`.

The combo ID (e.g. `llama3_1_8b-mi355-bs48-mbs6-bf16`) appears in every parametrized row: `test_training[llama3_1_8b-mi355-bs48-mbs6-bf16]`, `test_checkpoint[llama3_1_8b-mi355-bs48-mbs6-bf16]`, `test_metric[llama3_1_8b-mi355-bs48-mbs6-bf16]`, and `test_loss_curve[llama3_1_8b-mi355-bs48-mbs6-bf16]`.

### Multiple Precision Sweeps per Config

TorchTitan configs follow the same pattern as Megatron: one config file can contain multiple precision sweeps (BF16, FP8, MXFP8, MXFP4) for the same model. Example:

```json
"sweep": {
  "combinations": {
    "llama3_1_8b-mi355-bs48-mbs6-bf16": {
      "name": "llama3_1_8b_mbs6_gbs48_BF16",
      "global_batch_size": "48",
      "micro_batch_size": "6",
      "precision": "BF16"
    },
    "llama3_1_8b-mi355-bs48-mbs6-fp8": {
      "name": "llama3_1_8b_mbs6_gbs48_FP8",
      "global_batch_size": "48",
      "micro_batch_size": "6",
      "precision": "FP8"
    },
    "llama3_1_8b-mi355-bs48-mbs6-mxfp8": {
      "name": "llama3_1_8b_mbs6_gbs48_MXFP8",
      "global_batch_size": "48",
      "micro_batch_size": "6",
      "precision": "MXFP8"
    }
  },
  "runs": [
    "llama3_1_8b-mi355-bs48-mbs6-bf16",
    "llama3_1_8b-mi355-bs48-mbs6-fp8",
    "llama3_1_8b-mi355-bs48-mbs6-mxfp8"
  ]
}
```

## Metrics and PASS/FAIL

Each `test_metric[combo]` compares the parsed metric against its threshold spec and reports one of:

| Status | Meaning |
|---|---|
| PASS | value satisfies the threshold |
| FAIL | value violates the threshold (row is red; aggregated in the summary) |
| RECORD | no threshold defined, or `enforce_thresholds: false` — value logged, not gated |

Metrics surfaced (namespace `training.*`):

| Metric | Description |
|---|---|
| `training.throughput_per_gpu` | TFLOP/s per GPU |
| `training.tokens_per_sec` | Tokens per second (total across all GPUs) |
| `training.elapsed_time_per_iteration` | Wall time per training step (ms) |
| `training.mem_usage` | GPU memory usage |
| `training.scaling_efficiency_pct` | Multi-node scaling efficiency % vs single-node baseline (distributed only) |

Gating requires `enforce_thresholds: true` in the config. Set to `false` for record-only runs.

## Checkpoint Save/Resume Testing

When `checkpoint.enforce: true` in the config, `test_checkpoint[combo]` runs a two-phase test:

**Phase 1 — Save:**
- Train for `checkpoint.save_iters` iterations with periodic checkpoints every `checkpoint.save_interval` steps
- Parse checkpoint save I/O timing from logs
- Capture step-to-loss mapping

**Phase 2 — Resume:**
- Load the latest checkpoint and continue training for `checkpoint.resume_iters` iterations
- Verify step counter restoration
- Validate loss continuity (within `checkpoint.loss_rtol` tolerance)
- Parse checkpoint load I/O timing

Checkpoint I/O times (save and load) are logged but not gated. Loss continuity is a hard PASS/FAIL check.

**Config example:**

```json
"checkpoint": {
  "enforce": true,
  "save_interval": 20,
  "save_iters": 21,
  "resume_iters": 25,
  "loss_rtol": 0.05,
  "checkpoint_dir": "/shared/torchtitan/checkpoints"
}
```

Set `enforce: false` to skip checkpoint testing entirely.

## Convergence Tracking

When `convergence.target_value > 0` in the config, `test_training[combo]` computes and logs:

- **Steps to target:** First step where loss ≤ target_value
- **Time to target:** Cumulative wall-clock seconds to reach the target

Convergence metrics are informational and never gate PASS/FAIL.

**Config example:**

```json
"convergence": {
  "target_metric": "auto",
  "target_value": 2.0
}
```

- `target_metric`: `"auto"` (use eval_loss if available, else train_loss), `"train_loss"`, or `"eval_loss"`
- `target_value`: Target loss threshold; `0.0` disables convergence tracking

## Scaling Efficiency (distributed only)

`test_training` computes scaling efficiency as:

```
efficiency % = (actual_total_tok/s / (actual_nodes / baseline_nodes)) / baseline_total_tok/s × 100
```

Populate `scaling_baseline.tokens_per_sec_total` in the config from a completed single-node run (`tok/s × num_gpus`). Set to `0.0` to disable and collect data only.

## Training-Log Error Detection

During polling, each node's training log is scanned for known error patterns. Defaults cover:

- NCCL errors and timeouts
- GPU hardware faults and hangs
- PyTorch distributed errors
- TorchTitan-specific errors

A match fails that combo's `test_training` with the matched pattern name and the last lines of the log.

## Reports and Logs

- **Results table** — one row per test; metric rows show PASS/FAIL from the threshold check.
- **Full log** — each test row links to its own captured log.
- **Training logs** — written inside the container at `<log_dir>/torchtitan-logs/<combo_id>/training.log`.

Log path fields:

| Placeholder | Source |
|---|---|
| `<log_dir>` | `config.log_dir` in the config file |
| `<combo_id>` | Sweep run ID (e.g. `llama3_1_8b-mi355-bs48-mbs6-bf16`) |

## Config and Threshold Files

Located in `cvs/input/config_file/training/torchtitan/`:

| Config | Threshold | Model | Arch / mode |
|---|---|---|
| `mi355x_torchtitan_llama-3.1-8b_single.json` | `mi355x_torchtitan_llama-3.1-8b_single_threshold.json` | Llama-3.1-8B | MI355X, single-node |
| `mi355x_torchtitan_llama-3.3-70b_single.json` | `mi355x_torchtitan_llama-3.3-70b_single_threshold.json` | Llama-3.3-70B | MI355X, single-node |
| `mi355x_torchtitan_llama-3.3-70b_distributed.json` | `mi355x_torchtitan_llama-3.3-70b_distributed_threshold.json` | Llama-3.3-70B | MI355X, distributed |
| `mi355x_torchtitan_llama-3.1-405b_distributed.json` | `mi355x_torchtitan_llama-3.1-405b_distributed_threshold.json` | Llama-3.1-405B | MI355X, distributed |
| `mi355x_torchtitan_deepseek-v2-lite_single.json` | `mi355x_torchtitan_deepseek-v2-lite_single_threshold.json` | DeepSeek-V2-Lite | MI355X, single-node |
| `mi355x_torchtitan_qwen3-32b_single.json` | `mi355x_torchtitan_qwen3-32b_single_threshold.json` | Qwen3-32B | MI355X, single-node |
| `mi355x_torchtitan_mixtral-8x22b_single.json` | `mi355x_torchtitan_mixtral-8x22b_single_threshold.json` | Mixtral-8x22B | MI355X, single-node |

See [`cvs/input/config_file/training/torchtitan/README.md`](../../input/config_file/training/torchtitan/README.md) for the full variable reference and the values you must change for your cluster and container image.

## Prerequisites

- Passwordless SSH from the control host to each cluster node (key in the cluster file) and Docker available on the nodes.
- A container image bundling TorchTitan for ROCm (`container.image` in the config); TorchTitan must be present at `config.torchtitan_root` (default `/workspace/torchtitan`).
- A Hugging Face token file at `config.hf_token_file` (used to fetch the tokenizer and model). Tokenizer download requires network access on the nodes. For gated models (LLaMA, DeepSeek), model access must be granted on huggingface.co.
- For distributed runs: RDMA interfaces configured and reachable on all nodes; a shared filesystem path reachable from all nodes for logs, scripts, and checkpoints.

## Primus Support

TorchTitan includes factory dispatch for Primus-wrapped execution. When the container image name contains `primus` (case-insensitive), tests automatically use `PrimusTorchTitanTrainingJob` instead of the standard `TorchTitanTrainingJob`. The Primus integration is a placeholder for future development and currently delegates to the base TorchTitan execution.
