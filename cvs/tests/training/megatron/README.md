# Megatron Training Suite (single-node and distributed)

Cluster validation suite that runs Megatron-LM pre-training on AMD Instinct GPUs (single-node or multi-node) and gates the run on performance and correctness metrics with a PASS/FAIL HTML report.

## Overview

The suite drives a Megatron-LM training job inside a Docker container on one or more cluster nodes, then parses the training log to produce metrics and verdicts. It provides:

- **Two suites** — `megatron_single` (single-node) and `megatron_distributed` (multi-node, adds RDMA/NIC setup).
- **Parameter sweeps** — one full training run per enabled combo (e.g. FP8 and BF16), each with its own result rows in the report.
- **Loss curve** — a per-combo decreasing-trend check on `lm_loss` at steps 100 / 500 / 1k / 5k.
- **Training-log error scanning** — NCCL, GPU HW faults, OOM, and other signatures fail a run early with a clear reason.
- **HTML report** — per-test rows with linked logs and a consolidated metric results page.

The mode (single vs distributed) is determined by the config file's `framework` field (`megatron_single` or `megatron_distributed`).

## Quick Start

### Single-node

```bash
cvs run megatron_single \
  --cluster_file input/cluster_file/cluster.json \
  --config_file input/config_file/training/megatron/mi325x_megatron_llama-3.1-8b_single.json \
  --html ./logs/megatron_single.html --self-contained-html -vvv -s
```

### Distributed (multi-node)

```bash
cvs run megatron_distributed \
  --cluster_file input/cluster_file/cluster.json \
  --config_file input/config_file/training/megatron/mi325x_megatron_llama-3.3-70b_distributed.json \
  --html ./logs/megatron_distributed.html --self-contained-html -vvv -s
```

- `--cluster_file` — JSON describing the node(s); see `cvs/input/cluster_file/README.md`.
- `--config_file` — one of the config files in `cvs/input/config_file/training/megatron/`; see that folder's README for the full variable reference.
- `--html` / `--self-contained-html` — write the HTML report.

Use a single-node config with `megatron_single` and a distributed config with `megatron_distributed`. The config's `framework` field must match the suite.

### Run a specific stage

```bash
cvs run megatron_single test_smoke \
  --cluster_file input/cluster_file/cluster.json \
  --config_file input/config_file/training/megatron/mi325x_megatron_llama-3.1-8b_single.json
```

## The Two Suites

| Suite (`cvs run <name>`) | File | Distributed stages | Use with |
|---|---|---|---|
| `megatron_single` | `megatron_single.py` | none | single-node config (`framework: megatron_single`) |
| `megatron_distributed` | `megatron_distributed.py` | `test_setup_rdma` | multi-node config (`framework: megatron_distributed`) |

Both suites share fixtures and hooks from `conftest.py`.

## Test Lifecycle

Tests run in this pinned order. `[combo]` = one row per enabled sweep combo.

| Order | Test | Runs on | Purpose |
|---|---|---|---|
| 0 | `test_launch_container` | once | Launch and verify the container |
| 1 | `test_download_tokenizer` | once | Download HF tokenizer for models that require a local file (DeepSeek) |
| 2 | `test_smoke` | once | Fixed small run confirming the model loads and trains without error |
| 3 | `test_checkpoint` | once | Checkpoint save + resume with loss continuity check. Skipped when `checkpoint.enforce: false`. |
| 4 | `test_training[combo]` | per combo | Build cmd, train, poll logs, parse results; GPU memory freed between combos |
| 5 | `test_metric[combo]` | per combo | Threshold PASS/FAIL per metric |
| 6 | `test_loss_curve[combo]` | per combo | Gate on downward `lm_loss` trend at steps 100 / 500 / 1k / 5k |
| 7 | `test_teardown` | once | Tear the container down |

A training failure is isolated to that combo's `test_training` row; other combos still run. When a combo's training does not complete, its downstream `test_metric` and `test_loss_curve` rows are skipped. If an early lifecycle stage fails, all subsequent stages are skipped via `lifecycle.failed`.

On a training failure, lingering GPU processes are killed (`stop_training_processes`) so the next combo does not launch on top of them.

## Sweeps

A sweep combo is one full training run declared in `sweep.combinations`. `sweep.runs` is the ordered list of combo IDs to execute; set it to a subset to run only selected combos without editing `combinations`.

The combo ID (e.g. `llama3_1_8b-mi325x-bs128-mbs4-fp8`) appears in every parametrized row: `test_training[llama3_1_8b-mi325x-bs128-mbs4-fp8]`, `test_metric[llama3_1_8b-mi325x-bs128-mbs4-fp8]`, and `test_loss_curve[llama3_1_8b-mi325x-bs128-mbs4-fp8]`.

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
| `training.tokens_per_gpu` | Tokens per GPU per second |
| `training.elapsed_time_per_iteration` | Wall time per training step (ms) |
| `training.mem_usage` | GPU memory usage |
| `training.scaling_efficiency_pct` | Multi-node scaling efficiency % vs single-node baseline (distributed only) |

Gating requires `enforce_thresholds: true` in the config. Set to `false` for record-only runs.

## Scaling Efficiency (distributed only)

`test_training` computes scaling efficiency as:

```
efficiency % = (actual_total_tok/s / (actual_nodes / baseline_nodes)) / baseline_total_tok/s × 100
```

Populate `scaling_baseline.tokens_per_sec_total` in the config from a completed single-node run (`tok/s/GPU × 8`). Set to `0.0` to disable and collect data only.

## Loss Curve

`test_loss_curve[combo]` fits a least-squares line to `lm_loss` samples collected at `loss_curve.milestone_steps` (plus every `loss_curve.sample_every` steps) and passes when the slope is below `loss_curve.max_slope`. A value of `0.0` means any downward trend passes.

Set `loss_curve.enforce: false` to record the slope without gating.

## Convergence

`test_metric[combo]` also reports a convergence check when `convergence.target_value > 0`. It compares the final training loss (or eval loss when eval runs) against `convergence.target_value`. Set `target_value <= 0` to disable (record-only). `target_metric: "auto"` selects eval loss when available, otherwise training loss.

## Checkpoint Save and Resume

When `checkpoint.enforce` is `true`, the suite runs the training job twice:

1. **Save phase** — trains to `checkpoint.save_iters` steps, saving a checkpoint every `checkpoint.save_interval` steps to `checkpoint.checkpoint_dir`.
2. **Resume phase** — resumes from the saved checkpoint and trains to `checkpoint.resume_iters` steps.

The suite passes when the loss at `resume_iters` matches the save-phase loss within `checkpoint.loss_rtol` (relative tolerance). Set `enforce: false` to record checkpoint behavior without gating.

## Training-Log Error Detection

During polling, each node's `training.log` is scanned for known error patterns. Defaults cover:

- NCCL errors and timeouts
- GPU hardware faults and hangs
- PyTorch distributed errors

A match fails that combo's `test_training` with the matched pattern name and the last lines of the log.

## Reports and Logs

- **Results table** — one row per test; metric rows show PASS/FAIL from the threshold check.
- **Full log** — each test row links to its own captured log.
- **Training logs** — written inside the container at `<log_dir>/megatron-logs/<combo_id>/out-node<N>/training.log`.

Log path fields:

| Placeholder | Source |
|---|---|
| `<log_dir>` | `config.log_dir` in the config file |
| `<combo_id>` | Sweep run ID (e.g. `llama3_1_8b-mi325x-bs128-mbs4-fp8`) |
| `out-node<N>` | One directory per node; `out-node0` for single-node |

## Config and Threshold Files

Located in `cvs/input/config_file/training/megatron/`:

**MI300X**

| Config | Threshold | Mode |
|---|---|---|
| `mi300x_megatron_deepseek-v2-lite_single.json` | `mi300x_megatron_deepseek-v2-lite_single_threshold.json` | single-node |
| `mi300x_megatron_deepseek-v2-lite_distributed.json` | `mi300x_megatron_deepseek-v2-lite_distributed_threshold.json` | distributed |
| `mi300x_megatron_llama-3.1-8b_single.json` | `mi300x_megatron_llama-3.1-8b_single_threshold.json` | single-node |
| `mi300x_megatron_llama-3.1-8b_distributed.json` | `mi300x_megatron_llama-3.1-8b_distributed_threshold.json` | distributed |
| `mi300x_megatron_llama-3.1-405b_distributed.json` | `mi300x_megatron_llama-3.1-405b_distributed_threshold.json` | distributed |
| `mi300x_megatron_llama-3.3-70b_single.json` | `mi300x_megatron_llama-3.3-70b_single_threshold.json` | single-node |
| `mi300x_megatron_llama-3.3-70b_distributed.json` | `mi300x_megatron_llama-3.3-70b_distributed_threshold.json` | distributed |

**MI325X**

| Config | Threshold | Mode |
|---|---|---|
| `mi325x_megatron_deepseek-v2-lite_single.json` | `mi325x_megatron_deepseek-v2-lite_single_threshold.json` | single-node |
| `mi325x_megatron_deepseek-v2-lite_distributed.json` | `mi325x_megatron_deepseek-v2-lite_distributed_threshold.json` | distributed |
| `mi325x_megatron_llama-3.1-8b_single.json` | `mi325x_megatron_llama-3.1-8b_single_threshold.json` | single-node |
| `mi325x_megatron_llama-3.1-8b_distributed.json` | `mi325x_megatron_llama-3.1-8b_distributed_threshold.json` | distributed |
| `mi325x_megatron_llama-3.1-405b_distributed.json` | `mi325x_megatron_llama-3.1-405b_distributed_threshold.json` | distributed |
| `mi325x_megatron_llama-3.3-70b_single.json` | `mi325x_megatron_llama-3.3-70b_single_threshold.json` | single-node |
| `mi325x_megatron_llama-3.3-70b_distributed.json` | `mi325x_megatron_llama-3.3-70b_distributed_threshold.json` | distributed |

**MI355X**

| Config | Threshold | Mode |
|---|---|---|
| `mi355x_megatron_deepseek-v2-lite_single.json` | `mi355x_megatron_deepseek-v2-lite_single_threshold.json` | single-node |
| `mi355x_megatron_deepseek-v2-lite_distributed.json` | `mi355x_megatron_deepseek-v2-lite_distributed_threshold.json` | distributed |
| `mi355x_megatron_llama-3.1-8b_single.json` | `mi355x_megatron_llama-3.1-8b_single_threshold.json` | single-node |
| `mi355x_megatron_llama-3.1-8b_distributed.json` | `mi355x_megatron_llama-3.1-8b_distributed_threshold.json` | distributed |
| `mi355x_megatron_llama-3.1-405b_distributed.json` | `mi355x_megatron_llama-3.1-405b_distributed_threshold.json` | distributed |
| `mi355x_megatron_llama-3.3-70b_single.json` | `mi355x_megatron_llama-3.3-70b_single_threshold.json` | single-node |
| `mi355x_megatron_llama-3.3-70b_distributed.json` | `mi355x_megatron_llama-3.3-70b_distributed_threshold.json` | distributed |

See [`cvs/input/config_file/training/megatron/README.md`](../../input/config_file/training/megatron/README.md) for the full variable reference and the values you must change for your cluster and container image.

## Prerequisites

- Passwordless SSH from the control host to each cluster node (key in the cluster file) and Docker available on the nodes.
- A container image bundling Megatron-LM for ROCm (`container.image` in the config); Megatron-LM must be present at `config.megatron_root` (default `/workspace/Megatron-LM`).
- A Hugging Face token file at `config.hf_token_file` (used to fetch the tokenizer). Tokenizer download requires network access on the nodes. For gated models (LLaMA, DeepSeek), model access must be granted on huggingface.co.
- For distributed runs: RDMA interfaces configured and reachable on all nodes; a shared filesystem path reachable from all nodes for logs and scripts.
