# TorchTitan Training Suite (single-node and distributed)

Cluster validation suite that runs TorchTitan pre-training on AMD Instinct GPUs (single-node or multi-node) and gates the run on performance and correctness metrics with a PASS/FAIL HTML report.

## Overview

The suite drives a TorchTitan training job inside a Docker container on one or more cluster nodes, then parses the training log to produce metrics and verdicts. It provides:

- **Two suites** — `torchtitan_single` (single-node) and `torchtitan_distributed` (multi-node, adds RDMA/NIC setup).
- **Parameter sweeps** — one full training run per enabled combo (e.g. FP8 and BF16), each with its own result rows in the report.
- **Loss curve** — a per-combo decreasing-trend check on `lm_loss` at steps 100 / 500 / 1k / 5k.
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
| 5 | `test_metric[combo]` | per combo | Threshold PASS/FAIL per metric |
| 6 | `test_loss_curve[combo]` | per combo | Gate on downward `lm_loss` trend at steps 100 / 500 / 1k / 5k |
| 7 | `test_teardown` | once | Tear the container down |

A training failure is isolated to that combo's `test_training` row; other combos still run. When a combo's training does not complete, its downstream `test_metric` and `test_loss_curve` rows are skipped. If an early lifecycle stage fails, all subsequent stages are skipped via `lifecycle.failed`.

On a training failure, lingering GPU processes are killed (`stop_training_processes`) so the next combo does not launch on top of them.

## Sweeps

A sweep combo is one full training run declared in `sweep.combinations`. `sweep.runs` is the ordered list of combo IDs to execute; set it to a subset to run only selected combos without editing `combinations`.

The combo ID (e.g. `llama3_1_8b-mi355-bs128-mbs4-fp8`) appears in every parametrized row: `test_training[llama3_1_8b-mi355-bs128-mbs4-fp8]`, `test_metric[llama3_1_8b-mi355-bs128-mbs4-fp8]`, and `test_loss_curve[llama3_1_8b-mi355-bs128-mbs4-fp8]`.

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
| `training.tokens_per_sec` | Tokens per second per GPU |
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

## Training-Log Error Detection

During polling, each node's `training.log` is scanned for known error patterns. Defaults cover:

- NCCL errors and timeouts
- GPU hardware faults and hangs
- PyTorch distributed errors

A match fails that combo's `test_training` with the matched pattern name and the last lines of the log.

## Reports and Logs

- **Results table** — one row per test; metric rows show PASS/FAIL from the threshold check.
- **Full log** — each test row links to its own captured log.
- **Training logs** — written inside the container at `<log_dir>/torchtitan-logs/<combo_id>/out-node<N>/training.log`.

Log path fields:

| Placeholder | Source |
|---|---|
| `<log_dir>` | `config.log_dir` in the config file |
| `<combo_id>` | Sweep run ID (e.g. `llama3_1_8b-mi355-bs128-mbs4-fp8`) |
| `out-node<N>` | One directory per node; `out-node0` for single-node |

## Config and Threshold Files

Located in `cvs/input/config_file/training/torchtitan/`:

| Config | Threshold | Arch / mode |
|---|---|---|
| `mi355x_torchtitan_llama-3.1-8b_single.json` | `mi355x_torchtitan_llama-3.1-8b_single_threshold.json` | MI355X, single-node |
| `mi355x_torchtitan_llama-3.1-70b_single.json` | `mi355x_torchtitan_llama-3.1-70b_single_threshold.json` | MI355X, single-node |
| `mi355x_torchtitan_llama-3.3-70b_single.json` | `mi355x_torchtitan_llama-3.3-70b_single_threshold.json` | MI355X, single-node |
| `mi355x_torchtitan_llama-3.3-70b_distributed.json` | `mi355x_torchtitan_llama-3.3-70b_distributed_threshold.json` | MI355X, distributed |
| `mi355x_torchtitan_llama-3.1-405b_distributed.json` | `mi355x_torchtitan_llama-3.1-405b_distributed_threshold.json` | MI355X, distributed |
| `mi355x_torchtitan_deepseek-v2-lite_single.json` | `mi355x_torchtitan_deepseek-v2-lite_single_threshold.json` | MI355X, single-node |
| `mi355x_torchtitan_qwen3-32b_single.json` | `mi355x_torchtitan_qwen3-32b_single_threshold.json` | MI355X, single-node |
| `mi355x_torchtitan_mixtral-8x22b_single.json` | `mi355x_torchtitan_mixtral-8x22b_single_threshold.json` | MI355X, single-node |

See [`cvs/input/config_file/training/torchtitan/README.md`](../../../input/config_file/training/torchtitan/README.md) for the full variable reference and the values you must change for your cluster and container image.

## Prerequisites

- Passwordless SSH from the control host to each cluster node (key in the cluster file) and Docker available on the nodes.
- A container image bundling TorchTitan for ROCm (`container.image` in the config); TorchTitan must be present at `config.torchtitan_root` (default `/workspace/torchtitan`).
- A Hugging Face token file at `config.hf_token_file` (used to fetch the tokenizer). Tokenizer download requires network access on the nodes. For gated models (LLaMA, DeepSeek), model access must be granted on huggingface.co.
- For distributed runs: RDMA interfaces configured and reachable on all nodes; a shared filesystem path reachable from all nodes for logs and scripts.
