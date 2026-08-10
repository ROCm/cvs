# JAX MaxText Training Suite (single-node and distributed)

Cluster validation suite that runs **JAX MaxText** pre-training on AMD Instinct
GPUs (single-node or multi-node) and gates the run on performance and
correctness metrics with a PASS/FAIL HTML report.

## Overview

The suite drives a MaxText training job inside a container on one or more
cluster nodes, then parses the training log to produce metrics and verdicts. It
provides:

1. **Two suites** - `jaxmaxtext_single` (single node) and
   `jaxmaxtext_distributed` (multi-node, adds RDMA/NIC setup).
2. **Parameter sweeps** - one full training run per enabled sweep (e.g. BF16 and
   FP8), each with its own result rows in the report.
3. **Metric gating** - per-sweep, per-metric PASS/FAIL against a threshold file
   (throughput, TFLOP/s, step-time, loss, scaling efficiency, convergence, ...).
4. **Loss curve** - a per-sweep training-loss PNG plus a decreasing-trend check.
5. **Training-log error scanning** - configurable regex signatures (NCCL, GPU HW,
   OOM, segfault, ...) fail a run early with a clear reason.
6. **HTML report + console summary** - per-test rows, a consolidated metric
   results page, per-sweep loss curves, and an aggregated failure summary.

The mode (single vs distributed) is reflected in the suite file name, the metric
results HTML title, and the loss-curve titles/artifacts.

## Quick Start

Single-node run:

```bash
cvs run jaxmaxtext_single \
  --cluster_file ./p3_1n_cluster.json \
  --config_file cvs/input/config_file/training/jaxmaxtext/mi300x_jaxmaxtext_llama-3.3-70b_single.json \
  --html ./logs/jaxmaxtext_single.html --self-contained-html -vvv
```

Distributed (multi-node) run:

```bash
cvs run jaxmaxtext_distributed \
  --cluster_file ./p3_2n_cluster.json \
  --config_file cvs/input/config_file/training/jaxmaxtext/mi325x_jaxmaxtext_llama-3.3-70b_distributed.json \
  --html ./logs/jaxmaxtext_distributed.html --self-contained-html -vvv
```

- `--cluster_file` - JSON describing the node(s); the first node in `node_dict`
  is used as the JAX coordinator when `jax_distributed.coordinator_ip` is `auto`.
- `--config_file` - one of the config files in
  `cvs/input/config_file/training/jaxmaxtext/` (see that folder's README for the
  variable-by-variable reference and what to change for your cluster).
- `--html` / `--self-contained-html` - write the report; a `<name>_html/` bundle
  dir alongside it holds per-test logs, the metric results page, and loss-curve
  PNGs.

> Use a **single-node** config with `jaxmaxtext_single` and a **distributed**
> config with `jaxmaxtext_distributed`. The config's `training.distributed` flag
> must match the suite.

## The two suites

| Suite (`cvs run <name>`) | File | Distributed stages | Use with |
|---|---|---|---|
| `jaxmaxtext_single` | `jaxmaxtext_single.py` | none | single-node config (`distributed: false`) |
| `jaxmaxtext_distributed` | `jaxmaxtext_distributed.py` | `test_setup_rdma`, `test_setup_nic` | multi-node config (`distributed: true`) |

Both suites share their implementations from `_common.py`; sweep parametrization
and all fixtures/hooks live in `conftest.py`. `_common.py` and `conftest.py` are
helpers, not runnable suites.

## Test lifecycle (report rows)

Tests run in this pinned order. `[sweep]` = one row per enabled sweep;
`[sweep-metric]` = one row per metric per sweep.

| Order | Test | Runs on | Purpose |
|---|---|---|---|
| 1 | `test_launch_container` | once | Launch and verify the container |
| 2 | `test_setup_rdma` | distributed only | Copy RDMA lib into container (thor2 NIC) |
| 3 | `test_setup_nic` | distributed only | NIC setup scripts |
| 4 | `test_setup_tokenizer` | once | Download the HF tokenizer |
| 5 | `test_training_run[sweep]` | per sweep | Build cmd, train, poll, parse results |
| 6 | `test_metric[sweep-metric]` | per sweep x metric | Threshold PASS/FAIL per metric |
| 7 | `test_loss_curve[sweep]` | per sweep | Render loss PNG; gate on downward trend |
| 8 | `test_print_results_table` | once | Console tables + metric results HTML + failure summary |
| 9 | `test_teardown` | once | Tear the container down |

A training failure is isolated to that sweep's `test_training_run` row; other
sweeps still run. When a sweep's training does not complete, its downstream
`test_metric`/`test_loss_curve` rows are skipped.

## Sweeps

A **sweep** is one full training run with per-run MaxText overrides. Sweeps are
declared in the config under `training.sweeps` and selected with
`training.enabled_sweep_list`. The sweep `name` is also the **threshold cell
key**. For now precision is the swept dimension (BF16, FP8).

Each sweep gets a compact, unique **label** derived from its name -
`PRECISION[-SL<seqlen>][-B<batch>]`, e.g. `BF16-SL8192-B3`. The label appears in
every parametrized row: `test_training_run[BF16-SL8192-B3]`,
`test_metric[BF16-SL8192-B3-tflops_per_sec_per_gpu]`,
`test_loss_curve[BF16-SL8192-B3]`, and in the metric results/loss-curve reports.

## Metrics and PASS/FAIL

Each `test_metric[sweep-metric]` compares the parsed metric against its threshold
spec in the sweep's cell of the threshold file and reports one of:

| Status | Meaning |
|---|---|
| PASS | value satisfies the threshold |
| FAIL | value violates the threshold (row is red; also aggregated in the summary) |
| N/A | metric was not produced this run (feature disabled / rampup) - not a failure |
| RECORD | no threshold, or `enforce_thresholds` is false - value logged, not gated |

Metrics surfaced (namespace `training.*`): `tflops_per_sec_per_gpu`,
`tokens_per_sec_per_gpu`, `tokens_per_sec_total`, `scaling_efficiency_pct`,
`step_time_seconds`, `step_time_mean_ms`, `step_time_p50_ms`, `step_time_p95_ms`,
`final_loss`, `loss_decreased`, `eval_loss`, `steps_to_target`,
`time_to_target_seconds`.

Gating is threshold-driven and requires `enforce_thresholds: true` in the config.
A threshold entry with `"kind": "info"` always passes (record-only). See the
input-config README for threshold kinds and defaults.

## Reports and logs

- **Results table** - one row per test; metric rows show PASS/FAIL from the
  threshold check.
- **Full Log** - each test row links to its own captured log.
- **Metric Results** - every `test_metric` row also links to a single shared
  `metric_results.html` (Sweep | Metric | Expected | Actual | Unit | Status),
  titled with the mode (single/distributed).
- **Loss Curve** - each `test_loss_curve` row links to a per-sweep PNG.
- **Console summary** - `test_print_results_table` prints per-sweep tables and,
  via `globals.error_list` + `update_test_result()`, an aggregated list of all
  failed `(sweep, metric)` checks in the pytest final summary.

## Training-log error detection

During polling, each node's `training.log` is scanned for the regexes in
`training.error_patterns` (config-driven; falls back to built-in defaults).
Defaults cover NCCL, GPU HW faults, assertion/JAX stack traces, ROCm init
errors, Python fatal errors, TF coordination errors, `RESOURCE_EXHAUSTED`/OOM,
and segfault signatures. A match fails that sweep's `test_training_run` with the
matched signature name and the last part of the log. Add/remove signatures in
the config as you encounter new ones.

## Config and threshold files

Located in `cvs/input/config_file/training/jaxmaxtext/` (each config has a
sibling threshold file named by its `threshold_json` field):

| Config | Threshold | Arch / mode |
|---|---|---|
| `mi300x_jaxmaxtext_llama-3.3-70b_single.json` | `..._single_threshold.json` | MI300X, single-node |
| `mi300x_jaxmaxtext_llama-3.3-70b_distributed.json` | `..._distributed_threshold.json` | MI300X, distributed |
| `mi325x_jaxmaxtext_llama-3.3-70b_distributed.json` | `..._distributed_threshold.json` | MI325X, distributed |
| `mi355x_jaxmaxtext_llama-3.3-70b_distributed.json` | `..._distributed_threshold.json` | MI355X, distributed |

See `cvs/input/config_file/training/jaxmaxtext/README.md` for the full variable
reference and the values you must change for your cluster and container image.

## Prerequisites

- Passwordless SSH from the control host to each cluster node (key in the
  cluster file), and Docker available on the nodes.
- A container image bundling MaxText/JAX for ROCm (config `container.image`).
- A Hugging Face token file at `paths.hf_token_file` (used to fetch the
  tokenizer). The tokenizer download requires network access on the nodes.
- A shared filesystem path (`paths.shared_fs`) reachable from all nodes for
  distributed runs (models cache and logs).
