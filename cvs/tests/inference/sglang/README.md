# SGLang Inference Suite (single-node, distributed, and disaggregated)

Cluster validation suite that runs **SGLang** serving benchmarks on AMD Instinct
GPUs and gates each run on performance and lm-eval accuracy metrics with a
PASS/FAIL HTML report.

## Overview

The suite drives an SGLang server (unified or PD-disaggregated) inside a
container on one or more cluster nodes, then runs smoke checks, lm-eval accuracy
tasks, and randomized benchmark-serving workloads. It provides:

1. **Three suites** — `sglang_single` (one unified server), `sglang_distributed`
   (multi-node unified TP/PP), and `sglang_disagg_distributed` (prefill/decode
   PD roles with a proxy router).
2. **Benchmark variants** — model/TP/PP settings live under `benchmark_params`;
   the active variant is selected by `active_benchmark`, env
   `SGLANG_BENCHMARK_KEY`, or auto when only one key exists.
3. **Performance sweeps** — one benchmark run per threshold cell
   (ISL/OSL/TP/PP/concurrency), each with its own pytest row and metric subtests.
4. **Accuracy gating** — lm-eval HellaSwag and GSM8K compared against threshold
   cells `BENCH=lm_eval_hellaswag` and `BENCH=lm_eval_gsm8k`.
5. **OpenAI-compatible smoke tests** — HTTP endpoint checks before benchmarks.
6. **dmesg scanning** — time-bounded kernel log scan after the performance phase.
7. **HTML report + Run Deck** — pytest HTML rows with collapsible per-metric
   subtests, console results tables, and (when `--html` is set) an SGLang Run
   Deck bundle for interactive charts.

The mode (single vs distributed vs disaggregated) is determined by which suite
you run and the node-role fields in the config (`benchmark_serv_node`,
`server_node_list`, or prefill/decode/router lists).

## Quick Start

Single-node run:

```bash
cvs run sglang_single \
  --cluster_file cvs/input/cluster_file/cluster_container.json \
  --config_file cvs/input/config_file/inference/sglang/mi30x_sglang_llama_70b_single.json \
  --html ./logs/sglang_single.html --self-contained-html -vvv
```

Distributed (unified multi-node TP/PP) run:

```bash
cvs run sglang_distributed \
  --cluster_file cvs/input/cluster_file/cluster_container.json \
  --config_file cvs/input/config_file/inference/sglang/mi30x_sglang_llama_70b_distributed.json \
  --html ./logs/sglang_distributed.html --self-contained-html -vvv
```

Disaggregated (PD prefill/decode + router) run:

```bash
cvs run sglang_disagg_distributed \
  --cluster_file cvs/input/cluster_file/cluster_container.json \
  --config_file cvs/input/config_file/inference/sglang/mi30x_sglang_llama_70b_disaggregated.json \
  --html ./logs/sglang_disagg.html --self-contained-html -vvv
```

- `--cluster_file` — JSON describing the node(s); use
  `cluster_container.json` (or equivalent) with `"orchestrator": "container"`.
  The suite scopes SSH/container access to the nodes required by the config
  (benchmark node only for single-node; server+bench hosts for distributed; the
  union of prefill/decode/router/bench hosts for disagg).
- `--config_file` — one of the configs in
  `cvs/input/config_file/inference/sglang/` (replace every `<changeme>` placeholder
  before running).
- `--html` / `--self-contained-html` — write the pytest report; a sibling bundle
  directory holds per-test logs and the SGLang Run Deck artifacts.

> Use **`sglang_single`** with `nnodes: "1"` and a single `benchmark_serv_node`.
> Use **`sglang_distributed`** with `server_node_list` and matching `nnodes` for
> unified multi-node serving. Use **`sglang_disagg_distributed`** with
> `prefill_node_list`, `decode_node_list`, and `proxy_router_node`.

For smoke runs, filter performance cells with `-k`, for example
`-k "isl1024-osl1024-c64"`.

## The three suites

| Suite (`cvs run <name>`) | File | Topology | Use with |
|---|---|---|---|
| `sglang_single` | `sglang_single.py` | one unified server on `benchmark_serv_node` | single-node config (`nnodes: "1"`) |
| `sglang_distributed` | `sglang_distributed.py` | unified sharded server across `server_node_list` | multi-node TP/PP config |
| `sglang_disagg_distributed` | `sglang_disagg_distributed.py` | separate prefill, decode, and proxy router roles | PD disaggregated config |

All three suites share fixtures and hooks from `conftest.py` and results-table
helpers from `_shared.py`. Those files are helpers, not runnable suites.

## Test lifecycle (report rows)

Tests run in a pinned order per suite. `[cell]` = one row per performance
threshold cell (parametrize IDs like `isl1024-osl1024-c64`).

### `sglang_single`

| Order | Test | Runs on | Purpose |
|---|---|---|---|
| 1 | `test_launch_container` | once | Launch and verify the container |
| 2 | `test_rms_norm` | once | RMS-norm sanity check |
| 3 | `test_launch_server` | once | Start unified `sglang.launch_server` |
| 4 | `test_poll_for_server_ready` | once | Wait until the server is healthy |
| 5 | `test_openai_compatible_http_endpoints` | once | OpenAI-compatible HTTP smoke checks |
| 6 | `test_run_lm_eval_hellaswag_benchmark_test` | once | lm-eval HellaSwag accuracy |
| 7 | `test_run_lm_eval_gsm8k_benchmark_test` | once | lm-eval GSM8K accuracy |
| 8 | `test_run_performance_benchmark_test[cell]` | per cell | Randomized bench-serving + metric gates |
| 9 | `test_verify_dmesg_after_benchmark` | once | Time-bounded dmesg error scan |
| 10 | `test_print_results_table` | once | Console tables (smoke, accuracy, perf) |
| 11 | `test_teardown` | once | Tear the container down |

### `sglang_distributed`

Same as single-node through step 8, then:

| Order | Test | Purpose |
|---|---|---|
| 9 | `test_distributed_gpu_topology` | Verify GPU counts across server ranks |
| 10 | `test_print_results_table` | Console summary |
| 11 | `test_teardown` | Tear down containers |

### `sglang_disagg_distributed`

| Order | Test | Purpose |
|---|---|---|
| 1 | `test_launch_container` | Launch containers on role hosts |
| 2 | `test_rms_norm` | RMS-norm sanity check |
| 3 | `test_launch_prefill_servers` | Start prefill ranks |
| 4 | `test_launch_decode_servers` | Start decode ranks |
| 5 | `test_poll_for_server_ready` | Wait for PD servers |
| 6 | `test_launch_proxy_router` | Start the proxy router |
| 7 | `test_openai_compatible_http_endpoints` | HTTP smoke checks via router |
| 8 | `test_run_lm_eval_hellaswag_benchmark_test` | lm-eval HellaSwag |
| 9 | `test_run_lm_eval_gsm8k_benchmark_test` | lm-eval GSM8K |
| 10 | `test_run_performance_benchmark_test[cell]` | Per-cell bench-serving + gates |
| 11 | `test_verify_dmesg_after_benchmark` | dmesg scan |
| 12 | `test_disagg_gpu_topology` | Verify GPU counts on prefill/decode nodes |
| 13 | `test_print_results_table` | Console summary |
| 14 | `test_teardown` | Tear down containers |

On container or server failure, `lifecycle.failed` is set and the orchestrator
leak-guard tears down containers at module exit if `test_teardown` did not run.

## Performance sweeps

A **performance cell** is one `(ISL, OSL, TP, PP, concurrency)` pair declared
in the threshold file with a key like:

`ISL=1024,OSL=1024,TP=8,PP=1,CONC=64`

Cells are collected from the sibling threshold JSON referenced by
`benchmark_params.<variant>.threshold_file` and drive parametrization of
`test_run_performance_benchmark_test`. Each cell runs `bench_serv_random` with
that shape and concurrency, then gates every metric in the cell spec via pytest
subtests.

Accuracy thresholds use separate cells:

- `BENCH=lm_eval_hellaswag` — HellaSwag metric gate
- `BENCH=lm_eval_gsm8k` — GSM8K metric gate

Long-context NIAH cells (`ACC_ISL=...`) may appear in threshold files for
future use; they are not collected by the current suite parametrization.

## Metrics and PASS/FAIL

Each `test_run_performance_benchmark_test[cell]` evaluates metrics per node as
pytest subtests. Latency metrics (`*_ms`) must be **at or below** the threshold;
throughput and ratio metrics must be **at or above** it.

| Status | Meaning |
|---|---|
| PASS | value satisfies the threshold |
| FAIL | value violates the threshold (subtest fails; cell marked FAIL) |
| — | metric not present in the benchmark artifact (logged, not gated) |

**Performance metrics** (bare keys, not a namespace prefix):

`request_throughput_per_sec`, `output_throughput_per_sec`,
`output_throughput_per_gpu_per_sec`, `mean_ttft_ms`, `mean_tpot_ms`,
`p99_itl_ms`, `mean_e2e_latency_ms`, `goodput`, `mfu`.

**Accuracy metrics** come from lm-eval task output keys in the threshold file
(for example `acc_norm,none` for HellaSwag, `exact_match,flexible-extract` for
GSM8K).

Threshold specs support structured entries with `"kind"` (`min_tok_s`, `max_ms`,
`min`, ...) and a `"value"` field; bare numeric values are also accepted.

## Reports and logs

- **Results table** — one row per lifecycle test; performance rows are
  collapsible and show per-metric PASS/FAIL subtests.
- **Full Log** — each test row links to its captured log when HTML reporting is
  enabled.
- **Lifecycle stage table** — non-benchmark rows embed a stage/value/unit table
  (container launch, server ready, bench duration, ...).
- **Console summary** — `test_print_results_table` prints smoke, lm-eval accuracy,
  per-cell performance summary, and detailed metric tables.
- **Run Deck** — when `--html` is set, the inference report engine emits a suite-
  specific Run Deck at session end (`sglang_single_run_deck.html`,
  `sglang_distributed_run_deck.html`, or `sglang_disagg_run_deck.html`, plus JSON
  and an interactive viewer). Run Decks are render-only and do not affect gates.

Server and benchmark logs are written under the variant `paths.log_dir` on the
cluster nodes.

## Config and threshold files

Located in `cvs/input/config_file/inference/sglang/`. Each config references a
threshold file via `benchmark_params.<variant>.threshold_file` (often shared
across single/distributed/disaggregated stems for the same model):

| Config | Mode | Example model |
|---|---|---|
| `mi30x_sglang_llama_70b_single.json` | single-node | Llama 3.1 70B |
| `mi30x_sglang_llama_70b_distributed.json` | unified multi-node | Llama 3.1 70B |
| `mi30x_sglang_llama_70b_disaggregated.json` | PD disaggregated | Llama 3.1 70B |
| `mi30x_sglang_deepseek_r1_0528_single.json` | single-node | DeepSeek R1 |
| `mi30x_sglang_deepseek_r1_0528_distributed.json` | unified multi-node | DeepSeek R1 |
| `mi30x_sglang_deepseek_r1_0528_disaggregated.json` | PD disaggregated | DeepSeek R1 |
| `mi325x_sglang_kimi-k27_distributed.json` | unified multi-node | Kimi K2.7 |
| `mi35x_sglang_distributed.json` | unified multi-node | (MI35X template) |

Replace `<changeme>` placeholders (node IPs, NCCL interfaces, coordinator
addresses, container image pins) before running on your cluster.

## Prerequisites

- Passwordless SSH from the control host to each cluster node (key in the
  cluster file), and Docker available on the GPU nodes.
- A container image with SGLang for ROCm (config `config.container_image` or
  unified-schema `container.image`).
- A Hugging Face token file at `config.hf_token_file` when fetching remote
  models.
- Model weights reachable from server nodes (local path or HF download).
- For distributed/disaggregated runs: RDMA/NCCL settings (`nccl_ib_hca`,
  `nccl_socket_ifname`, `gloo_socket_ifname`, ...) matching your fabric; a shared
  or reachable log path (`config.log_dir`); and all role hosts listed in both
  the cluster file and the inference config.

## Related code

| Module | Purpose |
|---|---|
| `cvs/lib/inference/sglang/sglang_single_lib.py` | `SglangSingle` — single-node server lifecycle |
| `cvs/lib/inference/sglang/sglang_distributed_lib.py` | `SglangDistributed` — unified multi-node server |
| `cvs/lib/inference/sglang/sglang_disagg_lib.py` | `SglangDisaggPD` — PD prefill/decode/router |
| `cvs/lib/inference/sglang/sglang_config_loader.py` | Variant load, threshold cells, orchestrator bridge |
| `cvs/lib/inference/sglang/sglang_parsing.py` | Metric vocabulary and report column definitions |
| `cvs/lib/report/presets/sglang_*.py` | Run Deck presets per suite |
