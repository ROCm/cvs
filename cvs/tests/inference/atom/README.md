# ATOM Inference Suite (single-node and multinode)

Cluster validation suite that runs **ATOM** (and ATOM-coordinated vLLM / SGLang)
serving benchmarks on AMD Instinct GPUs and gates each sweep cell on tiered
performance and health metrics with a PASS/FAIL HTML report.

## Overview

The suite drives a serving + benchmark-serving job inside a container on one or
more cluster nodes, then parses the benchmark artifact to produce `client.*`
metrics and verdicts. It provides:

1. **One unified suite** — `atom` handles single-node and multinode PP from the
   same entry point; topology and driver behaviour come from the variant config.
2. **Execution drivers** — `params.driver` selects the server/client stack:
   `atom` (native openai_server), `vllm_atom` (vLLM PP coordinator + ATOM
   kernels), `sglang`, or interim `vllm`.
3. **Parameter sweeps** — one benchmark run per sweep cell (ISL/OSL shape ×
   concurrency), each with its own result rows in the report.
4. **Tiered metric gating** — one pytest row per **metric tier** per cell
   (throughput, TTFT, TPOT, health, scaling, record) against threshold specs.
5. **Server reuse** — optional reuse of a warm server across sweep cells when
   `reuse_server_across_sweep: true` and the session key matches.
6. **Multinode fabric discovery** — `test_discover_topology` resolves IB HCAs
   and socket netdev before the sweep when `params.nnodes > 1`.
7. **HTML report + Run Deck** — pytest HTML rows, console results tables, and
   (when `--html` is set) an ATOM Run Deck bundle for interactive charts.

Single-node vs multinode is determined by `params.nnodes` and the cluster file
host count, not by a separate suite name.

## Quick Start

Single-node W1 run (MI300X, `driver=atom`):

```bash
cvs run atom \
  --cluster_file ~/input/cluster_file/atom_cluster.json \
  --config_file ~/input/config_file/inference/atom/single/mi300x_atom_deepseek-r1_fp8_single.json \
  --html ~/cvs_results/atom-w1-single.html --self-contained-html -vvv
```

Multinode PP run (2-node, `driver=vllm_atom`):

```bash
cvs run atom \
  --cluster_file ~/input/cluster_file/atom_cluster.json \
  --config_file ~/input/config_file/inference/atom/distributed/mi300x_atom_deepseek-r1_fp8_distributed.json \
  --html ~/cvs_results/atom-w1-distributed.html --self-contained-html -vvv
```

- `--cluster_file` — JSON describing the node(s); `len(node_dict)` must match
  `params.nnodes` in the config.
- `--config_file` — a variant JSON under
  `cvs/input/config_file/inference/atom/` (see that folder's README for the
  variable-by-variable reference, copy-config flow, and lab prerequisites).
- `--html` / `--self-contained-html` — write the pytest report; a sibling
  bundle directory holds per-test logs and, when the report engine is enabled,
  the ATOM Run Deck artifacts.

> Use a **single-host** cluster file with `nnodes=1` variants and a **two-host**
> cluster file with multinode PP variants (`vllm_atom` or `sglang`). The config's
> `params.driver` and `params.nnodes` must match the intended topology.

For smoke runs, filter with `-k`, for example `-k "w1_1k_1k-conc128"`.

## Suite layout

| Item | File | Role |
|---|---|---|
| Suite (`cvs run atom`) | `atom.py` | Lifecycle tests, sweep inference, tiered metric gates |
| Fixtures / ordering | `conftest.py` | Cluster + variant load, orchestrator, lifecycle rank |
| Results table | `_shared.py` | Console + HTML results table (`test_print_results_table`) |

`conftest.py` and `_shared.py` are helpers, not runnable suites.

## Test lifecycle (report rows)

Tests run in this pinned order. `[cell]` = one row per sweep cell;
`[cell-tier]` = one row per metric tier per cell.

| Order | Test | Runs on | Purpose |
|---|---|---|---|
| 1 | `test_launch_container` | once | Launch and verify the container |
| 2 | `test_setup_sshd` | multinode | SSH daemon setup across nodes |
| 3 | `test_discover_topology` | once | Resolve IB HCAs + socket netdev (skipped work on single-node) |
| 4 | `test_model_fetch` | once | Verify / fetch the model cache |
| 5 | `test_atom_inference[cell]` | per cell | Build server env, start server, run bench client, parse results |
| 6 | `test_cell_metrics[cell-tier]` | per cell × tier | Threshold PASS/FAIL for that tier's metrics |
| 7 | `test_print_results_table` | once | Console tables + consolidated results |
| 8 | `test_teardown` | once | Tear the container down |

On inference failure, `lifecycle.failed` is set so downstream cells and metric
rows are skipped. Server reuse skips restart when the session key
(`server_session_key`) matches the prior cell.

## Sweeps

A **sweep cell** is one `(sequence shape, concurrency)` pair declared under
`sweep.sequence_combinations` and `sweep.runs`. Parametrize IDs look like
`w1_1k_1k-conc128` or, when metric tiers are collected,
`w1_1k_1k-conc128-throughput`.

Each cell's **threshold key** is built by `cell_key()`, for example:

- Single-node: `ISL=1024,OSL=1024,TP=8,CONC=128`
- Multinode PP: `ISL=1024,OSL=1024,TP=8,PP=2,NNODES=2,CONC=128`

That key must exist in the sibling threshold file referenced by
`threshold_json`.

## Metrics and PASS/FAIL

Each `test_cell_metrics[cell-tier]` evaluates metrics for one tier against the
cell's threshold specs and reports one of:

| Status | Meaning |
|---|---|
| PASS | value satisfies the threshold |
| FAIL | value violates the threshold (row is red) |
| skip | prior stage failed, cell did not run, or tier not applicable (e.g. scaling on single-node) |
| RECORD | `enforce_thresholds: false` or `record` tier — value logged, not gated |

**Metric tiers** (namespace `client.*` unless noted):

| Tier | Example metrics |
|---|---|
| `throughput` | `total_token_throughput`, `output_throughput`, `per_gpu_throughput`, `output_tput_per_gpu` |
| `ttft` | `mean_ttft_ms`, `p99_ttft_ms` |
| `tpot` | `mean_tpot_ms`, `p99_tpot_ms` |
| `health` | `success_rate`, `failed` |
| `scaling` | `scaling.efficiency_pct` (multinode) |
| `record` | remaining client metrics not in a gate tier |

Gating requires `enforce_thresholds: true` in the config. ATOM may omit some
tail percentiles even when `metric_percentiles` requests them; the suite only
gates metrics present in the benchmark artifact. See
`cvs/input/config_file/inference/atom/README.md` for threshold kinds and
variant-specific enforcement policy.

## Reports and logs

- **Results table** — one row per test; metric tier rows reflect threshold
  verdicts.
- **Full Log** — each test row links to its captured log when HTML reporting is
  enabled.
- **Console summary** — `test_print_results_table` prints per-cell tables with
  throughput, latency, and health columns.
- **Run Deck** — when `--html` is set, the inference report engine emits
  `atom_run_deck.html`, `.json`, and `_viewer.html` at session end (bundled
  into the pytest zip). See `cvs/lib/report/README.md`.

Server and client logs for each cell are written under the variant
`paths.log_dir` on the cluster nodes.

## Config and threshold files

Variant configs and thresholds live in `cvs/input/config_file/inference/atom/`
as flat sibling pairs:

```text
{gpu}_atom_{model}_{precision}[_{mode}].json
{gpu}_atom_{model}_{precision}[_{mode}]_threshold.json
```

On a lab machine, copy each variant into its **own subdirectory** so threshold
discovery is unambiguous (see the input-config README).

| Example config | Mode | Driver |
|---|---|---|
| `mi300x_atom_deepseek-r1_fp8_single` | single-node W1 | `atom` |
| `mi300x_atom_deepseek-r1_fp8_baseline_sweep` | DTNI baseline matrix | `atom` |
| `mi300x_atom_deepseek-r1_fp8_distributed` | 2-node PP W1 | `vllm_atom` |
| `mi300x_atom_deepseek-r1_fp8_sglang_distributed` | 2-node PP W1 | `sglang` |
| `mi300x_atom_deepseek-r1_fp8_mtp3` | single-node MTP3 | `atom` |

See `cvs/input/config_file/inference/atom/README.md` for the full variant
catalog, copy-config commands, cluster-file editing, and step-by-step lab run
recipes.

## Prerequisites

- Passwordless SSH from the control host to each cluster node (key in the
  cluster file), and Docker available on the GPU nodes.
- A container image with ATOM (and vLLM or SGLang when using those drivers);
  shipped configs use `<changeme>` until pinned for your lab.
- A Hugging Face token file at `paths.hf_token_file` when fetching models.
- Model cache at `paths.models_dir` on GPU nodes when `model.remote: 0`.
- For multinode runs: a shared or reachable log path, matching host count in
  the cluster file, `params.master_addr` set to the head VPC IP, and IB/socket
  interfaces discoverable (or explicit `roles.server.ib_hca_devices` /
  `roles.server.ib_netdev` overrides).

## Related code

| Module | Purpose |
|---|---|
| `cvs/lib/inference/atom/atom_orch.py` | `AtomJob` — server/client lifecycle, result parsing |
| `cvs/lib/inference/atom/atom_config_loader.py` | Typed variant load, sweep expansion, session keys |
| `cvs/lib/inference/atom/atom_parsing.py` | Metric tiers, `client.*` mapping, scaling efficiency |
| `cvs/lib/inference/utils/inference_suite_lifecycle.py` | Shared lifecycle stages (`test_launch_container`, …) |
| `cvs/lib/utils/ib_discovery.py` | Multinode IB HCA and socket netdev discovery |
