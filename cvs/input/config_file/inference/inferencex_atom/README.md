# InferenceX ATOM variants

W1 **DeepSeek R1 FP8** on 8× GPU, ISL=OSL=1024, TP8.

## Layout

**In the CVS repo**, all variants live as flat sibling pairs in **this directory**:

```text
{gpu}_inferencex-atom-single_{model}_{precision}[_{mode}]_config.json
{gpu}_inferencex-atom-single_{model}_{precision}[_{mode}]_threshold.json
```

**On your lab machine** (`~/input/config_file/inference/inferencex_atom/`), copy each variant into its **own subdirectory** so only one `*threshold.json` sits next to the config you pass to `--config_file`. `substitute_config` globs the config's parent directory; multiple `*threshold.json` files there raises `ValueError: multiple *threshold.json files … (ambiguous)`.

```text
~/input/.../inferencex_atom/smoke/   # smoke config + smoke threshold only
~/input/.../inferencex_atom/perf/    # perf config + perf threshold only
```

Each shipped config sets `"threshold_json"` to the sibling threshold filename (resolved relative to the config directory). You may also use an absolute path (vLLM-style).

Legacy nested layouts (`deepseek_r1_fp8_mi300x_atom_perf/`, `inferencemax/`, etc.) are **removed** from the repo tree. Use only the flat stems below.

**Config filename example:** `mi300x_inferencex-atom-single_deepseek-r1_fp8_perf_config.json`

| Variant | GPU | Notes |
|---------|-----|-------|
| `mi300x_inferencex-atom-single_deepseek-r1_fp8_smoke` | MI300X | Quick path check (C=128, 128 prompts) |
| `mi300x_inferencex-atom-single_deepseek-r1_fp8_perf` | MI300X | W1 perf, portable min-SLO thresholds, server reuse across sweep |
| `mi300x_inferencex-atom-single_deepseek-r1_fp8_perf_baseline_sweep` | MI300X | **DTNI baseline matrix:** 1K/1K + 8K/1K × C=4–256 (14 cells); `max_model_length=10240` |
| `mi300x_inferencex-atom-single_deepseek-r1_fp8_perf_baseline_sweep_multinode` | MI300X | **2-node** DTNI baseline matrix (same 14 cells); `PP=2`, `nnodes=2`, scaling gates |
| `mi300x_inferencex-atom-single_deepseek-r1_fp8_perf_multi` | MI300X | W1 **2-node** scaling (`nnodes=2`, `PP=2`); `enforce_thresholds: false` until lab confirm |
| `mi300x_inferencex-atom-single_deepseek-r1_fp8_mtp3` | MI300X | W1 FP8+MTP3 |
| `mi355x_inferencex-atom-single_deepseek-r1_fp8_perf` | MI355X | W1 perf (CI seeds, `enforce_thresholds: false`) |
| `mi355x_inferencex-atom-single_deepseek-r1_fp8_perf_baseline_sweep` | MI355X | **DTNI baseline matrix:** 1K/1K + 8K/1K × C=4–256 (14 cells); threshold seeds, `enforce_thresholds: false` |
| `mi355x_inferencex-atom-single_deepseek-r1_fp8_perf_multi` | MI355X | W1 **2-node** scaling (`nnodes=2`, `PP=2`); `enforce_thresholds: false` until lab confirm |
| `mi355x_inferencex-atom-single_deepseek-r1_fp8_mtp3` | MI355X | W1 FP8+MTP3 |
| `mi300x_inferencex-atom-single_gpt-oss-120b_bf16` | MI300X | GPT-OSS uplift placeholder (`driver: vllm`, inline `serve_args`) |
| `mi355x_inferencex-atom-single_gpt-oss-120b_bf16` | MI355X | GPT-OSS uplift placeholder |

ATOM server CLI is inline in each config under `roles.server.atom_args` (vLLM-style, same as `roles.server.serve_args` on `vllm_single`). MTP3 variants also set `params.bench_extra_args`.

**Model cache path:** shipped configs set `paths.models_dir` to `/home/models` and mount `/home/models:/home/models` into the container. Logs and HF token paths still use `{shared_fs}` under the SSH user home.

## Cluster file

Ship one template: `cvs/input/cluster_file/inferencex_atom_cluster.json`. Copy it to `~/input/cluster_file/inferencex_atom_cluster.json` and edit IPs, SSH user, and key path.

**Host count must match the variant:** `len(node_dict)` must equal `params.nnodes` in the config you pass to `--config_file`.

| Variant type | `params.nnodes` | `node_dict` |
|--------------|-----------------|-------------|
| Smoke, baseline sweep (single-node) | `1` (default) | **Head node only** — remove the worker entry |
| Baseline sweep multinode, multinode perf | `2` | Head + worker |

For multinode variants, set `params.master_addr` to the head VPC IP (same as `head_node_dict.mgmt_ip`). `test_setup_sshd` runs when `len(node_dict) > 1`. Variant config overrides cluster `container.image` and `container.name`.

## Shared suite helpers (reusable by other inference suites)

| Module | Purpose |
|--------|---------|
| `cvs/lib/inference/utils/inference_suite_lifecycle.py` | Lifecycle stage tests, `InferenceLifecycle`, pytest HTML hooks |
| `cvs/lib/inference/utils/inference_suite_results_table.py` | Configurable results table (`make_print_results_table`) |
| `cvs/lib/inference/unittests/fake_orch.py` | `FakeOrch` for Job parse unit tests |

`inferencex_atom` imports these today; `vllm_single` may adopt them in a follow-up without duplicating code.

## Pytest layout

1. `test_launch_container` → `test_setup_sshd` → `test_model_fetch`
2. `test_inferencex_atom_inference` (per sweep cell; reuses server when `reuse_server_across_sweep: true`)
3. `test_cell_metrics` (one HTML row per **metric tier** per cell: throughput, ttft, tpot, health, record)
4. `test_print_results_table` → `test_teardown`

W1 MI300X perf with two concurrency cells expects **~17** pytest rows (not one row per scalar metric).

## Before the first lab run

- On the **launcher** host after `git checkout` / `git pull`: run **`make install` first**, then **`source .cvs_venv/bin/activate`**. Do not activate `.cvs_venv` before `make install` — the Makefile manages that venv and install can fail if it is already active.

```bash
cd ~/cvs
git fetch origin hnimrama/ix-atom-multinode
git reset --hard origin/hnimrama/ix-atom-multinode
make install
source .cvs_venv/bin/activate
```

- Edit `~/input/cluster_file/inferencex_atom_cluster.json`: node IPs, `username`, `priv_key_file`. Trim `node_dict` to one host for single-node variants.
- **Launcher vs GPU node:** CVS pytest runs on the launcher; `ContainerOrchestrator` SSHes to cluster nodes and runs `sudo docker` there. Local Docker on the launcher is not used. Prerequisites split by host:

  | Item | Launcher | GPU node (cluster `mgmt_ip`) |
  |------|----------|------------------------------|
  | `cvs run`, venv, `~/input/`, `~/cvs_results/` | Yes | No |
  | `priv_key_file`, `~/.hf_token` (read locally by pytest) | Yes | No |
  | `/home/models` (when `model.remote: 0`) | No | Yes |
  | `rocm/atom-dev` image, `sudo docker` | No | Yes |
  | `~/LOGS/` (server/bench logs via volume mount) | No | Yes |

- Preflight from the launcher: `ssh -i ~/.ssh/<key> <user>@<mgmt_ip> 'sudo docker images | grep atom-dev; du -sh /home/models'`

## Smoke (MI300X)

One cell, 128 prompts — run this before the full perf matrix.

```bash
cd ~/cvs
make install   # after git pull only; run before activating venv
source .cvs_venv/bin/activate
mkdir -p ~/cvs_results ~/input/cluster_file

SMOKE_DIR=~/input/config_file/inference/inferencex_atom/smoke
mkdir -p "$SMOKE_DIR"

cvs copy-config inference/inferencex_atom/mi300x_inferencex-atom-single_deepseek-r1_fp8_smoke_config.json \
  --output "$SMOKE_DIR/mi300x_inferencex-atom-single_deepseek-r1_fp8_smoke_config.json"
cvs copy-config inference/inferencex_atom/mi300x_inferencex-atom-single_deepseek-r1_fp8_smoke_threshold.json \
  --output "$SMOKE_DIR/mi300x_inferencex-atom-single_deepseek-r1_fp8_smoke_threshold.json"
cvs copy-config inferencex_atom_cluster.json --output ~/input/cluster_file/inferencex_atom_cluster.json

TS=$(date +%Y%m%d_%H%M%S)
HTML=~/cvs_results/${TS}_ix-atom-smoke_mi300x.html
LOG=~/cvs_results/${TS}_ix-atom-smoke_mi300x.log

cvs run inferencex_atom \
  --cluster_file ~/input/cluster_file/inferencex_atom_cluster.json \
  --config_file "$SMOKE_DIR/mi300x_inferencex-atom-single_deepseek-r1_fp8_smoke_config.json" \
  --html="$HTML" \
  --self-contained-html \
  --log-file="$LOG" \
  -vvv -s

echo "HTML: $HTML"
echo "LOG:  $LOG"
```

When `--html` is set, the **IX Run Deck** (`inferencex_atom_run_deck.html`, `.json`,
`_viewer.html`) is generated at session end and bundled into the pytest zip.
See `cvs/lib/report/README.md` for wiring other suites. Open the pytest HTML **Reports**
section for links. Render-only; does not affect gates.

## W1 perf (MI300X)

Two concurrency cells (C=128, C=256), 1000 prompts. Second cell reuses the ATOM server when `reuse_server_across_sweep: true`.

```bash
cd ~/cvs
make install   # after git pull only; run before activating venv
source .cvs_venv/bin/activate

PERF_DIR=~/input/config_file/inference/inferencex_atom/perf
mkdir -p "$PERF_DIR"

cvs copy-config inference/inferencex_atom/mi300x_inferencex-atom-single_deepseek-r1_fp8_perf_config.json \
  --output "$PERF_DIR/mi300x_inferencex-atom-single_deepseek-r1_fp8_perf_config.json"
cvs copy-config inference/inferencex_atom/mi300x_inferencex-atom-single_deepseek-r1_fp8_perf_threshold.json \
  --output "$PERF_DIR/mi300x_inferencex-atom-single_deepseek-r1_fp8_perf_threshold.json"

TS=$(date +%Y%m%d_%H%M%S)
HTML=~/cvs_results/${TS}_ix-atom-w1-perf_mi300x.html
LOG=~/cvs_results/${TS}_ix-atom-w1-perf_mi300x.log

cvs run inferencex_atom \
  --cluster_file ~/input/cluster_file/inferencex_atom_cluster.json \
  --config_file "$PERF_DIR/mi300x_inferencex-atom-single_deepseek-r1_fp8_perf_config.json" \
  --html="$HTML" \
  --self-contained-html \
  --log-file="$LOG" \
  -vvv -s

echo "HTML: $HTML"
echo "LOG:  $LOG"
```

## W1 perf baseline sweep (MI300X) — DTNI matrix

DTNI baseline matrix: **1K/1K** and **8K/1K** at **C=4, 8, 16, 32, 64, 128, 256** (14 cells). `max_model_length=10240`. Expect a long run (~several hours); server is reused within each shape.

```bash
cd ~/cvs
make install
source .cvs_venv/bin/activate

BASELINE_DIR=~/input/config_file/inference/inferencex_atom/baseline_sweep
mkdir -p "$BASELINE_DIR"

cvs copy-config inference/inferencex_atom/mi300x_inferencex-atom-single_deepseek-r1_fp8_perf_baseline_sweep_config.json \
  --output "$BASELINE_DIR/mi300x_inferencex-atom-single_deepseek-r1_fp8_perf_baseline_sweep_config.json"
cvs copy-config inference/inferencex_atom/mi300x_inferencex-atom-single_deepseek-r1_fp8_perf_baseline_sweep_threshold.json \
  --output "$BASELINE_DIR/mi300x_inferencex-atom-single_deepseek-r1_fp8_perf_baseline_sweep_threshold.json"

TS=$(date +%Y%m%d_%H%M%S)
HTML=~/cvs_results/${TS}_ix-atom-baseline-sweep_mi300x.html
LOG=~/cvs_results/${TS}_ix-atom-baseline-sweep_mi300x.log

cvs run inferencex_atom \
  --cluster_file ~/input/cluster_file/inferencex_atom_cluster.json \
  --config_file "$BASELINE_DIR/mi300x_inferencex-atom-single_deepseek-r1_fp8_perf_baseline_sweep_config.json" \
  --html="$HTML" \
  --self-contained-html \
  --log-file="$LOG" \
  -vvv -s

echo "HTML: $HTML"
echo "LOG:  $LOG"
```

## W1 perf baseline sweep multinode (MI300X, 2-node)

Same **14-cell** DTNI matrix as single-node baseline sweep (1K/1K + 8K/1K × C=4–256), with `nnodes=2`, `PP=2`, and `scaling.efficiency_pct` gates. Expect a long run (~4–8 hours). Use a **2-host** `inferencex_atom_cluster.json` and set `params.master_addr` to the head VPC IP after `copy-config` (replace `{head-node-ip}` placeholder).

```bash
cd ~/cvs
make install
source .cvs_venv/bin/activate

BASELINE_MULTI_DIR=~/input/config_file/inference/inferencex_atom/baseline_sweep_multinode
mkdir -p "$BASELINE_MULTI_DIR"

cvs copy-config inference/inferencex_atom/mi300x_inferencex-atom-single_deepseek-r1_fp8_perf_baseline_sweep_multinode_config.json \
  --output "$BASELINE_MULTI_DIR/mi300x_inferencex-atom-single_deepseek-r1_fp8_perf_baseline_sweep_multinode_config.json"
cvs copy-config inference/inferencex_atom/mi300x_inferencex-atom-single_deepseek-r1_fp8_perf_baseline_sweep_multinode_threshold.json \
  --output "$BASELINE_MULTI_DIR/mi300x_inferencex-atom-single_deepseek-r1_fp8_perf_baseline_sweep_multinode_threshold.json"
cvs copy-config inferencex_atom_cluster.json --output ~/input/cluster_file/inferencex_atom_cluster.json

# Ensure node_dict lists head + worker. Edit cluster IPs and set master_addr in the copied config.

TS=$(date +%Y%m%d_%H%M%S)
HTML=~/cvs_results/${TS}_ix-atom-baseline-sweep-multinode_mi300x.html
LOG=~/cvs_results/${TS}_ix-atom-baseline-sweep-multinode_mi300x.log

cvs run inferencex_atom \
  --cluster_file ~/input/cluster_file/inferencex_atom_cluster.json \
  --config_file "$BASELINE_MULTI_DIR/mi300x_inferencex-atom-single_deepseek-r1_fp8_perf_baseline_sweep_multinode_config.json" \
  --html="$HTML" \
  --self-contained-html \
  --log-file="$LOG" \
  -vvv -s

echo "HTML: $HTML"
echo "LOG:  $LOG"
```

## W1 perf multinode (MI300X, 2-node)

Requires a 2-node cluster file, ATOM image with distributed serve support, and fabric env in `roles.server.env` (NCCL socket ifnames, etc.). One cell matches the single-node W1 reference (ISL=OSL=1024, C=128).

```bash
cd ~/cvs
make install   # after git pull only; run before activating venv
source .cvs_venv/bin/activate

MULTI_DIR=~/input/config_file/inference/inferencex_atom/multi
mkdir -p "$MULTI_DIR"

cvs copy-config inference/inferencex_atom/mi300x_inferencex-atom-single_deepseek-r1_fp8_perf_multi_config.json \
  --output "$MULTI_DIR/mi300x_inferencex-atom-single_deepseek-r1_fp8_perf_multi_config.json"
cvs copy-config inference/inferencex_atom/mi300x_inferencex-atom-single_deepseek-r1_fp8_perf_multi_threshold.json \
  --output "$MULTI_DIR/mi300x_inferencex-atom-single_deepseek-r1_fp8_perf_multi_threshold.json"
cvs copy-config inferencex_atom_cluster.json --output ~/input/cluster_file/inferencex_atom_cluster.json

# Ensure node_dict lists head + worker (params.nnodes=2 in multinode variant).

# Edit cluster + config: replace {head-node-ip} / {worker-node-ip} and set params.master_addr.

TS=$(date +%Y%m%d_%H%M%S)
HTML=~/cvs_results/${TS}_ix-atom-w1-perf-multi_mi300x.html
LOG=~/cvs_results/${TS}_ix-atom-w1-perf-multi_mi300x.log

cvs run inferencex_atom \
  --cluster_file ~/input/cluster_file/inferencex_atom_cluster.json \
  --config_file "$MULTI_DIR/mi300x_inferencex-atom-single_deepseek-r1_fp8_perf_multi_config.json" \
  --html="$HTML" \
  --self-contained-html \
  --log-file="$LOG" \
  -vvv -s

echo "HTML: $HTML"
echo "LOG:  $LOG"
```

## W1 perf multinode (MI355X, 2-node)

Same sweep matrix as MI300X multinode. Thresholds are seeded from the MI355X single-node CI reference; `enforce_thresholds` stays `false` until a 2-node MI355X lab run confirms.

```bash
cd ~/cvs
make install   # after git pull only; run before activating venv
source .cvs_venv/bin/activate

MULTI_DIR=~/input/config_file/inference/inferencex_atom/mi355x_multi
mkdir -p "$MULTI_DIR"

cvs copy-config inference/inferencex_atom/mi355x_inferencex-atom-single_deepseek-r1_fp8_perf_multi_config.json \
  --output "$MULTI_DIR/mi355x_inferencex-atom-single_deepseek-r1_fp8_perf_multi_config.json"
cvs copy-config inference/inferencex_atom/mi355x_inferencex-atom-single_deepseek-r1_fp8_perf_multi_threshold.json \
  --output "$MULTI_DIR/mi355x_inferencex-atom-single_deepseek-r1_fp8_perf_multi_threshold.json"
cvs copy-config inferencex_atom_cluster.json --output ~/input/cluster_file/inferencex_atom_cluster.json

# Ensure node_dict lists head + worker (params.nnodes=2 in multinode variant).

# Edit cluster + config: replace {head-node-ip} / {worker-node-ip} and set params.master_addr.

TS=$(date +%Y%m%d_%H%M%S)
HTML=~/cvs_results/${TS}_ix-atom-w1-perf-multi_mi355x.html
LOG=~/cvs_results/${TS}_ix-atom-w1-perf-multi_mi355x.log

cvs run inferencex_atom \
  --cluster_file ~/input/cluster_file/inferencex_atom_cluster.json \
  --config_file "$MULTI_DIR/mi355x_inferencex-atom-single_deepseek-r1_fp8_perf_multi_config.json" \
  --html="$HTML" \
  --self-contained-html \
  --log-file="$LOG" \
  -vvv -s

echo "HTML: $HTML"
echo "LOG:  $LOG"
```

## W1 perf (MI355X)

Thresholds are seeded from [ROCm/ATOM run 27912164002](https://github.com/ROCm/ATOM/actions/runs/27912164002). `enforce_thresholds` stays `false` until an MI355X lab run confirms.

```bash
cd ~/cvs
make install   # after git pull only; run before activating venv
source .cvs_venv/bin/activate

PERF_DIR=~/input/config_file/inference/inferencex_atom/mi355x_perf
mkdir -p "$PERF_DIR"

cvs copy-config inference/inferencex_atom/mi355x_inferencex-atom-single_deepseek-r1_fp8_perf_config.json \
  --output "$PERF_DIR/mi355x_inferencex-atom-single_deepseek-r1_fp8_perf_config.json"
cvs copy-config inference/inferencex_atom/mi355x_inferencex-atom-single_deepseek-r1_fp8_perf_threshold.json \
  --output "$PERF_DIR/mi355x_inferencex-atom-single_deepseek-r1_fp8_perf_threshold.json"
cvs copy-config inferencex_atom_cluster.json --output ~/input/cluster_file/inferencex_atom_cluster.json

TS=$(date +%Y%m%d_%H%M%S)
HTML=~/cvs_results/${TS}_ix-atom-w1-perf_mi355x.html
LOG=~/cvs_results/${TS}_ix-atom-w1-perf_mi355x.log

cvs run inferencex_atom \
  --cluster_file ~/input/cluster_file/inferencex_atom_cluster.json \
  --config_file "$PERF_DIR/mi355x_inferencex-atom-single_deepseek-r1_fp8_perf_config.json" \
  --html="$HTML" \
  --self-contained-html \
  --log-file="$LOG" \
  -vvv -s

echo "HTML: $HTML"
echo "LOG:  $LOG"
```
