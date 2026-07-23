# InferenceX ATOM variants

W1 **DeepSeek R1 FP8** on 8× GPU, ISL=OSL=1024, TP8.

## Layout

**In the CVS repo**, all variants live as flat sibling pairs in **this directory**:

```text
{gpu}_inferencex-atom_{model}_{precision}[_{mode}].json
{gpu}_inferencex-atom_{model}_{precision}[_{mode}]_threshold.json
```

Same convention as ``inference/vllm/`` (for example ``mi300x_vllm_llama31-70b_fp8_single.json`` / ``…_distributed.json``): flat sibling pairs, no ``_config`` suffix on the main JSON.

**On your lab machine** (`~/input/config_file/inference/inferencex_atom/`), copy each variant into its **own subdirectory** so only one `*threshold.json` sits next to the config you pass to `--config_file`. `substitute_config` globs the config's parent directory; multiple `*threshold.json` files there raises `ValueError: multiple *threshold.json files … (ambiguous)`.

```text
~/input/.../inferencex_atom/single/   # single-node config + threshold only
~/input/.../inferencex_atom/distributed/  # vllm_atom PP=2 config + threshold only
~/input/.../inferencex_atom/sglang_distributed/  # sglang PP=2 config + threshold only
```

Each shipped config sets `"threshold_json"` to the sibling threshold filename (resolved relative to the config directory). You may also use an absolute path (vLLM-style).

Legacy nested layouts (`deepseek_r1_fp8_mi300x_atom_perf/`, `inferencemax/`, etc.) are **removed** from the repo tree. Use only the flat stems below.

**Config filename example:** `mi300x_inferencex-atom_deepseek-r1_fp8_single.json`

| Variant | GPU | Driver | Notes |
|---------|-----|--------|-------|
| `mi300x_inferencex-atom_deepseek-r1_fp8_single` | MI300X | `atom` | W1 single-node, portable min-SLO thresholds, server reuse across sweep |
| `mi300x_inferencex-atom_deepseek-r1_fp8_baseline_sweep` | MI300X | `atom` | **DTNI baseline matrix:** 1K/1K + 8K/1K × C=4–256 (14 cells); `max_model_length=10240` |
| `mi300x_inferencex-atom_deepseek-r1_fp8_baseline_sweep_distributed` | MI300X | `vllm_atom` | **2-node** DTNI baseline (14 cells); `PP=2`, scaling gates |
| `mi300x_inferencex-atom_deepseek-r1_fp8_distributed` | MI300X | `vllm_atom` | W1 **2-node** PP=2; `enforce_thresholds: true` after lab recalibration |
| `mi300x_inferencex-atom_deepseek-r1_fp8_sglang_distributed` | MI300X | `sglang` | W1 **2-node** PP=2; `enforce_thresholds: false` until lab confirm |
| `mi300x_inferencex-atom_deepseek-r1_fp8_mtp3` | MI300X | `atom` | W1 FP8+MTP3 |
| `mi355x_inferencex-atom_deepseek-r1_fp8_single` | MI355X | `atom` | W1 single-node (CI seeds, `enforce_thresholds: false`) |
| `mi355x_inferencex-atom_deepseek-r1_fp8_baseline_sweep` | MI355X | `atom` | **DTNI baseline matrix:** 1K/1K + 8K/1K × C=4–256 (14 cells); threshold seeds, `enforce_thresholds: false` |
| `mi355x_inferencex-atom_deepseek-r1_fp8_distributed` | MI355X | `vllm_atom` | W1 **2-node** PP=2; `enforce_thresholds: false` until lab confirm |
| `mi355x_inferencex-atom_deepseek-r1_fp8_mtp3` | MI355X | `atom` | W1 FP8+MTP3 |
| `mi300x_inferencex-atom_gpt-oss-120b_bf16` | MI300X | `vllm` | GPT-OSS uplift placeholder |
| `mi355x_inferencex-atom_gpt-oss-120b_bf16` | MI355X | `vllm` | GPT-OSS uplift placeholder |

**Removed:** `*_smoke` variant (use `-k` on `single`, `distributed`, or `sglang_distributed` for a one-cell smoke). **Removed:** bare `driver=atom` multinode PP — use `vllm_atom` or `sglang` distributed stems above.

ATOM server CLI for **`driver=atom`** lives in `roles.server.atom_args`. Multinode **PP=2** variants use **`driver=vllm_atom`** (`roles.server.serve_args`) or **`driver=sglang`** (`roles.server.sglang_args`). MTP3 variants also set `params.bench_extra_args`.

## Execution drivers (`params.driver`)

Standalone ATOM has **no native pipeline parallel**. Multinode PP validation requires a framework coordinator:

| Driver | When to use | Server | Multinode PP |
|--------|-------------|--------|--------------|
| `atom` | W1 single-node (`*_single`, baseline sweep, MTP3) | `atom.entrypoints.openai_server` | No — single host only |
| `vllm_atom` | **2-node PP=2** (shipped multinode stems) | `vllm serve` + ATOM ROCm env | Yes — vLLM `--pipeline-parallel-size`, `--node-rank` |
| `sglang` | **2-node PP=2** SGLang path | `sglang.launch_server` | Yes — `--pp-size`, `--dist-init-addr` |
| `vllm` | GPT-OSS uplift placeholder only | `vllm serve` | Same PP flags as `vllm_atom` when configured |

**Before a multinode PP lab run**, set in the copied config:

- `container.image` — vLLM+ATOM or SGLang-capable image (shipped configs use `<changeme>`)
- `params.master_addr` — head node VPC IP (replace `{head-node-ip}`)

Multinode fabric is probed once per run in `test_discover_topology`:

- `roles.server.ib_hca_devices: "auto"` (default) → `NCCL_IB_HCA` from `ibv_devinfo`
- `roles.server.ib_netdev: "auto"` (default on distributed stems) → `GLOO_SOCKET_IFNAME` / `NCCL_SOCKET_IFNAME` from the cluster IP on each node

Override `ib_netdev` only when auto-discovery fails (asymmetric interface names) or you need a non-default NIC. Do **not** set `mlx5_*` — those are IB HCAs, not IP netdevs.

Threshold cell keys for multinode PP: `ISL=…,OSL=…,TP=8,PP=2,NNODES=2,CONC=…`.

**Model cache path:** shipped configs set `paths.models_dir` to `/home/models` and mount `/home/models:/home/models` into the container. Logs and HF token paths still use `{shared_fs}` under the SSH user home.

## Cluster file

Ship one template: `cvs/input/cluster_file/inferencex_atom_cluster.json`. Copy it to `~/input/cluster_file/inferencex_atom_cluster.json` and edit IPs, SSH user, and key path.

**Host count must match the variant:** `len(node_dict)` must equal `params.nnodes` in the config you pass to `--config_file`.

| Variant type | `params.nnodes` | `node_dict` |
|--------------|-----------------|-------------|
| Single-node (`*_single`, baseline sweep, MTP3) | `1` (default) | **Head node only** — remove the worker entry |
| Multinode PP (`*_distributed`, `*_baseline_sweep_distributed`, `*_sglang_distributed`) | `2` | Head + worker; use lab subdirs `distributed/` or `sglang_distributed/` |

For multinode PP variants, set `params.master_addr` to the head VPC IP and a coordinator-capable `container.image`. Fabric netdev/HCAs are discovered in `test_discover_topology` unless overridden. `test_setup_sshd` runs when `len(node_dict) > 1`.

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

W1 MI300X single with two concurrency cells expects **~17** pytest rows (not one row per scalar metric).

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

## W1 single-node (MI300X, `driver=atom`)

Two concurrency cells (C=128, C=256), 1000 prompts. Second cell reuses the ATOM server when `reuse_server_across_sweep: true`. For a quick smoke, add `-k "w1_1k_1k-conc128"`.

```bash
cd ~/cvs
make install   # after git pull only; run before activating venv
source .cvs_venv/bin/activate

SINGLE_DIR=~/input/config_file/inference/inferencex_atom/single
mkdir -p "$SINGLE_DIR"

cvs copy-config inference/inferencex_atom/mi300x_inferencex-atom_deepseek-r1_fp8_single.json \
  --output "$SINGLE_DIR/mi300x_inferencex-atom_deepseek-r1_fp8_single.json"
cvs copy-config inference/inferencex_atom/mi300x_inferencex-atom_deepseek-r1_fp8_single_threshold.json \
  --output "$SINGLE_DIR/mi300x_inferencex-atom_deepseek-r1_fp8_single_threshold.json"

TS=$(date +%Y%m%d_%H%M%S)
HTML=~/cvs_results/${TS}_ix-atom-w1-single_mi300x.html
LOG=~/cvs_results/${TS}_ix-atom-w1-single_mi300x.log

cvs run inferencex_atom \
  --cluster_file ~/input/cluster_file/inferencex_atom_cluster.json \
  --config_file "$SINGLE_DIR/mi300x_inferencex-atom_deepseek-r1_fp8_single.json" \
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

## W1 perf baseline sweep (MI300X) — DTNI matrix

DTNI baseline matrix: **1K/1K** and **8K/1K** at **C=4, 8, 16, 32, 64, 128, 256** (14 cells). `max_model_length=10240`. Expect a long run (~several hours); server is reused within each shape.

```bash
cd ~/cvs
make install
source .cvs_venv/bin/activate

BASELINE_DIR=~/input/config_file/inference/inferencex_atom/baseline_sweep
mkdir -p "$BASELINE_DIR"

cvs copy-config inference/inferencex_atom/mi300x_inferencex-atom_deepseek-r1_fp8_baseline_sweep.json \
  --output "$BASELINE_DIR/mi300x_inferencex-atom_deepseek-r1_fp8_baseline_sweep.json"
cvs copy-config inference/inferencex_atom/mi300x_inferencex-atom_deepseek-r1_fp8_baseline_sweep_threshold.json \
  --output "$BASELINE_DIR/mi300x_inferencex-atom_deepseek-r1_fp8_baseline_sweep_threshold.json"

TS=$(date +%Y%m%d_%H%M%S)
HTML=~/cvs_results/${TS}_ix-atom-baseline-sweep_mi300x.html
LOG=~/cvs_results/${TS}_ix-atom-baseline-sweep_mi300x.log

cvs run inferencex_atom \
  --cluster_file ~/input/cluster_file/inferencex_atom_cluster.json \
  --config_file "$BASELINE_DIR/mi300x_inferencex-atom_deepseek-r1_fp8_baseline_sweep.json" \
  --html="$HTML" \
  --self-contained-html \
  --log-file="$LOG" \
  -vvv -s

echo "HTML: $HTML"
echo "LOG:  $LOG"
```

## W1 perf baseline sweep multinode (MI300X, 2-node)

Same **14-cell** DTNI matrix as single-node baseline sweep (1K/1K + 8K/1K × C=4–256), with `nnodes=2`, **`driver=vllm_atom`**, **`pipeline_parallel_size=2`** (true pipeline parallel via vLLM coordinator + ATOM kernels; cell keys use `PP=2`), and `scaling.efficiency_pct` gates. Set `roles.server.ib_netdev` and a vLLM+ATOM container image before lab run. Expect a long run (~4–8 hours). Use a **2-host** `inferencex_atom_cluster.json` and set `params.master_addr` to the head VPC IP after `copy-config` (replace `{head-node-ip}` placeholder).

```bash
cd ~/cvs
make install
source .cvs_venv/bin/activate

BASELINE_MULTI_DIR=~/input/config_file/inference/inferencex_atom/baseline_sweep_distributed
mkdir -p "$BASELINE_MULTI_DIR"

cvs copy-config inference/inferencex_atom/mi300x_inferencex-atom_deepseek-r1_fp8_baseline_sweep_distributed.json \
  --output "$BASELINE_MULTI_DIR/mi300x_inferencex-atom_deepseek-r1_fp8_baseline_sweep_distributed.json"
cvs copy-config inference/inferencex_atom/mi300x_inferencex-atom_deepseek-r1_fp8_baseline_sweep_distributed_threshold.json \
  --output "$BASELINE_MULTI_DIR/mi300x_inferencex-atom_deepseek-r1_fp8_baseline_sweep_distributed_threshold.json"
cvs copy-config inferencex_atom_cluster.json --output ~/input/cluster_file/inferencex_atom_cluster.json

# Ensure node_dict lists head + worker. Edit cluster IPs and set master_addr in the copied config.

TS=$(date +%Y%m%d_%H%M%S)
HTML=~/cvs_results/${TS}_ix-atom-baseline-sweep-multinode_mi300x.html
LOG=~/cvs_results/${TS}_ix-atom-baseline-sweep-multinode_mi300x.log

cvs run inferencex_atom \
  --cluster_file ~/input/cluster_file/inferencex_atom_cluster.json \
  --config_file "$BASELINE_MULTI_DIR/mi300x_inferencex-atom_deepseek-r1_fp8_baseline_sweep_distributed.json" \
  --html="$HTML" \
  --self-contained-html \
  --log-file="$LOG" \
  -vvv -s

echo "HTML: $HTML"
echo "LOG:  $LOG"
```

## W1 perf multinode (MI300X, 2-node, `driver=vllm_atom`)

15-cell W1 scaling matrix with **`pipeline_parallel_size=2`**, **`driver=vllm_atom`**, and `scaling.efficiency_pct` gates. Requires vLLM+ATOM container, `ib_netdev`, and 2-node cluster. Recalibrate thresholds after the first true PP=2 lab run.

```bash
cd ~/cvs
make install   # after git pull only; run before activating venv
source .cvs_venv/bin/activate

DISTRIBUTED_DIR=~/input/config_file/inference/inferencex_atom/distributed
mkdir -p "$DISTRIBUTED_DIR"

cvs copy-config inference/inferencex_atom/mi300x_inferencex-atom_deepseek-r1_fp8_distributed.json \
  --output "$DISTRIBUTED_DIR/mi300x_inferencex-atom_deepseek-r1_fp8_distributed.json"
cvs copy-config inference/inferencex_atom/mi300x_inferencex-atom_deepseek-r1_fp8_distributed_threshold.json \
  --output "$DISTRIBUTED_DIR/mi300x_inferencex-atom_deepseek-r1_fp8_distributed_threshold.json"
cvs copy-config inferencex_atom_cluster.json --output ~/input/cluster_file/inferencex_atom_cluster.json

# Ensure node_dict lists head + worker. Edit cluster IPs, ib_netdev, container.image,
# and set params.master_addr in the copied config.

TS=$(date +%Y%m%d_%H%M%S)
HTML=~/cvs_results/${TS}_ix-atom-w1-perf-multi_mi300x.html
LOG=~/cvs_results/${TS}_ix-atom-w1-perf-multi_mi300x.log

cvs run inferencex_atom \
  --cluster_file ~/input/cluster_file/inferencex_atom_cluster.json \
  --config_file "$DISTRIBUTED_DIR/mi300x_inferencex-atom_deepseek-r1_fp8_distributed.json" \
  --html="$HTML" \
  --self-contained-html \
  --log-file="$LOG" \
  -vvv -s

echo "HTML: $HTML"
echo "LOG:  $LOG"
```

## W1 perf multinode SGLang (MI300X, 2-node, `driver=sglang`)

Same 15-cell sweep as vLLM-ATOM multinode, using SGLang pipeline parallel. `enforce_thresholds: false` until lab confirms — seed thresholds only.

```bash
cd ~/cvs
make install
source .cvs_venv/bin/activate

SGLANG_DIR=~/input/config_file/inference/inferencex_atom/sglang_distributed
mkdir -p "$SGLANG_DIR"

cvs copy-config inference/inferencex_atom/mi300x_inferencex-atom_deepseek-r1_fp8_sglang_distributed.json \
  --output "$SGLANG_DIR/mi300x_inferencex-atom_deepseek-r1_fp8_sglang_distributed.json"
cvs copy-config inference/inferencex_atom/mi300x_inferencex-atom_deepseek-r1_fp8_sglang_distributed_threshold.json \
  --output "$SGLANG_DIR/mi300x_inferencex-atom_deepseek-r1_fp8_sglang_distributed_threshold.json"
cvs copy-config inferencex_atom_cluster.json --output ~/input/cluster_file/inferencex_atom_cluster.json

# Edit cluster IPs, ib_netdev, SGLang container.image, params.master_addr.

TS=$(date +%Y%m%d_%H%M%S)
HTML=~/cvs_results/${TS}_ix-atom-w1-perf-multi-sglang_mi300x.html
LOG=~/cvs_results/${TS}_ix-atom-w1-perf-multi-sglang_mi300x.log

cvs run inferencex_atom \
  --cluster_file ~/input/cluster_file/inferencex_atom_cluster.json \
  --config_file "$SGLANG_DIR/mi300x_inferencex-atom_deepseek-r1_fp8_sglang_distributed.json" \
  --html="$HTML" \
  --self-contained-html \
  --log-file="$LOG" \
  -vvv -s

echo "HTML: $HTML"
echo "LOG:  $LOG"
```

## W1 perf multinode (MI355X, 2-node, `driver=vllm_atom`)

Same sweep matrix as MI300X multinode (`vllm_atom`, PP=2). Thresholds are seeded from the MI355X single-node CI reference; `enforce_thresholds` stays `false` until a 2-node MI355X lab run confirms.

```bash
cd ~/cvs
make install   # after git pull only; run before activating venv
source .cvs_venv/bin/activate

DISTRIBUTED_DIR=~/input/config_file/inference/inferencex_atom/mi355x_distributed
mkdir -p "$DISTRIBUTED_DIR"

cvs copy-config inference/inferencex_atom/mi355x_inferencex-atom_deepseek-r1_fp8_distributed.json \
  --output "$DISTRIBUTED_DIR/mi355x_inferencex-atom_deepseek-r1_fp8_distributed.json"
cvs copy-config inference/inferencex_atom/mi355x_inferencex-atom_deepseek-r1_fp8_distributed_threshold.json \
  --output "$DISTRIBUTED_DIR/mi355x_inferencex-atom_deepseek-r1_fp8_distributed_threshold.json"
cvs copy-config inferencex_atom_cluster.json --output ~/input/cluster_file/inferencex_atom_cluster.json

# Ensure node_dict lists head + worker (params.nnodes=2 in multinode variant).

# Edit cluster + config: replace {head-node-ip} / {worker-node-ip} and set params.master_addr.

TS=$(date +%Y%m%d_%H%M%S)
HTML=~/cvs_results/${TS}_ix-atom-w1-perf-multi_mi355x.html
LOG=~/cvs_results/${TS}_ix-atom-w1-perf-multi_mi355x.log

cvs run inferencex_atom \
  --cluster_file ~/input/cluster_file/inferencex_atom_cluster.json \
  --config_file "$DISTRIBUTED_DIR/mi355x_inferencex-atom_deepseek-r1_fp8_distributed.json" \
  --html="$HTML" \
  --self-contained-html \
  --log-file="$LOG" \
  -vvv -s

echo "HTML: $HTML"
echo "LOG:  $LOG"
```

## W1 single-node (MI355X, `driver=atom`)

Thresholds are seeded from [ROCm/ATOM run 27912164002](https://github.com/ROCm/ATOM/actions/runs/27912164002). `enforce_thresholds` stays `false` until an MI355X lab run confirms.

```bash
cd ~/cvs
make install   # after git pull only; run before activating venv
source .cvs_venv/bin/activate

SINGLE_DIR=~/input/config_file/inference/inferencex_atom/mi355x_single
mkdir -p "$SINGLE_DIR"

cvs copy-config inference/inferencex_atom/mi355x_inferencex-atom_deepseek-r1_fp8_single.json \
  --output "$SINGLE_DIR/mi355x_inferencex-atom_deepseek-r1_fp8_single.json"
cvs copy-config inference/inferencex_atom/mi355x_inferencex-atom_deepseek-r1_fp8_single_threshold.json \
  --output "$SINGLE_DIR/mi355x_inferencex-atom_deepseek-r1_fp8_single_threshold.json"
cvs copy-config inferencex_atom_cluster.json --output ~/input/cluster_file/inferencex_atom_cluster.json

TS=$(date +%Y%m%d_%H%M%S)
HTML=~/cvs_results/${TS}_ix-atom-w1-single_mi355x.html
LOG=~/cvs_results/${TS}_ix-atom-w1-single_mi355x.log

cvs run inferencex_atom \
  --cluster_file ~/input/cluster_file/inferencex_atom_cluster.json \
  --config_file "$SINGLE_DIR/mi355x_inferencex-atom_deepseek-r1_fp8_single.json" \
  --html="$HTML" \
  --self-contained-html \
  --log-file="$LOG" \
  -vvv -s

echo "HTML: $HTML"
echo "LOG:  $LOG"
```
