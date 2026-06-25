# InferenceX ATOM single-node variants

W1 **DeepSeek R1 FP8** on 8× GPU, ISL=OSL=1024, TP8.

## Layout

**In the CVS repo**, all variants live as flat sibling pairs in **this directory**:

```text
{gpu}_inferencex-atom-single_{model}_{precision}[_{mode}]_config.json
{gpu}_inferencex-atom-single_{model}_{precision}[_{mode}]_threshold.json
```

**On your lab machine** (`~/input/config_file/inference/inferencex_atom_single/`), copy each variant into its **own subdirectory** so only one `*threshold.json` sits next to the config you pass to `--config_file`. `substitute_config` globs the config's parent directory; multiple `*threshold.json` files there raises `ValueError: multiple *threshold.json files … (ambiguous)`.

```text
~/input/.../inferencex_atom_single/smoke/   # smoke config + smoke threshold only
~/input/.../inferencex_atom_single/perf/    # perf config + perf threshold only
```

Alternatively set `"threshold_json"` in the config to an explicit path (vLLM-style).

Legacy nested layouts (`deepseek_r1_fp8_mi300x_atom_perf/`, `inferencemax/`, etc.) are **removed** from the repo tree. Use only the flat stems below.

**Config filename example:** `mi300x_inferencex-atom-single_deepseek-r1_fp8_perf_config.json`

| Variant | IX recipe | GPU | Notes |
|---------|-----------|-----|-------|
| `mi300x_inferencex-atom-single_deepseek-r1_fp8_smoke` | `dsr1-fp8-mi300x-atom` | MI300X | Quick path check (C=128, 128 prompts) |
| `mi300x_inferencex-atom-single_deepseek-r1_fp8_perf` | `dsr1-fp8-mi300x-atom` | MI300X | W1 perf, calibrated thresholds, server reuse across sweep |
| `mi300x_inferencex-atom-single_deepseek-r1_fp8_mtp3` | `dsr1-fp8-mi300x-atom-mtp3` | MI300X | W1 FP8+MTP3 |
| `mi355x_inferencex-atom-single_deepseek-r1_fp8_perf` | `dsr1-fp8-mi355x-atom` | MI355X | W1 perf (CI seeds, `enforce_thresholds: false`) |
| `mi355x_inferencex-atom-single_deepseek-r1_fp8_mtp3` | `dsr1-fp8-mi355x-atom-mtp3` | MI355X | W1 FP8+MTP3 |
| `mi300x_inferencex-atom-single_gpt-oss-120b_bf16` | — | MI300X | GPT-OSS uplift placeholder (`driver: vllm`) |
| `mi355x_inferencex-atom-single_gpt-oss-120b_bf16` | — | MI355X | GPT-OSS uplift placeholder |

Recipe CLI fragments: `ix_recipes.json`.

## Cluster + container naming

Use `cvs/input/cluster_file/mi300x_atom_single.json` or `mi355x_atom_single.json`. The cluster `container.name` must match the variant (`inferencex_atom_mi300x` / `inferencex_atom_mi355x`); the suite deep-merges variant container settings over the cluster file.

## Shared suite helpers (reusable by other inference suites)

| Module | Purpose |
|--------|---------|
| `cvs/lib/inference/inference_suite_lifecycle.py` | Lifecycle stage tests, `InferenceLifecycle`, pytest HTML hooks |
| `cvs/lib/inference/inference_suite_results_table.py` | Configurable results table (`make_print_results_table`) |
| `cvs/lib/inference/unittests/fake_orch.py` | `FakeOrch` for Job parse unit tests |

`inferencex_atom_single` imports these today; `vllm_single` may adopt them in a follow-up without duplicating code.

## Pytest layout

1. `test_launch_container` → `test_setup_sshd` → `test_model_fetch`
2. `test_inferencex_atom_inference` (per sweep cell; reuses server when `reuse_server_across_sweep: true`)
3. `test_cell_metrics` (one HTML row per **metric tier** per cell: throughput, ttft, tpot, health, record)
4. `test_print_results_table` → `test_teardown`

W1 MI300X perf with two concurrency cells expects **~17** pytest rows (not one row per scalar metric).

## Before the first lab run

- `git checkout` the branch under test, then `make install` and `source .cvs_venv/bin/activate` on the **launcher** host.
- Edit `~/input/cluster_file/mi300x_atom_single.json` (or `mi355x_atom_single.json`): node IPs, `username`, `priv_key_file`, `container.image`.
- **Launcher vs GPU node:** CVS pytest runs on the launcher; `ContainerOrchestrator` SSHes to cluster nodes and runs `sudo docker` there. Local Docker on the launcher is not used. Prerequisites split by host:

  | Item | Launcher | GPU node (cluster `mgmt_ip`) |
  |------|----------|------------------------------|
  | `cvs run`, venv, `~/input/`, `~/cvs_results/` | Yes | No |
  | `priv_key_file`, `~/.hf_token` (read locally by pytest) | Yes | No |
  | `~/models` (when `model.remote: 0`) | No | Yes |
  | `rocm/atom-dev` image, `sudo docker` | No | Yes |
  | `~/LOGS/` (server/bench logs via volume mount) | No | Yes |

- Preflight from the launcher: `ssh -i ~/.ssh/<key> <user>@<mgmt_ip> 'sudo docker images | grep atom-dev; du -sh ~/models'`

## Smoke (MI300X)

One cell, 128 prompts — run this before the full perf matrix.

```bash
cd ~/cvs && source .cvs_venv/bin/activate
mkdir -p ~/cvs_results ~/input/cluster_file

SMOKE_DIR=~/input/config_file/inference/inferencex_atom_single/smoke
mkdir -p "$SMOKE_DIR"

cvs copy-config inference/inferencex_atom_single/mi300x_inferencex-atom-single_deepseek-r1_fp8_smoke_config.json \
  --output "$SMOKE_DIR/mi300x_inferencex-atom-single_deepseek-r1_fp8_smoke_config.json"
cvs copy-config inference/inferencex_atom_single/mi300x_inferencex-atom-single_deepseek-r1_fp8_smoke_threshold.json \
  --output "$SMOKE_DIR/mi300x_inferencex-atom-single_deepseek-r1_fp8_smoke_threshold.json"
cvs copy-config mi300x_atom_single.json --output ~/input/cluster_file/mi300x_atom_single.json

TS=$(date +%Y%m%d_%H%M%S)
HTML=~/cvs_results/${TS}_ix-atom-smoke_mi300x.html
LOG=~/cvs_results/${TS}_ix-atom-smoke_mi300x.log

cvs run inferencex_atom_single \
  --cluster_file ~/input/cluster_file/mi300x_atom_single.json \
  --config_file "$SMOKE_DIR/mi300x_inferencex-atom-single_deepseek-r1_fp8_smoke_config.json" \
  --html="$HTML" \
  --self-contained-html \
  --log-file="$LOG" \
  -vvv -s

echo "HTML: $HTML"
echo "LOG:  $LOG"
```

## W1 perf (MI300X)

Two concurrency cells (C=128, C=256), 1000 prompts. Second cell reuses the ATOM server when `reuse_server_across_sweep: true`.

```bash
cd ~/cvs && source .cvs_venv/bin/activate

PERF_DIR=~/input/config_file/inference/inferencex_atom_single/perf
mkdir -p "$PERF_DIR"

cvs copy-config inference/inferencex_atom_single/mi300x_inferencex-atom-single_deepseek-r1_fp8_perf_config.json \
  --output "$PERF_DIR/mi300x_inferencex-atom-single_deepseek-r1_fp8_perf_config.json"
cvs copy-config inference/inferencex_atom_single/mi300x_inferencex-atom-single_deepseek-r1_fp8_perf_threshold.json \
  --output "$PERF_DIR/mi300x_inferencex-atom-single_deepseek-r1_fp8_perf_threshold.json"

TS=$(date +%Y%m%d_%H%M%S)
HTML=~/cvs_results/${TS}_ix-atom-w1-perf_mi300x.html
LOG=~/cvs_results/${TS}_ix-atom-w1-perf_mi300x.log

cvs run inferencex_atom_single \
  --cluster_file ~/input/cluster_file/mi300x_atom_single.json \
  --config_file "$PERF_DIR/mi300x_inferencex-atom-single_deepseek-r1_fp8_perf_config.json" \
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
cd ~/cvs && source .cvs_venv/bin/activate

PERF_DIR=~/input/config_file/inference/inferencex_atom_single/mi355x_perf
mkdir -p "$PERF_DIR"

cvs copy-config inference/inferencex_atom_single/mi355x_inferencex-atom-single_deepseek-r1_fp8_perf_config.json \
  --output "$PERF_DIR/mi355x_inferencex-atom-single_deepseek-r1_fp8_perf_config.json"
cvs copy-config inference/inferencex_atom_single/mi355x_inferencex-atom-single_deepseek-r1_fp8_perf_threshold.json \
  --output "$PERF_DIR/mi355x_inferencex-atom-single_deepseek-r1_fp8_perf_threshold.json"
cvs copy-config mi355x_atom_single.json --output ~/input/cluster_file/mi355x_atom_single.json

TS=$(date +%Y%m%d_%H%M%S)
HTML=~/cvs_results/${TS}_ix-atom-w1-perf_mi355x.html
LOG=~/cvs_results/${TS}_ix-atom-w1-perf_mi355x.log

cvs run inferencex_atom_single \
  --cluster_file ~/input/cluster_file/mi355x_atom_single.json \
  --config_file "$PERF_DIR/mi355x_inferencex-atom-single_deepseek-r1_fp8_perf_config.json" \
  --html="$HTML" \
  --self-contained-html \
  --log-file="$LOG" \
  -vvv -s

echo "HTML: $HTML"
echo "LOG:  $LOG"
```
