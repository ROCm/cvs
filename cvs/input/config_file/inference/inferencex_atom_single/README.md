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

Each shipped config sets `"threshold_json"` to the sibling threshold filename (resolved relative to the config directory). You may also use an absolute path (vLLM-style).

Legacy nested layouts (`deepseek_r1_fp8_mi300x_atom_perf/`, `inferencemax/`, etc.) are **removed** from the repo tree. Use only the flat stems below.

**Config filename example:** `mi300x_inferencex-atom-single_deepseek-r1_fp8_perf_config.json`

| Variant | GPU | Notes |
|---------|-----|-------|
| `mi300x_inferencex-atom-single_deepseek-r1_fp8_smoke` | MI300X | Quick path check (C=128, 128 prompts) |
| `mi300x_inferencex-atom-single_deepseek-r1_fp8_perf` | MI300X | W1 perf, portable min-SLO thresholds, server reuse across sweep |
| `mi300x_inferencex-atom-single_deepseek-r1_fp8_mtp3` | MI300X | W1 FP8+MTP3 |
| `mi355x_inferencex-atom-single_deepseek-r1_fp8_perf` | MI355X | W1 perf (CI seeds, `enforce_thresholds: false`) |
| `mi355x_inferencex-atom-single_deepseek-r1_fp8_mtp3` | MI355X | W1 FP8+MTP3 |
| `mi300x_inferencex-atom-single_gpt-oss-120b_bf16` | MI300X | GPT-OSS uplift placeholder (`driver: vllm`, inline `serve_args`) |
| `mi355x_inferencex-atom-single_gpt-oss-120b_bf16` | MI355X | GPT-OSS uplift placeholder |
| `mi300x_inferencex-atom-single_kimi-k2.6_bf16_smoke` | MI300X | Kimi K2.6 BF16 — `/mnt/dtni/models/Kimi-K2.6`; 3 ISL/OSL combos × C=4,8,16,32,64 |

ATOM server CLI is inline in each config under `roles.server.atom_args` (vLLM-style, same as `roles.server.serve_args` on `vllm_single`). MTP3 variants also set `params.bench_extra_args`.

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

When `--html` is set, the **IX Run Deck** (`inferencex_atom_run_deck.html`, `.json`,
`_viewer.html`) is generated at session end and bundled into the pytest zip.
See `cvs/lib/report/README.md` for wiring other suites. Open the pytest HTML **Reports**
section for links. Render-only; does not affect gates.

## Kimi K2.6 smoke (MI300X, DTNI lab)

Local checkpoint at `/mnt/dtni/models/Kimi-K2.6`. The variant config mounts `/mnt/dtni/models`
into the container. **15 sweep cells** — three ISL/OSL pairs (`1024/1024`, `1024/8192`,
`8192/1024`) × concurrency `4, 8, 16, 32, 64`, server reused across cells — expect
**~95** pytest rows with `--html` (IX Run Deck included). Long run; plan several hours.

**Preflight (GPU node must have non-zero model dir):**

```bash
ssh -i ~/.ssh/<key> <user>@<gpu-node> \
  'du -sh /mnt/dtni/models/Kimi-K2.6; ls /mnt/dtni/models/Kimi-K2.6/config.json; sudo docker images | grep atom-dev'
```

`du` must not show `0`. If empty, stage the checkpoint on the node before running.

If the server log shows `load model runner failed` / `ModelRunner.*proc died`, check
`~/LOGS/.../atom_server.log` on the GPU node. Common causes: empty model path, missing
`tiktoken`, or `max_model_length` too small for the sweep (this variant uses
`max_model_length=16896` with `random_range_ratio=0` for ISL+OSL up to 8192+8192).

**`ValueError: Unsupported quant dtype: torch.bfloat16`** — the decompressed BF16
checkpoint at `/mnt/dtni/models/Kimi-K2.6` still carries a Quark `quantization_config`
where `weight.dtype` is `bfloat16` (Quark’s “do not quantize” spec). `rocm/atom-dev:latest`
mis-reads that as a quantized dtype and aborts during weight load. One-time fix on the GPU
node (back up first, then remove the metadata block; weights stay BF16 safetensors):

```bash
ssh -i ~/.ssh/<key> <user>@<gpu-node> 'python3 - <<'"'"'PY'"'"'
import json, shutil
from pathlib import Path
p = Path("/mnt/dtni/models/Kimi-K2.6/config.json")
cfg = json.loads(p.read_text())
qc = cfg.get("quantization_config")
if not qc:
    print("no quantization_config — nothing to do")
    raise SystemExit(0)
bak = p.with_suffix(".json.bak-cvs")
if not bak.exists():
    shutil.copy2(p, bak)
    print("backup:", bak)
cfg.pop("quantization_config", None)
p.write_text(json.dumps(cfg, indent=2) + "\n")
print("removed quantization_config from", p)
PY'
```

Re-run after that. Long-term fix belongs in ATOM (`QuarkParser` should treat
`bfloat16`/`float16` weight dtype as `QuantType.No`). Do **not** point this variant at
`Kimi-K2.6-MXFP4` if the goal is native K2.6 BF16.

Also remove nested `text_config.quantization_config` if present (same Quark metadata
issue; backup `config.json` first, use `sudo` when the tree is owned by another user).

**`HIP out of memory` during `FusedMoE` / `create_weights`** — the ~555G **BF16 safetensor**
tree at `/mnt/dtni/models/Kimi-K2.6` does **not** fit on 8× MI300X (192 GiB) with
`rocm/atom-dev:latest`. smoke8/smoke9 show ~181.7 GiB/GPU allocated before a final
+2.7 GiB `torch.empty` in `FusedMoE.create_weights`; `-tp 8 --enable-expert-parallel`
barely changes usage (~1 GiB). This is a **capacity limit**, not mmap or custom-AR.

**MI300X paths that work today:**

| Path | Model dir | Notes |
|------|-----------|--------|
| Pre-quant MXFP4 (fastest) | `/mnt/dtni/models/Kimi-K2.6-MXFP4` | Proven on this lab node (~521G) |
| Online quant (shipped config) | `/mnt/dtni/models/Kimi-K2.6` | `--online_quant_config` MXFP4 on `*experts*`, `fp8` KV, `ATOM_USE_TRITON_MXFP4_BMM=1` |

Pure BF16 weight residency on TP8 needs an ATOM/load fix or more aggregate VRAM (e.g. MI355X
recipe). vLLM’s Kimi K2.6 MI300X recipe uses **INT4/MXFP4**, not raw BF16 safetensors.

Also omit `ATOM_DISABLE_MMAP`, keep `serve_args.level=0`, one server only (~300 MiB/GPU idle).

**`hipIpcGetMemHandle` / custom all-reduce** — if this reappears, use
`ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION=0` and `--level 0` (already in the Kimi config).
Do **not** set `PYTORCH_ALLOC_CONF=expandable_segments:True` with custom AR on this stack.

```bash
cd ~/cvs && source .cvs_venv/bin/activate
mkdir -p ~/input/cluster_file

KIMI_DIR=~/input/config_file/inference/inferencex_atom_single/kimi_k26_smoke
mkdir -p "$KIMI_DIR"

cvs copy-config inference/inferencex_atom_single/mi300x_inferencex-atom-single_kimi-k2.6_bf16_smoke_config.json \
  --output "$KIMI_DIR/mi300x_inferencex-atom-single_kimi-k2.6_bf16_smoke_config.json" --force
cvs copy-config inference/inferencex_atom_single/mi300x_inferencex-atom-single_kimi-k2.6_bf16_smoke_threshold.json \
  --output "$KIMI_DIR/mi300x_inferencex-atom-single_kimi-k2.6_bf16_smoke_threshold.json" --force
cvs copy-config mi300x_atom_single.json --output ~/input/cluster_file/mi300x_atom_single.json

TS=$(date +%Y%m%d_%H%M%S)
HTML="$HOME/cvs_results/${TS}_kimi-k26-smoke_mi300x.html"
LOG="$HOME/cvs_results/${TS}_kimi-k26-smoke_mi300x.log"

cvs run inferencex_atom_single \
  --cluster_file ~/input/cluster_file/mi300x_atom_single.json \
  --config_file "$KIMI_DIR/mi300x_inferencex-atom-single_kimi-k2.6_bf16_smoke_config.json" \
  --html="$HTML" \
  --self-contained-html \
  --log-file="$LOG" \
  -vvv -s

echo "HTML: $HTML"
echo "LOG:  $LOG"
echo "Run deck: $HOME/cvs_results/inferencex_atom_run_deck.html"
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
