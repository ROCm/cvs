# InferenceX ATOM single-node variants

W1 **DeepSeek R1 FP8** on 8× GPU, ISL=OSL=1024, TP8.

Each variant is a `*_config.json` plus a sibling `*_threshold.json` in this directory.

**Config filename pattern:**

```text
{gpu}_inferencex-atom-single_{model}_{precision}[_{mode}]_config.json
```

Example: `mi300x_inferencex-atom-single_deepseek-r1_fp8_perf_config.json`

| Variant | IX recipe | GPU | Notes |
|---------|-----------|-----|-------|
| `mi300x_inferencex-atom-single_deepseek-r1_fp8_smoke` | `dsr1-fp8-mi300x-atom` | MI300X | Quick path check (C=128, 128 prompts) |
| `mi300x_inferencex-atom-single_deepseek-r1_fp8_perf` | `dsr1-fp8-mi300x-atom` | MI300X | W1 perf, calibrated thresholds |
| `mi300x_inferencex-atom-single_deepseek-r1_fp8_mtp3` | `dsr1-fp8-mi300x-atom-mtp3` | MI300X | W1 FP8+MTP3 |
| `mi355x_inferencex-atom-single_deepseek-r1_fp8_perf` | `dsr1-fp8-mi355x-atom` | MI355X | W1 perf (CI seeds, `enforce_thresholds: false`) |
| `mi355x_inferencex-atom-single_deepseek-r1_fp8_mtp3` | `dsr1-fp8-mi355x-atom-mtp3` | MI355X | W1 FP8+MTP3 |
| `mi300x_inferencex-atom-single_gpt-oss-120b_bf16` | — | MI300X | GPT-OSS uplift placeholder |
| `mi355x_inferencex-atom-single_gpt-oss-120b_bf16` | — | MI355X | GPT-OSS uplift placeholder |

Recipe CLI fragments: `ix_recipes.json`.

Before the first lab run: set `container.image`, cluster node IPs, and model paths. Install the branch under test (`make install`) so the lab uses current parser and threshold code.

## Smoke (MI300X)

One cell, 128 prompts — run this before the full perf matrix.

```bash
cvs copy-config inference/inferencex_atom_single/mi300x_inferencex-atom-single_deepseek-r1_fp8_smoke_config.json \
  --output ~/input/config_file/inference/inferencex_atom_single/mi300x_inferencex-atom-single_deepseek-r1_fp8_smoke_config.json
cvs copy-config inference/inferencex_atom_single/mi300x_inferencex-atom-single_deepseek-r1_fp8_smoke_threshold.json \
  --output ~/input/config_file/inference/inferencex_atom_single/mi300x_inferencex-atom-single_deepseek-r1_fp8_smoke_threshold.json

cvs run inferencex_atom_single \
  --cluster_file ~/input/cluster_file/mi300x_atom_single.json \
  --config_file ~/input/config_file/inference/inferencex_atom_single/mi300x_inferencex-atom-single_deepseek-r1_fp8_smoke_config.json \
  --html=/tmp/inferencex_atom_smoke_mi300x.html -vvv -s
```

## W1 perf (MI300X)

Two concurrency cells (C=128, C=256), 1000 prompts.

```bash
cvs copy-config inference/inferencex_atom_single/mi300x_inferencex-atom-single_deepseek-r1_fp8_perf_config.json \
  --output ~/input/config_file/inference/inferencex_atom_single/mi300x_inferencex-atom-single_deepseek-r1_fp8_perf_config.json
cvs copy-config inference/inferencex_atom_single/mi300x_inferencex-atom-single_deepseek-r1_fp8_perf_threshold.json \
  --output ~/input/config_file/inference/inferencex_atom_single/mi300x_inferencex-atom-single_deepseek-r1_fp8_perf_threshold.json
cvs copy-config mi300x_atom_single.json --output ~/input/cluster_file/mi300x_atom_single.json

cvs run inferencex_atom_single \
  --cluster_file ~/input/cluster_file/mi300x_atom_single.json \
  --config_file ~/input/config_file/inference/inferencex_atom_single/mi300x_inferencex-atom-single_deepseek-r1_fp8_perf_config.json \
  --html=/tmp/inferencex_atom_w1_mi300x.html -vvv -s
```

## W1 perf (MI355X)

Thresholds are seeded from [ROCm/ATOM run 27912164002](https://github.com/ROCm/ATOM/actions/runs/27912164002). `enforce_thresholds` stays `false` until an MI355X lab run confirms.

```bash
cvs copy-config inference/inferencex_atom_single/mi355x_inferencex-atom-single_deepseek-r1_fp8_perf_config.json \
  --output ~/input/config_file/inference/inferencex_atom_single/mi355x_inferencex-atom-single_deepseek-r1_fp8_perf_config.json
cvs copy-config inference/inferencex_atom_single/mi355x_inferencex-atom-single_deepseek-r1_fp8_perf_threshold.json \
  --output ~/input/config_file/inference/inferencex_atom_single/mi355x_inferencex-atom-single_deepseek-r1_fp8_perf_threshold.json
cvs copy-config mi355x_atom_single.json --output ~/input/cluster_file/mi355x_atom_single.json

cvs run inferencex_atom_single \
  --cluster_file ~/input/cluster_file/mi355x_atom_single.json \
  --config_file ~/input/config_file/inference/inferencex_atom_single/mi355x_inferencex-atom-single_deepseek-r1_fp8_perf_config.json \
  --html=/tmp/inferencex_atom_w1_mi355x.html -vvv -s
```
