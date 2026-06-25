# InferenceX ATOM single-node variants

W1 **DeepSeek R1 FP8** (`deepseek-ai/DeepSeek-R1-0528`), ISL=OSL=1024, TP8, FP8 KV cache.

Config layout matches `vllm_single`: flat `*_config.json` + sibling `*_threshold.json` in this directory.

**Filename pattern** (same as `vllm_single`):

```text
{gpu}_{framework}_{model}_{precision}[_{mode}]_config.json
```

Example: `mi300x_inferencex-atom-single_deepseek-r1_fp8_perf_config.json` (framework `inferencex_atom_single` → `inferencex-atom-single` in the stem).

| Config stem | IX recipe id | Arch | Mode |
|-------------|--------------|------|------|
| `mi300x_inferencex-atom-single_deepseek-r1_fp8_perf` | `dsr1-fp8-mi300x-atom` | MI300X | FP8 |
| `mi355x_inferencex-atom-single_deepseek-r1_fp8_perf` | `dsr1-fp8-mi355x-atom` | MI355X | FP8 |
| `mi300x_inferencex-atom-single_deepseek-r1_fp8_mtp3` | `dsr1-fp8-mi300x-atom-mtp3` | MI300X | FP8+MTP3 |
| `mi355x_inferencex-atom-single_deepseek-r1_fp8_mtp3` | `dsr1-fp8-mi355x-atom-mtp3` | MI355X | FP8+MTP3 |
| `mi300x_inferencex-atom-single_deepseek-r1_fp8_smoke` | `dsr1-fp8-mi300x-atom` | MI300X | FP8 smoke (C=128, 128 prompts) |
| `mi300x_inferencex-atom-single_gpt-oss-120b_bf16` | — | MI300X | GPT-OSS uplift (vLLM driver) |
| `mi355x_inferencex-atom-single_gpt-oss-120b_bf16` | — | MI355X | GPT-OSS uplift (vLLM driver) |

Recipe CLI fragments live in `ix_recipes.json` (pinned to ROCm/ATOM catalog + IX `amd-master.yaml` ids).

W1 configs pin `container.name` (`inferencex_atom_mi300x` / `inferencex_atom_mi355x`); set `container.image` and cluster node IPs before the first lab run.

**Lab runner:** install the branch under test (`make install` from repo root) so parser/threshold fixes are not shadowed by an older `site-packages/cvs` install.

## First lab run (smoke — shorter)

Use this before the full W1 perf matrix. One sweep cell (ISL=OSL=1024, C=128) with `num_prompts=128` instead of 1000. Server warmup still runs (`num_warmups = conc × 2` in ATOM bench).

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

After smoke passes, run `mi300x_inferencex-atom-single_deepseek-r1_fp8_perf` for calibration (C=128 + C=256, 1000 prompts).

## Copy configs to your host (full W1 perf)

```bash
cvs copy-config inference/inferencex_atom_single/mi300x_inferencex-atom-single_deepseek-r1_fp8_perf_config.json \
  --output ~/input/config_file/inference/inferencex_atom_single/mi300x_inferencex-atom-single_deepseek-r1_fp8_perf_config.json
cvs copy-config inference/inferencex_atom_single/mi300x_inferencex-atom-single_deepseek-r1_fp8_perf_threshold.json \
  --output ~/input/config_file/inference/inferencex_atom_single/mi300x_inferencex-atom-single_deepseek-r1_fp8_perf_threshold.json
cvs copy-config mi300x_atom_single.json --output ~/input/cluster_file/mi300x_atom_single.json
```

## Run (MI300X W1 FP8 example)

```bash
cvs run inferencex_atom_single \
  --cluster_file ~/input/cluster_file/mi300x_atom_single.json \
  --config_file ~/input/config_file/inference/inferencex_atom_single/mi300x_inferencex-atom-single_deepseek-r1_fp8_perf_config.json \
  --html=/tmp/inferencex_atom_w1_mi300x.html -vvv -s
```

## MI355X W1 (no local lab — ATOM CI seeds)

Thresholds in `mi355x_inferencex-atom-single_deepseek-r1_fp8_perf_threshold.json` and `mi355x_inferencex-atom-single_deepseek-r1_fp8_mtp3_threshold.json` are calibrated from [ROCm/ATOM run 27912164002](https://github.com/ROCm/ATOM/actions/runs/27912164002) (plan Section 4.3) with the same 10% margin as MI300X lab gates. `enforce_thresholds` stays `false` until an MI355X lab run confirms.

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

| Cell | Source (ATOM CI) | Gated output min (tok/s) | Gated mean TTFT max (ms) | Gated mean TPOT max (ms) |
|------|------------------|--------------------------|--------------------------|--------------------------|
| C=128 FP8 | 4449.62 / 329.25 / 27.64 | 4004.66 | 362.18 | 30.40 |
| C=256 FP8 | 6249.73 / 551.66 / 39.46 | 5624.76 | 606.83 | 43.41 |
| C=128 MTP3 | 5101.99 / 570.42 / 23.77 | 4591.79 | 627.46 | 26.15 |
| C=256 MTP3 | 7168.43 / 606.67 / 34.22 | 6451.59 | 667.34 | 37.64 |
