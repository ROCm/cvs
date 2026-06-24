# InferenceX ATOM single-node variants

W1 **DeepSeek R1 FP8** (`deepseek-ai/DeepSeek-R1-0528`), ISL=OSL=1024, TP8, FP8 KV cache.

| Variant dir | IX recipe id | Arch | Mode |
|-------------|--------------|------|------|
| `deepseek_r1_fp8_mi300x_atom_perf` | `dsr1-fp8-mi300x-atom` | MI300X | FP8 |
| `deepseek_r1_fp8_mi355x_atom_perf` | `dsr1-fp8-mi355x-atom` | MI355X | FP8 |
| `deepseek_r1_fp8_mi300x_atom_mtp3` | `dsr1-fp8-mi300x-atom-mtp3` | MI300X | FP8+MTP3 |
| `deepseek_r1_fp8_mi355x_atom_mtp3` | `dsr1-fp8-mi355x-atom-mtp3` | MI355X | FP8+MTP3 |
| `deepseek_r1_fp8_mi300x_atom_smoke` | `dsr1-fp8-mi300x-atom` | MI300X | FP8 smoke (C=128, 128 prompts) |

Recipe CLI fragments live in `ix_recipes.json` (pinned to ROCm/ATOM catalog + IX `amd-master.yaml` ids).

Each variant subdirectory has `<variant>_config.json` + `<variant>_threshold.json`. Set `container.image` / `container.name` and cluster node IPs before the first lab run.

## First lab run (smoke — shorter)

Use this before the full W1 perf matrix. One sweep cell (ISL=OSL=1024, C=128) with `num_prompts=128` instead of 1000. Server warmup still runs (`num_warmups = conc × 2` in ATOM bench).

```bash
cvs copy-config inference/inferencex_atom_single/deepseek_r1_fp8_mi300x_atom_smoke/deepseek_r1_fp8_mi300x_atom_smoke_config.json \
  --output ~/input/config_file/inference/inferencex_atom_single/deepseek_r1_fp8_mi300x_atom_smoke/deepseek_r1_fp8_mi300x_atom_smoke_config.json
cvs copy-config inference/inferencex_atom_single/deepseek_r1_fp8_mi300x_atom_smoke/deepseek_r1_fp8_mi300x_atom_smoke_threshold.json \
  --output ~/input/config_file/inference/inferencex_atom_single/deepseek_r1_fp8_mi300x_atom_smoke/deepseek_r1_fp8_mi300x_atom_smoke_threshold.json

cvs run inferencex_atom_single \
  --cluster_file ~/input/cluster_file/mi300x_atom_single.json \
  --config_file ~/input/config_file/inference/inferencex_atom_single/deepseek_r1_fp8_mi300x_atom_smoke/deepseek_r1_fp8_mi300x_atom_smoke_config.json \
  --html=/tmp/inferencex_atom_smoke_mi300x.html -vvv -s
```

After smoke passes, run `deepseek_r1_fp8_mi300x_atom_perf` for calibration (C=128 + C=256, 1000 prompts).

## Copy configs to your host (full W1 perf)

```bash
cvs copy-config inference/inferencex_atom_single/deepseek_r1_fp8_mi300x_atom_perf/deepseek_r1_fp8_mi300x_atom_perf_config.json \
  --output ~/input/config_file/inference/inferencex_atom_single/deepseek_r1_fp8_mi300x_atom_perf/deepseek_r1_fp8_mi300x_atom_perf_config.json
cvs copy-config inference/inferencex_atom_single/deepseek_r1_fp8_mi300x_atom_perf/deepseek_r1_fp8_mi300x_atom_perf_threshold.json \
  --output ~/input/config_file/inference/inferencex_atom_single/deepseek_r1_fp8_mi300x_atom_perf/deepseek_r1_fp8_mi300x_atom_perf_threshold.json
cvs copy-config mi300x_atom_single.json --output ~/input/cluster_file/mi300x_atom_single.json
```

## Run (MI300X W1 FP8 example)

```bash
cvs run inferencex_atom_single \
  --cluster_file ~/input/cluster_file/mi300x_atom_single.json \
  --config_file ~/input/config_file/inference/inferencex_atom_single/deepseek_r1_fp8_mi300x_atom_perf/deepseek_r1_fp8_mi300x_atom_perf_config.json \
  --html=/tmp/inferencex_atom_w1_mi300x.html -vvv -s
```

MI355X: use `mi355x_atom_single.json` and `deepseek_r1_fp8_mi355x_atom_perf/deepseek_r1_fp8_mi355x_atom_perf_config.json`.
