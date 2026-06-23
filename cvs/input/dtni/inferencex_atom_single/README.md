# InferenceX ATOM single-node variants (DTNI layout)

W1 **DeepSeek R1 FP8** (`deepseek-ai/DeepSeek-R1-0528`), ISL=OSL=1024, TP8, FP8 KV cache.

| Variant dir | IX recipe id | Arch | Mode |
|-------------|--------------|------|------|
| `deepseek_r1_fp8_mi300x_atom_perf` | `dsr1-fp8-mi300x-atom` | MI300X | FP8 |
| `deepseek_r1_fp8_mi355x_atom_perf` | `dsr1-fp8-mi355x-atom` | MI355X | FP8 |
| `deepseek_r1_fp8_mi300x_atom_mtp3` | `dsr1-fp8-mi300x-atom-mtp3` | MI300X | FP8+MTP3 |
| `deepseek_r1_fp8_mi355x_atom_mtp3` | `dsr1-fp8-mi355x-atom-mtp3` | MI355X | FP8+MTP3 |

Recipe CLI fragments live in `ix_recipes.json` (pinned to ROCm/ATOM catalog + IX `amd-master.yaml` ids).

Each subdirectory has `config.json` + `threshold.json`. Set `container.image` / `container.name` and cluster node IPs before the first lab run.

## Copy configs to your host

```bash
cvs copy-config inferencex_atom_single/deepseek_r1_fp8_mi300x_atom_perf/config.json \
  --output ~/input/dtni/deepseek_r1_fp8_mi300x_atom_perf/config.json
cvs copy-config inferencex_atom_single/deepseek_r1_fp8_mi300x_atom_perf/threshold.json \
  --output ~/input/dtni/deepseek_r1_fp8_mi300x_atom_perf/threshold.json
cvs copy-config mi300x_atom_single.json --output ~/input/cluster_file/mi300x_atom_single.json
```

## Run (MI300X W1 FP8 example)

```bash
cvs run inferencex_atom_single \
  --cluster_file ~/input/cluster_file/mi300x_atom_single.json \
  --config_file ~/input/dtni/deepseek_r1_fp8_mi300x_atom_perf/config.json \
  --html=/tmp/inferencex_atom_w1_mi300x.html -vvv -s
```

MI355X: use `mi355x_atom_single.json` and `deepseek_r1_fp8_mi355x_atom_perf/config.json`.
