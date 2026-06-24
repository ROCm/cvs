# InferenceX parity frameworks (M4)

ROCm **vLLM** and **SGLang** baseline engines on the **same workload cards** as `inferencex_atom_single` (ATOM). Each framework has its own suite id, config tree, and independently calibrated thresholds.

| Framework | Suite id | Engine |
|-----------|----------|--------|
| ATOM | `inferencex_atom_single` | `atom.entrypoints.openai_server` |
| vLLM parity | `inferencex_atom_vllm_single` | `vllm serve` + `vllm bench serve` |
| SGLang parity | `inferencex_atom_sglang_single` | `sglang.launch_server` + `sglang.bench_serving` |

## W1 MI300X parity triple

| ATOM reference | vLLM sibling | SGLang sibling |
|----------------|--------------|----------------|
| `deepseek_r1_fp8_mi300x_atom_perf` | `deepseek_r1_fp8_mi300x_atom_vllm_perf` | `deepseek_r1_fp8_mi300x_atom_sglang_perf` |

Same sweep: ISL=OSL=1024, TP=8, CONC=128 and 256, 1000 prompts. **Separate** `container.image` per engine (ATOM / vLLM / SGLang ROCm images).

## GPT-OSS 120B uplift (still supported)

| Config path | Suite |
|-------------|-------|
| `inferencex_atom_single/mi300x_gpt_oss_120b_single/` | ATOM suite with `params.driver=vllm` (interim uplift) |
| `inferencex_atom_vllm_single/mi300x_gpt_oss_120b_single/` | Dedicated vLLM parity suite (preferred for uplift) |

Sweep: ISL=7168, OSL=1024, CONC=64, TP=8. `enforce_thresholds: false` until W2 ATOM lands.

## Lab runner

From repo root: **`make install`** then use `.cvs_venv/bin/cvs`.

### vLLM W1 parity example

```bash
cvs copy-config inference/inferencex_atom_vllm_single/deepseek_r1_fp8_mi300x_atom_vllm_perf/deepseek_r1_fp8_mi300x_atom_vllm_perf_config.json \
  --output ~/input/config_file/inference/inferencex_atom_vllm_single/deepseek_r1_fp8_mi300x_atom_vllm_perf/deepseek_r1_fp8_mi300x_atom_vllm_perf_config.json
cvs copy-config inference/inferencex_atom_vllm_single/deepseek_r1_fp8_mi300x_atom_vllm_perf/deepseek_r1_fp8_mi300x_atom_vllm_perf_threshold.json \
  --output ~/input/config_file/inference/inferencex_atom_vllm_single/deepseek_r1_fp8_mi300x_atom_vllm_perf/deepseek_r1_fp8_mi300x_atom_vllm_perf_threshold.json

cvs run inferencex_atom_vllm_single \
  --cluster_file ~/input/cluster_file/mi300x_atom_single.json \
  --config_file ~/input/config_file/inference/inferencex_atom_vllm_single/deepseek_r1_fp8_mi300x_atom_vllm_perf/deepseek_r1_fp8_mi300x_atom_vllm_perf_config.json \
  --html=/tmp/inferencex_atom_vllm_w1_mi300x.html -vvv -s
```

### GPT-OSS 120B (vLLM parity suite)

```bash
cvs run inferencex_atom_vllm_single \
  --cluster_file ~/input/cluster_file/mi300x_atom_single.json \
  --config_file ~/input/config_file/inference/inferencex_atom_vllm_single/mi300x_gpt_oss_120b_single/mi300x_gpt_oss_120b_single_config.json \
  --html=/tmp/gpt_oss_120b_vllm.html -vvv -s
```

### SGLang W1 parity example

```bash
cvs run inferencex_atom_sglang_single \
  --cluster_file ~/input/cluster_file/mi300x_atom_single.json \
  --config_file ~/input/config_file/inference/inferencex_atom_sglang_single/deepseek_r1_fp8_mi300x_atom_sglang_perf/deepseek_r1_fp8_mi300x_atom_sglang_perf_config.json \
  --html=/tmp/inferencex_atom_sglang_w1_mi300x.html -vvv -s
```

Set `container.image` in each config before the first lab run. Thresholds ship **record-only** (`enforce_thresholds: false`) until per-engine lab calibration.
