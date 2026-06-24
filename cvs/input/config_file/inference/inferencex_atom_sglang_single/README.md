# InferenceX SGLang parity (`inferencex_atom_sglang_single`)

See `../inferencex_atom_vllm_single/README.md` for the full M4 parity overview (W1 triple + GPT-OSS uplift on the vLLM suite).

W1 variant: `deepseek_r1_fp8_mi300x_atom_sglang_perf` — same sweep cells as `deepseek_r1_fp8_mi300x_atom_perf`.

```bash
cvs run inferencex_atom_sglang_single \
  --cluster_file ~/input/cluster_file/mi300x_atom_single.json \
  --config_file ~/input/config_file/inference/inferencex_atom_sglang_single/deepseek_r1_fp8_mi300x_atom_sglang_perf/deepseek_r1_fp8_mi300x_atom_sglang_perf_config.json \
  --html=/tmp/inferencex_atom_sglang_w1_mi300x.html -vvv -s
```
