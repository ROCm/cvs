# Host-mounted vLLM server scripts (InferenceMax / VllmJob)

These scripts are meant to be copied (or bind-mounted) onto GPU nodes under the path
configured as `benchmark_server_script_path` in your inference JSON, for example:

`$HOME/benchmark_server_scripts/`

## GPT-OSS 120B on MI300X (`gptoss_fp4_mi300x.sh`)

- **`fixed_seq_len/gptoss_fp4_mi300x.sh`** — server-only `vllm serve` wrapper used with CVS
  (`InferenceMaxJob` with `use_host_mounted_server_script: true`). Defaults to
  **`--enforce-eager`** (via `VLLM_ENFORCE_EAGER`, default `1`) to avoid HIP launch
  resource errors during CUDA graph capture on some stacks. Set `VLLM_ENFORCE_EAGER=0`
  in the container `env_dict` if you need full graph capture performance.
  The script reads `VLLM_ENFORCE_EAGER` once, then **unsets** it before `exec vllm` so vLLM does
  not warn that `VLLM_ENFORCE_EAGER` is an unknown variable (only `--enforce-eager` is official).
- **`gptoss_fp4_mi300.sh`** — thin wrapper that runs the `fixed_seq_len` script.

Deploy the whole `benchmark_server_scripts/` tree so both paths exist. Example:

```bash
rsync -a cvs/input/benchmark_server_scripts/ "${GPU_HOST}:~/benchmark_server_scripts/"
```

Then in JSON: `use_host_mounted_server_script: true`, `benchmark_server_script_path` pointing
at that directory on the node (inside the container if you use the same bind mount as the host),
and `server_script` set to `fixed_seq_len/gptoss_fp4_mi300x.sh` or `gptoss_fp4_mi300.sh`.
