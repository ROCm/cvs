# Megatron configs

JSON configs and sibling `*_threshold.json` files for `megatron_single` / `megatron_distributed`.

MI300X and MI325X share `mi3xx_megatron_<model>_{single,distributed}.json`. Set `gpu_name` to `MI300X` or `MI325X` and `threshold_json` to the SKU-specific `mi300x_*_threshold.json` or `mi325x_*_threshold.json`. MI355X keeps `mi355x_*` files. Do not pass leftover `mi3xx_megatron_llama_{single,distributed}.json` or `mi35x_megatron_llama_single.json` to the unified suites; those nested files are for `megatron_llama3_1_*` only.

Llama 3.1 8B, Llama 3.3 70B, and DeepSeek V2 Lite support both Megatron-LM and Primus (same JSON; backend is `container.image`). Llama 3.1 405B (`*_llama-3.1-405b_distributed.json`) is distributed-only and Primus-only.

`container.env` NIC fields ship with example values plus `<changeme>`. Do not set `NNODES` in JSON.

Sweep combination keys (and matching `sweep.runs` entries and threshold cells) are `MBS=<micro_batch_size>,GBS=<global_batch_size>,PRECISION=<precision>`. Pytest parametrizes `sweep_name` from those keys. Packaged combination bodies set `"training_iterations": "20"`; other `train_params` overlays are optional. Omitting `sweep` runs one implicit `default` cell from `train_params`; the threshold file then needs a `"default"` cell when `enforce_thresholds` is true (optional when it is false).

- Schema: [docs/reference/configuration-files/training/megatron.rst](../../../../../docs/reference/configuration-files/training/megatron.rst)
- How to run: [docs/how-to/test-suites/training/megatron.rst](../../../../../docs/how-to/test-suites/training/megatron.rst)
