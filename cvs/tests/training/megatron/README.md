# Megatron training suite

Unified suites `megatron_single` and `megatron_distributed` run Megatron-LM or Primus pre-training. If `container.image` contains `primus`, the suite uses Primus; otherwise Megatron-LM. Use a `*_single.json` config with `megatron_single` and a `*_distributed.json` config with `megatron_distributed`. MI300X and MI325X share `mi3xx_megatron_<model>_{single,distributed}.json`; set `gpu_name` (`MI300X` / `MI325X` / `MI355X` only) and `threshold_json` to the SKU-specific `mi300x_*` or `mi325x_*` threshold file. MI355X packaged templates are Llama 3.1 8B and Llama 3.3 70B single-node only. `NNODES` is not a JSON field. Distributed runs need a shared `paths.data_cache_dir`. Llama 3.1 8B, Llama 3.3 70B, and DeepSeek V2 Lite support both backends; Llama 3.1 405B is Primus-only (distributed).

- How to run: [docs/how-to/test-suites/training/megatron.rst](../../../../docs/how-to/test-suites/training/megatron.rst)
- Schema: [docs/reference/configuration-files/training/megatron.rst](../../../../docs/reference/configuration-files/training/megatron.rst)
