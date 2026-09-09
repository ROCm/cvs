# Megatron configs

JSON configs and sibling `*_threshold.json` files for `megatron_single` / `megatron_distributed`.

MI300X and MI325X share `mi3xx_megatron_<model>_{single,distributed}.json`. Set `gpu_name` to `MI300X` or `MI325X` and `threshold_json` to the SKU-specific `mi300x_*_threshold.json` or `mi325x_*_threshold.json`. MI355X keeps `mi355x_*` files. Do not pass leftover `mi3xx_megatron_llama_{single,distributed}.json` or `mi35x_megatron_llama_single.json` to the unified suites; those nested files are for `megatron_llama3_1_*` only.

`container.env` NIC fields ship with example values plus `<changeme>`. Do not set `NNODES` in JSON.

Sweep combination keys (and matching `sweep.runs` entries and threshold cells) are `MBS=<micro_batch_size>,GBS=<global_batch_size>,PRECISION=<precision>`.

- Schema: [docs/reference/configuration-files/training/megatron.rst](../../../../../docs/reference/configuration-files/training/megatron.rst)
- How to run: [docs/how-to/test-suites/training/megatron.rst](../../../../../docs/how-to/test-suites/training/megatron.rst)
