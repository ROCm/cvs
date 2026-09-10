# ATOM inference configs

JSON variant and threshold files for the ``atom`` suite. Full documentation:

- **Configuration reference:** `docs/reference/configuration-files/inference/atom.rst`
- **How-to run:** `docs/how-to/test-suites/inference/atom.rst`

## Config inventory

Configs are lab-validated unless a row is marked pending.

| Stem | Files | Driver |
|------|-------|--------|
| `mi3xx_atom_deepseek-r1_fp8` | `_single` (profiles: `perf`, `mtp3`) | native `atom` |
| `mi3xx_atom_qwen3.5-397b-a17b_fp8` | `_single` (profiles: `perf`, `mtp3`) | native `atom` |
| `mi3xx_atom_vllm_deepseek-r1_fp8` | `_single`, `_distributed` | `vllm_atom` (serving schema; distributed uses PP=2) |
| `mi3xx_atom_vllm_gpt-oss-120b_mxfp4` | `_single` | `vllm_atom` (serving schema) |
| `mi3xx_atom_vllm_qwen3.5-397b-a17b_fp8` | `_single`, `_distributed` | `vllm_atom` (serving schema; lab pending) |
| `mi3xx_atom_sglang_deepseek-r1_fp8` | `_single`, `_distributed` | `sglang` (serving schema) |
| `mi3xx_atom_sglang_qwen3.5-397b-a17b_fp8` | `_single`, `_distributed` | `sglang` (serving schema; lab pending) |

Parity configs (`atom_vllm`, `atom_sglang`) use the unified serving schema:
`server_params`, `benchmark_params`, `sweeps`, `runs`. Run with ``cvs run atom``.

Config stems use the **family** prefix ``mi3xx``. Threshold files use the **platform**
prefix ``mi325x``. Copy each config + its ``threshold_json`` into a dedicated
subdirectory before running.
