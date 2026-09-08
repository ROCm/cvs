# ATOM inference configs

JSON variant and threshold files for the ``atom`` suite. Full documentation:

- **Configuration reference:** `docs/reference/configuration-files/inference/atom.rst`
- **How-to run:** `docs/how-to/test-suites/inference/atom.rst`

## Shipped inventory (lab-validated)

| Stem | Files | Driver |
|------|-------|--------|
| `mi3xx_atom_deepseek-r1_fp8` | `_single` (profiles: `perf`, `mtp3`), `_distributed` (`vllm_atom` PP=2) | native `atom` / `vllm_atom` |
| `mi3xx_atom_qwen3.5-397b-a17b_fp8` | `_single` | native `atom` |
| `mi3xx_atom_vllm_deepseek-r1_fp8` | `_single` | `vllm_atom` (M4 parity, serving schema) |
| `mi3xx_atom_sglang_deepseek-r1_fp8` | `_single`, `_distributed` | `sglang` (M4/M5 parity, serving schema) |

Parity configs (`atom_vllm`, `atom_sglang`) use the unified serving schema:
`server_params`, `benchmark_params`, `sweeps`, `runs`. Run with ``cvs run atom``.

Config stems use the **family** prefix ``mi3xx``. Threshold files use the **platform**
prefix ``mi325x``. Copy each config + its ``threshold_json`` into a dedicated
subdirectory before running.
