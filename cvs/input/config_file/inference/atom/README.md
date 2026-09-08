# ATOM inference configs

JSON variant and threshold files for the ``atom`` suite. Full documentation:

- **Configuration reference:** `docs/reference/configuration-files/inference/atom.rst`
- **How to run:** `docs/how-to/test-suites/inference/atom.rst`

## Shipped inventory (lab-validated)

| Stem | Files | Driver |
|------|-------|--------|
| `mi3xx_atom_deepseek-r1_fp8` | `_single` (profiles: `perf`, `mtp3`), `_vllm_single`, `_sglang_single`, `_distributed`, `_distributed_sglang` | `atom`, `vllm_atom`, or `sglang` |
| `mi3xx_atom_qwen3.5-397b-a17b_fp8` | `_single` | `atom` |

vLLM and SGLang parity configs are standalone flat JSON files — not profiles inside the native ATOM stems.

Config stems use the **family** prefix ``mi3xx``. Threshold files use the **platform** prefix ``mi325x`` (lab-validated on MI325X / gfx942). Copy each config + its ``threshold_json`` into a dedicated subdirectory before running (see how-to doc).
