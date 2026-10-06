# ATOM inference configs

JSON variant and threshold files for the ``atom`` suite. Full documentation:

- **Configuration reference:**
  [`docs/reference/configuration-files/inference/atom.rst`](../../../../../docs/reference/configuration-files/inference/atom.rst)
- **How-to run:**
  [`docs/how-to/test-suites/inference/atom.rst`](../../../../../docs/how-to/test-suites/inference/atom.rst)

## Config inventory

Configs are lab-validated unless a row is marked pending.

| Stem | Files | Driver |
|------|-------|--------|
| `mi3xx_atom_deepseek-r1_fp8` | `_single` (profiles: `perf`, `mtp3`), `_distributed` (`vllm_atom` PP=2) | native `atom` / `vllm_atom` |
| `mi3xx_atom_qwen3.5-397b-a17b_fp8` | `_single` (profiles: `perf`, `mtp3`) | native `atom` |
| `mi3xx_atom_vllm_deepseek-r1_fp8` | `_single` | `vllm_atom` (serving schema) |
| `mi3xx_atom_vllm_gpt-oss-120b_mxfp4` | `_single`, `_distributed` | `vllm_atom` (serving schema; distributed is PP=2 at the shipped TP4) |
| `mi3xx_atom_gpt-oss-120b_mxfp4` | `_single` | native `atom` (single GPU; bring-up thresholds) |
| `mi3xx_atom_vllm_glm-5.1_fp8` | `_single`, `_distributed` | `vllm_atom` (no native or SGLang serve recipe; bring-up thresholds) |
| `mi3xx_atom_glm-5.2_fp8` | `_single` (profiles: `perf`, `mtp3`) | native `atom` (MI300X/MI308X TP8; bring-up thresholds) |
| `mi3xx_atom_vllm_glm-5.2_fp8` | `_single`, `_distributed` | `vllm_atom` (bring-up thresholds) |
| `mi3xx_atom_sglang_glm-5.2_fp8` | `_single`, `_distributed` | `sglang` (bring-up thresholds) |
| `mi3xx_atom_glm-5.2_mxfp4` | `_single` (profiles: `perf`, `mtp3`) | native `atom` (MI355 TP4; `gpu_arch` mi355x; bring-up thresholds) |
| `mi3xx_atom_vllm_glm-5.2_mxfp4` | `_single`, `_distributed` | `vllm_atom` (MI355 TP4; bring-up thresholds) |
| `mi3xx_atom_sglang_glm-5.2_mxfp4` | `_single`, `_distributed` | `sglang` (MI355 TP4; bring-up thresholds) |
| `mi3xx_atom_minimax-m3_mxfp4` | `_single` (profiles: `perf`, `eagle3`) | native `atom` (MI355 TP4; bring-up thresholds) |
| `mi3xx_atom_vllm_minimax-m3_mxfp4` | `_single`, `_distributed` | `vllm_atom` (bring-up thresholds) |
| `mi3xx_atom_sglang_minimax-m3_mxfp4` | `_single`, `_distributed` | `sglang` (bring-up thresholds) |
| `mi3xx_atom_deepseek-r1_mxfp4` | `_single` (profiles: `perf`, `mtp3`) | native `atom` (`amd/DeepSeek-R1-0528-MXFP4`; bring-up thresholds) |
| `mi3xx_atom_vllm_deepseek-r1_mxfp4` | `_single`, `_distributed` | `vllm_atom` (`amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4`; bring-up thresholds) |
| `mi3xx_atom_sglang_deepseek-r1_mxfp4` | `_single`, `_distributed` | `sglang` (`amd/DeepSeek-R1-0528-MXFP4-v2`; bring-up thresholds) |
| `mi3xx_atom_vllm_qwen3.5-397b-a17b_mxfp4` | `_single`, `_distributed` | `vllm_atom` (MXFP4 plugin recipe; bring-up thresholds) |
| `mi3xx_atom_vllm_qwen3.5-397b-a17b_fp8` | `_single`, `_distributed` | `vllm_atom` (serving schema; lab pending) |
| `mi3xx_atom_sglang_deepseek-r1_fp8` | `_single`, `_distributed` | `sglang` (serving schema) |
| `mi3xx_atom_sglang_qwen3.5-397b-a17b_fp8` | `_single`, `_distributed` | `sglang` (serving schema; lab pending) |

Parity configs (`atom_vllm`, `atom_sglang`) use the unified serving schema:
`server_params`, `benchmark_params`, `sweeps`, `runs`. Run with ``cvs run atom``.

Config stems use the **family** prefix ``mi3xx``. Threshold files use the **platform**
prefix ``mi325x``. Copy each config + its ``threshold_json`` into a dedicated
subdirectory before running.
