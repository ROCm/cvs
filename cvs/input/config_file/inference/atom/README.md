# ATOM inference configs

JSON variant and threshold files for the ``atom`` suite. Full documentation:

- **Configuration reference:**
  [`docs/reference/configuration-files/inference/atom.rst`](../../../../../docs/reference/configuration-files/inference/atom.rst)
- **How-to run:**
  [`docs/how-to/test-suites/inference/atom.rst`](../../../../../docs/how-to/test-suites/inference/atom.rst)

## SKU split

| GPU | Native ATOM gate | Notes |
|-----|------------------|--------|
| MI300X / MI325X (gfx942) | DeepSeek-V4-**Flash-Base** TP8 FP8 KV | V4-Pro on 942 is not the qualification stem |
| MI355X (gfx950) | V4-**Pro** TP8; Kimi K2.7-Code MXFP4 **TP4** | Spur/managed compute labs use these stems |

Native ATOM is **node-local TP ≤ 8**. Cross-node pipeline parallel uses ``vllm_atom`` / ``sglang`` ``_distributed`` stems, not native ATOM. Atomesh P/D is not wired in this suite yet.

## Config inventory

Configs are lab-validated unless a row is marked pending or bring-up.

| Stem | Files | Driver |
|------|-------|--------|
| `mi3xx_atom_deepseek-r1_fp8` | `_single` (profiles: `perf`, `mtp3`) | native `atom` |
| `mi3xx_atom_deepseek-v4-flash` | `_single` | native `atom` (gfx942 V4 gate; bring-up thresholds) |
| `mi3xx_atom_qwen3.5-397b-a17b_fp8` | `_single` (profiles: `perf`, `mtp3`) | native `atom` |
| `mi355x_atom_kimi-k27-code_mxfp4` | `_single` | native `atom` (MI355X; bring-up thresholds) |
| `mi355x_atom_deepseek-v4-pro` | `_single` | native `atom` (MI355X; bring-up thresholds) |
| `mi3xx_atom_vllm_deepseek-r1_fp8` | `_single`, `_distributed` | `vllm_atom` (serving schema; distributed uses PP=2) |
| `mi3xx_atom_vllm_gpt-oss-120b_mxfp4` | `_single` | `vllm_atom` (serving schema) |
| `mi3xx_atom_vllm_qwen3.5-397b-a17b_fp8` | `_single`, `_distributed` | `vllm_atom` (serving schema; `_single` lab-validated; `_distributed` pending) |
| `mi3xx_atom_sglang_deepseek-r1_fp8` | `_single`, `_distributed` | `sglang` (serving schema) |
| `mi3xx_atom_sglang_qwen3.5-397b-a17b_fp8` | `_single`, `_distributed` | `sglang` (serving schema; `_single` lab-validated; `_distributed` pending) |

Parity configs (`atom_vllm`, `atom_sglang`) use the unified serving schema:
`server_params`, `benchmark_params`, `sweeps`, `runs`. Run with ``cvs run atom``.

Config stems use **family** prefixes ``mi3xx`` (gfx942) or ``mi355x`` (gfx950).
Threshold files use **platform** prefixes ``mi325x`` or ``mi355x``. Copy each
config + its ``threshold_json`` into a dedicated subdirectory before running.

New 355 / Flash stems ship ``enforce_thresholds: false`` until lab calibration.
