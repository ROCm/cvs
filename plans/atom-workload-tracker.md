# ATOM workload tracker — CVS automation map

Branch: `hnimrama/atom-accuracy`

Companion docs: [atom-tracker-excel-update.md](atom-tracker-excel-update.md) (Excel row-by-row),
[atom-accuracy-test-catalog.md](atom-accuracy-test-catalog.md) (ACC-* detail).

Maps the **Multi-Node Validation – ATOM (MI355X / MI350-class and above)** tracker
to CVS `cvs run atom` config stems. This is the **atom suite only** — not a separate
InferenceX runner. Cross-engine comparison uses `driver=vllm_atom` or `driver=sglang`
**inside the atom suite** (`*_vllm_single`, `*_sglang_single` stems).

**Status key**

| Status | Meaning |
|--------|---------|
| **Complete** | Config + harness; lab-validated gate (where applicable) |
| **In progress** | Config/harness in repo; lab or threshold calibration pending |
| **Not included** | No atom config stem yet |

GPU arch in CVS: `mi300x`, `mi355x` (MI350-class → use `mi355x_*` stems).

---

## Framework paths (#1–5)

| # | Path | P | CVS status | Atom stem / notes |
|---|------|---|------------|-------------------|
| 1 | vLLM-ATOM parity | P1 | In progress | `…_vllm_single`, `…_distributed` (`driver=vllm_atom`); M4 report panel |
| 2 | SGLang parity | P1 | In progress | `…_sglang_single`, `…_sglang_distributed` |
| 3 | ATOM native | P1 | Complete | `driver=atom`, W1 `…_single` |
| 4 | ATOM + MTP | P1 | In progress | `…_mtp3`, `test_atom_mtp_quality` |
| 5 | Disaggregated prefill/decode | P2 | Not included | Use SGLang disagg suite, not atom |

---

## Workloads (#6–23)

| # | Model / recipe | P | CVS status | Config stem(s) |
|---|----------------|---|------------|----------------|
| 6 | DeepSeek R1 FP8 MI355X 1K/1K | P1 | In progress | `mi355x_atom_deepseek-r1_fp8_{single,baseline_sweep,distributed,mtp3,accuracy}` |
| 7 | GPT-oss-120B MXFP4 8K/1K | P1 | In progress | `mi{300,355}x_atom_gpt-oss-120b_mxfp4_single`, `…_mxfp4_accuracy` |
| 8 | Qwen3.5-397B-A17B FP8 1K/8K | P1 | In progress | `mi{300,355}x_atom_qwen3.5-397b-a17b_fp8_single` |
| 9 | GLM 5.1 1K/4K (8K OSL in CVS W3) | P2 | In progress | `mi{300,355}x_atom_glm-5.1_single`, `…_accuracy` |
| 10 | DeepSeek V4 Pro 5000/1024 TP8 | P1 | Complete | `mi{300,355}x_atom_deepseek-v4-pro_{longctx_single,vllm_single,sglang_single,distributed}` |
| 11 | DeepSeek V4 Flash | P2 | In progress | `mi{300,355}x_atom_deepseek-v4-flash_single` |
| 13–17, 19–21, 23 | GLM 5.2, Qwen variants, etc. | P2 | In progress | `mi{300,355}x_atom_{glm-5.2-*,kimi-k2.5-*,qwen3.5-*,minimax-m3,mistral-large-3,mimo-v2.5-pro}_single` |
| 12 | Kimi K2.6 Thinking 1K/1K TP4 | P2 | In progress | `mi{300,355}x_atom_kimi-k2.6-thinking_{single,accuracy}` |
| 18 | Kimi K2.7 Code 1K/1K | P1 | In progress | `mi{300,355}x_atom_kimi-k2.7-code_single`, `…_longctx_single`, `…_code_accuracy` |
| 22 | DeepSeek R1 MXFP4 MI355X | P2 | In progress | `mi{300,355}x_atom_deepseek-r1_mxfp4_{single,accuracy}` |

---

## Performance metrics (#24–37)

| # | Metric | P | CVS status |
|---|--------|---|------------|
| 24–27 | Throughput / TTFT / TPOT (P1 core) | P1 | Complete — `client.*` in bench artifact |
| 29 | E2E latency | P2 | Complete — `client.mean_e2el_ms`, tails |
| 30 | Latency vs load curve | P2 | In progress — `*_baseline_sweep` (14 cells) |
| 31 | Goodput | P2 | Complete — `client.goodput` when present |
| 32 | Scaling efficiency | P2 | In progress — `*_distributed`, `scaling.efficiency_pct` |
| 33 | Peak GPU memory | P2 | In progress — `platform.gpu_metrics_poll` on P1 perf stems → `gpu.peak_gpu_memory_mb` |
| 35 | Success rate / failures | P2 | Complete — `client.success_rate`, `client.failed` |
| 36 | Model load / cache size | P2 | In progress — `model_fetch`, `server.model_cache_bytes` |
| 37 | Time-to-ready | P2 | In progress — `server.time_to_ready_s` lifecycle |
| 28, 34 | Prefill latency, KV footprint | P2 | Not included |

---

## Quality & accuracy (#38–50)

See [atom-accuracy-test-catalog.md](atom-accuracy-test-catalog.md) for ACC-* detail and
[atom-tracker-excel-update.md](atom-tracker-excel-update.md) for Excel Online row-by-row updates.

| Area | CVS status |
|------|------------|
| MTP quality (#38) | In progress — ACC-4/5/13 on `*_mtp3` |
| Quant parity (#39) | In progress — ACC-7 `quant_parity` config + `test_atom_quant_parity` probe scaffold |
| GSM8K / HellaSwag / MMLU-PRO (#45–46, #40) | In progress — ACC-1, ACC-14, ACC-15 on W1 `…_accuracy` |
| BBH, GPQA, MuSR, ARC, WinoGrande | In progress — P2 lm-eval on W1 `*_accuracy` (info gates); GPQA still needs gated HF access |
| Scale accuracy parity (#50) | In progress — `*_distributed_accuracy` + `CVS_ATOM_SCALE_ACCURACY_REF_JSON` panel |

---

## Lab priority (next runs)

1. `mi300x_atom_deepseek-r1_fp8_accuracy` — GSM8K + HellaSwag + MMLU-PRO baselines  
2. `mi300x_atom_deepseek-r1_fp8_single` — perf gate + GPU poll validation  
3. M4 triple: `_single` / `_vllm_single` / `_sglang_single`  
4. `mi355x_atom_deepseek-r1_fp8_single` — flip MI355X `enforce_thresholds` after confirm  
