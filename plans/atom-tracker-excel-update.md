# Multi-Node Validation – ATOM tracker — Excel update reference

**Branch:** `hnimrama/atom-accuracy` (pushed `9e840d69`)  
**Scope:** CVS `cvs run atom` only — not InferenceX runner. Cross-engine compare uses `driver=vllm_atom` / `driver=sglang` inside atom suite.  
**Date:** 2026-08-11

Use this file when updating the Excel Online tracker. Columns mirror the PDF:

| # | Category | Test / Metric | Priority | Automation Status (suggested) | Comments / CVS config stem |
|---|----------|---------------|----------|------------------------------|----------------------------|

**Status key**

| Value | Meaning |
|-------|---------|
| **Completed** | Harness + config in repo; lab-validated where applicable |
| **In Progress** | Config/harness in repo; lab or threshold calibration pending |
| **Not Started** | No atom config stem yet |

---

## ATOM – FRAMEWORK PATHS (#1–5)

| # | Category | Test / Metric | Priority | Automation Status (suggested) | Comments / CVS config stem |
|---|----------|---------------|----------|------------------------------|----------------------------|
| 1 | IX Path | vLLM (ROCm) – baseline engine for parity and cross-compare cards | P1 | **In Progress** | `mi300x_atom_deepseek-r1_fp8_vllm_single` (`driver=vllm_atom`); M4 report panel; lab pending |
| 2 | IX Path | SGLang (ROCm) – baseline engine for parity and cross-compare cards | P1 | **In Progress** | `mi300x_atom_deepseek-r1_fp8_sglang_single` (`driver=sglang`); lab pending |
| 3 | IX Path | ATOM – AiTer Optimized Model serving path | P1 | **Completed** | `driver=atom`; W1 `mi{300,355}x_atom_deepseek-r1_fp8_single` |
| 4 | IX Path | ATOM + MTP – speculative / multi-token prediction | P1 | **In Progress** | `mi300x_atom_deepseek-r1_fp8_mtp3`, `…_mtp3_accuracy`; `test_atom_mtp_quality`; lab pending |
| 5 | IX Path | ATOM-Disagg – disaggregated prefill/decode | P2 | **Not Started** | Use SGLang disagg suite, not atom |

---

## WORKLOAD COVERAGE – ATOM recipes (#6–23)

| # | Category | Test / Metric | Priority | Automation Status (suggested) | Comments / CVS config stem |
|---|----------|---------------|----------|------------------------------|----------------------------|
| 6 | Workload | DeepSeek R1 FP8 \| MI355X \| atom \| 1K/1K | P1 | **In Progress** | `mi355x_atom_deepseek-r1_fp8_{single,baseline_sweep,distributed,mtp3,accuracy}`; MI300X mirror on `mi300x_*`; lab gates pending |
| 7 | Workload | GPT-oss-120b – openai/gpt-oss-120b, TP=4 (MXFP4), 8K/1K | P1 | **In Progress** | `mi{300,355}x_atom_gpt-oss-120b_mxfp4_single`, `…_mxfp4_accuracy`; lab pending |
| 8 | Workload | Qwen3.5-397B-A17B-FP8 – amd/Qwen3.5-397B-A17B-FP8, TP=8, 1K/8K | P1 | **In Progress** | `mi{300,355}x_atom_qwen3.5-397b-a17b_fp8_single` + thresholds; committed `f33cd56c`; lab pending |
| 9 | Workload | GLM 5.1 – zai-org/GLM-5.1-FP8, TP=8, 1K/4K | P2 | **In Progress** | `mi{300,355}x_atom_glm-5.1_single`, `…_accuracy`; lab pending |
| 10 | Workload | DeepSeek V4 Pro – deepseek-ai/DeepSeek-V4-Pro, TP=8, 5000/1024 | P2 | **Not Started** | No atom stem |
| 11 | Workload | DeepSeek V4 Flash – deepseek-ai/DeepSeek-V4-Flash, TP=4, 1K/1K | P2 | **Not Started** | No atom stem |
| 12 | Workload | Kimi K2.6 Thinking – uniquealexx/Kimi-K2.6-Thinking-200x, TP=4, 1K/1K | P2 | **In Progress** | `mi{300,355}x_atom_kimi-k2.6-thinking_{single,accuracy}`; committed `109832a0`; lab pending |
| 13 | Workload | GLM 5.2 MXFP4 – amd/GLM-5.2-MXFP4, TP=8, 1K/1K | P2 | **Not Started** | No atom stem |
| 14 | Workload | Kimi K2.5 MXFP4 – amd/Kimi-K2.5-MXFP4, TP=4, 1K/1K | P2 | **Not Started** | No atom stem |
| 15 | Workload | Qwen 3.5 397B A17B – Qwen/Qwen3.5-397B-A17B, TP=8, 1K/1K | P2 | **Not Started** | No atom stem (distinct from #8 amd FP8 variant) |
| 16 | Workload | GLM 5.2 – zai-org/GLM-5.2-FP8, TP=8, 1K/1K | P2 | **Not Started** | No atom stem |
| 17 | Workload | GLM 5.2 – zai-org/GLM-5.2, TP=8 (MXFP4), 1K/1K | P2 | **Not Started** | No atom stem |
| 18 | Workload | Kimi-K2.7-Code – moonshotai/Kimi-K2.7-Code, TP=8 (MXFP4), 1K/1K | P1 | **In Progress** | `mi{300,355}x_atom_kimi-k2.7-code_single`, `…_longctx_single` (8K ISL), `…_code_accuracy` (NIAH); committed `fc5f00e3`; **highest lab priority** |
| 19 | Workload | MiniMax-M3 – MiniMaxAI/MiniMax-M3 (BF16), 1K/1K | P2 | **Not Started** | No atom stem |
| 20 | Workload | Qwen3.5-397B-A17B-MXFP4 – amd/Qwen3.5-397B-A17B-MXFP4, TP=8, 1K/1K | P2 | **Not Started** | No atom stem |
| 21 | Workload | Mistral Large 3 – mistralai/Mistral-Large-3-675B-Instruct-2512, TP=8, 1K/1K | P2 | **Not Started** | No atom stem |
| 22 | Workload | DeepSeek-R1-0528-MXFP4 – amd/DeepSeek-R1-0528-MXFP4, TP=8, 1K/1K | P2 | **In Progress** | `mi{300,355}x_atom_deepseek-r1_mxfp4_{single,accuracy}`; MI355X accuracy committed `2ae17d8d`; lab pending |
| 23 | Workload | MiMo-v2.5-Pro – XiaomiMiMo/MiMo-V2.5-Pro, TP=8 (BF16), 1K/1K | P2 | **Not Started** | No atom stem |

---

## PERFORMANCE BENCHMARK METRICS (#24–37)

| # | Category | Test / Metric | Priority | Automation Status (suggested) | Comments / CVS config stem |
|---|----------|---------------|----------|------------------------------|----------------------------|
| 24 | Performance | Throughput per GPU – total tokens/sec (`tput_per_gpu`) | P1 | **Completed** | `client.tput_per_gpu` in bench artifact |
| 25 | Performance | Output throughput per GPU – decode tokens/sec/GPU (`output_tput_per_gpu`) | P1 | **Completed** | `client.output_tput_per_gpu` |
| 26 | Performance | TTFT – mean & p99 (ms) | P1 | **Completed** | `client.mean_ttft_ms`, `client.p99_ttft_ms` |
| 27 | Performance | TPOT – mean & p95 (ms) | P1 | **Completed** | `client.mean_tpot_ms`, `client.p95_tpot_ms` |
| 28 | Performance | Prefill latency – p50 / p95 (ms) | P2 | **Not Started** | Not emitted separately from TTFT |
| 29 | Performance | End-to-end latency – mean / p95 / p99 (ms) | P2 | **Completed** | `client.mean_e2el_ms`, tails |
| 30 | Performance | Latency vs load – P95 & P99 at each QPS/concurrency step | P2 | **In Progress** | `*_baseline_sweep` (14 cells); sweep knee summary in report |
| 31 | Performance | Goodput – successful requests / total requests under load | P2 | **Completed** | `client.goodput` when present |
| 32 | Performance | Scaling efficiency % – multi-GPU / multi-node vs 1× reference | P2 | **In Progress** | `*_distributed`; `scaling.efficiency_pct` |
| 33 | Performance | Peak GPU memory – max allocated / reserved (MB) | P2 | **In Progress** | `platform.gpu_metrics_poll: true` → `gpu.peak_gpu_memory_mb`; committed `91ddd752`; enabled on W1 single |
| 34 | Performance | KV cache memory footprint (GB) at target batch × sequence | P2 | **Not Started** | — |
| 35 | Performance | Request success rate & error mix | P2 | **Completed** | `client.success_rate`, `client.failed` |
| 36 | Performance | Model load time (s) + load memory (MB) – cold start | P2 | **In Progress** | `model_fetch`, `server.model_cache_bytes` lifecycle |
| 37 | Performance | Time-to-ready — server (container start → readiness regex match) | P2 | **In Progress** | `server.time_to_ready_s` lifecycle metrics |

---

## OPTIONAL QUALITY GATES (#38–39)

| # | Category | Test / Metric | Priority | Automation Status (suggested) | Comments / CVS config stem |
|---|----------|---------------|----------|------------------------------|----------------------------|
| 38 | Quality | MTP acceptance rate / degenerate-spec decode checks | P2 | **In Progress** | ACC-4/5/13 via `test_atom_mtp_quality` on `*_mtp3`; floors TBD |
| 39 | Quality | Quantization output parity – FP8/FP4 vs BF16 reference | P2 | **Not Started** | ACC-7 not implemented |

---

## ACCURACY BENCHMARK METRICS (#40–50)

| # | Category | Test / Metric | Priority | Automation Status (suggested) | Comments / CVS config stem |
|---|----------|---------------|----------|------------------------------|----------------------------|
| 40 | MMLU-PRO | Refined MMLU, 10 choices; Accuracy % (5-shot) | P1 | **In Progress** | ACC-15; lm-eval `mmlu_pro` on W1 `*_accuracy`; info threshold → min after lab; committed `77897c2e` |
| 41 | BBH | Big Bench Hard; Normalized accuracy % (3-shot) | P2 | **Not Started** | — |
| 42 | MATH Level 5 | High-school competition math; Exact match % (4-shot) | P2 | **Not Started** | Partial: `hendrycks_math` on W7 thinking scaffold; not level-5 filter |
| 43 | GPQA | Graduate-level science Q&A; Normalized accuracy % (0-shot) | P2 | **Not Started** | Gated HF dataset |
| 44 | MuSR | Multistep soft reasoning; Normalized accuracy % (0-shot) | P2 | **Not Started** | — |
| 45 | GSM8K | Grade-school math; Exact-match accuracy % | P1 | **In Progress** | ACC-1..3; lm-eval on W1 `mi300x_atom_deepseek-r1_fp8_accuracy`; gate ≥ 0.94 flexible-extract; lab pending |
| 46 | HellaSwag | Commonsense sentence completion; Accuracy % | P1 | **In Progress** | ACC-14; lm-eval `hellaswag` 0-shot on W1 `*_accuracy`; info threshold → min after lab; committed `77897c2e` |
| 47 | MMLU | Legacy MMLU, 57 subjects; Accuracy % | P2 | **In Progress** | ACC-6 scaffold on W3 `glm-5.1_accuracy`; prefer ACC-15 (MMLU-PRO) for P1 |
| 48 | ARC-Challenge | AI2 Reasoning Challenge; Accuracy % | P2 | **Not Started** | — |
| 49 | WinoGrande | Commonsense pronoun resolution; Accuracy % | P2 | **Not Started** | — |
| 50 | Scale Parity – Accuracy | Same model/weights → scores match 1 GPU / multi-GPU / multi-node | P2 | **Not Started** | M4 is perf parity only; multinode PP accuracy deferred |

---

## INFRA / CROSS-CUTTING (not numbered in PDF)

| Item | Priority | Automation Status (suggested) | Comments |
|------|----------|------------------------------|----------|
| INF-6 time-bounded dmesg scan | — | **Completed** | Opt-in via env; committed `53117e3e` |
| INF-7 GPU metrics poll | — | **In Progress** | Same as tracker #33; committed `91ddd752` |
| Framework parity report panels | — | **Completed** | M4 compare + accuracy prev-run panels; committed `b87f31fe` |
| FUNC-1 API smoke | — | **Completed** | OpenAI-compatible chat/completion after `wait_ready` |
| FUNC-2 health check | — | **Completed** | `/health`, model list, max_tokens=1 liveness |
| ACC-12 NIAH long-context | — | **In Progress** | On K2.7 `…_code_accuracy`; lab pending |

---

## Lab priority (next runs — update Comments after lab)

1. `mi300x_atom_deepseek-r1_fp8_accuracy` — GSM8K + calibrate HellaSwag / MMLU-PRO (flip info → min)
2. `mi300x_atom_deepseek-r1_fp8_single` — perf gate + GPU poll validation
3. M4 triple: `_single` / `_vllm_single` / `_sglang_single`
4. `mi355x_atom_deepseek-r1_fp8_single` — flip MI355X `enforce_thresholds` after confirm
5. `mi300x_atom_kimi-k2.7-code_longctx_single` + `…_code_accuracy` — P1 K2.7

---

## Quick copy-paste summary (status counts)

| Status | Count (#1–50) |
|--------|---------------|
| Completed | 8 |
| In Progress | 24 |
| Not Started | 18 |

*Counts treat partial scaffolds as In Progress. Adjust after lab validation.*

---

## Related repo docs

- [`plans/atom-workload-tracker.md`](atom-workload-tracker.md) — CVS automation map
- [`plans/atom-accuracy-test-catalog.md`](atom-accuracy-test-catalog.md) — ACC-* detail
- [`cvs/input/config_file/inference/atom/README.md`](../cvs/input/config_file/inference/atom/README.md) — config stem inventory
