# Multi-Node Validation – ATOM tracker — Excel update reference

**Branch:** `hnimrama/atom-accuracy`  
**Scope:** CVS `cvs run atom` only — not InferenceX runner. Cross-engine compare uses `driver=vllm_atom` / `driver=sglang` inside the atom suite.  
**Last updated:** 2026-08-18

Use this file when updating the **Excel Online** tracker. Columns mirror the live sheet:

| # | Category | Test / Metric | Priority | Automation Status | Comments |
|---|----------|---------------|----------|-------------------|----------|

**Status key**

| Value | Meaning |
|-------|---------|
| **Completed** | CVS config stem + `cvs run atom` harness wired (automation shipped) |
| **In Progress** | Partial automation or lab/threshold work still open |
| *(blank)* | Not started — no atom stem yet |

Lab validation is tracked in **Comments** (`lab pending` until smoke passes); automation **Completed** does not require lab pass.

**P1-first rule:** Close all **P1** rows below before marking **P2** workloads/metrics **Completed**. P2 **In Progress** is fine while P1 lab runs are in flight.

**Lab order (MI300X jumphost):**

1. `mi300x_atom_deepseek-r1_fp8_single` — perf smoke (`w1_1k_1k-conc128`) → #3, #24–27, #33
2. `mi300x_atom_deepseek-r1_fp8_accuracy` — gsm8k smoke (`gsm8k_flex`) → #45; FUNC-1/2 on #3
3. Full W1 accuracy run → calibrate #40, #46
4. M4 triple: `_single` / `_vllm_single` / `_sglang_single` → #1, #2
5. P1 workloads #18, #7, #8, then MI355X #6

---

## ATOM – FRAMEWORK PATHS (cross-compare) (#1–5)

| # | Category | Test / Metric | Priority | Automation Status | Comments |
|---|----------|---------------|----------|-------------------|----------|
| 1 | ATOM Path | vLLM (ROCm) – baseline engine for InferenceX parity and cross-compare cards | P1 | **In Progress** | CVS M4: `mi300x_atom_deepseek-r1_fp8_vllm_single` (`driver=vllm_atom`); W2: `…_gpt-oss-120b_mxfp4_vllm_single`. Run-deck parity panel (`CVS_ATOM_PARITY_REF_JSON`). **Lab compare vs #3 pending.** |
| 2 | ATOM Path | SGLang (ROCm) – baseline engine for InferenceX parity and cross-compare cards | P1 | **In Progress** | CVS M4: `mi300x_atom_deepseek-r1_fp8_sglang_single`; W2: `…_gpt-oss-120b_mxfp4_sglang_single`. Same workload cards as ATOM where comparable. **Lab pending.** |
| 3 | ATOM Path | ATOM – InferenceX framework: atom (AiTer Optimized Model serving path) | P1 | **In Progress** | Primary path: `driver=atom`, `mi{300,355}x_atom_deepseek-r1_fp8_single`. **→ Completed after perf smoke passes.** INFRA: FUNC-1/2 on accuracy variant; optional INF-6 dmesg (`platform.dmesg_scan`). |
| 4 | ATOM Path | ATOM + MTP – InferenceX *-atom-mtp recipes (speculative / multi-token prediction) | P1 | **In Progress** | `mi300x_atom_deepseek-r1_fp8_mtp3`, `…_mtp3_accuracy`; `test_atom_mtp_quality` (ACC-4/5/13). MTP thresholds `"kind": "info"` until lab. Chat-template required. |
| 5 | ATOM Path | ATOM-Disagg – InferenceX framework: atom-disagg (disaggregated prefill/decode) | P2 | *(blank)* | Not in atom suite — use SGLang disagg path. Wide EP + custom comms kernels. |

---

## WORKLOAD COVERAGE – ATOM recipes (#6–23)

| # | Category | Test / Metric | Priority | Automation Status | Comments |
|---|----------|---------------|----------|-------------------|----------|
| 6 | Workload | DeepSeek R1 FP8 \| MI355X \| InferenceX framework: atom \| dsr1-fp8-mi355x-atom | P1 | **In Progress** | CVS: `mi{300,355}x_atom_deepseek-r1_fp8_{single,accuracy,distributed,mtp3,baseline_sweep}`. W1 **1K/1K** TP8. MI355X `enforce_thresholds: false` until lab confirm. |
| 7 | Workload | GPT-oss-120b – openai/gpt-oss-120b, TP=4 (MXFP4) | P1 | **In Progress** | `mi{300,355}x_atom_gpt-oss-120b_mxfp4_{single,accuracy}`; M4 `…_vllm_single` / `…_sglang_single`. **8K/1K** TP4 MXFP4. Lab pending. |
| 8 | Workload | Qwen3.5-397B-A17B-FP8 – amd/Qwen3.5-397B-A17B-FP8, TP=8 | P1 | **In Progress** | `mi{300,355}x_atom_qwen3.5-397b-a17b_fp8_single` + thresholds. **1K/8K** TP8. Lab pending. |
| 9 | Workload | GLM 5.1 – zai-org/GLM-5.1-FP8, TP=8 (FP8) | P2 | **In Progress** | `mi{300,355}x_atom_glm-5.1_{single,accuracy}`. Lab pending — **defer until P1 closed.** |
| 10 | Workload | DeepSeek V4 Pro – deepseek-ai/DeepSeek-V4-Pro, TP=8 (FP4+FP8) | P1 | **Completed** | `mi{300,355}x_atom_deepseek-v4-pro_{longctx_single,vllm_single,sglang_single,distributed}`; 5000/1024; V4 vLLM DSv4 recipe; lab pending |
| 11 | Workload | DeepSeek V4 Flash – deepseek-ai/DeepSeek-V4-Flash, TP=4 (FP4+FP8) | P2 | **In Progress** | `mi{300,355}x_atom_deepseek-v4-flash_single`. Lab pending. |
| 12 | Workload | Kimi K2.6 Thinking – uniquealexx/Kimi-K2.6-Thinking-200x, TP=4 (INT4) | P2 | **In Progress** | `mi{300,355}x_atom_kimi-k2.6-thinking_{single,accuracy}`. Lab pending. |
| 13 | Workload | GLM 5.2 MXFP4 – amd/GLM-5.2-MXFP4, TP=8 | P2 | **In Progress** | `mi{300,355}x_atom_glm-5.2-mxfp4_single`. MI355X + ATOM + vLLM parity TBD. |
| 14 | Workload | Kimi K2.5 MXFP4 – amd/Kimi-K2.5-MXFP4, TP=4 | P2 | **In Progress** | `mi{300,355}x_atom_kimi-k2.5-mxfp4_single`. ATOM + vLLM TBD. |
| 15 | Workload | Qwen 3.5 397B A17B – Qwen/Qwen3.5-397B-A17B, TP=8 (FP8) | P2 | **In Progress** | `mi{300,355}x_atom_qwen3.5-397b-a17b_single` (BF16); distinct from #8 amd FP8 variant. |
| 16 | Workload | GLM 5.2 – zai-org/GLM-5.2-FP8, TP=8 (FP8) | P2 | **In Progress** | `mi{300,355}x_atom_glm-5.2-fp8_single`. Lab pending. |
| 17 | Workload | GLM 5.2 – zai-org/GLM-5.2, TP=8 (MXFP4) | P2 | **In Progress** | `mi{300,355}x_atom_glm-5.2_single`. MI355X. Lab pending. |
| 18 | Workload | Kimi-K2.7-Code – moonshotai/Kimi-K2.7-Code, TP=8 (MXFP4) | P1 | **In Progress** | **Highest P1 workload lab priority.** `mi{300,355}x_atom_kimi-k2.7-code_{single,longctx_single,code_accuracy}`. ACC-12 NIAH on accuracy stem. Large-seq / agentic use case. |
| 19 | Workload | MiniMax-M3 – MiniMaxAI/MiniMax-M3 (BF16) | P2 | **In Progress** | `mi{300,355}x_atom_minimax-m3_single`. vLLM uplift TBD. |
| 20 | Workload | Qwen3.5-397B-A17B-MXFP4 – amd/Qwen3.5-397B-A17B-MXFP4, TP=8 | P2 | **In Progress** | `mi{300,355}x_atom_qwen3.5-397b-a17b-mxfp4_single`. ATOM + vLLM TBD. |
| 21 | Workload | Mistral Large 3 – mistralai/Mistral-Large-3-675B-Instruct-2512, TP=8 (FP8) | P2 | **In Progress** | `mi{300,355}x_atom_mistral-large-3_single`. Lab pending. |
| 22 | Workload | DeepSeek-R1-0528-MXFP4 – amd/DeepSeek-R1-0528-MXFP4, TP=8 MXFP4 | P2 | **In Progress** | `mi{300,355}x_atom_deepseek-r1_mxfp4_{single,accuracy}`. MI355X. Lab pending. |
| 23 | Workload | MiMo-v2.5-Pro – XiaomiMiMo/MiMo-V2.5-Pro, TP=8 (BF16) | P2 | **In Progress** | `mi{300,355}x_atom_mimo-v2.5-pro_single`. Lab pending. |

---

## PERFORMANCE BENCHMARK METRICS (#24–37)

| # | Category | Test / Metric | Priority | Automation Status | Comments |
|---|----------|---------------|----------|-------------------|----------|
| 24 | Performance | Throughput per GPU – total tokens/sec (tput_per_gpu) | P1 | **In Progress** | `client.per_gpu_throughput` in bench artifact. **→ Completed after W1 perf smoke passes.** |
| 25 | Performance | Output throughput per GPU – decode tokens/sec/GPU (output_tput_per_gpu) | P1 | **In Progress** | `client.output_tput_per_gpu`. **→ Completed with #24.** |
| 26 | Performance | TTFT – mean & p99 (ms) | P1 | **In Progress** | `client.mean_ttft_ms`, `client.p99_ttft_ms`. **→ Completed with #24.** |
| 27 | Performance | TPOT – mean & p95 (ms) | P1 | **In Progress** | `client.mean_tpot_ms`, `client.p95_tpot_ms`. **→ Completed with #24.** |
| 28 | Performance | Prefill latency – p50 / p95 (ms) | P2 | *(blank)* | Not emitted separately from TTFT on `driver=atom`. |
| 29 | Performance | End-to-end latency – mean / p95 / p99 (ms) | P1 | **Completed** | `client.mean_e2el_ms`, p95/p99 tails — emitted in artifact; gate optional. |
| 30 | Performance | Latency vs load – P95 & P99 at each QPS/concurrency step | P2 | **In Progress** | `*_baseline_sweep` (14 cells); sweep knee in report. Full vs trim sweep per workflow. |
| 31 | Performance | Goodput – successful requests / total requests under load | P1 | **Completed** | `client.goodput` when finite `request_rate`; record-only on W1 inf rate. |
| 32 | Performance | Scaling efficiency % – multi-GPU / multi-node vs 1× reference | P2 | **In Progress** | `*_distributed`; `scaling.efficiency_pct`. IB/UE fabric in run card + `test_discover_topology`. |
| 33 | Performance | Peak GPU memory – max allocated / reserved (MB) | P2 | **In Progress** | **INF-7:** `platform.gpu_metrics_poll: true` on W1 single → `gpu.peak_gpu_memory_mb`. Validate on perf smoke. |
| 34 | Performance | KV cache memory footprint (GB) at target batch × sequence | P2 | *(blank)* | No ATOM KV telemetry parser yet. |
| 35 | Performance | Request success rate & error mix | P2 | **Completed** | `client.success_rate`, `client.failed`. **INF-6:** optional `platform.dmesg_scan` → `test_verify_dmesg`. |
| 36 | Performance | Model load time (s) + load memory (MB) – cold start | P2 | **In Progress** | `gpu.model_load_s`, `gpu.model_load_memory_mb` when poll enabled; `server.model_cache_bytes`. |
| 37 | Performance | Time-to-ready — server (container start → readiness regex match) | P2 | **In Progress** | `server.time_to_ready_s` lifecycle metrics. |

---

## OPTIONAL QUALITY GATES (#38–39)

| # | Category | Test / Metric | Priority | Automation Status | Comments |
|---|----------|---------------|----------|-------------------|----------|
| 38 | Quality | MTP acceptance rate / degenerate-spec decode checks (chat-formatted prompts only) | P2 | **In Progress** | ACC-4/5/13 via `test_atom_mtp_quality` on `*_mtp3`; `"kind": "info"` until lab. W3/W4/W7/W15 MTP recipes. |
| 39 | Quality | Quantization output parity – FP8/FP4 vs BF16 reference | P2 | **In Progress** | **ACC-7:** `quant_parity` + `test_atom_quant_parity` on W1 `*_accuracy`; BF16 reference pairing TBD. |

---

## ACCURACY BENCHMARK METRICS (#40–50)

| # | Category | Test / Metric | Priority | Automation Status | Comments |
|---|----------|---------------|----------|-------------------|----------|
| 40 | MMLU-PRO | Refined MMLU, 10 choices; Accuracy % (5-shot) | P1 | **In Progress** | ACC-15; lm-eval `mmlu_pro` on W1 `mi300x_atom_deepseek-r1_fp8_accuracy`; **info → min after lab**. HF OLM v2 / LightEval. |
| 41 | BBH | Big Bench Hard; Normalized accuracy % (3-shot) | P2 | **In Progress** | ACC-16; lm-eval `bbh` on W1 accuracy stem; info threshold. |
| 42 | MATH Level 5 | High-school competition math; Exact match % (4-shot) | P2 | **In Progress** | ACC-11; `hendrycks_math` on W7 `…_thinking_accuracy` (Level 5 metadata). |
| 43 | GPQA | Graduate-level science Q&A; Normalized accuracy % (0-shot) | P2 | *(blank)* | Gated HF dataset — not wired in CVS. |
| 44 | MuSR | Multistep soft reasoning; Normalized accuracy % (0-shot) | P2 | **In Progress** | ACC-17; lm-eval `musr` on W1 accuracy stem; info threshold. |
| 45 | GSM8K | Grade-school math; Exact-match accuracy % | P1 | **In Progress** | ACC-1..3; lm-eval on `mi300x_atom_deepseek-r1_fp8_accuracy`; gate ≥ **0.94** flexible-extract. **→ Completed after gsm8k smoke passes.** HF LightEval. |
| 46 | HellaSwag | Commonsense sentence completion; Accuracy % | P1 | **In Progress** | ACC-14; lm-eval `hellaswag` 0-shot; info → min after first lab run. |
| 47 | MMLU | Legacy MMLU, 57 subjects; Accuracy % | P2 | **In Progress** | ACC-6 scaffold on W3 `glm-5.1_accuracy`; prefer #40 MMLU-PRO for P1. |
| 48 | ARC-Challenge | AI2 Reasoning Challenge; Accuracy % | P2 | **In Progress** | ACC-18; lm-eval `arc_challenge` on W1 accuracy stem. |
| 49 | WinoGrande | Commonsense pronoun resolution; Accuracy % | P2 | **In Progress** | ACC-19; lm-eval `winogrande` on W1 accuracy stem. |
| 50 | Scale Parity – Accuracy | Same model/weights → scores match 1 GPU / multi-GPU / multi-node | P2 | **In Progress** | `mi300x_atom_deepseek-r1_fp8_distributed_accuracy`; `CVS_ATOM_SCALE_ACCURACY_REF_JSON` / scale_accuracy report panel. |

---

## INFRA / CROSS-CUTTING (not numbered in Excel — map to Comments)

These items **do not get their own rows** in the Excel sheet. Copy the text into **Comments** on the row listed.

| Item | Priority | Automation Status | Map to Excel row # | Comments (paste into sheet) |
|------|----------|-----------------|--------------------|-----------------------------|
| INF-6 time-bounded dmesg scan | — | **Completed** | **#3**, **#35** | `platform.dmesg_scan: true` → `test_verify_dmesg` (opt-in) |
| INF-7 GPU metrics poll | — | **In Progress** | **#33** | `platform.gpu_metrics_poll: true` on W1 single → `gpu.peak_gpu_memory_mb` |
| Framework parity report panels | — | **Completed** | **#1**, **#2** | M4 run-deck: `CVS_ATOM_PARITY_REF_JSON`; `compare.vllm.*` / `compare.sglang.*` panels (render-only) |
| FUNC-1 API smoke | — | **Completed** | **#3**, **#45** | `functional.api_smoke: true` → `test_openai_compatible_smoke` after `wait_ready` |
| FUNC-2 health check | — | **Completed** | **#3**, **#45** | `functional.health_check: true` → `test_server_health` (/health, model list, max_tokens=1) |
| ACC-7 quant parity probe | — | **In Progress** | **#39** | `quant_parity` block + `test_atom_quant_parity` on W1 accuracy variant |
| ACC-12 NIAH long-context | — | **In Progress** | **#18** | `…_code_accuracy` + `test_atom_long_context_accuracy` |

---

## P1 closure checklist (flip to **Completed** in Excel)

| Step | Lab run | Rows to close |
|------|---------|---------------|
| 1 | Perf smoke: `mi300x_atom_deepseek-r1_fp8_single`, `-k w1_1k_1k-conc128` | **#3**, **#24–27**; partial **#33** |
| 2 | Accuracy gsm8k: `mi300x_atom_deepseek-r1_fp8_accuracy`, `-k gsm8k_flex` | **#45**; note FUNC-1/2 on **#3** |
| 3 | Full W1 accuracy (all lm-eval tasks) | Calibrate **#40**, **#46** (flip info → min gates) |
| 4 | M4 triple: `_single` / `_vllm_single` / `_sglang_single` | **#1**, **#2** |
| 5 | Kimi K2.7 lab | **#18** |
| 6 | GPT-oss / Qwen / MI355X R1 | **#7**, **#8**, **#6** |

---

## Summary counts (for Excel footer — update after lab)

**Framework + metric rows (#1–50), suggested as of 2026-08-18 (pre-lab):**

| Status | Count (#1–50) | Notes |
|--------|---------------|-------|
| **Completed** | 4 | #10 V4 Pro stems + #29, #31, #35 (code emits; INFRA FUNC-1/2/parity panels in repo) |
| **In Progress** | 42 | Includes remaining P1 rows until smokes pass |
| **Blank / Not started** | 4 | #5, #28, #34, #43 |

**P1 rows only (18 total in Excel):** #1–4, #6–8, #10, #18, #24–27, #29, #31, #40, #45–46

| P1 metric | Value |
|-----------|-------|
| # of P1 | 18 |
| # of P1 **Completed** (automation, pre-lab) | **3** (#10, #29, #31) — bump after perf + gsm8k smokes for #24–27, #45 |
| % of P1 automated | ~17% now → target **≥80%** after steps 1–4 above |

*Recalculate footer formulas after each lab milestone. Automation **Completed** = config + harness shipped; note `lab pending` in Comments until smoke passes.*

---

## Changelog

| Date | Summary |
|------|---------|
| 2026-08-18 | V4 Pro (#10) automation complete: longctx + vllm + sglang + distributed stems (mi300x/mi355x); orch V4 recipe guards |
| 2026-08-18 | Reconcile with live Excel sheet; P1-first statuses; INFRA → row mapping; fix #6 row; V4 Pro as P1 per tracker |
| 2026-08-11 | Initial cheat sheet (`357a82c8`) |

---

## Related repo docs

- [plans/atom-workload-tracker.md](atom-workload-tracker.md) — CVS automation map
- [plans/atom-accuracy-test-catalog.md](atom-accuracy-test-catalog.md) — ACC-* detail
- [cvs/input/config_file/inference/atom/README.md](../cvs/input/config_file/inference/atom/README.md) — config stem inventory + lab commands
