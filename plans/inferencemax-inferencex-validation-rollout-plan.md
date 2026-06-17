<!--
  Planning document only — no implementation in this change.
  Owner: InferenceX / ATOM validation in CVS (pytest + suite JSON + thresholds).
-->

# InferenceX ATOM — CVS automation implementation plan

## 1. Purpose and scope

This document captures a **single implementation plan** and **action items** for automating **InferenceX ATOM** workloads, baselines, and metrics inside the **CVS** repository: suite JSON under `cvs/input/config_file/inference/`, orchestration via `InferenceMaxJob` (to be renamed in a compatibility-preserving way), pytest entrypoints, thresholds, and reporting.

**In scope (InferenceX focus)**

- **IX paths:** vLLM (ROCm) baseline, SGLang (ROCm) baseline, ATOM, ATOM+MTP, ATOM-Disagg (when orchestration allows).
- **Workloads:** Aligned with InferenceX `amd-master`-style recipes (e.g. DeepSeek R1 FP8 ATOM on MI355X, Qwen 3.5 FP8, GLM5 FP8, Kimi K2.5 FP4, and follow-on W2/W6–W16 as prioritized).
- **Metrics:** Throughput per GPU, output throughput per GPU, TTFT/TPOT and percentiles, E2E/prefill where available, sweeps, goodput-related fields, and extensions listed in this doc.
- **Lab reality:** **Thor2 NIC first**; AINIC / ConnectX-7 called out as **future matrix** when hardware exists (AMD preference order remains AINIC → Thor2 → CX7 for reporting honesty).

**Explicitly out of scope for the first delivery waves**

- Full **Optimus / KVMGR / NIXL / hipFile tiering / MaaS / Gateway** product automation (see **Appendix B — deferred DI platform table** for tracking only).
- **AINIC / CX7** NIC parity runs until lab inventory supports them.

---

## 2. Current repository baseline (context)

- CVS already clones **SemiAnalysisAI/InferenceX** via `inferencemax_repo` and drives **bench_serving** separately from upstream `fixed_seq_len` scripts when using **host-mounted** server scripts.
- Naming is still **InferenceMax** in many places (`InferenceMaxJob`, `inferencemax_single`, `load_inferencemax_suite_raw`, JSON key `inferencemax_repo`) while docs reference InferenceX.
- Helpers exist or are emerging for **sweep** parametrization (`inferencemax_sweep.py`) and **metric parsing / `agg_bmk` aliases** (`bench_serving_metrics.py`); orchestrator result extraction should converge on a **single pipeline** for thresholds and HTML tables.
- **ATOM-Disagg** and some multi-node upstream flows may depend on **SLURM** or layouts not yet mapped to CVS docker-only paths — treat as **blocked** until a spike defines orchestration.

---

## 3. Phased implementation strategy

| Phase | Name | Goal | Depends on |
|-------|------|------|------------|
| **A** | **InferenceX naming + compatibility** | User-facing and API names say **InferenceX**; **legacy keys and test IDs keep working** for at least one release. | None |
| **B** | **Metrics and verification pipeline** | One path: bench client log + optional JSON (`agg_bmk`) → canonical metric names → `verify_inference_results` + session table. | Phase A optional in parallel |
| **C** | **P1 IX paths and workloads** | Thor2 / MI355X: vLLM + SGLang DSR1 FP8 baselines; ATOM W1, W5, W8, W10; pinned IX revision per variant. | B partial |
| **D** | **P1 thresholds and reporting** | Threshold JSON per cell; results table columns for dashboard metrics. | C |
| **E** | **P2 workloads and MTP** | FP4 variants, additional models, MTP recipes with **chat-template / dataset** discipline; optional separate CI job for flake isolation. | D stable on W1 |
| **F** | **Multi-node and scaling metrics** | `nnodes`>1 validation, scaling efficiency, fabric metadata on run card. | Lab cluster + B |
| **G** | **ATOM-Disagg + KV stack** | Upstream SLURM/docker story + Optimus/KV transfer — **spike first**, then rows from Appendix B as applicable. | Product + infra |

---

## 4. Master table — InferenceX ATOM engine and benchmarks

Use this as the **authoritative checklist** for the InferenceX automation track. **Automation status** is planning language (update as work lands).

| # | Category | Test / Metric | Priority | Automation status | Comments | New |
|---|----------|-----------------|----------|---------------------|----------|-----|
| 1 | IX Path | vLLM (ROCm) baseline for parity / cross-compare | P1 | In progress | Same workload cards as ATOM where comparable | No |
| 2 | IX Path | SGLang (ROCm) baseline | P1 | In progress | Second open engine | No |
| 3 | IX Path | ATOM — primary IX serving path | P1 | In progress | `amd-master` / `rocm/atom:*` images | No |
| 4 | IX Path | ATOM + MTP | P1 | Partial | Chat-formatted inputs per IX AGENTS.md | No |
| 5 | IX Path | ATOM-Disagg | P1 | Blocked | Wide EP + PD pools; upstream may require SLURM / non-docker path | No |
| 6 | Workload | W1 — DeepSeek R1 FP8 MI355X atom | P1 | In progress | Primary ATOM gate | No |
| 7 | Workload | W2 — DeepSeek R1 FP4 MI355X atom | P2 | Not started | Memory / throughput envelope | No |
| 8 | Workload | W3 — DeepSeek R1 FP8 atom-mtp | P2 | Partial | MTP + chat template | No |
| 9 | Workload | W4 — DeepSeek R1 FP4 atom-mtp | P2 | Not started | MTP + FP4 | No |
| 10 | Workload | W5 — Qwen 3.5 FP8 MI355X atom | P1 | In progress | Large MoE FP8 class | No |
| 11 | Workload | W6 — Qwen 3.5 FP4 MI355X atom | P2 | Not started | | No |
| 12 | Workload | W7 — Qwen 3.5 FP8 atom-mtp | P2 | Partial | | No |
| 13 | Workload | W8 — GLM5 FP8 MI355X atom | P1 | In progress | | No |
| 14 | Workload | W9 — GLM5.1 FP4 MI355X atom | P2 | Not started | | No |
| 15 | Workload | W10 — Kimi K2.5 FP4 MI355X atom | P1 | In progress | | No |
| 16 | Workload | W11 — MiniMax M2.5 FP8 MI355X atom | P2 | Not started | | No |
| 17 | Workload | W12 — MiniMax M2.5 FP4 MI355X atom | P2 | Not started | | No |
| 18 | Workload | W13 — GPT-OSS FP4 MI355X atom | P2 | Partial | May reuse GPT-OSS patterns from existing variants | No |
| 19 | Workload | W14 — DeepSeek V4 FP4 MI355X atom | P2 | Not started | gfx950 flags per upstream yaml | No |
| 20 | Workload | W15 — DeepSeek V4 FP4 atom-mtp | P2 | Not started | | No |
| 21 | Workload | W16 — DeepSeek V4 FP4 atom-disagg | P2 | Blocked | Disagg orchestration TBD | No |
| 22 | Performance | Throughput per GPU (`tput_per_gpu`) | P1 | Partial | `agg_bmk` / alias merge | No |
| 23 | Performance | Output throughput per GPU (`output_tput_per_gpu`) | P1 | Partial | | No |
| 24 | Performance | TTFT mean and p99 | P1 | Partial | Unify parser + thresholds | No |
| 25 | Performance | TPOT mean and p95 | P1 | Partial | | No |
| 26 | Performance | Prefill latency p50 / p95 | P2 | Partial | Patterns in bench_serving_metrics | No |
| 27 | Performance | E2E mean / p95 / p99 | P2 | Partial | | No |
| 28 | Performance | Latency vs load (p95/p99 per QPS or concurrency) | P2 | Partial | Suite `sweep` + per-step thresholds or curve policy | No |
| 29 | Performance | Goodput | P2 | Partial | Record ratio; policy vs hard fail | No |
| 30 | Performance | Scaling efficiency % | P2 | Not started | Multi-GPU / multi-node vs baseline | No |
| 31 | Performance | Peak GPU memory | P2 | Not started | rocm-smi or engine-reported | No |
| 32 | Performance | KV cache footprint | P2 | Not started | Engine-dependent | No |
| 33 | Performance | Model load time + memory | P2 | Not started | Readiness to first token path | No |
| 34 | Quality | MTP acceptance / degenerate decode | P2 | Not started | W3,W4,W7,W15 | No |
| 35 | Quality | Quant / logit parity vs BF16 | P2 | Not started | Optional low-frequency job | No |
| 36 | Reliability / performance | Request success rate and error mix | P1–P2 | Not started | Complements fail on non-zero failed requests | Yes |
| 37 | Performance | ITL tail beyond p99 (max or p99.9) | P2 | Not started | Stall outliers | Yes |
| 38 | Performance | Time-to-ready (start → readiness) | P1–P2 | Partial | Server startup health | Yes |
| 39 | Quality / observability | Token accounting sanity check | P2 | Not started | Duration × tok/s vs reported tokens | Yes |
| 40 | Performance | Prefill vs decode as first-class fields | P2 | Partial | When client exposes | Yes |
| 41 | Observability | Steady-state HW snapshot (e.g. rocm-smi) | P2 | Not started | Explain throughput regressions | Yes |
| 42 | Quality | Output smoke (fixed prompt, small max_tokens) | P2 | Not started | Cheap correctness | Yes |
| 43 | Quality (MTP) | MTP health beyond acceptance (fallback/disable events) | P2 | Not started | Log-dependent | Yes |
| 44 | Methodology | Warmup discipline | P2 | Not started | Reduces cold-cache variance | Yes |
| 45 | Performance | Achieved concurrency or actual QPS vs target | P2 | Not started | Latency-vs-load validity | Yes |
| 46 | Multi-node / scale | Straggler check (per-node throughput band) | P2 | Not started | | Yes |
| 47 | Observability | Server log error bucketing with counts | P2 | Not started | Triages NCCL/OOM/Python | Yes |

---

## 5. Appendix A — Recipe index (reference)

Maintain a **YAML or doc** that maps **recipe id** → upstream `fixed_seq_len` script → CVS **host-mounted** `*.sh` → **backend** notes (e.g. `cvs/input/config_file/inference/inferencemax_single/inferencex_atom_recipes.yaml` and `cvs/lib/dtni/inferencex_server_scripts/`). Pin **`inferencemax_repo` ref** (or successor `inferencex_repo` ref) per variant in suite JSON.

---

## 6. Appendix B — Deferred DI platform matrix (tracking only)

These rows align with **Optimus / DI architecture** slides. They are **not** required to close the first InferenceX ATOM engine gates; keep them in a **second tab** or section so CI scope stays clear. **New = Yes** for all.

| # | Category | Test / Metric | Priority | Automation status | Comments | New |
|---|----------|-----------------|----------|---------------------|----------|-----|
| B1 | Lab profile | Run card: NIC = Thor2; AINIC/CX7 unavailable documented | P1 | Not started | Honest vs AMD NIC preference order | Yes |
| B2 | NIC matrix | Same recipes on AINIC when available | P2 | Not started | Cross-suite when run | Yes |
| B3 | NIC matrix | Same recipes on CX7 when available | P2 | Not started | | Yes |
| B4 | Orchestration | Optimus — PD scheduling, SLO routing, health | P2–P3 | Not started | Beyond CVS bench-only path until integrated | Yes |
| B5 | Orchestration | llm-d community path where relevant | P3 | Not started | | Yes |
| B6 | KV Manager | Optimus–KVMGR tiering | P2–P3 | Not started | | Yes |
| B7 | KV (community) | LMCache / HiCache / Mooncake | P3 | Not started | Comparison stacks | Yes |
| B8 | KV Transfer | NIXL, MOR-HQ PD KV moves | P2–P3 | Not started | Pairs with disagg | Yes |
| B9 | KV → storage | hipFile tiering | P3 | Not started | | Yes |
| B10 | Expert parallelism | MoR-EP scaling / imbalance metrics | P2 | Not started | If logs expose | Yes |
| B11 | Transport | RCCL health + version on run card | P2 | Partial | | Yes |
| B12 | Transport | rocSHMEM vs RCCL-only A/B | P3 | Not started | If both used | Yes |
| B13 | GPU matrix | MI300X / MI325X / MI308X / MI4XXX where supported | P2 | Not started | Beyond MI355X-first | Yes |
| B14 | GPU kernels | AITER / hipBlasLt toggle regression matrix | P2 | Not started | | Yes |
| B15 | Gateway | Envoy / Istio or customer gateway | P3 | Not started | | Yes |
| B16 | MaaS | Catalog / auth / policy smoke | P3 | Not started | | Yes |

---

## 7. Action items (detailed)

Each item has an **ID** for tracking in issues/PRs. Check off in your tracker as you complete them.

### Phase A — Naming and compatibility

| ID | Action | Details | Owner |
|----|--------|---------|-------|
| A-1 | **Inventory rename surfaces** | Code: `InferenceMaxJob`, `inferencemax_orch`, `inferencemax_host_scripts`, `load_inferencemax_suite_raw`, `inferencemax_single` pytest, `inference_lib` framework key `inferencemax`. Docs: `inferencemax.rst`, `run-cvs-tests.rst`, `configure-config.rst`, install/how-to. Config paths: `inference/inferencemax_single/`. | TBD |
| A-2 | **Define public names** | Target: `InferenceXJob` (or keep class name with re-export), test suite id `inferencex_single`, config directory `inferencex_single/` **or** retain path with doc alias. | TBD |
| A-3 | **Compatibility policy** | Accept **`inferencemax_repo` and `inferencex_repo`** (same semantics). Deprecated CLI names forward to new names with warning. Minimum **one release** of dual support. | TBD |
| A-4 | **Migration note** | Short “Lab migration: rename keys and paths” section in docs with before/after examples. | TBD |

### Phase B — Metrics and verification

| ID | Action | Details | Owner |
|----|--------|---------|-------|
| B-1 | **Canonical metric vocabulary** | Document mapping: IX `agg_bmk` / JSON fields ↔ CVS threshold keys ↔ legacy `output_throughput_per_sec`, `mean_ttft_ms`, etc. | TBD |
| B-2 | **Unify extraction** | Route log + JSON parsing through `bench_serving_metrics` (or shared module) from a single call used by orchestrator verify path and pytest result aggregation. | TBD |
| B-3 | **Extend `verify_inference_results`** | Support **higher-is-better** vs **lower-is-better** by metric name rules; add aliases so one threshold can match multiple reported keys during transition. | TBD |
| B-4 | **Results table** | Extend `cvs/tests/inference/inferencemax/_shared.py` (or successor path) with columns for P1 dashboard metrics (`tput_per_gpu`, `output_tput_per_gpu`, p99 TTFT, p95 TPOT, etc.). | TBD |
| B-5 | **Failed-request policy** | Document: hard fail vs threshold on goodput; align with `poll_client_completion` behavior. | TBD |

### Phase C — P1 paths and workloads (Thor2 / MI355X)

| ID | Action | Details | Owner |
|----|--------|---------|-------|
| C-1 | **Run card template** | Fields: IX repo SHA, image, recipe id, NIC=Thor2, GPU SKU, nnodes, TP, ISL, OSL, concurrency, sweep label. | TBD |
| C-2 | **Pin IX revision per variant** | Add ref field in suite JSON; document update process when bumping IX. | TBD |
| C-3 | **W1 ATOM** | Suite + threshold + host script wiring; first green run on lab. | TBD |
| C-4 | **vLLM DSR1 FP8 baseline** | Same workload card as W1 for compare; `inferencex_atom_recipes` / `mi355x_ix_vllm_*` style config. | TBD |
| C-5 | **SGLang DSR1 FP8 baseline** | Same card where comparable; validate backend launch matches upstream server block intent. | TBD |
| C-6 | **W5, W8, W10** | Qwen 3.5 FP8, GLM5 FP8, Kimi K2.5 FP4 ATOM configs + scripts + thresholds. | TBD |
| C-7 | **Workspace volume** | Confirm `/workspace` or equivalent for server logs and optional JSON artifacts on host when needed. | TBD |

### Phase D — Thresholds and CI

| ID | Action | Details | Owner |
|----|--------|---------|-------|
| D-1 | **Threshold files** | One `*threshold.json` per variant (per existing loader rules); keys match verify after B-3. | TBD |
| D-2 | **Owner and update policy** | Who adjusts thresholds after known-good regressions (image bump, intentional kernel change). | TBD |
| D-3 | **CI job split** | Optional: `inferencex_atom_smoke` (single cell) vs `inferencex_atom_sweep` (longer); MTP in separate job if flaky. | TBD |

### Phase E — P2 workloads and MTP

| ID | Action | Details | Owner |
|----|--------|---------|-------|
| E-1 | **MTP checklist** | Dataset, `--use-chat-template` or equivalent, tokenizer mode; link IX AGENTS.md requirement in variant README. | TBD |
| E-2 | **W2, W6, W9, W11–W15** | Prioritize by lab image and model access; add rows to this doc when schedule is known. | TBD |
| E-3 | **W16 disagg** | Spike: SLURM vs docker multi-node; unblock before automation tasks. | TBD |

### Phase F — Multi-node and scaling

| ID | Action | Details | Owner |
|----|--------|---------|-------|
| F-1 | **Multi-node recipe set** | Define minimal `nnodes`>1 ATOM case; client log paths per rank already considered in orch — validate end-to-end. | TBD |
| F-2 | **Scaling efficiency** | Baseline single-node or single-GPU reference; record fabric metadata on run card. | TBD |
| F-3 | **Straggler detection** | Implement row 46 when multi-node is stable. | TBD |

### Phase G — Quality and observability (engine-adjacent)

| ID | Action | Details | Owner |
|----|--------|---------|-------|
| G-1 | **Output smoke test** | Optional pytest or post-step: fixed prompt, small `max_tokens`, basic assertions. | TBD |
| G-2 | **Time-to-ready metric** | Row 38: measure from server start to readiness regex. | TBD |
| G-3 | **MTP health logs** | Row 43: define what ATOM/vLLM exposes; parse or skip. | TBD |
| G-4 | **Quant parity** | Nightly or manual; tolerance doc; exclude or special-case MTP stacks per matrix notes. | TBD |

### Documentation and governance

| ID | Action | Details | Owner |
|----|--------|---------|-------|
| DOC-1 | **Link this plan** | From `docs/reference/configuration-files/inferencemax.rst` (or successor `inferencex.rst`) with one paragraph pointer. | TBD |
| DOC-2 | **Workload legend** | Keep W1–W16 mapping table in docs next to threshold examples. | TBD |

---

## 8. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Rename breaks existing labs | Dual keys and deprecated aliases for one release; migration doc (A-3, A-4). |
| MTP / chat-template flakes | Separate job; smaller smoke concurrency; explicit checklist (E-1). |
| Disagg blocked on SLURM | Track W5/W16 as blocked; spike E-3 before promising dates. |
| Metric drift between log and JSON | Single parser module (B-2); documented vocabulary (B-1). |
| Thor2 results misread as universal | Run card B1; future AINIC/CX7 rows (B2–B3). |

---

## 9. Suggested order of execution (summary)

1. **C-1, C-2, C-3** — first green path and reproducible run card.  
2. **B-1, B-2, B-4** in parallel with **C-4, C-5**.  
3. **D-1, D-2** once W1 is stable.  
4. **C-6** expand P1 workload set.  
5. **A-1–A-4** when ready to avoid thrashing active labs mid-sprint (or do early with shims).  
6. **E-*, F-*, G-*** per priority after P1 gate is green.

---

## 10. Document control

| Field | Value |
|-------|--------|
| **Location** | `plans/inferencex-atom-cvs-automation-plan.md` |
| **Type** | Planning only — implementation tracked separately in issues/PRs |
| **Related paths** | `cvs/lib/inference/inferencemax_orch.py`, `cvs/lib/inference/bench_serving_metrics.py`, `cvs/lib/dtni/inferencemax_sweep.py`, `cvs/lib/dtni/inferencex_server_scripts/`, `cvs/tests/inference/inferencemax/`, `cvs/input/config_file/inference/inferencemax_single/` |

When implementation lands, update **automation status** columns in §4 and §6 and mark action-item IDs complete in your external tracker.
