<!--
  Planning document only — no implementation in this change.
  Owner: InferenceX / ATOM validation in CVS (pytest + DTNI config + thresholds).
-->

# InferenceX ATOM — CVS automation implementation plan (DTNI-first)

## 1. Purpose and scope

This document is the **implementation and action-item plan** for **InferenceX ATOM** automation in CVS. **Implementation is aligned with the DTNI suite developer guide from the start**, not as a follow-on refactor of the legacy InferenceMax layout.

**Normative reference:** `plans/dtni-dev-guide.md` (phases, `orch`, Job shape, `cvs/input/dtni/<suite>/<variant>/`, `config.json` + `threshold.json`, `load_variant`, `evaluate_all`).

**In scope (InferenceX focus)**

- **IX paths:** vLLM (ROCm) baseline, SGLang (ROCm) baseline, ATOM, ATOM+MTP, ATOM-Disagg (when orchestration allows).
- **Workloads:** InferenceX `amd-master`-style recipes (W1–W16 and priorities in Section 5 of this plan).
- **Metrics:** Throughput per GPU, output throughput per GPU, TTFT/TPOT and percentiles, E2E/prefill where available, sweeps, goodput-related fields, and extensions in Section 5 of this plan.
- **Lab reality:** **Thor2 NIC first**; AINIC / ConnectX-7 as **future variant directories** or metadata when hardware exists.

**Explicitly out of scope for early waves**

- Full **Optimus / KVMGR / NIXL / hipFile / MaaS / Gateway** automation — **Appendix B** only.
- **Legacy-only** growth: new ATOM gates should **not** add net-new behavior to `InferenceBaseJob.verify_inference_results` except where unavoidable shims are required during migration.

---

## 2. DTNI alignment (non-negotiable for new work)

The following are **required** for this program—not optional later refactors:

| DTNI guide concept | InferenceX ATOM application |
|--------------------|-----------------------------|
| **DTNI guide Section 3: Load** | Each variant = `cvs/input/dtni/<suite>/<variant>/config.json` loaded via **`load_variant`** (Pydantic); no ad-hoc `json.load` in the test module. |
| **DTNI guide Section 3: Setup** | Module-scoped **`orch`** fixture: `setup_containers` on entry, `teardown_containers` on exit. **No** ordered `test_launch_*_containers` / `test_cleanup_*` lifecycle tests for this suite. |
| **DTNI guide Section 3: Generated tests** | **`pytest_generate_tests`** in `conftest.py` builds the sweep grid from typed **`benchmark_params`** (`concurrency_levels` × `sequence_combinations`). |
| **DTNI guide Section 3: Workload test** | Construct **`InferenceXJob(orch, variant_config, hf_token)`** (name as implemented), call verbs (`stop`/`start`/`wait_ready`/`run`/`wait_complete` or the agreed minimal set), then **`actuals = job.parse_results()`** — flat string-number dict acceptable if `evaluate_all` normalizes. |
| **DTNI guide Section 3: Verification** | **`evaluate_all(actuals, variant_config.thresholds, prefix=cell.id)`** (or equivalent per the DTNI guide). **Do not** use `InferenceBaseJob.verify_inference_results` as the primary gate for new DTNI variants. |
| **DTNI guide Section 5: Config vs threshold** | **`config.json`** = what to run (image, container, IX repo ref, model, sweep, server script references). **`threshold.json`** = pass/fail only, flat **`{cell.metric: {kind, value, ...}}`** namespace per DTNI guide Section 7. |
| **DTNI guide Section 4: Custom tests** | Liveness, firewall/NIC quirks, **`test_print_results_table`** — use **`orch.exec` / `orch.exec_on_head`**, not raw `phdl` / `docker_lib` in new suite code. |
| **DTNI guide Section 6: Job class** | **`InferenceXJob`** (or agreed name) is a **standalone** class under `cvs/lib/inference/` (or `cvs/lib/dtni/`) that takes **`orch` + typed config**; **does not inherit `InferenceBaseJob`** for the DTNI path. Reuse **parsing helpers** (`bench_serving_metrics`, etc.) as pure functions the Job calls. |
| **DTNI guide Section 8: Porting checklist** | Use the DTNI guide Section 8 porting checklist as the PR template for the first vertical slice (W1). |

**Suite and path naming (implementation detail)**

- Recommended suite folder: `cvs/input/dtni/inferencex_atom_single/<variant>/` (one variant directory per model × mode × purpose, per DTNI guide Section 7 conventions).
- Recommended pytest module: e.g. `cvs/tests/inference/inferencex_atom/inferencex_atom_single.py` with colocated `conftest.py`.
- **`cvs list` / `cvs run`** suite id should match the DTNI suite name chosen in code registration.

**Legacy `inferencemax_single`**

- Existing **`cvs/input/config_file/inference/inferencemax_single/`** and **`InferenceMaxJob`** remain for **backward compatibility** until explicitly deprecated.
- **New** InferenceX ATOM automation and thresholds **land under DTNI**; migration of old variants is **optional** and tracked separately (do not block P1 on porting historical GPT-OSS-only dirs unless required).

---

## 3. Current repository baseline (migration context only)

- **SemiAnalysisAI/InferenceX** is already cloned in legacy flows via `inferencemax_repo`; DTNI **`config.json`** should carry **`inferencex_repo`** (and accept **`inferencemax_repo`** as deprecated alias during transition).
- **Host-mounted** server scripts: `cvs/lib/dtni/inferencex_server_scripts/`; recipe index YAML may stay at `cvs/input/config_file/inference/inferencemax_single/inferencex_atom_recipes.yaml` **or** move under docs / variant `paths`—either is fine if **variants** remain under `cvs/input/dtni/...`.
- **`InferenceMaxJob` / `InferenceBaseJob`** remain in the tree for other suites; **new** code avoids inheriting them for the DTNI InferenceX Job.

---

## 4. Phased implementation strategy (DTNI-first order)

| Phase | Name | Goal | Depends on |
|-------|------|------|------------|
| **0** | **DTNI skeleton** | New suite: `conftest.py` (`orch`, `variant_config`, `pytest_generate_tests`), empty/smoke **`InferenceXJob`**, `load_variant` + Pydantic model for `framework: inferencex_atom_single` (or chosen id), **`evaluate_all`** wired with one smoke predicate. | None |
| **A** | **InferenceX naming + compatibility** | Public names **InferenceX**; JSON keys **`inferencex_repo`** primary, **`inferencemax_repo`** deprecated alias; CLI/docs point at DTNI paths. | 0 partial |
| **B** | **Metrics pipeline** | `parse_results()` merges bench log + optional **`agg_bmk`** JSON via **`bench_serving_metrics`**; output keys match **`threshold.json`** cell metric names. | 0 |
| **C** | **P1 variants** | Thor2 / MI355X: DTNI variant dirs for vLLM + SGLang DSR1 FP8 baselines; ATOM W1, W5, W8, W10; IX revision pinned in **config**. | B partial |
| **D** | **Thresholds + CI** | Full **`threshold.json`** per variant with typed predicates; negative threshold test per DTNI guide Section 9. | C |
| **E** | **P2 workloads + MTP** | Additional variant dirs; MTP = separate CI job if needed; chat-template checklist in variant README. | D |
| **F** | **Multi-node + scaling** | `nnodes` / distributed dimensions as **mode** or variant; scaling metrics + run-card fabric fields. | C + lab |
| **G** | **Disagg + DI stack** | Spike SLURM/docker; Appendix B rows when product integrates. | Infra |

---

## 5. Master table — InferenceX ATOM engine and benchmarks

Authoritative checklist. **Automation status** is planning language.

| # | Category | Test / Metric | Priority | Automation status | Comments | New |
|---|----------|-----------------|----------|---------------------|----------|-----|
| 1 | IX Path | vLLM (ROCm) baseline for parity / cross-compare | P1 | In progress | Same workload cards as ATOM where comparable | No |
| 2 | IX Path | SGLang (ROCm) baseline | P1 | In progress | Second open engine | No |
| 3 | IX Path | ATOM — primary IX serving path | P1 | In progress | `amd-master` / `rocm/atom:*` images | No |
| 4 | IX Path | ATOM + MTP | P1 | Partial | Chat-formatted inputs per IX AGENTS.md | No |
| 5 | IX Path | ATOM-Disagg | P1 | Blocked | PD pools; upstream may require SLURM | No |
| 6 | Workload | W1 — DeepSeek R1 FP8 MI355X atom | P1 | In progress | Primary ATOM gate | No |
| 7 | Workload | W2 — DeepSeek R1 FP4 MI355X atom | P2 | Not started | | No |
| 8 | Workload | W3 — DeepSeek R1 FP8 atom-mtp | P2 | Partial | MTP + chat template | No |
| 9 | Workload | W4 — DeepSeek R1 FP4 atom-mtp | P2 | Not started | | No |
| 10 | Workload | W5 — Qwen 3.5 FP8 MI355X atom | P1 | In progress | | No |
| 11 | Workload | W6 — Qwen 3.5 FP4 MI355X atom | P2 | Not started | | No |
| 12 | Workload | W7 — Qwen 3.5 FP8 atom-mtp | P2 | Partial | | No |
| 13 | Workload | W8 — GLM5 FP8 MI355X atom | P1 | In progress | | No |
| 14 | Workload | W9 — GLM5.1 FP4 MI355X atom | P2 | Not started | | No |
| 15 | Workload | W10 — Kimi K2.5 FP4 MI355X atom | P1 | In progress | | No |
| 16 | Workload | W11 — MiniMax M2.5 FP8 MI355X atom | P2 | Not started | | No |
| 17 | Workload | W12 — MiniMax M2.5 FP4 MI355X atom | P2 | Not started | | No |
| 18 | Workload | W13 — GPT-OSS FP4 MI355X atom | P2 | Partial | | No |
| 19 | Workload | W14 — DeepSeek V4 FP4 MI355X atom | P2 | Not started | gfx950 flags | No |
| 20 | Workload | W15 — DeepSeek V4 FP4 atom-mtp | P2 | Not started | | No |
| 21 | Workload | W16 — DeepSeek V4 FP4 atom-disagg | P2 | Blocked | | No |
| 22 | Performance | Throughput per GPU (`tput_per_gpu`) | P1 | Partial | Map to `evaluate_all` keys | No |
| 23 | Performance | Output throughput per GPU (`output_tput_per_gpu`) | P1 | Partial | | No |
| 24 | Performance | TTFT mean and p99 | P1 | Partial | | No |
| 25 | Performance | TPOT mean and p95 | P1 | Partial | | No |
| 26 | Performance | Prefill latency p50 / p95 | P2 | Partial | | No |
| 27 | Performance | E2E mean / p95 / p99 | P2 | Partial | | No |
| 28 | Performance | Latency vs load (p95/p99 per step) | P2 | Partial | Sweep cells = separate threshold prefixes | No |
| 29 | Performance | Goodput | P2 | Partial | `min_ratio` or assert + metric | No |
| 30 | Performance | Scaling efficiency % | P2 | Not started | | No |
| 31 | Performance | Peak GPU memory | P2 | Not started | | No |
| 32 | Performance | KV cache footprint | P2 | Not started | | No |
| 33 | Performance | Model load time + memory | P2 | Not started | | No |
| 34 | Quality | MTP acceptance / degenerate decode | P2 | Not started | | No |
| 35 | Quality | Quant / logit parity vs BF16 | P2 | Not started | Nightly / optional | No |
| 36 | Reliability / performance | Request success rate and error mix | P1–P2 | Not started | | Yes |
| 37 | Performance | ITL tail beyond p99 | P2 | Not started | | Yes |
| 38 | Performance | Time-to-ready (start → readiness) | P1–P2 | Partial | Liveness assert + optional metric | Yes |
| 39 | Quality / observability | Token accounting sanity | P2 | Not started | | Yes |
| 40 | Performance | Prefill vs decode first-class | P2 | Partial | | Yes |
| 41 | Observability | Steady-state rocm-smi snapshot | P2 | Not started | Phase-4 custom test | Yes |
| 42 | Quality | Output smoke | P2 | Not started | Phase-4 custom test | Yes |
| 43 | Quality (MTP) | MTP health (fallback / disable events) | P2 | Not started | | Yes |
| 44 | Methodology | Warmup discipline | P2 | Not started | Config, not threshold | Yes |
| 45 | Performance | Achieved concurrency / QPS vs target | P2 | Not started | | Yes |
| 46 | Multi-node / scale | Straggler check | P2 | Not started | | Yes |
| 47 | Observability | Server log error bucketing | P2 | Not started | | Yes |

---

## 6. Appendix A — Recipe index (reference)

Keep a **recipe id → upstream script → CVS `*.sh`** map (YAML or doc). Pin **IX git ref** in each variant’s **`config.json`** (not in `threshold.json`).

---

## 7. Appendix B — Deferred DI platform matrix (tracking only)

| # | Category | Test / Metric | Priority | Automation status | Comments | New |
|---|----------|-----------------|----------|---------------------|----------|-----|
| B1 | Lab profile | Run card: NIC = Thor2; document AINIC/CX7 unavailable | P1 | Not started | | Yes |
| B2 | NIC matrix | AINIC when available | P2 | Not started | | Yes |
| B3 | NIC matrix | CX7 when available | P2 | Not started | | Yes |
| B4 | Orchestration | Optimus | P2–P3 | Not started | | Yes |
| B5 | Orchestration | llm-d | P3 | Not started | | Yes |
| B6 | KV Manager | Optimus–KVMGR | P2–P3 | Not started | | Yes |
| B7 | KV (community) | LMCache / HiCache / Mooncake | P3 | Not started | | Yes |
| B8 | KV Transfer | NIXL, MOR-HQ | P2–P3 | Not started | | Yes |
| B9 | KV → storage | hipFile tiering | P3 | Not started | | Yes |
| B10 | Expert parallelism | MoR-EP | P2 | Not started | | Yes |
| B11 | Transport | RCCL on run card | P2 | Partial | | Yes |
| B12 | Transport | rocSHMEM vs RCCL A/B | P3 | Not started | | Yes |
| B13 | GPU matrix | MI300X / MI325X / MI308X / MI4XXX | P2 | Not started | | Yes |
| B14 | GPU kernels | AITER / hipBlasLt toggles | P2 | Not started | | Yes |
| B15 | Gateway | Envoy / Istio / customer | P3 | Not started | | Yes |
| B16 | MaaS | Catalog / auth / policy | P3 | Not started | | Yes |

---

## 8. Action items (detailed, DTNI-first)

### Phase 0 — DTNI skeleton (do first)

| ID | Action | Details |
|----|--------|---------|
| 0-1 | **Register suite in config_loader** | Pydantic models for `inferencex_atom_single` (or chosen id): `schema_version`, `framework`, `gpu_arch`, `paths`, `container`, `benchmark_params` sweep dims, IX clone ref + server script fields. Follow `dtni-dev-guide.md` Section 7 example shape; `extra="forbid"` except documented passthroughs. |
| 0-2 | **Create variant directory layout** | e.g. `cvs/input/dtni/inferencex_atom_single/<full_model_id>_mi355x_atom_perf/config.json` + `threshold.json`. |
| 0-3 | **`conftest.py`** | `pytest_generate_tests` for variants + cells; `orch` fixture with setup/teardown; `variant_config` indirect parametrize via **`load_variant`**. |
| 0-4 | **`InferenceXJob` skeleton** | Constructor `(orch, variant_config, hf_token)`; stubs for verbs; **`parse_results()` → `dict[str, str]`**; all remote I/O via **`orch`**. |
| 0-5 | **`evaluate_all` integration** | Import from `cvs.lib.dtni.verdict` (or current path); workload test calls **`evaluate_all(actuals, thresholds, prefix=cell.id)`** after parse. |
| 0-6 | **Smoke variant** | One cell, minimal thresholds (`smoke_*` per DTNI guide Section 7 anti-pattern avoidance: use **assert** for liveness, **thresholds** for numeric metrics only). |
| 0-7 | **Verification template (DTNI guide Section 9)** | `pytest --collect-only`, `cvs list`, hardware run HTML=0, docker lifecycle check, deliberate threshold fail. |

### Phase A — Naming and compatibility

| ID | Action | Details |
|----|--------|---------|
| A-1 | **Public names** | Suite id **`inferencex_atom_single`** (or chosen), docs, `cvs list` / run_plugin registration. |
| A-2 | **Repo key aliases** | **`inferencex_repo`** in typed config; accept **`inferencemax_repo`** with deprecation warning in loader or Job. |
| A-3 | **Migration doc** | “Legacy `inferencemax_single` vs DTNI `inferencex_atom_single`” for labs. |

### Phase B — Metrics pipeline

| ID | Action | Details |
|----|--------|---------|
| B-1 | **Canonical keys** | Document map: log/JSON field → **`parse_results`** keys → **`threshold.json`** metric suffixes per cell. |
| B-2 | **Implement `parse_results`** | Delegate to **`bench_serving_metrics`**; merge **`agg_bmk`** when present. |
| B-3 | **Predicate coverage** | Use guide kinds: `min`, `max_ms`, `within`, `min_tok_s`, `min_ratio`; extend **`verdict`** only if a new kind is required and review with DTNI owners. |
| B-4 | **Results table** | Phase-4 `test_print_results_table` reading **`inf_res_dict`** keyed by cell id; columns for P1 dashboard metrics. |

### Phase C — P1 variants (Thor2 / MI355X)

| ID | Action | Details |
|----|--------|---------|
| C-1 | **Run card / provenance** | Log or HTML: IX SHA, image tag, variant id, NIC=Thor2, `gpu_arch`, cell id — from **config** + cluster. |
| C-2 | **W1 ATOM** | Full `config.json` + `threshold.json`; host scripts wired via `paths` / container volumes per DTNI container block. |
| C-3 | **vLLM + SGLang baselines** | Separate variant dirs; same sweep cell ids for parity tables. |
| C-4 | **W5, W8, W10** | Same pattern as C-2. |

### Phase D — CI and governance

| ID | Action | Details |
|----|--------|---------|
| D-1 | **Threshold ownership** | Document who updates `threshold.json` on image/kernel bumps. |
| D-2 | **Job split** | Smoke vs sweep vs MTP optional jobs. |

### Phases E–G

| ID | Action | Details |
|----|--------|---------|
| E-1 | **MTP checklist** | Variant README + config flags; flaky → separate CI. |
| E-2 | **P2 variant dirs** | W2, W6–W15 as prioritized. |
| E-3 | **W16 disagg spike** | SLURM vs docker multi-node before automation. |
| F-1 | **Multi-node variants** | Distributed as mode or sibling variant dir per DTNI guide Section 7. |
| F-2 | **Scaling metrics** | Baseline variant + compare; fabric on run card. |
| G-1 | **Phase-4 quality tests** | Output smoke, rocm-smi snapshot, log bucketing — **`orch` only**. |

### Documentation

| ID | Action | Details |
|----|--------|---------|
| DOC-1 | **Cross-link** | This plan + `dtni-dev-guide.md` from `docs/reference/configuration-files/` (new or existing InferenceX page). |
| DOC-2 | **W1–W16 legend** | In docs next to DTNI threshold examples. |

---

## 9. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| **Pydantic / loader gap** | Phase 0-1 lands minimal model first; extend fields incrementally (DTNI guide Section 8, steps 3–4). |
| **Duplication with legacy Job** | Factor **pure** command-build + parse helpers shared by legacy and `InferenceXJob`; **no** new inheritance from `InferenceBaseJob`. |
| MTP flakes | Separate CI job (D-2). |
| Disagg blocked | E-3 before W16 promises. |
| Metric / threshold key drift | B-1 + code review rejects thresholds in `config.json`. |

---

## 10. Suggested execution order

1. **0-1 … 0-7** — DTNI shell + smoke variant green.  
2. **C-2** (W1) + **B-1 … B-4**.  
3. **C-3, C-4** + **D-1**.  
4. **A-1 … A-3** in parallel once suite id is stable.  
5. **E-*, F-*, G-*** per priority.

---

## 11. Document control

| Field | Value |
|-------|--------|
| **Location** | `plans/inferencex-atom-cvs-automation-plan.md` |
| **Normative DTNI reference** | `plans/dtni-dev-guide.md` |
| **Primary inputs** | `cvs/input/dtni/inferencex_atom_single/<variant>/config.json`, `threshold.json` |
| **Supporting assets** | `cvs/lib/dtni/inferencex_server_scripts/`, recipe YAML, `cvs/lib/inference/bench_serving_metrics.py`, `cvs/lib/dtni/verdict.py` (`evaluate_all`) |
| **Legacy (non-primary)** | `cvs/input/config_file/inference/inferencemax_single/`, `InferenceMaxJob`, `InferenceBaseJob` |

Update **Sections 5 and 7** of this plan (master table and Appendix B) automation status as variants land; use **Section 8** action-item IDs in PR descriptions.
