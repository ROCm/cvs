# InferenceMax / InferenceX (IX ATOM): validation rollout plan

**How this doc uses “phases”:** Timeline phases are **ordering buckets** (1, 2, 3, 4, …). The file spells out **Phases 1 to 2** in order, then **Phase 3** (advanced validation: longevity, accuracy, MTP/disagg), **Phase 4** (pipeline / ops / hardening), and **Phase 5+** if the backlog still won’t fit.

**First tranche:** priority goes to work that is **actually important** and **actually doable** on the clusters, images, and available time. A lot of that shows up as **P1** on the matrix, but that is not dogma. A **P2** row can move up when it blocks the gate that matters; a **P1** row can stay parked when multinode or tooling simply is not there yet.

**Matrix columns:** The tables stick to **P1** / **P2** the same way the InferenceMax / IX sheets in use are laid out (**P1** = gate type stuff, **P2** = breadth / heavier metrics / evalish rows). Some tabs also have **P0**, **P3**, weird one-off labels. Trace / the workload map should mirror what the sheet says rather than squashing everything into P1/P2. **Timeline phase ≠ matrix P1.** That distinction is easy to blur, so it stays spelled out up front.

**Scope:** **CVS AI cluster validation** for InferenceMax and InferenceX (IX ATOM). Same universe as the **CVS validation pipeline** inference workflow and the matrices tracked in sheets.  
**Starting point:** **8 GPUs**, single node first; Phase 2 is where multinode and vLLM/SGLang/ATOM parity land for real. Running through **ROCm + cvs** (and the internal configs/manifest) is the default path. **Tests and metrics** sit under each phase below; **`cvs run` names, configs, tickets** land in **Trace** as rows get automated.



## Contents

| # | Section |
|---|---------|
|   | [Matrix and timeline](#matrix_summary) |
|   | [Timeline overview](#timeline_high_level) |
| 1 | [Phase 1 goals and stages](#phase1_goals) |
| 1 | [Phase 1 order of work](#phase1_calendar) |
| 1 | [Phase 1 tests](#phase1_tests) |
| 1 | [Phase 1 exit checklist](#phase1_exit) |
| 2 | [Phase 2 goals and stages](#phase2_goals) |
| 2 | [Phase 2 order of work](#phase2_calendar) |
| 2 | [Phase 2 tests](#phase2_tests) |
| 2 | [Phase 2 exit checklist](#phase2_exit) |
| 3 | [Phase 3 advanced validation](#phase3_advanced) |
| 4 | [Phase 4 pipeline and hardening](#phase4_pipeline) |
| 5+ | [Phase 5+ overflow](#phase5_overflow) |
|   | [Traceability](#traceability_section) |


<a id="matrix_summary"></a>

## Matrix priority and timeline phases (summary)

| Matrix / sheet column | What it often means | How it maps to timeline (typical) |
|----------------------|---------------------|-----------------------------------|
| **P0** (or “everything stops until this is fixed”) | Blocks ship / program until cleared | **Earliest** timeline phase that has dependencies met, often same window as Phase 1 if it is infra/config, otherwise as soon as unblocked. |
| **P1** | Must-have for product gate: smoke, primary heroes, required multinode, key perf SLOs, IX `atom` + parity baselines | **Phase 1** = **smallest achievable slice** of P1 (single node + one IX `atom` + IX **P1** `agg_bmk` fields in this plan). **Remaining P1** continues in **Phase 2+** (multinode, vLLM/SGLang parity, second heroes, goodput / load). |
| **P2** | Secondary: FP4 variants, extra models, longevity, “extra” perf rows, HF only rows, accuracy suites, MTP/disagg recipes | **Phase 2** (late) and/or **Phases 3 to 5**, see phase test tables below. |
| **P3+** / deferred / other | Stretch, nice-to-have, or non-numeric columns on some tabs | **Phase 5+** or unscheduled; extra phase headings help when the backlog stays large. |

**Live sheet sync:** Whenever **any** priority column or row on the workbook changes, the phase tables here **are updated** on the same cadence. Common splits: **IX** MTP *framework* may stay **P1** while **`*-atom-mtp` recipes** stay **P2**; **IX goodput** is often **P2** while **InferenceMax goodput** may be **P1**. This doc is meant to track **what the sheet actually says**.

<a id="timeline_high_level"></a>

## Timeline phases (high level)

| Timeline phase | Name | Matrix coverage (typical) |
|----------------|------|---------------------------|
| **0** | Preconditions | Access, environments, naming (InferenceMax vs InferenceX), matrix row IDs ↔ configs. |
| **1** | Foundation (below) | **Smallest high-importance slice to land here:** InferenceMax smoke + multi GPU on **1×8 GPU** hero + one IX **`atom`** recipe + **IX P1 class** `agg_bmk` fields (`tput_per_gpu`, `output_tput_per_gpu`, `mean_ttft`/`p99_ttft`, `mean_tpot`/`p95_tpot`) + runbook. Rows that stay **open** after this slice (including remaining **P1** that need multinode/parity) roll to **Phase 2+**. |
| **2** | Breadth (below) | **Remaining gate items** (often **P1**): multinode serving, second heroes (e.g. R1 AITER, Mixtral), **vLLM + SGLang + ATOM** parity card, goodput / latency vs load. **P2** *early:* extra IX `atom` FP4 recipes, InferenceMax **P2** chat/perf rows as capacity allows. |
| **3** | Advanced validation (below) | Longevity (matrix continuous run), **P1/P2** accuracy suites, IX **MTP + disagg + quality** depth, quant/logit parity where the matrix asks for it. |
| **4** | Pipeline & hardening, **overlaps late Phase 3 OK** (below) | **Slurm**, **GPU idle** policy, **expanded manifest**, automation / matrix export so Phases 1 to 3 are rerunnable without heroics. |
| **5+** | Overflow | Anything that still does not fit (HF only depth, extra MTP recipes, **P3+** rows, second pass accuracy), **Phase 5+** table in this doc instead of overfilling Phase 3. |



<a id="phase1_goals"></a>

## Timeline Phase 1: goals (G1 to G5)

| ID | Goal |
|----|------|
| G1 | One **reliable smoke** path: model load + short generation + clear pass/fail and artifacts. |
| G2 | One **TP=8 hero workload** on a single node for **InferenceMax**; whatever row is the **main gate** for this push (**usually P1 on the sheet**): Qwen 3.5 397B FP8 unless another row fits the matrix better. |
| G3 | One **ATOM** recipe from IX matrix (plain `atom`, not MTP/disagg): **`qwen3.5-fp8-mi355x-atom`** *or* **`dsr1-fp8-mi355x-atom`** (one for the first tranche; a second is optional once the first is stable). |
| G4 | **InferenceMax:** throughput, TTFT, TPOT (mean + percentiles as tooling allows). **IX ATOM:** all **matrix P1** `agg_bmk.json` fields in Stage 1.4 (`tput_per_gpu`, `output_tput_per_gpu`, `mean_ttft`/`p99_ttft`, `mean_tpot`/`p95_tpot`). **Goodput** when load path exists (**matrix P1** on InferenceMax sheet), often **timeline Phase 2** if not ready before Phase 1 closes. |
| G5 | **Rerunnable** without hand-holding: documented `cvs run` / workflow inputs, config paths, and where logs/HTML/metrics land (Teams/S3 when the pipeline is hooked up). |



<a id="phase1_stages"></a>

## Timeline Phase 1: stages (what “done” means)

**What Phase 1 is really doing:** The goal is **not** to clear every “top priority” cell in the matrix in this phase alone. It is a **small vertical slice** that still counts (**often P1**, not always, see above). Anything else, including **P1** rows that want multinode or extra engines first, **slides to Phase 2+**; those rows do not vanish just because they are out of scope here.

### Stage 1.1: Preconditions & traceability

- [ ] **Phase 1 scope is written down and agreed** with whoever needs to care (row IDs / recipe ids **and** the sheet’s priority columns, **P1/P2/P0/…**).  
- [ ] **Workload map** (one table): `cvs run` name ↔ `config.json` path ↔ cluster topology ↔ IX recipe id (if applicable).  
- [ ] **GitHub Environment** + secrets for Inference workflow (or lab equivalent) are confirmed.  
- [ ] **Container image** / Artifactory rule for `ROCM_VERSION` for the suites in scope is confirmed.

**Exit:** An existing inference suite runs manually without guessing paths.



### Stage 1.2: Smoke (InferenceMax **matrix P1**)

- [ ] **Smoke** row: load + generate, single node, TP as required by model (often TP=8).  
- [ ] Failures are **actionable** (logs, container name, last command).  
- [ ] **Automation status** is updated on the sheet for the smoke row (e.g. `In CVS, manual dispatch` or team convention).

**Exit:** Smoke is the **first gate** for every subsequent change.



### Stage 1.3: Hero workload (InferenceMax **matrix P1**, single node)

- [ ] **`cvs run`** + Tier 3 config for **Qwen 3.5 397B-A17B FP8, 1 node, TP=8** is implemented or stabilized (or another **P1 single node** row from the matrix if Qwen is not the target row).  
- [ ] **Throughput**, **TTFT**, **TPOT** are parsed and recorded (mean first; percentiles as tooling allows).  
- [ ] Optional: **goodput** is in scope when the client supports success/failure accounting in the timeline Phase 1 window.

**Exit:** One non-smoke **matrix P1** workload is **repeatable** on the 8-GPU node class.



### Stage 1.4: IX ATOM baseline (**matrix P1**, one plain `atom` recipe + **matrix P1** perf fields)

- [ ] **`atom`** path has been run for **one** P1 recipe: `qwen3.5-fp8-mi355x-atom` **or** `dsr1-fp8-mi355x-atom` per `amd-master.yaml` / team procedure.  
- [ ] **IX matrix P1, performance (InferenceX aggregates):** for that recipe’s benchmark output (**`agg_bmk.json`** or documented equivalent), **all** of the following are captured and surfaced. If a field is missing from the JSON, **“N/A + reason”** is documented and a follow-up is filed; **matrix P1** perf rows are not dropped silently.

| Matrix **P1** | Metric | `agg_bmk.json` (or agg) field(s) |
|---------------|--------|-----------------------------------|
| Throughput per GPU, total tokens/sec | Total tok/s per GPU | `tput_per_gpu` |
| Output throughput per GPU, decode tok/s per GPU | Decode tok/s per GPU | `output_tput_per_gpu` |
| TTFT, mean & p99 (ms) | Time to first token | `mean_ttft`, `p99_ttft` |
| TPOT, mean & p95 (ms) | Time per output token | `mean_tpot`, `p95_tpot` |

- [ ] Any **aliases** or nested paths (if the file structure differs) are mapped in the **workload map** / runbook.  
- [ ] IX sheet **Automation** reflects the rows in play.

**Exit:** One ATOM recipe **runs end-to-end** with the **four matrix P1 performance metric groups** above **either in the report or called out as N/A with a ticket**, same bar as InferenceMax.



### Stage 1.5: Hardening & handoff

- [ ] **Runbook** (short, skimmable): how to trigger, overrides (`cluster_file` / `config_file`), cleanup, Teams on/off.  
- [ ] **One controlled failure** test (tight threshold or bad config) to validate signal.  
- [ ] Short **retro**: gaps for **timeline Phase 2** (second recipe, parity card, multinode) captured in backlog.

**Exit:** Phase 1 is rerunnable from the runbook alone (no shadow support).



<a id="phase1_calendar"></a>

## Order of work: timeline Phase 1

**Step numbers** are dependency order only; they are **not** tied to a rigid per-step calendar schedule.

### First stretch

| Step | Focus | Deliverables |
|-----|--------|----------------|
| **1** | Preconditions | Workload map draft; environment/image variables listed; blockers filed. |
| **2** | Preconditions + smoke start | Smoke config path is known; first smoke attempt is made; infra blockers are tracked to resolution. |
| **3** | Smoke | Smoke **passes** reliably; sheet row is current; minimal runbook section exists (smoke only). |
| **4** | Hero workload | Hero `cvs run` + config are wired; first full run is complete (parse/metrics gaps are OK on first try). |
| **5** | Hero workload | Second iteration: stable **pass** OR a documented **single open issue** with a named owner; core metrics appear in summary/logs. |

**After the first stretch:** **G1 + G2** in progress or met; **G5** started (runbook skeleton).

### Second stretch

| Step | Focus | Deliverables |
|-----|--------|----------------|
| **6** | Hero metrics | TTFT/TPOT/throughput aligned to matrix wording; thresholds or “report-only” documented. |
| **7** | IX ATOM | First ATOM recipe run (same hero model family if possible); field mapping doc. |
| **8** | IX ATOM | Repeatable ATOM run; **all timeline Phase 1 IX matrix P1 `agg_bmk` fields** (`tput_per_gpu`, `output_tput_per_gpu`, `mean_ttft`/`p99_ttft`, `mean_tpot`/`p95_tpot`) captured or N/A documented; IX sheet automation column updated. |
| **9** | Goodput / polish | Goodput is in scope where applicable; otherwise there is an explicit **timeline Phase 2** note; flaky teardown/cleanup is addressed. |
| **10** | Hardening | Controlled failure check; runbook complete; **G1 to G5** checklist sanity-checked with a reviewer or pair partner. |

**Phase 1 closure bar:** **G1 to G5** done for the **Phase 1 scope** that was committed (the important stuff that fits this slice; **usually P1**); **Phase 2+** backlog is written down with **agreed owners** (see **Timeline Phase 2** below).



<a id="phase1_tests"></a>

## Timeline Phase 1: tests & metrics (deliver in Phase 1)

**Trace:** each row below has `cvs run` name, config path, or ticket in **Trace** once wired. Everything else stays in later phases unless it is **explicitly** in scope for Phase 1.

### InferenceMax

Section order mirrors **DTNI Validation Tracker → InferenceMax**: **Inference frameworks** (#1–4 on the tab) first, then **Workload coverage** (ROCm / HF blocks), then **Performance benchmark metrics** as they apply to this phase.

#### Inference frameworks (InferenceMax tab #1–3 in scope here)

| Area | Item | Matrix | Trace |
|------|------|--------|-------|
| Framework | Smoke, model loads and generates | P1 | |
| Framework | Multi-GPU serving (multi GPU on **one** node) | P1 | Met when hero runs **TP=8** on 8-GPU node |

#### Workload + perf (Phase 1 slice; **IM Matrix** workload row when wired)

| Area | Item | IM | Matrix | Trace |
|------|------|----|--------|-------|
| Workload (ROCm) | Chat infer, Qwen 3.5 397B-A17B FP8, 1 node, TP=8 | W12 | P1 | |
| Workload (ROCm) | *(optional)* Smoke chat, Llama 3.1 8B FP8 1 GPU | W4 | P2 | |
| Perf | TTFT, p50 / p95 (ms) | — | P1 | |
| Perf | TPOT, p50 / p95 (ms) | — | P1 | |
| Perf | Global throughput, generated tokens/s | — | P1 | |
| Perf | Per-GPU throughput, tokens/s/GPU | — | P1 | |

**Not in Phase 1 (see later timeline phases):** Multinode framework row (**InferenceMax** tab #3; **IM Matrix** is separate from **W*** workload IDs); **Longevity** framework row (**tab #4**, usually **timeline Phase 3**); DeepSeek R1 AITER / Mixtral / other ROCm chat rows; all HF workloads; E2E / latency vs load / goodput / scaling / memory / HBM / MFU / quant parity / all accuracy; unless something is **forced into Phase 1** because it is on fire.

### IX ATOM

Section order mirrors **DTNI Validation Tracker → IX ATOM**: **InferenceX framework paths** (tab #1–5) first, then **Workload coverage** (recipes), then **Performance benchmark metrics (InferenceX aggregates)** as they apply to this phase.

#### InferenceX framework path (Phase 1 slice)

| Area | Item | Matrix | Trace |
|------|------|--------|-------|
| Framework | **ATOM**, `atom` (plain path only) | P1 | |

#### Workload + agg_bmk (Phase 1 slice; **IX ATOM Matrix** **W** when wired)

| Area | Item | IX | Matrix | Trace |
|------|------|-----|--------|-------|
| Recipe | **`qwen3.5-fp8-mi355x-atom`** *or* **`dsr1-fp8-mi355x-atom`** (one) | W5 / W1 | P1 | |
| Perf | Throughput per GPU, `tput_per_gpu` | — | P1 | |
| Perf | Output throughput per GPU, `output_tput_per_gpu` | — | P1 | |
| Perf | TTFT mean & p99, `mean_ttft`, `p99_ttft` | — | P1 | |
| Perf | TPOT mean & p95, `mean_tpot`, `p95_tpot` | — | P1 | |

**Not in Phase 1 (see later timeline phases):** vLLM / SGLang / MTP / disagg framework rows (**IX ATOM** tab #1–2, #4–5); second `atom` recipes (`glm5-*`, `kimi*`, alternate DSR1/Qwen); all FP4 `atom` recipes; IX prefill / E2E / latency curve / goodput / scaling / peak mem / KV / load-time metrics beyond the four P1 `agg_bmk` fields above.

**IX workload legend (for later phases):** Recipe **W1–W16** matches **IX ATOM Matrix** and **Timeline Phase 2** above (`W1`=`dsr1-fp8-mi355x-atom` … `W16`=`dsv4-fp4-mi355x-atom-disagg`). Full cross-product **Y/P** cells vs benchmark rows stay on the **spreadsheet**; the same perf fields are repeated for each recipe as it is promoted across **later timeline phases**.



<a id="phase1_exit"></a>

## Success checklist: end of timeline Phase 1

- [ ] Smoke green on target cluster class.  
- [ ] Hero InferenceMax workload green with **core metrics** visible.  
- [ ] One **ATOM** **matrix P1** recipe green with **documented** `agg_bmk.json` mapping for **matrix P1** IX performance: `tput_per_gpu`, `output_tput_per_gpu`, `mean_ttft` + `p99_ttft`, `mean_tpot` + `p95_tpot` (each present in artifacts/report or N/A + reason).  
- [ ] Runbook + sheet automation fields updated for completed rows.  
- [ ] **Phase 2+** scope is nailed down (**who / when** for multinode + parity, even if the dates slip).



<a id="phase2_goals"></a>

## Timeline Phase 2: goals (G6 to G9)

Build on Phase 1: **what’s left on the gate side** (usually a lot of **P1**: multinode, parity baselines, extra heroes) plus **some P2 breadth** when capacity allows. The heavy **P2** stuff (accuracy suites, MTP, disagg, longevity soaks) mostly waits for **Phases 3 to 5** unless tooling lands early enough for an early pull into an earlier phase.

| ID | Goal |
|----|------|
| G6 | **Second vertical slice:** second InferenceMax **matrix P1** chat row *and/or* second IX **matrix P1** `atom` recipe (e.g. `glm5-fp8-mi355x-atom`, `kimik2.5-fp4-mi355x-atom`, or alternate Qwen/DSR1). |
| G7 | **Multinode matrix P1:** at least one InferenceMax multinode row (e.g. DeepSeek-V3 2+ nodes, 405B multinode, Llama 70B multinode **P2**) **or** multinode IX, highest priority once `CLUSTER_MULTI_NODE_IPS` and the cluster are actually there. |
| G8 | **Parity card (IX matrix P1):** one frozen **parity set ID**, same model recipe / ISL to OSL / concurrency, run on **vLLM (ROCm)**, **SGLang (ROCm)**, and **ATOM** with results in one comparable report (metrics + run links). |
| G9 | **Reliability / load (matrix P1 / P2 mix):** **goodput** (often **matrix P1** on InferenceMax sheet) and/or **latency vs load** (**matrix P1** on some rows, **P2** elsewhere) on at least one hero workload where the client supports it. |



<a id="phase2_stages"></a>

## Timeline Phase 2: stages (what “done” means)

### Stage 2.1: Preconditions (multinode + parity)

- [ ] **Multinode:** cluster file template + secrets for **multinode IP list**; firewall / NIC notes per team runbook.  
- [ ] **Parity:** **parity set ID** v1 is documented: model id, weights revision, tokenizer/chat template, ISL, OSL, concurrency grid, precision, TP/EP.  
- [ ] **Manifest / workflow:** Inference category includes multinode suites; optional `cluster_file` override documented for topology-specific runs.

**Exit:** A **2+ node** run can be scheduled without ad hoc SSH; parity inputs are frozen in version control or the runbook.



### Stage 2.2: Second InferenceMax slice

- [ ] **`cvs run` + config** exists for the second P1 workload (e.g. **DeepSeek R1 FP8 TP=8 AITER MLA**, **Mixtral 8×7B FP8 TP+EP**, or whatever the **next P1 row** is on the plan).  
- [ ] **Metric parsing / thresholds** cover any new fields that row exposes.  
- [ ] **InferenceMax** sheet automation reflects the new rows.

**Exit:** Two distinct InferenceMax P1 workloads green on **single node** (or first multinode if Stage 2.3 is parallelized).



### Stage 2.3: Multinode serving (InferenceMax P1)

- [ ] **One** P1 **multinode** inference row is implemented or stabilized (see matrix: e.g. DeepSeek-V3 TP+EP multinode, Llama 405B FP8-KV multinode).  
- [ ] **Scaling efficiency** or **global throughput** capture vs 1× reference is validated if the matrix requires it (report-only is acceptable until a baseline exists).  
- [ ] **Node roles**, rendezvous failures, and cleanup for multinode teardown are documented.

**Exit:** One multinode P1 scenario passes with artifacts; sheet updated.



### Stage 2.4: IX ATOM breadth + parity baselines

- [ ] A **second `atom` recipe** is in scope (e.g. `glm5-fp8-mi355x-atom`, `kimik2.5-fp4-mi355x-atom`, or the alternate of Qwen vs DSR1 from **timeline Phase 1**).  
- [ ] **vLLM (ROCm)** and **SGLang (ROCm)** have been run on the **same parity card** as ATOM for that recipe (time boxed; partial metrics are OK if documented).  
- [ ] IX **agg_bmk** mapping covers any new fields (prefill p50/p95 if in scope).

**Exit:** Parity card v1 exists with three engine columns or explicit “blocked on engine X” with ticket link.



### Stage 2.5: Load curves & goodput

- [ ] A **concurrency or QPS sweep** exists for one hero (InferenceMax and/or IX per matrix).  
- [ ] **Goodput** (successful / total) under load is recorded for that scenario.  
- [ ] Optional: **latency vs load** P95/P99 per step (**matrix P1** on some InferenceMax rows, **matrix P2** on others) when the timeline allows.

**Exit:** At least one workload has a **load-shaped** result artifact; matrix rows updated or explicitly deferred to **timeline Phase 3** with reason.



### Stage 2.6: Hardening & handoff

- [ ] **Runbook** is updated: multinode, parity triggers, where to find comparison tables.  
- [ ] **Retro:** next timeline phase(s), **IX** `*-atom-mtp` / **atom-disagg**, **P2 class** perf depth (HBM/MFU, KV, cold start), **P2 class** chat/HF rows, **accuracy** suites, **longevity** (matrix class), **pipeline / Slurm / manifest** hardening.

**Exit:** G6 to G9 are either **done** or honestly **descoped** with notes; **Phases 3 to 5** backlog is ordered (split across **Phase 3** / **Phase 4** / **Phase 5+** headings in this doc so standups stay legible).



<a id="phase2_calendar"></a>

## Suggested order of work: timeline Phase 2

**Step numbers** are dependency order only; block sizing **stretches or shrinks** with how painful multinode + parity are in practice. Nothing here implies a fixed calendar mapping per step.

### Phase 2, first stretch

| Step | Focus | Deliverables |
|-----|--------|----------------|
| **1** | 2.1 Multinode + parity prep | Cluster multinode template; parity set ID doc v1. |
| **2** | 2.2 Second InferenceMax slice | Second workload first run is done; config/TP/EP issues are closed out. |
| **3** | 2.2 / 2.3 | Second workload stable **or** first multinode dry-run (infra). |
| **4** | 2.3 Multinode | First full multinode **matrix P1** pass **or** documented blocker + owner. |
| **5** | 2.3 / 2.4 | Multinode repeatability is shown **or** ATOM second recipe plus vLLM baseline on the parity card is underway. |

**After the first stretch:** G6 in progress or met; G7 started or multinode unblocked.

### Phase 2, second stretch

| Step | Focus | Deliverables |
|-----|--------|----------------|
| **6** | 2.4 Parity | vLLM + SGLang runs on parity card; results table draft. |
| **7** | 2.4 ATOM on same card | ATOM column is complete; gaps are logged. |
| **8** | 2.5 Goodput / sweep | Load sweep + goodput on hero (InferenceMax and/or IX). |
| **9** | 2.5 Polish | Latency vs load when in scope; flaky runs are addressed. |
| **10** | 2.6 Hardening | Runbook v2; controlled failure drill on multinode or parity path when policy allows; backlog for **Phases 3 to 5**. |

**Optional third stretch:** buffer for multinode or parity slips; second multinode row if required.

**Phase 2 closure bar:** **G6 to G9** are met **or** gaps are written down with owners (including what is **not** being done); **Phases 3 to 5** backlog (**P2 class** + MTP/disagg + accuracy + longevity + pipeline hardening, split across **Phase 3 / 4 / 5+** in this doc) is **ready to execute**, not just vibes.



<a id="phase2_tests"></a>

## Timeline Phase 2: tests & metrics (deliver in Phase 2)

**Trace:** each row below has `cvs run` name, config path, or ticket in **Trace** when the row is wired. **InferenceMax** tab lists **Inference frameworks** (#1–4) separately from **Workload coverage**; **`IM Matrix`** assigns **W1–W18** to workload rows only (not to the four framework lines). **IX ATOM** tab lists **InferenceX framework paths** (#1–5) separately from **Workload coverage**; **`IX ATOM Matrix`** assigns **W1–W16** to recipes. Cross-product **Y/P** cells on **IM Matrix** / **IX ATOM Matrix** stay authoritative; the grid is exported or linked when cells are automated.

### InferenceMax (Phase 2 target; aligned to tracker tabs)

Block order matches **InferenceMax** + **`IM Matrix`**: frameworks → ROCm workloads (**W1–W12**) → HF workloads (**W13–W18**) → performance benchmark metrics.

#### Inference frameworks (InferenceMax tab #1–4)

| Item | Tab # | Matrix | Trace |
|------|-------|--------|-------|
| Smoke test, model loads and generates | 1 | P1 | |
| Multi-GPU serving, multiple GPUs on one node | 2 | P1 | |
| Multi-Node serving, 2+ physical nodes | 3 | P1 | |
| Longevity, 24 h continuous serve (no crash / leak) | 4 | P2 | Usually **timeline Phase 3**; still the same tracker row for traceability |

#### Workload coverage — AMD ROCm (**IM Matrix** W1–W12)

| Item | IM | Matrix | Trace |
|------|----|--------|-------|
| Throughput, Llama 3.1 70B FP8-KV TP=8 in=128 out=2048 max_seqs=3200 | W1 | P2 | |
| Throughput, Llama 3.1 70B FP8-KV in=2048 out=2048 max_seqs=1500 | W2 | P2 | |
| Latency sweep, Llama 3.1 70B FP8-KV batch 1 to 128 | W3 | P2 | |
| Smoke chat, Llama 3.1 8B FP8 1 GPU ROCm | W4 | P2 | |
| Chat infer, DeepSeek R1 FP8 TP=8 AITER MLA | W5 | P1 | |
| Chat infer, DeepSeek-V3 2+ nodes TP=16 EP=16 | W6 | P1 | |
| Chat infer, Llama 3.1 405B FP8-KV multinode TP=8 | W7 | P1 | |
| Chat infer, Mixtral 8x7B FP8 TP+EP | W8 | P1 | |
| Chat infer, Llama 3.3 70B FP8 TP=8 | W9 | P2 | |
| Chat infer, Llama 4 Scout / Maverick MoE TP=8 | W10 | P2 | |
| Chat infer, DeepSeek-V4-Flash FP4+FP8 1 node TP=8 | W11 | P2 | |
| Chat infer, Qwen 3.5 397B-A17B FP8, 1 node, TP=8 | W12 | P1 | |

#### Workload coverage — HuggingFace (**IM Matrix** W13–W18)

| Item | IM | Matrix | Trace |
|------|----|--------|-------|
| Smoke chat, Llama 3.1 8B Instruct 1 GPU (HF path) | W13 | P2 | |
| Traffic, LLM-Perf baseline 256 / 64 | W14 | P2 | |
| Chat infer, Llama 3.1 70B multinode TP (HF) | W15 | P2 | |
| Cross-framework parity vs vLLM ROCm | W16 | P2 | |
| Long prefill, Llama 3.1 8B 32k in / 256 out | W17 | P2 | |
| Ecosystem, Gemma 3 8B + Qwen2.5 72B | W18 | P2 | |

#### Performance benchmark metrics (InferenceMax tab #23–42; **IM Matrix** metric rows)

Row order matches the **InferenceMax** performance block; **Y/P** per workload stays on **IM Matrix**.

| Item | Matrix | Trace |
|------|--------|-------|
| TTFT (Time to First Token), p50 / p95 (ms) | P2 | |
| TPOT (Time Per Output Token), p50 / p95 (ms) | P2 | |
| Prefill latency, p50 / p95 (ms) | P2 | |
| Normalized TTFT, ms / input token | P2 | |
| P99/P50 decode latency ratio | P2 | |
| Queue wait, p50 / p95 (ms) | P2 | |
| End-to-end request latency, P50 / P90 / P95 / P99 | P1 | |
| Latency vs load, P95 & P99 per QPS step | P1 | |
| Global throughput, generated tokens/s | P2 | |
| Decode throughput, tokens/s (p50) | P2 | |
| Per-GPU throughput, tokens/s/GPU | P2 | |
| Max concurrent requests @ target P95 latency | P2 | |
| Scaling efficiency % | P2 | |
| Peak GPU memory (MB) | P2 | |
| KV cache footprint (GB) | P2 | |
| Model load time + load memory | P2 | |
| GPU memory bandwidth util % | P2 | |
| GPU compute util % (MFU / TFLOPS) | P2 | *(often slips to a later phase if tooling lags)* |
| Goodput, successful / total under load | P1 | |
| Quantization output parity vs BF16 | P2 | |

### IX ATOM (Phase 2 target; aligned to tracker tabs)

Block order matches **IX ATOM** + **`IX ATOM Matrix`**: InferenceX framework paths → `amd-master` recipes (**W1–W16**) → performance benchmark metrics (aggregates).

#### InferenceX framework paths (IX ATOM tab #1–5)

| Item | Tab # | Matrix | Trace |
|------|-------|--------|-------|
| vLLM (ROCm), baseline parity | 1 | P1 | |
| SGLang (ROCm), baseline parity | 2 | P1 | |
| ATOM, `atom` (AiTer Optimized Model path) | 3 | P1 | |
| ATOM + MTP, `*-atom-mtp` recipes | 4 | P1 | |
| ATOM-Disagg, `atom-disagg` | 5 | P1 | |

#### Workload coverage — InferenceX ATOM recipes (**IX ATOM Matrix** W1–W16)

| Item | IX | Matrix | Trace |
|------|-----|--------|-------|
| `dsr1-fp8-mi355x-atom` | W1 | P1 | |
| `dsr1-fp4-mi355x-atom` | W2 | P2 | |
| `dsr1-fp8-mi355x-atom-mtp` | W3 | P2 | |
| `dsr1-fp4-mi355x-atom-mtp` | W4 | P2 | |
| `qwen3.5-fp8-mi355x-atom` | W5 | P1 | |
| `qwen3.5-fp4-mi355x-atom` | W6 | P2 | |
| `qwen3.5-fp8-mi355x-atom-mtp` | W7 | P2 | |
| `glm5-fp8-mi355x-atom` | W8 | P1 | |
| `glm5.1-fp4-mi355x-atom` | W9 | P2 | |
| `kimik2.5-fp4-mi355x-atom` | W10 | P1 | |
| `minimaxm2.5-fp8-mi355x-atom` | W11 | P2 | |
| `minimaxm2.5-fp4-mi355x-atom` | W12 | P2 | |
| `gptoss-fp4-mi355x-atom` | W13 | P2 | |
| `dsv4-fp4-mi355x-atom` | W14 | P2 | |
| `dsv4-fp4-mi355x-atom-mtp` | W15 | P2 | |
| `dsv4-fp4-mi355x-atom-disagg` | W16 | P2 | |

#### Performance benchmark metrics (InferenceX aggregates; IX ATOM tab #22–33)

| Item | Matrix | Trace |
|------|--------|-------|
| Throughput per GPU, `tput_per_gpu` | P1 | |
| Output throughput per GPU, `output_tput_per_gpu` | P1 | |
| TTFT mean & p99, `mean_ttft`, `p99_ttft` | P1 | |
| TPOT mean & p95, `mean_tpot`, `p95_tpot` | P1 | |
| Prefill latency, p50 / p95 | P2 | |
| E2E latency, mean / p95 / p99 | P2 | |
| Latency vs load | P2 | |
| Goodput | P2 | |
| Scaling efficiency % | P2 | |
| Peak GPU memory | P2 | |
| KV cache footprint | P2 | |
| Model load time + memory | P2 | |



<a id="phase2_exit"></a>

## Success checklist: end of timeline Phase 2

- [ ] Second InferenceMax **matrix P1** workload **or** second IX **matrix P1** `atom` recipe green (G6).  
- [ ] At least **one** multinode **matrix P1** scenario green **or** signed waiver with new date (G7).  
- [ ] Parity card v1: vLLM + SGLang + ATOM on same frozen inputs, with comparable metrics table (G8), **IX matrix P1** framework rows.  
- [ ] **Goodput** and/or **latency vs load** on at least one hero (G9), aligns **InferenceMax matrix P1** goodput where applicable.  
- [ ] Runbook v2; **Phases 3 to 5** backlog (see headings below) is ordered enough to actually execute, not a vague promise that evals happen later without a plan.



<a id="phase3_advanced"></a>

## Timeline Phase 3: advanced validation

> **Focus:** Long runs, **accuracy / quality** suites, and IX **MTP + disagg + parity** depth. The stuff that didn’t belong in Phase 2 because it burns calendar or needs fragile stacks stable first.

**When:** After Phase 2 exit checklist is green **or** honestly waived with owners/dates. Phase 3 stays a **bounded first pass**; when longevity and full accuracy land together, work can be spread across nodes or rows can shift to **Phase 5+**.

### Goals (G10 to G12)

| ID | Goal | “Done” looks like |
|----|------|-------------------|
| **G10** | **Longevity / soak** | At least one **full longevity (matrix) continuous** InferenceMax row green with artifacts + **no silent quality drift** (or documented acceptable drift + ticket). |
| **G11** | **Accuracy breadth** | **P1** accuracy rows (e.g. MMLU-Pro, GSM8K, HellaSwag) green on the agreed hero(s); **P2** suite either green or explicitly parked with matrix + Trace. |
| **G12** | **IX advanced paths** | **ATOM+MTP** and **ATOM-Disagg** paths exercised on **matrix P1** scope first; **P2** recipes (`dsr1-*-mtp`, `qwen3.5-fp8-*-mtp`, `dsv4-*-mtp`, `dsv4-fp4-mi355x-atom-disagg`, …) are **complete**, **blocked**, or **parked** with ticket + owner. |

### Stages (outline)

| Stage | Activities | Exit |
|-------|----------------|------|
| **3.1** | Longevity harness: job wrapper, checkpoints, **abort / resume** story, log retention | One **full matrix longevity** run completes with comparable perf + error budget |
| **3.2** | **P1** accuracy batch: MMLU-Pro, GSM8K, HellaSwag on frozen configs | Scores + run URLs in Trace; sheet rows updated |
| **3.3** | **P2** accuracy breadth (BBH, MATH L5, GPQA, …) **as matrix allows** | Table below: each row is **Y** or **N/A** with reason |
| **3.4** | IX **MTP**: framework + priority recipes; MTP-specific metrics / acceptance | No “works on one node only”; known bad specs are reproducible and documented |
| **3.5** | IX **Disagg**: prefill vs decode pools, pool metrics, failure modes | Disagg row(s) in matrix match reality |
| **3.6** | **Quality / parity**: MTP degenerate decode, quant/logit parity FP8/FP4 vs BF16 where in matrix | Findings linked; **won’t fix** explicitly signed if needed |

### Suggested sequence (Phase 3: sketch)

| Block | Focus |
|-------|--------|
| **Early** | 3.1 longevity dry-run to first full soak; 3.2 P1 accuracy can overlap when a second chunk of GPU exists |
| **Mid** | 3.2 completion; 3.4 MTP framework + first recipe; 3.3 P2 accuracy as capacity allows |
| **Late** | 3.5 disagg; 3.6 quality / parity; flaky rows get another pass |
| **Overflow** | Anything that slipped; second hero when the matrix calls for it |



### Timeline Phase 3: tests & metrics (InferenceMax)

Section order mirrors **InferenceMax** tab: **Longevity** sits with **Inference frameworks** (#4); **Accuracy benchmark metrics** follows the tab’s accuracy block (#43+).

#### Inference frameworks (longevity)

| Item | Tab # | Matrix | Trace |
|------|-------|--------|-------|
| Longevity (matrix continuous); 24 h style gate where the matrix asks for it | 4 | P2 | |

#### Accuracy benchmark metrics (InferenceMax tab #43–53)

| Item | Matrix | Trace |
|------|--------|-------|
| MMLU-Pro | P1 | |
| GSM8K | P1 | |
| HellaSwag | P1 | |
| BBH, MATH L5, GPQA, MuSR, MMLU (legacy), ARC-Challenge, WinoGrande | P2 | |
| Scale parity, accuracy across scales | P2 | |

### Timeline Phase 3: tests & metrics (IX, MTP, disagg, quality)

Section order mirrors **IX ATOM** tab: advanced paths and recipes first; **OPTIONAL QUALITY GATES** on the tab are broken out last.

#### IX advanced paths (MTP / disagg recipes and frameworks)

| Item | Matrix | Trace |
|------|--------|-------|
| Framework **ATOM + MTP** (`*-atom-mtp`) | P1 / P2 | |
| Recipes `dsr1-*-mtp`, `qwen3.5-fp8-*-mtp`, `dsv4-*-mtp` | P1 / P2 | |
| Framework **ATOM-Disagg** + `dsv4-fp4-mi355x-atom-disagg` | P1 / P2 | |

#### Optional quality gates (IX ATOM tab #34–35)

| Item | Matrix | Trace |
|------|--------|-------|
| MTP acceptance / degenerate-spec decode (quality) | P2 | |
| Disagg pool metrics (prefill vs decode) | P2 | |
| Quant / logit parity FP8/FP4 vs BF16 | P2 | |

### Success checklist: end of timeline Phase 3

- [ ] **G10:** InferenceMax longevity soak per matrix **or** signed deferral with new date + risk note.  
- [ ] **G11:** P1 accuracy trio (or sheet-equivalent) green **or** N/A with matrix alignment.  
- [ ] **G12:** IX MTP + disagg paths exercised per **P1** matrix scope; P2 recipes **done, blocked, or parked** in Trace.  
- [ ] **Trace** column is complete for every automated row above; sheet **W×metric** cells are reconciled for anything marked green without evidence.



<a id="phase4_pipeline"></a>

## Timeline Phase 4: pipeline & hardening

> **Focus:** The aim is for everything from Phases **1 to 3** to be **boring to rerun**. Slurm, manifests, idle policy, and automation hooks reduce the chance that one person becomes the single point of failure every time the matrix hiccups.

**When:** Overlap with **late Phase 3** is fine (e.g. manifest work while soak runs). Phase 4 is not a substitute for closing G6 to G9 unless priorities explicitly change.

### Goals (G13 to G15)

| ID | Goal | “Done” looks like |
|----|------|-------------------|
| **G13** | **Scheduler integration** | **Slurm** (or chosen scheduler) path documented: submit, hold, cancel, **GPU binding**, multinode **salloc/sbatch** templates match what Phase 2 proved manually. |
| **G14** | **GPU idle & hygiene** | **GPU idle policy** (or equivalent) encoded in runbook + pipeline checks so abandoned jobs don’t burn the pool; **cleanup** steps idempotent. |
| **G15** | **Manifest + reproducibility** | **Expanded manifest** (images, wheels, git SHAs, matrix slice IDs) travels with each **published** result; “what exactly ran?” is answerable from artifacts without archaeology. |

### Stages (outline)

| Stage | Activities | Exit |
|-------|----------------|------|
| **4.1** | Slurm job templates for InferenceMax + IX rows used in Phases 1 to 3 | `cvs run` / wrapper examples in runbook + repo |
| **4.2** | GPU idle / preemption hooks, **max runtime**, alerts on stuck jobs | Policy doc + at least one dry-run |
| **4.3** | Manifest schema bump: extra fields the sheet cares about | Sample JSON checked in next green run |
| **4.4** | **Automation:** **W×metric** grid export (CSV/artifact), optional CI gate on **P1** cells | Pipeline emits artifact URL per matrix hash |
| **4.5** | Runbook **v3**: on-call path, flake playbook, **rollback** of bad image | Someone else can rerun a P1 row cold |

### Timeline Phase 4: tests & metrics (pipeline / ops)

| Item | Matrix | Trace |
|------|--------|-------|
| CVS pipeline, **Slurm integration** |   | |
| CVS pipeline, **GPU idle policy** |   | |
| CVS pipeline, **expanded manifest** |   | |
| Matrix export, **W×metric** last-run URLs or CSV |   | |
| CI / scheduled re-run of **P1** smoke subset |   | |
| Controlled failure drills (cancel mid-run, node loss), **when cluster policy allows** |   | |

### Success checklist: end of timeline Phase 4

- [ ] **G13 to G15** met **or** waived with ticket + new date.  
- [ ] Runbook **v3** published; **Slurm + manifest + idle** sections are copy-pasteable by someone who wasn’t in Phase 1.  
- [ ] At least **one** automated artifact path (grid export and/or CI smoke) is live, **Trace** has the job name / workflow link.



<a id="phase5_overflow"></a>

## Timeline Phase 5+: overflow

> **Focus:** Anything that still doesn’t fit without turning Phase 3 or 4 into a novel, **extra accuracy**, **more MTP recipes**, **HF only** depth, **cross site** parity, **P3+** matrix rows, **program level** gates.

Use this bucket when **cleaner reporting** in standups helps (e.g. “Phase 5 = HF + chat breadth only”). Extra rows can live below, or headings can split (`Phase 5, HF depth`), instead of overfilling Phase 3.

| Item | Matrix | Trace |
|------|--------|-------|
| *(Rows that slip from Phase 3/4, or new matrix rows as the sheet grows, can be parked here.)* | | |



<a id="traceability_section"></a>

## Traceability & live-sheet reconciliation

**When the sheet and this doc disagree, the sheet wins.** Labels stay honest when phase-table drift is reconciled on the usual review cadence.

Phase tables in this doc follow the same block order as **InferenceMax**, **IM Matrix**, **IX ATOM**, and **IX ATOM Matrix** in **DTNI Validation Tracker.xlsx** (framework / path rows vs **W** workload rows vs performance vs accuracy / optional quality). When the live sheet differs, reconciliation rules in the table below still apply.

| Topic | This doc (typical) | If the **live** sheet differs |
|-------|-------------------|-------------------------------|
| **IX MTP** | Framework **ATOM+MTP** often **matrix P1**; individual `*-atom-mtp` recipes often **matrix P2**. | Recipe rows in **this doc’s tables** (any phase column) line up with the sheet; **Phase 5+** rows or sub-headings cover cases where Phase 3 would otherwise get unwieldy. |
| **IX Disagg** | **atom-disagg** path often **matrix P1**; `dsv4-fp4-mi355x-atom-disagg` often **matrix P2**. | The **Matrix** column in the **Phase 3 to 5** tables reflects the sheet. |
| **IX goodput** | Often **matrix P2** in IX perf rows. | IX Phase 2 perf rows show **P1** (or **P0**) when the sheet does. |
| **InferenceMax goodput / latency vs load** | Often **matrix P1** on IM perf rows. | The Phase 2 perf table shows **P2** when the sheet moved; **P0** appears when the sheet added it. |
| **Accuracy** | MMLU-Pro, GSM8K, HellaSwag = **P1** on a lot of sheets; other benchmarks **P2** (later phases). | After a matrix reshuffle, **Phase 3** (and **Phase 5+** peel rows) carry current tier labels rather than stale ones. |
| **Extra priority columns** | Tables show **P1/P2** to keep cells readable. | When the workbook uses **P0**, **P3**, or custom labels, the **Matrix** column (or **Sheet tag**) covers them so nothing is misrepresented. |

**Coverage hygiene**

- **Trace** has an entry on every phase-table row that is automated; **N/A** lines up with the sheet’s *No coverage reason* when that is the honest answer.
- **W×metric grids:** the live matrix (tab + hash) lands in `plans/artifacts/` or CI emits a CSV for each **Y/P** cell with last run URL; the full grid stays out of markdown.
- **Matrix churn:** Orphan rows tend not to accumulate when sheet diffs and doc updates stay paired.
