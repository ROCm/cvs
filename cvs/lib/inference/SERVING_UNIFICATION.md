# Serving suite unification (vLLM, SGLang, ATOM)

Minimal architecture proposal. xDiT is out of scope (Pssh + torchrun, not a serving sweep).
Reference implementations stay separate; this is what to share, what to stop duplicating, and what not to merge.

**Status:** sound after critic pass against `main` @ `9081b8b0` — cheapest-first sequence, not a rewrite.

---

## Target shape

```
cvs/lib/utils/                    gpu.py, ib_discovery.py, verdict.py, OpenAIProbe, BaseVariantConfig
cvs/lib/inference/utils/          serving_schema.py, cell_key.py, inference_suite_lifecycle.py
                                  bench_client_core.py (vLLM-bench JSON derivations only)
cvs/lib/inference/vllm/           job + loader + metrics  (move off utils/ later)
cvs/lib/inference/sglang/         job(s) + loader + parsing
cvs/lib/inference/atom/           native AtomJob + loader + parsing
```

One serving JSON skeleton for **new** configs (vLLM already; ATOM via `atom_serving_config.is_serving_config`. Packaged ATOM files are still `schema_version` 2 / profile sweep — do not pretend they are serving JSON yet):

`schema_version`, `enforce_thresholds`, `threshold_json`, `paths`, `model`, `container`,
`server_params`, `benchmark_params`, `sweeps` / `runs`.

Cell keys: `ISL=…,OSL=…,TP=…,PP=…,CONC=…` (one helper, three loaders).

Metric **names** stay per engine. Shared is the `{kind, value}` spec shape. Verdict helpers stay split: vLLM `metric_verdict` (registry `min`/`max` only); ATOM/SGLang `evaluate_all` (richer kinds). Do not force vLLM back onto `evaluate_all`.

---

## Do this (cheap)

| # | Change | Why |
|---|---|---|
| 1 | Split `inferencing_config_loader.py`: keep sweep/coverage helpers; **delete** unused `VariantConfig` / `Params` / `load_variant`. Retarget `ADDING_A_SUITE.md` to `vllm_config_loader`. | Dead vLLM-single schema still documented as the reference. ATOM is the only live importer of the sweep types. |
| 2 | One `format_perf_cell_key` / `PERF_CELL_RE` used by vLLM, ATOM serving, SGLang unified. | Same string already, three copies. |
| 3 | SGLang: `tests/inference/sglang/_stages.py` for the identical rms-norm / poll / OpenAI / lm-eval / perf / dmesg / table / teardown bodies. Keep three launch-sequence tests local. | Three modules, same functions. |
| 4 | Extract `run_openai_probe(exec_fn, …)` around `OpenAIProbe`. | Copied in `VllmJob`, `AtomJob`, `sglang_common`. |
| 5 | Delete `inference_lib.py` `InferenceJobFactory` and `InferenceBaseJob` (repoint `textwrap_for_yml` test). | Nothing constructs them. |
| 6 | vLLM/SGLang may import shared `test_launch_container` / `test_teardown`. **Do not** drop in shared `test_model_fetch` for vLLM: `_common.py` only asserts cache `du` on every orch host; lifecycle fetch can `hf download` when `model.remote==1` and records different HTML keys. Extend `InferenceLifecycle` with vLLM `live_server_*` / `ib_hcas` — do not replace `_Lifecycle` blindly. | Launch/teardown match; fetch and lifecycle state do not. |

---

## Do this next (GPU / topology — the actual gaps)

**SGLang can use `gpu.py`.** Nothing in `gpu.py` is vLLM-specific. It talks to an `Orchestrator` and runs host-side `amd-smi metric --json`.

SGLang does not today because it wants a **different product**: post-run occupancy vs TP×PP (`collect_sglang_gpu_topology` + `_host_exec` + **`sudo amd-smi metric --json`**), not the five HTML rows (`peak_gpu_memory_mb`, util %, model-load delta) that vLLM/ATOM attach to each cell.

`gpu.py` runs **`amd-smi metric --json` with no sudo** via `orch.exec_on_head` / `orch.exec`. That is the real integration snag on locked-down clusters, not an API gap. Decide per site: passwordless `amd-smi` (prefer, match vLLM/ATOM) vs a documented sudo wrapper — do not fork a second parser.

Recommended split:

- **Metrics (opt-in):** wrap the perf cell with `start_gpu_poller` / `stop_and_collect_gpu_poller` like `_common.py` / `atom.py`. Pass `nodes=` as hostnames (distributed) or labeled groups into `capture_gpu_metrics` for PD (`[("prefill", …), ("decode", …)]` already exists on capture; the **poller** still takes a flat host list — do not pretend they are the same API).
- **Topology check (keep):** `count_occupied_gpus_*` stays SGLang-local. Reuse `parse_mem_usage` / JSON unwrap per GPU entry. **Do not** call `parse_gpu_metrics` — that **sums** VRAM across GPUs and cannot count occupied devices.
- **Exec:** prefer `orch.exec_on_host` / `orch.exec(..., hosts=)` over `BaremetalOrchestrator.exec` hops. amd-smi stays host-side, never in-container.

**Topology module:** do **not** merge vLLM Ray/PP rules, ATOM `driver=atom` SPMD-DP, and SGLang PD roles into one resolver. Extract:

- `EffectiveServingTopology`: `hosts`, `nnodes`, `tp`, `pp`, `ib_hcas`, `socket_netdev`, optional `role_groups`
- suite adapters: `resolve_vllm_topology` (keep), ATOM driver checks (keep), `resolve_server_node_list` / `_disagg_role_hosts` (keep)

vLLM `test_discover_topology` should use `resolve_multinode_fabric` (HCAs **and** netdev), matching ATOM, instead of HCAs-only + requiring `NCCL_SOCKET_IFNAME` in config.

---

## Do not do

- One job ABC or resurrecting `InferenceBaseJob`.
- Merging `SglangSingle` / `SglangDistributed` / `SglangDisaggPD` (PD launch order is unique).
- Folding ATOM `client.*` actuals into vLLM bare metrics in one step (scaling keys, W1 gated superset, Run Deck `metric_prefix`).
- Forcing SGLang `*_per_sec` / `mfu` / `mean_e2e_latency_ms` onto `vllm_metrics.METRIC_REGISTRY` before SGLang stops gating through `benchmark_params.expected_results`.
- Collapsing ATOM `driver=vllm|sglang` into `AtomJob` forever — treat those as **migration adapters**. Native `AtomJob` should be `driver=atom` only; vLLM/SGLang coordinators belong in their suites plus `roles.server.env`.
- Unifying threshold **assertion** shapes (vLLM cell+registry subtests vs ATOM cell×tier vs SGLang fused perf subtests) until configs share one loader.

---

## ATOM drivers

Today `AtomJob` reimplements vLLM/SGLang serve+bench behind `params.driver`. That is the largest accidental coupling.

Target: ATOM suite = ATOM native path. Configs named `mi325x_atom_vllm_*` / `*_sglang_*` should either (a) run the vLLM/SGLang suites, or (b) stay as a thin env overlay on those jobs — not a second copy of argv/poll/parse inside `atom_orch.py`.

---

## Metrics / reports

- Per-engine registries (vLLM’s `METRIC_REGISTRY` is the template, not a global file).
- Shared **bench JSON derivations** used by both `project_vllm_metrics` and ATOM `to_client_metrics` (stop `vllm_parsing.py` vs `vllm_metrics.py` drift; make `vllm_parsing` a shim).
- Fix ATOM Run Deck lookups so bare threshold keys resolve against `client.*` actuals (`cell_build` vs pytest `evaluate_specs_for_actuals`).
- Deck profiles stay per stem (`vllm.json` / `sglang.json` / `atom.json`); shared render path is already `cvs/lib/report/`.

---

## Sequence

1. Delete dead loader/factory/base; extract cell-key + serving_schema; fix docs. **No behavior change.**
2. SGLang `_stages.py` + shared launch/teardown imports. **No behavior change** if hooks preserve log-dir cleanup and env-setup (`server` vs `bench`). Leave vLLM `test_model_fetch` and `_Lifecycle` extras local until explicitly aligned.
3. `scope_cluster_to_hosts` + SGLang/vLLM fabric discovery (`resolve_multinode_fabric`; drop SGLang `rdma0..7`/`eno0` setdefaults). **Behavior:** fewer hardcoded NICs; vLLM single still first-host unless we add an explicit host field (optional follow-up: `benchmark_serv_node`-style override for vLLM).
4. SGLang `gpu.py` poller on perf cells (record-only GPU rows first; same non-sudo `amd-smi` as vLLM). Keep occupancy tests on per-GPU `used_vram`.
5. Narrow `AtomJob`; shim `vllm_parsing`; Run Deck prefix bridge.
6. Only then: SGLang `variant.thresholds` instead of injected `expected_results`.

Stop after each step if configs or HTML rows change without an explicit migration.
