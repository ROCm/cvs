# CVS DTNI v1 Specification

**Status:** Draft for team review.

## Goal

Replace the current DTNI (data-center training and inference) stack — `cvs/lib/{megatron_training_lib,jax_training_lib,sglang_disagg_lib,inference_lib}.py` and `cvs/lib/inference/*.py`, plus the seven duplicated test wrappers under `cvs/tests/{training,inference}/` (~5000 LOC, ~80% duplication) — with a single lifecycle-driven framework: typed configs, on-disk manifests, deterministic cluster binder, and pytest-as-first-class test taxonomy. Land the security and correctness fixes that are currently shipping as latent bugs. Establish the foundation that lets v2 features bolt on without retrofit.

---

## W1. Core lifecycle and adapter framework

Centralize the prepare → launch → await → parse → verify → teardown lifecycle that every workload already executes in copy-pasted form today. One contract, one driver, no mode branching.

**Deliverables**

- `cvs/lib/adapter_protocol.py` — `WorkloadAdapter` Protocol with seven methods: `prepare`, `launch`, `progress_predicate`, `await_completion`, `parse`, `verify`, `teardown`. Every method takes the same `ctx` (the `RunContext`) and communicates through it; there is no separate run object threaded between methods. The Protocol shape is v1's commitment; it is not claimed to be a closed contract that handles every conceivable future workload.
- `cvs/lib/base_adapter.py` — `BaseWorkloadAdapter` (`abc.ABC`) providing concrete defaults that most adapters inherit:
  - `teardown` always captures container logs, dmesg snapshots, and GPU state, then removes containers by `run_id` label.
  - `await_completion` polls `progress_predicate` at a configurable interval and raises on timeout.
  - `prepare` is a no-op by default.
  - `_launch_role` / `_wait_http_pool` provide the shared multi-role launch + readiness plumbing (fan out one container per host bound to a role via the per-host runner; concurrent HTTP readiness across every handle of a role), so single-role and multi-role disagg adapters share one launch path.
- `cvs/lib/job.py` — `Job` driver running the six-step lifecycle. Mode-blind body: no `if mode == "training"` anywhere in the driver. Failures are classified at the boundary where they originate, not by post-hoc inspection of a stack trace. `teardown` always runs in `finally`.
- `cvs/lib/failure_taxonomy.py` — five disjoint failure categories, evaluated in priority order:
  - `setup_failure` — `prepare` or `launch` raised before the workload started.
  - `safety_violation` — `progress_predicate` broke mid-run.
  - `failure_pattern_matched` — a pattern from `failure_patterns.yaml` (W8) hit a log stream.
  - `liveness_failure` — `await_completion` timed out without the predicate breaking.
  - `verification_failure` — a `Threshold` (W3) evaluated to False at end-of-test.
- `cvs/lib/registry.py` — `INFERENCE_REGISTRY` and `TRAINING_REGISTRY` keyed on the `framework` Literal from the typed config. Driver dispatch is the only place that picks an adapter by name.

**Replaces**

- The per-wrapper `try / finally` blocks scattered across `cvs/tests/{training,inference}/**/*.py` for cleanup.
- The module-level `globals.error_list` aggregation pattern (full removal under W6).
- Implicit failure classification ("it printed an error, so it failed") in favor of explicit category assignment at the raise site.

**Dependencies:** none upstream.

---

## W2. Six concrete adapters

Port every existing DTNI workload onto the W1 lifecycle as a concrete adapter. Single-role workloads are direct `BaseWorkloadAdapter` subclasses. The one multi-role workload today (sglang disagg) ships as a single adapter with internal orchestration; the `CompositeAdapter` abstraction is explicitly deferred to v2.B.

**Deliverables**

- `VllmAdapter` — replaces `cvs/lib/inference/base.py` (the vLLM-flavored informal ABC) plus `cvs/lib/inference/vllm.py` plus four wrappers under `cvs/tests/inference/vllm/`. Lifts vLLM-specific env vars (`VLLM_USE_AITER_UNIFIED_ATTENTION`, `VLLM_ROCM_USE_AITER_MHA`, `VLLM_ROCM_USE_AITER_FUSED_MOE_A16W4`) out of the shared base into the adapter where they belong. Parse step captures per-request JSONL from `bench_serving` plus Prometheus queue depth and memory pressure (today emitted by the framework, today discarded by CVS).
- `InferenceMaxAdapter` — replaces `cvs/lib/inference/inference_max.py` plus `cvs/tests/inference/inferencemax/inferencemax_gpt_oss_120b_single.py`. Captures per-batch JSON (already emitted, currently discarded). Fixes the silent-pass-on-missing-threshold defect.
- `SglangDisaggAdapter` — replaces the 1200-LOC monolith at `cvs/lib/sglang_disagg_lib.py` plus the two near-identical wrappers `cvs/tests/inference/sglang/sglang_llama_70b_distributed.py` and `sglang_deepseek_r1_671b_distributed.py` (which differ by one line). **Single adapter, not Composite.** Internal orchestration manages the prefill / decode / router / bench roles. Parse step captures router queue trajectory, per-role startup times, decode KV-cache utilization. Refactoring into a Composite of sub-adapters is v2.B work, gated on a second multi-role workload landing.
- `PytorchXditAdapter` — replaces the two standalone wrappers `cvs/tests/inference/pytorch_xdit/{pytorch_xdit_flux1_dev_single,pytorch_xdit_wan22_14b_single}.py` (no shared library today; docker-run logic inlined per file). Eliminates the duplicated `LocalPssh`, `_is_local_target`, and `_redact_secrets` blocks across the two files. Parse step captures per-step latency trajectory for Flux and per-frame latency plus bitrate stats for WAN.
- `MegatronAdapter` — replaces `cvs/lib/megatron_training_lib.py` plus four wrappers under `cvs/tests/training/megatron/`. Single and distributed via the same adapter; `distributed: bool` is a config field, not a separate test file. Parse step captures full trajectory (loss, throughput, step time, grad norm) and per-rank logs (today only the last node's log is read, masking rank-0 failures).
- `JaxAdapter` — replaces `cvs/lib/jax_training_lib.py` plus three wrappers under `cvs/tests/training/jax/`. Parse step captures the per-step JSONL trajectory plus per-host metrics from the coordinator role.
- Delete `cvs/lib/inference_lib.py` — orphan `InferenceJobFactory` whose import target `cvs.lib.inference_max_lib` does not exist on disk; zero callers; would `ImportError` on first use.

**Replaces**

- The "framework emits, CVS throws away" defect for every adapter: each adapter's `parse` populates the manifest's `samples` and `trajectory` carriers (W4) with framework-native telemetry rather than tail-and-regex on console output.

**Dependencies:** W1 (Protocol, Job, base class), W3 (typed configs for adapter constructors), W4 (manifest carriers for parse output).

---

## W3. Typed config schema

One typed config per (framework, model). Mega-configs with multiple models per file are split. Stringified-key result dicts are replaced by typed threshold predicates. Typos fail at config load, not after a 20-minute workload run.

**Deliverables**

- `cvs/lib/config/base.py` — `BaseTestConfig` (Pydantic, `extra="forbid"`), `schema_version: "2"`, common fields: `target_gpu`, `cluster_ref`, `secrets`, `seed`, `thresholds`.
- `InferenceTestConfig` and `TrainingTestConfig` — discriminated unions on `framework`. Configs cannot route through the wrong mode; a typo like `framework: vlm` for inference is a `model_validate()` error at load, not a silent miss after the workload runs.
- First-class `knobs` field — `attention`, `quant`, `backend`, `fused_moe`, and similar. Lives at the top level, not buried inside per-framework `params`. These become pytest markers (W6) so users can slice with queries like `pytest -m "knob_aiter and quant_fp4"`.
- Per-framework `params` classes — `VllmParams`, `SglangDisaggParams`, `InferenceMaxParams`, `PytorchXditParams`, `MegatronParams`, `JaxParams`. Framework-specific knobs live here.
- Per-framework `SweepParams` classes — `VllmSweepParams`, `SglangDisaggSweepParams`, `MegatronSweepParams`, and so on. Sweep axes are typed and validated per framework: `concurrency` is valid for inference and rejected for training; `parallelism_combos` is valid for training and rejected for inference. Cross-axis constraints are expressed as Pydantic validators (for example, `product(parallelism) == total_gpus`).
- Six typed `Threshold` predicates, each carrying an explicit `op` field so direction is never inferred from the metric name:
  - `PercentileThreshold` — over `samples`; e.g. P99 TTFT ≤ 50 ms.
  - `MonotonicityThreshold` — over `trajectory`; e.g. loss non-increasing in the last quarter.
  - `ConvergenceThreshold` — over `trajectory`; e.g. loss reaches target ± epsilon by step N or wallclock T.
  - `StabilityThreshold` — rolling-variance bound over samples or trajectory.
  - `RateThreshold` — derived rate; e.g. throughput ≥ 1200 tokens/sec.
  - `GoodputThreshold` — filtered rate over samples (the MLPerf-shaped headline metric); requests/sec where TTFT ≤ X and TPOT ≤ Y.
- `cvs migrate-config` one-shot tool — rewrites every existing config under `cvs/input/config_file/{training,inference}/` from the current JSON schema to v2 YAML. Splits mega-configs (the vLLM 4-model file, the sglang 2-model file). Converts `result_dict["ISL=...,OSL=...,TP=...,CONC=..."]` lookups into typed `Threshold` predicates. Forbids `<changeme>` sentinels. Stamps `schema_version: "2"`.

**Replaces**

- The untyped `json.load` + `dict.setdefault` defaulting that currently lets typos like `percentiles_metrics` (vs `percentile_metrics`) silently fall through in `cvs/input/config_file/inference/inferencemax/*.json`.
- The substring-`"ms"` heuristic in `cvs/lib/inference/base.py` and `cvs/lib/sglang_disagg_lib.py` that infers comparison direction from the metric name.
- Stringified-param result lookup keys like `"ISL=1024,OSL=1024,TP=8,CONC=64"`.

**Dependencies:** none upstream; W2, W4, W5, W6 depend on it.

---

## W4. Manifest, sidecars, and cross-run export

State on disk, not in module globals. Per-run manifest as the small hierarchical index; bulky numeric arrays as Parquet sidecars; events as append-only JSONL; raw logs preserved. The schema is designed for cross-run regression analysis from day one.

**Deliverables**

- Per-run directory layout: `<artifact_dir>/<test_id>/<cell_id>/<short_hash>/<run_id>/`. Content-addressable; supports v2.A reuse-manifests without retrofit.
- `manifest.json` — small (5–50 KB) hierarchical metadata, verdicts, scalars, and pointers to sidecars. Pydantic-modeled; `cat`-able; `jq`-friendly. Eight content categories:
  - **Identity and provenance** — `run_id`, `test_id`, `cell_id`, `config_hash`, `workload_hash`, `verification_hash`, `cvs_version`, `cvs_git_sha`, `framework_image_digest`, `framework_versions`, timestamps, invoker.
  - **System fingerprint** — per host: CPU, memory, kernel, OS, GPUs, NICs, container runtime; topology hash for the cluster.
  - **Configuration and inputs** — pointer to `config.resolved.yaml`, dataset descriptors with shas, model descriptor with weight sha, redacted env, redacted command lines, seed.
  - **Phase timing** — start, end, duration, and status per lifecycle phase.
  - **Verdicts and result** — overall status, failure category, per-threshold verdicts with actual + expected + margin, pattern matches, derived scalars, categorical status flags.
  - **Resource summary** — per-host GPU utilization, HBM usage, power, network, OOM flag (lightweight aggregate; per-step data lives in the trajectory sidecar).
  - **Sidecar pointers** — paths to samples, trajectory, events, logs, dmesg snapshots, GPU state snapshots.
  - **Schema metadata** — `schema_version` for the manifest itself, versioned independently of the config schema.

  Both `workload_hash` (workload-defining inputs) and `verification_hash` (threshold and failure-pattern inputs) are recorded from day one even though v1 does not ship `--reuse-manifests`. This makes v2.A a pure tack-on with no historical migration.
- `events.jsonl` — append-only event stream with a **closed vocabulary**: `prepare.{start,done}`, `launch.{container_up,role_ready}`, `seed.logged`, `arrival.{start,end}`, `accuracy.{start,end}`, `step`, `request`, `safety.violated`, `pattern.matched`, `parse.done`, `verify.{passed,failed}`, `teardown.{start,done}`. Adding a new event name is a schema change reviewed in PR, not a free-for-all `log.info` call.
- `samples.parquet` — long-format, one row per request or sample (for example `{request_id, ts, ttft_ms, tpot_ms, itl_ms, e2el_ms, output_tokens, role, host}`). New metrics do not change schema.
- `trajectory.parquet` — long-format, one row per (step, metric, role, host). Same property.
- `config.resolved.yaml` alongside the manifest — full resolved config (post-override, post-substitution) for reproducibility. Hash for quick identity, resolved file for forensics.
- `logs/` — preserved raw artifacts per host and role: container stdout / stderr, dmesg pre and post snapshots, `rocm-smi` / `amd-smi` pre and post snapshots. Enables v2.C reparse-from-logs.
- `cvs export` — walks N run directories, joins manifest scalars with sidecar rows, writes one Parquet fact table partitioned by `experiment_id` / `cvs_git_sha` / `timestamp`. Pandas, Polars, DuckDB read it directly; no service required.

**Replaces**

- Module-level `inf_res_dict = {}` and `inference_dict["_test_output_dir"]` mutation that pass data between tests today.
- Tail-and-regex parsing as the only source of result data.
- The impossibility of `cvs run -k "verify"` against an already-completed benchmark (today, results die with the pytest process).

**Dependencies:** W3 (typed configs feed the manifest); W6, W8 depend on it.

---

## W5. Cluster pool, deterministic binder, sweep expansion

Decouple "what hardware do I have" (cluster) from "what workload do I want to run, parameterized how" (config). The binder maps role requirements to physical nodes at run time, deterministically, so two runs of the same config on the same cluster pick the same hosts.

**Deliverables**

- Cluster file schema — a pool of nodes only: `{nodes: {hostname: {ip, user, ssh_key, gpus, labels}}}`. No role assignments live here.
- Config-side `topology.roles` — `{role_name: {count, gpus_per_node, selector}}`. The selector is a label query against `cluster.nodes[].labels`. Examples: `decode: {count: 2, gpus_per_node: 8, selector: "mi355x"}`, `router: {count: 1, gpus_per_node: 0}`.
- Deterministic binder — first-fit by cluster-file declaration order. Same cluster plus same config always yields the same bindings. This determinism is load-bearing for v2.A reuse-manifests.
- Sweep expansion semantics:
  - Scalar lists default to cartesian product across axes.
  - Paired axes use a single list-of-objects (no cross); each entry is one cell. Optional `name:` field on each entry becomes the cell's pytest parametrize ID.
  - Multiple axis groups stack: cartesian between groups, paired within groups. Matches `pytest.mark.parametrize` semantics so the sweep block lowers cleanly to `pytest_generate_tests`.
  - Topology-changing axes (P/D split, node count) carry a per-cell `topology` block; the binder re-evaluates per cell.
- Per-cell skip with reason — when a cell's role requirements cannot be satisfied by the cluster pool, the cell is marked `status: skipped` in its manifest with a concrete reason (for example `insufficient_nodes (need 4, have 3)`). Other cells in the sweep proceed normally. A user with a small dev cluster gets useful partial coverage instead of an error.

**Replaces**

- Role-list fields baked into per-test configs today (`prefill_node_list`, `decode_node_list`, `proxy_router_node`, `benchmark_serv_node` in `cvs/input/config_file/inference/sglang/*.json`) that tie configs to specific hostnames.
- Ad-hoc per-file fixture logic for parametrization (the `pytest_generate_tests` block duplicated across four vLLM wrappers).

**Dependencies:** W3 (typed sweep params); W6 and W8 depend on it.

---

## W6. Pytest layer and test taxonomy

Pytest tests are first-class user surface: every claim about a workload is a named, sliceable test function. One workload run produces many independent pytest assertions by reading the manifest. Tier structure equals directory tree.

**Deliverables**

- Directory tree under `cvs/tests/`:
  - `logistics/` — claims that every config exercises (image pullable, container up, role ready, no orphan containers, dmesg clean).
  - `training/` — training-kind-only claims (loss finite, trajectory monotonic in expected window); subdir `training/test_distributed.py` for distributed-only claims (per-rank step sync, no straggler, collective health).
  - `inference/` — inference-kind-only claims (server health, request success rate, no 5xx burst); subdirs for `disagg` (router balance, P→D handoff, KV-cache utilization) and `distributed`.
  - `frameworks/` — one file per framework (`test_vllm.py`, `test_sglang.py`, `test_inferencemax.py`, `test_pytorch_xdit.py`, `test_megatron.py`, `test_jax.py`) for framework-specific knob assertions (AITER flags applied, XLA flags applied, attention backend matches the knob).
  - `benchmarks/` — one file per claim family (`test_throughput.py`, `test_latency.py`, `test_accuracy.py`, `test_convergence.py`). Opt-in via the config's `benchmarks: [...]` list.
  - `models/` — model-family edge cases (rarely populated; documents known quirks).
- Session-scoped `workload_run` fixture in `cvs/tests/conftest.py` — runs the `Job(adapter, cfg).run()` lifecycle once per (config, sweep cell), yields the manifest. One workload run feeds many test functions.
- Auto-applied pytest markers derived from config fields. Naming convention: `tier_1` through `tier_6`, `framework_<name>`, `workload_<kind>`, `topology_<kind>`, `model_<name>`, `knob_<key>_<value>`, `benchmark_<name>`, `gpu_<family>`. Registered via `pytest_configure` so `-m` queries do not warn about unknown markers.
- `collect-skip` hook in `pytest_collection_modifyitems` — items whose tier predicate does not match the cell's config are **deselected**, not skipped. Keeps `--collect-only` output sane: a vLLM cell does not show fifty deselected megatron / jax / sglang tests.
- `requires_benchmark("name")` decorator — tier-5 tests opt in via the config's `benchmarks` list. A vLLM config that declares `benchmarks: [throughput, ttft_p99]` runs the matching tier-5 functions and nothing else from tier 5.
- `pytest_terminal_summary` hook iterating per-test manifests at session end. Replaces module-level `globals.error_list` aggregation.
- One parametrized test pattern per suite consuming the typed config; adding a new model means adding a new config file, not a new test file.

**Replaces**

- The ~1120 LOC of byte-identical fixture boilerplate duplicated across seven training wrappers (the `cluster_file`, `training_config_file`, `cluster_dict`, `training_dict`, `model_params_dict`, `hf_token`, `phdl` fixture block).
- The pattern of one test file per (framework, model, single/distributed) tuple — collapsed into one parametrized test per suite.
- `globals.error_list = []` plus `fail_test()` plus `update_test_result()` across `cvs/lib/*_lib.py`.

**Dependencies:** W1 (Job and adapters), W3 (typed configs for marker derivation), W4 (manifest as fixture output), W5 (sweep expansion lowering to `pytest_generate_tests`).

---

## W7. Security and correctness fixes

Latent defects in today's code that are independent of the broader redesign. Ship as a standalone PR series before the rest of v1 if scheduling permits — none of these block on the architecture work.

**Deliverables**

- `SecretValue` class — token storage with type-level redaction. `repr` and `str` return `<SecretValue label=***>`; `.reveal()` is only called at `--env-file` write time. Removes plaintext HF tokens from logs (currently logged on every multi-node run via `phdl.exec` debug printing).
- `ContainerHandle` context manager — `docker run -d` with a `run_id` label; readiness probe blocks `__enter__`; `__exit__` always captures logs plus dmesg plus GPU state, then removes containers by label. Non-privileged default. Removes the `--privileged + seccomp=unconfined` default and the `docker system prune --force` call from `cvs/lib/docker_lib.py` (the prune currently wipes other users' containers on shared nodes).
- `fail_test()` actually calls `pytest.fail` — the current implementation appends to `globals.error_list` and returns; multi-step tests march past the first failure.
- Explicit `op:` field in `Threshold` (delivered as part of W3 but listed here for completeness) — removes the substring-`"ms"` direction inference that would flip comparison for any future metric name containing those letters (for example `latency_seconds`).
- `inferencemax` adapter (W2) raises on missing threshold instead of silently passing.
- HF-token fixture refactor — the current `UnboundLocalError` on a missing token file becomes a clean `pytest.skip` with a clear reason.
- `mi355` / `mi355x` literal normalization via a single GPU-family detection path. Today four files have inconsistent literals.
- Quote `$log_dir` in every `sudo rm -rf` call (today: `cvs/lib/docker_lib.py` runs unquoted on user-supplied paths).
- Sentinel-leak CI test — run a representative test with `HF_TOKEN=hf_LEAK_SENTINEL_xyz`; grep the full session log plus every in-container `/tmp/*.sh`. Fails if the sentinel appears anywhere.

**Replaces**

- The relevant defect lines in `cvs/lib/{sglang_disagg_lib.py, inference/base.py, docker_lib.py}` and the per-test HF-token fixture code paths.

**Dependencies:** none. Can land first.

---

## W8. Tooling and documentation

User-facing CLI commands and reference documentation that make the v1 surface usable without code-diving.

**Deliverables**

- `cvs plan` — dry-run binder. Parses config and cluster, expands the sweep, runs the binder, applies pytest collection rules, prints the planned matrix (cells × bindings × selected tests × skip reasons). Exits without launching any adapter methods. Same code path as `cvs run` minus execution.
- `cvs export` (delivered under W4; listed here for the user-facing surface) — flattens N run directories into one Parquet fact table for ad-hoc analysis.
- `cvs migrate-config` (delivered under W3; listed here for the user-facing surface) — one-shot rewriter for existing JSON configs to v2 YAML.
- `failure_patterns.yaml` — seed catalog of approximately eight entries for the failure-pattern scanner registered as a pytest hook. Each row carries `id`, `source` (`dmesg` or `framework_log`), `pattern` (regex), `severity` (`fatal` or `warn`), and `hint`. Seed coverage: OOM-killer, RCCL / NCCL collective timeout, HBM ECC, PCIe AER, thermal throttle, kernel panic, NCCL watchdog warning, container OOM. Adding a pattern is a YAML edit; no code changes.
- `FailurePatternScanner` — runs as a pytest hook (W6 seam); tails declared sources during the run; matches recorded as `pattern.matched` events with `{id, severity, line, node}`.
- User-facing documentation under `docs/`:
  - Config schema reference (per-framework `params` and `SweepParams`, `Threshold` predicates, `topology.roles`).
  - Pytest tier reference (what each directory covers, how to add a tier-5 benchmark, how a benchmark is opted in via config).
  - Marker reference (full list with derivation rules).
  - Sweep semantics (cartesian default, paired via list-of-objects, stacked axes, topology-changing cells).
  - "How to add a new framework" walkthrough (new adapter, registry entry, params class, sweep params class, optional framework-specific test file).

**Dependencies:** W3 (`cvs migrate-config`), W4 (`cvs export`), W5 (`cvs plan`), W6 (pattern scanner hook seam).

---

## Migration story

- A single one-shot run of `cvs migrate-config` rewrites every JSON config under `cvs/input/config_file/{training,inference}/` to v2 YAML. No backwards-compatible reader path. The new schema is the only schema.
- All seven existing test wrappers under `cvs/tests/{training,inference}/` are deleted. The new parametrized tests subsume them.
- The `cvs run` CLI surface is unchanged — same flags (`--cluster_file`, `--config_file`), same invocation pattern. Existing automation continues to work.
- New per-framework markers and the tier directory layout are additive on the pytest CLI side. Queries like `pytest -m "framework_vllm"` become possible; nothing previously possible breaks.

---

## Verification

Per workstream:

- **W1** — adapter Protocol conformance tests using fake adapters; `Job` driver behavior under each failure-class injection (raise from `prepare` → setup; broken predicate mid-run → safety; pattern match → pattern; timeout → liveness; failing threshold → verification).
- **W2** — per-adapter unit tests with fake `Pssh`; per-adapter end-to-end run of the smallest workload (for example vLLM gpt-oss-120b at concurrency 16) producing a complete manifest with `status: complete`.
- **W3** — `model_validate` against every migrated config; reject every malformed example fixture; `cvs migrate-config` round-trips every shipped config without semantic loss.
- **W4** — Pydantic round-trip for the manifest; Parquet schema test for samples and trajectory; closure test for the events vocabulary; one real run produces manifest plus sidecars that `pd.read_parquet(samples)` consumes with expected columns.
- **W5** — deterministic binding asserted across 100 randomized configs against a fixed cluster (same binding twice); per-cell skip-with-reason for under-resourced cells; `cvs plan` output matches `cvs run` actual bindings on a real cluster.
- **W6** — marker application verified against synthetic configs; `collect-skip` correctness for each tier predicate; `pytest --collect-only` for the example sweep prints the expected ID set.
- **W7** — sentinel-leak CI test; `--privileged` flag absent from every docker invocation in the codebase (grep-asserted); shared-host test confirms other users' containers are untouched.
- **W8** — `cvs plan` golden-file tests; `cvs export` schema test; a nightly run produces a Parquet that opens in a notebook.

Suite-level: a representative subset of today's configs (one of each shape — vLLM gpt-oss-120b, sglang Llama-70B disagg, megatron 8B distributed, jax 70B single) runs to `status: complete` on v1 CVS, with manifests inspected manually for correctness.

---

## Workstream dependencies

- W7 has no upstream dependencies — ship first as a standalone PR series.
- W1 enables W2 (adapters subclass the base) and W6 (the fixture runs `Job(adapter)`).
- W3 enables W2 (adapter constructors take typed configs), W6 (markers derived from config fields), and W5 (sweep params per framework).
- W4 enables W6 (the fixture yields the manifest) and W8 (`cvs export` reads the manifest tree).
- W5 enables W6 (sweep block lowers to `pytest_generate_tests`) and W8 (`cvs plan` is the binder's dry-run mode).
- W8 depends on W3, W4, W5, and W6 — ships last.
