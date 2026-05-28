# CVS DTNI Suite Expansion — PRD

> **Status:** Draft for team review.
> **Scope:** Architecture only — no sequencing, no slice plans, no effort estimates. Implementation rollout lives separately. Source of detail: `cvs-dtni-suite-expansion-design.md`.
> **Audience:** CVS contributors who will own, extend, or debug the inference and training suites.

Each architectural piece below carries two explicit labels: **In scope** — what this redesign commits to — and **Future seams** — what the design admits without warping, but is not building today.

---

## 1. Problem

The CVS DTNI (data-center training and inference) suites — inference (vllm, sglang, pytorch_xdit, inferencemax) and training (megatron, jax) — have grown by copy-paste-modify. Seven framework variants, ~5000 LOC across `cvs/tests/{inference,training}/`, ~80% duplication. Each new model is a new file. Each new framework is a new lib *and* a new test file.

Three problems compound:

1. **Robustness.** Silent-pass on missing thresholds (inferencemax). Pass/fail direction inferred from substring `"ms"` in the metric name (any future `latency_seconds` flips comparison). `fail_test()` does not actually call `pytest.fail` — multi-step tests continue past the first failure. HF tokens logged in plaintext on every multi-node run. `UnboundLocalError` in the HF-token fixture turns a missing-file error into a crash trace. `sudo rm -rf $log_dir` runs unquoted on user-supplied paths. Containers default to `--privileged` + `seccomp=unconfined`. `docker system prune --force` on shared nodes wipes other users' containers.
2. **Coverage.** Frameworks emit per-request JSONL, per-step trajectories, Prometheus metrics. CVS regex-greps the console and discards the rest. No loss-trajectory check, no per-rank jitter, no convergence sanity — a NaN-loss training run passes if the throughput line parsed.
3. **Scale.** Module-level `inf_res_dict = {}` and `inference_dict["_test_output_dir"]` mutation pass data between tests. Re-running `-k "verify"` against an already-run benchmark is impossible — the data is gone when the pytest process exits.

The dominant gap across all seven frameworks is "framework emits, CVS throws away" — not "framework doesn't emit."

---

## 2. Goals and non-goals

**Goals.**

- One typed config per (framework, model). No mega-configs. No stringified-param result dicts.
- One parametrized test per suite (`test_inference.py`, `test_training.py`). New model = new config file, not new test file.
- Trajectory and per-request data are first-class and retained on disk. Verification can be re-run without re-launching workloads.
- Failures classified into five disjoint categories the operator can act on (setup / safety / pattern-matched / liveness / verification).
- Every external resource (container, GPU, secret, ssh fan-out) is RAII-managed. Cleanup is guaranteed.
- Robustness over speed-to-first-deliverable.

**Non-goals.**

- Unifying CVS broadly. Scoped to DTNI; cross-suite changes (e.g. `globals.error_list` removal) ride along only as preconditions.
- Quality-metric implementations (CLIP / FID / LPIPS for xDiT; lm-eval-harness, GSM8K for LLM serving). The architecture exposes the seam; the implementations are follow-ups.
- Remote ingest deployment (Prometheus remote-write, OpenSearch, S3). The on-disk event schema is ingestion-ready; deployment is a follow-up.
- MLPerf official submission path (LoadGen, SPEC PTD power, TEST01/04/05, submission packaging). The Composite pattern and `accuracy_harness` seam keep this future-feasible; nothing lands today.

---

## 3. Core architecture

### 3.1 Lifecycle — Template Method

A single `Job` driver runs the same six steps in order for every workload, training or inference. No mode branching at the driver level.

**The `WorkloadAdapter` protocol.** Every adapter — Strategy or Composite, inference or training — satisfies this protocol. It is the closed contract between the `Job` driver and adapter implementations.

```python
# cvs/lib/adapter_protocol.py
class WorkloadAdapter(Protocol):
    def prepare(self, ctx: Context) -> None: ...
    def launch(self, ctx: Context) -> AdapterRun: ...
    def await_completion(self, run: AdapterRun) -> None: ...        # polls progress_predicate
    def progress_predicate(self, run: AdapterRun) -> ProgressStatus: ...
    def parse(self, run: AdapterRun, manifest: Manifest) -> WorkloadResult: ...
    def verify(self, result: WorkloadResult, thresholds: list[Threshold]) -> list[Verdict]: ...
    def teardown(self, run: AdapterRun) -> None: ...
```

The protocol has a **closed set** of methods. It does not gain new steps to accommodate one mode's needs. Cross-cutting behavior lives in hooks (§3.5), handles (§3.3), or adapter-internal helpers — never in the protocol. This is the guardrail that keeps a single spine safe across both training and inference.

**The `Job` driver.** Same six-step body for every workload. Failures are classified into the §4.5 five-category taxonomy at the boundary where they originate, not by post-hoc inspection of a stack trace. Teardown always runs.

```python
# cvs/lib/job.py
class Job:
    def __init__(self, adapter: WorkloadAdapter, cfg: BaseTestConfig,
                 cluster: ClusterConfig, gpu: GpuPlatform, secrets: Secrets):
        self.adapter = adapter
        self.ctx     = Context(cfg=cfg, cluster=cluster, gpu=gpu, secrets=secrets,
                               pssh=Pssh(cluster), run_id=mint_run_id())
        self.manifest = Manifest.create(self.ctx.test_id, cfg, cluster)

    def run(self) -> Manifest:
        run: AdapterRun | None = None
        try:
            self.manifest.append_event("prepare.start", source="job")
            self.adapter.prepare(self.ctx)

            run = self.adapter.launch(self.ctx)                     # raises -> setup_failure
            self.manifest.append_event("launch.role_ready", source="job")

            self._await_with_progress(run)                          # may raise safety_violation
                                                                    # or liveness_failure
            result = self.adapter.parse(run, self.manifest)
            self.manifest.record_result(result)

            verdicts = self.adapter.verify(result, self.ctx.cfg.thresholds)
            self.manifest.record_verdicts(verdicts)                 # status = pass | verification_failure

        except SetupFailure as e:
            self.manifest.record_failure("setup_failure", evidence=e.evidence)
        except SafetyViolation as e:
            self.manifest.record_failure("safety_violation",
                                         predicate_name=e.predicate, evidence=e.evidence)
        except LivenessFailure as e:
            self.manifest.record_failure("liveness_failure", evidence=e.evidence)
        except FailurePatternMatched as e:
            self.manifest.record_failure("failure_pattern_matched",
                                         pattern_id=e.pattern_id, evidence=e.line)
        finally:
            if run is not None:
                self.adapter.teardown(run)                          # RAII: always runs
            self.manifest.append_event("teardown.done", source="job")
            self.manifest.flush()

        return self.manifest

    def _await_with_progress(self, run: AdapterRun) -> None:
        deadline = time.monotonic() + self.ctx.cfg.await_timeout_sec
        while time.monotonic() < deadline:
            status = self.adapter.progress_predicate(run)
            if not status.ok:
                raise SafetyViolation(predicate=status.predicate_name,
                                      evidence=status.evidence)
            if self.adapter.await_completion_check(run):            # workload finished cleanly
                return
            time.sleep(self.ctx.cfg.poll_interval_sec)
        raise LivenessFailure(evidence=f"timeout after {self.ctx.cfg.await_timeout_sec}s")
```

Driver-level dispatch — the only place that picks an adapter by framework — is three lines. The driver itself is mode-blind:

```python
def run_workload(cfg, cluster, gpu, secrets) -> Manifest:
    registry = INFERENCE_REGISTRY if isinstance(cfg, InferenceTestConfig) else TRAINING_REGISTRY
    adapter  = registry[cfg.framework](cfg, cluster, gpu, secrets)
    return Job(adapter, cfg, cluster, gpu, secrets).run()
```

Whether `adapter` is a `VllmAdapter` (Strategy), a four-role sglang `CompositeAdapter`, or a sixteen-rank megatron `CompositeAdapter`, `Job.run()` is identical.

**In scope.**

- One `WorkloadAdapter` protocol (Python `Protocol`), six methods + `progress_predicate`.
- One `Job` driver. No `if mode == "training"` anywhere in the driver.
- Cross-cutting behavior goes through hooks (§3.5), handles (§3.3), or adapter-internal helpers — never into the protocol.

**Future seams.**

- If a future requirement genuinely needs a method only one mode needs, the answer is to split into `InferenceAdapter` and `TrainingAdapter` — a mechanical refactor in known places. The protocol does not drift into a `**kwargs` puddle.

### 3.2 Adapter topology — Strategy + Composite

Two patterns, chosen by a factory based on workload shape.

- **Strategy** for single-role workloads (vllm, inferencemax, xdit, single-node training).
- **Composite** for multi-role workloads (sglang disagg = prefill + decode + router + bench-client; distributed training = N symmetric workers). A `CompositeAdapter` *is itself* a `WorkloadAdapter`. The `Job` driver does not know which it has.

Strategy adapters are the unit of reuse. `SglangServerAdapter` is used twice inside the disagg Composite (once for prefill, once for decode, differing by a CLI flag). Today's `SglangDisaggPD` is monolithic — 1200 LOC at `cvs/lib/sglang_disagg_lib.py` that cannot be partially reused.

```python
@dataclass
class CompositeAdapter(WorkloadAdapter):
    sub_adapters:    list[WorkloadAdapter]
    sub_role_names:  list[str]
    launch_model:    Literal["barrier", "dag"]
    barrier_cohorts: list[list[int]]          # parallel within cohort; sequential between cohorts

    def progress_predicate(self, run):
        for sub in self.sub_adapters:          # conjunction over sub-roles
            s = sub.progress_predicate(run)
            if not s.ok: return s
        return Ok()
```

Two launch models:

- `"barrier"` — symmetric distributed training (torchrun, jax.distributed). All sub-adapters in a cohort start simultaneously to avoid racing collective bootstrap timeouts.
- `"dag"` — genuinely sequential pipelines (sglang disagg: servers → router → bench). Cohorts run in topological order; within a cohort, sub-adapters still launch in parallel.

**Worked example: sglang disaggregated PD as a Composite.**

The current `cvs/lib/sglang_disagg_lib.py` (1200 LOC) is replaced by a factory that builds four Strategy classes — `SglangServerAdapter` (reused for both prefill and decode roles, differing only by `mode`), `SglangRouterAdapter`, `SglangBenchAdapter` — and wires them into a single `CompositeAdapter`. The `Job` driver never sees the topology.

```python
def sglang_disagg_factory(cfg, cluster, gpu, secrets) -> CompositeAdapter:
    # 1. Per-role Strategy instances. SglangServerAdapter is reused twice.
    prefills = [SglangServerAdapter(node=n, mode="prefill",
                                    peer_addrs=PeerAddrs(decode_addrs=[d.addr for d in cluster.decode_nodes]))
                for n in cluster.prefill_nodes]
    decodes  = [SglangServerAdapter(node=n, mode="decode",
                                    peer_addrs=PeerAddrs(prefill_addrs=[p.addr for p in cluster.prefill_nodes]))
                for n in cluster.decode_nodes]
    router   =  SglangRouterAdapter(node=cluster.router_node,
                                    prefill_addrs=[p.addr for p in cluster.prefill_nodes],
                                    decode_addrs =[d.addr for d in cluster.decode_nodes])
    bench    =  SglangBenchAdapter(node=cluster.bench_node,
                                   router_addr=cluster.router_node.addr,
                                   workload=cfg.workload)

    # 2. Index every sub-adapter and lay out the DAG: servers -> router -> bench.
    sub_adapters   = [*prefills, *decodes, router, bench]
    sub_role_names = ([f"prefill-{i}" for i in range(len(prefills))] +
                      [f"decode-{i}"  for i in range(len(decodes))]  +
                      ["router", "bench"])
    server_idxs = list(range(len(prefills) + len(decodes)))
    router_idx  = len(server_idxs)
    bench_idx   = router_idx + 1

    return CompositeAdapter(
        sub_adapters=sub_adapters,
        sub_role_names=sub_role_names,
        launch_model="dag",
        barrier_cohorts=[
            server_idxs,         # cohort 1: every prefill + decode in parallel
            [router_idx],        # cohort 2: router (after servers ready)
            [bench_idx],         # cohort 3: bench-client (after router ready)
        ],
        depends_on={router_idx: server_idxs, bench_idx: [router_idx]},
        composite_progress_predicates=[
            RouterBalancePredicate(max_imbalance_pct=0.30),   # catches 90/10 routing
        ],
        composite_parsers=[
            PdHandoffJoinParser(),       # joins prefill/decode rows on request_id
            RouterTrajectoryParser(),    # router queue depth trajectory
        ],
    )
```

What this buys:

- **Reuse.** `SglangServerAdapter` is ~200 LOC and ships once; the disagg Composite invokes it twice (`mode="prefill"`, `mode="decode"`). Today's 1200-LOC monolith cannot be partially reused.
- **Correct launch ordering.** `barrier_cohorts` enforces that all servers come up before the router, and the router before the bench client — but every prefill and every decode launches *in parallel* within cohort 1. Today's sequential `test_launch_prefill_servers` → `test_launch_decode_servers` is unnecessarily serial.
- **Composite-level signal.** `RouterBalancePredicate` lifts a Composite-only invariant (no single server taking >70% of requests) into a `safety_violation` failure category. A stuck router with empty decode queues — silently passing today — surfaces mid-run.
- **Cross-role parsing.** `PdHandoffJoinParser` joins prefill-role rows and decode-role rows on `request_id`, producing the `cross_role_samples_rows` carrier (§4.1) with `{request_id, prefill_done_ns, decode_start_ns, kv_transfer_ms, e2e_ms}`. This is the only shape that can express prefill→decode handoff latency.
- **Driver indifference.** From the `Job` driver's perspective this returns a `WorkloadAdapter`. Same six lifecycle calls. No `if framework == "sglang_disagg"` branching anywhere.

**In scope.**

- `VllmAdapter`, `PytorchXditAdapter`, `InferenceMaxAdapter`, single-node training as Strategy.
- `SglangServerAdapter` (×2 in disagg), `SglangRouterAdapter`, `SglangBenchAdapter` inside a `dag`-mode `CompositeAdapter`.
- `MegatronWorkerAdapter`, `JaxWorkerAdapter` inside a `barrier`-mode `CompositeAdapter` (or as a cohort-of-1 Strategy when N=1).
- Factory per framework. The `Job` driver calls `REGISTRY[cfg.framework](...)` and gets back a `WorkloadAdapter` — never branches on topology.

**Future seams.**

- **MLPerf LoadGen** drops in as an additional sub-role under the existing Composite pattern — no protocol change.
- **TEST01/04/05 compliance scripts** drop in as decorators over an inner adapter — no protocol change.
- **TensorRT-LLM `trtllm-bench`** drops in as an alternative bench-client sub-role under the sglang/vllm Composite — same shape.

### 3.3 Resource handles — RAII

Four context-managed handles. Adapters use them; nothing else touches the underlying tools.

| Handle | Owns | Replaces |
|---|---|---|
| `Pssh` | All node communication. Localhost is opt-in via `cluster.localhost: bool`, not heuristic. Single log seam routes through the redactor. | Inline `subprocess.run`, duplicated `_is_local_target` and `LocalPssh` across xdit files |
| `GpuPlatform` | GPU vendor / family / device_count / device_paths / visible_env. `GpuPlatform.detect(phdl)` cross-checks PCI device ID + `rocm-smi`/`amd-smi` + `rocminfo` GFX version; disagreement fails config load. `SystemFingerprint.capture(pssh)` gathers CPU / RAM / kernel / NIC / container image digest for the manifest. | Inline `rocm-smi` parsing in 4 inference test files; `"mi355"` vs `"mi355x"` literal mismatch; no automated capture of "what changed between two runs?" forensic data |
| `SecretValue` | Token storage with type-level redaction. `repr`/`str` return `<SecretValue label=***>`. `.reveal()` only at `--env-file` write time. | Plaintext `HF_TOKEN` strings in command lines and logs |
| `ContainerHandle` | `docker run -d` with `run_id` label; readiness probe blocks `__enter__`; `__exit__` always captures logs + dmesg + `rocm-smi` and `docker rm` by label (not `system prune`). Non-privileged default. | `docker_lib.py` helpers, ad-hoc `try/finally` cleanup blocks, `docker system prune --force`, `--privileged` default |

**In scope.**

- All four handles; redactor registered at the Pssh log seam.
- `GpuPlatform.detect()` as the single source of truth for GPU identity.
- `SystemFingerprint.capture(pssh)` called once at `prepare.start`.
- Sentinel-leak guard: run with `HF_TOKEN=hf_LEAK_SENTINEL_xyz`; grep full session log + in-container `/tmp/*.sh` for the sentinel — must not find.

**Future seams.**

- **Wall-plug power capture** (SPEC PTD 1.11.1) lands as a new handle. Adapters use it; the protocol does not change.
- **K8s / Slurm orchestration backends** slot behind the `Pssh` interface. Adapters are unchanged.
- **New GPU SKUs** add a row to `PCI_ID_TO_FAMILY` in `handles.py`. No code changes.

### 3.4 Persistent manifest

Per-run `<artifact_dir>/<test_id>/manifest.json` + append-only `events.jsonl`. Manifest is the index; events are the time-ordered stream. State lives on disk, not in module globals.

```jsonc
// manifest.json (abbreviated)
{
  "run_id":  "20260115-221400-a7f3",
  "test_id": "test_inference[vllm-gpt-oss-120b-mi355x]",
  "config_hash": "sha256:...",
  "status":  "verification_failure",
  "system_desc": {
    "hosts": [{
      "node": "mi355x-node-01",
      "cpu": "AMD EPYC 9654 96-Core", "ram_gb": 1536,
      "kernel": "6.5.0-25-generic", "os": "Ubuntu 22.04.3 LTS",
      "gpus": [{ "vendor": "AMD", "family": "MI355X", "count": 8,
                 "vbios": "113-MI355X-...", "driver": "rocm-7.0.0" }],
      "nics": [{ "model": "Mellanox CX-7", "link_gbps": 400, "count": 8 }],
      "container": { "image": "rocm/vllm:7.0.0", "digest": "sha256:a91c..." }
    }]
  },
  "dataset_checksums": { "/datasets/gsm8k/test.jsonl": "sha256:e4f1b2..." },
  "failure": {
    "category": "verification_failure",
    "thresholds_failed": [
      { "kind": "Percentile", "metric": "ttft_ms", "value_seen": 78.2,
        "value_required": 50.0, "op": "<=" }
    ]
  },
  "result": {
    "samples":    { "ttft_ms": [...], "itl_ms": [...] },
    "trajectory": {},
    "scalars":    { "ttft_p99_ms": 78.2, "throughput_tokens_per_sec": 1184.0 },
    "status":     { "qps_sweep_complete": "yes" }
  }
}
```

`events.jsonl` is small and closed:

```jsonc
{ "ts": "...", "run_id": "...", "test_id": "...", "node": "n3", "role": "server",
  "event": "step", "source": "prometheus_poll",
  "fields": { "iter": 1024, "throughput": 4823.1, "loss": 2.34 } }
```

Event vocabulary: `prepare.{start,done}`, `launch.{container_up,role_ready}`, `seed.logged`, `arrival.{start,end}`, `accuracy.{start,end}`, `step`, `request`, `safety.violated`, `pattern.matched`, `parse.done`, `verify.{passed,failed}`, `teardown.{start,done}`.

**In scope.**

- `manifest.json` and `events.jsonl` per test, on devbox disk.
- `system_desc` (via `SystemFingerprint`) and `dataset_checksums` captured at `prepare.start` — a run is reconstructible from `<artifact_dir>/<test_id>/` alone.
- Re-runnability: `cvs run -k "parse and verify"` re-reads manifest + events without re-launching the workload. Re-verify against relaxed thresholds against the same samples is one command.
- Single closed event vocabulary above. Adding a new event name is a schema change, not a silent producer.

**Future seams.**

- **Remote ingest** — Prometheus remote-write, OpenSearch index, S3 archive. The event schema is unchanged; only the sink swaps.
- **MLPerf submission archive readiness** — `system_desc` + run reconstructibility is the precondition; submission packaging is the remaining work.
- **LoadGen event slot** — `loadgen.*` is intentionally not pre-allocated; added with LoadGen integration if/when that lands.

### 3.5 Hooks at the seam

Pytest hooks (`conftest.py`, `pytest_runtest_makereport`, `pytest_terminal_summary`) and the Pssh log seam are the single places where cross-cutting behavior lives.

- Failure-artifact capture (wrapping `ContainerHandle.__exit__` on the exception path).
- Cross-test results aggregation — replaces what `globals.error_list` did, via end-of-session iteration over per-test manifests.
- Secret redaction at the Pssh log seam — substring-replace any active `SecretValue.raw` before the log line is emitted.
- `FailurePatternScanner` (§4.4) registers here.

**In scope.**

- `pytest_runtest_makereport` writes failure context to manifest (best-effort, never propagates).
- `pytest_terminal_summary` reads every per-test manifest in the run directory and emits the cross-test summary.
- Redactor at `pssh.py:256` (the single `self.log.info(f'cmd = {full_cmd}')` line).
- `FailurePatternScanner` registered as a hook that scans during-run logs against `failure_patterns.yaml`.

**Future seams.**

- New cross-cutting behaviors (additional summary writers, alternative artifact destinations, in-process metric exporters) land here. They never land in the adapter protocol.

### 3.6 Typed config schema

Pydantic schemas, `extra = "forbid"`. Inference and training have separate registries so configs cannot route through the wrong mode. Typos fail at load, not after 20 minutes of compute.

```python
class BaseTestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["2"]
    target_gpu:     str                       # asserted against GpuPlatform.detect()
    cluster_ref:    Path
    secrets:        SecretsBlock
    seed:           SeedBlock = SeedBlock()
    thresholds:     list[Threshold]           # see §4.2

class InferenceTestConfig(BaseTestConfig):
    framework: Literal["vllm", "sglang_disagg", "inferencemax", "pytorch_xdit"]
    params:    VllmParams | SglangDisaggParams | InferenceMaxParams | PytorchXditParams
    scenario:  Literal["server", "offline", "single_stream", "multistream"] = "server"
    arrival:           ArrivalConfig | None = None
    accuracy_harness:  AccuracyHarnessConfig | None = None
    mlperf_inference_compliance: Literal["v6.0"] | None = None

class TrainingTestConfig(BaseTestConfig):
    framework: Literal["megatron", "jax"]
    params:    MegatronParams | JaxParams
    mlperf_training_compliance: Literal["v6.0"] | None = None
```

`scenario` (MLPerf-style traffic shape) and `framework` are orthogonal axes: any inference framework can serve under any scenario. `arrival` and `accuracy_harness` are inference-only optional sub-phase configs that the `Job` driver hands to `launch()` (see §4.3). `mlperf_*_compliance` is declarative only today — it documents which MLPerf round's SLO presets the config gates against; no enforcement validators.

**Intentionally absent.** `division` (CVS lives in MLPerf-Open methodology by default), `pair_hash` (single-config paired runs make pairing structural, not by-hash), `mode` enum (presence/absence of `accuracy_harness` is the discriminator).

**Rollout.** New schema is the only schema. A `cvs migrate-config` helper runs once per config: splits mega-configs (vllm 4-models, sglang 2-models) into one file per (framework, model), converts `result_dict[ISL=...,OSL=...,TP=...,CONC=...]` lookups into typed `Threshold` predicates (§4.2), forbids `<changeme>` sentinels, stamps `schema_version: "2"`. Worked example (inferencemax):

```jsonc
// Before
"result_dict": {
  "ISL=1024,OSL=1024,TP=8,CONC=64": { "throughput_tokens_per_sec": 1200, "ttft_ms": 50, "itl_ms": 25 }
}

// After
"thresholds": [
  { "kind": "Rate",       "metric": "throughput", "per_unit": "sec", "op": ">=", "min_rate": 1200 },
  { "kind": "Percentile", "metric": "ttft_ms",    "percentile": 99.0, "op": "<=", "value": 50 },
  { "kind": "Percentile", "metric": "itl_ms",     "percentile": 99.0, "op": "<=", "value": 25 }
]
```

**In scope.**

- `BaseTestConfig`, `InferenceTestConfig`, `TrainingTestConfig` with separate per-mode framework registries.
- `extra = "forbid"` everywhere — typo fails at `model_validate()`.
- `Threshold` union (§4.2). `target_gpu` asserted against detected GPU at config load.
- `seed`, `scenario`, `arrival`, `accuracy_harness`, `mlperf_*_compliance` fields land in the base schema even if their evaluators ship empty.
- `cvs migrate-config` helper covers every current config under `cvs/input/config_file/{training,inference}/`.

**Future seams.**

- **`mlperf_*_compliance` validator** — today declarative; future Pydantic validators can enforce HP-table / closed-division rules.
- **`threshold_table` syntactic sugar** for mega-configs — collapses a (ISL, OSL, concurrency) grid into a single block; expands to predicates at load. Pure sugar, no new evaluator.
- **`accuracy_harness.driver` extension** — `lm_eval_harness` is the first driver; MLPerf accuracy harnesses land alongside.

---

## 4. Result and verification model

### 4.1 Temporal result types

Results carry explicit temporal structure rather than flattening to scalars. Without this, "verify throughput ≥ X" and "verify loss decreased monotonically over the last 1000 steps" would have to be encoded the same way — which means encoding one of them badly.

```python
@dataclass
class WorkloadResult:
    samples:    dict[str, list[float]]              # iid: per-request latency, per-image FPS
    trajectory: dict[str, list[tuple[int, float]]]  # time-ordered: (step|ts, value)
    scalars:    dict[str, float]                    # derived: P99, median, slope, final
    status:     dict[str, str]                      # categorical: completion flags, validity
```

- **Inference** populates `samples` (per-request latencies) and `scalars` (derived P50/P99/mean).
- **Training** populates `trajectory` (per-iter loss, throughput, step time, grad norm) and `scalars` (median throughput, final loss).
- **`status`** carries categorical fields that don't coerce to floats — completion flags (`qps_sweep_complete: "yes"`), accuracy-mode markers, and forward-compatibility slots. Empty by default; populated by `parse()`.
- **Composite** nests by role name: `{role: WorkloadResult}`. Sub-role rows are *not* flattened into the top-level container (role attribution would be lost).

**In scope.**

- `WorkloadResult` with all four carriers.
- Per-rank trajectories for training adapters (today's `tail -15 last-node/training.log` loses per-rank signal; redesign reads from every rank).
- Composite nesting rule: sub-role results in `per_role`; only cross-role joined data (e.g. prefill→decode handoff) at the top level.

**Future seams.**

- **`artifacts`** field for side-channel data (paths to generated images / video frames) — read by quality-evaluator predicates without touching the temporal carriers.

### 4.2 Threshold predicates

A `Threshold` is a typed predicate, not a `(metric, op, value)` triple. Six kinds, each with its own temporal semantics.

```python
class PercentileThreshold:    # over samples
    percentile: float; op: Op; value: float
    # MLPerf-shaped default is p99 (Server/Multistream); p90 only for SingleStream.

class MonotonicityThreshold:  # over trajectory
    window: tuple[int, int] | Literal["last_quarter"]
    direction: Literal["non_increasing", "non_decreasing"]
    tolerance: float

class ConvergenceThreshold:   # over trajectory
    target: float; epsilon: float
    by_step: int | None = None             # one of by_step xor by_wallclock_sec
    by_wallclock_sec: float | None = None  # MLPerf training's primary metric

class StabilityThreshold:     # rolling-variance bound over samples or trajectory
    window_size: int; max_variance: float

class RateThreshold:          # derived from samples or trajectory
    per_unit: Literal["sec", "step"]; op: Op; min_rate: float

class GoodputThreshold:       # filtered rate over samples (the MLPerf headline metric)
    metric_pair: dict[Literal["ttft", "tpot"], str]
    ttft_max_ms: float; tpot_max_ms: float
    op: Op; min_qps: float
    # Evaluates: requests/sec where (ttft <= ttft_max_ms) AND (tpot <= tpot_max_ms).
    # Rate + Percentile cannot compose to express this filtered rate.
```

Worked example for inference (vllm Llama-2-70B, MLPerf v6.0 Interactive SLO):

```jsonc
"thresholds": [
  { "kind": "Goodput", "metric_pair": { "ttft": "ttft_ms", "tpot": "tpot_ms" },
    "ttft_max_ms": 450.0, "tpot_max_ms": 40.0, "op": ">=", "min_qps": 600.0 },
  { "kind": "Percentile", "metric": "ttft_ms", "percentile": 99.0, "op": "<=", "value": 450.0 },
  { "kind": "Percentile", "metric": "tpot_ms", "percentile": 99.0, "op": "<=", "value":  40.0 },
  { "kind": "Stability",  "metric": "ttft_ms", "window_size": 500, "max_variance": 2500.0 }
]
```

Worked example for training (megatron Llama 70B):

```jsonc
"thresholds": [
  { "kind": "Rate",         "metric": "throughput",   "per_unit": "step", "op": ">=", "min_rate": 4800 },
  { "kind": "Monotonicity", "metric": "loss",         "window": "last_quarter", "direction": "non_increasing", "tolerance": 0.02 },
  { "kind": "Convergence",  "metric": "loss",         "target": 2.1, "epsilon": 0.1, "by_wallclock_sec": 14400.0 },
  { "kind": "Stability",    "metric": "step_time_ms", "window_size": 50, "max_variance": 25.0 }
]
```

Both modes can use most predicates: inference may want `Stability` to detect warmup jitter; training may want `Percentile` on step time to bound worst-case per-iter slowness. `Goodput` is inference-only.

**In scope.**

- All six predicates above with explicit `op` (no substring-based direction inference).
- Named metric registry — unknown metric strings fail config load.
- Per-framework "already-emitted but discarded" data populated into `WorkloadResult`:

  | Framework | Lands now |
  |---|---|
  | vllm | Per-request JSONL → `samples`; Prometheus queue depth + memory pressure → `trajectory` |
  | sglang | Router queue trajectory; per-role startup times; decode KV-cache utilization |
  | xdit-Flux | Per-step latency trajectory; per-image generation time |
  | xdit-WAN | Per-frame latency; bitrate stats |
  | inferencemax | Per-batch JSON (already present but ignored) |
  | megatron | Full trajectory (loss, throughput, step time, grad norm); per-rank logs |
  | jax | Per-step JSONL trajectory; per-host metrics from the coordinator role |

**Future seams.**

- **`QualityThreshold(metric, evaluator)`** — a one-screen class. The evaluator is a callable that takes `WorkloadResult` plus optional side-channel data and returns a verdict. Implementations follow:
  - **CLIP / FID / LPIPS / GenEval** for xDiT-Flux image quality
  - **PSNR / SSIM / VBench** for xDiT-WAN video quality
  - **lm-eval-harness** (MMLU, GSM8K, HumanEval, IFEval) for LLM accuracy via the `accuracy_harness` sub-phase
  - **RULER @32K / @128K** for long-context accuracy
  - **HELM Safety / VHELM** for bias/toxicity
- **Memory headroom / collective comm fraction / MFU** for training — all inputs are in `WorkloadResult.trajectory`; the predicates are mechanical. Not landing today.
- **Tokens/sec/W** (`rocm-smi` package power) — indicative cross-arch perf/W; wall-plug power is separately a `Future seam` under §3.3.

### 4.3 Progress predicates

Small, code-defined invariants — *what it means for the test to be meaningfully running.* The `Job` driver polls these during `await_completion`; a violation becomes `safety_violation` in the manifest.

| Adapter | `progress_predicate` |
|---|---|
| `MegatronAdapter` / `JaxAdapter` | `step_counter_increasing ∧ loss_is_finite` |
| `VllmAdapter` / `SglangServerAdapter` / `InferenceMaxAdapter` | `server_responds_to_health_probe` |
| `PytorchXditAdapter` | `process_alive ∧ no_stuck_step` |
| `CompositeAdapter` | `∧` over sub-roles' predicates |

**Sub-phases inside `launch()`.** Optional `arrival` and `accuracy_harness` configs (§3.6) execute as *sequential sub-phases inside the adapter's `launch()`* against the same long-running SUT. They are **not** new lifecycle methods, **not** new progress predicates, and **not** separate `WorkloadAdapter` instances. The SUT's existing predicate (`server_responds_to_health_probe`) covers both phases; a sub-phase failure raises and the driver classifies as `setup_failure`. This preserves the §3.1 protocol-stability invariant: paired perf + accuracy is a single test with two clients hitting one server, not a `run_accuracy_pass()` method.

**In scope.**

- Per-adapter named predicates as above.
- `CompositeAdapter` conjunction over sub-roles.
- Sub-phase invocation inside `launch()` for `arrival` (Poisson sweep) and `accuracy_harness`.

**Future seams.**

- **Composite-level predicates** (e.g. `PerRankStepSyncWithin`, `PerRankThroughputSkewBelow`) extend the same predicate language without protocol changes — straggler-rank detection is a future addition under the existing seam.
- **Router-balance predicates** for sglang disagg (`RouterBalancePredicate(max_imbalance_pct=0.30)`) similarly extend without protocol changes.

### 4.4 Failure pattern catalog

Unbounded set of "bad things in log streams" (OOM, RCCL timeout, HBM ECC, PCIe AER, thermal throttle). YAML, not class hierarchy — adding a new pattern is a YAML edit, no code changes.

```yaml
- id: oom_kill
  source: dmesg
  pattern: "Out of memory: Killed process"
  severity: fatal
  hint: "Container OOM-killed; check workload memory footprint."

- id: rccl_timeout
  source: framework_log
  pattern: "NCCL.*Async operation timeout|RCCL.*timed out"
  severity: fatal
  hint: "RCCL/NCCL collective timed out; check network/topology and ulimit."
```

**In scope.**

- `failure_patterns.yaml` seeded with ~8 high-confidence patterns (OOM, RCCL timeout, HBM ECC, PCIe AER, thermal throttle, OOM-killer dmesg, kernel panic, NCCL watchdog warning).
- `FailurePatternScanner` registered as a hook (§3.5); scans `tail -n 200` of each declared `source_path` per node, per run, periodically.
- Matches recorded as `pattern.matched` events with `{id, severity, line, node}`.

**Future seams.**

- Pattern additions are pure data — no code review beyond the YAML diff.
- **Per-pattern auto-remediation** (e.g. `dmesg -C` + retry on transient ECC) lands as new fields on the pattern row; the scanner already has the hook.

### 4.5 Five-category failure taxonomy

Each comes from a different mechanism above and they are prioritized: setup > safety > pattern > liveness > verification.

| Category | Trigger |
|---|---|
| `setup_failure` | `prepare` or `launch` raised before workload started |
| `safety_violation` | Progress predicate broke mid-run |
| `failure_pattern_matched` | Catalog pattern hit |
| `liveness_failure` | `await_completion` timed out without progress predicate breaking |
| `verification_failure` | Threshold evaluated to `False` at end-of-test |

**In scope.**

- All five categories surface in `manifest.failure.category`.
- Single non-zero exit code today; category lives in the manifest.

**Future seams.**

- **Category-specific exit codes (2–6)** plus pytest-html badge for CI gating — small change once the manifest writes are stable.

---

## 5. MLPerf positioning

CVS adopts MLPerf **methodology** and explicitly does not adopt MLPerf **workflow choreography**. The two are separate decisions.

| Methodology (adopted now) | Workflow (not adopted) |
|---|---|
| Poisson arrival at target QPS via `bench_serving --request-rate <lambda>` | LoadGen as the canonical traffic generator |
| p99 tail-latency gating (Server / Multistream) | MT19937-specifically seeded traces |
| Goodput-at-SLO as the headline customer metric | Submission packaging (`results/`, `measurements/`, `system_desc_id.json`) |
| Paired (perf, accuracy) verdict against the same SUT | Two separate runs sharing a `pair_hash` — pairing is **structural** (one config, sub-phases in `launch()`), not adversarial |
| Per-model SLO presets from published MLPerf rounds (Conversational + Interactive) | Strict QSL sample counts; benchmark-specific MLPerf accuracy harness |
| Time-to-target quality in wallclock for training | Reference Convergence Point (RCP) envelopes |
| `rocm-smi` package-power telemetry for relative perf/W | MLPerf Power via SPEC PTD 1.11.1 (wall-plug, audit-grade) |
| Logged-seed convergence sanity | `mlperf_logging.mllog` as a runtime dependency |

CVS lives in the **MLPerf-Open** division by default (any framework, any model, any precision). MLPerf-Closed enforcement (HP verifier table, reference-impl framework gating, ECC/networking JSON) is out of scope — CVS users do not submit and are not adversarial.

**Paired runs are one test, not two.** A single config with both `arrival` and `accuracy_harness` blocks produces a single `Job` lifecycle: vllm starts, `bench_serving --request-rate <lambda>` Poisson sweep runs, `lm_eval --model local-completions` runs against the same vllm, both write into one `WorkloadResult`, all thresholds (perf + accuracy) evaluate against one verdict. The MLPerf "same code for perf and accuracy" rule is satisfied by *construction*, not by *enforcement*.

**In scope (methodology).** Every row in the left column above is reachable through the architecture in §3–§4. The vllm Llama-2-70B example in §4.2 evaluates the MLPerf v6.0 Interactive SLO directly. Per-model SLO presets (Conversational + Interactive) for the model families in production (Llama-3.1-8B/70B/405B, DeepSeek-R1, GPT-OSS-120B) are named gate presets selectable in config.

**Future seams (workflow).** Every row in the right column drops in without warping the design:

- **LoadGen** lands as a future sub-role under the existing Composite pattern (§3.2)
- **MLPerf accuracy harnesses** land alongside `lm_eval_harness` as additional `AccuracyHarnessConfig.driver` values (§3.6 / §4.3)
- **TEST01/04/05 compliance scripts** land as decorators over an inner adapter (§3.2)
- **Submission packaging** consumes `manifest.json` + `events.jsonl` + `system_desc` directly (§3.4)
- **SPEC PTD wall-plug power** lands as a new `PowerHandle` (§3.3)

**Open decision for this review: wrap vs absorb.**

- **Wrap.** `MLPerfTrainingAdapter` and `MLPerfInferenceAdapter` run the official MLPerf binaries as workloads. Two new adapters; coupling to MLPerf upstream is high; comparability with submitted numbers is exact.
- **Absorb.** Lift MLPerf workload definitions (model, dataset, target throughput, target accuracy) into typed CVS configs and run through existing per-framework adapters. New code is config + per-adapter knobs; coupling is low; comparability is approximate.

Both paths drop in cleanly. The first move sets the mental model contributors reach for. **If the intent is MLPerf submission / direct comparability with submitted numbers → wrap. If the intent is MLPerf workloads as a coverage matrix for harness-level validation → absorb.** Decision required at this review.

---

## 6. In scope now vs future seams — consolidated view

| Section | In scope now | Future seams |
|---|---|---|
| §3.1 Lifecycle | 6-step `WorkloadAdapter` protocol; `Job` driver; no mode branching | Split into `InferenceAdapter` / `TrainingAdapter` only if a method ever needs to diverge |
| §3.2 Adapter topology | Strategy + Composite; barrier-launch for training; DAG-launch for sglang disagg | MLPerf LoadGen sub-role; TEST01/04/05 decorators; `trtllm-bench` alternative client |
| §3.3 Resource handles | `Pssh` + redactor; `GpuPlatform` + `SystemFingerprint`; `SecretValue`; `ContainerHandle` | Wall-plug `PowerHandle` (SPEC PTD); K8s/Slurm backends behind `Pssh`; new SKUs via data-only edit |
| §3.4 Manifest | `manifest.json` + `events.jsonl` on disk; `system_desc` + `dataset_checksums`; re-runnable verify | Remote ingest (Prometheus remote-write, OpenSearch, S3); MLPerf submission archive; LoadGen event slot |
| §3.5 Hooks | `pytest_runtest_makereport`, `pytest_terminal_summary`, redactor seam, `FailurePatternScanner` | Additional summary writers / artifact destinations land here, not in the protocol |
| §3.6 Typed config | Discriminated union, `extra="forbid"`, `cvs migrate-config`, `scenario` / `arrival` / `accuracy_harness` / `seed` / `mlperf_*_compliance` fields | `mlperf_*_compliance` validators; `threshold_table` syntactic sugar; new `accuracy_harness.driver` values |
| §4.1 Result types | `samples` / `trajectory` / `scalars` / `status`; per-rank trajectories; composite nesting | `artifacts` field for side-channel data |
| §4.2 Thresholds | Six predicates (Percentile, Monotonicity, Convergence, Stability, Rate, Goodput); named metric registry; framework-emitted data populated | `QualityThreshold` for CLIP/FID/LPIPS/PSNR/SSIM/VBench; lm-eval-harness, RULER, HELM via `accuracy_harness`; memory headroom / collective comm fraction / MFU thresholds; tokens/sec/W |
| §4.3 Progress predicates | Per-adapter named predicates; Composite conjunction; `arrival` and `accuracy_harness` sub-phases inside `launch()` | Composite-level predicates: `PerRankStepSyncWithin`, `PerRankThroughputSkewBelow`, `RouterBalancePredicate` |
| §4.4 Failure patterns | `failure_patterns.yaml` with ~8 seed patterns; scanner as hook | Pattern additions are pure data; per-pattern auto-remediation hooks |
| §4.5 Failure taxonomy | Five categories, prioritized; single non-zero exit code | Category-specific exit codes (2–6) + pytest-html badge |
| §5 MLPerf | Methodology adopted (Poisson, p99, Goodput, paired verdict, SLO presets, wallclock convergence, package power, seed) | Workflow adoption — LoadGen, MLPerf accuracy harnesses, TEST01/04/05, SPEC PTD power, submission packaging. **Wrap vs absorb: decision required at this review.** |

---

*End of PRD.*
