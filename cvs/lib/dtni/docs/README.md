# DTNI v1 Architecture Walkthrough

A top-to-bottom tour of the DeTermINistic Inference subsystem: what it is, how a run flows through it, and where every moving part lives. Read this first; reach for the recipe docs in this directory once you know which seam you are working on.

## What DTNI is

DTNI (DeTermINistic Inference) is the CVS subsystem for validating LLM inference workloads — vLLM today, sglang and others later — against a fixed bar of correctness and performance. It lives under `cvs/lib/dtni/` and is invoked by the standard `cvs run <suite>` entrypoint. A "suite" here is a `(framework, model, variant)` triple expressed as a pytest test parametrized over a `threshold.json`; one CVS invocation produces one verdict per metric.

The "deterministic" in the name is operational, not algorithmic. Given the same `config.json`, the same `threshold.json`, the same cluster file, and the same image digest, a run produces the same verdict. State that would break that promise — timestamps, random ports, ad-hoc host paths — is either elided from the workload hash or pushed into `ctx.scratch` where it cannot leak into the verdict.

Without DTNI, validating each serving stack means bespoke launch scripts, bespoke result parsing, bespoke threshold checks, and no comparability between runs of vLLM vs sglang vs whatever ships next. DTNI gives the cluster team one CLI, one config shape, one verdict shape, and one inline threshold engine — so adding a new framework is a new adapter class and a registry line, not a new validation pipeline.

DTNI is also unit-testable end-to-end without a cluster. Every shell call goes through a `ctx.executor` seam (see Section 6); CI swaps in `LocalExecutor` and runs the adapter logic locally with no SSH, no docker. The same adapter code that drives a production MI300X cluster runs under pytest with a fake runner.

Recipe: see `overview.md` for the orientation pointer that brought you here.

## The mental model

A run is a pipeline. CVS resolves a suite name to a pytest test; the test builds a `RunContext` from the cluster file, the workload config, and the threshold file; the framework adapter is fetched from `FRAMEWORK_REGISTRY`; a `Job(adapter, ctx)` drives the adapter through five phases (`prepare`, `launch`, `await`, `parse`, `teardown`) with one inline `verify` between `parse` and `teardown`. The first phase to raise `WorkloadError` aborts the chain but `teardown` always runs. The result is a `JobResult` plus a verdict dict written to disk.

```
  cvs CLI                    DTNI layer                              Cluster
  -------                    ----------                              -------
  cvs run vllm_single
      |
      v
  pytest collection --> conftest.py + cvs/tests/dtni/vllm_single.py
      |                  (collects one node per threshold metric)
      v
  workload_outcome fixture --> execute_workload(cluster, config)
      |
      v
  RunContext built  (executor, bindings, thresholds, workload, run_id)
      |
      v
  adapter = FRAMEWORK_REGISTRY[framework]()
      |
      v
  Job(adapter, ctx).run()
      |
      +--> prepare      -- validate roles, probe GPU count
      +--> launch       -- pull image, start container per role  ----> docker run
      +--> await        -- progress_predicate poll loop
      +--> parse        -- smoke + benchmarks; fill ctx.result.scalars
      +--> verify       -- (inline in Job) evaluate_all vs thresholds
      +--> teardown     -- capture logs, docker rm by label
      |
      v
  JobResult --> verdict.json + log + parquet artifacts under ./cvs_artifacts/<run_id>/
      |
      v
  pytest emits one PASS/FAIL node per threshold metric
```

Two things are worth repeating because they trip people up. First, `verify` is not an adapter method in v1 — it is open-coded inside `Job.run()` in `cvs/lib/dtni/job.py`. Adapters populate `ctx.result.scalars` during `parse` and the Job calls `verdict.evaluate_all(ctx.thresholds, ctx.result.scalars)` itself. Second, the pytest layer collects one node *per threshold metric*, not one node per workload — a single workload run produces N pass/fail rows, sharing one `Job` invocation via a module-scoped fixture.

Recipe: see `overview.md`.

## Anatomy of a run

Walk through one concrete invocation: `cvs run vllm_single --cluster_file=cluster.json --config_file=cvs/input/dtni/vllm_single/llama_3_1_8b/llama_3_1_8b_bf16_single_1_perf/config.json`.

1. **CLI dispatch.** `cvs run vllm_single ...` resolves the suite name `vllm_single` to the pytest entrypoint at `cvs/tests/dtni/vllm_single.py` (note: no `test_` prefix; the suite resolver finds it by name) and runs pytest with the user's `--cluster_file` / `--config_file` arguments.

2. **Collection.** Pytest loads `cvs/tests/dtni/conftest.py`. The `pytest_generate_tests` hook reads `threshold.json` sibling of `--config_file`, parametrizes the `metric` argument with each threshold key, and emits one `test_threshold[<metric>]` node per key. The shadow conftest at `cvs/lib/dtni/unittests/conftest.py` is empty — it exists only to insulate the unit-test directory from the repo-wide conftest's HTML-report dependencies.

3. **Workload fixture.** The first `test_threshold` node triggers the module-scoped `workload_outcome` fixture, which calls `cvs.lib.dtni.runner.execute_workload(cluster_path, workload_config_path)`. Subsequent nodes reuse the cached fixture — the workload runs *once*.

4. **`execute_workload` orchestrates.** It loads the cluster file, builds a `MultiHostExecutor` for the head node, probes the GPU arch via `detect_arch_via`, then loads the workload config through `WorkloadConfig.model_validate` (Pydantic v2). Catalog cross-checks reject unknown `model.id` / `dataset.id` / benchmark ids with "did you mean..." hints. Paths are resolved (the `paths` block is templated; `{shared_fs}` etc. expand via `substitution.resolve_paths_block`). A `run_id` like `r_20260607T143000Z_a1b2` is generated.

5. **Bindings + executor.** `topology.resolve_bindings` walks the cluster's `node_dict` in key order, allocates `count` hosts per role in declaration order, and returns `{role: [hosts]}`. A new `MultiHostExecutor` is built spanning every host involved (including the head node if not already present). Image digests are resolved per host so the workload hash can pin them.

6. **Adapter resolved.** `adapter = FRAMEWORK_REGISTRY["vllm_single"]()` returns a fresh `VllmAdapter`. The registry is a plain dict in `cvs/lib/dtni/frameworks/registry.py`; adapters do not self-register, they are imported by the registry module.

7. **`Job(adapter, ctx).run()` enters the pipeline.** Each phase is a `_phase(name, fn)` call that tags any raised `WorkloadError` with its phase.

   - `prepare` validates the role contract (declared roles match `required_roles` / `optional_roles`) and probes `rocm-smi` per host to populate `ctx.scratch["host_gpus"]`.
   - `launch` for vLLM substitutes tokens in the role's `command` / `env` / `volumes`, bind-mounts the model and the host artifacts dir, and calls `self._launch_role("server", ...)` which `docker run -d`s the vLLM image with `network=host`, `ipc=host`, the requested GPU devices, and the run-id label. Then `_wait_http_pool("server", "/health", 8000, 1800s)` polls `curl localhost:8000/health` until 200.
   - `await_completion` for vLLM is the default no-op (`progress_predicate` returns `"done"`) — vLLM serves forever; the `parse` phase drives the lifetime.
   - `parse` runs the smoke completion (`POST /v1/completions` with prompt "Once upon a time"), writes `smoke_request_latency_ms` / `smoke_completion_tokens` / `smoke_throughput_tok_s` into `ctx.result.scalars`, then for each benchmark in `workload.benchmarks` runs `vllm bench serve` via `docker exec` inside the live container, reads `result.json`, and merges projected scalars (`serve_synth_short.ttft_p95_ms`, etc.).
   - `verify` (inline in `job.py`) calls `evaluate_all(ctx.thresholds, ctx.result.scalars)`. If any verdict fails, the Job raises `WorkloadError(phase="verify")` so the failure is captured even though it is not an adapter method.
   - `teardown` walks `ctx.containers`, captures `docker logs` / `dmesg` / `rocm-smi` snapshots into `ctx.logs`, then removes every container with the run-id label. Teardown errors do not mask a real earlier failure (`failed_phase` is only set to `"teardown"` if nothing else failed).

8. **Artifacts.** `execute_workload` flattens `ctx.logs` into one `.log` file, dumps `ctx.result.scalars` as a one-row parquet, and writes `verdict.json` under `./cvs_artifacts/<run_id>/` with the basename `{arch}_{model_id}_{framework}_{workload_hash}_{ts}`.

9. **Pytest emits verdicts.** Each `test_threshold[<metric>]` node looks up its metric in `result.verdicts` and asserts `passed`. Phase failures other than `verify` cause every metric node to *skip* with the phase error; only `verify` failures produce per-metric `FAIL`.

Recipe: see `add_workload.md` for the variant-authoring path.

## The phase pipeline in detail

Five adapter phases plus one inline verify phase, in the order `Job.run()` calls them. Pitfalls are the ones we have actually hit.

### prepare

The adapter validates that `ctx.bindings` covers `required_roles`, has no empty role bindings, and declares no roles outside `required_roles | optional_roles`. Then it probes `rocm-smi --showproductname` on every host and stores per-host GPU counts in `ctx.scratch["host_gpus"]`. Common pitfall: an adapter that adds a new required role but forgets the topology block in `config.json` will fail here with a clear `missing required roles` message — that is the intended behavior, not a bug to work around.

### launch

The adapter builds the per-role `docker run` invocation and starts one container per host bound to the role, using `_launch_role` to fan out. Each handle is registered in `ctx.containers` *and* `self.handles_by_role[role]`. The launch helper does substitution on the operator-authored `command` / `env` / `volumes` (`{model.path}`, `{run_id}`, etc.) and asks for HTTP readiness via `_wait_http_pool` when the role has a health endpoint. Common pitfall: bypassing `_launch_role` and calling `docker run` directly via `ctx.executor.exec` — this skips the run-id label, which means teardown cannot reclaim the container, and it skips `handles_by_role`, which means `parse` cannot find its own container.

### await_completion

The default loops on `progress_predicate(ctx)` returning `"done"` / `"running"` / `"broken"` with `poll_interval_s` between polls, `completion_timeout_s` as the deadline. vLLM overrides nothing and the predicate returns `"done"` immediately — vLLM serves forever, and there is no end-of-run signal to wait for. Frameworks that have one (a finite eval loop, a benchmark binary that exits) override `progress_predicate` and let `await` drive their lifetime. Common pitfall: putting the "is the server up" check here instead of in `launch`. That check belongs to `_wait_http_pool` in `launch`; `await` is for "has the workload finished doing what it came to do."

### parse

The adapter populates `ctx.result.scalars` from whatever the workload produced — log files, completion responses, benchmark JSON. The contract is mandatory: every adapter MUST emit `smoke_request_latency_ms` and `smoke_completion_tokens` here (see Section 11). Benchmark scalars come via `cvs.lib.dtni.benchmarks.runner.run_benchmarks`, which projects each harness's JSON output through `PROJECTORS[harness]` and merges the result. Common pitfall: writing scalar names that do not match the threshold file's keys — the verdict will report `metric not produced by adapter` (a FAIL) even though the adapter ran fine.

### verify (inline in job.py)

Not an adapter method. `Job.run()` calls `evaluate_all(ctx.thresholds, ctx.result.scalars)` directly. Each named threshold produces one `Verdict` dict (`metric`, `actual`, `threshold`, `kind`, `passed`, optional `note`). The Job raises `WorkloadError(phase="verify")` if any verdict fails so the failure flows through the same error path as the adapter phases. Common pitfall: trying to override `verify` in an adapter subclass — there is no such hook in v1; if you want custom verdict logic, project to scalars in `parse` and let the threshold engine handle it.

### teardown

Always runs, even when an earlier phase raised. The base implementation walks `ctx.containers`, calls `handle.capture()` (best-effort `docker logs`, `dmesg`, `rocm-smi`) into `ctx.logs[f"{name}.{artifact}"]`, then `handle.remove()` which does `docker rm -f` filtered by the run-id label. Teardown errors are swallowed if there is already a real failure; they become the `failed_phase` only when nothing else failed. Common pitfall: assuming `ctx.containers` is empty on the failure path — it is not; partial launches leave handles registered for forensics.

The ordering invariant: phases run in order; the first to raise `WorkloadError` aborts the rest of the *forward* chain, but `teardown` always runs from the `finally` block. See `cvs/lib/dtni/job.py` for the exact try/except/finally shape.

Recipe: see `framework/02_phases.md` for per-phase contracts and `framework/RUNBOOK.md` for adapter scaffolding.

## RunContext — the carrier object

`RunContext` (`cvs/lib/dtni/run_context.py`) is the mutable dataclass threaded through every phase. Adapters read from it, write to a known subset of it, and never replace it. Field by field:

| Field | Type | Written by | Purpose |
|---|---|---|---|
| `run_id` | str | Job builder | Unique tag like `r_20260607T143000Z_a1b2`. Used as the docker label and the artifacts subdir name. |
| `arch` | str | Job builder | Detected GPU arch (`mi300x`, `mi325x`, ...). Must match `workload.gpu_arch`. |
| `cluster` | dict | Job builder | Pruned cluster file (`username`, `head`, `node_dict`). |
| `workload` | dict | Job builder | Pydantic-validated `WorkloadConfig.model_dump()`. The role spec under `roles[<role>]` is mutated in-place to inject the run-specific volumes (model mount, artifacts mount). |
| `thresholds` | dict | Job builder | Raw `threshold.json` contents. Read-only after that. |
| `workload_name` | str | Job builder | `"<framework>/<variant>"`, used in verdict.json. |
| `workload_hash` | str | Job builder | SHA over `(image_digest, workload, thresholds)`. The determinism handle. |
| `bindings` | dict[str, list[str]] | Job builder | `role -> [host, ...]`. Adapters read this in `launch` to know where to start containers. |
| `executor` | MultiHostExecutor \| LocalExecutor | Job builder | The shell seam. See Section 6. |
| `artifacts_dir` | Path | Job builder | Per-run host directory; bind-mounted into containers as `/cvs_artifacts` (the OUTPUT_DIR_IN_CONTAINER constant). |
| `containers` | list[ContainerHandle] | Adapter (via `_register`) | Every launched handle. Drained by `teardown`. |
| `scratch` | dict | Adapter | Free-form per-run bookkeeping. The Job pre-seeds `sub_ctx`, `model_host_path`, `dataset_host_path`. |
| `logs` | dict[str, str] | Adapter / `teardown` | `{f"{container_name}.{artifact}": text}`. Drained into the run's `.log` by `artifact_writer`. |
| `result.scalars` | dict[str, float] | Adapter (in `parse`) | The one and only input to the threshold engine. |
| `events` | _NullEvents | Job builder | No-op sink in v1; reserved for structured events. |

`ctx.framework` is a derived property reading `workload["framework"]`. There is no setter — change the workload dict, not the property.

Recipe: see `framework/01_adapter_contract.md` for the full adapter-side view of the carrier.

## The executor seam

The executor is the single most important architectural decision in DTNI. Adapters never touch `subprocess`, `paramiko`, or `os.system` directly. Every shell command goes through one shape:

```python
host_exec = ctx.executor.executor_for(host)   # per-host view
out = host_exec.exec(cmd, timeout=...)         # str return, raises on non-zero
```

There are two implementations and they are duck-typed (no shared ABC):

- **`MultiHostExecutor`** (`cvs/lib/dtni/executor.py`) wraps `MultiProcessPssh`. Its `exec(cmd)` broadcasts to every host; `executor_for(host)` returns a `_PsshHostView` whose `exec(cmd) -> str` runs on that one host. Used by `cvs run` in production.
- **`LocalExecutor`** (same file) is `subprocess.run(["bash", "-c", cmd], ...)`. Returns stdout+stderr as a string, raises `RuntimeError` on non-zero exit. Used by unit tests and any host-local CVS deployment.

Both expose `exec(cmd, timeout=...) -> str | dict` and either an `executor_for(host)` method (MultiHost) or are themselves a per-host executor (Local). Adapters check `hasattr(ctx.executor, "executor_for")` to handle both shapes; the base class does this in `_discover_gpu_count`, `_launch_role`, and `_wait_http_pool` so subclasses do not have to.

This seam is what makes adapters unit-testable without a cluster. The DTNI unittests under `cvs/lib/dtni/unittests/` build a `RunContext` with a `LocalExecutor` (or a `unittest.mock.Mock` shaped like one) and exercise the same adapter code paths that ship to production. Adapters that bypass the seam — hardcoding `ssh foo` strings or shelling out to `subprocess.run` directly — silently break the test story. This is the single most common adapter-author mistake; review every new adapter for it.

`ContainerHandle` participates in the same seam: it stores a `runner` argument (a per-host executor), and every docker call goes through `self.runner.exec`. Unit tests pass a fake runner with a recorded `exec` and assert against the captured commands.

Recipe: see `framework/01_adapter_contract.md` (Executor seam section).

## Adapter architecture

`BaseWorkloadAdapter` (`cvs/lib/dtni/base_adapter.py`, 226 lines) is the contract. Subclasses set class attributes and implement / override phase methods; the base provides multi-host launch fan-out, HTTP readiness polling, GPU probing, and teardown.

### Class attributes adapters set

| Attribute | Type | Purpose |
|---|---|---|
| `framework` | str | Registry key. Must match the `framework` field in every `config.json` for this adapter. |
| `required_roles` | tuple[str, ...] | Roles that MUST be bound. Default: `("server",)`. |
| `optional_roles` | tuple[str, ...] | Roles that MAY be bound. Default: `()`. |
| `poll_interval_s` | float | `await_completion` poll interval. Default: 5.0. |
| `completion_timeout_s` | float | `await_completion` deadline. Default: 3600.0. |
| `http_pool_interval_s` | float | `_wait_http_pool` poll interval. Default: 5.0. |
| `gpu_probe_cmd` | str | Override only if not using `rocm-smi`. |

### Inherited helpers (do not re-implement)

| Helper | Behavior |
|---|---|
| `self._launch_role(ctx, role, *, image, env, command, volumes, ...)` | Starts one `ContainerHandle` per host in `ctx.bindings[role]`. Registers each into `ctx.containers` and `self.handles_by_role[role]`. Supports `per_host_kwargs_fn` for host-specific tweaks. |
| `self._wait_http_pool(role, path, port, timeout_s)` | Polls every handle in `self.handles_by_role[role]` in parallel until all return HTTP 200. Raises `WorkloadError` listing unready container names on timeout. |
| `self.handles_by_role[role]` | List of handles for a role. Use in `parse` to find your container instead of scanning `ctx.containers` by string match. |
| `self._discover_gpu_count(ctx)` | rocm-smi probe across every distinct host in bindings. Stored in `ctx.scratch["host_gpus"]` by `prepare`. |
| `self._validate_role_contract(ctx)` | Enforces `required_roles` / `optional_roles`. Called by `prepare`. |

### Registration

```python
# cvs/lib/dtni/frameworks/registry.py
from cvs.lib.dtni.frameworks.vllm_single_adapter import VllmAdapter
from cvs.lib.dtni.base_adapter import BaseWorkloadAdapter

FRAMEWORK_REGISTRY: dict[str, type[BaseWorkloadAdapter]] = {
    "vllm_single": VllmAdapter,
}
```

Adapter modules live FLAT under `cvs/lib/dtni/frameworks/` — `vllm_single_adapter.py`, not `vllm_single/adapter.py`. Adding a new framework means: write `frameworks/<name>_adapter.py`, import its class in `frameworks/registry.py`, add the `"name": Class` entry.

### Minimal adapter skeleton

```python
from cvs.lib.dtni.base_adapter import BaseWorkloadAdapter
from cvs.lib.dtni.substitution import substitute


class MyFrameworkAdapter(BaseWorkloadAdapter):
    framework = "my_framework"
    required_roles = ("server",)

    def launch(self, ctx) -> None:
        role_spec = ctx.workload["roles"]["server"]
        sub_ctx = ctx.scratch["sub_ctx"]
        self._launch_role(
            ctx, "server",
            image=ctx.workload["image"]["tag"],
            command=substitute(role_spec["command"], sub_ctx),
            env={k: substitute(v, sub_ctx) for k, v in role_spec.get("env", {}).items()},
            volumes={substitute(k, sub_ctx): substitute(v, sub_ctx)
                     for k, v in role_spec.get("volumes", {}).items()},
            ports={str(role_spec["port"]): str(role_spec["port"])},
            network="host",
        )
        self._wait_http_pool("server", role_spec["health_path"],
                             role_spec["port"], timeout_s=1800)

    def parse(self, ctx) -> None:
        # MUST emit smoke_request_latency_ms and smoke_completion_tokens
        ctx.result.scalars["smoke_request_latency_ms"] = ...
        ctx.result.scalars["smoke_completion_tokens"] = ...
```

`prepare`, `await_completion`, and `teardown` are inherited; override only when the defaults do not fit.

Recipe: see `framework/RUNBOOK.md` for the full scaffold-a-framework walkthrough.

## Configs and thresholds — the data layer

Workload configs and thresholds live under `cvs/input/dtni/<framework>/<model>/<variant>/{config,threshold}.json`. The variant directory name is a hard convention: it MUST end in `_perf` or `_accuracy`. Unsuffixed directories are migration debt.

Configs are Pydantic v2 models defined in `cvs/lib/dtni/config_loader.py`. Validation happens at load time, before any cluster contact, so a malformed config fails fast with a `ConfigError` that includes the file path and pydantic's structured error list.

### Top-level config shape

| Field | Type | Notes |
|---|---|---|
| `schema_version` | Literal[1] | Locked to 1 in v1. |
| `framework` | str | Registry key. Must match the parent directory name (cross-checked by `execute_workload`). |
| `gpu_arch` | str | One of `VALID_ARCHES`. Must match the cluster's detected arch. |
| `paths` | PathsBlock | `shared_fs`, `models_dir`, `datasets_dir`, `artifacts_dir`, `images_dir`. |
| `model` | ModelBlock | `id` (catalog-validated), `remote` 0/1, `precision`, optional `hf_token_env`. |
| `dataset` | DatasetBlock \| None | Optional; catalog-validated. |
| `image` | ImageBlock \| None | Top-level OR per-role; one or the other must be present. |
| `topology` | TopologyBlock | `roles: {role: {count, gpus_per_node}}`. |
| `roles` | dict[str, RoleSpec] | Keys MUST match `topology.roles` exactly. Per-role: `command`, `env`, `volumes`, `devices`, `port`, `health_path`, `extra_args`, optional `image`. |
| `benchmarks` | list[str] | Benchmark ids; validated against `BENCHMARK_REGISTRY` with did-you-mean hints. |
| `params` | dict[str, Any] | Free-form; exposed to substitution as `{params.<key>}`. |

The `paths` block is boilerplate today — identical across every current variant. It is known tech debt; v2 will consolidate it. Until then, copy it intact when adding a new variant.

### Substitution tokens

Two phases of substitution happen. Path-block self-references (`{shared_fs}` inside `models_dir` etc.) are resolved at config-load time by `substitution.resolve_paths_block`. Everything else expands at job-runtime via `substitution.substitute(value, sub_ctx)`:

| Token | Source |
|---|---|
| `{shared_fs}` `{models_dir}` `{datasets_dir}` `{artifacts_dir}` `{images_dir}` | `paths` block, post-resolution |
| `{run_id}` | Job builder |
| `{user-id}` | Cluster file username |
| `{model.id}` `{model.path}` `{model.precision}` | Resolved by `resource_resolver` + Job builder |
| `{params.<key>}` | Workload `params` block |

Unknown tokens fail loud with `ValueError`. Substitution recurses (up to depth 8) so a token may resolve to a string containing more tokens.

### Real config snippet

```json
"roles": {
  "server": {
    "shm_size": "64G",
    "devices": ["/dev/dri", "/dev/kfd"],
    "port": 8000,
    "health_path": "/health",
    "command": "vllm serve {model.path} --tensor-parallel-size {params.tensor_parallelism} --served-model-name {model.id}"
  }
},
"benchmarks": ["serve_synth_short", "serve_synth_long"],
"params": {"tensor_parallelism": 1, "output_len": 256}
```

### Thresholds

`threshold.json` is a flat `{metric_name: spec}` dict. Five spec kinds, defined in `cvs/lib/dtni/verdict.py`:

| `kind` | Spec shape | Pass condition |
|---|---|---|
| `min` | `{"kind": "min", "value": V}` | `actual >= V` |
| `max_ms` | `{"kind": "max_ms", "value": V}` | `actual <= V` |
| `within` | `{"kind": "within", "lo": L, "hi": H}` | `L <= actual <= H` |
| `min_tok_s` | `{"kind": "min_tok_s", "value": V}` | `actual >= V` (alias for `min` with semantic intent) |
| `min_ratio` | `{"kind": "min_ratio", "value": V}` | `actual >= V` (ratio semantics) |

Missing scalar produces a FAIL with `note="metric not produced by adapter"`. Unknown `kind` is rejected at load.

### Real threshold snippet

```json
{
  "smoke_request_latency_ms": {"kind": "max_ms", "value": 600000},
  "smoke_completion_tokens":  {"kind": "min", "value": 1},
  "serve_synth_short.ttft_p95_ms": {"kind": "max_ms", "value": 60000},
  "serve_synth_short.request_throughput": {"kind": "min", "value": 0.5}
}
```

Recipe: see `config_and_thresholds.md` for the full schema reference and `add_workload.md` for the variant-authoring recipe.

## Benchmarks subsystem

Benchmarks are decoupled into four registries, each in its own file. The split is intentional: it lets the same harness (vllm-bench-serve) back many benchmarks (serve_synth_short, serve_synth_long, ...) while each benchmark gets its own thresholds, and it makes every stage individually testable.

| Registry | File | Type | Purpose |
|---|---|---|---|
| `BENCHMARK_REGISTRY` | `cvs/lib/dtni/benchmarks/registry.py` | `dict[str, BenchmarkSpec]` | Catalog of named benchmarks. Each `BenchmarkSpec` has `id`, `harness`, `dataset_id`, `score_metric`, `score_filter`, `shots`, `extra`. |
| `HARNESS_INVOKERS` | `cvs/lib/dtni/benchmarks/harness_invokers.py` | `dict[str, Callable[[BenchmarkSpec, HarnessCtx], str]]` | Builds the in-container shell command for the harness. Pure function. |
| `_RESULT_GLOBS` | `cvs/lib/dtni/benchmarks/runner.py` | `dict[str, str]` | Per-harness output-file glob (e.g. `results*.json` for lm-eval, `result.json` for vllm-bench-serve). |
| `PROJECTORS` | `cvs/lib/dtni/benchmarks/projectors.py` | `dict[str, Callable[[BenchmarkSpec, dict], dict[str, float]]]` | Maps raw harness JSON to flat `{scalar_key: float}`. Pure function. |

Two harnesses ship in v1:

- **`lm-eval-harness`** drives `lm_eval` from EleutherAI's lm-evaluation-harness against the live OpenAI-compatible server. Powers `mmlu` and `gsm8k` accuracy benchmarks. Scalars project to the bare benchmark id (`scalars["mmlu"]`) plus an optional `_stderr` companion.
- **`vllm-bench-serve`** drives `vllm bench serve` against the same server for perf metrics (TTFT/TPOT/ITL/E2EL percentiles, throughputs, goodput). Scalars project to a dotted namespace (`scalars["serve_synth_short.ttft_p95_ms"]`) so multiple invocations of the same harness do not collide on metric names.

### Why the split

Building a harness command is pure and easy to unit-test in isolation. Finding the output file involves the host filesystem and benefits from being one line of glob configuration. Projecting JSON to scalars is pure too but harness-specific. Mashing all three into one giant function would couple test surfaces that have no reason to be coupled. The four-registry split means adding a new harness is three additive entries; adding a benchmark to an existing harness is one `BenchmarkSpec` entry in `BENCHMARK_REGISTRY`.

### Runtime flow

The adapter's `parse` calls `cvs.lib.dtni.benchmarks.runner.run_benchmarks(benchmarks=[...], server_handle=..., base_url=..., model_id=..., model_path=..., output_dir_in_container=..., output_dir_on_host=...)`. For each benchmark id, the runner looks up the spec, builds the harness command via `HARNESS_INVOKERS[spec.harness]`, runs it via `docker exec` inside the server container (which has the host artifacts dir bind-mounted at `/cvs_artifacts`), `find`s the result file using `_RESULT_GLOBS[spec.harness]`, reads it via the same per-host executor, parses JSON, and merges `PROJECTORS[spec.harness](spec, payload)` into the return dict. The adapter then `ctx.result.scalars.update(...)`s the result.

Recipe: see `add_benchmark.md` (existing harness) or `add_harness.md` (new harness).

## Topology and multi-role

Single-role single-host is the default and covers most v1 workloads. Topology lives in `config.json`:

```json
"topology": {"roles": {"server": {"count": 1, "gpus_per_node": 1}}}                       // single
"topology": {"roles": {"server": {"count": 4, "gpus_per_node": 8}}}                       // distributed single-role
"topology": {"roles": {"prefill": {"count": 1, "gpus_per_node": 8},
                       "decode":  {"count": 2, "gpus_per_node": 8}}}                      // disaggregated
```

Roles bind to hosts via `topology.resolve_bindings`, which walks `cluster.node_dict` keys in order and slices `count` hosts per role in declaration order. The result is `ctx.bindings[role] -> [host, ...]`. `BaseWorkloadAdapter._launch_role` handles fan-out automatically — pass it a role name and it launches one container per host in that role's binding.

### Distributed single-role (TP/DP serving)

One `required_roles = ("server",)`, multiple hosts. Use one `_launch_role("server", ...)` call followed by one `_wait_http_pool("server", ...)`. Decide up-front whether `parse` should reduce (sum tokens, average latencies) or sample one leader; the convention is single-leader sampling for smoke metrics (one host's `_run_smoke`) and reduction for aggregate benchmark metrics. See the patterns doc for the exact reduction rule.

### Disaggregated prefill/decode

`required_roles = ("prefill", "decode")`. The role names are a hard convention — `topology.roles`, `threshold` key prefixes, and downstream tooling all hard-code these strings. Launch order is critical: start `decode` first (it owns the KV cache receivers), wait for its health pool, then start `prefill` (which connects to decode for KV transport), then wait for prefill's health pool. Scalar naming convention: role-prefixed for per-role metrics (`prefill.tok_s`, `decode.queue_depth`), unprefixed for end-to-end metrics (`e2e.request_latency_ms`).

Recipe: see `patterns/distributed.md` and `patterns/disagg.md`.

## Artifacts and smoke

Adapters do not write to the host filesystem directly. They capture text into `ctx.logs[f"{container_name}.{artifact_name}"]` (typically via `container_handle.capture()` during teardown). `cvs/lib/dtni/artifact_writer.py` is the single owner of the filesystem write: it concatenates `ctx.logs` into one `.log` file, dumps `ctx.result.scalars` as a one-row parquet, and serializes the verdict dict to `verdict.json`, all under `./cvs_artifacts/<run_id>/` with a `{arch}_{model_id}_{framework}_{hash}_{ts}` basename. `ContainerHandle.capture` itself probes `docker logs`, `dmesg -T | tail -500`, and `rocm-smi || amd-smi monitor` — best-effort, never raises.

### Mandatory smoke metrics

Every adapter MUST emit these two scalars in `parse`, even when no benchmark runs:

| Metric | Threshold kind | What it proves |
|---|---|---|
| `smoke_request_latency_ms` | `max_ms` | The server processed at least one real request in bounded time. |
| `smoke_completion_tokens` | `min` (typically `value: 1`) | The server produced at least one real output token. |

If either is missing, the verdict FAILs even when every other threshold passes — the verdict for the missing metric is `passed=False, note="metric not produced by adapter"`. This is the smoke guard. It catches the failure mode where the workload technically ran (no exception, container exited 0) but produced no actual inference output — silent corruption of the verdict signal. The vLLM adapter's `_run_smoke` is the reference implementation: one `POST /v1/completions` with a fixed prompt, captures `elapsed_s` from a bash-level timer, reads `completion_tokens` from the `usage` block of the response.

Recipe: see `framework/03_artifacts_and_smoke.md`.

## Verdict evaluation

`cvs/lib/dtni/verdict.py` is 64 lines. After `parse`, the Job calls `evaluate_all(ctx.thresholds, ctx.result.scalars)` and gets back a list of dicts, one per threshold:

```python
{
  "metric": "serve_synth_short.ttft_p95_ms",
  "actual": 84.1,
  "threshold": 60000.0,
  "kind": "max_ms",
  "passed": True,
  "note": None
}
```

`all_passed(verdicts)` is `all(v["passed"] for v in verdicts)`. If false, the Job raises `WorkloadError(phase="verify")` with a summary message; the run fails. The verdict list is serialized into `verdict.json` under the `"verdicts"` key alongside `run_id`, `workload`, `arch`, `framework`, `workload_hash`, `failed_phase`, and `message`.

A missing metric produces a verdict with `actual=None, passed=False, note="metric not produced by adapter"` — this is the smoke guard's mechanism. A threshold with an unknown `kind` is rejected at config-load time, not at verify time; `evaluate_all` only sees validated specs.

## Test entrypoints

DTNI has two flavors of pytest tests, in different locations, with different conftests.

### Suite tests (real workload runs)

Files live at `cvs/tests/dtni/<framework>.py` with NO `test_` prefix. The suite resolver looks them up by framework name. Each file defines exactly one test function, `test_threshold(metric, workload_outcome)`, which the shared `cvs/tests/dtni/conftest.py` parametrizes over every key in the workload's `threshold.json`. The module-scoped `workload_outcome` fixture calls `execute_workload` once per module and every per-metric node reads from the same result.

```bash
cvs run vllm_single \
    --cluster_file=cluster.json \
    --config_file=cvs/input/dtni/vllm_single/llama_3_1_8b/llama_3_1_8b_bf16_single_1_perf/config.json
```

### Unittests (no-cluster)

Files live at `cvs/lib/dtni/unittests/test_*.py`. They build a `RunContext` with `LocalExecutor` (or a mock), drive adapter logic through fake docker/curl invocations, and assert against scalars / verdicts / captured shell commands. The shadow `conftest.py` in this directory is intentionally empty: it shadows the repo-wide conftest so HTML-report dependencies do not need to be importable for these tests to run.

```bash
pytest cvs/lib/dtni/unittests/ -v
```

### Discovery CLIs

`cvs list <framework>` collection-only listing of available variants. `cvs run <suite> --collect-only` does the pytest-level collection check (useful for verifying a threshold file is well-formed without running the workload).

Recipe: see `framework/04_registry_and_test_entrypoint.md`.

## Where things live

```
cvs/lib/dtni/
  __init__.py
  adapter_protocol.py        # WorkloadAdapter Protocol + Progress enum
  arch_detect.py             # detect_arch_via + VALID_ARCHES
  artifact_writer.py         # write_artifacts: parquet + log + verdict.json
  base_adapter.py            # BaseWorkloadAdapter (the contract)
  catalog.py                 # model/dataset catalog + did-you-mean
  config_loader.py           # Pydantic models, load_cluster/workload/thresholds
  container_handle.py        # ContainerHandle (docker run wrapper, label teardown)
  errors.py                  # WorkloadError, ConfigError, CatalogError
  executor.py                # MultiHostExecutor, LocalExecutor, _PsshHostView
  hashing.py                 # workload_hash
  job.py                     # Job.run() pipeline driver + inline verify
  resource_resolver.py       # resolve_model_path, resolve_image_on_host, ...
  run_context.py             # RunContext + RunResult dataclasses
  runner.py                  # execute_workload (library entrypoint)
  substitution.py            # substitute + build_context + resolve_paths_block
  topology.py                # resolve_bindings (role -> hosts)
  verdict.py                 # Verdict + evaluate_all + all_passed
  frameworks/
    __init__.py
    registry.py              # FRAMEWORK_REGISTRY
    vllm_single_adapter.py   # VllmAdapter (reference adapter)
  benchmarks/
    __init__.py
    registry.py              # BENCHMARK_REGISTRY (BenchmarkSpec entries)
    harness_invokers.py      # HARNESS_INVOKERS + HarnessCtx + OUTPUT_DIR_IN_CONTAINER
    projectors.py            # PROJECTORS (pure JSON -> scalars)
    runner.py                # run_benchmarks + _RESULT_GLOBS
  unittests/
    __init__.py
    conftest.py              # empty shadow (insulates from repo conftest)
    test_*.py                # adapter / projector / registry / verdict unit tests
  docs/                      # this directory
cvs/input/dtni/
  <framework>/<model>/<variant>/{config,threshold}.json
cvs/tests/dtni/
  __init__.py
  conftest.py                # pytest_generate_tests + workload_outcome fixture
  vllm_single.py             # suite-level test (no test_ prefix)
```

## Extension points

| I want to... | Recipe doc |
|---|---|
| New workload variant (existing framework + model) | `add_workload.md` |
| New benchmark in an existing harness | `add_benchmark.md` |
| New harness (new eval tool) | `add_harness.md` |
| New framework from scratch | `framework/RUNBOOK.md` |
| Distributed (single-role, multi-host) pattern conventions | `patterns/distributed.md` |
| Disagg (prefill/decode) pattern conventions | `patterns/disagg.md` |
| End-user docs (rocm.docs.amd.com) | `sphinx_rst.md` |
| Full schema reference for configs and thresholds | `config_and_thresholds.md` |

## Design properties

What this architecture buys, in one paragraph. **Determinism**: identical `config.json` + `threshold.json` + cluster + image digest produce identical verdicts; the workload hash makes that auditable. **Framework-agnostic surface**: `cvs run <suite>` is the same command regardless of the serving stack; adding vllm-disagg or sglang adds one adapter and zero new CLI surface. **Unit-testable adapters** via the duck-typed `ctx.executor.executor_for(host).exec(cmd)` seam — every adapter is exercisable in CI with `LocalExecutor` and a mock runner, no cluster, no docker. **Decoupled metrics**: adapters and projectors produce scalars; thresholds compare them; the Job neither knows nor cares what the metrics mean, so new metrics require no Job changes. **Single registry per concern**: frameworks, benchmark specs, harness invokers, harness projectors, and result globs each have exactly one home, and extension is purely additive — a new framework or benchmark touches one file per concern, never the Job, never the verdict engine.
