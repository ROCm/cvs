# 02 — Phase Invariants

Per-phase contract. What `ctx` looks like entering and exiting each phase,
what counts as success, what counts as failure. The Job
(`cvs/lib/dtni/job.py`) enforces this ordering and wraps every phase in
`_phase(name, fn)`.

See [`01_adapter_contract.md`](01_adapter_contract.md) for the class-level
contract and [`03_artifacts_and_smoke.md`](03_artifacts_and_smoke.md) for the
smoke-metric contract in `parse`.

## `ctx` state cheat sheet

`ctx` is a `RunContext` from `cvs/lib/dtni/run_context.py`. Fields an adapter
touches (all runner-populated unless noted):

| Field | Type | Adapter use |
|---|---|---|
| `run_id`, `arch`, `workload_name`, `workload_hash` | `str` | Read-only; label artifacts/containers. `arch` is GPU arch (`"mi300x"`, ...). |
| `workload` | `dict` | Resolved + substituted config (loader-set). Read in every phase. |
| `thresholds` | `dict` | DO NOT read; verify is inline in the Job. |
| `bindings` | `dict[str, list[str]]` | role -> [host], from topology. Read in `launch`. |
| `executor` | `MultiHostExecutor` / `LocalExecutor` | Always go through `.executor_for(host)` (see 01). |
| `artifacts_dir` | `pathlib.Path` | Per-run host dir. Bind-mount into containers as needed. |
| `containers` | `list[ContainerHandle]` | Append-only during `launch` via `_launch_role`. |
| `scratch` | `dict` | Free-form per-run bookkeeping. Base `prepare` writes `host_gpus`. |
| `logs` | `dict[str, str]` | Captured text artifacts (adapter / base `teardown`); drained by `artifact_writer`. |
| `result.scalars` | `dict[str, float]` | Adapter `parse` writes. The ONLY input to verify. |
| `events` | `_NullEvents` | Optional `events.emit(...)` sink; no-op by default. |

## `prepare(ctx)`

**Purpose.** Validate role bindings; probe per-host GPU count.

**Pre-state.** `ctx.bindings`, `ctx.executor`, `ctx.workload` populated.
`ctx.containers == []`, `ctx.result.scalars == {}`.

**Post-state.** No containers started. `ctx.scratch["host_gpus"]` populated
by base impl: `{host: int}`. Adapter-specific staging (rendered configs to
artifacts dir, pre-pulling images) done here.

**Failure semantics.** Missing required role, role bound to empty host list,
unknown declared role, or rocm-smi probe returning 0 GPUs all raise
`WorkloadError` from base. Job tags `failed_phase="prepare"`. `launch` is
skipped; `teardown` still runs (drains nothing since no containers).

**Common pitfalls.**

- Calling `subprocess.run` directly instead of `ctx.executor` — breaks unittests.
- Per-rank GPU probes — base already loops once per unique host in `ctx.bindings`.
- Overriding without `super().prepare(ctx)` first — drops role validation and host_gpus probe.

## `launch(ctx)`

**Purpose.** Start every container needed for the run; return only when the
workload is ready to receive its first request.

**Pre-state.** `prepare` succeeded. `ctx.scratch["host_gpus"]` available.
`ctx.containers == []`.

**Post-state.** One handle per host per role in `ctx.containers` and in
`self.handles_by_role[role]`. For HTTP-serving frameworks, every handle has
passed `_wait_http_pool` (server returning HTTP 200 on its health path).

**Failure semantics.** Image pull failure, container start failure, or
`_wait_http_pool` timeout raise `WorkloadError`. Handles that DID enter the
context manager are already in `ctx.containers`; `teardown` captures their
logs and removes them.

**Common pitfalls.**

- Forgetting readiness wait — `await_completion` / `parse` race the server, producing flaky errors.
- Hardcoding `docker run ...` instead of `_launch_role` — bypasses the `cvs_run_id` label that scopes teardown cleanup.
- Building role command from wrong substitution context — use pre-substituted `ctx.workload` (loader runs substitution before `Job.run`).

## `await_completion(ctx)`

**Purpose.** Block until the workload's notion of "done" is reached.

**Pre-state.** `launch` returned; container(s) running.

**Post-state.** Workload reports done (or `parse` will drive any remaining
work, for server-style frameworks).

**Default behaviour.** Base loops every `poll_interval_s` calling
`progress_predicate(ctx)`, which returns `'done'` (return cleanly),
`'running'` (keep polling), or `'broken'` (raise `WorkloadError`).
`completion_timeout_s` is a hard deadline; exceeding it raises `WorkloadError`.

**Server-style frameworks** (vllm) keep the inherited default
`progress_predicate -> 'done'`: the loop exits immediately, and `parse`
itself drives requests against the live server.

**Failure semantics.** Timeout, predicate returning `'broken'`, or an
exception during the predicate all raise `WorkloadError(phase="await")`.

**Common pitfalls.**

- Overriding `await_completion` to `time.sleep` directly instead of setting `progress_predicate` + `poll_interval_s` — loses the timeout contract.
- Tailing log files on the host — go through the container handle / executor so unittests still work.

## `parse(ctx)`

**Purpose.** Populate `ctx.result.scalars` with every metric the
`threshold.json` for any variant of this framework can name.

**Pre-state.** Server-style: container(s) live, await returned `'done'`
immediately. Batch-style: workload exited; result files in
`ctx.artifacts_dir` or readable via a container handle.

**Post-state.** `ctx.result.scalars` contains:

1. The two mandatory smoke metrics (see [03](03_artifacts_and_smoke.md)).
2. Every accuracy/perf scalar referenced by any threshold that could be
   evaluated against this run.

Missing a scalar that a threshold names does NOT raise — verify records a
verdict with `note="metric not produced by adapter"` and `passed=False`.
That is still a FAIL.

**Failure semantics.** Anything that prevents producing the smoke metrics
should raise `WorkloadError`. Per-benchmark partial failures should still
emit whatever scalars succeeded.

**Common pitfalls.**

- Returning early before emitting smoke metrics — smoke threshold then fails with a confusing `metric not produced` note.
- Calling the server from the host (firewall) instead of via the container handle when `network=host` is unavailable.
- Mixing string and float in `scalars` — keep values numeric; verdict evaluator does `float(actual)`.

## `verify` (inline, not an adapter method)

**Purpose.** Compare `ctx.result.scalars` against `ctx.thresholds`.

**Where.** `Job.run` in `cvs/lib/dtni/job.py`:

```python
verdicts = evaluate_all(ctx.thresholds, ctx.result.scalars)
if not all_passed(verdicts):
    raise WorkloadError(
        f"{sum(1 for v in verdicts if not v['passed'])}/{len(verdicts)} thresholds failed",
        phase="verify",
    )
```

Adapters cannot override this. Threshold kinds and behaviour are documented
in the sibling `config_and_thresholds.md`.

## `teardown(ctx)`

**Purpose.** Capture forensics; remove containers.

**Pre-state.** Any of: `prepare` failed, `launch` failed, `await` failed,
`parse` failed, verify failed, or everything passed.

**Post-state.** Every handle in `ctx.containers` has been `capture()`-ed —
its outputs drained into `ctx.logs[f"{handle.name}.{artifact}"]` — and then
removed. All containers labeled with this run's `cvs_run_id` are gone.

**Failure semantics.** Teardown ALWAYS runs (it is in the Job's `finally`).
A teardown exception does NOT overwrite an earlier failure: the Job only
records `failed_phase="teardown"` if no prior failure was tagged. This is
load-bearing — earlier phase failures must remain the diagnosed cause.

**Common pitfalls.**

- Raising inside teardown when the real failure was earlier — base impl swallows per-handle errors for exactly this reason.
- Skipping teardown if `launch` failed — DO NOT. Containers that DID start before the failure are in `ctx.containers` and must be reaped.

## Phase ordering invariants

From `cvs/lib/dtni/job.py`:

1. Phases run strictly in order: `prepare` → `launch` → `await` → `parse` → `verify` (inline).
2. **The first failing phase aborts the chain.** If `await` raises, `parse`
   does NOT run — partial results are lost on purpose (partial accuracy
   data was deemed worse than no data). If you need partial metrics, `parse`
   must be runnable from `await_completion` itself, or each chunk must be
   committed to `ctx.result.scalars` inline.
3. `teardown` ALWAYS runs (in `Job.run`'s `finally`), regardless of which earlier phase failed.
4. A teardown failure only becomes the reported `failed_phase` if no earlier phase already failed.
5. The verdict dict written to disk always carries
   `{run_id, workload, arch, framework, workload_hash, failed_phase, message, verdicts}` —
   even on early failure (`verdicts == []` in that case).
