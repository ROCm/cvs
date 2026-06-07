# 01 — Adapter Contract

This is the surface area you implement when adding a new framework. See
[`02_phases.md`](02_phases.md) for the per-phase pre/post-state, and
[`03_artifacts_and_smoke.md`](03_artifacts_and_smoke.md) for what `parse` must
emit.

## Class location

`cvs/lib/dtni/base_adapter.py` — `class BaseWorkloadAdapter(abc.ABC)`.

Subclass it under `cvs/lib/dtni/frameworks/<framework>_adapter.py` (flat
layout — single file per framework, no subdir). The reference is
`cvs/lib/dtni/frameworks/vllm_single_adapter.py`.

## Required class attributes

| Attr | Type | Meaning |
|---|---|---|
| `framework` | `str` (class var) | Registry key. MUST equal the key in `FRAMEWORK_REGISTRY` and the `framework` field in every config that targets this adapter. |
| `required_roles` | `tuple[str, ...]` | Roles the workload MUST declare. Single-host serving: `("server",)`. Disaggregated prefill/decode: `("prefill", "decode")`. |
| `optional_roles` | `tuple[str, ...]` | Defaults to `()`. Roles the workload MAY declare. Anything declared outside `required ∪ optional` raises `WorkloadError` in `prepare`. |

Optional tunables (inherited defaults usually fine):

| Attr | Default | Purpose |
|---|---|---|
| `poll_interval_s` | `5.0` | Loop sleep in default `await_completion`. |
| `completion_timeout_s` | `3600.0` | Hard timeout in default `await_completion`. |
| `launch_timeout_s` | adapter-defined | Used by `_wait_http_pool` when you call it from `launch`. |
| `gpu_probe_cmd` | `"rocm-smi --showproductname 2>/dev/null"` | Used by base `prepare` to count GPUs per host. |

## Phase methods

Each method has signature `(self, ctx: RunContext) -> None`. The Job calls
them in this exact order:

| Method | Required? | Notes |
|---|---|---|
| `prepare(ctx)` | Inherited — usually OK | Base impl validates `required_roles` vs `ctx.bindings`, then probes GPU count per host into `ctx.scratch["host_gpus"]`. Override only if you need pre-launch staging. |
| `launch(ctx)` | **Abstract — must override** | Start the container(s) for every required role. Should return only when each role is ready to receive traffic (use `_wait_http_pool` for HTTP servers). |
| `await_completion(ctx)` | Inherited — override only if not server-style | Default loops on `progress_predicate(ctx)` returning `'done'` / `'running'` / `'broken'`. Server-style frameworks (vllm) leave `progress_predicate` default (returns `'done'`); the lifetime is driven by `parse`. |
| `parse(ctx)` | **Abstract — must override** | Read whatever the run produced and populate `ctx.result.scalars`. MUST emit the two mandatory smoke metrics — see [`03_artifacts_and_smoke.md`](03_artifacts_and_smoke.md). |
| `teardown(ctx)` | Inherited — usually OK | Base impl iterates `ctx.containers`, calls `handle.capture()`, drains artifacts into `ctx.logs[f"{handle.name}.{artifact}"]`, then `handle.remove()`. Override only if you need extra cleanup. |

There is **no** `verify` method. The Job runs verify inline between `parse`
and `teardown`:

```python
verdicts = evaluate_all(ctx.thresholds, ctx.result.scalars)
```

Adapters never see thresholds; they only emit scalars. See `cvs/lib/dtni/job.py`.

## Inherited helpers — DO NOT re-implement

The multi-host launch and HTTP readiness machinery already lives on the base
class. Re-implementing it (calling `docker run` directly, using `time.sleep`
loops, hardcoding ssh) breaks the executor seam and the run_id cleanup
contract.

| Helper | Purpose |
|---|---|
| `self._launch_role(ctx, role, *, image, env=None, command=None, per_host_kwargs_fn=None, **spec_kwargs)` | Starts one container per host bound to `role`. Each handle is `__enter__`-ed, registered into `ctx.containers`, and appended to `self.handles_by_role[role]`. Returns the list of handles. `spec_kwargs` flow into `ContainerHandle` (volumes, devices, ports, network, ipc, shm_size, extra_args, ...). |
| `self._wait_http_pool(role, path, port, timeout_s)` | Polls every handle for `role` in parallel until each returns HTTP 200 from `http://localhost:{port}{path}`. Raises `WorkloadError` on timeout with the list of unready container names. |
| `self.handles_by_role: Dict[str, List[ContainerHandle]]` | Populated by `_launch_role`. Read it in `parse` to find a container by role; do not scan `ctx.containers` by name unless you have to. |
| `self._register(role, handle, ctx)` | Used by `_launch_role`; only call directly if you bypass `_launch_role` (rare). |

## Executor seam — CRITICAL

**Always** route shell-out through the per-host executor:

```python
per_host_exec = getattr(ctx.executor, "executor_for", None)
runner = per_host_exec(host) if per_host_exec else ctx.executor
out = runner.exec(cmd, timeout=timeout)
```

`ctx.executor` is one of:

- `MultiHostExecutor` (real runs) — `.executor_for(host)` returns a per-host
  view backed by pssh.
- `LocalExecutor` (unittests, single-node CVS) — no `.executor_for`; falls
  through to direct `.exec`.

Hardcoding `subprocess.run(["ssh", host, ...])` bypasses this and breaks
unittests, which inject `LocalExecutor`. The duck-typed seam is the contract.

## Error handling

Raise `cvs.lib.dtni.errors.WorkloadError(message)` from any phase to fail the
run. The Job tags the exception with the phase name (`prepare` / `launch` /
`await` / `parse` / `verify` / `teardown`) at the catch site if you do not set
`phase=` yourself, then writes a verdict dict with `failed_phase` and the
message. Teardown still runs.

```python
from cvs.lib.dtni.errors import WorkloadError
raise WorkloadError(f"{self.framework}: model returned empty completion")
```

Non-`WorkloadError` exceptions are wrapped: `WorkloadError(f"{name} failed: {exc}", phase=name)`.

## Minimal adapter skeleton

```python
# cvs/lib/dtni/frameworks/foo_single_adapter.py
from cvs.lib.dtni.base_adapter import BaseWorkloadAdapter
from cvs.lib.dtni.errors import WorkloadError


class FooAdapter(BaseWorkloadAdapter):
    framework = "foo_single"
    required_roles = ("server",)
    launch_timeout_s = 600.0

    def launch(self, ctx):
        role_spec = ctx.workload["roles"]["server"]
        self._launch_role(
            ctx, "server",
            image=ctx.workload["image"]["tag"],
            command=role_spec["command"],
            ports={str(role_spec["port"]): str(role_spec["port"])},
            network="host",
        )
        self._wait_http_pool(
            "server",
            path=role_spec.get("health_path", "/health"),
            port=role_spec["port"],
            timeout_s=self.launch_timeout_s,
        )

    def parse(self, ctx):
        # ... drive one request, measure latency, count tokens ...
        ctx.result.scalars["smoke_request_latency_ms"] = elapsed_ms
        ctx.result.scalars["smoke_completion_tokens"] = float(n_tokens)

    # prepare / await_completion / teardown inherited.
```

Once the class exists, register it (see
[`04_registry_and_test_entrypoint.md`](04_registry_and_test_entrypoint.md))
and seed a first config (see
[`05_seed_config_and_verification.md`](05_seed_config_and_verification.md)).
RUNBOOK sequences all of this.
