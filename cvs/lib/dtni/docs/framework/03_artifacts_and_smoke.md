# 03 — Artifacts and Smoke Metrics

Two contracts adapters must honor:

1. Drain container forensics into `ctx.logs` so the artifact writer can
   persist them on teardown.
2. Emit two mandatory smoke metrics in `parse` so every variant of the
   framework has a baseline liveness signal before any benchmark-specific
   threshold is evaluated.

See [`02_phases.md`](02_phases.md) for the broader phase contract.

## `ContainerHandle.capture()`

Defined in `cvs/lib/dtni/container_handle.py`. Best-effort snapshot of one
container's forensics. Never raises — failures are captured into the returned
dict as `"<capture failed: ...>"` strings.

Returns `Dict[str, str]` with these keys:

| Key | Source command | Purpose |
|---|---|---|
| `container.log` | `docker logs <name> 2>&1` | Full container stdout+stderr. |
| `dmesg.txt` | `dmesg -T | tail -n 500` | Recent host kernel ring buffer (XID-style errors, OOMs). |
| `gpu_state.txt` | `rocm-smi || amd-smi monitor` | GPU state right before container removal. |

You normally do not call `capture()` yourself — the inherited
`BaseWorkloadAdapter.teardown` does it for every handle in `ctx.containers`:

```python
# base_adapter.py
for handle in ctx.containers:
    try:
        artifacts = handle.capture()
        for name, text in artifacts.items():
            ctx.logs[f"{handle.name}.{name}"] = text
    except Exception:
        pass
    finally:
        handle.remove()
```

Call `capture()` manually only when you need forensics mid-run (e.g. after a
known-bad probe response, before a recovery attempt). Append the result into
`ctx.logs` the same way the base teardown does, so the artifact writer picks
it up.

## `ctx.logs` — the drain protocol

`ctx.logs: dict[str, str]` is the per-run keyed dict. Conventions:

- **Key shape**: `"{handle.name}.{artifact}"` for container-derived
  artifacts (base teardown does this automatically). For adapter-emitted
  blobs, use a short lowercase dotted key —
  `"smoke_request_raw.txt"`, `"server.health_probe.txt"`.
- **Role prefix when multi-role**: `"prefill.0.warmup_log.txt"` —
  `<role>.<index>.<artifact>` keeps multi-host runs disambiguatable.
- **Value type**: always `str`. If you must capture bytes, decode
  (`errors="replace"`) before assigning. Binary artifacts are out of scope
  for `ctx.logs`; bind-mount the host artifacts dir into the container and
  let the workload write them there directly (see vllm adapter).
- **No overwrites mid-phase**: if the same probe is run twice, suffix the
  key (`"server.health_probe.t1.txt"`); the artifact writer treats `ctx.logs`
  as a flat namespace and the last assignment wins.

The artifact writer drains `ctx.logs` to disk after `teardown` returns,
inside the per-run artifacts directory. Adapters never write to that
directory directly except for binary outputs the container itself produces
through a bind mount.

## Mandatory smoke metrics

Every adapter's `parse` MUST populate both of these in `ctx.result.scalars`:

| Metric | Type | Definition |
|---|---|---|
| `smoke_request_latency_ms` | `float` | End-to-end latency in milliseconds of one representative request through the workload's user-facing entrypoint (for serving frameworks: one completion request; for batch frameworks: one input through the inference path). |
| `smoke_completion_tokens` | `float` | Token count of the response to that representative request, as `float`. |

Every seed `threshold.json` ships a matching smoke threshold pair (see
[`05_seed_config_and_verification.md`](05_seed_config_and_verification.md)):

```json
"smoke_request_latency_ms": { "kind": "max_ms", "value": 600000 },
"smoke_completion_tokens":  { "kind": "min",    "value": 1 }
```

If either scalar is missing, the verdict for that threshold records
`passed: false, note: "metric not produced by adapter"` and the run is a
FAIL — the smoke guard is what catches "adapter launched but produced no
output". DO NOT skip emitting them, even on partial-failure paths inside
`parse`.

Adapters MAY emit additional smoke-prefixed metrics (vllm emits
`smoke_throughput_tok_s`); those are optional and only matter if a
`threshold.json` names them.

## Naming conventions for benchmark scalars

Beyond smoke, scalars come from benchmark projectors. Convention is
`<benchmark_id>.<metric>` (e.g. `serve_synth_short.request_throughput`,
`serve_synth_short.ttft_p95_ms`). Top-level accuracy task scores are bare
(`mmlu`, `gsm8k`). Threshold files reference the exact same key.

Adapters typically receive these scalars wholesale from
`cvs.lib.dtni.benchmarks.runner.run_benchmarks(...)` and merge them in:

```python
ctx.result.scalars.update(scalars)
```

— so the naming responsibility is on the benchmark projector, not the
adapter, for non-smoke metrics.

## Worked snippet — emit smoke metrics and capture raw output

Distilled from `cvs/lib/dtni/frameworks/vllm_single_adapter.py`:

```python
def parse(self, ctx) -> None:
    role_spec = ctx.workload["roles"]["server"]
    port = role_spec["port"]
    host = ctx.bindings["server"][0]
    executor = (
        ctx.executor.executor_for(host)
        if hasattr(ctx.executor, "executor_for") else ctx.executor
    )

    body = json.dumps({
        "model": ctx.workload["model"]["id"],
        "prompt": "Once upon a time",
        "max_tokens": 32,
        "temperature": 0.0,
    })
    cmd = (
        f"t0=$(date +%s.%N); "
        f"curl -s -X POST http://localhost:{port}/v1/completions "
        f"-H 'Content-Type: application/json' "
        f"-d {shlex.quote(body)}; "
        f"echo; t1=$(date +%s.%N); "
        f"echo \"ELAPSED=$(echo $t1 - $t0 | bc)\""
    )
    out = executor.exec(cmd, timeout=300)

    elapsed_s = _parse_elapsed(out)       # raises WorkloadError on miss
    n_tokens = _parse_completion_tokens(out)  # 0 on miss

    ctx.result.scalars["smoke_request_latency_ms"] = elapsed_s * 1000.0
    ctx.result.scalars["smoke_completion_tokens"] = float(n_tokens)
    ctx.logs["smoke_request_raw.txt"] = out
```

Three things to notice:

1. The probe goes through the executor seam, not direct `subprocess`.
2. The raw response is dropped into `ctx.logs` under a stable key so the
   artifact writer persists it next to the verdict.
3. Both mandatory smoke metrics are emitted unconditionally before any
   benchmark-driven scalars are merged in.
