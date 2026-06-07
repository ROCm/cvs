# Pattern: Disaggregated prefill/decode

Prefill/decode disaggregated serving: prefill workers handle the compute-bound
forward pass that fills the KV cache, decode workers handle the memory-bound
token generation, and a KV transport moves cache pages between them. This is
the multi-role analogue of `./distributed.md`.

## When this pattern applies

- vLLM, sglang, or any framework that splits prefill GPUs from decode GPUs
  and uses a KV transport (nixl, mooncake, RDMA, NVLink-Sharp) between them.
- NOT generic multi-role topologies (e.g. coordinator + workers) — those are
  bespoke and should not be jammed into the prefill/decode role names.

## Role declaration (HARD convention)

```python
from cvs.lib.dtni.base_adapter import BaseWorkloadAdapter

class FooDisaggAdapter(BaseWorkloadAdapter):
    framework = "foo_disagg"
    required_roles = ("prefill", "decode")
```

The role names MUST be exactly `"prefill"` and `"decode"`. `config.json`
`topology.roles` is keyed by these strings, threshold keys are prefixed with
these, and other tooling (dashboards, log scrapers) hard-code them.

## Topology binding shape

```text
ctx.bindings["prefill"] -> ["host-a"]
ctx.bindings["decode"]  -> ["host-b", "host-c", "host-d"]
```

Counts per role are independent; a `1:N` prefill:decode ratio is typical for
small-prompt high-throughput workloads. The base class fans `_launch_role`
out across each role's host list.

## Launch ordering (CRITICAL)

Decode workers MUST come up FIRST. They advertise the KV-receive endpoint;
prefill workers connect to that endpoint at startup. Launching prefill before
decode produces connection-refused failures that masquerade as flaky startup.

- Launch decode, then `_wait_http_pool("decode", ...)` until ready.
- Then launch prefill, then `_wait_http_pool("prefill", ...)` until ready.
- Put a code comment in `launch()` recording this ordering so a future reader
  does not "tidy up" the calls into parallel.

Anti-pattern:

```python
# WRONG — prefill cannot find KV receiver yet
self._launch_role(ctx, "prefill", ...)
self._launch_role(ctx, "decode", ...)
```

## KV transport naming convention

The adapter docstring MUST name the KV transport (and port, if non-default).
Oncall needs this when debugging connectivity — `ibstat`, `ss -lntp`, and
firewall checks all hinge on knowing which transport you picked.

```python
class SglangDisaggAdapter(BaseWorkloadAdapter):
    """sglang prefill/decode disagg.

    KV transport: mooncake over RDMA (port 7777 on decode hosts).
    """
```

Common values: `nixl`, `mooncake`, `rdma`, `nvlink-sharp`.

## Result key prefixing (HARD convention)

- Per-role scalars MUST be prefixed `prefill.<metric>` or `decode.<metric>`
  (e.g. `prefill.tok_s`, `decode.queue_depth`).
- End-to-end metrics observed from the client perspective use NO prefix.
- Mandatory smoke metrics — `smoke_request_latency_ms` and
  `smoke_completion_tokens` — are end-to-end, so unprefixed.
- Optionally also emit `prefill.smoke_latency_ms` / `decode.smoke_latency_ms`
  for diagnostic visibility, but the unprefixed e2e ones remain mandatory.

## Worked snippet

```python
def launch(self, ctx) -> None:
    # KV transport: nixl. Launch order: decode first (KV receivers),
    # then prefill (KV senders) — reversing causes connection-refused.
    self._launch_role(ctx, "decode", image=ctx.workload["roles"]["decode"]["image"]["tag"])
    self._wait_http_pool("decode", path="/health", port=8001, timeout_s=1800)
    self._launch_role(ctx, "prefill", image=ctx.workload["roles"]["prefill"]["image"]["tag"])
    self._wait_http_pool("prefill", path="/health", port=8000, timeout_s=1800)

def parse(self, ctx) -> None:
    # E2E smoke (unprefixed — observed at the client).
    client_result = self._read_client_result(ctx)
    ctx.result.scalars["smoke_request_latency_ms"] = client_result["e2e_ms"]
    ctx.result.scalars["smoke_completion_tokens"]  = client_result["tokens"]

    # Per-role scalars (prefixed).
    for role in ("prefill", "decode"):
        role_result = self._read_role_result(role, ctx)
        ctx.result.scalars[f"{role}.tok_s"] = role_result["tok_s"]
        ctx.result.scalars[f"{role}.queue_depth"] = role_result["queue_depth"]
```

## Threshold authoring

In `threshold.json`, role-prefixed metrics use role-prefixed threshold keys
(`prefill.tok_s`, `decode.queue_depth`). End-to-end metrics use unprefixed
threshold keys (`smoke_request_latency_ms`). Mixing the two — e.g. asserting
on `tok_s` when the adapter emits `prefill.tok_s` — silently fails to gate
anything. See `../config_and_thresholds.md` for the threshold schema.

## Per-role container images

Disagg deployments often pin different images per role: decode may need a
different transport library, a different vLLM build, or different env vars
than prefill. The config loader accepts `image` per role under
`workload.roles.<role>.image` (and `extra_args`, `env`, etc.); declare them
separately rather than smuggling role-specific flags into a shared image.
See `../config_and_thresholds.md`.

## Verification additions

Beyond the single-host baseline:

- Assert `len(ctx.bindings["prefill"]) >= 1` and
  `len(ctx.bindings["decode"]) >= 1` in `prepare` or early in `launch`.
- Add a launch-order assertion in unittests: decode handles populated before
  the prefill `_launch_role` call begins (mock `_launch_role` and record call
  order).
- Assert smoke metrics are emitted unprefixed and per-role metrics carry the
  `prefill.` / `decode.` prefix; a regex over `ctx.result.scalars.keys()`
  catches accidental drift.

## Pointer back

Use this pattern when filling in Step 4 of `../framework/RUNBOOK.md`.
