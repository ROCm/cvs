# Pattern: Distributed (multi-host, single-role)

Multi-host topologies where ONE role is replicated across N hosts: tensor-parallel
inference fronted by a load balancer, replicated stateless servers, data-parallel
sharded workers that each expose the same API. The multi-host machinery already
lives on `BaseWorkloadAdapter`; this doc is conventions, not new code.

## When this pattern applies

- Single role replicated across N hosts (TP serving, DP serving, replicated HTTP).
- NOT prefill/decode disaggregation — that is a multi-role topology; see
  `./disagg.md`.
- NOT independent shard processing with different roles per shard — just declare
  multiple roles and call `_launch_role` for each.

## Declaration

```python
from cvs.lib.dtni.base_adapter import BaseWorkloadAdapter

class FooDistributedAdapter(BaseWorkloadAdapter):
    framework = "foo_distributed"
    required_roles = ("server",)  # one role; multi-host comes from topology
```

Topology hosts arrive via `ctx.bindings["server"]` (list of host strings). A
single `_launch_role(ctx, "server", ...)` call launches one container per host
in that binding — fan-out is handled by the base class.

## Inherited helpers (DO NOT re-implement)

| Helper | Behavior |
|---|---|
| `self._launch_role(ctx, role, *, image, env, command, ...)` | Starts one container per host in `ctx.bindings[role]`, registers each handle into `ctx.containers` and `self.handles_by_role[role]`. |
| `self._wait_http_pool(role, path, port, timeout_s)` | Polls every handle for `role` in parallel until all return HTTP 200; raises `WorkloadError` listing unready container names on timeout. |
| `self.handles_by_role["server"]` | List of `ContainerHandle` populated by `_launch_role`. Use in `parse` / `teardown` instead of scanning `ctx.containers` by name. |

Reach for `_launch_role` + `_wait_http_pool` first. Only drop down to a custom
loop if you need per-host launch ordering (see `./disagg.md`).

## Reduction vs single-leader sampling

The key decision in a distributed adapter is how `parse()` collapses per-host
observations into the `ctx.result.scalars` dict.

- **Single-leader sampling** — pick ONE host (typically
  `ctx.bindings[role][0]`) and read the metric only from there. Correct for
  client-side latency where you want one observer's view; cheap and
  deterministic. Risk: hides skew between hosts.
- **Reduction** — collect a scalar from EACH host and aggregate
  (sum / mean / max / min) into one key. Correct for throughput-like metrics
  where the cluster-wide number is the answer. Risk: hides per-host variance;
  if one host dies, mean lies.

Mandatory smoke metrics (`smoke_request_latency_ms`, `smoke_completion_tokens`)
are ALWAYS single-leader — smoke is a single client observation. Aggregate
throughput, GPU utilization summaries, etc. use reduction.

## Worked snippet

```python
def parse(self, ctx) -> None:
    # Smoke: single-leader observation against host 0.
    leader = ctx.bindings["server"][0]
    leader_result = self._read_result_file(leader, ctx)
    ctx.result.scalars["smoke_request_latency_ms"] = leader_result["latency_ms"]
    ctx.result.scalars["smoke_completion_tokens"]  = leader_result["tokens"]

    # Reduction: aggregate per-host throughput into a cluster scalar.
    per_host = [self._read_result_file(h, ctx) for h in ctx.bindings["server"]]
    ctx.result.scalars["aggregate_tok_s"] = sum(r["tok_s"] for r in per_host)
    ctx.result.scalars["aggregate_requests"] = sum(r["requests"] for r in per_host)
```

`_read_result_file` is adapter-specific (read a JSON the container wrote, parse a
log line, etc.) — the pattern is host-indexed lookups, then collapse.

## Naming conventions

- Aggregate keys: `aggregate_<metric>` (e.g. `aggregate_tok_s`,
  `aggregate_requests`).
- Per-host keys, if you emit them: `host_<index>_<metric>` where `index` is the
  position in `ctx.bindings[role]` (e.g. `host_0_tok_s`). Avoid hostnames in
  keys — they vary across topologies and break threshold reuse.
- Smoke metrics are unprefixed and single-leader; do not emit per-host smoke
  metrics.

## Verification additions for distributed adapters

When the adapter ships, the verification checklist gets three extra items
beyond the single-host baseline:

- After `launch`, assert
  `len(self.handles_by_role["server"]) == len(ctx.bindings["server"])` — the
  fan-out actually opened the right number of containers.
- `_wait_http_pool` must return only after every host is ready; an early
  return means the pool was empty (likely you forgot to call `_launch_role`
  for that role).
- Unittest the reduction logic with synthetic per-host result dicts — pass in
  fixtures with deliberately skewed values and assert the aggregated scalar.

## Pointer back

Use this pattern when filling in Step 4 of `../framework/RUNBOOK.md`.
