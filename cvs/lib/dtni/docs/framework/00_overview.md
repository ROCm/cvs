# Framework Authoring — Overview

Adding a new framework end-to-end is the biggest single task in DTNI: it
touches an adapter class, the framework registry, a pytest entrypoint shim, a
first config/threshold pair, and an end-to-end smoke run. The entrypoint for
this work is [`RUNBOOK.md`](RUNBOOK.md) in this directory. RUNBOOK walks
six gated steps and points back at the per-topic docs below for the schema and
contract details — read RUNBOOK first; only come back here if RUNBOOK sends
you.

## What each doc in this directory covers

- [`01_adapter_contract.md`](01_adapter_contract.md) — the
  `BaseWorkloadAdapter` surface area: required attributes, the five phase
  methods you override, inherited helpers (`_launch_role`, `_wait_http_pool`),
  the executor seam, and a minimal adapter skeleton.
- [`02_phases.md`](02_phases.md) — per-phase invariants. Pre-state /
  post-state of `ctx` for each of `prepare`, `launch`, `await_completion`,
  `parse`, `teardown`, plus the inline `verify` the Job runs between `parse`
  and `teardown`.
- [`03_artifacts_and_smoke.md`](03_artifacts_and_smoke.md) —
  `ContainerHandle.capture()`, the `ctx.logs` drain protocol, and the two
  mandatory smoke metrics every adapter must emit
  (`smoke_request_latency_ms`, `smoke_completion_tokens`).
- [`04_registry_and_test_entrypoint.md`](04_registry_and_test_entrypoint.md) —
  how an adapter becomes discoverable: `FRAMEWORK_REGISTRY`, the pytest shim
  under `cvs/tests/dtni/`, the unittest `conftest.py` executor switch, and
  collection-time verification commands.
- [`05_seed_config_and_verification.md`](05_seed_config_and_verification.md) —
  the first `<model>_<variant>_perf/{config,threshold}.json` pair and the
  end-to-end command sequence to prove the new framework actually runs.

For the schema-of-record on config/threshold fields see the sibling
`config_and_thresholds.md` (one level up).
