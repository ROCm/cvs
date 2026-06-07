# Framework Scaffolding RUNBOOK

This is the single entrypoint to scaffold a new framework. Walk the 6 gated
steps in order. Each step has a definition-of-done you must satisfy before
moving on. Consult `01_adapter_contract.md` .. `05_seed_config_and_verification.md`
ONLY when a step points there. Stop and ask the developer if a stop-trigger fires.

## Inputs

Before starting, the developer (or invoking prompt) MUST commit to:

- `framework_name`: snake_case, e.g. `pytorch_single`, `sglang_disagg`
- `topology_kind`: `single_role` | `distributed_single_role` | `disagg`
- `serving_protocol`: `http` | `rpc` | `cli-only`
- `container_image`: full registry path OR `TBD` (must be set before Step 4)
- `first_model` + `first_variant`: e.g. `llama_3_1_8b` + `bf16_single_1_perf`

If any of these are TBD and cannot yet be resolved, STOP and ask.

## Pre-flight checks

Run these first; if any fail, STOP.

```bash
# 1. Worktree is clean
git status
# 2. Sibling adapters exist for reference (at minimum vllm_single_adapter.py)
ls cvs/lib/dtni/frameworks/
# 3. Baseline unittests green
pytest cvs/lib/dtni/unittests/ -x
# 4. The framework_name is NOT already registered
grep -r "<framework_name>" cvs/lib/dtni/frameworks/registry.py
```

## Step 1 — Adapter skeleton

- Files touched: `cvs/lib/dtni/frameworks/<framework_name>_adapter.py` (new)
- Subclass `BaseWorkloadAdapter`; set class attrs `framework` and `required_roles`.
- Implement all 5 phase methods (`prepare`, `launch`, `await_completion`, `parse`,
  `teardown`) as `pass` stubs. Do NOT add a `verify` method — `verify` is
  inline in `job.py` and adapters cannot override it.
- See `01_adapter_contract.md` for the full method signatures. If multi-host,
  also see `../patterns/distributed.md`. If disagg, also see `../patterns/disagg.md`.
- DoD: `python -c "from cvs.lib.dtni.frameworks.<name>_adapter import <Class>; print(<Class>.framework)"`
  prints the framework name.
- Stop-and-ask if: `required_roles` does not match the topology decided in
  Inputs, or you find yourself wanting a 6th phase method.

## Step 2 — Registry wiring

- Files touched: `cvs/lib/dtni/frameworks/registry.py`
- Import the new adapter class and add `FRAMEWORK_REGISTRY["<name>"] = <Class>`.
- See `04_registry_and_test_entrypoint.md`.
- DoD: `cvs list | grep <framework_name>` shows the entry.
- Stop-and-ask if: import-time error appears (almost always a circular import —
  do not paper over it by deferring the import).

## Step 3 — Pytest entrypoint shim

- Files touched: investigate where existing entrypoints live (likely
  `cvs/tests/dtni/<framework>.py` — confirm by listing that directory
  before writing) and create the corresponding shim for the new framework.
- See `04_registry_and_test_entrypoint.md` for the shim shape.
- DoD: `cvs run <suite> --collect-only` collects the test. It is expected to
  FAIL at run-time at this point — collection success is what matters.
- Stop-and-ask if: collection error (not a runtime error). A collect-only error
  means wiring is wrong; do not move to Step 4.

## Step 4 — Implement prepare + launch + teardown

- Files touched: `cvs/lib/dtni/frameworks/<framework_name>_adapter.py`
- `prepare`: pull image, render configs into `{artifacts_dir}/{run_id}/`.
- `launch`: call `self._launch_role(ctx, role, ...)` for each role. For
  multi-role ordering, follow `../patterns/disagg.md` or
  `../patterns/distributed.md`. Use `self._wait_http_pool(...)` for readiness
  on HTTP servers. Populate `self.handles_by_role`.
- `teardown`: iterate `self.handles_by_role` and call `handle.stop()` on each.
  Must be idempotent (safe to call twice, safe to call after partial launch).
- See `02_phases.md` for pre/post-state contracts and
  `03_artifacts_and_smoke.md` for the `container_handle.capture()` contract.
- DoD: launch a real run on a dev host; the container starts and teardown
  cleans it up. If no cluster access right now, skip to Step 5 and come back —
  do NOT mark Step 4 done until you have run it for real.
- Stop-and-ask if: you find yourself reaching for hardcoded `ssh` anywhere.
  All remote commands must go through `ctx.executor.executor_for(host).exec()`.

## Step 5 — Implement await + parse (mandatory smoke metrics)

- Files touched: `cvs/lib/dtni/frameworks/<framework_name>_adapter.py`
- `await_completion`: poll until the job is done. Every command must go through
  `ctx.executor` — no direct subprocess, no direct ssh.
- `parse`: read result files and populate `ctx.result.scalars`. You MUST
  emit `smoke_request_latency_ms` and `smoke_completion_tokens`, or the
  smoke guard FAILs and the suite cannot pass.
- See `03_artifacts_and_smoke.md`.
- DoD: smoke run succeeds and both smoke scalars are present in `JobResult`.
- Stop-and-ask if: parse logic performs I/O outside `ctx.executor` (this
  breaks unittests, which use the shadow conftest at
  `cvs/lib/dtni/unittests/conftest.py`).

## Step 6 — Seed config + threshold + final verification

- Files touched:
  - `cvs/input/dtni/<framework_name>/<model>/<model>_<variant>/config.json`
  - `cvs/input/dtni/<framework_name>/<model>/<model>_<variant>/threshold.json`
- Use a `_perf` variant for the seed (e.g. `bf16_single_1_perf`). Smoke
  thresholds only — no perf gates in the seed config.
- See `05_seed_config_and_verification.md` for templates and
  `../config_and_thresholds.md` for the full schema reference.
- DoD (all three must pass, in this order):

  ```bash
  cvs run <suite> --collect-only      # collects without error
  pytest cvs/lib/dtni/unittests/ -x   # no regressions
  cvs run <suite>                      # smoke run passes
  ```

- Stop-and-ask if: the smoke run fails. Diagnose before declaring done —
  do NOT loosen the threshold to make it pass.

## Definition of done (whole RUNBOOK)

- [ ] `cvs list <framework_name>` shows it
- [ ] `cvs run <suite> --collect-only` succeeds
- [ ] Smoke run emits both mandatory smoke metrics and verdict = `PASS`
- [ ] `pytest cvs/lib/dtni/unittests/ -x` is green
- [ ] No hardcoded `ssh` in the adapter
- [ ] Adapter is <= ~300 lines (if longer, consider extracting helpers, but
      do not over-abstract — see `../patterns/` for shared building blocks)

## End-user docs follow-up

Once the smoke run is green, update `docs/reference/configuration-files/*.rst`
per `../sphinx_rst.md` if this suite is user-facing.

## Stop-and-ask global triggers

- Topology does not match an existing pattern (single / distributed / disagg).
- Adapter needs to override `verify` — it cannot; `verify` is inline in
  `job.py`.
- Multi-role coordination requires more than `_launch_role` +
  `_wait_http_pool` provides.
- You need to mutate state outside `ctx.result.scalars` or `ctx.logs`.
