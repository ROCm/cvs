# Job driver & RunContext — AGENTS.md

> Scope: `cvs/lib/job.py`, `cvs/lib/run_context.py`. Workstream: W1 driver.
> Companion to `cvs/lib/manifest/AGENTS.md` (manifest shape) and the G5a
> contract files (`adapter_protocol.py`, `base_adapter.py`, `registry.py`).

## Invariants

- **Event ownership is split (B6).** `Job` owns *phase-boundary* events:
  `prepare.start`, `prepare.done`, `parse.done`, `teardown.start`,
  `teardown.done` (and `seed.logged` once at prepare). Adapters own
  *sub-phase* events: `launch.container_up`, `launch.role_ready`, `step`,
  `request`, `arrival.*`, `accuracy.*`, `safety.violated`, `verify.*`,
  `pattern.matched`. Do not emit phase-boundary events from an adapter; do
  not emit sub-phase events from `Job`. The split is what makes the events
  log analyzable across frameworks without duplicate-emit dedup.
- **Failures are classified at the raise site (B4).** `WorkloadFailure`
  subclasses carry their `category`; the driver records `exc.category.value`
  directly. Generic exceptions inside `prepare`/`launch` are re-raised as
  `SetupFailure` *at the boundary*, not classified post-hoc. A fatal pattern
  hit (`severity == "fatal"`) may upgrade an unclassified verdict
  (`complete`) or the generic-exception verdict to `failure_pattern_matched`
  -- but it MUST NOT clobber a classified `WorkloadFailure` (SafetyViolation
  / LivenessFailure / VerificationFailure). The taxonomy's
  earliest-raise-site rule wins; the override is an upgrade, not a relabel.
  The Job-owned phase events for `prepare` and `parse` are emitted only on
  the success path (`prepare.done`, `parse.done`); `teardown.start`/`done`
  always fire because teardown runs in `finally`. The events sidecar is
  always closed.
- **Teardown runs in `finally`.** Even when `prepare`/`launch`/`await`/
  `parse`/`verify` raises -- and even when teardown itself raises -- the
  manifest is still written, the events file is still closed, and the
  `teardown.start`/`teardown.done` boundary events still fire. A leaked
  container or an unflushed manifest is a regression.
- **`RunContext.executor` is a single optional slot (A1+A2 deferred).**
  The spine ships one `executor` field but no `stage_in`/`fetch` API and
  no helper to populate it. Real CVS clusters use a shared filesystem
  (Weka/NFS at the same path on devbox + nodes) or an end-of-run rsync,
  so per-file SFTP staging is not on the critical path. G3 already ships
  the `RunLayout.remote_artifact_dir` / `to_remote` plumbing, so adding
  the wrappers when the first no-shared-FS cluster lands is a pure
  addition. Per-role executors (A2) land alongside the first multi-role
  adapter (sglang-disagg / megatron / jax).
- **`ConfigInputs.env`/`commands` are recorded as-is (B7).** No redaction
  (W7 removed entirely). `env` is sourced from `config.container.env`;
  `commands` from `ctx.scratch["commands"]` (adapters append the
  docker/curl/torchrun lines they want forensically recorded). Do not
  reintroduce a redaction field or claim.

## Extension recipes

### Recipe: add a new adapter

Subclass `BaseWorkloadAdapter`, implement `launch` / `progress_predicate` /
`parse`, and `@register_adapter("<framework>", kind="inference"|"training")`.
Stage docker/torchrun lines onto `ctx.scratch.setdefault("commands", [])` if
you want them in the manifest. Do not emit phase-boundary events.

### Recipe: record a derived command for forensics

`ctx.scratch.setdefault("commands", []).append("<command>")`. `Job` copies
the list into `ConfigInputs.commands` verbatim.

### Recipe: add a sub-phase event

Add the name to `EVENT_VOCAB` in `manifest/events.py` (reviewed change),
then `ctx.events.emit("<name>", **fields)` from the adapter.

**Anti-patterns:** classifying failures by message text; emitting
phase-boundary events from an adapter; populating `ConfigInputs.env`
twice; building a per-role executors dict before a real multi-role
adapter needs one.
