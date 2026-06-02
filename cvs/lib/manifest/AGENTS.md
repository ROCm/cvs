# Manifest & sidecars — AGENTS.md

> Scope: `cvs/lib/manifest/` (`schema`, `sidecars`, `events`, `layout`,
> `export`). Workstream: W4. docs/ = how to USE; this file = how to EXTEND.

## Orientation

The durable output of every run. `Job` writes one `Manifest` (`schema.py`) — a
small, jq-friendly index — into the content-addressable `RunLayout` directory
(`layout.py`). Bulky numbers go to long-format Parquet **sidecars**
(`sidecars.py`), and lifecycle `events.jsonl` is written through `EventWriter`
against a CLOSED vocabulary (`events.py`). `export.py` flattens N manifests into
one fact table. This is the read surface for the pytest tiers (W6) and for
v2.A reuse — so its shape is a contract.

## Invariants (do not break)

- **The manifest stays small — pointers, not arrays.** `Manifest` holds
  identity, verdicts, scalars, phase timings, and `SidecarPointers`
  (`schema.py:142`). Per-sample/per-step arrays NEVER go in the manifest; they
  go to `samples.parquet` / `trajectory.parquet` referenced by path. Target
  5–50 KB, cat-able.
- **`events.jsonl` vocabulary is CLOSED.** `EventWriter.emit` raises
  `UnknownEventError` for any name outside `EVENT_VOCAB` (`events.py:19`). A new
  event name is a deliberate, reviewed edit to that frozenset — not an ad-hoc
  `log.info`. A file-backed writer also raises if `emit` is called after
  `close()` (no silent dropped writes) and streams (does NOT buffer `.records`);
  non-JSON-serializable `**fields` are persisted via `default=str` (lossy) — pass
  JSON-native values.
- **Manifest scalars stay strict-JSON.** `Verdicts.scalars` and
  `ResourceSummary.per_host` are `Optional[float]`; `Manifest.write` coerces any
  non-finite value (NaN loss, inf goodput) to `null` via `_finite_only` +
  `allow_nan=False`, so a failed run's manifest is still jq/duckdb-parseable. A
  reader must therefore guard `scalars[k]` for `None` before arithmetic.
- **Sidecars are long-format: a new metric is a new ROW, not a new column.**
  `TRAJECTORY_COLUMNS = [step, metric, value, role, host]` (`sidecars.py:20`).
  `write_trajectory` enforces it (a row with an off-schema key raises). Keeping
  the schema fixed is what makes the Parquet stable across frameworks. Per-sample
  rows stay wide (`SAMPLE_COLUMNS` is only the empty-input identity skeleton); new
  measurements arrive as data, not schema changes.
- **`workload_hash` + `verification_hash` are recorded day one.** `Identity`
  carries both (`schema.py:51`), populated by `Job._build_manifest` from
  `config.workload_hash()` / `config.verification_hash()`. They exist so v2.A
  reuse-manifests is a pure tack-on with no migration — never drop or repurpose
  them.
- **Commands/env are recorded as-is (no redaction).** Security was removed
  entirely (W7 / addendum B7, C4): `ConfigInputs` has `env`/`commands` and no
  redaction fields, and there is no secret-scrubbing helper. On closed clusters
  recorded values (including an inline token) are stored verbatim. Do not
  reintroduce redaction fields or a redaction claim.
- **`RunLayout` has a remote mirror for staging (A1).** `remote_root` /
  `to_remote()` mirror the content-addressable suffix on the remote node so
  G5's `stage_in`/`fetch` map local↔remote paths deterministically (local-keyed
  both ways; `to_remote` fail-closes with distinct errors). `RunLayout` itself
  does NO I/O — it only builds paths. `remote_artifact_dir` is G5-supplied;
  `remote_root` is `None` on single-host runs.
- **`ConfigInputs.env`/`commands` are G5-populated.** G3 ships the as-is fields
  (no redaction); G5's `_build_manifest` fills them. They default empty until
  then — do not assume the manifest carries env/commands in G3-only tests.

## Extension recipes

### Recipe: add a derived scalar or verdict field

Scalars are open: in the adapter's `parse`, put the value in
`ctx.result.scalars["<name>"]`; `Job` copies it into `Verdicts.scalars`
(`schema.py:131`) and `export._flatten_manifest` emits it as a `scalar_<name>`
fact-table column automatically. A *structural* verdict field (new typed key on
`Verdicts`/`Identity`) is a reviewed schema edit — add it with a default so old
manifests still validate (`extra="forbid"` is on every model).

**Verify:** `python -m unittest cvs.lib.unittests.test_dtni_manifest`

### Recipe: add a sidecar metric

Emit it as a row. For trajectory, append
`{"step": i, "metric": "<name>", "value": v, "role": r, "host": h}` to the list
you pass to `write_trajectory` (`sidecars.py:41`). For per-sample, add the key
to each sample dict consumed by `write_samples`; a threshold reads it via
`ResultView.sample_values("<name>")`. Do **not** add a column to
`TRAJECTORY_COLUMNS`.

**Verify:** `python -m unittest cvs.lib.unittests.test_dtni_manifest -k sidecar`

### Recipe: add an event name

Add the string to `EVENT_VOCAB` in `events.py` in a reviewed change, then
`ctx.events.emit("<name>", **fields)` at the call site. Names are
dotted/namespaced (`phase.detail`, e.g. `launch.role_ready`). Adding an
unsanctioned name will (correctly) raise at runtime.

**Anti-patterns:** stuffing arrays into the manifest; widening Parquet columns
per framework; emitting free-text events; computing `workload_hash` lazily or
omitting it; reintroducing redaction fields.

**Verify:** `python -c "from cvs.lib.manifest.events import EventWriter; EventWriter().emit('<name>')"` (no `UnknownEventError`).

## Reference

- Manifest layout and sidecar semantics: `docs/reference/dtni/` (`manifests`,
  `failure-patterns`). Don't copy `EVENT_VOCAB` here — read `events.py`.
