# cvs/lib/utils — shared, framework-agnostic config machinery

Pure functions and schema any CVS suite (inference, training, …) can call. No
suite-specific knowledge lives here. If something only an inference suite needs
crept in, it belongs in `cvs/lib/inference/utils/` instead — keep this boundary.

## What's here

- `config_loader.py` — the generic half of variant-config loading: the
  `paths`/`model`/`container` schema, `BaseVariantConfig`, the 3-pass
  placeholder substitution engine, and `substitute_config()`.
- `verdict.py` — `evaluate_all(actuals, thresholds)`: a tiny, framework-neutral
  threshold checker.

## Public entry points

- `substitute_config(config_path, cluster_dict) -> (raw_dict, thresholds)`
  Reads a variant `*_config.json` + its sibling `*threshold.json`, resolves all
  placeholders, strips threshold comment keys. Returns the substituted-but-
  **unvalidated** dict plus the parsed thresholds. A per-framework loader calls
  this, then builds its own typed `VariantConfig(**raw)`.
- `BaseVariantConfig` — subclass it per framework. Carries the shared fields
  (`schema_version`, `enforce_thresholds`, `paths`, `model`, `container`,
  `thresholds`) and the `model.remote==1` NotImplementedError guard. The image
  is declared once on `container.image` — there is no separate top-level block.
- `evaluate_all(actuals, thresholds)` — raises `ThresholdViolation` listing every
  failing metric. Threshold kinds: `min`, `max` (unit-agnostic ceiling, for
  counts like `failed`), `max_ms` (a ceiling that prints `ms`), `within`
  (±tolerance_pct), `min_tok_s`, `min_ratio` (compares against another metric
  named in `reference`).

## The seam (why this is split out)

`config_loader.py` used to be one monolith mixing generic machinery with
inference-only schema (`Sweep`/`SeqCombo`/`cell_key`). The generic half stayed
here; the inference half moved to
`cvs.lib.inference.utils.inferencing_config_loader`, which imports
`BaseVariantConfig`, `_Forbid`, and `substitute_config` from here. A second
framework (training) does the same: subclass `BaseVariantConfig`, reuse
`substitute_config` and `evaluate_all`, add its own schema in its own dir.

## Gotchas worth not re-discovering

- **Placeholder resolution is 3 ordered passes** (`substitute_config`):
  (1) cluster placeholders like `{user-id}` everywhere, (2) self-reference within
  the `paths` block (`{shared_fs}` → expanded inside other `paths.*`),
  (3) cross-block `{paths.models_dir}` → anywhere. An unknown `{token}` is left
  **verbatim**, not errored — a typo'd placeholder surfaces as a literal brace in
  a path, not a load failure.
- **Threshold file is found by sibling glob**, not by name: exactly one
  `*threshold.json` next to the config. Zero → `FileNotFoundError`; more than one
  → `ValueError` (ambiguous). So config and threshold can share a descriptive
  prefix (`llama31_70b_fp8_config.json` / `…_threshold.json`) but don't have to.
- **Threshold comment keys** (anything starting with `_`, e.g. `_comment`) are
  stripped before the coverage check, so you can document a threshold file inline.
- **`_Forbid` vs `_Allow`**: most schema classes forbid extra keys (a typo fails
  load loudly). `RuntimeSpec` is `_Allow` (orchestrator runtime args are
  open-ended). Don't loosen `_Forbid` to silence a "valid" key — add the field.
- **`BaseVariantConfig` validators run parent-first.** The remote-not-implemented
  check is intentionally first so a remote config fails fast before any
  subclass's (meaningless-for-a-rejected-config) coverage check runs. If you add
  a base validator, mind the ordering contract.
- **`container.model_dump()` is the orchestrator contract.** The loaded
  `container` field serialises to the exact `{lifetime, name, image, runtime:
  {name, args}}` shape `OrchestratorConfig.from_configs` consumes. Don't reshape
  it here.
- `model.remote==1` raises `NotImplementedError` (remote model download is
  unported PoC scope). The schema accepts it; the validator rejects it.

## When extending

- New generic threshold kind → add a branch in `verdict._check_one` and document
  it above. Keep it metric-name-agnostic.
- New shared config field every suite needs → add to `BaseVariantConfig`.
- Anything inference/training-specific → NOT here. See the sibling
  `cvs/lib/inference/utils/AGENTS.md`.
