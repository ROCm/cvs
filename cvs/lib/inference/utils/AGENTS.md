# cvs/lib/inference/utils — inference-specific helpers

The inference half of the config/parsing machinery. Generic machinery
(`BaseVariantConfig`, placeholder substitution, `substitute_config`,
`evaluate_all`) lives one level up in `cvs/lib/utils/` — import from there, don't
duplicate it. This dir holds what's specific to *serving* workloads: the sweep
schema and the vLLM client-metric vocabulary.

## What's here

- `inferencing_config_loader.py` — the inference config schema: the named-combo
  sweep selector (`SeqCombo`/`Run`/`Sweep`), `GoodputSlo`, `Params` (vLLM bench
  flags), the `server` role, and `VariantConfig(BaseVariantConfig)` with
  `cell_key`/`expected_cells` + the threshold-coverage check.
- `vllm_parsing.py` — pure parsers for vLLM benchmark artifacts:
  `to_client_metrics()`, the `client.*` namespace, `CLIENT_METRICS` (the
  display surface), and `GATED_METRICS` (the asserted SLO subset).

## Public entry points

- `load_variant(config_path, cluster_dict) -> VariantConfig`
  The inference loader. Delegates file-read + substitution to
  `substitute_config`, attaches thresholds, builds + validates the typed
  `VariantConfig`. This is what the suite's `variant_config` fixture calls.
- `to_client_metrics(raw, *, tp, isl) -> {client.*: value}`
  Maps a stock `vllm bench serve` results dict to the namespaced metric dict.
  **Pure** — no I/O, no orchestration. Caller fetches + json-loads the artifact.
- `CLIENT_METRICS` / `CLIENT_METRIC_UNITS`
  The ordered `(short_name, unit)` list that becomes one HTML row per metric. The
  single definition every vLLM flavour (single-node, distributed, disagg,
  InferenceMax) shares — don't re-list rows per suite.
- `GATED_METRICS`
  The asserted subset of `CLIENT_METRICS`: the perf+health SLO contract a
  calibrated run must *assert*, not merely display. Membership = "out of range
  means FAILURE". Today: throughput (total + per-request output), the full
  latency distribution (mean/median/p90/p95/p99 per ttft/tpot/itl/e2el — itl has
  no p90 producer), and run health (`success_rate` floor, `failed` ceiling).
  Closed-world default: a new metric is record-only until its name is added
  here, at which point the loader forces a spec for it in every cell.
- `GoodputSlo`, `SeqCombo`, `Run`, `Sweep`, `Params`, `Roles`, `VariantConfig` —
  the typed schema.

## The sweep selector (the headline design point)

The old schema expanded `sequence_combinations × concurrency_levels` into an NxM
cartesian. This replaces it with **named combos + an explicit `runs[]` list**:

```json
"sweep": {
  "sequence_combinations": [ {"name": "w1_isl=128_osl=2048", "isl":"128", "osl":"2048", ...} ],
  "runs": [ {"combo": "w1_isl=128_osl=2048", "concurrency": 16} ]
}
```

Each `run` is one `(combo, concurrency)` cell — you enumerate exactly the cells
you want, no explosion. `Sweep`'s `@model_validator` rejects duplicate combo
names and any `run.combo` that names no combo, **at load time**.

## Gotchas worth not re-discovering

- **`cell_key` is the single source of truth** for a cell's threshold key
  (`ISL=…,OSL=…,TP=…,CONC=…`). The loader's coverage check and the test's verdict
  lookup both call it, so they can't drift on whitespace/ordering. If you change
  the key format, everything keyed on it (threshold.json, `expected_cells`) moves
  together — change it in one place.
- **`_check_thresholds_cover_sweep` is what prevents a silent green.** It checks
  two axes: (1) cell coverage — every sweep cell has a threshold entry, no key
  names a non-existent cell; (2) gated-metric coverage — every present cell
  carries a spec for every `GATED_METRICS` member. Without (1) a mistyped key
  makes the test skip its verdict; without (2) a gated metric with no spec falls
  through `test_metric`'s record-only branch and reports PASS with zero
  assertions even under `enforce_thresholds=true`. When `enforce_thresholds=false`
  both are *warnings* (record-only mode), not errors — intentional for
  un-calibrated configs, not a bug to "fix".
- **`pytest_generate_tests` reads raw JSON and bypasses this loader** (it runs at
  collection time, before fixtures exist). So the validators here are
  **mirrored** by hand in the suite's `pytest_generate_tests` (dup-name check,
  unknown-combo check, `GoodputSlo(**…)` validation). If you add a validator to
  `Sweep`, mirror it there too or collection won't enforce it.
- **`GoodputSlo` is an INPUT, not a threshold.** It's passed to
  `vllm bench serve --goodput`; it lives in the sweep, not threshold.json.
  Per-combo because e2el scales with osl. `_Forbid`, so a typo'd SLO key fails
  load rather than silently dropping the gate.
- **`to_client_metrics` is deliberately I/O-free.** Artifact layout is
  job-specific (single-node `cat`, disagg prefill+decode, distributed rank-0);
  the *fetch* lives in the orchestrator (`vllm_single.parse_results`), the
  *transform* lives here so every job reuses it. Keep it that way — don't add
  file reads here.
- **Derived metrics degrade to `None`, never crash.** `_safe_div` returns `None`
  on missing/None/zero-divisor. `evaluate_all` then reports `None` as a loud
  violation (not a `float(None)` TypeError). A metric the run couldn't produce
  shows `-` in the table, not a stack trace.
- **`client.goodput` is an alias** for stock's `request_goodput` (the name the
  table + threshold file use). Stock leaves `request_goodput` null unless
  `--goodput` was passed.
- **`Params` is the only framework-flavoured schema class** (it's the `vllm bench
  serve` flag set). `Sweep`/`SeqCombo`/`GoodputSlo`/`Roles`/`cell_key` are
  serving-generic. When a second serving framework lands, subclass `Params`; the
  rest is reusable as-is.

## When extending

- New `client.*` metric → add the derivation in `to_client_metrics` (guard
  divisors with `_safe_div`) AND a `(name, unit)` entry in `CLIENT_METRICS`. If
  it's a pass/fail criterion, also add it to `GATED_METRICS` (then every cell's
  threshold.json needs a spec for it) — otherwise it stays record-only.
- New combo field → add to `SeqCombo` (`_Forbid`, so also update any raw-JSON
  reader like `pytest_generate_tests`).
- Second framework → subclass `Params`; reuse everything else here and
  `BaseVariantConfig`/`substitute_config` from `cvs/lib/utils/`.
