# cvs/lib/utils — framework-agnostic config machinery

**Boundary**: if every CVS suite (inference, training, ...) needs it, it belongs here.
Inference-only symbols belong in `cvs/lib/inference/utils/`; single-framework symbols in `cvs/lib/<framework>/utils/`.

---

## Files

### `config_loader.py`

#### Schema classes

| Class | Models | Extra keys |
|---|---|---|
| `_Forbid` | Base: extra keys forbidden | — |
| `_Allow` | Base: extra keys allowed | — |
| `Paths` | Shared filesystem paths | `_Forbid` |
| `ModelSpec` | Model identity and fetch mode | `_Forbid` |
| `RuntimeSpec` | Orchestrator runtime (name + open-ended args) | `_Allow` |
| `ContainerSpec` | Container lifecycle and image | `_Forbid` |
| `BaseVariantConfig` | Framework-agnostic skeleton all suites share | `_Forbid` |

**`Paths`** — `shared_fs: str`, `models_dir: str`, `log_dir: str`, `hf_token_file: str`

**`ModelSpec`** — `id: str`, `remote: Literal[0, 1]`

**`RuntimeSpec`** (`_Allow`) — `name: str`, `args: Dict[str, Any]` (defaults to `{}`).
`_Allow` because orchestrator runtime options are framework-specific.

**`ContainerSpec`** (`_Forbid`):

| Field | Type | Default | Meaning |
|---|---|---|---|
| `lifetime` | `Literal["no_launch", "per_run", "persistent"]` | `"per_run"` | `"no_launch"` — skip container management entirely; `"per_run"` — tear down and re-create each run; `"persistent"` — reuse an already-running container |
| `name` | `str` | required | Container name |
| `image` | `str` | required | Declared once here; no separate top-level image block |
| `runtime` | `RuntimeSpec` | required | Nested `RuntimeSpec`; its serialised form inside `container.model_dump()` is `{name, args}` — the full container dump is `{lifetime, name, image, runtime: {name, args}}` |

**`BaseVariantConfig`** (`_Forbid`) shared fields:

| Field | Type | Default | Notes |
|---|---|---|---|
| `schema_version` | `Literal[1]` | required | — |
| `enforce_thresholds` | `bool` | `True` | `False` → coverage failures become warnings; test runs record-only |
| `threshold_json` | `str` | required | Literal absolute path; see contract below |
| `paths` | `Paths` | required | — |
| `model` | `ModelSpec` | required | — |
| `container` | `ContainerSpec` | required | — |
| `thresholds` | `Dict[str, Dict[str, Any]]` | `{}` | Populated by the loader, not the config file |

`BaseVariantConfig` carries one `@model_validator(mode="after")`:

**`_check_remote_not_implemented`** — raises `NotImplementedError` when `model.remote == 1`.
Runs first (parent-class validators precede subclass validators), so a remote config fails fast
before any subclass coverage check runs on a config that will be rejected anyway.

---

#### `substitute_config(config_path, cluster_dict) -> (raw_dict, thresholds)`

- **Accepts**: path to a variant `_config.json` + a resolved cluster dict
- **`threshold_json` handling**: read as a literal absolute path from the raw (un-substituted) config
  before any substitution pass runs — **no placeholder substitution of any kind is applied to it**
- **3-pass substitution** (see `docs/placeholder-substitution.md` for worked example):
  1. Cluster placeholders (`{user-id}`, etc.) resolved everywhere in the document
  2. Self-reference within the `paths` block (`{shared_fs}` expanded inside other `paths.*` values)
  3. Cross-block references (`{paths.models_dir}`, etc.) resolved anywhere in the document
- **Strips** `_`-prefixed comment keys from thresholds before returning
- **Returns**: substituted-but-**unvalidated** dict + parsed thresholds
- **Does NOT**: validate, type-coerce, or build a typed config — that is the caller's job
- **Unknown `{token}`**: left verbatim (no error; typo surfaces as a literal brace in a path)

---

#### `_resolve_cluster_mapping(cluster_dict)`

- Returns `{"user-id": <username>}`
- Falls back to `getpass.getuser()` when `cluster_dict` has no `username` key (or it is falsy)
- This is how `{user-id}` resolves on clusters without an explicit `username` field

---

### `verdict.py`

**`ThresholdViolation(Exception)`**
- `.violations: list[str]` — all failure strings
- Exception message is the violation strings joined by newlines

**`evaluate_all(actuals, thresholds)`**

| Situation | Behaviour |
|---|---|
| Metric in `thresholds` but not in `actuals` | Violation string (not `KeyError`) |
| Metric present in `actuals` with value `None` | Loud violation string (not `float(None)` crash) |
| Metric present in `actuals` with a non-numeric, non-None value | Uncaught `ValueError` from `float()` — callers must ensure actuals values are numeric or `None` |
| `min_ratio` spec | `evaluate_all` injects `_actuals` into the spec dict before calling `_check_one`; callers never set `_actuals` |
| `min_ratio` — **reference** metric value is `None` | Caught inside `_check_one` (not in `evaluate_all`'s per-metric guard); `_check_one` returns a violation string when `actuals[ref_metric] is None`. The `evaluate_all` `None` guard covers only the **primary** metric being checked, not the reference metric for ratio specs. |
| Multiple failures | Raises `ThresholdViolation` listing ALL failures, not just the first |

- `actuals`: `{metric_name: value_or_None}`
- `thresholds`: `{metric_name: spec_dict}`

See `docs/threshold-kinds.md` for the full threshold kind reference.

---

## The boundary rule

| Question | Answer |
|---|---|
| Does every CVS suite need it? | `cvs/lib/utils/` |
| Do only serving/inference suites need it? | `cvs/lib/inference/utils/` |
| Does only one framework (vllm, megatron, jax) need it? | `cvs/lib/<framework>/utils/` |

When in doubt: "does any other suite need this?" → move it up one layer.

---

## Subclassing `BaseVariantConfig`

Contract for new suite authors:

**Must add:**
- `framework: Literal["your_name"]`
- `params` — your framework's CLI flags schema
- `sweep` — your sweep schema

**Must implement:**
- `cell_key(...)` — returns a string key matching threshold.json top-level keys
- `expected_cells()` — returns a list of all cell keys the sweep produces

**Must add:**
- A `@model_validator(mode="after")` that performs threshold-coverage checking
  (equivalent to `_check_thresholds_cover_sweep` in `inferencing_config_loader.py`).
  The check must cover **two axes**:
  1. **Cell coverage** — sweep cells with no threshold entry AND threshold keys
     that match no sweep cell (both directions; a one-way check silently skips
     orphaned threshold entries).
  2. **Gated-metric coverage** — for every cell that is present in both the sweep
     and the threshold file, every member of the framework's gated-metric set must
     have a spec. Without this axis, a gated metric with no spec falls through the
     `spec is None` record-only branch and silently reports a green PASS with zero
     assertions even under `enforce_thresholds=True`. For the vllm/inference
     framework this set is `GATED_METRICS` imported from
     `cvs.lib.inference.utils.vllm_parsing`; a new framework author must define an
     equivalent set.

**Validator ordering:** parent-class validators run before subclass validators.
`_check_remote_not_implemented` always fires first — do not add a base validator that
assumes a valid config before this check passes.

**`load_variant`:** always delegate to `substitute_config` — never reimplement file-read
or substitution. After calling `substitute_config`, attach `thresholds`, then build
`YourVariantConfig(**raw)`.

---

## Gotchas

- **`threshold_json` is a literal absolute path** — not a glob, not relative to the config
  file. It is read from the raw un-substituted config before Pass 1 runs, so no placeholder
  substitution (not even `{user-id}`) applies to it. If your threshold path needs to vary
  by user, it must be pre-resolved before being written into the config file.
- **Unknown `{token}` left verbatim** — a typo surfaces as a literal brace in a path at
  runtime, not a load failure. Check paths block values after loading if substitution is
  suspected to have silently failed.
- **`_Forbid` vs `_Allow`**: never loosen `_Forbid` to silence an "extra key" validation
  error — add the field explicitly.
- **Validator ordering**: `BaseVariantConfig` validators run before subclass validators;
  `_check_remote_not_implemented` always fires first. Do not add a subclass validator that
  assumes `model.remote == 0` without relying on this ordering guarantee.
- **`_resolve_cluster_mapping` fallback**: if the running user differs from the cluster user,
  verify `cluster_dict` has a `username` key; omitting it silently resolves `{user-id}` to
  the local OS user.
- **`container.model_dump()` is the orchestrator contract** — serialises to
  `{lifetime, name, image, runtime: {name, args}}` that `OrchestratorConfig.from_configs`
  consumes; do not reshape the dict before passing it.
