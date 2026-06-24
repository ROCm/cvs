# cvs/lib/inference/utils — inference-specific config and parsing

**Boundary**: this is the serving/inference half of the config machinery.
The generic half (`BaseVariantConfig`, `substitute_config`, `evaluate_all`, `Paths`, `ContainerSpec`)
lives in `cvs/lib/utils/` — import from there, never duplicate it here.

---

## Files

### `inferencing_config_loader.py`

#### Schema classes

**`RoleServer`** (`_Forbid`): per-model server overrides.
- `serve_args: Dict[str, Any]` — extra `vllm serve` flags; scalar → `--flag value`,
  `True` → bare `--flag`, list → flag repeated per element
- `env: Dict[str, str]` — env vars merged over orchestrator defaults
- Both default empty; fp8-kv cells set `--kv-cache-dtype` here to keep the generic driver model-agnostic

**`Roles`** (`_Forbid`): wraps `RoleServer`.
- `server: RoleServer` — defaults to empty `RoleServer()`

**`GoodputSlo`** (`_Forbid`): per-combo goodput gate, in milliseconds.
- `ttft_ms: float`, `tpot_ms: float`, `e2el_ms: float`
- **INPUT to the run** (passed to `vllm bench serve --goodput`), NOT a threshold to assert.
  Lives in the sweep, not `threshold.json`. `_Forbid` ensures a typo'd SLO key fails load
  rather than silently drop the SLO and run with the wrong gate on hardware.

**`SeqCombo`** (`_Forbid`): one named sequence-length combination.
- `name: str` — the join key referenced by `Run.combo`
- `isl: str`, `osl: str`
- `goodput_slo: Optional[GoodputSlo]` — omit when no goodput gate is needed

**`Run`** (`_Forbid`): one sweep cell — a named combo at a single concurrency.
- `combo: str` — references a `SeqCombo.name`
- `concurrency: int`
- Explicit `runs[]` replaces the old NxM cartesian (`sequence_combinations × concurrency_levels`);
  you enumerate exactly the cells you want

**`Sweep`** (`_Forbid`): the full sweep selector.
- `sequence_combinations: List[SeqCombo]`
- `runs: List[Run]`
- `@model_validator(mode="after")` delegates to `validate_sweep_selector`

**`Params`** (`_Forbid`): `vllm bench serve` CLI flags; all fields are `str`.

| Field | Default | Notes |
|---|---|---|
| `backend` | `"vllm"` | |
| `base_url` | `"http://0.0.0.0"` | |
| `port_no` | `"8888"` | |
| `dataset_name` | `"random"` | |
| `burstiness` | `"1.0"` | |
| `seed` | `"0"` | |
| `request_rate` | `"inf"` | |
| `random_range_ratio` | `"0.8"` | |
| `random_prefix_len` | `"0"` | |
| `tensor_parallelism` | `"1"` | used in `cell_key` and `per_gpu_throughput` |
| `tokenizer_mode` | `"auto"` | |
| `percentile_metrics` | `"ttft,tpot,itl,e2el"` | |
| `metric_percentiles` | `"50,90,95,99"` | |
| `num_prompts` | `"3200"` | overridden per-cell by `_num_prompts_for` |
| `client_poll_count` | `"20"` | see below |

`client_poll_count` semantics: total client wait budget =
**(client_poll_count × 60 s) + 120 s initial wait**.
The poll loop exits as soon as the client finishes, so raising this never slows down fast cells.
Raise it for high-osl cells where large-output runs take longer to complete. (Regression: REG-20260609-001)

**`VariantConfig(BaseVariantConfig)`**: the full typed config.
- Adds: `framework: Literal["vllm_single"]`, `gpu_arch: str`, `roles: Roles = Roles()`,
  `params: Params`, `sweep: Sweep`
- Implements: `cell_key(isl, osl, concurrency)`, `expected_cells()`
- Has: `@model_validator(mode="after") _check_thresholds_cover_sweep`

---

#### Public functions

**`load_variant(config_path, cluster_dict) -> VariantConfig`**

The function a suite's `variant_config` fixture calls.

1. Delegates file read + 3-pass placeholder substitution to `substitute_config`
2. Attaches thresholds returned by `substitute_config` to the raw dict
3. Builds and returns a typed, validated `VariantConfig`

Does not reimplement file reading or substitution — always calls `substitute_config`.
See `cvs/lib/utils/AGENTS.md` for the full `substitute_config` contract.

**`validate_sweep_selector(combo_names, run_combo_refs)`** — PUBLIC ENTRY POINT

Shared rule called by **both**:
- the typed `Sweep` validator at load time
- `pytest_generate_tests` at collection time (reads raw JSON before the loader runs)

Checks:
- combo names are unique (duplicate → `ValueError`)
- every `run.combo` names a known `sequence_combination` (unknown → `ValueError`)

Operates on plain `list[str]` so both call sites feed it without the full typed schema.
If you add a sweep check, add it here so both paths enforce it without drift.

---

#### `_check_thresholds_cover_sweep` — two-axis coverage check

`@model_validator(mode="after")` on `VariantConfig`. Fails at load time if the threshold file
does not match the sweep matrix.

**Axis 1 — cell coverage**
- Every sweep cell produced by `expected_cells()` has an entry in `threshold.json`
- No threshold key names a non-existent cell (catches typos in threshold key names)

**Axis 2 — gated-metric coverage**
- Every cell present in both sets has a spec for every `client.<GATED_METRICS member>` key
  (e.g. `client.total_token_throughput`, not `total_token_throughput`)
- Without this, a gated metric with no spec falls through `test_metric`'s `spec is None`
  record-only branch and reports PASS with zero assertions even under `enforce_thresholds=true`
- Only checked for cells present in both expected and threshold sets (missing cells are already
  reported by axis 1; no double-reporting)

When `enforce_thresholds=false`: both failures become warnings, not errors.
The config loads as a record-only scaffold (metrics captured, nothing asserted).

See `docs/cell-key-format.md` for the exact key format used in `threshold.json`.

---

### `vllm_parsing.py`

**`to_client_metrics(raw, *, tp, isl) -> dict`**

Pure — no I/O, no orchestration. `raw` is the already-parsed JSON the load generator
writes to its `--result-dir` results artifact. Caller is responsible for fetching and
`json.load`-ing the artifact.

- All stock scalars namespaced 1:1 as `client.<key>`
- `client.goodput` = alias for stock's `request_goodput` (the name threshold files reference)
- Derived metrics (all guarded by `_safe_div`; degrade to `None` on missing/`None`/zero-divisor):

| Metric | Formula |
|---|---|
| `client.per_gpu_throughput` | `total_token_throughput / tp` |
| `client.normalized_ttft_ms_per_tok` | `mean_ttft_ms / isl` |
| `client.decode_latency_ratio` | `p99_itl_ms / p50_itl_ms` |
| `client.decode_throughput_p50` | `1000.0 / median_tpot_ms` |
| `client.success_rate` | `completed / (completed + failed)` |

See `docs/derived-metrics.md` for per-metric inputs, None conditions, and gated/record-only rationale.

**`CLIENT_METRICS`** — ordered `list[(short_name, unit)]`.

The display surface: one HTML row per metric per cell. Single definition shared by all vLLM
flavours — do not re-list per suite.

**`CLIENT_METRIC_UNITS`** — `dict` form of `CLIENT_METRICS`.

**`GATED_METRICS`** — the asserted subset of `CLIENT_METRICS`.

Membership = "out of range means FAILURE".

Closed-world default: a new metric added to `CLIENT_METRICS` is record-only until its name
is explicitly added to `GATED_METRICS`. The loader's coverage check then forces a spec for
that metric in every cell before the suite can run green.

Currently gated:

| Category | Members |
|---|---|
| Throughput | `total_token_throughput`, `output_throughput` |
| TTFT latency | `mean`, `median`, `p90`, `p95`, `p99` |
| TPOT latency | `mean`, `median`, `p90`, `p95`, `p99` |
| ITL latency | `mean`, `median`, `p95`, `p99` (no p90 producer) |
| E2EL latency | `mean`, `median`, `p90`, `p95`, `p99` |
| Run health | `success_rate` (floor), `failed` (ceiling) |

Record-only by design: inputs (`num_prompts`), totals (`total_input_tokens`,
`total_output_tokens`), secondary throughputs (`per_gpu_throughput`, `request_throughput`,
`goodput`, `decode_throughput_p50`, `max_output_tokens_per_s`), diagnostic derivations
(`normalized_ttft_ms_per_tok`, `decode_latency_ratio`).

---

## The sweep selector

Named combos + explicit `runs[]` list replaces old NxM cartesian
(`sequence_combinations × concurrency_levels`). One `Run` = one `(combo, concurrency)` cell.
The sweep enumerates exactly the cells you want.

`sequence_combinations` names each ISL/OSL shape once (with an optional goodput SLO);
`runs` references those names at specific concurrencies. This lets you include only the
cells that matter for a given model — no silent NxM explosion, no empty cells.

---

## The serving-generic / vllm-specific seam

`Params` is the only vllm-specific class. Everything else (`Sweep`, `SeqCombo`, `GoodputSlo`,
`Roles`, `cell_key`) is serving-generic and reusable when a second serving framework lands.

**Second serving framework checklist:**
1. Subclass `Params` with your framework's CLI flags
2. Reuse `Sweep`/`SeqCombo`/`GoodputSlo`/`Roles` unchanged
3. Reuse `validate_sweep_selector` in your `pytest_generate_tests`
4. Write your own metric vocabulary (`<framework>_parsing.py`)
5. Define your own `GATED_METRICS`

---

## Lifecycle-as-tests model

Each stage of the test run is an **independent pytest test**, not fixture body code.
Each stage appears as a timed, independently pass/fail row in the HTML report.

Standard lifecycle order (pinned in `pytest_collection_modifyitems`):

| Rank | Test | Action |
|---|---|---|
| 0 | `test_launch_container` | `setup_containers()`; asserts container is running |
| 1 | `test_setup_sshd` | `setup_sshd()`; probes `:2224` for multinode only |
| 2 | `test_model_fetch` | ensures model bytes present; polls or downloads if remote |
| 3 | `test_vllm_inference` | benchmark loop per cell; stores results in `inf_res_dict` |
| 4 | `test_metric` | one test per metric per cell; reads `inf_res_dict`; asserts verdict |
| 5 | `test_print_results_table` | summary log; must run after all cells |
| 6 | `test_teardown` | `teardown_containers()`; sets `lifecycle.torn_down`; **never skips** |

Rules:
- Every test except `test_launch_container`, `test_teardown`, and `test_print_results_table`
  checks `lifecycle.failed` and skips if true. `test_launch_container` is the first stage
  and is itself responsible for setting `lifecycle.failed`; it has no prior stage to guard
  against. `test_teardown` must run even on failure. `test_print_results_table` guards only
  on whether `inf_res_dict` is empty and logs whatever results were recorded.
- `test_vllm_inference` catches exceptions, sets `lifecycle.failed = True`, re-raises
- `test_teardown` never skips — must run even on failure; sets `lifecycle.torn_down = True`
  to suppress the `orch` fixture's leak-guard finalizer (prevents double teardown)

`test_metric` verdict pattern:
- Reads value from `inf_res_dict`; attaches to `user_properties` for HTML rendering
- If `enforce_thresholds` and a spec exists for this cell+metric: calls `evaluate_all`
  with **the full per-cell actuals dict** (not just the single metric value) so a
  `min_ratio` spec can resolve its reference metric
- Otherwise: record-only PASS

---

## conftest fixtures

All fixtures are `scope="module"`.

| Fixture | Owns | Key detail |
|---|---|---|
| `cluster_dict` | reads `--cluster_file` JSON; resolves placeholders | calls `resolve_cluster_config_placeholders` |
| `variant_config` | calls `load_variant(config_file, cluster_dict)` | the sole entry point to the typed schema |
| `lifecycle` | `_Lifecycle` instance | shared cross-test state: `failed`, `torn_down`, `report` |
| `orch` | builds `ContainerOrchestrator`; registers leak-guard finalizer | deep-merges variant container block onto cluster container block |
| `hf_token` | reads `variant_config.paths.hf_token_file` | skips if file absent |
| `inf_res_dict` | module-scoped `{}` keyed by `(model_id, gpu_arch, isl, osl, combo_name, concurrency)` | populated by `test_vllm_inference`; consumed by `test_metric` |

`_deep_merge` helper: `OrchestratorConfig.from_configs` does a top-level `dict.update`,
so a bare variant container block wipes the cluster file's container settings. The conftest
deep-merges the variant block onto the cluster block so cluster-set scalar/dict keys survive
with the variant winning on conflicts. List keys (e.g. runtime args, volume mounts) are
replaced at the merge step and recombined additively downstream in `container.py`'s getters.

Required pytest hooks (all in `conftest.py`):
- `pytest_collection_modifyitems` — pins lifecycle order; imported functions (e.g.
  `test_print_results_table` from `_shared.py`) sort by source line, not insertion order,
  so explicit pinning is mandatory
- `pytest_runtest_makereport` — attaches lifecycle timing rows to the HTML detail panel
- `pytest_html_results_table_header` / `pytest_html_results_table_row` — adds Value/Unit
  columns; populated for `test_metric` rows, blank for lifecycle/inference rows

---

## `pytest_generate_tests` mirror rule

`pytest_generate_tests` runs at **collection time**, before any fixtures exist.
It reads the raw config JSON directly — it cannot use the `variant_config` fixture.

To keep collection-time validation aligned with load-time validation:

1. Call `validate_sweep_selector` on the raw combo names and run combo refs — mirrors the
   typed `Sweep` validator so duplicate names and unknown refs fail collection, not silently drop
2. Validate each raw `goodput_slo` dict through `GoodputSlo(**combo["goodput_slo"])` —
   mirrors the `_Forbid` model so a typo'd SLO key fails collection, not runs with the wrong gate

**Rule**: if you add a check to `Sweep`, add it to `validate_sweep_selector` (or an equivalent
call in `pytest_generate_tests`) so both paths enforce it without drift.

---

## Gotchas

- **`cell_key` is the single source of truth** — the loader coverage check (`expected_cells`)
  and the test verdict lookup (`test_metric`) both call it; change the format in one place and
  everything keyed on it moves together. See `docs/cell-key-format.md` for the exact format.

- **Both axes of `_check_thresholds_cover_sweep` must pass** — without axis 2 a gated metric
  with no spec reports zero-assertion PASS even under `enforce_thresholds=true`; the silent
  green is indistinguishable from a real pass.

- **`pytest_generate_tests` reads raw JSON** — it runs before fixtures exist; mirror every
  `Sweep` validator via `validate_sweep_selector` or the two paths drift.

- **`GoodputSlo` is an INPUT, not a threshold** — typo'd key fails load (`_Forbid`);
  lives in the sweep config, not `threshold.json`.

- **`to_client_metrics` is deliberately I/O-free** — the fetch lives in the job class;
  callers hand in an already-parsed dict.

- **Derived metrics degrade to `None` via `_safe_div`, never crash** — `None` renders as
  `-` in the HTML table; if the metric is gated, `evaluate_all` will report a loud violation.

- **All gated-metric threshold keys must use the `client.` prefix** — the axis 2 check
  resolves `client.<name>` in the threshold dict, so a bare key (e.g. `total_token_throughput`)
  is treated as absent and triggers a coverage failure even though the entry is present.

- **`client.goodput` is an alias** for stock's `request_goodput` — the names differ;
  threshold files must use `client.goodput`, not `client.request_goodput`.

- **`client_poll_count` controls the total client wait budget**; too low and a long-osl
  run times out before the client finishes. Raising it never slows down fast cells.
  (Regression: REG-20260609-001)
