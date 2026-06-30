# Spec: Integrate PR #244 report generation into the vLLM suite

**Branch:** `vllm-distributed-with-reports` (off `atnair/vllm-distributed`)
**Goal:** bring the render-only inference report library from PR #244
(ROCm/cvs, head `hnimrama/IX-atom-experimental-run-deck`) into the unified vLLM
suite so `cvs run vllm … --html` emits the HTML/JSON dashboard, interactive
viewer, and CI summary — **without changing any pass/fail behavior.**

---

## 0. Status of this branch

Already done (committed):
- Worktree/branch created off `atnair/vllm-distributed` (tip `79b80ae`).
- Commit `7b25aaf` carries the previously-unstaged `vllm_job.py` distributed
  fixes (worker ranks launch `--headless`; `is_ready()` greps only the head
  rank for the startup pattern when `nnodes > 1`).
- PR #244 head fetched as `FETCH_HEAD` (`432a689`) for cherry-copying files.

Remaining work is everything in §4.

---

## 1. Branch topology (why this is a file-copy, not a merge)

```
origin/main ──┬── (59 commits) ──> origin/main tip (e37f90d)
              │
              └── atnair/vllm-distributed (cut at daff5ab, +6 commits, 79b80ae)
                          │
                          └── vllm-distributed-with-reports  ← THIS branch
                                       └── 7b25aaf (carried vllm_job.py fix)

hnimrama/IX-atom (cut at daff5ab, +144 commits)
              └── hnimrama/IX-atom-experimental-run-deck = PR #244 (432a689)
```

`vllm-distributed` was cut from `main`, **not** from `IX-atom`. Therefore it
has **none** of PR #244's code. The PR (+5508/−313, 61 files) cannot be merged
or rebased in cheaply (histories diverge 144 vs 6 commits from the common base
`daff5ab`). We pull the report **library** in as a file copy on top of this
branch's tip. The common ancestor of both lines is `daff5ab`, so the report
package's transitive dependencies that already existed at `daff5ab` are present
on this branch too; only PR-*added* files are missing (enumerated in §3).

---

## 2. The integration contract (why it's low-risk)

PR #244's report system is **opt-in per suite** and **render-only**:

- A suite opts in only by adding `cvs/lib/report/presets/<run_stem>.py`
  defining an `InferenceReportConfig`. At session start, root `conftest.py`'s
  `try_auto_register_inference_suite_report(config)` imports
  `cvs.lib.report.presets.<stem>` where `stem == config._suite_name`.
- `config._suite_name` is derived in `cvs/conftest.py::pytest_configure` as the
  **basename of the test `.py` file** passed to pytest.
- `cvs run <name>` resolves `<name>` to a test file via
  `list_plugin.discover_tests()` (walks `cvs/tests/**`, maps each file's stem →
  module path) and runs `<file>`. For the unified suite, `cvs run vllm` →
  `cvs/tests/inference/vllm/vllm.py` → **stem `vllm`**.
- **Therefore the preset filename MUST be `presets/vllm.py`.** (verified on the
  TARGET branch: `cvs/tests/inference/vllm/vllm.py` exists; see §6 R1 for the
  `vllm_single` / legacy `vllm_distributed` stems.)

> **Reviewer trap — verify against the TARGET branch, not `FETCH_HEAD`.** The
> unified `vllm.py` suite lives on `atnair/vllm-distributed` (this branch),
> NOT on the PR/IX-atom line. `git show FETCH_HEAD:cvs/tests/inference/vllm/`
> shows only the OLD `vllm_single.py` (the PR predates the unification). All
> "vLLM suite provides…" claims below are verified against the working tree on
> THIS branch (`grep … cvs/tests/inference/vllm/vllm.py`), not against
> `FETCH_HEAD`. In particular the suite records `topology_discovery` and does
> NOT record `sshd_setup` — opposite to the old `vllm_single.py`.
- Every new conftest hook is gated behind
  `isinstance(get_suite_report_config(config), InferenceReportConfig)`. With no
  matching preset, every hook no-ops → **all other suites are unaffected.**

The engine reads only data the vLLM suite already produces:

| Engine expects | vLLM suite provides | Verified at |
|---|---|---|
| `inf_res_dict[(model, gpu, isl, osl, policy, conc)] → {host: actuals}` | identical 6-tuple key | `vllm.py` `test_vllm_inference`, key built lines ~255–263 |
| `actuals["client.<metric>"]` | `to_client_metrics` emits `client.*` | `vllm_parsing.py` |
| `variant_config.cell_key(isl, osl, conc)` (engine: `build_cell_record`) | `VariantConfig.cell_key` (adds `PP=` for distributed) | `vllm_config_loader.py:205` |
| `variant_config.thresholds` (dict keyed by cell_key) | present | `vllm_config_loader.py` |
| `variant_config.enforce_thresholds` | present (default True) | `vllm_config_loader.py:170` |
| `variant_config.model.id`, `.gpu_arch` | present | suite + loader |
| `lifecycle.report{nodeid → [(label, value, unit)]}` | `_Lifecycle.record` | `vllm/conftest.py` |

Fixtures the engine binds (`inf_res_dict`, `variant_config`, `lifecycle`) are
**all already module-scoped fixtures** in `cvs/tests/inference/vllm/conftest.py`.

---

## 3. Inventory: present vs missing on this branch

### Already present (no port needed) — verified on this branch
- `cvs/lib/report_plugins.py` with `HtmlReportManager`, `add_html_to_report`,
  `create_zip_bundle`, `is_enabled` (property), `_custom_test_reports`.
- `cvs/lib/utils/verdict.py` with `_check_one` and `evaluate_all`
  (engine dep `cvs.lib.utils.verdict._check_one` satisfied).
- vLLM fixtures + suite (`cvs/tests/inference/vllm/{vllm,_shared,conftest}.py`).
- `cvs/lib/inference/utils/vllm_parsing.py` with `CLIENT_METRICS`,
  `CLIENT_METRIC_UNITS`, `GATED_METRICS`.

### Missing — MUST be ported from `FETCH_HEAD`
1. **`cvs/lib/report/` package** — 43 PR-added files (engine, `presets/`,
   `viewer/`, `panels/`, `render/`, `unittests/`, docs).
2. **`HtmlReportManager.generate_suite_reports()`** — method the PR *adds* to
   `report_plugins.py` (target has the class but not this method).
3. **Root `cvs/conftest.py` report wiring** — the PR's conftest diff.
4. **`cvs/lib/inference/inference_suite_lifecycle.py`** — exists on the IX-atom
   line but NOT on this branch (absent from `main`). We need only its HTML
   helper `attach_lifecycle_html_table` (see §4.6 — trimmed port).
5. **vLLM report parsing helpers** — a vLLM `RESULTS_COLUMNS`, `METRIC_TIERS` /
   `METRIC_TIER_ORDER`, `tier_metric_specs`. (Note: a `VLLM_SINGLE_RESULTS_COLUMNS`
   tuple *does* exist on the IX-atom line in `inference_suite_results_table.py`,
   but it's IX-atom suite code we are not porting; and there is no
   tier/`tier_metric_specs` helper for vLLM anywhere — IX-atom's preset imported
   those from `inferencex_atom_parsing`. We author fresh vLLM versions in §4.3.)
6. **`cvs/lib/report/presets/vllm.py`** — the new preset (§4.4).

### Must NOT be copied (IX-atom-specific; would break imports)
- `cvs/lib/report/presets/inferencex_atom.py`
- `cvs/lib/report/presets/inferencex_atom_single.py`
- `cvs/lib/report/scripts/generate_ix_atom_sample_report.py`
(all import `cvs.lib.inference.utils.inferencex_atom_parsing`, which we do not
port). Keep `presets/_inference_suite_template.py` (docs only).

**`presets/__init__.py` MUST be rewritten (BLOCKER).** The PR's
`cvs/lib/report/presets/__init__.py` is
`from cvs.lib.report.presets.inferencex_atom import INFERENCEX_ATOM_REPORT_CONFIG`.
After deleting `inferencex_atom*.py`, importing the `presets` package raises
`ModuleNotFoundError` — and `auto_register` imports
`cvs.lib.report.presets.vllm`, which imports the parent package first. So §4.1
must overwrite `presets/__init__.py` with a docstring-only module (no
`inferencex_atom` import). See §4.1.

### PR files intentionally NOT ported (IX-atom suite code, not report deps)
The PR touches 61 files; only the report **library** + its generic wiring are in
scope. The following PR-added/-changed files are IX-atom *suite* code and are
deliberately skipped — verified (§4.1) that the report engine does **not**
import them: `cvs/lib/inference/base.py`,
`cvs/lib/inference/inference_suite_results_table.py` (PR edits — we keep our
branch's absence; the report engine does not import it),
`cvs/lib/inference/utils/inferencex_atom_parsing.py`,
`cvs/lib/inference/utils/inferencex_atom_config_loader.py`, the
`inferencex_atom` tests/configs, and any `dtni/vllm_benchmark_scripts/*` moves.
If a post-copy import check (§4.1) reveals the engine *does* need one of these,
treat that as a finding and port the minimal piece.

---

## 4. Implementation steps

### 4.1 Port the report library
From this worktree (PR fetched as `FETCH_HEAD`):
```
git checkout FETCH_HEAD -- cvs/lib/report
git rm -q cvs/lib/report/presets/inferencex_atom.py \
          cvs/lib/report/presets/inferencex_atom_single.py \
          cvs/lib/report/scripts/generate_ix_atom_sample_report.py
```
Then **overwrite `cvs/lib/report/presets/__init__.py`** so it no longer imports
the deleted IX-atom preset (BLOCKER — see §3):
```python
'''Per-suite report presets.'''
```
(docstring only; the `vllm` preset is imported by `auto_register`, not re-exported
from `__init__`). Then verify the remaining package has **no** residual import of
`inferencex_atom` (`grep -rn inferencex_atom cvs/lib/report` must be empty
except possibly README prose). The engine itself (`inference*.py`,
`cell_build.py`, `builder.py`, `registry.py`, `auto_register.py`, `types.py`,
`pytest_extras.py`, `provenance.py`, `ci_summary.py`, `artifacts.py`,
`json_io.py`, `compare.py`, `formatting.py`, `metrics.py`, `panels/`,
`render/`, `viewer/`) is suite-agnostic.

**Dependency check after copy:** the engine imports
`cvs.lib.utils.verdict._check_one` (present, §3). Confirm no other engine module
imports something only on the IX-atom line. Run
`python -c "import cvs.lib.report.inference"` and resolve any ImportError.

### 4.2 Port `report_plugins.generate_suite_reports`
Insert the PR's `generate_suite_reports(self, session)` method into this
branch's `cvs/lib/report_plugins.py` (additive — the surrounding API already
matches). It calls `publish_inference_suite_report` (from §4.1) and, when
`htmlpath` is set, writes the CI summary via `write_inference_ci_summary` and
registers it with `add_html_to_report`. Also apply the cosmetic
`_build_reports_section` quoting change (optional, harmless).

### 4.3 Add vLLM report parsing helpers
Add to `cvs/lib/inference/utils/vllm_parsing.py` (co-located with the metric
vocab so all vLLM flavors share one definition):

- `VLLM_RESULTS_COLUMNS`: `(label, key)` tuples. **First 7 must be the fixed
  positional columns** `("Model", None), ("GPU", None), ("ISL", None),
  ("OSL", None), ("Policy", None), ("Conc", None), ("Host", None)` — because
  `inference_payload.build_results_table` emits `model, gpu, isl, osl, policy,
  conc, host` for the first 7 and only looks up `metric_keys[7:]` in the per-host
  dict. Then append `client.*` metric columns mirroring the existing console
  table in `_shared.py::test_print_results_table`
  (Req/s=`client.request_throughput`, Total tok/s=`client.total_token_throughput`,
  Mean/P95 TTFT, Mean/P95 TPOT, P99 ITL=`client.p99_itl_ms`,
  Goodput=`client.goodput`).
  > **Contract note:** the 7-fixed-column shape is a hard requirement of
  > `build_results_table`; a unit test must pin it (§4.7).
- `METRIC_TIERS: dict[str, tuple[str,...]]` with keys `throughput`, `ttft`,
  `tpot`, `health`; `METRIC_TIER_ORDER = tuple(METRIC_TIERS) + ("record",)`.
  **Seed tier membership from the existing `GATED_METRICS` set** so the report's
  gate matrix exactly mirrors what the suite enforces. Every name in
  `GATED_METRICS` must land in exactly one non-record tier (§4.7 unit test).
- `tier_metric_specs(thresholds_cell, tier) -> dict[str, dict]`: for a
  non-record tier, return `{f"client.{m}": thresholds_cell[f"client.{m}"]}` for
  each `m` in that tier that has a spec; for `"record"`, return the non-tiered
  metrics that have specs. Mirror IX-atom's helper shape (engine calls it as
  `config.tier_metric_specs(thresholds_cell, tier)` in `cell_build.tier_status`).

> **Tier-order invariant (MAJOR).** `cell_build.tier_status` iterates
> `config.metric_tier_order`; any tier name returned by `METRIC_TIERS` that is
> NOT in `metric_tier_order` silently drops those metrics from the gate matrix.
> So `set(METRIC_TIERS) ⊆ set(METRIC_TIER_ORDER)` MUST hold (it does by
> construction here, since `METRIC_TIER_ORDER = tuple(METRIC_TIERS) + ("record",)`)
> — pin it with the §4.7 unit test so a future tier edit can't break it.

### 4.4 Create the preset `cvs/lib/report/presets/vllm.py`
Filename MUST be `vllm.py`. Minimal version via the builder:
```python
from cvs.lib.inference.utils.vllm_parsing import (
    CLIENT_METRIC_UNITS, METRIC_TIER_ORDER, VLLM_RESULTS_COLUMNS, tier_metric_specs,
)
from cvs.lib.report.presets.builder import make_inference_report_config

VLLM_REPORT_CONFIG = make_inference_report_config(
    suite_id="vllm",
    report_basename="vllm_run_deck",
    title="vLLM Run Deck",
    results_columns=VLLM_RESULTS_COLUMNS,
    metric_units=CLIENT_METRIC_UNITS,
    tier_metric_specs=tier_metric_specs,
    metric_tier_order=METRIC_TIER_ORDER,
    inference_test_substring="test_vllm_inference",
    row_card_test_names=("test_metric",),
    # Lifecycle labels MUST match what the vLLM suite actually records (see below).
    session_lifecycle_labels=(
        "container_launch", "topology_discovery", "model_fetch",
        "server_ready", "teardown",
    ),
    cell_lifecycle_labels=("server_ready",),
)
```
`make_inference_report_config` forwards `**kwargs` to `InferenceReportConfig`, so
the explicit `session_lifecycle_labels` / `cell_lifecycle_labels` above override
the builder defaults.

> **Lifecycle-label reconciliation (MAJOR — the §2 contract table was
> optimistic).** The builder's default `_SESSION_LIFECYCLE` is
> `(container_launch, sshd_setup, model_fetch, server_ready, client_complete,
> teardown)` and default `_CELL_LIFECYCLE` is `(server_ready, client_complete)`.
> But the unified vLLM suite records (verified, `vllm.py`): `container_launch`,
> `topology_discovery`, `model_fetch`, `model_size` (unit `GB`, not `s`),
> `server_ready`, `teardown` — it does **not** record `sshd_setup` or
> `client_complete`. `aggregate_lifecycle`/`lifecycle_for_cell` keep only labels
> present with unit `s`, so with the builder defaults the timeline would silently
> (a) show empty `sshd_setup`/`client_complete` slots and (b) omit vLLM's real
> `topology_discovery`. Fixing the label sets in the preset (above) makes the
> timeline reflect reality. `model_size` is intentionally excluded (unit `GB`,
> filtered out by `aggregate_lifecycle`'s `unit != "s"` guard).
> **Optional richer timeline:** add a `client_complete` record in
> `test_vllm_inference` after the client finishes (mirrors IX-atom) — only if we
> also re-add it to `cell_lifecycle_labels`. Out of scope for the minimal port.

`auto_register._find_preset_in_module` picks up any `InferenceReportConfig` in
the module (prefers a `*_REPORT_CONFIG` name), so this single file is enough; no
`_single` shim is needed because the unified suite's stem is just `vllm`.

`inference_test_substring="test_vllm_inference"` and
`row_card_test_names=("test_metric",)` MUST match real suite test names
(verified present in `vllm.py:223,266`). These drive
`cell_build.resolve_pytest_nodeids_for_cell` and the row-card extras; a wrong
substring silently yields empty pytest links (not a crash), so pin via the live
run check in §5. `test_metric` carries `seq_combo` + `concurrency` funcargs
(verified) which `pytest_extras.attach_inference_cell_row_extra` requires — if
those funcargs were absent the cell-card extra would silently no-op.

**Chart/headline sanity (MAJOR).** The builder auto-derives `cell_highlights`
and `chart_series` from `results_columns`, and `InferenceReportConfig` defaults
`headline_metric`/`sweep_throughput_metric` to `client.output_throughput` and
`sweep_ttft_metric` to `client.mean_ttft_ms`. All three exist in vLLM's
`CLIENT_METRICS` (verified), so charts/sweep summaries populate. Confirm the
auto-derived highlights are sensible (throughput + latency) in the dry-render
(§5.4 dry-render); if not, pass explicit `cell_highlights`/`chart_series`.

### 4.5 Root `cvs/conftest.py` wiring
Apply the PR's conftest changes, **with three corrections to PR bugs — do not
copy the PR conftest verbatim:**

- refactor `pytest_configure` into `_ensure_html_report_manager(config)` +
  the existing manager-creation hook, **and add auto-register without defining a
  second `pytest_configure`.**
  > **PR BUG (MAJOR) — double `pytest_configure`.** The PR's `cvs/conftest.py`
  > defines `pytest_configure` **twice** at module scope (one `tryfirst`, one
  > `trylast`). At module scope the second binding *shadows* the first —
  > `vars(module)["pytest_configure"]` is only the `trylast` one, so the
  > manager-creation hookimpl never runs as `pytest_configure` (it's masked
  > today only because `_ensure_html_report_manager` is also called lazily from
  > `sessionstart`/`makereport`/`sessionfinish`). **Fix:** put both actions in a
  > single `pytest_configure` body (create manager first, then
  > `try_auto_register_inference_suite_report(config)`), OR give the second a
  > distinct hook (it doesn't need to be `pytest_configure` — call
  > auto-register from the one `pytest_configure`). Either way: exactly one
  > `pytest_configure`.
- add autouse module fixture `_cvs_inference_suite_report_session` that, when a
  preset is registered, pulls `inf_res_dict`/`variant_config`/`lifecycle` via
  `request.getfixturevalue` (catching `FixtureLookupError`) and calls
  `bind_session_results`;
- in `pytest_sessionstart`, **call `clear_session_results()`** before tests run
  (see §4.5a — session-global leak);
- extend `pytest_runtest_makereport` to call
  `attach_inference_suite_report_row_extra` when a preset is active, and
  **conditionally** `attach_lifecycle_html_table` (see double-table decision
  below);
- `pytest_sessionfinish` → wrap `mgr.generate_suite_reports(session)` in
  try/except (see §4.5b), then **always** call `mgr.create_zip_bundle(session)`.

#### Double-table decision (R3 — resolved, not deferred)
The vLLM *suite* conftest (`cvs/tests/inference/vllm/conftest.py`) ALREADY
defines a `hookwrapper=True` `pytest_runtest_makereport` that attaches a
lifecycle HTML table, plus `pytest_html_results_table_header/_row` that insert
the Value+Unit columns. The PR's root conftest *also* attaches a lifecycle table
in its (separate, also `hookwrapper`) `makereport`. Both wrappers run → **two
identical stage tables per row.** The column inserts are NOT duplicated (only the
suite conftest touches columns), so there is no header/row count mismatch — but
the duplicate table is a visible regression.
**Decision: the root conftest must NOT attach the lifecycle table for suites
whose own conftest already does.** Concretely, in the root
`pytest_runtest_makereport`, call only `attach_inference_suite_report_row_extra`
(the cell-card extra, which the suite conftest does NOT provide) and **omit**
`attach_lifecycle_html_table`. **Also drop the PR conftest's
`from cvs.lib.inference.inference_suite_lifecycle import attach_lifecycle_html_table`
import line** (it becomes unused → lint failure). The vLLM suite already renders
the lifecycle table itself, so nothing is lost. (If a future inference suite has
no per-test lifecycle table of its own, it can opt in — but that's out of scope
here. Since auto-register only fires when `presets.<stem>` exists and `vllm` is
the only preset, no other suite can activate this path today.) Confirm zero
duplicate tables on a real `--html` report (§5.5).

#### 4.5a Session-global state leak (MINOR→MAJOR)
`cvs/lib/report/registry._SESSION` is a module-global dict populated by
`bind_session_results` and never cleared automatically (`clear_session_results`
exists but no hook calls it). Across a multi-module / `--reruns` / xdist session,
a stale `inf_res_dict` could render into a later report. Mitigation: call
`clear_session_results()` from `pytest_sessionstart` so every session starts
clean. Low absolute risk (generation is gated by the active preset and re-reads
the live store) but cheap to make correct.

#### 4.5b `sessionfinish` must not let report errors kill the zip (MAJOR)
The PR's `pytest_sessionfinish` calls `generate_suite_reports(session)` then
`create_zip_bundle(session)` with **no** try/except. The render engine can raise
on malformed `inf_res_dict`/thresholds; an exception there would **skip
`create_zip_bundle` entirely**, losing the run's results archive. Wrap report
generation so it can never gate artifact bundling:
```python
from cvs.lib import globals as _cvs_globals  # module-level; target conftest has no `log`

@pytest.hookimpl(hookwrapper=True)
def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    yield
    mgr = _ensure_html_report_manager(session.config)
    try:
        mgr.generate_suite_reports(session)
    except Exception:
        _cvs_globals.log.exception("suite report generation failed; continuing to zip bundle")
    mgr.create_zip_bundle(session)
```
> **NOTE:** the current target `cvs/conftest.py` imports only
> `importlib.metadata, sys, Path, pytest, HtmlReportManager` — there is **no**
> module-level `log`. The guard's `except` branch must use an explicitly-bound
> logger (e.g. `cvs.lib.globals.log` as above, or
> `logging.getLogger(__name__)`), otherwise it raises `NameError` and defeats
> the guard. Render-only means render failures must be non-fatal to the run.

### 4.6 Trimmed `inference_suite_lifecycle.py`
Root conftest (and, transitively, `pytest_extras`) import from
`cvs.lib.inference.inference_suite_lifecycle`. The full PR module imports
`cvs.lib.inference.cache_probe.du_bytes` at module top-level, and
**`cache_probe.py` is absent on this branch.** Port a **trimmed** module that
imports cleanly. Keep-list (BLOCKER if incomplete):
- `InferenceLifecycle`
- `attach_lifecycle_html_table`
- `html_metric_table_header`, `html_metric_table_row`
- `sort_lifecycle_items`
- **`sweep_cell_result_key`** — REQUIRED. `cvs/lib/report/pytest_extras.py:8`
  does `from cvs.lib.inference.inference_suite_lifecycle import
  sweep_cell_result_key` (used at `:57`), and `pytest_extras` is reached via
  `inference_wiring.attach_inference_suite_report_row_extra` — which the root
  conftest calls. If omitted, importing `inference_wiring` raises ImportError
  and the entire report path dies at the first metric test. The function (PR
  module line 69) is a pure helper defined **before** the `cache_probe` import
  (line 81), so it ports without dragging in `du_bytes`.

Drop the stage tests (`test_launch_container`, etc.) and the `du_bytes` import —
the unified vLLM suite implements its own stage tests inline, so nothing
collects the module's stage tests. Verify with
`python -c "import cvs.lib.inference.inference_suite_lifecycle,
cvs.lib.report.inference_wiring, cvs.lib.report.pytest_extras"`.
(Alternative: port `cache_probe.py` too and move its import function-local;
trimming is lower-footprint. Either way the module must import cleanly.)

### 4.7 Unit tests
- Port `cvs/lib/report/unittests/` (in §4.1). Remove/skip any test that imports
  IX-atom specifics; check `grep -rn inferencex_atom cvs/lib/report/unittests`.
  These tests use the suite-agnostic builder/engine (no IX-atom fixtures), so
  they should run as-is once the package imports cleanly.
- Add `cvs/lib/inference/unittests/test_vllm_report_preset.py`:
  - `auto_register` returns `VLLM_REPORT_CONFIG` for stem `vllm`;
  - `VLLM_RESULTS_COLUMNS` first 7 entries are the fixed positional columns
    (`Model, GPU, ISL, OSL, Policy, Conc, Host` with `None` keys);
  - **`set(METRIC_TIERS) ⊆ set(METRIC_TIER_ORDER)`** and every `GATED_METRICS`
    member appears in exactly one non-record tier (no overlap, none orphaned);
  - the preset's `session_lifecycle_labels` are a subset of the labels the suite
    actually records (guard against future drift) — assert
    `set(VLLM_REPORT_CONFIG.session_lifecycle_labels) <= {"container_launch",
    "topology_discovery", "model_fetch", "server_ready", "teardown"}`;
  - `build_inference_report_payload` over a synthetic 2-cell `inf_res_dict`
    yields `schema_version==1`, `suite_id=="vllm"`, non-empty `cells`, and a
    gate matrix consistent with `enforce_thresholds`.
- Run `pytest cvs/lib/report/unittests cvs/lib/inference/unittests/test_vllm_report_preset.py -q`.

---

## 5. Verification gate (independent of test self-report)

Per CVS discipline, tests are **not** trustworthy oracles — verify independently.

1. **Import smoke (covers the two blockers + the trimmed module):**
   `python -c "import cvs.lib.report.inference, cvs.lib.report.presets,
   cvs.lib.report.presets.vllm, cvs.lib.report.inference_wiring,
   cvs.lib.report.pytest_extras, cvs.lib.inference.inference_suite_lifecycle"`
   resolves with no ImportError. (If `presets/__init__.py` still imports
   `inferencex_atom`, or the trimmed lifecycle module is missing
   `sweep_cell_result_key`, this fails here.)
2. **Build-order check.** After EACH commit (§7), `import cvs.conftest` and the
   smoke import above must succeed — no commit may leave the root conftest
   referencing a module not yet ported. In particular C3 (conftest wiring) must
   land **after** C2 (report package) + C4 (trimmed lifecycle); see §7 ordering.
3. **Offline unit:** §4.7 suite green.
4. **Dry render (no hardware):** synthesize `inf_res_dict` (2 ISL/OSL combos,
   ascending concurrency, ≥1 multi-host cell) + a minimal `variant_config`
   stub; call `cvs.lib.report.inference.write_report(...)` — the low-level
   writer (the session wrapper `publish_inference_suite_report` used by
   `generate_suite_reports` adds provenance + manager registration on top of it,
   and needs a live pytest config; `write_report` is the right entrypoint for an
   offline render); assert
   `vllm_run_deck.html`, `.json`, `_viewer.html` produced; JSON
   `schema_version==1`, `suite_id=="vllm"`, cells non-empty, results table
   header count == row width, gate matrix matches thresholds, **lifecycle
   timeline shows the vLLM labels (`container_launch`/`topology_discovery`/
   `model_fetch`/`server_ready`/`teardown`) and no empty `sshd_setup`/
   `client_complete` slots**, and auto-derived charts have a throughput + a
   latency series.
5. **Live single-node** (`cvs run vllm` only — no raw torchrun, always
   `--self-contained-html`):
   `cvs run vllm --cluster_file … --config_file <…vllm…single….json>
   --html report.html --self-contained-html`. Confirm:
   - `vllm_run_deck.{html,json}` + viewer beside `report.html`, linked in the
     **Reports** section;
   - JSON cells/lifecycle/results-table populated;
   - **exactly one** lifecycle stage table per test row (no duplicate from the
     root vs suite conftest — the R3 decision in §4.5);
   - results-table header cell count == each row's cell count (no column drift);
   - **pass/fail identical** to a pre-port / no-`--html` run (render is
     side-effect-free) — diff the pytest summary line;
   - a deliberately-broken render (e.g. temporarily feed malformed thresholds)
     still produces the results zip (validates the §4.5b guard);
   - Surface all log paths (discipline §run-discipline).
6. **Live distributed** (the point of this branch): 2-node PP config
   (`mi300x_vllm…distributed…json`) with `--html`. Confirm:
   - carried `vllm_job.py` fixes hold (head logs startup; workers `--headless`,
     don't block readiness);
   - cell keys carry `PP=`; results table + gate matrix render per-host rows for
     multi-host `inf_res_dict` values (guards against thin/empty multi-host cells
     if readiness reports early — R7).
7. **Independent log-scan:** grep run logs for `Traceback`, `Error`, `assert`,
   teardown-failure signatures even when pytest reports PASS.

---

## 6. Risks & open questions

- **R1 — run stem coverage.** Preset `vllm.py` matches only `cvs run vllm`.
  `cvs/tests/inference/vllm/vllm_single.py` and
  `cvs/tests/inference/vllm_distributed/vllm_distributed.py` still exist; if
  those stems are still invoked, they get no report unless we add
  `presets/vllm_single.py` / `presets/vllm_distributed.py`. **Decision needed:**
  is the unified `vllm` the only supported entrypoint now? If legacy files are
  dead, consider deleting them (separate change) to avoid confusion.
- **R2 — tier/threshold fidelity.** The report gate matrix must mirror real
  enforcement. Seed `METRIC_TIERS` from `GATED_METRICS` and unit-test the
  partition (every gated metric in exactly one tier; no record-tier overlap).
- **R3 — duplicate HTML extras. RESOLVED in §4.5** (root conftest omits
  `attach_lifecycle_html_table`; suite conftest owns the lifecycle table). Listed
  here only as the highest-likelihood regression to re-confirm on a real report
  (§5.5). Column inserts are owned solely by the suite conftest → no count drift.
- **R4 — zip inclusion.** PR description says reports are bundled into the
  results zip, but the PR's `create_zip_bundle` is **byte-identical** to this
  branch's (verified): it zips only the main HTML, `assets/`, and the per-test
  log dir. `generate_suite_reports` writes the run-deck `.html/.json/_viewer.html`
  to `htmlpath.parent` and only *links* the CI summary via `add_html_to_report`
  — the run-deck artifacts are **not** in the zip. **Decision needed:** if the
  artifacts must live inside the results zip, extend `create_zip_bundle` to also
  add each `self._custom_test_reports` path (and the JSON/viewer). If "beside the
  HTML report + linked" is acceptable, no change. Flagged because the PR
  description over-claims; do not assume the zip contains them.
- **R5 — `_suite_name` edge case.** `_suite_name` defaults to `"test"` when no
  `*.py` arg is found; `cvs run` always passes the file, so stem is `vllm`. But
  if anyone runs `pytest cvs/tests/inference/vllm/` (directory, no file), the
  preset won't auto-register. Acceptable (out of the `cvs run` path); note it.
- **R6 — drift from main (59 commits behind).** Out of scope here, but the
  eventual upstream PR targets `main`; keep the port additive/isolated to ease a
  later rebase. The report package is self-contained under `cvs/lib/report/`,
  which minimizes conflict surface.
- **R7 — thin multi-host cells if readiness reports early.** The carried
  `vllm_job.py` fix changes distributed readiness to head-rank-only. Reports read
  `inf_res_dict`/`lifecycle` *after* tests complete, so there is no functional
  coupling — but if a distributed run is declared ready prematurely and a cell's
  per-host actuals come back partial/empty, the report renders thin cells (not a
  crash; `build_results_table` only looks up `metric_keys[7:]`). Covered by the
  §5.6 per-host-rows assertion.
- **R8 — faithfully-copied PR bugs. HANDLED in §4.5** (do not copy verbatim):
  (a) double `pytest_configure` at module scope — collapse to one;
  (b) unguarded `generate_suite_reports` in `sessionfinish` — wrap so it can't
  abort the zip;
  (c) uncleared session-global `registry._SESSION` — clear at `sessionstart`.
  These are real defects in PR #244's conftest/wiring; the integration must fix
  them rather than inherit them. (Consider also flagging (a)/(b) upstream on the
  PR.)

---

## 7. Deliverable / commit plan

**Ordering is load-bearing** (each commit must leave the tree importable — §5.2):
the root conftest (C4) references the report package, the trimmed lifecycle
module, and the vLLM preset, so all three must exist first.

- `7b25aaf` carry `vllm_job.py` distributed fixes ✅ (done)
- C2: port `cvs/lib/report/` (minus IX-atom files) **+ rewrite
  `presets/__init__.py`** (§4.1 blocker). After C2, `import cvs.lib.report.inference`
  works but `presets.vllm` not yet present.
- C3: trimmed `cvs/lib/inference/inference_suite_lifecycle.py` **including
  `sweep_cell_result_key`** (§4.6 blocker). After C3,
  `import cvs.lib.report.inference_wiring` / `pytest_extras` works.
- C4: vLLM report helpers (`VLLM_RESULTS_COLUMNS`, `METRIC_TIERS`,
  `METRIC_TIER_ORDER`, `tier_metric_specs`) in `vllm_parsing.py` **+**
  `cvs/lib/report/presets/vllm.py`. After C4, `import cvs.lib.report.presets.vllm`
  works → auto-register has a target.
- C5: `report_plugins.generate_suite_reports` **+** root `cvs/conftest.py` wiring
  with the three R8 fixes (single `pytest_configure`, guarded `sessionfinish`,
  `clear_session_results` at `sessionstart`) and the R3 decision (no root
  lifecycle table). This is the only edit to a shared file every suite loads;
  keep all of it in one commit so a revert is atomic.
- C6: unit tests (`cvs/lib/report/unittests/` port + `test_vllm_report_preset.py`).

(Commit boundaries may merge C2–C4 if preferred, but C5 must come last among the
wiring commits and must not precede the modules it imports.)

Net: `cvs run vllm … --html` produces an IX Run Deck-style dashboard for vLLM
single-node and distributed runs; **all gates and thresholds unchanged**, and
non-inference suites are provably unaffected (gated wiring + guarded sessionfinish).
