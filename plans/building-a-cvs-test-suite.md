**Building a CVS test suite — a reference guide**

This guide documents the base infrastructure for writing a new **inference** or
**training** test suite in CVS, using the `vllm_single` suite (PR against
`dev/dtni`) as the worked reference implementation. If you are adding a serving
framework, a training benchmark, or any new workload, base your structure on the
patterns here rather than on the older `InferenceBaseJob` / `if_dict` style.

> **Status note.** CVS is experimental. Flags and config keys change, and the
> built-in pass/fail is not a trustworthy oracle on its own — always verify a
> run's artifacts independently. This guide describes the structure, not a frozen
> API.

> **Training is not ported yet.** Only `vllm_single` (inference) exists today.
> The library layout, the generic↔domain seam, and the suite skeleton below are
> deliberately framework-neutral so a training suite drops into the same shape —
> this guide is the blueprint for that port, not a record of it.

### What changed in the restructure

This PR reorganizes where shared vs. workload-specific code lives, so the next
suite reuses machinery instead of copy-pasting it.

**`cvs/lib/dtni` → `cvs/lib/utils` (renamed).** `dtni` was a leftover project
codename; the directory actually holds pure, framework-agnostic helpers. The new
name says what it is: utilities any suite (inference, training, …) can call.

**`cvs/lib/utils/` — the shared/common layer.** Everything here is generic: no
suite knows about vLLM, Megatron, ISL/OSL, or goodput. It holds:
- `config_loader.py` — the generic config skeleton (`BaseVariantConfig`), the
  `paths`/`model`/`image`/`container` schema, the 3-pass placeholder
  substitution engine, and `substitute_config()` (read a config + its sibling
  threshold file, resolve placeholders).
- `verdict.py` — `evaluate_all(actuals, thresholds)`, a metric-name-agnostic
  threshold checker (`min`, `max_ms`, `within`, `min_tok_s`, `min_ratio`).

**`cvs/lib/inference/utils/` — the domain-specific layer.** Helpers only an
*inference* suite needs live here, one level down from the shared dir. It holds:
- `inferencing_config_loader.py` — the inference config schema: the named-combo
  sweep selector, `GoodputSlo`, the vLLM bench `Params`, and
  `VariantConfig(BaseVariantConfig)` with the ISL/OSL/TP/CONC `cell_key`. It
  *imports* `BaseVariantConfig` + `substitute_config` from `cvs/lib/utils` rather
  than re-implementing them.
- `vllm_parsing.py` — the `client.*` metric vocabulary (`to_client_metrics`,
  `CLIENT_METRICS`): pure transforms from a vLLM benchmark artifact to a
  namespaced metric dict.

**The rule for where a helper goes.** If every workload could use it (config
plumbing, threshold math) → `cvs/lib/utils/`. If it only makes sense for one
domain (sweep shapes, serving metrics) → `cvs/lib/<domain>/utils/`. A training
port adds `cvs/lib/training/utils/` the same way inference did — subclass
`BaseVariantConfig`, reuse `substitute_config` and `evaluate_all`, add its own
schema and metric vocabulary. The shared layer never grows a dependency on a
specific framework.

```
cvs/lib/
  utils/                         # shared, framework-agnostic (was: dtni)
    config_loader.py             #   BaseVariantConfig, substitute_config, placeholders
    verdict.py                   #   evaluate_all (generic threshold kinds)
  inference/
    utils/                       # inference-only helpers
      inferencing_config_loader.py   #   sweep selector, Params, VariantConfig, cell_key
      vllm_parsing.py            #   client.* metrics, to_client_metrics
    vllm_orch.py                 #   the driver (VllmJob)
  training/                      # (future) same shape:
    utils/                       #   training_config_loader.py, <metric>_parsing.py
```

### The layered architecture

A suite is built from six layers. Each has one job; the seams between them are
the reusable surface a second suite plugs into.

```
  cluster file (node IP + user/key/orchestrator)   <- per-environment, OUT of repo
        |
  variant config (*_config.json + *threshold.json) <- per-workload, IN repo
        |
  config loader  ── cvs/lib/utils/config_loader.py                  (generic: schema, substitution)
        |          └ cvs/lib/inference/utils/inferencing_config_loader.py (domain schema + sweep)
        |
  orchestrator   ── cvs/core/orchestrators (ContainerOrchestrator; provided)
        |
  job / driver   ── cvs/lib/inference/vllm_orch.py   (launch server, run client, fetch artifact)
        |
  suite / pytest ── cvs/tests/inference/vllm/        (lifecycle-as-tests, parametrization, HTML)
        |
  parsing + verdict ── vllm_parsing.to_client_metrics + cvs/lib/utils/verdict.evaluate_all
```

The **generic↔domain seam** is the key idea: anything every workload shares lives
in `cvs/lib/utils/`; anything specific to *your* workload lives in your own
`cvs/lib/<domain>/utils/` module that subclasses/reuses it. The config loader
docstring literally calls this the "generalization seam," and this PR executes
the split.

### The two config files (per workload)

A workload is described by a pair of JSON files in the same directory:

- `*_config.json` — the variant: paths, model, container (with `container.image`),
  params, sweep.
- `*threshold.json` — the per-cell pass/fail thresholds.

The loader finds the threshold by **sibling glob** (exactly one `*threshold.json`
next to the config), so the two share a descriptive prefix
(`llama31_70b_fp8_config.json` / `llama31_70b_fp8_threshold.json`).

#### Placeholders

Config values use `{...}` placeholders resolved in three passes:
`{user-id}` (from the cluster file) → `{shared_fs}` (self-reference within
`paths`) → `{paths.models_dir}` (cross-block). An unknown placeholder is left
literal — there is no error, so check for a stray brace in a resolved path if
something doesn't mount.

#### enforce_thresholds

`enforce_thresholds: false` makes the suite **record-only**: it captures every
metric and skips assertions, and a threshold/sweep mismatch warns instead of
failing the load. Use it for un-calibrated workloads (e.g. throughput
characterization where the published numbers are curves, not tabulated cells).
Flip to `true` once you have real numbers — the coverage check then guarantees
both that every sweep cell has a threshold entry AND that every cell carries a
spec for every gated metric (`GATED_METRICS`), so a green run can't have
silently skipped its verdict.

### The sweep selector (named combos + runs)

The sweep enumerates exactly the `(sequence-shape, concurrency)` cells to run. It
replaces the old `sequence_combinations × concurrency_levels` cartesian:

```json
"sweep": {
  "sequence_combinations": [
    { "name": "w1_isl=128_osl=2048", "isl": "128", "osl": "2048",
      "goodput_slo": { "ttft_ms": 1000000000.0, "tpot_ms": 1000000000.0, "e2el_ms": 1000000000.0 } }
  ],
  "runs": [
    { "combo": "w1_isl=128_osl=2048", "concurrency": 16 }
  ]
}
```

- Combos are named once; `runs` cherry-picks `(combo, concurrency)` pairs. No NxM
  explosion — you list precisely the cells you want.
- Load-time validation rejects duplicate combo names and any `run.combo` that
  names no combo. (This is mirrored in the suite's `pytest_generate_tests`, which
  reads raw JSON at collection time before the typed loader runs.)
- `goodput_slo` is an **input** to the run (passed to `vllm bench serve
  --goodput`), not a threshold — that's why it lives in the sweep.

Each cell's threshold key is produced by `VariantConfig.cell_key()`
(`ISL=…,OSL=…,TP=…,CONC=…`). It is the single source of truth shared by the
coverage check and the verdict lookup, so threshold.json keys must match it
exactly.

### The job/driver (self-contained, no external .sh)

`vllm_orch.VllmJob` is the reference driver. It talks only to an injected
orchestrator (`orch.exec`, which routes into the running container) and a typed
`VariantConfig`. Lifecycle highlights to copy:

- **The server command is built in Python** (`_server_argv`), not cloned from an
  external `.sh` repo. A run is self-contained. Per-model quirks come from
  `roles.server.serve_args` (a `{flag: value}` map) / `roles.server.env` in
  config.
- **`/tmp/server_env_script.sh`** is written by `build_server_cmd` and **sourced
  by both server and client**, so the two share one environment (HF token, cache
  pin, AITER flags). Each value is `shlex.quote`d.
- **`--max-model-len` is derived per cell** from isl/osl/random_range_ratio so a
  sweep change stays self-consistent.
- **Readiness/completion are detected by scanning the whole log**, with narrow
  failure markers (don't match bare `error:` — ROCm/vLLM logs benign ones).
- **Results come from the stock `results` artifact**, not console-regex.
  `parse_results` fetches the extensionless JSON `vllm bench serve` writes to
  `--result-dir` and hands it to the pure `to_client_metrics`. Missing/empty/
  unparseable → hard-fail the cell (never a silently-green empty row).

The fetch lives in the job (artifact layout is job-specific); the transform lives
in `inference/utils` (so distributed/disagg/InferenceMax reuse it).

### The suite (lifecycle-as-tests)

In `cvs/tests/inference/vllm/`, each lifecycle stage is its own pytest test so it
shows up as a timed, pass/fail row in the HTML report:

```
test_launch_container → test_setup_sshd → test_model_fetch
  → test_vllm_inference (per cell) → test_metric (per metric per cell)
  → test_print_results_table → test_teardown
```

Patterns to copy:

- **`pytest_generate_tests`** (in the suite module, not conftest) parametrizes
  from the sweep selector. It runs at collection time and reads raw JSON, so it
  re-validates combos by hand (mirroring the typed loader).
- **`test_vllm_inference` runs the benchmark once per cell** and stashes results
  in a module-scoped `inf_res_dict`. **`test_metric` reads one cached metric** and
  is one HTML row per metric per cell — no GPU work, asserts only when
  `enforce_thresholds` is true and a spec exists.
- **`_Lifecycle`** (`conftest`) carries cross-test state: `failed` lets a broken
  stage skip the rest instead of cascading; `torn_down` lets explicit teardown
  suppress the fixture leak-guard finalizer.
- **The `orch` fixture owns only the teardown safety net** — launch/sshd happen
  in tests so they're timed rows. A mid-sweep failure still tears the container
  down via the finalizer.
- **Single-node guards**: `test_setup_sshd` only probes port 2224 when
  `len(orch.hosts) > 1` (in-container sshd exists only for inter-node MPI).
- **HTML Value/Unit columns** come from `pytest_html_results_table_header` /
  `_row` hooks scoped to this conftest, populated from
  `metric_value`/`metric_unit` user-properties.

### Parsing + verdict

- **`to_client_metrics(raw, *, tp, isl)`** — pure: stock keys namespaced
  `client.*` 1:1, plus derived metrics (`per_gpu_throughput`,
  `decode_throughput_p50`, `success_rate`, …) via `_safe_div` (degrades to
  `None`, never crashes). `CLIENT_METRICS` is the ordered display surface.
- **`evaluate_all(actuals, thresholds)`** — generic, framework-neutral. Kinds:
  `min`, `max`, `max_ms`, `within`, `min_tok_s`, `min_ratio`. Raises
  `ThresholdViolation` listing every failure. A `None` actual is a loud
  violation, not a TypeError.
- **`GATED_METRICS`** (in `vllm_parsing`) — the asserted SLO subset of
  `CLIENT_METRICS`. The loader's coverage check requires a threshold spec for
  every gated metric in every present cell, so a gated metric can't silently
  fall through to a zero-assertion record-only row. A new metric is record-only
  until added to the set.

### To add your own suite — checklist

1. **Schema.** If serving: reuse `inferencing_config_loader` (subclass `Params`
   for a new framework's flags). If a new domain (training): create
   `cvs/lib/<domain>/utils/<domain>_config_loader.py`, subclass
   `BaseVariantConfig`, reuse `substitute_config` + `evaluate_all` from
   `cvs/lib/utils/`. Define your own `cell_key` + coverage check.
2. **Config pair.** Write `*_config.json` + `*threshold.json` under
   `cvs/input/config_file/<domain>/<suite>/<variant>/`. Start
   `enforce_thresholds: false` until calibrated.
3. **Driver.** Write a `Job` class that takes an `orch` + typed config and owns
   launch → run → fetch-artifact → `parse`. Build commands in Python; keep the
   pure transform in your `utils`. Hard-fail on missing artifacts.
4. **Suite.** Under `cvs/tests/<domain>/<suite>/`: `conftest.py` (fixtures +
   lifecycle + HTML hooks), the suite module (`pytest_generate_tests` +
   lifecycle-as-tests), `_shared.py` (the results table). Pin test order in
   `pytest_collection_modifyitems`.
5. **Metrics.** Define your metric vocabulary + units list once in your `utils`,
   the way `CLIENT_METRICS` does — don't re-list rows per suite.
6. **Cluster file.** Keep it minimal and OUT of the repo: node IP + user + key +
   orchestrator. The variant config supplies the container block; don't ship a
   bespoke per-suite cluster file.
7. **Verify independently.** After a run, read the artifact + the HTML cells
   yourself. CVS's PASS is not a trustworthy oracle.

### Reference files (this PR)

| Layer | File |
|---|---|
| generic schema + substitution | `cvs/lib/utils/config_loader.py` |
| generic verdict | `cvs/lib/utils/verdict.py` |
| inference schema + sweep | `cvs/lib/inference/utils/inferencing_config_loader.py` |
| client metric vocabulary | `cvs/lib/inference/utils/vllm_parsing.py` |
| driver | `cvs/lib/inference/vllm_orch.py` |
| suite | `cvs/tests/inference/vllm/{vllm_single,conftest,_shared}.py` |
| config pair | `cvs/input/config_file/inference/vllm_single/w1_llama31_70b_fp8kv/` |

Agent-facing summaries (entry points + gotchas) live in each package's
`AGENTS.md`.
