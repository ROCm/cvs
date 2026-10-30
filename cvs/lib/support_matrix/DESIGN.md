# HW/SW Support-Matrix Gate — Design Plan

Status: proposal (design only — no implementation yet); decisions in §12 locked to recommended defaults; amended per the filename-convention review (see §2.1)
Branch: `solaiys/dtni-hwsw-support-matrix-check`
Scope: Inference and Training workload suites run through CVS.

## 1. Problem & goals

Users running Inference/Training workloads want CVS to **verify hardware and
software compatibility before a workload runs**, and to **stop immediately with
a clear tag** when the workload is not supported on the detected platform.

Concrete examples from the request:
- MXFP4 is not supported on MI300/MI325 → must not be allowed to run.
- Latest models will not run on older ROCm releases → must fail fast.

Two independent checks:
- **Check-A (HW / GPU):** is the GPU present on the execution nodes supported for
  this workload?
- **Check-B (SW / ROCm):** is the ROCm version — *where the workload actually
  runs* (inside the container for the docker orchestrator) — new enough for this
  workload?

Goals:
- One **generic, shared library** that gates both suites; the library is **never
  edited** to add/remove a supported or unsupported combination (design req #8).
- Declarative data drives the decision; adding coverage = editing data, not code.
- Non-breaking: existing configs keep working; the gate is opt-in per config and
  can be globally overridden.
- Runs **after container launch** for the docker orchestrator (design req #7).

Non-goals:
- Measuring correctness/performance (handled by the existing suites).
- Auto-discovering support from the tracker spreadsheet at runtime (the tracker
  is the human source of truth; its rows are transcribed into config data).

## 2. What already exists (reuse, don't reinvent)

- GPU arch from `rocm-smi`: `get_model_from_rocm_smi_output()` →
  `mi300x | mi325 | mi350 | mi355` (handles the MI350/MI355 marketing-name
  omission via Device ID). `cvs/lib/utils_lib.py`.
- ROCm version from `amd-smi version --json` → `rocm_version`, already wrapped as
  a preflight check: `RocmVersionCheck` in `cvs/lib/preflight/version_check.py`
  (has a reusable `_extract_rocm_version`).
- Inference suites already detect the GPU post-launch through the orchestrator:
  `gpu_type` fixture in `cvs/tests/inference/sglang/conftest.py`
  (`orch.all.exec("rocm-smi -a ...")` → `get_model_from_rocm_smi_output`); same in
  `cvs/tests/inference/xdit/conftest.py`. RCCL has `_detect_gpu_arch()`.
- Training configs already carry a structured `gpu_name` validated against an
  allowed set (`_ALLOWED_GPU_NAMES` in
  `cvs/lib/training/megatron/utils/training_config_loader.py`; torchtitan and
  jaxmaxtext loaders have `gpu_name` too).
- Suites follow a **lifecycle-as-tests** model: a module-scoped `orch` fixture
  plus ordered stage tests, where the container is launched by a *stage* and
  later stages `pytest.skip` once `lifecycle.failed` is set. Order is pinned by
  each suite's `pytest_collection_modifyitems` (e.g. torchtitan "launch → …").
- Config validation is pydantic-based and "fails fast" (`cvs/parsers/schemas.py`),
  and `resolve_test_config_placeholders()` (`cvs/lib/utils_lib.py`) is the shared
  early entry most suites already call (it hard-exits on unresolved `<changeme>`).

Gaps: there is no declared `(supported arch, ROCm range)` data attached to a
workload, and there is no single gate that stops an unsupported run. Precision
(MXFP4/FP8) is **not a structured field** in any suite today — it appears in some
config *filenames* by convention but may be absent, so it cannot be relied on.

## 2.1 Config filenames are not a reliable signal (researched)

The gate **must not parse the config filename** to learn the framework, GPU,
model, or precision. Findings from the repo:

- Filenames are **free-form**. Nothing in the loaders, the `cvs config`
  discovery, or the pytest metadata parses them for semantics; `cvs config
  copy/list` and the HTML metadata use the name only for display.
- The **one exception** is ATOM: `gpu_arch_from_config_path()`
  (`cvs/lib/inference/atom/atom_config_loader.py`) requires `{gpu}_atom_*.json`,
  but **only as a fallback** when the config omits an explicit `gpu_arch` field.
  Every other suite ignores the filename.
- The docs describe naming *patterns* (`mi3xx_sglang_`, `mi3xx_vllm_…_fp8_…`) as a
  human identification convention, not a contract the code enforces.
- The authoritative, structured signals that already exist are: training configs'
  `gpu_name` field, ATOM's `gpu_arch` field, and **runtime detection** of the GPU
  (`rocm-smi`) and ROCm (`amd-smi` / `/opt/rocm/.info/version`).

Consequence for this design: the gate's decision depends only on (a) the
per-config `support_matrix` block the author writes and (b) values detected at
runtime. The report tag is built from the suite name plus the detected arch/ROCm
(and the block's `notes`); no framework/model/precision string is read from the
filename or required for the gate to work.

## 3. The coverage data (from the DTNI Validation Tracker)

Per the JAX/SGLang/vLLM/Megatron/TorchTitan/XDIT **workload/model-coverage**
sheets (matrix sheets ignored), each row is a workload identified by
`model + precision (+ TP/seqlen)`, with per-GPU support in the coverage column:

- JAX `Llama-3.1-8B | Precision=BF16` → `MI300X: Batch=4 | MI325X: Batch=10 | MI355X: Batch=12` (all supported).
- JAX `Llama-3.1-8B | Precision=MXFP4` → `MI300X: — | MI325X: — | MI355X: Batch=6` (**only MI355X**).
- SGLang `GPT-oss-120b … TP=4(MXP4)`, `Kimi-K2.5-MXFP4 … TP=4`, etc.

Takeaways that drive the data model:
- The **unit of support is a workload** (framework + model + precision), which is
  exactly what one CVS config file represents. The workload identity lives in the
  config's *contents* and its author-declared `support_matrix` block — **not** in
  the filename, which is free-form (see §2.1). A name like
  `mi3xx_vllm_kimi-k27-code_mxfp4_single.json` is a human convention only.
- `—` / `-` in a GPU column means **unsupported** on that arch.
- So each config's supported arch set = the GPUs with a non-`—` entry in its
  tracker row; MXFP4 workloads carry `required_rocm_min_version` from the ROCm
  the MXFP4 path needs.

## 4. Data model — per-config `support_matrix` block (recommended)

Add an optional, non-breaking block to each workload config:

```json
"support_matrix": {
  "enforce": true,
  "gpu": {
    "supported_arch": ["mi355x"],
    "unsupported_arch": []
  },
  "required_rocm_min_version": "7.0",
  "required_rocm_max_version": null,
  "notes": "MXFP4 requires gfx950 (MI350/MI355); unsupported on MI300/MI325."
}
```

Semantics:
- `enforce` (default `true`): on violation the workload is tagged and stopped.
  `false` → violations are logged as warnings and the run continues.
- `gpu.supported_arch` (allow-list): detected arch must be in it. **Absent →
  Check-A skipped**, and the detected GPU is reported as a warning that the
  support list was not specified (mirrors the ROCm rule below).
- `gpu.unsupported_arch` (deny-list): optional negative form that maps directly to
  the tracker's `—` cells; detected arch must not be in it. `supported_arch` and
  `unsupported_arch` are mutually exclusive per config.
- `required_rocm_min_version` (Check-B): detected ROCm (inside the container) must
  be `>=` this. **Absent → Check-B skipped**, and the detected ROCm is reported as
  a warning that `required_rocm_min_version` was not specified (design req #4).
- `required_rocm_max_version` (optional): upper bound for deprecations.

Arch tokens are canonical (`mi300x, mi325x, mi350x, mi355x`) with an alias/gfx
table (`mi325`→`mi325x`, `mi350x/mi355x`→`gfx950`, `mi300x/mi325x`→`gfx942`) kept
as **shipped data** (see §6) so authors can write either a model name or a gfx
family, and so a new GPU generation is a data edit, not a code edit.

A pydantic `SupportMatrix` model (in `cvs/parsers/schemas.py`, consistent with the
existing "single source of truth") validates the block early and rejects unknown
arch tokens / malformed versions with a clear path, exactly like other configs.

## 4.1 Future: precision / sweep-param support per arch

Precision (MXFP4 / FP8 / BF16) is becoming a **sweep parameter** in the training
configs, not a top-level or filename token. A later iteration will extend
`support_matrix` so support can be declared **per sweep-param value per arch** —
e.g. BF16 on mi300x/mi325x/mi355x but MXFP4 only on mi355x — mirroring the tracker
rows where batch/precision vary per GPU column.

The v1 block above gates at the **config (workload) granularity**. The extension
adds an optional per-sweep-cell map (keyed by the sweep key, or by a param such as
`precision`) whose entries carry their own `supported_arch` /
`required_rocm_min_version`, evaluated for the specific cell about to run. This
stays data-driven and backward compatible: configs without the per-cell map keep
the config-level behaviour, and the library still reads data only (req #8).

## 5. Check-A vs the two proposed methods (answering design req #6)

**Recommendation: Method 1 (per-config block) as the primary mechanism.**

- A config already is one workload (framework+model+precision+topology); the
  support declaration belongs with it, co-located and bounded at **O(configs)**.
- Non-breaking and incremental: unannotated configs simply skip the checks with a
  warning; annotate as the tracker is transcribed.
- The tracker's per-GPU coverage maps 1:1 onto `supported_arch` for that row.

**Method 2 (one central `support_matrix.json` of unsupported combos) — not
recommended as primary.** It grows **O(models × precisions × archs)**, becomes a
*second* source of truth that drifts from both the configs and the tracker, and
(as you suspected) becomes unmanageable at scale. It also couples unrelated
workloads into one file with merge-conflict hot-spots.

**Optional future augmentation (better than either alone): a small capability
*rules* file** — not an enumerated combo list — for cross-cutting hardware facts,
e.g. `precision: mxfp4 → requires gfx950, rocm_min 7.0`. A config then only
declares `precision: mxfp4` and the library merges the intrinsic rule. This keeps
the common constraint DRY (a handful of rules) while the per-config block handles
the rest. Start with the per-config block; add the rules file only if duplication
becomes real. Either way the **library reads data and never changes** (req #8).

## 6. Generic library — `cvs/lib/support_matrix/`

No embedded mappings. Everything the gate decides on comes from the config block
(and optional rules/alias data). Proposed layout:

```
cvs/lib/support_matrix/
  __init__.py
  schema.py          # SupportMatrix pydantic model (or re-export from parsers/schemas.py)
  version.py         # parse_rocm_version(), compare() — tolerant of "7.0", "6.4.0-123"
  arch.py            # normalize_arch(), gfx_family(); table loaded from data/arch_aliases.json
  detect.py          # detect_gpu_arch(handle), detect_rocm_version(handle) via .exec
  evaluate.py        # evaluate(support_matrix, platform) -> Result(violations[], warnings[])
  gate.py            # run_gate(handle, config_dict, context, report=None) -> Result; enforce/skip/warn
  data/
    arch_aliases.json   # canonical arch <-> alias/gfx map (data-driven)
  unittests/
    test_version.py test_arch.py test_detect.py test_evaluate.py test_gate.py
```

Public API (stable, data-agnostic):
- `detect_platform(handle_host, handle_container) -> Platform{gpu_arch, gfx, rocm_version}`.
- `evaluate(support_matrix, platform) -> Result` — pure function, trivially unit-tested.
- `run_gate(handle, config_dict, context, report=None)` — the one call a suite
  makes; reads `config_dict["support_matrix"]`, detects, evaluates, applies the
  enforce/warn/skip policy, emits the tag, and returns the `Result`.

Guarantee (req #8): adding/removing support = editing a config's `support_matrix`
block (or the optional rules/alias data). `cvs/lib/support_matrix/*.py` is never
touched for a new mapping.

## 7. Detection — where and how (design req #7)

Both checks run **after container launch** for the docker orchestrator, using the
same `orch` handle the suites already have:

- **Check-A (GPU arch):** the GPU is physical, so detection can run on the node
  (`orch.all.exec("rocm-smi -a | head -30")`, as the sglang `gpu_type` fixture
  already does) or inside the container; parsed by the existing
  `get_model_from_rocm_smi_output()`.
- **Check-B (ROCm):** must reflect the **container's** ROCm, so it runs
  **inside** the container (`orch.exec(...)` routes into it post-launch). Probe
  order for robustness:
  1. `cat /opt/rocm/.info/version` (present in ROCm images, e.g. `6.4.0-123`),
  2. fallback `amd-smi version --json` → `rocm_version` (reuse
     `RocmVersionCheck._extract_rocm_version`),
  3. fallback `hipconfig --version`.
  Optionally also report the **host** ROCm for context.

Version compare: parse `major.minor[.patch][-build]` into a tuple, ignore build
suffix for ordering; `"7.0" >= "7.0"` and `"6.4.1" < "7.0"`. No new deps
required (small local parser; `packaging` only if already vendored).

All detection takes a handle exposing `.exec()`, so it works with the `orch`
fixture (new suites) and with legacy `phdl` (older inference paths) without
change, and is unit-tested by mocking `.exec()` outputs (no live cluster).

## 8. Placement in each suite (design req #2)

Uniform rule: **one early lifecycle stage** `test_support_matrix`, pinned by the
suite's existing `pytest_collection_modifyitems` to run **immediately after the
container-launch stage and before the first workload stage**. It calls
`support_matrix.gate.run_gate(orch, config_dict, context)`.

Why a stage (not an autouse fixture): in the lifecycle-as-tests model the
container is brought up by a *stage*, so "after container launch" is naturally a
stage ordering; a stage is also a visible report row (the "tag"), and failing it
sets `lifecycle.failed`, which the existing guard uses to `pytest.skip` every
subsequent workload stage — i.e. the workload is cleanly stopped.

Concrete hooks:
- Training (orch + lifecycle): add the stage via the shared `_common.py`
  (jaxmaxtext already centralizes stages there) and the megatron/torchtitan stage
  files; pin order right after `launch`.
- Inference sglang/vLLM (orch + lifecycle): add the stage after the container-up
  stage and before server launch/benchmark; reuse the existing `gpu_type`
  detection, add the in-container ROCm probe.
- ATOM / xDiT: same stage in their conftest/stage modules.
- Any legacy `phdl` inference path using `InferenceBaseJob` directly: call
  `run_gate(phdl, if_dict, context)` right after `launch_docker_container()` and
  before `launch_server()`.

Each suite change is a thin 2–3 line stage that delegates to the shared lib.

## 9. Enforcement, tagging, and the escape hatch

- Policy precedence: global override → per-config `enforce` → default (enforce).
  Global override via CLI `--support-matrix={enforce|warn|off}` (and/or env
  `CVS_SUPPORT_MATRIX`), so a user can disable gating and "just print the warning
  and continue" (design req #6). Default = `enforce`.
- On **violation + enforce**: log a distinctive tagged line
  `CVS SUPPORT-MATRIX: UNSUPPORTED <suite> on <arch>/ROCm <ver> — <reason>`,
  where `<reason>` is the block's `notes`; the workload is identified by the suite
  name and the detected values, never parsed from the filename. Record a prominent
  row in the HTML report /
  Run Deck, set `lifecycle.failed`, and `fail_test(...)` the stage → the workload
  stages skip. (Prefer this over `pytest.exit()` so the report/zip still finalize;
  `pytest.exit()` remains an option if a truly immediate process exit is wanted.)
- On **violation + warn/off**, or on a **skipped check** (missing
  `supported_arch` or `required_rocm_min_version`): log
  `CVS SUPPORT-MATRIX: WARNING …` with the detected values and continue.
- Always report the detected `gpu_arch` and `rocm_version` so the row is useful
  even when it passes.

## 10. Phased implementation plan

1. **Library core (pure, no cluster):** `schema.py` (`SupportMatrix` model in
   `parsers/schemas.py`), `version.py`, `arch.py` + `data/arch_aliases.json`,
   `evaluate.py`; unit tests for version compare, arch normalization, and
   evaluate() allow/deny/missing-field/enforce-vs-warn.
2. **Detection + gate:** `detect.py` (mock `.exec()` in tests), `gate.py` with the
   enforce/warn/off policy and the report hook; unit tests with mocked handles.
3. **Suite wiring:** add the `test_support_matrix` stage + order pin to training
   (jaxmaxtext/megatron/torchtitan) and inference (vllm/sglang, then atom/xdit);
   add the CLI/env override in `cvs/conftest.py` / the `cvs run` plumbing.
4. **Seed data:** annotate a first set of configs with `support_matrix`
   (the MXFP4 ones get `supported_arch: ["mi355x"]` + `required_rocm_min_version`;
   latest-model configs get their ROCm floor), transcribed from the tracker.
5. **Docs + CHANGELOG:** document the `support_matrix` block in
   `docs/reference/configuration-files/`, add a short how-to note, and an "Added"
   CHANGELOG entry.

Every library function/class ships with unit tests in
`cvs/lib/support_matrix/unittests/` (repo rule), mocking all `.exec()` calls.

## 11. Edge cases, risks, backward-compat

- **No block present** → both checks skip with warnings; nothing breaks.
- **Detection fails** (no rocm-smi / no amd-smi in image) → treat as a warning,
  not a hard stop, unless a strict mode is set, to avoid false negatives bricking
  valid runs.
- **ROCm string variance** (`6.4.0-123`, `7.0`, `7.0.0`) → tolerant parser with
  tests.
- **Arch aliasing** (`mi325` vs `mi325x`, gfx families) → single normalization
  table in data; unknown arch rejected at validation time with a clear message.
- **Multi-node / disagg** → detect on every participating node; if nodes disagree
  on arch/ROCm, that itself is a violation (report per node).
- **Filenames are free-form and are never parsed by the gate** (see §2.1). The
  structured `support_matrix` block is the sole authority for the decision; the tag
  identifies the workload by suite name and detected values.

## 12. Decisions (resolved — recommended defaults)

These are locked for the initial implementation; revisit only if a suite needs
otherwise.

1. **Allow-list** (`supported_arch`) is the default transcription style from the
   tracker; the `—` cells are simply its complement. `unsupported_arch` stays
   available as an optional negative form.
2. `required_rocm_min_version` lives **inside** the `support_matrix` block, so
   Check-A and Check-B are declared together (req #4 satisfied; the parameter is
   still a single config key, just namespaced).
3. The optional capability **rules** file is **deferred**; v1 uses the per-config
   block only. Add the rules file later if duplication becomes painful.
4. On **detection failure** (no `rocm-smi` / no `amd-smi` in the image): default is
   **warn-and-continue**, so a probe gap never bricks an otherwise-valid run. A
   strict mode can be added later.
5. The gate **never parses the config filename**. The decision uses only the
   author-declared `support_matrix` block plus runtime-detected arch/ROCm; the tag
   identifies the workload by suite name and detected values. ATOM keeps its
   existing `{gpu}_atom_*.json` fallback for its own `gpu_arch` field — unchanged
   and independent of this gate.
