# Design-2 — Single support-matrix data file per category

Status: proposal (design only — no implementation yet)
Branch: `solaiys/dtni-hwsw-support-matrix-check`
Scope: Inference and Training workload suites run through CVS.

This is an alternative to `DESIGN.md` (Design-1, per-config `support_matrix`
block). Design-2 keeps **all** support data in **two shipped files** — one for
training, one for inference — instead of annotating each config.

- `training_support_matrix.json` — jaxmaxtext, megatron, torchtitan (+ future).
- `inference_support_matrix.json` — vllm, sglang, atom, xdit (+ future).

A sample `training_support_matrix.json` is committed beside this doc, transcribed
from the DTNI validation-tracker coverage for the three training frameworks.

## 1. Why a single file per category (vs Design-1)

| | Design-1 (per-config block) | Design-2 (single data file) |
|---|---|---|
| Source of truth | scattered across every config | one file per category |
| Transcribing the tracker | edit N configs | edit 1 file, model-by-model |
| New/unannotated config | silently skips the gate | still gated if its model is in the file |
| Who maintains it | each config author | CVS team, in one review-friendly place |
| Config filename | irrelevant (see §6) | irrelevant (see §6) |

Design-2 matches how the data actually originates — one tracker, maintained by
the CVS team — and gates a workload even when its config was never annotated. The
unit of lookup is **(framework, model)**, which the suite already knows at run
time (framework = the suite; model = a config field).

## 2. File layout and lookup

```
cvs/lib/support_matrix/
  training_support_matrix.json     # this sample (final home may be data/)
  inference_support_matrix.json    # same shape; created when inference coverage is supplied
```

Top level:

```json
{
  "schema_version": 1,
  "category": "training",
  "known_arch_tokens": ["mi300x", "mi325x", "mi355x"],
  "enforce_default": true,
  "frameworks": {
    "<framework>": { "models": { "<model-key>": { ...model entry... } } }
  }
}
```

Lookup at run time: `frameworks[framework].models[model]`. The model is matched
on the normalized key (lower-cased, trimmed) or any entry in its optional
`aliases` list (so a suite can pass either a short name like `llama-3.1-8b` or an
HF id like `meta-llama/Llama-3.1-70B`). A framework or model **absent** from the
file is an *unknown* workload → checks are skipped with a warning (never blocks a
run we have no data for).

## 3. Model entry (deny-list)

Each entry records **only what is unsupported**. Everything not listed — including
GPU archs added in the future — is allowed, so most models have a tiny or empty
entry.

```json
"llama-3.1-8b": {
  "aliases": ["meta-llama/Llama-3.1-8B"],
  "unsupported_gpu_arch": [],
  "required_rocm_min_version": "7.0",
  "required_rocm_max_version": null,
  "unsupported_sweep_params": {
    "precision": {
      "mxfp8": ["mi300x", "mi325x"],
      "mxfp4": ["mi300x", "mi325x"]
    }
  },
  "notes": "optional free text"
}
```

Field semantics (every rule is a **deny**; the run is stopped if any matches):

- **`unsupported_gpu_arch`** (model-level deny / Check-A): the model does not run
  on these archs at all (any sweep). Detected arch in this list → stop. Absent/`[]`
  → the model runs on every arch (e.g. Llama-3.1-405B lists `["mi300x"]`).
- **`required_rocm_min_version`** / **`required_rocm_max_version`** (Check-B):
  detected ROCm (inside the container, where the workload runs) must be within
  `[min, max]`. Either bound absent → unbounded on that side; both absent → Check-B
  skipped.
- **`unsupported_sweep_params.<param>.<value>`** (Check-C, per sweep param): the
  **list of archs on which that param value is not supported** (even though the
  model otherwise runs there). If the run requests `precision=mxfp4` and the
  detected arch is in `unsupported_sweep_params.precision.mxfp4`, the run is
  stopped. A value or param not listed imposes no constraint. The same
  `param → value → [archs]` shape extends to any future sweep param (`seqlen`,
  `tp`, `ep`, attention backend, ...) with **no schema change**.

`known_arch_tokens` at the top level is only a registry for validation (reject
typos); it is **not** the universe used to invert anything — a new token simply is
not denied, hence allowed, until a deny entry is added.

Performance values (Batch, GBS, tok/s/gpu) from the tracker are **not** stored
here — they are tuning defaults, not support signals.

## 4. Point 2 — deny-list is the right shape for this data

The entry above uses an **unsupported (deny) list**
(`unsupported_sweep_params.<param>.<value> = [unsupported arch]`, plus a
model-level `unsupported_gpu_arch`), not an allow-list. Three reasons specific to
this data and the intended posture:

1. **The unsupported set is sparse.** Almost every `(model, precision)` is
   supported on every GPU; only a few cells (MXFP4/MXFP8, some FP8) are restricted.
   A deny-list stores just those cells, so most entries are tiny or empty — far
   less to write and review than listing every supported arch per precision.
2. **Future archs need no edits.** A new GPU generation is expected to support
   everything. With a deny-list it is allowed automatically (not denied anywhere);
   with an allow-list every model×precision row would have to be edited to add it.
   Default-allow is also exactly the requested behaviour — "block MXFP4 on
   MI300/MI325", everything else runs.
3. **Extensible to new gating params.** `unsupported_sweep_params` is keyed
   `param → value → [archs]`, so a future constraint (a sequence length or
   parallelism unsupported on some arch) is a new key, not a new structure.

Trade-off (accepted): a deny-list cannot distinguish "confirmed supported
everywhere" from "not yet characterised" — both are empty entries. The positive
coverage stays in the tracker; this file records only what to block. TBD rows
carry a `notes` string for human context and otherwise impose no gate.

Shape chosen: `param → value → [archs]` (value-first) because each tracker row is
one precision, so a `—` cell is transcribed as "add this arch under that
precision". Arch-first (`arch → [values]`) is equivalent but reads less directly
from the sheet.

## 5. Point 4 — where and how the gate runs

The check runs **after the container is launched** (so Check-B sees the
container's ROCm), using the same `orch`/handle the suites already hold.

Flow, per suite, as one early lifecycle stage placed right after the
container-launch stage and before the first workload stage:

1. The suite calls the library with what it already knows:
   `support_matrix.gate(category="training", framework="jaxmaxtext",
   model=<from config>, sweep={"precision": <from config/sweep>, ...},
   handle=orch)`.
2. The library **detects at run time on the GPU nodes**:
   - GPU arch per node via `rocm-smi` → `get_model_from_rocm_smi_output()`
     (existing helper in `cvs/lib/utils_lib.py`).
   - ROCm version inside the container via `cat /opt/rocm/.info/version`, falling
     back to `amd-smi version --json` (reuse `RocmVersionCheck._extract_rocm_version`)
     then `hipconfig --version`.
3. It looks up `frameworks[framework].models[model]` and evaluates Check-A
   (`unsupported_gpu_arch`), Check-B (ROCm range), and Check-C
   (`unsupported_sweep_params`) against the detected arch/ROCm — each is a deny
   rule, so a match means "stop".
4. On a violation with enforcement on, it logs a distinctive tag
   `CVS SUPPORT-MATRIX: UNSUPPORTED <framework>/<model> precision=<v> on
   <arch>/ROCm <ver> — <reason>`, records a report row, and **stops the run**
   (fails the stage → the lifecycle guard skips every later workload stage). With
   enforcement off (or an unknown/`null` entry), it logs a warning and continues.

Multi-node: detect on every participating node; if nodes disagree on arch/ROCm,
that is itself a violation (reported per node).

Enforcement precedence: CLI/env override (`--support-matrix={enforce|warn|off}`)
→ file `enforce_default` → default `enforce`.

## 6. Config filenames are NOT used (carried over from Design-1 §2.1)

The gate never parses the config filename. The only authoritative inputs are this
data file plus the runtime-detected arch/ROCm. Framework comes from the suite,
model from a config field (with `aliases` to bridge naming). (ATOM's existing
`{gpu}_atom_*.json` fallback for its own `gpu_arch` field is unrelated and
unchanged.)

## 7. Library shape (data-driven; code never changes to add a mapping)

```
cvs/lib/support_matrix/
  loader.py      # load + pydantic-validate the category file
  version.py     # parse_rocm_version(), compare() — tolerant of "7.0", "6.4.0-123"
  arch.py        # normalize_arch(); alias/gfx table as data
  detect.py      # detect_gpu_arch(handle), detect_rocm_version(handle) via .exec
  evaluate.py    # evaluate(model_entry, requested_sweep, platform) -> Result(violations[], warnings[])
  gate.py        # gate(category, framework, model, sweep, handle, report=None) -> Result
  unittests/     # one test module per file; mock all .exec()
```

Adding or removing support = editing the JSON data file only. The schema is a
pydantic model in `cvs/parsers/schemas.py` (consistent with the existing single
source of truth) that rejects unknown arch tokens and malformed versions early.

## 8. Transcribed sample — `training_support_matrix.json`

The committed sample encodes the three coverage tables as deny rules. Notable
gated cases:

- **MXFP8 / MXFP4** (jaxmaxtext & megatron, Llama-3.1-8B):
  `unsupported_sweep_params.precision.{mxfp8,mxfp4} = ["mi300x","mi325x"]` —
  allowed only on MI355X; a run on MI300X/MI325X is stopped (the motivating case).
- **FP8 on Llama-3.3-70B**: `unsupported_sweep_params.precision.fp8 = ["mi300x"]`
  (MI300X `—`), while FP8 on Llama-3.1-8B has no deny — model-dependent FP8, which
  is why a single global precision table cannot express it.
- **Llama-3.1-405B** (jaxmaxtext): `unsupported_gpu_arch: ["mi300x"]` — model-level
  HW deny (not supported on MI300X at all).
- **TBD rows** (Mixtral-8x22B; megatron/torchtitan 405B; torchtitan
  DeepSeek-V2-Lite, Qwen3-32B; BF16 parity runs): only a `notes` string, no deny
  rules → they run unrestricted until characterised.
- **Everything else** (most rows): no deny entry (just a
  `required_rocm_min_version`) → supported on all archs.

## 9. Phased plan

1. Library core + pydantic schema + `version.py`/`arch.py` + `evaluate.py`; unit
   tests (allow/deny/missing/unknown/enforce-vs-warn).
2. `detect.py` + `gate.py` (mock `.exec()`); CLI/env override wiring.
3. Suite stage wiring: training (jaxmaxtext/megatron/torchtitan) first, then
   inference; pass framework + model + sweep to `gate()` after launch.
4. Fill `training_support_matrix.json` from the tracker (this sample) and create
   `inference_support_matrix.json` when inference coverage is supplied.
5. Docs (`docs/reference/configuration-files/`) + CHANGELOG "Added".

## 10. Open decisions

1. File home: beside the library (`cvs/lib/support_matrix/`) vs a `data/`
   subdir vs `cvs/input/support_matrix/`. Recommendation: `cvs/lib/support_matrix/`
   (team-maintained reference data shipped with the library, not per-run user
   input).
2. Unknown model/framework → warn-and-continue (recommended) vs hard-skip-silent.
3. Whether to also keep Design-1's per-config block as an override for one-off
   workloads not worth a matrix row. Recommendation: no — one mechanism.
