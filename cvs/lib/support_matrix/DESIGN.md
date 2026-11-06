# Support-matrix compatibility gate — design

Status: accepted (single-data-file design); implemented and wired into the training
and inference suites.
Branch: `solaiys/dtni-hwsw-support-matrix-check`
Scope: Inference and Training workload suites run through CVS.

CVS stops a workload **before it runs** when the framework/model is not supported
on the detected GPU architecture or ROCm version — e.g. MXFP4 must not run on
MI300/MI325, and the latest models must not run on older ROCm. The check runs
**after the container is launched** (so it sees the container's ROCm), reads a
single shipped data file per category, and compares the declared constraints
against the GPU arch and ROCm version **detected at run time on the GPU nodes**.

All support data lives in **two shipped files**, keyed `framework → model`:

- `training_support_matrix.json` — jaxmaxtext, megatron, torchtitan (+ future).
- `inference_support_matrix.json` — vllm, sglang, atom, xdit (+ future).

Only **jaxmaxtext** carries data today; the other frameworks/models are empty and
filled in later by their owners.

## 1. Why a single data file per category

- One review-friendly place that matches how the DTNI tracker is maintained (one
  sheet per category), rather than annotating every workload config.
- A workload is gated even when its config was never annotated.
- Deny-list entries stay sparse, and a **new GPU generation needs zero edits**
  (see §4).

The unit of lookup is **(framework, model)**, which the suite already knows at run
time: the framework is the suite, and the model is a config field.

## 2. File layout and lookup

```
cvs/lib/support_matrix/
  support_matrix_lib.py            # the whole library (load, detect, evaluate, gate)
  training_support_matrix.json     # data: training frameworks
  inference_support_matrix.json    # data: inference frameworks
  README.md                        # user-facing: what the gate does + the env switch
  unittests/test_support_matrix_lib.py
```

Top level of a data file:

```json
{
  "schema_version": 1,
  "category": "training",
  "known_arch_tokens": ["mi300x", "mi325x", "mi350x", "mi355x"],
  "frameworks": {
    "<framework>": { "models": { "<model-key>": { ...model entry... } } }
  }
}
```

Lookup at run time: `frameworks[framework].models[model]`. Matching is
**separator- and path-insensitive**: the model string is lower-cased, reduced to the
basename of a path or `org/model` HF id, and stripped of `-`, `_`, `.` and spaces.
So one entry keyed `llama-3.1-8b` matches every spelling the suites actually pass —
jaxmaxtext `llama3.1-8b`, megatron `llama3.1_8B`, torchtitan `llama3_1_8b`, a local
path `/root/models/Llama-3.1-8B`, and the HF id `meta-llama/Llama-3.1-8B`. An
`aliases` entry is needed only for a genuinely different name (e.g. a MaxText preset
like `deepseek2-16b`). The matrix lists **only models that have an input config**.

A framework or model **absent** from the file is **allowed to run** (never
blocked), and the gate logs:

> the framework or model under test is not available in the support matrix. This
> means this framework/model is not validated / characterized yet with CVS.

## 3. Model entry (deny-list)

Each entry records **only what is unsupported**. Everything not listed — including
GPU archs added in the future — is allowed, so most models have a tiny or empty
entry.

```json
"llama-3.1-8b": {
  "aliases": ["meta-llama/Llama-3.1-8B"],
  "unsupported_gpu_arch": [],
  "required_rocm_min_version": "",
  "required_rocm_max_version": "",
  "unsupported_params": {
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
  on these archs at all. Detected arch in this list → stop. Empty `[]` → the model
  runs on every arch. Kept present (even when empty) so the field is discoverable.
  An **undetected GPU is itself a violation** (the gate cannot certify a GPU it
  cannot identify) — see §6.
- **`required_rocm_min_version`** / **`required_rocm_max_version`** (Check-B):
  detected ROCm (inside the container, where the workload runs) must be within
  `[min, max]`. An empty string `""` means that bound is unset; **both are empty by
  default**, so ROCm is not gated until a model owner sets a real minimum (or a
  `max` when a model is **deprecated**).
- **`unsupported_params.<param>.<value>`** (Check-C, per parameter): the **list of
  archs on which that parameter value is not supported**. If the run uses
  `precision=mxfp4` and the detected arch is in
  `unsupported_params.precision.mxfp4`, the run is stopped. A value or param not
  listed imposes no constraint. The same `param → value → [archs]` shape extends to
  any parameter (`precision`, `seqlen`, `tp`, `ep`, ...) with **no schema change**.

Naming: the field is `unsupported_params` (not `unsupported_sweep_params`) because
a gated parameter may live anywhere in the config, not only inside a sweep block.

`known_arch_tokens` at the top level is only a registry for validation (reject
typos); it is **not** the universe used to invert anything — a new token simply is
not denied, hence allowed, until a deny entry is added.

Performance values (Batch, GBS, tok/s/gpu) from the tracker are **not** stored here
— they are tuning defaults, not support signals.

## 4. Why a deny-list

1. **The unsupported set is sparse.** Almost every `(model, precision)` is
   supported on every GPU; only a few cells (MXFP4/MXFP8, some FP8) are restricted.
   A deny-list stores just those cells, so most entries are tiny or empty.
2. **Future archs need no edits.** A new GPU generation is allowed automatically
   (not denied anywhere); an allow-list would need every row edited to add it.
   Default-allow is exactly the requested behaviour — "block MXFP4 on MI300/MI325",
   everything else runs.
3. **Extensible to new gating params.** `unsupported_params` is keyed
   `param → value → [archs]`, so a new constraint is a new key, not a new structure.

Trade-off (accepted): a deny-list cannot distinguish "confirmed supported
everywhere" from "not yet characterised" — both are empty entries. Positive
coverage stays in the tracker; this file records only what to block.

## 5. Enforcement — environment switch only

There is **no CLI flag**. Enforcement is controlled by one environment variable:

```
CVS_SUPPORT_MATRIX_GATE = 1 | 0 | ON | OFF | TRUE | FALSE   (case-insensitive)
```

- **Unset (default) or ON/1/TRUE:** the gate **enforces** — on a violation it fails
  the `test_support_matrix_gate` stage and stops the run (the lifecycle guard then
  skips every later workload stage).
- **OFF/0/FALSE:** the gate **does not block** — it logs each violation as a warning
  and continues, plus a one-line NOTE that `CVS_SUPPORT_MATRIX_GATE` is disabled.
  This lets a user do an experiment run without being blocked by the gate.

A data file that fails to load is itself treated as a violation: the stage fails
when the gate is on, and warns-and-continues when it is off.

This switch is documented for users in `README.md` in this folder.

## 6. Where and how the gate runs

Each suite adds one early stage, `test_support_matrix_gate`, pinned by the suite's
`pytest_collection_modifyitems` to run **right after container launch** and before
the first workload stage. The stage:

1. Extracts what it knows from the config: `framework` (the suite), `model`, and the
   `params` the run will use (e.g. the set of `precision` values across the enabled
   sweeps). It calls the library:
   `run_support_matrix_gate("training", framework, model, params, orch)`.
2. The library **detects at run time on every participating node**, through the same
   `orch` handle (its `.exec()` routes into the launched container), with a short
   probe timeout:
   - GPU arch via `rocm-smi`, falling back to `amd-smi` — both parsed by the existing
     `get_model_from_rocm_smi_output()` (`cvs/lib/utils_lib.py`) and normalized to a
     canonical token. That helper defaults to `mi300x` for unrecognized output, so
     the gate trusts `mi300x` only when the probe actually names it; anything else
     (a banner, or an arch the parser does not know) is reported as **undetected**,
     never defaulted.
   - ROCm version via `cat /opt/rocm/.info/version`, falling back to
     `amd-smi version --json` then `hipconfig --version`.
3. It looks up `frameworks[framework].models[model]` and evaluates Check-A
   (`unsupported_gpu_arch`), Check-B (ROCm range), and Check-C (`unsupported_params`)
   against each node's detected arch/ROCm — each is a deny rule, so a match is a
   violation. An **undetected GPU arch is a violation**; undetected ROCm is only a
   warning (a model gates ROCm only when it declares a min/max).
4. It applies the env switch: enforce → return "must stop" with the violation list;
   OFF → log warnings + the disabled-NOTE and return "may proceed". An absent
   framework/model logs the not-characterized message and proceeds.

The suite then fails the stage (and sets `lifecycle.failed`) when the result says
"must stop"; otherwise it records a pass and continues.

Multi-node: detect on every node; a violation on any node (or nodes disagreeing on
arch/ROCm) is reported per node.

## 7. Library shape (one main file)

Everything lives in **`support_matrix_lib.py`** — the file is small enough that
splitting it into loader/version/arch/detect/evaluate/gate modules would add
indirection without benefit. It contains:

- data loading + light structural validation (reject unknown arch tokens / malformed
  versions),
- arch normalization and ROCm version parse/compare,
- runtime detection (`detect_platform(handle)`),
- pure evaluation (`evaluate(entry, params, platform) -> violations`),
- the env switch (`gate_enforced()`), and
- the single entry point the suites call,
  `run_support_matrix_gate(category, framework, model, params, handle)`.

Adding or removing support = editing the JSON data file only; the library code
never changes for a new mapping. Unit tests live in
`unittests/test_support_matrix_lib.py` and mock all `.exec()` calls (no live
cluster).

## 8. Data today

Each framework lists only the models that have an input config (pruned to the
config-backed set). `training_support_matrix.json` carries **jaxmaxtext** deny data
transcribed from the DTNI tracker (MXFP4/MXFP8 denied on MI300X/MI325X for
Llama-3.1-8B; FP8 denied on MI300X for Llama-3.3-70B; Llama-3.1-405B model-level deny
on MI300X), with MaxText-preset aliases where a config uses a different name. The
megatron and torchtitan models, and the whole `inference_support_matrix.json`, are
present as empty templates (no denies, `required_rocm_min_version` empty) for owners
to fill later.

## 9. Config filenames are not used

The gate never parses the config filename. The only inputs are the data file plus
the runtime-detected arch/ROCm. Framework comes from the suite, model from a config
field (with `aliases` to bridge naming).

## 10. Open decisions (resolved)

1. Data home: `cvs/lib/support_matrix/` — team-maintained reference data shipped
   with the library, not per-run user input.
2. Unknown model/framework → allow-and-log (never block a run we have no data for).
3. One mechanism only — no per-config override block.
