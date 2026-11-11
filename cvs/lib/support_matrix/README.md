# HW/SW support-matrix compatibility gate

CVS checks, **before a workload runs**, whether the framework/model under test is
supported on the GPU architecture and ROCm version of the nodes it is about to run
on. If it is not, the run is stopped with a clear message instead of failing later
with an obscure error.

Examples of what it catches:

- A precision that needs newer hardware (e.g. **MXFP4 on MI300X/MI325X**).
- A model that is not supported on a given GPU at all.
- A model that needs a newer ROCm release than the one installed.

This README is the canonical reference for the feature — the data-file format,
the runtime behaviour, the environment switch, and the library API are all
documented here.

## How it runs

Each training and inference suite runs a `test_support_matrix_gate` stage, pinned
to execute **right after the container is launched** and before the first workload
stage (`pytest_collection_modifyitems` in each suite's `conftest.py` sets the
order). The stage gathers what the suite already knows — the framework (the suite
itself), the model (a config field), and the parameter values the run will use
(e.g. the set of precisions across the enabled sweeps; inference suites pass no
parameters today) — and calls the library:

```python
from cvs.lib.support_matrix.support_matrix_lib import run_support_matrix_gate

result = run_support_matrix_gate(
    "training", "jaxmaxtext", model_name, {"precision": ["bf16", "mxfp4"]}, orch
)
if not result.passed:
    lifecycle.failed = True
    pytest.fail("support matrix gate blocked this run: " + "; ".join(result.violations))
```

The library detects the platform on every participating node — through the same
`orch` handle, whose `.exec()` routes into the launched container, so both probes
reflect the container the workload will run in:

- **GPU arch** via `rocm-smi`, falling back to `amd-smi` (both parsed by the
  existing `get_model_from_rocm_smi_output()` in `cvs/lib/utils_lib.py`), normalized
  to a canonical token. The probes carry a short timeout. An **unrecognized GPU is
  reported as undetected** — it is never defaulted to MI300X.
- **ROCm version** via `cat /opt/rocm/.info/version`, falling back to
  `amd-smi version --json`, then `hipconfig --version`.

It then evaluates three independent deny checks against each node's detected
arch/ROCm; any match is a violation:

- **Check-A — GPU arch:** detected arch is in the model's `unsupported_gpu_arch`.
  An **undetected GPU is itself a violation** — the gate cannot certify a GPU it
  cannot identify.
- **Check-B — ROCm range:** detected ROCm is below `required_rocm_min_version` or
  above `required_rocm_max_version`.
- **Check-C — parameter value:** a value the run uses (e.g. `precision=mxfp4`) is in
  `unsupported_params.<param>.<value>` for the detected arch.

Undetected ROCm produces a warning, not a violation (a model only gates ROCm when it
declares a min/max). The config **filename is never parsed** — only the data file and
the runtime-detected values drive the decision.

## The data files

All support data lives in two shipped files in this folder, keyed
`framework → model`:

- `training_support_matrix.json` — training frameworks (jaxmaxtext, megatron, torchtitan)
- `inference_support_matrix.json` — inference frameworks (vllm, sglang, atom, xdit)

Top-level shape:

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

`known_arch_tokens` is only a registry used to reject typos at load time; it is
**not** the universe used to invert anything.

### Model entry (deny-list)

An entry records **only what is unsupported**. Everything not listed — including GPU
archs added in the future — is allowed, so most entries are tiny or empty.

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

Field semantics (every rule is a deny; a match stops the run):

- **`aliases`** (optional): alternate names the suite may pass (e.g. an HF id). The
  lookup is **separator- and path-insensitive**: it lower-cases, takes the basename
  of a path or `org/model` HF id, and drops `-`, `_`, `.` and spaces, so
  `llama-3.1-8b` (jaxmaxtext), `llama3.1_8B` (megatron), `llama3_1_8b` (torchtitan),
  `/root/models/Llama-3.1-8B` and `meta-llama/Llama-3.1-8B` all match the one entry.
  Add an alias only for a genuinely different name (e.g. a MaxText preset like
  `deepseek2-16b`).
- **`unsupported_gpu_arch`** (Check-A): archs on which the model does not run at all.
  Empty `[]` means it runs on every arch. Kept present even when empty so the field
  is discoverable.
- **`required_rocm_min_version`** / **`required_rocm_max_version`** (Check-B): the
  detected ROCm must be within `[min, max]`. An empty string `""` means that bound is
  unset — **both are empty by default**, so no ROCm gating happens until a model
  owner sets a real minimum (or a `max` when a model is **deprecated**). Versions
  tolerate a build suffix (`6.4.0-123`) and a missing patch (`7.0`).
- **`unsupported_params.<param>.<value>`** (Check-C): the archs on which that
  parameter value is unsupported. The shape is `param → value → [archs]`, so it
  extends to any parameter (`precision`, `seqlen`, `tp`, `ep`, ...) with no schema
  change. The field is named `unsupported_params` (not `unsupported_sweep_params`)
  because a gated parameter may live anywhere in the config, not only in a sweep.
- **`notes`** (optional): free text for humans; ignored by the gate.

Arch tokens are canonical `mi300x / mi325x / mi350x / mi355x`. `rocm-smi` sometimes
reports `mi325 / mi350 / mi355`; the library normalizes those to the `…x` form, so
either spelling works in the data.

Performance values (batch, GBS, tok/s) are **not** stored here — they are tuning
defaults, not support signals.

### Why a deny-list (not an allow-list)

- The unsupported set is sparse: almost every `(model, precision)` is supported on
  every GPU; only a few cells (MXFP4/MXFP8, some FP8) are restricted.
- A new GPU generation needs **zero edits** — it is allowed automatically because it
  is not denied anywhere. An allow-list would need every row edited to add it.
- It matches the request: "block MXFP4 on MI300/MI325", everything else runs.

Trade-off (accepted): a deny-list cannot tell "confirmed supported everywhere" from
"not yet characterised" — both are empty entries. Positive coverage lives in the
validation tracker; this file records only what to block.

### Editing the data

Adding or removing support is a **data edit only** — the library code never changes
for a new mapping. Add a model under its framework, keyed by the name its input
config carries (matching is separator/path-insensitive — see `aliases` above), fill
the deny fields that apply, and add an alias only for a genuinely different preset
name. The matrix lists **only models that have an input config**. Today jaxmaxtext
carries the transcribed deny data; megatron, torchtitan and the inference frameworks
are present as empty templates for their owners to fill.

## Turning the gate off (experiment runs)

Enforcement is controlled by **one environment variable** (there is no CLI flag):

```
CVS_SUPPORT_MATRIX_GATE
```

| Value (case-insensitive) | Behaviour |
|---|---|
| unset (default), `1`, `ON`, `TRUE` | **Enforce** — on a violation the gate fails and the run stops. |
| `0`, `OFF`, `FALSE` | **Disabled** — violations are printed as warnings and the run continues, with a NOTE that the gate is disabled. |

If the data file itself fails to load, the stage **fails** when the gate is on, and
warns-and-continues when it is off.

Use the disabled mode when you deliberately want to run a combination the matrix
flags (for example, an experiment on hardware that is not yet characterized) without
being blocked:

```bash
CVS_SUPPORT_MATRIX_GATE=OFF cvs run jaxmaxtext_single \
  --cluster_file <cluster>.json --config_file <config>.json
```

With the gate disabled you still see each violation in the log, followed by:

```
NOTE: CVS_SUPPORT_MATRIX_GATE is disabled - continuing despite the violation(s) above.
```

## When a workload is not in the matrix

If the framework or model under test has no entry in the data file, the run is
**allowed** and the log records:

```
the framework or model under test is not available in the support matrix. This
means this framework/model is not validated / characterized yet with CVS.
```

## Library API

Everything lives in one module, `support_matrix_lib.py`:

- `run_support_matrix_gate(category, framework, model, params, handle)` — the single
  entry point the suites call. Loads the data file, looks up `(framework, model)`,
  detects the platform, evaluates the deny rules, applies the env switch, logs the
  outcome, and returns a `GateResult`.
- `GateResult` — `passed` (may the run proceed?), `violations`, `warnings`,
  `not_in_matrix`, `enforced`, `platform`.
- `detect_platform(handle)` — `{host: {"arch": <token|None>, "rocm": <version|None>}}`.
- `evaluate(entry, params, platform)` — pure function returning
  `(violations, warnings)`; the core logic, trivially unit-tested.
- `gate_enforced()` — reads `CVS_SUPPORT_MATRIX_GATE`.
- `served_model_name(variant_config)` — best-effort served-model id from a suite's
  config. The key that names the model is not unified across suites yet, so it tries
  `model_id` / `hf_model_id` / `model_name` / `hf_model_name` / `model` both at the
  top level and inside `model` / `server_params` / `model_params` / `train_params` /
  `inference` / `tokenizer`. (`model_repo` is excluded — it is a filesystem path.)
- `load_matrix(category)`, `find_model_entry(matrix, framework, model)` (lenient,
  separator/path-insensitive), `normalize_arch(token)`, `parse_version(text)`.

## Files in this folder

```
support_matrix_lib.py            the whole library
training_support_matrix.json     data: training frameworks
inference_support_matrix.json    data: inference frameworks
README.md                        this document (canonical reference)
DESIGN.md                        design rationale (may be removed later; README is canonical)
unittests/test_support_matrix_lib.py
```

Unit tests mock all `.exec()` calls (no live cluster) and cover version parsing,
arch normalization and confident detection, the env switch, lenient model matching
(config spellings and path basenames), evaluation (arch/ROCm/param denies and
undetected arch as a violation), served-model extraction, and the end-to-end gate in
enforce, disabled, supported, absent-entry, unknown-GPU, and load-failure modes. Run
them with:

```bash
python -m unittest cvs.lib.support_matrix.unittests.test_support_matrix_lib
```
