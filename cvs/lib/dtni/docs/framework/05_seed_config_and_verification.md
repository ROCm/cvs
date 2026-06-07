# 05 — Seed Config and End-to-End Verification

Once the adapter class exists ([01](01_adapter_contract.md)), is registered
and has a pytest shim ([04](04_registry_and_test_entrypoint.md)), the last
step is a seed `config.json` + `threshold.json` pair and an end-to-end run
that proves the framework actually emits the mandatory smoke metrics on real
hardware.

Schema-of-record for both files is the sibling `config_and_thresholds.md`.
This page only covers the **seed pair** — minimum viable, smoke-only.

## File layout

```
cvs/input/dtni/<framework>/<model>/<model>_<variant>_perf/
  config.json
  threshold.json
```

Conventions:

- Per-framework input root: `cvs/input/dtni/<framework>/` (e.g.
  `cvs/input/dtni/vllm_single/`).
- First variant for a model is always `_perf`. `_perf` runs the workload
  with smoke metrics only — no benchmarks, no accuracy. `_accuracy`
  variants come later, after smoke is stable.
- Variant directory name pattern: `<model>_<modifier>_<variant_kind>`,
  e.g. `llama_3_1_8b_bf16_single_1_perf`. The trailing token (`_perf` or
  `_accuracy`) is load-bearing; it is referenced by tooling.
- `config.json` and `threshold.json` are siblings; the conftest reads
  `threshold.json` next to `--config_file` to drive parametrize.

## Seed `config.json` skeleton

Copy from the nearest existing sibling rather than typing from scratch.
This is the minimum surface for a `_perf` seed (no `benchmarks` array):

```json
{
  "schema_version": 1,
  "framework": "foo_single",
  "gpu_arch": "mi300x",
  "paths": {
    "shared_fs":     "/mnt/dtni/{user-id}/cvs",
    "models_dir":    "/mnt/dtni/{user-id}/models",
    "datasets_dir":  "{shared_fs}/datasets",
    "artifacts_dir": "{shared_fs}/artifacts",
    "images_dir":    "{shared_fs}/images"
  },
  "model":   { "id": "<hf-id>", "remote": 0, "precision": "bf16" },
  "image":   { "tag": "<registry>/<image>:<tag>", "remote": 1 },
  "topology": { "roles": { "server": { "count": 1, "gpus_per_node": 1 } } },
  "roles":   {
    "server": {
      "command": "...",
      "port": 8000,
      "health_path": "/health",
      "volumes": { "{models_dir}": "/models" },
      "devices": ["/dev/kfd", "/dev/dri"]
    }
  },
  "params":  { "input_len": 16, "output_len": 32 }
}
```

Notes:

- `paths` is boilerplate; the 5 keys (`shared_fs`, `models_dir`,
  `datasets_dir`, `artifacts_dir`, `images_dir`) are copied verbatim across
  every variant in the repo. Do not invent extra keys.
- `framework` MUST match the registry key from
  [04](04_registry_and_test_entrypoint.md).
- For per-field semantics (substitution tokens, `model.remote`,
  benchmarks/datasets, role command syntax) see the sibling
  `config_and_thresholds.md`.

## Seed `threshold.json` skeleton

```json
{
  "smoke_request_latency_ms": { "kind": "max_ms", "value": 600000 },
  "smoke_completion_tokens":  { "kind": "min",    "value": 1 }
}
```

These two thresholds match the mandatory smoke metrics from
[03](03_artifacts_and_smoke.md). The 10-minute (`600000` ms) ceiling is the
de facto floor used across every existing variant in the repo — start
there, tighten later. Do not add benchmark or accuracy thresholds to the
seed; those come with the `_accuracy` variant and require the benchmark
registry coupling described in `add_workload.md`.

## End-to-end verification sequence

Run these in order. Each one catches a distinct failure mode; do not skip
ahead.

```bash
# 1. Config parses and substitutions resolve.
cd cvs && python -m cvs.lib.dtni.config_loader \
    cvs/input/dtni/foo_single/<model>/<model>_<variant>_perf/config.json

# 2. Framework is registered and adapter resolves.
cvs list foo_single

# 3. Pytest collects the entrypoint with this config.
cvs run foo_single \
    --cluster_file=<cluster.json> \
    --config_file=cvs/input/dtni/foo_single/<model>/<model>_<variant>_perf/config.json \
    --collect-only

# 4. DTNI unittests still pass — no regressions from new code.
pytest cvs/lib/dtni/unittests/ -x

# 5. Real smoke run on the cluster. Expect both mandatory smoke
#    metrics emitted and both smoke thresholds PASS.
cvs run foo_single \
    --cluster_file=<cluster.json> \
    --config_file=cvs/input/dtni/foo_single/<model>/<model>_<variant>_perf/config.json \
    --html=report.html
```

| Step | Failure mode caught |
|---|---|
| 1 | JSON syntax; missing required field; unresolvable substitution token; unknown paths key. |
| 2 | Registry key missing; import-time error in `<framework>_adapter.py`. |
| 3 | Test shim under `cvs/tests/dtni/` missing or misnamed; `threshold.json` not next to `config.json`; threshold JSON unparseable. |
| 4 | A change to the adapter or registry broke an existing unittest (import side effect, signature drift). |
| 5 | The adapter actually launches the workload, the workload actually answers, and `parse` emits both smoke metrics. Failing here with `failed_phase="verify"` and a missing-metric note in `verdicts` means `parse` returned without setting one of the two smoke scalars. |

## After smoke is green

Once step 5 is PASS:

- Tighten `smoke_request_latency_ms` toward a realistic ceiling for the
  model + GPU combo.
- Add the first `_accuracy` variant under the same model directory
  following `add_workload.md` (catalog coupling to `BENCHMARK_REGISTRY`
  becomes mandatory at that point).
- If this framework is the first one exposed to external users, update
  `docs/reference/configuration-files/*.rst` per `docs/sphinx_rst.md`.
