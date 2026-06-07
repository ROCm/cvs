# 04 — Registry and Test Entrypoint

An adapter class on disk is invisible until two things happen:

1. The class is in `FRAMEWORK_REGISTRY` so the Job knows which adapter
   handles a `framework: "<name>"` config.
2. A pytest entrypoint shim under `cvs/tests/dtni/` exposes the framework
   to `cvs run <framework>` / `cvs list <framework>`.

See [`01_adapter_contract.md`](01_adapter_contract.md) for the class itself
and [`05_seed_config_and_verification.md`](05_seed_config_and_verification.md)
for the first config/threshold pair.

## Registry — `cvs/lib/dtni/frameworks/registry.py`

The registry is a flat dict mapping framework name to adapter class. Adding
a framework is a two-line diff: one import, one dict entry.

```python
# cvs/lib/dtni/frameworks/registry.py
from __future__ import annotations

from cvs.lib.dtni.frameworks.vllm_single_adapter import VllmAdapter
from cvs.lib.dtni.frameworks.foo_single_adapter import FooAdapter   # NEW
from cvs.lib.dtni.base_adapter import BaseWorkloadAdapter

FRAMEWORK_REGISTRY: dict[str, type[BaseWorkloadAdapter]] = {
    "vllm_single": VllmAdapter,
    "foo_single":  FooAdapter,                                       # NEW
}
```

Rules:

- The dict key MUST match the adapter's `framework` class attribute and the
  `framework` field in every config that targets it. Mismatch is a silent
  "framework not found" at runtime.
- One key per adapter class. If you have both single-host and disaggregated
  variants of the same engine, use distinct keys (`vllm_single`,
  `vllm_disagg`) and distinct adapter classes.
- Imports here are eager — a broken import in your new adapter module
  fails the whole registry. Keep adapter top-level imports light; defer
  optional dependencies into the phase methods.

## Pytest entrypoint shim — `cvs/tests/dtni/<framework>.py`

Each framework needs a small test file under `cvs/tests/dtni/`. It is what
`cvs run <framework>` and `cvs list <framework>` actually collect. The
template is `cvs/tests/dtni/vllm_single.py`:

```python
# cvs/tests/dtni/foo_single.py
"""Customer-facing pytest entrypoint for the DTNI foo_single framework."""

from __future__ import annotations

import pytest


def test_threshold(metric: str, workload_outcome):
    """One node per threshold metric."""
    if metric.startswith("__"):
        pytest.skip(f"workload not collectible: {metric}")

    result = workload_outcome.job_result
    if result.failed_phase and result.failed_phase != "verify":
        pytest.skip(
            f"phase {result.failed_phase!r} failed before verify: {result.message}"
        )

    verdict = next((v for v in result.verdicts if v["metric"] == metric), None)
    if verdict is None:
        pytest.fail(f"threshold {metric!r} defined but no verdict produced")

    assert verdict["passed"], (
        f"{metric}: actual={verdict['actual']} threshold={verdict['threshold']} "
        f"kind={verdict['kind']}"
        + (f" — {verdict['note']}" if verdict.get("note") else "")
    )
```

That is the whole file. All the heavy lifting is in
`cvs/tests/dtni/conftest.py`:

- `pytest_generate_tests` parametrizes `metric` from the sibling
  `threshold.json` of the `--config_file` flag.
- `workload_outcome` fixture (module-scope) calls
  `cvs.lib.dtni.runner.execute_workload(cluster_path=..., workload_config_path=...)`
  exactly once per module and shares the result across every per-metric
  node.

Because the conftest is shared, you do NOT add a new conftest per framework
— just the `test_threshold` shim.

## Unittest conftest sentinel — `cvs/lib/dtni/unittests/conftest.py`

This is a deliberately near-empty file:

```python
"""Empty conftest to shadow the repo-level cvs/conftest.py for these unit tests.

The repo conftest imports HTML reporting deps not needed for pure-logic tests.
"""
```

Its only job is to **shadow** the repo's top-level `cvs/conftest.py` so the
pure-logic unittests under `cvs/lib/dtni/unittests/` can be run without
pulling in the HTML-report dependency tree the customer-facing pytest entry
needs.

There is no fixture in this conftest that swaps `ctx.executor` to
`LocalExecutor` — each unittest constructs the `RunContext` directly with
whatever executor (real or fake) it needs. The conftest exists only as a
shadow file. When adding a new adapter, you usually need no change here at
all; instead, add module-level fixtures inside your own
`test_<framework>_adapter.py` that build a `RunContext` with a stub
executor of your choice.

## Collection verification

After registering the adapter and adding the test shim, three commands
prove the framework is wired up before you ever try to run it:

```bash
# 1. The framework appears in `cvs list`.
cvs list foo_single

# 2. Pytest can collect the entrypoint with a real config + threshold.
cvs run foo_single \
    --cluster_file=<cluster.json> \
    --config_file=cvs/input/dtni/foo_single/<model>/<model>_<variant>_perf/config.json \
    --collect-only

# 3. The DTNI unittests still pass (no import-time regressions from
#    the new adapter or registry entry).
pytest cvs/lib/dtni/unittests/ -x
```

Each command targets a distinct failure mode:

| Command | Catches |
|---|---|
| `cvs list foo_single` | Framework key missing from registry; import-time error in the adapter file. |
| `cvs run --collect-only` | Test shim missing; `threshold.json` unparseable; config path wrong. Per-metric nodes appear as `test_threshold[<metric>]`. |
| `pytest cvs/lib/dtni/unittests/ -x` | New import broke an existing unittest module via the registry side effects. |

Only after all three are green should you progress to a real smoke run
(see [`05_seed_config_and_verification.md`](05_seed_config_and_verification.md)).
