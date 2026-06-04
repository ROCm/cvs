# Workload adapters — AGENTS.md

> Scope: `cvs/lib/adapters/` — every concrete workload adapter and the
> contract every adapter must follow. Companion: `cvs/lib/AGENTS.md`
> (Job driver, RunContext, failure taxonomy), `cvs/lib/manifest/AGENTS.md`
> (sidecar shapes), `cvs/tests/dtni/AGENTS.md` (how tier tests consume
> a run).

## What an adapter is

A class that implements the seven-method `WorkloadAdapter` Protocol
(`cvs/lib/adapter_protocol.py`) and registers itself under a framework
name via `@register_adapter("<framework>", kind=...)`. The `Job` driver
in `cvs/lib/job.py` calls the methods in a fixed six-phase order:

```
prepare -> launch -> await_completion -> parse -> verify -> teardown
```

The adapter owns only the framework-specific behaviour:

- `prepare` -- pre-launch setup (no-op by default; the base class handles
  the empty case).
- `launch` -- start the container(s), kick the benchmark client.
- `progress_predicate` -- tri-state `RUNNING` / `DONE` / `BROKEN` polled
  by `await_completion`. `DONE` means the workload finished; `BROKEN`
  means it died (raises `SafetyViolation`).
- `parse` -- read framework-native telemetry into a `ResultView`
  (scalars + long-format samples).
- `verify` -- default base implementation evaluates configured
  thresholds against `ResultView`; override only for custom verdict
  shapes.
- `teardown` -- default base implementation captures logs + removes
  containers; override only for extra forensics.

Everything else (event emission for phase boundaries, failure-category
classification, manifest construction) is owned by `Job` -- adapters
emit only sub-phase events (`launch.container_up`, `launch.role_ready`,
...) and never write the manifest themselves.

## Registration

```python
from cvs.lib.registry import register_adapter
from cvs.lib.base_adapter import BaseWorkloadAdapter

@register_adapter("my_framework", kind="inference")
class MyAdapter(BaseWorkloadAdapter):
    framework = "my_framework"
    ...
```

Re-registering the same name (in either registry) raises `ValueError`.
The framework name must match the config dispatch key
(`cfg.framework` in the YAML).

**Auto-loading:** the package's `__init__.py` imports every adapter
module so a single `import cvs.lib.adapters` triggers registration.
Code that calls `get_adapter(name)` must therefore import the package
first; the production tier-test path already does this via the
conftest's `from cvs.lib.registry import get_adapter` + the
`_ensure_adapters_loaded` shim in `registry.py`.

## Sidecar policy (G3.2 -- per-framework column vocabulary)

G3 froze the `samples.parquet` / `trajectory.parquet` *shape*
(long-format, off-schema reject) and the cross-run `FACT_COLUMNS` fact
table, but the per-framework *column vocabulary* was unspecified.
**Policy (addendum §4.5): adapter-private samples, public-scalar fact
table.**

- `samples.parquet` / `trajectory.parquet` columns are **adapter-private**.
  Each adapter documents its own column vocabulary in this AGENTS.md
  (or in its own per-framework AGENTS.md under
  `cvs/lib/adapters/<framework>/`).
- `cvs export` flattens **only** `Verdicts.scalars` and `Identity`
  fields into `fact.parquet`. Cross-framework comparability lives in
  the verdict layer (named metrics like `ttft_p99_ms`, `loss_final`),
  not in raw sidecars.
- No per-framework column registry in `cvs/lib/manifest/`. If a future
  need arises for cross-framework sidecar queries, add a G3.3 brief --
  do not silently introduce one.

### Per-framework column vocabularies

- **vLLM** (`vllm_adapter.py`): `samples.parquet` schema is
  `request_id`, `ttft_ms`, `tpot_ms`, `itl_ms`, `e2el_ms`,
  `output_tokens`, `role` (always `"server"` in single-role configs).
  Population would come from per-request arrays in the bench result
  JSON (`ttfts`/`tpots`/`itls`/`e2els`/`output_lens`); the new
  `vllm bench serve` CLI dropped those arrays from its
  `--save-result` output, so `samples.parquet` is empty today and
  percentile thresholds resolve via the framework-scalar fallback in
  `PercentileThreshold.evaluate` (`p99_ttft_ms` etc., always present
  in the JSON). The `verdict.detail` field annotates which path
  produced each verdict. Scalars promoted to `Verdicts.scalars`
  (public): `elapsed_s`, `request_throughput`, `output_throughput`,
  `total_throughput`, `mean_ttft_ms`, `p99_ttft_ms`, `mean_tpot_ms`,
  `p99_tpot_ms`. No `trajectory.parquet` (vLLM is request-batch, not
  step-trajectory).

## Standing bake-ins (PR-Y vLLM)

These are documented here because every later adapter PR must reconcile
its own design against them; they were resolved on the reference
vertical.

- **C1 -- HTTP-200 readiness.** Readiness must assert HTTP 200 via
  `curl -s -o /dev/null -w "%{http_code}"` against
  `params.base_url:params.port_no`. Do not hardcode `:8888`; do not
  treat "any non-empty output" as ready.
- **C2 -- bench via `docker exec`, dual-site result delivery.** The
  bench client runs inside the launched container via
  `docker exec -d <name> sh -c "<bench-argv> > bench.log 2>&1"`. Result
  delivery works under either site shape:
  1. **Shared FS** (Weka/NFS at the same path on devbox + nodes): the
     local-existence check in `progress_predicate` succeeds with no
     round-trip and `parse` reads the file in place. The run's
     `logs_dir` is bind-mounted into the container at the same path
     it has on devbox; the bench's `--save-result` lands at a path
     both sides see.
  2. **No shared FS**: `progress_predicate` falls back to `ssh test -f`
     on the bound node; `parse` SFTP-fetches via
     `ctx.executor.download(remote, local)` (thin passthrough to
     `Pssh.download_file` on `_SingleHostExecutor` in the dtni
     conftest). Both paths are wired today; when a *second* adapter
     needs the same fetch shape, promote into a
     `RunContext.fetch(remote) -> local` helper.
- **C3 -- single-cell params scalars (sweep deferred).** The adapter
  reads `params.tensor_parallelism` / `params.concurrency` /
  `params.isl` / `params.osl` as scalars and exposes `CVS_TP` to the
  container env. The multi-cell sweep machinery in
  `cvs/lib/config/sweep.py` exists and is unit-tested but is not
  wired into the conftest's cell-parametrize hook -- PR-Z lifts
  these scalars to swept axes (list-typed `sweep:` block on the
  workload YAML) without changing the adapter contract.
- **C4 descoped (security removed, W7, addendum §4 B7 / §5 C4).** The
  HF token rides in `cfg.container.env['HF_TOKEN']` and is recorded
  verbatim in pssh logs / manifest commands / container.log / docker
  inspect. Accepted on closed/internal clusters; do not re-introduce
  `SecretValue` / `redact_secrets` / `env_file` plumbing.

## How to add a new adapter

1. Create `cvs/lib/adapters/<framework>_adapter.py`. Subclass
   `BaseWorkloadAdapter` and implement at minimum `launch`,
   `progress_predicate`, `parse`. Override `prepare`/`teardown` only
   if you need extra side-effects.
2. Decorate with `@register_adapter("<framework>", kind=...)`. The
   `kind` chooses the inference vs training registry; the name must
   match the config's `framework` dispatch key.
3. Add an import line to `cvs/lib/adapters/__init__.py` so the
   adapter self-registers on `import cvs.lib.adapters`. The
   `_ensure_adapters_loaded` shim in `registry.py` guarantees this
   import runs before any `get_adapter(name)` lookup.
4. Document your `samples.parquet` / `trajectory.parquet` columns
   under "Per-framework column vocabularies" above (or in a
   per-framework `cvs/lib/adapters/<framework>/AGENTS.md` if the
   adapter grows into a package).
5. Author the offline gate at
   `cvs/lib/adapters/unittests/test_<framework>_adapter.py`:
   - `test_registered` -- `get_adapter("<framework>") is <Class>`.
   - One happy-path test of any pure parser (`read_bench_result`-style).
   - One parse-edge test (missing file, missing keys, ...).
   Do not unit-test `launch` / `progress_predicate` against a fake
   executor unless the wiring is non-trivial: those are exercised by
   the real-HW gate.
6. Author the tier tests at
   `cvs/tests/dtni/<suite_name>/test_*.py`. Each test takes
   `workload_run` and asserts one claim against the Manifest.

## Anti-patterns

- **Emitting phase-boundary events from the adapter.** `Job` owns
  `prepare.*`, `parse.done`, `teardown.*` (B6). Adapters emit only
  sub-phase events (`launch.container_up`, `step`, `request`, ...).
- **Catching an exception in a lifecycle method and returning silently.**
  Failure classification happens at the raise site (B4). Let the
  exception escape; `Job` wraps it into the right `WorkloadFailure`
  subclass.
- **Hand-rolling result parsing inline in `parse`.** Keep the JSON ->
  scalars/samples mapping as a `@staticmethod` (like `read_bench_result`)
  so it is unit-testable without a cluster.
- **Reading the config file or binding inside the adapter.** The
  `RunContext` carries the typed config (`ctx.config`), the bound
  hosts (`ctx.bindings`), and the executor (`ctx.executor`). Adapters
  never re-read either file.
- **Re-introducing `SecretValue` / `env_file` / `redact_*` plumbing.**
  Security was removed entirely; the closed-cluster acceptance is
  documented in the addendum.
