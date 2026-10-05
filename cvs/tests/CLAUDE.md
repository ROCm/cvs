# cvs/tests/ — test suite orientation

## Shape of a test domain

Each `cvs/tests/<domain>/` is one or more pytest modules discovered by
`cli_plugins/list_plugin.py::discover_tests()` (see `cvs/CLAUDE.md`) —
one runnable "test name" per `.py` file (stem of the filename), excluding
`__init__.py`/`conftest.py` and anything under a `resources/` subdirectory.
A domain can be a single flat file (`benchmark/test_aorta.py`, `mori/
mori_benchmark_test.py`), several sibling files (`rccl/rccl_pairwise.py`,
`rccl_perf.py`, `rccl_regression.py`), or nested subpackages grouping
related suites (`anc/cpu/`, `anc/gpu/`; `inference/{inferencemax,
pytorch_xdit,sglang,vllm}/`; `training/{jax,megatron,torchtitan}/`). Most
domains ship their own `README.md` documenting that domain's checks/config
in detail — check there before `cvs/lib/CLAUDE.md`/source for domain
specifics.

## The cluster_file / config_file / fixture pattern

Every domain is invoked as `cvs run <test_name> --cluster_file <f>
--config_file <f>`; the two options are registered once, globally, in
`cvs/conftest.py::pytest_addoption`. From there the pattern is:

1. Module- or session-scoped fixtures (defined per-domain, e.g. in the test
   module itself or a domain `conftest.py`) call
   `pytestconfig.getoption("cluster_file")`/`("config_file")` and
   `json.load` them into `cluster_dict`/`config_dict`.
2. `config_dict` (and sometimes `cluster_dict`) is validated/normalized
   through a pydantic model in `cvs/parsers/schemas.py` — e.g.
   `ClusterConfigFile`, `PreflightConfigFile`, `AortaBenchmarkConfigFile`.
   Unknown keys / missing required fields raise before any node is touched.
   Some domains (e.g. preflight) additionally accept legacy/deprecated
   config paths and migrate them with a logged warning rather than failing.
3. A fixture builds the remote-execution handle (`phdl`/`shdl`, backed by
   `cvs/lib/parallel/multiprocess_pssh.py::MultiProcessPssh`) from
   `cluster_dict`, and yields it to the test functions or lib check/job
   classes. Some domains additionally use the `orch` fixture
   (`cvs/tests/conftest.py`) from `cvs/core/orchestrators/` to set up a
   container-based runtime before tests execute.
4. Test functions call into `cvs/lib/<domain>_lib.py` (or
   `cvs/lib/preflight/`, `cvs/lib/inference/`, etc.) helpers/classes with
   `phdl`/`shdl` plus plain config values — the lib layer owns command
   construction and result parsing; the test layer owns fixtures,
   assertions/result aggregation, and report triggering.
5. `cvs/conftest.py` hooks (`pytest_sessionstart`, `pytest_runtest_makereport`,
   `pytest_sessionfinish`) wire up `HtmlReportManager`
   (`cvs/lib/report_plugins.py`) for per-test log capture and a final zip
   bundle, independent of any domain-specific HTML report a suite builds on
   top (e.g. preflight's `PreflightReportGenerator`).

`cvs/tests/conftest.py` itself only defines the `orch` fixture; most
domain-specific fixtures live in the test module or a nested `conftest.py`
(e.g. `cvs/tests/anc/conftest.py`) rather than here.

## Domain map (one line each)

- `anc/` — "AMD Node Check" validation suite (root-required); `cpu/`/`gpu/`
  subpackages plus `anc_installation.py`.
- `benchmark/` — Aorta-based distributed training benchmark in a Docker
  container with RCCL; validates iteration time, compute ratio, overlap
  ratio, rank balance against configurable thresholds.
- `health/` — single-node burn-in diagnostics (AGFHC, CSP-qual AGFHC, RVS,
  TransferBench) validating hardware/firmware functionality and performance
  against reference bandwidth/latency numbers; burn-in tools themselves are
  not vendored in-repo to avoid version coupling.
- `ibperf/` — InfiniBand bandwidth/latency perf tests
  (`ib_perf_bw_test.py`) plus tool installation (`install_ibperf_tools.py`).
- `inference/` — distributed inference workloads: `inferencemax/`,
  `pytorch_xdit/`, `sglang/`, `vllm/`.
- `mori/` — MORI benchmark test (`mori_benchmark_test.py`).
- `ompi/` — OpenMPI installation (`install_ompi.py`).
- `platform/` — host OS/BIOS/firmware/driver/network config checks
  (`host_configs_cvs.py`).
- `preflight/` — pre-workload cluster validation (node health, RDMA, IFoE,
  connectivity). See `cvs/tests/preflight/CLAUDE.md` for its own
  conventions — this is the most actively developed domain in this
  worktree.
- `rccl/` — RCCL collective performance/regression/pairwise tests;
  entry points documented in `rccl/README.md`
  (`cvs.lib.rccl_lib.run_rccl`, etc.).
- `training/` — distributed training: `jax/`, `megatron/`, `torchtitan/`.

## Adding a new domain/suite

Prefer copying the fixture/config-validation pattern from the closest
existing domain above rather than inventing a new one; see `cvs/CLAUDE.md`
for the full "adding a new test domain" checklist (schema, sample config,
`cvs/lib/parallel/` reuse).
