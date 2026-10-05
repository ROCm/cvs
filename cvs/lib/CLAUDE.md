# cvs/lib/ — shared library modules

## Shape of a "lib module"

A `cvs/lib/*.py` module is plain Python imported by one or more
`cvs/tests/<domain>/*.py` test files — it holds helper functions and/or a
"job"/"check" class, never a `test_*` function itself and never reads
`--cluster_file`/`--config_file` directly (that's the test module's/fixture's
job; lib functions take an already-connected `phdl`/`shdl` handle and plain
Python args). Most modules follow one of two shapes:
- a flat set of functions taking `phdl` (parallel-SSH handle, see
  `cvs/lib/parallel/`) and returning parsed dicts (e.g. `docker_lib.py`,
  `rocm_plib.py`, `linux_utils.py`), or
- a class encapsulating a multi-step workload/check with its own state
  (e.g. `JaxTrainingJob` in `jax_training_lib.py`, `MoriBenchmark` in
  `mori_lib.py`, `PreflightCheck` subclasses in `cvs/lib/preflight/`).

## Module map (one line each)

- `anc_lib.py` — shared "AMD Node Check" machinery reused across the ANC CVS
  suites (`cvs/tests/anc/`).
- `docker_lib.py` — inspect/kill Docker containers and images on cluster
  nodes via `phdl`.
- `env_lib.py` — safely build shell-compatible env-var export strings,
  including controlled self-referential expansion (e.g. `PATH`).
- `globals.py` — process-wide `log` (stdlib `logging.getLogger()`) and
  shared mutable state (e.g. `error_list`) used across lib/test modules.
- `html_lib.py` — low-level HTML page/report building blocks (headers,
  footers, RCCL heatmaps/amCharts graphs) used by report generators.
- `ibperf_lib.py` — IB perf test helpers: ROCm path detection, dmabuf
  support checks, bandwidth/latency parsing and verification, consumed by
  `cvs/tests/ibperf/`.
- `inference_lib.py` — thin wrapper importing `InferenceMaxJob`/`VllmJob`
  from `inference_max_lib`, consumed by `cvs/tests/inference/`.
- `inference/` — package with `base.py`, `inference_max.py`, `vllm.py` job
  classes for inference test domains.
- `jax_training_lib.py` — `JaxTrainingJob` class driving JAX distributed
  training runs for `cvs/tests/training/jax/`.
- `linux_utils.py` — general Linux/RDMA introspection helpers (e.g.
  `get_rdma_nic_dict(phdl)` parsing `rdma link` output); imports
  `rocm_plib`/`utils_lib`; widely reused (including by
  `cvs/lib/preflight/interface_consistency.py`).
- `megatron_training_lib.py` — `MegatronLlamaTrainingJob` class and result
  parsing for `cvs/tests/training/megatron/`.
- `mori_lib.py` — `MoriBenchmark` class plus output-table parsing for MORI
  benchmark tests (`cvs/tests/mori/`).
- `node_scraper_adapter.py` — adapter around AMD node-scraper's offline
  dmesg analyzer (CVS still collects raw dmesg over its own parallel-SSH
  layer; this module only does the analysis).
- `parallel/` — the parallel remote-execution layer: `multiprocess_pssh.py`
  (`MultiProcessPssh`, the `phdl` most tests/checks are built on), `pssh.py`,
  `pssh_sharder.py`, `scp.py`, `config.py`, `interfaces.py`. This is the one
  mechanism to reuse for "run a command across cluster nodes" — do not
  hand-roll a new SSH loop.
- `parallel_ssh_lib.py` — deprecated compatibility shim; new code should
  import from `cvs/lib/parallel/` directly instead.
- `preflight/` — the preflight-checks subsystem (base class, one module per
  check, report generator). See `cvs/lib/preflight/CLAUDE.md` for its own,
  more detailed conventions — not repeated here.
- `rccl_lib.py` — RCCL test orchestration helpers: process cleanup, result
  JSON save/load to the head node, MPI/UCX detection, output-flag detection,
  for `cvs/tests/rccl/`.
- `report_plugins.py` — `HtmlReportManager`, the pytest-html integration
  used from `cvs/conftest.py` (per-test log capture, report zip bundling).
- `rocm_plib.py` — ROCm/`amd-smi` JSON command wrappers (`get_rocm_smi_dict`,
  `get_gpu_partition_dict`, `get_amd_smi_fw_dict`, ...).
- `run_config_paths.py` — resolves `run_config`/baseline-CSV paths so they
  work from any clone location or after a pip install.
- `scriptlet.py` — `ScriptLet` class: batches script distribution +
  execution across cluster nodes to minimize SSH round-trips.
- `sglang_disagg_lib.py` — `SglangDisaggPD` class for disaggregated
  prefill/decode SGLang inference tests (`cvs/tests/inference/sglang/`).
- `torchtitan_training_lib.py` — `TorchTitanTrainingJob` class and result
  parsing for `cvs/tests/training/torchtitan/`.
- `utils_lib.py` — general-purpose test utilities: `fail_test`,
  `update_test_result`, `scan_test_results`, JSON/dict conversion helpers;
  imported with `import *` by several test modules.
- `verify_lib.py` — hardware/health verification helpers (PCIe bus
  width/errors, dmesg error scanning, NIC link-flap detection, `lspci`
  checks) used across health/platform/preflight-adjacent checks.

## Unit tests (`cvs/lib/unittests/`)

One `unittest.TestCase` file per lib module (e.g. `test_linux_utils.py`,
`test_rccl_lib.py`, `test_html_lib.py`). Convention: mock the remote-execution
handle rather than hitting real hosts — `phdl = MagicMock()`,
`phdl.exec.return_value = {"node1": "<canned output>"}`, then assert both the
command string passed to `phdl.exec` and the parsed return value. Never spin
up real SSH/subprocess connections in these tests. Run via `make test` (also
executed by `run_all_unittests.py` at the repo root).

`cvs/lib/preflight/unittests/` is the same convention applied to the
preflight subsystem specifically — see `cvs/lib/preflight/CLAUDE.md`.
