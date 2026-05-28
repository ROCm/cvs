# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Cluster Validation Suite (CVS)** is a pytest-based framework for validating AMD AI clusters end-to-end: single-node burn-in (AGFHC, RocBLAS, RVS, rocHPL, Transferbench), multi-node RCCL, InfiniBand performance, distributed training (JAX, Megatron-LM, TorchTitan), and distributed inference (vLLM, SGLang). It is distributed as a Python package with a `cvs` CLI entry point and is designed to run from a head node (or any Linux box with SSH reach into the cluster).

## Commands

### Build / Install
```bash
make build          # fmt-check → lint → sdist
make sdist          # source distribution only
make install        # build + create .cvs_venv + install CVS
make cvs-venv       # just the venv
make clean          # remove venvs, build artifacts, caches
```

### Lint / Format
```bash
make fmt            # auto-format with ruff (version pinned in Makefile)
make fmt-check      # check formatting without modifying
make lint           # ruff check + pylint scoped to E1205/E1206 (logging arg counts) on cvs/
make lint-fix       # auto-fix ruff issues (code quality, not formatting)
```

### Tests
```bash
make test           # unit tests + CLI tests — this is what CI runs
make ut             # unit tests only (python run_all_unittests.py)

# Single unit test file
python -m unittest cvs/lib/unittests/test_<module>.py
```

Cluster/integration tests require real infrastructure and run via the CLI, not pytest directly:
```bash
cvs run <test_name> --cluster_file <path> --config_file <path>
```
Both `--cluster_file` and `--config_file` are mandatory and enforced by `cli_plugins/run_plugin.py`.

## Repository Layout

```
cvs/
├── main.py                  # CLI entry point
├── cli_plugins/             # Subcommand plugins (run, list, generate, exec, scp, monitor, copy-config)
├── core/
│   ├── orchestrators/       # Baremetal vs container orchestration (factory.py picks)
│   └── runtimes/            # Runtime abstraction for test execution
├── lib/                     # Reusable library modules — most logic lives here
├── tests/                   # Pytest cluster validation tests; shared hooks in conftest.py
│   └── benchmark/ health/ ibperf/ inference/ mori/ platform/ preflight/ rccl/ training/
├── parsers/                 # Output parsers and Pydantic config schemas (schemas.py)
├── runners/                 # Concrete benchmark runner implementations
├── monitors/                # Cluster monitoring scripts
├── input/                   # Config/cluster file templates (cluster_file/, config_file/, env_file/, generate/)
├── schema/                  # RCCL and other test schemas
└── unittests/               # Top-level unit tests for CLI and extension loading
docs/                        # Published ROCm documentation source
```

### Library Modules at a Glance (`cvs/lib/`)
| Module | Purpose |
|--------|---------|
| `parallel_ssh_lib.py` | `Pssh` class — the parallel-SSH workhorse threaded through every test |
| `parallel/` | Sharded SSH internals (pssh, sharder, multiprocess, scp); auto-engages at 32+ hosts |
| `docker_lib.py` | `launch_docker_container`, `kill_docker_container`, `delete_all_containers_and_volumes` |
| `globals.py` | Global `log` handle and `error_list` |
| `utils_lib.py` | `fail_test`, `update_test_result`, placeholder resolution, `get_model_from_rocm_smi_output`, `json_to_dict` |
| `verify_lib.py` | `verify_dmesg_for_errors`, GPU/NIC/BIOS verification |
| `linux_utils.py` | RDMA / ethtool stats collection |
| `env_lib.py` | Environment variable management |
| `html_lib.py` | HTML report generation |
| `report_plugins.py` | Extensible report generation |
| `scriptlet.py` | Script generation and execution |
| `rocm_plib.py` | ROCm platform utilities |
| `jax_training_lib.py` | JAX distributed training |
| `megatron_training_lib.py` | Megatron-LM distributed training |
| `torchtitan_training_lib.py` | TorchTitan (PyTorch-native) training |
| `rccl_lib.py` | RCCL collective communications validation |
| `ibperf_lib.py` | InfiniBand performance testing |
| `inference_lib.py` + `inference/` | Inference validation (base, vllm, inference_max) |
| `sglang_disagg_lib.py` | SGLang disaggregated serving |
| `mori_lib.py` | Memory and bandwidth testing |
| `preflight/` | Preflight checks (RDMA, GID, interface consistency, version) |

## Architecture

### CLI plugin discovery
`cvs/main.py` dynamically loads subcommands from `cvs/cli_plugins/`. Each plugin subclasses `SubcommandPlugin` (`cli_plugins/base.py`) and self-registers. Plugins include `run`, `list`, `generate`, `exec`, `scp`, `monitor`, `copy-config`.

### Test execution flow
1. `cvs run <test>` → `run_plugin.py` invokes pytest on a file under `cvs/tests/`.
2. `cvs/tests/conftest.py` installs hooks for HTML report management and test metadata.
3. Per-test fixtures load the two required JSON files, run them through placeholder resolution (`resolve_cluster_config_placeholders` / `resolve_test_config_placeholders` in `cvs/lib/utils_lib.py`), and build a `Pssh` handle (`phdl`) over the node list.
4. Tests instantiate workload classes from `cvs/lib/` (e.g., `MegatronLlamaTrainingJob`, `TorchTitanTrainingJob`, `Pssh`) and walk their lifecycle methods.
5. Lib methods shell out via `phdl.exec(cmd)` (same cmd → all nodes) or `phdl.exec_cmd_list([per-node cmds])` (positional pairing with `host_list`), returning `{node: stdout}`.
6. Failures call `fail_test(msg)` from `utils_lib`, which appends to `globals.error_list`; tests end with `update_test_result()` to roll up errors. Tests do **not** raise to signal failure.
7. `cvs/lib/report_plugins.py` and `html_lib.py` render the HTML report.

### Workload class lifecycle
All training job classes (`MegatronLlamaTrainingJob`, `TorchTitanTrainingJob`, JAX equivalents) share the same shape:
```
__init__(phdl, model_name, training_config_dict, model_params_dict,
         hf_token, gpu_type, distributed_training, tune_model_params)
.run_pretraining_tasks()      # snapshot RDMA/ethtool stats (distributed only)
.exec_nic_setup_scripts()     # vendor NIC workarounds inside container
.build_training_job_cmd()     # populate self.job_cmd / self.job_cmd_list
.start_training_job()         # write wrapper scripts + docker exec them
.poll_for_training_completion()
.verify_training_results()    # perf vs expected_result_dict, NaN/Inf, dmesg, net counters
```

### Orchestration layer
`cvs/core/orchestrators/` abstracts where tests run:
- `baremetal.py` — exec directly via SSH on the node
- `container.py` — exec inside a per-node Docker container
- `factory.py` — picks the orchestrator from `OrchestratorConfig` and resolves placeholders like `{user-id}`

Cluster file templates mirror this split: `cluster.json` (baremetal) vs `cluster_container.json` (container). Many existing tests still construct `phdl` and call `docker_lib` directly rather than going through the orchestrator layer — orchestrators are the newer pattern. **For new tests, prefer the orchestrator layer; for modifications, mirror what the file already uses.**

### Configuration model
Every test requires two JSON files:
- **cluster_file** — node IPs/hostnames, SSH user, SSH key, env vars. Templates: `cvs/input/cluster_file/`
- **config_file** — workload parameters (model, batch sizes, NIC names, container image, expected perf thresholds). Templates: `cvs/input/config_file/<category>/`

Both support placeholders (`{user-id}`, `{home-mount-dir}`, etc.). Resolution happens in test fixtures **before** dicts are passed into lib classes — lib classes assume their input is already resolved.

Training configs follow a consistent shape: a `config` block (orchestration: container image, log_dir, NIC settings, nnodes) and a `model_params` block keyed `single_node|multi_node → <model_name> → <gpu_type> → {params..., result_dict}`. `result_dict` holds expected performance thresholds that `verify_*_results` checks against.

Config schemas live in `cvs/parsers/schemas.py` (Pydantic). New config keys must be added to the schema.

### Extension system
Third-party packages can ship tests/configs/generators via an `extension.ini`. CVS discovers them through the `CVS_EXTENSION_PKG_NAMES` env var and `cvs/extension.py`.

### Environment variables
- `CVS_HOSTS_PER_SHARD` (default 32) — hosts processed per parallel-SSH shard
- `CVS_WORKERS_PER_CPU` (default 4) — workers per CPU core (total = CPU × this)
- `CVS_EXTENSION_PKG_NAMES` — comma-separated extension packages to load
- `CLUSTER_FILE` — fallback for `--cluster_file` (used by `cvs exec`)

## Conventions

These patterns are non-obvious and routinely tripped over:

- **Logging.** Use `from cvs.lib import globals; log = globals.log`. Write `log.info("%s", x)` — pylint enforces logging arg counts (E1205/E1206) on `cvs/`; f-strings inside `log.info(...)` lint-fail.
- **Fixture duplication is intentional.** Every training test redefines the same ~8 fixtures (`cluster_file`, `training_config_file`, `cluster_dict`, `training_dict`, `model_params_dict`, `hf_token`, `phdl`, `gpu_type`). This is an existing project convention; don't refactor across test files unless asked.
- **`fail_test` over `raise`.** Failures accumulate in `globals.error_list`; tests keep running. Don't raise from test bodies.
- **`host_list[-1]` is the authoritative log source.** Training libs read the "last node's" logs as source of truth. To change which node is authoritative, update both `megatron_training_lib.py` and `torchtitan_training_lib.py` and call it out in the commit message.
- **Shared constants across training libs are currently duplicated.** `training_err_dict`, `err_counters_pattern`, and `detect_rocm_path` exist as exact copies in `megatron_training_lib.py` and `torchtitan_training_lib.py`. If you change *behavior* in one copy, update the other in the same PR; if you only fix style, leave the sibling alone.
- **String booleans in configs.** Some config values arrive as `'True'`/`'False'` strings (e.g., `verify_network_errors`). Compare with `== 'True'`, not truthiness.
- **Wildcard imports in tests.** `from cvs.lib.utils_lib import *` / `from cvs.lib.verify_lib import *` is standard in tests (`F403`/`F405` are ignored in `pyproject.toml`). Don't replace with explicit imports unless asked.
- **Scripts dir is rebuilt per run.** Training job `__init__` does `rm -rf scripts_dir; mkdir; chmod 777`. Nothing persists across runs.
- **`nproc_per_node = 8`** is hardcoded in training libs (one GPU per process, 8 GPUs per AMD node).
- **Placeholder resolution boundary.** Fixtures resolve placeholders; lib classes consume already-resolved dicts. Don't add placeholder logic inside lib classes.
- **Trace one-line edits before applying them.** Before changing an `__init__` assignment (e.g., `self.X = kwarg` → `self.X = literal`), grep for the LHS attribute and every callsite that passes the corresponding kwarg. Flag any branches that become dead code *before* the edit, not after.
- **Re-Read files before a second Edit in the same turn.** IDE format-on-save and hooks may rewrite a file between your Read and Edit. When editing a file you previously Read in this session, re-Read the target window before any subsequent Edit.
- **Diff against sibling files before calling something "duplication".** When you spot apparent duplication in `cvs/tests/<category>/`, diff against the matching file in a sibling framework (`megatron/` ↔ `torchtitan/` ↔ `jax/`) before proposing consolidation. If they're identical, it's convention — leave it alone.

## Where to look when…

| Task | Start here |
|------|-----------|
| Adding a new training framework | `cvs/lib/<framework>_training_lib.py` mirroring `megatron_training_lib.py` |
| Changing parallel-SSH behavior | `cvs/lib/parallel/` (pssh, sharder, multiprocess) |
| Adding a config field | `cvs/parsers/schemas.py` (Pydantic) + the template under `cvs/input/config_file/` |
| Debugging a failed run | `globals.error_list`, `{log_dir}/<framework>-logs/out-node*/training.log`, dmesg via `verify_dmesg_for_errors` |
| Understanding container vs baremetal | `cvs/core/orchestrators/factory.py` and `cvs/input/cluster_file/README.md` |

## Code style (non-defaults)

- Pylint scoped to logging-arg-count rules (E1205/E1206) on `cvs/` — see "Logging" under Conventions
- Ignored ruff rules: `F403`, `F405` (star imports in tests), `E722` (bare except) — don't "fix" these
- Other rules (line length 120, Python 3.9+, preserve quote style) are enforced by ruff at lint time

## Further reading

- `README.md` — user-facing install/run guide (do not duplicate that content here)
- `UNIT_TESTING_GUIDE.md` — unit-test workflow
- `CONTRIBUTORS.md` — contribution guidelines
- `docs/` — source for published ROCm CVS docs
- `cvs/input/cluster_file/README.md` — cluster-file reference (baremetal vs container)
- Published docs: <https://rocm.docs.amd.com/projects/cvs/en/latest/>
