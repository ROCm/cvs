# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Cluster Validation Suite (CVS)** is a pytest-based testing framework for validating AMD AI clusters — from single-node burn-in to multi-node distributed training and inference. It is distributed as a Python package with a `cvs` CLI entry point.

## Project Structure

```
cvs/
├── main.py                  # CLI entry point
├── cli_plugins/             # Subcommand plugins (run, list, generate, exec, scp, monitor)
├── core/
│   ├── orchestrators/       # Baremetal and container orchestration
│   └── runtimes/            # Runtime abstraction for test execution
├── lib/                     # Reusable library modules
│   ├── docker_lib.py        # Docker container orchestration helpers
│   ├── env_lib.py           # Environment variable management
│   ├── globals.py           # Global constants
│   ├── html_lib.py          # HTML report generation
│   ├── ibperf_lib.py        # InfiniBand performance testing
│   ├── inference_lib.py     # Inference validation logic
│   ├── jax_training_lib.py  # JAX distributed training
│   ├── linux_utils.py       # Linux host utilities
│   ├── megatron_training_lib.py  # Megatron-LM distributed training
│   ├── mori_lib.py          # Memory and bandwidth testing
│   ├── parallel_ssh_lib.py  # Parallel SSH entry point
│   ├── rccl_lib.py          # RCCL collective communications validation
│   ├── report_plugins.py    # Extensible report generation
│   ├── rocm_plib.py         # ROCm platform utilities
│   ├── scriptlet.py         # Script generation and execution
│   ├── sglang_disagg_lib.py # SGLang disaggregated serving
│   ├── utils_lib.py         # General utilities
│   ├── verify_lib.py        # GPU, NIC, BIOS verification
│   ├── inference/           # Inference sub-library (base, vllm, inference_max)
│   ├── parallel/            # Parallel SSH internals (pssh, pssh_sharder, multiprocess_pssh, scp)
│   └── preflight/           # Preflight checks (RDMA, GID, interface consistency, version check)
├── tests/                   # Cluster validation tests (pytest)
│   ├── benchmark/
│   ├── health/
│   ├── ibperf/
│   ├── inference/
│   ├── mori/
│   ├── platform/
│   ├── preflight/
│   ├── rccl/
│   └── training/
├── parsers/                 # Output parsers and Pydantic config schemas
├── runners/                 # Concrete benchmark runner implementations
├── monitors/                # Cluster monitoring scripts
├── input/                   # Config/cluster file templates
│   ├── cluster_file/
│   ├── config_file/
│   ├── env_file/
│   └── generate/
├── schema/                  # RCCL and other test schemas
└── unittests/               # Top-level unit tests for CLI and extension loading
```

## Commands

### Build
```bash
make build          # fmt-check → lint → sdist
make sdist          # build source distribution only
```

### Lint & Format
```bash
make fmt            # auto-format with ruff
make fmt-check      # check formatting without modifying
make lint           # ruff check + pylint (logging arg counts on cvs/)
make lint-fix       # auto-fix ruff issues
```

### Test
```bash
make test           # unit tests + CLI tests (what CI runs)
make ut             # unit tests only (python run_all_unittests.py)
```

To run a single unit test file:
```bash
python -m unittest cvs/lib/unittests/test_<module>.py
```

Cluster/integration tests require real infrastructure and are invoked via the CLI:
```bash
cvs run <test_name> --cluster_file <path> --config_file <path>
```

### Environment
```bash
make cvs-venv       # create .cvs_venv and install CVS
make clean          # remove all venvs, build artifacts, and cache
```

## Architecture

### CLI Plugin System
`cvs/main.py` loads subcommands dynamically from `cvs/cli_plugins/`. Each plugin inherits from `SubcommandPlugin` (`cvs/cli_plugins/base.py`) and registers itself. Key plugins:

- **run_plugin.py** — wraps pytest; enforces `--cluster_file` and `--config_file` args
- **list_plugin.py** — discovers tests across core + extension packages
- **generate_plugin.py** — generates cluster/config file templates
- **exec_plugin.py** / **scp_plugin.py** — parallel SSH execution and file copy across cluster nodes

### Orchestration Layer
`cvs/core/orchestrators/` abstracts baremetal vs. container execution:
- `factory.py` builds the right orchestrator from config
- `baremetal.py` / `container.py` implement the abstract interface in `base.py`
- `factory.py` resolves `{user-id}` and similar placeholders in `OrchestratorConfig`

### Test Execution Flow
1. `cvs run <test>` → run_plugin invokes pytest on the target test file
2. `cvs/conftest.py` hooks set up HTML report management and test metadata
3. Test functions in `cvs/tests/` load configs via pytest fixtures
4. Library functions in `cvs/lib/` execute SSH, run benchmarks, and parse results
5. `cvs/lib/report_plugins.py` / `html_lib.py` generate HTML reports

### Test Categories (`cvs/tests/`)
| Directory | What it validates |
|-----------|------------------|
| `health/` | Single-node burn-in (AGFHC, RocBLAS, RVS) |
| `rccl/` | Multi-node GPU collective communications |
| `training/` | Distributed training — JAX and Megatron-LM |
| `inference/` | vLLM, SGLang, PyTorch inference |
| `ibperf/` | InfiniBand network performance |
| `benchmark/` | Aorta benchmark runner |
| `mori/` | Memory and bandwidth |
| `platform/` | Host OS, BIOS, firmware checks |
| `preflight/` | Pre-test cluster validation |

### Library Modules (`cvs/lib/`)
Reusable functional modules consumed by both tests and CLI tools:
- `parallel_ssh_lib.py` — sharded parallel SSH across large node counts
- `rccl_lib.py`, `jax_training_lib.py`, `megatron_training_lib.py` — workload-specific logic
- `docker_lib.py` — container orchestration helpers
- `verify_lib.py` — GPU, NIC, BIOS verification utilities
- `schemas.py` (in `cvs/parsers/`) — Pydantic models for all JSON/YAML config validation

### Configuration System
All tests are driven by two required files:
- **cluster_file** — cluster topology (IPs, hostnames, SSH keys); templates in `cvs/input/cluster_file/`
- **config_file** — test-specific parameters; templates in `cvs/input/config_file/`

Use `cvs copy-config` to copy sample templates, and `cvs generate` to scaffold new ones.

### Extension System
Third-party packages can add tests, configs, and generators by shipping an `extension.ini`. CVS discovers extensions via the `CVS_EXTENSION_PKG_NAMES` environment variable and the `cvs/extension.py` loader.

## Code Style
- Line length: 120 characters (ruff)
- Target: Python 3.9+
- Quote style: preserve (ruff)
- Pylint enforces correct logging argument counts (E1205/E1206) in `cvs/`
