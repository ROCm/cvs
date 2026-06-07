# DTNI

DeTermINistic Inference — CVS workload runner for LLM inference validation.

## Docs

| Goal | Doc |
|---|---|
| Get oriented | [docs/overview.md](docs/overview.md) |
| Add a workload variant | [docs/add_workload.md](docs/add_workload.md) |
| Add a benchmark | [docs/add_benchmark.md](docs/add_benchmark.md) |
| Add a harness | [docs/add_harness.md](docs/add_harness.md) |
| Scaffold a new framework | [docs/framework/RUNBOOK.md](docs/framework/RUNBOOK.md) |
| Schema reference | [docs/config_and_thresholds.md](docs/config_and_thresholds.md) |
| End-user docs (rocm.docs) | [docs/sphinx_rst.md](docs/sphinx_rst.md) |
| Distributed pattern | [docs/patterns/distributed.md](docs/patterns/distributed.md) |
| Disagg pattern | [docs/patterns/disagg.md](docs/patterns/disagg.md) |

## Framework deep-dives

| Doc | Purpose |
|---|---|
| [docs/framework/00_overview.md](docs/framework/00_overview.md) | Pointer to RUNBOOK |
| [docs/framework/01_adapter_contract.md](docs/framework/01_adapter_contract.md) | BaseWorkloadAdapter surface |
| [docs/framework/02_phases.md](docs/framework/02_phases.md) | Per-phase invariants |
| [docs/framework/03_artifacts_and_smoke.md](docs/framework/03_artifacts_and_smoke.md) | container_handle + smoke metrics |
| [docs/framework/04_registry_and_test_entrypoint.md](docs/framework/04_registry_and_test_entrypoint.md) | Registry + pytest shim |
| [docs/framework/05_seed_config_and_verification.md](docs/framework/05_seed_config_and_verification.md) | First config + verify |
