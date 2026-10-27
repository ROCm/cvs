# Changelog

All notable changes to ROCm Cluster Validation Suite (CVS) are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - Unreleased

### Added

- Run Deck for most suites: health, platform, preflight, ibperf, MORI, ANC, ARC, RCCL, training and inference runs write a Run Deck beside the pytest HTML report. Install suites do not.
- `cvs run` supports Slurm and SPUR: launch it with `srun` or `spur run`, and supported suites run without passwordless SSH.
- ibperf bandwidth and latency tests support vPOD-aware node pairing (`ibperf.pairing_mode`).
- RCCL:
  - Scale-up bandwidth references (`cvs_params.fabric`).
  - Bus-bandwidth tolerance (`cvs_params.bus_bw_tolerance`, default `0.95`).
  - Perf and regression compare the reported topology with the launched topology (`cvs_params.topology_check`, default `warn`).
  - Perf and regression can scan each node's dmesg between that test's start and end markers (`cvs_params.dmesg_during_test`, default `"False"`).
  - The sample config gates scale-out `all_reduce_perf` at the published MI350X/MI355X acceptance bar in `results.thor`; MI300X clusters must override those entries. `verify_bus_bw` remains `"False"` by default.
  - Single-node 8-GPU guide configs `mi300x_rccl_single_node_config.json` and `mi350x_rccl_single_node_config.json`.
- Platform `test_check_be_nic_link_speed` checks backend NIC link speed against `nic_link_speed` (default `400000` Mb/s). Existing configs fail on NICs running at another speed, or on a node with fewer backend NICs unless `nic_link_interfaces` is set.

### Changed

- vLLM configs that set `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` in `container.env` now fail to load; set `paths.hf_token_file` instead, on a path mounted in the container on each node.
- ibperf:
  - A node with no partner in the discovered NIC set fails and is excluded, instead of running as a server with no client.
  - Each node's perftest binary must support ROCm DMA-BUF, unless `ibperf.require_dmabuf` is `"False"`.
- RCCL:
  - Perf, regression and pairwise run through the orchestrator, so they also run with the container backend.
  - Result files default to the run directory, and each run checks that the directory is shared before launching. Container runs need it mounted at the same host and container path.
  - The shipped `verify_bw_dip` and `verify_lat_dip` defaults are now `"False"`. With either set to `"True"` and no matching `results` reference, the test fails instead of warning and skipping; re-baseline thresholds first.
  - A `null` `results` entry means "no published threshold" and skips the bandwidth gate; absent or empty entries still fail.
  - Graphs and result tables chart the rows the checks evaluate, with one series per data type (`<collective> [type=<dtype>]`) when several `data_types` are configured. Heatmap reference JSONs saved from such runs must be regenerated.
- RCCL and Aorta report each check as a pytest sub-test, shown as per-case rows in the HTML report and in JUnit XML. Sub-tests add to the JUnit `tests` total, and a failed one adds its own `<failure>`.
- Custom `Orchestrator` subclasses must implement `exec_on_host`, `upload_to_head`, `download_from_head` and `download_file`.
- Node Smoke Tier 1, Tier 2 and Tier 3 now run by default. To keep them off, set `node_smoke_tier1.connectivity_mode` and `node_smoke_tier3.connectivity_mode` to `"skip"`.
- `cvs run` writes a self-contained pytest HTML report and a text log under `<workspace>/cvs_runs/<run_id>/` by default; `--no-html` and `--no-log-file` suppress them.
- Generic ANC install and log helpers moved to `cvs/lib/amdtools_lib.py`; `anc_lib` names unchanged.
- Logged command lines show `<redacted>` in place of the values of token, password, secret and API-key variables and flags; the commands sent to nodes are unchanged.

### Fixed

- Aorta script mode (the `aorta_single` default) now applies `training_overrides`; previously runs silently used the YAML values. The applied config is saved as `effective_config.yaml`.
- Kernel-log health checks no longer echo each node's full dmesg or journal into the run log and HTML report. Each scan logs one summary line, and detected errors are still reported.
- jaxmaxtext no longer writes the Hugging Face token into the per-node launcher script or the tokenizer-download command; it reads the token from `paths.hf_token_file` inside the container at run time. The token file must be on a path mounted in the container on each node.
- ibperf:
  - Runs on nodes with fewer than 8 GPUs instead of failing with `KeyError: 'card4'`.
  - `verify_bw` now enforces `expected_results`; previously tests passed regardless of measured bandwidth. Configs with `verify_bw` set to `"True"` may now fail; re-baseline thresholds for your cluster.
  - The sample config sets `duration` to `10` (was `30`), which caused failures; lower it in configs copied from the old sample.
- RCCL:
  - Perf and regression thresholds and dip checks read the configured top-level `rccl.results` table, and regression reads its collectives from `rccl_test_params`.
  - NIC- and rank-keyed `results` entries are validated like flat entries, and invalid values fail with their config path.
  - Bus-bandwidth verification fails when a configured message size has no result row, instead of skipping it.
  - Perf and regression result files include the collective name, so a run keeps every collective's results instead of only the last. Tools that read the old result paths must use the new names.
  - Pairwise and incremental runs no longer apply the full cluster's flat bandwidth thresholds to each sub-run; only NIC- and rank-keyed thresholds carry over.
  - Malformed JSON results are reported, and a run with no result rows fails on missing bandwidth, instead of crashing.
- Preflight:
  - Node Smoke rows pass or fail only for checks Primus reported, and a tier failure stays in the HTML report when the rows do not show it.
  - Failure messages name the hosts that failed, not config sections such as `vpod_membership`.
- vLLM:
  - DeepSeek V4 sample configs no longer set `HF_HUB_OFFLINE` or `TRANSFORMERS_OFFLINE`, which broke accuracy tasks.
  - Suites accept the `spec_decode_*` statistics that `vllm bench serve` reports, instead of failing the sweep.
  - `vllm_single` accepts `pipeline_parallel_size` greater than 1, and `vllm_distributed` on one host serves with the configured pipeline width.
  - A server that exits during startup fails the cell at once, instead of waiting out the readiness timeout.
- vLLM, ATOM and SGLang no longer reset the xGMI error counters when reading GPU metrics.
- Pytest sub-test reports no longer replace the parent test's Full Log link.
- IB topology discovery recognizes AMD Pollara (Ionic) HCA names such as `ionic_0`.
- Containers with `runtime.args.privileged: false` start on hosts without `/dev/infiniband`; InfiniBand devices are passed only where the host has them.
- The platform `test_check_dmesg_driver_errors` check no longer fails on err-level `[Firmware Bug]:` lines or zero-valued `WALKER_ERROR`, `MAPPING_ERROR` and `PROTECTION_FAULT_STATUS` lines. Non-zero values still fail.

### Removed

- The legacy dmesg parser. Every dmesg and kernel-journal scan, including the platform `test_check_dmesg_driver_errors` check, now uses AMD node-scraper. `CVS_DMESG_PARSER` is no longer read; remove it from run scripts.

## [0.2.0] - 2026-09-23

### Added

- Installable Python package layout: tests, lib, and input live under `cvs/`; `cvs` CLI; Makefile install/test targets.
- Training: JAX MaxText, TorchTitan, Megatron (single & multi-node).
- Aorta / benchmark training microbenchmarks with schema, configs, and runner.
- Preflight suite: MI4XX node-health and IFoE checks, nodesmoke tiers (including Primus CLI tier 1 and tier 3), ROCm version consistency.
- AMD Node Check (ANC) suite: CPU/GPU group suites, multi-format install, HTML reports, fail-fast install/ldconfig, inactivity timeouts.
- Inference: vLLM (single & distributed), SGLang (single, distributed and disaggregated), xDiT (single & distributed and ATOM (single).
- MORI RDMA performance tests.
- Pairwise RCCL tests.
- Container backend: `container.lifetime` schema, `setup_script` provisioning, persistent mode, CVS container image.
- Rack-aware execution with switch-tray support.
- Cluster Health Monitor (`cvs/monitors/cluster-mon`): TCP probe, log search, host reconnect, UI without setup-key; control-plane monitoring (Slurm/K8s); RCCL RAS and Inspector plugins; nginx TLS overlay for public-IP monitoring VMs.
- Per-test pytest-html reports; RCCL performance reports integrated into the CVS HTML bundle.
- Configurable `cvs_exec_timeout`, rccl-tests `-T` timeout, and `-A` algoproto output on `rccl_perf` / `rccl_regression`.
- Unified logging and CLI output across CVS.
- Install, how-to, and configuration documentation on the ROCm docs site.

### Changed

- Megatron and JAX training suites moved onto the `orch` fixture.
- RCCL suites refactored (perf vs regression parametrization, standardized naming, config file names aligned to `framework_model_size_single|distributed`).
- Health configs and TransferBench/RVS install paths updated for newer ROCm (including 7.11) and alternate folder layouts (including Rock).
- JAX training library layout and commands updated.
- Automatic per-command sudo fallback (replaces manual `orch_sudo` config).
- Dmesg scanning via amd-node-scraper (`full_dmesg_scan` / `CVS_DMESG_PARSER`), with remaining journal scans migrated off the legacy path.
- RDMA device names may include underscores.

### Fixed

- RCCL: skip sudo-only checks without passwordless sudo; accept Reduce and HyperCube collectives; post-test JSON save for payloads larger than 30 KB; `--mca pml` when OpenMPI is built with UCX; oversubscription surfaced as WARN with a per-test dmesg window and per-user hostfile.
- Dmesg error-pattern matching.
- TransferBench output parsing on newer ROCm versions.
- Continue running when some cluster nodes are unreachable.
- HTML report: missing link in the ENV table.
- SSH sessions left open by temporary subset PSSH handles.

## [0.1.0] - 2025-12-14

First public CVS release (`release/cvs-0.1.0`).

### Added

- Platform tests: host OS, BIOS, firmware/driver, and network configuration checks.
- Burn-in health tests: AGFHC, TransferBench, RVS, single-node RCCL.
- Network tests: ping checks and multi-node RCCL collectives.
- IB Perf bandwidth tests.
- Distributed and single-node training: JAX and Megatron for Llama 3.1 8B / 70B / 405B (MI300 / MI350 / MI355 configs), including MI300 405B distributed configs and expected tokens/sec for 70B.
- PyTest runner launched from a head node over SSH, with parallel-SSH for cluster-wide single-node tests.
- Cluster JSON and per-suite config JSON inputs, with `<changeme>` / `{user-id}` placeholder resolution (fail fast on unresolved placeholders).
- `version.txt` and CVS version in the HTML report.
- Recommended AINIC environment settings and extra RCCL env vars sourced from an env file.
- Optional `NCCL_SOCKET_IFNAME` pass-through; OpenMPI older than 5.x uses `--mca oob_tcp_if_include`.
- `env_source_script` compatibility; Hugging Face token from `hf_token_file`; Megatron master address localhost handling.
- Custom `gst_single.conf` for MI355X / MI350X.
