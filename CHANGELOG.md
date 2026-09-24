# Changelog

All notable changes to ROCm Cluster Validation Suite (CVS) are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- RCCL: compare reported topology with launcher settings in performance and regression runs, with configurable `warn` (default), `strict`, and `off` modes and a separate JSON audit that preserves producer metadata. Count GPUs across all threads in regression launches, and skip topology checks when every row omits topology.
- RCCL: accept single-node results without a topology block, reject mixed result shapes regardless of row order, and report malformed JSON result structures without crashing row processing.
- RCCL: return cleanly when a run produces no result rows instead of raising `IndexError` from the bandwidth-dip check. Log scanning still fails these runs for missing bandwidth numbers.

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
