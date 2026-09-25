.. meta::
  :description: Release notes for Cluster Validation Suite (CVS) 0.2.0: new test suites, container backend, Cluster Health Monitor, and bug fixes since the previous release.
  :keywords: CVS, ROCm, release notes, changelog, AMD, cluster validation, GPU, AMD Instinct

***************************************************
Cluster Validation Suite (CVS) 0.2.0 release notes
***************************************************

These release notes describe notable changes since the previous CVS release.

Release highlights
==================

This release focuses on expanding CVS into a full-featured cluster qualification platform, adding distributed training suites (JAX MaxText, TorchTitan, and Megatron), inference suites (vLLM, SGLang, xDiT, and ATOM), and a container execution backend alongside the existing bare-metal path.

New cluster-wide observability is provided by the Cluster Health Monitor, with support for control-plane monitoring, RCCL Reliability, Availability, and Serviceability (RAS) and Inspector plugins, and an nginx Transport Layer Security (TLS) overlay.

Further, rack-aware execution with switch-tray support, pairwise RCCL tests, and MORI RDMA performance tests extend validation coverage to large-scale cluster topologies.

Added
=====

- Installable Python package layout: tests, lib, and input live under ``cvs/``; ``cvs`` CLI; Makefile install/test targets.
- Training: JAX MaxText, TorchTitan, Megatron (single and multi-node).
- Aorta and benchmark training microbenchmarks with schema, configs, and runner.
- Preflight suite: MI4XX node-health and InfiniBand over Ethernet (IFoE) checks, nodesmoke tiers (including Primus CLI tier 1 and tier 3), ROCm version consistency.
- Inference: vLLM (single and distributed), SGLang (single, distributed, and disaggregated), xDiT (single and distributed), and ATOM (single).
- MORI RDMA performance tests.
- Pairwise RCCL tests.
- Container backend: ``container.lifetime`` schema, ``setup_script`` provisioning, persistent mode, CVS container image.
- Rack-aware execution with switch-tray support.
- Cluster Health Monitor (``cvs/monitors/cluster-mon``): TCP probe, log search, host reconnect, UI without setup-key; control-plane monitoring (Slurm/Kubernetes (K8s)); RCCL Reliability, Availability, and Serviceability (RAS) and Inspector plugins; nginx Transport Layer Security (TLS) overlay for public-IP monitoring VMs.
- Per-test pytest-html reports; RCCL performance reports integrated into the CVS HTML bundle.
- Configurable ``cvs_exec_timeout``, ``rccl-tests -T`` timeout, and ``-A algoproto`` output on ``rccl_perf`` and ``rccl_regression``.
- Unified logging and CLI output across CVS.
- Install, how-to, and configuration documentation on the ROCm docs site.

Changed
=======

- Megatron and JAX training suites moved onto the ``orch`` fixture.
- RCCL suites refactored (perf versus regression parametrization, standardized naming, config file names aligned to ``framework_model_size_single|distributed``).
- Health configs and TransferBench/RVS install paths updated for newer ROCm (including 7.11) and alternate folder layouts (including Rock).
- JAX training library layout and commands updated.
- Automatic per-command sudo fallback (replaces manual ``orch_sudo`` config).
- Dmesg scanning via ``amd-node-scraper`` (``full_dmesg_scan`` / ``CVS_DMESG_PARSER``), with remaining journal scans migrated off the legacy path.
- RDMA device names might include underscores.

Fixed
=====

- RCCL: skip sudo-only checks without passwordless sudo; accept Reduce and HyperCube collectives; post-test JSON save for payloads larger than 30 KB; ``--mca pml`` when OpenMPI is built with Unified Communication X (UCX); oversubscription surfaced as WARN with a per-test dmesg window and per-user hostfile.
- Dmesg error-pattern matching.
- TransferBench output parsing on newer ROCm versions.
- Continue running when some cluster nodes are unreachable.
- HTML report: missing link in the ENV table.
- SSH sessions left open by temporary subset PSSH handles.
