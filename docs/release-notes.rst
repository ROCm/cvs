.. meta::
  :description: Release notes for Cluster Validation Suite (CVS) 0.3.0: Run Deck, Slurm and SPUR support, RCCL topology checks, and bug fixes since the previous release.
  :keywords: CVS, ROCm, release notes, changelog, AMD, cluster validation, GPU, AMD Instinct

***************************************************
Cluster Validation Suite (CVS) 0.3.0 release notes
***************************************************

These release notes describe notable changes since the previous CVS release.

Release highlights
==================

This release adds Run Deck output for most suites and first-class Slurm and SPUR support, letting ``cvs run`` launch as a job step without passwordless SSH. RCCL gains topology verification, per-node dmesg scanning during tests, and scale-up bandwidth references. ibperf adds vPOD-aware node pairing and tighter bandwidth enforcement. The legacy dmesg parser is removed; all kernel-log scanning now uses AMD node-scraper.

Added
=====

- Run Deck output for health, platform, preflight, ibperf, MORI, ANC, RCCL, training, and inference suites. Install suites do not produce a Run Deck.
- ``cvs run`` supports Slurm and SPUR: launch with ``srun`` or ``spur run`` and supported suites run without passwordless SSH. See :doc:`/how-to/run-on-managed-compute`.
- ibperf: vPOD-aware node pairing using ``ibperf.pairing_mode``.
- RCCL:

  - Scale-up bandwidth references (``cvs_params.fabric``).
  - Bus-bandwidth tolerance (``cvs_params.bus_bw_tolerance``, default ``0.95``).
  - Topology verification: perf and regression compare the reported topology with the launched topology (``cvs_params.topology_check``, default ``warn``).
  - Per-test dmesg scanning between each test's start and end markers (``cvs_params.dmesg_during_test``, default ``"False"``).
  - Sample config gates scale-out ``all_reduce_perf`` at the published MI350X/MI355X acceptance bar in ``results.thor``; MI300X clusters must override those entries.
  - Single-node 8-GPU guide configs ``mi300x_rccl_single_node_config.json`` and ``mi350x_rccl_single_node_config.json``.

- Platform ``test_check_be_nic_link_speed`` checks backend NIC link speed against ``nic_link_speed`` (default ``400000`` Mb/s).

Changed
=======

- vLLM configs that set ``HF_TOKEN`` or ``HUGGING_FACE_HUB_TOKEN`` in ``container.env`` fail to load; set ``paths.hf_token_file`` instead, pointing to a path mounted in the container on each node.
- ibperf:

  - A node with no partner in the discovered NIC set fails and is excluded, instead of running as a server with no client.
  - Each node's perftest binary must support ROCm DMA-BUF, unless ``ibperf.require_dmabuf`` is ``"False"``.

- RCCL:

  - Perf, regression, and pairwise run through the orchestrator and work with the container backend.
  - Result files default to the run directory; each run verifies the directory is shared before launching. Container runs must mount it at the same host and container path.
  - ``verify_bw_dip`` and ``verify_lat_dip`` default to ``"False"``. Setting either to ``"True"`` with no matching ``results`` reference now fails instead of warning and skipping; re-baseline thresholds first.
  - A ``null`` ``results`` entry means no published threshold and skips the bandwidth gate; absent or empty entries still fail.
  - Graphs and result tables chart rows the checks evaluate, with one series per data type (``<collective> [type=<dtype>]``) when multiple ``data_types`` are configured. Heatmap reference JSONs saved from earlier runs must be regenerated.

- RCCL and Aorta report each check as a pytest sub-test, shown as per-case rows in the HTML report and JUnit XML. Sub-tests add to the JUnit ``tests`` total.
- Custom ``Orchestrator`` subclasses must implement ``exec_on_host``, ``upload_to_head``, ``download_from_head``, and ``download_file``.
- Node Smoke Tier 1, Tier 2, and Tier 3 run by default. To disable, set ``node_smoke_tier1.connectivity_mode`` and ``node_smoke_tier3.connectivity_mode`` to ``"skip"``.
- ``cvs run`` writes a self-contained pytest HTML report and a text log under ``<workspace>/cvs_runs/<run_id>/`` by default. Use ``--no-html`` and ``--no-log-file`` to suppress them.
- Logged command lines show ``<redacted>`` in place of token, password, secret, and API-key variable values; commands sent to nodes are unchanged.

Fixed
=====

- Aorta script mode (the ``aorta_single`` default) now applies ``training_overrides``; previously runs silently used YAML values. The applied config is saved as ``effective_config.yaml``.
- Kernel-log health checks no longer echo each node's full dmesg or journal into the run log and HTML report; each scan logs one summary line.
- ibperf:

  - Runs on nodes with fewer than 8 GPUs instead of failing with ``KeyError: 'card4'``.
  - ``verify_bw`` now enforces ``expected_results``; previously tests passed regardless of measured bandwidth. Configs with ``verify_bw: "True"`` might now fail; re-baseline thresholds for your cluster.
  - The sample config sets ``duration`` to ``10`` (was ``30``); lower it in configs copied from the old sample.

- RCCL:

  - Perf and regression thresholds and dip checks read the configured top-level ``rccl.results`` table; regression reads its collectives from ``rccl_test_params``.
  - NIC- and rank-keyed ``results`` entries are validated like flat entries; invalid values fail with their config path.
  - Bus-bandwidth verification fails when a configured message size has no result row, instead of skipping it.
  - Result files include the collective name, preserving results from every collective instead of only the last.
  - Pairwise and incremental runs no longer apply the full cluster's flat bandwidth thresholds to each sub-run; only NIC- and rank-keyed thresholds carry over.
  - Malformed JSON results are reported, and a run with no result rows fails on missing bandwidth instead of crashing.

- Preflight:

  - Node Smoke rows pass or fail only for checks Primus reported; a tier failure stays in the HTML report when rows do not show it.
  - Failure messages name the hosts that failed, not config sections such as ``vpod_membership``.

- vLLM:

  - DeepSeek V4 sample configs no longer set ``HF_HUB_OFFLINE`` or ``TRANSFORMERS_OFFLINE``, which broke accuracy tasks.
  - Suites accept ``spec_decode_*`` statistics that ``vllm bench serve`` reports, instead of failing the sweep.
  - ``vllm_single`` accepts ``pipeline_parallel_size`` greater than ``1``; ``vllm_distributed`` on one host serves with the configured pipeline width.
  - A server that exits during startup fails the cell immediately, instead of waiting out the readiness timeout.

- vLLM, ATOM, and SGLang no longer reset xGMI error counters when reading GPU metrics.
- Pytest sub-test reports no longer replace the parent test's Full Log link.
- IB topology discovery recognizes AMD Pollara (Ionic) HCA names such as ``ionic_0``.
- Containers with ``runtime.args.privileged: false`` start on hosts without ``/dev/infiniband``; InfiniBand devices are passed only where the host has them.
- Platform ``test_check_dmesg_driver_errors`` no longer fails on ``[Firmware Bug]:`` lines or zero-valued ``WALKER_ERROR``, ``MAPPING_ERROR``, and ``PROTECTION_FAULT_STATUS`` lines; non-zero values still fail.

Removed
=======

- The legacy dmesg parser. All dmesg and kernel-journal scans now use AMD node-scraper. ``CVS_DMESG_PARSER`` is no longer read; remove it from run scripts.
