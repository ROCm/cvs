.. meta::
  :description: Run ATOM LLM inference benchmarks with CVS on AMD Instinct GPUs for single-node and multinode pipeline-parallel deployments using ROCm.
  :keywords: CVS, ATOM, inference, benchmark, AMD Instinct, ROCm, AMD, GPU, LLM, multinode, pipeline parallel, DeepSeek

*********************************************************************
Run ATOM LLM inference benchmarks with Cluster Validation Suite (CVS)
*********************************************************************

The ATOM suite benchmarks LLM serving on AMD Instinct GPUs using the ATOM stack
(``atom.entrypoints.openai_server`` + ``atom.benchmarks.benchmark_serving`` on
single-node variants, or a PP coordinator with ``params.driver: vllm_atom`` /
``sglang`` on multinode stems). One suite name — ``atom`` — covers single-node
and multinode pipeline-parallel runs; topology comes from the config and cluster
file.

This page walks through a first run. For the full schema, profiles, thresholds,
and parameter reference, see :doc:`/reference/configuration-files/inference/atom`.

Prerequisites
=============

In addition to the :ref:`common inference prerequisites <inference-common-prerequisites>`:

- The container image must provide the selected serving stack: native ATOM,
  vLLM-ATOM, or SGLang.
- When ``model.remote`` is ``0``, place the weights at ``paths.models_dir``.
- For multinode pipeline parallelism, mount the shared log path on every node.

Run ATOM tests
==============

Copy a config template, fill in your cluster details and model paths, then
run the suite. CVS produces a pytest HTML report and a Run Deck with
per-cell results and bandwidth charts.

Copy config and cluster files
------------------------------

List the shipped ATOM configs to find the template for your GPU and model:

.. code:: bash

  cvs config list inference/atom

Copy each variant into its own subdirectory so only one ``*threshold.json``
sits beside the config you pass to ``--config_file``:

.. code:: bash

  SINGLE_DIR=~/input/config_file/inference/atom/single
  mkdir -p "$SINGLE_DIR"

  cvs config copy inference/atom/mi3xx_atom_deepseek-r1_fp8_single.json \
    --output "$SINGLE_DIR/mi3xx_atom_deepseek-r1_fp8_single.json"
  cvs config copy inference/atom/mi325x_atom_deepseek-r1_fp8_single_threshold.json \
    --output "$SINGLE_DIR/mi325x_atom_deepseek-r1_fp8_single_threshold.json"
  cvs config copy cluster_file/atom_cluster.json --output ~/input/cluster_file/atom_cluster.json

Config stems use ``mi3xx_*`` (gfx942) or ``mi355x_*`` (gfx950). Threshold
files use ``mi325x_*`` or ``mi355x_*`` — match the platform you calibrated on.

Edit placeholders
-----------------

Replace cluster node IPs and trim ``node_dict`` to one host for single-node
runs (two hosts for distributed PP). In the config, set at minimum:

- ``container.image`` — your ATOM ROCm image.
- ``container.runtime.args.volumes`` — replace ``<changeme-models-mount>`` with
  the host models directory (for example ``/it-share-prj2-1/models``).
- ``paths.shared_fs``, ``paths.log_dir``, ``paths.hf_token_file``.
- ``model.id`` — model under test.

``paths.models_dir`` is ``/models``, the in-container weights mount. Keep it
as shipped and customize only the host side of the models volume. ATOM
defaults ``HF_HUB_CACHE`` to that path for Hub weight lookups. SGLang
``bench_serving`` with ``dataset_name=random`` downloads ShareGPT into
``HF_HUB_CACHE``; Qwen SGLang samples override ``HF_HUB_CACHE`` and
``HF_HOME`` to ``{paths.shared_fs}/.cache/huggingface``.

For multinode PP, also set:

- ``params.nnodes`` — number of hosts; keep ``node_dict`` the same length.
- ``params.master_addr`` — head-node address reachable from every node.
- ``roles.server.ib_hca_devices`` and ``roles.server.ib_netdev`` — verify
  automatic fabric discovery or set these explicitly. The netdev must be a
  network interface name, not an ``mlx5_*`` HCA device name.

.. tip::

  Leave ``enforce_thresholds: false`` on first runs until thresholds are
  calibrated for your hardware. MI355X shipped stems are often record-only.

Run the suite
-------------

Single-node W1 (``driver=atom``):

.. code:: bash

  cvs run atom \
    --cluster_file ~/input/cluster_file/atom_cluster.json \
    --config_file "$SINGLE_DIR/mi3xx_atom_deepseek-r1_fp8_single.json" \
    -vvv

Multinode PP (``driver=vllm_atom``):

.. code:: bash

  cvs run atom \
    --cluster_file ~/input/cluster_file/atom_cluster.json \
    --config_file ~/input/config_file/inference/atom/distributed/mi3xx_atom_vllm_deepseek-r1_fp8_distributed.json \
    --html ~/cvs_results/atom-w1-distributed.html --self-contained-html -vvv

MTP-3 speculative decode (``schema_version: 2`` profile):

.. code:: bash

  cvs run atom \
    --cluster_file ~/input/cluster_file/atom_cluster.json \
    --config_file "$SINGLE_DIR/mi3xx_atom_deepseek-r1_fp8_single.json" \
    --config_profile mtp3 \
    -vvv

vLLM and SGLang parity configs live in ``inference/atom/`` (for example
``mi3xx_atom_vllm_deepseek-r1_fp8_single.json``) and run with ``cvs run atom``,
not ``cvs run vllm`` or ``cvs run sglang``.

To smoke-test a single sweep cell, pass ``-k``, for example
``-k "w1_1k_1k-conc128"``.

After ``git pull``, run ``make install`` before
``source .cvs_venv/bin/activate``. On SPUR or Slurm, run ``make install``
on the GPU node inside the job step if the login node cannot create a venv.

Run on managed compute
~~~~~~~~~~~~~~~~~~~~~~

Follow :doc:`/how-to/run-on-managed-compute` to run ATOM as a Slurm or SPUR
job step.

Test lifecycle
==============

Tests run in a fixed order. ``[cell]`` = one row per sweep cell; ``[cell-tier]`` =
one row per metric tier per cell.

.. list-table::
   :widths: 1 3 6
   :header-rows: 1

   * - Order
     - Test
     - Purpose
   * - 1
     - ``test_launch_container``
     - Launch and verify the container
   * - 2
     - ``test_setup_sshd``
     - Multinode SSH setup (skipped on single-node)
   * - 3
     - ``test_discover_topology``
     - Resolve IB HCAs and socket netdev
   * - 4
     - ``test_model_fetch``
     - Verify model cache on nodes
   * - 5
     - ``test_atom_inference[cell]``
     - Server start (or reuse), bench client, parse results
   * - 6
     - ``test_cell_metrics[cell-tier]``
     - Threshold PASS/FAIL per metric tier
   * - 7
     - ``test_print_results_table``
     - Console summary tables
   * - 8
     - ``test_teardown``
     - Tear down the container

On inference failure, ``lifecycle.failed`` skips downstream cells and metric
rows. When ``reuse_server_across_sweep: true``, a warm server is kept across cells
with the same ``server_session_key``.

Sweeps and metrics
==================

Each **sweep cell** is one ``(ISL, OSL, concurrency)`` pair from ``sweep.runs``.
Parametrize IDs look like ``w1_1k_1k-conc128`` or ``w1_1k_1k-conc128-throughput``.

Threshold **cell keys** (must match the sibling threshold file):

- Single-node: ``ISL=1024,OSL=1024,TP=8,PP=1,CONC=128``
- Multinode PP: ``ISL=1024,OSL=1024,TP=8,PP=2,CONC=128``

Each ``test_cell_metrics[cell-tier]`` gates one tier when ``enforce_thresholds:
true``:

.. list-table::
   :widths: 2 6
   :header-rows: 1

   * - Tier
     - Example metrics (bare names in thresholds)
   * - ``throughput``
     - ``output_throughput``, ``per_gpu_throughput``, …
   * - ``ttft`` or ``tpot``
     - ``mean_ttft_ms``, ``p99_ttft_ms``, …
   * - ``health``
     - ``success_rate``, ``failed``
   * - ``scaling``
     - ``scaling.efficiency_pct`` (multinode)
   * - ``record``
     - Remaining metrics — logged, not gated

Benchmark artifacts still expose ``client.*`` keys internally; threshold JSON
uses bare metric names. ATOM might omit tail percentiles even when
``metric_percentiles`` requests them — only present metrics are gated.

Reports and logs
================

- **pytest HTML** — one row per lifecycle stage and per metric tier.
- **Console tables** — ``test_print_results_table`` prints per-cell throughput and latency.
- **Run Deck** — ``atom_run_deck.html``, ``.json``,
  and ``_viewer.html``. See :ref:`pytest-html-report-and-bundle` for generic Run
  Deck behavior and packaging.
- **Per-cell logs** — under ``paths.log_dir`` on cluster nodes (server + client logs).

Launcher vs GPU node
====================

SSH and jumphost labs: ``cvs run`` and the venv live on the **launcher**. SPUR or
Slurm managed labs: ``cvs run`` and ``make install`` run **inside the job step**
on the GPU node (shared home for ``~/input/`` and reports).

.. list-table::
   :widths: 4 2 2 3
   :header-rows: 1

   * - Item
     - SSH launcher
     - GPU node (SSH lab)
     - GPU node (SPUR or Slurm step)
   * - ``cvs run``, venv, reports
     - Yes
     - No
     - Yes
   * - ``~/input/`` (shared home)
     - Yes
     - If NFS-mounted
     - If NFS-mounted
   * - ``priv_key_file`` (unmanaged SSH)
     - Yes
     - No
     - No (HTTP agents)
   * - HF token file
     - Yes
     - If NFS-mounted
     - If NFS-mounted
   * - Models host mount (``<changeme-models-mount>`` → ``/models`` in container)
     - No
     - Yes
     - Yes
   * - Container image, ``sudo docker``
     - No
     - Yes
     - Yes
   * - ``~/LOGS/`` (volume mount)
     - No
     - Yes
     - Yes

Related resources
=================

- :doc:`/reference/configuration-files/inference/atom` — configuration schema and thresholds
- :doc:`/reference/cluster/cluster-file` — cluster file format
- :doc:`/how-to/run-with-containers` — container backend
- :doc:`/how-to/test-suites/index` — other CVS suites
