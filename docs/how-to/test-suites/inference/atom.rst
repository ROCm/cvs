.. meta::
  :description: Run ATOM inference benchmarks with CVS, single-node and multinode
  :keywords: CVS, ATOM, inference, benchmark, multinode, LLM, ROCm

******************************
Run ATOM inference benchmarks
******************************

The ATOM suite benchmarks LLM serving on AMD Instinct GPUs using the ATOM stack
(``atom.entrypoints.openai_server`` + ``atom.benchmarks.benchmark_serving`` on
single-node variants, or a PP coordinator with ``params.driver: vllm_atom`` /
``sglang`` on multinode stems). One suite name ΓÇö ``atom`` ΓÇö covers single-node
and multinode pipeline-parallel runs; topology comes from the config and cluster
file.

This page walks through a first run. For the full schema, profiles, thresholds,
and parameter reference, see :doc:`/reference/configuration-files/inference/atom`.

Prerequisites
=============

On every **GPU node**:

- Docker with ROCm device passthrough (``sudo docker`` or docker group).
- ATOM (or vLLM-ATOM / SGLang) container image available on the node.
- Model weights at ``paths.models_dir`` when ``model.remote: 0``.
- Shared log path mounted when using multinode PP.

On the **launcher** (where you run ``cvs run``):

- CVS installed (see :doc:`/getting-started/install`).
- SSH key access to cluster nodes (``priv_key_file`` in the cluster file).
- Hugging Face token file at ``paths.hf_token_file`` when required.

For **multinode PP** (``params.nnodes: 2``, ``pipeline_parallel_size: 2``):

- Two hosts in ``node_dict`` matching ``params.nnodes``.
- ``params.master_addr`` set to the head node VPC IP.
- IB/socket fabric discoverable, or explicit ``roles.server.ib_hca_devices`` /
  ``roles.server.ib_netdev`` (socket netdev must **not** be an ``mlx5_*`` HCA name).

Step 1: Copy config and cluster files
=====================================

List shipped ATOM configs:

.. code:: bash

  cvs config list inference/atom

Copy each variant into its **own subdirectory** so only one ``*threshold.json``
sits beside the config you pass to ``--config_file``:

.. code:: bash

  SINGLE_DIR=~/input/config_file/inference/atom/single
  mkdir -p "$SINGLE_DIR"

  cvs config copy inference/atom/mi325x_atom_deepseek-r1_fp8_single.json \
    --output "$SINGLE_DIR/mi325x_atom_deepseek-r1_fp8_single.json"
  cvs config copy inference/atom/mi325x_atom_deepseek-r1_fp8_threshold.json \
    --output "$SINGLE_DIR/mi325x_atom_deepseek-r1_fp8_threshold.json"
  cvs config copy cluster_file/atom_cluster.json --output ~/input/cluster_file/atom_cluster.json

Use ``mi325x_*`` stems on MI325X (gfx942). Threshold filenames must match the
platform you calibrated on.

Step 2: Edit placeholders
=========================

Replace cluster node IPs and trim ``node_dict`` to one host for single-node runs
(two hosts for distributed PP). In the config, set at minimum:

- ``container.image`` ΓÇö your ATOM ROCm image.
- ``paths.shared_fs``, ``paths.models_dir``, ``paths.log_dir``, ``paths.hf_token_file``.
- ``model.id`` ΓÇö model under test.

For multinode PP, also set ``params.master_addr`` and verify
``roles.server.ib_netdev`` (``"auto"`` is the default on shipped distributed stems).

.. tip::

  Leave ``enforce_thresholds: false`` on first lab runs until thresholds are
  calibrated for your hardware. MI355X shipped stems often ship record-only.

Step 3: Run the suite
=====================

Single-node W1 (``driver=atom``):

.. code:: bash

  cvs run atom \
    --cluster_file ~/input/cluster_file/atom_cluster.json \
    --config_file "$SINGLE_DIR/mi325x_atom_deepseek-r1_fp8_single.json" \
    --html ~/cvs_results/atom-w1-single.html --self-contained-html -vvv

Multinode PP (``driver=vllm_atom``):

.. code:: bash

  cvs run atom \
    --cluster_file ~/input/cluster_file/atom_cluster.json \
    --config_file ~/input/config_file/inference/atom/distributed/mi325x_atom_deepseek-r1_fp8_distributed.json \
    --html ~/cvs_results/atom-w1-distributed.html --self-contained-html -vvv

Multi-profile configs (``schema_version: 2``) select a job shape with
``--config_profile``:

.. code:: bash

  cvs run atom \
    --cluster_file ~/input/cluster_file/atom_cluster.json \
    --config_file "$SINGLE_DIR/mi325x_atom_deepseek-r1_fp8_single.json" \
    --config_profile accuracy \
    --html ~/cvs_results/atom-w1-accuracy.html --self-contained-html -vvv

Smoke one cell with pytest ``-k``, for example ``-k "w1_1k_1k-conc128"``.

After ``git pull``, run ``make install`` before ``source .cvs_venv/bin/activate``.

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
     - ``output_throughput``, ``per_gpu_throughput``, ΓÇª
   * - ``ttft`` / ``tpot``
     - ``mean_ttft_ms``, ``p99_ttft_ms``, ΓÇª
   * - ``health``
     - ``success_rate``, ``failed``
   * - ``scaling``
     - ``scaling.efficiency_pct`` (multinode)
   * - ``record``
     - Remaining metrics ΓÇö logged, not gated

Benchmark artifacts still expose ``client.*`` keys internally; threshold JSON
uses bare metric names. ATOM may omit tail percentiles even when
``metric_percentiles`` requests them ΓÇö only present metrics are gated.

Reports and logs
================

- **pytest HTML** ΓÇö one row per lifecycle stage and per metric tier.
- **Console tables** ΓÇö ``test_print_results_table`` prints per-cell throughput and latency.
- **Run Deck** ΓÇö when ``--html`` is set, ``atom_run_deck.html`` / ``.json`` /
  ``_viewer.html`` are bundled into the pytest zip (render-only; does not affect gates).
- **Per-cell logs** ΓÇö under ``paths.log_dir`` on cluster nodes (server + client logs).

Launcher vs GPU node
====================

.. list-table::
   :widths: 4 2 2
   :header-rows: 1

   * - Item
     - Launcher
     - GPU node
   * - ``cvs run``, venv, ``~/input/``, reports
     - Yes
     - No
   * - ``priv_key_file``, HF token file
     - Yes
     - No
   * - ``/home/models`` (when ``model.remote: 0``)
     - No
     - Yes
   * - Container image, ``sudo docker``
     - No
     - Yes
   * - ``~/LOGS/`` (volume mount)
     - No
     - Yes

See also
========

- :doc:`/reference/configuration-files/inference/atom` — configuration schema and thresholds
- :doc:`/reference/configuration-files/cluster-file` ΓÇö cluster file format
- :doc:`/how-to/run-with-containers` ΓÇö container backend
- :doc:`/how-to/run-cvs-tests` ΓÇö other CVS suites
