.. meta::
  :description: Configure RCCL benchmark configuration file variables
  :keywords: RCCL, ROCm, benchmark, CVS

**********************************************************************
ROCm Communication Collectives Library (RCCL) test configuration files
**********************************************************************

RCCL tests in CVS validate distributed GPU communication performance across AMD GPU clusters. The suites run RCCL collectives, optionally validate against expected thresholds, and generate HTML artifacts (graph and heatmap reports).

RCCL test suites
================

CVS provides the following RCCL suites:

.. list-table::
   :widths: 3 7
   :header-rows: 1

   * - Test suite
     - What it does
   * - ``rccl_multinode_cvs``
     - Runs multinode RCCL with a parameter sweep over collective, algorithm, protocol, queue-pair scale, and PXN setting.
   * - ``rccl_multinode_default_cvs``
     - Runs multinode RCCL collectives using RCCL default algorithm/protocol behavior.
   * - ``rccl_singlenode_cvs``
     - Runs single-node RCCL collectives and validates output against configured expectations.
   * - ``rccl_heatmap_cvs``
     - Runs a sweep over collective, GPU count, data type, and channel configuration; then generates a heatmap against a golden reference JSON.

All suites also collect host/network information and check firewall state before performance runs.

How to run
==========

From the CVS repo root (directory containing ``cvs`` and ``input``):

.. code-block:: bash

  cvs list rccl_multinode_cvs
  cvs list rccl_multinode_default_cvs
  cvs list rccl_singlenode_cvs
  cvs list rccl_heatmap_cvs

Run the multinode sweep:

.. code-block:: bash

  cvs run rccl_multinode_cvs \
      --cluster_file input/cluster_file/cluster.json \
      --config_file input/config_file/rccl/rccl_config.json \
      --html=/var/www/html/cvs/rccl_multinode.html --capture=tee-sys --self-contained-html \
      --log-file=/tmp/rccl_multinode.log -vvv -s

Run multinode with RCCL defaults:

.. code-block:: bash

  cvs run rccl_multinode_default_cvs \
      --cluster_file input/cluster_file/cluster.json \
      --config_file input/config_file/rccl/rccl_config.json \
      --html=/var/www/html/cvs/rccl_multinode_default.html --capture=tee-sys --self-contained-html \
      --log-file=/tmp/rccl_multinode_default.log -vvv -s

Run the single-node suite:

.. code-block:: bash

  cvs run rccl_singlenode_cvs \
      --cluster_file input/cluster_file/cluster.json \
      --config_file input/config_file/rccl/single_node_mi355_rccl.json \
      --html=/var/www/html/cvs/rccl_singlenode.html --capture=tee-sys --self-contained-html \
      --log-file=/tmp/rccl_singlenode.log -vvv -s

Run the heatmap suite:

.. code-block:: bash

  cvs run rccl_heatmap_cvs \
      --cluster_file input/cluster_file/cluster.json \
      --config_file input/config_file/rccl/rccl_config.json \
      --html=/var/www/html/cvs/rccl_heatmap.html --capture=tee-sys --self-contained-html \
      --log-file=/tmp/rccl_heatmap.log -vvv -s

.. note::

  Update expected thresholds in ``results`` for your platform and cluster size before relying on pass/fail status.
  Placeholders such as ``{user-id}`` in config paths are resolved by CVS at runtime.

``rccl_config.json``
====================

Main config file used by ``rccl_multinode_cvs``, ``rccl_multinode_default_cvs``, and ``rccl_heatmap_cvs``:

.. dropdown:: ``rccl_config.json`` (example)

  .. code:: json

    {
      "rccl": {
        "no_of_nodes": "2",
        "no_of_global_ranks": "16",
        "no_of_local_ranks": "8",
        "rccl_dir": "/opt/rccl-tests/",
        "rccl_tests_dir": "/opt/rccl-tests/build",
        "mpi_dir": "/usr/bin",
        "mpi_path_var": "/usr",
        "rocm_path_var": "/opt/rocm/",
        "rccl_collective": ["all_reduce_perf", "all_gather_perf", "broadcast_perf"],
        "rccl_algo": ["ring", "tree"],
        "rccl_protocol": ["simple"],
        "qp_scale": ["1", "2"],
        "gpu_count_list": ["8", "16", "32"],
        "data_type_list": ["float", "bfloat16"],
        "channel_config_list": ["default"],
        "golden_reference_json_file": "/home/{user-id}/JSONS/mi300_reference.json",
        "start_msg_size": "1024",
        "end_msg_size": "16g",
        "step_function": "2",
        "warmup_iterations": "10",
        "no_of_iterations": "20",
        "verify_bus_bw": "False",
        "verify_bw_dip": "True",
        "verify_lat_dip": "True",
        "env_source_script": "/root/env_source_file.sh",
        "results": {}
      }
    }

Parameters
----------

Exhaustive parameter list for ``rccl_config.json``:

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameter
     - Example/default
     - Description
   * - ``no_of_nodes``
     - ``2``
     - Number of nodes participating in multinode runs.
   * - ``no_of_global_ranks``
     - ``16``
     - Total MPI ranks across all nodes.
   * - ``no_of_local_ranks``
     - ``8``
     - MPI ranks per node.
   * - ``ranks_per_node``
     - ``8``
     - GPU/rank density per node (topology reference).
   * - ``rccl_dir``
     - ``/opt/rccl-tests/``
     - RCCL test install root.
   * - ``rccl_tests_dir``
     - ``/opt/rccl-tests/build``
     - Directory containing RCCL test binaries.
   * - ``mpi_dir``
     - ``/usr/bin``
     - MPI executable directory.
   * - ``mpi_path_var``
     - ``/usr``
     - MPI root path used when constructing runtime environment.
   * - ``mpi_pml``
     - ``auto``
     - MPI point-to-point messaging layer (`auto`, `ucx`, `ob1`).
   * - ``rocm_path_var``
     - ``/opt/rocm/``
     - ROCm installation path.
   * - ``rccl_path_var``
     - ``/opt/rccl-tests/``
     - RCCL path exported to runtime environment.
   * - ``gpu_count_list``
     - ``["8", "16", "32"]``
     - GPU-count sweep list for ``rccl_heatmap_cvs``.
   * - ``data_type_list``
     - ``["float", "bfloat16"]``
     - Data-type sweep list for ``rccl_heatmap_cvs``.
   * - ``nic_model``
     - ``thor``
     - NIC model tag used in validation logic/reporting.
   * - ``heatmap_title``
     - ``RCCL heatmap comparison between MI300s``
     - Title rendered in heatmap HTML.
   * - ``golden_reference_json_file``
     - ``/home/{user-id}/JSONS/mi300_reference.json``
     - Baseline reference JSON for heatmap comparison.
   * - ``cluster_snapshot_debug``
     - ``False``
     - Enables before/after cluster metric snapshots around tests.
   * - ``env_source_script``
     - ``/root/env_source_file.sh``
     - Optional script sourced before test execution.
   * - ``rccl_collective``
     - ``["all_reduce_perf", ..., "broadcast_perf"]``
     - RCCL collectives to execute.
   * - ``rccl_algo``
     - ``["ring", "tree"]``
     - RCCL algorithm sweep values.
   * - ``rccl_protocol``
     - ``["simple"]``
     - RCCL protocol sweep values.
   * - ``qp_scale``
     - ``["1", "2"]``
     - Queue-pair scaling sweep values.
   * - ``ib_hca_list``
     - ``bnxt_re0,...,bnxt_re7``
     - RDMA HCA device list used by launch/runtime setup.
   * - ``net_dev_list``
     - ``ens28np0,...,ens22np0``
     - Network interface list aligned with IB devices.
   * - ``oob_port``
     - ``eth0``
     - Out-of-band control interface.
   * - ``nccl_socket_ifname``
     - ``""``
     - Optional NCCL bootstrap/socket fallback interface.
   * - ``gid_index``
     - ``1``
     - GID index for RDMA/IB communication.
   * - ``start_msg_size``
     - ``1024``
     - Start message size for sweep.
   * - ``end_msg_size``
     - ``16g``
     - End message size for sweep.
   * - ``step_function``
     - ``2``
     - Message-size progression rule.
   * - ``threads_per_gpu``
     - ``1``
     - Worker threads per GPU for RCCL test binaries.
   * - ``warmup_iterations``
     - ``10``
     - Warmup iteration count before measured iterations.
   * - ``no_of_iterations``
     - ``20``
     - Number of measured iterations.
   * - ``no_of_cycles``
     - ``1``
     - Number of full run cycles.
   * - ``check_iteration_count``
     - ``1``
     - Validation/check iteration count used by parsers.
   * - ``nccl_ib_timeout``
     - ``30``
     - NCCL/RCCL IB timeout.
   * - ``ib_rx_queue_len``
     - ``8192``
     - IB receive queue length.
   * - ``ucx_tls``
     - ``tcp``
     - UCX transport selection.
   * - ``hcoll_enable_mcast_all``
     - ``0``
     - HCOLL multicast setting.
   * - ``nccl_cumem_enable``
     - ``0``
     - NCCL memory behavior control.
   * - ``nccl_ib_sl``
     - ``0``
     - IB service level.
   * - ``nccl_ib_tc``
     - ``0``
     - IB traffic class.
   * - ``nccl_ib_split_data_on_qps``
     - ``0``
     - Split-data policy across queue pairs.
   * - ``nccl_pxn_disable``
     - ``["0", "1"]``
     - PXN enable/disable sweep values.
   * - ``nccl_net_plugin``
     - ``none``
     - NCCL/RCCL network plugin selection.
   * - ``channel_config_list``
     - ``["default"]``
     - Channel sweep list (for example ``"16-16"``, ``"64-64"``, ``"default"``).
   * - ``verify_bus_bw``
     - ``False``
     - Enable bus-bandwidth threshold validation.
   * - ``verify_bw_dip``
     - ``True``
     - Enable bandwidth-dip validation.
   * - ``verify_lat_dip``
     - ``True``
     - Enable latency-dip validation.
   * - ``debug_level``
     - ``ERROR``
     - RCCL/NCCL test debug verbosity.
   * - ``rccl_result_file``
     - ``/tmp/rccl_result_file.json``
     - Base output file path for RCCL parsed results.
   * - ``_comments_results``
     - informational text
     - Notes that expected results vary by cluster size and hardware.
   * - ``results``
     - per-collective threshold map
     - Expected threshold values used for pass/fail validation.

Expected results format
-----------------------

The ``results`` section is used for threshold validation. Values are keyed by collective and message size (bytes), with expected bus bandwidth values.

.. dropdown:: ``results`` snippet

  .. code:: json

    "results": {
      "all_reduce_perf": {
        "bus_bw": {
          "8589934592": "330.00",
          "17179869184": "350.00"
        }
      }
    }

Collective meanings
-------------------

- ``all_reduce_perf``: all ranks reduce then receive the reduced result.
- ``all_gather_perf``: each rank receives data from all ranks.
- ``scatter_perf``: root rank distributes shards to all ranks.
- ``gather_perf``: all ranks send data to a root rank.
- ``reduce_scatter_perf``: reduction followed by scatter.
- ``sendrecv_perf``: point-to-point pair communication.
- ``alltoall_perf``: equal-sized all-to-all exchange.
- ``alltoallv_perf``: variable-sized all-to-all exchange.
- ``broadcast_perf``: one-to-all data broadcast.

``single_node_mi355_rccl.json``
===============================

This file is tuned for ``rccl_singlenode_cvs`` and keeps only single-node relevant fields (collective list, message sweep, iteration controls, and expected thresholds).

Parameters
----------

Exhaustive parameter list for ``single_node_mi355_rccl.json``:

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameter
     - Example/default
     - Description
   * - ``no_of_local_ranks``
     - ``8``
     - Local ranks (GPU processes) used for single-node tests.
   * - ``rccl_dir``
     - ``/opt/rccl-tests/``
     - RCCL test install root.
   * - ``rccl_tests_dir``
     - ``/opt/rccl-tests/build``
     - Directory containing RCCL test binaries.
   * - ``rocm_path_var``
     - ``/opt/rocm/``
     - ROCm installation path.
   * - ``rccl_path_var``
     - ``/opt/rccl-tests/``
     - RCCL runtime path.
   * - ``env_source_script``
     - ``/root/env_source_file.sh``
     - Optional script sourced before test execution.
   * - ``rccl_collective``
     - ``["all_reduce_perf", ..., "broadcast_perf"]``
     - Collectives executed in single-node sweep.
   * - ``start_msg_size``
     - ``1024``
     - Start message size for sweep.
   * - ``end_msg_size``
     - ``16g``
     - End message size for sweep.
   * - ``step_function``
     - ``2``
     - Message-size progression rule.
   * - ``warmup_iterations``
     - ``10``
     - Warmup iteration count before measured iterations.
   * - ``no_of_iterations``
     - ``1``
     - Number of measured iterations.
   * - ``check_iteration_count``
     - ``1``
     - Validation/check iteration count used by parsers.
   * - ``verify_bus_bw``
     - ``False``
     - Enable bus-bandwidth threshold validation.
   * - ``verify_bw_dip``
     - ``True``
     - Enable bandwidth-dip validation.
   * - ``verify_lat_dip``
     - ``True``
     - Enable latency-dip validation.
   * - ``debug_level``
     - ``ERROR``
     - RCCL/NCCL test debug verbosity.
   * - ``rccl_result_file``
     - ``/tmp/rccl_result_file.json``
     - Output file path for parsed results.
   * - ``_comments_results``
     - informational text
     - Notes that expected results vary by system and test profile.
   * - ``results``
     - per-collective threshold map
     - Expected threshold values used for pass/fail validation.

Use this command:

.. code-block:: bash

  cvs run rccl_singlenode_cvs \
      --cluster_file input/cluster_file/cluster.json \
      --config_file input/config_file/rccl/single_node_mi355_rccl.json \
      --html=/var/www/html/cvs/rccl_singlenode.html --capture=tee-sys --self-contained-html \
      --log-file=/tmp/rccl_singlenode.log -vvv -s

AINIC/ANP net plugin setup (optional)
=====================================

To run with AINIC + ANP plugin:

1. Ensure AMD ANP is installed and available on all target nodes.
2. Edit ``input/config_file/rccl/ainic_env_script.sh`` and set ``ANP_HOME_DIR`` to your ANP install path.
3. Set ``env_source_script`` in RCCL JSON config to that script path.
4. Run RCCL using ``cvs run ...`` commands; CVS sources the script automatically.

If AINIC/ANP is not used, set ``env_source_script`` to your standard environment script or ``"None"`` to skip script sourcing.

Validation and artifacts
========================

- Test-level pass/fail is based on command execution plus enabled validations (``results``, bandwidth checks, dip checks).
- Graph reports are generated under ``/tmp/rccl_perf_report_*.html`` (or ``/tmp/rccl_singlenode_perf_report_*.html``).
- Heatmap runs produce ``/tmp/rccl_heatmap_*.html`` and JSON result artifacts, and may copy them to ``output_dir`` when configured.
- Artifacts generated under ``/tmp`` can be copied to a web server path (for example ``/var/www/html/cvs``) for browser access.
