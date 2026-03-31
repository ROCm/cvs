.. meta::
  :description: Configure RCCL benchmark configuration file variables
  :keywords: RCCL, ROCm, benchmark, CVS

**********************************************************************
ROCm Communication Collectives Library (RCCL) test configuration files
**********************************************************************

RCCL in CVS now uses a single suite, ``rccl_cvs``, and a reduced configuration surface. The RCCL JSON describes benchmark intent only. NCCL, UCX, plugin, and site-specific tuning should live in ``env_script`` instead of the CVS config.

RCCL test suite
===============

CVS provides one RCCL suite:

.. list-table::
   :widths: 3 7
   :header-rows: 1

   * - Test suite
     - What it does
   * - ``rccl_cvs``
     - Runs either a single-node or multi-node RCCL benchmark flow, performs lightweight preflight, validates rccl-tests JSON output, and writes one consolidated result artifact.

How to run
==========

From the CVS repo root:

.. code-block:: bash

  cvs list rccl_cvs

Run RCCL with the simplified config:

.. code-block:: bash

  cvs run rccl_cvs \
      --cluster_file input/cluster_file/cluster.json \
      --config_file input/config_file/rccl/rccl_config.json \
      --html=/var/www/html/cvs/rccl.html --capture=tee-sys --self-contained-html \
      --log-file=/tmp/rccl.log -vvv -s

.. note::

  Update expected thresholds in ``results`` for your platform and cluster size before relying on pass/fail status.
  Placeholders such as ``{user-id}`` in config paths are resolved by CVS at runtime.

``rccl_config.json``
====================

``rccl_config.json`` is used for both single-node and multi-node RCCL runs. Switch between them by changing ``mode``.

.. dropdown:: ``rccl_config.json`` (example)

  .. code:: json

    {
      "rccl": {
        "mode": "multi_node",
        "rccl_tests_dir": "/opt/rccl-tests/build",
        "mpi_root": "/usr",
        "rocm_path": "/opt/rocm",
        "env_script": "/root/env_source_file.sh",
        "num_ranks": "16",
        "ranks_per_node": "8",
        "collectives": ["all_reduce_perf", "all_gather_perf", "broadcast_perf"],
        "datatype": "float",
        "start_size": "1024",
        "end_size": "16g",
        "step_factor": "2",
        "warmups": "10",
        "iterations": "20",
        "cycles": "1",
        "verify_bus_bw": "False",
        "verify_bw_dip": "True",
        "verify_lat_dip": "True",
        "output_json": "/tmp/rccl_result.json",
        "results": {}
      }
    }

Parameters
----------

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameter
     - Example/default
     - Description
   * - ``mode``
     - ``multi_node``
     - RCCL execution mode. Use ``single_node`` or ``multi_node``.
   * - ``rccl_tests_dir``
     - ``/opt/rccl-tests/build``
     - Directory containing RCCL benchmark binaries.
   * - ``mpi_root``
     - ``/usr``
     - MPI installation root used to derive ``mpirun`` and runtime library paths.
   * - ``mpirun_path``
     - ``/usr/bin/mpirun``
     - Optional explicit ``mpirun`` path. Use this instead of ``mpi_root`` when preferred.
   * - ``rocm_path``
     - ``/opt/rocm``
     - ROCm installation path used to populate runtime environment variables.
   * - ``env_script``
     - ``/root/env_source_file.sh``
     - User-owned script that sets NCCL, UCX, plugin, and site-specific tuning.
   * - ``num_ranks``
     - ``16``
     - Total number of ranks to launch.
   * - ``ranks_per_node``
     - ``8``
     - Number of ranks launched per node.
   * - ``collectives``
     - ``["all_reduce_perf", "all_gather_perf"]``
     - RCCL collectives to execute in one run.
   * - ``datatype``
     - ``float``
     - Single rccl-tests datatype to benchmark.
   * - ``start_size``
     - ``1024``
     - Start message size for the benchmark sweep.
   * - ``end_size``
     - ``16g``
     - End message size for the benchmark sweep.
   * - ``step_factor``
     - ``2``
     - Message-size growth factor.
   * - ``warmups``
     - ``10``
     - Warmup iterations before measured iterations.
   * - ``iterations``
     - ``20``
     - Number of measured iterations.
   * - ``cycles``
     - ``1``
     - Number of benchmark cycles.
   * - ``verify_bus_bw``
     - ``False``
     - Enable minimum bus-bandwidth validation against ``results``.
   * - ``verify_bw_dip``
     - ``True``
     - Enable bandwidth-dip checks for message sizes present in ``results``.
   * - ``verify_lat_dip``
     - ``True``
     - Enable latency-dip checks for message sizes present in ``results``.
   * - ``results``
     - per-collective threshold map
     - Inline thresholds used for validation.
   * - ``results_file``
     - ``/path/to/thresholds.json``
     - Optional external threshold JSON file. Use this instead of inline ``results`` when preferred.
   * - ``output_json``
     - ``/tmp/rccl_result.json``
     - Final consolidated result artifact written by ``rccl_cvs``.

Expected results format
-----------------------

The ``results`` section is keyed by collective and message size. Each message size contains the minimum expected bus bandwidth.

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

AINIC/ANP net plugin setup (optional)
=====================================

To run with AINIC + ANP plugin:

1. Ensure AMD ANP is installed and available on all target nodes.
2. Edit ``input/config_file/rccl/ainic_env_script.sh`` and set ``ANP_HOME_DIR`` to your ANP install path.
3. Point ``env_script`` at that file.
4. Run ``cvs run rccl_cvs ...``; CVS sources the script before benchmark execution.

Validation and artifacts
========================

- Test-level pass/fail is based on command execution, output schema validation, optional threshold checks, and dmesg/preflight checks.
- ``rccl_cvs`` writes one JSON artifact to ``output_json``.
- The artifact contains run metadata, one entry per collective, parsed rccl-tests rows, and validation summaries.
