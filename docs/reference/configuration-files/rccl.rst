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
   * - ``rccl_perf``
     - User-facing performance suite that runs configured collectives with environment script staging.
   * - ``rccl_regression``
     - Regression suite with Cartesian product sweep using environment variable combinations from JSON configuration.

All suites also collect host/network information and check firewall state before performance runs.

How to run
==========

From the CVS repo root (directory containing ``cvs`` and ``input``):

.. code-block:: bash

  cvs list rccl_perf
  cvs list rccl_regression

Run the performance suite:

.. code-block:: bash

  cvs run rccl_perf \
      --cluster_file input/cluster_file/cluster.json \
      --config_file input/config_file/rccl/rccl_config.json \
      --html=/var/www/html/cvs/rccl_perf.html --capture=tee-sys --self-contained-html \
      --log-file=/tmp/rccl_perf.log -vvv -s

Run the regression suite:

.. code-block:: bash

  cvs run rccl_regression \
      --cluster_file input/cluster_file/rccl/rccl_regression.json \
      --config_file input/config_file/rccl/rccl_config.json \
      --html=/var/www/html/cvs/rccl_regression.html --capture=tee-sys --self-contained-html \
      --log-file=/tmp/rccl_regression.log -vvv -s

Generate heatmap from results:

.. code-block:: bash

  cvs generate heatmap \
      --actual /tmp/rccl_perf_results.json \
      --reference /path/to/reference.json \
      --output /var/www/html/cvs/rccl_heatmap.html \
      --title "RCCL Performance Heatmap"

.. note::

  Update expected thresholds in ``results`` for your platform and cluster size before relying on pass/fail status.
  Placeholders such as ``{user-id}`` in config paths are resolved by CVS at runtime.

``rccl_config.json``
====================

Main config file used by ``rccl_perf`` and ``rccl_regression``:

.. dropdown:: ``rccl_config.json`` (example)

  .. code:: json

    {
      "rccl": {
        "env_source_script": "/home/{user-id}/thor2_env_script.sh",
        "mpi_params": {
          "no_of_nodes": "2",
          "no_of_local_ranks": "8",
          "mpi_pml": "auto",
          "mpi_dir": "/home/{user-id}/openmpi/bin",
          "mpi_oob_port": "eth0"
        },
        "rccl_test_params": {
          "rccl_collective": ["all_reduce_perf", "all_gather_perf", "broadcast_perf"],
          "rccl_tests_dir": "/home/{user-id}/rccl-tests/build",
          "start_msg_size": "1024",
          "end_msg_size": "16g",
          "step_function": "2",
          "warmup_iterations": "10",
          "no_of_iterations": "20",
          "data_types": ["float"]
        },
        "cvs_params": {
          "verify_bus_bw": "False",
          "verify_bw_dip": "True",
          "verify_lat_dip": "True",
          "cluster_snapshot_debug": "False",
          "rccl_result_file": "/home/{user-id}/rccl_result_file.json"
        },
        "results": {}
      }
    }

.. dropdown:: ``rccl_config.json`` with regression (example)

  .. code:: json

    {
      "rccl": {
        "env_source_script": "/home/{user-id}/thor2_env_script.sh",
        "mpi_params": {
          "no_of_nodes": "2",
          "no_of_local_ranks": "8",
          "mpi_pml": "auto",
          "mpi_dir": "/home/{user-id}/openmpi/bin"
        },
        "rccl_test_params": {
          "rccl_collective": ["all_reduce_perf"],
          "rccl_tests_dir": "/home/{user-id}/rccl-tests/build",
          "start_msg_size": "1024",
          "end_msg_size": "16g",
          "step_function": "2",
          "warmup_iterations": "10",
          "no_of_iterations": "20"
        },
        "regression": {
          "NCCL_ALGO": ["Ring", "Tree"],
          "NCCL_PROTO": ["Simple"],
          "NCCL_IB_QPS_PER_CONNECTION": ["1", "2"],
          "NCCL_PXN_DISABLE": ["0", "1"],
          "NCCL_MIN_NCHANNELS": ["8", "16"],
          "NCCL_MAX_NCHANNELS": ["8", "16"]
        },
        "cvs_params": {
          "verify_bus_bw": "False",
          "verify_bw_dip": "True", 
          "verify_lat_dip": "True",
          "rccl_result_file": "/home/{user-id}/rccl_result_file.json"
        },
        "results": {}
      }
    }

Parameters
----------

Configuration parameters for RCCL suites:

**Core Configuration:**

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameter
     - Example/default
     - Description
   * - ``env_source_script``
     - ``"/home/{user-id}/thor2_env_script.sh"``
     - Environment script sourced before test execution (contains RCCL/NCCL/UCX tuning parameters).
   * - ``rccl_collective``
     - ``["all_reduce_perf", "all_gather_perf"]``
     - RCCL collectives to execute.
   * - ``rccl_result_file``
     - ``"/tmp/rccl_result.json"``
     - Output file path for RCCL parsed results.
   * - ``start_msg_size``
     - ``"1024"``
     - Start message size for sweep.
   * - ``end_msg_size``
     - ``"16g"``
     - End message size for sweep.
   * - ``regression``
     - ``{"NCCL_ALGO": ["Ring", "Tree"], "NCCL_PROTO": ["Simple"]}``
     - Cartesian product sweep using NCCL/RCCL environment variable combinations (rccl_regression only).

**MPI Parameters (mpi_params):**

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameter
     - Example/default
     - Description
   * - ``mpi_pml``
     - ``"auto"``
     - MPI point-to-point messaging layer (auto, ucx, ob1).

**RCCL Test Parameters (rccl_test_params):**

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameter
     - Example/default
     - Description
   * - ``warmup_iterations``
     - ``"10"``
     - Warmup iteration count before measured iterations.
   * - ``no_of_iterations``
     - ``"20"``
     - Number of measured iterations.
   * - ``step_function``
     - ``"2"``
     - Message-size progression rule.

**CVS Parameters (cvs_params):**

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameter
     - Example/default
     - Description
   * - ``verify_bus_bw``
     - ``"False"``
     - Enable bus-bandwidth threshold validation.
   * - ``verify_bw_dip``
     - ``"True"``
     - Enable bandwidth-dip validation.
   * - ``verify_lat_dip``
     - ``"True"``
     - Enable latency-dip validation.
   * - ``cluster_snapshot_debug``
     - ``"False"``
     - Enables before/after cluster metric snapshots around tests.
   * - ``results``
     - ``{}``
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

Regression testing
==================

The ``rccl_regression`` suite performs Cartesian product sweeps using the ``regression`` object in the configuration. Each key-value pair represents an NCCL/RCCL environment variable and its possible values:

.. code:: json

  "regression": {
    "NCCL_ALGO": ["Ring", "Tree"],
    "NCCL_PROTO": ["Simple"],
    "NCCL_IB_QPS_PER_CONNECTION": ["1", "2"],
    "NCCL_PXN_DISABLE": ["0", "1"],
    "NCCL_MIN_NCHANNELS": ["8", "16"],
    "NCCL_MAX_NCHANNELS": ["8", "16"]
  }

**Key features:**

- **Cartesian product**: All combinations of the environment variables are tested
- **Paired channels**: ``NCCL_MIN_NCHANNELS`` and ``NCCL_MAX_NCHANNELS`` are paired (not Cartesian)
- **Tree filtering**: Tree algorithm only runs with ``all_reduce_perf`` collective
- **Environment variables**: Use actual NCCL/RCCL environment variable names as keys

Single-node testing
===================

Single-node RCCL testing can be achieved by configuring a single-node cluster in your cluster JSON file and running either ``rccl_perf`` or ``rccl_regression`` suites. The tests will automatically adapt to single-node execution when only one node is present in the cluster configuration.

Heatmap generation
==================

Generate performance heatmaps by comparing actual test results against a golden reference:

.. code-block:: bash

  cvs generate heatmap \
      --actual /tmp/rccl_perf_results.json \
      --reference /path/to/golden_reference.json \
      --output /var/www/html/cvs/rccl_heatmap.html \
      --title "RCCL Performance Comparison" \
      --metadata

**Options:**

- ``--actual``: Path to actual test results JSON file (required)
- ``--reference``: Path to golden reference JSON file (required)  
- ``--output``: Output HTML file path (optional, defaults to /tmp)
- ``--title``: Custom heatmap title (optional)
- ``--metadata``: Include metadata table if actual JSON has 'metadata' key
- ``--no-data-table``: Exclude data table from output

Environment script setup
=========================

All cluster configurations require an environment script to be sourced before RCCL tests, regardless of NIC type (Broadcom/ConnectX/AINIC):

1. **For AINIC clusters:** Ensure AMD ANP is installed and available on all target nodes, then edit ``input/config_file/rccl/ainic_env_script.sh`` and set ``ANP_HOME_DIR`` to your ANP install path.

2. **For other NIC types:** Use the appropriate environment script (``thor2_env_script.sh`` for Broadcom, ``cx7_env_script.sh`` for ConnectX-7, etc.).

3. Set ``env_source_script`` in RCCL JSON config to the correct script path for your hardware.

4. Run RCCL using ``cvs run ...`` commands; CVS sources the script automatically.

The environment script contains essential RCCL/NCCL/UCX tuning parameters and paths required for proper test execution.

Validation and artifacts
========================

- Test-level pass/fail is based on command execution plus enabled validations (``results``, bandwidth checks, dip checks).
- Performance reports are generated under ``/tmp/rccl_perf_report_*.html``.
- Regression reports are generated under ``/tmp/rccl_perf_report_*.html`` with "RCCL Multi Node Performance Report" title.
- Heatmaps are generated using ``cvs generate heatmap`` with customizable output paths.
- All artifacts generated under ``/tmp`` can be copied to a web server path (for example ``/var/www/html/cvs``) for browser access.
