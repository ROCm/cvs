.. meta::
  :description: Run CVS test suites on AMD Instinct GPU clusters: health, RCCL, training, and inference tests using cvs run with cluster and config files.
  :keywords: CVS, ROCm, health, network, tests, RCCL, AMD Instinct, GPU, AMD, training, inference, cluster, CLI

***************************************************************************
Run Cluster Validation Suite (CVS) test suites on AMD Instinct GPU clusters
***************************************************************************

To run a test suite over SSH you need two files: a cluster file (``--cluster_file``) that describes your nodes and SSH access, and a test suite config (``--config_file``) with suite-specific settings. 
Set those up first — see :doc:`/how-to/configure/cluster-config` and :doc:`/how-to/configure/test-suite-config/index`. 

To choose the right template, see :doc:`/how-to/configure/test-suite-config/pick-config-file`. 

On Slurm or SPUR, launch ``cvs run`` as a job step instead and the cluster file becomes optional (see :doc:`/how-to/run-on-managed-compute`).

For SSH mode, use ``cvs run`` on the head node to execute tests across the cluster.

List available suites
=====================

Run the following command to see all available test suites:

.. code:: bash

  cvs list

You can also run ``cvs run`` with no arguments to see the same catalog.

Run a suite
===================

Pass the cluster file and test suite config from your workspace:

.. code:: bash

  cvs run agfhc_cvs \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/health_config.json \
    --capture=tee-sys -vvv -s

``cvs run`` writes the pytest HTML report, text log, and zip bundle under the run directory. See :ref:`pytest-html-report-and-bundle`.

List test cases in a suite
==========================

``cvs list <suite>`` lists the test functions in that suite, including parameterized tests generated from your config. Pass the same ``--cluster_file`` and ``--config_file`` you use for ``cvs run``:

.. code:: bash

  cvs list agfhc_cvs \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/health_config.json

Run one test function
=====================

Add the test function name after the suite name:

.. code:: bash

  cvs run agfhc_cvs test_agfhc_hbm \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/health_config.json \
    --capture=tee-sys -vvv -s

Run with containers
===================

The cluster file has an ``orchestrator`` key. Suites differ: some follow it and run the workload inside the container, some start a container and keep their checks on the host, some force a container, and some stay on host SSH. The table is in :doc:`/reference/cluster/cluster-file`.

- **Bare metal** — commands run on the host filesystem. Use this when the suite follows ``orchestrator`` and the ROCm stack is installed on each host.
- **Container** — commands for a suite that enters the container run with ``docker exec`` in one container per host. ``per_run`` (the default lifetime) removes that container at the end of the run.

See :doc:`/how-to/run-with-containers` for the cluster file, image, and a health-suite example.

Run on managed compute
======================

Launch one CVS task per node as a job step and give every task the same shared
workspace. CVS generates the cluster file from the allocated node list; see
:doc:`/how-to/run-on-managed-compute` for setup and a complete quickstart.

.. code:: bash

  srun -A <account> -p <partition> -N 2 --ntasks-per-node 1 --exclusive --mpi=none \
    bash -lc 'source <cvs-checkout>/.cvs_venv/bin/activate &&
      cvs run host_configs_cvs test_check_os_release test_check_kernel_version \
        --config_file /shared/<user>/cvs/host_config.json \
        --workspace /shared/<user>/cvs'

Common ``cvs run`` options
==========================

.. include:: /_includes/common-cvs-run-flags.rst

Test suites
===========

Run suites in the order shown: validate single-node health before exercising the network, and validate the network before distributed training or inference. Each suite's page includes setup and run steps.

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Category
     - What it validates
     - Suites
   * - :doc:`Burn-in and diagnostics </how-to/test-suites/burn-in-diag/index>`
     - Single-node GPU and host health: OS config, BIOS/firmware, driver load, GPU burn-in, and device access. Run before any cluster-wide workload.
     - Platform, Health, Preflight
   * - :doc:`Network </how-to/test-suites/network/index>`
     - Interconnect bandwidth, latency, and GPU collective communication across all nodes. Run after burn-in passes and before distributed workloads.
     - IB Perf, RCCL, MORI
   * - :doc:`Training </how-to/test-suites/training/index>`
     - Multi-node distributed training throughput, scaling efficiency, and model convergence. Run after network validation.
     - Aorta, JAX MaxText, Megatron, TorchTitan
   * - :doc:`Inference </how-to/test-suites/inference/index>`
     - LLM serving throughput, latency, accuracy, and diffusion model performance on AMD Instinct GPUs.
     - vLLM, ATOM, SGLang, xDiT

Scalability
===========

For clusters with 32 or more nodes, CVS automatically shards work across parallel worker processes. See :doc:`/reference/cvs-at-scale` for tuning ``CVS_HOSTS_PER_SHARD`` and ``CVS_WORKERS_PER_CPU``.

Test results
============

CVS writes a pytest HTML report after each run. The report includes per-node pass/fail status, captured output, and links to any custom suite-specific reports.

.. _pytest-html-report-and-bundle:

Pytest HTML report and bundle
-----------------------------

Each run produces the following artifacts in the run directory.

.. list-table::
   :header-rows: 1

   * - Artifact
     - Default location
     - Notes
   * - Pytest HTML report
     - ``<run_dir>/<test-file-stem>.html``
     - For example,
       ``<workspace>/cvs_runs/<run_id>/rccl_perf.html``.
       The auto-derived report is self-contained.
   * - Text log
     - ``<run_dir>/<test-file-stem>.log``
     - Pytest run log.
   * - Attachments
     - ``<run_dir>/<test-file-stem>_html/``
     - Per-test "Full Log" pages and a copy of the run log.
       When the suite supports a Run Deck, this directory
       also holds the Run Deck HTML and JSON, and might
       include a summary page or an interactive viewer.
       The pytest report's **Reports** section links the
       Run Deck HTML. The other deck files stay in this
       directory and in the zip. Recreated at the start
       of each run.
   * - Zip bundle
     - ``<run_dir>/<test-file-stem>_<YYYY-mm-ddTHHMMSS>.zip``
     - Contains the report, the attachments directory,
       and the cluster and config files. Links still work
       after extraction.

``<run_dir>`` is ``<workspace>/cvs_runs/<run_id>``. ``<run_id>`` is ``local-<YYYYmmdd-HHMMSS>`` outside a scheduler, or the job ID under SPUR or Slurm.

A Run Deck summarizes results collected during the run. It is included when the selected suite supports it. Run Deck generation does not affect the suite result.

See :doc:`/reference/cli/cvs-run` for the report and log options.

Test output examples
--------------------

.. include:: /_includes/test-results.rst

Sample reports for each category are on that category's page, under **Test results**:

- :doc:`Burn-in and diagnostics </how-to/test-suites/burn-in-diag/index>` — pytest HTML report and Run Deck
- :doc:`Network </how-to/test-suites/network/index>` — RCCL pytest HTML report, Run Deck, and interactive viewer
- :doc:`Training </how-to/test-suites/training/index>` — pytest HTML report, Run Deck, and interactive viewer
- :doc:`Inference </how-to/test-suites/inference/index>` — pytest HTML report, Run Deck, and interactive viewer
