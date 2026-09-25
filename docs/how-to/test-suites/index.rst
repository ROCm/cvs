.. meta::
  :description: Run CVS test suites on AMD Instinct GPU clusters: health, RCCL, training, and inference tests using cvs run with cluster and config files.
  :keywords: CVS, ROCm, health, network, tests, RCCL, AMD Instinct, GPU, AMD, training, inference, cluster, CLI

***************************************************************************
Run Cluster Validation Suite (CVS) test suites on AMD Instinct GPU clusters
***************************************************************************

To run a test suite you need two files: a **cluster file** (``--cluster_file``) that describes your nodes and SSH access, and a **test suite config** (``--config_file``) with suite-specific settings. Set those up first — see :doc:`/how-to/configure/cluster-config` and :doc:`/how-to/configure/test-suite-config/index`. To choose the right template, see :doc:`/how-to/configure/test-suite-config/pick-config-file`.

Then use ``cvs run`` on the head node to execute tests across the cluster.

List available suites
=============================

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
    --config_file ~/cvs_workspace/mi300_health_config.json \
    --html=/var/www/html/cvs/agfhc.html --capture=tee-sys --self-contained-html \
    --log-file=/tmp/test.log -vvv -s

List test cases in a suite
==========================

``cvs list <suite>`` lists the test functions in that suite, including parameterized tests generated from your config. Pass the same ``--cluster_file`` and ``--config_file`` you use for ``cvs run``:

.. code:: bash

  cvs list agfhc_cvs \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/mi300_health_config.json

Run one test function
=============================

Add the test function name after the suite name:

.. code:: bash

  cvs run agfhc_cvs test_agfhc_hbm \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/mi300_health_config.json \
    --html=/var/www/html/cvs/agfhc.html --capture=tee-sys --self-contained-html \
    --log-file=/tmp/test.log -vvv -s

Run with containers
===================

CVS selects an execution backend in the cluster file:

- **Bare metal** — use when you can install or upgrade the ROCm stack on each host and run tests on the host filesystem.
- **Container** — use when you cannot install or change ROCm on the hosts, or when you have a Docker image with a pinned ROCm version and framework dependencies (for example PyTorch) that you run on each node.

See :doc:`/how-to/run-with-containers` for ``cluster_container.json``, image selection, and container run commands.

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
   * - :doc:`Burn-in / Diag </how-to/test-suites/burn-in-diag/index>`
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

CVS writes a pytest HTML report after each run. The report includes per-node pass/fail status, captured output, and links to any custom suite-specific reports (for example RCCL performance charts).

.. include:: /_includes/test-results.rst
