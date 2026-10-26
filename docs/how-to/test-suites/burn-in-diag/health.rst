.. meta::
  :description: Run AGFHC, TransferBench, and RVS health burn-in tests to validate GPU hardware, firmware, and memory bandwidth on AMD Instinct clusters with CVS.
  :keywords: CVS, health, burn-in, AGFHC, RVS, TransferBench, AMD Instinct, ROCm, AMD, GPU, HBM, PCIe

*******************************************************
Run AGFHC, TransferBench, and RVS health burn-in tests
*******************************************************

Health burn-in tests run AMD GPU Field Health Check (AGFHC), `TransferBench <https://rocm.docs.amd.com/projects/TransferBench/en/latest/index.html>`_, and `ROCm Validation Suite (RVS) <https://rocm.docs.amd.com/projects/ROCmValidationSuite/en/latest/index.html>`_ on every node in the cluster. Use these tests to validate GPU hardware, memory bandwidth, and compute throughput before production workloads.

.. _health-set-up-config:

Set up config
=============

1. Copy the health configuration file:

   .. code:: bash

     cvs config copy health/health_config.json --output ~/cvs_workspace/health/health_config.json

2. Edit paths and any remaining ``<changeme>`` placeholders:

   - Under ``agfhc``: ``path``, ``package_tar_ball``, ``install_dir``, ``env``, ``log_dir``. A ``path`` other than ``/opt/amd/agfhc`` installs with ``./install --rocm-tar -o`` and no sudo; its parent and ``log_dir`` must be writable on the nodes. Update ``env`` when ``path`` changes.
   - Under ``transferbench``: see :ref:`health-transferbench`. Leave ``rocm_path`` as ``<changeme>`` to auto-detect ROCm.
   - Under ``rvs``: see :ref:`health-rvs`. Leave ``rocm_path`` and ``path`` placeholders to follow the detected ROCm install.

For the complete field reference including install paths and threshold values, see :doc:`/reference/configuration-files/burn-in-diag/health`.

.. _health-run-tests:

Run tests
=========

Use these tests to validate single-node diagnostics for GPU cluster health.
For the performance validation, use the reference bandwidth or latency numbers provided as part of the input ``config_file`` for the relevant test.

Use these scripts to run each health test. These CVS test scripts have two parts: installing the functionality and running the tests.

.. note::

  These health suites honor ``orchestrator`` in the cluster file. With ``container``, the workload runs inside the per-host container (``docker exec``). See :doc:`/how-to/run-with-containers`.

AGFHC
~~~~~

See the `AGFHC (AMD GPU Field Health Check) <https://instinct.docs.amd.com/projects/gpu-operator/en/latest/test/agfhc.html>`_ docs for more information.

You can list all available AGFHC test cases using the CLI:

.. code:: bash

  cvs list agfhc_cvs

.. code:: text

  Available tests in agfhc_cvs:
    - test_agfhc_hbm
    - test_agfhc_hbm1_lvl5
    - test_agfhc_hbm2_lvl5
    - test_agfhc_hbm3_lvl3
    - test_agfhc_dma_all_lvl1
    - test_agfhc_dma_lvl1
    - test_agfhc_gfx_lvl1
    - test_agfhc_pcie_lvl1
    - test_agfhc_pcie_lvl3
    - test_agfhc_xgmi_lvl1
    - test_agfhc_all_perf
    - test_agfhc_all_lvl5

Run the AGFHC test:

1. Run the installation:

   .. code:: bash

     cvs run install_agfhc \
       --cluster_file ~/cvs_workspace/cluster.json \
       --config_file ~/cvs_workspace/health/health_config.json \
       --capture=tee-sys -vvv -s

2. Run the AGFHC test:

   .. code:: bash

     cvs run agfhc_cvs \
       --cluster_file ~/cvs_workspace/cluster.json \
       --config_file ~/cvs_workspace/health/health_config.json \
       --capture=tee-sys -vvv -s

3. Run the CSP qualification test:

   .. code:: bash

     cvs run csp_qual_agfhc \
       --cluster_file ~/cvs_workspace/cluster.json \
       --config_file ~/cvs_workspace/health/health_config.json \
       --capture=tee-sys -vvv -s

.. _health-transferbench:

TransferBench
~~~~~~~~~~~~~

`TransferBench <https://rocm.docs.amd.com/projects/TransferBench/en/latest/index.html>`_ measures GPU memory bandwidth on every node. The suite runs six presets. Bandwidth presets fail when a measurement is below the matching key in ``transferbench.results``.

In the copied health file, replace or confirm these ``transferbench`` values:

.. list-table::
   :header-rows: 1
   :widths: 4 6

   * - Key
     - What to put there
   * - ``git_install_path``
     - Directory the installer clones into. The packaged value is ``/home/{user-id}/INSTALL/``. The clone lands in ``TransferBench`` under that directory.
   * - ``path``
     - Directory that contains the ``TransferBench`` binary. Keep it equal to ``<git_install_path>/TransferBench``.
   * - ``git_url`` and ``git_tag``
     - Repository and tag to build. ``git_tag`` is required. The packaged tag is ``v1.67.00``.
   * - ``results.*``
     - Bandwidth gates for your GPUs. The packaged file fills every gate. ``test_transfer_bench_a2a`` requires ``results.gpu_to_gpu_a2a_per_link_min`` (the legacy name ``gpu_to_gpu_a2a_avg`` is still accepted).

Leave these unless you are changing the layout:

.. list-table::
   :header-rows: 1
   :widths: 3 7

   * - Key
     - Packaged meaning
   * - ``rocm_path``
     - ``<changeme>`` auto-detects ``/opt/rocm`` or ``/opt/rocm/core-*``. A real path is used only when that directory contains ``libamdhip64``.
   * - ``auto_num_cpu_devices``
     - ``true``. When ``NUM_CPU_DEVICES`` is unset, CVS counts NUMA nodes that have CPUs and exports that count.
   * - ``a2asweep_timeout``
     - ``1500`` seconds for ``test_transfer_bench_a2asweep``. Omit the key to derive the timeout from ``a2asweep_combos``, ``a2asweep_sec_per_combo``, and ``a2asweep_timeout_buffer``.
   * - ``p2p_preset``
     - ``p2p``. ``podp2p`` is GPU-only and is not a substitute for the CPU-to-GPU check.

Install, then run. Install clones, checks out ``git_tag``, and builds on every node:

.. code:: bash

  cvs run install_transferbench \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/health/health_config.json

  cvs run transferbench_cvs \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/health/health_config.json

.. note::

  ``cvs run`` writes the pytest HTML report described in :ref:`pytest-html-report-and-bundle`.

Read the results
^^^^^^^^^^^^^^^^

Each preset is one row:

.. list-table::
   :header-rows: 1
   :widths: 4 6

   * - Row
     - What it checks
   * - ``test_transfer_bench_a2a``
     - Every GPU's all-to-all per-link bandwidth is at least ``gpu_to_gpu_a2a_per_link_min``.
   * - ``test_transfer_bench_p2p``
     - UniDir and BiDir GPU-to-GPU averages meet ``avg_gpu_to_gpu_p2p_unidir_bw`` and ``avg_gpu_to_gpu_p2p_bidir_bw``.
   * - ``test_transfer_bench_healthcheck``
     - The ``healthcheck`` preset finishes. Output that matches the suite failure scan fails the row.
   * - ``test_transfer_bench_a2asweep``
     - The ``a2asweep`` preset finishes inside ``a2asweep_timeout``.
   * - ``test_transfer_bench_scaling``
     - GPU 0 peak bandwidth meets ``best_gpu0_bw``.
   * - ``test_transfer_bench_schmoo``
     - The 32-CU local and remote read, write, and copy values meet the ``32_cu_*`` keys.

``results.gpu_to_gpu_a2a_rtotal``, when set, is recorded and is not a pass/fail gate.

On a managed-compute step, the nodes need a writable ``git_install_path`` and a route to ``git_url``. See :ref:`health-managed-compute`.

A missing all-to-all row fails with ``RTotal row not found``. A missing per-link gate fails with ``Missing gpu_to_gpu_a2a_per_link_min``. A p2p run that aborts while enumerating CPU NUMA nodes fails looking for UniDir or BiDir averages. Set ``num_cpu_devices``, or ``env.NUM_CPU_DEVICES``, to the number of NUMA nodes that have CPUs. ``auto_num_cpu_devices`` does that when the key is unset.

.. _health-rvs:

RVS
~~~

`ROCm Validation Suite (RVS) <https://rocm.docs.amd.com/projects/ROCmValidationSuite/en/latest/index.html>`_ checks GPU presence, memory, and stress on every node. The packaged file sets ``rvs_test_level`` to ``4``. On RVS 1.3.0 or newer that runs the level suite and skips the individual module rows. ``test_rvs_gpu_enumeration`` runs either way.

In the copied health file, replace or confirm these ``rvs`` values:

.. list-table::
   :header-rows: 1
   :widths: 4 6

   * - Key
     - What to put there
   * - ``rocm_path``
     - ``<changeme>`` auto-detects ``/opt/rocm`` or ``/opt/rocm/core-*``. Set a path only when that tree contains ``libamdhip64``.
   * - ``path``
     - Directory of the ``rvs`` binary. The packaged value is ``<changeme>/bin``. Install replaces ``<changeme>`` with the detected ROCm path.
   * - ``git_install_path``
     - Writable directory used when install builds from the repository. The packaged value is ``/home/{user-id}/INSTALL/rvs``.
   * - ``rvs_test_level``
     - ``0`` runs the individual module rows and skips the level row. ``1`` through ``5`` run ``rvs -r <level>`` when RVS is 1.3.0 or newer.

Leave ``config_path_mi300x`` and ``config_path_default`` unless the RVS conf tree is not under the detected ROCm prefix. Install rewrites a leading ``/opt/rocm`` in those paths to the detected prefix.

Install, then run. When ``rvs`` or ``gst_single.conf`` is already on the node, install verifies them and does not rebuild:

.. code:: bash

  cvs run install_rvs \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/health/health_config.json

  cvs run rvs_cvs \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/health/health_config.json

.. note::

  ``cvs run`` writes the pytest HTML report described in :ref:`pytest-html-report-and-bundle`.

Read the results
^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 4 6

   * - Row
     - When it runs
   * - ``test_rvs_level_config``
     - ``rvs_test_level`` is ``1``–``5`` and RVS is 1.3.0 or newer. Skipped at level ``0``, and on older RVS, with the reason in the row.
   * - ``test_rvs_gpu_enumeration``
     - Always. Fails with ``No GPUs detected`` when RVS reports no supported GPUs.
   * - ``test_rvs_mem_test``
     - Level ``0``, or RVS older than 1.3.0. Uses ``tests[]`` entry ``mem_test`` and ``mem.conf``.
   * - ``test_rvs_gst_single``
     - Same window as memory. Uses ``gst_single`` and ``gst_single.conf``.
   * - ``test_rvs_iet_stress``
     - Same window. Uses ``iet_stress`` and ``iet_stress.conf``.
   * - ``test_rvs_pebb_single``
     - Same window. Uses ``pebb_single`` and ``pebb_single.conf``.
   * - ``test_rvs_pbqt_single``
     - Same window. Uses ``pbqt_single`` and ``pbqt_single.conf``.
   * - ``test_rvs_babel_stream``
     - Same window. Uses ``babel_stream`` and ``babel.conf``.

With the packaged level ``4`` and a current RVS, the module rows skip with ``Running LEVEL-4 test instead``. Set ``rvs_test_level`` to ``0`` to run them. A module whose name is missing from ``rvs.tests`` fails with ``Test configuration for <name> not found``. A missing conf file fails with ``Configuration file [<file>] for <name> not found``.

On a managed-compute step, install does not use apt. It installs the RVS tarball when ``rvs`` or ``gst_single.conf`` is missing. See :ref:`health-managed-compute`.

.. _health-managed-compute:

Run on managed compute
~~~~~~~~~~~~~~~~~~~~~~

Follow :doc:`/how-to/run-on-managed-compute` to run these health suites on managed compute.

TransferBench adds these requirements:

- ``git_install_path`` must be writable on every allocated node.
- Every node must be able to clone ``git_url``.

RVS adds this requirement:

- On a managed-compute step, ``install_rvs`` does not use apt. When ``rvs`` or ``gst_single.conf`` is missing, it installs the RVS tarball. ``git_install_path`` must be writable on every node.
