.. meta::
  :description: Run CVS burn-in and diagnostic test suites to validate GPU cluster health before network, training, or inference workloads on AMD Instinct hardware.
  :keywords: CVS, burn-in, diagnostic, GPU, AMD Instinct, ROCm, AMD, health, AGFHC, RVS, TransferBench, preflight

*********************************************************************
Run Cluster Validation Suite (CVS) burn-in and diagnostic test suites
*********************************************************************

Host validation, GPU burn-in, and preflight checks run before network, training, or inference workloads.

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - Suite
     - How to run
     - Config reference
   * - Platform
     - :doc:`/how-to/test-suites/burn-in-diag/platform`
     - :doc:`/reference/configuration-files/burn-in-diag/platform`
   * - Health (burn-in)
     - :doc:`/how-to/test-suites/burn-in-diag/health`
     - :doc:`/reference/configuration-files/burn-in-diag/health`
   * - Preflight
     - :doc:`/how-to/test-suites/burn-in-diag/preflight`
     - :doc:`/reference/configuration-files/burn-in-diag/preflight`

See also :doc:`/how-to/test-suites/index` for common ``cvs run`` flags and workflow.

Test results
============

Sample output snapshots from a health burn-in run.

.. important::

   The numbers in these sample reports and screenshots (bandwidth, GFLOPS,
   and every other metric) are illustrative sample values, not final tuned
   results. They are shown only to illustrate the report layout.

HTML test report
----------------

.. image:: /images/health_pytest_summary.png
   :alt: RVS pytest HTML report summary

The pytest HTML report shows the environment block (CVS version, cluster file,
config file), a link to the Run Deck, and a per-test result table with pass,
fail, and skip verdicts for each RVS test and duration.

Run Deck report
---------------

The Run Deck dashboard: run overview, measurement charts, heatmaps, and
detailed results.

.. image:: /images/health_rundeck_summary.png
   :alt: RVS Run Deck overview

The overview shows the cluster-wide pass rate, pass and fail counts by
node-group cell, and filter controls for narrowing results by node, group,
measurement, or status.

.. image:: /images/health_measurements.png
   :alt: RVS Run Deck measurement charts

The measurements view shows one bar chart per precision type (fp8, fp16, bf16,
fp32) with per-node bars and a dashed acceptance threshold line.

.. image:: /images/health_heatmaps.png
   :alt: RVS Run Deck heatmaps

The heatmaps view shows peer-to-peer bandwidth matrices for PBQT xGMI
(GPU-to-GPU, in GB/s) and memory bandwidth tables for Babel (read, write, copy,
and compute kernels, in MB/s), one panel per node.

.. image:: /images/health_results.png
   :alt: RVS Run Deck detailed results

The detailed results table lists every node-group cell with its pass or fail
status and the message returned by RVS.
