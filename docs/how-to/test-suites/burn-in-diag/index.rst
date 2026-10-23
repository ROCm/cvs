.. meta::
  :description: Run CVS burn-in and diagnostic test suites to validate GPU cluster health before network, training, or inference workloads on AMD Instinct hardware.
  :keywords: CVS, burn-in, diagnostic, GPU, AMD Instinct, ROCm, AMD, health, AGFHC, RVS, TransferBench, preflight

*******************************************************************
Run Cluster Validation Suite (CVS) burn-in and diagnostic test suites
*******************************************************************

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
   and every other metric) are **illustrative sample values, not final tuned
   results**. They are shown only to illustrate the report layout.

HTML test report
----------------

.. image:: /images/health_pytest_summary.png
   :alt: RVS pytest HTML report summary

Run Deck report
---------------

The Run Deck dashboard: run overview, measurement charts, heatmaps, and
detailed results.

.. image:: /images/health_rundeck_summary.png
   :alt: RVS Run Deck overview

.. image:: /images/health_measurements.png
   :alt: RVS Run Deck measurement charts

.. image:: /images/health_heatmaps.png
   :alt: RVS Run Deck heatmaps

.. image:: /images/health_results.png
   :alt: RVS Run Deck detailed results
