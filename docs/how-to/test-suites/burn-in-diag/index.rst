.. meta::
  :description: Run burn-in and diagnostic CVS test suites
  :keywords: CVS, platform, health, preflight, burn-in, diag

********************
Burn-in / Diag tests
********************

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

See also :doc:`/how-to/run-tests/index` for common ``cvs run`` flags and workflow.
