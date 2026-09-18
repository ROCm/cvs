.. meta::
  :description: Run CVS burn-in and diagnostic test suites to validate GPU cluster health before network, training, or inference workloads on AMD Instinct hardware.
  :keywords: CVS, burn-in, diagnostic, GPU, AMD Instinct, ROCm, AMD, health, AGFHC, RVS, TransferBench, preflight, ANC

*******************************************************************
Run Cluster Validation Suite (CVS) burn-in and diagnostic test suites
*******************************************************************

Host validation, GPU burn-in, preflight checks, and AMD Node Check (ANC) diagnostics run before network, training, or inference workloads.

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
   * - ANC (AMD Node Check)
     - :doc:`/how-to/test-suites/burn-in-diag/anc`
     - :doc:`/reference/configuration-files/burn-in-diag/anc`

See also :doc:`/how-to/test-suites/index` for common ``cvs run`` flags and workflow.
