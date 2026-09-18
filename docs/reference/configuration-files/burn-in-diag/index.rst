.. meta::
  :description: JSON configuration schemas for CVS burn-in and diagnostic test suites, including platform, health, preflight, and ANC tests for AMD GPU clusters.
  :keywords: CVS, burn-in, diagnostic, platform, health, preflight, ANC, JSON, ROCm, GPU, AMD, AGFHC, RVS

***************************************************************************
Cluster Validation Suite (CVS) burn-in and diagnostic test configuration schemas
***************************************************************************

JSON configuration schemas for host validation, GPU burn-in, preflight, and ANC suites under ``cvs/input/config_file/``.

- :doc:`Platform </reference/configuration-files/burn-in-diag/platform>` — host OS, BIOS, firmware, and PCIe checks
- :doc:`Health </reference/configuration-files/burn-in-diag/health>` — AGFHC, TransferBench, and RVS burn-in configs
- :doc:`Preflight </reference/configuration-files/burn-in-diag/preflight>` — node smoke and cluster preflight checks
- :doc:`ANC </reference/configuration-files/burn-in-diag/anc>` — AMD Node Check CPU and GPU diagnostic groups

How to run these suites: :doc:`/how-to/test-suites/burn-in-diag/index`.
