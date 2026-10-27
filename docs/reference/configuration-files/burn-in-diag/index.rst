.. meta::
  :description: JSON configuration schemas for CVS burn-in and diagnostic test suites, including platform, health, and preflight tests for AMD GPU clusters.
  :keywords: CVS, burn-in, diagnostic, platform, health, preflight, JSON, ROCm, GPU, AMD, AGFHC, RVS

********************************************************************************
Cluster Validation Suite (CVS) burn-in and diagnostic test configuration schemas
********************************************************************************

JSON configuration schemas for host validation, GPU burn-in, and preflight under |cvs-input-config-file|_.

- :doc:`Platform </reference/configuration-files/burn-in-diag/platform>` — host OS, BIOS, firmware, and PCIe checks
- :doc:`Health </reference/configuration-files/burn-in-diag/health>` — AGFHC, TransferBench, and RVS burn-in configs
- :doc:`Preflight </reference/configuration-files/burn-in-diag/preflight>` — node smoke and cluster preflight checks

How to run these suites: :doc:`/how-to/test-suites/burn-in-diag/index`.


.. |cvs-input-config-file| replace:: ``cvs/input/config_file/``
.. _cvs-input-config-file: https://github.com/ROCm/cvs/tree/release/cvs-0.3.0/cvs/input/config_file/
