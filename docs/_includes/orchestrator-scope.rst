.. note::

  **Current scope**: Of the test suites shipped with CVS, ``rvs_cvs`` / ``install_rvs``, ``transferbench_cvs``, ``agfhc_cvs`` / ``install_agfhc``, and ``csp_qual_agfhc`` consume the orchestrator and honor the ``orchestrator`` key in the cluster file (including Spur managed HTTP transport and container ``docker exec``). Other ``cvs run`` suites and the ``cvs exec`` CLI may still run on the host regardless of the ``orchestrator`` value. Migrating additional suites is tracked separately. Custom Python scripts can use the ``OrchestratorFactory`` API directly as an escape hatch.
