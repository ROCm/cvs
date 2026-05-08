.. note::

  **Scope today.** Of the test suites shipped with CVS, only ``rvs_cvs`` consumes the orchestrator and honors the ``orchestrator`` key in the cluster file. All other ``cvs run`` test suites and the ``cvs exec`` CLI run on the host regardless of the ``orchestrator`` value. Migrating additional suites to the orchestrator is tracked separately. Custom Python scripts can use the ``OrchestratorFactory`` API directly as an escape hatch.
