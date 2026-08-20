**Mandatory**

- ``--config_file``: Per-suite test configuration under ``cvs/input/config_file/``. See :doc:`/reference/configuration-files/index`.
- ``--cluster_file``: Cluster JSON with node list, SSH credentials, and execution backend. Required outside scheduler-managed (Slurm/spur) runs. See :doc:`/reference/cluster/cluster-file`.

**Optional**

- ``--workspace``: Shared-filesystem root for this run. The HTML report and text log default to ``<workspace>/cvs_runs/<run_id>/``. Falls back to ``$CVS_WORKSPACE`` then the venv parent directory. Scheduler-managed runs in a container must set this explicitly; the venv parent is not on shared storage there. In a scheduler-managed run (Slurm or SPUR) the container orchestrator bind-mounts the run directory into the container at the same path, so ``{run_dir}`` resolves identically on the host and inside the container.
- ``--html``: HTML report path override. Omit to use the run-directory default; ``--no-html`` skips the auto-derived report.
- ``--self-contained-html``: Embed CSS/JS in the HTML report for a portable single file. Implied when the HTML path is auto-derived; pass it with an explicit ``--html`` path if you want a portable override.
- ``--log-file``: Text log path override. Omit to use the run-directory default; ``--no-log-file`` skips the auto-derived file (console logging is unaffected).
- ``--capture=tee-sys``: Capture stdout/stderr from tests while still printing to the console.
- ``-vvv``: Increase pytest verbosity.
- ``-s``: Disable pytest output capturing (print statements appear in the console).

All pytest options can be passed through ``cvs run``. For the full option list see :doc:`/reference/cli/cvs-run`.
