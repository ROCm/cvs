These arguments are commonly passed to ``cvs run``:

- ``--cluster_file``: Cluster JSON with node list, SSH credentials, and execution backend. See :doc:`/reference/cluster/cluster-file`.
- ``--config_file``: Per-suite test configuration under ``cvs/input/config_file/``. See :doc:`/reference/configuration-files/index`.
- ``--html``: PyTest HTML report path with pass/fail summary and log links.
- ``--capture=tee-sys``: Capture stdout/stderr from tests.
- ``--self-contained-html``: Single HTML report with embedded styling and images.
- ``--log-file``: Text log file for Python logger output.
- ``-vvv``: Increase pytest verbosity.
- ``-s``: Disable output capturing (print statements appear in the console).

All pytest options can be passed through ``cvs run``. For the full option list see :doc:`/reference/cli/cvs-run`.
