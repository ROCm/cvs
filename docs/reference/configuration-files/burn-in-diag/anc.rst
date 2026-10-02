.. meta::
  :description: Reference for the ANC (AMD Node Check) configuration file in CVS, covering install options, CPU and GPU diagnostic groups, timeouts, and log collection.
  :keywords: ANC, AMD Node Check, CVS, ROCm, burn-in, GPU, CPU, diagnostic, cluster, validation, JSON, AMD

*************************************************
ANC (AMD Node Check) configuration file reference
*************************************************

AMD Node Check (ANC) suites install the ANC tool on every node, then run CPU or GPU diagnostic groups. Shared logic lives in ``cvs/lib/anc_lib.py``; group names come from ``CPU_GROUPS`` and ``GPU_GROUPS``.

Configuration file location: ``cvs/input/config_file/anc/anc_config.json``

For the suite overview, install flavours, and artifact layout, see
`cvs/tests/anc/README.md <https://github.com/ROCm/cvs/blob/main/cvs/tests/anc/README.md>`_
in the repository.

Run ANC
=======

Run the three ANC suites in order: install first, then CPU and GPU diagnostics.

.. code:: bash

  cvs run anc_installation \
    --cluster_file cluster.json \
    --config_file cvs/input/config_file/anc/anc_config.json

  cvs run anc_test_cpu \
    --cluster_file cluster.json \
    --config_file cvs/input/config_file/anc/anc_config.json

  cvs run anc_test_gpu \
    --cluster_file cluster.json \
    --config_file cvs/input/config_file/anc/anc_config.json

Replace ``<changeme>`` in ``anc_release_url`` and ``log_folder_path`` before running. An unresolved placeholder aborts the run before any node is contacted.

Sample configuration
====================

Keys prefixed with ``_comment`` are documentation only and ignored at runtime.

.. note::

  In this configuration file, ``{home}`` and ``{user-id}`` are resolved at load time. A leading ``~`` is expanded to the runner user's home directory.

.. dropdown:: ``anc_config.json`` (example)

  .. code:: json

    {
      "anc": {
        "description": "AMD Node Check",
        "inactivity_timeout": 900,
        "install_timeout": 1800,
        "anc_version": "1.4.9",
        "anc_release_url": "<changeme>",
        "ANC_INSTALL_PATH": "",
        "print_all_to_console": "True",
        "log_folder_path": "<changeme>",
        "ADD_ANC_LOGS_TO_HTML_REPORTS": "False",
        "COLLECT_HTML_REPORTS": "True"
      }
    }

Key parameters
==============

The following table describes each key in the ``anc`` configuration block.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Key
     - Meaning
   * - ``inactivity_timeout``
     - Per-group inactivity timeout in seconds (default ``900``). A group is aborted only after this many seconds with **no new ANC output**. There is no total wall-clock cap.
   * - ``install_timeout``
     - Download and install inactivity timeout in seconds (default ``1800``), used only by ``anc_installation`` / the install pre-task. Not a total budget: a progress heartbeat keeps a slow download alive; a genuine stall still fails.
   * - ``anc_version``
     - Expected ANC version. Install skips when this version is already present and post-verifies the match. When set, it must equal the version in ``anc_release_url``.
   * - ``anc_release_url``
     - URL of the ANC release archive. Packaging is auto-detected from the filename: **legacy (≤1.4.x)** outer tarballs include a ``-deb-`` / ``-rpm-`` / ``-tar-`` token; **direct (1.5.0+)** URLs point at a ``.deb`` / ``.rpm`` / ``.tar.gz`` with no flavour token.
   * - ``ANC_INSTALL_PATH``
     - **Tar installs only:** relocatable prefix (entrypoint ``<prefix>/anc/anc.py``). Deb/rpm packages ignore this key and install under ``/opt/amdtools``. Leave blank to keep the default ``/opt/amdtools``.
   * - ``print_all_to_console``
     - ``True`` echoes ANC group output to the console; ``False`` suppresses it (install and ldconfig diagnostics still print).
   * - ``log_folder_path``
     - Controller-side destination **prefix** for collected logs and the auto-collected HTML report. Required. CVS appends ``anc_logs/<node>/<test_name>/<timestamp>`` and ``html_reports/<node>/<test_name>/<timestamp>/``.
   * - ``ADD_ANC_LOGS_TO_HTML_REPORTS``
     - ``True`` always bundles each node's ANC log tarball into the pytest-html report. ``False`` (default) bundles tarballs only when the test fails.
   * - ``COLLECT_HTML_REPORTS``
     - ``True`` (default) auto-generates a pytest-html report under ``log_folder_path`` even without ``--html``. An explicit ``--html`` always overrides that path.

Install location
================

The following list shows where ANC is installed depending on the package format.

- **deb / rpm** — always ``/opt/amdtools/anc``
- **tar** — ``ANC_INSTALL_PATH`` (default ``/opt/amdtools``), giving ``<prefix>/anc/anc.py``

The download is staged in a private temp directory on each node and removed after install (success or failure).

Related resources
=================

- :doc:`/how-to/test-suites/burn-in-diag/anc` — ``cvs run`` examples and group lists
- :doc:`/reference/cluster/cluster-file` — Cluster topology and SSH
- `cvs/tests/anc/README.md <https://github.com/ROCm/cvs/blob/main/cvs/tests/anc/README.md>`_ — Install flavours, pass/fail criteria, and artifact layout
