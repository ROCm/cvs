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

``anc_release_url`` is optional: set it only when ANC must be downloaded and installed. When ANC is already installed, each suite runs against the installed version without a URL. The URL, ``anc_version``, and ``ANC_INSTALL_PATH`` are each validated only at the point they are used, not before the run.

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
        "anc_version": "1.7.3",
        "anc_release_url": "",
        "ANC_INSTALL_PATH": "",
        "print_all_to_console": "True",
        "ADD_ANC_LOGS_TO_HTML_REPORTS": "False"
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
     - Optional minimum ANC version. When unset, each suite runs against whatever version is installed. When set, a node below it triggers an install **only if** ``anc_release_url`` can satisfy it (its version >= ``anc_version``); otherwise the ``anc_installation`` suite fails, while normal group/item suites log the shortfall and proceed with the installed version. A release candidate counts as its base release (``1.7.0-rc.1`` is treated as ``1.7.0``); the rc number is never compared. Validated (must be a dotted release like ``1.7.0``) only when it is compared.
   * - ``anc_release_url``
     - **Optional** URL of the ANC release archive, used only when ANC must be installed (absent on a node, or below ``anc_version`` with a URL new enough to upgrade). Leave blank when ANC is preinstalled. Packaging is auto-detected from the filename: **legacy (≤1.4.x)** outer tarballs include a ``-deb-`` / ``-rpm-`` / ``-tar-`` token; **direct (1.5.0+)** URLs point at a ``.deb`` / ``.rpm`` / ``.tar.gz`` with no flavour token. Parsed and shell-safety-checked only on the install path.
   * - ``ANC_INSTALL_PATH``
     - **Tar installs only:** relocatable prefix (entrypoint ``<prefix>/anc/anc.py``), also where each suite probes for an existing install when set. Deb/rpm packages ignore this key and install under ``/opt/amdtools``. Validated (shell-safe, non-root) when the prefix is resolved for use. Leave blank to keep the default ``/opt/amdtools``.
   * - ``print_all_to_console``
     - ``True`` echoes ANC group output to the console; ``False`` suppresses it (install and ldconfig diagnostics still print).
   * - ``ADD_ANC_LOGS_TO_HTML_REPORTS``
     - ``True`` always bundles each node's ANC log tarball into the pytest-html report. ``False`` (default) bundles tarballs only when the test fails.

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
