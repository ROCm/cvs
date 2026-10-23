:orphan:

.. meta::
  :description: ARC configuration keys for controller-based rack validation in CVS.
  :keywords: ARC, AMD Rack Check, CVS, rack, JSON, controllers

***************************************************
ARC (AMD Rack Check) configuration file reference
***************************************************

The sample file is ``cvs/input/config_file/arc/arc_config.json``. Replace both ``<changeme>`` values before running. Unresolved placeholders stop the run at startup.

.. dropdown:: ``arc_config.json`` (example)

   .. code:: json

      {
        "arc": {
          "description": "AMD Rack Check",
          "inactivity_timeout": 900,
          "run_timeout": 14400,
          "install_timeout": 1800,
          "arc_version": "1.8.1",
          "arc_release_url": "<changeme>",
          "ARC_INSTALL_PATH": "",
          "controller_nodes": [],
          "rack_config": "<changeme>",
          "spec_mode": "full",
          "mock": "False",
          "continue_on_failure": "False",
          "allow_disruptive": "False",
          "print_all_to_console": "True",
          "ADD_ARC_LOGS_TO_HTML_REPORTS": "False"
        }
      }

Configuration keys
==================

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Key
     - Meaning
   * - ``description``
     - Human-readable suite label.
   * - ``inactivity_timeout``
     - Seconds the SSH channel may stay silent before CVS abandons the run (default 900, minimum 120). CVS prints a heartbeat every 60 s while ARC is alive, so this fires only when the controller shell or SSH stalls.
   * - ``run_timeout``
     - Wall-clock limit for one ARC run, in seconds (default 14400). ``0`` disables it. ARC enforces its own per-item limits, for example 5400 s for ``ANCGPUTest``. Raise this for flows that chain long items.
   * - ``install_timeout``
     - Seconds without installer output before an install times out (default 1800).
   * - ``arc_version``
     - Minimum installed release. Every controller must satisfy it; the release URL version must be at least this version.
   * - ``arc_release_url``
     - Direct ARC ``.deb``, ``.rpm``, or ``.tar.gz`` URL.
   * - ``ARC_INSTALL_PATH``
     - Tar-only prefix. Blank uses ``/opt/amdtools``; deb/rpm packages use their fixed ``/opt/amdtools/arc`` location.
   * - ``controller_nodes``
     - List of cluster ``node_dict`` keys to act as rack controllers. Empty selects the first node. Only these hosts are contacted.
   * - ``rack_config``
     - Absolute path to a rack JSON on every controller. Each file describes its controller's rack. CVS does not generate or distribute these files yet; stage one per controller before the run.
   * - ``spec_mode``
     - ``full`` or ``partial`` rack validation.
   * - ``mock``
     - ``True`` sets ``MOCK=1``. ARC release builds reject ``--mock``; mock data may only cover example content.
   * - ``continue_on_failure``
     - ``True`` passes ARC's ``--continue-on-failure`` flag.
   * - ``allow_disruptive``
     - ``True`` enables state-changing flow and item suites; default ``False`` skips them before install.
   * - ``print_all_to_console``
     - ``True`` streams ARC output. ``False`` hides it. Both modes print a heartbeat line every 60 s and the run sentinels.
   * - ``ADD_ARC_LOGS_TO_HTML_REPORTS``
     - ``True`` attaches controller log tarballs on all tests; ``False`` attaches them on failures only.

Timeouts and cleanup
====================

CVS starts ARC in its own process group on the controller. When ``arc.run_timeout`` passes, or CVS stops listening because of an inactivity abort or interrupted run, the controller shell sends SIGTERM to that group, waits up to 30 s, then sends SIGKILL. A run that exceeds ``arc.run_timeout`` fails with a ``timed out after N s`` reason.

Processes ARC starts as root through ``sudo``, in a separate session, or on trays over its own SSH (for example ANC on the GPUs) are outside that group and may keep running. After a timeout, check the trays with ``pgrep -af anc`` and stop leftovers manually.

Install location and controllers
================================

Deb and rpm packages install at ``/opt/amdtools/arc``. Tar packages extract under ``ARC_INSTALL_PATH`` (default ``/opt/amdtools``), giving ``<prefix>/arc/arc.py``. The download is staged in a private temp directory and cleaned afterward. Controllers must have passwordless SSH and passwordless sudo for ARC tools. The cluster-file schema is unchanged.

For suite commands and disruptive content, see :doc:`/how-to/test-suites/burn-in-diag/arc`.
