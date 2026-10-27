:orphan:

.. meta::
  :description: ARC configuration keys for controller-based rack validation in CVS.
  :keywords: ARC, AMD Rack Check, CVS, rack, JSON, controllers

***************************************************
ARC (AMD Rack Check) configuration file reference
***************************************************

ARC reads two files: ``arc_config.json`` for ARC settings, and the cluster file's ``racks`` block for each rack's inventory. CVS builds one ARC rack file per rack from that block, see `Rack inventory in the cluster file`_.

The sample file is ``cvs/input/config_file/arc/arc_config.json``. Replace the ``<changeme>`` value before running. Unresolved placeholders stop the run at startup.

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
          "rack_config": "",
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
     - Deprecated. Used only when the cluster file has no ``racks`` block: a list of ``node_dict`` keys to act as controllers, empty for the first node. Set ``racks.<rack_id>.arc_controller`` instead.
   * - ``rack_config``
     - Optional. Path to a rack JSON you staged on every controller, at the same path on each. When set, CVS uses it and does not generate rack files. Leave empty to generate them from the cluster file.
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

Deb and rpm packages install at ``/opt/amdtools/arc``. Tar packages extract under ``ARC_INSTALL_PATH`` (default ``/opt/amdtools``), giving ``<prefix>/arc/arc.py``. The download is staged in a private temp directory and cleaned afterward. Controllers must have passwordless SSH and passwordless sudo for ARC tools.

Each rack has one controller: ``racks.<rack_id>.arc_controller``, or the first ``node_dict`` entry with that ``rack_id`` when it is not set. Only controllers are contacted.

.. _arc-rack-inventory:

Rack inventory in the cluster file
==================================

ARC needs a rack file that names each tray's addresses and logins. Instead of writing one per rack, describe the racks in the cluster file. CVS starts from the ARC template for each rack's ``platform``, fills it from the fields below, validates it on that rack's controller, and runs ARC against it. To create or update the file, see :doc:`/reference/cluster/rack-cluster-file`.

.. dropdown:: ``racks`` block (example)

   .. code:: json

      {
        "node_dict": {
          "10.0.0.1": {"bmc_ip": "10.0.1.1", "vpc_ip": "10.0.0.1", "rack_id": "rack-01"},
          "10.0.0.2": {"bmc_ip": "10.0.1.2", "vpc_ip": "10.0.0.2", "rack_id": "rack-01"}
        },
        "racks": {
          "switch_ssh_user": "admin",
          "switch_ssh_password": "<changeme>",
          "bmc_user": "<changeme>",
          "bmc_password": "<changeme>",
          "rack-01": {
            "platform": "HeliosP",
            "switch_model": "CelesticaPolarisSwitch",
            "switch_trays": ["10.0.2.1", "10.0.2.2", "10.0.2.3", "10.0.2.4", "10.0.2.5", "10.0.2.6"],
            "switch_bmc": ["10.0.3.1", "10.0.3.2", "10.0.3.3", "10.0.3.4", "10.0.3.5", "10.0.3.6"],
            "rmc": ["10.0.4.1"]
          }
        }
      }

Only ``platform`` and the compute nodes are required. Leave out any device list for hardware you cannot reach, and set ``arc.spec_mode`` to ``partial``: ``full`` requires every device the platform declares. Content that needs a missing device cannot run.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Meaning
   * - ``node_dict`` entries with a ``rack_id``
     - Compute trays of that rack, **in tray order**. ``vpc_ip`` (or the node key when it is not set) becomes the tray's address as the controller reaches it, ``bmc_ip`` its BMC (``NA`` leaves it out), and the cluster file's ``username`` and ``priv_key_file`` (or ``password``) its login. ARC checks that ``priv_key_file`` exists on the controller.
   * - ``racks.bmc_user``, ``racks.bmc_password``
     - One login for every BMC in the file: compute ``bmc_ip``, ``switch_bmc``, ``rmc``, and ``power_shelves``. Required when ``rmc`` or ``power_shelves`` is set. Without it, CVS leaves every BMC out.
   * - ``racks.switch_ssh_user``, ``racks.switch_ssh_password``, ``racks.switch_ssh_key_file``
     - Login for every switch host. ``cvs exec --target switches`` uses the same login.
   * - ``platform``
     - ARC platform: ``HeliosP``, ``HeliosR``, or ``HeliosM``.
   * - ``arc_controller``
     - Host that runs ARC for this rack. Default: the first ``node_dict`` entry with this ``rack_id``.
   * - ``switch_model``
     - Scale-up switch model, needed with ``switch_trays`` when the platform accepts more than one.
   * - ``switch_trays``
     - Scale-up switch host addresses, in tray order.
   * - ``switch_bmc``
     - Scale-up switch BMC addresses, in the same order as ``switch_trays``.
   * - ``rmc``
     - Rack management controller addresses, in tray order. HeliosM has four.
   * - ``front_end_switches``, ``back_end_switches``
     - Front-end and back-end switch host addresses, in tray order.
   * - ``power_shelves``
     - HeliosM only. Power shelf addresses, in tray order.

Tray order
----------

The position of a device in its list is its ARC tray number, and ARC ties each tray number to one slot in the rack. Scale-up tests derive accelerator IDs, MAC addresses, and switch ports from it, so a node listed out of order fails those tests or reports faults against the wrong tray. List devices in the order of their slots:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Device
     - Order on HeliosP, HeliosR, and HeliosM
   * - Compute trays
     - U9 to U17, then U25 to U33 (18 trays)
   * - Scale-up switches
     - U19 to U24

For other device types, run ``arc.py --generate <platform> --id-location`` on a controller. It lists each type's slots in tray order with their row and position.

Serial numbers are generated from the rack and slot, for example ``rack-01-compute-0``. A list longer than the platform's slots, or a field still set to ``<changeme>``, stops the run before ARC starts.

For suite commands and disruptive content, see :doc:`/how-to/test-suites/burn-in-diag/arc`.
