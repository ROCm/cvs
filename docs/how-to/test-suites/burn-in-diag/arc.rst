.. meta::
  :description: Run AMD Rack Check (ARC) rack-level diagnostic flows and items from CVS controllers.
  :keywords: ARC, AMD Rack Check, CVS, rack, diagnostics, controller

***********************************
Run AMD Rack Check (ARC) rack tests
***********************************

ARC runs one controller per rack against a rack JSON. CVS installs ARC on the controllers, runs selected flows or items, and collects the full ARC log session into the CVS run directory. Non-controller nodes are not contacted.

Set up
======

Use baremetal orchestration, passwordless SSH to each controller, passwordless sudo for ARC's tools, and Python 3.9 or newer. Generate each controller's rack JSON with ``arc.py --generate <platform> --rack <file>``, then check it with ``arc.py --rack <file> --validate --spec-mode <mode>``. Each controller has its own rack description at the same configured path.

Copy ``arc/arc_config.json`` with ``cvs config copy`` and replace both ``<changeme>`` values. ``arc.controller_nodes`` lists keys from the cluster file's ``node_dict``; an empty list selects its first key. ``arc.rack_config`` is the path on each controller. The cluster file needs no rack-specific fields. See :doc:`/reference/configuration-files/burn-in-diag/arc` for all settings.

Run tests
=========

.. code:: bash

   cvs list arc_test_diag
   cvs run arc_installation --cluster_file cluster.json --config_file arc_config.json
   cvs run arc_test_examples test_example_pass --cluster_file cluster.json --config_file arc_config.json
   cvs run arc_test_diag test_PingTest --cluster_file cluster.json --config_file arc_config.json

The eight generated suites are ``arc_test_examples``, ``arc_test_flows``, ``arc_test_disruptive_flows``, ``arc_test_diag``, ``arc_test_telemetry``, ``arc_test_anc``, ``arc_test_workload``, and ``arc_test_disruptive``. The example suite is an integration self-test; ``example_failure`` fails by design. Diagnostic flows ``diags_lvl1`` and ``diags_lvl2`` clear AMC log stores as PRE/POST conditions. Content missing on a platform reports n/a and fails its test.

.. warning::

   ``arc_test_disruptive_flows`` and ``arc_test_disruptive`` require ``arc.allow_disruptive`` set to ``"True"``. Their tests skip before installation when it is false, even in mock mode.

   ``heliosm_mfg_l11``, ``heliosp_mfg_l11``, ``heliosr_mfg_l11``, and ``reboot_all`` reboot or power-cycle rack hardware. ``scale_up_config`` and ``scale_up_verify`` reconfigure EAM, AFM, or switches.

   ``ComputeDCCycle``, ``RebootComputeTest``, ``RebootRmcTest``, and ``RebootSwitchTest`` cycle components. ``PortToggle``, ``ScaleOutConnectivityTestID``, ``ScaleOutConnectivityTestL2``, and ``ScaleOutConnectivityTestL3`` change ports, host namespaces, or routes.

   ``ScaleUpConnectivityTest``, ``ScaleupEAMConfig``, and ``ScaleupSwitchConfig`` reconfigure scale-up fabric. ``PRBSCheck``, ``PRBSCheckEAMFirst``, and ``SBRTest`` run link stress diagnostics. ``SwitchInternalLoopback`` changes switch loopback.

Verdicts and logs
=================

A controller passes when the ARC exit code is zero and ``ARC/summary.json`` reports ``SUCCESS`` or ``WARNING``. CVS collects ``<run_dir>/arc_logs/<node>/<test_name>/<timestamp>/<session>/ARC/`` and links ``summary.json`` and ``errors.json`` in the HTML report. Failures attach the entire log tree as a tarball. The Run Deck shows an ARC version row and per-controller item failures. ARC's controller-side ``$HOME/arc_logs`` is retained for review.

A run that exceeds ``arc.run_timeout`` fails with a timed-out reason; see the :doc:`configuration reference </reference/configuration-files/burn-in-diag/arc>` for what CVS stops.
