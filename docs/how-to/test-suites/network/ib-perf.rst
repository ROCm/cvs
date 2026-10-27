.. meta::
  :description: Run CVS InfiniBand bandwidth and latency performance tests to validate RDMA fabric throughput and latency across AMD Instinct GPU cluster nodes.
  :keywords: CVS, InfiniBand, IB, RDMA, bandwidth, latency, AMD Instinct, ROCm, AMD, GPU, network, performance

**************************************************
Run CVS InfiniBand bandwidth and latency IB tests
**************************************************

InfiniBand (IB Perf) tests measure raw bandwidth and latency across the cluster fabric with the ``perftest`` tools (``ib_write_bw``, ``ib_read_bw``, ``ib_send_bw``, ``ib_write_lat``, ``ib_read_lat``, and ``ib_send_lat``). Run IB Perf after burn-in passes and before RCCL tests to validate that the interconnect delivers expected throughput.

The same tools run on RoCE. The shipped config expects GID type ``RoCE v2``.

.. _ib-perf-set-up-config:

Prerequisites
=============

- A cluster file with at least two nodes. One node cannot form a server/client pair.
- SSH key access from the machine where you run ``cvs``, as in your cluster file.
- A ROCm install visible on the nodes. Set ``rocm_dir`` in the config below, or leave it empty to auto-detect.
- ``rocm-smi`` on every node, on ``PATH`` or at ``/opt/rocm/bin``. GPU discovery uses it. ``install_system_packages`` never installs it, and the install suite fails on a node without it.
- ``numactl``, ``lshw``, the ``rdma`` command, ``libibverbs``, and ``librdmacm`` on every node. The install suite can ``apt install`` build and runtime packages when ``install_system_packages`` is ``True``. That install is skipped inside a Spur or Slurm job step.
- Passwordless sudo on the nodes if you use AFM vPOD discovery (``sudo -n afmctl``) or the dmesg scan (``sudo dmesg``). Without sudo, dmesg is skipped and AFM pairing stops.
- ``install_dir`` as a path that is the same on every node. The install suite clones and builds perftest on the head node, then every node runs ``<install_dir>/perftest/bin/<tool>``.

Set up config
=============

1. Copy the IB performance configuration file:

   .. code:: bash

     cvs config copy ibperf/ibperf_config.json --output ~/cvs_workspace/ibperf/ibperf_config.json

2. Edit ``~/cvs_workspace/ibperf/ibperf_config.json``:

   - Replace ``rocm_dir``. The value ``<changeme>`` stops the run during placeholder resolution, before auto-detect. Set a real path such as ``/opt/rocm``, or ``""`` to auto-detect ``/opt/rocm/core-*`` and then ``/opt/rocm``.
   - Set ``install_dir`` if ``/home/{user-id}/`` is not the shared directory you want. ``{user-id}`` becomes the cluster file ``username`` (then ``$USER``, then ``root``).
   - Set ``ib_bw_test_list`` and ``ib_lat_test_list`` to the perftest binaries you want. These lists are the cases that run. See :doc:`/reference/configuration-files/network/ib`.
   - Replace the ``expected_results`` numbers for your fabric before you treat a pass as a performance gate. The shipped numbers apply only to ``ib_write_bw`` at 8192 bytes.

3. Keep every other ``<changeme>`` out of this file. A remaining ``<changeme>`` exits at startup with ``Unresolved placeholder found in test config``.

For the field reference, pairing, and pass/fail rules, see :doc:`/reference/configuration-files/network/ib`.

Topology
========

Nodes are paired as server and client. With the shipped ``pairing_mode`` of ``sequential``, pair 1 is the first and second node in the cluster file, pair 2 is the third and fourth, and so on. The first node of each pair is the server. An odd last node is left out and logged. It does not by itself fail the test.

Each GPU that maps to an active RDMA NIC in its PCI domain runs its own perftest process. The client waits 5 seconds, then connects to the server. Ports start at ``port_no`` and increase by one per GPU. Bandwidth passes ``-D <duration>``, ``-q <qp>``, ``-s <bytes>``, and ``--report_gbits``. Latency passes ``-s <bytes>`` and ignores ``duration`` and ``qp_count_list``.

To test scale-out links across vPODs, set pairing in the same ``ibperf_config.json`` you copied above. The SSH user needs passwordless sudo for ``afmctl show device --json`` (often ``/usr/sbin/afmctl`` on MI4XX). The suite also supports ``intra_vpod`` and explicit ``node_dict[<node>].vpod_id`` labels with ``vpod_source`` ``cluster_file``. GPU instance count follows discovery.

.. code:: json

  "pairing_mode": "inter_vpod",
  "vpod_source": "afm",
  "afmctl_path": "/usr/sbin/afmctl"

.. _ib-perf-run-tests:

Run tests
=========

List the cases for the file you edited. ``configured_bw_tests`` and ``configured_lat_tests`` read ``ib_bw_test_list`` and ``ib_lat_test_list`` from that file. Without ``--config_file``, the list is the built-in default (``ib_write_bw``, ``ib_read_bw``, ``ib_send_bw``, ``ib_write_lat``, ``ib_send_lat``), which is not what the shipped file runs.

.. code:: bash

  cvs list ib_perf_bw_test \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/ibperf/ibperf_config.json

For the shipped lists, collection looks like this (``cvs list`` sorts names; the suite runs bandwidth cases, then latency cases, then the two chart tests):

.. code:: text

  Available tests in ib_perf_bw_test:
  --------------------------------------------------------------------------------
    • test_build_ib_bw_perf_chart
    • test_build_ib_lat_perf_chart
    • test_ib_bw_perf[ib_send_bw]
    • test_ib_bw_perf[ib_write_bw]
    • test_ib_lat_perf[ib_read_lat]
    • test_ib_lat_perf[ib_send_lat]
    • test_ib_lat_perf[ib_write_lat]

  Total: 7 tests

``test_ib_bw_perf[ib_write_bw]`` is one pytest node id. The brackets are part of the id. Quote them so the shell does not treat ``[`` as a glob.

Install perftest once, then run the suite. Both commands use the config you edited:

.. code:: bash

  cvs run install_ibperf_tools \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/ibperf/ibperf_config.json \
    --workspace ~/cvs_workspace \
    --capture=tee-sys -vvv

  cvs run ib_perf_bw_test \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/ibperf/ibperf_config.json \
    --workspace ~/cvs_workspace \
    --capture=tee-sys -vvv

Set ``install_perf_package`` to ``False`` when ``<install_dir>/perftest/bin`` already contains the ROCm build you want. The install test then logs ``Skipping perftest installation`` and does not clone again. A second install on an existing checkout runs ``git clone`` anyway and fails if ``perftest`` is already there.

Run one bandwidth case and its spreadsheet in the same command. The chart test reads measurements stored in that process:

.. code:: bash

  cvs run ib_perf_bw_test \
    'test_ib_bw_perf[ib_write_bw]' \
    test_build_ib_bw_perf_chart \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/ibperf/ibperf_config.json \
    --workspace ~/cvs_workspace \
    --capture=tee-sys -vvv

A latency case is ``'test_ib_lat_perf[ib_read_lat]'`` plus ``test_build_ib_lat_perf_chart``. An unknown name, a duplicate, or a non-list value fails collection with the ``ValueError`` from ``configured_bw_tests`` or ``configured_lat_tests``. An empty list collects no cases for that side. The chart tests are still collected.

What a pass produces
====================

``cvs run`` writes the pytest HTML report, the pytest log, and a zip under ``<workspace>/cvs_runs/<run_id>/``:

- ``ib_perf_bw_test.html`` — pytest pass/fail. A GPU under a configured bandwidth threshold fails this report.
- ``ib_perf_bw_test_html/ib_perf_bw_test_run_deck.html`` and ``.json`` — the IB Perf run deck, linked from the pytest report as ``ibperf Run Deck``. The deck is render-only. An outlier flag (bandwidth under 95% of the fleet median at the largest size, or p99 latency over 2× the median p99) does not fail the test.
- ``ib_perf_bw_test.log`` — the pytest log.
- ``ib_perf_bw_test_<timestamp>.zip`` — the pytest HTML, the run deck, and per-test logs. It does not contain the spreadsheets.

On a machine that is not inside a job step, ``<run_id>`` is ``local-YYYYMMDD-HHMMSS``. Pass ``--workspace`` (or set ``CVS_WORKSPACE``) to choose the parent directory. A Spur or Slurm step uses the job id and requires that workspace.

The chart tests write workbooks in the directory where you started ``cvs``, not in the run directory:

- ``ib_bw_pps_perf.xlsx`` — one sheet per bandwidth test and QP count, bandwidth in Gbps and the perftest message rate, plus a cluster-average chart.
- ``ib_lat_perf.xlsx`` — one sheet per latency test, with min, max, average, stdev, and 99th percentile in microseconds.

Per-node perftest output is ``/tmp/ib_perf_<gpu-index>_logs``. Each message size deletes and rewrites those files. They are not copied into the run directory.

Troubleshooting
===============

Search the pytest log for ``FAIL -``. The test then ends with ``Following FAILURES seen``.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - What you see
     - What it means
   * - ``Unresolved placeholder found in test config``
     - A ``<changeme>`` is still in the config, including ``rocm_dir``.
   * - ``At least 2 nodes are required to run this test``
     - The install suite saw fewer than two nodes in the cluster file.
   * - ``No ibperf node pairs could be formed``
     - Pairing produced no server/client pair. Check ``pairing_mode`` and vPOD labels.
   * - ``ibperf.vpod_source afm needs passwordless sudo for afmctl``
     - AFM mode could not run ``sudo -n afmctl``.
   * - ``perftest runtime dependencies missing on node``
     - ``numactl``, ``lshw``, ``rdma``, ``libibverbs``, or ``librdmacm`` is missing and ``install_system_packages`` is not installing them.
   * - ``rocm-smi not found on node``
     - ``rocm-smi`` is neither on ``PATH`` nor at ``/opt/rocm/bin``. Without it GPU discovery finds no cards. Install ROCm with ``rocm-smi`` on that node or in the image.
   * - ``ib_write_bw not showing expected use_rocm output``
     - The in-tree binary ``<install_dir>/perftest/ib_write_bw`` did not advertise GPUDirect RDMA. The perf tests invoke ``<install_dir>/perftest/bin/<tool>``.
   * - ``lacks ROCm DMA-BUF support``
     - ``require_dmabuf`` is on and ``--use_rocm_dmabuf`` is missing from ``--help``. Rebuild perftest or set ``require_dmabuf`` to ``False``.
   * - ``No active backend RDMA NIC in the PCI domain of``
     - That GPU has no active backend NIC in its PCI domain. The test stops before perftest.
   * - ``has different or missing GPU cards``
     - The two nodes of a pair discovered different card lists. That pair is excluded and the test fails.
   * - ``GPU Buffer allocation failed or Connection not setup``
     - The perftest log has no ``bytes of GPU buffer`` line. Check the GID index, port, and that the server started.
   * - ``Client did not complete for msg size``
     - No result row for that size. Bandwidth waits ``duration`` plus 60 seconds for perftest to exit, then up to 80 seconds for a line. Latency waits up to 60 seconds for exit and up to 20 seconds for a line. ``duration`` is not part of the latency wait.
   * - ``Actual BW ... Gbps less than the expected BW``
     - A GPU instance was under ``expected_results`` for that test, size, and QP count. Units are Gbps.
   * - ``Actual latency ... us greater than the expected latency``
     - A GPU instance's average latency (``t_avg``) was over the threshold. Units are microseconds.
   * - ``ERROR - Failue pattern``
     - The dmesg scan matched a known error line. The word ``Failue`` is the text CVS prints.
   * - ``No latency results for``
     - ``test_build_ib_lat_perf_chart`` ran for a test that recorded no rows.

``No expected result configured ... not gated`` is informational. That size ran and was not compared to a threshold.

Run on managed compute
======================

Follow :doc:`/how-to/run-on-managed-compute` to run IB Perf on managed compute. Inside a job step, set ``--workspace`` (or ``CVS_WORKSPACE``) to storage every node can see. ``install_system_packages`` does not run ``apt`` in that step.

Run in containers
=================

Follow :doc:`/how-to/run-with-containers` and set ``"orchestrator": "container"`` in the cluster file. The install and test suites start the container on every node in their first step, or attach to the one already running, and run the perftest build, perftest, and the GPU and NIC checks inside it. With ``persistent`` the container keeps running after the test suite; remove it yourself. The dmesg scan and AFM vPOD discovery stay on the host, so the dmesg scan needs passwordless sudo on the hosts, as on baremetal.

IB Perf needs these container settings:

- ``container.lifetime`` ``persistent``. The install and test suites are separate runs, and a ``per_run`` container is removed between them with anything installed in it.
- ``install_dir`` on a volume in ``container.runtime.args.volumes`` that every node shares, mounted at the same path in the container. The build runs in the head node's container and every node runs the result.
- An image with ROCm at ``rocm_dir`` and ``rocm-smi``, even when ``install_system_packages`` is ``True``. The install suite never installs either, and it fails on a node without ``rocm-smi``. The image also needs ``lshw``, ``rdma``, ``numactl``, ``libibverbs``, and ``librdmacm``, unless ``install_system_packages`` is ``True`` and the image has ``sudo``.

.. _ib-perf-dmesg-scan:

Kernel log (dmesg) scan
=======================

After each bandwidth and latency run, the suite reads ``dmesg`` on every node from the minute the run started and fails the test if it finds a known error pattern. The end time is not applied (``till_end_flag=True``), so the scan continues through the kernel log collected at the end of that message size. The log is parsed with AMD node-scraper (the ``amd-node-scraper`` package installed with CVS), extended with the CVS error patterns.

If any node lacks passwordless sudo, the scan is skipped and the log says ``SKIPPING the dmesg error check: no passwordless sudo to read dmesg on one or more nodes``. The bandwidth and latency numbers can still pass.
