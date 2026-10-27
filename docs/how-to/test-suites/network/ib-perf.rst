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
- A ROCm install on every node. Set ``rocm_dir`` in the config below, or ``""`` to auto-detect.
- ``amd-smi`` on every node, on ``PATH`` or at ``/opt/rocm/bin``. GPU discovery uses it. ``install_system_packages`` never installs it, and the install suite fails on a node without it.
- On every node: ``numactl``, ``libibverbs``, ``librdmacm``, ``lshw``, and the ``rdma`` command. The install suite can ``apt install`` the perftest build and runtime packages, ``lshw``, and ``iproute2`` (for ``rdma``) when ``install_system_packages`` is ``True``. That install is skipped inside a SPUR or Slurm job step.
- On the head node, when ``install_perf_package`` is ``True``: ``git``, ``build-essential``, the autotools build chain, ``pkg-config``, the ``libibverbs``, ``librdmacm``, ``libibumad``, and ``libpci`` development packages (the set ``install_system_packages`` installs), and access to ``https://github.com/linux-rdma/perftest``.
- Each backend NIC has a network interface (on InfiniBand, an IPoIB interface). Ports without one are not used.
- Each client reaches its server at the address the cluster file uses for that node, on TCP ports ``port_no`` up to ``port_no`` plus the GPU count minus one.
- ``install_dir`` on a filesystem shared by every node at the same path, such as a shared home. The install suite builds perftest on the head node only, and every node runs ``<install_dir>/perftest/bin/<tool>``.
- Passwordless sudo on the nodes if you use AFM vPOD discovery (``sudo -n afmctl``) or want the dmesg scan (``sudo dmesg``). Without sudo, the dmesg scan is skipped and AFM pairing stops.

Set up config
=============

1. Copy the IB performance configuration file:

   .. code:: bash

     cvs config copy ibperf/ibperf_config.json --output ~/cvs_workspace/ibperf/ibperf_config.json

2. Edit ``~/cvs_workspace/ibperf/ibperf_config.json``:

   - Replace ``rocm_dir``. The value ``<changeme>`` stops the run during placeholder resolution, before auto-detect. Set a real path such as ``/opt/rocm``, or ``""`` to auto-detect ``/opt/rocm/core-*`` and then ``/opt/rocm``.
   - Set ``install_dir`` if ``/home/{user-id}/`` is not the shared directory you want. ``{user-id}`` becomes the cluster file ``username`` (then ``$USER``, then ``root``).
   - Set ``ib_bw_test_list`` and ``ib_lat_test_list`` to the perftest binaries you want. These lists are the cases that run. See :doc:`/reference/configuration-files/network/ib`.
   - Set ``gid_index`` for your NICs, or ``auto`` to select the lowest index usable on every NIC. The shipped ``3`` is checked against ``gid_type`` before perftest starts.
   - Replace the ``expected_results`` numbers for your fabric before you treat a pass as a performance gate. The shipped numbers apply only to ``ib_write_bw`` at 8192 bytes.

3. No other shipped field contains ``<changeme>``. A remaining ``<changeme>`` exits at startup with ``Unresolved placeholder found in test config``.

For the field reference, pairing, and pass/fail rules, see :doc:`/reference/configuration-files/network/ib`.

Topology
========

Nodes are paired as server and client. With the shipped ``pairing_mode`` of ``sequential``, pair 1 is the first and second node in the cluster file, pair 2 is the third and fourth, and so on. The first node of each pair is the server. An odd last node is left out and logged. It does not by itself fail the test.

Each GPU runs its own perftest process on its local NIC. The server starts first and each client connects after about 5 seconds. Ports start at ``port_no`` and increase by one per GPU.

Bandwidth runs are bidirectional (``-b``) and pass ``-D <duration>``, ``-q <qp>``, ``-s <bytes>``, and ``--report_gbits``. The reported Gbps is the sum of both directions, so set ``expected_results`` against bidirectional numbers. Latency runs pass ``-s <bytes>`` and ignore ``duration`` and ``qp_count_list``.

To pair across or within vPODs, set ``pairing_mode`` to ``inter_vpod`` or ``intra_vpod`` in the same config. See :ref:`the vPOD-aware pairing reference <ib-vpod-pairing>` for ``vpod_source``, ``afmctl_path``, and the sudo it needs.

.. _ib-perf-run-tests:

Run tests
=========

List the cases for the file you edited. Pass ``--config_file``; without it, ``cvs list`` shows the built-in default lists, not the shipped file's (see :doc:`/reference/configuration-files/network/ib`).

.. code:: bash

  cvs list ib_perf_bw_test \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/ibperf/ibperf_config.json

For the shipped lists, collection looks like this (``cvs list`` sorts names; the suite runs ``test_launch_containers``, the bandwidth cases, the latency cases, the two chart tests, then ``test_teardown_containers``). The two container steps skip on bare metal:

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
    • test_launch_containers
    • test_teardown_containers

  Total: 9 tests

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

Set ``install_perf_package`` to ``False`` when ``<install_dir>/perftest/bin`` already contains the ROCm build you want. The install test then logs ``Skipping perftest installation``. With ``True`` and an existing ``<install_dir>/perftest`` checkout, ``git clone`` fails without failing the test, and the existing checkout is rebuilt as is. Delete ``<install_dir>/perftest`` first to build a fresh copy.

Run one bandwidth case and its spreadsheet in the same command. The chart test reads measurements stored in that process. A chart test run in a separate ``cvs run`` has no measurements to chart:

.. code:: bash

  cvs run ib_perf_bw_test \
    'test_ib_bw_perf[ib_write_bw]' \
    test_build_ib_bw_perf_chart \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/ibperf/ibperf_config.json \
    --workspace ~/cvs_workspace \
    --capture=tee-sys -vvv

A latency case is ``'test_ib_lat_perf[ib_read_lat]'`` plus ``test_build_ib_lat_perf_chart``. An unknown name, a duplicate, or a non-list value fails collection with the ``ValueError`` from ``configured_bw_tests`` or ``configured_lat_tests``. An empty list collects no cases for that side. The chart tests are still collected. On the container backend, also select ``test_launch_containers``; it attaches to a persistent container that is already running.

Run on managed compute
----------------------

Follow :doc:`/how-to/run-on-managed-compute` to run IB Perf on managed compute. Inside a job step, ``install_system_packages`` does not run ``apt``; install the perftest packages in the node image beforehand.

Run in containers
-----------------

Follow :doc:`/how-to/run-with-containers` and set ``"orchestrator": "container"`` in the cluster file. The install and test suites start the container on every node in their first step, or attach to the one already running, and run the perftest build, perftest, and the GPU and NIC checks inside it. With ``persistent`` the container keeps running after the test suite; remove it yourself. The dmesg scan and AFM vPOD discovery stay on the host, so the dmesg scan needs passwordless sudo on the hosts, as on baremetal.

IB Perf needs these container settings:

- ``container.lifetime`` ``persistent``. The install and test suites are separate runs, and a ``per_run`` container is removed between them with anything installed in it.
- ``install_dir`` on a volume in ``container.runtime.args.volumes`` that every node shares, mounted at the same path in the container. The build runs in the head node's container and every node runs the result.
- An image with ROCm at ``rocm_dir`` and ``amd-smi``, even when ``install_system_packages`` is ``True``. The install suite never installs either, and it fails on a node without ``amd-smi``. The image also needs ``lshw``, ``rdma`` (from ``iproute2``), ``numactl``, ``libibverbs``, and ``librdmacm``, unless ``install_system_packages`` is ``True`` and the image has ``sudo``. ``install_ibperf_tools`` fails when any of them is still missing, with or without apt.

What a pass produces
====================

``cvs run`` writes the pytest HTML report, log, and zip bundle described in :ref:`pytest-html-report-and-bundle`, and includes the IB Perf Run Deck. A GPU under a configured threshold fails the pytest test.

The chart tests write two workbooks in the directory where you started ``cvs``, not in the run directory, and they are not in the zip:

- ``ib_bw_pps_perf.xlsx``: one sheet per bandwidth test and QP count, with bandwidth in Gbps, the perftest message rate, and a cluster-average chart.
- ``ib_lat_perf.xlsx``: one sheet per latency test, with min, max, average, stdev, and 99th percentile in microseconds.

Each node keeps perftest output in ``/tmp/ib_perf_<n>_logs``, one file per GPU instance numbered from 0 in card order. Every bandwidth run (message size and QP count) and every latency run (message size) deletes and rewrites these files, so after the suite they hold only the last run. They are not copied into the run directory.

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
     - ``numactl``, ``lshw``, ``rdma``, ``libibverbs``, or ``librdmacm`` is missing. Either ``install_system_packages`` is not installing them, or the apt install did not provide them.
   * - ``amd-smi not found on node``
     - ``amd-smi`` is neither on ``PATH`` nor at ``/opt/rocm/bin``. Without it GPU discovery finds no cards. Install ROCm with ``amd-smi`` on that node or in the image.
   * - ``ib_write_bw not showing expected use_rocm output``
     - The in-tree binary ``<install_dir>/perftest/ib_write_bw`` did not advertise GPUDirect RDMA. The perf tests invoke ``<install_dir>/perftest/bin/<tool>``.
   * - ``lacks ROCm DMA-BUF support``
     - ``require_dmabuf`` is on and ``--use_rocm_dmabuf`` is missing from ``--help``. Rebuild perftest or set ``require_dmabuf`` to ``False``.
   * - ``No active backend RDMA NIC in the PCI domain of``
     - That GPU has no active backend NIC in its PCI domain. The test stops before perftest.
   * - ``has different or missing GPU cards``
     - The two nodes of a pair discovered different card lists, or none. No cards usually means ``amd-smi`` is missing or failed on that node. That pair is excluded and the test fails.
   * - ``Could not probe perftest binary``
     - The binary is missing on that node, does not load with ``rocm_dir``, or was built without ``--enable-rocm``. Check that ``install_dir`` is shared by every node and that the install suite passed.
   * - ``perftest instance ... exited with code``
     - That perftest process exited non-zero, even if it printed a result. Check that instance's ``/tmp/ib_perf_<n>_logs`` on that node.
   * - ``perftest configure failed on node``
     - The head-node build failed with and without DMA-BUF. The last configure lines follow the message. Check ``rocm_dir`` and the head-node build packages.
   * - ``ibperf gid_index is invalid`` or ``Node ... NIC ...`` (GID)
     - The configured ``gid_index`` is missing, all-zero, or not ``gid_type`` on that NIC. Set ``gid_index`` to ``auto``, or set the right ``gid_type``.
   * - ``No GID index is usable on every NIC`` or ``no RoCE v2 GID with an IPv4-mapped address``
     - ``gid_index`` ``auto`` found no index shared by every NIC. Check ``gid_type`` and that each NIC has an IPv4 address.
   * - ``GPU Buffer allocation failed or Connection not setup``
     - The perftest log has no ``bytes of GPU buffer`` line. Check the GID index, port, and that the server started.
   * - ``Client did not complete for msg size``
     - No result row for that size. On bandwidth runs, ``no result within 80s`` means perftest was still running after the exit wait, and ``perftest exited without a result`` means it ended early. The waits are listed under ``duration`` in :doc:`/reference/configuration-files/network/ib`. Check that instance's ``/tmp/ib_perf_<n>_logs`` on that node.
   * - ``Actual BW ... Gbps less than the expected BW``
     - A GPU instance was under ``expected_results`` for that test, size, and QP count. Units are Gbps.
   * - ``Actual latency ... us greater than the expected latency``
     - A GPU instance's average latency (``t_avg``) was over the threshold. Units are microseconds.
   * - ``ERROR - Failure pattern *** ... *** seen in``
     - The dmesg scan matched a known error line on that node during the run.
   * - ``No latency results for``
     - ``test_build_ib_lat_perf_chart`` ran for a test that recorded no rows.

``No expected result configured ... not gated`` is informational. That size ran and was not compared to a threshold.

.. _ib-perf-dmesg-scan:

Kernel log (dmesg) scan
=======================

After each bandwidth and latency run, the suite reads ``dmesg`` on every node from the minute the run started and fails the test if it finds a known error pattern. The end time is not applied (``till_end_flag=True``), so the scan continues through the kernel log collected at the end of that message size. The log is parsed with AMD node-scraper (the ``amd-node-scraper`` package installed with CVS), extended with the CVS error patterns.

If any node lacks passwordless sudo, the scan is skipped and the log says ``SKIPPING the dmesg error check: no passwordless sudo to read dmesg on one or more nodes``. The bandwidth and latency numbers can still pass.
