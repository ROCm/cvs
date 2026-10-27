.. meta::
  :description: Reference for the CVS InfiniBand IB Perf test configuration file, covering bandwidth and latency benchmarks, queue pairs, message sizes, and expected results.
  :keywords: CVS, InfiniBand, IB, RDMA, ROCm, bandwidth, latency, benchmark, GPU, AMD, JSON, configuration, network

*******************************************************************************
InfiniBand (IB Perf) test configuration file for Cluster Validation Suite (CVS)
*******************************************************************************

IB Perf measures throughput and latency with perftest. Bandwidth tests report Gbps. Latency tests report microseconds.

The suite reads the ``ibperf`` object. The shipped file is ``input/config_file/ibperf/ibperf_config.json``. Keys whose names start with ``_`` (the ``_comment_*`` entries in that file) are not settings.

.. note::

  ``{user-id}`` is replaced with the cluster file ``username``, then ``$USER``, then ``root``. ``{user}``, ``{home}``, ``{home-mount-dir}``, ``{node-dir-name}``, and ``{run_dir}`` are replaced in the same pass.
  Any string that still contains ``<changeme>`` stops the run with ``Unresolved placeholder found in test config``. That check runs before ROCm auto-detect. The shipped ``rocm_dir`` value ``<changeme>`` does not auto-detect. Set a path, or ``""``, or omit ``rocm_dir``.

.. dropdown:: ``ibperf_config.json``

  The block below matches the shipped values the suite reads. The file on disk also has ``_comment_*`` keys.

  .. code:: json

    {
        "ibperf":
        {
          "install_perf_package": "True",
          "install_system_packages": "False",
          "install_dir": "/home/{user-id}/",
          "rocm_dir": "<changeme>",
          "require_dmabuf": "True",
          "pairing_mode": "sequential",
          "vpod_source": "afm",
          "afmctl_path": "afmctl",
          "qp_count_list": [ "8", "16" ],
          "ib_bw_test_list": [ "ib_write_bw", "ib_send_bw" ],
          "ib_lat_test_list": [ "ib_write_lat", "ib_send_lat", "ib_read_lat" ],
          "msg_size_list": [ 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536 ],
          "gid_index": "3",
          "gid_type": "RoCE v2",
          "port_no": "1516",
          "duration": "10",
          "verify_bw": "True",
          "expected_results": {
            "ib_write_bw": {
              "8192": {
                "8": "180.0",
                "16": "200.0"
              }
            }
          }
        }
    }

Which tests run
===============

``configured_bw_tests`` reads ``ib_bw_test_list``. ``configured_lat_tests`` reads ``ib_lat_test_list``. Each name becomes one pytest case, ``test_ib_bw_perf[<name>]`` or ``test_ib_lat_perf[<name>]``.

.. list-table::
   :widths: 3 7
   :header-rows: 1

   * - List value
     - What collection does
   * - Key omitted
     - Bandwidth defaults to ``ib_write_bw``, ``ib_read_bw``, ``ib_send_bw``. Latency defaults to ``ib_write_lat``, ``ib_send_lat``.
   * - A list of supported names
     - Those binaries run, in that order. The shipped file omits ``ib_read_bw`` and includes ``ib_read_lat``.
   * - ``[]``
     - No cases on that side. The chart tests are still collected.
   * - Unknown name, duplicate, or a non-list
     - Collection fails with ``ValueError`` from ``configured_bw_tests`` or ``configured_lat_tests``.

Supported bandwidth names are ``ib_write_bw``, ``ib_read_bw``, and ``ib_send_bw``. Supported latency names are ``ib_write_lat``, ``ib_read_lat``, and ``ib_send_lat``.

Parameters
==========

``verify_bw`` is on when the value contains ``true`` in any case. ``require_dmabuf`` accepts ``true`` / ``1`` / ``yes`` / ``on`` and ``false`` / ``0`` / ``no`` / ``off``. Any other ``require_dmabuf`` value is treated as true.

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Parameter
     - If you omit the key
     - What it changes
   * - ``install_perf_package``
     - The install test errors (the key is required there)
     - ``True`` clones, configures, and installs perftest on the head node. ``False`` skips that work.
   * - ``install_system_packages``
     - ``False``
     - ``True`` runs ``sudo apt install`` for the perftest build and runtime packages on every node before the build. Skipped inside Spur/Slurm job steps. The install test fails if ``numactl``, ``lshw``, ``rdma``, ``libibverbs``, or ``librdmacm`` is missing, after the apt install when true. It always fails on a node without ``amd-smi`` on ``PATH`` or at ``/opt/rocm/bin``, which apt never installs. The head node also needs the build tools and the ``libibverbs`` / ``librdmacm`` development packages when you install from source.
   * - ``install_dir``
     - The perf tests error (the key is required there)
     - Directory for the perftest checkout. Tests run ``<install_dir>/perftest/bin/<tool>``. The install check looks at ``<install_dir>/perftest/ib_write_bw``.
   * - ``rocm_dir``
     - Auto-detect
     - ROCm prefix passed to perftest ``--with-rocm`` and prepended to ``LD_LIBRARY_PATH``. ``""`` auto-detects ``/opt/rocm/core-*``, then ``/opt/rocm``. The shipped ``<changeme>`` never reaches auto-detect.
   * - ``require_dmabuf``
     - True
     - Fail before launch if any node's binary was built without ``--use_rocm_dmabuf``. ``False`` runs without DMA-BUF and records ``dmabuf=off``. Configure tries ``--enable-rocm --enable-rocm-dmabuf``, then retries without DMA-BUF.
   * - ``pairing_mode``
     - ``sequential``
     - ``sequential`` pairs cluster-file order. ``inter_vpod`` pairs across vPODs. ``intra_vpod`` pairs inside one vPOD.
   * - ``vpod_source``
     - ``afm``
     - Used only for vPOD modes. ``afm`` runs ``sudo -n afmctl show device --json``. ``cluster_file`` reads ``node_dict[<node>].vpod_id``.
   * - ``afmctl_path``
     - ``afmctl``
     - ``afmctl`` binary for AFM discovery. An absolute path is used as given. A bare name is resolved with ``command -v`` and then run under ``sudo -n``.
   * - ``qp_count_list``
     - The bandwidth test errors
     - Queue-pair counts for bandwidth only. Latency does not pass ``-q``. Strings or integers both work.
   * - ``ib_bw_test_list``
     - The three write/read/send bandwidth tools
     - Bandwidth cases. See the table above.
   * - ``ib_lat_test_list``
     - ``ib_write_lat`` and ``ib_send_lat``
     - Latency cases. See the table above.
   * - ``msg_size_list``
     - The perf tests error
     - Message sizes in bytes, passed as perftest ``-s``, for both bandwidth and latency.
   * - ``gid_index``
     - ``auto``
     - GID index passed as perftest ``-x``, checked on every selected NIC before launch. ``auto``, ``""``, or omitting the key selects the lowest index common to every NIC: ``gid_type`` with an IPv4-mapped address on RoCE, or the lowest populated index (normally 0) on InfiniBand.
   * - ``gid_type``
     - ``RoCE v2``
     - Expected sysfs GID type. ``any`` skips the type check. InfiniBand link layers are not type-checked. An all-zero GID still fails on every link layer.
   * - ``port_no``
     - The perf tests error
     - First TCP port. Each GPU on a node uses the next port.
   * - ``duration``
     - The bandwidth test errors
     - Seconds for perftest ``-D`` on bandwidth only. Latency ignores it. After launch, bandwidth waits this value plus 60 seconds for the processes to exit, then reads the logs. If exit is not confirmed, it polls up to 80 seconds for a result line. Latency waits up to 60 seconds for exit and up to 20 seconds for a result line. The shipped file uses ``10``. A larger value runs each bandwidth case for that many seconds, plus the 60 second exit wait.
   * - ``verify_bw``
     - Off (only a value containing ``true`` turns it on)
     - Apply ``expected_results`` to bandwidth and to latency.
   * - ``expected_results``
     - Nothing is gated
     - Thresholds. Shape and direction are below.

``expected_results.<bw_test>.<msg_size>.<qp_count>`` is the minimum bandwidth in Gbps for each GPU instance. The test fails if any instance is below it. An equal value passes.

``expected_results.<lat_test>.<msg_size>`` is the maximum average latency in microseconds (perftest ``t_avg``) for each GPU instance. The test fails if any instance is above it. A latency entry must be a number. A QP object under a latency test fails with ``must be a latency in usec``.

JSON keys are strings. The checker stringifies the message size and the QP count from the sweep, so ``8192`` and ``"8192"`` in the sweep match the key ``"8192"``.

Message sizes and QP counts in ``expected_results`` that are outside ``msg_size_list`` and ``qp_count_list`` log ``expected_results.<path> is not covered by msg_size_list/qp_count_list and will not be checked`` when ``verify_bw`` is on. A sweep point with no threshold logs ``No expected result configured for <test> msg_size=<size>; not gated`` and does not fail.

The shipped sample gates only ``ib_write_bw`` at 8192 bytes for QP counts 8 and 16, at 180.0 Gbps and 200.0 Gbps. ``configured_bw_tests`` still runs ``ib_send_bw``, and ``configured_lat_tests`` still runs the three latency tests. Those cases are not gated by this sample. Replace the two numbers for your fabric. The checker uses the numbers in the file you pass. It does not keep a second threshold table.

For ``ib_read_lat``, only the client side of each pair has results. An empty server result is skipped. Other latency tests fail a node that returns no rows.

The failure lines are ``Actual BW <bw> Gbps less than the expected BW <threshold> Gbps`` and ``Actual latency <t_avg> us greater than the expected latency <threshold> us``, including the node and GPU index.

vPOD-aware pairing
==================

``sequential`` pairs consecutive nodes in cluster-file order. ``inter_vpod`` pairs nodes from different vPODs for scale-out tests. ``intra_vpod`` pairs nodes within each vPOD. Nodes left without a partner are excluded and logged. Discovery errors and pairs excluded during the test cause a test failure.

With ``vpod_source`` ``afm``, the suite runs ``sudo -n afmctl show device --json`` on each node. AFM vPOD identity is ``(scaleup_fabric.physical_pod.id, scaleup_fabric.virtual_pod.id)``. Older AFM builds without these IDs fall back to ``vpod_accelerators``, which is unique only within one physical pod. Overlapping accelerator IDs stop discovery and recommend ``vpod_source`` ``cluster_file``. The SSH user must have passwordless sudo for ``afmctl``.

For an explicit cluster-file mapping, set ``vpod_source`` to ``cluster_file`` and label each node:

.. code:: json

   "node_dict": {
     "n1": {"vpod_id": "A"},
     "n2": {"vpod_id": "A"},
     "n3": {"vpod_id": "B"},
     "n4": {"vpod_id": "B"}
   }

``inter_vpod`` then forms ``(n1, n3)`` and ``(n2, n4)``.

GPUs per node
=============

The number of GPU instances follows GPU discovery. Both nodes of a pair must report the same GPU cards, and all active pairs must use the same card list. A mismatched pair fails and is excluded while eligible pairs continue.

When GPU and backend-NIC PCI domains overlap, each GPU is paired only with an active NIC in its own domain (nearest bus, then nearest device). A GPU with no eligible NIC is left unmapped. The bandwidth or latency test then fails before perftest with ``No active backend RDMA NIC in the PCI domain of <card>``. If a domain has more GPUs than NICs, a NIC can be shared. That is logged (``shares NIC``) and the test still launches.

``HIP_VISIBLE_DEVICES``, ``ROCR_VISIBLE_DEVICES``, ``CUDA_VISIBLE_DEVICES``, and ``GPU_DEVICE_ORDINAL`` are unset for the perftest launch. If one of them is set in the probe, the test fails via ``fail_test`` with ``GPU visibility env set on node`` and falls back to the card index for ``--use_rocm`` so the run can continue.

Current behavior
================

- Perftest command lines and the in-memory results live for one ``cvs run`` process. ``test_build_ib_bw_perf_chart`` and ``test_build_ib_lat_perf_chart`` write ``ib_bw_pps_perf.xlsx`` and ``ib_lat_perf.xlsx`` in the current directory. A later process does not see the measurements.
- The latency chart fails with ``No latency results for <test>`` when that test is present and has no rows.
- Node-local logs ``/tmp/ib_perf_<gpu-index>_logs`` are removed at the start of every message size.
- Outlier marks in the HTML run deck do not fail the pytest test. Threshold misses do, when ``verify_bw`` is on and a threshold exists for that point.
