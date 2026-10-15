.. meta::
  :description: Reference for the CVS InfiniBand IB Perf test configuration file, covering bandwidth and latency benchmarks, queue pairs, message sizes, and expected results.
  :keywords: CVS, InfiniBand, IB, RDMA, ROCm, bandwidth, latency, benchmark, GPU, AMD, JSON, configuration, network

*******************************************************************************
InfiniBand (IB Perf) test configuration file for Cluster Validation Suite (CVS)
*******************************************************************************

IB Perf and latency tests measure network performance. Perf tests measure throughput (bandwidth), and latency tests measure delay. 

The following sample shows the ``ibperf_config.json`` structure:

.. note::

  In this configuration file, ``{user-id}`` resolves to the current username at runtime. You can also manually change this value to your username. 

.. dropdown:: ``ibperf_config.json``

  .. code:: json
      
    {
        "ibperf":
        {
          "install_perf_package": "True",
          "install_dir": "/home/{user-id}/",
          "rocm_dir": "<changeme>",
          "require_dmabuf": "True",
          "pairing_mode": "sequential",
          "vpod_source": "afm",
          "afmctl_path": "afmctl",
          "qp_count_list": [ "8", "16" ],
          "ib_bw_test_list": [ "ib_write_bw", "ib_read_bw", "ib_send_bw" ],
          "ib_lat_test_list": [ "ib_write_lat", "ib_send_lat" ],
          "msg_size_list": [ 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536 ],
          "gid_index": "3",
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

Parameters
==========

Here's an exhaustive list of the available parameters in the IB Perf configuration file.

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameters
     - Default values
     - Description
   * - ``install_perf_package``
     - True
     - Build and install perftest from source with ROCm GPU support. CVS configures perftest with ``--enable-rocm-dmabuf`` so that ``--use_rocm_dmabuf`` is available, and falls back to a ROCm-only build if that configure fails.
   * - ``install_dir``
     - ``/home/{user-id}/``
     - Installation directory for performance testing tools
   * - ``rocm_dir``
     - ``<changeme>``
     - 	Set the path of rocm
   * - ``require_dmabuf``
     - True
     - Fail before launching if the perftest binary on any test node was built without ROCm DMA-BUF support (``--use_rocm_dmabuf``). Set to False to run without DMA-BUF; the run records ``dmabuf=off``.
   * - ``pairing_mode``
     - ``sequential``
     - Pair consecutive cluster-file nodes, pair across vPODs with ``inter_vpod``, or pair within a vPOD with ``intra_vpod``.
   * - ``vpod_source``
     - ``afm``
     - For vPOD modes, discover the scale-up fabric ``(physical_pod.id, virtual_pod.id)`` key using AFM, or read ``node_dict[<node>].vpod_id`` with ``cluster_file``.
   * - ``afmctl_path``
     - ``afmctl``
     - Path to ``afmctl`` for AFM discovery. The SSH user needs passwordless sudo for ``sudo -n afmctl show device --json``; MI4XX installations may use ``/usr/sbin/afmctl``.
   * - ``qp_count_list``
     - Values:
        - 8 
        - 16
     - Queue Pair counts to test
   * - ``ib_bw_test_list``
     - Values:
        - ``ib_write_bw``
        - ``ib_read_bw``
        - ``ib_send_bw``
     - IB bandwidth tests to run, one test case each. Supported values: ``ib_write_bw``, ``ib_read_bw``, ``ib_send_bw``. If omitted, all three run. An empty list skips bandwidth tests. Unsupported or duplicate names fail test collection.
   * - ``ib_lat_test_list``
     - Values:
        - ``ib_write_lat`` 
        - ``ib_send_lat`` 
        - ``ib_read_lat``
     - IB latency tests to run, one test case each. Supported values: ``ib_write_lat``, ``ib_read_lat``, ``ib_send_lat``. If omitted, ``ib_write_lat`` and ``ib_send_lat`` run. An empty list skips latency tests. Unsupported or duplicate names fail test collection.
   * - ``msg_size_list``
     - Values:
        - 2 
        - 4 
        - 8 
        - 16 
        - 32
        - 64 
        - 128 
        - 256 
        - 512
        - 1024 
        - 2048 
        - 4096 
        - 8192 
        - 16384
        - 32768
        - 65536 
     - Test message sizes in bytes
   * - ``gid_index``
     - 3
     - GID index validated on every NIC before perftest; ``auto`` selects the lowest common RoCE v2 IPv4-mapped index on RoCE, or the lowest populated index (normally 0) on InfiniBand
   * - ``gid_type``
     - RoCE v2
     - Expected sysfs GID type; ``any`` disables type checking, and InfiniBand link layers are exempt
   * - ``port_no``
     - 1516
     - Port number for test communication
   * - ``duration``
     - 10
     - Run time in seconds of each bandwidth test (perftest ``-D``). Latency tests ignore it. Keep it at 10 or less: the suite reads bandwidth results about 20 seconds after starting perftest, so longer runs are reported as failures.
   * - ``verify_bw``
     - True
     - Enforce ``expected_results`` thresholds (bandwidth and latency). Every GPU instance
       on every node must meet the threshold.

``expected_results.<bw_test>.<msg_size>.<qp_count>`` sets the minimum bandwidth in Gbps
for each GPU instance. The test fails if any instance falls below it.
``expected_results.<lat_test>.<msg_size>`` sets the maximum average latency in microseconds
for each GPU instance.

Message sizes and QP counts must appear in ``msg_size_list`` and ``qp_count_list``.
Entries outside the configured sweep produce a warning and are not checked.
The sample enables ``verify_bw``. Re-baseline its thresholds for your cluster before running it.

The sample ``ib_write_bw`` thresholds are:

.. dropdown:: ib_write_bw

  .. code:: json

    "8192": {
      "8": "180.0",
      "16": "200.0"
    }

vPOD-aware pairing
==================

``sequential`` pairs consecutive nodes in cluster-file order. ``inter_vpod`` pairs nodes from different vPODs for scale-out tests; ``intra_vpod`` pairs nodes within each vPOD. Nodes left without a partner are excluded and logged. Discovery errors and pairs excluded during the test cause a test failure.

With ``vpod_source=afm``, the suite runs ``sudo -n afmctl show device --json`` on each node. AFM vPOD identity is ``(scaleup_fabric.physical_pod.id, scaleup_fabric.virtual_pod.id)``. Older AFM builds without these IDs fall back to ``vpod_accelerators``, which is unique only within one physical pod. Overlapping accelerator IDs stop discovery and recommend ``vpod_source=cluster_file``. The SSH user must have passwordless sudo for ``afmctl``.

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

Known limitation: GPU-to-NIC mapping compares PCI bus numbers only and ignores the PCI domain. On MI450 this can map every GPU to the same NIC, so 4-GPU traffic measurements are not meaningful until the shared mapping code is corrected.
