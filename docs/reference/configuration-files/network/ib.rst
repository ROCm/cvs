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
          "install_system_packages": "False",
          "install_dir": "/home/{user-id}/",
          "rocm_dir": "<changeme>",
          "require_dmabuf": "True",
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
     - Build and install perftest from source with ROCm GPU support on the head node. ``install_dir`` must be on storage shared by every node. CVS configures perftest with ``--enable-rocm-dmabuf`` so that ``--use_rocm_dmabuf`` is available, and falls back to a ROCm-only build if that configure fails.
   * - ``install_system_packages``
     - False
     - Run ``sudo apt install`` for the perftest build and runtime packages on every node before the build. Skipped inside Spur/Slurm job steps. When False, no packages are installed and the install test fails if ``numactl``, ``libibverbs``, or ``librdmacm`` is missing on any node. The head node also needs the build tools and the ``libibverbs``/``librdmacm`` development packages.
   * - ``install_dir``
     - ``/home/{user-id}/``
     - Installation directory for performance testing tools
   * - ``rocm_dir``
     - ``<changeme>``
     - 	Set the path of rocm
   * - ``require_dmabuf``
     - True
     - Fail before launching if the perftest binary on any test node was built without ROCm DMA-BUF support (``--use_rocm_dmabuf``). Set to False to run without DMA-BUF; the run records ``dmabuf=off``.
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

