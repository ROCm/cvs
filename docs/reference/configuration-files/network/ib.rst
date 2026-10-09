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
          "pairing_mode": "sequential",
          "vpod_source": "afm",
          "afmctl_path": "afmctl",
          "qp_count_list": [ "8", "16" ],
          "ib_bw_test_list": [ "ib_write_bw", "ib_send_bw"],
          "ib_lat_test_list": [ "ib_write_lat", "ib_send_lat", "ib_read_lat" ],
          "msg_size_list": [ 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536 ],
          "gid_index": "3",
          "port_no": "1516",
          "duration": "30",
          "verify_bw": "True",
          "expected_results":
          {
          "ib_write_bw":
          {
              "8192":
                    {
            "8": "180.0",
            "16": "200.0"
                    },
              "8388608":
                    {
            "8": "280.0",
            "16": "300.0"
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
     - Enable automatic installation of InfiniBand performance tools
   * - ``install_dir``
     - ``/home/{user-id}/``
     - Installation directory for performance testing tools
   * - ``rocm_dir``
     - ``<changeme>``
     - 	Set the path of rocm
   * - ``pairing_mode``
     - ``sequential``
     - Pair consecutive nodes in cluster-file order, across vPODs with ``inter_vpod``, or within vPODs with ``intra_vpod``.
   * - ``vpod_source``
     - ``afm``
     - For vPOD modes, read membership from AFM or use ``cluster_file`` to read each node's ``vpod_id``.
   * - ``afmctl_path``
     - ``afmctl``
     - Path to the AFM command when ``vpod_source`` is ``afm``.
   * - ``qp_count_list``
     - Values:
        - 8 
        - 16
     - Queue Pair counts to test
   * - ``ib_bw_test_list``
     - Values:
        - ``ib_write_bw`` 
        - ``ib_send_bw``
     - IB bandwidth tests
   * - ``ib_lat_test_list``
     - Values:
        - ``ib_write_lat`` 
        - ``ib_send_lat`` 
        - ``ib_read_lat``
     - IB latency tests
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
     - Global Identifier index for InfiniBand
   * - ``port_no``
     - 1516
     - Port number for test communication
   * - ``duration``
     - 30
     - Test duration in seconds
   * - ``verify_bw``
     - True
     - Bandwidth verification 

The ``expected_results`` section also contains the ``ib_write_bw`` parameter. It describes the bandwith expectation, and it has these default values in the JSON file:

.. dropdown:: ib_write_bw

  .. code:: json

    "8192":
                    {
            "8": "180.0",
            "16": "200.0"
                    },
              "8388608":
                    {
            "8": "280.0",
            "16": "300.0"
                    }


vPOD-aware pairing
==================

``sequential`` keeps the existing consecutive server/client pairs. ``inter_vpod`` pairs nodes in different vPODs to test the scale-out path; ``intra_vpod`` pairs nodes within the same vPOD. Nodes without a partner are excluded.

For explicit membership, set ``vpod_source`` to ``cluster_file`` and add ``vpod_id`` to each ``node_dict`` entry in the cluster file. For example, label ``n1`` and ``n2`` with ``"A"``, and ``n3`` and ``n4`` with ``"B"``. With ``inter_vpod``, the pairs are ``(n1, n3)`` and ``(n2, n4)``.

The default AFM source uses ``afmctl show device --json``. AFM accelerator IDs are only unique within one scale-up domain; on multi-rack clusters with ambiguous AFM membership, use ``cluster_file`` labels. The ibperf suite still assumes 8 GPUs per node.
