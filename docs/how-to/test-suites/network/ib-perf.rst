.. meta::
  :description: Run CVS InfiniBand bandwidth and latency performance tests to validate RDMA fabric throughput and latency across AMD Instinct GPU cluster nodes.
  :keywords: CVS, InfiniBand, IB, RDMA, bandwidth, latency, AMD Instinct, ROCm, AMD, GPU, network, performance

**************************************************
Run CVS InfiniBand bandwidth and latency IB tests
**************************************************

InfiniBand (IB Perf) tests measure raw bandwidth and latency across the cluster's InfiniBand fabric using ``perftest`` tools. Run IB Perf after burn-in passes and before RCCL tests to validate that the interconnect delivers expected throughput.

.. _ib-perf-set-up-config:

Set up config
=============

Follow these steps to set up the IB performance configuration.

1. Copy the IB performance configuration file:

   .. code:: bash

     cvs config copy ibperf/ibperf_config.json --output ~/cvs_workspace/ibperf/ibperf_config.json

2. Edit the file and update ``install_dir`` to your desired location.
3. Change any other parameters relevant to your testing requirements.

For the complete field reference, see :doc:`/reference/configuration-files/network/ib`.

To test traffic between vPODs, assign each node a ``vpod_id`` in the cluster file and select explicit labels in the ibperf config:

.. code:: json

  {
    "ibperf": {
      "pairing_mode": "inter_vpod",
      "vpod_source": "cluster_file"
    }
  }

The default ``sequential`` mode retains cluster-file order. ``intra_vpod`` pairs nodes within each vPOD. AFM is the default vPOD source for one scale-up domain; use ``cluster_file`` labels when AFM IDs may repeat across domains. The suite currently assumes 8 GPUs per node.

.. _ib-perf-run-tests:

Run tests
=========

You can list all available IB Perf test cases using the CLI:

.. code:: bash

  cvs list ib_perf_bw_test

.. code:: text

  Available tests in ib_perf_bw_test:
    - test_ib_bw_perf
    - test_ib_bw_perf
    - test_ib_bw_perf
    - test_ib_lat_perf
    - test_ib_lat_perf
    - test_build_ib_bw_perf_chart
    - test_build_ib_lat_perf_chart

.. note::

   At least two nodes are required to run IB Perf installation and tests.

Run the IB Perf suite:

1. Run the installation:

   .. code:: bash

     cvs run install_ibperf_tools \
       --cluster_file ~/cvs_workspace/cluster.json \
       --config_file ~/cvs_workspace/ibperf/ibperf_config.json \
       --capture=tee-sys -vvv -s

2. Start the IB Perf test:

   .. code:: bash

     cvs run ib_perf_bw_test \
       --cluster_file ~/cvs_workspace/cluster.json \
       --config_file ~/cvs_workspace/ibperf/ibperf_config.json \
       --capture=tee-sys -vvv -s
