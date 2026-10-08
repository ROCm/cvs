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
3. Change any other parameters relevant to your testing requirements. Keep ``duration`` at 10 seconds or less; see the ``duration`` entry in :doc:`/reference/configuration-files/network/ib`.

For the complete field reference, see :doc:`/reference/configuration-files/network/ib`.

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

``test_ib_bw_perf`` runs once each for ``ib_write_bw``, ``ib_read_bw``, and ``ib_send_bw``, and ``test_ib_lat_perf`` runs once each for ``ib_write_lat`` and ``ib_send_lat``. This set is fixed in the suite; the ``ib_bw_test_list`` and ``ib_lat_test_list`` config fields don't change it.

.. note::

   At least two nodes are required to run IB Perf installation and tests.

Run the IB Perf suite:

1. Run the installation:

   .. code:: bash

     cvs run install_ibperf_tools \
       --cluster_file ~/cvs_workspace/cluster.json \
       --config_file ~/cvs_workspace/ibperf/ibperf_config.json \
       --capture=tee-sys -vvv

2. Start the IB Perf test:

   .. code:: bash

     cvs run ib_perf_bw_test \
       --cluster_file ~/cvs_workspace/cluster.json \
       --config_file ~/cvs_workspace/ibperf/ibperf_config.json \
       --capture=tee-sys -vvv

.. _ib-perf-dmesg-scan:

Kernel log (dmesg) scan
=======================

After each bandwidth and latency run, the suite reads ``dmesg`` on every node from the minute the run started and fails the test if it finds a known error pattern. The ``CVS_DMESG_PARSER`` environment variable selects the parser:

- Unset or ``node-scraper`` (default): AMD node-scraper (the ``amd-node-scraper`` package installed with CVS), extended with the CVS error patterns.
- ``legacy``: the CVS regex patterns only. ``cvs``, ``0``, ``false``, ``off``, and ``no`` also select this parser.

Any other value selects node-scraper. CVS reads the variable on the host that runs ``cvs run``, so set it in that shell:

.. code:: bash

  CVS_DMESG_PARSER=legacy cvs run ib_perf_bw_test \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/ibperf/ibperf_config.json \
    --capture=tee-sys -vvv
