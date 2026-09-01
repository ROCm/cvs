.. meta::
  :description: Run IB bandwidth and latency performance tests
  :keywords: CVS, ib

**************************
InfiniBand (IB Perf) tests
**************************

.. _ib-perf-set-up-config:

Set up config
=============

1. Copy the IB performance configuration file:

   .. code:: bash

     cvs config copy ibperf/ibperf_config.json --output ~/cvs_workspace/ibperf/ibperf_config.json

2. Edit the file and update ``install_dir`` to your desired location.
3. Change any other parameters relevant to your testing requirements.

Full parameter list: :doc:`/reference/configuration-files/network/ib`.

.. _ib-perf-run-tests:

Run tests
=========

InfiniBand (IB Perf) test script
--------------------------------

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

Use these scripts to start the test:
Note: At least two nodes are required to run IB Perf installation and tests.

1. Run the installation: 

   .. code:: bash

     cvs run install_ibperf_tools --cluster_file input/cluster_file/cluster.json --config_file input/config_file/ibperf/ibperf_config.json --html=/var/www/html/cvs/ib.html --capture=tee-sys --self-contained-html --log-file=/tmp/test.log -vvv -s

2. Start the IB Perf test:

   .. code:: bash

     cvs run ib_perf_bw_test --cluster_file input/cluster_file/cluster.json --config_file input/config_file/ibperf/ibperf_config.json --html=/var/www/html/cvs/ib.html --capture=tee-sys --self-contained-html --log-file=/tmp/test.log -vvv -s
