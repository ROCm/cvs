.. meta::
  :description: Run CVS MORI RDMA benchmark tests to measure put, write, and I/O throughput over InfiniBand fabric on AMD Instinct GPU cluster nodes.
  :keywords: CVS, MORI, RDMA, InfiniBand, AMD Instinct, ROCm, AMD, GPU, benchmark, network, IB, throughput

***********************************************
Run CVS MORI RDMA benchmark tests
***********************************************

MORI tests measure RDMA put, write, and I/O throughput over the InfiniBand fabric using AMD Pensando AINIC and other RDMA-capable devices. Run MORI to validate raw RDMA performance before training or inference workloads.

.. _mori-set-up-config:

Set up config
=============

1. Copy the MORI RDMA configuration file:

   .. code:: bash

     cvs config copy mori/mi35x_mori_config.json --output ~/cvs_workspace/mori/mi35x_mori_config.json

2. Edit the file and configure:

   - ``no_of_nodes`` — number of nodes in the cluster
   - Every field still set to ``<changeme>`` — replace with cluster-specific values before running

For the complete field reference, see :doc:`/reference/configuration-files/network/mori`.

.. _mori-run-tests:

Run tests
=========

MORI test scripts
-----------------

You can list all available MORI test cases using the CLI:

.. code:: bash

  cvs list mori_benchmark_test

.. code:: text

  Available tests in mori_benchmark_test:
    - test_cleanup_stale_containers
    - test_concurrent_put_imm_threads
    - test_concurrent_put_signal_thread
    - test_concurrent_put_threads
    - test_ibgda_write_test
    - test_install_container_packages
    - test_io_read[16384-128-1]
    - test_io_read[16384-128-8]
    - test_io_read[32768-128-1]
    - test_io_read[32768-128-8]
    - test_io_read[32768-256-1]
    - test_io_read[32768-256-8]
    - test_io_write[16384-128-1]
    - test_io_write[16384-128-8]
    - test_io_write[32768-128-1]
    - test_io_write[32768-128-8]
    - test_io_write[32768-256-1]
    - test_io_write[32768-256-8]
    - test_launch_mori_container
    - test_setup_env
    - test_setup_ibv_devices
    - test_shmem_api

Run the MORI benchmark suite:

.. code:: bash

  cvs run mori_benchmark_test \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/mori/mi35x_mori_config.json \
    --html=/var/www/html/cvs/mori.html --capture=tee-sys --self-contained-html \
    --log-file=/tmp/mori.log -vvv -s
