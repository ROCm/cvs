.. meta::
  :description: Run MORI RDMA benchmark tests
  :keywords: CVS, mori

**********
MORI tests
**********

.. _mori-set-up-config:

Set up config
=============

1. Copy the MORI RDMA configuration file:

   .. code:: bash

     cvs config copy mori/mi35x_mori_config.json --output ~/cvs_workspace/mori/mi35x_mori_config.json

2. Edit the file and configure:

   - ``no_of_nodes`` — number of nodes in the cluster
   - Every field still set to ``<changeme>`` — replace with cluster-specific values before running

Full parameter list: :doc:`/reference/configuration-files/network/mori`.

.. _mori-run-tests:

Run tests
=========

Mori test scripts
------------------------------

You can list all available Mori test cases using the CLI:

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

Use these scripts to run the Mori tests.

.. code:: bash

  cvs run mori_benchmark_test --cluster_file input/cluster_file/cluster.json --config_file input/config_file/mori/mi35x_mori_config.json --html=/var/www/html/cvs/mori.html --capture=tee-sys --self-contained-html --log-file=/tmp/mori.log -vvv -s
