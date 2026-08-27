.. meta::
  :description: Run RCCL performance and regression suites
  :keywords: CVS, rccl

**********
RCCL tests
**********

.. _rccl-set-up-config:

Set up config
=============

1. Copy the RCCL configuration file:

   .. code:: bash

     cvs config copy rccl/rccl_config.json --output ~/cvs_workspace/rccl/rccl_config.json

2. Edit directory paths and any ``<changeme>`` placeholders:

   - ``rccl_dir``, ``rccl_tests_dir``, ``mpi_dir``
   - ``mpi_path_var``, ``rccl_path_var``, ``rocm_path_var``

Full parameter list: :doc:`/reference/configuration-files/network/rccl`.

.. _rccl-run-tests:

Run tests
=========

ROCm Communication Collectives Library (RCCL) tests
---------------------------------------------------

You can list all available RCCL test cases using the CLI:

.. code:: bash

  cvs list rccl_perf

.. code:: text

  Available tests in rccl_perf:
    - test_collect_hostinfo
    - test_collect_networkinfo
    - test_disable_firewall
    - test_gen_graph
    - test_print_env_once
    - test_rccl_perf[all_gather_perf]
    - test_rccl_perf[all_reduce_perf]
    - test_rccl_perf[alltoall_perf]
    - test_rccl_perf[alltoallv_perf]
    - test_rccl_perf[broadcast_perf]
    - test_rccl_perf[gather_perf]
    - test_rccl_perf[reduce_scatter_perf]
    - test_rccl_perf[scatter_perf]
    - test_rccl_perf[sendrecv_perf]

.. code:: bash

  cvs list rccl_regression

.. code:: text

  Available tests in rccl_regression:
    - test_collect_hostinfo
    - test_collect_networkinfo
    - test_disable_firewall
    - test_gen_graph
    - test_print_env_once
    - test_rccl_perf

Use these scripts to start RCCL tests with CVS:

**Prerequisites: Environment Script Staging**

Before running RCCL tests, ensure the environment script specified in ``env_source_script`` is available on all cluster nodes:

- **With shared storage**: Place the environment script in a shared directory accessible from all nodes
- **Without shared storage**: Use ``cvs scp`` to copy the environment script to all nodes. See :doc:`/how-to/copy-to-cluster` for detailed instructions and examples.

1. Run RCCL performance suite:

.. code:: bash

  cvs run rccl_perf --cluster_file input/cluster_file/cluster.json --config_file input/config_file/rccl/rccl_config.json --html=/var/www/html/cvs/rccl_perf.html --capture=tee-sys --self-contained-html --log-file=/tmp/rccl_perf.log -vvv -s

2. Run RCCL regression suite:

.. code:: bash

  cvs run rccl_regression --cluster_file input/cluster_file/cluster.json --config_file input/config_file/rccl/rccl_regression.json --html=/var/www/html/cvs/rccl_regression.html --capture=tee-sys --self-contained-html --log-file=/tmp/rccl_regression.log -vvv -s

3. Generate RCCL performance heatmap:

.. code:: bash

  cvs generate heatmap --actual /tmp/rccl_perf_results.json --reference /path/to/golden_reference.json --output /var/www/html/cvs/rccl_heatmap.html --title "RCCL Performance Comparison"
