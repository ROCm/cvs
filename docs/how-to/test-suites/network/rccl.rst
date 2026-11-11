.. meta::
  :description: Run CVS RCCL performance and regression test suites to validate collective communication bandwidth across AMD Instinct GPU cluster nodes with ROCm.
  :keywords: CVS, RCCL, collective communication, AMD Instinct, ROCm, AMD, GPU, InfiniBand, all-reduce, all-gather, performance

***********************************************
Run CVS RCCL performance and regression tests
***********************************************

RCCL tests validate collective communication bandwidth and correctness across AMD Instinct GPU nodes using all-reduce, all-gather, broadcast, and other collectives. Run RCCL after IB Perf passes to confirm that GPU-to-GPU communication meets performance thresholds.

- ``rccl_perf`` runs configured collectives across ``rccl_test_params.data_types``.
- ``rccl_regression`` runs the Cartesian product of the ``regression`` environment axes and collectives.
- ``rccl_pairwise`` runs Phase 0 reference sanity, Phase 1 reference/candidate pairs, and a Phase 2 incremental group.

.. _rccl-set-up-config:

Set up config
=============

1. Copy the RCCL configuration file:

   .. code:: bash

     cvs config copy rccl/rccl_config.json --output ~/cvs_workspace/rccl/rccl_config.json

2. Edit directory paths and any ``<changeme>`` placeholders:

   - ``rccl_dir``, ``rccl_tests_dir``, ``mpi_dir``
   - ``mpi_path_var``, ``rccl_path_var``, ``rocm_path_var``

3. All three suites read the same file. ``rccl_regression`` additionally requires a non-empty ``regression`` object; ``rccl_pairwise`` uses ``cvs_params.pairwise_min_bw`` and ``cvs_params.pairwise_results_file``.

For the complete field reference, see :doc:`/reference/configuration-files/network/rccl`.

.. _rccl-run-tests:

Run tests
=========

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

.. code:: bash

  cvs list rccl_pairwise

.. code:: text

  Available tests in rccl_pairwise:
  --------------------------------------------------------------------------------
    • test_collect_hostinfo
    • test_collect_networkinfo
    • test_gen_graph
    • test_rccl_incremental
    • test_rccl_pairwise

  Total: 5 tests

Prerequisites for environment script staging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before running RCCL tests, ensure the environment script specified in ``env_source_script`` is available on all cluster nodes:

With shared storage
"""""""""""""""""""

Place the environment script in a shared directory accessible from all nodes.

Without shared storage
""""""""""""""""""""""

Use ``cvs scp`` to copy the environment script to all nodes. See :doc:`/how-to/copy-to-cluster` for detailed instructions and examples.

.. _rccl-run-baremetal:

Run on bare metal or with the container backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Run RCCL performance suite:

   .. code:: bash

     cvs run rccl_perf \
       --cluster_file ~/cvs_workspace/cluster.json \
       --config_file ~/cvs_workspace/rccl/rccl_config.json \
       --capture=tee-sys -vvv -s

2. Run RCCL regression suite:

   .. code:: bash

     cvs run rccl_regression \
       --cluster_file ~/cvs_workspace/cluster.json \
       --config_file ~/cvs_workspace/rccl/rccl_config.json \
       --capture=tee-sys -vvv -s

3. Run RCCL pairwise suite:

   .. code:: bash

     cvs run rccl_pairwise \
       --cluster_file ~/cvs_workspace/cluster.json \
       --config_file ~/cvs_workspace/rccl/rccl_config.json \
       --capture=tee-sys -vvv -s

4. Generate RCCL performance heatmap:

   .. code:: bash

     cvs generate heatmap --actual /tmp/rccl_perf_results.json --reference /path/to/golden_reference.json --output /var/www/html/cvs/rccl_heatmap.html --title "RCCL Performance Comparison"

Each ``cvs run`` writes its pytest HTML report and log as described in :ref:`pytest-html-report-and-bundle`. Pass ``--workspace <shared path>`` to choose where the run directory is created.

.. _rccl-run-managed:

Run on managed compute
~~~~~~~~~~~~~~~~~~~~~~

Follow :doc:`/how-to/run-on-managed-compute` to run RCCL on managed compute.

On SPUR 0.11, pairwise and incremental RCCL tests are skipped. These tests run under Slurm.

Related resources
=================

- :doc:`/reference/configuration-files/network/rccl`
- :doc:`/reference/cli/cvs-run`
- :doc:`/how-to/copy-to-cluster`
