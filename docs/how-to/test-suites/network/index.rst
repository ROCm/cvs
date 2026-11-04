.. meta::
  :description: Run CVS network and collective communication test suites to validate InfiniBand, RCCL, and RDMA interconnect before distributed workloads on AMD Instinct clusters.
  :keywords: CVS, network, InfiniBand, RCCL, MORI, RDMA, AMD Instinct, ROCm, AMD, GPU, IB, collective communication

****************************************************************************
Run Cluster Validation Suite (CVS) network and collective communication suites
****************************************************************************

InfiniBand performance, RCCL collectives, and MORI RDMA benchmarks validate cluster interconnect before distributed workloads.

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - Suite
     - How to run
     - Config reference
   * - InfiniBand (IB Perf)
     - :doc:`/how-to/test-suites/network/ib-perf`
     - :doc:`/reference/configuration-files/network/ib`
   * - RCCL
     - :doc:`/how-to/test-suites/network/rccl`
     - :doc:`/reference/configuration-files/network/rccl`
   * - MORI (RDMA)
     - :doc:`/how-to/test-suites/network/mori`
     - :doc:`/reference/configuration-files/network/mori`

See also :doc:`/how-to/test-suites/index` for common ``cvs run`` flags and workflow.

Test results
============

Sample output snapshots from an RCCL performance run.

.. important::

   The numbers in these sample reports and screenshots (bandwidth, latency, and
   every other metric) are illustrative sample values, not final tuned
   results. They are shown only to illustrate the report layout.

HTML test report
----------------

The pytest HTML report lists each RCCL perf test and links to the RCCL Run
Deck.

.. image:: /images/network_rccl_pytest_report.png
   :alt: RCCL pytest HTML test report

The report shows the environment block (CVS version, cluster file, config
file), links to the RCCL Performance Report and Run Deck, and a per-test
result table covering host info collection, firewall checks, and each
collective parametrized by algorithm and protocol.

Run Deck report
---------------

The static Run Deck HTML summarizes the run and links to the interactive
viewer.

.. image:: /images/network_rccl_rundeck_run_card.png
   :alt: RCCL Run Deck run card

The run card shows the cluster topology (nodes, MPI ranks, local ranks),
the collectives tested, message size range, NIC model, threshold mode, and
links to the pytest report, run log, cluster file, and config file.

Run Deck interactive viewer
---------------------------

The interactive viewer adds series and search filters, charts with legend
swatches and reset zoom, and a paginated full results table.

.. image:: /images/network_rccl_rundeck_viewer_filters.png
   :alt: RCCL Run Deck interactive viewer filters

The filter bar narrows results by collective and searches any column value.
It shows the total row and series counts and instructions for scroll-to-zoom
and drag-to-pan on the charts.

.. image:: /images/network_rccl_rundeck_viewer_charts.png
   :alt: RCCL Run Deck interactive viewer bandwidth charts

The bus bandwidth chart plots GB/s against message size for each collective,
with one color-coded series per collective and dot markers at each measured
point.

.. image:: /images/network_rccl_rundeck_full_results.png
   :alt: RCCL Run Deck interactive viewer full results table

The full results table lists every collective and message size with the
measured bus bandwidth, algorithm bandwidth, and time in microseconds.
