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
   every other metric) are **illustrative sample values, not final tuned
   results**. They are shown only to illustrate the report layout.

HTML test report
----------------

The pytest HTML report lists each RCCL perf test and links to the bundled RCCL
performance report.

.. image:: /images/network_rccl_pytest_report.png
   :alt: RCCL pytest HTML test report

RCCL performance report
---------------------

The bundled RCCL performance report summarizes collective bandwidth and
latency across message sizes.

**Bandwidth graph:**

.. image:: /images/network_rccl_perf_bandwidth_graph.png
   :alt: RCCL performance bandwidth graph

**Results table:**

.. image:: /images/network_rccl_perf_results_table.png
   :alt: RCCL performance results table
