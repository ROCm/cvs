.. meta::
  :description: Run network and collective communication CVS test suites
  :keywords: CVS, IB Perf, RCCL, MORI, network

**************
Network tests
**************

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

See also :doc:`/how-to/run-tests/index` for common ``cvs run`` flags and workflow.
