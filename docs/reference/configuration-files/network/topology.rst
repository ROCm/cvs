.. meta::
  :description: CVS topology NIC bandwidth JSON configuration schema, covering coverage mode, median threshold, RDMA traffic class, and TransferBench settings.
  :keywords: CVS, topology, NIC, bandwidth, configuration, JSON, TransferBench, traffic class, ROCm, AMD, GPU

****************************************************
Topology NIC bandwidth configuration
****************************************************

Schema for ``topology/topology_config.json``. Every field has a working default; copy the shipped file and change only what you need. See :doc:`/how-to/test-suites/network/topology` for running the suite.

All fields live under the top-level ``topology`` key.

Coverage and thresholds
=======================

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Field
     - Default
     - Meaning
   * - ``coverage``
     - ``auto``
     - Which pairs to measure. ``full`` measures every cross-node NIC pair. ``quick`` measures two reference nodes against every other node. ``auto`` uses ``full`` while a full run stays within a pair budget of 2000 and ``quick`` above that. The budget counts measured pairs rather than nodes because a full run grows with the square of both the node count and the NICs per node: with 8 backend NICs per node ``auto`` stays on ``full`` up to 6 nodes, and with 2 it stays on ``full`` up to 22. Set ``coverage`` to ``full`` explicitly to measure every pair regardless of how long that takes.
   * - ``flag_below_pct_of_median``
     - ``90``
     - A pair is flagged when its bandwidth is below this percentage of the median of all measured pairs. It is also the heatmap threshold: cells below it are red.

Fabric
======

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Field
     - Default
     - Meaning
   * - ``traffic_class``
     - ``auto``
     - RDMA traffic class, passed to TransferBench as ``NIC_TRAFFIC_CLASS``. ``auto`` uses ``NCCL_IB_TC`` from the cluster file's ``env_vars`` when set, otherwise ``0``. You can also set an integer from 0 to 255. The report prints the class used.
   * - ``nic_filter``
     - ``""``
     - Restricts TransferBench to matching NICs (``TB_NIC_FILTER``). Empty means every selected backend NIC.

TransferBench
=============

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Field
     - Default
     - Meaning
   * - ``tb_binary``
     - ``TransferBench``
     - Executable name or absolute path. A bare name is resolved on each node.
   * - ``use_sudo``
     - ``false``
     - Run TransferBench under ``sudo``. Needs passwordless sudo on every node.
   * - ``transferbench_env``
     - ``{}``
     - Extra environment for the TransferBench process, such as ``LD_LIBRARY_PATH``. Keys must be valid shell environment variable names.

Advanced
========

Under ``topology.advanced``:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Field
     - Default
     - Meaning
   * - ``message_size``
     - ``64M``
     - Transfer size per measurement.
   * - ``queue_pairs``
     - ``4``
     - RDMA queue pairs per transfer.
   * - ``iterations``
     - ``10``
     - Measured iterations per pair.
   * - ``warmup_iterations``
     - ``3``
     - Unmeasured warmup iterations per pair.
   * - ``timeout``
     - ``1800``
     - Seconds allowed for the TransferBench run.
   * - ``master_port``
     - ``31337``
     - TCP port for the multi-rank bootstrap. The ``CVS_TOPO_TB_PORT`` environment variable overrides it.

See :doc:`/reference/configuration-files/network/index` for the other network suites.
