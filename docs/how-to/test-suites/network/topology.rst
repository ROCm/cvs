.. meta::
  :description: Run the CVS topology NIC bandwidth test to measure cross-node RDMA bandwidth for every NIC pair and find a weak cable, NIC, or uplink on an AMD Instinct cluster.
  :keywords: CVS, topology, NIC, RDMA, bandwidth, heatmap, TransferBench, AMD Instinct, ROCm, AMD, GPU, network

****************************************************
Run CVS topology NIC bandwidth tests
****************************************************

The topology ``nic_bandwidth`` suite measures RDMA bandwidth between every cross-node NIC pair with TransferBench, then compares each pair against the cluster median. A single weak pair points at one link, a NIC that is slow in every direction points at that NIC or its cable, and a node pair that is slow on every NIC points at an uplink or placement problem.

Run it after IB Perf passes, when you want to know which nodes to place a job on rather than whether the fabric works at all.

.. _topology-prerequisites:

Prerequisites
=============

- A cluster file with at least two nodes. The suite skips on a single node, because only cross-node pairs are measured.
- SSH key access from the machine where you run ``cvs``, as in your cluster file.
- ``amd-smi`` on every node, on ``PATH`` or at ``/opt/rocm/bin``. GPU discovery uses it.
- TransferBench on every node, on ``PATH`` or at the path you set in ``tb_binary``. The health suite's ``install_transferbench`` test can install it.
- Backend NICs with the link up. A NIC whose link is down, or that has no GPU in its own PCI domain (typically the management NIC), is left out and listed in the report with the reason.

Set up config
=============

Copy the shipped config and edit your copy:

.. code:: bash

   cvs config list                       # show config file paths
   cp <path>/topology/topology_config.json ./my_topology_config.json

The shipped defaults run everywhere. See :doc:`/reference/configuration-files/network/topology` for every field.

Run the tests
=============

.. code:: bash

   cvs run nic_bandwidth --cluster_file ./my_cluster.json \
           --config_file ./my_topology_config.json

Run on managed compute
----------------------

See :doc:`/how-to/run-on-managed-compute` for running CVS under a scheduler. This suite adds no scheduler-specific requirement.

Test results
============

The test fails when any pair, NIC, or node pair falls below ``flag_below_pct_of_median`` percent of the cluster median. Each failure names the pairs, NICs, or nodes involved.

The run writes ``topology.json`` with the full matrix and the flagged entries, next to the run's other artefacts.

The Run Deck adds a ``nic_bandwidth`` group:

- a **cluster-wide NIC-to-NIC grid**, every source NIC against every destination NIC
- a **per-node heatmap** for each node, its own NICs against every peer NIC
- a **node-to-node rollup**, the worst NIC pair between each ordered node pair

Cells show bandwidth as a percentage of the cluster median, so one threshold reads the same on any cluster. Colour runs from red through orange and yellow-green to green as a pair approaches the median; red is below the threshold.

Same-node cells are blank. Only cross-node pairs are measured: TransferBench's ``nicp2p`` preset cannot skip NIC self-loopback, and a loopback run has been implicated in a host-fatal machine check, so CVS generates its own transfer list that never pairs a node with itself.

See :doc:`/how-to/test-suites/index` for common ``cvs run`` flags, the report bundle, and the workflow.
