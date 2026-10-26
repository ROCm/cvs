.. meta::
  :description: Run CVS cluster topology discovery to probe every node's GPU XGMI mesh, CPU, NUMA, and NIC layout and view the stitched cluster graph.
  :keywords: CVS, topology, cluster, GPU, XGMI, NUMA, NIC, amd-smi, AMD Instinct, ROCm, AMD

*******************************************
Run CVS cluster topology discovery
*******************************************

The ``topology_discover`` suite probes every node in the cluster file for its hardware topology — the GPU XGMI mesh from ``amd-smi`` and the CPU, NUMA, and NIC layout from sysfs — then stitches the per-node results into one cluster graph grouped by rack and chassis. It writes the graph as JSON under the run directory and links a graphical view in the HTML report. Run it to inventory a cluster before burn-in, or to confirm every node is reachable and reports the GPUs it should.

The verdict covers discovery, not hardware pass/fail: the test passes when every node in the cluster file responded and returned usable data, and fails listing the nodes that did not.

.. _topology-set-up-config:

Set up config
=============

1. Copy the topology configuration file:

   .. code:: bash

     cvs config copy topology/topology_config.json --output ~/cvs_workspace/topology/topology_config.json

2. The defaults work as-is; there are no ``<changeme>`` placeholders. Adjust ``probe_timeout`` only for very large clusters. For the field reference, see :doc:`/reference/configuration-files/burn-in-diag/topology`.

.. _topology-run-tests:

Run tests
=========

Pass the cluster file and the topology config:

.. code:: bash

  cvs run topology_discover \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/topology/topology_config.json \
    --capture=tee-sys -vvv -s

The cluster graph is written to ``<run_dir>/topology/cluster_topology.json`` and each node's raw probe output under ``<run_dir>/topology/<node>/``. The HTML report links a **Cluster Topology** view showing the rack, chassis, node, and GPU hierarchy with the intra-node XGMI links drawn between GPUs. For report and bundle details, see :ref:`pytest-html-report-and-bundle`.

Scale-up and scale-out
======================

The report classifies each node's fabrics so you can decide how to shard a model:

- **Scale-up** is the intra-node GPU fabric. The GPU XGMI mesh is the scale-up network, and each node also reports its partition mode (for example ``SPX/NPS1`` from ``amd-smi partition``). Keep tensor-parallel shards on this fabric.
- **Scale-out** is the inter-node network. Each node reports its backend RDMA NICs — count, fabric type (RoCE or InfiniBand), line rate, and link state (for example ``8x 400G RoCE (NDR), all ACTIVE``). Data- and pipeline-parallel traffic crosses this fabric.

.. note::

   A node can report its scale-out *capacity* (how many NICs, how fast) but not the *wiring* — which switch each NIC connects to — because that lives in the switches. When an LLDP daemon runs on the nodes, the suite resolves each uplink to its top-of-rack switch and fills ``inter_node_edges`` (grouping nodes by switch); otherwise that section reads "not discovered". Full rack cabling needs a fabric-aware source such as ARC. The ``inter_node_edges`` records use ARC's edge shape so an ARC-derived graph can populate them later.

A node whose amdgpu driver is not loaded still succeeds: it reports its CPU, NUMA, and NIC layout with zero GPUs rather than failing.
