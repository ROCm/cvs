.. meta::
  :description: Run CVS preflight and node smoke check tests to validate cluster health, RDMA inventory, and GPU node readiness before performance or training workloads.
  :keywords: CVS, preflight, node smoke, AMD Instinct, ROCm, AMD, GPU, RDMA, InfiniBand, TransferBench, health

*************************************************************************
Run CVS preflight and node smoke check tests on AMD Instinct GPU clusters
*************************************************************************

Preflight checks validate cluster health and configuration consistency before performance, RCCL, training, or inference workloads. Checks include GPU node health, optional MI4XX fabric admission, IFoE and TransferBench gates, RDMA inventory, and Node Smoke tiers. Node Smoke Tier 1, Tier 2, and Tier 3 run by default.

.. _preflight-set-up-config:

Set up config
=============

Follow these steps to set up the preflight configuration.

1. Copy the preflight configuration file:

   .. code:: bash

     cvs config copy preflight/preflight_config.json --output ~/cvs_workspace/preflight/preflight_config.json

2. Edit paths, thresholds, and Node Smoke settings. Replace every ``<changeme>`` placeholder. Node Smoke runs unless ``connectivity_mode`` is ``"skip"``.

3. Review the ``host_readiness`` sections: kernel parameters, CPU idle states, persistent environment variables, GPU idle state, hardware inventory, and platform extras. Add ``sku_profiles`` that describe your node hardware to gate inventory. Without profiles, inventory is only recorded. DIMM and PCIe AtomicOps checks need passwordless ``sudo`` for ``dmidecode`` and ``lspci`` on every node.

For the complete field reference including Node Smoke tier options and RDMA inventory settings, see :doc:`/reference/configuration-files/burn-in-diag/preflight`.

.. _preflight-run-tests:

Run tests
=========

Run the following command to list available checks:

.. code:: bash

  cvs list preflight_checks

Run the full preflight suite:

.. code:: bash

  cvs run preflight_checks \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/preflight/preflight_config.json \
    --capture=tee-sys -vvv -s

Run individual Node Smoke tiers:

.. code:: bash

  cvs run preflight_checks test_node_smoke_tier1 \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/preflight/preflight_config.json
  cvs run preflight_checks test_node_smoke_tier3 \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/preflight/preflight_config.json

Run on managed compute
~~~~~~~~~~~~~~~~~~~~~~

See :doc:`/how-to/run-on-managed-compute` for guidance on running these preflight checks on managed compute.
