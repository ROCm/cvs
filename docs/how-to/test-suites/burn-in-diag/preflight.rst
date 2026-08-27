.. meta::
  :description: Run CVS preflight and node smoke checks
  :keywords: CVS, preflight, node smoke

****************
Preflight tests
****************

Preflight checks validate cluster health and configuration consistency before performance, RCCL, training, or inference workloads. Checks include GPU node health, optional MI4XX fabric admission, IFoE and TransferBench gates, RDMA inventory, and optional Node Smoke tiers.

.. _preflight-set-up-config:

Set up config
=============

1. Copy the preflight configuration file:

   .. code:: bash

     cvs config copy preflight/preflight_config.json --output ~/cvs_workspace/preflight/preflight_config.json

2. Edit paths, thresholds, and optional Node Smoke settings. Replace every ``<changeme>`` placeholder.

Full parameter list: :doc:`/reference/configuration-files/burn-in-diag/preflight`.

.. _preflight-run-tests:

Run tests
=========

List available checks:

.. code:: bash

  cvs list preflight_checks

Run the full preflight suite:

.. code:: bash

  cvs run preflight_checks \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/preflight/preflight_config.json \
    --html=/var/www/html/cvs/preflight.html --capture=tee-sys --self-contained-html \
    --log-file=/tmp/preflight.log -vvv -s

Run individual Node Smoke tiers:

.. code:: bash

  cvs run preflight_checks test_node_smoke_tier1 \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/preflight/preflight_config.json

  cvs run preflight_checks test_node_smoke_tier3 \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/preflight/preflight_config.json
