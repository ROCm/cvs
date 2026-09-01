.. meta::
  :description: Run TorchTitan pre-training benchmarks
  :keywords: CVS, TorchTitan, training

***********************
TorchTitan training tests
***********************

TorchTitan validates single-node and multi-node pre-training on AMD Instinct GPUs. CVS drives training inside a container, parses logs, and gates on throughput, loss-curve, and optional checkpoint metrics.

There are two suites:

- ``torchtitan_single`` — single-node configs (``framework: torchtitan_single``)
- ``torchtitan_distributed`` — multi-node configs (``framework: torchtitan_distributed``); adds RDMA setup

.. _torchtitan-set-up-config:

Set up config
=============

1. List available TorchTitan configuration files:

   .. code:: bash

     cvs config list training/torchtitan

2. Copy the configuration file you need, for example:

   .. code:: bash

     cvs config copy training/torchtitan/mi355x_torchtitan_llama-3.1-8b_single.json --output ~/cvs_workspace/training/torchtitan/mi355x_torchtitan_llama-3.1-8b_single.json

3. Replace every ``<changeme>`` with cluster-specific values (container image, HF token path, NCCL/RDMA fields on distributed configs).
4. Edit the sibling ``*_threshold.json`` file if you gate on performance metrics.
5. Set ``sweep.runs`` to the precision combos you want to execute.

Full parameter list: :doc:`/reference/configuration-files/training/torchtitan`.

.. _torchtitan-run-tests:

Run tests
=========

List available stages
---------------------

.. code:: bash

  cvs list torchtitan_single

.. code:: text

  Available tests in torchtitan_single:
    - test_launch_container
    - test_download_tokenizer
    - test_smoke
    - test_training[...]
    - test_checkpoint[...]
    - test_metric[...]
    - test_loss_curve[...]
    - test_teardown

.. code:: bash

  cvs list torchtitan_distributed

.. code:: text

  Available tests in torchtitan_distributed:
    - test_launch_container
    - test_setup_rdma
    - test_download_tokenizer
    - test_smoke
    - test_training[...]
    - test_checkpoint[...]
    - test_metric[...]
    - test_loss_curve[...]
    - test_teardown

Single-node example
-------------------

.. code:: bash

  cvs run torchtitan_single \
    --cluster_file input/cluster_file/cluster.json \
    --config_file input/config_file/training/torchtitan/mi355x_torchtitan_llama-3.1-8b_single.json \
    --html ./logs/torchtitan_single.html --self-contained-html -vvv -s

Distributed example
-------------------

.. code:: bash

  cvs run torchtitan_distributed \
    --cluster_file input/cluster_file/cluster.json \
    --config_file input/config_file/training/torchtitan/mi355x_torchtitan_llama-3.3-70b_distributed.json \
    --html ./logs/torchtitan_distributed.html --self-contained-html -vvv -s

Run a single stage
------------------

.. code:: bash

  cvs run torchtitan_single test_smoke \
    --cluster_file input/cluster_file/cluster.json \
    --config_file input/config_file/training/torchtitan/mi355x_torchtitan_llama-3.1-8b_single.json

Use a single-node config with ``torchtitan_single`` and a distributed config with ``torchtitan_distributed``. The config ``framework`` field must match the suite name.
