.. meta::
  :description: Run JAX MaxText training benchmarks
  :keywords: CVS, jax, jaxmaxtext, MaxText

**********************
JAX MaxText tests
**********************

JAX training in CVS is **jaxmaxtext**. The legacy ``jax_llama3_1_*`` suites have
been removed. Use ``jaxmaxtext_single`` with a single-node config
(``training.distributed: false``) and ``jaxmaxtext_distributed`` with a
distributed config (``training.distributed: true``). Copy the matching
``*_threshold.json`` into the same directory as the suite config.

.. _jax-set-up-config:

Set up config
=============

1. List available JAX MaxText configuration files:

   .. code:: bash

     cvs config list training/jaxmaxtext

2. Copy the configuration file and its sibling threshold file, for example:

   .. code:: bash

     cvs config copy training/jaxmaxtext/mi300x_jaxmaxtext_llama-3.3-70b_single.json --output ~/cvs_workspace/training/jaxmaxtext/mi300x_jaxmaxtext_llama-3.3-70b_single.json
     cvs config copy training/jaxmaxtext/mi300x_jaxmaxtext_llama-3.3-70b_single_threshold.json --output ~/cvs_workspace/training/jaxmaxtext/mi300x_jaxmaxtext_llama-3.3-70b_single_threshold.json

3. Replace every ``<changeme>`` with cluster-specific values (especially NCCL/RDMA fields on distributed configs).
4. Change any other parameters relevant to your testing requirements.

Full parameter list: :doc:`/reference/configuration-files/training/jaxmaxtext`.

.. _jax-run-tests:

Run tests
=========

JAX MaxText test scripts
------------------------

You can list all available JAX MaxText test cases using the CLI:

.. code:: bash

  cvs list jaxmaxtext_single

.. code:: text

  Available tests in jaxmaxtext_single:
    - test_launch_container
    - test_setup_tokenizer
    - test_smoke
    - test_training_run
    - test_metric
    - test_loss_curve
    - test_checkpoint_resume
    - test_print_results_table
    - test_teardown

.. code:: bash

  cvs list jaxmaxtext_distributed

.. code:: text

  Available tests in jaxmaxtext_distributed:
    - test_launch_container
    - test_setup_rdma
    - test_setup_tokenizer
    - test_smoke
    - test_training_run
    - test_metric
    - test_loss_curve
    - test_checkpoint_resume
    - test_print_results_table
    - test_teardown

Use these scripts to run the JAX MaxText tests.

Single-node:

.. code:: bash

  cvs run jaxmaxtext_single --cluster_file input/cluster_file/cluster.json --config_file input/config_file/training/jaxmaxtext/mi300x_jaxmaxtext_llama-3.3-70b_single.json --html=/var/www/html/cvs/jaxmaxtext_single.html --capture=tee-sys --self-contained-html --log-file=/tmp/jaxmaxtext_single.log -vvv -s

Distributed:

.. code:: bash

  cvs run jaxmaxtext_distributed --cluster_file input/cluster_file/cluster.json --config_file input/config_file/training/jaxmaxtext/mi325x_jaxmaxtext_llama-3.3-70b_distributed.json --html=/var/www/html/cvs/jaxmaxtext_distributed.html --capture=tee-sys --self-contained-html --log-file=/tmp/jaxmaxtext_distributed.log -vvv -s
