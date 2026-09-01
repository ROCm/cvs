.. meta::
  :description: Run SGLang disaggregated inference tests
  :keywords: CVS, sglang

************
SGLang tests
************

.. _sglang-set-up-config:

Set up config
=============

1. List available SGLang configs:

   .. code:: bash

     cvs config list inference/sglang

2. Copy the configuration file for your workload, for example:

   .. code:: bash

     cvs config copy inference/sglang/mi35x_sglang_distributed.json --output ~/cvs_workspace/inference/sglang/mi35x_sglang_distributed.json

3. Edit the file — set ``container_image`` and replace every ``<changeme>`` with cluster-specific values.

Full parameter list: :doc:`/reference/configuration-files/inference/sglang`.

.. _sglang-run-tests:

Run tests
=========

Sglang test scripts
------------------------------

You can list all available Sglang test cases using the CLI:

.. code:: bash

  cvs list sglang_deepseek_r1_671b_distributed

.. code:: text

  Available tests in sglang_deepseek_r1_671b_distributed:
    - test_cleanup_stale_containers
    - test_launch_decode_servers
    - test_launch_inference_containers
    - test_launch_prefill_servers
    - test_launch_proxy_router
    - test_poll_for_server_ready
    - test_rms_norm
    - test_run_benchmark_test
    - test_run_gsm8k_benchmark_test
    - test_setup_ibv_devices

.. code:: bash

  cvs list sglang_llama_70b_distributed

.. code:: text

  Available tests in sglang_llama_70b_distributed:
    - test_cleanup_stale_containers
    - test_launch_decode_servers
    - test_launch_inference_containers
    - test_launch_prefill_servers
    - test_launch_proxy_router
    - test_poll_for_server_ready
    - test_rms_norm
    - test_run_benchmark_test
    - test_run_gsm8k_benchmark_test
    - test_setup_ibv_devices
    
Use these scripts to run the Sglang tests.

.. code:: bash

  cvs run sglang_deepseek_r1_671b_distributed --cluster_file input/cluster_file/cluster.json --config_file input/config_file/inference/sglang/mi35x_sglang_distributed.json --html=/var/www/html/cvs/sglang.html --capture=tee-sys --self-contained-html --log-file=/tmp/sglang.log -vvv -s

.. code:: bash

  cvs run sglang_llama_70b_distributed --cluster_file input/cluster_file/cluster.json --config_file input/config_file/inference/sglang/mi35x_sglang_distributed.json --html=/var/www/html/cvs/sglang.html --capture=tee-sys --self-contained-html --log-file=/tmp/sglang.log -vvv -s
