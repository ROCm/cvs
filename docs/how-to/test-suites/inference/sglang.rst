.. meta::
<<<<<<< HEAD
  :description: Run SGLang disaggregated inference tests
  :keywords: CVS, sglang

************
SGLang tests
************
=======
  :description: Run SGLang inference benchmarks with CVS on MI30X clusters
  :keywords: CVS, SGLang, inference, benchmark, distributed, disaggregated, LLM, ROCm

*************************
Run SGLang inference tests
*************************

CVS provides three SGLang suites under ``cvs/tests/inference/sglang/``. Each suite is a
separate pytest module; pick the one that matches your topology, then point ``--config_file``
at a template from ``cvs/input/config_file/inference/sglang/``.

For the full configuration schema, threshold format, and parameter reference, see
:doc:`/reference/configuration-files/sglang`.

Test suites
===========

.. list-table::
   :widths: 2 3 5
   :header-rows: 1

   * - CVS suite name
     - Source module
     - What it runs
   * - ``sglang_single``
     - ``sglang_single.py``
     - One unified ``sglang.launch_server`` on a single ``benchmark_serv_node`` (local TP).
   * - ``sglang_distributed``
     - ``sglang_distributed.py``
     - One unified multi-node server (TP/PP across ``server_node_list``).
   * - ``sglang_disagg_distributed``
     - ``sglang_disagg_distributed.py``
     - Disaggregated prefill/decode with a proxy router.
>>>>>>> cddd5e14 (Adding IB device-8)

.. _sglang-set-up-config:

Set up config
=============

<<<<<<< HEAD
1. List available SGLang configs:

   .. code:: bash

     cvs config list inference/sglang

2. Copy the configuration file for your workload, for example:

   .. code:: bash

     cvs config copy inference/sglang/mi35x_sglang_distributed.json --output ~/cvs_workspace/inference/sglang/mi35x_sglang_distributed.json

3. Edit the file — set ``container_image`` and replace every ``<changeme>`` with cluster-specific values.

Full parameter list: :doc:`/reference/configuration-files/inference/sglang`.
=======
1. List available SGLang templates:

   .. code:: bash

     cvs copy-config --list | grep inference/sglang

2. Copy the configuration (and threshold file, if you edit thresholds locally):

   .. code:: bash

     cvs copy-config inference/sglang/mi3xx_sglang_llama_70b_single.json \
       --output ~/cvs_workspace/mi3xx_sglang_llama_70b_single.json

     cvs copy-config inference/sglang/mi3xx_sglang_llama_70b_threshold.json \
       --output ~/cvs_workspace/mi3xx_sglang_llama_70b_threshold.json

3. Copy a cluster file (container backend recommended):

   .. code:: bash

     cvs copy-config cluster_container.json --output ~/cvs_workspace/cluster.json

4. Edit the config — set ``container_image``, replace every ``<changeme>`` with
   cluster-specific values, and ensure ``threshold_file`` resolves to your threshold JSON.

Shipped config templates:

.. list-table::
   :widths: 3 2
   :header-rows: 1

   * - Config file
     - Use with suite
   * - ``mi3xx_sglang_llama_70b_single.json``
     - ``sglang_single``
   * - ``mi3xx_sglang_deepseek_r1_0528_single.json``
     - ``sglang_single``
   * - ``mi3xx_sglang_llama_70b_distributed.json``
     - ``sglang_distributed``
   * - ``mi3xx_sglang_deepseek_r1_0528_distributed.json``
     - ``sglang_distributed``
   * - ``mi3xx_sglang_llama_70b_disaggregated.json``
     - ``sglang_disagg_distributed``
   * - ``mi3xx_sglang_deepseek_r1_0528_disaggregated.json``
     - ``sglang_disagg_distributed``

.. note::

  When a config's ``benchmark_params`` block has more than one variant key, export
  ``SGLANG_BENCHMARK_KEY`` (for example ``llama-70b`` or ``deepseek-r1``) before running.
>>>>>>> cddd5e14 (Adding IB device-8)

.. _sglang-run-tests:

Run tests
=========

<<<<<<< HEAD
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
=======
List stages in a suite:

.. code:: bash

  cvs list sglang_single

``sglang_single`` stages
------------------------

.. code:: text

  Available tests in sglang_single:
    - test_launch_container
    - test_rms_norm
    - test_launch_server
    - test_poll_for_server_ready
    - test_openai_compatible_http_endpoints
    - test_run_lm_eval_hellaswag_benchmark_test
    - test_run_lm_eval_gsm8k_benchmark_test
    - test_run_performance_benchmark_test
    - test_verify_dmesg_after_benchmark
    - test_print_results_table
    - test_teardown

``test_run_performance_benchmark_test`` is parametrized once per ISL/OSL/concurrency cell
defined in the threshold file (for example ``isl1024-osl1024-c64``).

Example run:

.. code:: bash

  cvs run sglang_single \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/mi3xx_sglang_llama_70b_single.json \
    --html ~/cvs_results/sglang_single.html --self-contained-html \
    --log-file /tmp/sglang.log -vvv

``sglang_distributed`` stages
-----------------------------

.. code:: text

  Available tests in sglang_distributed:
    - test_launch_container
    - test_setup_ibv_devices
    - test_rms_norm
    - test_launch_server
    - test_poll_for_server_ready
    - test_openai_compatible_http_endpoints
    - test_run_lm_eval_hellaswag_benchmark_test
    - test_run_lm_eval_gsm8k_benchmark_test
    - test_run_performance_benchmark_test
    - test_verify_dmesg_after_benchmark
    - test_distributed_gpu_topology
    - test_print_results_table
    - test_teardown

Example run:

.. code:: bash

  cvs run sglang_distributed \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/mi3xx_sglang_llama_70b_distributed.json \
    --html ~/cvs_results/sglang_distributed.html --self-contained-html \
    --log-file /tmp/sglang.log -vvv

``sglang_disagg_distributed`` stages
------------------------------------

.. code:: text

  Available tests in sglang_disagg_distributed:
    - test_launch_container
    - test_setup_ibv_devices
    - test_rms_norm
    - test_launch_prefill_servers
    - test_launch_decode_servers
    - test_poll_for_server_ready
    - test_launch_proxy_router
    - test_openai_compatible_http_endpoints
    - test_run_lm_eval_hellaswag_benchmark_test
    - test_run_lm_eval_gsm8k_benchmark_test
    - test_run_performance_benchmark_test
    - test_verify_dmesg_after_benchmark
    - test_disagg_gpu_topology
    - test_print_results_table
    - test_teardown

Example run:

.. code:: bash

  cvs run sglang_disagg_distributed \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/mi3xx_sglang_llama_70b_disaggregated.json \
    --html ~/cvs_results/sglang_disagg.html --self-contained-html \
    --log-file /tmp/sglang.log -vvv

Direct pytest invocation
------------------------

Each module can also be run with pytest:

.. code:: bash

  pytest cvs/tests/inference/sglang/sglang_single.py \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/mi3xx_sglang_llama_70b_single.json \
    --html ~/cvs_results/sglang_single.html

Read the results
================

With ``--html``, CVS writes an HTML report plus ``sglang_run_deck.html`` (interactive viewer)
using the shared ``sglang`` report profile.

Key lifecycle stages to watch:

- **Container launch** — ``test_launch_container`` must pass before any server work runs.
- **IB setup** — ``test_setup_ibv_devices`` (distributed and disaggregated only) validates RDMA
  inside the container.
- **Server ready** — ``test_poll_for_server_ready`` waits for the SGLang server log to show ready.
- **Smoke** — ``test_openai_compatible_http_endpoints`` probes the OpenAI-compatible API.
- **Performance** — ``test_run_performance_benchmark_test`` runs ``sglang.bench_serving`` for each
  threshold cell. Set ``enforce_thresholds: false`` under ``bench_serv_random`` to record metrics
  without failing on unc calibrated gates.
- **Accuracy** — ``test_run_lm_eval_hellaswag_benchmark_test`` and
  ``test_run_lm_eval_gsm8k_benchmark_test`` run lm-eval tasks configured in ``inference_tests``.
- **Summary** — ``test_print_results_table`` prints throughput/latency/accuracy in the console and
  report.
- **Teardown** — ``test_teardown`` stops containers even when a prior stage failed.

Logs are written under ``log_dir`` from the config (default ``/home/{user-id}/LOGS/sglang``).
>>>>>>> cddd5e14 (Adding IB device-8)
