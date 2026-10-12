.. meta::
  :description: Run SGLang LLM inference benchmarks with CVS on AMD Instinct GPU clusters, covering single-node, distributed, and disaggregated prefill/decode topologies.
  :keywords: CVS, SGLang, inference, benchmark, AMD Instinct, ROCm, AMD, GPU, LLM, disaggregated, distributed, Llama

**********************************************
Run SGLang LLM inference benchmarks with CVS
**********************************************

CVS provides three SGLang suites under ``cvs/tests/inference/sglang/``. Each suite is a
separate pytest module; pick the one that matches your topology, then point ``--config_file``
at a template from ``cvs/input/config_file/inference/sglang/``.

For the full configuration schema, threshold format, and parameter reference, see
:doc:`/reference/configuration-files/inference/sglang`.

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
     - One unified ``sglang.launch_server`` on the first host in ``cluster.json`` (local TP). Extra hosts are ignored.
   * - ``sglang_distributed``
     - ``sglang_distributed.py``
     - One unified multi-node server. The first ``server_params.nnodes`` hosts in ``cluster.json`` are the ranks. Rank 0 is the benchmark node.
   * - ``sglang_disagg_distributed``
     - ``sglang_disagg_distributed.py``
     - Disaggregated prefill/decode. The first even ``nnodes`` hosts are split into equal prefill and decode groups. Rank 0 is prefill coordinator, proxy, and benchmark. Rank 1 is decode coordinator.

.. _sglang-set-up-config:

Set up config
=============

1. List available SGLang templates:

   .. code:: bash

     cvs config copy --list | grep inference/sglang

2. Copy the configuration (and threshold file, if you edit thresholds locally):

   .. code:: bash

     cvs config copy inference/sglang/mi3xx_sglang_llama_70b_single.json \
       --output ~/cvs_workspace/mi3xx_sglang_llama_70b_single.json

     cvs config copy inference/sglang/mi325_sglang_llama_70b_threshold.json \
       --output ~/cvs_workspace/mi325_sglang_llama_70b_threshold.json

3. Copy a cluster file (container backend recommended):

   .. code:: bash

     cvs config copy cluster_container.json --output ~/cvs_workspace/cluster.json

4. Edit the config — set ``container.image``, replace every ``<changeme>`` with
   cluster-specific values, and ensure ``threshold_json`` resolves to your threshold JSON.
   Do not set ``benchmark_serv_node``, ``server_node_list``, or the disaggregated role
   lists. The suite assigns those from ``cluster.json`` ``node_dict`` order and ``nnodes``.

Shipped templates are named ``mi3xx_sglang_<model>_<topology>.json`` and
``mi35x_sglang_<model>_<topology>.json``. ``<model>`` is ``llama_70b``,
``deepseek_r1_0528``, or ``qwen35_397b``. ``<topology>`` is ``single``, ``distributed``,
or ``disaggregated``. MI30x configs use the ``mi325_sglang_*_threshold.json`` files.
MI35x configs use the ``mi355_sglang_*_threshold.json`` files.

.. list-table::
   :widths: 2 3
   :header-rows: 1

   * - Topology suffix
     - Suite
   * - ``single``
     - ``sglang_single``
   * - ``distributed``
     - ``sglang_distributed``
   * - ``disaggregated``
     - ``sglang_disagg_distributed``

.. note::

  Each file is one workload. ``sweep.runs`` selects which ``ISL,OSL,TP,PP,CONC`` cells
  to parametrize. Empty ``runs`` uses every performance cell in the threshold JSON.
  Order the hosts in ``cluster.json`` the way the suite should assign ranks.
  Disaggregated ``nnodes`` must be even. HTTP defaults to port 8000; set
  ``server_params.serve_port`` to change the client port (MI35x multi-node templates
  use ``8010``). ``HSA_FORCE_FINE_GRAIN_PCIE`` is read from ``container.runtime.args.env``
  and defaults to ``1`` when the key is omitted or blank.

.. _sglang-run-tests:

Run tests
=========

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
    - test_scan_inference_logs_for_failure
    - test_openai_compatible_http_endpoints
    - test_run_lm_eval_hellaswag_benchmark_test
    - test_run_lm_eval_gsm8k_benchmark_test
    - test_run_performance_benchmark_test
    - test_verify_dmesg_after_benchmark
    - test_scan_inference_logs_after_workload
    - test_print_results_table
    - test_teardown

``test_run_performance_benchmark_test`` is parametrized once per combo in ``sweep.runs``
(for example ``isl1024-osl1024-c64``). Empty ``runs`` uses every performance cell in the
threshold file. TP/PP in the combo may differ from the threshold-file key; matching is
exact, then unique ``ISL,OSL,CONC``.

Example run:

.. code:: bash

  cvs run sglang_single \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/mi3xx_sglang_llama_70b_single.json \
    -vvv

``sglang_distributed`` stages
-----------------------------

.. code:: text

  Available tests in sglang_distributed:
    - test_launch_container
    - test_rms_norm
    - test_launch_server
    - test_poll_for_server_ready
    - test_scan_inference_logs_for_failure
    - test_openai_compatible_http_endpoints
    - test_run_lm_eval_hellaswag_benchmark_test
    - test_run_lm_eval_gsm8k_benchmark_test
    - test_run_performance_benchmark_test
    - test_verify_dmesg_after_benchmark
    - test_distributed_gpu_topology
    - test_scan_inference_logs_after_workload
    - test_print_results_table
    - test_teardown

Example run:

.. code:: bash

  cvs run sglang_distributed \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/mi3xx_sglang_llama_70b_distributed.json \
    -vvv

``sglang_disagg_distributed`` stages
------------------------------------

.. code:: text

  Available tests in sglang_disagg_distributed:
    - test_launch_container
    - test_rms_norm
    - test_launch_prefill_servers
    - test_launch_decode_servers
    - test_poll_for_server_ready
    - test_scan_inference_logs_for_failure
    - test_launch_proxy_router
    - test_openai_compatible_http_endpoints
    - test_run_long_context_accuracy
    - test_run_lm_eval_hellaswag_benchmark_test
    - test_run_lm_eval_gsm8k_benchmark_test
    - test_run_performance_benchmark_test
    - test_verify_dmesg_after_benchmark
    - test_disagg_gpu_topology
    - test_scan_inference_logs_after_workload
    - test_print_results_table
    - test_teardown

Example run:

.. code:: bash

  cvs run sglang_disagg_distributed \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/mi3xx_sglang_llama_70b_disaggregated.json \
    -vvv

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

``cvs run`` writes a pytest HTML report under the run directory by default, plus
``sglang_run_deck.html`` (interactive viewer) using the shared ``sglang`` report
profile. Direct ``pytest`` still needs ``--html`` for that bundle.

Key lifecycle stages to watch:

- **Container launch** — ``test_launch_container`` starts the containers and resets ``paths.log_dir``.
  Cleanup uses ``sudo -n`` when passwordless sudo is available, and a plain ``rm`` otherwise.
- **Server ready** — ``test_poll_for_server_ready`` waits until the server log contains
  ``server is fired up and ready to roll``. ``test_scan_inference_logs_for_failure`` runs only
  when that poll fails.
- **Smoke** — ``test_openai_compatible_http_endpoints`` probes the OpenAI-compatible API.
- **Long context** — ``test_run_long_context_accuracy`` (disaggregated only) runs when
  ``server_params.lng_ctx_activate`` is true. Shipped templates leave it false, so the stage skips.
- **Performance** — ``test_run_performance_benchmark_test`` runs ``sglang.bench_serving`` for each
  selected sweep cell. Set top-level ``enforce_thresholds: false`` to record metrics
  without failing on uncalibrated gates.
- **Accuracy** — ``test_run_lm_eval_hellaswag_benchmark_test`` and
  ``test_run_lm_eval_gsm8k_benchmark_test`` run lm-eval tasks configured in ``accuracy.tasks``.
- **Logs and GPUs** — after the benchmark, the suite scans dmesg and the inference logs.
  Distributed and disaggregated runs also record GPU occupancy with ``amd-smi metric --json``
  on the host (no sudo).
- **Summary** — ``test_print_results_table`` prints throughput, latency, and accuracy.
- **Teardown** — ``test_teardown`` stops containers even when a prior stage failed.

Logs are written under ``paths.log_dir`` from the config (default ``/home/{user-id}/LOGS/sglang``).
