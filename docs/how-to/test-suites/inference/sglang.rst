.. meta::
  :description: Run SGLang inference benchmarks with CVS on MI30X clusters
  :keywords: CVS, SGLang, inference, benchmark, distributed, disaggregated, LLM, ROCm

*************************
Run SGLang inference tests
*************************

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
     - Independent full-model ``sglang.launch_server`` on every ``cluster.json`` host (local TP).
   * - ``sglang_distributed``
     - ``sglang_distributed.py``
     - One unified multi-node server (TP/PP across the first ``nnodes`` hosts in ``cluster.json``).
   * - ``sglang_disagg_distributed``
     - ``sglang_disagg_distributed.py``
     - Disaggregated prefill/decode (even ``nnodes`` from ``cluster.json``; rank-0 is proxy/benchmark).

.. _sglang-set-up-config:

Set up config
=============

1. List available SGLang templates:

   .. code:: bash

     cvs config list inference/sglang

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
   ``sglang_single`` uses every host in ``cluster.json`` (full model on each node) and
   defaults the HTTP port to ``8000``; do not set ``benchmark_serv_node`` or
   ``proxy_router_serv_port``. ``sglang_distributed`` takes the first ``nnodes`` hosts
   from ``cluster.json``; rank-0 is master and benchmark. Dist-init defaults to ``40001``.
   ``sglang_disagg_distributed`` requires even ``nnodes`` (>= 2); rank-0 is prefill
   coordinator, proxy, and benchmark, rank-1 is decode coordinator, and remaining hosts
   split equally (1P/1D, 2P/2D, 3P/3D). Ports default; do not pin node lists or PD ports.

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

  Shipped templates are a single workload per file. ``sweep.runs`` selects which
  ``ISL,OSL,TP,PP,CONC`` cells to parametrize; empty ``runs`` uses every performance
  cell in the threshold JSON.

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
    - test_openai_compatible_http_endpoints
    - test_run_lm_eval_hellaswag_benchmark_test
    - test_run_lm_eval_gsm8k_benchmark_test
    - test_run_performance_benchmark_test
    - test_verify_dmesg_after_benchmark
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
    - test_run_long_context_accuracy
    - test_run_lm_eval_hellaswag_benchmark_test
    - test_run_lm_eval_gsm8k_benchmark_test
    - test_run_performance_benchmark_test
    - test_verify_dmesg_after_benchmark
    - test_disagg_gpu_topology
    - test_print_results_table
    - test_teardown

``test_run_long_context_accuracy`` is parametrized from ``ACC_ISL=…,OSL=…`` cells in the
threshold JSON. It runs only when ``server_params.lng_ctx_activate`` is ``true``
(DeepSeek disaggregated template). Llama disaggregated skips this stage.

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
  selected sweep cell. Set top-level ``enforce_thresholds: false`` to record metrics
  without failing on uncalibrated gates.
- **Accuracy** — ``test_run_lm_eval_hellaswag_benchmark_test`` and
  ``test_run_lm_eval_gsm8k_benchmark_test`` run lm-eval tasks configured in ``accuracy.tasks``.
  Disaggregated runs also include ``test_run_long_context_accuracy`` (needle-in-a-haystack)
  when ``lng_ctx_activate`` is ``true``; cells come from ``ACC_ISL=…`` keys in the
  threshold file.
- **Summary** — ``test_print_results_table`` prints throughput/latency/accuracy in the console and
  report.
- **Teardown** — ``test_teardown`` stops containers even when a prior stage failed.

Logs are written under ``paths.log_dir`` from the config (default ``/home/{user-id}/LOGS/sglang``).
