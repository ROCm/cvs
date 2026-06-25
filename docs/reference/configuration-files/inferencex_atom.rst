.. meta::  :description: Configure the variables in the InferenceX ATOM configuration files
  :keywords: inference, ROCm, install, cvs, InferenceX ATOM, vLLM

***************************************
InferenceX ATOM inference configuration file
***************************************

InferenceX ATOM tests validate inference performance for large language models (LLMs) using vLLM backend on AMD GPU clusters. These tests ensure optimal inference throughput, latency, and token generation performance for AI serving workloads.

The InferenceX ATOM tests check:

- **Container orchestration**: Docker setup with ROCm for inference workloads
- **Model serving**: vLLM backend initialization and model loading
- **Performance metrics**: Output throughput, Time to First Token (TTFT), and Time Per Output Token (TPOT)
- **Benchmarking**: Load testing with various concurrency levels and sequence lengths
- **Result verification**: Expected throughput and latency metrics

InferenceX ATOM inputs use flat ``*_config.json`` + sibling ``*_threshold.json`` pairs under ``cvs/input/config_file/inference/inferencex_atom_single/`` (same layout and naming as ``vllm_single``). Filename pattern: ``{gpu}_{framework}_{model}_{precision}[_{mode}]_config.json`` (framework ``inferencex_atom_single`` → ``inferencex-atom-single`` in the stem). Pass ``--config_file`` pointing at the ``*_config.json``; the loader discovers the sole sibling ``*threshold.json`` via :func:`cvs.lib.inference.utils.inferencex_atom_config_loader.load_variant`. For example, MI300X GPT-OSS 120B uses ``mi300x_inferencex-atom-single_gpt-oss-120b_bf16_config.json`` plus ``mi300x_inferencex-atom-single_gpt-oss-120b_bf16_threshold.json``.

**InferenceX ATOM / MI300X note:**

  - Parameters with the ``<changeme>`` value must have that value modified to your specifications.
  - ``{user-id}`` will be resolved to the current username in the runtime. You can also manually change this value to your username.
  - **Server**: ``roles.server.serve_args`` and ``roles.server.env`` drive a Python-built ``vllm serve`` command inside the container (same pattern as ``vllm_single``). MI300-class defaults include ``enforce-eager``, ``block-size``, and ``no-enable-prefix-caching``.
  - **Thresholds**: sibling ``*threshold.json`` next to the config file (exactly one file; multiple files is an error). Loaded by :func:`cvs.lib.inference.utils.inferencex_atom_config_loader.load_variant`. Cell keys use ``ISL=<isl>,OSL=<osl>,TP=<tp>,CONC=<conc>`` with ``client.*`` metric specs (see vLLM threshold examples). ``test_metric`` asserts via :func:`cvs.lib.utils.verdict.evaluate_all` when ``enforce_thresholds`` is true.
  - **Sweep**: ``sweep.sequence_combinations`` (named ISL/OSL pairs) plus ``sweep.runs`` (explicit ``{combo, concurrency}`` list). Model id comes from ``model.id``.

Pytest and HTML layout (inferencex_atom_single)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 10 35 55
   :header-rows: 1

   * - Stage
     - Test
     - Notes
   * - 1
     - ``test_launch_container``
     - Host ``docker_lib`` launch; records ``container_launch``.
   * - 2
     - ``test_setup_sshd``
     - Multinode only; single-node skips sshd probe.
   * - 3
     - ``test_model_fetch``
     - Ensures model bytes are present under ``paths.models_dir``.
   * - 4
     - ``test_inferencex_atom_inference``
     - Parametrized cell; records ``server_ready`` then ``client_complete``.
   * - 5
     - ``test_metric``
     - One HTML row per ``client.*`` metric per cell.
   * - 6
     - ``test_print_results_table``
     - Session results grid from ``inf_res_dict``.
   * - 7
     - ``test_teardown``
     - Explicit teardown; sets ``lifecycle.torn_down`` after verify.

Example variant layout
======================

Each config stem has ``<stem>_config.json`` (``schema_version: 1``) and a sibling ``<stem>_threshold.json``. See ``mi300x_inferencex-atom-single_gpt-oss-120b_bf16_config.json`` for a GPT-OSS 120B reference, or ``mi300x_inferencex-atom-single_deepseek-r1_fp8_perf_config.json`` for W1 DeepSeek R1 FP8.

.. dropdown:: Example ``mi300x_inferencex-atom-single_gpt-oss-120b_bf16_threshold.json`` (excerpt)

  .. code:: json

    {
      "ISL=7168,OSL=1024,TP=8,CONC=64": {
        "client.output_throughput": {"kind": "min_tok_s", "value": 4200},
        "client.mean_ttft_ms": {"kind": "max_ms", "value": 500},
        "client.mean_tpot_ms": {"kind": "max_ms", "value": 15}
      }
    }

  Every member of ``GATED_METRICS`` needs a spec in each cell (see the shipped file for the full set). Set ``enforce_thresholds: false`` in ``*_config.json`` until numbers are calibrated.

Parameters
==========

Top-level blocks match the vLLM single-node schema (see ``plans/building-a-cvs-test-suite.md``). InferenceX ATOM-specific notes:

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Block / key
     - Example
     - Description
   * - ``framework``
     - ``inferencex_atom_single``
     - Suite identifier passed to the loader.
   * - ``enforce_thresholds``
     - ``false``
     - When false, ``test_metric`` records ``client.*`` values without asserting. Flip to ``true`` after calibrating ``*_threshold.json``.
   * - ``paths.*``
     - ``shared_fs``, ``models_dir``, ``log_dir``, ``hf_token_file``
     - Placeholder-substituted paths. ``models_dir`` is the HF cache pin for serve and fetch.
   * - ``model.id``
     - ``openai/gpt-oss-120b``
     - HuggingFace model id passed to ``vllm serve`` and ``vllm bench serve``.
   * - ``container.image`` / ``container.name``
     - ``<changeme>``
     - Docker image and container name (set per environment).
   * - ``container.runtime.args``
     - ``shm_size``, ``volumes``, ``devices``
     - Bind **only** ``/home/{user-id}:/home/{user-id}`` in ``volumes``; the orchestrator also mounts ``/home/<user>:/workspace``.
   * - ``roles.server.serve_args``
     - ``enforce-eager``, ``gpu-memory-utilization``, ``block-size``
     - Extra ``vllm serve`` flags (Python-built; no host ``.sh`` staging).
   * - ``roles.server.env``
     - ``VLLM_ROCM_USE_AITER``, etc.
     - Container env merged into ``/tmp/server_env_script.sh`` before launch.
   * - ``params.tensor_parallelism``
     - ``8``
     - Tensor-parallel size for serve and bench; appears in threshold cell keys as ``TP``.
   * - ``params.max_model_length``
     - ``8192``
     - Passed to ``vllm serve --max-model-len``. ISL + OSL (with ``random_range_ratio``) must fit.
   * - ``params.num_prompts``
     - ``1000``
     - Prompt count for ``vllm bench serve``.
   * - ``params.client_poll_count`` / ``client_poll_wait_time``
     - ``50`` / ``60``
     - Client completion poll budget.
   * - ``params.bench_max_failed_requests``
     - ``0``
     - After the bench summary, fail when ``Failed requests`` exceeds this cap.
   * - ``sweep.sequence_combinations``
     - named ``isl`` / ``osl`` pairs
     - Named combos referenced by ``sweep.runs``.
   * - ``sweep.runs``
     - ``{combo, concurrency}``
     - Explicit list of cells to run (not a cartesian product).

Legacy monolithic JSON (``config`` + ``benchmark_params``) is no longer supported by ``inferencex_atom_single``.
