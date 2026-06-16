.. meta::
  :description: Configure the variables in the InferenceMAX configuration files
  :keywords: inference, ROCm, install, cvs, InferenceMAX, vLLM

***************************************
InferenceMAX inference configuration file
***************************************

InferenceMAX tests validate inference performance for large language models (LLMs) using vLLM backend on AMD GPU clusters. These tests ensure optimal inference throughput, latency, and token generation performance for AI serving workloads.

The InferenceMAX tests check:

- **Container orchestration**: Docker setup with ROCm for inference workloads
- **Model serving**: vLLM backend initialization and model loading
- **Performance metrics**: Output throughput, Time to First Token (TTFT), and Time Per Output Token (TPOT)
- **Benchmarking**: Load testing with various concurrency levels and sequence lengths
- **Result verification**: Expected throughput and latency metrics

InferenceMAX inputs use per-variant directories under ``cvs/input/config_file/inference/inferencemax_single/<variant>/`` with a suite JSON passed as ``--config_file`` (typically ``<variant>_config.json``). Optional pass/fail numbers live in the **sole** ``*threshold.json`` in that directory when present (same ``glob`` rule as ``load_variant`` in ``cvs.lib.dtni.config_loader``; merged by ``load_inferencemax_suite_raw``). For example, MI300X GPT-OSS 120B single-node uses ``mi300x_gpt_oss_120b_single/mi300x_gpt_oss_120b_single_config.json`` plus ``mi300x_gpt_oss_120b_single_threshold.json``. A preserved MI355x reference lives at ``mi355x_inferencemax_gpt_oss_120b_single/`` with the same naming pattern; verify ``server_script`` against your InferenceX revision.

.. note::

  - Parameters with the ``<changeme>`` value must have that value modified to your specifications.
  - ``{user-id}`` will be resolved to the current username in the runtime. You can also manually change this value to your username.
  - ``server_script`` is interpreted relative to ``benchmarks/single_node/`` (or ``benchmarks/multi_node/`` when multi-node) inside the cloned ``inferencemax_repo``. It must exist at that path in the repo revision you use; upstream layouts change. On current ``SemiAnalysisAI/InferenceX`` ``main``, MI300X GPT-OSS-style server entrypoints often live under ``fixed_seq_len/`` (for example ``fixed_seq_len/gptoss_fp4_mi300x.sh``). CVS ships a **flat** ``benchmark_server_scripts/`` tree (``gptoss_fp4_mi300x.sh`` at the top level); when ``use_host_mounted_server_script`` is set, ``cvs.lib.inference.inferencemax_orch.InferenceMaxJob`` maps ``fixed_seq_len/`` in ``server_script`` to that layout. If the path is wrong, the server log shows ``No such file or directory``.
  - InferenceX single-node scripts typically write under ``/workspace`` (e.g. ``server.log``, ``gpu_metrics.csv``). CVS creates ``/workspace`` inside the container before launching the server; add a volume mapping for ``/workspace`` in ``container_config.volume_dict`` only if you need those files on the host after the run. For vLLM 0.21+ nightlies, upstream ``benchmarks/single_node/fixed_seq_len/gptoss_fp4_mi300x.sh`` invokes ``vllm serve`` **without** ``--enforce-eager``, so graph capture can still run even when ``vllm_enforce_eager`` adds ``VLLM_ENFORCE_EAGER=1`` to the env; the shipped MI300X suite defaults ``use_host_mounted_server_script`` so the CVS wrapper passes ``--enforce-eager`` on the CLI.
  - Host-mount server wrappers for GPT-OSS MI300X ship under ``cvs/input/config_file/inference/inferencemax_single/mi300x_gpt_oss_120b_single/benchmark_server_scripts/``. The shipped ``mi300x_gpt_oss_120b_single_config.json`` sets ``benchmark_server_script_path`` to ``auto`` so ``InferenceMaxJob`` resolves that directory next to the installed ``cvs`` package (works for editable installs and ``site-packages`` as long as the run host mounts the parent of that tree, e.g. full ``/home/{user-id}``). Set an explicit absolute path, or rsync the directory to a stable location and point ``benchmark_server_script_path`` there, if auto-resolution fails.
  - **Thresholds**: optional ``*threshold.json`` in the variant directory (same discovery as ``load_variant`` in ``cvs.lib.dtni.config_loader``: exactly one ``*threshold.json`` if present; multiple files is an error). Merged by ``load_inferencemax_suite_raw``. Keys must match :meth:`~cvs.lib.inference.base.InferenceBaseJob.verify_inference_results` — ``ISL=<isl>,OSL=<osl>,TP=<tp>,CONC=<conc>`` (string values as in JSON). You can instead keep ``result_dict`` inline in the suite JSON only when no threshold file is used.
  - **Which model block runs**: if ``benchmark_params`` has a single top-level model key (for example ``gpt-oss-120b``), that block is used automatically. If you add more than one model object under ``benchmark_params``, set a top-level string ``benchmark_model`` to the key you want ``inferencemax_single`` to run (validated by ``cvs.lib.dtni.config_loader.inferencemax_benchmark_model_name``).

Pytest and HTML layout (inferencemax_single)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
     - ``test_inferencemax_inference``
     - Parametrized cell; records ``server_ready`` then ``client_complete``.
   * - 3
     - ``test_print_results_table``
     - Session results grid from ``inf_res_dict``.
   * - 4
     - ``test_teardown``
     - Explicit teardown; sets ``lifecycle.torn_down`` after verify.

``mi300x_singlenode_inferencemax.json``
========================================

Legacy monolithic reference (historical). Current variants split **run** parameters into ``*_config.json`` and **expected metrics** into ``*_threshold.json`` beside it.

.. dropdown:: Example ``mi300x_gpt_oss_120b_single_threshold.json``

  .. code:: json

    {
        "_comment": "Pass/fail expectations; keys must match verify_inference_results.",
        "result_dict": {
            "ISL=7168,OSL=1024,TP=8,CONC=64": {
                "output_throughput_per_sec": "4200",
                "mean_ttft_ms": "500",
                "mean_tpot_ms": "15"
            }
        }
    }

.. dropdown:: ``mi300x_singlenode_inferencemax.json``

  .. code:: json

    {
        "config": {
            "container_image": "rocm/7.0:rocm7.0_ubuntu_22.04_vllm_0.10.1_instinct_20250927_rc1",
            "container_name": "inference_max_rocm",
            "_example_nnodes": "4",
            "nnodes": "4",
            "inferencemax_repo": "https://github.com/InferenceMAX/InferenceMAX.git",
            "benchmark_script_repo": "https://github.com/kimbochen/bench_serving.git",
            "hf_token_file": "/home/{user-id}/.hf_token",
            "shm_size": "128G",
            "log_dir": "/home/{user-id}/LOGS",
            "container_config": {
                "device_list": [
                    "/dev/dri",
                    "/dev/kfd"
                ],
                "volume_dict": {
                    "/home/{user-id}": "/home/{user-id}"
                },
                "env_dict": {}
            }
        },
        "benchmark_params": {
            "gpt-oss-120b": {
                "backend": "vllm",
                "base_url": "http://0.0.0.0",
                "port_no": "8000",
                "_example_dataset_name": "sharegpt|hf|random|sonnet|burstgpt",
                "dataset_name": "random",
                "max_concurrency": "64",
                "model": "openai/gpt-oss-120b",
                "num_prompts": "1000",
                "input_sequence_length": "8192",
                "output_sequence_length": "1024",
                "burstiness": "1.0",
                "seed": "0",
                "max_model_length": "9216",
                "random_range_ratio": "0.8",
                "random_prefix_len": "0",
                "tensor_parallelism": "8",
                "_example_tokenizer_mode": "auto|slow|mistral|custom",
                "tokenizer_mode": "auto",
                "percentile_metrics": "ttft,tpot,itl,e2el",
                "metric_percentiles": "99",
                "server_script": "fixed_seq_len/gptoss_fp4_mi300x.sh",
                "bench_serv_script": "benchmark_serving.py"
            }
        }
    }

Parameters
==========

Use the parameters in this table to configure the InferenceMAX configuration file.

.. |br| raw:: html

    <br />

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameters
     - Default values
     - Description
   * - ``container_image``
     - rocm/7.0:rocm7.0_ubuntu_22.04_ |br| vllm_0.10.1_instinct_20250927_rc1
     - Docker container image with ROCm and vLLM for inference
   * - ``container_name``
     - inference_max_rocm
     - Name of the Docker container instance
   * - ``nnodes``
     - 4
     - Number of nodes in the cluster
   * - ``inferencemax_repo``
     - https://github.com/ |br| SemiAnalysisAI/InferenceX.git
     - Git repository URL for the InferenceX tree CVS clones into ``/app`` inside the container (the legacy ``InferenceMAX/InferenceMAX`` stub only redirects here; override in JSON if you use a fork or pin a tag)
   * - ``benchmark_script_repo``
     - https://github.com/kimbochen/ |br| bench_serving.git
     - Git repository URL for benchmarking scripts
   * - ``hf_token_file``
     - ``/home/{user-id}/`` |br| ``.hf_token``
     - Path to HuggingFace authentication token file for model access
   * - ``shm_size``
     - 128G
     - Shared memory size allocated to the container
   * - ``log_dir``
     - ``/home/{user-id}/LOGS``
     - Directory where inference logs are stored
   * - ``vllm_enforce_eager``
     - (omit / false)
     - When true, CVS adds ``export VLLM_ENFORCE_EAGER=1`` to ``/tmp/server_env_script.sh`` before the InferenceX server script runs, which typically disables CUDA graph capture and avoids many ROCm + vLLM nightly failures during engine startup (may reduce throughput).
   * - ``container_config.`` |br| ``device_list``
     - Values: |br| - ``"/dev/dri"`` |br| - ``"/dev/kfd"``
     - List of device paths to mount in the container for GPU access
   * - ``container_config.`` |br| ``volume_dict``
     - ``{"/home/{user-id}": "/home/{user-id}"}`` plus optional |br| ``"/home/{user-id}/LOGS/inference-max/workspace": "/workspace"``
     - Host to container volume mounts. Map ``{log_dir}/inference-max/workspace`` to ``/workspace`` so InferenceX vLLM artifacts (for example ``server.log``, ``gpu_metrics.csv``) stay on the host after teardown.
   * - ``/home/{user-id}``
     - ``/home/{user-id}``
     - User home directory mount
   * - ``container_config.`` |br| ``env_dict``
     - Empty
     - Dictionary of environment variables to set in the container
   * - ``benchmark_params.`` |br| ``gpt-oss-120b.backend``
     - vllm
     - Inference backend to use (vLLM)
   * - ``benchmark_params.`` |br| ``gpt-oss-120b.base_url``
     - http://0.0.0.0
     - Base URL for the inference server
   * - ``benchmark_params.`` |br| ``gpt-oss-120b.port_no``
     - 8000
     - Port number for the inference server
   * - ``benchmark_params.`` |br| ``gpt-oss-120b.`` |br| ``dataset_name``
     - random
     - Dataset type for benchmarking (sharegpt, hf, random, sonnet, burstgpt)
   * - ``benchmark_params.`` |br| ``gpt-oss-120b.`` |br| ``max_concurrency``
     - 64
     - Maximum number of concurrent requests during benchmarking
   * - ``benchmark_params.`` |br| ``gpt-oss-120b.model``
     - openai/gpt-oss-120b
     - HuggingFace model identifier or path
   * - ``benchmark_params.`` |br| ``gpt-oss-120b.`` |br| ``num_prompts``
     - 1000
     - Total number of prompts to send during the benchmark
   * - ``benchmark_params.`` |br| ``gpt-oss-120b.`` |br| ``input_sequence_length``
     - 8192
     - Length of input sequences in tokens
   * - ``benchmark_params.`` |br| ``gpt-oss-120b.`` |br| ``output_sequence_`` |br| ``length``
     - 1024
     - Expected length of output sequences in tokens
   * - ``benchmark_params.`` |br| ``gpt-oss-120b.burstiness``
     - 1.0
     - Request burstiness factor (1.0 = uniform distribution)
   * - ``benchmark_params.`` |br| ``gpt-oss-120b.seed``
     - 0
     - Random seed for reproducible benchmark results
   * - ``benchmark_params.`` |br| ``gpt-oss-120b.`` |br| ``max_model_length``
     - 9216
     - Maximum total sequence length the model can handle
   * - ``benchmark_params.`` |br| ``gpt-oss-120b.`` |br| ``random_range_ratio``
     - 0.8
     - Range ratio for random dataset generation
   * - ``benchmark_params.`` |br| ``gpt-oss-120b.`` |br| ``random_prefix_len``
     - 0
     - Prefix length for random dataset generation
   * - ``benchmark_params.`` |br| ``gpt-oss-120b.`` |br| ``tensor_parallelism``
     - 8
     - Number of GPUs to use for tensor parallelism
   * - ``benchmark_params.`` |br| ``gpt-oss-120b.`` |br| ``tokenizer_mode``
     - auto
     - Tokenizer mode (auto, slow, mistral, custom)
   * - ``benchmark_params.`` |br| ``gpt-oss-120b.`` |br| ``percentile_metrics``
     - ttft,tpot,itl,e2el
     - Comma-separated list of metrics to compute percentiles for (ttft: Time to First Token, tpot: Time Per Output Token, itl: Inter-Token Latency, e2el: End-to-End Latency)
   * - ``benchmark_params.`` |br| ``gpt-oss-120b.`` |br| ``metric_percentiles``
     - 99
     - Percentile values to compute for metrics (e.g., 99 for 99th percentile)
   * - ``benchmark_params.`` |br| ``gpt-oss-120b.server_script``
     - fixed_seq_len/ |br| gptoss_fp4_mi300x.sh
     - Path under ``benchmarks/<single_node|multi_node>/`` to the shell script inside the clone (must exist in ``inferencemax_repo``). With ``use_host_mounted_server_script``, CVS resolves ``fixed_seq_len/`` to the flat scripts under ``benchmark_server_script_path``.
   * - ``benchmark_params.`` |br| ``gpt-oss-120b.`` |br| ``bench_serv_script``
     - benchmark_serving.py
     - Script to run the benchmarking client
   * - ``benchmark_params.`` |br| ``gpt-oss-120b.result_dict.`` |br| ``output_throughput_`` |br| ``per_sec``
     - 4200
     - Under each ``ISL=...,OSL=...,TP=...,CONC=...`` key (see ``verify_inference_results``): expected output tok/s (higher is better). Usually supplied from ``*_threshold.json`` merged at pytest load time.
   * - ``benchmark_params.`` |br| ``gpt-oss-120b.result_dict.`` |br| ``mean_ttft_ms``
     - 500
     - Same keyed map: expected mean TTFT in ms (lower is better).
   * - ``benchmark_params.`` |br| ``gpt-oss-120b.result_dict.`` |br| ``mean_tpot_ms``
     - 15
     - Same keyed map: expected mean TPOT in ms (lower is better).
