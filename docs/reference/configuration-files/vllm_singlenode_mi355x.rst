.. meta::
  :description: Configure the variables in the MI355X Single-Node vLLM configuration file
  :keywords: inference, ROCm, install, cvs, vLLM, MI355X, LLM, single-node

*************************************************
MI355X single-node vLLM inference configuration file
*************************************************

MI355X single-node vLLM tests validate LLM inference performance using vLLM on AMD MI355X GPU systems. These tests ensure optimal throughput, latency, and scalability for large language model serving workloads on single-node configurations.

The MI355X vLLM tests check:

- **Container orchestration**: Docker setup with vLLM for single-node inference
- **Model serving**: LLM deployment with PagedAttention and continuous batching
- **Performance metrics**: Throughput, TTFT, TPOT, ITL, and E2EL
- **Multiple models**: GPT-OSS-120B, Qwen3-235B, Qwen3-80B, DeepSeek-V3.1
- **Workload scenarios**: Balanced, long generation, and long context
- **Result verification**: Expected throughput and latency metrics

Change the parameters as needed in the MI355X vLLM configuration file: ``mi355x_singlenode_vllm.json`` for single-node LLM serving.

.. note::

  - ``{user-id}`` will be resolved to the current username in the runtime. You can also manually change this value to your username.

``mi355x_singlenode_vllm.json``
================================

Here's a code snippet of the ``mi355x_singlenode_vllm.json`` file for reference:

.. dropdown:: ``mi355x_singlenode_vllm.json``

  .. code:: json

    {
        "config": {
            "container_image": "rocm/7.0:rocm7.0_ubuntu_22.04_vllm_0.10.1_instinct_20250927_rc1",
            "container_name": "vllm_inference_rocm",
            "nnodes": "1",
            "benchmark_server_script_path": "/home/{user-id}/benchmark_server_scripts/",
            "benchmark_script_repo": "https://github.com/kimbochen/bench_serving.git",
            "hf_token_file": "/home/{user-id}/.hf_token",
            "shm_size": "16G",
            "log_dir": "/home/{user-id}/LOGS",
            "data_cache_dir": "/it-share/models/",
            "container_config": {
                "device_list": [
                    "/dev/dri",
                    "/dev/kfd",
                    "/dev/mem"
                ],
                "volume_dict": {
                    "/home/{user-id}": "/home/{user-id}",
                    "/it-share/models/": "/models"
                },
                "env_dict": {
                    "HF_HUB_CACHE": "/models/huggingface-cache"
                }
            }
        },
        "benchmark_params": {
            "gpt-oss-120b": {
                "container_image": "rocm/7.0:rocm7.0_ubuntu_22.04_vllm_0.10.1_instinct_20250927_rc1",
                "backend": "vllm",
                "base_url": "http://0.0.0.0",
                "port_no": "8888",
                "dataset_name": "random",
                "concurrency_levels": [16, 32, 64],
                "model": "openai/gpt-oss-120b",
                "num_prompts": "3200",
                "sequence_combinations": [
                    {"isl": "1024", "osl": "1024", "name": "balanced"},
                    {"isl": "1024", "osl": "8192", "name": "long_generation"},
                    {"isl": "8192", "osl": "1024", "name": "long_context"}
                ],
                "burstiness": "1.0",
                "seed": "0",
                "request_rate": "inf",
                "max_model_length": "9216",
                "random_range_ratio": "0.8",
                "tensor_parallelism": "1",
                "tokenizer_mode": "auto",
                "percentile_metrics": "ttft,tpot,itl,e2el",
                "metric_percentiles": "99",
                "result_dict": {
                    "ISL=1024,OSL=1024,TP=1,CONC=16": {
                        "total_throughput_per_sec": "4651",
                        "mean_ttft_ms": "70",
                        "mean_tpot_ms": "8"
                    }
                }
            }
        }
    }

Parameters
==========

Use the parameters in this table to configure the MI355X vLLM configuration file.

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
     - Docker container image with vLLM for MI355X GPUs
   * - ``container_name``
     - vllm_inference_rocm
     - Name of the Docker container instance
   * - ``nnodes``
     - 1
     - Number of nodes (single-node configuration)
   * - ``benchmark_server_`` |br| ``script_path``
     - ``/home/{user-id}/`` |br| ``benchmark_server_scripts/``
     - Path to benchmark server scripts
   * - ``benchmark_script_repo``
     - https://github.com/kimbochen/ |br| bench_serving.git
     - GitHub repository for benchmark scripts
   * - ``hf_token_file``
     - ``/home/{user-id}/`` |br| ``.hf_token``
     - Path to HuggingFace authentication token file
   * - ``shm_size``
     - 16G
     - Shared memory size for the container
   * - ``log_dir``
     - ``/home/{user-id}/LOGS``
     - Directory for vLLM logs
   * - ``data_cache_dir``
     - /it-share/models/
     - Directory for model cache
   * - ``container_config.`` |br| ``device_list``
     - Values: |br| - ``"/dev/dri"`` |br| - ``"/dev/kfd"`` |br| - ``"/dev/mem"``
     - List of device paths to mount in the container for GPU access
   * - ``container_config.`` |br| ``volume_dict``
     - ``{"/home/{user-id}":`` |br| ``"/home/{user-id}",`` |br| ``"/it-share/models/": "/models"}``
     - Dictionary mapping host paths to container paths for volume mounts
   * - ``container_config.`` |br| ``env_dict.HF_HUB_CACHE``
     - /models/huggingface-cache
     - HuggingFace model cache directory
   * - ``benchmark_params.`` |br| ``<model>.container_image``
     - Model-specific container image
     - Container image for specific model benchmarks (overrides global container_image if set)
   * - ``benchmark_params.`` |br| ``<model>.backend``
     - vllm
     - Inference backend to use (vLLM)
   * - ``benchmark_params.`` |br| ``<model>.base_url``
     - http://0.0.0.0
     - Base URL for the vLLM server
   * - ``benchmark_params.`` |br| ``<model>.port_no``
     - 8888
     - Port number for the vLLM server
   * - ``benchmark_params.`` |br| ``<model>.dataset_name``
     - random
     - Dataset type for benchmarking (sharegpt, hf, random, sonnet, burstgpt)
   * - ``benchmark_params.`` |br| ``<model>.`` |br| ``concurrency_levels``
     - [16, 32, 64]
     - List of concurrent request levels to test
   * - ``benchmark_params.`` |br| ``<model>.model``
     - Model identifier
     - HuggingFace model identifier or local path (e.g., openai/gpt-oss-120b, Qwen/Qwen3-235B-A22B-Instruct-2507, Qwen/Qwen3-Next-80B-A3B-Instruct, deepseek-ai/DeepSeek-V3.1)
   * - ``benchmark_params.`` |br| ``<model>.num_prompts``
     - 3200
     - Total number of prompts to send during benchmarking
   * - ``benchmark_params.`` |br| ``<model>.`` |br| ``sequence_combinations``
     - Three scenarios
     - List of input/output sequence length combinations with scenario names: balanced (ISL=1024, OSL=1024), long_generation (ISL=1024, OSL=8192), long_context (ISL=8192, OSL=1024)
   * - ``benchmark_params.`` |br| ``<model>.burstiness``
     - 1.0
     - Request burstiness factor (1.0 = uniform distribution, higher values create more bursty traffic)
   * - ``benchmark_params.`` |br| ``<model>.seed``
     - 0
     - Random seed for reproducible benchmark results
   * - ``benchmark_params.`` |br| ``<model>.request_rate``
     - inf
     - Maximum request rate (inf = unlimited, or specify QPS)
   * - ``benchmark_params.`` |br| ``<model>.`` |br| ``max_model_length``
     - 9216
     - Maximum total sequence length the model can handle
   * - ``benchmark_params.`` |br| ``<model>.`` |br| ``random_range_ratio``
     - 0.8
     - Ratio for randomizing input/output lengths around specified values
   * - ``benchmark_params.`` |br| ``<model>.`` |br| ``random_prefix_len``
     - 0
     - Length of random prefix for shared prefix testing
   * - ``benchmark_params.`` |br| ``<model>.`` |br| ``tensor_parallelism``
     - Varies per model
     - Number of GPUs to use for tensor parallelism (1 for GPT-OSS-120B and Qwen3-80B, 8 for Qwen3-235B and DeepSeek-V3.1)
   * - ``benchmark_params.`` |br| ``<model>.tokenizer_mode``
     - auto
     - Tokenizer mode (auto, slow, mistral, custom)
   * - ``benchmark_params.`` |br| ``<model>.`` |br| ``percentile_metrics``
     - ttft,tpot,itl,e2el
     - Comma-separated list of metrics to compute percentiles for (ttft: Time to First Token, tpot: Time Per Output Token, itl: Inter-Token Latency, e2el: End-to-End Latency)
   * - ``benchmark_params.`` |br| ``<model>.`` |br| ``metric_percentiles``
     - 99
     - Percentile values to compute for metrics (e.g., 99 for 99th percentile)
   * - ``benchmark_params.`` |br| ``<model>.server_script``
     - Model-specific script
     - Shell script to launch vLLM server with model-specific configuration
   * - ``benchmark_params.`` |br| ``<model>.`` |br| ``bench_serv_script``
     - benchmark_serving.py
     - Python script to run the benchmarking client
   * - ``benchmark_params.`` |br| ``<model>.result_dict``
     - Model-specific baselines
     - Dictionary of expected performance results for each workload scenario, specifying total_throughput_per_sec, mean_ttft_ms, and mean_tpot_ms for different combinations of ISL (Input Sequence Length), OSL (Output Sequence Length), TP (Tensor Parallelism), and CONC (Concurrency)
