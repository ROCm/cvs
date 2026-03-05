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

Change the parameters as needed in the InferenceMAX configuration file: ``mi300x_singlenode_inferencemax.json`` for single node inference configurations.

.. note::

  - Parameters with the ``<changeme>`` value must have that value modified to your specifications.
  - ``{user-id}`` will be resolved to the current username in the runtime. You can also manually change this value to your username.

``mi300x_singlenode_inferencemax.json``
========================================

Here's a code snippet of the ``mi300x_singlenode_inferencemax.json`` file for reference:

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
                "percentiles_metrics": "ttft,tpot,itl,e2el",
                "metric_percentiles": "99",
                "server_script": "gptoss_fp4_mi300x_docker.sh",
                "bench_serv_script": "benchmark_serving.py",
                "result_dict": {
                    "output_throughput_per_sec": "4200",
                    "mean_ttft_ms": "500",
                    "mean_tpot_ms": "15"
                }
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
     - https://github.com/InferenceMAX/ |br| InferenceMAX.git
     - Git repository URL for InferenceMAX framework
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
   * - ``container_config.`` |br| ``device_list``
     - Values: |br| - ``"/dev/dri"`` |br| - ``"/dev/kfd"``
     - List of device paths to mount in the container for GPU access
   * - ``container_config.`` |br| ``volume_dict``
     - ``{"/home/{user-id}": "/home/{user-id}"}``
     - Dictionary mapping host paths to container paths for volume mounts
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
   * - ``benchmark_params.`` |br| ``gpt-oss-120b.`` |br| ``percentiles_metrics``
     - ttft,tpot,itl,e2el
     - Comma-separated list of metrics to compute percentiles for (ttft: Time to First Token, tpot: Time Per Output Token, itl: Inter-Token Latency, e2el: End-to-End Latency)
   * - ``benchmark_params.`` |br| ``gpt-oss-120b.`` |br| ``metric_percentiles``
     - 99
     - Percentile values to compute for metrics (e.g., 99 for 99th percentile)
   * - ``benchmark_params.`` |br| ``gpt-oss-120b.server_script``
     - gptoss_fp4_mi300x_docker.sh
     - Script to launch the inference server
   * - ``benchmark_params.`` |br| ``gpt-oss-120b.`` |br| ``bench_serv_script``
     - benchmark_serving.py
     - Script to run the benchmarking client
   * - ``benchmark_params.`` |br| ``gpt-oss-120b.result_dict.`` |br| ``output_throughput_`` |br| ``per_sec``
     - 4200
     - Expected number of output tokens generated per second
   * - ``benchmark_params.`` |br| ``gpt-oss-120b.result_dict.`` |br| ``mean_ttft_ms``
     - 500
     - Expected mean Time to First Token in milliseconds
   * - ``benchmark_params.`` |br| ``gpt-oss-120b.result_dict.`` |br| ``mean_tpot_ms``
     - 15
     - Expected mean Time Per Output Token in milliseconds
