.. meta::
  :description: Configure SGLang inference benchmarks on AMD MI30X clusters
  :keywords: inference, ROCm, cvs, SGLang, LLM, MI30X, distributed, disaggregated, prefill, decode

**********************************
SGLang inference configuration
**********************************

CVS ships three SGLang inference suites for AMD MI30X clusters. Each suite reads a JSON
configuration file from ``cvs/input/config_file/inference/sglang/`` and a matching
threshold file referenced by ``benchmark_params.<variant>.threshold_file``.

.. list-table::
   :widths: 2 3 5
   :header-rows: 1

   * - CVS suite
     - Test module
     - Topology
   * - ``sglang_single``
     - ``cvs/tests/inference/sglang/sglang_single.py``
     - One unified ``sglang.launch_server`` on a single ``benchmark_serv_node`` (TP across local GPUs).
   * - ``sglang_distributed``
     - ``cvs/tests/inference/sglang/sglang_distributed.py``
     - One unified multi-node server (TP/PP + ``nnodes``); all ``server_node_list`` ranks participate.
   * - ``sglang_disagg_distributed``
     - ``cvs/tests/inference/sglang/sglang_disagg_distributed.py``
     - Disaggregated prefill/decode with a proxy router; separate prefill and decode node groups.

Run any suite with:

.. code:: bash

  cvs run <suite> \
    --cluster_file cvs/input/cluster_file/<cluster>.json \
    --config_file cvs/input/config_file/inference/sglang/<config>.json \
    --html=~/cvs_results/sglang.html

Copy a template locally with ``cvs copy-config`` (see ``cvs copy-config --list`` for names).

.. note::

  - ``{user-id}`` in path strings is resolved to the current username at runtime.
  - Replace every ``<changeme>`` placeholder before running; unresolved placeholders cause a hard exit at startup.
  - When ``benchmark_params`` contains more than one variant key, set ``SGLANG_BENCHMARK_KEY`` (for example ``llama-70b`` or ``deepseek-r1``) to select which block to run.

Configuration files
===================

All SGLang templates live under ``cvs/input/config_file/inference/sglang/``.

Model and topology templates
----------------------------

.. list-table::
   :widths: 3 2 2
   :header-rows: 1

   * - Config file
     - Model variant key
     - Use with suite
   * - ``mi3xx_sglang_llama_70b_single.json``
     - ``llama-70b``
     - ``sglang_single``
   * - ``mi3xx_sglang_deepseek_r1_0528_single.json``
     - ``deepseek-r1``
     - ``sglang_single``
   * - ``mi3xx_sglang_llama_70b_distributed.json``
     - ``llama-70b``
     - ``sglang_distributed``
   * - ``mi3xx_sglang_deepseek_r1_0528_distributed.json``
     - ``deepseek-r1``
     - ``sglang_distributed``
   * - ``mi3xx_sglang_llama_70b_disaggregated.json``
     - ``llama-70b``
     - ``sglang_disagg_distributed``
   * - ``mi3xx_sglang_deepseek_r1_0528_disaggregated.json``
     - ``deepseek-r1``
     - ``sglang_disagg_distributed``

Threshold files
---------------

Performance cells and pass/fail limits are stored separately. Each ``benchmark_params`` block
points at its threshold file via ``threshold_file``.

.. list-table::
   :widths: 3 5
   :header-rows: 1

   * - Threshold file
     - Referenced by
   * - ``mi3xx_sglang_llama_70b_threshold.json``
     - Llama 3.1 70B configs (single, distributed, disaggregated)
   * - ``mi3xx_sglang_deepseek_r1_0528_threshold.json``
     - DeepSeek-R1-0528 configs (single, distributed, disaggregated)
   * - ``mi3xx_sglang_gpt_oss_120b_threshold.json``
     - Custom / future GPT-OSS 120B runs (point ``threshold_file`` at this path)
   * - ``mi3xx_sglang_glm_52_fp8_threshold.json``
     - Custom / future GLM 5.2 FP8 runs
   * - ``mi3xx_sglang_kimi_k26_threshold.json``
     - Custom / future Kimi K2.6 runs

Threshold keys use the form ``ISL=<n>,OSL=<n>,TP=<n>,PP=<n>,CONC=<n>``. Each value is a
metric map (for example ``output_throughput_per_sec``, ``mean_ttft_ms``, ``mean_tpot_ms``,
``goodput``, ``mfu``) with ``kind`` and ``value`` fields.

File structure
==============

Every template has two top-level keys:

.. list-table::
   :widths: 2 6
   :header-rows: 1

   * - Key
     - Description
   * - ``config``
     - Cluster topology, container image, networking, and ``container_config`` (devices/volumes).
   * - ``benchmark_params``
     - One or more named model variants (``llama-70b``, ``deepseek-r1``). Each variant holds model
       settings, ``threshold_file``, and ``inference_tests``.

The variant key (for example ``llama-70b``) is an internal label; the HuggingFace or filesystem
path loaded by SGLang is ``benchmark_params.<key>.model``.

Example: single-node template
=============================

.. dropdown:: ``mi3xx_sglang_llama_70b_single.json`` (abbreviated)

  .. code:: json

    {
        "config": {
            "container_image": "rocm/sgl-dev:v0.5.12.post1-rocm720-mi30x-20260603",
            "container_name": "sglang_container",
            "nnodes": "1",
            "hf_token_file": "/home/{user-id}/.hf_token",
            "shm_size": "128G",
            "log_dir": "/home/{user-id}/LOGS/sglang",
            "log_level": "info",
            "nccl_debug": "ERROR",
            "benchmark_serv_node": "<changeme>",
            "proxy_router_serv_port": "8000",
            "container_config": {
                "device_list": [ "/dev/dri", "/dev/kfd" ],
                "volume_dict": {
                    "/home/{user-id}": "/home/{user-id}",
                    "/mnt/dtni/models": "/root/models"
                },
                "env_dict": {}
            }
        },
        "benchmark_params": {
            "llama-70b": {
                "backend": "sglang",
                "threshold_file": "cvs/input/config_file/inference/sglang/mi3xx_sglang_llama_70b_threshold.json",
                "max_concurrency": "256",
                "model": "meta-llama/Llama-3.1-70B-Instruct",
                "tensor_parallelism": "8",
                "pipeline_parallelism": "1",
                "memory_fraction": "0.85",
                "inference_tests": {
                    "bench_serv_random": {
                        "backend": "sglang",
                        "enforce_thresholds": false,
                        "data_set_name": "random",
                        "num_prompts": "25",
                        "random_range_ratio": "0.5",
                        "model_num_params": "70000000000",
                        "peak_gpu_tflops": "2615"
                    },
                    "lm_eval_hellaswag": { "...": "..." },
                    "lm_eval_gsm8k": { "...": "..." }
                }
            }
        }
    }

General ``config`` parameters
=============================

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Parameter
     - Example
     - Description
   * - ``container_image``
     - ``rocm/sgl-dev:…``
     - Docker image with SGLang and ROCm for MI30X.
   * - ``container_name``
     - ``sglang_container``
     - Container instance name on each participating node.
   * - ``nnodes``
     - ``1``, ``2``, ``4``, …
     - Server rank count (must match node lists for multi-node suites).
   * - ``hf_token_file``
     - ``/home/{user-id}/.hf_token``
     - HuggingFace token file for model download.
   * - ``shm_size``
     - ``128G``
     - Docker shared memory size.
   * - ``log_dir``
     - ``/home/{user-id}/LOGS/sglang``
     - Shared log root (must be visible from benchmark nodes).
   * - ``log_level``
     - ``info``
     - SGLang server log level.
   * - ``nccl_debug``
     - ``ERROR``
     - NCCL log level (multi-node only).
   * - ``benchmark_serv_node``
     - node hostname/IP
     - Node that runs smoke tests, lm-eval, and ``bench_serving`` (required for all suites).
   * - ``proxy_router_serv_port``
     - ``8000``
     - HTTP port for the unified server (single/distributed) or proxy router client port (disaggregated).
   * - ``container_config.device_list``
     - ``[ "/dev/dri", "/dev/kfd" ]`` (single)
     - GPU devices passed into the container. Multi-node configs also include ``/dev/infiniband/rdma_cm``.
   * - ``container_config.volume_dict``
     - host → container map
     - Bind mounts for home, models, and (multi-node) RDMA libraries. See :ref:`sglang-volume-mounts`.

Single-node only (``sglang_single``)
------------------------------------

.. list-table::
   :widths: 3 5
   :header-rows: 1

   * - Parameter
     - Description
   * - ``benchmark_serv_node``
     - Exactly one host; only this node receives a container. Other cluster nodes are ignored.
   * - ``nnodes``
     - Must be ``1``.

Unified multi-node (``sglang_distributed``)
-------------------------------------------

Additional ``config`` fields beyond the single-node set:

.. list-table::
   :widths: 3 5
   :header-rows: 1

   * - Parameter
     - Description
   * - ``server_node_list``
     - All ranks of the unified ``sglang.launch_server`` (length must equal ``nnodes``).
   * - ``dist_init_port``
     - Distributed init port on rank-0 (default ``40001``).
   * - ``nic_type``
     - ``thor2`` (Broadcom Thor) or ``ainic`` (AMD Pensando). Drives IB setup behavior.
   * - ``nccl_ib_hca``, ``nccl_ib_gid_index``
     - NCCL InfiniBand/RoCE device list and GID index.
   * - ``nccl_socket_ifname``, ``gloo_socket_ifname``
     - Ethernet interfaces for socket/Gloo fallback.
   * - ``hca_id_prefix``, ``mount_vol``
     - Used by ``test_setup_ibv_devices`` when ``nic_type`` matches Broadcom/Thor; ``mount_vol`` is the in-container path to ``libbnxt_re-rdmav34.so``.

Disaggregated prefill-decode (``sglang_disagg_distributed``)
------------------------------------------------------------

Uses the multi-node network fields above, plus:

.. list-table::
   :widths: 3 5
   :header-rows: 1

   * - Parameter
     - Description
   * - ``prefill_node_list``, ``decode_node_list``
     - Node groups for prefill and decode servers.
   * - ``proxy_router_node``
     - Host running the PD proxy router.
   * - ``prefill_serv_port``, ``decode_serv_port``, ``proxy_router_port``
     - Internal service ports (defaults ``30001``, ``30002``, ``8000``).
   * - ``prefill_coordinator_addr``, ``decode_coordinator_addr``
     - Rank-0 addresses for each role group.
   * - ``prefill_coordinator_port``, ``decode_coordinator_port``
     - Coordinator ports (defaults ``40001``, ``40002``).
   * - ``nccl_ib_hca_list``
     - RDMA devices for disaggregation transfer (in addition to ``nccl_ib_hca``).

``benchmark_params`` / model settings
=====================================

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Parameter
     - Example
     - Description
   * - ``model``
     - ``meta-llama/Llama-3.1-70B-Instruct``
     - HuggingFace ID or container path (for example ``/root/models/DeepSeek-R1-0528``).
   * - ``threshold_file``
     - path under ``cvs/input/config_file/inference/sglang/``
     - External JSON with per-cell performance thresholds.
   * - ``tensor_parallelism``, ``pipeline_parallelism``
     - ``8``, ``1`` or ``2``
     - TP size per node; PP across nodes for distributed/disaggregated runs.
   * - ``memory_fraction``
     - ``0.85``
     - Static KV-cache memory fraction passed to ``launch_server``.
   * - ``max_concurrency``
     - ``256``
     - ``bench_serving`` concurrency sweep upper bound.
   * - ``add_export_env``, ``add_flags``
     - ROCm/SGLang tuning (for example ``SGLANG_USE_AITER=1``, ``--attention-backend aiter``).
   * - ``context_length``
     - ``205000``
     - Long-context cap (distributed / disaggregated DeepSeek and Llama templates).

Inference tests
===============

Each ``benchmark_params`` variant defines ``inference_tests``:

``bench_serv_random``
  Random synthetic load via ``sglang.bench_serving``. ISL/OSL/concurrency cells come from the
  threshold file; ``input_length`` and ``output_length`` are injected at collection time.

  - ``enforce_thresholds``: when ``false``, measured throughput/latency is recorded and reported
    but does not fail the run. When ``true``, results are compared against the threshold file.
  - ``num_prompts``, ``random_range_ratio``, ``model_num_params``, ``peak_gpu_tflops``: bench workload
    and MFU calculation inputs.

``lm_eval_hellaswag``, ``lm_eval_gsm8k``
  Accuracy tasks via lm-eval. Thresholds for accuracy metrics are always enforced when configured
  in the threshold file.

.. _sglang-volume-mounts:

Volume mounts
=============

**Single-node** configs mount only user home and model storage—no InfiniBand or RDMA verb libraries.

**Distributed and disaggregated** configs add RDMA-related mounts for Thor/Broadcom NICs:

.. code:: json

    {
        "volume_dict": {
            "/dev/infiniband": "/dev/infiniband",
            "/usr/local/lib/libbnxt_re-rdmav34.so": "/usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so:ro",
            "/usr/lib/x86_64-linux-gnu/libibverbs.so.1": "/usr/lib/x86_64-linux-gnu/libibverbs.so.1:ro",
            "/lib/libibverbs.d": "/lib/libibverbs.d"
        }
    }

``test_setup_ibv_devices`` (distributed and disaggregated suites only) validates IB visibility inside
the container after these mounts are applied.

Disaggregated architecture overview
====================================

SGLang disaggregated prefill-decode separates inference into:

1. **Prefill nodes** — process prompts and build KV cache.
2. **Decode nodes** — autoregressive token generation from cached KV states.
3. **Proxy router** — routes requests between prefill and decode clusters.

Use ``sglang_disagg_distributed`` with ``mi3xx_sglang_*_disaggregated.json`` templates. Unified
multi-node serving (no PD split) uses ``sglang_distributed`` instead.

Performance metrics
===================

The results table and threshold files use:

- **Output throughput** (``output_throughput_per_sec``) — output tokens per second.
- **TTFT** (``mean_ttft_ms``) — mean time to first token.
- **TPOT** (``mean_tpot_ms``) — mean time per output token.
- **E2E latency** (``mean_e2e_latency_ms``) — end-to-end request latency.
- **Goodput** — fraction of successful requests.
- **MFU** — model FLOPs utilization derived from ``model_num_params`` and ``peak_gpu_tflops``.

Troubleshooting
===============

**Container launch**
  Verify ``container_image`` on all nodes, ``device_list`` GPU paths, and ``shm_size``. Single-node
  runs need only ``/dev/dri`` and ``/dev/kfd``.

**Multi-node networking**
  Confirm RDMA devices with ``ibv_devinfo`` inside the container after ``test_setup_ibv_devices``.
  Match ``nccl_ib_hca`` / ``nccl_ib_hca_list`` to your cluster. For Thor NICs, ensure
  ``libbnxt_re-rdmav34.so`` mounts are present and ``nic_type`` is set correctly.

**Variant selection**
  If startup fails with multiple ``benchmark_params`` keys, export ``SGLANG_BENCHMARK_KEY``.

**Model access**
  Set ``hf_token_file`` for HuggingFace models or mount local weights under ``/root/models`` via
  ``volume_dict``.
