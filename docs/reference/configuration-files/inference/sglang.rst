.. meta::
  :description: Configure SGLang inference benchmarks on AMD MI30X clusters
  :keywords: inference, ROCm, cvs, SGLang, LLM, MI30X, distributed, disaggregated, prefill, decode

**********************************
SGLang inference configuration
**********************************

CVS ships three SGLang inference suites for AMD MI30X clusters. Each suite reads a JSON
configuration file from ``cvs/input/config_file/inference/sglang/`` and a matching
threshold file referenced by top-level ``threshold_json``.

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

How to run: :doc:`/how-to/test-suites/inference/sglang`.

Run any suite with:

.. code:: bash

  cvs run <suite> \
    --cluster_file cvs/input/cluster_file/<cluster>.json \
    --config_file cvs/input/config_file/inference/sglang/<config>.json \
    --html=~/cvs_results/sglang.html

Copy a template locally:

.. code:: bash

  cvs config list inference/sglang
  cvs config copy inference/sglang/mi3xx_sglang_llama_70b_single.json \
    --output ~/cvs_workspace/inference/sglang/mi3xx_sglang_llama_70b_single.json
  cvs config copy inference/sglang/mi325_sglang_llama_70b_threshold.json \
    --output ~/cvs_workspace/inference/sglang/mi325_sglang_llama_70b_threshold.json

.. note::

  - ``{user-id}`` in path strings is resolved to the current username at runtime.
  - Replace every ``<changeme>`` placeholder before running; unresolved placeholders cause a hard exit at startup.

Configuration files
===================

All SGLang templates live under ``cvs/input/config_file/inference/sglang/``.

Model and topology templates
----------------------------

.. list-table::
   :widths: 3 3 2
   :header-rows: 1

   * - Config file
     - Threshold JSON
     - Use with suite
   * - ``mi3xx_sglang_llama_70b_single.json``
     - ``mi325_sglang_llama_70b_threshold.json``
     - ``sglang_single``
   * - ``mi3xx_sglang_deepseek_r1_0528_single.json``
     - ``mi325_sglang_deepseek_r1_0528_threshold.json``
     - ``sglang_single``
   * - ``mi3xx_sglang_llama_70b_distributed.json``
     - ``mi325_sglang_llama_70b_threshold.json``
     - ``sglang_distributed``
   * - ``mi3xx_sglang_deepseek_r1_0528_distributed.json``
     - ``mi325_sglang_deepseek_r1_0528_threshold.json``
     - ``sglang_distributed``
   * - ``mi3xx_sglang_llama_70b_disaggregated.json``
     - ``mi325_sglang_llama_70b_threshold.json``
     - ``sglang_disagg_distributed``
   * - ``mi3xx_sglang_deepseek_r1_0528_disaggregated.json``
     - ``mi325_sglang_deepseek_r1_0528_threshold.json``
     - ``sglang_disagg_distributed``

Threshold files
---------------

Performance cells and pass/fail limits are stored separately. Each workload points at its
threshold file via top-level ``threshold_json`` (a filename beside the config).

.. list-table::
   :widths: 3 5
   :header-rows: 1

   * - Threshold file
     - Referenced by
   * - ``mi325_sglang_llama_70b_threshold.json``
     - Llama 3.1 70B configs (single, distributed, disaggregated)
   * - ``mi325_sglang_deepseek_r1_0528_threshold.json``
     - DeepSeek-R1-0528 configs (single, distributed, disaggregated)
   * - ``mi325_sglang_gpt_oss_120b_threshold.json``
     - Custom / future GPT-OSS 120B runs (point ``threshold_json`` at this path)
   * - ``mi325_sglang_glm_52_fp8_threshold.json``
     - Custom / future GLM 5.2 FP8 runs
   * - ``mi325_sglang_kimi_k26_threshold.json``
     - Custom / future Kimi K2.6 runs

Threshold keys use the form ``ISL=<n>,OSL=<n>,TP=<n>,PP=<n>,CONC=<n>``. Each value is a
metric map (for example ``output_throughput_per_sec``, ``mean_ttft_ms``, ``mean_tpot_ms``,
``goodput``, ``mfu``) with ``kind`` and ``value`` fields. Accuracy cells use
``BENCH=lm_eval_hellaswag`` and ``BENCH=lm_eval_gsm8k``.

File structure
==============

Shipped templates use these top-level keys:

.. list-table::
   :widths: 2 6
   :header-rows: 1

   * - Key
     - Description
   * - ``enforce_thresholds``
     - When ``false``, performance metrics are recorded but do not fail the run. When ``true``,
       results are compared against the threshold file. Does not gate lm-eval accuracy.
   * - ``threshold_json``
     - Filename of the threshold JSON in the same directory.
   * - ``paths``
     - ``shared_fs``, ``models_dir``, ``log_dir``, ``hf_token_file``.
   * - ``container``
     - Image, name, lifetime, and ``runtime.args`` (volumes, devices, env).
   * - ``server_params``
     - Model, TP/PP, node lists, ports, and launch flags.
   * - ``benchmark_params``
     - ``bench_serving`` workload (``num_prompts``, ``data_set_name``, MFU inputs).
   * - ``accuracy``
     - ``tasks`` list (``lm_eval_hellaswag``, ``lm_eval_gsm8k``).
   * - ``sweeps``
     - Optional per-combo overrides (for example ``num_prompts``).
   * - ``sweep``
     - ``runs`` list of combo keys to parametrize. Empty ``runs`` uses every performance cell
       in the threshold JSON.

The HuggingFace or filesystem path loaded by SGLang is ``server_params.model``.

Example: single-node template
=============================

.. dropdown:: ``mi3xx_sglang_llama_70b_single.json`` (abbreviated)

  .. code:: json

    {
        "enforce_thresholds": false,
        "threshold_json": "mi325_sglang_llama_70b_threshold.json",
        "paths": {
            "shared_fs": "/home/{user-id}",
            "models_dir": "/root/models",
            "log_dir": "{shared_fs}/LOGS/sglang",
            "hf_token_file": "{shared_fs}/.hf_token"
        },
        "container": {
            "lifetime": "per_run",
            "name": "sglang_container",
            "image": "<changeme>",
            "runtime": {
                "name": "docker",
                "args": {
                    "network": "host",
                    "ipc": "host",
                    "privileged": true,
                    "shm_size": "128G",
                    "volumes": [
                        "/home/{user-id}:/home/{user-id}",
                        "/mnt/dtni/models:/root/models"
                    ],
                    "devices": [ "/dev/dri", "/dev/kfd" ],
                    "env": {
                        "NCCL_DEBUG": "ERROR",
                        "ADD_EXPORT_ENV": [ "SGLANG_USE_AITER=1" ]
                    }
                }
            }
        },
        "server_params": {
            "backend": "sglang",
            "nnodes": "1",
            "model": "meta-llama/Llama-3.1-70B-Instruct",
            "tensor_parallelism": "8",
            "pipeline_parallelism": "1",
            "benchmark_serv_node": "<changeme>",
            "proxy_router_serv_port": "8000",
            "add_flags": [ "--attention-backend aiter" ]
        },
        "benchmark_params": {
            "backend": "sglang",
            "data_set_name": "random",
            "num_prompts": "25",
            "model_num_params": "70000000000",
            "peak_gpu_tflops": "2615"
        },
        "accuracy": { "tasks": [ { "id": "lm_eval_hellaswag" }, { "id": "lm_eval_gsm8k" } ] },
        "sweep": {
            "runs": [
                { "combo": "ISL=1024,OSL=1024,TP=8,PP=1,CONC=64" }
            ]
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
   * - ``container.image``
     - ``rocm/sgl-dev:…``
     - Docker image with SGLang and ROCm for MI30X.
   * - ``container.name``
     - ``sglang_container``
     - Container instance name on each participating node.
   * - ``server_params.nnodes``
     - ``1``, ``2``, ``4``, …
     - Server rank count. For ``sglang_distributed``, must match ``server_node_list`` length.
       Disaggregated launch uses the lengths of ``prefill_node_list`` and ``decode_node_list``.
   * - ``paths.hf_token_file``
     - ``/home/{user-id}/.hf_token``
     - HuggingFace token file for model download.
   * - ``container.runtime.args.shm_size``
     - ``128G``
     - Docker shared memory size.
   * - ``paths.log_dir``
     - ``/home/{user-id}/LOGS/sglang``
     - Shared log root (must be visible from benchmark nodes).
   * - ``server_params.log_level``
     - ``info``
     - SGLang server log level.
   * - ``container.runtime.args.env.NCCL_DEBUG``
     - ``ERROR``
     - NCCL log level (multi-node).
   * - ``server_params.benchmark_serv_node``
     - node hostname/IP
     - Node that runs smoke tests, lm-eval, and ``bench_serving`` (required for all suites).
   * - ``server_params.proxy_router_serv_port``
     - ``8000``
     - HTTP port for the unified server (single/distributed) or proxy router client port (disaggregated).
   * - ``container.runtime.args.devices``
     - ``[ "/dev/dri", "/dev/kfd" ]`` (single)
     - GPU devices passed into the container. Multi-node configs also include ``/dev/infiniband/rdma_cm``.
   * - ``container.runtime.args.volumes``
     - list of ``host:container[:opts]`` strings
     - Bind mounts for home, models, and (multi-node) RDMA libraries. See :ref:`sglang-volume-mounts`.

Single-node only (``sglang_single``)
------------------------------------

.. list-table::
   :widths: 3 5
   :header-rows: 1

   * - Parameter
     - Description
   * - ``server_params.benchmark_serv_node``
     - Exactly one host; only this node receives a container. Other cluster nodes are ignored.
   * - ``server_params.nnodes``
     - Must be ``1``.

Unified multi-node (``sglang_distributed``)
-------------------------------------------

Additional ``server_params`` / ``container`` env fields beyond the single-node set:

.. list-table::
   :widths: 3 5
   :header-rows: 1

   * - Parameter
     - Description
   * - ``server_params.server_node_list``
     - All ranks of the unified ``sglang.launch_server`` (length must equal ``nnodes``).
   * - ``server_params.dist_init_port``
     - Distributed init port on rank-0 (default ``40001``).
   * - ``NCCL_IB_HCA``, ``NCCL_IB_GID_INDEX``
     - NCCL InfiniBand/RoCE device list and GID index (``container.runtime.args.env``).
   * - ``NCCL_SOCKET_IFNAME``, ``GLOO_SOCKET_IFNAME``, ``GLOO_TCP_IFNAME``
     - Ethernet interfaces for socket/Gloo fallback.
   * - ``HCA_ID_PREFIX``
     - Used by ``test_setup_ibv_devices`` to match ``ibv_devinfo`` HCA names. The host
       ``libbnxt_re-rdmav34.so`` is bind-mounted via ``volumes``.

Disaggregated prefill-decode (``sglang_disagg_distributed``)
------------------------------------------------------------

Uses the multi-node network env fields above, plus:

.. list-table::
   :widths: 3 5
   :header-rows: 1

   * - Parameter
     - Description
   * - ``server_params.prefill_node_list``, ``decode_node_list``
     - Node groups for prefill and decode servers. ``--nnodes`` / ``--node-rank`` follow these list lengths.
   * - ``server_params.proxy_router_node``
     - Host running the PD proxy router.
   * - ``prefill_serv_port``, ``decode_serv_port``, ``proxy_router_port``
     - Internal service ports (defaults ``30001``, ``30002``, ``8000``).
   * - ``prefill_coordinator_addr``, ``decode_coordinator_addr``
     - Rank-0 addresses for each role group.
   * - ``prefill_coordinator_port``, ``decode_coordinator_port``
     - Coordinator ports (defaults ``40001``, ``40002``).

``benchmark_params`` / model settings
=====================================

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Parameter
     - Example
     - Description
   * - ``server_params.model``
     - ``meta-llama/Llama-3.1-70B-Instruct``
     - HuggingFace ID or container path (for example ``/root/models/DeepSeek-R1-0528``).
   * - ``threshold_json``
     - filename under ``cvs/input/config_file/inference/sglang/``
     - External JSON with per-cell performance thresholds.
   * - ``server_params.tensor_parallelism``, ``pipeline_parallelism``
     - ``8``, ``1`` or ``2``
     - TP size per node; PP across nodes for distributed/disaggregated runs. Sweep combo TP/PP
       labels the cell; they do not relaunch the server.
   * - ``server_params.memory_fraction``
     - ``0.85`` (Llama) / ``0.7`` (DeepSeek)
     - Static KV-cache memory fraction passed to ``launch_server``.
   * - ``server_params.max_concurrency``
     - ``256``
     - ``bench_serving`` concurrency sweep upper bound.
   * - ``server_params.tokenizer_mode``
     - ``auto``
     - Tokenizer mode passed to ``launch_server``.
   * - ``server_params.inference_poll_iterations``
     - ``16``
     - Server-ready poll attempts.
   * - ``ADD_EXPORT_ENV``, ``server_params.add_flags``
     - ROCm/SGLang tuning (for example ``SGLANG_USE_AITER=1``, ``--attention-backend aiter``). DeepSeek templates also set ``GPU_ARCHS=gfx942``.
   * - ``server_params.context_length``
     - ``205000``
     - Long-context cap (distributed / disaggregated Llama and DeepSeek templates).
   * - ``server_params.prefill_policy``, ``decode_policy``
     - ``cache_aware``
     - Disaggregated templates only; PD routing policy.

Inference tests
===============

``benchmark_params``
  Random synthetic load via ``sglang.bench_serving``. ISL/OSL/concurrency cells come from
  ``sweep.runs`` (or every performance cell in the threshold file if ``runs`` is empty).
  ``input_length`` and ``output_length`` are injected at collection time.

  Combo keys are matched exactly, then by unique ``ISL,OSL,CONC`` if TP/PP in the combo
  differs from the threshold-file key. ``sweeps`` supplies per-combo overrides such as
  ``num_prompts``.

  - ``enforce_thresholds`` (top-level): when ``false``, measured throughput/latency is recorded
    and reported but does not fail the run. When ``true``, results are compared against the
    matched threshold cell.
  - ``num_prompts``, ``random_range_ratio``, ``model_num_params``, ``peak_gpu_tflops``: bench workload
    and MFU calculation inputs.

``accuracy.tasks`` (``lm_eval_hellaswag``, ``lm_eval_gsm8k``)
  Accuracy tasks via lm-eval. Thresholds for accuracy metrics are always enforced when configured
  in the threshold file.

.. _sglang-volume-mounts:

Volume mounts
=============

**Single-node** configs mount only user home and model storage—no InfiniBand or RDMA verb libraries.

**Distributed and disaggregated** configs add RDMA-related mounts for Thor/Broadcom NICs:

.. code:: json

    {
        "volumes": [
            "/dev/infiniband:/dev/infiniband",
            "/usr/local/lib/libbnxt_re-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so:ro",
            "/usr/lib/x86_64-linux-gnu/libibverbs.so.1:/usr/lib/x86_64-linux-gnu/libibverbs.so.1:ro",
            "/lib/libibverbs.d:/lib/libibverbs.d"
        ]
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

The performance summary TP/PP columns come from the ``sweep.runs`` combo string. Expected
values still come from the matched threshold cell (ISL/OSL/CONC).

Troubleshooting
===============

**Container launch**
  Verify ``container.image`` on all nodes, ``devices`` GPU paths, and ``shm_size``. Single-node
  runs need only ``/dev/dri`` and ``/dev/kfd``. Keep ``ADD_EXPORT_ENV`` as a JSON list; do not
  put it as a ``docker run -e`` scalar.

**Multi-node networking**
  Confirm RDMA devices with ``ibv_devinfo`` inside the container after ``test_setup_ibv_devices``.
  Match ``NCCL_IB_HCA`` to your cluster. For Thor NICs, ensure ``libbnxt_re-rdmav34.so`` mounts
  are present.

**Sweep collection**
  Each listed ``sweep.runs`` combo must match a threshold cell exactly or uniquely by
  ``ISL,OSL,CONC``. Empty ``runs`` selects every performance cell in the threshold JSON.

**Model access**
  Set ``paths.hf_token_file`` for HuggingFace models or mount local weights under ``/root/models`` via
  ``volumes``.
