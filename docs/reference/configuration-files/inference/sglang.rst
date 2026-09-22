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
     - Full-model ``sglang.launch_server`` on the first ``cluster.json`` host (TP across local GPUs). Extra hosts are ignored.
   * - ``sglang_distributed``
     - ``cvs/tests/inference/sglang/sglang_distributed.py``
     - One unified multi-node server (TP/PP across the first ``nnodes`` hosts in ``cluster.json``).
   * - ``sglang_disagg_distributed``
     - ``cvs/tests/inference/sglang/sglang_disagg_distributed.py``
     - Disaggregated prefill/decode: even ``nnodes`` from ``cluster.json``; rank-0 is
       prefill coordinator, proxy, and benchmark; rank-1 is decode coordinator.

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

Threshold keys use the form ``ISL=<n>,OSL=<n>,TP=<n>,PP=<n>,CONC=<n>``. Each value is a
metric map (for example ``output_throughput_per_sec``, ``mean_ttft_ms``, ``mean_tpot_ms``,
``goodput``, ``mfu``) with ``kind`` and ``value`` fields. Accuracy cells use
``BENCH=lm_eval_hellaswag`` and ``BENCH=lm_eval_gsm8k``. Long-context NIAH cells use
``ACC_ISL=<n>,OSL=<n>`` (shipped files currently have ``ACC_ISL=131072,OSL=1024`` with
``pass_rate``). Those cells parametrize ``test_run_long_context_accuracy`` on
``sglang_disagg_distributed``; the stage still skips unless ``lng_ctx_activate`` is
``true``.

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
       results are compared against the threshold file. Does not gate lm-eval or NIAH
       accuracy (those always use their threshold cells when the stage runs).
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
   * - ``long_ctx_niah``
     - Optional NIAH workload for ``test_run_long_context_accuracy`` (DeepSeek disaggregated
       template). Loaded into ``inference_tests.long_ctx_niah``.
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
                        "SGLANG_USE_AITER": "1"
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
     - Server rank count. For ``sglang_distributed`` and ``sglang_disagg_distributed``,
       the first this many hosts from ``cluster.json`` participate. Must be at least 2
       and must not exceed the cluster size. Disaggregated also requires an even
       ``nnodes`` so prefill and decode groups stay equal (1P/1D, 2P/2D, …).
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
     - unused in packaged configs
     - ``sglang_single`` uses the first cluster host. ``sglang_distributed`` and
       ``sglang_disagg_distributed`` also derive hosts from ``cluster.json`` order.
   * - ``server_params.proxy_router_serv_port``
     - ``8000``
     - HTTP port for the unified server or proxy router. Optional; defaults to
       ``8000``. Omit from packaged SGLang configs.
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
   * - Cluster hosts
     - Only the first ``node_dict`` entry in ``cluster.json`` gets a container and a
       full-model server. Extra hosts are ignored. ``server_params.benchmark_serv_node``
       is unused.
   * - ``server_params.nnodes``
     - Must be ``1`` (local TP only; nodes do not form one multi-rank server).
   * - HTTP port
     - Defaults to ``8000``. Do not set ``proxy_router_serv_port``.

Unified multi-node (``sglang_distributed``)
-------------------------------------------

Additional ``server_params`` / ``container`` env fields beyond the single-node set:

.. list-table::
   :widths: 3 5
   :header-rows: 1

   * - Parameter
     - Description
   * - ``server_params.nnodes``
     - How many ``cluster.json`` hosts to use (``node_dict`` order). Rank-0 is master
       and the benchmark node. The suite fails immediately if this is larger than
       the cluster or less than 2.
   * - HTTP / dist-init ports
     - HTTP defaults to ``8000``. Dist-init defaults to ``40001``. Do not set
       ``server_node_list``, ``benchmark_serv_node``, ``dist_init_port``, or
       ``proxy_router_serv_port``.
   * - ``NCCL_IB_HCA``, ``NCCL_IB_GID_INDEX``
     - NCCL InfiniBand/RoCE device list and GID index (``container.runtime.args.env``).
   * - ``NCCL_SOCKET_IFNAME``, ``GLOO_SOCKET_IFNAME``, ``GLOO_TCP_IFNAME``
     - Ethernet interfaces for socket/Gloo fallback.

Disaggregated prefill-decode (``sglang_disagg_distributed``)
------------------------------------------------------------

Uses the multi-node network env fields above, plus:

.. list-table::
   :widths: 3 5
   :header-rows: 1

   * - Parameter
     - Description
   * - ``server_params.nnodes``
     - Even count of ``cluster.json`` hosts (``node_dict`` order). Rank-0 is prefill
       coordinator, proxy router, and benchmark. Rank-1 is decode coordinator.
       Remaining hosts split equally into prefill and decode. Fails if ``nnodes``
       is odd, less than 2, or larger than the cluster.
   * - Ports
     - Prefill serve ``30001``, decode serve ``30002``, HTTP/proxy ``8000``,
       prefill coordinator ``40001``, decode coordinator ``40002``. Do not set
       node lists, coordinator addresses, or those port fields.

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
   * - ``container.runtime.args.env``, ``server_params.add_flags``
     - ROCm/SGLang tuning as scalar env (for example ``SGLANG_USE_AITER``, ``AMDGCN_USE_BUFFER_OPS``, ``ROCM_QUICK_REDUCE_QUANTIZATION``) plus ``--attention-backend aiter``. DeepSeek templates also set ``GPU_ARCHS=gfx942``.
   * - ``server_params.context_length``
     - ``205000`` (Llama / unified DeepSeek) or ``163840`` (DeepSeek disaggregated)
     - KV-cache context cap passed to ``launch_server`` as ``--context-length``. On
       disaggregated runs this flag is applied to both prefill and decode. Required when
       ``lng_ctx_activate`` is ``true``.
   * - ``server_params.lng_ctx_activate``
     - ``true`` (DeepSeek disaggregated only)
     - Enables ``test_run_long_context_accuracy`` and injects long-context CLI flags.
       Omit or set to anything other than ``true`` to skip NIAH (Llama disaggregated).
   * - ``server_params.chunked_prefill_size``
     - ``8192``
     - Required when ``lng_ctx_activate`` is ``true``. Passed as ``--chunked-prefill-size``
       on prefill (and unified) servers only — not on decode.
   * - ``server_params.max_prefill_tokens``
     - ``8192``
     - Optional with ``lng_ctx_activate``. Passed as ``--max-prefill-tokens`` on prefill
       (and unified) servers only.
   * - ``long_ctx_niah.num_prompts``, ``seed``, ``request_timeout_sec``, ``exec_timeout_sec``, ``tolerance_frac``
     - ``6``, ``42``, ``7200``, ``21600``, ``0.05``
     - NIAH client settings. ISL/OSL come from the ``ACC_ISL=…`` threshold cell, not this block.
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

``long_ctx_niah`` (``sglang_disagg_distributed``)
  Needle-in-a-haystack long-context accuracy. Collection parametrizes one pytest case per
  ``ACC_ISL=<n>,OSL=<n>`` threshold cell. The test skips unless ``server_params.lng_ctx_activate``
  is ``true``. When it runs, ``pass_rate`` from that cell is always enforced.
  ``mi3xx_sglang_deepseek_r1_0528_disaggregated.json`` ships with this enabled.

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

Disaggregated architecture overview
====================================

SGLang disaggregated prefill-decode separates inference into:

1. **Prefill nodes** — process prompts and build KV cache.
2. **Decode nodes** — autoregressive token generation from cached KV states.
3. **Proxy router** — routes requests between prefill and decode clusters.

Use ``sglang_disagg_distributed`` with ``mi3xx_sglang_*_disaggregated.json`` templates. Unified
multi-node serving (no PD split) uses ``sglang_distributed`` instead.

When ``lng_ctx_activate`` is ``true``, prefill launch includes ``--context-length``,
``--chunked-prefill-size``, and optional ``--max-prefill-tokens``. Decode launch includes
``--context-length`` only so it can hold the transferred KV cache.

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
  Verify ``container.image`` on all nodes, ``devices`` GPU paths, and ``shm_size``. Single-node
  runs need only ``/dev/dri`` and ``/dev/kfd``. Put ROCm/SGLang knobs as scalar ``env``
  keys (``SGLANG_USE_AITER``, ``GPU_ARCHS``, ...), not as an ``ADD_EXPORT_ENV`` list.

**Multi-node networking**
  Confirm RDMA devices with ``ibv_devinfo`` inside the container.
  Match ``NCCL_IB_HCA`` to your cluster. For Thor NICs, ensure ``libbnxt_re-rdmav34.so`` mounts
  are present.

**Sweep collection**
  Each listed ``sweep.runs`` combo must match a threshold cell exactly or uniquely by
  ``ISL,OSL,CONC``. Empty ``runs`` selects every performance cell in the threshold JSON.

**Long-context NIAH**
  ``sglang_disagg_distributed`` requires at least one ``ACC_ISL=…,OSL=…`` cell in the
  threshold file (collection fails without it). The stage then skips unless
  ``lng_ctx_activate`` is ``true``. If the server OOMs at 131k ISL, confirm
  ``context_length`` / ``chunked_prefill_size`` on the DeepSeek disaggregated template.

**Model access**
  Set ``paths.hf_token_file`` for HuggingFace models or mount local weights under ``/root/models`` via
  ``volumes``.
