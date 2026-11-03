.. meta::
  :description: Reference for CVS SGLang inference benchmark configuration, covering single-node, distributed, and disaggregated prefill-decode topology and threshold files.
  :keywords: CVS, SGLang, inference, ROCm, LLM, GPU, AMD, distributed, disaggregated, prefill, decode, JSON, benchmark

*************************************************************************
SGLang inference benchmark configuration for Cluster Validation Suite (CVS)
*************************************************************************

CVS ships three SGLang inference suites for AMD MI30x and MI35x clusters. Each suite reads a JSON
configuration file from ``cvs/input/config_file/inference/sglang/`` and a matching
threshold file referenced by top-level ``threshold_json``. Hosts come from ``cluster.json``
``node_dict`` order. Role lists in the workload JSON are ignored.

.. list-table::
   :widths: 2 3 5
   :header-rows: 1

   * - CVS suite
     - Test module
     - Topology
   * - ``sglang_single``
     - ``cvs/tests/inference/sglang/sglang_single.py``
     - One unified ``sglang.launch_server`` on the first ``cluster.json`` host (TP across local GPUs). Extra hosts are ignored.
   * - ``sglang_distributed``
     - ``cvs/tests/inference/sglang/sglang_distributed.py``
     - One unified multi-node server. The first ``server_params.nnodes`` hosts are the ranks. Rank 0 is the benchmark node.
   * - ``sglang_disagg_distributed``
     - ``cvs/tests/inference/sglang/sglang_disagg_distributed.py``
     - Disaggregated prefill/decode. The first even ``nnodes`` hosts split into equal prefill and decode groups. Rank 0 is prefill coordinator, proxy, and benchmark. Rank 1 is decode coordinator.

See :doc:`/how-to/test-suites/inference/sglang`  for more information on running these tests.

Run any suite with:

.. code:: bash

  cvs run <suite> \
    --cluster_file cvs/input/cluster_file/<cluster>.json \
    --config_file cvs/input/config_file/inference/sglang/<config>.json

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

Templates are named ``<gpu>_sglang_<model>_<topology>.json``. ``<model>`` is
``llama_70b``, ``deepseek_r1_0528``, or ``qwen35_397b``. ``<topology>`` is ``single``,
``distributed``, or ``disaggregated``.

.. list-table::
   :widths: 2 3 3
   :header-rows: 1

   * - Config prefix
     - Threshold prefix
     - GPU family
   * - ``mi3xx_sglang_``
     - ``mi325_sglang_``
     - MI30x
   * - ``mi35x_sglang_``
     - ``mi355_sglang_``
     - MI35x

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
     - MI30x Llama 3.1 70B configs (single, distributed, disaggregated)
   * - ``mi325_sglang_deepseek_r1_0528_threshold.json``
     - MI30x DeepSeek-R1-0528 configs (single, distributed, disaggregated)
   * - ``mi325_sglang_qwen35_397b_threshold.json``
     - MI30x Qwen3.5-397B configs (single, distributed, disaggregated)
   * - ``mi355_sglang_llama_70b_threshold.json``
     - MI35x Llama 3.1 70B configs (single, distributed, disaggregated)
   * - ``mi355_sglang_deepseek_r1_0528_threshold.json``
     - MI35x DeepSeek-R1-0528 configs (single, distributed, disaggregated)
   * - ``mi355_sglang_qwen35_397b_threshold.json``
     - MI35x Qwen3.5-397B configs (single, distributed, disaggregated)

Threshold keys use the form ``ISL=<n>,OSL=<n>,TP=<n>,PP=<n>,CONC=<n>``. Each value is a
metric map (for example ``output_throughput_per_sec``, ``mean_ttft_ms``, ``mean_tpot_ms``,
``goodput``, ``mfu``) with ``kind`` and ``value`` fields. Accuracy cells use
``BENCH=lm_eval_hellaswag`` and ``BENCH=lm_eval_gsm8k``.

File structure
==============

Shipped templates use these top-level keys:

The following table describes each top-level key in a SGLang configuration file.

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
   * - ``gpu_name``
     - Cluster GPU label. Shipped values contain ``<changeme>`` and must be replaced.
   * - ``server_params``
     - Model, TP/PP, ``nnodes``, optional ``serve_port``, and launch flags. Do not set node lists.
   * - ``benchmark_params``
     - ``bench_serving`` workload (``num_prompts``, ``data_set_name``, MFU inputs).
   * - ``long_ctx_niah``
     - Disaggregated only. Needle-in-a-haystack probe settings. The stage runs only when
       ``server_params.lng_ctx_activate`` is true.
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
     - Docker image with SGLang and ROCm. Use an MI30x image for ``mi3xx`` configs and an MI35x image for ``mi35x`` configs.
   * - ``container.name``
     - ``sglang_container``
     - Container instance name on each participating node.
   * - ``server_params.nnodes``
     - ``1``, ``2``, ``4``, …
     - How many hosts to take from ``cluster.json``, in ``node_dict`` order. Single-node configs use ``1``. Distributed configs need at least ``2``. Disaggregated configs need an even count of at least ``2``.
   * - ``server_params.serve_port``
     - ``8010``
     - Client HTTP port. Omitted configs use ``8000``. On disaggregated runs this is the proxy port. Prefill and decode stay on ``30001`` and ``30002``.
   * - ``paths.hf_token_file``
     - ``/home/{user-id}/.hf_token``
     - HuggingFace token file for model download.
   * - ``container.runtime.args.shm_size``
     - ``128G``
     - Docker shared memory size.
   * - ``paths.log_dir``
     - ``/home/{user-id}/LOGS/sglang``
     - Shared log root. ``test_launch_container`` deletes and recreates this directory. Passwordless sudo uses ``sudo -n rm``; otherwise the suite runs a plain ``rm``.
   * - ``server_params.log_level``
     - ``info``
     - SGLang server log level.
   * - ``container.runtime.args.env.HSA_FORCE_FINE_GRAIN_PCIE``
     - ``0`` or ``1``
     - Exported into every server, proxy, and benchmark script. Blank or omitted values become ``1``.
   * - ``container.runtime.args.env.NCCL_DEBUG``
     - ``ERROR``
     - NCCL log level (multi-node).
   * - ``container.runtime.args.devices``
     - ``[ "/dev/dri", "/dev/kfd" ]`` (single)
     - GPU devices passed into the container. MI30x multi-node configs also include ``/dev/infiniband/rdma_cm``. MI35x multi-node configs include ``/dev/infiniband``.
   * - ``container.runtime.args.volumes``
     - list of ``host:container[:opts]`` strings
     - Bind mounts for home, models, and (multi-node) RDMA libraries. See :ref:`sglang-volume-mounts`.

Single-node only (``sglang_single``)
------------------------------------

The suite uses the first host in ``cluster.json`` and ignores the rest. Leave ``benchmark_serv_node`` unset. ``server_params.nnodes`` must be ``1``.

Unified multi-node (``sglang_distributed``)
-------------------------------------------

The suite takes the first ``nnodes`` hosts from ``cluster.json``. Rank 0 is the dist-init master and the benchmark node. Leave ``server_node_list`` and ``benchmark_serv_node`` unset. If ``nnodes`` is larger than the cluster, the suite fails at load time.

.. list-table::
   :widths: 3 5
   :header-rows: 1

   * - Parameter
     - Description
   * - ``server_params.dist_init_port``
     - Distributed init port on rank 0. Omitted configs use ``40001``.
   * - ``NCCL_IB_HCA``, ``NCCL_IB_GID_INDEX``
     - NCCL InfiniBand/RoCE device list and GID index (``container.runtime.args.env``).
   * - ``NCCL_SOCKET_IFNAME``, ``GLOO_SOCKET_IFNAME``, ``GLOO_TCP_IFNAME``
     - Ethernet interfaces for socket/Gloo fallback.
   * - Other scalar ``env`` entries
     - Copied into the in-container ``export`` block. The NCCL, Gloo, and ``HSA_FORCE_FINE_GRAIN_PCIE`` keys above are applied on their own and are not duplicated in that block.

Disaggregated prefill-decode (``sglang_disagg_distributed``)
------------------------------------------------------------

Uses the multi-node network env fields above. The loader drops any configured role lists and assigns them from the first ``nnodes`` hosts:

1. Rank 0 is the prefill coordinator, the proxy router, and the benchmark node.
2. Rank 1 is the decode coordinator.
3. The remaining hosts split in half: the first half joins prefill, the second half joins decode.

``nnodes`` must be even and at least 2, and the cluster must contain at least that many hosts.

.. list-table::
   :widths: 3 5
   :header-rows: 1

   * - Parameter
     - Description
   * - ``server_params.serve_port``
     - Proxy client port. Omitted configs use ``8000``. MI35x templates set ``8010``.
   * - ``prefill_serv_port``, ``decode_serv_port``
     - Internal service ports. Omitted configs use ``30001`` and ``30002``.
   * - ``prefill_coordinator_port``, ``decode_coordinator_port``
     - Coordinator ports. Omitted configs use ``40001`` and ``40002``.
   * - ``server_params.prefill_policy``, ``decode_policy``
     - PD routing policy. Shipped templates use ``cache_aware``.
   * - ``MC_GID_INDEX``, ``MC_DISABLE_HIP``
     - Mooncake Transfer Engine settings on MI35x disaggregated templates (``container.runtime.args.env``). ``MC_GID_INDEX`` is the RoCEv2 GID index shared by the configured HCAs. ``MC_DISABLE_HIP=1`` keeps cross-node KV transfer on RDMA. SGLang selects Mooncake when ``--disaggregation-transfer-backend`` is omitted.
   * - ``server_params.lng_ctx_activate``
     - When true, ``test_run_long_context_accuracy`` runs the ``long_ctx_niah`` probe. Shipped templates set this to false, so the stage skips.

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
     - ``0.85`` or ``0.7``
     - Static KV-cache memory fraction passed to ``launch_server``. Llama templates use ``0.85``. DeepSeek templates use ``0.7``. Qwen single-node uses ``0.85``; Qwen multi-node uses ``0.7``.
   * - ``server_params.max_concurrency``
     - ``256``
     - ``bench_serving`` concurrency sweep upper bound.
   * - ``server_params.tokenizer_mode``
     - ``auto``
     - Tokenizer mode passed to ``launch_server``.
   * - ``server_params.inference_poll_iterations``
     - ``16``
     - Server-ready poll attempts. The poll looks for ``server is fired up and ready to roll`` in the server log.
   * - ``container.runtime.args.env``, ``server_params.add_flags``
     - Scalar env such as ``SGLANG_USE_AITER``, ``AMDGCN_USE_BUFFER_OPS``, and ``ROCM_QUICK_REDUCE_QUANTIZATION``, plus ``--attention-backend aiter``. MI30x DeepSeek templates and MI30x multi-node Qwen templates set ``GPU_ARCHS=gfx942``. MI35x disaggregated templates use SGLang's default Mooncake transfer backend and set ``MC_GID_INDEX`` and ``MC_DISABLE_HIP``.
   * - ``server_params.context_length``
     - ``205000`` or ``163840``
     - Long-context cap on distributed (``205000``) and disaggregated (``163840``) templates.

Inference tests
===============

``benchmark_params``
  Random synthetic load via ``sglang.bench_serving``. ISL/OSL/concurrency cells come from
  ``sweep.runs`` (or every performance cell in the threshold file if ``runs`` is empty).
  ``input_length`` and ``output_length`` are injected at collection time.

  Combo keys match a threshold cell by the full ``ISL,OSL,TP,PP,CONC`` key. If that key is missing and exactly one threshold cell has the same ``ISL,OSL,CONC``, that cell is used. TP and PP on the combo then only label the run; they do not relaunch the server. If several threshold cells share that shape, collection fails and the combo must use the exact key. ``sweeps`` supplies per-combo overrides such as ``num_prompts``. Empty ``runs`` selects each performance cell in the threshold JSON once.

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

**Single-node** configs mount only user home and model storage.

**MI30x distributed and disaggregated** configs add Thor/Broadcom RDMA mounts:

.. code:: json

    {
        "volumes": [
            "/dev/infiniband:/dev/infiniband",
            "/usr/local/lib/libbnxt_re-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so:ro",
            "/usr/lib/x86_64-linux-gnu/libibverbs.so.1:/usr/lib/x86_64-linux-gnu/libibverbs.so.1:ro",
            "/lib/libibverbs.d:/lib/libibverbs.d"
        ]
    }

**MI35x distributed and disaggregated** configs mount the Ionic RDMA libraries instead:

.. code:: json

    {
        "volumes": [
            "/dev/infiniband:/dev/infiniband",
            "/usr/lib/x86_64-linux-gnu/libionic.so.1:/usr/lib/x86_64-linux-gnu/libionic.so.1:ro",
            "/usr/lib/x86_64-linux-gnu/libionic.so:/usr/lib/x86_64-linux-gnu/libionic.so:ro",
            "/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:ro",
            "/etc/libibverbs.d/ionic.driver:/etc/libibverbs.d/ionic.driver:ro"
        ]
    }

Those host paths must exist on every participating node. Change the mount source if the library lives somewhere else.

Disaggregated architecture overview
====================================

SGLang disaggregated prefill-decode separates inference into:

1. **Prefill nodes** — process prompts and build KV cache.
2. **Decode nodes** — autoregressive token generation from cached KV states.
3. **Proxy router** — routes requests between prefill and decode clusters.

Use ``sglang_disagg_distributed`` with a ``*_disaggregated.json`` template. Unified
multi-node serving (no PD split) uses ``sglang_distributed``. The node split follows
``cluster.json`` order and ``server_params.nnodes``, as described above.

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
  keys (``SGLANG_USE_AITER``, ``GPU_ARCHS``, ``HSA_FORCE_FINE_GRAIN_PCIE``, ...).
  If ``paths.log_dir`` is owned by root and passwordless sudo is unavailable, the cleanup
  ``rm`` fails.

**Multi-node networking**
  Match ``NCCL_IB_HCA`` to the fabric. MI30x templates expect Broadcom ``bnxt`` mounts and
  ``rdma*`` devices. MI35x templates expect Ionic library mounts and ``ionic_*`` devices.
  Confirm the mounted ``.so`` files exist on the host before the container starts.

**Host selection**
  Put hosts in ``cluster.json`` in rank order. The config does not choose nodes.
  ``sglang_single`` uses the first host. ``sglang_distributed`` uses the first ``nnodes``
  hosts. ``sglang_disagg_distributed`` needs an even ``nnodes`` and splits that prefix
  into equal prefill and decode groups.

**Sweep collection**
  Each listed ``sweep.runs`` combo must match a threshold cell by the full key, or be the
  only cell with that ``ISL,OSL,CONC``. Empty ``runs`` selects every performance cell in
  the threshold JSON once. A sweep TP/PP that differs from the server TP/PP does not
  relaunch the server.

**Model access**
  Set ``paths.hf_token_file`` for HuggingFace models or mount local weights under ``/root/models`` via
  ``volumes``.
