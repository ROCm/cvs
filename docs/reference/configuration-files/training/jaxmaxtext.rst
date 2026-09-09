.. meta::
  :description: Configure the variables in the JAX MaxText training configuration files
  :keywords: training, ROCm, cvs, JAX, MaxText

****************************************
JAX MaxText training configuration files
****************************************

The JAX MaxText suites (``jaxmaxtext_single`` / ``jaxmaxtext_distributed``) run
`MaxText <https://github.com/AI-Hypercomputer/maxtext>`_ pre-training inside a
container on one or more nodes and gate the run on performance and correctness
metrics with a PASS/FAIL HTML report.

.. note::

  JAX training in CVS is **jaxmaxtext**. The legacy ``jax`` suites
  (``jax_llama3_1_*``) have been removed; use ``jaxmaxtext_single`` /
  ``jaxmaxtext_distributed``.

The JAX MaxText tests check:

- **Container orchestration**: Docker setup with ROCm/RDMA
- **Model load + smoke**: the model loads and trains a few steps with no error/NaN signature
- **Per-sweep training**: one full run per enabled sweep (e.g. BF16, FP8)
- **Performance targets**: TFLOP/s, tokens/s, step time, and multi-node scaling efficiency
- **Convergence**: final loss / loss-decreasing trend, optional time-to-target
- **Checkpoint save/resume** (opt-in): resume correctness + checkpoint I/O timing

Use ``cvs config list training/jaxmaxtext`` to list available templates, or
``cvs config copy training/jaxmaxtext/<name>`` to copy one to your working directory.

.. note::

  - Any value containing ``<changeme>`` must be replaced for your setup.
    ``container.image`` ships with a ``<changeme>`` tag (e.g.
    ``rocm/jax-training:maxtext-v26.4 <changeme>``) — set the image tag your
    cluster has before running; unlike the env placeholders it is not caught at
    config load but fails at container launch until replaced.
    Distributed configs additionally ship the NCCL RDMA/NIC device-selection
    vars in ``container.env`` with an example value plus a ``<changeme>`` tag
    (``NCCL_IB_HCA``, ``NCCL_SOCKET_IFNAME``,
    ``GLOO_SOCKET_IFNAME``, ``NCCL_IB_GID_INDEX``); the config **hard-exits at
    load** while any ``container.env`` value still contains ``<changeme>``.
  - ``{user-id}`` resolves to the cluster/OS username at runtime, and
    ``{shared_fs}`` / ``{paths.*}`` self-references resolve from the ``paths`` block.
  - Keys prefixed with ``_`` (e.g. ``_env_comment``, ``_train_params_comment``)
    are inline comments and are ignored by the loader.

The suite/lifecycle reference is in
:doc:`/how-to/test-suites/training/jax`. The config files themselves live in
``cvs/input/config_file/training/jaxmaxtext/`` (each config plus a sibling
``_threshold.json``); this page documents every block and the threshold format.

Available configurations
=========================

Config files follow the naming pattern ``<gpu>_jaxmaxtext_<model>_<mode>.json``.
Each config has a sibling ``_threshold.json`` referenced by ``threshold_json``.
The **mode is inferred from the config**: distributed configs carry the NCCL
RDMA device-selection vars in ``container.env`` (and add the ``test_setup_rdma``
stage); single-node configs omit them. Run single-node configs with
``jaxmaxtext_single`` and distributed configs with ``jaxmaxtext_distributed``.

.. list-table::
   :widths: 5 2 2 2
   :header-rows: 1

   * - Config file
     - GPU
     - Mode
     - Precisions
   * - ``mi300x_jaxmaxtext_llama-3.1-8b_distributed.json``
     - MI300X
     - distributed
     - BF16, FP8
   * - ``mi300x_jaxmaxtext_llama-3.1-70b_single.json``
     - MI300X
     - single
     - BF16, FP8
   * - ``mi300x_jaxmaxtext_llama-3.1-70b_distributed.json``
     - MI300X
     - distributed
     - BF16, FP8
   * - ``mi300x_jaxmaxtext_llama-3.3-70b_single.json``
     - MI300X
     - single
     - BF16, FP8
   * - ``mi300x_jaxmaxtext_llama-3.3-70b_distributed.json``
     - MI300X
     - distributed
     - BF16, FP8
   * - ``mi300x_jaxmaxtext_deepseek-v2-lite_distributed.json``
     - MI300X
     - distributed
     - BF16
   * - ``mi325x_jaxmaxtext_llama-3.1-8b_distributed.json``
     - MI325X
     - distributed
     - BF16, FP8
   * - ``mi325x_jaxmaxtext_llama-3.1-405b_distributed.json``
     - MI325X
     - distributed
     - BF16, FP8
   * - ``mi325x_jaxmaxtext_llama-3.3-70b_distributed.json``
     - MI325X
     - distributed
     - BF16, FP8
   * - ``mi325x_jaxmaxtext_deepseek-v2-lite_distributed.json``
     - MI325X
     - distributed
     - BF16
   * - ``mi325x_jaxmaxtext_deepseek-v4-284b_distributed.json``
     - MI325X
     - distributed
     - BF16
   * - ``mi35x_jaxmaxtext_llama-3.1-70b_single.json``
     - MI35X
     - single
     - BF16, FP8

Config layout
=============

A config groups its keys into five areas:

1. **CVS params at the root** — ``gpu_name``, ``gpus_per_node``, ``threshold_json``,
   ``enforce_thresholds``, and ``paths``.
2. **container** — image, Docker runtime args, and the static ``env`` exported
   into the container on every node.
3. **train_params** — the tokenizer source, train-script candidates, the
   ``maxtext_config`` passthrough written verbatim to the MaxText YAML, and the
   structured ``xla_flags`` (exported as one ``XLA_FLAGS`` env var).
4. **tests blocks at the root** — ``scaling_baseline``, ``convergence``,
   ``loss_curve``, ``smoke``, ``checkpoint_resume``, ``error_patterns``.
5. **sweeps + runs** — ``sweeps`` is a ``{key: overrides}`` map (one full
   training run each) whose key encodes the primary params
   (``BS=..,PRECISION=..,SL=..``, parsed by CVS); ``runs`` selects which sweep
   keys to execute.

Example configuration
=====================

A representative distributed config
(``mi325x_jaxmaxtext_llama-3.3-70b_distributed.json``, abridged):

.. dropdown:: ``mi325x_jaxmaxtext_llama-3.3-70b_distributed.json`` (abridged)

  .. code:: json

    {
      "gpu_name": "mi325x",
      "threshold_json": "mi325x_jaxmaxtext_llama-3.3-70b_distributed_threshold.json",
      "enforce_thresholds": false,
      "gpus_per_node": 8,

      "paths": {
        "shared_fs": "/home/{user-id}",
        "models_dir": "{shared_fs}/cache/maxtext",
        "log_dir": "{shared_fs}/LOGS/jaxmaxtext",
        "hf_token_file": "{shared_fs}/.hf_token",
        "temp_dir": "/tmp/{user-id}/jaxmaxtext"
      },

      "container": {
        "lifetime": "per_run",
        "name": "rocm-jaxmaxtext-llama3.3-70b",
        "image": "rocm/jax-training:maxtext-v26.4 <changeme>",
        "runtime": { "name": "docker", "args": { "network": "host", "ipc": "host", "privileged": true, "shm-size": "256G", "ulimit": ["nofile=65535:65535"], "volumes": ["..."] } },
        "env": {
          "GPU_MAX_HW_QUEUES": "2",
          "HSA_FORCE_FINE_GRAIN_PCIE": "1",
          "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.97",

          "NCCL_IB_HCA": "rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7 <changeme>",
          "NCCL_SOCKET_IFNAME": "eno0 <changeme>",
          "GLOO_SOCKET_IFNAME": "eno0 <changeme>",
          "NCCL_IB_GID_INDEX": "3 <changeme>",

          "NCCL_IB_DISABLE": "0",
          "NCCL_IB_TC": "41",
          "NCCL_IB_SL": "0",

          "NVTE_FUSED_ATTN": "1",

          "JAX_COORDINATOR_PORT": "12346",
          "JAX_DISTRIBUTED_INITIALIZATION_TIMEOUT_SECONDS": "1800",
          "JAX_DISTRIBUTED_HEARTBEAT_TIMEOUT_SECONDS": "900"
        }
      },

      "train_params": {
        "hf_model_id": "NousResearch/Meta-Llama-3-70B",
        "train_script_paths": ["/workspace/maxtext/src/maxtext/trainers/pre_train/train.py", "/workspace/maxtext/src/MaxText/train.py"],
        "maxtext_config": {
          "base_config": "base.yml",
          "model_name": "llama3.3-70b",
          "tokenizer_path": "{paths.models_dir}/Meta-Llama-70-B",
          "hardware": "gpu",
          "steps": 30,
          "enable_checkpointing": false,
          "attention": "cudnn_flash_te",
          "dtype": "bfloat16",
          "weight_dtype": "bfloat16",
          "dataset_type": "synthetic",
          "quantization": "",
          "per_device_batch_size": 3,
          "max_target_length": 8192,
          "remat_policy": "full",
          "scan_layers": true,
          "ici_fsdp_parallelism": 8,
          "dcn_data_parallelism": -1
        },
        "xla_flags": { "xla_gpu_autotune_level": "0", "...": "..." }
      },

      "scaling_baseline": { "tokens_per_sec_total": 394000.0, "num_nodes": 1 },
      "convergence": { "target_metric": "auto", "target_value": 10.0 },
      "loss_curve": { "sample_every": 10, "milestone_steps": [100, 500, 1000, 5000], "max_slope": 0.0, "enforce": true },
      "smoke": { "enabled": true, "steps": 5, "per_device_batch_size": 1, "max_target_length": 2048 },
      "checkpoint_resume": { "enabled": false, "steps_before_ckpt": 6, "steps_after_resume": 6, "checkpoint_period": 5, "loss_tolerance": 0.1, "delete_ckpt_dir": true },
      "error_patterns": { "NCCL ERROR": "NCCL ERROR|NCCL timeout", "...": "..." },

      "sweeps": {
        "BS=3,PRECISION=BF16,SL=8192": { "_comment": "extra maxtext_config overrides go here, e.g. \"steps\": 300" },
        "BS=3,PRECISION=FP8,SL=8192":  { "_comment": "extra maxtext_config overrides go here, e.g. \"steps\": 300" }
      },
      "runs": ["BS=3,PRECISION=BF16,SL=8192", "BS=3,PRECISION=FP8,SL=8192"]
    }

Top-level (CVS) fields
======================

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Field
     - Example
     - Description
   * - ``gpu_name``
     - ``mi325x``
     - GPU architecture label (informational; also used in run/report labels).
   * - ``gpus_per_node``
     - ``8``
     - GPUs per node; ``num_gpus = num_nodes × gpus_per_node`` feeds
       ``tokens_per_sec_total`` and scaling efficiency. Do not assume a fixed topology.
   * - ``enforce_thresholds``
     - ``false``
     - If ``false``, ``test_metric`` records values but does not fail.
   * - ``threshold_json``
     - ``mi325x_jaxmaxtext_llama-3.3-70b_distributed_threshold.json``
     - Companion threshold filename, resolved next to the config.

``paths``
---------

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Field
     - Example
     - Description
   * - ``shared_fs``
     - ``/home/{user-id}``
     - Base path reachable from all nodes. Self-referenced by other ``paths`` fields as ``{shared_fs}``.
   * - ``models_dir``
     - ``{shared_fs}/cache/maxtext``
     - Tokenizer/model cache directory.
   * - ``log_dir``
     - ``{shared_fs}/LOGS/jaxmaxtext``
     - Training log output directory (per-node logs are namespaced under it).
   * - ``hf_token_file``
     - ``{shared_fs}/.hf_token``
     - Hugging Face token file (for the tokenizer download).
   * - ``temp_dir``
     - ``/tmp/{user-id}/jaxmaxtext``
     - Host-user-namespaced in-container scratch for launcher scripts / MaxText YAML. Keep ``{user-id}`` so shared nodes never collide on ``/tmp/root``.

``container``
-------------

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Field
     - Example
     - Description
   * - ``lifetime``
     - ``per_run``
     - Launched once per session, torn down after.
   * - ``name``
     - ``rocm-jaxmaxtext-llama3.3-70b``
     - Container instance name (any unique string).
   * - ``image``
     - ``rocm/jax-training:maxtext-v26.4 <changeme>``
     - **Required** — the MaxText/JAX ROCm image present on all nodes. Ships with a ``<changeme>`` tag; replace it with the image tag your cluster has before running (not validated at config load, but the container launch fails until it is a valid image reference).
   * - ``runtime.args``
     - *(see snippet)*
     - Docker args: ``network: host``, ``ipc: host``, ``privileged: true``, ``shm-size``, ``ulimit``, and ``volumes``. Distributed configs mount ``/dev/infiniband`` and the NIC ``libibverbs`` provider (``:ro``); ``volumes`` also bind-mounts the home dir and the training-output dir.
   * - ``env``
     - *(dict)*
     - Static environment exported into the container on every node at ``docker run`` time (see :ref:`jax-container-env`). The mode is inferred from it: a config that sets the NCCL IB device vars is treated as distributed.

.. _jax-container-env:

``container.env``
-----------------

A flat ``{NAME: value}`` map exported into the container on every node at
``docker run`` time and inherited by ``docker exec`` (there is no env script to
source). Per-node/dynamic vars (``JAX_COORDINATOR_IP``, ``NNODES``,
``NODE_RANK``, ``JAX_PROCESS_INDEX``) and credentials (``HF_TOKEN``, ``HF_HOME``,
``LD_LIBRARY_PATH``, ``PYTHONPATH``) are injected by the launcher and must **not**
be set here. ``XLA_FLAGS`` is not set here either — it is built from the
structured ``train_params.xla_flags`` map. The vars are grouped for readability:

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Group
     - Examples
     - Description
   * - GPU / memory
     - ``GPU_MAX_HW_QUEUES``, ``HSA_FORCE_FINE_GRAIN_PCIE``, ``HIP_FORCE_DEV_KERNARG``, ``HSA_NO_SCRATCH_RECLAIM``, ``XLA_PYTHON_CLIENT_MEM_FRACTION``
     - ROCm/HIP tuning and the fraction of GPU memory JAX may allocate (e.g. ``0.97``).
   * - NCCL device selection *(distributed)*
     - ``NCCL_IB_HCA``, ``NCCL_SOCKET_IFNAME``, ``GLOO_SOCKET_IFNAME``, ``NCCL_IB_GID_INDEX``
     - Cluster-specific RDMA/NIC device selection, shipped as ``<example> <changeme>`` (see the :ref:`NCCL device selection note <jax-nccl-devices>`). Their presence marks the config as distributed.
   * - NCCL tuning
     - ``NCCL_DEBUG``, ``NCCL_IB_DISABLE``, ``NCCL_PROTO``, ``NCCL_IB_TC``, ``NCCL_IB_SL``, ``NCCL_CHECKS_DISABLE``, ``NCCL_CROSS_NIC``
     - RoCE/IB transport tuning (``NCCL_IB_DISABLE: 0`` uses IB/RoCE; ``NCCL_IB_TC`` / ``NCCL_IB_SL`` are the RoCE traffic class / service level). On MI355-class images ``RCCL_WARP_SPEED_AUTO`` also lives here.
   * - Transformer-Engine / Composable-Kernel
     - ``NVTE_*``, ``NVTE_CK_*``
     - Fused-attention numerics controls (numerics-sensitive; tune per model/BKC).
   * - JAX coordinator
     - ``JAX_COORDINATOR_PORT``, ``JAX_DISTRIBUTED_INITIALIZATION_TIMEOUT_SECONDS``, ``JAX_DISTRIBUTED_HEARTBEAT_TIMEOUT_SECONDS``
     - JAX distributed coordinator port and the init-rendezvous / heartbeat timeouts.

.. _jax-nccl-devices:

.. note::

  **NCCL device selection (distributed).** Each of
  ``NCCL_IB_HCA``, ``NCCL_SOCKET_IFNAME``, ``GLOO_SOCKET_IFNAME``, and
  ``NCCL_IB_GID_INDEX`` ships with an example value followed by a ``<changeme>``
  tag (e.g. ``"rdma0,...,rdma7 <changeme>"``, ``"eno0 <changeme>"``,
  ``"3 <changeme>"``). Replace the whole value with your cluster's setting — the
  config **hard-exits at load** while any ``container.env`` value still contains
  ``<changeme>``. Discover them with ``ibv_devices`` (HCAs) and ``ip -br link``
  (host interface). ``GLOO_SOCKET_IFNAME`` accepts a single interface only.

``train_params``
================

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Field
     - Example
     - Description
   * - ``hf_model_id``
     - ``NousResearch/Meta-Llama-3-70B``
     - Hugging Face repo the tokenizer is downloaded from. Download is **skipped** when every enabled run uses ``dataset_type: synthetic``.
   * - ``train_script_paths``
     - *(list)*
     - Candidate in-container MaxText entrypoints; the job picks the first that exists (list newest-first, e.g. the v26.4+ path before the v26.3 path).
   * - ``model_id``
     - ``llama3.3-70b``
     - Optional CVS-side label for run names / report / loss-curve filenames. Defaults to ``maxtext_config.model_name``; set it only for a friendlier label.
   * - ``maxtext_branch``
     - ``""``
     - Optional. When set, after container launch the job runs ``git reset --hard`` + ``git checkout <branch>`` in ``maxtext_root`` on every node and verifies the checkout. Empty (default) = use the image's baked-in MaxText unchanged.
   * - ``maxtext_root``
     - ``/workspace/maxtext``
     - MaxText repository root inside the container (used by the branch checkout / install command).
   * - ``maxtext_install_cmd``
     - ``""``
     - Optional shell command run verbatim from ``maxtext_root`` after launch (e.g. ``python3 -m pip install --no-deps -e .`` to make a checked-out branch the imported package). Runs whenever set, even without a branch checkout. Empty (default) = no install step.

``train_params.maxtext_config``
-------------------------------

Written verbatim into the MaxText YAML, so any valid MaxText parameter can be
set here (including ``steps``, ``enable_checkpointing``, and ``tokenizer_path``).
``run_name`` and ``base_output_directory`` are injected by the driver and must
**not** be set here. The most-edited keys:

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Key
     - Example
     - Description
   * - ``base_config``
     - ``base.yml``
     - MaxText base config to inherit defaults from.
   * - ``model_name``
     - ``llama3.3-70b``
     - MaxText model preset (layers/heads/dims). Must exist in the image's MaxText.
   * - ``tokenizer_path``
     - ``{paths.models_dir}/Meta-Llama-70-B``
     - In-container directory the tokenizer is written to / read from (the download target).
   * - ``hardware``
     - ``gpu``
     - Target backend.
   * - ``steps``
     - ``30``
     - Training steps; also drives completion detection and the poll budget.
   * - ``enable_checkpointing``
     - ``false``
     - Whether MaxText writes checkpoints during the run.
   * - ``attention``
     - ``cudnn_flash_te``
     - Attention kernel: ``dot_product`` / ``flash`` / ``cudnn_flash_te`` (use ``dot_product`` for models whose ``attention_type`` is compressed, e.g. DeepSeek V4).
   * - ``dtype`` / ``weight_dtype``
     - ``bfloat16``
     - Compute dtype and master-weight dtype.
   * - ``quantization``
     - ``""``
     - ``""`` = BF16; ``nanoo_fp8`` (MI300X/MI325X, CDNA3) for FP8.
   * - ``dataset_type``
     - ``synthetic``
     - ``synthetic`` = random token ids (no tokenizer or data download; good for throughput/functional runs). For a real loss curve use ``hf`` with a dataset (see the real-data note below); any non-synthetic value requires the tokenizer.
   * - ``per_device_batch_size`` / ``max_target_length``
     - ``3`` / ``8192``
     - Per-GPU batch and sequence length (typically overridden per sweep).
   * - ``remat_policy``
     - ``full``
     - Activation rematerialization (memory vs. recompute trade-off).
   * - ``scan_layers``
     - ``true``
     - Scan the decoder stack (memory/compile savings).
   * - ``ici_*_parallelism``
     - ``ici_fsdp_parallelism: 8``
     - Intra-node parallelism dims (fsdp/data/tensor/sequence/pipeline/expert).
   * - ``dcn_*_parallelism``
     - ``dcn_data_parallelism: -1``
     - Cross-node parallelism dims; ``-1`` fills the remaining mesh axis.

Other passthrough keys seen in the configs: ``packing``, ``megablox`` /
``sparse_matmul`` / ``capacity_factor`` / ``sharding_tolerance`` (MoE kernel
path), ``profiler`` / ``skip_first_n_steps_for_profiler`` / ``profiler_steps``,
``shardy``, ``logits_dot_in_fp32``, ``param_scan_axis``, ``max_segments_per_seq``,
``kv_quant_*``, ``optimizer_memory_host_offload``, ``async_checkpointing``,
``log_period``, ``enable_goodput_recording`` / ``monitor_goodput``.

.. note::

  **Using real data (HuggingFace).** Configs default to ``dataset_type:
  synthetic`` (random tokens; no data/tokenizer download) for throughput and
  functional runs. For a genuine loss curve, set these keys inside
  ``maxtext_config`` (the HF pipeline streams data — no full download):

  .. code:: json

    "dataset_type": "hf",
    "hf_path": "allenai/c4",
    "hf_data_dir": "en",
    "train_split": "train",
    "tokenizer_type": "huggingface"

  ``tokenizer_type: huggingface`` is required so MaxText loads the HF
  ``tokenizer.json`` named by ``hf_model_id`` (the ``sentencepiece`` default
  would mismatch it), and real data triggers the tokenizer download. Do **not**
  place comment (``_``-prefixed) keys inside ``maxtext_config`` — every key there
  is written verbatim to the run YAML and MaxText rejects unknown keys.

``train_params.xla_flags``
--------------------------

A structured ``{flag: value}`` map emitted as a single ``XLA_FLAGS`` env var
(``--<flag>=<value> ...``) into ``container.env``. An empty map omits
``XLA_FLAGS`` entirely (so XLA's own defaults are not clobbered). Notable entries
include ``xla_gpu_autotune_level``, ``xla_gpu_enable_latency_hiding_scheduler``,
and the all-gather / reduce-scatter combine thresholds.

Tests blocks
============

``scaling_baseline`` (distributed)
----------------------------------

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``tokens_per_sec_total``
     - ``0.0``
     - Single-node total tok/s baseline (``tok/s/GPU × GPUs``). ``0.0`` disables the scaling-efficiency metric (record-only).
   * - ``num_nodes``
     - ``1``
     - Nodes used to produce the baseline (``1`` for a single-node baseline).

``convergence``
---------------

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``target_metric``
     - ``auto``
     - ``auto`` uses eval loss when eval runs, else training loss (also ``train_loss`` / ``eval_loss``).
   * - ``target_value``
     - ``0.0``
     - Loss target for ``steps_to_target`` / ``time_to_target``. ``<= 0`` disables convergence (record-only). Eval loss needs ``eval_interval > 0`` and a validation dataset in ``maxtext_config``.

``loss_curve``
--------------

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``sample_every``
     - ``10``
     - Sample training loss every N steps for the slope check.
   * - ``milestone_steps``
     - ``[100, 500, 1000, 5000]``
     - Steps always included in the sampled curve.
   * - ``max_slope``
     - ``0.0``
     - Least-squares slope must be ``< max_slope`` to pass; ``0.0`` = any downward trend passes.
   * - ``enforce``
     - ``true``
     - ``false`` makes the loss-curve check record-only.

``smoke``
---------

The smoke test (``test_smoke``) loads the model and runs a few steps at a small
fixed batch/seqlen in BF16, passing only if no error/NaN signature fires (no
metric checks). A failure gates the rest of the suite. When the block is omitted,
the schema defaults apply (enabled).

.. list-table::
   :widths: 3 2 5
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``enabled``
     - ``true``
     - Runs by default (opt-OUT). Set ``false`` to skip, or use ``-k "not smoke"``.
   * - ``steps``
     - ``5``
     - Steps for the smoke run.
   * - ``per_device_batch_size``
     - ``1``
     - Small fixed batch.
   * - ``max_target_length``
     - ``2048``
     - Small fixed sequence length.

``checkpoint_resume``
---------------------

Opt-in (``enabled: false``). Runs one sweep twice: Phase 1 trains
``steps_before_ckpt`` with checkpointing on (saved at ``checkpoint_period``);
Phase 2 resumes and trains ``steps_after_resume`` more. Passes when Phase 2
restarts at the checkpoint step and the boundary loss matches Phase 1 within
``loss_tolerance``.

.. list-table::
   :widths: 3 2 5
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``enabled``
     - ``false``
     - Opt-in switch.
   * - ``sweep``
     - ``""``
     - Which sweep to exercise (``""`` = first enabled).
   * - ``steps_before_ckpt``
     - ``6``
     - Phase-1 steps (checkpoint saved at ``checkpoint_period``).
   * - ``steps_after_resume``
     - ``6``
     - Phase-2 steps after resuming.
   * - ``checkpoint_period``
     - ``5``
     - Save frequency; must be ``<= steps_before_ckpt`` or Phase 1 saves nothing.
   * - ``loss_tolerance``
     - ``0.1``
     - Max loss delta at the resume boundary.
   * - ``max_save_seconds`` / ``max_load_seconds``
     - ``0.0``
     - I/O time gates for ``checkpoint_save_seconds`` / ``checkpoint_load_seconds``; ``0`` = record-only.
   * - ``delete_ckpt_dir``
     - ``true``
     - Delete the checkpoint dir after the test (``false`` keeps it for inspection).
   * - ``smoke_model_overrides``
     - ``{}``
     - Optional shrink of the model (same tokenizer/vocab) for a fast I/O check.

``error_patterns``
------------------

A ``{name: regex}`` dict scanned in each node's ``training.log`` during polling;
a match fails that sweep's ``test_training_run`` with the matched name. Remove
the block to use the built-in defaults (NCCL, GPU HW faults, assertion/JAX stack
traces, ROCm init errors, Python fatal errors, TF coordination errors,
``RESOURCE_EXHAUSTED``/OOM, and segfaults).

Sweeps and runs
===============

``sweeps`` is a ``{key: overrides}`` map — each entry is one full training run,
and its **key** is both the parsed sweep spec and the **threshold cell key**.
``runs`` is the list of sweep keys to actually execute.

The key is a comma-separated, parseable spec that CVS turns into
``maxtext_config`` overrides for that run:

.. list-table::
   :widths: 2 3 5
   :header-rows: 1

   * - Token
     - Maps to
     - Notes
   * - ``BS``
     - ``per_device_batch_size``
     - Per-GPU batch size.
   * - ``PRECISION``
     - ``quantization``
     - ``BF16`` → ``""``; ``FP8`` → the GPU's FP8 flavor (``nanoo_fp8`` on
       MI300X/MI325X CDNA3, ``fp8`` on MI350-class CDNA4).
   * - ``SL``
     - ``max_target_length``
     - Sequence length.

``dtype`` / ``weight_dtype`` are always ``bfloat16`` (the ``maxtext_config``
default) and are no longer repeated per sweep. Add **any extra** override inside
the sweep's ``{}`` (it takes precedence over the parsed key), e.g. a per-sweep
``steps`` — which CVS also uses for the run's timeout/poll budget and completion
detection.

.. code:: json

  "sweeps": {
    "BS=3,PRECISION=FP8,SL=8192": {
      "_comment": "extra maxtext_config overrides go here, e.g. \"steps\": 300"
    }
  },
  "runs": ["BS=3,PRECISION=FP8,SL=8192"]

Threshold files
===============

Each config has a sibling ``<config-stem>_threshold.json`` referenced by
``threshold_json``. It maps each **sweep key** (cell key) to a dict of
``{metric: spec}``, one spec per line. A metric is gated (PASS/FAIL) only when
``enforce_thresholds: true`` **and** it has a numeric spec whose ``kind`` is not
``info``; otherwise it is recorded. The cell key must match the sweep key
exactly, or the metric falls back to ``RECORD``. Metrics not produced by a run
report ``N/A`` (not a failure).

.. note::

  The shipped threshold values were captured on a **2-node (2N)** run
  (single-node configs: **1N**; the large ``llama-3.1-405b`` and
  ``deepseek-v4-284b`` configs: **4N**). Throughput and step-time scale with the
  GPU count, so if you run a different number of nodes, update the values to the
  appropriate targets for that node count (see the ``_node_count_comment`` in
  each threshold file).

.. code:: json

  "BS=3,PRECISION=BF16,SL=8192": {
    "training.tflops_per_sec_per_gpu": {"kind": "min", "value": 260.0},
    "training.tokens_per_sec_per_gpu": {"kind": "min", "value": 1217.0},
    "training.final_loss": {"kind": "max", "value": 15.0},
    "training.loss_decreased": {"kind": "min", "value": 1},
    "training.step_time_p95_ms": {"kind": "info", "value": 3600000.0}
  }

Threshold kinds
---------------

.. list-table::
   :widths: 2 3 5
   :header-rows: 1

   * - Kind
     - Passes when
     - Notes
   * - ``min``
     - ``actual >= value``
     - Lower bound.
   * - ``max``
     - ``actual <= value``
     - Upper bound.
   * - ``max_ms``
     - ``actual <= value``
     - Upper bound, ``ms`` in the message.
   * - ``min_tok_s``
     - ``actual >= value``
     - Lower bound, ``tok/s`` in the message.
   * - ``within``
     - ``value ± tolerance_pct%``
     - Needs a ``tolerance_pct`` key.
   * - ``min_ratio``
     - ``actual / actuals[reference] >= value``
     - Needs a ``reference`` key.
   * - ``info``
     - always
     - Record-only; keeps a default ``value`` placeholder to calibrate later.

Tracked metrics
---------------

All metrics use the ``training.`` namespace.

.. list-table::
   :widths: 3 1 1 5
   :header-rows: 1

   * - Metric
     - Single
     - Dist.
     - Description
   * - ``tflops_per_sec_per_gpu``
     - ✓
     - ✓
     - TFLOP/s per GPU (typically ``min``).
   * - ``tokens_per_sec_per_gpu``
     - ✓
     - ✓
     - Tokens/s per GPU (typically ``min``).
   * - ``tokens_per_sec_total``
     - ✓
     - ✓
     - Total tokens/s across all GPUs.
   * - ``scaling_efficiency_pct``
     - —
     - ✓
     - Multi-node scaling efficiency % vs. ``scaling_baseline``.
   * - ``step_time_seconds``
     - ✓
     - ✓
     - Mean step wall time (s).
   * - ``step_time_mean_ms`` / ``step_time_p50_ms`` / ``step_time_p95_ms``
     - ✓
     - ✓
     - Step-time mean / p50 / p95 (ms).
   * - ``final_loss``
     - ✓
     - ✓
     - Final training loss (typically ``max``).
   * - ``loss_decreased``
     - ✓
     - ✓
     - ``1`` if loss decreased over the run (``min`` = ``1``).
   * - ``eval_loss``
     - ✓
     - ✓
     - Final eval loss (only when eval is enabled).
   * - ``steps_to_target`` / ``time_to_target_seconds``
     - ✓
     - ✓
     - Convergence metrics (only when ``convergence.target_value > 0``).

To start gating a metric currently marked ``info``: replace ``"kind": "info"``
with ``min`` / ``max`` / etc. and set a calibrated ``value``. Checkpoint I/O
timings (``checkpoint_save_seconds`` / ``checkpoint_load_seconds``) are gated by
the ``checkpoint_resume`` block, not the threshold file.
