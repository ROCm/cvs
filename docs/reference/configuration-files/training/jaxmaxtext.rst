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

  JAX training in CVS is now **jaxmaxtext**. The legacy ``jax`` suites
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

  - Parameters with the ``<changeme>`` value must be modified to your setup;
    distributed runs hard-exit at config load until the ``nccl.*`` placeholders
    (``ib_hca``, ``socket_ifname``, ``gloo_socket_ifname``, ``ib_gid_index``)
    are set.
  - ``{user-id}`` resolves to the cluster/OS username at runtime.
  - Keys prefixed with ``_`` (e.g. ``_nccl_comment``, ``_example_ib_hca``) are
    inline comments/examples and are ignored by the loader.

The suite/lifecycle reference is in ``cvs/tests/training/jaxmaxtext/README.md``.
The config files themselves live in
``cvs/input/config_file/training/jaxmaxtext/`` (each config plus a sibling
``_threshold.json``); this page documents every block and the threshold format.

Available configurations
=========================

Config files follow the naming pattern ``<gpu>_jaxmaxtext_<model>_<mode>.json``.
Each config has a sibling ``_threshold.json`` referenced by ``threshold_json``.

.. list-table::
   :widths: 4 2 2 2 2
   :header-rows: 1

   * - Model
     - MI300X
     - MI325X
     - MI355X
     - Mode
   * - Llama 3.3 70B
     - ✓
     - ✓
     - ✓
     - single (MI300X), distributed
   * - Llama 3.1 8B
     - ✓
     - ✓
     - ✓
     - distributed
   * - Llama 3.1 405B
     - —
     - ✓
     - ✓
     - distributed only
   * - DeepSeek V2 Lite
     - ✓
     - ✓
     - ✓
     - distributed

Single-node configs set ``training.distributed: false`` and run with
``jaxmaxtext_single``; distributed configs set ``training.distributed: true``,
add the ``test_setup_rdma`` stage, and require the ``nccl.*`` network fields.
Use a single-node config with ``jaxmaxtext_single`` and a distributed config
with ``jaxmaxtext_distributed`` — the flag must match the suite.

Example configuration
=====================

The blocks below are common to every JAX MaxText config; only ``model.id``,
``maxtext_config.model_name``, the parallelism dims, and the sweeps differ
between models. A representative distributed config
(``mi325x_jaxmaxtext_llama-3.3-70b_distributed.json``):

.. dropdown:: ``mi325x_jaxmaxtext_llama-3.3-70b_distributed.json`` (abridged)

  .. code:: json

    {
      "schema_version": 1,
      "framework": "jaxmaxtext",
      "gpu_arch": "mi325x",
      "enforce_thresholds": true,
      "threshold_json": "mi325x_jaxmaxtext_llama-3.3-70b_distributed_threshold.json",
      "paths": {
        "shared_fs": "/home/{user-id}",
        "models_dir": "{shared_fs}/cache/maxtext",
        "log_dir": "{shared_fs}/LOGS/jaxmaxtext",
        "hf_token_file": "{shared_fs}/.hf_token",
        "temp_dir": "/tmp/{user-id}/jaxmaxtext"
      },
      "model": { "id": "llama3.3-70b", "remote": 0, "precision": "bfloat16" },
      "container": {
        "lifetime": "per_run",
        "name": "rocm-jaxmaxtext-llama3.3-70b",
        "image": "rocm/jax-training:maxtext-v26.4",
        "runtime": { "name": "docker", "args": { "network": "host", "ipc": "host", "privileged": true, "shm-size": "256G", "volumes": ["..."] } }
      },
      "training": {
        "distributed": true,
        "gpus_per_node": 8,
        "steps": 30,
        "enable_checkpointing": false,
        "maxtext_config": {
          "base_config": "base.yml",
          "model_name": "llama3.3-70b",
          "hardware": "gpu",
          "attention": "cudnn_flash_te",
          "dtype": "bfloat16",
          "weight_dtype": "bfloat16",
          "quantization": "",
          "dataset_type": "synthetic",
          "per_device_batch_size": 3,
          "max_target_length": 8192,
          "remat_policy": "full",
          "scan_layers": true,
          "ici_fsdp_parallelism": 8,
          "dcn_data_parallelism": -1
        },
        "tokenizer": { "hf_model_id": "NousResearch/Meta-Llama-3-70B", "tokenizer_path": "{paths.models_dir}/Meta-Llama-70-B" },
        "nic_type": "thor2",
        "rdma_lib": { "host_source_file": "...", "container_mount_file": "...", "container_dest_file": "..." },
        "env_vars": { "NNODES": "auto", "NCCL_IB_DISABLE": "0", "NCCL_IB_TC": "41", "NCCL_IB_SL": "0", "...": "..." },
        "xla_flags": { "xla_gpu_autotune_level": "0", "...": "..." },
        "nccl": { "ib_hca_list": "<changeme>", "ib_hca": "<changeme>", "socket_ifname": "<changeme>", "gloo_socket_ifname": "<changeme>", "ib_gid_index": "<changeme>" },
        "jax_distributed": { "coordinator_ip": "auto", "coordinator_port": "12346" },
        "scaling_baseline": { "tokens_per_sec_total": 394000.0, "num_nodes": 1 },
        "convergence": { "target_metric": "auto", "target_value": 10.0 },
        "loss_curve": { "sample_every": 10, "milestone_steps": [100, 500, 1000, 5000], "max_slope": 0.0, "enforce": true },
        "smoke": { "enabled": true, "steps": 5, "per_device_batch_size": 1, "max_target_length": 2048 },
        "checkpoint_resume": { "enabled": false, "steps_before_ckpt": 6, "steps_after_resume": 6, "checkpoint_period": 5, "loss_tolerance": 0.1, "delete_ckpt_dir": true },
        "error_patterns": { "NCCL ERROR": "NCCL ERROR|NCCL timeout", "...": "..." },
        "sweeps": [ { "name": "NNODES=2,STEPS=30,PRECISION=BF16,BATCH=3,GBS=48,SEQLEN=8192", "maxtext_overrides": { "per_device_batch_size": 3, "max_target_length": 8192, "quantization": "" } } ],
        "enabled_sweep_list": ["NNODES=2,STEPS=30,PRECISION=BF16,BATCH=3,GBS=48,SEQLEN=8192"]
      }
    }

Top-level fields
================

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Field
     - Example
     - Description
   * - ``schema_version``
     - ``1``
     - Config schema version. Must be ``1``.
   * - ``framework``
     - ``jaxmaxtext``
     - Framework selector.
   * - ``gpu_arch``
     - ``mi325x``
     - GPU architecture label (informational).
   * - ``enforce_thresholds``
     - ``true``
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
     - Training log output directory.
   * - ``hf_token_file``
     - ``{shared_fs}/.hf_token``
     - Hugging Face token file (for the tokenizer download).
   * - ``temp_dir``
     - ``/tmp/{user-id}/jaxmaxtext``
     - Host-user-namespaced in-container scratch for launcher scripts / MaxText YAML. Keep ``{user-id}`` so shared nodes never collide on ``/tmp/root``.

``model``
---------

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Field
     - Example
     - Description
   * - ``id``
     - ``llama3.3-70b``
     - Model id used in run names / labels.
   * - ``remote``
     - ``0``
     - ``0`` = weights/tokenizer already cached locally.
   * - ``precision``
     - ``bfloat16``
     - Label only (the effective precision is set in ``maxtext_config`` / sweeps).

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
     - ``rocm/jax-training:maxtext-v26.4``
     - **Required** — the MaxText/JAX ROCm image present on all nodes.
   * - ``runtime.args``
     - *(see snippet)*
     - Docker args: ``network: host``, ``ipc: host``, ``privileged: true``, ``shm-size``, ``ulimit``, and ``volumes``. Distributed configs mount ``/dev/infiniband`` and the NIC ``libibverbs`` provider (``:ro``); ``volumes`` also bind-mounts the home dir and the training-output dir.

``training`` block
==================

Top-level scalar fields of the ``training`` block. The nested blocks
(``maxtext_config``, ``tokenizer``, ``nccl``, …) are documented in their own
sections below.

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``distributed``
     - ``true``
     - ``true`` = multi-node (adds the ``test_setup_rdma`` stage); ``false`` = single-node.
   * - ``gpus_per_node``
     - ``8``
     - GPUs per node; ``num_gpus = num_nodes × gpus_per_node`` feeds ``tokens_per_sec_total`` and scaling efficiency. Do not assume a fixed topology.
   * - ``verify_dmesg``
     - ``true``
     - Scan host ``dmesg`` on all nodes for GPU/HW/kernel faults over the run window. Set ``false`` on clusters without passwordless ``sudo`` for ``dmesg``.
   * - ``steps``
     - ``30``
     - Training steps; also drives completion detection and the poll budget.
   * - ``enable_checkpointing``
     - ``false``
     - Whether MaxText writes checkpoints during the run.
   * - ``train_script_paths``
     - *(list)*
     - Candidate in-container MaxText entrypoints; the job picks the first that exists (list newest-first, e.g. the v26.4+ path before the v26.3 path).
   * - ``nic_type``
     - ``thor2``
     - NIC family; ``thor2`` (Broadcom) enables the backup RDMA-lib copy, ``none`` skips it.
   * - ``error_patterns``
     - *(dict)*
     - ``{name: regex}`` scanned in each node's ``training.log`` during polling; a match fails that sweep's ``test_training_run``. Remove the block to use the built-in defaults.

``maxtext_config``
------------------

Written verbatim into the MaxText YAML, so any valid MaxText parameter can be
set. ``steps``, ``enable_checkpointing``, ``run_name``, ``base_output_directory``,
and ``tokenizer_path`` are injected by the driver and must **not** be set here.
The most-edited keys:

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
   * - ``hardware``
     - ``gpu``
     - Target backend.
   * - ``attention``
     - ``cudnn_flash_te``
     - Attention kernel: ``dot_product`` / ``flash`` / ``cudnn_flash_te`` (use ``dot_product`` for models whose ``attention_type`` is compressed, e.g. DeepSeek V4).
   * - ``dtype`` / ``weight_dtype``
     - ``bfloat16``
     - Compute dtype and master-weight dtype.
   * - ``quantization``
     - ``""``
     - ``""`` = BF16; ``nanoo_fp8`` (MI300X/MI325X, CDNA3) or ``fp8`` (MI355X/MI350X, CDNA4) for FP8.
   * - ``dataset_type``
     - ``synthetic``
     - ``synthetic`` = random token ids (no tokenizer download). Any other value (or unset → MaxText tfds/C4) requires the tokenizer.
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
   * - ``opt_type``
     - ``adamw``
     - Optimizer (``adamw`` default; ``sgd`` used for some from-scratch synthetic runs).

Other passthrough keys seen in the configs: ``packing``, ``megablox`` /
``sparse_matmul`` / ``capacity_factor`` (MoE kernel path), ``profiler``,
``shardy``, ``logits_dot_in_fp32``, ``param_scan_axis``, ``max_segments_per_seq``,
``kv_quant_*``, ``optimizer_memory_host_offload``, ``async_checkpointing``,
``enable_goodput_recording`` / ``monitor_goodput``.

``tokenizer``
-------------

.. list-table::
   :widths: 3 4 5
   :header-rows: 1

   * - Field
     - Example
     - Description
   * - ``hf_model_id``
     - ``NousResearch/Meta-Llama-3-70B``
     - Hugging Face repo the tokenizer is downloaded from. Download is **skipped** when every enabled run uses ``dataset_type: synthetic``.
   * - ``tokenizer_path``
     - ``{paths.models_dir}/Meta-Llama-70-B``
     - In-container directory the tokenizer is written to.

``rdma_lib`` (backup)
---------------------

Distributed only, and used **only** when a direct read-only ``.so`` bind-mount
is not allowed. In that case the host lib is mounted as ``<name>.so.host``
(``container_mount_file``) and copied to the real ``<name>.so``
(``container_dest_file``) inside the container after launch. When the ``.so`` is
bind-mounted ``:ro`` directly (the default in ``container.runtime.args.volumes``),
this block is unused.

.. list-table::
   :widths: 3 5
   :header-rows: 1

   * - Field
     - Description
   * - ``host_source_file``
     - Path to the NIC's ``libibverbs`` provider on the host.
   * - ``container_mount_file``
     - Where the host lib is mounted inside the container (``…so.host``).
   * - ``container_dest_file``
     - Real ``.so`` path the mounted file is copied to at launch.

``env_vars`` and ``xla_flags``
------------------------------

``env_vars`` is a dict exported before training; ``xla_flags`` is emitted as a
single ``XLA_FLAGS`` string (``--<key>=<value>``). Notable entries:

.. list-table::
   :widths: 3 5
   :header-rows: 1

   * - Variable
     - Description
   * - ``NNODES``
     - ``auto`` — replaced at runtime with the cluster node count.
   * - ``XLA_PYTHON_CLIENT_MEM_FRACTION``
     - Fraction of GPU memory JAX may allocate (e.g. ``0.97``).
   * - ``NCCL_IB_DISABLE``
     - ``0`` uses IB/RoCE; ``1`` forces the TCP socket path.
   * - ``NCCL_IB_TC`` / ``NCCL_IB_SL``
     - RoCE traffic class / service level. (``NCCL_IB_GID_INDEX`` is driven by ``nccl.ib_gid_index`` instead — do not set it here.)
   * - ``NVTE_*`` / ``NVTE_CK_*``
     - Transformer-Engine / Composable-Kernel fused-attention controls (numerics-sensitive; tune per model/BKC).
   * - ``xla_gpu_autotune_level``
     - XLA GEMM autotuning level.

``nccl`` (distributed)
----------------------

Cluster-specific RDMA/NIC devices. Each field ships as ``<changeme>`` with a
sibling ``_example_*`` value; distributed runs **hard-exit at config load** until
they are set. Discover them with ``ibv_devices`` (HCAs) and ``ip -br link``
(host interface).

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Field
     - Example
     - Description / export
   * - ``ib_hca_list``
     - ``rdma0,…,rdma7``
     - Comma-separated RDMA HCA list → ``NCCL_IB_HCA_LIST``.
   * - ``ib_hca``
     - ``rdma0,…,rdma7``
     - Primary HCA(s) → ``NCCL_IB_HCA``.
   * - ``socket_ifname``
     - ``eno0``
     - Control interface → ``NCCL_SOCKET_IFNAME`` (accepts a comma list).
   * - ``gloo_socket_ifname``
     - ``eno0``
     - Gloo control interface → ``GLOO_SOCKET_IFNAME`` (single interface only).
   * - ``ib_gid_index``
     - ``3``
     - RoCE GID index → ``NCCL_IB_GID_INDEX``.

``jax_distributed``
-------------------

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``coordinator_ip``
     - ``auto``
     - ``auto`` uses the first node in the cluster ``node_dict``; or set a specific IP.
   * - ``coordinator_port``
     - ``12346``
     - JAX coordinator port.
   * - ``initialization_timeout_seconds``
     - ``1800``
     - Distributed init rendezvous timeout.
   * - ``heartbeat_timeout_seconds``
     - ``900``
     - Coordination-service heartbeat timeout.

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
metric checks). A failure gates the rest of the suite.

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

Sweeps
======

Each sweep is one full training run; ``maxtext_overrides`` merges onto
``maxtext_config`` for that run. ``enabled_sweep_list`` selects which sweeps to
run. The sweep ``name`` is also the **threshold cell key**.

.. list-table::
   :widths: 3 5
   :header-rows: 1

   * - Field
     - Description
   * - ``sweeps[].name``
     - Cell key, format ``NNODES=..,STEPS=..,PRECISION=..,BATCH=..,GBS=..,SEQLEN=..``. ``NNODES`` (cluster), ``STEPS`` (``training.steps``) and ``GBS`` (= ``per_device_batch_size × total GPUs``) are labels only.
   * - ``sweeps[].maxtext_overrides``
     - Per-run overrides merged onto ``maxtext_config`` (typically ``per_device_batch_size``, ``max_target_length``, ``dtype`` / ``weight_dtype``, ``quantization``).
   * - ``enabled_sweep_list``
     - Subset of sweep ``name`` s to actually run.

.. code:: json

  "sweeps": [
    {
      "name": "NNODES=2,STEPS=30,PRECISION=FP8,BATCH=3,GBS=48,SEQLEN=8192",
      "maxtext_overrides": {
        "per_device_batch_size": 3,
        "max_target_length": 8192,
        "dtype": "bfloat16",
        "weight_dtype": "bfloat16",
        "quantization": "nanoo_fp8"
      }
    }
  ],
  "enabled_sweep_list": ["NNODES=2,STEPS=30,PRECISION=FP8,BATCH=3,GBS=48,SEQLEN=8192"]

Threshold files
===============

Each config has a sibling ``<config-stem>_threshold.json`` referenced by
``threshold_json``. It maps each **sweep name** (cell key) to a dict of
``{metric: spec}``. A metric is gated (PASS/FAIL) only when
``enforce_thresholds: true`` **and** it has a numeric spec whose ``kind`` is not
``info``; otherwise it is recorded. The cell key must match the sweep ``name``
exactly (including ``NNODES``), or the metric falls back to ``RECORD``. Metrics
not produced by a run report ``N/A`` (not a failure).

.. code:: json

  "NNODES=2,STEPS=30,PRECISION=BF16,BATCH=3,GBS=48,SEQLEN=8192": {
    "training.tflops_per_sec_per_gpu": { "kind": "min", "value": 260.0 },
    "training.tokens_per_sec_per_gpu": { "kind": "min", "value": 1217.0 },
    "training.final_loss":             { "kind": "max", "value": 15.0 },
    "training.loss_decreased":         { "kind": "min", "value": 1 },
    "training.step_time_p95_ms":       { "kind": "info", "value": 3600000.0 }
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
