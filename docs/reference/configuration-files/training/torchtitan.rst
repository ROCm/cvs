.. meta::
  :description: Configure TorchTitan training configuration files
  :keywords: training, ROCm, CVS, TorchTitan

****************************************
TorchTitan training configuration files
****************************************

TorchTitan configs live under ``cvs/input/config_file/training/torchtitan/``. Each config has a sibling ``*_threshold.json`` referenced by ``threshold_json``. One config file can hold multiple precision sweeps (BF16, FP8, MXFP8, MXFP4) for the same model.

Keys prefixed with ``_`` (for example ``_scaling_baseline_comment``) are inline comments and are ignored by the loader.

Use ``cvs config list training/torchtitan`` to list templates, or ``cvs config copy training/torchtitan/<name>`` to copy one to your working directory.

.. note::

  - Replace every ``<changeme>`` placeholder before running; unresolved placeholders cause a hard exit at startup.
  - ``{user-id}`` in path fields is resolved to the cluster username at load time.

Available configurations
========================

MI355X
------

.. list-table::
   :widths: 5 3 2
   :header-rows: 1

   * - Config
     - Model
     - Mode
   * - ``mi355x_torchtitan_llama-3.1-8b_single.json``
     - Llama-3.1-8B
     - single
   * - ``mi355x_torchtitan_llama-3.3-70b_single.json``
     - Llama-3.3-70B
     - single
   * - ``mi355x_torchtitan_llama-3.3-70b_distributed.json``
     - Llama-3.3-70B
     - distributed
   * - ``mi355x_torchtitan_llama-3.1-405b_distributed.json``
     - Llama-3.1-405B
     - distributed
   * - ``mi355x_torchtitan_deepseek-v2-lite_single.json``
     - DeepSeek-V2-Lite
     - single
   * - ``mi355x_torchtitan_qwen3-32b_single.json``
     - Qwen3-32B
     - single
   * - ``mi355x_torchtitan_mixtral-8x22b_single.json``
     - Mixtral-8x22B
     - single

MI3XX
-----

.. list-table::
   :widths: 5 3 2
   :header-rows: 1

   * - Config
     - Model family
     - Mode
   * - ``mi3xx_torchtitan_llama_single.json`` / ``mi3xx_torchtitan_llama_distributed.json``
     - Llama
     - single / distributed
   * - ``mi3xx_torchtitan_deepseek_single.json`` / ``mi3xx_torchtitan_deepseek_distributed.json``
     - DeepSeek
     - single / distributed
   * - ``mi3xx_torchtitan_qwen3_single.json`` / ``mi3xx_torchtitan_qwen3_distributed.json``
     - Qwen3
     - single / distributed

Cluster-specific edits
======================

.. list-table::
   :widths: 2 2 4
   :header-rows: 1

   * - Where
     - Field
     - Change to
   * - ``container``
     - ``image``
     - Your TorchTitan ROCm image tag, accessible on all nodes
   * - ``config``
     - ``hf_token_file``
     - Path to your Hugging Face token file on the nodes
   * - ``config``
     - ``log_dir`` / ``scripts_dir`` / ``data_cache_dir``
     - Replace ``{user-id}`` with your username
   * - ``config``
     - ``nnodes``, ``master_address``
     - Node count and head-node IP (**distributed only**)
   * - ``config``
     - ``nic_type``, ``nccl_ib_hca_list``, ``nccl_socket_ifname``
     - Your NIC family and RDMA device names (**distributed only**)
   * - ``scaling_baseline``
     - ``tokens_per_sec_total``
     - Measured single-node total tok/s; ``0.0`` disables scaling efficiency (**distributed only**)
   * - Threshold JSON
     - per-metric bounds
     - Calibrated PASS/FAIL limits for your hardware

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
     - ``torchtitan_single`` / ``torchtitan_distributed``
     - Selects the test suite. Must match ``cvs run <name>``.
   * - ``gpu_arch``
     - ``MI355X``
     - GPU architecture label (informational).
   * - ``enforce_thresholds``
     - ``true``
     - If ``false``, ``test_metric`` logs results but does not fail.
   * - ``threshold_json``
     - ``mi355x_torchtitan_llama-3.1-8b_single_threshold.json``
     - Companion threshold file in the same directory.
   * - ``loss_curve``
     - (block)
     - Milestone-step loss decrease validation.
   * - ``convergence``
     - (block)
     - Steps/time to reach a target loss (informational).
   * - ``checkpoint``
     - (block)
     - Optional save/resume checkpoint test.
   * - ``scaling_baseline``
     - (block)
     - Single-node baseline for scaling efficiency (**distributed only**).
   * - ``config``
     - (block)
     - Runtime paths, NCCL, and training settings.
   * - ``model_params``
     - (block)
     - Model architecture defaults; sweep combos override fields.
   * - ``container``
     - (block)
     - Docker container settings.
   * - ``sweep``
     - (block)
     - Training combinations and ordered run list.

``config`` block
================

.. list-table::
   :widths: 3 2 5
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``hf_token_file``
     - ``/home/{user-id}/.hf_token``
     - Hugging Face access token file path.
   * - ``log_dir``
     - ``/home/{user-id}/LOGS/torchtitan``
     - Training log output directory.
   * - ``scripts_dir``
     - ``/home/{user-id}/SCRIPTS/torchtitan``
     - Per-node wrapper scripts directory.
   * - ``data_cache_dir``
     - ``/home/{user-id}/cache``
     - Tokenizer and dataset cache directory.
   * - ``torchtitan_root``
     - ``/workspace/torchtitan``
     - TorchTitan path inside the container.
   * - ``training_iterations``
     - ``"10"``
     - Training iterations per combo.
   * - ``nnodes``
     - ``"1"`` / ``<changeme>``
     - Node count; must match the cluster file.
   * - ``master_address``
     - ``127.0.0.1`` / ``<changeme>``
     - Head-node IP for distributed coordination.
   * - ``nic_type``
     - ``thor2``
     - NIC family; ``thor2`` triggers Broadcom RDMA-lib copy.
   * - ``nccl_ib_hca_list``
     - ``<changeme>``
     - Comma-separated RDMA HCA list.
   * - ``nccl_socket_ifname``
     - ``ensf1np1``
     - Control-plane interface name.
   * - ``use_generated_config``
     - ``"True"``
     - Auto-generate TorchTitan TOML from ``model_params`` when ``"True"``.

``model_params`` block
======================

Defaults applied to every sweep combo; individual combos override matching keys.

.. list-table::
   :widths: 3 5
   :header-rows: 1

   * - Field
     - Description
   * - ``model_name``
     - Friendly name used in log paths and labels.
   * - ``hf_model_name``
     - Hugging Face repo ID (for example ``meta-llama/Llama-3.1-8B``).
   * - ``sequence_length``
     - Sequence length in tokens.
   * - ``dataset``
     - Dataset name (for example ``c4``).
   * - ``lr`` / ``warmup_steps``
     - Learning rate and warmup steps.
   * - ``tensor_parallel_degree`` / ``pipeline_parallel_degree`` / ``context_parallel_degree`` / ``expert_parallel_degree``
     - Parallelism degrees (TP, PP, CP, EP for MoE).
   * - ``data_parallel_shard_degree``
     - Data parallel (FSDP) degree.
   * - ``precision``
     - Default precision; overridden per sweep combo.

``container`` block
===================

.. list-table::
   :widths: 3 5
   :header-rows: 1

   * - Field
     - Description
   * - ``lifetime``
     - ``per_run`` — launched once per session, torn down after.
   * - ``name``
     - Docker container name.
   * - ``image``
     - **Required** — Docker image URI.
   * - ``runtime.args.volumes``
     - Host paths volume-mounted into the container.
   * - ``runtime.args.devices``
     - Host devices exposed (``/dev/kfd``, ``/dev/dri`` for AMD GPUs).

Distributed configs additionally mount the Broadcom RDMA library and expose ``/dev/infiniband/rdma_cm``.

Optional blocks
===============

``loss_curve``
  Sample loss every ``sample_every`` steps and check a decreasing trend at ``milestone_steps``. Set ``enforce: true`` to gate PASS/FAIL.

``convergence``
  Track steps and wall-clock time to reach ``target_value`` loss. Informational only; never gates PASS/FAIL.

``checkpoint``
  When ``enforce: true``, run a two-phase save/resume test with step-counter and loss-continuity validation.

``scaling_baseline`` (distributed only)
  ``tokens_per_sec_total`` is your measured single-node total tok/s. Used to compute ``training.scaling_efficiency_pct``.

Sweeps
======

Each entry in ``sweep.combinations`` is one parametrized training run. ``sweep.runs`` is the ordered list of combo IDs to execute; omit combos from ``runs`` to skip them without editing ``combinations``.

.. list-table::
   :widths: 3 5
   :header-rows: 1

   * - Combo field
     - Description
   * - ``name``
     - Human-readable label (used in reports).
   * - ``global_batch_size`` / ``micro_batch_size``
     - Batch sizes for this combo.
   * - ``precision``
     - Precision override (``BF16``, ``FP8``, ``MXFP8``, ``MXFP4``, etc.).

Threshold files
===============

Threshold files map each sweep combo to per-metric pass/fail limits. Cell keys use the format ``MBS=<mbs>,GBS=<gbs>,PRECISION=<precision>``.

.. list-table::
   :widths: 3 5
   :header-rows: 1

   * - Metric
     - Description
   * - ``training.throughput_per_gpu``
     - TFLOP/s per GPU.
   * - ``training.tokens_per_sec``
     - Tokens per second (total across all GPUs).
   * - ``training.elapsed_time_per_iteration``
     - Wall time per training step (ms).
   * - ``training.mem_usage``
     - GPU memory usage.
   * - ``training.scaling_efficiency_pct``
     - Multi-node scaling efficiency vs single-node baseline (**distributed only**).

Threshold kinds: ``min`` (actual ≥ value), ``max`` (actual ≤ value), ``min_ratio`` (actual / reference ≥ value).

How to run: :doc:`/how-to/test-suites/training/torchtitan`.
