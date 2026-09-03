.. meta::
  :description: Configure the variables in the Megatron training configuration files
  :keywords: training, ROCm, install, cvs, Megatron,

*************************************
Megatron training configuration files
*************************************

JSON configs and sibling ``*_threshold.json`` files for ``megatron_single`` and ``megatron_distributed``. One file is one GPU architecture, model, and mode: ``mi{gpu}_megatron_{model}_{single|distributed}.json``. Match ``framework`` to the suite you run. Keep the sibling threshold file next to the config (``threshold_json`` is resolved relative to the config file).

How to run the suites: :doc:`/how-to/test-suites/training/megatron`.

Use ``cvs config list training/megatron`` to list available templates, or
``cvs config copy training/megatron/<name>`` to copy one to your working
directory. Copy the sibling ``*_threshold.json`` into the same directory as the
suite config.

Backends
========

The suite selects the training backend from ``container.image`` (substring ``primus``, case-insensitive):

* **Megatron-LM** — image name does not contain ``primus``. Training scripts live under ``config.megatron_root``. Log files use ``<log_dir>/megatron-logs/<combo_id>/out-node<N>/training.log``.
* **Primus** — image name contains ``primus``. In-image YAML lives under ``examples/megatron/configs/{gpu_arch}/``. Log files use ``<log_dir>/primus-logs/<combo_id>/out-node<N>/training.log``.

.. note::

  - Parameters with the ``<changeme>`` value must have that value modified to your specifications. Unresolved placeholders cause a hard exit at load time.
  - ``{user-id}`` will be resolved to the cluster username (or the local OS user as fallback). You can also set this value yourself.
  - Keys prefixed with ``_`` (for example ``_checkpoint_comment``) are inline comments and are ignored by the loader.
  - ``sweep.runs`` is required. It must be a subset of (or equal to) the keys in ``sweep.combinations``. Omitting it fails config load.

Available configurations
========================

Config files follow the naming pattern ``<gpu>_megatron_<model>_<mode>.json``. The table below lists all available combinations. Each model also has a corresponding ``_threshold.json`` file referenced by ``threshold_json``.

.. list-table::
   :widths: 4 2 2 2 2
   :header-rows: 1

   * - Model
     - MI300X
     - MI325X
     - MI355X
     - Mode
   * - Llama 3.1 8B
     - ✓
     - ✓
     - ✓
     - single, distributed
   * - Llama 3.3 70B
     - ✓
     - ✓
     - ✓
     - single, distributed
   * - DeepSeek V2 Lite
     - ✓
     - ✓
     - ✓
     - single, distributed
   * - Llama 3.1 405B
     - ✓
     - ✓
     - ✓
     - distributed only

Single-node configs set ``framework: megatron_single`` and use ``nnodes: 1`` with ``master_address: 127.0.0.1``. Distributed configs set ``framework: megatron_distributed``, require the network fields (``nic_type``, ``nccl_ib_hca_list``, etc.), and add a ``scaling_baseline`` section and ``checkpoint_dir`` to the ``checkpoint`` block.

Leftover files ``mi3xx_megatron_llama_*.json`` and ``mi35x_megatron_llama_single.json`` are nested-schema configs for the legacy ``megatron_llama3_1_*`` suites only. Do not pass them to ``megatron_single`` or ``megatron_distributed``.

Required edits
==============

Set these before a run (full field tables are under `Common parameters`_):

* ``container.image`` — Megatron-LM or Primus ROCm image on all nodes.
* ``config.training_iterations`` — training steps (for example ``"30"``).
* ``config.hf_token_file`` — Hugging Face token path on the nodes.
* ``config.nccl_socket_ifname`` / ``config.gloo_socket_ifname`` — control NIC.
* ``sweep.runs`` — combo IDs to execute.
* **Distributed only:** ``config.nnodes``, ``config.master_address``, ``config.nic_type``, ``config.nccl_ib_hca_list`` / ``nccl_ib_hca``. When ``checkpoint.enforce`` is ``true``, also set ``checkpoint.checkpoint_dir`` and replace the last ``<changeme>:<changeme>`` volume with that shared path.

Top-level fields
================

These fields appear at the root of every config file.

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
     - ``megatron_single`` / ``megatron_distributed``
     - Selects the test class. Use ``megatron_single`` for one-node runs and ``megatron_distributed`` for multi-node runs.
   * - ``gpu_arch``
     - ``MI300X``
     - GPU architecture string used for logging and Primus EXP config path resolution.
   * - ``enforce_thresholds``
     - ``true``
     - If ``false``, threshold checks in ``test_metric`` log results but do not fail the test.
   * - ``threshold_json``
     - ``mi300x_megatron_llama-3.1-8b_single_threshold.json``
     - Filename of the companion threshold file, looked up in the same directory as the config file.

Model configurations
====================

Each model section below shows only the ``model_params`` and ``sweep`` blocks, which are the parts that differ between models. All other sections (``config``, ``container``, ``smoke``, ``loss_curve``, ``convergence``, ``checkpoint``, ``scaling_baseline``) are identical in structure across models and are documented in `Common parameters`_.

Llama 3.1 8B
------------

Available as ``mi300x_megatron_llama-3.1-8b_{single,distributed}.json``, ``mi325x_…``, ``mi355x_…``.

.. dropdown:: ``mi300x_megatron_llama-3.1-8b_single.json`` (representative)

  .. code:: json

    {
      "schema_version": 1,
      "framework": "megatron_single",
      "gpu_arch": "MI300X",
      "enforce_thresholds": true,
      "threshold_json": "mi300x_megatron_llama-3.1-8b_single_threshold.json",
      "model_params": {
        "model_name": "llama3.1_8B",
        "tokenizer_model": "meta-llama/Llama-3.1-8B",
        "model_size": "8",
        "sequence_length": "8192",
        "recompute": "0",
        "fsdp": "0",
        "tensor_parallelism": "1",
        "pipeline_parallelism": "1",
        "precision": "FP8"
      },
      "sweep": {
        "combinations": {
          "llama3_1_8b-mi300x-bs128-mbs4-fp8": {
            "name": "llama3_1_8b_mbs4_gbs128_FP8",
            "global_batch_size": "128",
            "micro_batch_size": "4",
            "precision": "FP8"
          },
          "llama3_1_8b-mi300x-bs128-mbs4-bf16": {
            "name": "llama3_1_8b_mbs4_gbs128_BF16",
            "global_batch_size": "128",
            "micro_batch_size": "4",
            "precision": "BF16"
          },
          "llama3_1_8b-mi300x-bs128-mbs4-mxfp4": {
            "name": "llama3_1_8b_mbs4_gbs128_MXFP4",
            "global_batch_size": "128",
            "micro_batch_size": "4",
            "precision": "MXFP4"
          },
          "llama3_1_8b-mi300x-bs128-mbs4-mxfp8": {
            "name": "llama3_1_8b_mbs4_gbs128_MXFP8",
            "global_batch_size": "128",
            "micro_batch_size": "4",
            "precision": "MXFP8"
          }
        },
        "runs": [
          "llama3_1_8b-mi300x-bs128-mbs4-fp8",
          "llama3_1_8b-mi300x-bs128-mbs4-bf16",
          "llama3_1_8b-mi300x-bs128-mbs4-mxfp4",
          "llama3_1_8b-mi300x-bs128-mbs4-mxfp8"
        ]
      }
    }

``model_params``
~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Parameter
     - Value
     - Description
   * - ``model_name``
     - ``llama3.1_8B``
     - Used in log labels and report filenames.
   * - ``tokenizer_model``
     - ``meta-llama/Llama-3.1-8B``
     - HuggingFace repo ID for the tokenizer and model weights.
   * - ``model_size``
     - ``8``
     - Model size in billions of parameters.
   * - ``sequence_length``
     - ``8192``
     - Maximum context length.
   * - ``tensor_parallelism``
     - ``1``
     - Fits on a single GPU; no tensor splitting needed.
   * - ``pipeline_parallelism``
     - ``1``
     - Single pipeline stage.
   * - ``precision``
     - ``FP8``
     - Default precision (overridden per sweep cell).


Llama 3.3 70B
-------------

Available as ``mi300x_megatron_llama-3.3-70b_{single,distributed}.json``, ``mi325x_…``, ``mi355x_…``.

.. dropdown:: ``mi300x_megatron_llama-3.3-70b_single.json`` (representative)

  .. code:: json

    {
      "schema_version": 1,
      "framework": "megatron_single",
      "gpu_arch": "MI300X",
      "enforce_thresholds": true,
      "threshold_json": "mi300x_megatron_llama-3.3-70b_single_threshold.json",
      "model_params": {
        "model_name": "llama3.3_70B",
        "tokenizer_model": "meta-llama/Llama-3.3-70B-Instruct",
        "model_size": "70",
        "sequence_length": "8192",
        "recompute": "0",
        "fsdp": "0",
        "tensor_parallelism": "8",
        "pipeline_parallelism": "1",
        "precision": "FP8"
      },
      "sweep": {
        "combinations": {
          "llama3_3_70b-mi300x-bs96-mbs3-fp8": {
            "name": "llama3_3_70b_mbs3_gbs96_FP8",
            "global_batch_size": "96",
            "micro_batch_size": "3",
            "precision": "FP8"
          },
          "llama3_3_70b-mi300x-bs96-mbs3-bf16": {
            "name": "llama3_3_70b_mbs3_gbs96_BF16",
            "global_batch_size": "96",
            "micro_batch_size": "3",
            "precision": "BF16"
          }
        },
        "runs": [
          "llama3_3_70b-mi300x-bs96-mbs3-fp8",
          "llama3_3_70b-mi300x-bs96-mbs3-bf16"
        ]
      }
    }

``model_params``
~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Parameter
     - Value
     - Description
   * - ``model_name``
     - ``llama3.3_70B``
     - Used in log labels and report filenames.
   * - ``tokenizer_model``
     - ``meta-llama/Llama-3.3-70B-Instruct``
     - HuggingFace repo ID for the tokenizer and model weights.
   * - ``model_size``
     - ``70``
     - Model size in billions of parameters.
   * - ``sequence_length``
     - ``8192``
     - Maximum context length.
   * - ``tensor_parallelism``
     - ``8``
     - Splits the model across all 8 GPUs on the node.
   * - ``pipeline_parallelism``
     - ``1``
     - Single pipeline stage.
   * - ``precision``
     - ``FP8``
     - Default precision (overridden per sweep cell).


DeepSeek V2 Lite
----------------

Available as ``mi300x_megatron_deepseek-v2-lite_{single,distributed}.json``, ``mi325x_…``, ``mi355x_…``.

.. dropdown:: ``mi300x_megatron_deepseek-v2-lite_single.json`` (representative)

  .. code:: json

    {
      "schema_version": 1,
      "framework": "megatron_single",
      "gpu_arch": "MI300X",
      "enforce_thresholds": true,
      "threshold_json": "mi300x_megatron_deepseek-v2-lite_single_threshold.json",
      "model_params": {
        "model_name": "deepseek_v2_lite",
        "tokenizer_model": "deepseek-ai/DeepSeek-V2-Lite",
        "model_size": "16",
        "sequence_length": "4096",
        "recompute": "0",
        "fsdp": "0",
        "tensor_parallelism": "1",
        "pipeline_parallelism": "1",
        "precision": "BF16"
      },
      "sweep": {
        "combinations": {
          "deepseek_v2_lite-mi300x-bs128-mbs4-bf16": {
            "name": "deepseek_v2_lite_mbs4_gbs128_BF16",
            "global_batch_size": "128",
            "micro_batch_size": "4",
            "precision": "BF16"
          },
          "deepseek_v2_lite-mi300x-bs128-mbs4-fp8": {
            "name": "deepseek_v2_lite_mbs4_gbs128_FP8",
            "global_batch_size": "128",
            "micro_batch_size": "4",
            "precision": "FP8"
          }
        },
        "runs": [
          "deepseek_v2_lite-mi300x-bs128-mbs4-bf16",
          "deepseek_v2_lite-mi300x-bs128-mbs4-fp8"
        ]
      }
    }

``model_params``
~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Parameter
     - Value
     - Description
   * - ``model_name``
     - ``deepseek_v2_lite``
     - Used in log labels and report filenames.
   * - ``tokenizer_model``
     - ``deepseek-ai/DeepSeek-V2-Lite``
     - HuggingFace repo ID. CVS downloads the ``tokenizer.model`` file locally before launch (``test_download_tokenizer``).
   * - ``model_size``
     - ``16``
     - Model size in billions of parameters.
   * - ``sequence_length``
     - ``4096``
     - Shorter context than Llama due to DeepSeek V2's attention architecture.
   * - ``tensor_parallelism``
     - ``1``
     - Fits on a single GPU.
   * - ``pipeline_parallelism``
     - ``1``
     - Single pipeline stage.
   * - ``precision``
     - ``BF16``
     - Default precision; FP8 also available as a sweep cell.


Llama 3.1 405B
--------------

Available as ``mi300x_megatron_llama-3.1-405b_distributed.json``, ``mi325x_…``, ``mi355x_…`` (distributed only).

.. dropdown:: ``mi325x_megatron_llama-3.1-405b_distributed.json`` (representative)

  .. code:: json

    {
      "schema_version": 1,
      "framework": "megatron_distributed",
      "gpu_arch": "MI325X",
      "enforce_thresholds": true,
      "threshold_json": "mi325x_megatron_llama-3.1-405b_distributed_threshold.json",
      "model_params": {
        "model_name": "llama3.1_405B",
        "tokenizer_model": "meta-llama/Llama-3.1-405B",
        "model_size": "405",
        "sequence_length": "8192",
        "recompute": "0",
        "fsdp": "0",
        "tensor_parallelism": "8",
        "pipeline_parallelism": "4",
        "precision": "FP8"
      },
      "sweep": {
        "combinations": {
          "llama3_1_405b-mi325x-bs64-mbs1-fp8": {
            "name": "llama3_1_405b_mbs1_gbs64_FP8",
            "global_batch_size": "64",
            "micro_batch_size": "1",
            "precision": "FP8"
          },
          "llama3_1_405b-mi325x-bs64-mbs1-bf16": {
            "name": "llama3_1_405b_mbs1_gbs64_BF16",
            "global_batch_size": "64",
            "micro_batch_size": "1",
            "precision": "BF16"
          }
        },
        "runs": [
          "llama3_1_405b-mi325x-bs64-mbs1-fp8",
          "llama3_1_405b-mi325x-bs64-mbs1-bf16"
        ]
      }
    }

``model_params``
~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Parameter
     - Value
     - Description
   * - ``model_name``
     - ``llama3.1_405B``
     - Used in log labels and report filenames.
   * - ``tokenizer_model``
     - ``meta-llama/Llama-3.1-405B``
     - HuggingFace repo ID for the tokenizer and model weights.
   * - ``model_size``
     - ``405``
     - Model size in billions of parameters.
   * - ``sequence_length``
     - ``8192``
     - Maximum context length.
   * - ``tensor_parallelism``
     - ``8``
     - Splits across all 8 GPUs per node.
   * - ``pipeline_parallelism``
     - ``4``
     - Splits the model across 4 pipeline stages (requires at least 4 nodes).
   * - ``precision``
     - ``FP8``
     - Default precision (overridden per sweep cell).


Common parameters
=================

These sections appear in all config files. The parameter names and semantics are identical across models and GPU variants.

``config`` (single-node)
------------------------

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``hf_token_file``
     - ``/home/{user-id}/.hf_token``
     - Path to a Hugging Face token file for gated models and datasets.
   * - ``log_dir``
     - ``/home/{user-id}/LOGS/megatron``
     - Host path where per-node training logs are written. Must be volume-mounted into the container. Megatron-LM writes ``<log_dir>/megatron-logs/<combo_id>/out-node<N>/training.log``; Primus writes ``<log_dir>/primus-logs/<combo_id>/out-node<N>/training.log``.
   * - ``scripts_dir``
     - ``/home/{user-id}/SCRIPTS/megatron``
     - Host path where the lib writes per-rank wrapper scripts. Must be volume-mounted into the container.
   * - ``data_cache_dir``
     - ``/home/{user-id}/cache``
     - Dataset and tokenizer cache directory.
   * - ``rocm_dir``
     - ``""``
     - ROCm installation path inside the container. Leave empty for auto-detection.
   * - ``megatron_root``
     - ``/workspace/Megatron-LM``
     - Root directory of the Megatron-LM checkout inside the container. Used by the Megatron-LM backend; Primus jobs use in-image YAML under ``examples/megatron/configs/{gpu_arch}/`` instead.
   * - ``training_iterations``
     - ``<changeme>``
     - Number of training steps to run.
   * - ``nnodes``
     - ``1``
     - Always ``1`` for single-node configs.
   * - ``master_address``
     - ``127.0.0.1``
     - Loopback address for single-node runs.
   * - ``nccl_socket_ifname``
     - ``<changeme>``
     - Network interface for NCCL control channels (required even for single-node).
   * - ``gloo_socket_ifname``
     - ``<changeme>``
     - Network interface for Gloo control channels.
   * - ``nccl_ib_gid_index``
     - ``3``
     - GID index for InfiniBand addressing.
   * - ``nccl_debug``
     - ``ERROR``
     - NCCL log verbosity level.
   * - ``verify_network_errors``
     - ``False``
     - Disabled for single-node; no RDMA counters to compare.

``config`` (distributed — additional fields)
--------------------------------------------

Distributed configs include all single-node fields above plus the following required network fields.

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``nnodes``
     - ``<changeme>``
     - Number of nodes in the distributed job. Must match the cluster file.
   * - ``master_address``
     - ``<changeme>``
     - IP address of the rank-0 (master) node.
   * - ``nic_type``
     - ``<changeme>``
     - NIC hardware type (e.g. ``thor2``, ``cx7``). Controls Broadcom-specific in-container workarounds.
   * - ``nccl_ib_hca_list``
     - ``<changeme>``
     - Comma-separated list of InfiniBand HCA device names for NCCL multi-rail.
   * - ``nccl_ib_hca``
     - ``<changeme>``
     - Primary HCA name passed to ``NCCL_IB_HCA``.
   * - ``hca_id_pattern``
     - ``bnxt_|rocep``
     - ``|``-separated NIC-name prefixes checked against ``ibv_devinfo`` after the libbnxt copy. Add ``|mlx5_`` for Mellanox/RoCE NICs.
   * - ``verify_network_errors``
     - ``True``
     - Compare RDMA and ethtool error counters before and after training.

``container``
-------------

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``lifetime``
     - ``per_run``
     - When to create and destroy the container. ``per_run`` launches a fresh container for each test session.
   * - ``name``
     - *(model-specific)*
     - Container instance name, e.g. ``megatron_llama3_1_8b_single``.
   * - ``image``
     - ``<changeme>``
     - Docker image to run. If the image name contains ``primus`` (case-insensitive), the suite uses the Primus backend; otherwise it uses Megatron-LM. Set this to the image available in your environment.
   * - ``runtime.name``
     - ``docker``
     - Container runtime. Currently only ``docker`` is supported.
   * - ``runtime.args.network``
     - ``host``
     - Use host networking so NCCL and Gloo can reach other nodes directly.
   * - ``runtime.args.ipc``
     - ``host``
     - Share the host IPC namespace for GPU shared memory.
   * - ``runtime.args.privileged``
     - ``true``
     - Required for ROCm GPU and InfiniBand device access.
   * - ``runtime.args.volumes``
     - *(see below)*
     - List of ``host:container`` bind mounts. At minimum, mount the user home directory (``/home/{user-id}:/home/{user-id}``) so ``log_dir``, ``scripts_dir``, and ``data_cache_dir`` are accessible inside the container. Distributed configs also require ``/dev/infiniband:/dev/infiniband`` and the Broadcom driver library mount. The last entry (``<changeme>:<changeme>``) is the shared filesystem bind mount for ``checkpoint.checkpoint_dir`` — replace both sides with the same shared path (e.g. ``/mnt/shared/ckpt:/mnt/shared/ckpt``). Required only when ``checkpoint.enforce`` is ``true``; the loader skips this entry when ``enforce`` is ``false``.
   * - ``runtime.args.devices``
     - ``/dev/kfd``, ``/dev/dri``
     - GPU device nodes to expose. Distributed configs also add ``/dev/infiniband/rdma_cm``.

``checkpoint``
--------------

Controls the checkpoint save and resume test (``test_checkpoint``). The test is Primus-only: it is skipped when ``enforce`` is ``false``, and also skipped when the container image name does not contain ``primus``. On Primus it runs in two phases: a save phase that trains for ``save_iters`` steps writing a checkpoint every ``save_interval`` steps, followed by a resume phase that loads the last checkpoint and trains to ``resume_iters`` steps. Continuity is checked at the first resume step (``last_ckpt_step + 1``), which must not exceed the checkpoint-step loss by more than ``loss_rtol``.

``checkpoint_dir`` is only present in distributed configs. On Primus distributed runs it must be a shared filesystem path visible on every node. Single-node Primus ignores that field and writes under ``{log_dir}/ckpt_primus``. Megatron-LM never runs ``test_checkpoint``.

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``enforce``
     - ``false``
     - If ``false``, ``test_checkpoint`` is skipped. Set to ``true`` to enable checkpoint save/resume verification.
   * - ``save_interval``
     - ``20``
     - How often (in steps) to write a checkpoint during the save phase. The last checkpoint lands at ``floor(save_iters / save_interval) * save_interval``.
   * - ``save_iters``
     - ``21``
     - Steps to train in the save phase. Must not be an exact multiple of ``save_interval`` so the final checkpoint is not the last step.
   * - ``resume_iters``
     - ``25``
     - Steps to train in the resume phase, continuing from the last checkpoint.
   * - ``loss_rtol``
     - ``0.05``
     - Relative tolerance for the loss continuity check. The first step of the resume phase must not exceed the checkpoint-step loss by more than ``loss_rtol * max(abs(save_loss), 1e-9)``.
   * - ``checkpoint_dir``
     - ``<changeme>``
     - *(Distributed only)* Shared filesystem path for checkpoints. Must be volume-mounted into the container at the same path on all nodes. Required only when ``checkpoint.enforce`` is ``true``; exempted from the placeholder check when ``enforce`` is ``false``.

``smoke``
---------

Controls ``test_smoke``: a small fixed cell (not a ``sweep.runs`` entry) that loads the model and trains a few steps with no metric gating. Packaged configs set this explicitly; if the block is omitted the loader defaults to enabled with ``iters`` 10, MBS ``1``, precision ``BF16``, and an empty ``global_batch_size`` (the suite then uses 8 on single-node and 16 on distributed).

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``enabled``
     - ``true``
     - If ``false``, ``test_smoke`` is skipped.
   * - ``iters``
     - ``10``
     - Training steps for the smoke cell.
   * - ``micro_batch_size``
     - ``"1"``
     - Micro-batch size for the smoke cell.
   * - ``global_batch_size``
     - ``"8"`` (single) / ``"16"`` (distributed) in packaged files; empty in schema default
     - Global batch size. Empty string lets the suite pick the topology default above.
   * - ``precision``
     - ``BF16``
     - Precision tag passed into the smoke training command.

``loss_curve``
--------------

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``sample_every``
     - ``10``
     - Sample a loss point every N steps for the slope check.
   * - ``milestone_steps``
     - ``[100, 500, 1000, 5000]``
     - Additional steps always included in the sampled loss curve regardless of ``sample_every``.
   * - ``max_slope``
     - ``0.0``
     - Maximum allowed least-squares slope of the sampled loss curve. A positive slope (loss increasing) fails the check.
   * - ``enforce``
     - ``true``
     - If ``false``, the loss curve check is record-only and does not fail the test.

``convergence``
---------------

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``target_metric``
     - ``auto``
     - Metric tracked for convergence. ``auto`` uses eval loss when ``--eval-interval`` is set in the training script, otherwise falls back to training loss.
   * - ``target_value``
     - ``0.0``
     - Loss value at which the model is considered converged. ``0.0`` or negative disables convergence checking (record-only).

``scaling_baseline``
--------------------

*(Distributed configs only.)*

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``tokens_per_sec_total``
     - ``0.0``
     - Total tokens/sec from a prior single-node run (``tokens/GPU/s × GPUs_per_node``). Used to compute scaling efficiency as nodes increase. ``0.0`` disables the metric (record-only).
   * - ``num_nodes``
     - ``1``
     - Number of nodes used to produce ``tokens_per_sec_total``. Must be ``1`` for a single-node baseline.

``sweep``
---------

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``combinations``
     - N/A
     - Dict of named sweep cells. Each cell specifies ``global_batch_size``, ``micro_batch_size``, and optionally ``precision`` and ``name``. The combination key is used as the pytest parametrize ID.
   * - ``runs``
     - N/A
     - Required ordered list of combination keys to execute. Must be a subset of (or equal to) the keys in ``combinations``. Reorder or trim this list to run only specific cells. Omitting ``runs`` fails config load.

Any key in a sweep combo overrides the matching ``model_params`` field (for example ``precision`` or ``tensor_parallelism``).

Threshold files
---------------

Each suite JSON names a sibling file in ``threshold_json``. Cell keys must match ``MBS=<mbs>,GBS=<gbs>,PRECISION=<precision>`` exactly, or that combo is record-only.

A metric is gated only when ``enforce_thresholds`` is ``true`` and the cell has a numeric spec:

.. list-table::
   :widths: 2 5
   :header-rows: 1

   * - Kind
     - Passes when
   * - ``min``
     - actual ≥ value
   * - ``max``
     - actual ≤ value
   * - ``info``
     - always; recorded only
   * - ``min_ratio``
     - actual / ``reference`` ≥ value

Tracked metrics (namespace ``training.*``): ``throughput_per_gpu``, ``tokens_per_gpu``, ``elapsed_time_per_iteration``, ``mem_usage``, and on distributed configs ``scaling_efficiency_pct`` (always ``info``).
