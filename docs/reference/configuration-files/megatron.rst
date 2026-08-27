.. meta::
  :description: Configure the variables in the Megatron training configuration files
  :keywords: training, ROCm, install, cvs, Megatron,

*************************************
Megatron training configuration files
*************************************

Megatron training enables scaling transformer models from millions to trillions of parameters by efficiently utilizing hundreds or thousands of GPUs across multiple nodes.

The Megatron tests check:

- **Container orchestration**: Docker setup with ROCm/RDMA
- **Multi-node communication**: NCCL/RCCL initialization
- **Model convergence**: Loss decreases and no NaN/Inf values
- **Performance targets**: Throughput and memory usage within expected ranges
- **Result verification**: Expected tokens/sec and TFLOPS metrics

Use ``cvs copy-config --list`` to list available templates, or ``cvs copy-config <name>`` to copy one to your working directory.

.. note::

  - Parameters with the ``<changeme>`` value must have that value modified to your specifications.
  - ``{user-id}`` will be resolved to the current username in the runtime. You can also manually change this value to your username.

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

Each model section below shows only the ``model_params`` and ``sweep`` blocks, which are the parts that differ between models. All other sections (``config``, ``container``, ``checkpoint``, ``loss_curve``, ``convergence``, ``scaling_baseline``) are identical in structure across models and are documented in `Common parameters`_.

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
     - Host path where per-node training logs are written. Must be volume-mounted into the container.
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
     - Root directory of the Megatron-LM checkout inside the container.
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
     - Docker image to run. Must be set to the Megatron-LM or Primus image available in your environment.
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
     - List of ``host:container`` bind mounts. At minimum, mount the user home directory (``/home/{user-id}:/home/{user-id}``) so ``log_dir``, ``scripts_dir``, and ``data_cache_dir`` are accessible inside the container. Distributed configs also require ``/dev/infiniband:/dev/infiniband`` and the Broadcom driver library mount.
   * - ``runtime.args.devices``
     - ``/dev/kfd``, ``/dev/dri``
     - GPU device nodes to expose. Distributed configs also add ``/dev/infiniband/rdma_cm``.

``checkpoint``
--------------

Controls the checkpoint save and resume test (``test_checkpoint``). The test runs in two phases: a save phase that trains for ``save_iters`` steps writing a checkpoint every ``save_interval`` steps, followed by a resume phase that loads the last checkpoint and trains to ``resume_iters`` steps, then verifies that loss does not spike across the boundary. Supported for both Primus and Megatron (llama2/llama3) backends.

``checkpoint_dir`` is only present in distributed configs, where a shared filesystem path is required for all nodes to access the same checkpoint. Single-node configs omit it; the training script uses its default local path.

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
     - *(Distributed only)* Shared filesystem path for checkpoints. Must be volume-mounted into the container at the same path on all nodes.

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
     - Ordered list of combination keys to execute. Must be a subset of (or equal to) the keys in ``combinations``. Reorder or trim this list to run only specific cells.
