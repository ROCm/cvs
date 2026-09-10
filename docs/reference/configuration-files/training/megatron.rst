.. meta::
  :description: Configure the variables in the Megatron training configuration files
  :keywords: training, ROCm, install, cvs, Megatron,

*************************************
Megatron training configuration files
*************************************

JSON configs and sibling ``*_threshold.json`` files for ``megatron_single`` and ``megatron_distributed``. MI300X and MI325X share one config per model and mode (``mi3xx_megatron_{model}_{single|distributed}.json``); set ``gpu_name`` and ``threshold_json`` to the SKU. MI355X ships Llama 3.1 8B and Llama 3.3 70B single-node configs only (``mi355x_megatron_llama-3.1-8b_single.json``, ``mi355x_megatron_llama-3.3-70b_single.json``). Use a ``*_single.json`` file with ``megatron_single`` and a ``*_distributed.json`` file with ``megatron_distributed``. ``threshold_json`` is resolved relative to the config file.

How to run the suites: :doc:`/how-to/test-suites/training/megatron`.

Use ``cvs config list training/megatron`` to list available templates, or
``cvs config copy training/megatron/<name>`` to copy one to your working
directory. Copy the matching SKU ``*_threshold.json`` into the same directory as
the suite config (``mi300x_*`` or ``mi325x_*`` for ``mi3xx_`` templates;
``mi355x_*`` for MI355X).

Backends
========

The suite selects the training backend from ``container.image`` (substring ``primus``, case-insensitive):

* **Megatron-LM** — image name does not contain ``primus``. Training scripts live under ``/workspace/Megatron-LM`` inside the image. Log files use ``<log_dir>/megatron-logs/<combo_id>/out-node<N>/training.log``.
* **Primus** — image name contains ``primus``. In-image YAML lives under ``examples/megatron/configs/{gpu_arch}/``. Log files use ``<log_dir>/primus-logs/<combo_id>/out-node<N>/training.log``.

Llama 3.1 8B, Llama 3.3 70B, and DeepSeek V2 Lite run on **both** Megatron-LM and Primus (same JSON; pick the backend with ``container.image``). **Llama 3.1 405B** is Primus only (distributed).

``<combo_id>`` on disk is the sweep combination key with non-filename characters replaced (``=`` and ``,`` become ``_``). Pytest still uses the unsanitized key (for example ``MBS=4,GBS=128,PRECISION=FP8``).

.. note::

  - Parameters with the ``<changeme>`` value must have that value modified to your specifications. Unresolved placeholders cause a hard exit at load time.
  - ``{user-id}`` will be resolved to the cluster username (or the local OS user as fallback). You can also set this value yourself.
  - Keys prefixed with ``_`` (for example ``_checkpoint_comment``) are inline comments and are ignored by the loader.
  - ``sweep.runs`` is required when ``sweep.combinations`` is non-empty. It must be a subset of (or equal to) those keys. Omitting ``sweep`` entirely (or using empty ``combinations``) runs one implicit ``default`` cell from ``train_params``.
  - Each key in ``sweep.combinations`` must be ``MBS=<micro_batch_size>,GBS=<global_batch_size>,PRECISION=<precision>``. The suite parses those three values from the key; do not repeat them in the combination body.

Available configurations
========================

MI300X and MI325X share ``mi3xx_megatron_<model>_<mode>.json``. MI355X ships ``mi355x_megatron_llama-3.1-8b_single.json`` and ``mi355x_megatron_llama-3.3-70b_single.json``. Thresholds stay SKU-specific (``mi300x_*_threshold.json``, ``mi325x_*_threshold.json``, ``mi355x_*_threshold.json``) and are selected with ``threshold_json``.

.. list-table::
   :widths: 4 3 3 2
   :header-rows: 1

   * - Model
     - MI300X / MI325X config
     - MI355X config
     - Mode
   * - Llama 3.1 8B
     - ``mi3xx_…``
     - ``mi355x_…`` (single only)
     - single, distributed (Megatron-LM or Primus); MI355X packaged as single-node
   * - Llama 3.3 70B
     - ``mi3xx_…``
     - ``mi355x_…`` (single only)
     - single, distributed (Megatron-LM or Primus); MI355X packaged as single-node
   * - DeepSeek V2 Lite
     - ``mi3xx_…``
     - —
     - single, distributed (Megatron-LM or Primus)
   * - Llama 3.1 405B
     - ``mi3xx_…``
     - —
     - distributed (Primus only)

Single-node configs set ``container.env.MASTER_ADDR`` to ``127.0.0.1``. ``NNODES`` is not in the JSON: the suite sets it from the cluster host count into ``docker run -e``. Distributed configs require ``MASTER_ADDR`` and ``NCCL_IB_HCA`` and add a ``scaling_baseline`` section and ``checkpoint_dir`` to the ``checkpoint`` block. NIC type is not in the JSON: Megatron-LM defaults to ``thor2`` for MI300X/MI325X and ``ainic`` for MI355X from ``gpu_name``.

Leftover files ``mi3xx_megatron_llama_*.json`` and ``mi35x_megatron_llama_single.json`` are nested-schema configs for the legacy ``megatron_llama3_1_*`` suites only. Do not pass them to ``megatron_single`` or ``megatron_distributed``.

Required edits
==============

Set these before a run (full field tables are under `Common parameters`_):

* ``gpu_name`` / ``threshold_json`` — on ``mi3xx_`` templates, set ``MI300X`` or ``MI325X`` and the matching SKU threshold filename. ``gpu_name`` must be exactly ``MI300X``, ``MI325X``, or ``MI355X`` after uppercase.
* ``container.image`` — Megatron-LM or Primus ROCm image on all nodes.
* ``train_params.training_iterations`` — default training steps (for example ``"30"``). Each ``sweep.combinations`` body sets ``training_iterations`` (packaged value ``"20"``); that overlay wins for that cell.
* ``paths.hf_token_file`` — Hugging Face token path on the nodes.
* ``container.env.NCCL_SOCKET_IFNAME`` / ``GLOO_SOCKET_IFNAME`` / ``NCCL_IB_GID_INDEX`` / ``NCCL_DEBUG`` — templates include example values plus ``<changeme>`` (for example ``enp193s0f1np1 <changeme>``, ``3 <changeme>``, ``ERROR <changeme>``). Remove ``<changeme>`` and keep or edit the example. Required on single-node and distributed.
* ``sweep.runs`` — combination keys to execute (same ``MBS=…,GBS=…,PRECISION=…`` strings as in ``sweep.combinations``).
* Do not set ``NNODES`` in JSON.
* **Distributed only:** ``container.env.MASTER_ADDR``, ``container.env.NCCL_IB_HCA`` (example HCA list plus ``<changeme>``). ``paths.data_cache_dir`` must be a shared filesystem. When ``checkpoint.enforce`` is ``true``, also set ``checkpoint.checkpoint_dir`` and replace the last ``<changeme>:<changeme>`` volume with that shared path.

Top-level fields
================

These fields appear at the root of every config file.

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Field
     - Example
     - Description
   * - ``gpu_name``
     - ``MI300X`` / ``<changeme>``
     - GPU architecture string used for logging and Primus YAML path resolution (``examples/megatron/configs/{gpu_name}/``). Loaded as uppercase (``mi300x`` becomes ``MI300X``). Allowed values: ``MI300X``, ``MI325X``, ``MI355X``. Extra tokens fail at load. On ``mi3xx_`` templates this is ``<changeme>`` until you set ``MI300X`` or ``MI325X``.
   * - ``paths``
     - see `Common parameters`_
     - Host paths: ``hf_token_file``, ``log_dir``, ``scripts_dir``, ``data_cache_dir``.
   * - ``container``
     - see `Common parameters`_
     - Image, runtime mounts, and ``container.env``. Packaged files keep this block immediately after ``paths``.
   * - ``train_params``
     - see per-model tables
     - Model knobs plus ``training_iterations``. With a ``sweep``, MBS/GBS/precision live on the combination key. With no ``sweep``, optional ``micro_batch_size``, ``global_batch_size``, and ``precision`` here feed the implicit ``default`` cell.
   * - ``verify_network_errors``
     - omitted (single) / ``True`` (distributed)
     - Compare RDMA and ethtool error counters before and after training. Single-node templates omit it (schema/lib default ``False``).
   * - ``enforce_thresholds``
     - ``true``
     - If ``false``, threshold checks in ``test_metric`` log results but do not fail the test.
   * - ``threshold_json``
     - ``mi300x_megatron_llama-3.1-8b_single_threshold.json`` / ``<changeme>``
     - Filename of the companion threshold file, looked up in the same directory as the config file. On ``mi3xx_`` templates this is ``<changeme>`` until you point it at the ``mi300x_*`` or ``mi325x_*`` threshold for the SKU.

Model configurations
====================

Each model section below shows only the ``train_params`` and ``sweep`` blocks, which are the parts that differ between models. All other sections (``paths``, ``container``, ``smoke``, ``loss_curve``, ``convergence``, ``checkpoint``, ``scaling_baseline``) are identical in structure across models and are documented in `Common parameters`_.

Llama 3.1 8B
------------

Available as ``mi3xx_megatron_llama-3.1-8b_{single,distributed}.json`` (MI300X/MI325X) and ``mi355x_megatron_llama-3.1-8b_single.json`` (MI355X).

.. dropdown:: ``mi3xx_megatron_llama-3.1-8b_single.json`` (representative)

  .. code:: json

    {
      "gpu_name": "<changeme>",
      "enforce_thresholds": true,
      "threshold_json": "<changeme>",
      "train_params": {
        "model_name": "llama3.1_8B",
        "tokenizer_model": "meta-llama/Llama-3.1-8B",
        "model_size": "8",
        "sequence_length": "8192",
        "recompute": "0",
        "fsdp": "0",
        "tensor_parallelism": "1",
        "pipeline_parallelism": "1",
        "training_iterations": "<changeme>"
      },
      "sweep": {
        "combinations": {
          "MBS=4,GBS=128,PRECISION=FP8": {
            "training_iterations": "20"
          },
          "MBS=4,GBS=128,PRECISION=BF16": {
            "training_iterations": "20"
          },
          "MBS=4,GBS=128,PRECISION=MXFP4": {
            "training_iterations": "20"
          },
          "MBS=4,GBS=128,PRECISION=MXFP8": {
            "training_iterations": "20"
          }
        },
        "runs": [
          "MBS=4,GBS=128,PRECISION=FP8",
          "MBS=4,GBS=128,PRECISION=BF16",
          "MBS=4,GBS=128,PRECISION=MXFP4",
          "MBS=4,GBS=128,PRECISION=MXFP8"
        ]
      }
    }

``train_params``
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
   * - ``training_iterations``
     - ``<changeme>``
     - Training steps (same field on every model).


Llama 3.3 70B
-------------

Available as ``mi3xx_megatron_llama-3.3-70b_{single,distributed}.json`` (MI300X/MI325X) and ``mi355x_megatron_llama-3.3-70b_single.json`` (MI355X).

.. dropdown:: ``mi3xx_megatron_llama-3.3-70b_single.json`` (representative)

  .. code:: json

    {
      "gpu_name": "<changeme>",
      "enforce_thresholds": true,
      "threshold_json": "<changeme>",
      "train_params": {
        "model_name": "llama3.3_70B",
        "tokenizer_model": "meta-llama/Llama-3.3-70B-Instruct",
        "model_size": "70",
        "sequence_length": "8192",
        "recompute": "0",
        "fsdp": "0",
        "tensor_parallelism": "8",
        "pipeline_parallelism": "1",
        "training_iterations": "<changeme>"
      },
      "sweep": {
        "combinations": {
          "MBS=3,GBS=96,PRECISION=FP8": {
            "training_iterations": "20"
          },
          "MBS=3,GBS=96,PRECISION=BF16": {
            "training_iterations": "20"
          }
        },
        "runs": [
          "MBS=3,GBS=96,PRECISION=FP8",
          "MBS=3,GBS=96,PRECISION=BF16"
        ]
      }
    }

``train_params``
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
   * - ``training_iterations``
     - ``<changeme>``
     - Training steps (same field on every model).


DeepSeek V2 Lite
----------------

Available as ``mi3xx_megatron_deepseek-v2-lite_{single,distributed}.json`` (MI300X/MI325X).

.. dropdown:: ``mi3xx_megatron_deepseek-v2-lite_single.json`` (representative)

  .. code:: json

    {
      "gpu_name": "<changeme>",
      "enforce_thresholds": true,
      "threshold_json": "<changeme>",
      "train_params": {
        "model_name": "deepseek_v2_lite",
        "tokenizer_model": "deepseek-ai/DeepSeek-V2-Lite",
        "model_size": "16",
        "sequence_length": "4096",
        "recompute": "0",
        "fsdp": "0",
        "tensor_parallelism": "1",
        "pipeline_parallelism": "1",
        "training_iterations": "<changeme>"
      },
      "sweep": {
        "combinations": {
          "MBS=4,GBS=128,PRECISION=BF16": {
            "training_iterations": "20"
          },
          "MBS=4,GBS=128,PRECISION=FP8": {
            "training_iterations": "20"
          }
        },
        "runs": [
          "MBS=4,GBS=128,PRECISION=BF16",
          "MBS=4,GBS=128,PRECISION=FP8"
        ]
      }
    }

``train_params``
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
   * - ``training_iterations``
     - ``<changeme>``
     - Training steps (same field on every model).


Llama 3.1 405B
--------------

Available as ``mi3xx_megatron_llama-3.1-405b_distributed.json`` (MI300X/MI325X; distributed only). Unlike the other models, 405B runs on **Primus only**: ``container.image`` must contain ``primus``. Megatron-LM does not support 405B.

.. dropdown:: ``mi3xx_megatron_llama-3.1-405b_distributed.json`` (representative)

  .. code:: json

    {
      "gpu_name": "<changeme>",
      "enforce_thresholds": true,
      "threshold_json": "<changeme>",
      "train_params": {
        "model_name": "llama3.1_405B",
        "tokenizer_model": "meta-llama/Llama-3.1-405B",
        "model_size": "405",
        "sequence_length": "8192",
        "recompute": "0",
        "fsdp": "0",
        "tensor_parallelism": "8",
        "pipeline_parallelism": "4",
        "training_iterations": "<changeme>"
      },
      "sweep": {
        "combinations": {
          "MBS=1,GBS=64,PRECISION=FP8": {
            "training_iterations": "20"
          },
          "MBS=1,GBS=64,PRECISION=BF16": {
            "training_iterations": "20"
          }
        },
        "runs": [
          "MBS=1,GBS=64,PRECISION=FP8",
          "MBS=1,GBS=64,PRECISION=BF16"
        ]
      }
    }

``train_params``
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
   * - ``training_iterations``
     - ``<changeme>``
     - Training steps.


Common parameters
=================

These sections appear in all config files. The parameter names and semantics are identical across models and GPU variants.

``paths``
---------

Host paths used by the job. They must be volume-mounted into the container (typically via the home-directory bind mount).

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
     - Host path where per-node training logs are written. Megatron-LM writes ``<log_dir>/megatron-logs/<combo_id>/out-node<N>/training.log``; Primus writes ``<log_dir>/primus-logs/<combo_id>/out-node<N>/training.log``. On disk ``<combo_id>`` is the sanitized combination key (see `Backends`_).
   * - ``scripts_dir``
     - ``/home/{user-id}/SCRIPTS/megatron``
     - Host path where the lib writes per-rank wrapper scripts.
   * - ``data_cache_dir``
     - ``/home/{user-id}/cache``
     - Dataset and tokenizer cache directory. On distributed runs this path must be a shared filesystem.
   * - ``rocm_dir``
     - ``""``
     - ROCm installation path inside the container. Leave empty for auto-detection.

``container.env``
-----------------

These values are passed into the container as ``docker run -e`` flags and also flattened into the training job dict. ``NNODES`` is injected at launch from the cluster host count (not a JSON field). Packaged templates put example NIC strings in the values and keep ``<changeme>`` in the same string so load fails until you edit them. Megatron-LM and Primus wrappers do not re-export ``NCCL_*``, ``GLOO_SOCKET_IFNAME``, ``MASTER_ADDR``, or ``NNODES``. Megatron-LM still re-exports ``NCCL_IB_GID_INDEX`` after Broadcom NIC setup.

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``MASTER_ADDR``
     - ``127.0.0.1`` (single) / ``<changeme>`` (distributed)
     - Rank-0 address (loopback on single-node).
   * - ``NCCL_SOCKET_IFNAME``
     - ``enp193s0f1np1 <changeme>``
     - Network interface for NCCL control channels. Replace ``<changeme>``; the ``enp193s0f1np1`` token is an example.
   * - ``GLOO_SOCKET_IFNAME``
     - ``enp193s0f1np1 <changeme>``
     - Network interface for Gloo control channels. Same example pattern as ``NCCL_SOCKET_IFNAME``.
   * - ``NCCL_IB_GID_INDEX``
     - ``3 <changeme>``
     - GID index for InfiniBand addressing. Replace ``<changeme>`` (typical value ``3``).
   * - ``NCCL_DEBUG``
     - ``ERROR <changeme>``
     - NCCL log verbosity. Replace ``<changeme>`` (typical value ``ERROR``).
   * - ``NCCL_IB_HCA``
     - ``bnxt_re0,...,bnxt_re8 <changeme>``
     - *(Distributed)* Comma-separated InfiniBand HCA device names. The packaged list is an example; replace ``<changeme>``.

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

Controls the checkpoint save and resume test (``test_checkpoint``). The test is Primus-only: it is skipped when ``enforce`` is ``false``, and also skipped when the container image name does not contain ``primus``. On Primus it runs in two phases: a save phase that trains for ``save_iters`` steps writing a checkpoint every ``save_interval`` steps, followed by a resume phase that loads the last checkpoint and trains to ``resume_iters`` steps. Continuity is checked at the first resume step (``last_ckpt_step + 1``), which must not exceed the checkpoint-step loss by more than ``loss_rtol``. Micro-batch size, global batch size, and precision are the same as ``test_smoke`` (the ``smoke`` block).

``checkpoint_dir`` is only present in distributed configs. On Primus distributed runs it must be a shared filesystem path visible on every node. Single-node Primus ignores that field and writes under ``{log_dir}/ckpt_primus``. Megatron-LM never runs ``test_checkpoint``.

Load I/O timing is taken from the node-0 Primus log. Single-node resume lines say ``loading checkpoint from``; distributed resume lines say ``loading distributed checkpoint from``. Both end with ``successfully loaded checkpoint from``. A missing load-start line yields a warning, not a test failure.

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

Controls ``test_smoke``: a small fixed cell (not a ``sweep.runs`` entry) that loads the model and trains a few steps with no metric gating. ``test_checkpoint`` uses the same ``micro_batch_size``, ``global_batch_size``, and ``precision`` (it still uses ``checkpoint.*`` for step counts). Packaged configs set this explicitly; if the block is omitted the loader defaults to enabled with ``iters`` 10, MBS ``1``, precision ``BF16``, and an empty ``global_batch_size`` (the suite then uses 8 on single-node and 16 on distributed).

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
     - Micro-batch size for the smoke cell and for ``test_checkpoint``.
   * - ``global_batch_size``
     - ``"8"`` (single) / ``"16"`` (distributed) in packaged files; empty in schema default
     - Global batch size for smoke and ``test_checkpoint``. Empty string lets the suite pick the topology default above.
   * - ``precision``
     - ``BF16``
     - Precision tag for the smoke cell and for ``test_checkpoint``.

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
     - Dict of sweep cells. The key must be ``MBS=<micro_batch_size>,GBS=<global_batch_size>,PRECISION=<precision>`` (the ``sweep_name`` pytest ID and threshold cell).
   * - ``combinations.*.training_iterations``
     - ``"20"``
     - Per-cell step count. Overrides ``train_params.training_iterations`` for that run. Other ``train_params`` keys may also appear in the body.
   * - ``runs``
     - N/A
     - Ordered list of combination keys to execute. Required when ``combinations`` is non-empty. Omit ``sweep`` (or leave ``combinations`` empty) to run the implicit ``default`` cell.

Pytest parametrizes ``sweep_name`` from ``sweep.runs``. The suite parses ``micro_batch_size``, ``global_batch_size``, and ``precision`` from each combination key. Additional fields in the combination body override the corresponding ``train_params`` values for that run, including ``training_iterations``. If ``sweep`` is omitted or ``combinations`` is empty, one cell named ``default`` trains with base ``train_params`` (MBS ``2``, GBS ``128``, precision ``BF16`` unless those keys are set in ``train_params``). The matching threshold cell is ``default`` and is required when ``enforce_thresholds`` is ``true``.

Threshold files
---------------

Each suite JSON names a sibling file in ``threshold_json``. With a declared ``sweep``, top-level cell keys must be the same ``MBS=<mbs>,GBS=<gbs>,PRECISION=<precision>`` strings as ``sweep.combinations``. With no ``sweep`` (generic / implicit ``default`` run), the threshold cell is the literal key ``default`` — not an ``MBS=…`` string built from ``train_params``. A missing or extra cell fails load when ``enforce_thresholds`` is ``true``. When it is ``false``, the ``default`` cell may be omitted: the loader warns and ``test_metric`` is record-only.

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
   * - ``optional`` (boolean on a spec, not a kind)
     - if true, Megatron ``test_metric`` skips the metric when it is missing or None; a present value is still gated by ``kind``.

Tracked metrics (namespace ``training.*``): ``throughput_per_gpu``, ``tokens_per_gpu``, ``elapsed_time_per_iteration``, ``mem_usage`` (Megatron-LM ``mem usages:``; Primus does not emit it; packaged specs set ``optional: true``), and on distributed configs ``scaling_efficiency_pct`` (``kind: info`` and ``optional: true`` in packaged files).
