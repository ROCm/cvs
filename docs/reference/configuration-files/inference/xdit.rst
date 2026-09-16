.. meta::
  :description: Configure PyTorch xDiT FLUX and WAN 2.2 inference benchmarks
  :keywords: inference, ROCm, cvs, xDiT, FLUX, WAN, text-to-image, image-to-video

**********************************
xDiT inference configuration
**********************************

CVS ships five xDiT suites under ``cvs/tests/inference/xdit/``. Each suite reads a JSON
file from ``cvs/input/config_file/inference/xdit/``. Latency thresholds live in sibling
``*_threshold.json`` files referenced by top-level ``threshold_json``.

- **Single-node** templates run one independent container+torchrun job on every node in
  the cluster file.
- **Distributed** templates run one coordinated torchrun job (``topology: distributed``,
  ``nnodes >= 2``). Replace every ``<changeme>`` (image, model path, NCCL/network fields)
  before running.

How to run: :doc:`/how-to/test-suites/inference/xdit`.

.. note::

  - ``{user-id}``, ``{home}``, and ``{paths.*}`` placeholders are resolved at startup.
  - Models must already be staged on every participating node. Prefer an absolute path in
    ``model.id``; a Hugging Face repo id requires a pre-populated cache under ``paths.models_dir``.
  - FLUX.1-dev and FLUX.2-dev share ``pytorch_xdit_flux_dev_*``; pick the matching JSON.
  - There is no packaged WAN-native distributed workload; only the five implemented suites
    have templates in this directory.

Configuration files
===================

.. list-table::
   :widths: 3 2
   :header-rows: 1

   * - Config file
     - Use with suite
   * - ``mi3xx_pytorch_xdit_flux1_dev_single.json``
     - ``pytorch_xdit_flux_dev_single``
   * - ``mi3xx_pytorch_xdit_flux1_dev_distributed.json``
     - ``pytorch_xdit_flux_dev_distributed``
   * - ``mi3xx_pytorch_xdit_flux2_dev_single.json``
     - ``pytorch_xdit_flux_dev_single``
   * - ``mi3xx_pytorch_xdit_flux2_dev_distributed.json``
     - ``pytorch_xdit_flux_dev_distributed``
   * - ``mi3xx_pytorch_xdit_wan22_14b_single.json``
     - ``pytorch_xdit_wan22_14b_single``
   * - ``mi3xx_pytorch_xdit_wan22_14b_diffusers_single.json``
     - ``pytorch_xdit_wan22_14b_diffusers_single``
   * - ``mi3xx_pytorch_xdit_wan22_14b_diffusers_distributed.json``
     - ``pytorch_xdit_wan22_14b_diffusers_distributed``

Copy a template and its threshold file together:

.. code:: bash

  cvs config list inference/xdit
  cvs config copy inference/xdit/mi3xx_pytorch_xdit_flux1_dev_single.json \
    --output ~/cvs_workspace/inference/xdit/mi3xx_pytorch_xdit_flux1_dev_single.json
  cvs config copy inference/xdit/mi3xx_pytorch_xdit_flux1_dev_single_threshold.json \
    --output ~/cvs_workspace/inference/xdit/mi3xx_pytorch_xdit_flux1_dev_single_threshold.json

File structure
==============

Unified templates use the same top-level layout as SGLang and vLLM inference configs:

.. list-table::
   :widths: 2 6
   :header-rows: 1

   * - Key
     - Description
   * - ``schema_version``
     - Unified config version (``1``).
   * - ``framework``
     - ``xdit``.
   * - ``gpu_arch``
     - Target GPU family (templates ship ``mi3xx``).
   * - ``topology``
     - ``single`` or ``distributed``.
   * - ``enforce_thresholds``
     - When ``true``, benchmark results must satisfy the selected GPU thresholds.
   * - ``threshold_json``
     - Sibling threshold filename resolved relative to the config file.
   * - ``paths``
     - ``shared_fs``, ``models_dir``, ``log_dir``, ``hf_token_file``.
   * - ``model``
     - ``id`` (HF repo id or absolute host path) and ``remote`` (``0`` = offline/local).
   * - ``container``
     - ``lifetime``, ``name``, ``image``, and ``runtime.name`` / ``runtime.args``
       (including ``env``, matching SGLang).
   * - ``params``
     - ``flux1_dev_t2i`` (FLUX.1 and FLUX.2) or ``wan22_i2v_a14b`` (WAN native and Diffusers).
   * - ``inference``
     - Optional runtime fields such as ``model_rev`` for pinned HF snapshots.
   * - ``benchmark_serv_node``
     - Required cluster ``node_dict`` key for single-node suites.
   * - ``nnodes``, ``master_addr``, ``master_port``
     - Distributed torchrun rendezvous. NCCL/IB env lives under
       ``container.runtime.args.env``.

Legacy ``config`` + ``benchmark_params`` files with embedded ``expected_results`` still
validate through ``PytorchXditWanConfigFile`` / ``PytorchXditFluxConfigFile`` for backward
compatibility.

Example: FLUX.1-dev single-node
===============================

.. dropdown:: ``mi3xx_pytorch_xdit_flux1_dev_single.json`` (abbreviated)

  .. code:: json

    {
        "schema_version": 1,
        "framework": "xdit",
        "topology": "single",
        "benchmark_serv_node": "<changeme>",
        "enforce_thresholds": true,
        "threshold_json": "mi3xx_pytorch_xdit_flux1_dev_single_threshold.json",
        "paths": {
            "shared_fs": "{home}",
            "models_dir": "{home}/.cache/huggingface",
            "log_dir": "{home}/cvs_flux_output",
            "hf_token_file": "{home}/.hf_token"
        },
        "model": {
            "id": "black-forest-labs/FLUX.1-dev",
            "remote": 0
        },
        "container": {
            "lifetime": "per_run",
            "name": "flux-benchmark",
            "image": "<changeme>",
            "runtime": {
                "name": "docker",
                "args": {
                    "network": "host",
                    "ipc": "host",
                    "privileged": true,
                    "shm_size": "128G",
                    "volumes": [
                        "{paths.models_dir}:/hf_home",
                        "{paths.log_dir}:/outputs"
                    ],
                    "devices": ["/dev/dri", "/dev/kfd"],
                    "env": {
                        "NCCL_DEBUG": "ERROR"
                    }
                }
            }
        },
        "params": {
            "flux1_dev_t2i": {
                "prompt": "A small cat",
                "num_inference_steps": 25,
                "num_repetitions": 25,
                "height": 1024,
                "width": 1024,
                "ulysses_degree": 8,
                "ring_degree": 1,
                "use_torch_compile": true,
                "torchrun_nproc": 8
            }
        }
    }

Threshold file
--------------

``mi3xx_pytorch_xdit_flux1_dev_single_threshold.json``:

.. code:: json

    {
        "auto": { "max_avg_pipe_time_s": 10.0 },
        "mi300x": { "max_avg_pipe_time_s": 3.0 },
        "mi350": { "max_avg_pipe_time_s": 2.0 },
        "mi355": { "max_avg_pipe_time_s": 7.0 }
    }

General ``paths`` and ``container`` parameters
==============================================

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Parameter
     - Example
     - Description
   * - ``paths.shared_fs``
     - ``/home/{user-id}`` or ``{home}``
     - Cluster-visible home or scratch root.
   * - ``paths.models_dir``
     - ``{home}/.cache/huggingface``
     - Host HF cache or staged model directory (mounted at ``/hf_home``).
   * - ``paths.log_dir``
     - ``{home}/cvs_flux_output``
     - Host directory for ``flux_<target>_outputs`` or ``wan_22_<target>_outputs``.
   * - ``paths.hf_token_file``
     - ``{home}/.hf_token``
     - Hugging Face token for gated models.
   * - ``container.image``
     - ``<changeme>``
     - PyTorch xDiT image. FLUX.1/WAN native: ``amdsiloai/pytorch-xdit:v25.11.2``. FLUX.2/WAN Diffusers: ``rocm/ufb-private:…``.
   * - ``container.name``
     - ``flux-benchmark``
     - Docker name (distributed ranks use ``{container.name}-rankN``).
   * - ``container.runtime.args.devices``
     - ``["/dev/dri", "/dev/kfd"]``
     - GPU device nodes. Distributed templates also pass ``/dev/infiniband/rdma_cm``.
   * - ``container.runtime.args.volumes``
     - host:container list
     - Bind mounts. FLUX.2 mounts ``flux2_example.py``; WAN Diffusers mounts ``wan_i2v_example.py``.
       Distributed templates also mount InfiniBand libraries, matching SGLang.
   * - ``container.runtime.args.env``
     - ``{"NCCL_DEBUG": "ERROR"}``
     - Container environment. Distributed templates put NCCL/Gloo interface
       settings here (``NCCL_IB_HCA``, ``NCCL_SOCKET_IFNAME``, and so on).

Distributed fields
------------------

Present on ``topology: distributed`` templates:

.. list-table::
   :widths: 3 5
   :header-rows: 1

   * - Parameter
     - Description
   * - ``topology``
     - Must be ``distributed``.
   * - ``nnodes``
     - Participating node count (must be ``>= 2``). Optional ``server_node_list`` can subset the cluster.
   * - ``master_addr``, ``master_port``
     - torchrun rendezvous (port default ``29500``). ``master_addr`` is ``<changeme>``.
   * - ``container.runtime.args.env.NCCL_IB_HCA``
     - InfiniBand/RoCE devices. Templates include an example list plus ``<changeme>``.
   * - ``container.runtime.args.env.NCCL_SOCKET_IFNAME`` / ``GLOO_SOCKET_IFNAME`` / ``GLOO_TCP_IFNAME``
     - Ethernet interfaces for socket/Gloo fallback.
   * - ``container.runtime.args.env.NCCL_IB_GID_INDEX``
     - GID index for IB/RoCE (templates use ``3``).
   * - ``container.runtime.args.env.NCCL_DEBUG``
     - NCCL log level (templates use ``ERROR``).

``params.flux1_dev_t2i``
========================

Used by all four FLUX templates. FLUX.2 sets ``model_type: flux2``.

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Parameter
     - Example
     - Description
   * - ``model_type``
     - ``flux2``
     - FLUX.2 only. Selects ``flux2_example.py`` instead of ``run_usp.py``.
   * - ``prompt``, ``seed``
     - ``A small cat``, ``42``
     - Generation prompt and RNG seed.
   * - ``guidance_scale``
     - ``4.0``
     - FLUX.2 guidance (FLUX.1 templates omit this).
   * - ``num_inference_steps``
     - ``25`` / ``50``
     - Denoising steps (FLUX.1 ``25``, FLUX.2 ``50``).
   * - ``max_sequence_length``
     - ``256`` / ``512``
     - Text encoder sequence length.
   * - ``no_use_resolution_binning``
     - ``true``
     - Disable resolution binning.
   * - ``warmup_steps``, ``warmup_calls``, ``num_repetitions``
     - ``1``, ``5``, ``25``
     - Warmup then measured repetitions.
   * - ``height``, ``width``
     - ``1024``
     - Output image size.
   * - ``ulysses_degree``, ``ring_degree``
     - ``8``, ``1`` (single) / ``8``, ``2`` (2-node)
     - Sequence-parallel layout. Product with pipefusion/TP/DP must equal ``nnodes × torchrun_nproc``.
   * - ``use_torch_compile``
     - ``true``
     - Enable ``torch.compile``.
   * - ``torchrun_nproc``
     - ``8``
     - Processes (GPUs) per node.

FLUX threshold metric: ``max_avg_pipe_time_s`` in the sibling threshold JSON.

``params.wan22_i2v_a14b``
=========================

Native WAN (``mi3xx_pytorch_xdit_wan22_14b_single.json``)
--------------------------------------------------------

Runs ``/app/Wan2.2/run.py``. Threshold metric is ``max_avg_total_time_s``.

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Parameter
     - Example
     - Description
   * - ``prompt``
     - (long I2V prompt)
     - Image-to-video prompt.
   * - ``size``
     - ``720*1280``
     - Frame size.
   * - ``frame_num``
     - ``81``
     - Number of video frames.
   * - ``num_benchmark_steps``
     - ``5``
     - Measured steps after compile/warmup.
   * - ``compile``
     - ``true``
     - Enable compile on the native launcher.
   * - ``torchrun_nproc``
     - ``8``
     - GPUs per node.

Diffusers xFuser WAN
--------------------

``mi3xx_pytorch_xdit_wan22_14b_diffusers_*.json`` additionally set:

.. list-table::
   :widths: 3 5
   :header-rows: 1

   * - Parameter
     - Description
   * - ``model_format``
     - ``diffusers``.
   * - ``wan_diffusers_launcher``
     - ``xfuser_example``.
   * - ``wan_diffusers_run_script``
     - In-container path to ``wan_i2v_example.py`` (default ``/benchmark/wan_i2v_example.py``).
   * - ``wan_xfuser_auto_input_image``
     - Generate an in-container input image when true.
   * - ``wan_xfuser_install_video_deps``
     - Install video encode deps inside the container when true.
   * - ``wan_xfuser_output_type``
     - ``pil``.
   * - ``wan_diffusers_save_video_path``
     - ``/outputs/results/video_i2v.mp4``.
   * - ``wan_diffusers_timing_json_path``
     - ``results/timing.json``.
   * - ``require_video_artifact``
     - Fail parse if ``video_i2v.mp4`` is missing.
   * - ``num_inference_steps``, ``warmup_steps``
     - Denoising and warmup (Diffusers templates: ``40`` and ``1``).
   * - ``ulysses_size``, ``ring_size``
     - Parallel layout. Distributed: product must equal ``nnodes × torchrun_nproc``.

WAN Diffusers threshold metric: ``max_avg_pipe_time_s`` (``auto``, ``mi325`` in the shipped templates).

Volume mounts
=============

**FLUX.1 / WAN native** templates mount ``paths.models_dir`` and ``paths.log_dir`` only.
When ``model.id`` (or ``server_params.model``) is an absolute host path outside those
mounts, the loader bind-mounts it at ``/model`` and passes ``/model`` to the workload.
Adding an explicit ``container.runtime.args.volumes`` entry that covers the path
overrides this and the container path is derived from that mount instead.

**FLUX.2** mounts the in-tree example when the image lacks it:

.. code:: json

  {
      "volumes": [
          "/home/{user-id}/cvs/cvs/lib/inference/xdit/scripts/flux2_example.py:/benchmark/flux2_example.py"
      ]
  }

**WAN Diffusers** mounts the xFuser example:

.. code:: json

  {
      "volumes": [
          "/home/{user-id}/cvs/cvs/lib/inference/xdit/scripts/wan_i2v_example.py:/benchmark/wan_i2v_example.py"
      ]
  }

Adjust the host path to your CVS checkout.

Performance metrics
===================

GPU type is detected from ``rocm-smi``. Lookup order: exact key → ``auto``.

- **FLUX** — average ``pipe_time`` vs ``max_avg_pipe_time_s``; artifacts ``results/timing.json`` and ``flux_*.png``.
- **WAN native** — average ``total_time`` vs ``max_avg_total_time_s``; ``rank0_step*.json`` and ``video.mp4``.
- **WAN Diffusers** — average pipe/epoch time vs ``max_avg_pipe_time_s``; ``results/timing.json`` and ``results/video_i2v.mp4``.

Shipped numbers are starting points; tune the sibling threshold JSON for your stack before production gating.

Troubleshooting
===============

**``/dev/kfd not found``**
  Run on GPU compute nodes, not login nodes.

**Container image not found locally**
  ``docker pull`` the configured ``container.image`` on every execution node.

**Local model path not found**
  Stage weights on every participating node. Diffusers WAN requires ``model.id`` as an absolute path.

**Parallel degree product != world_size**
  Align ``ulysses`` / ``ring`` (and FLUX pipefusion/TP/DP) with ``nnodes × torchrun_nproc``.

**Missing ``timing.json`` / ``video.mp4``**
  The benchmark docker exit code was non-zero or artifacts were written elsewhere; inspect the log tail on the failing node.
