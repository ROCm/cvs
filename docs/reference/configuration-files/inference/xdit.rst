.. meta::
  :description: Configure PyTorch xDiT FLUX and WAN 2.2 inference benchmarks
  :keywords: inference, ROCm, cvs, xDiT, FLUX, WAN, text-to-image, image-to-video

**********************************
xDiT inference configuration
**********************************

CVS ships five xDiT suites under ``cvs/tests/inference/xdit/``. Each suite reads a JSON
file from ``cvs/input/config_file/inference/xdit/``. There is no separate threshold file;
``expected_results`` live inside ``benchmark_params``.

- **Single-node** templates run one independent docker+torchrun job on every node in the
  cluster file.
- **Distributed** templates run one coordinated torchrun job (``nnodes >= 2``). Replace
  every ``<changeme>`` (NCCL/network fields) before running.

How to run: :doc:`/how-to/test-suites/inference/xdit`.

.. note::

  - ``{user-id}`` and ``{home}`` in path strings are resolved at runtime.
  - Models must already be staged on every participating node. Prefer an absolute path in
    ``model_repo``; a Hugging Face repo id requires a pre-populated cache under ``hf_home``.
  - FLUX.1-dev and FLUX.2-dev share ``pytorch_xdit_flux_dev_*``; pick the matching JSON.

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

Copy a template:

.. code:: bash

  cvs config list inference/xdit
  cvs config copy inference/xdit/mi3xx_pytorch_xdit_flux1_dev_single.json \
    --output ~/cvs_workspace/inference/xdit/mi3xx_pytorch_xdit_flux1_dev_single.json

File structure
==============

Every template has two top-level keys:

.. list-table::
   :widths: 2 6
   :header-rows: 1

   * - Key
     - Description
   * - ``config``
     - Image, model path, output dir, optional distributed rendezvous/NCCL, ``container_config``.
   * - ``benchmark_params``
     - ``flux1_dev_t2i`` (FLUX.1 and FLUX.2) or ``wan22_i2v_a14b`` (WAN native and Diffusers).

Example: FLUX.1-dev single-node
===============================

.. dropdown:: ``mi3xx_pytorch_xdit_flux1_dev_single.json`` (abbreviated)

  .. code:: json

    {
        "config": {
            "container_image": "amdsiloai/pytorch-xdit:v25.11.2",
            "container_name": "flux-benchmark",
            "hf_token_file": "{home}/.hf_token",
            "hf_home": "{home}/.cache/huggingface",
            "output_base_dir": "{home}/cvs_flux_output",
            "model_repo": "black-forest-labs/FLUX.1-dev",
            "model_rev": "",
            "container_config": {
                "device_list": ["/dev/dri", "/dev/kfd"],
                "volume_dict": {},
                "env_dict": {}
            }
        },
        "benchmark_params": {
            "flux1_dev_t2i": {
                "prompt": "A small cat",
                "num_inference_steps": 25,
                "num_repetitions": 25,
                "height": 1024,
                "width": 1024,
                "ulysses_degree": 8,
                "ring_degree": 1,
                "use_torch_compile": true,
                "torchrun_nproc": 8,
                "expected_results": {
                    "auto": { "max_avg_pipe_time_s": 10.0 },
                    "mi300x": { "max_avg_pipe_time_s": 3.0 },
                    "mi350": { "max_avg_pipe_time_s": 2.0 },
                    "mi355": { "max_avg_pipe_time_s": 7.0 }
                }
            }
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
   * - ``container_image``
     - ``amdsiloai/pytorch-xdit:v25.11.2``
     - Image with PyTorch xDiT. FLUX.2 and WAN Diffusers templates use ``rocm/ufb-private:…``.
   * - ``container_name``
     - ``flux-benchmark``
     - Docker name (distributed ranks use ``{container_name}-rankN``).
   * - ``hf_token_file``
     - ``{home}/.hf_token``
     - Hugging Face token for gated models (FLUX.2 chat template).
   * - ``hf_home``
     - ``{home}/.cache/huggingface``
     - Host HF cache (mounted at ``/hf_home``). Must contain ``hub/`` when ``model_repo`` is a repo id.
   * - ``output_base_dir``
     - ``{home}/cvs_flux_output``
     - Host directory for ``flux_<target>_outputs`` or ``wan_22_<target>_outputs``.
   * - ``model_repo``
     - HF id or ``/data/models/…``
     - Repo id (offline cache) or absolute host path (preferred). WAN Diffusers **requires** an absolute path.
   * - ``model_rev``
     - snapshot hash or ``""``
     - HF snapshot id when using cache mode. WAN native template pins ``206a9ee1…``.
   * - ``container_config.device_list``
     - ``["/dev/dri", "/dev/kfd"]``
     - GPU device nodes passed into the container.
   * - ``container_config.volume_dict``
     - host → container map
     - Extra bind mounts. FLUX.2 mounts ``flux2_example.py``; WAN Diffusers mounts ``wan_i2v_example.py``.
   * - ``container_config.env_dict``
     - ``{"NCCL_PROTO": "Simple"}``
     - Extra environment variables inside the container.

Distributed ``config`` fields
-----------------------------

Present on ``*_distributed.json`` templates (FLUX.1, FLUX.2, WAN Diffusers):

.. list-table::
   :widths: 3 5
   :header-rows: 1

   * - Parameter
     - Description
   * - ``nnodes``
     - Participating node count (must be ``>= 2``). Optional ``server_node_list`` can subset the cluster.
   * - ``master_addr``, ``master_port``
     - torchrun rendezvous (port default ``29500``). ``master_addr`` is ``<changeme>``.
   * - ``nccl_ib_hca``, ``nccl_ib_gid_index``
     - NCCL InfiniBand/RoCE devices and GID index.
   * - ``nccl_socket_ifname``, ``gloo_socket_ifname``
     - Ethernet interfaces for socket/Gloo fallback.
   * - ``nccl_debug``
     - NCCL log level (templates use ``INFO``).

``benchmark_params.flux1_dev_t2i``
==================================

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
   * - ``expected_results``
     - ``mi300x.max_avg_pipe_time_s``
     - Pass/fail on average ``pipe_time`` from ``results/timing.json``. Keys: ``auto``, ``mi300x``, ``mi350``, ``mi355``.

``benchmark_params.wan22_i2v_a14b``
===================================

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
   * - ``expected_results``
     - ``mi300x.max_avg_total_time_s``
     - Average ``total_time`` from ``rank0_step*.json``. Requires ``video.mp4``.

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
   * - ``expected_results``
     - ``max_avg_pipe_time_s`` (``auto``, ``mi325`` in the shipped templates).

Volume mounts
=============

**FLUX.1 / WAN native** templates ship an empty ``volume_dict``. Bind-mount models via
``model_repo`` as an absolute path, or rely on ``hf_home``.

**FLUX.2** mounts the in-tree example when the image lacks it:

.. code:: json

  {
      "volume_dict": {
          "/home/{user-id}/cvs/cvs/lib/inference/xdit/scripts/flux2_example.py": "/benchmark/flux2_example.py"
      }
  }

**WAN Diffusers** mounts the xFuser example:

.. code:: json

  {
      "volume_dict": {
          "/home/{user-id}/cvs/cvs/lib/inference/xdit/scripts/wan_i2v_example.py": "/benchmark/wan_i2v_example.py"
      }
  }

Adjust the host path to your CVS checkout.

Performance metrics
===================

GPU type is detected from ``rocm-smi``. Lookup order: exact key → ``auto``.

- **FLUX** — average ``pipe_time`` vs ``max_avg_pipe_time_s``; artifacts ``results/timing.json`` and ``flux_*.png``.
- **WAN native** — average ``total_time`` vs ``max_avg_total_time_s``; ``rank0_step*.json`` and ``video.mp4``.
- **WAN Diffusers** — average pipe/epoch time vs ``max_avg_pipe_time_s``; ``results/timing.json`` and ``results/video_i2v.mp4``.

Shipped numbers are starting points; tune ``expected_results`` for your stack before production gating.

Troubleshooting
===============

**``/dev/kfd not found``**
  Run on GPU compute nodes, not login nodes.

**Container image not found locally**
  ``docker pull`` the configured ``container_image`` on every execution node.

**Local model path not found**
  Stage weights on every participating node. Diffusers WAN requires ``model_repo`` as an absolute path.

**Parallel degree product != world_size**
  Align ``ulysses`` / ``ring`` (and FLUX pipefusion/TP/DP) with ``nnodes × torchrun_nproc``.

**Missing ``timing.json`` / ``video.mp4``**
  The benchmark docker exit code was non-zero or artifacts were written elsewhere; inspect the log tail on the failing node.
