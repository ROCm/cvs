.. meta::
  :description: Reference for CVS xDiT inference benchmark configuration, covering FLUX.1, FLUX.2 text-to-image, and WAN 2.2 image-to-video suites on AMD GPU clusters.
  :keywords: CVS, xDiT, inference, ROCm, FLUX, WAN, text-to-image, image-to-video, GPU, AMD, distributed, JSON, benchmark

*************************************************************************
xDiT inference benchmark configuration for Cluster Validation Suite (CVS)
*************************************************************************

CVS ships five xDiT suites under ``cvs/tests/inference/xdit/``. Each suite reads a JSON
file from ``cvs/input/config_file/inference/xdit/``. Latency limits live in a sibling
threshold file named by top-level ``threshold_json``. ``mi3xx_*`` and ``mi35x_*``
templates for the same workload share that file.

- **Single-node** templates run one independent docker+torchrun job on every node in the
  cluster file.
- **Distributed** templates run one coordinated torchrun job (``server_params.nnodes >= 2``).
  Replace every ``<changeme>`` (image, model, GPU label, and NCCL/network fields) before running.
  Stage model weights on every participating node before the run.

See :doc:`/how-to/test-suites/inference/xdit` for more information on running these tests.

.. note::

  - ``{user-id}`` and ``{home}`` in path strings are resolved at runtime. ``{paths.<name>}``
    references expand from the ``paths`` block.
  - ``server_params.model`` is a Hugging Face repo id or an absolute host path. A repo id
    requires a pre-populated cache under ``paths.models_dir``. An absolute path is
    bind-mounted into the container.
  - FLUX.1-dev and FLUX.2-dev share ``xdit_flux_dev_*``; pick the matching JSON.
  - A relative ``threshold_json`` is loaded from the config's directory. If that copy is
    missing, CVS falls back to the packaged file of the same name.

Configuration files
===================

Each xDiT suite maps to a ``mi3xx`` template and a ``mi35x`` template. ``mi3xx`` covers
MI300X, MI325X, and MI350X (``gpu_name`` hint ``mi325``). ``mi35x`` covers MI355X
(``gpu_name`` hint ``mi355``) and ships Ionic RDMA settings on distributed configs.

.. list-table::
   :widths: 4 3 3
   :header-rows: 1

   * - Config file
     - Threshold file
     - Use with suite
   * - ``mi3xx_xdit_flux1_dev_single.json``, ``mi35x_xdit_flux1_dev_single.json``
     - ``xdit_flux1_dev_single_threshold.json``
     - ``xdit_flux_dev_single``
   * - ``mi3xx_xdit_flux1_dev_distributed.json``, ``mi35x_xdit_flux1_dev_distributed.json``
     - ``xdit_flux1_dev_distributed_threshold.json``
     - ``xdit_flux_dev_distributed``
   * - ``mi3xx_xdit_flux2_dev_single.json``, ``mi35x_xdit_flux2_dev_single.json``
     - ``xdit_flux2_dev_single_threshold.json``
     - ``xdit_flux_dev_single``
   * - ``mi3xx_xdit_flux2_dev_distributed.json``, ``mi35x_xdit_flux2_dev_distributed.json``
     - ``xdit_flux2_dev_distributed_threshold.json``
     - ``xdit_flux_dev_distributed``
   * - ``mi3xx_xdit_wan22_14b_single.json``, ``mi35x_xdit_wan22_14b_single.json``
     - ``xdit_wan22_14b_single_threshold.json``
     - ``xdit_wan22_14b_single``
   * - ``mi3xx_xdit_wan22_14b_diffusers_single.json``, ``mi35x_xdit_wan22_14b_diffusers_single.json``
     - ``xdit_wan22_14b_diffusers_single_threshold.json``
     - ``xdit_wan22_14b_diffusers_single``
   * - ``mi3xx_xdit_wan22_14b_diffusers_distributed.json``, ``mi35x_xdit_wan22_14b_diffusers_distributed.json``
     - ``xdit_wan22_14b_diffusers_distributed_threshold.json``
     - ``xdit_wan22_14b_diffusers_distributed``

Copy a template and its threshold file:

.. code:: bash

  cvs config list inference/xdit
  cvs config copy inference/xdit/mi3xx_xdit_flux1_dev_single.json \
    --output ~/cvs_workspace/inference/xdit/mi3xx_xdit_flux1_dev_single.json
  cvs config copy inference/xdit/xdit_flux1_dev_single_threshold.json \
    --output ~/cvs_workspace/inference/xdit/xdit_flux1_dev_single_threshold.json

File structure
==============

Packaged templates use these top-level keys:

.. list-table::
   :widths: 2 6
   :header-rows: 1

   * - Key
     - Description
   * - ``gpu_name``
     - Label shipped as ``<changeme> mi325`` or ``<changeme> mi355``. Remove ``<changeme>``. Threshold selection uses the detected GPU, not this string.
   * - ``enforce_thresholds``
     - When ``true`` (the packaged default), latency must meet the selected GPU limit. ``false`` records the metric.
   * - ``threshold_json``
     - Sibling threshold filename. Both GPU families for a workload share one file.
   * - ``paths``
     - ``shared_fs``, ``models_dir``, ``log_dir``, ``hf_token_file``.
   * - ``container``
     - Image, name, and ``runtime.args`` (network, IPC, shm, volumes, devices, env).
   * - ``server_params``
     - ``backend`` (``xdit``), ``nnodes``, ``log_level``, and ``model``.
   * - ``benchmark_params``
     - Flat FLUX or WAN workload fields. The loader wraps them as ``flux1_dev_t2i`` or ``wan22_i2v_a14b``.

Example: FLUX.1-dev single-node
===============================

.. dropdown:: ``mi3xx_xdit_flux1_dev_single.json`` (abbreviated)

  .. code:: json

    {
        "_comment": "FLUX.1-dev single-node xDiT workload. Thresholds live in the sibling threshold JSON. Download model before run.",
        "gpu_name": "<changeme> mi325",
        "enforce_thresholds": true,
        "threshold_json": "xdit_flux1_dev_single_threshold.json",
        "paths": {
            "shared_fs": "{home}",
            "models_dir": "{home}/.cache/huggingface",
            "log_dir": "{home}/cvs_flux1_output",
            "hf_token_file": "{home}/.hf_token"
        },
        "container": {
            "lifetime": "per_run",
            "name": "flux-benchmark_single",
            "_image_example": "amdsiloai/pytorch-xdit:v25.11.2",
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
                    "env": { "NCCL_DEBUG": "ERROR" }
                }
            }
        },
        "server_params": {
            "backend": "xdit",
            "nnodes": "1",
            "log_level": "info",
            "model": "<changeme>"
        },
        "benchmark_params": {
            "prompt": "A small cat",
            "seed": 42,
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

``mi35x_xdit_flux1_dev_single.json`` matches that layout with ``gpu_name`` hint ``mi355``,
an example model path, and ``use_torch_compile`` set to ``false``.

General parameters
==================

The following parameters appear in every packaged xDiT template.

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Parameter
     - Example
     - Description
   * - ``container.image``
     - ``<changeme>``
     - Image to run. ``_image_example`` documents a known tag: ``amdsiloai/pytorch-xdit:v25.11.2`` for FLUX.1 and native WAN; a ``rocm/ufb-private`` gfx942 tag on ``mi3xx`` FLUX.2 and WAN Diffusers; a gfx950 tag on ``mi35x`` FLUX.2 and WAN Diffusers.
   * - ``container.name``
     - ``flux-benchmark_single``
     - Docker name.
   * - ``paths.hf_token_file``
     - ``{home}/.hf_token``
     - Host token file for gated models. The token is read on the runner and is not passed as ``HF_TOKEN`` on ``docker run``.
   * - ``paths.models_dir``
     - ``{home}/.cache/huggingface``
     - Host HF cache or model root, mounted at ``/hf_home``.
   * - ``paths.log_dir``
     - ``{home}/cvs_flux1_output``
     - Host directory for ``flux_<hostname>_outputs`` or ``wan_22_<hostname>_outputs``. Mounted at ``/outputs``.
   * - ``server_params.model``
     - HF id or ``/data/models/…``
     - Repo id (offline cache) or absolute host path. Stage the weights before the run.
   * - ``server_params.nnodes``
     - ``"1"`` / ``"2"``
     - Participating node count. Distributed templates require ``>= 2``.
   * - ``container.runtime.args.devices``
     - ``["/dev/dri", "/dev/kfd"]``
     - GPU device nodes. Distributed templates also pass InfiniBand devices.
   * - ``container.runtime.args.volumes``
     - ``host:container`` list
     - Bind mounts. FLUX.2 adds ``flux2_example.py``; WAN Diffusers adds ``wan_i2v_example.py``.
   * - ``container.runtime.args.env``
     - ``{"NCCL_DEBUG": "ERROR"}``
     - Extra environment variables inside the container.

Distributed network fields
--------------------------

Distributed templates put NCCL and socket settings in ``container.runtime.args.env``.
CVS copies ``NCCL_IB_HCA``, ``NCCL_IB_GID_INDEX``, ``NCCL_SOCKET_IFNAME``,
``GLOO_SOCKET_IFNAME``, ``GLOO_TCP_IFNAME``, and ``NCCL_DEBUG`` into the launch environment.

.. list-table::
   :widths: 3 5
   :header-rows: 1

   * - Template family
     - Shipped example
   * - ``mi3xx_*_distributed``
     - Broadcom bnxt library mounts, ``NCCL_IB_HCA`` ``rdma0``–``rdma7``, socket ``eno0``, GID index ``3``, ``NCCL_DEBUG`` ``ERROR``.
   * - ``mi35x_*_distributed``
     - Ionic userspace library mounts, ``NCCL_IB_HCA`` ``ionic_0``–``ionic_7``, socket ``ens3``, GID index ``1``, ``NCCL_DMABUF_ENABLE`` ``0``, ``NCCL_DEBUG`` ``ERROR``. The versioned ``libionic.so`` mount contains ``<changeme>``.

Replace every ``<changeme>`` value, and edit the Ionic or bnxt library paths when the host
build differs. ``master_addr`` is optional; an empty value probes the first address on
the rank-0 node. ``master_port`` defaults to ``29500``.

``benchmark_params`` for FLUX
=============================

Used by all FLUX templates. The loader stores the flat object as ``flux1_dev_t2i``.
FLUX.2 sets ``model_type`` to ``flux2``, which selects ``flux2_example.py``. FLUX.1 uses
``run_usp.py``. A repo or path containing ``flux2`` or ``flux.2`` is also treated as FLUX.2
when ``model_type`` is omitted.

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Parameter
     - Example
     - Description
   * - ``model_type``
     - ``flux2``
     - FLUX.2 only.
   * - ``prompt``, ``seed``
     - ``A small cat``, ``42``
     - Generation prompt and RNG seed.
   * - ``guidance_scale``
     - ``4.0``
     - FLUX.2 guidance. FLUX.1 templates omit this.
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
     - Warmup, then measured repetitions. FLUX.1 parse requires ``num_repetitions``, and ``timing.json`` must contain that many samples.
   * - ``height``, ``width``
     - ``1024``
     - Output image size.
   * - ``ulysses_degree``, ``ring_degree``
     - ``8``, ``1`` (single) / ``8``, ``2`` (2-node)
     - Sequence-parallel layout. Product with pipefusion, tensor-parallel, and data-parallel degrees (each default ``1``) must equal ``nnodes × torchrun_nproc``.
   * - ``use_torch_compile``
     - ``true`` / ``false``
     - ``mi3xx`` FLUX templates and ``mi35x`` FLUX.2 single set ``true``. ``mi35x`` FLUX.1 and ``mi35x`` FLUX.2 distributed set ``false``.
   * - ``torchrun_nproc``
     - ``8``
     - Processes (GPUs) per node.

``benchmark_params`` for WAN 2.2
================================

The loader stores the flat object as ``wan22_i2v_a14b``.

Native WAN (``*_xdit_wan22_14b_single.json``)
---------------------------------------------

Runs ``/app/Wan2.2/run.py``. The threshold metric is ``max_avg_total_time_s``.
``num_benchmark_steps`` is required, and the output must contain that many ``rank0_step*.json`` files.

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
     - Measured steps after compile/warmup. Parse requires this count of rank-0 step JSONs.
   * - ``compile``
     - ``true`` / ``false``
     - ``mi3xx`` ships ``true``. ``mi35x`` ships ``false``.
   * - ``torchrun_nproc``
     - ``8``
     - GPUs per node. Single-node native WAN infers ulysses from this value and ring ``1``.

Diffusers xFuser WAN
--------------------

``*_xdit_wan22_14b_diffusers_*.json`` additionally set:

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
     - In-container path to ``wan_i2v_example.py`` when you override the default ``/benchmark/wan_i2v_example.py``.
   * - ``wan_xfuser_auto_input_image``
     - Generate an in-container input image when true.
   * - ``wan_xfuser_install_video_deps``
     - Install video encode deps inside the container when true.
   * - ``wan_xfuser_output_type``
     - ``pil``.
   * - ``require_video_artifact``
     - Fail parse if ``video_i2v.mp4`` is missing. Packaged templates set ``true``.
   * - ``num_inference_steps``, ``warmup_steps``
     - Denoising and warmup (packaged templates: ``40`` and ``1``).
   * - ``num_benchmark_steps``
     - Packaged Diffusers templates set ``1``.
   * - ``compile``
     - Packaged Diffusers templates set ``false``.
   * - ``ulysses_size``, ``ring_size``
     - Parallel layout. Distributed: product must equal ``nnodes × torchrun_nproc``. Single-node templates ship ``8`` and ``1``.
   * - ``torchrun_nproc``
     - GPUs per node (``8``).

Volume mounts
=============

**FLUX.1 / WAN native** templates mount the model cache and the log directory:

.. code:: json

  "volumes": [
      "{paths.models_dir}:/hf_home",
      "{paths.log_dir}:/outputs"
  ]

An absolute ``server_params.model`` path that is not already mounted is bind-mounted at
``/model``.

**FLUX.2** also mounts the in-tree example. Adjust the host path to your CVS checkout:

.. code:: json

  "{home}/cvs/cvs/lib/inference/xdit/scripts/flux2_example.py:/benchmark/flux2_example.py"

**WAN Diffusers** mounts the xFuser example the same way:

.. code:: json

  "{home}/cvs/cvs/lib/inference/xdit/scripts/wan_i2v_example.py:/benchmark/wan_i2v_example.py"

Distributed templates add the host home directory, ``/dev/infiniband``, and the fabric
provider libraries (bnxt on ``mi3xx``, Ionic on ``mi35x``).

Threshold files
===============

Each threshold file is a map of GPU key to one latency limit. Keys that start with ``_``
are comments and are ignored. Allowed specific keys are ``mi300x``, ``mi325``, ``mi350``,
and ``mi355``. ``auto`` is the fallback when the detected key is absent.

GPU type comes from the PCI device ID under ``/sys/class/drm/card*/device/device``.
``0x74a5`` selects ``mi325``, ``0x75a0`` selects ``mi350``, and ``0x75a3`` selects
``mi355``. Any other ID, including MI300X, selects ``mi300x``. A threshold key applies
only when detection reports that same string.

.. list-table::
   :widths: 4 2 6
   :header-rows: 1

   * - Threshold file
     - Metric
     - Shipped keys (seconds)
   * - ``xdit_flux1_dev_single_threshold.json``
     - ``max_avg_pipe_time_s``
     - ``auto`` 10, ``mi300x`` 3, ``mi350`` 2, ``mi355`` 7
   * - ``xdit_flux1_dev_distributed_threshold.json``
     - ``max_avg_pipe_time_s``
     - ``auto`` 12, ``mi300x`` 10, ``mi355`` 22
   * - ``xdit_flux2_dev_single_threshold.json``
     - ``max_avg_pipe_time_s``
     - ``auto`` 10, ``mi300x`` 8, ``mi355`` 7
   * - ``xdit_flux2_dev_distributed_threshold.json``
     - ``max_avg_pipe_time_s``
     - ``auto`` 20, ``mi300x`` 16, ``mi355`` 50
   * - ``xdit_wan22_14b_single_threshold.json``
     - ``max_avg_total_time_s``
     - ``auto`` 15, ``mi300x`` 200, ``mi325`` 160, ``mi350`` 120, ``mi355`` 120
   * - ``xdit_wan22_14b_diffusers_single_threshold.json``
     - ``max_avg_pipe_time_s``
     - ``auto`` 300, ``mi325`` 300, ``mi355`` 300
   * - ``xdit_wan22_14b_diffusers_distributed_threshold.json``
     - ``max_avg_pipe_time_s``
     - ``auto`` 300, ``mi325`` 300, ``mi355`` 900

A detected GPU with no key of its own uses ``auto``. MI350X on
``xdit_flux1_dev_distributed_threshold.json`` has no ``mi350`` entry, so it uses the
``auto`` limit of 12 seconds. MI325X (``0x74a5``) uses the ``mi325`` limit when that key
exists, and ``auto`` otherwise.

Comments in the FLUX.1 single and native WAN files mark which keys were measured and
which remain placeholders. Tune the file for your stack before production gating.
Set ``enforce_thresholds`` to ``false`` to record latency without failing the parse stage.

Performance metrics
===================

- **FLUX** — average ``pipe_time`` vs ``max_avg_pipe_time_s``; artifacts ``results/timing.json`` and ``flux_*.png``. FLUX.1 also requires ``num_repetitions`` samples.
- **WAN native** — average ``total_time`` vs ``max_avg_total_time_s``; ``rank0_step*.json`` (count must equal ``num_benchmark_steps``) and ``video.mp4``.
- **WAN Diffusers** — average pipe/epoch time vs ``max_avg_pipe_time_s``; ``results/timing.json`` and ``results/video_i2v.mp4``.

Troubleshooting
===============

**``/dev/kfd not found``**
  Run on GPU compute nodes, not login nodes. ``test_verify_prerequisites`` also requires ``torchrun`` inside the container.

**Container image not found locally**
  ``docker pull`` the configured ``container.image`` on every execution node. The image field must replace ``<changeme>``.

**Local model path not found**
  Stage weights on every participating node before the run. A Hugging Face repo id must already exist under ``paths.models_dir``. An absolute ``server_params.model`` path must exist on each participating host.

**Parallel degree product != world_size**
  Align ``ulysses`` / ``ring`` (and FLUX pipefusion, tensor-parallel, and data-parallel degrees) with ``nnodes × torchrun_nproc``.

**FLUX.1 requires num_repetitions / native WAN requires num_benchmark_steps**
  Set the field in ``benchmark_params``. The parsed artifact count must match that value.

**Missing ``timing.json`` / ``video.mp4``**
  The benchmark docker exit code was non-zero or artifacts were written elsewhere; inspect the log tail on the failing node.

**MI355X threshold uses ``auto`` or ``mi300x``**
  Confirm the node exposes device ID ``0x75a3``. A missing ``mi355`` key in the threshold file selects ``auto``. An unrecognized device ID selects ``mi300x``. MI325X is ``0x74a5`` and selects ``mi325``.
