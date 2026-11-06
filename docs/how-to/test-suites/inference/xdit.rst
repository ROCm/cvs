.. meta::
  :description: Run xDiT diffusion inference tests for FLUX.1, FLUX.2, and WAN 2.2 workloads on AMD Instinct GPU clusters with CVS and ROCm.
  :keywords: CVS, xDiT, inference, AMD Instinct, ROCm, AMD, GPU, FLUX, WAN, diffusion, distributed, PyTorch, CLIPScore

**********************************************************************
Run xDiT diffusion inference tests with Cluster Validation Suite (CVS)
**********************************************************************

CVS provides seven xDiT suites under |cvs-tests-inference-xdit|_. Each suite is a
separate pytest module; pick the one that matches your topology and launcher, then point
``--config_file`` at a template from |cvs-input-config-file-inference-xdit|_.

- **Single-node** suites run one independent docker+torchrun job on **every** node in
  the cluster file (full model on each node).
- **Distributed** suites run one coordinated torchrun job across ``server_params.nnodes``
  (``nnodes >= 2``), using the first that many cluster hosts, or ``server_node_list`` when set.

FLUX.1-dev and FLUX.2-dev share the ``xdit_flux_dev_*`` suites; choose the matching
``flux1`` or ``flux2`` JSON. Stage model weights on every participating node before the
run. Packaged configs keep Hugging Face downloads off.

``mi3xx_*`` templates target MI300X, MI325X, and MI350X. ``mi35x_*`` templates target
MI355X, including Ionic RDMA examples for distributed runs. Both families point
``threshold_json`` at the same sibling threshold file for that workload.

Config reference: :doc:`/reference/configuration-files/inference/xdit`.

Test suites
===========

The following suites are available. Every suite runs the same stages except
``xdit_wan22_14b_single``, which has no FID vs steps stage.

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - CVS suite name
     - Source module
     - What it runs
   * - ``xdit_flux_dev_single``
     - ``xdit_flux_dev_single.py``
     - FLUX.1 (bind-mounted ``run_usp.py``) or FLUX.2 (``flux2_example.py``); one job per cluster node.
   * - ``xdit_flux_dev_distributed``
     - ``xdit_flux_dev_distributed.py``
     - Unified FLUX.1 and FLUX.2 torchrun across ``nnodes``.
   * - ``xdit_wan22_14b_single``
     - ``xdit_wan22_14b_single.py``
     - WAN 2.2 I2V native (``/app/Wan2.2/run.py``); one job per cluster node. No FID vs steps stage.
   * - ``xdit_wan22_14b_diffusers_single``
     - ``xdit_wan22_14b_diffusers_single.py``
     - WAN Diffusers xFuser (``wan_i2v_example.py``); one job per cluster node.
   * - ``xdit_wan22_14b_diffusers_distributed``
     - ``xdit_wan22_14b_diffusers_distributed.py``
     - Unified WAN Diffusers xFuser torchrun across ``nnodes``.
   * - ``xdit_wan21_t2v_14b_diffusers_single``
     - ``xdit_wan21_t2v_14b_diffusers_single.py``
     - WAN 2.1 T2V Diffusers (bind-mounted ``wan_t2v_example.py``); one job per cluster node.
   * - ``xdit_wan21_t2v_14b_diffusers_distributed``
     - ``xdit_wan21_t2v_14b_diffusers_distributed.py``
     - Unified WAN 2.1 T2V Diffusers torchrun across ``nnodes``.

.. _xdit-set-up-config:

Set up config
=============

Follow these steps to set up the xDiT configuration.

1. List available xDiT templates:

   .. code:: bash

     cvs config list inference/xdit

2. Copy the workload JSON and its sibling threshold file into the same directory.
   FLUX.1-dev and FLUX.2-dev share the same suites; copy the matching ``flux1`` or
   ``flux2`` JSON. Use ``mi35x_*`` on MI355X.

   .. code:: bash

     cvs config copy inference/xdit/mi3xx_xdit_flux1_dev_single.json \
       --output ~/cvs_workspace/inference/xdit/mi3xx_xdit_flux1_dev_single.json

     cvs config copy inference/xdit/xdit_flux1_dev_single_threshold.json \
       --output ~/cvs_workspace/inference/xdit/xdit_flux1_dev_single_threshold.json

     cvs config copy inference/xdit/mi35x_xdit_flux1_dev_distributed.json \
       --output ~/cvs_workspace/inference/xdit/mi35x_xdit_flux1_dev_distributed.json

     cvs config copy inference/xdit/xdit_flux1_dev_distributed_threshold.json \
       --output ~/cvs_workspace/inference/xdit/xdit_flux1_dev_distributed_threshold.json

3. Copy a cluster file (GPU compute nodes only):

   .. code:: bash

     cvs config copy cluster_container.json --output ~/cvs_workspace/cluster.json

4. Edit the config:

   - Set ``container.image`` (``_image_example`` names a known image for that template).
   - Set ``server_params.model`` to a Hugging Face repo id or an absolute host path, and
     stage those weights before the run.
   - Set ``paths.models_dir`` by removing ``<changeme>``. It is the host directory that holds
     the model weights and the scoring checkpoints from step 5, mounted at ``/hf_home``.
   - Set ``gpu_name`` by removing ``<changeme>`` (templates hint ``mi325`` or ``mi355``).
   - On distributed templates, replace every ``<changeme>`` NCCL/socket field and confirm
     the RDMA library bind mounts match the host.
   - Resolve ``{user-id}`` and ``{home}`` or leave them for CVS to expand.
   - Keep ``threshold_json`` as the sibling filename unless you point it at your own file.

5. Download the scoring checkpoints into ``paths.models_dir`` on every participating node.
   CVS never downloads them, and the stages that need them fail when they are missing.

   - **CLIP** ``openai/clip-vit-base-patch32`` is the default ``clip_model``. ``test_clip_score``
     scores with it, and ``test_verify_model`` checks it is already present offline.

     .. code:: bash

       HF_HOME=/data/models huggingface-cli download openai/clip-vit-base-patch32

     This writes ``/data/models/hub/models--openai--clip-vit-base-patch32``, which is
     ``/hf_home`` inside the container. A local copy of the model directory also works when
     you set ``clip_model`` to its container path, for example ``/hf_home/clip-vit-base-patch32``.

   - **Inception-v3** ``inception_v3.pth`` is used by ``test_fid_curve`` when
     ``fid_nfe_steps`` is set. Place it at ``/data/models/inception_v3.pth`` so the sample
     ``fid_inception_path`` of ``/hf_home/inception_v3.pth`` resolves. A missing file skips
     ``test_fid_curve``. CVS does not download it. It must be a torchvision
     ``inception_v3`` state dict from ``Inception_V3_Weights.IMAGENET1K_V1``, saved with
     ``torch.save(model.state_dict(), ...)``. The xDiT image ships torch and torchvision, so
     you can generate it there:

     .. code:: bash

       docker run --rm -v /data/models:/hf_home rocm/pytorch-xdit:v26.9 python3 -c \
         "import torch, torchvision as tv; m = tv.models.inception_v3(weights=tv.models.Inception_V3_Weights.IMAGENET1K_V1); torch.save(m.state_dict(), '/hf_home/inception_v3.pth')"

   - **FID reference images** for ``fid_reference_dir`` (``/hf_home/fid_reference`` in the samples)
     are optional. When that directory is missing or has fewer than two images, the score uses
     the highest step-count samples this run saved under ``results/fid/``. A staged set of real
     PNG or JPG files is used instead when it has at least two images.

Shipped config templates and the suites they target:

.. list-table::
   :widths: 4 3
   :header-rows: 1

   * - Config file
     - Use with suite
   * - ``mi3xx_xdit_flux1_dev_single.json`` or ``mi35x_xdit_flux1_dev_single.json``
     - ``xdit_flux_dev_single``
   * - ``mi3xx_xdit_flux1_dev_distributed.json`` or ``mi35x_xdit_flux1_dev_distributed.json``
     - ``xdit_flux_dev_distributed``
   * - ``mi3xx_xdit_flux2_dev_single.json`` or ``mi35x_xdit_flux2_dev_single.json``
     - ``xdit_flux_dev_single``
   * - ``mi3xx_xdit_flux2_dev_distributed.json`` or ``mi35x_xdit_flux2_dev_distributed.json``
     - ``xdit_flux_dev_distributed``
   * - ``mi3xx_xdit_wan22_14b_single.json`` or ``mi35x_xdit_wan22_14b_single.json``
     - ``xdit_wan22_14b_single``
   * - ``mi3xx_xdit_wan22_14b_diffusers_single.json`` or ``mi35x_xdit_wan22_14b_diffusers_single.json``
     - ``xdit_wan22_14b_diffusers_single``
   * - ``mi3xx_xdit_wan22_14b_diffusers_distributed.json`` or ``mi35x_xdit_wan22_14b_diffusers_distributed.json``
     - ``xdit_wan22_14b_diffusers_distributed``
   * - ``mi3xx_xdit_wan21_t2v_14b_diffusers_single.json`` or ``mi35x_xdit_wan21_t2v_14b_diffusers_single.json``
     - ``xdit_wan21_t2v_14b_diffusers_single``
   * - ``mi3xx_xdit_wan21_t2v_14b_diffusers_distributed.json`` or ``mi35x_xdit_wan21_t2v_14b_diffusers_distributed.json``
     - ``xdit_wan21_t2v_14b_diffusers_distributed``

Each workload has one shared threshold file (``xdit_<workload>_threshold.json``).
``mi3xx`` and ``mi35x`` configs for the same workload reference that same file.
GPU keys inside it (``mi300x``, ``mi325``, ``mi350``, ``mi355``, ``auto``) select the limit.

.. note::

  FLUX.1 configs bind-mount |cvs-lib-inference-xdit-scripts-run-usp-py|_ at
  ``/benchmark/run_usp.py`` and ``flux_utils.py`` beside it at ``/benchmark/flux_utils.py``.
  FLUX.2 configs bind-mount |cvs-lib-inference-xdit-scripts-flux2-example-py|_ at
  ``/benchmark/flux2_example.py``. The copy baked into the image does not accept
  the benchmark arguments.
  WAN 2.2 Diffusers suites mount
  |cvs-lib-inference-xdit-scripts-wan-i2v-example-py|_. WAN 2.1 T2V Diffusers suites mount
  |cvs-lib-inference-xdit-scripts-wan-t2v-example-py|_. Configs that set ``fid_nfe_steps``
  also mount ``fid_sweep.py`` at ``/benchmark/fid_sweep.py``. Adjust the host side of those
  volume entries to your CVS checkout.

  ``mi3xx`` distributed templates ship Broadcom bnxt RDMA mounts and ``rdma*`` HCA names.
  ``mi35x`` distributed templates ship Ionic library mounts and ``ionic_*`` HCA names.
  Replace the ``<changeme>`` interface, HCA, and GID values for your fabric.

  On shared clusters, skip aggressive docker prune during container setup:

  .. code:: bash

    export CVS_SKIP_DOCKER_SYSTEM_PRUNE=1

.. _xdit-run-tests:

Run tests
=========

List stages in a suite:

.. code:: bash

  cvs list xdit_flux_dev_single

Every xDiT suite uses this order, except that ``xdit_wan22_14b_single`` omits ``test_fid_curve``:

.. code:: text

  Available tests in xdit_flux_dev_single:
    - test_launch_container
    - test_verify_prerequisites
    - test_verify_model
    - test_verify_parallelism
    - test_run_benchmark
    - test_parse_thresholds
    - test_clip_score
    - test_fid_curve
    - test_print_results
    - test_teardown

``test_verify_parallelism`` checks the parallel-degree product:

- **FLUX** (single and distributed): ``ulysses_degree × ring_degree × pipefusion × tensor_parallel × data_parallel == nnodes × torchrun_nproc``. Single-node uses ``nnodes = 1``.
- **WAN distributed**: ``ulysses_size × ring_size == nnodes × torchrun_nproc``. Single-node WAN logs the layout and skips that product check.

Example run (FLUX.1-dev on an MI300-family node; use the flux2 JSON for FLUX.2-dev, or ``mi35x_*`` on MI355X):

.. code:: bash

  cvs run xdit_flux_dev_single \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/inference/xdit/mi3xx_xdit_flux1_dev_single.json \
    -vvv

Distributed FLUX.1-dev on MI355X:

.. code:: bash

  cvs run xdit_flux_dev_distributed \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/inference/xdit/mi35x_xdit_flux1_dev_distributed.json \
    -vvv

WAN 2.2 native:

.. code:: bash

  cvs run xdit_wan22_14b_single \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/inference/xdit/mi3xx_xdit_wan22_14b_single.json \
    -vvv

WAN Diffusers:

.. code:: bash

  cvs run xdit_wan22_14b_diffusers_single \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/inference/xdit/mi3xx_xdit_wan22_14b_diffusers_single.json \
    -vvv

  cvs run xdit_wan22_14b_diffusers_distributed \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/inference/xdit/mi35x_xdit_wan22_14b_diffusers_distributed.json \
    -vvv

WAN 2.1 T2V Diffusers:

.. code:: bash

  cvs run xdit_wan21_t2v_14b_diffusers_single \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/inference/xdit/mi3xx_xdit_wan21_t2v_14b_diffusers_single.json \
    -vvv

  cvs run xdit_wan21_t2v_14b_diffusers_distributed \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/inference/xdit/mi3xx_xdit_wan21_t2v_14b_diffusers_distributed.json \
    -vvv

Direct pytest invocation
------------------------

Each module can also be run with pytest:

.. code:: bash

  pytest cvs/tests/inference/xdit/xdit_flux_dev_single.py \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/inference/xdit/mi3xx_xdit_flux1_dev_single.json \
    --html ~/cvs_results/xdit_flux1_single.html

Run on managed compute
----------------------

Follow :doc:`/how-to/run-on-managed-compute` to run xDiT on managed compute.

Read the results
================

``cvs run`` writes the pytest HTML report described in :ref:`pytest-html-report-and-bundle`. Benchmark pass/fail uses the docker
exit code plus parsed artifacts and the GPU key in the sibling threshold file.

The suite reads PCI device IDs from ``/sys/class/drm/card*/device/device``. ``0x74a5``
selects ``mi325``, ``0x75a0`` selects ``mi350``, and ``0x75a3`` selects ``mi355``. Any
other ID, including MI300X, selects ``mi300x``. Lookup then uses that key, or ``auto``
when the threshold file has no entry for it. ``test_print_results`` prints PASS or FAIL for each host. With
``enforce_thresholds`` set to ``false``, the same row is RECORDED and does not fail the stage.

Each stage corresponds to a test function in the HTML report.

.. list-table::
   :header-rows: 1
   :widths: 16 30 54

   * - Stage
     - Test function
     - Notes
   * - Launch
     - ``test_launch_container``
     - Starts the named container. Setup may run ``docker system prune``
       unless ``CVS_SKIP_DOCKER_SYSTEM_PRUNE=1``.
   * - Prerequisites
     - ``test_verify_prerequisites``
     - Requires ``/dev/kfd`` and ``torchrun`` inside the container on every
       participating node.
   * - Model
     - ``test_verify_model``
     - Checks staged weights and that the CLIP checkpoint is in the
       container cache. A missing cache or host path fails the stage. The
       suite does not download weights. ``enforce_thresholds: false`` does
       not skip a missing CLIP checkpoint.
   * - Parallelism
     - ``test_verify_parallelism``
     -
   * - Benchmark
     - ``test_run_benchmark``
     -
   * - Parse
     - ``test_parse_thresholds``
     - Compares average latency to the threshold file. FLUX.1 and FLUX.2
       require ``num_repetitions`` and that ``timing.json`` contains that
       many ``pipe_time`` samples. FLUX.2 records them in the same torchrun
       session. Native WAN requires ``num_benchmark_steps`` and that many
       ``rank0_step*.json`` files.
   * - CLIP score
     - ``test_clip_score``
     - Scores saved media inside the benchmark container. FLUX scores
       ``flux_*.png`` and ``flux2_*.png`` (``clip_score`` against
       ``min_clip_score``). WAN samples 8 frames from ``video.mp4``
       (native) or ``video_i2v.mp4`` (Diffusers) and gates
       ``mean_frame_clip_score`` against ``min_mean_frame_clip_score``.
       Uses ``openai/clip-vit-base-patch32``; prompts longer than 77
       tokens are truncated (``results/clipscore.json`` sets
       ``prompt_truncated``). WAN requires ``imageio`` with
       ``imageio-ffmpeg``, OpenCV, or ``torchvision`` to decode the MP4.
       The shipped floor of 20 is an unvalidated placeholder.
       ``enforce_thresholds: false`` records a score that misses the floor
       without failing the stage, but does not skip a missing CLIP
       checkpoint. A failed ``test_parse_thresholds`` does not skip this
       stage. A failed earlier stage still does.
   * - FID vs steps
     - ``test_fid_curve``
     - Skipped unless ``fid_nfe_steps`` lists two or more sampling step counts.
       Not part of ``xdit_wan22_14b_single``, which fails the benchmark stage if
       ``fid_nfe_steps`` is set. The benchmark saves extra samples under ``results/fid/nfe_<steps>/``
       after the timed run, then stops at least 180 seconds before the
       benchmark command timeout. ``test_fid_curve`` scores those files
       against ``fid_reference_dir`` with the Inception checkpoint at
       ``fid_inception_path``; both are staged in step 5. A single-node WAN run
       that sets ``fid_nfe_steps`` uses a two-hour benchmark timeout so the extra
       videos fit after the timed pass. A missing Inception checkpoint skips the stage.
       CVS does not download it. A missing or short ``fid_reference_dir`` uses the highest
       step-count samples from this run. The interactive viewer plots FID
       against the step counts, one line per host. A sweep that the
       timeout cuts short is skipped, not failed.
   * - Teardown
     - ``test_teardown``
     - Stops the container.

The metrics, threshold keys, and output artifacts for each benchmark family are listed below.

.. list-table::
   :widths: 2 3 3 3
   :header-rows: 1

   * - Family
     - Metric
     - Threshold key
     - Artifacts
   * - FLUX
     - average ``pipe_time`` from ``results/timing.json``
     - ``max_avg_pipe_time_s``
     - ``timing.json``, ``flux_*.png``
   * - WAN native
     - average ``total_time`` from ``rank0_step*.json``
     - ``max_avg_total_time_s``
     - step JSONs, ``video.mp4``
   * - WAN Diffusers
     - average epoch and pipe time from ``results/timing.json``
     - ``max_avg_pipe_time_s``
     - ``results/timing.json``, ``results/video_i2v.mp4``
   * - FLUX
     - mean CLIPScore of the saved PNGs
     - ``min_clip_score``
     - ``results/clipscore.json``
   * - WAN native
     - mean CLIPScore of 8 frames from ``video.mp4``
     - ``min_mean_frame_clip_score``
     - ``results/clipscore.json``
   * - WAN Diffusers
     - mean CLIPScore of 8 frames from ``video_i2v.mp4``
     - ``min_mean_frame_clip_score``
     - ``results/clipscore.json``

Single-node output dirs use the node hostname, for example
``${paths.log_dir}/flux_<hostname>_outputs`` or ``wan_22_<hostname>_outputs``.
Distributed runs write to the rank-0 hostname directory.


.. |cvs-tests-inference-xdit| replace:: ``cvs/tests/inference/xdit/``
.. _cvs-tests-inference-xdit: https://github.com/ROCm/cvs/tree/main/cvs/tests/inference/xdit/

.. |cvs-input-config-file-inference-xdit| replace:: ``cvs/input/config_file/inference/xdit/``
.. _cvs-input-config-file-inference-xdit: https://github.com/ROCm/cvs/tree/main/cvs/input/config_file/inference/xdit/

.. |cvs-lib-inference-xdit-scripts-run-usp-py| replace:: ``cvs/lib/inference/xdit/scripts/run_usp.py``
.. _cvs-lib-inference-xdit-scripts-run-usp-py: https://github.com/ROCm/cvs/blob/main/cvs/lib/inference/xdit/scripts/run_usp.py

.. |cvs-lib-inference-xdit-scripts-flux2-example-py| replace:: ``cvs/lib/inference/xdit/scripts/flux2_example.py``
.. _cvs-lib-inference-xdit-scripts-flux2-example-py: https://github.com/ROCm/cvs/blob/main/cvs/lib/inference/xdit/scripts/flux2_example.py

.. |cvs-lib-inference-xdit-scripts-wan-i2v-example-py| replace:: ``cvs/lib/inference/xdit/scripts/wan_i2v_example.py``
.. _cvs-lib-inference-xdit-scripts-wan-i2v-example-py: https://github.com/ROCm/cvs/blob/main/cvs/lib/inference/xdit/scripts/wan_i2v_example.py

.. |cvs-lib-inference-xdit-scripts-wan-t2v-example-py| replace:: ``cvs/lib/inference/xdit/scripts/wan_t2v_example.py``
.. _cvs-lib-inference-xdit-scripts-wan-t2v-example-py: https://github.com/ROCm/cvs/blob/main/cvs/lib/inference/xdit/scripts/wan_t2v_example.py
