.. meta::
  :description: Run xDiT diffusion inference tests for FLUX.1, FLUX.2, and WAN 2.2 workloads on AMD Instinct GPU clusters with CVS and ROCm.
  :keywords: CVS, xDiT, inference, AMD Instinct, ROCm, AMD, GPU, FLUX, WAN, diffusion, distributed, PyTorch

**************************************************
Run xDiT diffusion inference tests with CVS
**************************************************

CVS provides five xDiT suites under ``cvs/tests/inference/xdit/``. Each suite is a
separate pytest module; pick the one that matches your topology and launcher, then point
``--config_file`` at a template from ``cvs/input/config_file/inference/xdit/``.

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

The following suites are available. Every suite runs the same eight stages.

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - CVS suite name
     - Source module
     - What it runs
   * - ``xdit_flux_dev_single``
     - ``xdit_flux_dev_single.py``
     - FLUX.1 (``run_usp.py``) or FLUX.2 (``flux2_example.py``); one job per cluster node.
   * - ``xdit_flux_dev_distributed``
     - ``xdit_flux_dev_distributed.py``
     - Unified FLUX.1 / FLUX.2 torchrun across ``nnodes``.
   * - ``xdit_wan22_14b_single``
     - ``xdit_wan22_14b_single.py``
     - WAN 2.2 I2V native (``/app/Wan2.2/run.py``); one job per cluster node.
   * - ``xdit_wan22_14b_diffusers_single``
     - ``xdit_wan22_14b_diffusers_single.py``
     - WAN Diffusers xFuser (``wan_i2v_example.py``); one job per cluster node.
   * - ``xdit_wan22_14b_diffusers_distributed``
     - ``xdit_wan22_14b_diffusers_distributed.py``
     - Unified WAN Diffusers xFuser torchrun across ``nnodes``.

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
   - Set ``gpu_name`` by removing ``<changeme>`` (templates hint ``mi325`` or ``mi355``).
   - On distributed templates, replace every ``<changeme>`` NCCL/socket field and confirm
     the RDMA library bind mounts match the host.
   - Resolve ``{user-id}`` / ``{home}`` or leave them for CVS to expand.
   - Keep ``threshold_json`` as the sibling filename unless you point it at your own file.

Shipped config templates:

.. list-table::
   :widths: 4 3
   :header-rows: 1

   * - Config file
     - Use with suite
   * - ``mi3xx_xdit_flux1_dev_single.json`` / ``mi35x_xdit_flux1_dev_single.json``
     - ``xdit_flux_dev_single``
   * - ``mi3xx_xdit_flux1_dev_distributed.json`` / ``mi35x_xdit_flux1_dev_distributed.json``
     - ``xdit_flux_dev_distributed``
   * - ``mi3xx_xdit_flux2_dev_single.json`` / ``mi35x_xdit_flux2_dev_single.json``
     - ``xdit_flux_dev_single``
   * - ``mi3xx_xdit_flux2_dev_distributed.json`` / ``mi35x_xdit_flux2_dev_distributed.json``
     - ``xdit_flux_dev_distributed``
   * - ``mi3xx_xdit_wan22_14b_single.json`` / ``mi35x_xdit_wan22_14b_single.json``
     - ``xdit_wan22_14b_single``
   * - ``mi3xx_xdit_wan22_14b_diffusers_single.json`` / ``mi35x_xdit_wan22_14b_diffusers_single.json``
     - ``xdit_wan22_14b_diffusers_single``
   * - ``mi3xx_xdit_wan22_14b_diffusers_distributed.json`` / ``mi35x_xdit_wan22_14b_diffusers_distributed.json``
     - ``xdit_wan22_14b_diffusers_distributed``

Each workload has one shared threshold file (``xdit_<workload>_threshold.json``).
``mi3xx`` and ``mi35x`` configs for the same workload reference that same file.
GPU keys inside it (``mi300x``, ``mi325``, ``mi350``, ``mi355``, ``auto``) select the limit.

.. note::

  FLUX.2 configs bind-mount ``cvs/lib/inference/xdit/scripts/flux2_example.py`` when the
  image does not ship ``/benchmark/flux2_example.py``. WAN Diffusers suites mount
  ``cvs/lib/inference/xdit/scripts/wan_i2v_example.py``. Adjust the host side of those
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

Every xDiT suite uses this order:

.. code:: text

  Available tests in xdit_flux_dev_single:
    - test_launch_container
    - test_verify_prerequisites
    - test_verify_model
    - test_verify_parallelism
    - test_run_benchmark
    - test_parse_thresholds
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

Direct pytest invocation
------------------------

Each module can also be run with pytest:

.. code:: bash

  pytest cvs/tests/inference/xdit/xdit_flux_dev_single.py \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/inference/xdit/mi3xx_xdit_flux1_dev_single.json \
    --html ~/cvs_results/xdit_flux1_single.html

Read the results
================

``cvs run`` writes a pytest HTML report under the run directory by default. Benchmark pass/fail uses the docker
exit code plus parsed artifacts and the GPU key in the sibling threshold file.

The suite reads PCI device IDs from ``/sys/class/drm/card*/device/device``. ``0x74a5``
selects ``mi325``, ``0x75a0`` selects ``mi350``, and ``0x75a3`` selects ``mi355``. Any
other ID, including MI300X, selects ``mi300x``. Lookup then uses that key, or ``auto``
when the threshold file has no entry for it. ``test_print_results`` prints PASS or FAIL for each host. With
``enforce_thresholds`` set to ``false``, the same row is RECORDED and does not fail the stage.

Key stages to watch:

- **Launch** — ``test_launch_container`` starts the named container. Container setup may
  run ``docker system prune`` unless ``CVS_SKIP_DOCKER_SYSTEM_PRUNE=1``.
- **Prerequisites** — ``test_verify_prerequisites`` requires ``/dev/kfd`` and ``torchrun``
  inside the container on every participating node.
- **Model** — ``test_verify_model`` checks the staged weights. A missing cache or host
  path fails the stage. The suite does not download weights.
- **Parallelism** — ``test_verify_parallelism``.
- **Benchmark** — ``test_run_benchmark``.
- **Parse** — ``test_parse_thresholds`` compares the average latency to the threshold file.
  FLUX.1 requires ``num_repetitions``, and ``timing.json`` must contain that many
  ``pipe_time`` samples. Native WAN requires ``num_benchmark_steps``, and that many
  ``rank0_step*.json`` files must be present.
- **Teardown** — ``test_teardown`` stops the container.

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
     - average epoch / pipe time from ``results/timing.json``
     - ``max_avg_pipe_time_s``
     - ``results/timing.json``, ``results/video_i2v.mp4``

Single-node output dirs use the node hostname, for example
``${paths.log_dir}/flux_<hostname>_outputs`` or ``wan_22_<hostname>_outputs``.
Distributed runs write to the rank-0 hostname directory.
