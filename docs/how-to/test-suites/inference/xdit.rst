.. meta::
  :description: Run Flux.1, Flux.2, and WAN 2.2 xDiT inference tests
  :keywords: CVS, flux1_t2i, flux2, wan22, xdit

*************************
Run xDiT inference tests
*************************

CVS provides five xDiT suites under ``cvs/tests/inference/xdit/``. Each suite is a
separate pytest module; pick the one that matches your topology and launcher, then point
``--config_file`` at a template from ``cvs/input/config_file/inference/xdit/``.

- **Single-node** suites run one independent torchrun job inside the orchestrated
  container on every node in the cluster ``node_dict``, and report per-host results.
- **Distributed** suites run one coordinated torchrun job across ``nnodes`` (``nnodes >= 2``),
  using ``server_node_list`` when set.

FLUX.1-dev and FLUX.2-dev share the ``pytorch_xdit_flux_dev_*`` suites; choose the matching
``flux1`` or ``flux2`` JSON. ``server_params.model`` may be a Hugging Face repo id
(downloaded into ``paths.models_dir`` / ``HF_HOME`` during ``test_verify_model``) or an
absolute host path (bind-mounted at ``/model``). Gated repos need ``paths.hf_token_file``.

Config reference: :doc:`/reference/configuration-files/inference/xdit`.

Test suites
===========

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - CVS suite name
     - Source module
     - What it runs
   * - ``pytorch_xdit_flux_dev_single``
     - ``pytorch_xdit_flux_dev_single.py``
     - FLUX.1 (``run_usp.py``) or FLUX.2 (``flux2_example.py``); one job per cluster node.
   * - ``pytorch_xdit_flux_dev_distributed``
     - ``pytorch_xdit_flux_dev_distributed.py``
     - Unified FLUX.1 / FLUX.2 torchrun across ``nnodes``.
   * - ``pytorch_xdit_wan22_14b_single``
     - ``pytorch_xdit_wan22_14b_single.py``
     - WAN 2.2 I2V native (``/app/Wan2.2/run.py``); one job per cluster node.
   * - ``pytorch_xdit_wan22_14b_diffusers_single``
     - ``pytorch_xdit_wan22_14b_diffusers_single.py``
     - WAN Diffusers xFuser (``wan_i2v_example.py``); one job per cluster node.
   * - ``pytorch_xdit_wan22_14b_diffusers_distributed``
     - ``pytorch_xdit_wan22_14b_diffusers_distributed.py``
     - Unified WAN Diffusers xFuser torchrun across ``nnodes``.

.. _xdit-set-up-config:

Set up config
=============

1. List available xDiT templates:

   .. code:: bash

     cvs config list inference/xdit

2. Copy the configuration for your workload. FLUX.1-dev and FLUX.2-dev share the same
   suites; copy the matching ``flux1`` or ``flux2`` JSON:

   .. code:: bash

     cvs config copy inference/xdit/mi3xx_pytorch_xdit_flux1_dev_single.json \
       --output ~/cvs_workspace/inference/xdit/mi3xx_pytorch_xdit_flux1_dev_single.json

     cvs config copy inference/xdit/mi3xx_pytorch_xdit_wan22_14b_single.json \
       --output ~/cvs_workspace/inference/xdit/mi3xx_pytorch_xdit_wan22_14b_single.json

3. Copy a cluster file (GPU compute nodes only):

   .. code:: bash

     cvs config copy cluster_container.json --output ~/cvs_workspace/cluster.json

4. Copy the sibling ``*_threshold.json`` beside each workload config and keep both files
   in the same directory.

5. Edit the config — set ``container.image``, ``model.id`` / ``paths.models_dir``,
   ``nnodes`` for distributed runs, and replace every ``<changeme>``. Resolve
   ``{user-id}`` / ``{home}`` or leave them for CVS to expand.

Shipped config templates:

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

.. note::

  FLUX.2 configs bind-mount ``cvs/lib/inference/xdit/scripts/flux2_example.py`` when the
  image does not ship ``/app/external/xdit/examples/flux2_example.py``. WAN Diffusers
  mounts ``cvs/lib/inference/xdit/scripts/wan_i2v_example.py`` from the CVS checkout on
  the cluster (not a copy baked into the image). Keep that file current on every
  execution node; current xFuser images require the in-tree launcher.

  On shared clusters, skip aggressive docker prune during cleanup:

  .. code:: bash

    export CVS_PYTORCH_XDIT_SKIP_DOCKER_SYSTEM_PRUNE=1

.. _xdit-run-tests:

Run tests
=========

List stages in a suite:

.. code:: bash

  cvs list pytorch_xdit_flux_dev_single

``pytorch_xdit_flux_dev_single`` stages
---------------------------------------

.. code:: text

  Available tests in pytorch_xdit_flux_dev_single:
    - test_launch_container
    - test_verify_prerequisites
    - test_verify_model
    - test_verify_parallelism
    - test_run_benchmark
    - test_parse_thresholds
    - test_print_results
    - test_teardown

Example run (FLUX.1-dev; use the flux2 JSON for FLUX.2-dev):

.. code:: bash

  cvs run pytorch_xdit_flux_dev_single \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/inference/xdit/mi3xx_pytorch_xdit_flux1_dev_single.json \
    --html ~/cvs_results/pytorch_xdit_flux1_single.html --self-contained-html \
    --log-file /tmp/pytorch_xdit_flux1_single.log -vvv

``pytorch_xdit_flux_dev_distributed`` stages
--------------------------------------------

.. code:: text

  Available tests in pytorch_xdit_flux_dev_distributed:
    - test_launch_container
    - test_verify_prerequisites
    - test_verify_model
    - test_verify_parallelism
    - test_run_benchmark
    - test_parse_thresholds
    - test_print_results
    - test_teardown

``test_verify_parallelism`` checks that
``ulysses × ring × pipefusion × tp × dp == nnodes × torchrun_nproc``.

Example run:

.. code:: bash

  cvs run pytorch_xdit_flux_dev_distributed \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/inference/xdit/mi3xx_pytorch_xdit_flux1_dev_distributed.json \
    --html ~/cvs_results/pytorch_xdit_flux1_distributed.html --self-contained-html \
    --log-file /tmp/pytorch_xdit_flux1_distributed.log -vvv

``pytorch_xdit_wan22_14b_single`` stages
----------------------------------------

.. code:: text

  Available tests in pytorch_xdit_wan22_14b_single:
    - test_launch_container
    - test_verify_prerequisites
    - test_verify_model
    - test_verify_parallelism
    - test_run_benchmark
    - test_parse_thresholds
    - test_print_results
    - test_teardown

Example run:

.. code:: bash

  cvs run pytorch_xdit_wan22_14b_single \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/inference/xdit/mi3xx_pytorch_xdit_wan22_14b_single.json \
    --html ~/cvs_results/pytorch_xdit_wan22.html --self-contained-html \
    --log-file /tmp/pytorch_xdit_wan22.log -vvv

``pytorch_xdit_wan22_14b_diffusers_single`` stages
--------------------------------------------------

.. code:: text

  Available tests in pytorch_xdit_wan22_14b_diffusers_single:
    - test_launch_container
    - test_verify_prerequisites
    - test_verify_model
    - test_verify_parallelism
    - test_run_benchmark
    - test_parse_thresholds
    - test_print_results
    - test_teardown

Example run:

.. code:: bash

  cvs run pytorch_xdit_wan22_14b_diffusers_single \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/inference/xdit/mi3xx_pytorch_xdit_wan22_14b_diffusers_single.json \
    --html ~/cvs_results/pytorch_xdit_wan22_diffusers_single.html --self-contained-html \
    --log-file /tmp/pytorch_xdit_wan22_diffusers_single.log -vvv

``pytorch_xdit_wan22_14b_diffusers_distributed`` stages
-------------------------------------------------------

.. code:: text

  Available tests in pytorch_xdit_wan22_14b_diffusers_distributed:
    - test_launch_container
    - test_verify_prerequisites
    - test_verify_model
    - test_verify_parallelism
    - test_run_benchmark
    - test_parse_thresholds
    - test_print_results
    - test_teardown

``test_verify_parallelism`` checks that ``ulysses_size × ring_size == nnodes × torchrun_nproc``.

Example run:

.. code:: bash

  cvs run pytorch_xdit_wan22_14b_diffusers_distributed \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/inference/xdit/mi3xx_pytorch_xdit_wan22_14b_diffusers_distributed.json \
    --html ~/cvs_results/pytorch_xdit_wan22_diffusers_distributed.html --self-contained-html \
    --log-file /tmp/pytorch_xdit_wan22_diffusers_distributed.log -vvv

Direct pytest invocation
------------------------

Each module can also be run with pytest:

.. code:: bash

  pytest cvs/tests/inference/xdit/pytorch_xdit_flux_dev_single.py \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/inference/xdit/mi3xx_pytorch_xdit_flux1_dev_single.json \
    --html ~/cvs_results/pytorch_xdit_flux1_single.html

Read the results
================

With ``--html``, CVS writes a pytest HTML report. Benchmark pass/fail uses the docker
exit code plus parsed artifacts and GPU-specific thresholds (``mi300x``, ``mi350``,
``mi355``, or ``auto``).

Key stages to watch:

- **Container lifecycle** — ``test_launch_container`` starts the scoped containers;
  ``test_teardown`` removes only the suite-owned per-run containers.
- **Model preflight** — ``test_verify_model`` downloads a Hugging Face repo id into
  ``HF_HOME`` when needed, or checks an absolute local model path on every
  participating node.
- **Parallelism** — ``test_verify_parallelism`` validates the configured topology.
- **Benchmark** — ``test_run_benchmark`` deletes previous
  ``${paths.log_dir}/flux_*_outputs`` or ``wan_22_*_outputs`` trees, then runs
  torchrun inside the containers.
- **Parse / print** — ``test_parse_thresholds`` compares average latency to the
  sibling threshold JSON. ``test_print_results`` prints Host, Model, topology,
  Ulysses, and Ring. The xDiT Run Deck run card shows **Server nodes**, **nnodes**,
  **Benchmark node** (all execution hosts on single; rank-0 on distributed),
  Ulysses, and Ring.

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

Single-node output dirs use the cluster SSH target, for example
``${paths.log_dir}/flux_<target>_outputs`` or ``wan_22_<target>_outputs``.
Distributed runs write to the rank-0 target directory.
