.. meta::
  :description: Configure the Aorta benchmark configuration file variables
  :keywords: Aorta, ROCm, RCCL, benchmark, CVS

********************************************
Aorta benchmark test configuration file
********************************************

The Aorta benchmark runs distributed training with RCCL in a container, collects PyTorch profiler traces, and validates iteration time and compute/communication overlap. Metrics are derived from host-side trace parsing (raw traces or TraceLens reports when available).

``aorta_benchmark.yaml``
========================

Here's a code snippet of the ``aorta_benchmark.yaml`` file for reference:

.. note::

  Set ``aorta_path`` to the absolute path of your Aorta repository on the host. The runner bind-mounts this path into the container. Do not leave the default ``<changeme>``.

.. dropdown:: ``aorta_benchmark.yaml``

  .. code:: yaml

    # Path to Aorta repository on host (bind-mounted into container)
    aorta_path: <changeme>
    container_mount_path: /mnt
    base_config: config/distributed.yaml

    docker:
      image: jeffdaily/pytorch:torchrec-dlrm-complete
      container_name: aorta-benchmark
      shm_size: 17G
      network_mode: host
      privileged: true

    rccl:
      clone_url: https://github.com/rocm/rccl.git
      branch: develop
      build_path: /mnt/rccl

    environment:
      NCCL_MAX_NCHANNELS: 112
      NCCL_MAX_P2P_NCHANNELS: 112
      NCCL_DEBUG: VERSION
      TORCH_NCCL_HIGH_PRIORITY: 1
      OMP_NUM_THREADS: 1
      RCCL_MSCCL_ENABLE: 0

    training_overrides:
      training.max_steps: 100
      profiling.active: 10

    build_script: scripts/build_rccl.sh
    experiment_script: scripts/launch_rocm.sh
    gpus_per_node: 8
    timeout_seconds: 10800
    skip_rccl_build: false

    analysis:
      enable_tracelens: false
      enable_gemm_analysis: false
      tracelens_script: scripts/tracelens_single_config/run_tracelens_single_config.sh
      skip_if_exists: false

    expected_results:
      max_avg_iteration_ms: 7000
      min_compute_ratio: 0.8
      min_overlap_ratio: 0.0
      max_time_variance_ratio: 0.2

Parameters
==========

Here's an exhaustive list of the available parameters in the Aorta benchmark configuration file.

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameters
     - Default values
     - Description
   * - ``aorta_path``
     - (required)
     - Absolute path to Aorta repository on host; bind-mounted into container
   * - ``container_mount_path``
     - ``/mnt``
     - Mount point inside container for ``aorta_path``
   * - ``base_config``
     - ``config/distributed.yaml``
     - Aorta config file path relative to ``aorta_path``
   * - ``docker.image``
     - ``jeffdaily/pytorch:torchrec-dlrm-complete``
     - Docker image for the benchmark container
   * - ``docker.container_name``
     - ``aorta-benchmark``
     - Name of the container
   * - ``docker.shm_size``
     - ``17G``
     - Shared memory size for the container
   * - ``docker.network_mode``
     - ``host``
     - Docker network mode
   * - ``docker.privileged``
     - true
     - Run container in privileged mode
   * - ``rccl.clone_url``
     - ``https://github.com/rocm/rccl.git``
     - RCCL git repository URL (used if building RCCL inside container)
   * - ``rccl.branch``
     - ``develop``
     - RCCL branch to build
   * - ``rccl.build_path``
     - ``/mnt/rccl``
     - Path inside container for RCCL build
   * - ``environment.NCCL_MAX_NCHANNELS``
     - 112
     - Maximum NCCL channels
   * - ``environment.NCCL_MAX_P2P_NCHANNELS``
     - 112
     - Maximum NCCL P2P channels
   * - ``environment.NCCL_DEBUG``
     - ``VERSION``
     - NCCL debug level
   * - ``environment.TORCH_NCCL_HIGH_PRIORITY``
     - 1
     - Enable high-priority NCCL streams
   * - ``environment.OMP_NUM_THREADS``
     - 1
     - OpenMP thread count
   * - ``environment.RCCL_MSCCL_ENABLE``
     - 0
     - Enable MSCCL
   * - ``training_overrides``
     - (key-value overrides)
     - Overrides passed to Aorta via ``--override`` (e.g. ``training.max_steps``, ``profiling.active``)
   * - ``build_script``
     - ``scripts/build_rccl.sh``
     - RCCL build script path relative to container mount
   * - ``experiment_script``
     - ``scripts/launch_rocm.sh``
     - Experiment/launch script path relative to container mount
   * - ``gpus_per_node``
     - 8
     - Number of GPUs per node
   * - ``timeout_seconds``
     - 10800
     - Benchmark timeout in seconds
   * - ``skip_rccl_build``
     - false
     - If true, skip RCCL build (use existing build in ``aorta_path``)
   * - ``analysis.enable_tracelens``
     - false
     - Run TraceLens analysis after benchmark (optional, host parsing works without it)
   * - ``analysis.enable_gemm_analysis``
     - false
     - Run GEMM analysis (for sweep experiments)
   * - ``analysis.tracelens_script``
     - ``scripts/tracelens_single_config/run_tracelens_single_config.sh``
     - TraceLens script path relative to ``aorta_path``
   * - ``analysis.gemm_script``
     - ``scripts/gemm_analysis/run_tracelens_analysis.sh``
     - GEMM analysis script path relative to ``aorta_path``
   * - ``analysis.skip_if_exists``
     - false
     - Skip analysis if ``tracelens_analysis`` directory already exists
   * - ``expected_results.max_avg_iteration_ms``
     - e.g. 7000
     - Maximum acceptable average iteration time (ms); validation fails if exceeded
   * - ``expected_results.min_compute_ratio``
     - e.g. 0.8
     - Minimum acceptable compute ratio (compute time / total iteration time)
   * - ``expected_results.min_overlap_ratio``
     - e.g. 0.0
     - Minimum acceptable compute-communication overlap ratio
   * - ``expected_results.max_time_variance_ratio``
     - e.g. 0.2
     - Maximum acceptable iteration time variance (e.g. std/mean); used for rank balance

How to run
==========

From the CVS repo root (directory containing ``cvs`` and ``input``):

.. code-block:: bash

  pytest cvs/tests/benchmark/test_aorta.py \
      --cluster_file input/cluster_file/cluster.json \
      --config_file input/config_file/aorta/aorta_benchmark.yaml \
      -v --log-cli-level=INFO

Provide a valid ``cluster_file`` and ensure ``aorta_path`` in the config points to an existing Aorta checkout. The runner will build RCCL (unless ``skip_rccl_build`` is true), run the experiment script, collect ``torch_traces`` (PyTorch profiler output), and optionally run TraceLens in the container. Results are parsed on the host from raw traces or from TraceLens reports when present.

Expected results and artifacts
==============================

Validation uses the ``expected_results`` thresholds: iteration time must be within ``max_avg_iteration_ms``, compute and overlap ratios must meet the minimums, and time variance across ranks must not exceed ``max_time_variance_ratio``. Exact pass values depend on cluster size and hardware.

Artifacts produced under the configured output directory include training logs, ``torch_profiler`` (or equivalent) trace data, and optionally ``tracelens_analysis`` when TraceLens is enabled. The test report (e.g. ``aorta_benchmark_report.json``) summarizes metrics and pass/fail per threshold.
