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

    multi_node:
      master_launch_mode: auto
      train_script: train.py
      extra_torchrun_args: []
      extra_train_args: []
      extra_env: {}
      collect_traces: true

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
   * - ``multi_node.master_launch_mode``
     - ``auto``
     - ``auto`` picks ``script`` for single-node clusters and ``torchrun`` for multi-node clusters. Set to ``script`` to force the single-node ``experiment_script`` path (errors out on >1 node), or ``torchrun`` to always build a disaggregated ``torchrun`` command.
   * - ``multi_node.nproc_per_node``
     - ``null`` (defaults to ``gpus_per_node``)
     - Processes/GPUs per node passed as ``torchrun --nproc_per_node``.
   * - ``multi_node.master_port``
     - ``null`` (free ephemeral port)
     - Port for the ``torchrun`` rendezvous (``--master_port``). Pin this when you need a deterministic port (e.g., firewalled environments).
   * - ``multi_node.master_addr``
     - ``null`` (head node from cluster.json)
     - Override the rendezvous address (``--master_addr``).
   * - ``multi_node.train_script``
     - ``train.py``
     - Aorta training entry script relative to ``aorta_path``. Used in ``torchrun`` mode.
   * - ``multi_node.extra_torchrun_args``
     - ``[]``
     - Additional ``torchrun`` flags appended before the training script.
   * - ``multi_node.extra_train_args``
     - ``[]``
     - Additional ``train.py`` flags appended after ``--config``.
   * - ``multi_node.extra_env``
     - ``{}``
     - Extra environment variables exported inside each container before ``torchrun``. Use for transport tuning (``NCCL_SOCKET_IFNAME``, ``NCCL_IB_HCA``, ``NCCL_IB_GID_INDEX``, ...).
   * - ``multi_node.collect_traces``
     - ``true``
     - When true, the runner pulls each node's ``torch_profiler/`` trees back to ``<aorta_path>/combined_traces/node_<rank>/`` on the head node so host parsers see one unified trace tree.
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

  cvs run test_aorta \
      --cluster_file input/cluster_file/cluster.json \
      --config_file input/config_file/aorta/aorta_benchmark.yaml \
      -v --log-cli-level=INFO

Provide a valid ``cluster_file`` and ensure ``aorta_path`` in the config points to an existing Aorta checkout. The runner will build RCCL (unless ``skip_rccl_build`` is true), run the experiment script, collect ``torch_traces`` (PyTorch profiler output), and optionally run TraceLens in the container. Results are parsed on the host from raw traces or from TraceLens reports when present.

Multi-node disaggregated launch
===============================

By default, when the cluster file contains more than one node, ``test_aorta`` runs a disaggregated launch: a single Aorta container is started on every node, then the runner kicks off ``torchrun`` in parallel on each container with ``--nnodes``, ``--node_rank``, ``--master_addr``, and ``--master_port`` set so the ranks rendezvous on the head node. This mirrors Aorta's own ``scripts/multi_node/local_launch.sh`` pattern and brings the benchmark in line with the other multi-node CVS suites (sglang, pytorch-xdit), which only require **one** ``cluster.json`` for a multi-node run.

The multi-node behavior is controlled by the ``multi_node`` block in ``aorta_benchmark.yaml``:

.. code:: yaml

  multi_node:
    master_launch_mode: auto      # auto | script | torchrun
    nproc_per_node: 8             # defaults to gpus_per_node
    master_port: 29500            # default: free ephemeral port
    master_addr: 10.0.0.1         # default: head node from cluster.json
    train_script: train.py
    extra_torchrun_args: []
    extra_train_args: []
    extra_env:
      NCCL_SOCKET_IFNAME: bond0
      NCCL_IB_HCA: rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7
      NCCL_IB_GID_INDEX: "3"
    collect_traces: true

Single-node clusters keep using the configured ``experiment_script`` (``master_launch_mode: auto`` resolves to ``script``). Force the disaggregated path with ``master_launch_mode: torchrun`` if you want it for a single-node cluster too.

When ``collect_traces`` is true, every node's ``torch_profiler/`` directories are rsynced back to ``<aorta_path>/combined_traces/node_<rank>/`` on the head node and exposed as the ``torch_traces`` artifact, so the existing host parsers and threshold checks see one unified tree without further configuration.

Expected results and artifacts
==============================

Validation uses the ``expected_results`` thresholds: iteration time must be within ``max_avg_iteration_ms``, compute and overlap ratios must meet the minimums, and time variance across ranks must not exceed ``max_time_variance_ratio``. Exact pass values depend on cluster size and hardware.

Artifacts produced under the configured output directory include training logs, ``torch_profiler`` (or equivalent) trace data, and optionally ``tracelens_analysis`` when TraceLens is enabled. The test report (e.g. ``aorta_benchmark_report.json``) summarizes metrics and pass/fail per threshold.
