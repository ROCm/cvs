.. meta::
  :description: Configure the Aorta benchmark configuration file variables
  :keywords: Aorta, ROCm, RCCL, benchmark, CVS

********************************************
Aorta benchmark test configuration file
********************************************

The Aorta benchmark runs distributed training with RCCL in a container, collects PyTorch profiler traces, and validates iteration time and compute/communication overlap. Metrics are derived from host-side trace parsing (raw traces or TraceLens reports when available).

``aorta_benchmark.yaml``
========================

The shipped sample is ``cvs/input/config_file/aorta/aorta_benchmark.yaml`` (path relative to the CVS package directory; see *How to run* below).

Path placeholders
-----------------

When you run ``test_aorta``, the suite loads the cluster file, resolves cluster placeholders (e.g. ``{user-id}`` in ``username``), then resolves **Aorta YAML** placeholders with the same helper used by other CVS test configs: ``{user-id}``, ``{user}``, ``{home}``, ``{home-mount-dir}``, ``{node-dir-name}``. Replacement values come from the validated cluster model (username and optional ``home_mount_dir_name`` / ``node_dir_name``). Manual ``<changeme>`` markers are rejected.

You may instead use fully absolute paths with no placeholders. Other entry points that validate YAML directly (without ``test_aorta``) do not perform this step unless they call the resolver explicitly.

.. note::

  ``aorta_path`` must exist on the host unless ``aorta_auto_clone`` is true and ``aorta_clone_url`` is set; the runner can then clone into ``aorta_path`` during setup.

.. dropdown:: Example ``aorta_benchmark.yaml`` (aligned with the shipped sample)

  .. code:: yaml

    aorta_path: /home/{user-id}/aorta
    aorta_auto_clone: false
    aorta_clone_url: null

    container_mount_path: /mnt
    base_config: config/profile_overlap_2gpu.yaml

    docker:
      image: jeffdaily/pytorch:torchrec-dlrm-complete
      container_name: aorta-benchmark
      shm_size: 17G
      network_mode: host
      privileged: true

    rccl:
      clone_url: https://github.com/ROCmSoftwarePlatform/rccl.git
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
      training.max_steps: 15
      profiling.active: 6

    build_script: scripts/launch_rocm.sh
    experiment_script: scripts/launch_rocm.sh
    gpus_per_node: 8
    timeout_seconds: 3600
    skip_rccl_build: true

    analysis:
      enable_tracelens: false
      enable_gemm_analysis: false
      tracelens_script: scripts/tracelens_single_config/run_tracelens_single_config.sh
      gemm_script: scripts/gemm_analysis/run_tracelens_analysis.sh
      skip_if_exists: false

    multi_node:
      master_launch_mode: auto
      train_script: train.py
      extra_torchrun_args: []
      extra_train_args: []
      extra_env: {}
      collect_traces: true

    expected_results:
      max_avg_iteration_ms: 12000
      min_compute_ratio: 0.01
      min_overlap_ratio: 0.0
      max_time_variance_ratio: 0.5

Parameters
==========

The **middle column** is the **schema default**: the value Pydantic applies when you **omit** that key from your YAML. It is not a promise about the checked-in sample file.

The **dropdown above** is the full **shipped** ``aorta_benchmark.yaml``. Where the sample lists a key, that value wins for that file; compare the sample to the middle column to see explicit overrides.

.. note::

  The shipped sample commonly overrides schema defaults for ``base_config``, ``build_script``, ``experiment_script``, ``timeout_seconds``, ``skip_rccl_build``, ``training_overrides``, ``analysis.enable_tracelens``, and ``expected_results``.

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameters
     - Schema default if omitted
     - Description
   * - ``aorta_path``
     - (required)
     - Absolute path to Aorta on the host; bind-mounted into the container. Placeholders resolved in ``test_aorta`` as described above.
   * - ``aorta_auto_clone``
     - ``false``
     - If true and ``aorta_path`` is missing, clone from ``aorta_clone_url`` during runner setup.
   * - ``aorta_clone_url``
     - ``null``
     - Git URL for Aorta when using auto-clone.
   * - ``container_mount_path``
     - ``/mnt``
     - Mount point inside the container for ``aorta_path``.
   * - ``base_config``
     - ``config/distributed.yaml``
     - Aorta config file path relative to ``aorta_path``.
   * - ``docker.image``
     - ``jeffdaily/pytorch:torchrec-dlrm-complete``
     - Docker image for the benchmark container.
   * - ``docker.container_name``
     - ``aorta-benchmark``
     - Container name.
   * - ``docker.shm_size``
     - ``17G``
     - Shared memory size for the container.
   * - ``docker.network_mode``
     - ``host``
     - Docker network mode.
   * - ``docker.privileged``
     - true
     - Run the container in privileged mode.
   * - ``rccl.clone_url``
     - ``https://github.com/ROCmSoftwarePlatform/rccl.git``
     - RCCL Git URL (used when building RCCL in the container).
   * - ``rccl.branch``
     - ``develop``
     - RCCL branch to build.
   * - ``rccl.build_path``
     - ``/mnt/rccl``
     - Path inside the container for the RCCL build.
   * - ``environment.NCCL_MAX_NCHANNELS``
     - 112
     - Maximum NCCL channels.
   * - ``environment.NCCL_MAX_P2P_NCHANNELS``
     - 112
     - Maximum NCCL P2P channels.
   * - ``environment.NCCL_DEBUG``
     - ``VERSION``
     - NCCL debug level.
   * - ``environment.TORCH_NCCL_HIGH_PRIORITY``
     - 1
     - High-priority NCCL streams.
   * - ``environment.OMP_NUM_THREADS``
     - 1
     - OpenMP thread count.
   * - ``environment.RCCL_MSCCL_ENABLE``
     - 0
     - MSCCL enable flag.
   * - ``training_overrides``
     - ``{}``
     - Overrides passed to Aorta via ``--override`` (e.g. ``training.max_steps``, ``profiling.active``).
   * - ``build_script``
     - ``scripts/build_rccl.sh``
     - RCCL build script path relative to the container mount (skipped when ``skip_rccl_build`` is true).
   * - ``experiment_script``
     - ``scripts/rccl_exp.sh``
     - Experiment/launch script path relative to the container mount.
   * - ``gpus_per_node``
     - 8
     - GPUs per node.
   * - ``timeout_seconds``
     - 10800
     - Benchmark timeout in seconds.
   * - ``skip_rccl_build``
     - ``false``
     - If true, skip building RCCL (use an existing build / container setup).
   * - ``analysis.enable_tracelens``
     - ``true``
     - Run TraceLens in the container when available (shipped sample sets ``false``).
   * - ``analysis.enable_gemm_analysis``
     - ``false``
     - Run GEMM analysis (sweep workflows).
   * - ``analysis.tracelens_script``
     - ``scripts/tracelens_single_config/run_tracelens_single_config.sh``
     - TraceLens script relative to ``aorta_path``.
   * - ``analysis.gemm_script``
     - ``scripts/gemm_analysis/run_tracelens_analysis.sh``
     - GEMM analysis script relative to ``aorta_path``.
   * - ``analysis.skip_if_exists``
     - ``false``
     - Skip analysis if ``tracelens_analysis`` already exists.
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
     - optional
     - Maximum acceptable average iteration time (ms).
   * - ``expected_results.min_compute_ratio``
     - optional
     - Minimum compute ratio (compute time / iteration time).
   * - ``expected_results.min_overlap_ratio``
     - optional
     - Minimum compute–communication overlap ratio.
   * - ``expected_results.max_time_variance_ratio``
     - optional
     - Maximum iteration time variance across ranks (e.g. std/mean).

How to run
==========

Use the **CVS package directory** as the working directory: the directory that contains the ``input`` tree (in a typical clone, the inner ``cvs`` directory next to ``tests`` and ``lib``). Example:

.. code-block:: bash

  cd /path/to/your/cvs-checkout/cvs
  cvs run test_aorta \
      --cluster_file input/cluster_file/cluster.json \
      --config_file input/config_file/aorta/aorta_benchmark.yaml \
      -v --log-cli-level=INFO

Provide a valid ``cluster_file``. Ensure ``aorta_path`` exists after placeholder resolution, or enable auto-clone with a valid URL. With ``skip_rccl_build: false``, the runner builds RCCL from ``rccl.clone_url`` unless skipped; with ``skip_rccl_build: true``, the experiment script runs without that build step. The runner collects ``torch_traces`` (PyTorch profiler output) and optionally runs TraceLens inside the container. Parsing and threshold checks run on the host.

Alternate mirrors for ``rccl.clone_url`` may work if they track the same upstream; the canonical default string in schema, runner, and sample is ``https://github.com/ROCmSoftwarePlatform/rccl.git``.

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

Validation uses ``expected_results`` when fields are set. Artifact layout depends on the Aorta run; the test report (e.g. ``aorta_benchmark_report.json`` under the runner output directory) summarizes metrics. Prefer scratch or local disk for ``aorta_path`` when NFS ``root_squash`` prevents the container from writing ``artifacts/`` under your tree.
