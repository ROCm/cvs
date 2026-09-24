.. meta::
  :description: Configure Aorta benchmark JSON variants and performance thresholds
  :keywords: Aorta, ROCm, RCCL, benchmark, CVS

Aorta benchmark configuration
=============================

Aorta uses JSON variants in ``cvs/input/config_file/benchmark/aorta/``:

* ``mi3xx_aorta_profile_overlap_2gpu_single.json``
* ``mi3xx_aorta_profile_overlap_2gpu_distributed.json``
* ``mi3xx_aorta_profile_overlap_2gpu_threshold.json``

The variants preserve the short profiling workload from the previous sample: the base file
is ``config/profile_overlap_2gpu.yaml``, with ``training.max_steps=15`` and
``profiling.active=6``. Set the GPU count explicitly and adapt the base YAML for the intended
node count. All ``<changeme>`` values must be replaced before launch.

Paths and containers
--------------------

The shared loader resolves ``{user-id}`` from the cluster username, references within
``paths`` such as ``{shared_fs}``, and cross-block references such as ``{paths.shared_fs}``.
Unresolved placeholders fail configuration loading. The previous ``{home}``, ``{user}``,
``{home-mount-dir}`` and ``{node-dir-name}`` Aorta placeholders should be replaced by explicit
paths or the shared loader's supported references.

.. list-table:: Main fields
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Meaning
   * - ``aorta_path``
     - Absolute repository directory on each cluster host. It need not exist on the CVS machine.
   * - ``container_mount_path``
     - Repository path inside the container; default ``/mnt``. Declare the corresponding writable host/container volume explicitly.
   * - ``aorta_auto_clone``, ``aorta_clone_url``
     - Clone a missing repository on every host before container launch when enabled. A URL and host ``git`` installation are required.
   * - ``container``
     - Shared container schema: ``name``, ``image``, ``lifetime``, ``runtime: {name, args}``, and string-valued ``env``.
   * - ``container.runtime.args``
     - Devices, volumes, network, IPC, privileges, capabilities, groups, security options and ulimits. The sample uses ``network: host`` and ``ipc: host``.
   * - ``container.lifetime``
     - ``per_run`` launches/removes containers; ``persistent`` keeps them; ``no_launch`` attaches to containers already running.
   * - ``paths`` and ``model``
     - Shared schema fields. Aorta does not use ``models_dir`` or ``hf_token_file`` to download models; ``model.remote`` must be zero.
   * - ``output_dir``
     - Local CVS output root; default ``aorta_results``. Each invocation gets a unique run subdirectory.

Use storage writable by the container user. Root-squashed NFS may prevent root containers
from writing output. The suite restores repository ownership to each host's login user
before delegating container teardown to the orchestrator.
At launch, CVS also resolves each host's numeric ``render`` group ID and adds the
discovered IDs to the containers so GPU device access does not depend on the image's
group-name mapping.

Workload and launch
-------------------

.. list-table:: Aorta execution fields
   :header-rows: 1
   :widths: 35 65

   * - Field
     - Meaning
   * - ``base_config``
     - Aorta YAML path relative to the repository.
   * - ``experiment_script``
     - Script-mode launcher. Receives the base config followed by one ``--override`` group.
   * - ``build_script``, ``skip_rccl_build``
     - Optional RCCL build script and skip gate. The sample skips the build and uses container-native RCCL.
   * - ``rccl``
     - ``clone_url``, ``branch``, ``build_path``. Exposed to scripts through ``RCCL_CLONE_URL``, ``RCCL_BRANCH``, and ``rccl_path``; the build script must consume its checkout/build settings.
   * - ``training_overrides``
     - Key/value settings forwarded together after one ``--override`` argument.
   * - ``gpus_per_node``
     - Required positive GPU count used per node. Match it to the workload and configured hardware.
   * - ``timeout_seconds``
     - Positive bound for each build, benchmark or analysis phase; default 3600 seconds.
   * - ``multi_node.master_launch_mode``
     - ``auto`` selects script mode for one node and torchrun for multiple nodes. Explicit ``script`` requires one node; explicit ``torchrun`` also works with one node.
   * - ``multi_node.nproc_per_node``
     - Torchrun processes per node. Omit or use zero to select ``gpus_per_node``.
   * - ``multi_node.master_addr``
     - Optional rendezvous override. Default: first node's ``vpc_ip``, then its node identifier.
   * - ``multi_node.master_port``
     - Rendezvous port in 1024..65535. Omit or use zero to select an available port on the first node.
   * - ``multi_node.train_script``
     - Training entry point relative to the repository; default ``train.py``.
   * - ``multi_node.extra_torchrun_args``, ``extra_train_args``
     - Additional shell argument fragments, preserving the previous launcher convention.
   * - ``multi_node.extra_env``
     - Environment overrides, including cluster-specific ``NCCL_SOCKET_IFNAME`` and ``NCCL_IB_HCA``.
   * - ``multi_node.collect_traces``
     - Default true: gather all node traces in torchrun mode. False: collect only the head's newest tree; use true for complete distributed metrics.

Environment values move from the old ``environment`` block to ``container.env``.
``TENSILE_STREAMK_MAX_CUS`` defaults to 256 minus ``NCCL_MAX_NCHANNELS``. The launch environment
prepends the RCCL build and ROCm library paths unless ``LD_LIBRARY_PATH`` is explicitly set.
The benchmark requires passwordless ``sudo -n journalctl -k`` access for bounded kernel-error
scanning on every node.

Analysis and thresholds
-----------------------

``analysis.enable_tracelens`` enables the configured ``tracelens_script``;
``analysis.enable_gemm_analysis`` enables ``gemm_script``. Both scripts run in the head
container against its original output tree. ``analysis.skip_if_exists`` permits reuse of an
existing ``tracelens_analysis`` directory. Analysis failures are warnings; raw traces remain
available. Multi-node metrics always come from raw traces from the collected nodes.

``threshold_json`` points to a sibling JSON file with an ``expected_results`` block:

.. code-block:: json

   {
     "expected_results": {
       "max_avg_iteration_ms": 12000,
       "min_compute_ratio": 0.01,
       "min_overlap_ratio": 0.0,
       "max_time_variance_ratio": 0.5
     }
   }

These starting thresholds come from the previous gfx942 sample. Calibrate them for the
selected hardware and workload. Ratios are in 0..1; time and variance limits are non-negative.
With ``enforce_thresholds: true`` at least one non-null threshold is required. False records
metrics without threshold assertions. Unknown threshold names are rejected.

Migration and backend support
-----------------------------

Replace the former YAML runner configuration with a JSON variant plus threshold file. Move
Docker settings under ``container``, environment values under ``container.env``, and
``expected_results`` into the threshold file. Drop ``shm_size`` and use ``ipc: host``.
Run ``cvs run aorta_single`` or ``cvs run aorta_distributed`` with the matching node count.

Artifacts now live under the local ``output_dir/<run-id>/``. The distributed parser layout
remains ``combined_traces/node_<rank>/<original-output>/torch_profiler/``. The suite excludes
stale profiler files using each node's benchmark-start timestamp and preserves surviving
artifacts after execution failures.

Runtime and transport support comes from the shared orchestrator. The Enroot runtime in
this checkout is not implemented; this migration does not add that backend.
