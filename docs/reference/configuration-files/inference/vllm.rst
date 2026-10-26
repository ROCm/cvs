.. meta::
  :description: Reference for the vLLM inference benchmark configuration in CVS, covering container setup, sweep cells, threshold files, accuracy tests, and multi-node execution.
  :keywords: CVS, vLLM, inference, ROCm, LLM, benchmark, GPU, AMD, threshold, multinode, accuracy, JSON, configuration

******************************************************************************
vLLM inference benchmark configuration file for Cluster Validation Suite (CVS)
******************************************************************************

The vLLM suites benchmark LLM serving throughput, latency, and accuracy on AMD Instinct GPUs. ``vllm_single`` runs on the first cluster host and ignores additional hosts. ``vllm_distributed`` uses every host in the cluster file. One host does not raise: ``resolve_vllm_topology`` returns an effective single-node topology, and ``vllm serve`` omits the multi-host flags. Packaged distributed recipes are laid out for two hosts. Their threshold values are zeros, not measured pass/fail limits.

For a mapping-style ``node_dict``, "first cluster host" means the first JSON key
in insertion order. ``vllm_single`` scopes execution to that host and rewrites
``head_node_dict.mgmt_ip`` to match, even if the original head names another
node. Put the intended single-node target first in ``node_dict``.

Run it with:

.. code:: bash

  cvs run vllm_single --cluster_file <cluster.json> --config_file <config.json>

For a step-by-step walkthrough of a first run, see :doc:`/how-to/test-suites/inference/vllm`. This page is the schema and metric reference.

Lifecycle
=========

Each stage of the run is an independent test, so every stage becomes its own timed, pass/fail row in the HTML report. The suite pins this order explicitly rather than relying on definition order:

.. list-table::
   :widths: 1 3 6
   :header-rows: 1

   * - Order
     - Test
     - Purpose
   * - 0
     - ``test_launch_container``
     - Pull/load the image and start the container on every node.
   * - 1
     - ``test_setup_sshd``
     - Always skipped for vLLM (see note below).
   * - 2
     - ``test_discover_topology``
     - Resolve IB HCA devices; no-op for effective single-node execution.
   * - 3
     - ``test_model_fetch``
     - Stage model weights.
   * - 4
     - ``test_openai_compatible_smoke``
     - Short-lived server; verifies the OpenAI-compatible API answers.
   * - 5
     - ``test_vllm_inference``
     - Run one benchmark cell (parametrized per sweep run) and verify its metrics; configured threshold gates are listed as subtests.
   * - 6
     - ``test_accuracy_eval``
     - lm-eval accuracy tasks, if any are configured.
   * - 7
     - ``test_print_results_table``
     - Console + report summary table.
   * - 8
     - ``test_teardown``
     - Stop the server and tear down the container.

.. note::

  ``test_setup_sshd`` always skips in this suite. The skip reason is ``vLLM uses host-network NCCL/gloo rather than in-container sshd``. A skipped row here is expected.

.. note::

  vLLM suite execution is serial and single-pass. Do not use xdist workers or
  ``pytest-repeat`` counts above one: later stages consume results collected
  earlier in the same pytest process.

If a stage fails, later stages are skipped rather than cascading into confusing downstream errors. The container is still torn down by a leak-guard even when a mid-sweep test fails.

Configuration file structure
============================

A vLLM configuration file has these top-level keys:

.. list-table::
   :widths: 3 2 5
   :header-rows: 1

   * - Key
     - Required
     - Description
   * - ``enforce_thresholds``
     - no (default ``true``)
     - When ``false``, metrics record and no threshold spec is asserted. Packaged Llama files set ``false``.
   * - ``threshold_json``
     - yes
     - Explicit path to the threshold file. See :ref:`vllm-threshold-discovery`.
   * - ``container``
     - yes
     - Container/Docker settings. See :ref:`vllm-container`.
   * - ``paths``
     - yes
     - Filesystem locations. See :ref:`vllm-paths`.
   * - ``server_params``
     - yes
     - Harness-owned server fields plus snake-case ``vllm serve`` options.
   * - ``benchmark_params``
     - no
     - Benchmark defaults plus snake-case ``vllm bench serve`` options.
   * - ``sweeps``
     - yes
     - Canonical run-cell keys mapped to benchmark overrides.
   * - ``runs``
     - yes
     - Nonempty ordered list of the sweep cells to execute.
   * - ``thresholds``
     - no
     - Per-cell pass/fail specs. See :ref:`vllm-thresholds`.
   * - ``accuracy``
     - no
     - lm-eval task selection. See :ref:`vllm-accuracy`.

.. important::

  Top-level keys, ``paths``, ``container``, ``container.runtime``, and
  ``container.runtime.args`` **forbid unknown keys**. ``server_params``,
  ``benchmark_params``, and individual sweep overrides accept extra snake-case
  vLLM option names. CVS converts those extras to kebab-case CLI flags.
  ``null`` omits a flag, ``true`` emits a bare flag, scalars emit one value,
  lists emit one flag followed by values, and mappings emit compact JSON.
  ``false`` is rejected on those extra options. The harness booleans
  ``benchmark_params.ignore_eos`` and ``trust_remote_code`` do accept ``false``.
  Use an option's negative form instead of ``false`` for any other flag.

Placeholder substitution
------------------------

Values are resolved in three passes, so later forms can reference earlier ones:

1. **Cluster placeholders** — ``{user-id}`` resolves to the current username.
2. **Self-reference within** ``paths`` — ``{shared_fs}`` expands to the already-resolved ``paths.shared_fs``.
3. **Cross-block** — ``{paths.models_dir}`` expands anywhere else in the file, such as in a volume mount.

.. code:: json

    {
      "paths": {
        "shared_fs": "/mnt/dtni/{user-id}",
        "models_dir": "{shared_fs}/models",
        "log_dir": "{shared_fs}/LOGS",
        "hf_token_file": "{shared_fs}/.cache/huggingface/token"
      },
      "container": {
        "runtime": {
          "args": {
            "volumes": ["{paths.models_dir}:/models"]
          }
        }
      }
    }

.. _vllm-backends:

Execution backends
==================

Four different things in this stack are called a "backend". They are unrelated, and confusing them is the most common configuration mistake.

.. list-table::
   :widths: 3 2 5
   :header-rows: 1

   * - Setting
     - Values
     - What it selects
   * - ``benchmark_params.backend``
     - ``"vllm"`` (default)
     - The **client** backend passed to ``vllm bench serve --backend``. Nothing to do with distribution.
   * - ``server_params.distributed_executor_backend``
     - ``"mp"`` (default), ``"ray"``
     - How vLLM distributes the model across nodes. This is the multinode setting.
   * - ``container.runtime.name``
     - ``"docker"``
     - The container runtime.
   * - Cluster file ``orchestrator``
     - ``"baremetal"``, ``"container"``
     - Whether CVS runs commands on the host or inside a container. See :doc:`/reference/cluster/cluster-file`.

.. warning::

  The vLLM schema accepts only ``container.runtime.name`` ``"docker"``. ``enroot`` and podman are rejected at config load, before ``test_launch_container``.

Distributed executor: mp and ray
--------------------------------

Multinode runs support **two** executor backends. ``mp`` is the default and requires no configuration key at all.

**mp (default).** Used when ``server_params.distributed_executor_backend`` is absent or ``"mp"``. On two or more hosts the suite injects the full distributed block into each rank's ``vllm serve`` command:

.. code:: bash

  vllm serve <model> --tensor-parallel-size <tp> --port <port> \
    --node-rank <rank> --master-addr <addr> --master-port <port> \
    --nnodes <n> --pipeline-parallel-size <pp> \
    --distributed-executor-backend mp

Every rank above 0 additionally gets ``--headless``. Two or more hosts on this path **require pipeline parallelism** (``pipeline_parallel_size`` greater than 1). One host does not: the serve command stays single-node, and ``--pipeline-parallel-size`` is added only when that size is greater than 1. ``distributed_executor_backend`` is not passed on one host.

**ray (opt-in).** Selected by setting the snake_case key ``server_params.distributed_executor_backend`` to the exact lowercase string ``"ray"``. Other values are configuration errors. On two or more hosts, Ray takes a different route:

1. Bootstrap the cluster head: ``ray start --head --port=<master_port>``
2. Bootstrap each worker: ``ray start --address=<cluster-head>:<dist_init_port>``
3. Launch ``vllm serve`` on the **head node only** — workers run no serve process
4. On teardown, broadcast ``ray stop`` after the process kill

Under ray, none of the mp distributed flags are emitted. ``--pipeline-parallel-size`` is added only when ``server_params.pipeline_parallel_size`` is greater than 1.

.. note::

  Ray does not *enable* multinode — it **relaxes** the pipeline-parallelism requirement. With ray, ``pipeline_parallel_size`` of 1 is legal and is the expected configuration for pure tensor-parallel multinode serving. With mp, pipeline parallelism is mandatory.

Because only the head node serves under ray, worker ranks produce no per-rank server log. That is expected.

Topology validation rules
-------------------------

These rules are enforced when the configuration file loads, before anything starts:

.. list-table::
   :widths: 4 6
   :header-rows: 1

   * - Condition
     - Rule
   * - two or more cluster hosts, backend is not ray
     - ``pipeline_parallel_size`` **must** be greater than 1
   * - two or more cluster hosts, backend is ray
     - ``pipeline_parallel_size`` of 1 is valid
   * - two or more cluster hosts, either backend
     - ``container.env.NCCL_SOCKET_IFNAME`` is **required**

One host never hits these rules. ``resolve_vllm_topology`` returns mode ``single`` and does not raise. The exact multi-host errors are:

.. code:: text

  vllm_distributed requires pipeline_parallel_size>1 unless distributed-executor-backend=ray
  vllm_distributed requires container.env.NCCL_SOCKET_IFNAME on multi-host clusters

The first message names the flag with hyphens. The JSON key you set is ``distributed_executor_backend``. A legacy top-level ``ib_netdev`` also satisfies the socket-interface check. If ``container.env`` sets any of ``NCCL_SOCKET_IFNAME``, ``GLOO_SOCKET_IFNAME``, or ``TP_SOCKET_IFNAME``, it must set all three, and none of them can be empty.

Multinode prerequisites
-----------------------

Beyond the validation rules, a multinode run needs:

- ``server_params.dist_init_port`` — default ``29501``; CVS derives the head address from the cluster.
- ``container.env.NCCL_IB_HCA`` — the comma-separated RDMA HCA names available on every node. The packaged MI3xx configurations set ``rdma0`` through ``rdma7``.
- ``container.env.NCCL_SOCKET_IFNAME`` — the Linux netdev associated with the selected RNICs.
- ``container.env.GLOO_SOCKET_IFNAME`` and ``TP_SOCKET_IFNAME`` — generally the frontend/control-plane interface.
- ``container.env.NCCL_IB_GID_INDEX`` — the index for the intended RoCE/IB fabric. Use ``show_gids`` inside the container and choose an entry available on every selected HCA and node. If that command is unavailable, inspect ``ibv_devinfo -v`` and ``/sys/class/infiniband/<hca>/ports/<port>/gid_attrs/``.

``test_discover_topology`` runs only for two or more hosts. It checks that every node has the same non-empty HCA list (``ibv_devinfo -l``, otherwise ``/sys/class/infiniband``) and that every requested ``NCCL_IB_HCA`` name is in that list. A missing name fails with ``ib_discovery preflight: requested HCA devices <missing> not found on <host>. Available: <devs>``. It does not check socket-interface names, the GID index, or link state. One host, including one-host ``vllm_distributed``, skips this stage. When ``NCCL_IB_HCA`` is set, discovery does not overwrite it. When it is unset, the discovered list is exported as ``NCCL_IB_HCA`` in the per-command env script.

.. _vllm-container:

Container and Docker configuration
==================================

The vLLM ``container`` block selects the image, the container name, the lifetime, ``container.env``, and a small set of ``docker run`` arguments. The container orchestrator adds further flags from its defaults and from the cluster file. Packaged MI3xx configurations leave Hugging Face hub access online. Their volume list includes both NIC families, labeled by ``_comment_bnxt_mount`` (Broadcom Thor2, ``libbnxt_re-rdmav34.so``) and ``_comment_ionic_mounts`` (AMD Pensando Ionic / AINIC: ``libionic.so``, ``libionic.so.1``, the versioned ``libionic.so.1.*`` soname, ``libionic-rdmav34.so``, and ``ionic.driver``). Replace ``<changeme-libionic-soname>`` with the filename from ``readlink -f /usr/lib/x86_64-linux-gnu/libionic.so.1``, and delete the family that is not installed. ``docker run`` fails when a listed host path is missing.

.. code:: json

    {
      "container": {
        "lifetime": "per_run",
        "name": "vllm_llama33-70b_fp8_single_mi325x",
        "image": "<changeme>",
        "runtime": {
          "name": "docker",
          "args": {
            "network": "host",
            "ipc": "host",
            "privileged": true,
            "volumes": [
              "/home/{user-id}:/home/{user-id}",
              "<changeme-models-mount>:/models:ro"
            ]
          }
        }
      }
    }

Container block keys
--------------------

``ContainerConfig`` accepts only these keys. Anything else, including ``image_tar``, is a validation error.

.. list-table::
   :widths: 3 2 5
   :header-rows: 1

   * - Key
     - Default
     - Description
   * - ``image``
     - none
     - Container image. **Required.** An empty string makes the orchestrator log ``Container image not specified in config`` and ``test_launch_container`` fails because ``setup_containers()`` returned false.
   * - ``name``
     - required
     - Container name. The orchestrator's ``<user>_<sanitized-image>`` fallback applies only when a container block reaches it without a name. The vLLM schema requires ``name``.
   * - ``lifetime``
     - ``"per_run"``
     - One of ``no_launch``, ``per_run``, ``persistent``.
   * - ``runtime.name``
     - ``"docker"``
     - The only value the vLLM schema accepts.
   * - ``runtime.args``
     - see below
     - Only ``network``, ``ipc``, ``privileged``, ``volumes``, and ``devices``.
   * - ``env``
     - ``{}``
     - Environment variables passed as ``docker run -e``. **On** ``container``, not under ``runtime.args``. ``HF_TOKEN`` and ``HUGGING_FACE_HUB_TOKEN`` are rejected; set ``paths.hf_token_file`` instead.

``image_tar``, ``runtime.args.registry``, ``runtime.args.shm_size``, and ``runtime.args.cap_add`` are not fields of this model. Putting them in the vLLM file fails config load. Set those on the cluster file's ``container`` block. The vLLM conftest deep-merges the variant onto that block and keeps cluster keys the variant does not set. The orchestrator still emits them: ``image_tar`` loads an image, ``registry`` logs in before a pull, ``shm_size`` becomes ``--shm-size``, and ``cap_add`` appends to the orchestrator's default capabilities. See :doc:`/reference/cluster/cluster-file` for the orchestrator container keys that page documents.

.. warning::

  Put ``env`` on ``container``. ``RuntimeArgs`` forbids unknown keys, so ``env`` under ``runtime.args`` is rejected before launch.

Runtime arguments
-----------------

``RuntimeArgs`` accepts only the keys below. Schema defaults are ``network`` ``host``, ``ipc`` ``host``, ``privileged`` true, and empty ``volumes`` and ``devices``. The orchestrator then appends its own mounts and devices; you cannot drop those defaults from the vLLM file.

.. list-table::
   :widths: 3 7
   :header-rows: 1

   * - Key
     - What the vLLM file may set
   * - ``volumes``
     - Bind mounts, ``host:container[:ro]``. The orchestrator also adds ``/home/$USER/.ssh:/host_ssh``.
   * - ``devices``
     - Extra ``--device`` paths, appended to the orchestrator defaults ``/dev/kfd``, ``/dev/dri``, and each ``/dev/infiniband/*`` node the host has. Packaged files list ``/dev/kfd`` and ``/dev/dri`` again; that appends a second copy, it does not replace the defaults.
   * - ``network``
     - ``--network``. Schema default ``host``.
   * - ``ipc``
     - ``--ipc``. Schema default ``host``.
   * - ``privileged``
     - ``--privileged`` when true. Schema default true.

The assembled command is:

.. code:: bash

  docker run -d --name <name> <args> <image> sleep infinity

The container is a long-lived sidecar; every workload command runs through ``docker exec`` inside it. InfiniBand devices are additionally passed through by per-host shell expansion at launch time, so each node mounts the devices it actually has.

.. note::

  ``--gpus`` is deliberately never emitted. GPU access on AMD hardware comes from the ``/dev/kfd`` and ``/dev/dri`` device mounts plus the ``video`` group, which the orchestrator adds. Capabilities, security options, groups, ulimits, shared-memory size, registry login, and image-tar load are orchestrator behavior. Set them in the cluster file, not in the vLLM schema. ``runtime.args.shm_size`` in the vLLM file is rejected; the same key on the cluster file is emitted as ``--shm-size``.

Container lifetime
------------------

The ``lifetime`` key controls when the container is started and stopped relative to the test lifecycle.

.. list-table::
   :widths: 2 4 4
   :header-rows: 1

   * - ``lifetime``
     - Setup behavior
     - Teardown behavior
   * - ``no_launch``
     - Verifies a container of that name is already running; never starts one.
     - No-op
   * - ``per_run``
     - Force-removes any stale container of the same name, then launches.
     - ``docker rm -f``
   * - ``persistent``
     - Attaches if running on all hosts; cold-starts if absent on all hosts; **refuses** on partial or failed probe.
     - No-op

.. tip::

  With ``persistent``, always pin ``container.name`` explicitly. The default name is derived from the image, so bumping an image tag silently abandons the old container and starts a new one.

Registry authentication
-----------------------

``runtime.args.registry`` and top-level ``image_tar`` are rejected in the vLLM file. Set them on the cluster file's ``container`` block if you need a private pull or a ``docker load``. The orchestrator then does the following. ``username`` and ``password_file`` are required; ``server`` defaults to Docker Hub. The password is read from a file on each remote host. There is no inline password key, and the login is kept out of the logs. Login is skipped when ``image_tar`` is set, because a tar load does not pull.

Image resolution order at launch:

1. If ``image_tar`` is set and the image is absent, ``docker load`` it.
2. Otherwise, if ``registry`` is set, log in.
3. Check whether the image exists on **all** hosts.
4. If not, ``docker pull`` it, with no retry or fallback.

.. note::

  The image-exists check matches ``Repository:Tag`` exactly, so an image referenced without a tag or by digest never matches and is pulled on every run.

Cluster file merge
------------------

The variant's ``container`` block is deep-merged **onto** the cluster file's block: dictionaries merge key-wise, while scalars and lists are replaced. Cluster-set values survive unless the variant sets the same key.

.. warning::

  A loaded vLLM ``container`` block always contributes ``lifetime``, ``name``, and ``image``. ``name`` and ``image`` are required, so omitting them fails validation. An empty ``image`` string overwrites a cluster-file image. The orchestrator then logs ``Container image not specified in config`` and ``test_launch_container`` fails because ``setup_containers()`` returned false. Set ``image`` in the vLLM file. The packaged Llama files use ``<changeme>`` there.

.. _vllm-paths:

Paths
=====

All four keys are required.

.. list-table::
   :widths: 3 7
   :header-rows: 1

   * - Key
     - Description
   * - ``shared_fs``
     - Root of the shared filesystem, typically the anchor other paths reference.
   * - ``models_dir``
     - In-container directory exported as ``HF_HUB_CACHE``. ``test_model_fetch`` runs ``du -sb`` on this path inside the container. In the packaged Llama files it is ``/models``, not the host weight directory. The host directory is the volume source ``<changeme-models-mount>:/models:ro``.
   * - ``log_dir``
     - Root for run artifacts. In the packaged Llama files it is ``{shared_fs}/LOGS``, and ``shared_fs`` is ``/home/{user-id}``, which is also bind-mounted into the container. ``per_run`` teardown deletes the container, not that host directory, so the logs remain.
   * - ``hf_token_file``
     - Path of the Hugging Face token file as seen inside the container. In the packaged Llama files it is ``{shared_fs}/.hf_token``. CVS does not read the file on the machine that starts ``cvs run``. The env script reads it inside the container when a server or benchmark command is sourced. A missing or empty file exports an empty ``HF_TOKEN`` and the run continues.

``{user-id}`` becomes the login name of the ``cvs run`` process unless the cluster file sets ``username``. Place the token file on each node at the substituted host path. The home volume makes that path visible inside the container. Do not put the token in ``container.env``.

Per-cell artifacts land in::

  <log_dir>/vllm/out-node<rank>/isl<isl>_osl<osl>_conc<conc>/
    vllm_serve_server.log
    client.log
    results

.. _vllm-model:

Model
=====

Model configuration keys and their defaults are listed below.

.. list-table::
   :widths: 3 2 5
   :header-rows: 1

   * - Key
     - Default
     - Description
   * - ``server_params.model``
     - none
     - Path ``vllm serve`` opens inside the container. Packaged Llama files set ``<changeme>``. With their mount, use a path under ``/models``.

.. _vllm-roles:

Server role
===========

``server_params`` controls the ``vllm serve`` process. Harness-owned fields
are ``model``, ``tensor_parallel_size``, ``pipeline_parallel_size``, ``port``,
``dist_init_port``, polling controls, and ``distributed_executor_backend``.
Every other snake-case key is passed through to ``vllm serve``.

.. list-table::
   :widths: 3 2 5
   :header-rows: 1

   * - Key
     - Default
     - Description
   * - ``model``
     - none
     - Local model path supplied as the positional ``vllm serve`` argument.
   * - ``tensor_parallel_size``
     - none
     - Tensor-parallel degree.
   * - ``pipeline_parallel_size``
     - ``1``
     - Pipeline-parallel degree. On one host, vLLM needs
       ``tensor_parallel_size`` × ``pipeline_parallel_size`` GPUs on that
       machine, so a TP 8 × PP 2 recipe needs 16.
   * - ``port``
     - ``8888``
     - OpenAI-compatible server port. The client calls ``http://0.0.0.0:<port>``.
   * - ``dist_init_port``
     - ``29501``
     - Head-node port for multi-host startup: ``--master-port`` on mp, the Ray
       head port on ray. Ignored on one host.
   * - ``server_poll_iterations``
     - ``60``
     - Readiness polls after warmup. The packaged Llama single file keeps 60; the distributed file sets 360.
   * - ``server_poll_wait_s``
     - ``60``
     - Seconds between readiness polls, including after the last unreadiness check. Packaged Llama single keeps 60; distributed sets 10.
   * - ``server_warmup_wait_s``
     - ``330``
     - Seconds after a 30-second log precheck and before the readiness polls. Packaged Llama single keeps 330; distributed sets 30.
   * - ``distributed_executor_backend``
     - ``"mp"``
     - Snake-case JSON key. ``"mp"`` or ``"ray"``. See :ref:`vllm-backends`.
       Ignored on one host: CVS passes no ``--distributed-executor-backend``.

How server_params are flattened
-------------------------------

.. list-table::
   :widths: 3 3 4
   :header-rows: 1

   * - JSON value
     - Emitted
     - Example
   * - Scalar
     - ``--flag value``
     - ``"kv_cache_dtype": "fp8"`` → ``--kv-cache-dtype fp8``
   * - ``true``
     - ``--flag`` (bare)
     - ``"enforce_eager": true`` → ``--enforce-eager``
   * - ``false``
     - rejected
     - Omit the setting or use vLLM's explicit negative option.
   * - List
     - One flag followed by its values.
     - ``"x": ["a","b"]`` → ``--x a b``

When ``server_params.max_model_len`` is absent, CVS derives one for each
benchmark cell from its effective parameters after sweep overrides:

.. code:: text

  ceil((ISL + OSL) * (1 + random_range_ratio)) + random_prefix_len + 8

An explicit non-null ``server_params.max_model_len`` takes precedence and
emits exactly one ``--max-model-len`` flag. An explicit null value is an
intentional opt-out: the generic option serializer emits no flag, and CVS
suppresses the fallback so vLLM uses the model or image default.

Environment variables: two mechanisms
-------------------------------------

These are separate and are frequently confused.

.. list-table::
   :widths: 2 4 4
   :header-rows: 1

   * -
     - ``container.env``
     - generated per-command environment
   * - Applied by
     - ``docker run -e``
     - A sourced shell script inside the container after HCA discovery.
   * - Scope
     - Every command in the container, for its whole lifetime.
     - Hugging Face path/token variables and legacy network fallbacks.
   * - Changing it
     - Requires recreating the container.
     - Takes effect on the next command.
   * - Defaults
     - Orchestrator adds ``GPUS=8`` and ``MULTINODE=true`` as ``docker run -e`` unless ``container.env`` overrides them.
     - No network overrides unless a legacy top-level field is set.

The generated per-command environment always exports:

.. code:: bash

  if [ -s <paths.hf_token_file> ]; then export HF_TOKEN="$(tr -d '[:space:]' < <paths.hf_token_file>)"; else export HF_TOKEN=; fi
  export HF_HUB_CACHE=<paths.models_dir>

Packaged multi-host MI3xx files set the HCA, socket-interface, GID, and NCCL
debug settings in ``container.env`` so every command inherits them and
topology discovery does not overwrite ``NCCL_IB_HCA``. The packaged Llama
single-node file sets no ``container.env``. The legacy top-level
``ib_hca_devices`` and ``ib_netdev`` fields remain fallbacks for external
configurations that omit their corresponding container environment variables.
Put other static ROCm, NCCL, and vLLM exports in ``container.env``.

Packaged ``container.env`` AITER settings differ by recipe. The commit id
``4bdc8a788`` appears only as documentation context; it is not stored in the
JSON. Follow the env block of the file you copied. Current files that set
AITER-related variables:

- DeepSeek V4 Flash, DeepSeek V4 Pro, GLM 5.1 FP8, and GLM 5.2 FP8 single-node
  files set ``VLLM_ROCM_USE_AITER=1``, ``VLLM_ROCM_USE_AITER_MHA=0``, and
  ``GPU_ARCHS=gfx942``. Their distributed twins, and GLM 5.1 BF16 distributed,
  add ``VLLM_ROCM_USE_AITER_FP4BMM=0``.
- Mistral Large 3 FP8 distributed sets ``VLLM_ROCM_USE_AITER=1``,
  ``VLLM_ROCM_USE_AITER_MHA=0``, and ``GPU_ARCHS=gfx942``.
- DeepSeek R1 0528 FP8 distributed sets ``VLLM_ROCM_USE_AITER=1`` and disables
  ``VLLM_ROCM_USE_AITER_MHA``, ``VLLM_ROCM_USE_AITER_MLA``, and
  ``VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS``, with ``GPU_ARCHS=gfx942``.
- Kimi K2.5 sets ``VLLM_ROCM_USE_AITER=1`` and disables
  ``VLLM_ROCM_USE_AITER_MHA``, ``VLLM_ROCM_USE_AITER_FP4BMM``,
  ``VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS``, and
  ``VLLM_ROCM_USE_AITER_MLA``. The distributed file also sets ``GPU_ARCHS=gfx942``.
- Kimi K2.6 and Kimi K2.7 Code distributed set ``VLLM_ROCM_USE_AITER=0`` and
  disable the same four feature variables, with ``GPU_ARCHS=gfx942``.
- DeepSeek R1 0528 MXFP4 distributed and Qwen3.5 397B MXFP4 distributed set
  only ``VLLM_ROCM_USE_AITER_FP4BMM=0``.
- The MI455X DeepSeek V4 Flash single file sets ``VLLM_ROCM_USE_AITER=1`` and
  ``AITER_JIT_DIR``.

Llama 3.3 70B and the other packaged recipes that do not appear above set no
AITER variables. Do not copy an AITER block from another recipe without
checking the image you actually run.

.. _vllm-params:

Benchmark parameters
====================

``benchmark_params`` holds client defaults. Per-cell entries in ``sweeps``
override these values. CVS owns endpoint construction, result paths, and
percentile reporting.

.. list-table::
   :widths: 3 2 5
   :header-rows: 1

   * - Key
     - Default
     - Description
   * - ``backend``
     - ``"vllm"``
     - Client backend for ``vllm bench serve``.
   * - ``dataset_name``
     - ``"random"``
     - Dataset for the load generator.
   * - ``num_prompts``
     - ``3200``
     - Total prompts per cell. The packaged Llama single file sets 320; the distributed file sets 64.
   * - ``burstiness``
     - ``1.0``
     - 1.0 is a uniform arrival process; lower is burstier.
   * - ``seed``
     - ``0``
     - Random seed.
   * - ``request_rate``
     - ``"inf"``
     - Arrival rate; ``inf`` sends as fast as concurrency allows.
   * - ``random_range_ratio``
     - ``0.0``
     - Length jitter around ISL/OSL; also feeds the derived max-model-len.
   * - ``random_prefix_len``
     - ``0``
     - Shared prefix length.
   * - ``tokenizer_mode``
     - ``"auto"``
     - Tokenizer mode.
   * - ``client_poll_iterations``
     - ``20``
     - Client completion polls. Packaged Llama single sets 90; distributed sets 540.
   * - ``client_poll_wait_s``
     - ``60``
     - Seconds between client polls. Packaged Llama single keeps 60; distributed sets 10.
   * - ``client_initial_wait_s``
     - ``120``
     - Seconds before the first client poll. Packaged Llama single keeps 120; distributed sets 10.
   * - ``trust_remote_code``
     - ``false``
     - Emits ``--trust-remote-code`` when true.
   * - ``ignore_eos``
     - ``true``
     - Emits ``--ignore-eos`` when true.

``base_url`` is not a config key. ``VllmJob`` always uses ``http://0.0.0.0``. Setting it, or any other reserved benchmark name, fails load with ``benchmark_params cannot override harness fields``. Reserved names are ``backend``, ``base_url``, ``model``, ``max_concurrency``, ``random_input_len``, ``random_output_len``, ``result_dir``, ``result_filename``, ``percentile_metrics``, and ``metric_percentiles``. Reserved server names, which CVS adds itself on multi-host mp, are ``master_addr``, ``master_port``, ``nnodes``, ``node_rank``, and ``headless``.

The client gives up with ``client did not complete before poll cap`` after the initial wait plus one sleep per poll. The server gives up with ``vllm server did not become ready before timeout`` after a fixed 30-second log precheck, the warmup, and one sleep per unreadiness poll. Failed requests above 1 percent raise ``client failed:``, including ``Failed requests: <n>/<total> (<percent>)``. At or below 1 percent the cell continues.

.. tip::

  Arbitrary snake-case keys under ``benchmark_params`` and a cell override are
  translated to ``vllm bench serve`` options. They cannot override model,
  endpoint, sequence lengths, concurrency, result paths, or harness-owned
  percentile reporting.

.. _vllm-sweep:

Sweep
=====

The sweep is an explicit list of canonical cells, not a cartesian product.
``sweeps`` defines optional overrides and ``runs`` selects the cells to execute.

.. code:: json

    {
      "sweeps": {
        "ISL=1000,OSL=1000,TP=8,PP=2,CONC=16": { "num_prompts": 50 },
        "ISL=1000,OSL=1000,TP=8,PP=2,CONC=32": {}
      },
      "runs": [
        "ISL=1000,OSL=1000,TP=8,PP=2,CONC=16",
        "ISL=1000,OSL=1000,TP=8,PP=2,CONC=32"
      ]
    }

.. list-table::
   :widths: 3 7
   :header-rows: 1

   * - Key
     - Description
   * - ``sweeps.<cell>``
     - Per-cell benchmark override object.
   * - ``runs[]``
     - Canonical key declared in ``sweeps``.

An undeclared, malformed, duplicated, or TP/PP-inconsistent cell key is a
load-time error.

Cell keys
---------

Each run is one **cell**, identified by a canonical key used to look up thresholds:

.. code:: text

  ISL=<isl>,OSL=<osl>,TP=<tp>,PP=<pp>,CONC=<conc>

``PP=`` is always present, including single-node and Ray runs with ``PP=1``.
The host count is placement information, not a threshold dimension.
Examples::

  ISL=1000,OSL=1000,TP=8,PP=1,CONC=16
  ISL=1000,OSL=1000,TP=8,PP=2,CONC=16

Server reuse
------------

Cells with identical server arguments share a server identity, so the suite
reuses the running server instead of stopping it, restarting, and reloading
weights. The derived ``--max-model-len`` is part of that identity: different
derived values force a restart, while cells with equal derived values reuse the
server. For example, with zero range ratio and prefix, 1024/8192 and 8192/1024
both derive ``9224`` and can share. An explicit
non-null ``server_params.max_model_len`` can also allow different ISL/OSL
cells to share. Explicit null removes the max-model-len option from the server
identity entirely, so different ISL/OSL cells share when their other server
arguments match. Concurrency remains client-only, so ordering runs with
concurrency varying fastest makes a sweep substantially quicker.

.. _vllm-thresholds:

Thresholds
==========

Thresholds are keyed by canonical cell, then by a bare metric name:

.. code:: json

    {
      "ISL=1000,OSL=1000,TP=8,PP=1,CONC=16": {
        "output_throughput": {"kind": "min", "value": 4000},
        "mean_ttft_ms": {"kind": "max", "value": 500},
        "failed": {"kind": "max", "value": 0}
      }
    }

The ``4000`` and ``500`` figures only illustrate the grammar. Packaged threshold files use ``0``.

A cell may contain any subset of the registry, including an empty object. When
``enforce_thresholds`` is true, every selected run must have a threshold cell,
but only specs present in that cell create verification subtests. When it is
false, recorded metrics keep status ``pass`` and no threshold subtest is
emitted.

Sweep specs are strict at load time regardless of enforcement:

- A spec contains exactly ``kind`` and ``value``.
- ``kind`` must equal the registry direction, exactly ``min`` or ``max``.
- ``value`` must be a finite JSON number. Booleans, strings, null, arrays,
  objects, NaN, and infinity are rejected.
- Cell-level keys beginning with ``_comment`` or ``_example`` are metadata and
  are ignored by metric validation; all other keys must name a registered
  metric.
- Prefixed names (``client.*``, ``gpu.*``, ``prom.*``), unknown names, legacy
  kinds (including ``min_tok_s`` and ``max_ms``), references, tolerances,
  units, ``info``, and extra fields are rejected.

``min`` fails below its value and passes at or above it. ``max`` fails above
its value and passes at or below it. A gated actual that is missing, null,
boolean, string, collection, NaN, or infinity fails for every datasource.

The optional top-level ``accuracy`` block remains task-qualified and is exempt
from these vLLM sweep-name and spec rules.

.. _vllm-threshold-discovery:

Threshold file discovery
------------------------

Set ``threshold_json`` to the threshold file. A relative path resolves against
the configuration file's directory. If that file is missing, CVS warns and
uses the packaged file of the same name. Copy the sibling beside the config
when you want to edit the numbers.

There are 33 packaged threshold files. Every numeric ``value`` in them is
``0``. Those zeros are uncalibrated placeholders, not measured limits. The
Llama 3.3 70B FP8 single file has two cells
(``ISL=1024,OSL=1024,TP=8,PP=1,CONC=16`` and ``CONC=32``), each with the same
32-metric subset, and ``enforce_thresholds`` false. Its distributed sibling
has one cell, ``ISL=1024,OSL=1024,TP=8,PP=2,CONC=16``, with the same 32 zeros.
Most other MI3xx files also use that 32-metric subset, with one, two, or three
cells. ``mi455x_vllm_deepseek-v4-flash_fp8_single_threshold.json`` is the
exception: six cells, each gating only ``failed`` at ``0``, and the paired
config sets ``enforce_thresholds`` to true. All 56 registry metrics remain
supported, and every finite produced value is reported whether or not it has
a threshold spec. You can add any other registered metric or remove any
packaged entry.

Metrics
=======

vLLM uses one ordered 56-entry bare-name registry. The metric name owns its
raw source or derivation inputs, display unit, datasource, display category,
and exact threshold direction. ``median_*`` and ``p50_*`` remain separate.

Run and health (7)
------------------

``max_concurrency``, ``max_concurrent_requests``, ``num_prompts``,
``completed``, ``failed``, ``success_rate``, ``duration``.

Throughput and totals (10)
--------------------------

``request_throughput``, ``goodput``, ``output_throughput``,
``total_token_throughput``, ``per_gpu_throughput``,
``decode_throughput_p50``, ``max_output_tokens_per_s``, ``rtfx``,
``total_input_tokens``, ``total_output_tokens``.

TTFT (8)
--------

``mean_ttft_ms``, ``median_ttft_ms``, ``std_ttft_ms``, ``p50_ttft_ms``,
``p90_ttft_ms``, ``p95_ttft_ms``, ``p99_ttft_ms``,
``normalized_ttft_ms_per_tok``.

TPOT (7)
--------

``mean_tpot_ms``, ``median_tpot_ms``, ``std_tpot_ms``, ``p50_tpot_ms``,
``p90_tpot_ms``, ``p95_tpot_ms``, ``p99_tpot_ms``.

ITL (8)
-------

``mean_itl_ms``, ``median_itl_ms``, ``std_itl_ms``, ``p50_itl_ms``,
``p90_itl_ms``, ``p95_itl_ms``, ``p99_itl_ms``, ``decode_latency_ratio``.

End-to-end latency (7)
----------------------

``mean_e2el_ms``, ``median_e2el_ms``, ``std_e2el_ms``, ``p50_e2el_ms``,
``p90_e2el_ms``, ``p95_e2el_ms``, ``p99_e2el_ms``.

GPU (5)
-------

``peak_gpu_memory_mb``, ``model_load_memory_mb``, ``model_load_s``,
``gpu_bandwidth_util_pct``, ``gpu_compute_util_pct``. Load time and memory are
captured once after successful readiness and reused for cells sharing that
server. Load time remains available whenever the elapsed measurement is finite.
Load memory requires finite pre/post VRAM snapshots; missing snapshots do not
disable server reuse or the elapsed measurement. A real zero memory delta
remains zero.

Prometheus (4)
--------------

``queue_time_p50_ms``, ``queue_time_p95_ms``, ``prefill_time_p50_ms``, and
``prefill_time_p95_ms``. CVS diffs before/after histogram scrapes so reused
server counters remain isolated to one cell.

Directions
----------

The following are ``min``: ``max_concurrency``,
``max_concurrent_requests``, ``num_prompts``, ``completed``, ``success_rate``,
``request_throughput``, ``goodput``, every token throughput/total metric,
``rtfx``, ``gpu_bandwidth_util_pct``, and ``gpu_compute_util_pct``. Every other
registry metric is ``max``.

Projection and derivation
-------------------------

The vLLM result projector accepts only finite built-in integer and float values;
booleans are not numeric. Known metadata is ignored: ``date``,
``endpoint_type``, ``backend``, ``label``, ``model_id``, ``tokenizer_id``,
``burstiness``, and ``request_rate`` (including a finite request rate). Any
unknown finite top-level numeric field fails parsing and names the artifact.
The raw ``request_goodput`` field maps only to ``goodput``.

Derived metrics are emitted only when their result is finite:

.. code:: text

  per_gpu_throughput          = total_token_throughput / (tp * pp)
  normalized_ttft_ms_per_tok  = mean_ttft_ms / isl
  decode_latency_ratio        = p99_itl_ms / p50_itl_ms
  decode_throughput_p50       = 1000 / median_tpot_ms
  success_rate                = completed / (completed + failed)

Reporting and compatibility
---------------------------

Each ``test_vllm_inference[<cell>]`` row verifies its own cell. The ``<cell>``
id is the canonical key, such as ``ISL=1024,OSL=1024,TP=8,PP=1,CONC=16``.
After the benchmark completes, CVS builds every host row before emitting one
subtest for each enforced spec, so one failure does not hide sibling verdicts.
A missed threshold fails that cell's row but does not skip later cells.

With ``enforce_thresholds`` false, ``evaluate_metric_verdicts`` sets ``status``
to ``pass`` and emits no threshold subtest. The HTML Result column shows
Passed. The Note column is one of:

- ``metric produced without a configured threshold``
- ``threshold enforcement disabled; no threshold asserted``
- ``<metric>: unavailable or non-finite actual recorded; no threshold asserted``

A spec that is not enforced is labeled ``reference: <kind> <value>`` in the
gate column. With enforcement on, finite values without a spec still pass, and
each enforced spec is a single subtest. A non-finite gated actual fails with
``<metric>: actual must be a finite built-in int or float, got <value>``.

The row also emits one compact JUnit property with ``actuals_by_host`` and
metric contract ``{"id":"vllm-bare","version":1}``.

Run Deck tables, charts, and highlights use selected registry metrics rather
than all 56 columns. Historical namespaced vLLM Run Deck artifacts do not match
the contract: previous-run and manually selected viewer baselines display an
explicit incompatibility and suppress performance comparisons. A compatible
task-qualified accuracy section in the same baseline remains independently
eligible for accuracy comparison.

Results table
-------------

The summary table uses ``VLLM_RESULTS_COLUMNS``: Model, GPU, ISL, OSL, Policy,
Conc, Host, then Req/s, Total tok/s, Mean TTFT (ms), P95 TTFT (ms), Mean TPOT
(ms), P95 TPOT (ms), P99 ITL (ms), and Goodput (req/s). The GPU column is an
empty string. The Policy column is the full cell key.

.. _vllm-accuracy:

Accuracy tests
==============

Accuracy evaluation runs `lm-evaluation-harness <https://github.com/EleutherAI/lm-evaluation-harness>`_ against the live server after the performance sweep. Task **selection** lives in the configuration file; **gating values** live in the threshold file.

.. code:: json

    {
      "accuracy": {
        "tasks": [
          {
            "id": "gsm8k_strict",
            "tasks": "gsm8k",
            "backend": "vllm",
            "lm_eval_model": "local-completions",
            "num_fewshot": 5,
            "batch_size": "auto",
            "limit": 100,
            "num_concurrent": 8,
            "exec_timeout_sec": 7200,
            "extra_model_args": "tokenizer_backend=huggingface"
          }
        ]
      }
    }

.. list-table::
   :widths: 3 2 5
   :header-rows: 1

   * - Key
     - Default
     - Description
   * - ``id``
     - none
     - Unique label for this entry; duplicates are rejected.
   * - ``tasks`` (or legacy ``task``)
     - none
     - lm-eval task name or task list.
   * - ``lm_eval_model``
     - endpoint-derived
     - ``local-completions`` or ``local-chat-completions``
   * - ``num_fewshot``
     - lm-eval default
     - Few-shot example count, if explicitly set.
   * - ``num_concurrent``
     - ``8``
     - Concurrent requests.
   * - ``apply_chat_template``
     - ``false``
     - Enables or names the chat template.
   * - ``metadata``
     - ``{}``
     - Passed through to lm-eval.
   * - ``include_path``
     - ``""``
     - Directory of custom task definitions.
   * - ``gen_kwargs``
     - ``{}``
     - Generation arguments.

``lm_eval_model`` selects the API surface. If omitted, CVS derives it from
``apply_chat_template``:

.. list-table::
   :widths: 2 3 4
   :header-rows: 1

   * - Value
     - lm-eval model
     - Endpoint
   * - ``false``
     - ``local-completions``
     - ``/v1/completions``
   * - ``true``
     - ``local-chat-completions``
     - ``/v1/chat/completions``

CVS pins runtime installation to ``lm-eval[api,math]==0.4.12``. The shared
accuracy schema also exposes that release's evaluation controls, including
``batch_size``, ``max_batch_size``, ``device``, ``limit``, ``samples``,
``use_cache``, ``cache_requests``, ``check_integrity``,
``system_instruction``, ``fewshot_as_multiturn``, ``predict_only``, ``seed``,
``trust_remote_code``, ``confirm_run_unsafe_code``, ``metadata``, and
``gen_kwargs``. CVS owns the endpoint, model path, output path, and sample
logging. Results land under ``<log_dir>/accuracy``.

.. note::

  Most lm-eval tasks download their dataset from the Hugging Face Hub, and
  lm-eval inherits ``container.env``. If ``HF_HUB_OFFLINE=1`` or
  ``TRANSFORMERS_OFFLINE=1`` is set there, a dataset that is not already cached
  fails to load with ``OfflineModeIsEnabled``, and ``test_accuracy_eval`` fails
  after the whole performance sweep. Set these flags only on clusters without
  Hub access, and pre-cache every task's dataset first, for example by mounting
  a populated datasets cache and pointing ``HF_DATASETS_CACHE`` at it. The
  server needs no Hub access, because ``server_params.model`` is a local path.

Accuracy metric keys
--------------------

Accuracy metrics are keyed ``<lm_task_name>.<metric_key>``, with any comma in the name replaced by a double underscore. For example, gsm8k's ``exact_match,strict-match`` becomes:

.. code:: text

  gsm8k.exact_match__strict-match

Gate them in the threshold file's top-level ``accuracy`` block, keyed by the task ``id``:

.. code:: json

    {
      "accuracy": {
        "gsm8k_strict": {
          "gsm8k.exact_match__strict-match": { "kind": "min", "value": 0.75 }
        }
      }
    }

.. note::

  An accuracy failure does not mark the shared lifecycle as failed, so the remaining stages — the results table and teardown — still run normally.

Troubleshooting
===============

Common error messages and their causes and fixes are listed below.

.. list-table::
   :widths: 5 5
   :header-rows: 1

   * - Message
     - Cause and fix
   * - ``vllm_distributed requires pipeline_parallel_size>1 unless distributed-executor-backend=ray``
     - Two or more hosts on ``mp``. Raise ``pipeline_parallel_size``, or set the snake_case key ``distributed_executor_backend`` to ``"ray"``. One host does not raise this.
   * - ``vllm_distributed requires container.env.NCCL_SOCKET_IFNAME on multi-host clusters``
     - Set the socket interface. If you set any of the three socket variables, set all of them. A legacy ``ib_netdev`` also satisfies this check.
   * - ``ib_discovery preflight: requested HCA devices <missing> not found on <host>``
     - Multi-host only. A name in ``NCCL_IB_HCA`` is absent on that node. Preflight does not check socket-interface names, GID index, or link state, and it does not run for one host.
   * - ``vllm server did not become ready before timeout``
     - Readiness polls were exhausted. See ``server_poll_iterations``, ``server_poll_wait_s``, and ``server_warmup_wait_s``, plus the fixed 30-second log precheck.
   * - ``client did not complete before poll cap``
     - The benchmark client did not print a serving result before ``client_initial_wait_s`` plus ``client_poll_iterations`` × ``client_poll_wait_s``.
   * - ``client failed:`` … ``Failed requests:``
     - Failed requests were more than 1% of successful plus failed requests.
   * - ``Container image not specified in config``
     - ``container.image`` is empty. Note that a variant ``container`` block with no ``image`` overwrites the cluster file's value.
   * - ``runs must be a nonempty explicit list``
     - ``runs`` is missing or empty. List at least one canonical cell key from ``sweeps``.
   * - ``runs contains duplicate cells``
     - The same cell key appears twice in ``runs``.
   * - ``runs reference unknown sweeps``
     - A ``runs`` entry is not a key in ``sweeps``.
   * - ``run cell must be canonical ISL=<n>,OSL=<n>,TP=<n>,PP=<n>,CONC=<n>``
     - A sweep or run key is malformed.
   * - ``conflicts with server_params tensor/pipeline parallel size``
     - The TP or PP in a cell key does not match ``server_params``.
   * - ``duplicate task id(s)``
     - Two ``accuracy.tasks`` entries share an ``id``.
   * - ``unknown vLLM threshold metric '<metric>'``
     - The name is not one of the 56 registry metrics. ``_comment`` and ``_example`` keys are ignored.
   * - ``<metric> threshold kind must be '<direction>', got '<kind>'``
     - ``kind`` does not match the registry direction. Use the required ``min`` or ``max`` value shown in the message.
   * - ``<metric>: actual must be a finite built-in int or float, got <value>``
     - The gated datasource did not produce a valid value. Inspect the benchmark artifact, GPU telemetry, or server metrics for that cell; percentile collection is harness-owned.
   * - Unknown key under ``container`` or ``runtime.args``
     - ``ContainerConfig`` and ``RuntimeArgs`` forbid extra keys. ``env`` belongs on ``container``. ``image_tar``, ``registry``, ``shm_size``, and ``cap_add`` belong on the cluster file. See :doc:`/reference/cluster/cluster-file`.
   * - ``runtime.name`` is not ``docker``
     - The vLLM schema accepts only ``docker``.

Related resources
=================

- :doc:`/how-to/test-suites/inference/vllm` — step-by-step first run
- :doc:`/reference/cluster/cluster-file` — cluster file and orchestrator backends
- :doc:`/how-to/run-with-containers` — container backend walkthrough
- :doc:`/how-to/test-suites/index` — running other CVS suites
