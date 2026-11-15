.. meta::
  :description: Reference for the vLLM inference benchmark configuration in CVS, covering container setup, sweeps and runs, threshold files, accuracy tests, and multi-node execution.
  :keywords: CVS, vLLM, inference, ROCm, LLM, benchmark, GPU, AMD, threshold, multinode, accuracy, JSON, configuration

******************************************************************************
vLLM inference benchmark configuration file for Cluster Validation Suite (CVS)
******************************************************************************

The vLLM suites benchmark LLM serving throughput, latency, and accuracy on AMD Instinct GPUs. ``vllm_single`` runs on the first node in the cluster file and ignores the others. ``vllm_distributed`` runs on the first ``server_params.nnodes`` nodes, at least two, and ignores the rest. A cluster file with fewer nodes fails the run. Packaged distributed recipes run on Ray across two nodes. Their threshold values are zeros, not measured pass/fail limits.

Nodes are taken in ``node_dict`` order: the first JSON keys of a mapping, or
the first entries of a list. The first selected node is the head. CVS
rewrites ``head_node_dict.mgmt_ip`` to that node, even if the cluster file
names another head. Put the intended nodes first in ``node_dict``.

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
     - Resolve IB HCA devices; no-op for ``vllm_single``.
   * - 3
     - ``test_model_fetch``
     - Stage model weights.
   * - 4
     - ``test_openai_compatible_smoke``
     - Short-lived server; verifies the OpenAI-compatible API answers.
   * - 5
     - ``test_vllm_inference``
     - Run one benchmark cell (parametrized per entry in ``runs``) and verify its metrics; configured threshold gates are listed as subtests.
   * - 6
     - ``test_accuracy_eval``
     - The ``accuracy.lm_eval`` tasks, if any are configured.
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

If a stage fails, later stages are skipped rather than cascading into confusing downstream errors. The container is still torn down at the end of the test module even when a mid-run test fails.

.. _vllm-config-structure:

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
     - When ``false``, or without ``threshold_json``, metrics record and no threshold spec is asserted. Packaged Llama files set ``false``.
   * - ``threshold_json``
     - no
     - Path to the threshold file. See :ref:`vllm-threshold-file`.
   * - ``container``
     - yes
     - Container/Docker settings. See :ref:`vllm-container`.
   * - ``paths``
     - yes
     - Filesystem locations. See :ref:`vllm-paths`.
   * - ``server_params``
     - yes
     - Server fields. ``engine_args`` holds every other ``vllm serve`` option. See :ref:`vllm-roles`.
   * - ``benchmark_params``
     - yes
     - Benchmark fields. ``client_args`` holds every other ``vllm bench serve`` option. See :ref:`vllm-params`.
   * - ``timeouts``
     - no
     - Seconds the server and client waits may take. See :ref:`vllm-timeouts`.
   * - ``sweeps``
     - yes
     - Each cell key, mapped to that cell's overrides. See :ref:`vllm-sweep`.
   * - ``runs``
     - yes
     - Ordered list of the ``sweeps`` keys to run. See :ref:`vllm-sweep`.
   * - ``vllm``
     - yes
     - Names the engine and holds ``distributed_executor_backend``. See :ref:`vllm-backends`.
   * - ``thresholds``
     - no
     - Per-cell pass/fail specs. See :ref:`vllm-thresholds`.
   * - ``accuracy``
     - no
     - lm-eval tasks and the other accuracy checks. See :ref:`vllm-accuracy`.

A configuration file in the old format, with keys such as
``server_params.tensor_parallel_size``,
``server_params.distributed_executor_backend``,
``benchmark_params.ignore_eos``, or ``accuracy.tasks``, fails to load. So does
a threshold file with task ids directly under ``accuracy``. The error names
the old keys and the ``cvs config migrate vllm`` command that converts the
file. See :doc:`Convert an old configuration file </reference/configuration-files/inference/index>`.

For vLLM, the command keeps ``sweeps`` and ``runs`` as they are and makes
these moves:

- ``server_params.tensor_parallel_size``, ``pipeline_parallel_size``, and
  ``dist_init_port`` to ``tp``, ``pp``, and ``master_port``.
- ``server_params.distributed_executor_backend`` to
  ``vllm.distributed_executor_backend``, with the same value. A file that does
  not set it gets an empty ``vllm`` block, so it runs on ``ray``; set ``mp``
  there to keep the old launch.
- ``server_params.max_model_len`` to ``server_params.max_model_length``.
- The three server wait fields to ``timeouts.server_ready_s`` and the three
  client wait fields to ``timeouts.client_s``, as described below.
- ``accuracy.tasks`` to ``accuracy.lm_eval.tasks``. In a threshold file, each
  task-id section under ``accuracy`` moves under ``accuracy.lm_eval``.
- A ``pipeline_parallel_size`` above 1 also sets ``server_params.nnodes`` to
  that value.
- Other ``server_params`` and ``benchmark_params`` keys that are not fields
  move to ``engine_args`` and ``client_args``.

Move each field that migrate leaves in ``_unmapped`` as follows:

- ``ib_netdev``: set ``NCCL_SOCKET_IFNAME``, ``GLOO_SOCKET_IFNAME``, and
  ``TP_SOCKET_IFNAME`` in ``container.env`` to its value.
- ``ib_hca_devices``: set ``container.env.NCCL_IB_HCA`` to its devices,
  comma-separated.
- ``benchmark_params.dataset_name``, ``ignore_eos``, or ``trust_remote_code``:
  these settings are gone. The client always runs ``--dataset-name random``
  with ``--ignore-eos``, and passes ``--trust-remote-code`` exactly when
  ``server_params.engine_args.trust_remote_code`` is true. Migrate drops the
  field when it already says that; delete it, or set ``trust_remote_code`` in
  ``engine_args``.
- ``server_params.backend`` or ``benchmark_params.backend`` other than
  ``vllm``: delete it.
- A server or client wait field, including one inside a ``sweeps`` override:
  set ``timeouts.server_ready_s`` to ``server_warmup_wait_s`` +
  ``server_poll_iterations`` × ``server_poll_wait_s``, and
  ``timeouts.client_s`` to ``client_initial_wait_s`` +
  ``client_poll_iterations`` × ``client_poll_wait_s``. Count a missing field
  at its old default: 330, 60, and 60 for the server, and 120, 20, and 60 for
  the client. See :ref:`vllm-timeouts`.

.. important::

  Top-level keys, ``paths``, ``container``, ``container.runtime``,
  ``container.runtime.args``, ``server_params``, ``benchmark_params``,
  ``vllm``, and ``timeouts`` **forbid unknown keys**. Other ``vllm serve`` options go in
  ``server_params.engine_args``, and other ``vllm bench serve`` options in
  ``benchmark_params.client_args``, as snake-case names. CVS converts them to
  kebab-case CLI flags. ``null`` omits a flag, ``true`` emits a bare flag,
  scalars emit one value, lists emit one flag followed by values, and mappings
  emit compact JSON. ``false`` is rejected in ``engine_args``,
  ``client_args``, and per-cell overrides. Use an option's negative form
  instead. Keys that start with ``_`` are comments and are ignored, except in
  ``container.env``, which keeps them, and in ``engine_args`` and
  ``client_args``, which reject them. In a threshold cell, only metric names
  starting with ``_comment`` or ``_example`` are comments. A ``<changeme>`` or
  ``<changeme-...>`` left in any value, in the configuration file or its
  threshold file, fails config load, and the error names every such field.
  Values under comment keys, such as ``_volumes_example``, are not checked.

Placeholder substitution
------------------------

Values are resolved in three passes, so later forms can reference earlier ones:

1. **Cluster placeholders** — ``{user-id}`` resolves to the cluster file ``username``, then ``$USER``, then ``root``. ``{user}``, ``{home}``, ``{home-mount-dir}``, ``{node-dir-name}``, and ``{run_dir}`` are replaced in the same pass.
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

Three different things in this stack are called a "backend". They are unrelated, and confusing them is the most common configuration mistake. The client backend passed to ``vllm bench serve --backend`` is always ``vllm`` and is not a config key.

.. list-table::
   :widths: 3 2 5
   :header-rows: 1

   * - Setting
     - Values
     - What it selects
   * - ``vllm.distributed_executor_backend``
     - ``"ray"`` (default), ``"mp"``
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

Multinode runs support **two** executor backends, set by ``vllm.distributed_executor_backend``. The ``vllm`` block is required and has only this key. An empty ``vllm`` block selects ``ray``. The packaged distributed files set ``ray``, and the packaged single-node files set ``mp``. ``vllm_single`` runs one node, where the setting changes nothing: CVS passes no ``--distributed-executor-backend`` and starts no Ray cluster. On one node, ``--pipeline-parallel-size`` is added only when ``server_params.pp`` is greater than 1.

**mp.** Used when ``vllm.distributed_executor_backend`` is ``"mp"``. On ``vllm_distributed`` the suite injects the full distributed block into each rank's ``vllm serve`` command:

.. code:: bash

  vllm serve <model> --tensor-parallel-size <tp> --port <port> \
    --max-model-len <len> \
    --node-rank <rank> --master-addr <addr> --master-port <port> \
    --nnodes <n> --pipeline-parallel-size <pp> \
    --distributed-executor-backend mp

``<addr>`` is the head node and ``<n>`` is ``server_params.nnodes``. Every rank above 0 additionally gets ``--headless``. This path **requires pipeline parallelism** (``server_params.pp`` greater than 1).

**ray (default).** Used when ``vllm.distributed_executor_backend`` is ``"ray"`` or left out. Values other than ``"mp"`` and ``"ray"`` are configuration errors. On ``vllm_distributed``, Ray takes a different route:

1. Bootstrap the cluster head: ``ray start --head --port=<master_port>``
2. Bootstrap each worker: ``ray start --address=<cluster-head>:<master_port>``
3. Launch ``vllm serve`` on the **head node only**, with ``--distributed-executor-backend ray`` — workers run no serve process
4. On teardown, broadcast ``ray stop`` after the process kill

Under ray, none of the other mp distributed flags (``--node-rank``, ``--master-addr``, ``--master-port``, ``--nnodes``, ``--headless``) are emitted. ``--pipeline-parallel-size`` is added only when ``server_params.pp`` is greater than 1.

.. note::

  Ray does not *enable* multinode — it **relaxes** the pipeline-parallelism requirement. With ray, ``server_params.pp`` of 1 is legal, for pure tensor-parallel multinode serving. The packaged distributed files use ray with ``pp`` 2. With mp, pipeline parallelism is mandatory.

Because only the head node serves under ray, worker ranks produce no per-rank server log. That is expected.

Topology validation rules
-------------------------

These rules apply to ``vllm_distributed``. CVS checks them after the configuration file loads and before the container starts:

.. list-table::
   :widths: 4 6
   :header-rows: 1

   * - Condition
     - Rule
   * - always
     - ``server_params.nnodes`` **must** be 2 or more, and the cluster file must list at least that many nodes
   * - backend is mp
     - ``server_params.pp`` **must** be greater than 1
   * - backend is ray
     - ``server_params.pp`` of 1 is valid
   * - either backend
     - ``container.env.NCCL_SOCKET_IFNAME`` is **required**

``vllm_single`` never hits these rules. The exact errors are:

.. code:: text

  distributed runs need server_params.nnodes of 2 or more
  the cluster file has <n> node(s); distributed needs <nnodes>
  vllm_distributed requires server_params.pp>1 unless distributed_executor_backend=ray
  vllm_distributed requires container.env.NCCL_SOCKET_IFNAME on multi-host clusters

If ``container.env`` sets any of ``NCCL_SOCKET_IFNAME``, ``GLOO_SOCKET_IFNAME``, or ``TP_SOCKET_IFNAME``, it must set all three, and none of them can be empty.

Multinode prerequisites
-----------------------

Beyond the validation rules, a multinode run needs:

- ``server_params.nnodes`` — the node count, 2 or more. The cluster file must list at least that many nodes; CVS uses the first ``nnodes`` of them.
- ``server_params.master_port`` — default ``29501``; the head address is the first selected node, and every other node must reach it.
- ``container.env.NCCL_IB_HCA`` — the comma-separated RDMA HCA names available on every node. The packaged MI3xx configurations set ``rdma0`` through ``rdma7``.
- ``container.env.NCCL_SOCKET_IFNAME`` — the Linux netdev associated with the selected RNICs.
- ``container.env.GLOO_SOCKET_IFNAME`` and ``TP_SOCKET_IFNAME`` — generally the frontend/control-plane interface.
- ``container.env.NCCL_IB_GID_INDEX`` — the index for the intended RoCE/IB fabric. Use ``show_gids`` inside the container and choose an entry available on every selected HCA and node. If that command is unavailable, inspect ``ibv_devinfo -v`` and ``/sys/class/infiniband/<hca>/ports/<port>/gid_attrs/``.

``test_discover_topology`` runs only for ``vllm_distributed``. It checks that every node has the same non-empty HCA list (``ibv_devinfo -l``, otherwise ``/sys/class/infiniband``) and that every requested ``NCCL_IB_HCA`` name is in that list. A missing name fails with ``ib_discovery preflight: requested HCA devices <missing> not found on <host>. Available: <devs>``. It does not check socket-interface names, the GID index, or link state. ``vllm_single`` skips this stage. When ``NCCL_IB_HCA`` is set, discovery does not overwrite it. When it is unset, the discovered list is exported as ``NCCL_IB_HCA`` in the per-command env script.

.. _vllm-container:

Container and Docker configuration
==================================

The vLLM ``container`` block selects the image, the container name, the lifetime, ``container.env``, and a small set of ``docker run`` arguments. The container orchestrator adds further flags from its defaults and from the cluster file. Packaged MI3xx configurations leave Hugging Face hub access online. Their volume list includes both NIC families, labeled by ``_comment_bnxt_mount`` (Broadcom Thor2, ``libbnxt_re-rdmav34.so``) and ``_comment_ionic_mounts`` (AMD Pensando Ionic / AINIC: ``libionic.so``, ``libionic.so.1``, the versioned ``libionic.so.1.*`` soname, ``libionic-rdmav34.so``, and ``ionic.driver``).

Two volume entries ship as a bare ``"<changeme>"``: the models mount and the versioned Ionic library. Replace each with a whole ``host:container[:ro]`` entry. The sibling ``_volumes_example`` key shows one for each: ``/data/models:/models:ro``, and the ``libionic.so.1.*`` file that ``readlink -f /usr/lib/x86_64-linux-gnu/libionic.so.1`` prints, mounted at the same path. ``_volumes_example`` is a comment and is not mounted. Delete the NIC family that is not installed. ``docker run`` fails when a listed host path is missing.

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
              "<changeme>"
            ],
            "_volumes_example": [
              "/data/models:/models:ro"
            ]
          }
        }
      }
    }

Container block keys
--------------------

``container`` accepts only these keys. Anything else, including ``image_tar``, is a validation error.

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
     - The Docker keys under :ref:`vllm-runtime-args`. ``env`` is rejected.
   * - ``env``
     - ``{}``
     - Environment variables passed as ``docker run -e``. **On** ``container``, not under ``runtime.args``; it is the only environment block the vLLM file reads. ``HF_TOKEN`` and ``HUGGING_FACE_HUB_TOKEN``, in any letter case, are rejected in the vLLM file; set ``paths.hf_token_file`` instead. A token in the cluster file's ``container.env`` is passed to the container, and logs redact it.

``image_tar`` is not a field of this model; set it on the cluster file's ``container`` block. ``runtime.args`` keys such as ``registry``, ``shm_size``, and ``cap_add`` can be set in either file. CVS deep-merges the vLLM ``container`` block onto the cluster file's block and keeps cluster keys the vLLM file does not set. The orchestrator emits them: ``image_tar`` loads an image, ``registry`` logs in before a pull, ``shm_size`` becomes ``--shm-size``, and ``cap_add`` appends to the orchestrator's default capabilities. See :doc:`/reference/cluster/cluster-file` for the orchestrator container keys that page documents.

.. warning::

  Put ``env`` on ``container``. ``env`` under ``runtime.args`` fails config load with ``container.runtime.args.env is not read; set these variables in container.env``.

.. _vllm-runtime-args:

Runtime arguments
-----------------

``runtime.args`` accepts only the keys below. ``volumes``, ``devices``, ``network``, ``ipc`` and ``privileged`` come from the vLLM file or its schema defaults. ``shm_size``, ``cap_add`` and ``registry`` keep a cluster value when the vLLM file leaves them out. The orchestrator then appends its own mounts and devices; you cannot drop those defaults from the vLLM file.

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
   * - ``cap_add``, ``security_opt``, ``group_add``, ``ulimit``
     - Lists appended to the orchestrator's default capabilities, security options, groups, and ulimits.
   * - ``user``
     - ``--user``.
   * - ``shm_size``
     - ``--shm-size``.
   * - ``registry``
     - Registry login before a pull. See `Registry authentication`_.

The assembled command is:

.. code:: bash

  docker run -d --name <name> <args> <image> sleep infinity

The container is a long-lived sidecar; every workload command runs through ``docker exec`` inside it. InfiniBand devices are additionally passed through by per-host shell expansion at launch time, so each node mounts the devices it actually has.

.. note::

  ``--gpus`` is deliberately never emitted. GPU access on AMD hardware comes from the ``/dev/kfd`` and ``/dev/dri`` device mounts plus the ``video`` group, which the orchestrator adds. Image-tar load is orchestrator behavior; set ``image_tar`` in the cluster file.

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

Set ``runtime.args.registry`` in either file, and ``image_tar`` on the cluster file's ``container`` block, if you need a private pull or a ``docker load``. The orchestrator then does the following. ``username`` and ``password_file`` are required; ``server`` defaults to Docker Hub. The password is read from a file on each remote host. There is no inline password key, and the login is kept out of the logs. Login is skipped when ``image_tar`` is set, because a tar load does not pull.

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

  A loaded vLLM ``container`` block always contributes ``lifetime``, ``name``, and ``image``. ``name`` and ``image`` are required, so omitting them fails validation. An empty ``image`` string overwrites a cluster-file image. The orchestrator then logs ``Container image not specified in config`` and ``test_launch_container`` fails because ``setup_containers()`` returned false. Set ``image`` in the vLLM file. The packaged files ship ``<changeme>`` there, which fails config load until you replace it.

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
     - In-container directory exported as ``HF_HUB_CACHE``. ``test_model_fetch`` runs ``du -sb`` on this path inside the container. In the packaged Llama files it is ``/models``, not the host weight directory. The host directory is the source of the ``/models`` volume, which ships as ``<changeme>``; ``_volumes_example`` shows ``/data/models:/models:ro``.
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

``server_params`` controls the ``vllm serve`` process. Its fields are
``model``, ``tp``, ``pp``, ``dp``, ``ep``, ``port``, ``master_port``,
``nnodes``, ``prefill_nnodes``, ``max_model_length``, ``long_context``, and
``engine_args``. Every other ``vllm serve`` option
goes in ``engine_args``; a key outside this table fails config load. The
executor backend is set in the top-level ``vllm`` block; see
:ref:`vllm-backends`. The wait before the server is ready is set in
:ref:`vllm-timeouts`.

.. list-table::
   :widths: 3 2 5
   :header-rows: 1

   * - Key
     - Default
     - Description
   * - ``model``
     - none
     - Local model path supplied as the positional ``vllm serve`` argument.
   * - ``tp``
     - none
     - Tensor-parallel degree, passed as ``--tensor-parallel-size``.
   * - ``pp``
     - ``1``
     - Pipeline-parallel degree, passed as ``--pipeline-parallel-size``. On one
       host, vLLM needs ``tp`` × ``pp`` GPUs on that machine, so a TP 8 × PP 2
       recipe needs 16.
   * - ``dp``, ``ep``
     - ``1``
     - Must stay ``1``. Put vLLM data- and expert-parallel options in ``engine_args``.
   * - ``port``
     - ``8888``
     - OpenAI-compatible server port. The client calls ``http://0.0.0.0:<port>``.
   * - ``master_port``
     - ``29501``
     - Head-node port for multi-host startup: ``--master-port`` on mp, the Ray
       head port on ray. Ignored by ``vllm_single``.
   * - ``nnodes``
     - ``null``
     - Node count for ``vllm_distributed``, which runs on the first
       ``nnodes`` nodes of the cluster file and needs 2 or more. ``null``
       means one node. ``vllm_single`` always runs on one node.
   * - ``prefill_nnodes``
     - ``null``
     - For disaggregated serving. vLLM does not use it.
   * - ``max_model_length``
     - ``null``
     - Passed as ``--max-model-len``, right after ``--port``. When ``null``,
       CVS derives one for each run; see below.
   * - ``long_context``
     - ``{}``
     - Server flags for long-context evals, ``chunked_prefill_size`` and
       ``max_prefill_tokens``. vLLM does not use them.
   * - ``engine_args``
     - ``{}``
     - Every other ``vllm serve`` option, as snake-case names. These names
       fail config load with ``cannot set harness fields``: the fields above
       except ``prefill_nnodes`` and ``long_context``;
       ``tensor_parallel_size``, ``pipeline_parallel_size``, and
       ``max_model_len``; ``backend``, ``dist_init_port``, and
       ``distributed_executor_backend``; and the flags CVS sets on
       multi-host mp, ``master_addr``, ``node_rank``, and ``headless``. The
       old server wait fields, ``server_warmup_wait_s``,
       ``server_poll_iterations``, and ``server_poll_wait_s``, fail the same
       way. ``trust_remote_code: true`` here also gives the benchmark client
       ``--trust-remote-code``.

How engine_args and client_args are flattened
---------------------------------------------

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
   * - Mapping
     - One flag followed by compact JSON.
     - ``"limit_mm_per_prompt": {"image": 0}`` → ``--limit-mm-per-prompt {"image":0}``
   * - ``null``
     - nothing
     - ``"max_num_seqs": null`` emits no flag.

When ``server_params.max_model_length`` is ``null``, CVS derives one for each
run from its ISL, OSL, and that run's ``random_range_ratio`` and
``random_prefix_len``:

.. code:: text

  ceil((ISL + OSL) * (1 + random_range_ratio)) + random_prefix_len + 8

A set ``max_model_length`` is passed for every run instead. Every benchmark
run gets exactly one ``--max-model-len``.

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
     - Hugging Face path and token variables, and discovered HCAs.
   * - Changing it
     - Requires recreating the container.
     - Takes effect on the next command.
   * - Defaults
     - Orchestrator adds ``GPUS=8`` and ``MULTINODE=true`` as ``docker run -e`` unless ``container.env`` overrides them.
     - ``NCCL_IB_HCA`` only when ``container.env`` does not set it and topology discovery found HCAs.

The generated per-command environment always exports:

.. code:: bash

  if [ -s <paths.hf_token_file> ]; then export HF_TOKEN="$(tr -d '[:space:]' < <paths.hf_token_file>)"; else export HF_TOKEN=; fi
  export HF_HUB_CACHE=<paths.models_dir>

Packaged multi-host MI3xx files set the HCA, socket-interface, GID, and NCCL
debug settings in ``container.env`` so every command inherits them and
topology discovery does not overwrite ``NCCL_IB_HCA``. The packaged Llama
single-node file sets no ``container.env``. Put other static ROCm, NCCL, and
vLLM exports in ``container.env``.

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

``benchmark_params`` holds the client settings every run starts from. A
cell's overrides in ``sweeps`` replace them for that run; see
:ref:`vllm-sweep`. CVS owns endpoint construction, result paths, and
percentile reporting.

.. list-table::
   :widths: 3 2 5
   :header-rows: 1

   * - Key
     - Default
     - Description
   * - ``num_prompts``
     - none
     - Required. Total prompts per run. The packaged Llama single file sets 320; the distributed file sets 64.
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
   * - ``scaling_baseline_output_throughput``
     - ``null``
     - Not implemented for vLLM. Setting it fails config load with
       ``not implemented for vLLM``.
   * - ``client_args``
     - ``{}``
     - Every other ``vllm bench serve`` option, as snake-case names.

The client always gets ``--dataset-name random`` and ``--ignore-eos``. It gets
``--trust-remote-code`` exactly when ``server_params.engine_args.trust_remote_code``
is true.

``base_url`` is not a config key. ``VllmJob`` always uses ``http://0.0.0.0``. In ``client_args``, a field in this table other than ``scaling_baseline_output_throughput``, or a flag CVS sets itself, fails load with ``cannot set harness fields``. Those flags are ``backend``, ``base_url``, ``model``, ``dataset_name``, ``max_concurrency``, ``random_input_len``, ``random_output_len``, ``save_result``, ``result_dir``, ``result_filename``, ``percentile_metrics``, ``metric_percentiles``, ``ignore_eos``, and ``trust_remote_code``. The old client wait fields, ``client_initial_wait_s``, ``client_poll_iterations``, and ``client_poll_wait_s``, fail the same way. For the names ``engine_args`` refuses, see :ref:`vllm-roles`.

Failed requests above 1 percent raise ``client failed:``, including ``Failed requests: <n>/<total> (<percent>)``. At or below 1 percent the cell continues.

.. tip::

  Snake-case keys under ``benchmark_params.client_args`` are translated to
  ``vllm bench serve`` options. They cannot override model, endpoint, sequence
  lengths, concurrency, result paths, or harness-owned percentile reporting.

.. _vllm-timeouts:

Timeouts
========

``timeouts`` sets how long each wait may take, in whole seconds greater than
zero. A budget counts the waiting time between checks, so the time each check
takes comes on top. A field left out or set to ``null`` uses its default.

.. list-table::
   :widths: 3 2 5
   :header-rows: 1

   * - Key
     - Default
     - Description
   * - ``server_ready_s``
     - ``3930``
     - Waiting time for the server to become ready. CVS first waits a fixed 30
       seconds for the server log, outside this budget, then checks readiness
       every 10 seconds until those waits add up to ``server_ready_s``. The
       packaged Llama single file sets 3930; the distributed file sets 3630.
   * - ``client_s``
     - ``1320``
     - Waiting time for one cell's benchmark client to print its result,
       checked every 10 seconds until those waits add up to ``client_s``. The
       packaged Llama single file sets 5520; the distributed file sets 5410.

The server gives up with ``vllm server did not become ready before timeout``,
and the client with ``client did not complete before poll cap``. A server that
exits or logs a fatal error, or a client that crashes, stops its wait early.

.. _vllm-sweep:

Sweeps and runs
===============

``sweeps`` maps each canonical cell key to that cell's overrides. ``runs``
lists the ``sweeps`` keys to run, in order. Each run uses ``benchmark_params``
plus its cell's overrides. Neither is a cartesian product: list every cell.

.. code:: json

    {
      "sweeps": {
        "ISL=1024,OSL=1024,TP=8,PP=1,CONC=16": {},
        "ISL=1024,OSL=1024,TP=8,PP=1,CONC=32": {}
      },
      "runs": [
        "ISL=1024,OSL=1024,TP=8,PP=1,CONC=16",
        "ISL=1024,OSL=1024,TP=8,PP=1,CONC=32"
      ]
    }

``runs`` needs at least one key, and each key must be a ``sweeps`` key, listed
once. A ``sweeps`` key that ``runs`` does not list is not run. A malformed
cell key, or one whose TP or PP differs from ``server_params``, is a load-time
error, in ``sweeps`` or in the threshold file.

Per-cell overrides
------------------

A cell's object changes only that run. A key may be a ``benchmark_params``
field such as ``num_prompts``, or any other ``vllm bench serve`` option in
snake case, as in ``client_args``. Its value replaces the one from
``benchmark_params`` or ``client_args``, and follows the ``client_args``
rules. Keys that start with ``_`` are comments.

.. code:: json

    {
      "sweeps": {
        "ISL=1024,OSL=1024,TP=8,PP=1,CONC=16": {},
        "ISL=1024,OSL=1024,TP=8,PP=1,CONC=32": {"num_prompts": 640}
      }
    }

A cell may not set a flag CVS sets itself or a client wait field: ``backend``,
``base_url``, ``model``, ``dataset_name``, ``random_input_len``,
``random_output_len``, ``max_concurrency``, ``save_result``, ``result_dir``,
``result_filename``, ``percentile_metrics``, ``metric_percentiles``,
``ignore_eos``, ``trust_remote_code``, ``client_poll_iterations``,
``client_poll_wait_s``, or ``client_initial_wait_s``. Those fail config load
with ``cannot set harness fields``.

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
both derive ``9224`` and can share. A set ``server_params.max_model_length``
is the same for every cell, so cells with different ISL/OSL share when their
other server arguments match. Concurrency and per-cell overrides are
client-only, except that ``random_range_ratio`` and ``random_prefix_len``
change a derived ``--max-model-len``. Ordering ``runs`` with concurrency
varying fastest makes a run substantially quicker.

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
``enforce_thresholds`` is true and a threshold file is loaded, every entry in ``runs`` must have a threshold cell,
but only specs present in that cell create verification subtests. When it is
false, recorded metrics keep status ``pass`` and no threshold subtest is
emitted.

Cell specs are strict at load time regardless of enforcement:

- A spec contains exactly ``kind`` and ``value``.
- ``kind`` must equal the registry direction, exactly ``min`` or ``max``, or be
  ``info``. An ``info`` spec records its value and never fails.
- ``value`` must be a finite JSON number. Booleans, strings, null, arrays,
  objects, NaN, and infinity are rejected.
- Cell-level keys beginning with ``_comment`` or ``_example`` are comments and
  are ignored; all other keys must name a registered metric.
- Prefixed names (``client.*``, ``gpu.*``, ``prom.*``), unknown names, legacy
  kinds (including ``min_tok_s`` and ``max_ms``), references, tolerances,
  units, ``optional``, and extra fields are rejected. A legacy kind fails with
  the ``cvs config migrate vllm`` command that converts the threshold file.

``min`` fails below its value and passes at or above it. ``max`` fails above
its value and passes at or below it. A gated actual that is missing, null,
boolean, string, collection, NaN, or infinity fails for every datasource.

The optional top-level ``accuracy`` block holds only ``lm_eval``,
``custom_eval``, ``quant_parity``, and ``mtp_quality``, and its metric names
are not checked against the vLLM registry. See :ref:`vllm-accuracy`. A file
with task ids directly under ``accuracy`` is in the old format and is refused
with the ``cvs config migrate vllm`` command that converts it.

.. _vllm-threshold-file:

Threshold file
--------------

Set ``threshold_json`` to the threshold file. A relative path resolves against
the configuration file's directory. If that file is missing, CVS warns and
uses the packaged file of the same name. Copy the sibling beside the config
when you want to edit the numbers. A ``threshold_json`` that names a missing
file with no packaged copy fails the load.

Without ``threshold_json``, metrics record without gating, as if
``enforce_thresholds`` were ``false``, and CVS logs a warning. CVS does not
look for a threshold file beside the configuration file.

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

Accuracy evaluation runs `lm-evaluation-harness <https://github.com/EleutherAI/lm-evaluation-harness>`_ against the live server after the performance runs. Task **selection** lives in the configuration file's ``accuracy.lm_eval.tasks``; **gating values** live in the threshold file's ``accuracy.lm_eval``.

.. code:: json

    {
      "accuracy": {
        "lm_eval": {
          "tasks": [
            {
              "id": "gsm8k_flex",
              "task": "gsm8k",
              "num_fewshot": 5,
              "num_concurrent": 4
            },
            {
              "id": "gsm8k_strict",
              "task": "gsm8k",
              "num_fewshot": 5,
              "num_concurrent": 4,
              "metadata": {
                "filter": "strict-match"
              }
            }
          ]
        }
      }
    }

The ``accuracy`` block has four keys:

- ``lm_eval.tasks``: the lm-eval runs, described below. Each runs as one
  ``test_accuracy_eval[<id>]`` row.
- ``custom_eval``: a list of other evals, each picked by its ``type``. The only
  type is ``niah``. vLLM accepts the list but does not run it.
- ``quant_parity`` and ``mtp_quality``: off by default. Setting ``enabled`` to
  true in either fails config load with ``not implemented for vLLM``.

Each ``lm_eval.tasks`` entry takes these keys:

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
     - none
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

Accuracy metrics are keyed ``<lm_task_name>.<metric_key>``, with any comma in the name replaced by a double underscore. For example, gsm8k's ``exact_match,flexible-extract`` becomes:

.. code:: text

  gsm8k.exact_match__flexible-extract

Gate them in the threshold file's ``accuracy.lm_eval`` block, keyed by the task ``id``. These specs are checked only when ``enforce_thresholds`` is true:

.. code:: json

    {
      "accuracy": {
        "lm_eval": {
          "gsm8k_flex": {
            "gsm8k.exact_match__flexible-extract": {
              "kind": "min",
              "value": 0
            }
          }
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
   * - ``old config format: <keys>; run cvs config migrate vllm <config> <new config>``
     - The file uses the old format. Run the printed command, then fix any field it reports. See :doc:`Convert an old configuration file </reference/configuration-files/inference/index>`.
   * - ``_unmapped.<field>: has no place in the new format``
     - ``cvs config migrate`` could not place an old field. Move it to its new place or delete it, then delete the ``_unmapped`` block.
   * - ``distributed runs need server_params.nnodes of 2 or more``
     - ``vllm_distributed`` with ``server_params.nnodes`` unset or below 2. Set it to the node count.
   * - ``the cluster file has <n> node(s); distributed needs <nnodes>``
     - The cluster file lists fewer nodes than ``server_params.nnodes``. Add nodes, or lower ``nnodes``.
   * - ``vllm_distributed requires server_params.pp>1 unless distributed_executor_backend=ray``
     - ``vllm_distributed`` on ``mp`` with ``server_params.pp`` of 1. Raise ``server_params.pp``, or set ``vllm.distributed_executor_backend`` to ``"ray"``, the default.
   * - ``vllm_distributed requires container.env.NCCL_SOCKET_IFNAME on multi-host clusters``
     - Set the socket interface. If you set any of the three socket variables, set all of them.
   * - ``ib_discovery preflight: requested HCA devices <missing> not found on <host>``
     - ``vllm_distributed`` only. A name in ``NCCL_IB_HCA`` is absent on that node. Preflight does not check socket-interface names, GID index, or link state, and it does not run for ``vllm_single``.
   * - ``vllm server did not become ready before timeout``
     - The server was not ready once the waits between readiness checks, after the fixed 30-second log precheck, added up to ``timeouts.server_ready_s``. See :ref:`vllm-timeouts`.
   * - ``client did not complete before poll cap``
     - The benchmark client did not print a serving result before the waits between checks added up to ``timeouts.client_s``. See :ref:`vllm-timeouts`.
   * - ``client failed:`` … ``Failed requests:``
     - Failed requests were more than 1% of successful plus failed requests.
   * - ``Container image not specified in config``
     - ``container.image`` is empty. Note that a variant ``container`` block with no ``image`` overwrites the cluster file's value.
   * - ``runs: needs at least one sweeps key to run``
     - ``runs`` is empty. List at least one ``sweeps`` key.
   * - ``cell keys listed more than once``
     - The same cell key appears twice in ``runs``.
   * - ``runs names cell keys that sweeps does not define``
     - Add the key to ``sweeps``, or remove it from ``runs``.
   * - ``cell key '<key>' ...; expected ISL=<n>,OSL=<n>,TP=<n>,PP=<n>,CONC=<n>``
     - A cell key, in ``sweeps`` or the threshold file, is malformed.
   * - ``conflicts with server_params.tp or server_params.pp``
     - The TP or PP in a cell key does not match ``server_params``.
   * - ``runs missing threshold coverage``
     - ``enforce_thresholds`` is true and a run has no threshold cell.
   * - ``cannot set harness fields``
     - ``engine_args``, ``client_args``, or a cell in ``sweeps`` names a field or a flag CVS sets itself. Use the field instead.
   * - ``not implemented for vLLM``
     - ``accuracy.quant_parity.enabled``, ``accuracy.mtp_quality.enabled``, or ``benchmark_params.scaling_baseline_output_throughput`` is set. vLLM does not implement it.
   * - ``duplicate task id(s)``
     - Two ``accuracy.lm_eval.tasks`` entries share an ``id``.
   * - ``unknown vLLM threshold metric '<metric>'``
     - The name is not one of the 56 registry metrics. Names beginning with ``_comment`` or ``_example`` are ignored.
   * - ``<metric> threshold kind must be '<direction>' or 'info'``
     - ``kind`` does not match the registry direction. Use the ``min`` or ``max`` value shown in the message, or ``info`` to record without gating.
   * - ``<metric>: actual must be a finite built-in int or float, got <value>``
     - The gated datasource did not produce a valid value. Inspect the benchmark artifact, GPU telemetry, or server metrics for that cell; percentile collection is harness-owned.
   * - Unknown key under ``container`` or ``runtime.args``
     - ``container`` and ``runtime.args`` forbid keys they do not define. ``image_tar`` belongs on the cluster file. See :doc:`/reference/cluster/cluster-file`.
   * - ``container.runtime.args.env is not read; set these variables in container.env``
     - Move ``env`` from ``runtime.args`` to ``container``.
   * - ``<field>: still a <changeme> placeholder; replace it with the value for your setup``
     - A template value was not replaced. The error lists every such field.
   * - ``runtime.name`` is not ``docker``
     - The vLLM schema accepts only ``docker``.

Related resources
=================

- :doc:`/how-to/test-suites/inference/vllm` — step-by-step first run
- :doc:`/reference/cluster/cluster-file` — cluster file and orchestrator backends
- :doc:`/how-to/run-with-containers` — container backend walkthrough
- :doc:`/how-to/test-suites/index` — running other CVS suites
