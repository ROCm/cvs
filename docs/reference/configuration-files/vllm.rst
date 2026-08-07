.. meta::
  :description: Configure the vLLM inference benchmark suite in CVS
  :keywords: inference, ROCm, cvs, vLLM, LLM, benchmark, multinode, thresholds, metrics, accuracy

**********************************
vLLM inference configuration file
**********************************

The vLLM suite benchmarks LLM serving throughput, latency, and accuracy on AMD Instinct GPUs. It is a **single parametrized suite**: the same test file covers single-node and multinode pipeline-parallel runs, and the topology is determined entirely by the configuration file. There is no separate "single-node" and "distributed" suite to choose between.

Run it with:

.. code:: bash

  cvs run vllm --cluster_file <cluster.json> --config_file <config.json>

For a step-by-step walkthrough of a first run, see :doc:`/how-to/run-vllm-benchmarks`. This page is the schema and metric reference.

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
     - Pull/load the image and start the container on every node
   * - 1
     - ``test_setup_sshd``
     - Always skipped for vLLM (see note below)
   * - 2
     - ``test_discover_topology``
     - Resolve IB HCA devices; no-op when ``nnodes`` is 1
   * - 3
     - ``test_model_fetch``
     - Stage model weights
   * - 4
     - ``test_openai_compatible_smoke``
     - Short-lived server; verifies the OpenAI-compatible API answers
   * - 5
     - ``test_vllm_inference``
     - Run one benchmark cell (parametrized per sweep run)
   * - 6
     - ``test_metric``, ``test_gpu_metric``, ``test_prom_metric``
     - One row per metric, per cell
   * - 7
     - ``test_accuracy_eval``
     - lm-eval accuracy tasks, if any are configured
   * - 8
     - ``test_print_results_table``
     - Console + report summary table
   * - 9
     - ``test_teardown``
     - Stop the server and tear down the container

.. note::

  ``test_setup_sshd`` always skips in this suite. vLLM uses ``--distributed-executor-backend mp`` with NCCL over the host network, so no inter-container sshd is needed. A skipped row here is expected, not a problem.

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
   * - ``schema_version``
     - yes
     - Must be ``1``
   * - ``framework``
     - yes
     - Must be ``"vllm"``
   * - ``gpu_arch``
     - yes
     - GPU architecture label, for example ``"mi300x"``. Reported, not enforced
   * - ``enforce_thresholds``
     - no (default ``true``)
     - When ``false``, threshold failures and coverage gaps become warnings
   * - ``threshold_json``
     - no
     - Explicit path to the threshold file. See :ref:`vllm-threshold-discovery`
   * - ``container``
     - no
     - Container/Docker settings. See :ref:`vllm-container`
   * - ``paths``
     - yes
     - Filesystem locations. See :ref:`vllm-paths`
   * - ``model``
     - yes
     - Model identifier. See :ref:`vllm-model`
   * - ``roles``
     - yes
     - Server arguments and environment. See :ref:`vllm-roles`
   * - ``params``
     - yes
     - Client and topology parameters. See :ref:`vllm-params`
   * - ``sweep``
     - yes
     - Sequence combinations and runs. See :ref:`vllm-sweep`
   * - ``thresholds``
     - no
     - Per-cell pass/fail specs. See :ref:`vllm-thresholds`
   * - ``accuracy``
     - no
     - lm-eval task selection. See :ref:`vllm-accuracy`

.. important::

  Every block except ``container`` **forbids unknown keys**. A misspelled key is a hard validation error at load time, not a silently ignored setting. The ``container`` block is permissive because it passes ``runtime`` and other keys through to the orchestrator untouched.

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
   * - ``params.backend``
     - ``"vllm"`` (default)
     - The **client** backend passed to ``vllm bench serve --backend``. Nothing to do with distribution
   * - ``roles.server.serve_args.``\ ``distributed-executor-backend``
     - ``"mp"`` (default), ``"ray"``
     - How vLLM distributes the model across nodes. This is the multinode setting
   * - ``container.runtime.name``
     - ``"docker"`` (default), ``"enroot"``
     - The container runtime
   * - Cluster file ``orchestrator``
     - ``"baremetal"``, ``"container"``
     - Whether CVS runs commands on the host or inside a container. See :doc:`/reference/configuration-files/cluster-file`

.. warning::

  ``enroot`` is registered but **not implemented**. Every method is a stub that returns failure, so a run with ``runtime.name: "enroot"`` fails at ``test_launch_container``. Podman is not supported. Use ``docker``.

Distributed executor: mp and ray
--------------------------------

Multinode runs support **two** executor backends. ``mp`` is the default and requires no configuration key at all.

**mp (default).** Used whenever ``distributed-executor-backend`` is absent from ``serve_args``. The suite injects the full distributed block into each rank's ``vllm serve`` command:

.. code:: bash

  vllm serve <model> --tensor-parallel-size <tp> --port <port> \
    --node-rank <rank> --master-addr <addr> --master-port <port> \
    --nnodes <n> --pipeline-parallel-size <pp> \
    --distributed-executor-backend mp

Every rank above 0 additionally gets ``--headless``. This path **requires pipeline parallelism** (``pipeline_parallel_size`` greater than 1).

**ray (opt-in).** Selected by setting ``distributed-executor-backend`` to the exact lowercase string ``"ray"``. Any other spelling, including ``"Ray"``, falls back to the mp path. Ray takes a completely different route:

1. Bootstrap the cluster head: ``ray start --head --port=<master_port>``
2. Bootstrap each worker: ``ray start --address=<master_addr>:<master_port>``
3. Launch ``vllm serve`` on the **head node only** — workers run no serve process
4. On teardown, broadcast ``ray stop`` after the process kill

Under ray, none of the mp distributed flags are emitted; the backend flag reaches vLLM through normal ``serve_args`` flattening. ``--pipeline-parallel-size`` is added only when ``pipeline_parallel_size`` is greater than 1.

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
   * - ``nnodes`` > 1, backend is not ray
     - ``pipeline_parallel_size`` **must** be greater than 1
   * - ``nnodes`` > 1, backend is ray
     - ``pipeline_parallel_size`` of 1 is valid
   * - ``pipeline_parallel_size`` > 1
     - ``nnodes`` **must** be greater than 1
   * - ``nnodes`` > 1, either backend
     - ``roles.server.ib_netdev`` is **required**

The corresponding error messages are:

.. code:: text

  nnodes=2 > 1 requires pipeline_parallel_size > 1 (got pp=1)
  pipeline_parallel_size=2 > 1 requires nnodes > 1 (got nnodes=1)
  ib_netdev is required in roles.server when nnodes > 1. Set it to the Linux
  network interface name for NCCL_SOCKET_IFNAME (e.g. "ens51f1np1"). Cannot be
  auto-derived from HCA names.

Multinode prerequisites
-----------------------

Beyond the validation rules, a multinode run needs:

- ``params.master_addr`` — the head node's address, reachable from every worker.
- ``params.master_port`` — default ``"29501"``.
- ``roles.server.ib_netdev`` — the Linux interface name. There is deliberately no ``"auto"`` value; it cannot be derived reliably from HCA names. This value populates ``NCCL_SOCKET_IFNAME``, ``GLOO_SOCKET_IFNAME``, and ``TP_SOCKET_IFNAME``.
- ``roles.server.ib_hca_devices`` — ``"auto"``, an explicit list, or ``null``. When set, populates ``NCCL_IB_HCA``.

.. _vllm-container:

Container and Docker configuration
==================================

The container block controls image selection, lifetime, and the ``docker run`` flags.

.. code:: json

    {
      "container": {
        "lifetime": "per_run",
        "name": "vllm_perf_inference_rocm",
        "image": "rocm/vllm:latest",
        "runtime": {
          "name": "docker",
          "args": {
            "network": "host",
            "ipc": "host",
            "privileged": true,
            "volumes": [
              "/home/{user-id}:/home/{user-id}",
              "{paths.models_dir}:/models"
            ]
          }
        }
      }
    }

Container block keys
--------------------

.. list-table::
   :widths: 3 2 5
   :header-rows: 1

   * - Key
     - Default
     - Description
   * - ``image``
     - none
     - Container image. **Required** — launch fails with ``Container image not specified in config``
   * - ``name``
     - ``<user>_<sanitized-image>``
     - Container name
   * - ``lifetime``
     - ``"per_run"``
     - One of ``no_launch``, ``per_run``, ``persistent``
   * - ``runtime.name``
     - ``"docker"``
     - Container runtime
   * - ``runtime.args``
     - ``{}``
     - Docker flags; see the table below
   * - ``env``
     - ``{}``
     - Container-level environment variables. **Top level, not under** ``runtime.args``
   * - ``image_tar``
     - absent
     - Path on each host to a saved image tar to ``docker load`` instead of pulling. **Top level**

.. warning::

  Put ``env`` at the **container top level**. Placing it under ``runtime.args`` crashes container launch: the code iterates that value as a sequence of pairs, which raises ``ValueError: too many values to unpack`` for any key longer than two characters.

Runtime arguments
-----------------

All keys under ``runtime.args`` are optional. **List-valued keys append to the defaults; scalar keys override them.** There is no way to remove a default device or capability.

.. list-table::
   :widths: 2 1 3 4
   :header-rows: 1

   * - Key
     - Merge
     - Default
     - Emitted flag
   * - ``volumes``
     - append
     - ``/home/$USER/.ssh:/host_ssh`` (always added)
     - ``-v <host>:<ctr>[:ro]``
   * - ``devices``
     - append
     - ``/dev/kfd``, ``/dev/dri``, ``/dev/infiniband``
     - ``--device <path>``
   * - ``cap_add``
     - append
     - ``SYS_PTRACE``, ``IPC_LOCK``, ``SYS_ADMIN``
     - ``--cap-add <cap>``
   * - ``security_opt``
     - append
     - ``seccomp=unconfined``, ``apparmor=unconfined``
     - ``--security-opt <opt>``
   * - ``group_add``
     - append
     - ``video``
     - ``--group-add <group>``
   * - ``ulimit``
     - append
     - ``memlock=-1``
     - ``--ulimit <limit>``
   * - ``network``
     - override
     - ``host``
     - ``--network <mode>``
   * - ``ipc``
     - override
     - ``host``
     - ``--ipc <mode>``
   * - ``privileged``
     - override
     - ``true``
     - ``--privileged``
   * - ``registry``
     - n/a
     - none
     - Triggers ``docker login``; see below

The assembled command is:

.. code:: bash

  docker run -d --name <name> <args> <image> sleep infinity

The container is a long-lived sidecar; every workload command runs through ``docker exec`` inside it. InfiniBand devices are additionally passed through by per-host shell expansion at launch time, so each node mounts the devices it actually has.

.. note::

  ``--gpus`` is deliberately never emitted — GPU access on AMD hardware comes from the ``/dev/kfd`` and ``/dev/dri`` device mounts plus the ``video`` group.

  ``shm_size`` is **not supported** on this path. Setting ``runtime.args.shm_size`` is silently ignored; ``--shm-size`` is never emitted.

Container lifetime
------------------

.. list-table::
   :widths: 2 4 4
   :header-rows: 1

   * - ``lifetime``
     - Setup behavior
     - Teardown behavior
   * - ``no_launch``
     - Verifies a container of that name is already running; never starts one
     - No-op
   * - ``per_run``
     - Force-removes any stale container of the same name, then launches
     - ``docker rm -f``
   * - ``persistent``
     - Attaches if running on all hosts; cold-starts if absent on all hosts; **refuses** on partial or failed probe
     - No-op

.. tip::

  With ``persistent``, always pin ``container.name`` explicitly. The default name is derived from the image, so bumping an image tag silently abandons the old container and starts a new one.

Registry authentication
-----------------------

Set ``runtime.args.registry`` to log in before pulling:

.. code:: json

    {
      "registry": {
        "username": "myuser",
        "password_file": "/path/on/each/host/to/token",
        "server": "registry.example.com"
      }
    }

``username`` and ``password_file`` are required; ``server`` defaults to Docker Hub. The password is read from a **file path on each remote host** — there is no inline password or token key, and the login is kept out of the logs. Login is skipped entirely when ``image_tar`` is set, since a tar load never pulls.

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

  A ``container`` block in the variant always contributes ``lifetime``, ``name``, and ``image`` — including empty defaults. If your variant defines ``container`` but omits ``image``, it overwrites a cluster-file ``image`` with an empty string and the launch fails. Set ``image`` in whichever file defines the block.

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
     - Root of the shared filesystem, typically the anchor other paths reference
   * - ``models_dir``
     - Model weight cache; exported into the server as ``HF_HUB_CACHE``
   * - ``log_dir``
     - Root for run artifacts
   * - ``hf_token_file``
     - Path to a file containing the Hugging Face token

If ``hf_token_file`` does not exist and the model is pre-staged (``model.remote`` of 0), the run continues with an empty token and the server sets ``HF_HUB_OFFLINE=1``. If the model is remote, the suite skips instead.

Per-cell artifacts land in::

  <log_dir>/vllm/out-node<rank>/isl<isl>_osl<osl>_conc<conc>/
    vllm_serve_server.log
    client.log
    results

.. _vllm-model:

Model
=====

.. list-table::
   :widths: 3 2 5
   :header-rows: 1

   * - Key
     - Default
     - Description
   * - ``id``
     - none
     - Hugging Face model ID or local path, for example ``amd/Llama-3.1-70B-Instruct-FP8-KV``
   * - ``remote``
     - none
     - ``0`` for a pre-staged model

.. important::

  ``remote: 1`` is **not implemented** and raises ``NotImplementedError`` at load time. Stage weights under ``paths.models_dir`` and use ``remote: 0``.

.. _vllm-roles:

Server role
===========

``roles.server`` controls the ``vllm serve`` process.

.. list-table::
   :widths: 3 2 5
   :header-rows: 1

   * - Key
     - Default
     - Description
   * - ``serve_args``
     - ``{}``
     - Flags passed through to ``vllm serve``
   * - ``env``
     - ``{}``
     - Environment for the server process and the benchmark client
   * - ``ib_hca_devices``
     - ``null``
     - ``"auto"``, an explicit list, or ``null``; sets ``NCCL_IB_HCA``
   * - ``ib_netdev``
     - ``null``
     - Interface name; required when ``nnodes`` is greater than 1

How serve_args are flattened
----------------------------

.. list-table::
   :widths: 3 3 4
   :header-rows: 1

   * - JSON value
     - Emitted
     - Example
   * - Scalar
     - ``--flag value``
     - ``"kv-cache-dtype": "fp8"`` → ``--kv-cache-dtype fp8``
   * - ``true``
     - ``--flag`` (bare)
     - ``"enforce-eager": true`` → ``--enforce-eager``
   * - ``false``
     - nothing
     - ``"enforce-eager": false`` → omitted entirely
   * - List
     - flag repeated per element
     - ``"x": ["a","b"]`` → ``--x a --x b``

``serve_args.log-level``, if set, must be one of ``debug``, ``info``, ``warning``, ``error``, ``critical``.

Derived max-model-len
---------------------

``--max-model-len`` is computed and emitted **only when** ``serve_args`` does not already set ``max-model-len``:

.. code:: text

  ceil((isl + osl) * (1 + random_range_ratio)) + random_prefix_len + 8

Setting ``max-model-len`` explicitly in ``serve_args`` suppresses the derived value, so the flag never appears twice.

Environment variables: two mechanisms
-------------------------------------

These are separate and are frequently confused.

.. list-table::
   :widths: 2 4 4
   :header-rows: 1

   * -
     - ``container.env``
     - ``roles.server.env``
   * - Applied by
     - ``docker run -e``
     - A sourced shell script inside the container
   * - Scope
     - Every command in the container, for its whole lifetime
     - The ``vllm serve`` processes and the benchmark client
   * - Changing it
     - Requires recreating the container
     - Takes effect on the next run
   * - Defaults
     - ``GPUS=8``, ``MULTINODE=true``
     - See below

The server environment script always exports:

.. code:: bash

  export HF_TOKEN=<token>
  export HF_HUB_CACHE=<paths.models_dir>
  export VLLM_USE_AITER_UNIFIED_ATTENTION=1
  export VLLM_ROCM_USE_AITER_MHA=0
  export VLLM_ROCM_USE_AITER_FUSED_MOE_A16W4=1

then, conditionally, ``NCCL_IB_HCA`` (from ``ib_hca_devices``) and ``NCCL_SOCKET_IFNAME`` / ``GLOO_SOCKET_IFNAME`` / ``TP_SOCKET_IFNAME`` (from ``ib_netdev``). Entries from ``roles.server.env`` are appended **last**, so they override any of the above.

.. _vllm-params:

Parameters
==========

``params`` holds the client knobs and the topology.

.. list-table::
   :widths: 3 2 5
   :header-rows: 1

   * - Key
     - Default
     - Description
   * - ``backend``
     - ``"vllm"``
     - Client backend for ``vllm bench serve``
   * - ``base_url``
     - ``"http://0.0.0.0"``
     - Server base URL
   * - ``port_no``
     - ``"8888"``
     - Server port
   * - ``dataset_name``
     - ``"random"``
     - Dataset for the load generator
   * - ``num_prompts``
     - ``"3200"``
     - Total prompts per cell
   * - ``burstiness``
     - ``"1.0"``
     - 1.0 is a uniform arrival process; lower is burstier
   * - ``seed``
     - ``"0"``
     - Random seed
   * - ``request_rate``
     - ``"inf"``
     - Arrival rate; ``inf`` sends as fast as concurrency allows
   * - ``random_range_ratio``
     - ``"0.8"``
     - Length jitter around ISL/OSL; also feeds the derived max-model-len
   * - ``random_prefix_len``
     - ``"0"``
     - Shared prefix length
   * - ``tensor_parallelism``
     - ``"8"``
     - TP degree
   * - ``pipeline_parallel_size``
     - ``"1"``
     - PP degree; see :ref:`vllm-backends`
   * - ``nnodes``
     - ``"1"``
     - Node count
   * - ``master_addr``
     - ``"localhost"``
     - Head node address for multinode
   * - ``master_port``
     - ``"29501"``
     - Rendezvous port
   * - ``tokenizer_mode``
     - ``"auto"``
     - Tokenizer mode
   * - ``percentile_metrics``
     - ``"ttft,tpot,itl,e2el"``
     - Metric families to compute percentiles for
   * - ``metric_percentiles``
     - ``"50,90,95,99"``
     - Percentiles to emit
   * - ``client_poll_count``
     - ``"20"``
     - Client completion polls before giving up

.. tip::

  ``metric_percentiles`` must emit every percentile your thresholds gate. The default ``"50,90,95,99"`` covers all gated latency metrics. Narrowing it to ``"99"`` makes p50/p90/p95 unavailable, and any threshold on them then fails loudly.

.. _vllm-sweep:

Sweep
=====

The sweep is an explicit list of runs, not a cartesian product. Named sequence combinations are declared once, then referenced by the runs list.

.. code:: json

    {
      "sweep": {
        "sequence_combinations": [
          {
            "name": "balanced",
            "isl": "1000",
            "osl": "1000",
            "goodput_slo": { "ttft_ms": 2000.0, "tpot_ms": 50.0, "e2el_ms": 60000.0 }
          }
        ],
        "runs": [
          { "combo": "balanced", "concurrency": 16 },
          { "combo": "balanced", "concurrency": 32 }
        ]
      }
    }

.. list-table::
   :widths: 3 7
   :header-rows: 1

   * - Key
     - Description
   * - ``sequence_combinations[].name``
     - Unique label; duplicates are rejected
   * - ``sequence_combinations[].isl``
     - Input sequence length
   * - ``sequence_combinations[].osl``
     - Output sequence length
   * - ``sequence_combinations[].goodput_slo``
     - Optional; ``ttft_ms``, ``tpot_ms``, ``e2el_ms``, all required together
   * - ``runs[].combo``
     - Must name a declared combination
   * - ``runs[].concurrency``
     - Integer max concurrency for this cell

A ``combo`` that names no declared combination is a load-time error listing the known names.

When ``goodput_slo`` is set, the client is invoked with ``--goodput ttft:<v> tpot:<v> e2el:<v>`` and ``client.goodput`` becomes meaningful.

Cell keys
---------

Each run is one **cell**, identified by a canonical key used to look up thresholds:

.. code:: text

  Single-node:  ISL=<isl>,OSL=<osl>,TP=<tp>,CONC=<conc>
  Distributed:  ISL=<isl>,OSL=<osl>,TP=<tp>,PP=<pp>,CONC=<conc>

The ``PP=`` segment appears **only** when ``pipeline_parallel_size`` is greater than 1, which keeps single-node keys backward compatible. Examples::

  ISL=1000,OSL=1000,TP=8,CONC=16
  ISL=1000,OSL=1000,TP=8,PP=2,CONC=16

Server reuse
------------

Cells that differ **only** in concurrency share a server identity, so the suite reuses the running server instead of stopping it, restarting, and reloading weights. Changing ISL, OSL, TP, PP, or any server argument forces a restart. Ordering runs so that concurrency varies fastest therefore makes a sweep substantially quicker.

.. _vllm-thresholds:

Thresholds
==========

Thresholds turn measurements into pass/fail results. They are keyed by cell key, then by fully-qualified metric name:

.. code:: json

    {
      "ISL=1000,OSL=1000,TP=8,CONC=16": {
        "client.total_token_throughput": { "kind": "min_tok_s", "value": 4000 },
        "client.mean_ttft_ms":           { "kind": "max_ms",    "value": 500 },
        "client.failed":                 { "kind": "max",       "value": 0 },
        "client.success_rate":           { "kind": "min",       "value": 0.99 },
        "gpu.gpu_compute_util_pct":      { "kind": "within",    "value": 90, "tolerance_pct": 10 },
        "client.output_throughput":      { "kind": "min_ratio", "value": 0.8,
                                           "reference": "client.total_token_throughput" }
      }
    }

Threshold kinds
---------------

.. list-table::
   :widths: 2 2 6
   :header-rows: 1

   * - ``kind``
     - Extra keys
     - Fails when
   * - ``min``
     - —
     - ``actual < value``
   * - ``max``
     - —
     - ``actual > value``. Unit-agnostic upper bound, for counts such as ``failed``
   * - ``max_ms``
     - —
     - ``actual > value``. Identical comparison to ``max``, but the message says "ms"
   * - ``min_tok_s``
     - —
     - ``actual < value``. Identical comparison to ``min``, but the message says "tok/s"
   * - ``within``
     - ``tolerance_pct``
     - ``actual`` falls outside ``value ± tolerance_pct`` percent
   * - ``min_ratio``
     - ``reference``
     - ``actual / <reference metric>`` is less than ``value``

An unrecognized ``kind`` is a violation, not a silent skip. A metric that is missing from the results, or whose value is ``None``, is also a loud violation rather than a pass.

For ``min_ratio``, the ``reference`` names another metric in the same cell. If that reference is missing, ``None``, or zero, the check fails with a message naming the reason.

Coverage checking
-----------------

Threshold files are validated against the sweep on two axes:

1. **Cell coverage** — every sweep cell must have a threshold entry, and no threshold key may name a cell that the sweep does not produce. This catches keys left behind after a sweep edit.
2. **Metric coverage** — every cell must carry a spec for each gated metric.

The ``accuracy`` key is exempt from cell-coverage checking, since it is keyed by task rather than by cell.

Setting ``enforce_thresholds`` to ``false`` downgrades **both** axes to warnings and stops threshold violations from failing tests. The run still measures and records everything, which makes it the right setting for a first calibration run on new hardware.

A metric whose spec is ``null`` is explicitly record-only: it is measured and reported but never asserted.

.. _vllm-threshold-discovery:

Threshold file discovery
------------------------

The threshold file is located in one of two ways:

- **Explicit** — set ``threshold_json`` to a path. A relative path resolves against the configuration file's directory.
- **Implicit** — if ``threshold_json`` is absent, the loader looks for exactly one file matching ``*threshold.json`` beside the configuration file. Finding more than one is an error, so add ``threshold_json`` when several coexist in a directory.

Metrics
=======

Metrics live in namespaces. Each numeric metric becomes one test, and therefore one row in the HTML report. A metric that could not be measured skips rather than failing. Read the measured values from the results table and the per-cell logs; the report's Value and Unit columns are currently disabled.

Client metrics
--------------

Measured by the load generator (``vllm bench serve``) and namespaced ``client.*``. **Gated** metrics require a threshold spec in every cell; the rest are record-only by design.

.. list-table::
   :widths: 4 1 1 4
   :header-rows: 1

   * - Metric
     - Unit
     - Gated
     - Notes
   * - ``client.total_token_throughput``
     - tok/s
     - yes
     - Input plus output tokens per second
   * - ``client.output_throughput``
     - tok/s
     - yes
     - Generated tokens per second
   * - ``client.mean_ttft_ms``
     - ms
     - yes
     - Time to first token
   * - ``client.median_ttft_ms``
     - ms
     - yes
     -
   * - ``client.p90_ttft_ms``
     - ms
     - yes
     -
   * - ``client.p95_ttft_ms``
     - ms
     - yes
     -
   * - ``client.p99_ttft_ms``
     - ms
     - yes
     -
   * - ``client.mean_tpot_ms``
     - ms
     - yes
     - Time per output token
   * - ``client.median_tpot_ms``
     - ms
     - yes
     -
   * - ``client.p90_tpot_ms``
     - ms
     - yes
     -
   * - ``client.p95_tpot_ms``
     - ms
     - yes
     -
   * - ``client.p99_tpot_ms``
     - ms
     - yes
     -
   * - ``client.mean_itl_ms``
     - ms
     - yes
     - Inter-token latency
   * - ``client.median_itl_ms``
     - ms
     - yes
     -
   * - ``client.p95_itl_ms``
     - ms
     - yes
     -
   * - ``client.p99_itl_ms``
     - ms
     - yes
     - ITL has no p90 producer
   * - ``client.mean_e2el_ms``
     - ms
     - yes
     - End-to-end latency
   * - ``client.median_e2el_ms``
     - ms
     - yes
     -
   * - ``client.p90_e2el_ms``
     - ms
     - yes
     -
   * - ``client.p95_e2el_ms``
     - ms
     - yes
     -
   * - ``client.p99_e2el_ms``
     - ms
     - yes
     -
   * - ``client.success_rate``
     - \-
     - yes
     - Derived; see below
   * - ``client.failed``
     - \-
     - yes
     - Failed request count
   * - ``client.max_concurrency``
     - \-
     - no
     -
   * - ``client.max_concurrent_requests``
     - \-
     - no
     -
   * - ``client.num_prompts``
     - \-
     - no
     -
   * - ``client.completed``
     - \-
     - no
     -
   * - ``client.duration``
     - s
     - no
     -
   * - ``client.request_throughput``
     - req/s
     - no
     -
   * - ``client.goodput``
     - req/s
     - no
     - Alias of the stock ``request_goodput``; meaningful only with ``goodput_slo``
   * - ``client.per_gpu_throughput``
     - tok/s
     - no
     - Derived; see below
   * - ``client.decode_throughput_p50``
     - tok/s
     - no
     - Derived; see below
   * - ``client.max_output_tokens_per_s``
     - tok/s
     - no
     -
   * - ``client.total_input_tokens``
     - \-
     - no
     -
   * - ``client.total_output_tokens``
     - \-
     - no
     -
   * - ``client.normalized_ttft_ms_per_tok``
     - ms/tok
     - no
     - Derived; see below
   * - ``client.decode_latency_ratio``
     - \-
     - no
     - Derived; see below

Derived client metrics
~~~~~~~~~~~~~~~~~~~~~~

.. code:: text

  per_gpu_throughput          = total_token_throughput / (tp * pp)
  normalized_ttft_ms_per_tok  = mean_ttft_ms / isl
  decode_latency_ratio        = p99_itl_ms / p50_itl_ms
  decode_throughput_p50       = 1000 / median_tpot_ms
  success_rate                = completed / (completed + failed)

Every division is guarded: a missing, ``None``, or zero divisor yields ``None`` — reported as ``-`` — rather than a bogus zero or a crash.

.. note::

  ``client.request_rate`` is not surfaced as a metric row, because the stock benchmark emits the string ``inf`` rather than a number.

  A new metric is **record-only by default**. It stays informational until its name is explicitly added to the gated set, at which point every cell must supply a spec for it.

GPU metrics
-----------

Sampled from ``amd-smi`` during the run and namespaced ``gpu.*``. None are gated by default.

.. list-table::
   :widths: 4 1 5
   :header-rows: 1

   * - Metric
     - Unit
     - Description
   * - ``gpu.peak_gpu_memory_mb``
     - MB
     - Peak VRAM observed
   * - ``gpu.model_load_memory_mb``
     - MB
     - VRAM attributable to loading weights
   * - ``gpu.model_load_s``
     - s
     - Weight load duration
   * - ``gpu.gpu_bandwidth_util_pct``
     - %
     - Memory bandwidth utilization
   * - ``gpu.gpu_compute_util_pct``
     - %
     - Compute utilization

Server metrics
--------------

Scraped from the vLLM ``/metrics`` Prometheus endpoint and namespaced ``prom.*``.

.. list-table::
   :widths: 4 1 5
   :header-rows: 1

   * - Metric
     - Unit
     - Source histogram
   * - ``prom.queue_time_p50_ms``
     - ms
     - ``vllm:request_queue_time_seconds``
   * - ``prom.queue_time_p95_ms``
     - ms
     - ``vllm:request_queue_time_seconds``
   * - ``prom.prefill_time_p50_ms``
     - ms
     - ``vllm:request_prefill_time_seconds``
   * - ``prom.prefill_time_p95_ms``
     - ms
     - ``vllm:request_prefill_time_seconds``

vLLM's Prometheus counters are cumulative over the server process lifetime, so a raw scrape after cell three would include cells one and two. The suite therefore scrapes **before and after each cell** and diffs the histogram buckets, giving per-cell quantiles. Quantiles are computed with the same interpolation PromQL's ``histogram_quantile`` uses.

If the endpoint cannot be reached, all four report ``-`` and skip rather than failing the run.

Results table
-------------

The summary table emits seven fixed columns — Model, GPU, ISL, OSL, Policy, Conc, Host — followed by Req/s, Total tok/s, Mean TTFT, P95 TTFT, Mean TPOT, P95 TPOT, P99 ITL, and Goodput.

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
            "task": "gsm8k",
            "num_fewshot": 5,
            "num_concurrent": 8,
            "apply_chat_template": false
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
     - Unique label for this entry; duplicates are rejected
   * - ``task``
     - none
     - lm-eval task name
   * - ``num_fewshot``
     - ``0``
     - Few-shot example count
   * - ``num_concurrent``
     - ``8``
     - Concurrent requests
   * - ``apply_chat_template``
     - ``false``
     - Selects the endpoint; see below
   * - ``metadata``
     - ``{}``
     - Passed through to lm-eval
   * - ``include_path``
     - ``""``
     - Directory of custom task definitions
   * - ``gen_kwargs``
     - ``{}``
     - Generation arguments

``apply_chat_template`` selects the API surface:

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

lm-eval is probed for at run time and installed into the container if absent. Each task has a four-hour timeout. Results land under ``<log_dir>/accuracy``.

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

.. list-table::
   :widths: 5 5
   :header-rows: 1

   * - Message
     - Cause and fix
   * - ``nnodes=N > 1 requires pipeline_parallel_size > 1``
     - Multinode on the mp backend needs pipeline parallelism. Either raise ``pipeline_parallel_size``, or set ``distributed-executor-backend`` to ``"ray"``
   * - ``pipeline_parallel_size=N > 1 requires nnodes > 1``
     - Pipeline parallelism spans nodes. Raise ``nnodes`` or reset ``pipeline_parallel_size`` to 1
   * - ``ib_netdev is required in roles.server when nnodes > 1``
     - Set ``roles.server.ib_netdev`` to the interface name. There is no ``"auto"``
   * - ``Container image not specified in config``
     - ``container.image`` is empty. Note that a variant ``container`` block with no ``image`` overwrites the cluster file's value
   * - ``duplicate sequence_combination names``
     - Two entries in ``sequence_combinations`` share a ``name``
   * - ``run.combo names no sequence_combination``
     - A ``runs[].combo`` does not match any declared name; the message lists the valid ones
   * - ``duplicate task id(s)``
     - Two ``accuracy.tasks`` entries share an ``id``
   * - ``<metric>: unknown threshold kind``
     - Typo in ``kind``. Valid values are ``min``, ``max``, ``max_ms``, ``min_tok_s``, ``within``, ``min_ratio``
   * - ``<metric>: missing from actuals``
     - A threshold gates a metric this run did not produce. Common cause: ``metric_percentiles`` omits the gated percentile
   * - ``NotImplementedError: model.remote=1``
     - Remote model download is unimplemented. Pre-stage weights and set ``remote: 0``
   * - ``ValueError: too many values to unpack``
     - ``env`` was placed under ``runtime.args``. Move it to the ``container`` top level
   * - Extra-key validation error
     - A misspelled key. Every block except ``container`` forbids unknown keys

See also
========

- :doc:`/how-to/run-vllm-benchmarks` — step-by-step first run
- :doc:`/reference/configuration-files/cluster-file` — cluster file and orchestrator backends
- :doc:`/how-to/run-with-containers` — container backend walkthrough
- :doc:`/how-to/run-cvs-tests` — running other CVS suites
