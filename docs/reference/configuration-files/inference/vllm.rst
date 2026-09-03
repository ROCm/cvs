.. meta::
  :description: Configure the vLLM inference benchmark suite in CVS
  :keywords: inference, ROCm, cvs, vLLM, LLM, benchmark, multinode, thresholds, metrics, accuracy

**********************************
vLLM inference configuration file
**********************************

The vLLM suites benchmark LLM serving throughput, latency, and accuracy on AMD Instinct GPUs. ``vllm_single`` runs on the first cluster host and ignores additional hosts. ``vllm_distributed`` supports one-host fallback or the current two-host distributed recipes. Larger clusters require an explicit recipe and calibrated thresholds.

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
     - Pull/load the image and start the container on every node
   * - 1
     - ``test_setup_sshd``
     - Always skipped for vLLM (see note below)
   * - 2
     - ``test_discover_topology``
     - Resolve IB HCA devices; no-op for effective single-node execution
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
   * - ``enforce_thresholds``
     - no (default ``true``)
     - When ``false``, metrics record without requiring calibrated threshold cells
   * - ``threshold_json``
     - yes
     - Explicit path to the threshold file. See :ref:`vllm-threshold-discovery`
   * - ``ib_hca_devices`` / ``ib_netdev``
     - no / distributed
     - RDMA HCA selection and distributed socket interface
   * - ``container``
     - yes
     - Container/Docker settings. See :ref:`vllm-container`
   * - ``paths``
     - yes
     - Filesystem locations. See :ref:`vllm-paths`
   * - ``server_params``
     - yes
     - Harness-owned server fields plus snake-case ``vllm serve`` options
   * - ``benchmark_params``
     - no
     - Benchmark defaults plus snake-case ``vllm bench serve`` options
   * - ``sweeps``
     - yes
     - Canonical run-cell keys mapped to benchmark overrides
   * - ``runs``
     - yes
     - Nonempty ordered list of the sweep cells to execute
   * - ``thresholds``
     - no
     - Per-cell pass/fail specs. See :ref:`vllm-thresholds`
   * - ``accuracy``
     - no
     - lm-eval task selection. See :ref:`vllm-accuracy`

.. important::

  Top-level and structural blocks **forbid unknown keys**. ``server_params``,
  ``benchmark_params``, and individual sweep overrides deliberately accept
  arbitrary snake-case vLLM option names. CVS converts them to kebab-case CLI
  flags. ``null`` omits a flag, ``true`` emits a bare flag, scalars emit one
  value, lists emit one flag followed by values, and mappings emit compact JSON.
  Use an option's negative form instead of ``false``.

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
     - The **client** backend passed to ``vllm bench serve --backend``. Nothing to do with distribution
   * - ``server_params.distributed_executor_backend``
     - ``"mp"`` (default), ``"ray"``
     - How vLLM distributes the model across nodes. This is the multinode setting
   * - ``container.runtime.name``
     - ``"docker"``
     - The container runtime
   * - Cluster file ``orchestrator``
     - ``"baremetal"``, ``"container"``
     - Whether CVS runs commands on the host or inside a container. See :doc:`/reference/cluster/cluster-file`

.. warning::

  ``enroot`` is registered but **not implemented**. Every method is a stub that returns failure, so a run with ``runtime.name: "enroot"`` fails at ``test_launch_container``. Podman is not supported. Use ``docker``.

Distributed executor: mp and ray
--------------------------------

Multinode runs support **two** executor backends. ``mp`` is the default and requires no configuration key at all.

**mp (default).** Used whenever ``server_params.distributed_executor_backend`` is absent. The suite injects the full distributed block into each rank's ``vllm serve`` command:

.. code:: bash

  vllm serve <model> --tensor-parallel-size <tp> --port <port> \
    --node-rank <rank> --master-addr <addr> --master-port <port> \
    --nnodes <n> --pipeline-parallel-size <pp> \
    --distributed-executor-backend mp

Every rank above 0 additionally gets ``--headless``. This path **requires pipeline parallelism** (``pipeline_parallel_size`` greater than 1).

**ray (opt-in).** Selected by setting ``server_params.distributed_executor_backend`` to the exact lowercase string ``"ray"``. Other values are configuration errors. Ray takes a completely different route:

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
   * - exactly two cluster hosts, backend is not ray
     - ``pipeline_parallel_size`` **must** be greater than 1
   * - exactly two cluster hosts, backend is ray
     - ``pipeline_parallel_size`` of 1 is valid
   * - ``pipeline_parallel_size`` > 1
     - The distributed suite requires more than one cluster host
   * - exactly two cluster hosts, either backend
     - Top-level ``ib_netdev`` is **required**; larger clusters are rejected until a dedicated recipe exists

The corresponding error messages are:

.. code:: text

  multi-host distributed execution requires pipeline_parallel_size > 1 unless using ray
  pipeline_parallel_size > 1 requires a multi-host distributed suite
  ib_netdev is required for multi-host distributed execution. Set it to the Linux
  network interface name for NCCL_SOCKET_IFNAME (e.g. "ens51f1np1"). Cannot be
  auto-derived from HCA names.

Multinode prerequisites
-----------------------

Beyond the validation rules, a multinode run needs:

- ``server_params.dist_init_port`` — default ``29501``; CVS derives the head address from the cluster.
- Top-level ``ib_netdev`` — the Linux interface name. There is deliberately no ``"auto"`` value; it cannot be derived reliably from HCA names. This value populates ``NCCL_SOCKET_IFNAME``, ``GLOO_SOCKET_IFNAME``, and ``TP_SOCKET_IFNAME``.
- Top-level ``ib_hca_devices`` — ``"auto"``, an explicit list, or ``null``. When set, populates ``NCCL_IB_HCA``.

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

If ``hf_token_file`` does not exist, the run continues with an empty token. vLLM
configs always serve a pre-staged model mounted under ``paths.models_dir``.

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
   * - ``server_params.model``
     - none
     - Local path, for example ``/models/Llama-3.1-70B-Instruct-FP8-KV``

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
     - Local model path supplied as the positional ``vllm serve`` argument
   * - ``tensor_parallel_size``
     - none
     - Tensor-parallel degree
   * - ``pipeline_parallel_size``
     - ``1``
     - Pipeline-parallel degree
   * - ``port``
     - ``8888``
     - OpenAI-compatible server port

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
     - ``"kv_cache_dtype": "fp8"`` → ``--kv-cache-dtype fp8``
   * - ``true``
     - ``--flag`` (bare)
     - ``"enforce_eager": true`` → ``--enforce-eager``
   * - ``false``
     - rejected
     - Omit the setting or use vLLM's explicit negative option
   * - List
     - one flag followed by its values
     - ``"x": ["a","b"]`` → ``--x a b``

CVS does not derive ``max_model_len``. Set it in ``server_params`` whenever the
image or model needs an explicit context limit.

Environment variables: two mechanisms
-------------------------------------

These are separate and are frequently confused.

.. list-table::
   :widths: 2 4 4
   :header-rows: 1

   * -
     - ``container.env``
     - generated process environment
   * - Applied by
     - ``docker run -e``
     - A sourced shell script inside the container after HCA discovery
   * - Scope
     - Every command in the container, for its whole lifetime
     - NCCL/Gloo/TP network variables plus Hugging Face path/token variables
   * - Changing it
     - Requires recreating the container
     - Takes effect on the next command
   * - Defaults
     - ``GPUS=8``, ``MULTINODE=true``
     - HCA and netdev selections

The server environment script always exports:

.. code:: bash

  export HF_TOKEN=<token>
  export HF_HUB_CACHE=<paths.models_dir>
then, conditionally, ``NCCL_IB_HCA`` (from top-level ``ib_hca_devices``) and
``NCCL_SOCKET_IFNAME`` / ``GLOO_SOCKET_IFNAME`` / ``TP_SOCKET_IFNAME`` (from
top-level ``ib_netdev``). Put static ROCm, NCCL, and vLLM exports in
``container.env``; it may not override those generated network variables.

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
     - Client backend for ``vllm bench serve``
   * - ``base_url``
     - ``"http://0.0.0.0"``
     - Server base URL
   * - ``num_prompts``
     - ``3200``
     - Total prompts per cell
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
   * - ``tokenizer_mode``
     - ``"auto"``
     - Tokenizer mode
   * - ``client_poll_iterations``
     - ``20``
     - Client completion polls before giving up

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
      }
    }

.. list-table::
   :widths: 3 7
   :header-rows: 1

   * - Key
     - Description
   * - ``sweeps.<cell>``
     - Per-cell benchmark override object
   * - ``runs[]``
     - Canonical key declared in ``sweeps``

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

.. _vllm-threshold-coverage:

Coverage checking
-----------------

At load time, every threshold cell must name a key in ``sweeps``. When
``enforce_thresholds`` is true, every selected ``runs`` cell must have a
threshold entry. Record-only runs may select uncalibrated cells.

There is no per-metric coverage requirement. A cell's entry may spec a single metric or two dozen — a threshold file is free to gate only the metrics you care about rather than every member of every family.

The ``accuracy`` key is exempt from cell-coverage checking, since it is keyed by task rather than by cell.

Setting ``enforce_thresholds`` to ``false`` stops threshold violations from
failing tests. The run still measures and records everything, which makes it
the right setting for a first calibration run on new hardware.

Which metrics are asserted is then decided per metric at evaluation time, not at load time. A metric is checked only when its cell carries a spec for it; with no spec it is measured and reported but never asserted. A spec of ``null`` is the explicit way to say the same thing.

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

Measured by the load generator (``vllm bench serve``) and namespaced ``client.*``. **Gated** marks the metrics designated as pass/fail criteria: they populate the report's gate matrix and they are the ones a threshold file normally specs. The mark does not make a metric mandatory — any metric, gated or not, is asserted only when its cell carries a spec for it (see :ref:`vllm-threshold-coverage`).

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

  A new metric is **record-only by default**. Adding its name to the gated set marks it as a pass/fail criterion and files it under one of the report's gate-matrix tiers; it still only fails a run in those cells whose threshold entry specs it.

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
     - Unique label for this entry; duplicates are rejected
   * - ``tasks`` (or legacy ``task``)
     - none
     - lm-eval task name or task list
   * - ``lm_eval_model``
     - endpoint-derived
     - ``local-completions`` or ``local-chat-completions``
   * - ``num_fewshot``
     - lm-eval default
     - Few-shot example count, if explicitly set
   * - ``num_concurrent``
     - ``8``
     - Concurrent requests
   * - ``apply_chat_template``
     - ``false``
     - Enables or names the chat template
   * - ``metadata``
     - ``{}``
     - Passed through to lm-eval
   * - ``include_path``
     - ``""``
     - Directory of custom task definitions
   * - ``gen_kwargs``
     - ``{}``
     - Generation arguments

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
   * - Distributed execution requires ``pipeline_parallel_size > 1``
     - Multi-host ``vllm_distributed`` on the mp backend needs pipeline parallelism. Either raise ``pipeline_parallel_size``, or set ``distributed-executor-backend`` to ``"ray"``
   * - ``vllm_single requires pipeline_parallel_size=1``
     - Use ``vllm_distributed`` when the config requires pipeline parallelism
   * - ``vllm_distributed requires roles.server.ib_netdev``
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

- :doc:`/how-to/test-suites/inference/vllm` — step-by-step first run
- :doc:`/reference/cluster/cluster-file` — cluster file and orchestrator backends
- :doc:`/how-to/run-with-containers` — container backend walkthrough
- :doc:`/how-to/run-tests/index` — running other CVS suites
