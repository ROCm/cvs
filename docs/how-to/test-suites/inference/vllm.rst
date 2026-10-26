.. meta::
  :description: Run vLLM LLM inference tests with CVS on AMD Instinct GPUs, covering single-node and multinode distributed serving with ROCm and InfiniBand.
  :keywords: CVS, vLLM, inference, benchmark, AMD Instinct, ROCm, AMD, GPU, LLM, multinode, InfiniBand, RDMA

***************************************
Run vLLM LLM inference tests with Cluster Validation Suite (CVS)
***************************************

The vLLM suites measure LLM serving throughput, latency, and accuracy on AMD Instinct GPUs. ``vllm_single`` runs on the first cluster host and ignores additional hosts. ``vllm_distributed`` uses every host in the cluster file. One host does not raise an error: the run becomes a single-node serve, and the multi-host flags are left off. Packaged distributed recipes are laid out for two hosts (tensor parallel 8, pipeline parallel 2). Their threshold numbers are zeros, not measured pass/fail limits.

For a mapping-style ``node_dict``, "first cluster host" is the first JSON key
in insertion order. ``vllm_single`` rewrites ``head_node_dict.mgmt_ip`` to that
host, so put the intended single-node target first in ``node_dict``.

This page walks through a first run of the packaged Llama 3.3 70B FP8 files. For the full schema, every metric, and the threshold grammar, see :doc:`/reference/configuration-files/inference/vllm`.

Prerequisites
=============

In addition to the :ref:`common inference prerequisites <inference-common-prerequisites>`:

- The container image must provide ``vllm`` on ``PATH``. The suite runs
  ``vllm serve`` and ``vllm bench serve`` inside the container.
- Stage the model weights on the host before the run. The suite does not download models.

For a multinode run, also identify:

- The head-node address reachable from every worker. CVS takes this from the cluster head.
- The HCA names on every node (``ibv_devinfo -l`` inside the image, or ``/sys/class/infiniband``).
- The Linux netdev for ``NCCL_SOCKET_IFNAME``, and the frontend interfaces for ``GLOO_SOCKET_IFNAME`` and ``TP_SOCKET_IFNAME``.
  Find netdevs with ``ip -br addr`` on a node.

.. _vllm-set-up-config:

Run vLLM tests
==============

Copy a config template, fill in your cluster details and model paths, then
run the suite. CVS produces a pytest HTML report with per-cell metric rows
and threshold verdicts.

Copy a configuration file
--------------------------

List the shipped vLLM configs to find the template for your GPU and topology:

.. code:: bash

  cvs config list inference/vllm

Copy the Llama 3.3 70B FP8 pair that matches your topology. ``threshold_json``
is a filename relative to the config, so put the threshold file in the same
directory if you want to edit it. If that sibling is missing, CVS warns and
uses the packaged copy.

.. code:: bash

  # Single node
  cvs config copy inference/vllm/mi3xx_vllm_llama33-70b_fp8_single.json \
    --output /tmp/cvs/mi3xx_vllm_llama33-70b_fp8_single.json
  cvs config copy inference/vllm/mi325x_vllm_llama33-70b_fp8_single_threshold.json \
    --output /tmp/cvs/mi325x_vllm_llama33-70b_fp8_single_threshold.json

  # Two-host layout
  cvs config copy inference/vllm/mi3xx_vllm_llama33-70b_fp8_distributed.json \
    --output /tmp/cvs/mi3xx_vllm_llama33-70b_fp8_distributed.json
  cvs config copy inference/vllm/mi325x_vllm_llama33-70b_fp8_distributed_threshold.json \
    --output /tmp/cvs/mi325x_vllm_llama33-70b_fp8_distributed_threshold.json

Copy a cluster file. Use the container template, since the vLLM suite always
runs inside a container:

.. code:: bash

  cvs config copy cluster_container.json --output /tmp/cvs/cluster.json

Fill in the placeholders
-------------------------

Replace every ``<changeme>`` before the run. ``{user-id}`` is not a
``<changeme>``: it substitutes to the login name of the ``cvs run`` process
unless the cluster file sets ``username``.

In the **cluster file**, set your SSH user, private key path, and node
addresses. Unresolved ``<changeme>`` strings in the cluster file stop
startup. The vLLM loader does not apply the same hard-exit to the workload
file, except when ``threshold_json`` itself still contains ``<changeme>``.
An unreplaced image name or mount fails later, at pull or ``docker run``.
See :doc:`/how-to/run-with-containers` for the cluster-file walkthrough.

In the **Llama single-node file**, replace these placeholders:

.. list-table::
   :header-rows: 1
   :widths: 4 6

   * - Placeholder
     - What to put there
   * - ``container.image``
     - Full image reference that provides ``vllm``. The file ships
       ``<changeme>``, not a tag.
   * - ``server_params.model``
     - Model path inside the container. With the packaged mount, this is a
       path under ``/models``.
   * - ``<changeme-models-mount>``
     - Host directory that holds the weights. It is mounted read-only at
       ``/models``.
   * - ``<changeme-libionic-soname>``
     - Filename from ``readlink -f /usr/lib/x86_64-linux-gnu/libionic.so.1``
       on the nodes. Delete the Ionic volume lines if the nodes have no Ionic
       stack. Delete the ``libbnxt_re-rdmav34.so`` line if they have no
       Broadcom Thor2 library. ``docker run`` fails when a listed host path
       is missing.

Leave these paths as shipped unless you are changing the layout:

.. list-table::
   :header-rows: 1
   :widths: 3 7

   * - Key
     - Packaged meaning
   * - ``paths.shared_fs``
     - ``/home/{user-id}`` on the nodes. Not a ``<changeme>``.
   * - ``paths.models_dir``
     - ``/models``, the in-container Hugging Face cache. CVS exports it as
       ``HF_HUB_CACHE`` and checks that the directory contains bytes. This
       is not the host weight path.
   * - ``paths.hf_token_file``
     - ``/home/{user-id}/.hf_token`` after substitution. CVS does not read
       this file on the machine running ``cvs run``. Each node reads it
       inside the container when the server env script is sourced. The home
       volume makes the host file visible at the same path. A missing or
       empty file continues the run with an empty token.
   * - ``paths.log_dir``
     - ``/home/{user-id}/LOGS``. The same home volume bind-mounts it into
       the container. ``lifetime`` is ``per_run``, so teardown runs
       ``docker rm -f`` and deletes the container writable layer. Files
       under this host path remain.

The single-node file also runs three lm-eval tasks (``gsm8k_flex``,
``gsm8k_strict``, ``gsm8k_stderr``) after the sweep. Remove the
``accuracy`` block to skip accuracy evaluation on the first run.

For the **distributed** file, replace the same image, model, models mount,
and NIC lines, and also replace:

- ``container.env.NCCL_SOCKET_IFNAME``, ``GLOO_SOCKET_IFNAME``,
  ``TP_SOCKET_IFNAME``, and ``NCCL_IB_GID_INDEX`` (each is ``<changeme>``).
- ``container.env.NCCL_IB_HCA`` if the nodes do not expose ``rdma0``
  through ``rdma7``. This value is not a ``<changeme>``; a wrong name fails
  HCA preflight.

.. _vllm-thresholds-first-run:

Threshold file
~~~~~~~~~~~~~~

Both Llama threshold files set every ``value`` to ``0`` — an uncalibrated
placeholder, including ``max`` limits. The paired configs set
``enforce_thresholds`` to ``false``, so those zeros are not applied.

The single-node sibling has two cells
(``ISL=1024,OSL=1024,TP=8,PP=1,CONC=16`` and
``ISL=1024,OSL=1024,TP=8,PP=1,CONC=32``). The distributed sibling has one
cell (``ISL=1024,OSL=1024,TP=8,PP=2,CONC=16``). Each cell lists the same
32 metrics. Copy the sibling beside the config before editing it. Replace
the zeros with measured limits before setting ``enforce_thresholds`` to
``true``.

.. _vllm-run-tests:

Run the suite
--------------

.. code:: bash

  cvs run vllm_single \
    --cluster_file /tmp/cvs/cluster.json \
    --config_file /tmp/cvs/mi3xx_vllm_llama33-70b_fp8_single.json

Use ``vllm_distributed`` to run one service across the cluster file:

.. code:: bash

  cvs run vllm_distributed \
    --cluster_file /tmp/cvs/cluster.json \
    --config_file /tmp/cvs/mi3xx_vllm_llama33-70b_fp8_distributed.json

.. note::

  ``cvs run`` writes the pytest HTML report described in
  :ref:`pytest-html-report-and-bundle`. Any flag CVS does not recognize is
  passed straight through to pytest, so options such as ``-vvv`` and
  ``--capture=tee-sys`` work as usual.

Run on managed compute
~~~~~~~~~~~~~~~~~~~~~~

Follow :doc:`/how-to/run-on-managed-compute` to run vLLM as a Slurm or
SPUR job step.

Read the results
-----------------

Open the HTML report. Each lifecycle stage and benchmark cell is its own
row. The row names correspond to the test functions:

.. list-table::
   :header-rows: 1
   :widths: 4 6

   * - Row
     - What it covers
   * - ``test_launch_container``
     - Start the container.
   * - ``test_setup_sshd``
     - Always skipped. vLLM uses the host network and does not need an
       inter-container sshd.
   * - ``test_discover_topology``
     - HCA preflight. A no-op when the effective topology is one host.
   * - ``test_model_fetch``
     - Verify that ``paths.models_dir`` (``/models`` in the packaged file)
       contains bytes inside the container.
   * - ``test_openai_compatible_smoke``
     - Short-lived server; the OpenAI-compatible API must respond.
   * - ``test_vllm_inference[<cell>]``
     - One benchmark cell. ``<cell>`` is the canonical key, for example
       ``ISL=1024,OSL=1024,TP=8,PP=1,CONC=16``.
   * - ``test_accuracy_eval[<task id>]``
     - One lm-eval task, if ``accuracy`` is configured.
   * - ``test_print_results_table``
     - Console summary.
   * - ``test_teardown``
     - Stop the container.

Expand a benchmark row to see the metric table. With ``enforce_thresholds``
false, every recorded metric's status is set to ``pass`` and the HTML
Result column shows **Passed**. No threshold subtest is emitted. The Note
column shows one of:

- ``metric produced without a configured threshold`` — the cell has no spec
  for that metric.
- ``threshold enforcement disabled; no threshold asserted`` — a spec exists
  and the actual value is finite.
- ``<metric>: unavailable or non-finite actual recorded; no threshold
  asserted`` — a spec exists but the actual value is missing or not finite.

A configured gate appears as ``reference: min 0`` or ``reference: max 0``,
not as a failure. With enforcement on, each configured spec becomes one
subtest, and a missed threshold fails that cell's row without skipping
later cells.

The console summary columns are: Model, GPU, ISL, OSL, Policy, Conc, Host,
Req/s, Total tok/s, Mean TTFT (ms), P95 TTFT (ms), Mean TPOT (ms), P95
TPOT (ms), P99 ITL (ms), and Goodput (req/s). The GPU column is blank. The
Policy column is the full cell key.

Per-cell logs land under ``paths.log_dir``. For the packaged file that is
``/home/<you>/LOGS`` on the host, which is why they persist after
``docker rm -f``::

  <log_dir>/vllm/out-node<rank>/isl<isl>_osl<osl>_conc<conc>/
    vllm_serve_server.log
    client.log
    results

.. important::

  When a run fails, read ``vllm_serve_server.log`` on the node, not just
  the CVS client log. GPU exceptions, weight-loading failures, and
  out-of-memory kills appear only in the server log.

Packaged time budget
~~~~~~~~~~~~~~~~~~~~

These sleeps are in addition to model load and the benchmark itself. The
server path always waits 30 seconds for the log to appear
(``_precheck_wait``), then the warmup, then one sleep after every
unreadiness poll including the last one.

.. list-table::
   :header-rows: 1
   :widths: 3 4 5

   * - File
     - Server sleeps before timeout
     - Client sleeps before timeout
   * - Llama single
     - 30 s + ``server_warmup_wait_s`` 330 + 60 × ``server_poll_wait_s``
       60 = 3960 s
     - ``client_initial_wait_s`` 120 + 90 × ``client_poll_wait_s`` 60
       = 5520 s
   * - Llama distributed
     - 30 s + warmup 30 + 360 × 10 s = 3660 s
     - initial 10 s + 540 × 10 s = 5410 s

Timeout errors:

- ``vllm server did not become ready before timeout``
- ``client did not complete before poll cap``

A cell also stops when failed requests exceed 1% of completed plus failed
requests. The error starts with ``client failed:`` and includes
``Failed requests: <n>/<total> (<percent>)``. At or below 1%, the cell
continues and logs a warning.

Going multinode
===============

``vllm_distributed`` with one host calls ``resolve_vllm_topology`` and gets an effective ``single`` topology. That path does not raise. ``vllm serve`` is the single-node command: no ``--node-rank``, ``--master-addr``, ``--master-port``, ``--nnodes``, or ``--distributed-executor-backend``. ``--pipeline-parallel-size`` is added only when pipeline parallel size is greater than 1. Ray is not started.

Two or more hosts on the default ``mp`` backend require pipeline parallel size greater than 1. The JSON key is ``server_params.distributed_executor_backend`` (snake case). The exact error text still uses a hyphen:

.. code:: text

  vllm_distributed requires pipeline_parallel_size>1 unless distributed-executor-backend=ray

Set the value to the lowercase string ``"ray"`` to take the Ray path, or raise ``pipeline_parallel_size``. Other strings are rejected when the config loads.

Multi-host runs also require a socket interface. The exact error is:

.. code:: text

  vllm_distributed requires container.env.NCCL_SOCKET_IFNAME on multi-host clusters

A legacy top-level ``ib_netdev`` also satisfies that check. Set ``GLOO_SOCKET_IFNAME`` and ``TP_SOCKET_IFNAME`` as well: if you set any one of the three, the config load requires all three.

HCA preflight
-------------

``test_discover_topology`` runs only when the effective host list has two or more hosts. It does not run for ``vllm_single`` or for one-host ``vllm_distributed``.

On a multi-host run it checks:

- Each node has a non-empty HCA list, from ``ibv_devinfo -l``, or from ``/sys/class/infiniband`` when that command is missing or empty.
- The discovered name lists match across nodes.
- Every name in ``container.env.NCCL_IB_HCA`` (or a legacy ``ib_hca_devices`` list) is present on every node. A miss fails with ``ib_discovery preflight: requested HCA devices <missing> not found on <host>. Available: <devs>``.

It does not check that the socket-interface names exist, that ``NCCL_IB_GID_INDEX`` is valid, or that the links are up. When ``NCCL_IB_HCA`` is set, preflight does not rewrite it; the container environment keeps the value you wrote.

Using the default backend (mp)
------------------------------

If you set nothing else, the suite uses ``mp``. Two or more hosts require pipeline parallelism. The JSON key stays snake case:

.. code:: json

    {
      "container": {
        "env": {
          "NCCL_IB_HCA": "rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7",
          "NCCL_SOCKET_IFNAME": "eno0",
          "GLOO_SOCKET_IFNAME": "eno0",
          "TP_SOCKET_IFNAME": "eno0",
          "NCCL_IB_GID_INDEX": "3"
        }
      },
      "server_params": {
        "tensor_parallel_size": 8,
        "pipeline_parallel_size": 2,
        "distributed_executor_backend": "mp"
      }
    }

CVS launches ``vllm serve`` on every node with the correct rank, adding ``--headless`` to every rank above 0. The client base URL is always ``http://0.0.0.0`` plus ``server_params.port`` (8888 in the packaged file). You cannot set ``base_url`` in the config; it is a reserved benchmark key.

Using ray
---------

Ray is opt-in. Set the snake_case key ``distributed_executor_backend`` to ``"ray"``:

.. code:: json

    {
      "server_params": {
        "tensor_parallel_size": 8,
        "pipeline_parallel_size": 1,
        "distributed_executor_backend": "ray"
      }
    }

CVS then bootstraps a Ray cluster before serving: ``ray start --head --port=<dist_init_port>`` on rank 0, ``ray start --address=<head>:<dist_init_port>`` on each worker, and ``ray stop`` at teardown. Only the head node runs ``vllm serve``, so worker nodes produce no server log. ``--pipeline-parallel-size`` is added only when pipeline parallel size is greater than 1.

.. note::

  Ray is not required for multinode. ``mp`` is the default and works across nodes when pipeline parallel size is greater than 1. Ray removes that requirement, so pipeline parallel size 1 is valid. Use ray for pure tensor-parallel serving across nodes.

  Only the exact lowercase string ``"ray"`` selects it.

Troubleshooting
===============

Validation error on startup
----------------------------

``container``, ``paths``, and the top-level config reject unknown keys.
``server_params``, ``benchmark_params``, and sweep overrides accept extra
snake_case vLLM options. Keys such as ``image_tar``, ``registry``,
``shm_size``, and ``cap_add`` belong on the cluster file, not in the vLLM
config. See :doc:`/reference/cluster/cluster-file`.

Distributed topology validation fails
--------------------------------------

.. code-block:: text

  vllm_distributed requires pipeline_parallel_size>1 unless distributed-executor-backend=ray

You have two or more hosts on ``mp`` without pipeline parallelism. Raise
``pipeline_parallel_size``, or set ``distributed_executor_backend`` to
``"ray"``.

Missing NCCL socket interface
------------------------------

.. code-block:: text

  vllm_distributed requires container.env.NCCL_SOCKET_IFNAME on multi-host clusters

Set the socket interface. If you set any of ``NCCL_SOCKET_IFNAME``,
``GLOO_SOCKET_IFNAME``, or ``TP_SOCKET_IFNAME``, set all three.

Container image not specified
------------------------------

.. code-block:: text

  Container image not specified in config

``container.image`` is empty. The orchestrator logs this, then
``test_launch_container`` fails because ``setup_containers()`` returned
false. The vLLM schema requires ``image``, so omitting the key fails
validation instead. An empty string overwrites a cluster-file image. The
packaged Llama files ship ``<changeme>``—not a real image until you
replace it.

``env`` placed under ``container.runtime.args``
------------------------------------------------

``RuntimeArgs`` rejects unknown keys, so ``env`` there fails config load.
Move ``env`` to ``container``, not ``container.runtime.args``.

Threshold fails with non-finite actual
---------------------------------------

.. code-block:: text

  actual must be a finite built-in int or float

The gated datasource did not produce a valid value for that metric. Inspect
the raw benchmark artifact, GPU telemetry, or server metrics for that cell.
The suite owns benchmark percentile collection; workload configuration
cannot override it.

Sweep is slower than expected
------------------------------

Cells that differ only in concurrency reuse the running server. Changing
ISL, OSL, TP, PP, or any server argument forces a restart and a weight
reload. Order ``runs`` so concurrency varies fastest to avoid needless
reloads. The packaged single-node file already does this (concurrency 16,
then 32).

``runtime.name`` is not ``docker``
------------------------------------

The vLLM schema accepts only ``"docker"`` as the runtime name.

Related resources
=================

- :doc:`/reference/configuration-files/inference/vllm` — full configuration schema, metrics, and thresholds
- :doc:`/reference/cluster/cluster-file` — cluster file schema and orchestrator container keys
- :doc:`/how-to/run-with-containers` — container backend in depth
- :doc:`/how-to/test-suites/index` — running other CVS suites
