.. meta::
  :description: Run vLLM inference benchmarks with CVS, single-node and multinode
  :keywords: CVS, vLLM, inference, benchmark, multinode, ray, LLM, ROCm

*****************************
Run vLLM inference benchmarks
*****************************

The vLLM suites measure LLM serving throughput, latency, and accuracy on AMD Instinct GPUs. ``vllm_single`` runs on the first cluster host and ignores additional hosts; ``vllm_distributed`` runs one service across all hosts, and falls back to single-node behavior on a one-host cluster.

This page walks through a first run. For the full schema, every metric, and the threshold grammar, see :doc:`/reference/configuration-files/inference/vllm`.

Prerequisites
=============

On every cluster node:

- **Docker** installed, with the SSH user able to run it (passwordless ``sudo docker`` or membership in the ``docker`` group).
- **Host driver** loaded, so ``/dev/kfd``, ``/dev/dri/*``, and ``/dev/infiniband/*`` are present for passthrough.
- **A vLLM image** either already loaded or pullable from a reachable registry. It must contain ``vllm`` on the path — the suite invokes ``vllm serve`` and ``vllm bench serve`` inside the container.
- **Model weights** staged on a shared filesystem that every node mounts at the same path. Remote model download is not implemented, so weights must be present before the run.
- **A Hugging Face token file**, if the model needs one. A pre-staged model can run without it.

On the head node where you launch ``cvs run``:

- CVS installed (see :doc:`/getting-started/install`).
- SSH key-based access to every cluster node.

For a multinode run, additionally have on hand:

- The **head node address** reachable from every worker.
- The **network interface name** used for inter-node traffic, for example ``ens51f1np1``. Find it with ``ip -br addr`` on a node. CVS cannot derive this automatically.

.. _vllm-set-up-config:

Step 1: Copy a configuration file
=================================

CVS ships single-node and distributed vLLM configurations. List them:

.. code:: bash

  cvs config list inference/vllm

Copy the one that matches your topology:

.. code:: bash

  # Single node
  cvs config copy inference/vllm/mi3xx_vllm_llama33-70b_fp8_single.json \
    --output /tmp/cvs/vllm_singlenode_config.json

  # Multiple nodes
  cvs config copy inference/vllm/mi3xx_vllm_llama33-70b_fp8_distributed.json \
    --output /tmp/cvs/vllm_multinode_config.json

You also need a cluster file describing your nodes. Use the container template, since the vLLM suite always runs inside a container:

.. code:: bash

  cvs config copy cluster_container.json --output /tmp/cvs/cluster.json

Step 2: Fill in the placeholders
================================

Every value marked ``<changeme>`` must be replaced before the run.

In the **cluster file**, set your SSH user, private key path, and node addresses. See :doc:`/how-to/run-with-containers` for a walkthrough.

In the **configuration file**, set:

- ``container.image`` — your vLLM image. Cite the full tag; do not abbreviate it.
- ``paths.shared_fs`` — the shared filesystem root. The other paths derive from it by default.
- ``paths.models_dir`` — where the weights live. Make sure this path is also mounted into the container by ``container.runtime.args.volumes``.
- ``paths.hf_token_file`` — path to your token file.
- ``model.id`` — the model to serve.

For a **multinode** configuration, also set:

- ``params.master_addr`` — the head node address.
- ``roles.server.ib_netdev`` — the interface name you looked up in the prerequisites.

.. tip::

  Leave ``enforce_thresholds`` set to ``false`` for your first run on new hardware. The run then measures and records everything without failing on thresholds you have not calibrated yet. Set it to ``true`` once you know what good looks like.

.. _vllm-run-tests:

Step 3: Run the suite
=====================

.. code:: bash

  cvs run vllm_single \
    --cluster_file /tmp/cvs/cluster.json \
    --config_file /tmp/cvs/vllm_singlenode_config.json \
    --html /tmp/cvs/vllm.html --self-contained-html \
    --log-file /tmp/cvs/cvs.log

Use ``vllm_distributed`` for one distributed service across a multi-host cluster:

.. code:: bash

  cvs run vllm_distributed \
    --cluster_file /tmp/cvs/cluster.json \
    --config_file /tmp/cvs/vllm_multinode_config.json \
    --html /tmp/cvs/vllm.html --self-contained-html \
    --log-file /tmp/cvs/cvs.log

.. note::

  ``--self-contained-html`` only takes effect together with ``--html``. Always pass both, so the report is a single file you can attach or copy off the cluster.

  Any flag CVS does not recognize is passed straight through to pytest, so options such as ``-vvv`` and ``--capture=tee-sys`` work as usual.

Step 4: Read the results
========================

Open the HTML report. Each lifecycle stage and each metric is its own row:

- **Lifecycle rows** — container launch, topology discovery, model fetch, the OpenAI-compatible smoke test, then teardown. These tell you *how far* the run got.
- **Inference rows** — one per sweep cell, labelled ``<combo>-conc<N>``.
- **Metric rows** — one per metric per cell. A metric that could not be measured is skipped rather than failed.
- **Results table** — the summary near the end, also printed to the console. This is where you read the measured numbers.

Per-cell logs land under your configured ``log_dir``::

  <log_dir>/vllm/out-node<rank>/isl<isl>_osl<osl>_conc<conc>/
    vllm_serve_server.log    <- the server's own log
    client.log               <- the load generator
    results                  <- raw benchmark JSON

.. important::

  When a run fails, read ``vllm_serve_server.log`` on the node, not just the CVS client log. Some faults — GPU exceptions, weight-loading failures, out-of-memory kills — appear only in the server log.

A skipped ``test_setup_sshd`` row is expected. vLLM communicates over the host network and needs no inter-container sshd.

Going multinode
===============

The cluster file determines the host count. For distributed runs, configure ``params.pipeline_parallel_size`` and ``roles.server.ib_netdev``. Which parallelism combination is valid depends on the distributed executor backend.

Using the default backend (mp)
------------------------------

If you set nothing else, the suite uses ``mp``. It requires pipeline parallelism across the nodes:

.. code:: json

    {
      "params": {
        "tensor_parallelism": "8",
        "pipeline_parallel_size": "2",
        "master_addr": "10.0.0.1"
      },
      "roles": {
        "server": {
          "ib_netdev": "ens51f1np1"
        }
      }
    }

CVS launches ``vllm serve`` on every node with the correct rank, adding ``--headless`` to every rank above 0.

Using ray
---------

Ray is opt-in. Add it to ``serve_args``:

.. code:: json

    {
      "roles": {
        "server": {
          "serve_args": {
            "distributed-executor-backend": "ray"
          },
          "ib_netdev": "ens51f1np1"
        }
      },
      "params": {
        "tensor_parallelism": "8",
        "pipeline_parallel_size": "1",
        "master_addr": "10.0.0.1"
      }
    }

CVS then bootstraps a Ray cluster before serving: ``ray start --head`` on rank 0, ``ray start --address=...`` on each worker, and ``ray stop`` at teardown. Only the head node runs ``vllm serve``, so worker nodes produce no server log.

.. note::

  Ray is not required for multinode — ``mp`` is the default and works across nodes. What ray changes is that it **removes the pipeline-parallelism requirement**, so ``pipeline_parallel_size`` of 1 becomes valid. Use ray when you want pure tensor-parallel serving across nodes; use the default otherwise.

  Only the exact lowercase string ``"ray"`` selects it. ``"Ray"`` silently falls back to ``mp`` and then fails validation if ``pipeline_parallel_size`` is 1.

Common pitfalls
===============

**The run fails immediately with a validation error.** Configuration files are validated before anything launches, and every block except ``container`` rejects unknown keys — so a misspelled key is a hard error rather than a silently ignored setting. Read the message: it names the offending key.

**Distributed topology validation fails.** You configured multiple hosts on the default ``mp`` backend without pipeline parallelism. Either raise ``pipeline_parallel_size``, or switch to ray.

**"vllm_distributed requires roles.server.ib_netdev".** Set it to the interface name. There is deliberately no ``"auto"`` value — it cannot be derived reliably from HCA names.

**"Container image not specified in config".** ``container.image`` is empty. Watch for this specific trap: if your configuration file has a ``container`` block that omits ``image``, it overwrites the cluster file's image with an empty string. Set ``image`` in whichever file defines the block.

**Container launch crashes with "too many values to unpack".** You placed ``env`` under ``container.runtime.args``. It belongs at the ``container`` top level.

**A threshold fails with "missing from actuals".** The threshold gates a metric this run did not produce. The usual cause is ``params.metric_percentiles`` omitting a percentile that a threshold references — the default ``"50,90,95,99"`` covers all gated latency metrics.

**Every metric row skips.** The benchmark produced no parseable results. Check ``client.log`` and the server log for the cell.

**The sweep is slower than expected.** Cells that differ only in concurrency reuse the running server; changing ISL, OSL, TP, PP, or any server argument forces a restart and a weight reload. Ordering ``runs`` so concurrency varies fastest avoids needless reloads.

**Using** ``runtime.name: "enroot"``. Enroot is registered but not implemented, and the run fails at container launch. Use ``docker``.

See also
========

- :doc:`/reference/configuration-files/inference/vllm` — full configuration schema, metrics, and thresholds
- :doc:`/reference/cluster/cluster-file` — cluster file schema
- :doc:`/how-to/run-with-containers` — container backend in depth
- :doc:`/how-to/run-tests/index` — running other CVS suites
