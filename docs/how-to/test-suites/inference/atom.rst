.. meta::
  :description: Run ATOM LLM inference benchmarks with CVS on AMD Instinct GPUs for single-node and multinode pipeline-parallel deployments using ROCm.
  :keywords: CVS, ATOM, inference, benchmark, AMD Instinct, ROCm, AMD, GPU, LLM, multinode, pipeline parallel, DeepSeek

******************************************
Run ATOM LLM inference benchmarks with CVS
******************************************

The ATOM suite measures LLM serving throughput, latency, and accuracy on AMD Instinct GPUs. One suite name, ``atom``, covers native ATOM, vLLM-ATOM, and SGLang. The driver comes from the config. This page walks through a first run of the packaged DeepSeek-R1 FP8 single-node files. For the schema, shipped models, and threshold grammar, see :doc:`/reference/configuration-files/inference/atom`.

Prerequisites
=============

In addition to the :ref:`common inference prerequisites <inference-common-prerequisites>`:

- The container image must provide the serving stack selected by the config. The packaged single-node file uses native ATOM (``params.driver`` ``atom``).
- Stage the model weights on the host before the run. The packaged file sets ``model.remote`` to ``0`` and checks that ``paths.models_dir`` contains bytes inside the container. It does not download the model.

For a multinode run, also:

- Keep the number of hosts in the cluster file equal to ``params.nnodes`` (native profiles) or ``server_params.nnodes`` (serving-schema files).
- Set ``master_addr`` to an address reachable from every participating node.
- Mount ``paths.log_dir`` on every node. The packaged files put it under ``paths.shared_fs``.

.. _atom-set-up-config:

Step 1: Copy a configuration file
=================================

CVS ships single-node and distributed ATOM configurations. List them:

.. code:: bash

  cvs config list inference/atom

Copy the DeepSeek-R1 FP8 single-node pair. ``threshold_json`` is a filename relative to the config, so put the threshold file in the same directory. The config stem is ``mi3xx_*`` and the sibling threshold file is ``mi325x_*``. If that sibling is missing, CVS warns and uses the packaged copy.

.. code:: bash

  cvs config copy inference/atom/mi3xx_atom_deepseek-r1_fp8_single.json \
    --output /tmp/cvs/mi3xx_atom_deepseek-r1_fp8_single.json
  cvs config copy inference/atom/mi325x_atom_deepseek-r1_fp8_single_threshold.json \
    --output /tmp/cvs/mi325x_atom_deepseek-r1_fp8_single_threshold.json

Copy one pair into a directory. A directory with more than one ``*threshold.json``, and a config that does not name ``threshold_json``, fails to load.

You also need a cluster file. The ATOM suite runs inside a container:

.. code:: bash

  cvs config copy cluster_container.json --output /tmp/cvs/cluster.json

Step 2: Fill in the placeholders
================================

Replace every ``<changeme>`` before the run. ``{user-id}`` is not one of those: it becomes the login name of the ``cvs run`` process, unless the cluster file sets ``username``.

In the **cluster file**, set your SSH user, private key path, and node addresses. For this single-node file, leave one host in ``node_dict``. Unresolved ``<changeme>`` strings in the cluster file stop startup. See :doc:`/how-to/run-with-containers`.

The config is ``schema_version`` ``2``. ``default_profile`` is ``perf``. The same placeholders appear under ``profiles.perf`` and ``profiles.mtp3``. Replace every occurrence.

.. list-table::
   :header-rows: 1
   :widths: 4 6

   * - Placeholder
     - What to put there
   * - ``container.image``
     - Full image reference that provides native ATOM. The file ships ``<changeme>``, not a tag.
   * - ``<changeme-models-mount>``
     - Host directory that holds the weights. It is mounted read-only at ``/models``.

Leave these as they are for the first run:

.. list-table::
   :header-rows: 1
   :widths: 3 7

   * - Key
     - Packaged meaning
   * - ``model.id``
     - ``deepseek-ai/DeepSeek-R1-0528``. ``model.remote`` is ``0``.
   * - ``paths.shared_fs``
     - ``/home/{user-id}`` on the nodes.
   * - ``paths.models_dir``
     - ``/models``, the in-container weight path. CVS checks that the directory contains bytes.
   * - ``paths.hf_token_file``
     - ``/home/{user-id}/.hf_token`` after substitution. The home volume makes the host file visible at the same path inside the container.
   * - ``paths.log_dir``
     - ``/home/{user-id}/LOGS``. ``lifetime`` is ``per_run``, so teardown removes the container. Files under this host path remain.
   * - ``params.driver``
     - ``atom`` on both profiles.

The ``perf`` profile also runs the lm-eval tasks in ``accuracy.tasks`` and a quant-parity probe (``quant_parity.enabled`` is ``true``). Remove the ``accuracy`` block, or set ``quant_parity.enabled`` to ``false``, to skip those rows on a first run.

.. _atom-thresholds-first-run:

Threshold file
--------------

Both profiles set ``enforce_thresholds`` to ``false``, so the sibling numbers are not applied. The threshold file has a ``profiles.perf`` block and a ``profiles.mtp3`` block. ``--config_profile`` selects the matching block. Each block lists these cells:

- ``ISL=1024,OSL=1024,TP=8,PP=1,CONC=128``
- ``ISL=1024,OSL=1024,TP=8,PP=1,CONC=256``
- ``ISL=128,OSL=32,TP=8,PP=1,CONC=1``

``runs`` is empty, so the suite runs every cell key under ``sweeps``. Copy the sibling beside the config before you edit it. Replace the specs with measured limits before you set ``enforce_thresholds`` to ``true``.

.. _atom-run-tests:

Step 3: Run the suite
=====================

``perf`` is the default profile:

.. code:: bash

  cvs run atom \
    --cluster_file /tmp/cvs/cluster.json \
    --config_file /tmp/cvs/mi3xx_atom_deepseek-r1_fp8_single.json

The ``mtp3`` profile adds speculative decode (``--method mtp`` and ``--num-speculative-tokens 3`` in ``roles.server.atom_args``) and runs MTP quality checks:

.. code:: bash

  cvs run atom \
    --cluster_file /tmp/cvs/cluster.json \
    --config_file /tmp/cvs/mi3xx_atom_deepseek-r1_fp8_single.json \
    --config_profile mtp3

.. note::

  ``cvs run`` writes the pytest HTML report described in :ref:`pytest-html-report-and-bundle`.

  Any flag CVS does not recognize is passed straight through to pytest, so options such as ``-vvv`` and ``--capture=tee-sys`` work as usual.

vLLM-ATOM and SGLang stems are still ``cvs run atom``. See :ref:`atom-other-files`.

Run on managed compute
----------------------

Follow :doc:`/how-to/run-on-managed-compute` to run ATOM on managed compute.

ATOM adds these requirements:

- Request one node for the packaged single-node file. A distributed file needs the same host count as ``server_params.nnodes`` (or ``params.nnodes`` on a native profile).
- Replace ``{head-node-ip}`` in ``master_addr`` with an address every node can reach.
- Keep the config, its sibling threshold file, and the Hugging Face token file on storage every node sees at the same path. The models volume must exist on each GPU node.

Step 4: Read the results
========================

Open the HTML report. Each lifecycle stage, benchmark cell, and metric tier is its own row. For the packaged ``perf`` profile the rows are:

.. list-table::
   :header-rows: 1
   :widths: 4 6

   * - Row
     - What it is
   * - ``test_launch_container``
     - Start the container.
   * - ``test_setup_sshd``
     - Passes without starting sshd on one host, and on a managed-compute step.
   * - ``test_discover_topology``
     - No-op when ``params.nnodes`` is ``1``.
   * - ``test_model_fetch``
     - Check that ``paths.models_dir`` (``/models`` in the packaged file) has bytes inside the container.
   * - ``test_openai_compatible_smoke``
     - OpenAI-compatible API probe. Runs because ``functional.api_smoke`` is ``true``.
   * - ``test_server_health``
     - ``/health``, ``/v1/models``, and a one-token completion. Runs because ``functional.health_check`` is ``true``.
   * - ``test_atom_inference[<cell>]``
     - One benchmark cell. ``<cell>`` looks like ``isl1024_osl1024-conc128``.
   * - ``test_cell_metrics[<cell>-<tier>]``
     - One metric tier for that cell. ``<tier>`` is ``throughput``, ``ttft``, ``tpot``, ``health``, ``scaling``, or ``record``.
   * - ``test_accuracy_eval[<task id>]``
     - One lm-eval task, when ``accuracy`` is configured.
   * - ``test_atom_quant_parity``
     - Short completion probe. Runs because ``quant_parity.enabled`` is ``true``.
   * - ``test_print_results_table``
     - Console summary.
   * - ``test_teardown``
     - Stop the container.

Rows for stages the profile does not enable are skipped. On this file that is ``test_atom_long_context_accuracy`` (no ``long_context_accuracy`` cells), ``test_atom_mtp_quality`` (enabled on ``mtp3``, not on ``perf``), and ``test_verify_dmesg`` (``platform.dmesg_scan`` is off). A failed stage skips later stages with ``a prior lifecycle stage failed``. Teardown still runs.

With ``enforce_thresholds`` false, each ``test_cell_metrics`` row passes and no threshold is asserted. With enforcement on, a tier that has no spec fails that row. The ``scaling`` tier skips on a single-node run. A metric missing from the benchmark artifact is not gated. A tier whose gated metrics are all missing fails with ``no assertable threshold specs``.

Per-cell logs land under ``paths.log_dir``. For the packaged file that is ``/home/<you>/LOGS`` on the host::

  <log_dir>/atom/out-node<rank>/isl<isl>_osl<osl>_conc<conc>/
    atom_server.log
    client.log
    results

A vLLM-ATOM driver writes ``vllm_serve_server.log``. An SGLang driver writes ``sglang_server.log``.

.. important::

  When a run fails, read the server log on the node, not just the CVS client log. Weight-loading failures and GPU exceptions appear there.

The server and client give up with:

- ``server did not become ready before timeout``
- ``atom server warmup did not complete before timeout``
- ``client did not complete before poll cap``
- ``client failed:``

.. _atom-other-files:

Other packaged files
====================

Every stem under ``inference/atom/`` runs with ``cvs run atom``. The reference lists them. Two shapes:

- **Native profiles** (``schema_version`` ``2``), such as the DeepSeek file on this page. ``params.driver`` is ``atom``. Select ``perf`` or ``mtp3`` with ``--config_profile``.
- **Serving schema** (``server_params`` and ``benchmark_params``), used by ``mi3xx_atom_vllm_*`` and ``mi3xx_atom_sglang_*`` stems. ``server_params.backend`` is ``vllm_atom`` or ``sglang``. These files reject ``--config_profile``.

A distributed serving file needs two hosts and a reachable ``server_params.master_addr``. The shipped distributed recipes use pipeline parallel size ``2``. Example:

.. code:: bash

  cvs run atom \
    --cluster_file /tmp/cvs/cluster.json \
    --config_file /tmp/cvs/mi3xx_atom_vllm_deepseek-r1_fp8_distributed.json

Copy that config with its sibling ``mi325x_atom_vllm_deepseek-r1_fp8_distributed_threshold.json`` the same way as the single-node pair. ``mi355x_*`` configs pair with ``mi355x_*`` threshold files.

Native ``driver`` ``atom`` with more than one node is data parallel. Pipeline parallel size greater than 1 requires ``vllm_atom`` or ``sglang``. On two or more hosts, ``test_discover_topology`` resolves InfiniBand HCAs and the socket interface on the host, before the container starts the server. ``roles.server.ib_hca_devices`` and ``roles.server.ib_netdev`` may be ``auto``. An ``ib_netdev`` value that looks like ``mlx5_*`` is treated as ``auto``.

Common pitfalls
===============

**"Container image not specified" or the pull fails.** ``container.image`` is still ``<changeme>`` or is not a reference the node can pull. On a profile file, replace it under the profile you run.

**"threshold_json not found".** The sibling named by ``threshold_json`` is not in the same directory, and CVS has no packaged file of that name. Copy the threshold file next to the config.

**"multiple \*threshold.json files next to config (ambiguous)".** The config does not set ``threshold_json``, and the directory has more than one threshold file. Point ``threshold_json`` at one file, or keep a single sibling in the directory.

**"serving-schema atom configs do not support --config_profile".** Drop ``--config_profile`` for a ``mi*_atom_vllm_*`` or ``mi*_atom_sglang_*`` file.

**"unknown config profile".** ``--config_profile`` is not ``perf`` or ``mtp3`` on this DeepSeek file. The error lists the profiles in the file.

**"params.driver='atom' with nnodes>1 uses ATOM SPMD data parallel".** A native ATOM profile has ``pipeline_parallel_size`` greater than 1. Use a ``vllm_atom`` or ``sglang`` distributed file for pipeline parallel.

**"requires pipeline_parallel_size>1".** A ``vllm_atom`` or ``sglang`` file has two or more nodes and pipeline parallel size 1, and it is not using the Ray backend. Raise ``pipeline_parallel_size``, or set the distributed executor backend to ``ray``.

**"model.remote=1".** Remote download is not available. Keep ``model.remote`` at ``0`` and stage the weights on the host mount.

See also
========

- :doc:`/reference/configuration-files/inference/atom` — configuration schema, shipped models, and thresholds
- :doc:`/reference/cluster/cluster-file` — cluster file schema
- :doc:`/how-to/run-with-containers` — container backend
- :doc:`/how-to/test-suites/index` — other CVS suites
