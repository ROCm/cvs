.. meta::
  :description: Reference for the ATOM inference benchmark configuration in CVS, covering drivers, sweep cells, threshold files, and multi-node pipeline parallelism for AMD GPUs.
  :keywords: CVS, ATOM, inference, ROCm, LLM, benchmark, AMD Instinct, GPU, threshold, vLLM, SGLang, JSON, multinode

******************************************************************************
ATOM inference benchmark configuration file for Cluster Validation Suite (CVS)
******************************************************************************

The ATOM suite validates LLM serving on AMD Instinct GPUs. One suite name, ``atom``, covers native ATOM, vLLM-ATOM, and SGLang. Native single-node files set ``params.driver`` to ``atom``. Shipped multinode pipeline-parallel files set ``server_params.backend`` to ``vllm_atom`` or ``sglang``.

Run the suite with:

.. code:: bash

  cvs run atom --cluster_file <cluster.json> --config_file <config.json>

For a step-by-step first run of the packaged DeepSeek-R1 FP8 single-node files, see :doc:`/how-to/test-suites/inference/atom`.

Lifecycle
=========

Each stage is an independent test, so every stage is its own row in the HTML report. The suite orders the stages explicitly:

.. list-table::
   :widths: 1 3 6
   :header-rows: 1

   * - Order
     - Test
     - Purpose
   * - 1
     - ``test_launch_container``
     - Start the container on every node.
   * - 2
     - ``test_setup_sshd``
     - In-container sshd for multinode SSH. Passes without starting sshd on one host and on a managed-compute step.
   * - 3
     - ``test_discover_topology``
     - Resolve InfiniBand HCAs and the socket interface. No-op when ``nnodes`` is 1.
   * - 4
     - ``test_model_fetch``
     - Check that ``paths.models_dir`` contains bytes when ``model.remote`` is ``0``.
   * - 5
     - ``test_openai_compatible_smoke``
     - OpenAI-compatible API probe when ``functional.api_smoke`` is true.
   * - 6
     - ``test_server_health``
     - ``/health``, ``/v1/models``, and a one-token completion when ``functional.health_check`` is true.
   * - 7
     - ``test_atom_inference``
     - One benchmark cell. The row id looks like ``isl1024_osl1024-conc128``.
   * - 8
     - ``test_cell_metrics``
     - One metric tier per cell (``throughput``, ``ttft``, ``tpot``, ``health``, ``scaling``, ``record``).
   * - 9
     - ``test_accuracy_eval``
     - One lm-eval task, when ``accuracy.tasks`` is set.
   * - 10
     - ``test_atom_long_context_accuracy``
     - One long-context cell, when ``long_context_accuracy.cells`` is set.
   * - 11
     - ``test_atom_mtp_quality``
     - MTP acceptance checks when ``mtp_quality.enabled`` is true.
   * - 12
     - ``test_atom_quant_parity``
     - Short completion probe when ``quant_parity.enabled`` is true.
   * - 13
     - ``test_verify_dmesg``
     - Kernel-log scan for the run window when ``platform.dmesg_scan`` is true.
   * - 14
     - ``test_print_results_table``
     - Console summary.
   * - 15
     - ``test_teardown``
     - Stop the container.

A failed stage skips later stages with ``a prior lifecycle stage failed``. Teardown still runs. Stages whose config block is off are skipped with a reason that names the block, for example ``mtp_quality not enabled for this variant``.

File layout
===========

Shipped files live under ``cvs/input/config_file/inference/atom/``.

Config stems use ``mi3xx_*`` (gfx942) or ``mi355x_*`` (gfx950). ``mi3xx_*`` configs name an ``mi325x_*`` threshold file. ``mi355x_*`` configs name an ``mi355x_*`` threshold file. ``threshold_json`` is that filename, resolved next to the config. It is not passed through placeholder substitution. A missing sibling falls back to the packaged file of the same name, with a warning. A config that omits ``threshold_json`` requires exactly one ``*threshold.json`` in the same directory.

Two JSON shapes ship:

- **Native profiles.** ``schema_version`` is ``2``. Job shapes live under ``profiles``. ``default_profile`` selects the shape when ``--config_profile`` (or ``CVS_CONFIG_PROFILE``) is omitted. The packaged DeepSeek-R1 and Qwen FP8 single-node stems use ``perf`` and ``mtp3``.
- **Serving schema.** Top-level ``server_params``, ``benchmark_params``, ``sweeps``, and ``runs``. Used by ``*_atom_vllm_*`` and ``*_atom_sglang_*`` stems. ``--config_profile`` is rejected.

Top-level keys that start with ``_`` (for example ``_comment``) are ignored.

Shipped model inventory
=======================

List the directory with ``cvs config list inference/atom``. Each config's ``threshold_json`` names the sibling to copy beside it.

.. list-table::
   :widths: 4 3 3
   :header-rows: 1

   * - Config stem
     - Files
     - Driver
   * - ``mi3xx_atom_deepseek-r1_fp8``
     - ``_single`` (``perf``, ``mtp3``)
     - ``atom``
   * - ``mi355x_atom_deepseek-r1_fp8``
     - ``_single`` (``perf``, ``mtp3``)
     - ``atom``
   * - ``mi3xx_atom_qwen3.5-397b-a17b_fp8``
     - ``_single`` (``perf``, ``mtp3``)
     - ``atom``
   * - ``mi355x_atom_qwen3.5-397b-a17b_fp8``
     - ``_single`` (``perf``, ``mtp3``)
     - ``atom``
   * - ``mi355x_atom_kimi-k27-code_mxfp4``
     - ``_single``
     - ``atom``
   * - ``mi355x_atom_deepseek-v4-pro``
     - ``_single``
     - ``atom``
   * - ``mi3xx_atom_vllm_deepseek-r1_fp8``
     - ``_single``, ``_distributed``
     - ``vllm_atom``
   * - ``mi355x_atom_vllm_deepseek-r1_fp8``
     - ``_single``, ``_distributed``
     - ``vllm_atom``
   * - ``mi3xx_atom_vllm_gpt-oss-120b_mxfp4``
     - ``_single``
     - ``vllm_atom``
   * - ``mi3xx_atom_vllm_glm-5.1_fp8``
     - ``_single``, ``_distributed``
     - vLLM recipe only; MI325X single-node passed smoke, health, short cells, and GSM8K; distributed bring-up thresholds
   * - ``mi3xx_atom_vllm_deepseek-v4-flash_fp8``
     - ``_single``
     - ``vllm_atom``
   * - ``mi3xx_atom_vllm_qwen3.5-397b-a17b_fp8``
     - ``_single``, ``_distributed``
     - ``vllm_atom``
   * - ``mi355x_atom_vllm_qwen3.5-397b-a17b_fp8``
     - ``_single``, ``_distributed``
     - ``vllm_atom``
   * - ``mi3xx_atom_sglang_deepseek-r1_fp8``
     - ``_single``, ``_distributed``
     - ``sglang``
   * - ``mi355x_atom_sglang_deepseek-r1_fp8``
     - ``_single``, ``_distributed``
     - ``sglang``
   * - ``mi3xx_atom_sglang_qwen3.5-397b-a17b_fp8``
     - ``_single``, ``_distributed``
     - ``sglang``
   * - ``mi355x_atom_sglang_qwen3.5-397b-a17b_fp8``
     - ``_single``, ``_distributed``
     - ``sglang``

Native Qwen threshold files omit ``_single`` in the filename: ``mi325x_atom_qwen3.5-397b-a17b_fp8_threshold.json`` and ``mi355x_atom_qwen3.5-397b-a17b_fp8_threshold.json``.

Config profiles
===============

On a ``schema_version`` ``2`` file, ``--config_profile NAME`` (or ``CVS_CONFIG_PROFILE``) selects one profile. The DeepSeek-R1 FP8 and Qwen FP8 single-node stems use the same names:

.. list-table::
   :widths: 2 2 4
   :header-rows: 1

   * - ``--config_profile``
     - Driver
     - Use when
   * - ``perf`` (default)
     - ``atom``
     - Serving sweep plus the accuracy tasks in that profile
   * - ``mtp3``
     - ``atom``
     - Speculative decode (``--method mtp``, ``--num-speculative-tokens 3``) plus MTP quality checks

Serving-schema files raise ``serving-schema atom configs do not support --config_profile``. An unknown name raises ``unknown config profile``, and the message lists the names in the file.

Fields you must customize
=========================

Replace these before a run. The first-run walkthrough for the DeepSeek single-node file is :doc:`/how-to/test-suites/inference/atom`.

.. list-table::
   :widths: 3 4
   :header-rows: 1

   * - Where
     - Change to
   * - ``container.image``
     - Image on the GPU nodes that provides the selected driver. Shipped files use ``<changeme>``.
   * - ``container.runtime.args.volumes`` (``<changeme-models-mount>``)
     - Host directory of weights, mounted read-only at ``/models``.
   * - Cluster file
     - SSH user, private key, and node addresses. Host count matches ``nnodes``.
   * - ``master_addr``
     - Address reachable from every node on a multinode file. Shipped distributed files use ``{head-node-ip}``, which you replace by hand.
   * - Threshold values
     - Measured limits for your hardware, before you set ``enforce_thresholds`` to ``true``.

Placeholder substitution
========================

- ``{user-id}`` becomes the cluster ``username``, or the local login name when that field is empty.
- ``{shared_fs}`` expands inside ``paths``.
- ``{paths.models_dir}`` and other ``{paths.*}`` expand anywhere else in the file.
- ``<changeme>`` and ``<changeme-models-mount>`` are literal strings you replace. They are not substituted.
- ``{head-node-ip}`` is not substituted. Replace it in copied multinode configs.
- ``threshold_json`` is a literal filename.

``model.remote`` must stay ``0``. ``model.remote=1`` is rejected. Stage the weights on the host mount.

Configuration schema
====================

Native profile files put the workload keys inside each profile. Serving-schema files put the same settings at the top level, in ``server_params`` and ``benchmark_params``. After load, both shapes use one driver, one node count, and one sweep.

Fields:

.. list-table::
   :widths: 3 2 5
   :header-rows: 1

   * - Field
     - Required
     - Meaning
   * - ``schema_version``
     - on profile files
     - ``2`` for multi-profile files. Serving-schema files omit it.
   * - ``framework``
     - no
     - ``atom`` when set.
   * - ``default_profile``
     - on profile files
     - Profile used when ``--config_profile`` is omitted. Packaged stems use ``perf``.
   * - ``profiles``
     - on profile files
     - Map of profile name to a workload body.
   * - ``enforce_thresholds``
     - no (default ``true``)
     - Inside the profile on a ``schema_version`` ``2`` file, and at the top level on a serving-schema file. ``false`` records metrics and asserts no threshold spec. Packaged files set ``false``.
   * - ``threshold_json``
     - yes
     - Sibling threshold filename.
   * - ``gpu_arch``
     - no
     - Taken from the ``{gpu}_atom_`` filename prefix when omitted.
   * - ``paths``
     - yes
     - ``shared_fs``, ``models_dir``, ``log_dir``, ``hf_token_file``.
   * - ``model``
     - yes
     - ``id`` and ``remote`` (``0``).
   * - ``container``
     - yes
     - Image, lifetime, and Docker runtime. ``runtime.name`` is ``docker``.
   * - ``server_params``
     - serving schema
     - Server driver, parallelism, ``master_addr``, ``serve_args`` or ``sglang_args``, and server env.
   * - ``benchmark_params``
     - serving schema
     - Client workload: prompts, length, percentiles, poll waits.
   * - ``roles.server``
     - native ``atom``
     - ``atom_args`` (required for ``driver`` ``atom``), ``env``, and optional ``ib_hca_devices`` / ``ib_netdev``.
   * - ``params``
     - native profiles
     - Driver and client workload. See below.
   * - ``sweeps``
     - yes
     - Map of cell keys to benchmark overrides.
   * - ``runs``
     - no
     - Cell keys to execute. Empty runs every ``ISL=`` key in ``sweeps``.
   * - ``functional``
     - no
     - ``api_smoke`` and ``health_check``. The packaged ``perf`` profile sets both to ``true``.
   * - ``accuracy``
     - no
     - ``tasks[]`` of lm-eval tasks. Each task becomes one ``test_accuracy_eval`` row.
   * - ``mtp_quality``
     - no
     - MTP checks. The packaged ``mtp3`` profile sets ``enabled`` to ``true``.
   * - ``quant_parity``
     - no
     - Short completion probe. The packaged DeepSeek ``perf`` profile sets ``enabled`` to ``true``.
   * - ``long_context_accuracy``
     - no
     - ``cells[]`` for needle-in-a-haystack checks.
   * - ``platform``
     - no
     - ``dmesg_scan`` (default ``false``) and ``gpu_metrics_poll``.

``params`` and ``server_params``
--------------------------------

Native profiles use ``params``. Serving-schema files use ``server_params`` for the server and ``benchmark_params`` for the client. The loader maps ``server_params.backend`` to the driver: ``atom`` stays ``atom``, ``vllm`` and ``vllm_atom`` become ``vllm_atom``, and ``sglang`` stays ``sglang``.

.. list-table::
   :widths: 3 5
   :header-rows: 1

   * - Field
     - Meaning
   * - ``driver`` / ``server_params.backend``
     - ``atom``, ``vllm_atom``, or ``sglang``. ``vllm`` is accepted on ``backend`` and runs as ``vllm_atom``.
   * - ``tensor_parallelism``
     - Tensor-parallel size. Appears as ``TP=`` in threshold keys. The packaged DeepSeek file uses ``8``.
   * - ``pipeline_parallel_size`` / ``pipeline_parallelism``
     - Pipeline-parallel size. Appears as ``PP=``. ``1`` on single-node stems. ``2`` on shipped distributed stems.
   * - ``nnodes``
     - Node count. Must match the number of hosts in the cluster file. It is not part of the threshold cell key.
   * - ``master_addr`` / ``master_port``
     - Coordinator address and port on multinode runs.
   * - ``atom_args``
     - Extra CLI tokens for native ATOM. Required when ``driver`` is ``atom``.
   * - ``serve_args``
     - Server flags for ``vllm_atom``, including ``distributed-executor-backend`` (``mp`` or ``ray``).
   * - ``ib_hca_devices`` / ``ib_netdev``
     - ``auto`` or an explicit list. ``auto`` is resolved by ``test_discover_topology`` on the host. A value that looks like ``mlx5_*`` is treated as ``auto``.
   * - ``num_prompts``, ``max_model_length``, ``metric_percentiles``
     - Benchmark client workload.
   * - ``reuse_server_across_sweep``
     - ``true`` keeps the server across cells that share model, shape, and server arguments.
   * - ``scaling_baseline_output_throughput``
     - Single-node output tok/s used to compute ``scaling.efficiency_pct`` on multinode runs.
   * - ``server_precheck_wait_s``, ``server_warmup_wait_s``, ``server_poll_count``, ``server_poll_wait_time``
     - How long the suite waits for the server log and for ready.
   * - ``client_initial_wait_s``, ``client_poll_count``, ``client_poll_wait_time``
     - How long the suite waits for the benchmark client.

Sweep cells
-----------

A cell key is:

.. code:: text

  ISL=<isl>,OSL=<osl>,TP=<tensor_parallelism>,PP=<pipeline_parallel_size>,CONC=<concurrency>

``PP=1`` on single-node cells. ``runs`` lists the keys to execute. An empty ``runs`` list executes every ``ISL=`` key in ``sweeps``. The pytest row id shortens the key to ``isl<isl>_osl<osl>-conc<concurrency>``.

Per-cell logs:

.. code:: text

  <log_dir>/atom/out-node<rank>/isl<isl>_osl<osl>_conc<concurrency>/
    atom_server.log          # driver atom
    vllm_serve_server.log    # driver vllm_atom
    sglang_server.log        # driver sglang
    client.log
    results

Drivers
-------

.. list-table::
   :widths: 2 3 3
   :header-rows: 1

   * - Driver
     - When
     - Server
   * - ``atom``
     - Native single-node, including ``perf`` and ``mtp3``
     - Native ATOM OpenAI server
   * - ``vllm_atom``
     - Shipped vLLM stems, including 2-node pipeline parallel
     - ``vllm serve`` with the ATOM image environment
   * - ``sglang``
     - Shipped SGLang stems, including 2-node pipeline parallel
     - SGLang launch server

Native ``atom`` with ``nnodes`` greater than 1 is data parallel. ``pipeline_parallel_size`` greater than 1 on that driver is rejected. ``vllm_atom`` and ``sglang`` require ``pipeline_parallel_size`` greater than 1 when ``nnodes`` is greater than 1, unless ``serve_args`` sets ``distributed-executor-backend`` to ``ray``. ``pipeline_parallel_size`` greater than 1 also requires ``nnodes`` greater than 1.

Threshold files
===============

A native threshold file maps ``profiles.<name>`` to cell keys. A serving-schema threshold file maps cell keys at the top level. Each cell maps a bare metric name to ``{kind, value}``.

With ``enforce_thresholds`` true, every gated metric in a cell needs a spec. Metrics absent from the benchmark artifact are skipped. A tier whose gated metrics are all absent fails that ``test_cell_metrics`` row. The ``scaling`` tier skips when ``nnodes`` is 1.

Gated tiers:

.. list-table::
   :widths: 2 6
   :header-rows: 1

   * - Tier
     - Bare metric names
   * - ``throughput``
     - ``total_token_throughput``, ``output_throughput``, ``per_gpu_throughput``, ``output_tput_per_gpu``
   * - ``ttft``
     - ``mean_ttft_ms``, ``p99_ttft_ms``
   * - ``tpot``
     - ``mean_tpot_ms``, ``p99_tpot_ms``
   * - ``health``
     - ``success_rate``, ``failed``
   * - ``scaling``
     - ``scaling.efficiency_pct`` (multinode)
   * - ``record``
     - Remaining client metrics. Logged, and not gated.

Threshold kinds:

.. list-table::
   :widths: 2 4
   :header-rows: 1

   * - ``kind``
     - Passes when
   * - ``min``
     - Actual is greater than or equal to ``value``.
   * - ``max``
     - Actual is less than or equal to ``value``.
   * - ``max_ms``
     - Actual is less than or equal to ``value`` (milliseconds).
   * - ``min_tok_s``
     - Actual is greater than or equal to ``value`` (tokens per second).
   * - ``within``
     - Actual is inside the tolerance fields on the spec.
   * - ``min_ratio``
     - Actual meets the ratio against the reference metric named on the spec.
   * - ``info``
     - Always recorded.

Example single-node cell:

.. code:: json

  "ISL=1024,OSL=1024,TP=8,PP=1,CONC=128": {
    "output_throughput": {"kind": "min_tok_s", "value": 1500},
    "p99_ttft_ms": {"kind": "max_ms", "value": 5000},
    "success_rate": {"kind": "min", "value": 1},
    "failed": {"kind": "max", "value": 0}
  }

Example multinode cell:

.. code:: json

  "ISL=1024,OSL=1024,TP=8,PP=2,CONC=128": {
    "output_throughput": {"kind": "min_tok_s", "value": 2500},
    "scaling.efficiency_pct": {"kind": "min", "value": 80}
  }

Non-sweep keys in the same file are ``accuracy``, ``mtp_quality``, ``quant_parity``, and ``long_context_accuracy``. They are consulted only when ``enforce_thresholds`` is true and that stage is enabled.

Errors
======

.. list-table::
   :widths: 4 6
   :header-rows: 1

   * - Message
     - What to change
   * - ``serving-schema atom configs do not support --config_profile``
     - Omit ``--config_profile`` on a vLLM or SGLang stem.
   * - ``unknown config profile``
     - Use a name listed in the error. Packaged profile stems define ``perf`` and ``mtp3``.
   * - ``threshold_json not found``
     - Copy the named sibling next to the config.
   * - ``multiple *threshold.json files next to config (ambiguous)``
     - Set ``threshold_json``, or leave one threshold file in the directory.
   * - ``params.driver='atom' requires roles.server.atom_args``
     - Add the native server CLI tokens, or switch to a serving-schema file.
   * - ``params.driver='atom' with nnodes>1 uses ATOM SPMD data parallel``
     - Pipeline parallel needs ``vllm_atom`` or ``sglang``.
   * - ``requires pipeline_parallel_size>1``
     - Raise pipeline parallel size, or set ``distributed-executor-backend`` to ``ray``.
   * - ``pipeline_parallel_size=… > 1 requires nnodes > 1``
     - Add hosts until ``nnodes`` matches the pipeline width.
   * - ``model.remote=1``
     - Keep ``remote`` at ``0`` and stage weights on the host mount.
   * - ``server did not become ready before timeout``
     - Read ``atom_server.log``, ``vllm_serve_server.log``, or ``sglang_server.log`` under ``paths.log_dir``.
   * - ``atom server warmup did not complete before timeout``
     - Same server log. Raise ``server_warmup_wait_s`` or ``server_poll_count`` if the model is still loading.
   * - ``client did not complete before poll cap``
     - Read ``client.log`` in the same per-cell directory.
   * - ``client failed:``
     - The client log names the failed requests.
   * - ``no threshold specs for tier``
     - Add that tier's metrics to the cell, or leave ``enforce_thresholds`` false.
   * - ``no assertable threshold specs for tier``
     - The benchmark artifact omitted every gated metric for that tier.

Related resources
=================

- :doc:`/how-to/test-suites/inference/atom` — first run of the packaged DeepSeek-R1 FP8 single-node files
- :doc:`/reference/cluster/cluster-file` — cluster file schema
- :doc:`/how-to/run-with-containers` — container backend
