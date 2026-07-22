.. meta::  :description: Configure the variables in the InferenceX ATOM configuration files
  :keywords: inference, ROCm, install, cvs, InferenceX ATOM, ATOM

***************************************
InferenceX ATOM inference configuration file
***************************************

InferenceX ATOM tests validate LLM serving on AMD GPU clusters using the **ATOM** stack
(``atom.entrypoints.openai_server`` + ``atom.benchmarks.benchmark_serving``). W1 workloads
use ``params.driver: atom``. Multinode **pipeline parallel** (``PP=2``) uses a framework
coordinator: ``params.driver: vllm_atom`` (vLLM + ATOM ROCm kernels) or ``params.driver: sglang``.
A legacy ``params.driver: vllm`` path remains for GPT-OSS uplift only.

The suite checks:

- **Container orchestration**: Docker with ROCm; cluster + variant container blocks merged
- **Model serving**: ATOM OpenAI-compatible server, health + warmup probes
- **Performance metrics**: Throughput, per-GPU throughput, TTFT/TPOT (including p99/p95 tails)
- **Benchmarking**: Named ISL/OSL combos with explicit concurrency sweep cells
- **Result verification**: Tiered ``client.*`` thresholds when ``enforce_thresholds`` is true

Configs use flat ``*_config.json`` + sibling ``*_threshold.json`` pairs under
``cvs/input/config_file/inference/inferencex_atom/``. Filename pattern:
``{gpu}_inferencex-atom_{model}_{precision}[_{mode}]_config.json``.
Pass ``--config_file`` to the ``*_config.json``; :func:`cvs.lib.utils.config_loader.substitute_config`
discovers the sole sibling ``*threshold.json`` in the **config file's parent directory** when
``threshold_json`` is omitted. If that directory contains more than one ``*threshold.json``,
loading fails with an ambiguous-threshold ``ValueError``.

**Lab ``~/input`` layout:** the repo keeps every variant flat in one tree, but after
``cvs copy-config`` you should place each run's config + threshold pair in a dedicated
subdirectory (for example ``~/input/.../inferencex_atom/smoke/``) so only one
threshold file sits beside the config you pass to ``--config_file``. Alternatively set
``"threshold_json"`` in the config to an explicit path. See the in-tree README at
``cvs/input/config_file/inference/inferencex_atom/README.md`` for copy-paste commands.

**Cluster file:** use ``cvs/input/cluster_file/inferencex_atom_cluster.json``. Edit ``node_dict`` so host count matches variant ``params.nnodes`` (one host for single-node sweeps; two for multinode).
Container ``name`` must match the variant (``inferencex_atom_mi300x`` / ``inferencex_atom_mi355x``);
the suite deep-merges variant ``container`` over the cluster file.

**Launcher vs GPU node:** pytest and HTML/log output run on the host where you invoke
``cvs run``. :class:`cvs.core.orchestrators.container.ContainerOrchestrator` SSHes to
cluster nodes (``cluster_file`` ``mgmt_ip`` / ``node_dict``) and runs ``sudo docker`` there.
``paths.models_dir`` and the ATOM image must exist on the GPU node; ``priv_key_file`` and
``paths.hf_token_file`` are read on the launcher. Local Docker on the launcher is not required.

**ATOM server CLI:** set ``roles.server.atom_args`` inline in the config (vLLM-style, analogous to
``roles.server.serve_args`` on ``vllm_single``). When ``params.driver`` is ``atom``, ``atom_args``
is required. MTP3 variants also set ``params.bench_extra_args`` (for example ``--use-chat-template``).

Pytest and HTML layout (inferencex_atom)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 10 35 55
   :header-rows: 1

   * - Stage
     - Test
     - Notes
   * - 1
     - ``test_launch_container``
     - Host Docker launch; records ``container_launch``.
   * - 2
     - ``test_setup_sshd``
     - Multinode only; single-node skips sshd probe.
   * - 3
     - ``test_model_fetch``
     - Ensures model bytes under ``paths.models_dir``.
   * - 4
     - ``test_inferencex_atom_inference``
     - One parametrized cell; server start (or reuse), bench, parse ``results.json``.
   * - 5
     - ``test_cell_metrics``
     - One HTML row per **metric tier** per cell (throughput, ttft, tpot, health, record).
   * - 6
     - ``test_print_results_table``
     - Session results grid from ``inf_res_dict``.
   * - 7
     - ``test_teardown``
     - Explicit teardown; sets ``lifecycle.torn_down``.

Example variant layout
======================

Each stem has ``<stem>_config.json`` (``schema_version: 1``, ``framework: inferencex_atom``)
and sibling ``<stem>_threshold.json``. In the CVS source tree many stems share one directory;
on a lab machine, copy only the pair you need into a per-variant subdirectory (or set
``threshold_json``). See ``mi300x_inferencex-atom_deepseek-r1_fp8_perf_config.json``
for the W1 MI300X reference.

.. dropdown:: Example ``mi300x_inferencex-atom_deepseek-r1_fp8_perf_threshold.json`` (excerpt)

  .. code:: json

    {
      "ISL=1024,OSL=1024,TP=8,CONC=128": {
        "client.output_throughput": {"kind": "min_tok_s", "value": 2590.98},
        "client.per_gpu_throughput": {"kind": "min_tok_s", "value": 648.65},
        "client.p99_ttft_ms": {"kind": "max_ms", "value": 1834.3},
        "client.p95_tpot_ms": {"kind": "max_ms", "value": 53.76},
        "client.success_rate": {"kind": "min", "value": 1},
        "client.failed": {"kind": "max", "value": 0}
      }
    }

  Every member of :data:`cvs.lib.inference.inferencex_atom.inferencex_atom_parsing.GATED_METRICS` needs a
  spec in each cell when ``enforce_thresholds`` is true. W1 perf gates include
  ``per_gpu_throughput``, ``output_tput_per_gpu``, ``p99_ttft_ms``, and ``p95_tpot_ms``.

Parameters
==========

Top-level blocks follow the DTNI variant schema. InferenceX ATOM-specific keys:

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Block / key
     - Example
     - Description
   * - ``framework``
     - ``inferencex_atom``
     - Suite identifier for :func:`load_variant`.
   * - ``gpu_arch``
     - ``mi300x``
     - Hardware profile for the variant.
   * - ``roles.server.atom_args``
     - ``["-tp", "8", "--kv_cache_dtype", "fp8"]``
     - Inline ATOM ``openai_server`` CLI tokens after ``--model`` / ``--server-port``.
   * - ``roles.server.serve_args``
     - ``{"enforce-eager": true}``
     - vLLM uplift path only (``params.driver: vllm``).
   * - ``enforce_thresholds``
     - ``true`` / ``false``
     - When true, ``test_cell_metrics`` asserts via :func:`cvs.lib.utils.verdict.evaluate_all`.
   * - ``paths.*``
     - ``shared_fs``, ``models_dir`` (``/home/models``), ``log_dir``, ``hf_token_file``
     - ``models_dir`` is an absolute HF hub cache path on GPU nodes; variant configs also bind-mount ``/home/models`` into the container.
   * - ``model.id``
     - ``deepseek-ai/DeepSeek-R1-0528``
     - HuggingFace model id for ATOM server and bench.
   * - ``container.image`` / ``container.name``
     - ``rocm/atom-dev:latest``, ``inferencex_atom_mi300x``
     - Docker image and container name (override cluster file defaults).
   * - ``roles.server.atom_args``
     - ``-tp``, ``--kv_cache_dtype``
     - Extra CLI tokens after ``--model`` / ``--server-port`` (ATOM driver).
   * - ``roles.server.env``
     - ``ATOM_DISABLE_MMAP``
     - Merged into ``/tmp/server_env_script.sh`` before server launch.
   * - ``params.driver``
     - ``atom`` / ``vllm`` / ``vllm_atom`` / ``sglang``
     - ``atom`` = standalone ATOM server + ``benchmark_serving`` (no native PP). ``vllm_atom`` = vLLM multinode PP coordinator + ATOM kernels. ``sglang`` = SGLang PP coordinator. ``vllm`` = interim uplift.
   * - ``params.tensor_parallelism``
     - ``8``
     - TP size; appears in threshold cell keys as ``TP``.
   * - ``params.reuse_server_across_sweep``
     - ``true``
     - Skip server restart when only concurrency changes between sweep cells.
   * - ``params.nnodes`` / ``params.pipeline_parallel_size``
     - ``2`` / ``2`` (multinode PP)
     - True multinode pipeline parallel: set ``driver=vllm_atom`` or ``sglang`` with ``nnodes=2`` and ``pipeline_parallel_size=2`` (cell keys use ``PP=2``). Requires ``roles.server.ib_netdev``. Standalone ``driver=atom`` multinode uses SPMD data parallel (``DP`` in cell keys), not PP.
   * - ``params.master_addr`` / ``params.master_port``
     - head VPC IP / ``29501``
     - Rendezvous for distributed ATOM/vLLM executor; defaults to cluster head when empty.
   * - ``params.scaling_baseline_output_throughput``
     - ``1500``
     - Single-node reference ``output_throughput`` for ``scaling.efficiency_pct`` (record-only).
   * - ``params.server_warmup_wait_s`` / ``client_initial_wait_s``
     - ``330`` / ``120``
     - Config-driven server warmup and client poll floor (shorter on smoke configs).
   * - ``params.metric_percentiles``
     - ``95,99``
     - Tail percentiles for W1 gates (p95 TPOT, p99 TTFT).
   * - ``params.bench_max_failed_requests``
     - ``0``
     - Runtime cap on bench ``Failed requests``; pair with threshold ``client.failed``.
   * - ``sweep.sequence_combinations`` / ``sweep.runs``
     - named ISL/OSL + ``{combo, concurrency}``
     - Explicit cell list (not a cartesian product).

Metric tiers and parsing live in :mod:`cvs.lib.inference.inferencex_atom.inferencex_atom_parsing`
(see ``cvs/lib/inference/utils/docs/inferencex-atom-parsing.md``). Legacy monolithic JSON
(``config`` + ``benchmark_params``) and the deprecated ``inferencemax`` suite are not used.
