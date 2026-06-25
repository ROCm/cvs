.. meta::  :description: Configure the variables in the InferenceX ATOM configuration files
  :keywords: inference, ROCm, install, cvs, InferenceX ATOM, ATOM

***************************************
InferenceX ATOM inference configuration file
***************************************

InferenceX ATOM tests validate LLM serving on AMD GPU clusters using the **ATOM** stack
(``atom.entrypoints.openai_server`` + ``atom.benchmarks.benchmark_serving``). W1 workloads
use ``params.driver: atom``. A legacy ``params.driver: vllm`` path (``vllm serve`` +
``vllm bench serve``) remains for GPT-OSS uplift only.

The suite checks:

- **Container orchestration**: Docker with ROCm; cluster + variant container blocks merged
- **Model serving**: ATOM OpenAI-compatible server, health + warmup probes
- **Performance metrics**: Throughput, per-GPU throughput, TTFT/TPOT (including p99/p95 tails)
- **Benchmarking**: Named ISL/OSL combos with explicit concurrency sweep cells
- **Result verification**: Tiered ``client.*`` thresholds when ``enforce_thresholds`` is true

Configs use flat ``*_config.json`` + sibling ``*_threshold.json`` pairs under
``cvs/input/config_file/inference/inferencex_atom_single/``. Filename pattern:
``{gpu}_inferencex-atom-single_{model}_{precision}[_{mode}]_config.json``.
Pass ``--config_file`` to the ``*_config.json``; :func:`cvs.lib.utils.config_loader.substitute_config`
discovers the sole sibling ``*threshold.json`` when ``threshold_json`` is omitted.

**Cluster file:** use ``cvs/input/cluster_file/mi300x_atom_single.json`` (or ``mi355x_atom_single.json``).
Container ``name`` must match the variant (``inferencex_atom_mi300x`` / ``inferencex_atom_mi355x``);
the suite deep-merges variant ``container`` over the cluster file.

**W1 recipe wiring:** set ``ix_recipe_id`` (e.g. ``dsr1-fp8-mi300x-atom``) to merge server
``atom_args`` and client ``bench_extra_args`` from ``ix_recipes.json``.

Pytest and HTML layout (inferencex_atom_single)
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

Each stem has ``<stem>_config.json`` (``schema_version: 1``, ``framework: inferencex_atom_single``)
and sibling ``<stem>_threshold.json``. See ``mi300x_inferencex-atom-single_deepseek-r1_fp8_perf_config.json``
for the W1 MI300X reference.

.. dropdown:: Example ``mi300x_inferencex-atom-single_deepseek-r1_fp8_perf_threshold.json`` (excerpt)

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

  Every member of :data:`cvs.lib.inference.utils.inferencex_atom_parsing.GATED_METRICS` needs a
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
     - ``inferencex_atom_single``
     - Suite identifier for :func:`load_variant`.
   * - ``gpu_arch``
     - ``mi300x``
     - Must match ``ix_recipe_id`` when a recipe is set.
   * - ``ix_recipe_id``
     - ``dsr1-fp8-mi300x-atom``
     - Merges ``atom_args`` / ``bench_extra_args`` from ``ix_recipes.json``.
   * - ``enforce_thresholds``
     - ``true`` / ``false``
     - When true, ``test_cell_metrics`` asserts via :func:`cvs.lib.utils.verdict.evaluate_all`.
   * - ``paths.*``
     - ``shared_fs``, ``models_dir``, ``log_dir``, ``hf_token_file``
     - Placeholder-substituted paths (``{user-id}`` resolved at load).
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
     - ``atom`` / ``vllm``
     - ``atom`` = ATOM server + ``benchmark_serving``; ``vllm`` = interim uplift path.
   * - ``params.tensor_parallelism``
     - ``8``
     - TP size; appears in threshold cell keys as ``TP``.
   * - ``params.reuse_server_across_sweep``
     - ``true``
     - Skip server restart when only concurrency changes between sweep cells.
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

Metric tiers and parsing live in :mod:`cvs.lib.inference.utils.inferencex_atom_parsing`
(see ``cvs/lib/inference/utils/docs/inferencex-atom-parsing.md``). Legacy monolithic JSON
(``config`` + ``benchmark_params``) and the deprecated ``inferencemax`` suite are not used.
