.. meta::
  :description: Configure the ATOM inference benchmark suite in CVS
  :keywords: inference, ROCm, cvs, ATOM, LLM, benchmark, multinode, thresholds

**********************************
ATOM inference configuration file
**********************************

The ATOM suite validates LLM serving on AMD Instinct GPUs. Single-node variants
use ``params.driver: atom`` (native ``openai_server``). Shipped multinode
**pipeline parallel** stems use ``params.driver: vllm_atom`` or ``sglang`` with
``nnodes: 2`` and ``pipeline_parallel_size: 2``.

Run the suite with:

.. code:: bash

  cvs run atom --cluster_file <cluster.json> --config_file <config.json>

For a step-by-step first run, see :doc:`/how-to/test-suites/inference/atom`. Config
JSON uses full param names (for example ``tensor_parallelism``); threshold cell
keys use abbreviations (for example ``TP=``, ``PP=``).

File layout
===========

Shipped files live under ``cvs/input/config_file/inference/atom/``:

.. code:: text

  {gpu}_atom_{model}_{precision}_single.json
  {gpu}_atom_{model}_{precision}_distributed.json   # when multinode PP is supported
  {platform}_atom_{model}_{precision}_threshold.json

Config stems use the **family** prefix ``mi3xx``. Threshold files use a
**platform** prefix (shipped: ``mi325x``, lab-validated on MI325X / gfx942).
Each config's ``threshold_json`` points at the matching ``mi325x_*_threshold.json``.

Multi-profile configs (``schema_version: 2``) embed job shapes under ``profiles``.
Select one at runtime with ``--config_profile NAME`` (or ``CVS_CONFIG_PROFILE``).
Flat ``schema_version: 1`` files use an implicit ``perf`` profile.

:func:`cvs.lib.utils.config_loader.substitute_config` resolves ``threshold_json``
next to the config. Multiple ``*threshold.json`` files in one directory raises an
ambiguous-threshold error — on lab machines, copy each variant pair into its own
subdirectory (see :doc:`/how-to/test-suites/inference/atom`).

Shipped model inventory
=======================

Lab-validated configs only. Native ATOM ``driver=atom`` workloads use flat
``schema_version: 1`` JSON or ``schema_version: 2`` profiles (``perf``, ``mtp3``).

Framework parity (vLLM / SGLang) uses the unified serving schema under
``inference/atom/`` (stems ``mi3xx_atom_vllm_*``, ``mi3xx_atom_sglang_*``) and
still runs with ``cvs run atom``.

.. list-table::
   :widths: 3 3 4
   :header-rows: 1

   * - Model stem
     - Config files
     - Notes
   * - ``mi3xx_atom_deepseek-r1_fp8``
     - ``_single`` (``perf`` + ``mtp3`` profiles), ``_distributed`` (``vllm_atom`` PP=2)
     - Native ATOM + multinode PP
   * - ``mi3xx_atom_deepseek-v4-pro``
     - ``_single``
     - Native ATOM long-context
   * - ``mi3xx_atom_qwen3.5-397b-a17b_fp8``
     - ``_single``
     - Native ATOM perf + accuracy
   * - ``mi3xx_atom_vllm_deepseek-r1_fp8``
     - ``_single``
     - vLLM parity (serving schema)
   * - ``mi3xx_atom_vllm_gpt-oss-120b_mxfp4``
     - ``_single``
     - GPT-OSS MXFP4 vLLM parity (serving schema)
   * - ``mi3xx_atom_sglang_deepseek-r1_fp8``
     - ``_single``, ``_distributed``
     - SGLang parity (serving schema)

Config profiles
===============

``schema_version: 2`` atom files embed job shapes under ``profiles``. Select one at
runtime with ``--config_profile`` (or ``CVS_CONFIG_PROFILE``). Flat
``schema_version: 1`` files use an implicit ``perf`` profile.

DeepSeek R1 FP8 — ``mi3xx_atom_deepseek-r1_fp8_single.json`` profiles:

.. list-table::
   :widths: 2 2 4
   :header-rows: 1

   * - ``--config_profile``
     - Driver
     - Use when
   * - ``perf`` (default)
     - ``atom``
     - Daily perf gate + full accuracy matrix
   * - ``mtp3``
     - ``atom``
     - Speculative-decode perf + gsm8k quality

``atom_vllm`` / ``atom_sglang`` parity (serving schema, same suite):

.. code:: bash

  cvs run atom \
    --config_file ~/input/.../mi3xx_atom_sglang_deepseek-r1_fp8_single.json \
    --cluster_file ~/input/cluster_file/atom_cluster.json

  cvs run atom \
    --config_file ~/input/.../mi3xx_atom_vllm_deepseek-r1_fp8_single.json \
    --cluster_file ~/input/cluster_file/atom_cluster.json

  cvs run atom \
    --config_file ~/input/.../mi3xx_atom_vllm_gpt-oss-120b_mxfp4_single.json \
    --cluster_file ~/input/cluster_file/atom_cluster.json

.. code:: bash

  cvs run atom \
    --config_file ~/input/.../mi3xx_atom_deepseek-r1_fp8_single.json \
    --config_profile mtp3 \
    --cluster_file ~/input/cluster_file/atom_cluster.json

Keys prefixed with ``_`` (for example ``_comment``) are ignored by the loader.

Cluster and lab setup
=====================

**Cluster file:** ``cvs/input/cluster_file/atom_cluster.json``. Edit
``node_dict`` so host count matches ``params.nnodes``.

.. list-table::
   :widths: 3 2 3
   :header-rows: 1

   * - Variant type
     - ``params.nnodes``
     - ``node_dict``
   * - Single-node / baseline / MTP3
     - ``1``
     - Head node only
   * - Multinode PP
     - ``2``
     - Head + worker

Set ``enforce_thresholds`` to ``true`` for PASS/FAIL gating or ``false`` for
record-only (MI355X stems often ship record-only until lab calibration).

Fields you must customize
-------------------------

.. list-table::
   :widths: 3 4
   :header-rows: 1

   * - Where
     - Change to
   * - ``container.image``
     - Your ATOM ROCm image on GPU nodes
   * - ``container.runtime.args.volumes`` (``<changeme-models-mount>``)
     - Host directory holding HF cache / weights; mounted read-only at ``/models``
   * - ``paths.shared_fs``, ``paths.log_dir``, ``paths.hf_token_file``
     - Lab paths; ``{user-id}`` resolves to cluster username
   * - ``model.id``
     - Model under test
   * - ``params.driver``
     - ``atom`` (single-node) or ``vllm_atom`` / ``sglang`` (multinode PP)
   * - ``params.nnodes``, ``params.master_addr``, ``params.pipeline_parallel_size``
     - Match topology (``2`` / head IP / ``2`` on distributed PP stems)
   * - ``params.scaling_baseline_output_throughput``
     - Single-node output tok/s for ``scaling.efficiency_pct`` (multinode)
   * - ``roles.server.atom_args``
     - Inline ATOM server CLI when ``driver=atom``
   * - ``roles.server.serve_args``
     - Multinode server flags when ``driver=vllm_atom``
   * - ``roles.server.ib_hca_devices``, ``roles.server.ib_netdev``
     - ``"auto"`` or explicit; netdev must not be ``mlx5_*``
   * - Threshold JSON values
     - Calibrated PASS/FAIL bounds for your hardware

Placeholder substitution
========================

- ``{user-id}`` — cluster username (or local OS user fallback).
- ``{shared_fs}`` — self-reference within ``paths``.
- ``{paths.models_dir}`` and other ``{paths.*}`` — cross-referenced anywhere.
- ``<changeme-models-mount>`` — host path to the HF cache / weights tree; mounted at
  ``/models`` in the container. Shipped configs set ``paths.models_dir`` to ``/models``
  (the in-container path used for ``HF_HUB_CACHE``).
- ``{head-node-ip}`` — replace manually in copied multinode configs.

``threshold_json`` is a literal filename (no placeholder substitution).

Configuration schema
====================

Top-level fields:

.. list-table::
   :widths: 3 5
   :header-rows: 1

   * - Field
     - Meaning
   * - ``schema_version``
     - ``1`` (flat) or ``2`` (multi-profile)
   * - ``framework``
     - ``atom``
   * - ``gpu_arch``
     - Inferred from the config filename prefix (``{gpu}_atom_…``); optional override in JSON
   * - ``enforce_thresholds``
     - ``true`` = gate metrics; ``false`` = record-only
   * - ``threshold_json``
     - Sibling threshold filename
   * - ``paths``, ``model``, ``container``, ``roles``, ``params``, ``sweep``
     - See below

``params`` block
----------------

.. list-table::
   :widths: 3 5
   :header-rows: 1

   * - Field
     - Meaning
   * - ``driver``
     - ``atom``, ``vllm_atom``, ``sglang``, or interim ``vllm``
   * - ``tensor_parallelism``
     - TP size (W1: ``8``); appears as ``TP=`` in threshold keys
   * - ``pipeline_parallel_size``
     - PP size (``1`` single-node; ``2`` on distributed stems); ``PP=`` in keys
   * - ``nnodes``
     - Node count (not part of threshold cell key)
   * - ``master_addr`` / ``master_port``
     - Multinode PP coordinator
   * - ``num_prompts``, ``max_model_length``, ``random_range_ratio``
     - Benchmark client workload
   * - ``metric_percentiles``
     - Tail percentiles requested (e.g. ``95,99``)
   * - ``reuse_server_across_sweep``
     - Keep server warm across cells with matching session key
   * - ``scaling_baseline_output_throughput``
     - Baseline for ``scaling.efficiency_pct``
   * - ``server_*`` / ``client_*`` poll waits
     - Server ready and client completion timeouts

``sweep`` block
---------------

.. list-table::
   :widths: 3 5
   :header-rows: 1

   * - Field
     - Meaning
   * - ``sequence_combinations``
     - Named ``{name, isl, osl}`` shapes
   * - ``runs``
     - ``{combo, concurrency}`` — one benchmark cell each

Each cell's threshold key is built by :meth:`cvs.lib.inference.atom.atom_config_loader.AtomVariantConfig.cell_key`:

- Single-node: ``ISL=1024,OSL=1024,TP=8,PP=1,CONC=128``
- Multinode PP: ``ISL=1024,OSL=1024,TP=8,PP=2,CONC=128``

Always include ``PP=`` (use ``PP=1`` on single-node). ``params.nnodes`` is still
required for multinode runs but is **not** part of the threshold key.

Execution drivers
-----------------

.. list-table::
   :widths: 2 3 3 2
   :header-rows: 1

   * - Driver
     - When
     - Server
     - Native PP
   * - ``atom``
     - Single-node W1, baseline, MTP3
     - ``atom.entrypoints.openai_server``
     - No
   * - ``vllm_atom``
     - Shipped 2-node PP stems
     - Multinode serve + ATOM ROCm env
     - Yes

Multinode fabric is probed in ``test_discover_topology`` on the **cluster host
OS** (not inside the container).

Optional blocks
---------------

.. list-table::
   :widths: 3 5
   :header-rows: 1

   * - Block
     - Purpose
   * - ``functional.api_smoke`` / ``functional.health_check``
     - FUNC-1 / FUNC-2 before sweep
   * - ``platform.dmesg_scan`` / ``platform.gpu_metrics_poll``
     - INF-6 dmesg / INF-7 GPU metrics
   * - ``accuracy.tasks[]``
     - lm-eval tasks (thresholds under top-level ``accuracy`` key)
   * - ``long_context_accuracy.cells[]``
     - NIAH long-context cells (ACC-12)
   * - ``mtp_quality``
     - MTP acceptance checks (ACC-4/5/13)

Threshold files
===============

A threshold file maps each **cell key** to ``{metric: spec}``. Perf metrics use
**bare names** (``output_throughput``, ``mean_ttft_ms``, ΓÇª). Multinode cells
may also gate ``scaling.efficiency_pct``. Non-sweep keys include ``accuracy``,
``mtp_quality``, and ``long_context_accuracy``.

Threshold kinds:

.. list-table::
   :widths: 2 3 4
   :header-rows: 1

   * - ``kind``
     - Passes when
     - Notes
   * - ``min`` / ``max``
     - ``actual >=`` / ``actual <= value``
     - e.g. ``success_rate``, ``failed``
   * - ``max_ms``
     - ``actual <= value``
     - Latency upper bound
   * - ``min_tok_s``
     - ``actual >= value``
     - Throughput lower bound
   * - ``within`` / ``min_ratio``
     - Tolerance or ratio vs reference
     - Needs extra fields
   * - ``info``
     - Always record
     - Calibrate later

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

When ``enforce_thresholds: true``, every member of
:data:`cvs.lib.inference.atom.atom_parsing.GATED_METRICS` needs a spec in each
gated cell. Metrics missing from the benchmark artifact are skipped at gate time.

Run deck comparison (optional)
==============================

Set these env vars or sibling files for render-only comparison panels (no effect
on pytest gates):

.. list-table::
   :widths: 3 2 5
   :header-rows: 1

   * - Env / file
     - Panel
     - Purpose
   * - ``CVS_INFERENCE_PREV_REPORT_JSON``
     - ``panels.prev_run``
     - Per-cell throughput delta vs baseline
   * - Baseline ``accuracy`` block + same env
     - ``panels.accuracy_prev_run``
     - gsm8k flexible-extract delta
   * - ``CVS_ATOM_PARITY_REF_JSON``
     - ``panels.framework_parity``
     - Framework parity ratios

See also
========

- :doc:`/how-to/test-suites/inference/atom` — step-by-step first run
- :doc:`/reference/configuration-files/cluster-file` — cluster file schema
- :mod:`cvs.lib.inference.atom.atom_config_loader` — loader and ``cell_key``
- :mod:`cvs.lib.inference.atom.atom_parsing` — metric tiers and parsing
