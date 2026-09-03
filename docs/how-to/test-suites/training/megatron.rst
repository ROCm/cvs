.. meta::
  :description: Run Megatron Llama and DeepSeek training benchmarks
  :keywords: CVS, megatron

***********************
Megatron training tests
***********************

Cluster validation that runs Megatron-LM or Primus pre-training on AMD Instinct GPUs (single-node or multi-node) and gates the run on performance and correctness metrics with a PASS/FAIL HTML report.

The suite drives a training job inside a Docker container on one or more cluster nodes, then parses the training log to produce metrics and verdicts. It provides:

- **Two suites** — ``megatron_single`` (single-node) and ``megatron_distributed`` (multi-node, adds RDMA/NIC setup).
- **Megatron-LM or Primus** — if ``container.image`` contains ``primus`` (case-insensitive), the suite uses Primus; otherwise Megatron-LM. Primus reads YAML from ``examples/megatron/configs/{gpu_arch}/`` inside the image.
- **Parameter sweeps** — one full training run per enabled combo (for example FP8 and BF16), each with its own result rows in the report.
- **Loss curve** — a per-combo decreasing-trend check on ``lm_loss`` at steps 100 / 500 / 1k / 5k.
- **Training-log error scanning** — NCCL, GPU HW faults, OOM, and other signatures fail a run early with a clear reason.
- **HTML report** — per-test rows with linked logs and a consolidated metric results page.

The mode (single vs distributed) is determined by the config file's ``framework`` field (``megatron_single`` or ``megatron_distributed``). Match the suite name to that field.

Prerequisites
=============

- Passwordless SSH from the control host to each cluster node (key in the cluster file) and Docker available on the nodes.
- A container image for ROCm (``container.image`` in the config). A Megatron-LM image must provide Megatron-LM at ``config.megatron_root`` (default ``/workspace/Megatron-LM``). A Primus image (name contains ``primus``) uses in-image YAML under ``examples/megatron/configs/{gpu_arch}/`` instead.
- A Hugging Face token file at ``config.hf_token_file`` (used to fetch the tokenizer). Tokenizer download requires network access on the nodes. For gated models (LLaMA, DeepSeek), model access must be granted on huggingface.co.
- For distributed runs: RDMA interfaces configured and reachable on all nodes; a shared filesystem path reachable from all nodes for logs and scripts.

.. _megatron-set-up-config:

Set up config
=============

1. List available training configuration files:

   .. code:: bash

     cvs config list training/megatron

2. Copy the configuration file and its sibling threshold file, for example:

   .. code:: bash

     cvs config copy training/megatron/mi325x_megatron_llama-3.1-8b_single.json --output ~/cvs_workspace/training/megatron/mi325x_megatron_llama-3.1-8b_single.json
     cvs config copy training/megatron/mi325x_megatron_llama-3.1-8b_single_threshold.json --output ~/cvs_workspace/training/megatron/mi325x_megatron_llama-3.1-8b_single_threshold.json

3. Replace every ``<changeme>`` with cluster-specific values (container image, NCCL/RDMA fields on distributed configs).
4. Change any other parameters relevant to your testing requirements.

The same folder also has DeepSeek V2 Lite (single and distributed) and Llama 3.1 405B (distributed only) configs for MI300X, MI325X, and MI355X. See `Config and threshold files`_ for the full inventory.

Full parameter list: :doc:`/reference/configuration-files/training/megatron`.

Primus vs Megatron-LM
=====================

The same eight test stages run for both backends. ``_make_training_job`` in ``megatron_single.py`` / ``megatron_distributed.py`` picks the job class from ``container.image`` (substring ``primus``, case-insensitive).

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * - Stage
     - Megatron-LM (image does not contain ``primus``)
     - Primus (image contains ``primus``)
   * - ``test_launch_container``
     - Launch the Docker container on all suite hosts
     - Same
   * - ``test_download_tokenizer``
     - Downloads ``tokenizer.model`` into ``data_cache_dir`` for DeepSeek/Mixtral. Llama/Qwen skip (HF repo ID is enough)
     - Always a no-op (``_needs_local_tokenizer()`` is false; Primus takes HF repo IDs in YAML)
   * - ``test_smoke``
     - Runs the cell from the config ``smoke`` block (default 10 iters) via the Megatron-LM training script under ``config.megatron_root``. Skipped when ``smoke.enabled`` is ``false``
     - Same cell via ``primus-cli direct -- train pretrain --config examples/megatron/configs/{gpu_arch}/{model}-{precision}-pretrain.yaml``. If the MI325X YAML is missing, Primus retries ``MI300X``. Skipped when ``smoke.enabled`` is ``false``
   * - ``test_checkpoint``
     - Skipped (``checkpoint test is Primus-only``)
     - Runs only when ``checkpoint.enforce`` is ``true``. Single-node writes under ``{log_dir}/ckpt_primus``. Distributed requires ``checkpoint.checkpoint_dir`` on a shared filesystem
   * - ``test_training[combo]``
     - Wrapper script + Megatron-LM shell. Distributed also runs ``exec_nic_setup_scripts()`` (Broadcom ``libbnxt_re`` copy when ``nic_type`` matches thor/broadcom)
     - Wrapper script + ``primus-cli``. Distributed sets ``NCCL_IB_*`` env vars; no Broadcom lib copy
   * - ``test_metric`` / ``test_loss_curve``
     - Parse Megatron-LM log metrics
     - Parse Primus log metrics (same ``training.*`` names)
   * - ``test_teardown``
     - Tear down the container
     - Same

Logs for both backends use per-node files: ``<log_dir>/{megatron-logs|primus-logs}/<combo_id>/out-node<N>/training.log``.

.. _megatron-run-tests:

Run tests
=========

You can list all available test stages using the CLI:

.. code:: bash

  cvs list megatron_single

.. code:: text

  Available tests in megatron_single:
    - test_launch_container
    - test_download_tokenizer
    - test_smoke
    - test_checkpoint
    - test_training
    - test_metric
    - test_loss_curve
    - test_teardown

.. code:: bash

  cvs list megatron_distributed

.. code:: text

  Available tests in megatron_distributed:
    - test_launch_container
    - test_download_tokenizer
    - test_smoke
    - test_checkpoint
    - test_training
    - test_metric
    - test_loss_curve
    - test_teardown

Use a single-node config with ``megatron_single`` and a distributed config with ``megatron_distributed``. The config's ``framework`` field must match the suite.

- ``--cluster_file`` — JSON describing the node(s); see :doc:`/how-to/configure/cluster-config`.
- ``--config_file`` — one of the files under ``input/config_file/training/megatron/``; field reference: :doc:`/reference/configuration-files/training/megatron`.
- ``--html`` / ``--self-contained-html`` — write the HTML report.

Single-node — MI300X
~~~~~~~~~~~~~~~~~~~~

.. code:: bash

  cvs run megatron_single \
    --cluster_file input/cluster_file/cluster.json \
    --config_file input/config_file/training/megatron/mi300x_megatron_llama-3.1-8b_single.json \
    --html ./logs/megatron_single.html --self-contained-html -vvv -s

Single-node — MI325X
~~~~~~~~~~~~~~~~~~~~

.. code:: bash

  cvs run megatron_single \
    --cluster_file input/cluster_file/cluster.json \
    --config_file input/config_file/training/megatron/mi325x_megatron_llama-3.1-8b_single.json \
    --html ./logs/megatron_single.html --self-contained-html -vvv -s

Single-node — MI355X
~~~~~~~~~~~~~~~~~~~~

.. code:: bash

  cvs run megatron_single \
    --cluster_file input/cluster_file/cluster.json \
    --config_file input/config_file/training/megatron/mi355x_megatron_llama-3.1-8b_single.json \
    --html ./logs/megatron_single.html --self-contained-html -vvv -s

Distributed — MI300X
~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

  cvs run megatron_distributed \
    --cluster_file input/cluster_file/cluster.json \
    --config_file input/config_file/training/megatron/mi300x_megatron_llama-3.3-70b_distributed.json \
    --html ./logs/megatron_distributed.html --self-contained-html -vvv -s

Distributed — MI325X
~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

  cvs run megatron_distributed \
    --cluster_file input/cluster_file/cluster.json \
    --config_file input/config_file/training/megatron/mi325x_megatron_llama-3.3-70b_distributed.json \
    --html ./logs/megatron_distributed.html --self-contained-html -vvv -s

Distributed — MI355X
~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

  cvs run megatron_distributed \
    --cluster_file input/cluster_file/cluster.json \
    --config_file input/config_file/training/megatron/mi355x_megatron_llama-3.3-70b_distributed.json \
    --html ./logs/megatron_distributed.html --self-contained-html -vvv -s

Run a specific stage
~~~~~~~~~~~~~~~~~~~~

.. code:: bash

  cvs run megatron_single test_smoke \
    --cluster_file input/cluster_file/cluster.json \
    --config_file input/config_file/training/megatron/mi325x_megatron_llama-3.1-8b_single.json

Test lifecycle
==============

Tests run in this pinned order. ``[combo]`` = one row per enabled sweep combo.

.. list-table::
   :header-rows: 1
   :widths: 8 28 12 52

   * - Order
     - Test
     - Runs on
     - Purpose
   * - 0
     - ``test_launch_container``
     - once
     - Launch and verify the container
   * - 1
     - ``test_download_tokenizer``
     - once
     - Megatron-LM: download ``tokenizer.model`` for DeepSeek/Mixtral. Primus: always skip (HF repo IDs).
   * - 2
     - ``test_smoke``
     - once
     - Small run from the ``smoke`` config block confirming the model loads and trains without error. Skipped when ``smoke.enabled`` is ``false``
   * - 3
     - ``test_checkpoint``
     - once
     - Primus-only checkpoint save + resume with loss continuity at the first resume step. Skipped when ``checkpoint.enforce: false`` or the image name does not contain ``primus``.
   * - 4
     - ``test_training[combo]``
     - per combo
     - Build cmd, train, poll logs, parse results; GPU memory freed between combos
   * - 5
     - ``test_metric[combo]``
     - per combo
     - Threshold PASS/FAIL per metric
   * - 6
     - ``test_loss_curve[combo]``
     - per combo
     - Gate on downward ``lm_loss`` trend at steps 100 / 500 / 1k / 5k
   * - 7
     - ``test_teardown``
     - once
     - Tear the container down

A training failure is isolated to that combo's ``test_training`` row; other combos still run. When a combo's training does not complete, its downstream ``test_metric`` and ``test_loss_curve`` rows are skipped. If an early lifecycle stage fails, all subsequent stages are skipped via ``lifecycle.failed``.

On a training failure, lingering GPU processes are killed (``stop_training_processes``) so the next combo does not launch on top of them.

Sweeps
======

A sweep combo is one full training run declared in ``sweep.combinations``. ``sweep.runs`` is the ordered list of combo IDs to execute; set it to a subset to run only selected combos without editing ``combinations``.

The combo ID (for example ``llama3_1_8b-mi325x-bs128-mbs4-fp8``) appears in every parametrized row: ``test_training[...]``, ``test_metric[...]``, and ``test_loss_curve[...]``.

Metrics and PASS/FAIL
=====================

Each ``test_metric[combo]`` compares the parsed metric against its threshold spec and reports one of:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Status
     - Meaning
   * - PASS
     - value satisfies the threshold
   * - FAIL
     - value violates the threshold (row is red; aggregated in the summary)
   * - RECORD
     - no threshold defined, or ``enforce_thresholds: false`` — value logged, not gated

Metrics surfaced (namespace ``training.*``):

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Metric
     - Description
   * - ``training.throughput_per_gpu``
     - TFLOP/s per GPU
   * - ``training.tokens_per_gpu``
     - Tokens per GPU per second
   * - ``training.elapsed_time_per_iteration``
     - Wall time per training step (ms)
   * - ``training.mem_usage``
     - GPU memory usage
   * - ``training.scaling_efficiency_pct``
     - Multi-node scaling efficiency % vs single-node baseline (distributed only)

Gating requires ``enforce_thresholds: true`` in the config. Set to ``false`` for record-only runs.

Scaling efficiency (distributed only)
=====================================

``test_training`` computes scaling efficiency as:

.. code:: text

  efficiency % = (actual_total_tok/s / (actual_nodes / baseline_nodes)) / baseline_total_tok/s × 100

Populate ``scaling_baseline.tokens_per_sec_total`` in the config from a completed single-node run (``tok/s/GPU × 8``). Set to ``0.0`` to disable and collect data only.

Loss curve
==========

``test_loss_curve[combo]`` fits a least-squares line to ``lm_loss`` samples collected at ``loss_curve.milestone_steps`` (plus every ``loss_curve.sample_every`` steps) and passes when the slope is below ``loss_curve.max_slope``. A value of ``0.0`` means any downward trend passes.

Set ``loss_curve.enforce: false`` to record the slope without gating.

Convergence
===========

``test_metric[combo]`` also reports a convergence check when ``convergence.target_value > 0``. It compares the final training loss (or eval loss when eval runs) against ``convergence.target_value``. Set ``target_value <= 0`` to disable (record-only). ``target_metric: "auto"`` selects eval loss when available, otherwise training loss.

Checkpoint save and resume
==========================

``test_checkpoint`` runs only for Primus images (``container.image`` contains ``primus``) and only when ``checkpoint.enforce`` is ``true``. Megatron-LM images skip this stage. When it runs, the suite launches the training job twice:

1. **Save phase** — trains to ``checkpoint.save_iters`` steps, saving a checkpoint every ``checkpoint.save_interval`` steps. Single-node uses ``{log_dir}/ckpt_primus``. Distributed uses ``checkpoint.checkpoint_dir`` (required; must be a shared path such as NFS).
2. **Resume phase** — resumes from the saved checkpoint and trains to ``checkpoint.resume_iters`` steps.

The suite passes when the loss at the first resume step (``last_ckpt_step + 1``) does not exceed the checkpoint-step loss by more than ``checkpoint.loss_rtol`` (relative tolerance).

Training-log error detection
============================

During polling, each node's ``training.log`` is scanned for known error patterns. Defaults cover:

- NCCL errors and timeouts
- GPU hardware faults and hangs
- PyTorch distributed errors

A match fails that combo's ``test_training`` with the matched pattern name and the last lines of the log.

Reports and logs
================

- **Results table** — one row per test; metric rows show PASS/FAIL from the threshold check.
- **Full log** — each test row links to its own captured log.
- **Training logs** — Megatron-LM writes ``<log_dir>/megatron-logs/<combo_id>/out-node<N>/training.log``. Primus writes ``<log_dir>/primus-logs/<combo_id>/out-node<N>/training.log``.

Log path fields:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Placeholder
     - Source
   * - ``<log_dir>``
     - ``config.log_dir`` in the config file
   * - ``<combo_id>``
     - Sweep run ID (for example ``llama3_1_8b-mi325x-bs128-mbs4-fp8``)
   * - ``out-node<N>``
     - One directory per node; ``out-node0`` for single-node

Config and threshold files
==========================

Located in ``cvs/input/config_file/training/megatron/``. Field-level schema: :doc:`/reference/configuration-files/training/megatron`.

Do not use leftover ``mi3xx_megatron_llama_*.json`` / ``mi35x_megatron_llama_single.json`` files with these suites. Those nested configs belong only to the legacy ``megatron_llama3_1_*`` test modules.

MI300X
~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 45 45 10

   * - Config
     - Threshold
     - Mode
   * - ``mi300x_megatron_deepseek-v2-lite_single.json``
     - ``mi300x_megatron_deepseek-v2-lite_single_threshold.json``
     - single-node
   * - ``mi300x_megatron_deepseek-v2-lite_distributed.json``
     - ``mi300x_megatron_deepseek-v2-lite_distributed_threshold.json``
     - distributed
   * - ``mi300x_megatron_llama-3.1-8b_single.json``
     - ``mi300x_megatron_llama-3.1-8b_single_threshold.json``
     - single-node
   * - ``mi300x_megatron_llama-3.1-8b_distributed.json``
     - ``mi300x_megatron_llama-3.1-8b_distributed_threshold.json``
     - distributed
   * - ``mi300x_megatron_llama-3.1-405b_distributed.json``
     - ``mi300x_megatron_llama-3.1-405b_distributed_threshold.json``
     - distributed
   * - ``mi300x_megatron_llama-3.3-70b_single.json``
     - ``mi300x_megatron_llama-3.3-70b_single_threshold.json``
     - single-node
   * - ``mi300x_megatron_llama-3.3-70b_distributed.json``
     - ``mi300x_megatron_llama-3.3-70b_distributed_threshold.json``
     - distributed

MI325X
~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 45 45 10

   * - Config
     - Threshold
     - Mode
   * - ``mi325x_megatron_deepseek-v2-lite_single.json``
     - ``mi325x_megatron_deepseek-v2-lite_single_threshold.json``
     - single-node
   * - ``mi325x_megatron_deepseek-v2-lite_distributed.json``
     - ``mi325x_megatron_deepseek-v2-lite_distributed_threshold.json``
     - distributed
   * - ``mi325x_megatron_llama-3.1-8b_single.json``
     - ``mi325x_megatron_llama-3.1-8b_single_threshold.json``
     - single-node
   * - ``mi325x_megatron_llama-3.1-8b_distributed.json``
     - ``mi325x_megatron_llama-3.1-8b_distributed_threshold.json``
     - distributed
   * - ``mi325x_megatron_llama-3.1-405b_distributed.json``
     - ``mi325x_megatron_llama-3.1-405b_distributed_threshold.json``
     - distributed
   * - ``mi325x_megatron_llama-3.3-70b_single.json``
     - ``mi325x_megatron_llama-3.3-70b_single_threshold.json``
     - single-node
   * - ``mi325x_megatron_llama-3.3-70b_distributed.json``
     - ``mi325x_megatron_llama-3.3-70b_distributed_threshold.json``
     - distributed

MI355X
~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 45 45 10

   * - Config
     - Threshold
     - Mode
   * - ``mi355x_megatron_deepseek-v2-lite_single.json``
     - ``mi355x_megatron_deepseek-v2-lite_single_threshold.json``
     - single-node
   * - ``mi355x_megatron_deepseek-v2-lite_distributed.json``
     - ``mi355x_megatron_deepseek-v2-lite_distributed_threshold.json``
     - distributed
   * - ``mi355x_megatron_llama-3.1-8b_single.json``
     - ``mi355x_megatron_llama-3.1-8b_single_threshold.json``
     - single-node
   * - ``mi355x_megatron_llama-3.1-8b_distributed.json``
     - ``mi355x_megatron_llama-3.1-8b_distributed_threshold.json``
     - distributed
   * - ``mi355x_megatron_llama-3.1-405b_distributed.json``
     - ``mi355x_megatron_llama-3.1-405b_distributed_threshold.json``
     - distributed
   * - ``mi355x_megatron_llama-3.3-70b_single.json``
     - ``mi355x_megatron_llama-3.3-70b_single_threshold.json``
     - single-node
   * - ``mi355x_megatron_llama-3.3-70b_distributed.json``
     - ``mi355x_megatron_llama-3.3-70b_distributed_threshold.json``
     - distributed

Legacy suite names
==================

Earlier releases shipped per-model suite names such as ``megatron_llama3_1_8b_single``,
``megatron_llama3_1_70b_distributed``, and ``megatron_llama3_1_8b_distributed``.
These suites are superseded by the unified ``megatron_single`` and ``megatron_distributed``
suites documented above. Use the unified suites for all new deployments.
