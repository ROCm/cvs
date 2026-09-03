.. meta::
  :description: Run JAX MaxText training benchmarks
  :keywords: CVS, jax, jaxmaxtext, MaxText

**********************
JAX MaxText tests
**********************

JAX training in CVS is **jaxmaxtext**. The legacy ``jax_llama3_1_*`` suites have
been removed. Use ``jaxmaxtext_single`` with a single-node config
(``training.distributed: false``) and ``jaxmaxtext_distributed`` with a
distributed config (``training.distributed: true``). Copy the matching
``*_threshold.json`` into the same directory as the suite config.

.. _jax-set-up-config:

Set up config
=============

1. List available JAX MaxText configuration files:

   .. code:: bash

     cvs config list training/jaxmaxtext

2. Copy the configuration file and its sibling threshold file, for example:

   .. code:: bash

     cvs config copy training/jaxmaxtext/mi300x_jaxmaxtext_llama-3.3-70b_single.json --output ~/cvs_workspace/training/jaxmaxtext/mi300x_jaxmaxtext_llama-3.3-70b_single.json
     cvs config copy training/jaxmaxtext/mi300x_jaxmaxtext_llama-3.3-70b_single_threshold.json --output ~/cvs_workspace/training/jaxmaxtext/mi300x_jaxmaxtext_llama-3.3-70b_single_threshold.json

3. Replace every ``<changeme>`` with cluster-specific values (especially NCCL/RDMA fields on distributed configs).
4. Change any other parameters relevant to your testing requirements.

Full parameter list: :doc:`/reference/configuration-files/training/jaxmaxtext`.

.. _jax-run-tests:

Run tests
=========

JAX MaxText test scripts
------------------------

You can list all available JAX MaxText test cases using the CLI:

.. code:: bash

  cvs list jaxmaxtext_single

.. code:: text

  Available tests in jaxmaxtext_single:
    - test_launch_container
    - test_setup_tokenizer
    - test_smoke
    - test_training_run
    - test_metric
    - test_loss_curve
    - test_checkpoint_resume
    - test_print_results_table
    - test_teardown

.. code:: bash

  cvs list jaxmaxtext_distributed

.. code:: text

  Available tests in jaxmaxtext_distributed:
    - test_launch_container
    - test_setup_rdma
    - test_setup_tokenizer
    - test_smoke
    - test_training_run
    - test_metric
    - test_loss_curve
    - test_checkpoint_resume
    - test_print_results_table
    - test_teardown

Use these scripts to run the JAX MaxText tests.

Single-node:

.. code:: bash

  cvs run jaxmaxtext_single --cluster_file input/cluster_file/cluster.json --config_file input/config_file/training/jaxmaxtext/mi300x_jaxmaxtext_llama-3.3-70b_single.json --html=/var/www/html/cvs/jaxmaxtext_single.html --capture=tee-sys --self-contained-html --log-file=/tmp/jaxmaxtext_single.log -vvv -s

Distributed:

.. code:: bash

  cvs run jaxmaxtext_distributed --cluster_file input/cluster_file/cluster.json --config_file input/config_file/training/jaxmaxtext/mi325x_jaxmaxtext_llama-3.3-70b_distributed.json --html=/var/www/html/cvs/jaxmaxtext_distributed.html --capture=tee-sys --self-contained-html --log-file=/tmp/jaxmaxtext_distributed.log -vvv -s

Use a single-node config (``training.distributed: false``) with
``jaxmaxtext_single`` and a distributed config (``training.distributed: true``)
with ``jaxmaxtext_distributed`` — the flag must match the suite.

Prerequisites
=============

- Passwordless SSH from the control host to each node (key in the cluster file),
  and Docker available on the nodes.
- A container image bundling MaxText/JAX for ROCm (config ``container.image``).
- A Hugging Face token file at ``paths.hf_token_file`` (used to fetch the
  tokenizer; needs network access on the nodes). Skipped when every enabled run
  uses ``dataset_type: synthetic``.
- A shared filesystem (``paths.shared_fs``) reachable from all nodes for the
  models cache and logs.

Test lifecycle
==============

The tests run in this fixed order. ``[sweep]`` = one row per enabled sweep;
``[sweep-metric]`` = one row per metric per sweep.

.. list-table::
   :widths: 1 3 2 5
   :header-rows: 1

   * - Order
     - Test
     - Runs on
     - Purpose
   * - 1
     - ``test_launch_container``
     - once
     - Launch and verify the container (and check out ``maxtext_branch`` if set).
   * - 2
     - ``test_setup_rdma``
     - distributed only
     - Copy the RDMA lib into the container (thor2 NIC) and verify ``ibv_devinfo``.
   * - 3
     - ``test_setup_tokenizer``
     - once
     - Download the HF tokenizer. Skipped when every enabled run is ``dataset_type: synthetic``.
   * - 4
     - ``test_smoke``
     - once
     - Small fixed run (BF16, few steps): the model loads and trains with no error/NaN signature. Enabled by default; a failure gates the rest of the suite. Skip with ``training.smoke.enabled: false`` or ``-k "not smoke"``.
   * - 5
     - ``test_training_run[sweep]``
     - per sweep
     - Build the command, train, poll, and parse results.
   * - 6
     - ``test_metric[sweep-metric]``
     - per sweep × metric
     - Threshold PASS/FAIL per metric.
   * - 7
     - ``test_loss_curve[sweep]``
     - per sweep
     - Render the loss PNG and gate on a downward trend.
   * - 8
     - ``test_checkpoint_resume``
     - once
     - Opt-in (``training.checkpoint_resume.enabled``): checkpoint save+resume correctness + I/O timing; skipped when disabled.
   * - 9
     - ``test_print_results_table``
     - once
     - Console tables + metric-results HTML + failure summary.
   * - 10
     - ``test_teardown``
     - once
     - Tear the container down.

A training failure is isolated to that sweep's ``test_training_run`` row; other
sweeps still run, and the failed sweep's downstream ``test_metric`` /
``test_loss_curve`` rows are skipped. Lingering ranks are killed before the next
sweep launches.

Sweeps
======

A **sweep** is one full training run with per-run MaxText overrides, declared in
``training.sweeps`` and selected with ``training.enabled_sweep_list``. The sweep
``name`` is also the threshold cell key. Each sweep gets a compact label derived
from its name (``PRECISION[-SL<seqlen>][-B<batch>]``, e.g. ``BF16-SL8192-B3``)
that appears in every parametrized row.

Metrics and PASS/FAIL
=====================

Each ``test_metric[sweep-metric]`` compares the parsed metric against its spec
in the sweep's cell of the threshold file:

.. list-table::
   :widths: 2 5
   :header-rows: 1

   * - Status
     - Meaning
   * - PASS
     - Value satisfies the threshold.
   * - FAIL
     - Value violates the threshold (row is red; aggregated in the summary).
   * - N/A
     - Metric not produced this run (feature disabled / rampup) — not a failure.
   * - RECORD
     - No threshold, or ``enforce_thresholds: false`` — value logged, not gated.

Metrics use the ``training.`` namespace (``tflops_per_sec_per_gpu``,
``tokens_per_sec_per_gpu``, ``tokens_per_sec_total``, ``scaling_efficiency_pct``,
``step_time_*``, ``final_loss``, ``loss_decreased``, ``eval_loss``,
``steps_to_target``, ``time_to_target_seconds``). Gating requires
``enforce_thresholds: true``; an ``info`` threshold always passes (record-only).
See :doc:`/reference/configuration-files/training/jaxmaxtext` for threshold kinds
and the full metric list.

Reports and logs
================

- **Results table** — one row per test; metric rows show PASS/FAIL.
- **Full Log** — each test row links to its own captured log.
- **Metric Results** — every ``test_metric`` row links to a shared
  ``metric_results.html`` (Sweep | Metric | Expected | Actual | Unit | Status).
- **Loss Curve** — each ``test_loss_curve`` row links to a per-sweep PNG.
- **Console summary** — ``test_print_results_table`` prints per-sweep tables and
  an aggregated list of failed ``(sweep, metric)`` checks.

Training-log error detection
============================

During polling, each node's ``training.log`` is scanned for the regexes in
``training.error_patterns`` (config-driven; falls back to built-in defaults
covering NCCL, GPU HW faults, assertion/JAX stack traces, ROCm init errors,
Python fatal errors, TF coordination errors, ``RESOURCE_EXHAUSTED``/OOM, and
segfaults). Fatal Python crashes (tracebacks / import errors) are always scanned
so an early failure fails fast instead of running to the poll timeout. A match
fails that sweep's ``test_training_run`` with the matched signature name.
