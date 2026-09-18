.. meta::
  :description: Run the PyTorch Vision training CVS test suites
  :keywords: CVS, PyTorch, torchvision, ResNet-50, rocAL, ImageNet, training

**************************
PyTorch Vision training
**************************

``pytorch_vision_single`` and ``pytorch_vision_distributed`` run public
torchvision training workloads in an AMD ROCm PyTorch container and gate the
run on performance and correctness metrics with a PASS/FAIL HTML report.

Overview
========

The suites drive a torchvision training job (ResNet-50 for W1, ViT-B/16 for
W3) inside a container on one or more
cluster nodes, then parse the rank-zero result artifact to produce metrics and
verdicts. They provide:

#. **Config-selected workload** — the model, FLOPs/image, optimizer, and
   scorecard workload id all come from the config, so the same suite code
   serves W1, W3, and future vision rows without modification.
#. **Two suites** — ``pytorch_vision_single`` (one node) and
   ``pytorch_vision_distributed`` (two or more nodes, one ``torchrun`` rank
   group per node).
#. **Parameter sweeps** — one full training run per enabled sweep (microbatch
   size, gradient accumulation, rocAL CPU vs GPU decode, standard vs heavy
   augmentation), each with its own result rows.
#. **Metric gating** — per-sweep, per-metric PASS/FAIL against a threshold
   file.
#. **Loss curve** — a per-sweep training-loss PNG plus a decreasing-trend
   check.
#. **Training-log error scanning** — configurable regex signatures (NCCL, GPU
   HW, OOM, segfault) fail a run early with a clear reason.
#. **HTML report + console summary** — per-test rows, a consolidated metric
   results page, per-sweep loss curves, and the full metric table.

The mode (single vs distributed) is reflected in the suite name, the metric
results HTML title, and the loss-curve titles and artifact names.

Quick start
===========

Single-node run:

.. code-block:: bash

   cvs run pytorch_vision_single \
     --cluster_file ./mi325x_1n_cluster.json \
     --config_file cvs/input/config_file/training/pytorch_vision/mi325x_pytorch_vision_resnet50_single_perf_config.json \
     --html ./logs/pytorch_vision_single.html --self-contained-html --capture=tee-sys

Distributed (multi-node) run:

.. code-block:: bash

   cvs run pytorch_vision_distributed \
     --cluster_file ./mi325x_2n_cluster.json \
     --config_file cvs/input/config_file/training/pytorch_vision/mi325x_pytorch_vision_resnet50_distributed_perf_config.json \
     --html ./logs/pytorch_vision_distributed.html --self-contained-html --capture=tee-sys

Use a **single-node** config with ``pytorch_vision_single`` and a
**distributed** config with ``pytorch_vision_distributed``. The config's
``training.distributed`` flag must match the suite and the cluster node count,
or the run fails immediately — a distributed config launched under the
single-node suite would otherwise train on one node and report it as a scaled
result.

The two suites
==============

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Suite (``cvs run <name>``)
     - File
     - Use with
   * - ``pytorch_vision_single``
     - ``pytorch_vision_single.py``
     - single-node config (``distributed: false``), exactly one cluster node
   * - ``pytorch_vision_distributed``
     - ``pytorch_vision_distributed.py``
     - multi-node config (``distributed: true``), two or more cluster nodes

Both suites share their implementations from ``_common.py``; sweep
parametrization and all fixtures and hooks live in ``conftest.py``.
``_common.py`` and ``conftest.py`` are helpers, not runnable suites.

Test lifecycle (report rows)
============================

Tests run in this pinned order. ``[sweep]`` = one row per enabled sweep;
``[sweep-metric]`` = one row per metric per sweep.

.. list-table::
   :header-rows: 1
   :widths: 8 35 20 37

   * - Order
     - Test
     - Runs on
     - Purpose
   * - 1
     - ``test_launch_container``
     - once
     - Launch and verify the container on every node
   * - 2
     - ``test_verify_environment``
     - once
     - Verify torch, torchvision, HIP, GPU count, and GPU architecture
   * - 3
     - ``test_real_data_smoke[sweep]``
     - smoke profiles
     - Prove real ImageNet flows end to end before longer profiles are trusted
   * - 4
     - ``test_training[sweep]``
     - per sweep
     - Stage, train, scan the log, and parse the result artifact
   * - 5
     - ``test_rocal_overhead_comparisons``
     - once
     - Require a GPU-standard baseline for every CPU-decode and heavy-augmentation sweep
   * - 6
     - ``test_metric[sweep-metric]``
     - per sweep x metric
     - Threshold PASS/FAIL per metric
   * - 7
     - ``test_loss_curve[sweep]``
     - per sweep
     - Render loss PNG; gate on downward trend
   * - 8
     - ``test_convergence[sweep]``
     - per sweep
     - Steps and wall-clock time to the configured accuracy target
   * - 9
     - ``test_print_results_table``
     - once
     - Console table + metric results HTML
   * - 10
     - ``test_teardown``
     - once
     - Tear the container down

On a training failure or timeout, lingering ranks are killed so the next sweep
does not launch on top of them. A training failure is isolated to that sweep's
``test_training`` row; other sweeps still run, and that sweep's downstream
``test_metric`` and ``test_loss_curve`` rows are skipped.

Multi-node topology
===================

``torchrun`` is launched once per node, each with its own ``--node-rank``,
because a single broadcast command cannot carry per-node identity. The first
host in the cluster file is the rendezvous endpoint on
``training.master_port``; single-node runs use ``--standalone`` instead and
reserve no port.

``NNODES`` and ``NODE_RANK`` are exported by the launcher after any
``training.env_vars``, so a single-node pair left in a config cannot silently
collapse a multi-node launch to one rank.

The dataset path must resolve identically on every node, and rank zero owns the
result artifact.

The node count is never declared in the config. Sweep names encode only
per-device dimensions, so one distributed config runs unchanged on 2, 4, or
more nodes; ``world_size`` and the effective global batch are derived as
``gpus_per_node x nodes`` at run time and cross-checked against the artifact.
This mirrors jaxmaxtext, whose sweep keys likewise omit topology.

Energy is measured by CodeCarbon on rank zero's node only, so on a multi-node
run ``energy_kwh`` and ``images/kWh`` are per-node figures; images are scaled
to that node's share so the ratio stays comparable across topologies.

Known limitation: unlike jaxmaxtext there is no RDMA setup stage, so the suite
relies on whatever transport RCCL negotiates. A 4-node run spanning two
subnets (10.32.80.x plus 10.32.81.x) hit a collective timeout, while the same
config on four same-subnet nodes passed. Keep multi-node sets on one fabric
until an RDMA stage is added.

Scaling efficiency needs a calibrated reference. Set
``training.scaling_baseline.images_per_sec_total`` from a prior single-node
run's ``results.json``, and ``num_nodes`` to that run's node count, before the
metric appears. Left at ``0.0`` it is omitted rather than reported as a
misleading zero, so a distributed run never claims an efficiency it cannot
substantiate.

Evaluation label correctness
============================

Every evaluation records a per-class label histogram as
``eval_label_classes_observed`` plus ``eval_label_min_per_class`` and
``eval_label_max_per_class``, and fails outright on any label outside
``[0, num_classes)``. Sample count alone cannot detect a bad label mapping: a
truncated or scrambled label space still totals the right number of images.

On a full ImageNet-1k validation set a correct pass reports 1000 classes with
min == max == 50.

The evaluation batch size is chosen automatically as the largest divisor of
the per-rank shard that does not exceed the training batch size, so evaluation
never leaves a partial batch. This matters because rocAL's default
``LAST_BATCH_FILL`` pads a partial final batch by repeating samples, and those
repeats land in the accuracy numbers. Measured over 8 shards at batch 128
(6,250 per shard): ``FILL`` yielded 50,176 samples with 50-72 images per class,
``LAST_BATCH_PARTIAL`` 48,976 with 16 classes missing entirely, and
``LAST_BATCH_DROP`` 48,128 with 32 missing. Sizing the batch to divide exactly
yields 50,000 with a uniform 50 per class. Evaluation throughput is not a
reported metric, so the batch size costs nothing.

Threshold files for the full-evaluation profiles gate
``eval_label_classes_observed`` at 1000 so a regression in this path fails the
run rather than quietly skewing accuracy. Smoke profiles cap evaluation with
``eval_steps`` and therefore cannot observe every class, so they record the
value without gating it.

Metrics and PASS/FAIL
=====================

Each ``test_metric[sweep-metric]`` compares the parsed metric against its
threshold spec in the sweep's cell of the threshold file and reports one of:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Status
     - Meaning
   * - PASS
     - value satisfies the threshold
   * - FAIL
     - value violates the threshold
   * - RECORD
     - no threshold, or ``enforce_thresholds`` is false — value logged, not gated

Metrics surfaced (namespace ``training.*``): ``images_per_sec``,
``images_per_sec_per_gpu``, ``tflops_per_sec_per_gpu``, ``mfu_pct``,
``step_time_ms_{mean,p50,p95}``,
``peak_memory_{allocated,reserved,used}_mb``,
``data_loader_images_per_sec``, ``checkpoint_{save,load}_seconds``,
``checkpoint_state_match``, ``checkpoint_loss_delta``,
``gradient_accumulation_overhead_pct``, ``augmentation_overhead_pct``,
``rocal_cpu_overhead_pct``, ``loss_{initial,final}``,
``loss_step_{100,500,1000,5000}``, ``top1_accuracy_pct``,
``top5_accuracy_pct``, ``eval_loss``, ``eval_sample_count``,
``convergence_{step,time_seconds}``, ``energy_kwh``, ``images_per_kwh``.

Gating is threshold-driven and requires ``enforce_thresholds: true``. A
threshold entry with ``"kind": "info"`` always passes (record-only).

Reports and logs
================

- **Results table** — one row per test; metric rows show PASS/FAIL from the
  threshold check.
- **Metric Results** — every ``test_metric`` row links to a single shared
  ``metric_results.html`` (Sweep | Metric | Expected | Actual | Unit |
  Status), titled with the mode.
- **Loss Curve** — each ``test_loss_curve`` row links to and inlines a
  per-sweep PNG.
- **Result artifacts** — rank zero writes ``results.json`` and
  ``training.log`` below ``{paths.log_dir}/pytorch_vision/<sweep-label>/<run-id>/``.

Prerequisites
=============

- Passwordless SSH from the control host to each cluster node (key in the
  cluster file), and Docker available on the nodes.
- A container image bundling PyTorch and rocAL for ROCm
  (``container.image``). The public ``rocm/pytorch`` images do **not** ship
  rocAL; see ``build_tools/pytorch_vision/`` for the image build.
- ImageNet-1k reachable at the same path on every node, laid out as
  ``train/<label>/*.JPEG`` and ``val/<label>/*.JPEG``. CVS does not download
  it, because its distribution requires separate access terms.

See also :doc:`/how-to/run-tests/index` for common ``cvs run`` flags and
workflow, and
:doc:`/reference/configuration-files/training/pytorch_vision` for the
variable-by-variable config reference.
