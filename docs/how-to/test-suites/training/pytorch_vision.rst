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

The suites drive a vision training job (ResNet-50 for W1/W2, ViT-B/16 for W3,
ViT-base-patch16-224 for W6) inside a container on one or more
cluster nodes, then parse the rank-zero result artifact to produce metrics and
verdicts. They provide:

#. **Config-selected workload** — the model, FLOPs/image, optimizer, and
   scorecard workload id all come from the config, so the same suite code
   serves W1, W3, W6, and future vision rows without modification.
#. **Two training harnesses** — ``training.harness`` selects ``native`` (our
   own DDP loop, W1-W3) or ``hf_trainer`` (HuggingFace Trainer with FSDP, W6).
   Both emit the same result artifact so their scorecard rows stay comparable.
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
- **Run Deck** — ``pytest --html`` also writes ``pytorch_vision_run_deck.html``,
  ``.json``, and ``pytorch_vision_run_deck_viewer.html`` next to the pytest
  report. Both ``pytorch_vision_single`` and ``pytorch_vision_distributed``
  share ``cvs/lib/report/profiles/pytorch_vision.json``. Cells are the sweep
  names (model, precision, microbatch, gradient accumulation, resolution, and
  rocAL data path). The headline metric is images/s/GPU. Single-node runs omit
  scaling efficiency. Each ``test_metric`` row expands to the same collapsible
  pass/fail panel the other training suites use, and still links to
  ``metric_results.html``.
- **Loss curve on the deck** — sampled ``(step, loss)`` points are copied into
  the deck cell, so the viewer can draw loss against step. The PNG linked from
  ``test_loss_curve`` is unchanged.

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

FSDP sharding strategy
======================

``training.fsdp_strategy`` selects ``full_shard`` (the default, parameters
spread over every rank) or ``hybrid_shard`` (shard within a node, replicate
across nodes). For a model as small as ViT-base the choice dominates multi-node
throughput: on four nodes W6 measured 11,616 images/s with ``full_shard``
against 24,059 with ``hybrid_shard``, or 31.7% against 68.5% scaling efficiency.
``full_shard`` spends more on cross-node all-gathers than it recovers in
parallelism, to the point that two nodes are slower in absolute terms than one.
Prefer ``hybrid_shard`` for multi-node runs unless the model does not fit in a
single node's memory.


Sweep ordering and page-cache warming
=====================================

Sweeps in a profile run back to back against the same ImageNet shards on the
shared filesystem, so later sweeps read data the earlier ones have already
pulled into the page cache. For workloads whose step time is bounded by input
throughput this dominates the measurement: running W2's five sweeps in reverse
moved the same ``MBS=128`` rocAL-GPU sweep from 18,096 to 34,351 images/s, and
the ``MBS=256`` sweep from 21,997 to 47,266, purely by position. The
rocAL-CPU-decode sweep was unaffected (13,746 against 13,437) because its
bottleneck is decode rather than I/O, which is what identifies the cause.

The consequence is that cross-sweep comparisons within a loader-bound profile
are not trustworthy as measured: W2's accumulation, augmentation, and CPU-decode
overhead rows came out negative, implying the more expensive configuration was
faster. Treat them as unreliable rather than as results.

Compute-bound profiles are far less exposed. W3's five sweeps spread about 19%
against W2's 190%, and its overhead rows are plausible (1.65% accumulation,
9.64% CPU decode, 3.39% heavy augmentation).

``training.sweep_repeats`` addresses this. With a value above 1 the suite runs
every enabled sweep once per round, reshuffling the order each round with
``training.sweep_order_seed``, and reports the median per metric. Every sweep
therefore takes a turn in each cache state instead of one sweep always reading
cold. Repeating a sweep consecutively would not work, because it would leave
that sweep in the same cache state for all of its repeats.

W1 and W2's perf profiles use three repeats. On W2 this moved the CPU-decode
overhead from -24.4% to 23.1% and the accumulation overhead from -3.6% to 41.3%,
both of which are now the right sign and the right order of magnitude. The
augmentation row remains slightly negative at -5.7%, so it is closer to noise
than to a measurement; more repeats would tighten it further.


The ``hf_trainer`` harness (W6)
===============================

W6 runs ViT-base-patch16-224 through the HuggingFace ``Trainer`` with FSDP
rather than our own DDP loop. It differs from W3 in more than node count:

* **Image** — ``transformers``, ``accelerate``, and ``datasets`` are not in the
  rocAL image. W6 uses ``cvs/pytorch-vision-hf``, built by
  ``build_tools/pytorch_vision/Dockerfile.hf-7.2.4`` on top of the rocAL image
  so both harnesses share one ROCm/PyTorch baseline.
* **Data** — ``Trainer`` builds its own ``DataLoader`` from a torchvision
  ``ImageFolder``, so the rocAL pipeline is not used. The data loader
  throughput row is filled by timing that loader in isolation instead.
* **Sharding** — FSDP with ``transformer_layer_cls_to_wrap=["ViTLayer"]``,
  replacing the parameter replication DDP performs in W1-W3.
* **Checkpoints** — FSDP shards parameters, so save/resume parity is verified
  against a gathered ``FULL_STATE_DICT`` rather than each rank's local shard.
* **Weights** — ``sweep.pretrained`` selects the initialization. The smoke
  profile loads ``google/vit-base-patch16-224`` and reports real accuracy from
  the first step; the perf and ``train_5k`` profiles train the same
  architecture from scratch so the loss curve, loss milestones, and convergence
  rows carry information and stay comparable with W3. Fine-tuning a converged
  checkpoint makes those rows meaningless, which is why they are split across
  profiles rather than taken from a single run.

Scaling on this cluster is poor: ``full_shard`` spreads an 86M-parameter model
across every rank, so the cross-node all-gather traffic outweighs the added
compute and two nodes reach roughly half the ideal throughput. ``HYBRID_SHARD``
(shard within a node, replicate across nodes) is the usual remedy for a model
this size and is not yet evaluated here.

Because ``Trainer`` owns the loop, it also owns the optimizer and LR schedule.
The ``training.lr_schedule`` block and the learning-rate trajectory assertion
apply to the ``native`` harness only; for W6 the scheduler actually used is
recorded in the artifact as ``hf_lr_scheduler_type``.

