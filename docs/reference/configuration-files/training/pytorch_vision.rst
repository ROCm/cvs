.. meta::
  :description: PyTorch Vision training configuration and threshold reference
  :keywords: CVS, PyTorch, torchvision, rocAL, ImageNet, config schema

**************************
PyTorch Vision training
**************************

The PyTorch Vision suites use the same configuration split as JAX MaxText:

- the cluster file contains SSH access and the selected node(s)
- ``*_config.json`` contains the container and training workload
- the sibling ``*_threshold.json`` contains per-sweep metric gates

Files live in ``cvs/input/config_file/training/pytorch_vision/`` and are named
``<gpu>_pytorch_vision_<model>_<mode>_<profile>_config.json`` with a matching
``_threshold.json``.

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Config
     - Purpose
   * - ``mi325x_pytorch_vision_resnet50_single_smoke_config.json``
     - 5-step real-data smoke check
   * - ``mi325x_pytorch_vision_resnet50_single_perf_config.json``
     - 50-step performance sweeps: GA, rocAL CPU vs GPU, standard vs heavy augmentation
   * - ``mi325x_pytorch_vision_resnet50_single_train5k_config.json``
     - 5,000-step loss-curve and accuracy profile
   * - ``mi325x_pytorch_vision_resnet50_single_to-accuracy_disabled_config.json``
     - Protected train-to-target-accuracy profile (disabled by default)
   * - ``mi325x_pytorch_vision_resnet50_single_synthetic_config.json``
     - Synthetic on-GPU input; measures model, optimizer, and DDP without storage
   * - ``mi325x_pytorch_vision_resnet50_distributed_perf_config.json``
     - Multi-node scaling pair for the single-node MBS=128 GPU-standard cell
   * - ``mi325x_pytorch_vision_vit-b-16_single_smoke_config.json``
     - W3 ViT-B/16 smoke check
   * - ``mi325x_pytorch_vision_vit-b-16_single_perf_config.json``
     - W3 ViT-B/16 50-step performance sweeps
   * - ``mi325x_pytorch_vision_vit-b-16_single_train5k_config.json``
     - W3 ViT-B/16 5,000-step loss-curve and accuracy profile

Top-level fields
================

- ``framework``: must be ``pytorch_vision_training``.
- ``gpu_arch``: exact device family expected inside the container.
- ``enforce_thresholds``: gates every configured metric when ``true``.
- ``threshold_json``: sibling threshold filename.
- ``paths``: shared model, log, and token locations.
- ``model``: public model identity; ``remote`` remains ``0`` because the suites
  use random torchvision weights.
- ``container``: lifecycle, image, and runtime mounts and devices.

Training block
==============

- ``workload``: scorecard workload id (``W1`` ResNet-50, ``W3`` ViT-B/16, ...).
  Recorded in the result artifact and cross-checked on parse, so a result
  cannot be attributed to the wrong scorecard row. It also scopes the staged
  script path, letting two workloads share a host without colliding.
- ``optimizer``: ``sgd`` (default) or ``adamw``. Transformer backbones do not
  train usefully under SGD at these learning rates; ``momentum`` is ignored
  when ``adamw`` is selected.
- ``distributed``: ``false`` for ``pytorch_vision_single``, ``true`` for
  ``pytorch_vision_distributed``. Must match the suite and the cluster node
  count.
- ``master_port``: rendezvous port on the first cluster host. Multi-node only;
  single-node runs use ``--standalone`` and reserve no port.
- ``enabled``: module-level safety switch. Disabled profiles skip before
  container launch.
- ``allow_long_run``: second explicit opt-in for target-accuracy training.
- ``phase``: ``smoke``, ``performance``, or ``accuracy``.
- ``run_mode``: ``smoke``, ``perf``, ``train_5k``, or ``train_to_accuracy``.
- ``gpus_per_node``: expected visible GPU count and ``torchrun`` rank count per
  node.
- ``steps``: measured optimizer steps.
- ``warmup_steps``: untimed optimizer steps before performance measurement.
  Accuracy profiles require zero, because those updates are not part of the
  declared epoch or step count.
- ``max_epochs``, ``max_duration_seconds``: safety caps for target-accuracy
  training.
- ``eval_enabled``, ``eval_every_epochs``, ``eval_steps``: streaming validation
  cadence and optional validation-step cap.
- ``milestone_steps``: optimizer steps whose losses are copied into the
  artifact.
- ``num_classes``, ``channels_last``: classifier outputs and NHWC-compatible
  tensor storage.
- ``learning_rate``, ``momentum``, ``weight_decay``: SGD settings.
- ``lr_schedule``: constant or epoch-based multistep decay, with optional
  linear warmup and explicit epoch milestones.
- ``timeout_s``: hard ``torchrun`` timeout.
- ``omp_num_threads``: CPU threads per rank.
- ``verify_dmesg``: enables bounded host-kernel error scanning.
- ``peak_tflops_per_gpu``: dense-precision hardware peak used for MFU.
- ``checkpoint_enabled``, ``checkpoint_keep_file``,
  ``checkpoint_loss_tolerance``: save/load and resume-parity validation.
- ``loss_curve``: sampling cadence, minimum points, and decreasing-slope check.
- ``accuracy``: explicit final Top-1 and Top-5 qualification targets.
- ``convergence``: optional Top-1 and evaluation-loss targets, and
  ``stop_when_reached``.
- ``scaling_baseline``: reference throughput for scaling efficiency %.
  ``images_per_sec_total`` is the TOTAL images/sec from a prior run on
  ``num_nodes`` nodes, taken from that run's ``results.json``. Efficiency % =
  throughput_N / ((N / num_nodes) x images_per_sec_total) x 100, matching the
  JAX MaxText definition so the two suites are comparable. Leave
  ``images_per_sec_total`` at ``0.0`` to disable it; the metric is then omitted
  rather than reported as a misleading zero.
- ``codecarbon``: CodeCarbon 3.2.4 AMDSMI activation, sampling, and
  required/optional policy.
- ``env_vars``: exported only for the training process. ``NNODES`` and
  ``NODE_RANK`` are overwritten per node by the launcher.
- ``error_patterns``: named regular expressions checked against
  ``training.log``.
- ``sweeps``: full training runs. Each ``name`` must match a threshold cell.
- ``enabled_sweep_list``: exact sweep names to execute; omit or empty runs all.

Sweep fields
============

- ``name``: canonical threshold cell key. Include topology and material
  workload dimensions so results cannot be compared against the wrong gate.
- ``label``: compact unique pytest identifier, also the log directory name.
- ``model``: torchvision model factory name.
- ``backend``: currently ``torchvision``.
- ``precision``: ``BF16``, ``FP16``, or ``FP32``.
- ``batch_size``: local images per GPU.
- ``image_size``: square input resolution.
- ``gradient_accumulation_steps``: microbatches per optimizer step.
- ``training_flops_per_image``: forward+backward FLOPs used for TFLOPS/s/GPU.
  ResNet-50 uses 4.1 GMAC forward x two FLOPs per multiply-add x three for
  training = 24.6 GFLOP/image; ViT-B/16 uses 16.85 GMAC = 101.1 GFLOP/image.
  Both were measured in-container with ``torch.utils.flop_counter``, so the
  convention is consistent across workloads.
- ``data_mode``: ``synthetic`` or ``rocal``.
- ``dataset_path``: container-visible ImageNet root for rocAL.
- ``rocal_device``: ``cpu`` or ``gpu``.
- ``augmentation``: ``standard`` or ``heavy`` for controlled A/B sweeps.
- ``rocal_num_threads``, ``loader_warmup_steps``, ``loader_benchmark_steps``:
  input-pipeline settings and the isolated loader benchmark window.

Thresholds
==========

Each threshold file maps a sweep ``name`` to a cell of metric specs. Supported
kinds are ``min``, ``max``, ``max_ms``, and ``info``; ``info`` always passes and
records the value. Start with ``info`` while calibrating and enable numeric
gates only after repeated runs with the same image, model, hardware, and batch
shape.

The shipped rocAL and multi-node thresholds are record-only, because storage
and fabric behaviour materially affect throughput. Calibrate on the target
filesystem and fabric before setting ``enforce_thresholds`` to ``true``.

Adding a sweep
==============

Add the sweep object, add its exact ``name`` to ``enabled_sweep_list``, and add
a matching top-level cell to the threshold file.

How to run these suites:
:doc:`/how-to/test-suites/training/pytorch_vision`.
