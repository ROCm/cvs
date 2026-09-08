# PyTorch Vision training configuration

The PyTorch Vision suite uses the same configuration split as JAX MaxText:

- the cluster file contains SSH access and the selected node
- `*_config.json` contains the container and training workload
- the sibling `*_threshold.json` contains per-sweep metric gates

Run the single-node suite with:

```bash
cvs run pytorch_vision_training \
  --cluster_file /path/to/mi325x_1n_cluster.json \
  --config_file /path/to/mi325x_resnet50_w1_config.json \
  --html /path/to/results/pytorch_vision_w1.html \
  --self-contained-html \
  --capture=tee-sys
```

## Top-level fields

- `framework`: must be `pytorch_vision_training`.
- `gpu_arch`: exact device family expected inside the container.
- `enforce_thresholds`: gates every configured metric when `true`.
- `threshold_json`: sibling threshold filename.
- `paths`: shared model, log, and token locations. W1 only writes `log_dir`.
- `model`: public model identity; `remote` remains `0` because W1 uses random
  torchvision weights.
- `container`: lifecycle, digest-pinned AMD image, and runtime mounts/devices.

## Training block

- `distributed`: must be `false` for the current single-node suite.
- `gpus_per_node`: expected visible GPU count and torchrun rank count.
- `steps`: measured optimizer steps.
- `warmup_steps`: untimed steps before measurement.
- `num_classes`: classifier output classes; W1 uses 1000.
- `channels_last`: enables NHWC-compatible tensor storage.
- `learning_rate`, `momentum`, `weight_decay`: SGD settings.
- `timeout_s`: hard torchrun timeout.
- `omp_num_threads`: CPU threads per rank.
- `verify_dmesg`: enables bounded host-kernel error scanning.
- `peak_tflops_per_gpu`: provisional dense-precision hardware peak used for MFU.
- `checkpoint_enabled`: enables save/load and resume-parity validation.
- `checkpoint_keep_file`: retains the checkpoint after validation when `true`.
- `checkpoint_loss_tolerance`: maximum resumed-loss delta.
- `env_vars`: exported only for the training process.
- `error_patterns`: named regular expressions checked against `training.log`.
- `sweeps`: full training runs. Each `name` must match a threshold cell.
- `enabled_sweep_list`: exact sweep names to execute; omit or empty runs all.

## Sweep fields

- `name`: canonical threshold cell key. Include topology and material workload
  dimensions so results cannot be compared against the wrong gate.
- `label`: compact unique pytest/run-deck identifier.
- `model`: torchvision model factory name.
- `backend`: currently `torchvision`.
- `precision`: `BF16`, `FP16`, or `FP32`.
- `batch_size`: local images per GPU.
- `image_size`: square synthetic input resolution.
- `gradient_accumulation_steps`: microbatches per optimizer step.
- `training_flops_per_image`: provisional forward+backward FLOPs used for
  TFLOPS/s/GPU. W1 uses 4.1 GMAC forward × two FLOPs per multiply-add × three
  for training = 24.6 GFLOP/image.

The supplied W1 sweeps use one node, 50 measured steps, ResNet-50 BF16, global
batch 2048, and 224×224 input. GA=1 uses microbatch 256; GA=4 uses microbatch 64
so the comparison holds effective global batch constant.

## Adding a sweep

Add the sweep object, add its exact `name` to `enabled_sweep_list`, and add a
matching top-level cell to the threshold file. Start with `kind: info` while
calibrating. Enable `min`, `max`, or `max_ms` only after repeated runs with the
same image, model, hardware, and batch shape.
