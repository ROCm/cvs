# PyTorch Vision training

`pytorch_vision_training` runs public torchvision training workloads in an AMD
ROCm PyTorch container. The first workload is W1:

- ResNet-50 with random initialization
- ImageNet-shaped synthetic input (`3 × 224 × 224`, 1000 classes)
- BF16 autocast and channels-last tensors
- one MI325X node with eight DDP ranks
- SGD with momentum

The synthetic batch is created once on each GPU. This intentionally measures the
model, optimizer, and DDP path without storage or DataLoader variance. It does
not measure ImageNet accuracy or input-pipeline performance.

## Container

The checked-in config pins this public AMD image:

```text
rocm/pytorch:rocm7.14_ubuntu24.04_py3.12_pytorch_release_2.12.0@sha256:c38eeda81d85f00fbe35d3d50ce42ce59c524e87d810624f4eb5c52fddb3b9ad
```

CVS launches and removes the container through `ContainerOrchestrator`. The
benchmark never invokes Docker or installs packages itself.

## Run

Use a cluster file containing exactly one eight-GPU node:

```bash
cvs run pytorch_vision_training \
  --cluster_file /path/to/cluster.json \
  --config_file cvs/input/config_file/training/pytorch_vision/mi325x_resnet50_w1_config.json \
  --html /tmp/pytorch_vision_w1.html \
  --self-contained-html \
  --capture=tee-sys
```

The config owns the image, batch size, warmup/measured step counts, optimizer,
environment overrides, and selected sweep cells. The cluster file only owns
node access.

## Results

Rank zero writes `results.json` below:

```text
{paths.log_dir}/pytorch_vision/<combo>/<run-id>/
```

The same directory contains `training.log`. The structured artifact records:

- total and per-GPU images/second
- mean, p50, and p95 distributed step time
- peak allocated and reserved GPU memory
- initial and final measured loss
- raw per-step critical-path times and runtime metadata

Step throughput uses the slowest rank for each measured step. Warmup steps are
excluded, and the benchmark adds no per-step barrier or metric collective to
the timed region. Total throughput uses the slowest rank's wall-clock duration
for the complete measured window, so Python launch overhead and rank skew are
included. CVS also scans host dmesg over the bounded training window for GPU,
driver, and hardware errors.

## Run deck

When `--html` is supplied, CVS automatically loads
`cvs.lib.report.presets.pytorch_vision_training` and adds these SGLang-style
sidecars to the report bundle:

```text
pytorch_vision_training_run_deck.html
pytorch_vision_training_run_deck.json
pytorch_vision_training_run_deck_summary.html
```

The run deck presents the pinned image and topology, lifecycle timing, threshold
status by throughput/latency/memory tier, per-cell gate margins, and the full
metric table. It is render-only: pytest metric rows remain the source of
pass/fail.

## Thresholds

The checked-in MI325X thresholds were calibrated from three consecutive runs
with the pinned image. Those runs measured 25.83–25.86k images/s total and
79.19–79.27 ms mean step time. The enforced limits leave approximately 5%
throughput headroom and 10% p95-latency headroom:

- total throughput: at least 24,500 images/s
- per-GPU throughput: at least 3,062.5 images/s/GPU
- p95 step time: at most 88 ms
- peak allocated/reserved memory: at most 13,000/15,000 MB

Mean/p50 timing and loss values remain informational. Recalibrate the gated
limits when changing the image, GPU architecture, batch size, or workload.

## Future data support

A later workload can mount an operator-provided ImageNet directory through the
container config and add DataLoader/accuracy stages. ImageNet is not downloaded
by this suite because its distribution requires separate access terms.
