# PyTorch Vision training

`pytorch_vision_training` runs public torchvision training workloads in an AMD
ROCm PyTorch container. W1 has phased synthetic-performance and real
ImageNet/rocAL coverage:

- ResNet-50 with random initialization
- synthetic input or streaming ImageNet-1k train/validation input
- BF16 autocast and channels-last tensors
- one MI325X node with eight DDP ranks
- SGD with momentum
- smoke, performance, one-epoch, 5k-step, protected 90-epoch, and protected
  24-hour run modes

In `perf` mode the synthetic batch is created once on each GPU. This intentionally measures the
model, optimizer, and DDP path without storage or DataLoader variance. It does
not measure ImageNet accuracy or input-pipeline performance. Real-data modes
stream rocAL batches and validation without retaining all images.

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

The config follows the JAX MaxText training layout: `training.distributed` and
`training.gpus_per_node` define topology, `training.steps` and optimizer fields
define execution, `training.env_vars` and `training.error_patterns` control the
runtime, and `training.enabled_sweep_list` selects entries from
`training.sweeps`. Each sweep `name` is also its threshold-file cell key; its
short `label` is used in pytest rows and the run deck. The cluster file only
owns node access.

## Results

Rank zero writes `results.json` below:

```text
{paths.log_dir}/pytorch_vision/<sweep-label>/<run-id>/
```

The same directory contains `training.log`. The structured artifact records:

- total and per-GPU images/second
- provisional TFLOPS/s/GPU and MFU
- mean, p50, and p95 distributed step time
- peak PyTorch allocated/reserved memory and observed device-wide used memory
- checkpoint save/load time, exact state parity, and resumed-loss delta
- gradient-accumulation overhead against the GA=1 fixed-global-batch baseline
- initial and final measured loss
- raw per-step critical-path times and runtime metadata
- sampled loss/time points and losses at steps 100/500/1000/5000
- streamed evaluation loss, globally reduced Top-1/Top-5 accuracy, and sample count
- convergence step/time when a target is configured
- continuous AMD-SMI memory/utilization/energy and CodeCarbon 3.2.4 emissions

Step throughput uses the slowest rank for each measured step. Warmup steps are
excluded. Accuracy is never formed by averaging rank percentages: ranks SUM
correct counts, loss sums, and sample counts. Total throughput uses the slowest rank's wall-clock duration
for the complete measured window, so Python launch overhead and rank skew are
included. CVS also scans host dmesg over the bounded training window for GPU,
driver, and hardware errors.

`device_memory_used_mb_observed` is the larger of the CUDA device-wide samples
taken immediately before and after the measured window. It includes non-PyTorch
occupancy and is deliberately not labeled as an in-window peak; allocated and
reserved values use PyTorch's true peak counters.

The provisional compute metrics use 24.6 GFLOP per ResNet-50 training image
(4.1 GMAC forward × two FLOPs per multiply-add × three for forward/backward)
and AMD's published 1307.4 dense BF16 TFLOPS/GPU peak for MI325X. These
assumptions are stored in the config and shown in the run deck. They are
informational until a performance methodology owner approves them.

Checkpoint validation saves model, optimizer, step, and RNG state, flushes the
file to storage, restores it into a fresh model/optimizer, restores RNG state,
takes one optimizer step on both the original and resumed paths, and checks
exact state immediately after load plus post-step model/optimizer maximum
absolute deltas and loss parity. The resumed GPU step uses calibrated numerical
tolerances because independent convolution backward executions are not
bitwise-deterministic. The checkpoint is deleted after verification by default;
timing and parity remain in `results.json`.

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
status by throughput, latency, accuracy, convergence, data, energy, and
continuous-memory tier, per-cell gate margins, charts, artifact links, and the
full metric table. It is render-only: pytest metric rows remain the source of
pass/fail.

## Thresholds

The checked-in MI325X thresholds were calibrated from three consecutive runs
per GA cell with the pinned image. GA=1 measured 25.76–25.80k images/s and GA=4
measured 16.75–17.44k images/s at the same effective global batch. Proper GA
suppresses DDP synchronization on the first three microbatches and added
47.7–53.9% optimizer-step overhead. Checkpoint state loaded exactly; the
independent resumed step had zero loss delta and stayed within the calibrated
model/optimizer state tolerances.

- total throughput: at least 24,500 images/s
- per-GPU throughput: at least 3,062.5 images/s/GPU
- p95 step time: at most 88 ms
- GA=1 allocated/reserved/observed-device memory: at most 13,000/15,000/21,000 MB
- GA=4 throughput: at least 15,500 images/s
- GA=4 p95 step time and GA overhead: at most 136 ms and 60%
- checkpoint state match: exactly represented by a minimum value of 1

TFLOPS, MFU, checkpoint I/O time, mean/p50 timing, and loss values remain
informational. Recalibrate the gated limits when changing the image, GPU
architecture, batch size, storage target, or workload.

## Phased profiles and safety

The config directory provides smoke, rocAL performance, one-epoch, and 5k-step
profiles plus disabled 90-epoch and 24-hour profiles. Long profiles ship with
both `training.enabled=false` and `training.allow_long_run=false`; both settings
must be changed explicitly before they can run. ImageNet is not downloaded by
CVS because its distribution requires separate access terms.

Real-data smoke, loss-curve, and convergence checks have separate pytest rows.
Numerical accuracy, convergence, data, and energy values are record-only until
calibrated. Evaluation completion, finite values, positive sample counts,
AMDSMI activation when requested, and result artifacts are structural checks.
