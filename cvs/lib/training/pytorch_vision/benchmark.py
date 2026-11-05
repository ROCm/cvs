"""Container-side ResNet training benchmark launched with torchrun."""

from __future__ import annotations

import argparse
import contextlib
import datetime
import importlib.metadata
import json
import math
import os
import statistics
import sys
import threading
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=["torchvision", "transformers", "mamba"], default="torchvision")
    parser.add_argument("--precision", choices=["BF16", "FP16", "FP32"], default="BF16")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--measure-steps", type=int, default=50)
    parser.add_argument(
        "--run-mode",
        choices=["smoke", "perf", "train_5k", "finetune", "train_to_accuracy"],
        default="perf",
    )
    parser.add_argument("--phase", choices=["smoke", "performance", "accuracy"], default="performance")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--max-duration-seconds", type=int)
    parser.add_argument("--eval-enabled", action="store_true")
    parser.add_argument("--eval-steps", type=int)
    parser.add_argument("--eval-every-epochs", type=int, default=1)
    parser.add_argument("--eval-sample-count", type=int, default=50000)
    parser.add_argument("--milestone-steps", default="100,500,1000,5000")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--zero-init-residual", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--harness", choices=["native", "hf_trainer", "task"], default="native")
    parser.add_argument(
        "--task", choices=["classification", "detection", "segmentation", "vlm_sft"], default="classification"
    )
    parser.add_argument("--task-options", default="{}", help="JSON options for the task harness")
    parser.add_argument(
        "--model-id", default="", help="HuggingFace model id (hf_trainer, task, or a native HF backbone)"
    )
    parser.add_argument(
        "--fsdp-strategy",
        choices=["full_shard", "hybrid_shard"],
        default="full_shard",
        help="FSDP sharding strategy for the hf_trainer harness",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="load pretrained weights; omit to train the same architecture from scratch",
    )
    parser.add_argument("--workload", default="W1", help="scorecard workload id recorded in the artifact")
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--lr-schedule", choices=["constant", "multistep"], default="constant")
    parser.add_argument("--lr-warmup-epochs", type=int, default=0)
    parser.add_argument("--lr-milestones-epochs", default="")
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--training-flops-per-image", type=float, required=True)
    parser.add_argument("--peak-tflops-per-gpu", type=float, required=True)
    parser.add_argument("--data-mode", choices=["synthetic", "rocal", "dataset"], default="synthetic")
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--rocal-device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--augmentation", choices=["standard", "heavy"], default="standard")
    parser.add_argument("--rocal-num-threads", type=int, default=8)
    parser.add_argument("--loader-warmup-steps", type=int, default=5)
    parser.add_argument("--loader-benchmark-steps", type=int, default=20)
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--checkpoint-loss-tolerance", type=float, default=1e-5)
    parser.add_argument("--keep-checkpoint", action="store_true")
    parser.add_argument("--loss-curve-sample-every", type=int, default=1)
    parser.add_argument("--loss-curve-minimum-points", type=int, default=2)
    parser.add_argument("--convergence-top1", type=float)
    parser.add_argument("--convergence-eval-loss", type=float)
    parser.add_argument("--convergence-train-loss", type=float)
    parser.add_argument("--target-top5", type=float)
    parser.add_argument("--stop-on-convergence", action="store_true")
    parser.add_argument("--codecarbon-enabled", action="store_true")
    parser.add_argument("--codecarbon-required", action="store_true")
    parser.add_argument("--codecarbon-measure-power-secs", type=int, default=5)
    parser.add_argument("--codecarbon-country-iso-code")
    parser.add_argument("--collective-timeout-seconds", type=int, default=120)
    parser.add_argument("--startup-timeout-seconds", type=int, default=300)
    parser.add_argument("--setup-timeout-seconds", type=int, default=1800)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _autocast(precision):
    dtypes = {"BF16": torch.bfloat16, "FP16": torch.float16}
    if precision == "FP32":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtypes[precision])


def _percentile(values, percentile):
    index = min(len(values) - 1, math.ceil((percentile / 100.0) * len(values)) - 1)
    return values[max(index, 0)]


def _train_step(model, optimizer, loss_fn, image_batches, label_batches, precision):
    optimizer.zero_grad(set_to_none=True)
    losses = []
    accumulation_steps = len(image_batches)
    for index, (images, labels) in enumerate(zip(image_batches, label_batches)):
        sync_context = (
            model.no_sync()
            if index < accumulation_steps - 1 and hasattr(model, "no_sync")
            else contextlib.nullcontext()
        )
        with sync_context:
            with _autocast(precision):
                output = model(images)
                loss = loss_fn(output, labels)
            (loss / accumulation_steps).backward()
        losses.append(loss.detach())
    optimizer.step()
    return torch.stack(losses).mean()


def _gather_step_times(local_times, rank, world_size, device):
    local = torch.tensor(local_times, dtype=torch.float64, device=device)
    gathered = [torch.empty_like(local) for _ in range(world_size)] if rank == 0 else None
    dist.gather(local, gather_list=gathered, dst=0)
    if rank != 0:
        return None
    return torch.stack(gathered).amax(dim=0).cpu().tolist()


def _reduce_mean(value, world_size, device):
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item() / world_size


def _reduce_max(value, device):
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.item()


def _tasks_module():
    """Load tasks.py, staged beside this script as ``<stem>_tasks.py``."""
    import importlib.util

    here = Path(__file__)
    for candidate in (here.with_name(here.stem + "_tasks.py"), here.with_name("tasks.py")):
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location("cvs_pytorch_vision_tasks", candidate)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError(f"task module not staged beside {here}")


def _task_options(args):
    return json.loads(getattr(args, "task_options", "") or "{}")


class _LogitsOnly(torch.nn.Module):
    """Present an HF image classifier as ``images -> logits`` for the native loop."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, images):
        return self.inner(pixel_values=images).logits


def _build_model(args, device):
    if args.model.startswith("vim_"):
        tasks = _tasks_module()
        options = _task_options(args)
        checkpoint = (
            tasks.resolve_hf_file(args.model_id, options["pretrained_file"])
            if args.pretrained and options.get("pretrained_file")
            else ""
        )
        model = tasks.build_vim(args.model, args.num_classes, args.image_size, checkpoint)
    elif args.model_id:
        from transformers import AutoConfig, AutoModelForImageClassification

        if args.pretrained:
            inner = AutoModelForImageClassification.from_pretrained(
                args.model_id, num_labels=args.num_classes, ignore_mismatched_sizes=True
            )
        else:
            inner = AutoModelForImageClassification.from_config(
                AutoConfig.from_pretrained(args.model_id, num_labels=args.num_classes)
            )
        model = _LogitsOnly(inner)
    else:
        from torchvision.models import get_model

        extra = {"zero_init_residual": True} if args.zero_init_residual else {}
        model = get_model(args.model, weights=None, num_classes=args.num_classes, **extra)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    return model.to(device)


def _build_optimizer(model, args):
    if args.optimizer == "adamw":
        # Transformer backbones (ViT and friends) do not train usefully under
        # SGD at these learning rates; momentum is unused in this branch.
        return torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    return torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )


def _build_lr_scheduler(optimizer, args, steps_per_epoch):
    if args.lr_schedule == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    if steps_per_epoch is None:
        raise ValueError("multistep learning-rate schedule requires epoch-based training")
    warmup_steps = args.lr_warmup_epochs * steps_per_epoch
    milestone_steps = [int(epoch) * steps_per_epoch for epoch in args.lr_milestones_epochs.split(",") if epoch.strip()]
    if not milestone_steps:
        raise ValueError("multistep learning-rate schedule requires milestones")

    def multiplier(step):
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / warmup_steps
        decays = sum(step >= milestone for milestone in milestone_steps)
        return args.lr_gamma**decays

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _nested_equal(left, right):
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_nested_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_nested_equal(a, b) for a, b in zip(left, right))
    return left == right


def _nested_max_abs_delta(left, right):
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if left.shape != right.shape:
            return float("inf")
        if left.numel() == 0:
            return 0.0
        return float((left.detach().double().cpu() - right.detach().double().cpu()).abs().max())
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return float("inf")
        return max((_nested_max_abs_delta(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return float("inf")
        return max((_nested_max_abs_delta(a, b) for a, b in zip(left, right)), default=0.0)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    return 0.0 if left == right else float("inf")


def _to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def _exact_eval_batch_size(shard_samples, preferred):
    """Largest divisor of ``shard_samples`` that is <= ``preferred``.

    With no partial batch there is nothing for rocAL's LAST_BATCH_FILL to pad,
    so evaluation consumes each image exactly once regardless of how the
    training batch size happens to divide the shard. Evaluation throughput is
    not a reported metric, so trading batch size for exactness costs nothing.
    """
    if shard_samples <= 0:
        return max(1, preferred)
    for size in range(min(preferred, shard_samples), 0, -1):
        if shard_samples % size == 0:
            return size
    return 1


def _build_rocal_loader(args, rank, world_size, split="train", batch_size=None):
    import amd.rocal.fn as fn
    import amd.rocal.types as types
    from amd.rocal.pipeline import Pipeline
    from amd.rocal.plugin.pytorch import ROCALClassificationIterator

    data_path = str(Path(args.dataset_path) / split)
    cpu = args.rocal_device == "cpu"
    batch_size = batch_size or args.batch_size
    # HF ViT checkpoints were trained on inputs scaled to [-1, 1]; everything
    # else in the suite uses the ImageNet statistics.
    if _task_options(args).get("input_normalization") == "half":
        mean, std = [127.5] * 3, [127.5] * 3
    else:
        mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    pipe = Pipeline(
        batch_size=batch_size,
        num_threads=args.rocal_num_threads,
        device_id=int(os.environ["LOCAL_RANK"]),
        seed=2026 + rank,
        rocal_cpu=cpu,
        tensor_dtype=types.FLOAT,
        tensor_layout=types.NCHW,
        prefetch_queue_depth=6,
        mean=mean,
        std=std,
        output_memory_type=types.HOST_MEMORY if cpu else types.DEVICE_MEMORY,
    )
    with pipe:
        jpegs, _labels = fn.readers.file(
            file_root=data_path,
            pad_last_batch=split == "train",
        )
        if split == "train":
            # image_slice draws a random window and the old path then stretched
            # that window to 224. A 90-epoch ResNet-50 run on that pipeline
            # finished at 13% Top-1. Decode the full frame and apply the
            # torchvision RandomResizedCrop defaults instead.
            #
            # rocAL shards are contiguous runs of the class-sorted file list and
            # it shuffles only within a shard, so sharded training gives every
            # rank a fixed eighth of the classes. BatchNorm batch statistics
            # then encode the shard: a 5,000-step ResNet-50 run reached train
            # loss 2.6 at 5.8% validation Top-1. Each rank instead reads the
            # whole set in its own seeded order, reshuffled on every pass.
            decoded = fn.decoders.image(
                jpegs,
                output_type=types.RGB,
                file_root=data_path,
                shard_id=0,
                num_shards=1,
                random_shuffle=True,
                stick_to_shard=True,
            )
            resized = fn.random_resized_crop(
                decoded,
                resize_width=args.image_size,
                resize_height=args.image_size,
                crop_area_factor=[0.08, 1.0],
                crop_aspect_ratio=[0.75, 1.333333],
                interpolation_type=types.LINEAR_INTERPOLATION,
                output_layout=types.NHWC,
                output_dtype=types.UINT8,
            )
        else:
            # Evaluation must consume each image exactly once.
            #
            # Measured on the 50,000-image validation set over 8 shards at
            # batch 128 (ideal 6,250/shard, 50 per class):
            #   FILL    50,176 total, 1000 classes, 50..72 per class
            #   PARTIAL 48,976 total,  984 classes,  0..50 per class
            #   DROP    48,128 total,  968 classes,  0..50 per class
            #
            # Only FILL omits nothing; its surplus is padding appended to each
            # shard's final batch, which the caller's local-target cap trims off.
            # PARTIAL and DROP silently lose whole classes, which accuracy alone
            # would never reveal.
            #
            # stick_to_shard must be explicit: without it a shard bleeds into
            # its neighbour, which both duplicates and omits images and is what
            # produced a 32..72 per-class spread before this was pinned down.
            decoded = fn.decoders.image(
                jpegs,
                output_type=types.RGB,
                file_root=data_path,
                shard_id=rank,
                num_shards=world_size,
                random_shuffle=False,
                stick_to_shard=True,
                pad_last_batch=True,
                last_batch_policy=types.LAST_BATCH_FILL,
            )
            # 256 for a 224 crop; small-image sets such as CIFAR keep the same
            # 0.875 crop ratio instead of being upscaled to ImageNet size.
            resized = fn.resize(
                decoded,
                resize_shorter=round(args.image_size * 256 / 224),
                scaling_mode=types.SCALING_MODE_NOT_SMALLER,
                output_layout=types.NHWC,
                output_dtype=types.UINT8,
                interpolation_type=types.TRIANGULAR_INTERPOLATION,
            )
        flip = fn.random.coin_flip(probability=0.5 if split == "train" else 0.0)
        normalized = fn.crop_mirror_normalize(
            resized,
            output_layout=types.NCHW,
            output_dtype=types.FLOAT,
            crop=(args.image_size, args.image_size),
            mirror=flip,
            mean=mean,
            std=std,
        )
        pipe.set_outputs(normalized)
    pipe.build()
    return ROCALClassificationIterator(
        pipe,
        device="cpu" if cpu else "cuda",
        device_id=int(os.environ["LOCAL_RANK"]),
    )


def _prepare_rocal_batch(images, labels, device, channels_last, heavy=False):
    if images.device != device:
        if images.device.type == "cpu" and not images.is_pinned():
            images = images.pin_memory()
        images = images.to(device, non_blocking=True)
    if labels.device != device:
        if labels.device.type == "cpu" and not labels.is_pinned():
            labels = labels.pin_memory()
        labels = labels.to(device, non_blocking=True)
    if heavy:
        mask = torch.rand(images.shape[0], device=device) < 0.25
        if mask.any():
            width = max(1, images.shape[-1] // 8)
            images[mask, :, :width, :width] = 0
    if channels_last:
        images = images.contiguous(memory_format=torch.channels_last)
    return images, labels.long().reshape(-1)


def _next_rocal_batches(loader, accumulation_steps, channels_last, device, heavy=False, reset_on_end=True):
    image_batches = []
    label_batches = []
    for _ in range(accumulation_steps):
        try:
            [images], labels = next(loader)
        except StopIteration:
            if not reset_on_end:
                if image_batches:
                    break
                raise
            loader.reset()
            [images], labels = next(loader)
        images, labels = _prepare_rocal_batch(images, labels, device, channels_last, heavy)
        if accumulation_steps > 1:
            # rocAL reuses the same output tensors on every __next__ call.
            # Gradient accumulation must own each microbatch until the optimizer
            # step has consumed all of them.
            images = images.clone(memory_format=torch.preserve_format)
            labels = labels.clone()
        image_batches.append(images)
        label_batches.append(labels)
    return image_batches, label_batches


def _rocal_samples_per_rank(loader, world_size=1):
    # The training loader holds the whole set on every rank, so one epoch is
    # each rank's share of it rather than a full pass of its iterator.
    return math.ceil(int(loader.iterator_length) / world_size)


def _rocal_batches_per_epoch(loader, world_size=1):
    sample_count = _rocal_samples_per_rank(loader, world_size)
    batch_size = int(loader.batch_size)
    if sample_count <= 0 or batch_size <= 0:
        raise ValueError(f"invalid rocAL iterator dimensions: samples={sample_count}, batch_size={batch_size}")
    return math.ceil(sample_count / batch_size)


def _distributed_next_rocal_batches(
    loader,
    accumulation_steps,
    channels_last,
    device,
    heavy=False,
    reset_on_end=True,
):
    batches = None
    error = None
    status = 1
    try:
        batches = _next_rocal_batches(
            loader,
            accumulation_steps,
            channels_last,
            device,
            heavy,
            reset_on_end,
        )
    except StopIteration:
        status = 0
    except Exception as exc:
        status = -1
        error = f"{type(exc).__name__}: {exc}"
    status_tensor = torch.tensor(status, dtype=torch.int32, device=device)
    dist.all_reduce(status_tensor, op=dist.ReduceOp.MIN)
    global_status = int(status_tensor.item())
    if global_status < 0:
        raise RuntimeError(f"rocAL iterator failed on at least one rank; local_error={error}")
    if global_status == 0:
        return None
    return batches


def _measure_rocal_loader(loader, args, device, rank, world_size):
    for _ in range(args.loader_warmup_steps):
        _distributed_next_rocal_batches(
            loader,
            args.gradient_accumulation_steps,
            args.channels_last,
            device,
            args.augmentation == "heavy",
        )
    critical_times_ms = []
    for _ in range(args.loader_benchmark_steps):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        _distributed_next_rocal_batches(
            loader,
            args.gradient_accumulation_steps,
            args.channels_last,
            device,
            args.augmentation == "heavy",
        )
        torch.cuda.synchronize(device)
        critical_times_ms.append((time.perf_counter() - start) * 1000.0)
    critical_times_ms = _gather_step_times(critical_times_ms, rank, world_size, device)
    if rank != 0:
        return None
    mean_ms = statistics.fmean(critical_times_ms)
    images_per_step = args.batch_size * args.gradient_accumulation_steps * world_size
    return {
        "data_loader_images_per_sec": images_per_step * 1000.0 / mean_ms,
    }


def _accuracy_from_totals(top1_correct, top5_correct, sample_count):
    if sample_count <= 0:
        raise ValueError("evaluation produced no samples")
    if not 0 <= top1_correct <= top5_correct <= sample_count:
        raise ValueError("invalid distributed accuracy totals")
    return 100.0 * top1_correct / sample_count, 100.0 * top5_correct / sample_count


def _evaluation_meets_convergence_target(evaluation, args):
    return (args.convergence_top1 is None or evaluation["top1_accuracy_pct"] >= args.convergence_top1) and (
        args.convergence_eval_loss is None or evaluation["eval_loss"] <= args.convergence_eval_loss
    )


def _evaluation_meets_stop_target(evaluation, args):
    return _evaluation_meets_convergence_target(evaluation, args) and (
        args.target_top5 is None or evaluation["top5_accuracy_pct"] >= args.target_top5
    )


def _scorecard_milestone_metrics(milestone_losses):
    return {
        f"loss_step_{milestone}": milestone_losses[str(milestone)]
        for milestone in (100, 500, 1000, 5000)
        if str(milestone) in milestone_losses
    }


class _EvalLoader:
    """Hands ``_evaluate`` a rocAL validation iterator positioned at its shard start.

    ``reset()`` is only safe when every shard is the same size. With unequal
    shards rocAL fills the short final batch from the next shard, and a reset
    then restarts the stream off its boundary: on Food-101 (25,250 images over
    8 ranks) a reset pass saw 238-261 images per class instead of 250. Those
    sets get a freshly built pipeline per pass instead.

    Retired pipelines stay referenced until exit: destroying a rocAL pipeline
    mid-run while its prefetch workers are live corrupted the heap
    (``malloc(): unsorted double linked list corrupted``) on a W8 fine-tune.
    Each one keeps its GPU buffers, so full evaluation avoids rebuilding by
    running on ``ranks`` that divide the set evenly; see ``_eval_ranks``.
    Ranks outside that group have no pipeline and evaluate nothing.
    """

    def __init__(self, build, rebuild_each_pass, ranks):
        self.ranks = ranks
        self._build = build
        self._rebuild = rebuild_each_pass
        self._loader = build() if build is not None else None
        self._retired = []
        self._fresh = True

    def next_pass(self):
        if self._loader is None:
            return None
        if not self._fresh:
            if self._rebuild:
                self._retired.append(self._loader)
                self._loader = self._build()
            else:
                self._loader.reset()
        self._fresh = False
        return self._loader


def _eval_ranks(sample_count, world_size):
    """Largest rank count up to ``world_size`` that splits the eval set evenly.

    Unequal shards force a fresh pipeline on every pass (see ``_EvalLoader``),
    and the retired ones exhaust GPU memory over a long run: ImageNet's 50,000
    validation images over 32 ranks failed a ``hipMalloc`` partway through a
    four-node target-accuracy run. 25 of those ranks take 2,000 images each.
    """
    if not sample_count:
        return world_size
    return max(count for count in range(1, world_size + 1) if sample_count % count == 0)


def _evaluate(model, eval_loader, loss_fn, args, device):
    model.eval()
    # The first DDP forward after a training step broadcasts the buffers, which
    # is a collective, and ranks outside the eval group never make that call:
    # a four-node run hung on it at the first evaluation. Every rank takes
    # rank zero's BatchNorm statistics here, as that broadcast would, and the
    # forward runs on the unwrapped module.
    network = getattr(model, "module", model)
    for buffer in network.buffers():
        dist.broadcast(buffer, src=0)
    loader = eval_loader.next_pass()
    totals = torch.zeros(4, dtype=torch.float64, device=device)
    # Per-class label histogram. A correct label mapping over a full ImageNet
    # validation set covers every class; a scrambled or truncated label space
    # shows up here as missing classes or a lopsided distribution, neither of
    # which accuracy alone would reveal.
    label_hist = torch.zeros(args.num_classes, dtype=torch.int64, device=device)
    steps = 0
    rank = int(os.environ["RANK"])
    eval_ranks = eval_loader.ranks
    # rocAL starts shard i at floor(i * N / shards) and pads a short final
    # batch by reading on into the next shard, so each rank keeps exactly the
    # images between its own boundaries.
    if rank >= eval_ranks:
        local_target = 0
    elif args.eval_steps is None:
        local_target = (rank + 1) * args.eval_sample_count // eval_ranks - rank * args.eval_sample_count // eval_ranks
    else:
        local_target = None
    local_samples = 0
    try:
        while True:
            # Shards differ by one image when the eval set does not divide by
            # world size, so a rank can finish a batch early. It keeps joining
            # the collective without fetching until no rank needs more.
            need = local_samples < local_target if local_target is not None else steps < args.eval_steps
            batches, status, error = None, 1, None
            if need:
                try:
                    batches = _next_rocal_batches(loader, 1, args.channels_last, device, False, False)
                except StopIteration:
                    status = 0
                except Exception as exc:
                    status, error = -1, f"{type(exc).__name__}: {exc}"
            flags = torch.tensor([int(need), -status], dtype=torch.int32, device=device)
            dist.all_reduce(flags, op=dist.ReduceOp.MAX)
            if int(flags[1].item()) > 0:
                raise RuntimeError(f"rocAL eval iterator failed on at least one rank; local_error={error}")
            if not int(flags[0].item()) or int(flags[1].item()) == 0:
                break
            if not need:
                continue
            images, labels = batches
            if local_target is not None:
                remaining = local_target - local_samples
                images[0] = images[0][:remaining]
                labels[0] = labels[0][:remaining]
            with torch.no_grad(), _autocast(args.precision):
                output = network(images[0])
                loss = loss_fn(output, labels[0])
            count = labels[0].numel()
            flat = labels[0].reshape(-1)
            if count and (int(flat.min().item()) < 0 or int(flat.max().item()) >= args.num_classes):
                raise RuntimeError(
                    f"evaluation labels outside [0, {args.num_classes}): "
                    f"min={int(flat.min().item())} max={int(flat.max().item())}"
                )
            label_hist += torch.bincount(flat, minlength=args.num_classes)
            predictions = output.topk(min(5, output.shape[1]), dim=1).indices
            correct = predictions.eq(labels[0].view(-1, 1))
            totals[0] += correct[:, :1].any(dim=1).sum()
            totals[1] += correct.any(dim=1).sum()
            totals[2] += loss.detach().double() * count
            totals[3] += count
            local_samples += count
            steps += 1
    finally:
        model.train()
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    dist.all_reduce(label_hist, op=dist.ReduceOp.SUM)
    samples = int(totals[3].item())
    if args.eval_steps is None and samples != args.eval_sample_count:
        raise RuntimeError(f"evaluation consumed {samples} samples, expected exactly {args.eval_sample_count}")
    top1, top5 = _accuracy_from_totals(int(totals[0].item()), int(totals[1].item()), samples)
    observed = int((label_hist > 0).sum().item())
    present = label_hist[label_hist > 0]
    return {
        "top1_accuracy_pct": top1,
        "top5_accuracy_pct": top5,
        "eval_loss": totals[2].item() / samples,
        "eval_sample_count": samples,
        "eval_completed": 1.0,
        "eval_label_classes_observed": float(observed),
        "eval_label_min_per_class": float(present.min().item()) if observed else 0.0,
        "eval_label_max_per_class": float(present.max().item()) if observed else 0.0,
    }


def _start_codecarbon(args, rank, local_gpu_count):
    payload = {
        "version": None,
        "tracker": "amdsmi",
        "gpu_count": 0,
        "active": False,
        "reason": "disabled",
    }
    tracker = None
    if rank == 0 and args.codecarbon_enabled:
        try:
            version = importlib.metadata.version("codecarbon")
            if version != "3.2.4":
                raise RuntimeError(f"CodeCarbon 3.2.4 required, found {version}")
            import amdsmi
            from codecarbon import EmissionsTracker

            amdsmi.amdsmi_init()
            handles = amdsmi.amdsmi_get_processor_handles()
            if len(handles) < local_gpu_count:
                raise RuntimeError(f"AMDSMI found {len(handles)} GPUs, expected {local_gpu_count}")
            kwargs = {
                "measure_power_secs": args.codecarbon_measure_power_secs,
                "tracking_mode": "machine",
                "gpu_ids": list(range(local_gpu_count)),
                "log_level": "error",
                "save_to_file": False,
            }
            if args.codecarbon_country_iso_code:
                kwargs["country_iso_code"] = args.codecarbon_country_iso_code
            tracker = EmissionsTracker(**kwargs)
            tracker.start()
            payload.update(
                {
                    "version": version,
                    "gpu_count": len(handles),
                    "active": True,
                    "reason": "",
                }
            )
        except Exception as exc:
            payload["reason"] = f"{type(exc).__name__}: {exc}"
    state = [payload]
    dist.broadcast_object_list(state, src=0)
    if args.codecarbon_enabled and args.codecarbon_required and not state[0]["active"]:
        raise RuntimeError(f"required CodeCarbon tracking unavailable: {state[0]['reason']}")
    return tracker, state[0]


def _stop_codecarbon(tracker, payload, rank):
    if rank == 0 and tracker is not None:
        try:
            emissions = tracker.stop()
            if emissions is None or not math.isfinite(float(emissions)):
                raise RuntimeError(f"invalid emissions result: {emissions!r}")
            payload["emissions_kg_co2eq"] = float(emissions)
            final_data = getattr(tracker, "final_emissions_data", None)
            energy_kwh = getattr(final_data, "energy_consumed", None)
            if energy_kwh is not None and math.isfinite(float(energy_kwh)) and float(energy_kwh) > 0:
                payload["energy_kwh"] = float(energy_kwh)
            duration_seconds = getattr(final_data, "duration", None)
            if duration_seconds is not None and math.isfinite(float(duration_seconds)) and float(duration_seconds) > 0:
                payload["duration_seconds"] = float(duration_seconds)
        except Exception as exc:
            payload.update({"active": False, "reason": f"{type(exc).__name__}: {exc}"})
    state = [payload]
    dist.broadcast_object_list(state, src=0)
    return state[0]


def _checkpoint_roundtrip(model, optimizer, scheduler, loss_fn, image_batches, label_batches, args, device, rank):
    metrics = {
        "checkpoint_save_seconds": 0.0,
        "checkpoint_load_seconds": 0.0,
        "checkpoint_state_match": 1.0,
        "checkpoint_loss_delta": 0.0,
        "checkpoint_resume_model_max_abs_delta": 0.0,
        "checkpoint_resume_optimizer_max_abs_delta": 0.0,
    }
    if not args.checkpoint_path:
        raise ValueError("--checkpoint-path is required for checkpoint validation")

    dist.barrier()
    error = ""
    if rank == 0:
        checkpoint_path = Path(args.checkpoint_path)
        try:
            raw_model = model.module
            optimizer_step = args.warmup_steps + getattr(args, "completed_steps", args.measure_steps)

            save_start = time.perf_counter()
            model_state = _to_cpu(raw_model.state_dict())
            optimizer_state = _to_cpu(optimizer.state_dict())
            scheduler_state = _to_cpu(scheduler.state_dict())
            checkpoint = {
                "model": model_state,
                "optimizer": optimizer_state,
                "scheduler": scheduler_state,
                "optimizer_step": optimizer_step,
                "cpu_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state(device).cpu(),
            }
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with checkpoint_path.open("wb") as stream:
                torch.save(checkpoint, stream)
                stream.flush()
                os.fsync(stream.fileno())
            metrics["checkpoint_save_seconds"] = time.perf_counter() - save_start

            restored_model = _build_model(args, device)
            restored_optimizer = _build_optimizer(restored_model, args)
            restored_scheduler = _build_lr_scheduler(restored_optimizer, args, args.steps_per_epoch)
            load_start = time.perf_counter()
            loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            restored_model.load_state_dict(loaded["model"], strict=True)
            restored_optimizer.load_state_dict(loaded["optimizer"])
            restored_scheduler.load_state_dict(loaded["scheduler"])
            torch.cuda.synchronize(device)
            metrics["checkpoint_load_seconds"] = time.perf_counter() - load_start

            initial_model_match = _nested_equal(model_state, restored_model.state_dict())
            initial_optimizer_match = _nested_equal(optimizer_state, restored_optimizer.state_dict())
            initial_scheduler_match = _nested_equal(scheduler_state, restored_scheduler.state_dict())
            step_match = loaded["optimizer_step"] == optimizer_step

            torch.set_rng_state(loaded["cpu_rng_state"])
            torch.cuda.set_rng_state(loaded["cuda_rng_state"], device)
            raw_model.train()
            reference_loss = float(
                _train_step(
                    raw_model,
                    optimizer,
                    loss_fn,
                    image_batches,
                    label_batches,
                    args.precision,
                )
            )
            scheduler.step()
            reference_model_state = _to_cpu(raw_model.state_dict())
            reference_optimizer_state = _to_cpu(optimizer.state_dict())
            reference_scheduler_state = _to_cpu(scheduler.state_dict())

            torch.set_rng_state(loaded["cpu_rng_state"])
            torch.cuda.set_rng_state(loaded["cuda_rng_state"], device)
            restored_model.train()
            resumed_loss = float(
                _train_step(
                    restored_model,
                    restored_optimizer,
                    loss_fn,
                    image_batches,
                    label_batches,
                    args.precision,
                )
            )
            restored_scheduler.step()
            torch.cuda.synchronize(device)

            metrics["checkpoint_loss_delta"] = abs(reference_loss - resumed_loss)
            resumed_model_match = _nested_equal(
                reference_model_state,
                restored_model.state_dict(),
            )
            resumed_optimizer_match = _nested_equal(
                reference_optimizer_state,
                restored_optimizer.state_dict(),
            )
            resumed_scheduler_match = _nested_equal(
                reference_scheduler_state,
                restored_scheduler.state_dict(),
            )
            resumed_model_delta = _nested_max_abs_delta(
                reference_model_state,
                restored_model.state_dict(),
            )
            resumed_optimizer_delta = _nested_max_abs_delta(
                reference_optimizer_state,
                restored_optimizer.state_dict(),
            )
            metrics["checkpoint_state_match"] = float(
                initial_model_match
                and initial_optimizer_match
                and initial_scheduler_match
                and step_match
                and resumed_scheduler_match
            )
            metrics["checkpoint_resume_model_max_abs_delta"] = resumed_model_delta
            metrics["checkpoint_resume_optimizer_max_abs_delta"] = resumed_optimizer_delta
            print(
                "CHECKPOINT_PARITY "
                f"initial_model={initial_model_match} "
                f"initial_optimizer={initial_optimizer_match} "
                f"initial_scheduler={initial_scheduler_match} "
                f"step={step_match} "
                f"resumed_model={resumed_model_match} "
                f"resumed_optimizer={resumed_optimizer_match} "
                f"resumed_scheduler={resumed_scheduler_match} "
                f"resumed_model_max_abs_delta={resumed_model_delta} "
                f"resumed_optimizer_max_abs_delta={resumed_optimizer_delta} "
                f"loss_delta={metrics['checkpoint_loss_delta']}",
                flush=True,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if not args.keep_checkpoint and checkpoint_path.exists():
                try:
                    checkpoint_path.unlink()
                except OSError as exc:
                    error = error or f"{type(exc).__name__}: {exc}"

    error_payload = [error]
    dist.broadcast_object_list(error_payload, src=0)
    if error_payload[0]:
        raise RuntimeError(f"checkpoint round trip failed on rank 0: {error_payload[0]}")
    return metrics


def _hf_image_transforms(image_size, train, heavy=False):
    """ImageNet transforms; ``heavy`` adds RandAugment and random erasing."""
    from torchvision import transforms

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    if train:
        steps = [transforms.RandomResizedCrop(image_size), transforms.RandomHorizontalFlip()]
        if heavy:
            steps.append(transforms.RandAugment(num_ops=2, magnitude=9))
        steps += [transforms.ToTensor(), normalize]
        if heavy:
            steps.append(transforms.RandomErasing(p=0.25))
        return transforms.Compose(steps)
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


class _HFStepTimer:
    """Collects per-step wall time and loss from the Trainer loop.

    HF Trainer owns the training loop, so the metric contract is populated from
    callbacks rather than from our own timing around each step.
    """

    def __init__(self, warmup_steps, sample_every, milestones):
        self.warmup_steps = warmup_steps
        self.sample_every = max(1, sample_every)
        # milestone_steps arrives as the raw CLI string; iterating it directly
        # would yield characters, so no step would ever match a milestone.
        if isinstance(milestones, str):
            milestones = [part for part in milestones.split(",") if part.strip()]
        self.milestones = {int(value) for value in (milestones or [])}
        self.step_times_ms = []
        self.loss_series = []
        self.milestone_losses = {}
        self.learning_rates = []
        self._last = None
        # The measured window is bounded by the first and last post-warmup step
        # so it covers the same work as step_times_ms. Timing the whole
        # trainer.train() call instead would fold Trainer startup and the warmup
        # steps into the denominator while the numerator counted neither.
        self._window_start = None
        self._window_end = None

    def on_step_begin(self):
        self._last = time.perf_counter()

    def on_step_end(self, step):
        if self._last is None:
            return
        now = time.perf_counter()
        elapsed_ms = (now - self._last) * 1000.0
        if step > self.warmup_steps:
            if self._window_start is None:
                self._window_start = self._last
            self._window_end = now
            self.step_times_ms.append(elapsed_ms)

    def on_log(self, step, loss, learning_rate=None):
        if learning_rate is not None:
            self.learning_rates.append(float(learning_rate))
        if loss is None:
            return
        if step % self.sample_every == 0 or step in self.milestones:
            self.loss_series.append({"step": step, "loss": float(loss), "time_seconds": self.elapsed()})
        if step in self.milestones:
            self.milestone_losses[str(step)] = float(loss)

    def elapsed(self):
        return 0.0 if self._window_start is None else time.perf_counter() - self._window_start

    def measured_window_s(self):
        if self._window_start is None or self._window_end is None:
            return 0.0
        return self._window_end - self._window_start


def _hf_evaluate(model, dataset, args, device, rank, world_size):
    """Distributed top-1/top-5/loss plus the label histogram, matching the
    native path's contract so W6 rows are comparable with W1/W3."""
    from torch.utils.data import DataLoader, DistributedSampler

    shard = len(dataset) // world_size
    batch_size = _exact_eval_batch_size(shard, args.batch_size)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=8, pin_memory=True)

    totals = torch.zeros(4, dtype=torch.float64, device=device)
    label_hist = torch.zeros(args.num_classes, dtype=torch.int64, device=device)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long().reshape(-1)
            with _autocast(args.precision):
                output = model(pixel_values=images).logits
                loss = loss_fn(output, labels)
            count = labels.numel()
            if count and (int(labels.min().item()) < 0 or int(labels.max().item()) >= args.num_classes):
                raise RuntimeError(f"evaluation labels outside [0, {args.num_classes})")
            label_hist += torch.bincount(labels, minlength=args.num_classes)
            predictions = output.topk(min(5, output.shape[1]), dim=1).indices
            correct = predictions.eq(labels.view(-1, 1))
            totals[0] += correct[:, :1].any(dim=1).sum()
            totals[1] += correct.any(dim=1).sum()
            totals[2] += loss.detach().double() * count
            totals[3] += count
    model.train()
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    dist.all_reduce(label_hist, op=dist.ReduceOp.SUM)
    samples = int(totals[3].item())
    top1, top5 = _accuracy_from_totals(int(totals[0].item()), int(totals[1].item()), samples)
    observed = int((label_hist > 0).sum().item())
    present = label_hist[label_hist > 0]
    return {
        "top1_accuracy_pct": top1,
        "top5_accuracy_pct": top5,
        "eval_loss": totals[2].item() / samples,
        "eval_sample_count": samples,
        "eval_completed": 1.0,
        "eval_label_classes_observed": float(observed),
        "eval_label_min_per_class": float(present.min().item()) if observed else 0.0,
        "eval_label_max_per_class": float(present.max().item()) if observed else 0.0,
    }


def _measure_hf_loader(trainer, args, device, world_size):
    """Data-loader-only throughput for the HF path.

    The native path measures rocAL; W6 feeds HF Trainer from a torch
    DataLoader, so the same row is filled by timing that loader in isolation
    with no model work in the loop.
    """
    loader = trainer.get_train_dataloader()
    iterator = iter(loader)
    images_per_step = args.batch_size * args.gradient_accumulation_steps * world_size

    def _drain(steps):
        for _ in range(steps):
            for _ in range(args.gradient_accumulation_steps):
                try:
                    next(iterator)
                except StopIteration:
                    return False
        return True

    # The DataLoader prefetches prefetch_factor batches per worker, so a short
    # window only reads batches that were decoded during warmup. Timing at
    # least four full prefetch queues measures the decode rate instead.
    prefetched = max(1, loader.num_workers) * (loader.prefetch_factor or 2)
    measured_steps = max(args.loader_benchmark_steps, 4 * prefetched // args.gradient_accumulation_steps)
    if not _drain(max(args.loader_warmup_steps, prefetched)):
        return {}
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    if not _drain(measured_steps):
        return {}
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    if elapsed <= 0 or measured_steps <= 0:
        return {}
    mean_ms = elapsed * 1000.0 / measured_steps
    return {"data_loader_images_per_sec": images_per_step * 1000.0 / mean_ms}


def _hf_checkpoint_roundtrip(trainer, args, device, rank):
    """Save, reload, and verify model state for the FSDP path.

    FSDP shards parameters across ranks, so parity has to be checked against a
    gathered FULL_STATE_DICT rather than each rank's local shard. Timings are
    measured on rank 0 and broadcast so every rank reports the same figures.
    """
    from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType

    model = trainer.model
    path = Path(args.checkpoint_path)
    is_fsdp = isinstance(model, FSDP)
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    def _gather():
        if not is_fsdp:
            return {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            return {k: v.detach().to("cpu") for k, v in model.state_dict().items()}

    original = _gather()

    timings = torch.zeros(2, dtype=torch.float64, device=device)
    if rank == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        torch.save({"model": original}, path)
        with open(path, "rb") as handle:
            os.fsync(handle.fileno())
        timings[0] = time.perf_counter() - start

        start = time.perf_counter()
        restored = torch.load(path, map_location="cpu", weights_only=False)["model"]
        timings[1] = time.perf_counter() - start
    else:
        restored = None
    dist.broadcast(timings, src=0)

    # Exact parity is checked on rank 0, which is the only rank holding the
    # gathered state, then broadcast so the verdict is consistent everywhere.
    verdict = torch.zeros(2, dtype=torch.float64, device=device)
    if rank == 0:
        match = _nested_equal(original, restored)
        delta = _nested_max_abs_delta(original, restored)
        verdict[0] = 1.0 if match else 0.0
        verdict[1] = float(delta)
    dist.broadcast(verdict, src=0)

    if rank == 0 and not args.keep_checkpoint:
        path.unlink(missing_ok=True)

    return {
        "checkpoint_save_seconds": float(timings[0].item()),
        "checkpoint_load_seconds": float(timings[1].item()),
        "checkpoint_state_match": float(verdict[0].item()),
        # A gathered save/load is a pure serialization round trip: there is no
        # resumed optimizer step here, so the deltas are the measured state
        # difference rather than a post-step comparison.
        "checkpoint_loss_delta": 0.0,
        "checkpoint_resume_model_max_abs_delta": float(verdict[1].item()),
        "checkpoint_resume_optimizer_max_abs_delta": 0.0,
    }


def _run_hf_trainer(args, rank, world_size, device, local_rank):
    """W6: ViT-base-patch16-224 through HF Trainer with FSDP.

    HF Trainer owns the training loop, so instead of driving steps ourselves we
    attach callbacks and reconstruct the same artifact the native path emits.
    That keeps W6's scorecard rows directly comparable with W1 and W3 rather
    than introducing a second, differently-defined set of numbers.
    """
    from torchvision.datasets import ImageFolder
    from transformers import (
        Trainer,
        TrainerCallback,
        TrainingArguments,
        ViTConfig,
        ViTForImageClassification,
    )

    torch.cuda.reset_peak_memory_stats(device)

    # Pretrained weights give a real accuracy reading from the first step, but
    # they also mean the loss curve starts converged, so the loss-milestone and
    # convergence rows only carry information on a from-scratch run.
    if args.pretrained:
        model = ViTForImageClassification.from_pretrained(
            args.model_id,
            num_labels=args.num_classes,
            ignore_mismatched_sizes=True,
        )
    else:
        config = ViTConfig.from_pretrained(args.model_id, num_labels=args.num_classes)
        model = ViTForImageClassification(config)
    model.to(device)

    train_dataset = ImageFolder(
        str(Path(args.dataset_path) / "train"),
        transform=_hf_image_transforms(args.image_size, True, heavy=args.augmentation == "heavy"),
    )
    eval_dataset = (
        ImageFolder(str(Path(args.dataset_path) / "val"), transform=_hf_image_transforms(args.image_size, False))
        if args.eval_enabled
        else None
    )
    if rank == 0:
        print(
            f"PYTORCH_VISION_HF_DATA train={len(train_dataset)} val={len(eval_dataset) if eval_dataset else 0}",
            flush=True,
        )
    _wait_for_all_ranks("hf_datasets", args.setup_timeout_seconds)

    timer = _HFStepTimer(args.warmup_steps, args.loss_curve_sample_every, args.milestone_steps)
    energy = {}

    class _Callback(TrainerCallback):
        def on_step_begin(self, _a, _s, _c, **_kw):
            timer.on_step_begin()

        def on_step_end(self, _a, state, _c, **_kw):
            timer.on_step_end(int(state.global_step))
            # RCCL and CodeCarbon's polling thread both use AMDSMI, and FSDP
            # creates its communicators during the first steps, so the tracker
            # starts after warmup. That also scopes energy to the measured
            # window, as on the native path.
            if not energy and state.global_step >= max(1, args.warmup_steps):
                energy["tracker"], energy["payload"] = _start_codecarbon(args, rank, torch.cuda.device_count())

        def on_log(self, _a, state, _c, logs=None, **_kw):
            if logs:
                timer.on_log(int(state.global_step), logs.get("loss"), logs.get("learning_rate"))

    output_dir = str(Path(args.output).parent / "hf_trainer")
    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=args.measure_steps + args.warmup_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        bf16=args.precision == "BF16",
        logging_steps=max(1, args.loss_curve_sample_every),
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
        # tqdm renders progress bars with carriage returns and non-UTF8 bytes
        # that the log transport cannot decode; the loss curve comes from the
        # callback, so the bars carry no information we need.
        disable_tqdm=True,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        # FSDP is the point of W6: shard parameters across every rank rather
        # than replicating them the way W1/W3's DDP path does. full_shard
        # spreads an 86M-parameter model over every rank, so on more than one
        # node the cross-node all-gathers can cost more than the added compute;
        # hybrid_shard shards within a node and replicates across nodes.
        fsdp=f"{args.fsdp_strategy} auto_wrap",
        fsdp_config={"transformer_layer_cls_to_wrap": ["ViTLayer"]},
        seed=2026,
    )

    def _collate(batch):
        pixel_values = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        return {"pixel_values": pixel_values, "labels": labels}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=_collate,
        callbacks=[_Callback()],
    )

    metrics_loader = _measure_hf_loader(trainer, args, device, world_size)

    trainer.train()
    measured_window_s = _reduce_max(timer.measured_window_s(), device)
    codecarbon = _stop_codecarbon(energy["tracker"], energy["payload"], rank)

    step_times = sorted(timer.step_times_ms)
    images_per_step = args.batch_size * args.gradient_accumulation_steps * world_size
    measured_images = images_per_step * max(1, len(timer.step_times_ms))
    images_per_sec = measured_images / measured_window_s if measured_window_s > 0 else 0.0
    images_per_sec_per_gpu = images_per_sec / world_size
    metrics = {
        "images_per_sec": images_per_sec,
        "images_per_sec_per_gpu": images_per_sec_per_gpu,
        "tflops_per_sec_per_gpu": images_per_sec_per_gpu * args.training_flops_per_image / 1e12,
        "mfu_pct": (images_per_sec_per_gpu * args.training_flops_per_image / 1e12) / args.peak_tflops_per_gpu * 100.0,
        "step_time_ms_mean": sum(step_times) / len(step_times) if step_times else 0.0,
        "step_time_ms_p50": _percentile(step_times, 50),
        "step_time_ms_p95": _percentile(step_times, 95),
        "peak_memory_allocated_mb": torch.cuda.max_memory_allocated(device) / 1e6,
        "peak_memory_reserved_mb": torch.cuda.max_memory_reserved(device) / 1e6,
    }
    if timer.loss_series:
        metrics["loss_initial"] = timer.loss_series[0]["loss"]
        metrics["loss_final"] = timer.loss_series[-1]["loss"]
        metrics["loss_curve_points"] = float(len(timer.loss_series))
        metrics["loss_curve_decreased"] = float(timer.loss_series[-1]["loss"] < timer.loss_series[0]["loss"])
    metrics.update(metrics_loader)
    if timer.learning_rates:
        metrics["learning_rate_initial"] = timer.learning_rates[0]
        metrics["learning_rate_final"] = timer.learning_rates[-1]
    metrics.update(_scorecard_milestone_metrics(timer.milestone_losses))

    if args.checkpoint_path:
        metrics.update(_hf_checkpoint_roundtrip(trainer, args, device, rank))

    evaluation = {}
    if eval_dataset is not None:
        evaluation = _hf_evaluate(trainer.model, eval_dataset, args, device, rank, world_size)
        metrics.update(evaluation)

    convergence = None
    if args.convergence_train_loss is not None:
        convergence = next((p for p in timer.loss_series if p["loss"] <= args.convergence_train_loss), None)
    if convergence is not None:
        metrics["convergence_step"] = convergence["step"]
        metrics["convergence_time_seconds"] = convergence["time_seconds"]

    return {
        "schema_version": 1,
        "workload": args.workload,
        "model": args.model,
        "backend": args.backend,
        "precision": args.precision,
        "image_size": args.image_size,
        "batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_global_batch_size": images_per_step,
        "world_size": world_size,
        "phase": args.phase,
        "run_mode": args.run_mode,
        "synthetic_data": False,
        "data_mode": args.data_mode,
        "rocal_device": args.rocal_device,
        "augmentation": args.augmentation,
        "dataset_path": args.dataset_path,
        "harness": "hf_trainer",
        "pretrained": bool(args.pretrained),
        "fsdp_strategy": args.fsdp_strategy,
        # Recorded for provenance: HF Trainer's own scheduler, not our
        # training.lr_schedule block.
        "hf_lr_scheduler_type": str(training_args.lr_scheduler_type),
        "hf_warmup_steps": int(training_args.get_warmup_steps(training_args.max_steps)),
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "device": torch.cuda.get_device_name(device),
        "measurement_window_s": measured_window_s,
        "completed_steps": args.measure_steps,
        "processed_images": measured_images,
        "loss_time_series": timer.loss_series,
        "milestone_losses": timer.milestone_losses,
        "evaluations": [dict(evaluation, step=args.measure_steps, time_seconds=measured_window_s)]
        if evaluation
        else [],
        "step_times_ms": timer.step_times_ms,
        "training_flops_per_image": args.training_flops_per_image,
        "peak_tflops_per_gpu": args.peak_tflops_per_gpu,
        "codecarbon": codecarbon,
        "metrics": metrics,
    }


def main():
    args = _parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    _start_distributed(device, local_rank, args.collective_timeout_seconds, args.startup_timeout_seconds)

    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)

    if args.harness in {"hf_trainer", "task"}:
        # HF Trainer and the task harness drive their own loops; both write the
        # same artifact contract and return.
        if args.harness == "task":
            import sys

            artifact = _tasks_module().run_task(sys.modules[__name__], args, rank, world_size, device, local_rank)
        else:
            artifact = _run_hf_trainer(args, rank, world_size, device, local_rank)
        artifact_error = ""
        if rank == 0:
            try:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.output).write_text(json.dumps(artifact, indent=2))
                print("PYTORCH_VISION_OK " + json.dumps(artifact["metrics"]), flush=True)
            except Exception as exc:
                artifact_error = f"{type(exc).__name__}: {exc}"
        _finish(rank, world_size, artifact_error, args.collective_timeout_seconds)
        return

    expected_backend = "mamba" if args.model.startswith("vim_") else "transformers" if args.model_id else "torchvision"
    if args.backend != expected_backend:
        raise ValueError(f"model {args.model!r} needs backend {expected_backend!r}, got {args.backend!r}")
    model = _build_model(args, device)
    model = DistributedDataParallel(model, device_ids=[local_rank], gradient_as_bucket_view=True)
    model.train()

    optimizer = _build_optimizer(model, args)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    # Validation loss stays plain cross-entropy so it compares across recipes.
    eval_loss_fn = torch.nn.CrossEntropyLoss()

    rocal_loader = None
    validation_loader = None
    data_loader_metrics = {}
    if args.data_mode == "rocal":
        rocal_loader = _build_rocal_loader(args, rank, world_size, split="train")
        if args.eval_enabled:
            # Size the eval batch so the per-rank shard divides exactly; see
            # _exact_eval_batch_size. Without this the final partial batch is
            # padded with repeats that corrupt the per-class distribution.
            eval_ranks = world_size if args.eval_steps is not None else _eval_ranks(args.eval_sample_count, world_size)
            shard = args.eval_sample_count // eval_ranks if args.eval_sample_count else 0
            eval_bs = _exact_eval_batch_size(shard, args.batch_size)
            if rank == 0 and (eval_bs != args.batch_size or eval_ranks != world_size):
                print(
                    f"PYTORCH_VISION_EVAL_BATCH shard={shard} train_bs={args.batch_size} eval_bs={eval_bs} "
                    f"eval_ranks={eval_ranks}/{world_size}",
                    flush=True,
                )
            validation_loader = _EvalLoader(
                (lambda: _build_rocal_loader(args, rank, eval_ranks, split="val", batch_size=eval_bs))
                if rank < eval_ranks
                else None,
                rebuild_each_pass=bool(args.eval_sample_count and args.eval_sample_count % eval_ranks),
                ranks=eval_ranks,
            )
        _wait_for_all_ranks("rocal_loaders", args.setup_timeout_seconds)
        measured_loader_metrics = _measure_rocal_loader(rocal_loader, args, device, rank, world_size)
        if measured_loader_metrics is not None:
            data_loader_metrics = measured_loader_metrics
        rocal_loader.reset()
        image_batches, label_batches = _distributed_next_rocal_batches(
            rocal_loader,
            args.gradient_accumulation_steps,
            args.channels_last,
            device,
            args.augmentation == "heavy",
        )
    else:
        generator = torch.Generator(device=device)
        generator.manual_seed(2026 + rank)
        image_batches = [
            torch.randn(
                args.batch_size,
                3,
                args.image_size,
                args.image_size,
                device=device,
                generator=generator,
            )
            for _ in range(args.gradient_accumulation_steps)
        ]
        label_batches = [
            torch.randint(
                0,
                args.num_classes,
                (args.batch_size,),
                device=device,
                generator=generator,
            )
            for _ in range(args.gradient_accumulation_steps)
        ]
        if args.channels_last:
            image_batches = [images.contiguous(memory_format=torch.channels_last) for images in image_batches]

    batches_per_epoch = _rocal_batches_per_epoch(rocal_loader, world_size) if rocal_loader is not None else None
    steps_per_epoch = (
        math.ceil(batches_per_epoch / args.gradient_accumulation_steps) if batches_per_epoch is not None else None
    )
    args.steps_per_epoch = steps_per_epoch
    scheduler = _build_lr_scheduler(optimizer, args, steps_per_epoch)
    learning_rate_initial = optimizer.param_groups[0]["lr"]

    dist.barrier()
    for _ in range(args.warmup_steps):
        if rocal_loader is not None:
            image_batches, label_batches = _distributed_next_rocal_batches(
                rocal_loader,
                args.gradient_accumulation_steps,
                args.channels_last,
                device,
                args.augmentation == "heavy",
            )
        _train_step(
            model,
            optimizer,
            loss_fn,
            image_batches,
            label_batches,
            args.precision,
        )
        scheduler.step()
    torch.cuda.synchronize(device)
    dist.barrier()
    torch.cuda.reset_peak_memory_stats(device)
    if rocal_loader is not None:
        rocal_loader.reset()

    local_times = []
    losses = []
    loss_time_series = []
    milestone_losses = {}
    evaluations = []
    all_finite = True
    # Energy is scoped to the measured window only. Warmup, the loader benchmark, and
    # checkpoint validation are excluded so images/kWh divides the same work it measures.
    # Energy is measured on rank 0's node only; on multi-node runs it is a
    # per-node figure, not a cluster total.

    tracker, codecarbon = _start_codecarbon(args, rank, torch.cuda.device_count())
    window_start = time.perf_counter()
    milestones = {int(value) for value in args.milestone_steps.split(",") if value.strip()}
    target_steps = args.measure_steps
    if args.max_epochs is not None:
        if rocal_loader is None:
            raise ValueError("epoch-based modes require rocAL data")
        target_steps = steps_per_epoch * args.max_epochs
    step = 0
    local_images_processed = 0
    eval_seconds = 0.0
    while step < target_steps:
        if args.max_duration_seconds is not None and time.perf_counter() - window_start >= args.max_duration_seconds:
            break
        wall_start = time.perf_counter()
        if rocal_loader is not None:
            batches = _distributed_next_rocal_batches(
                rocal_loader,
                args.gradient_accumulation_steps,
                args.channels_last,
                device,
                args.augmentation == "heavy",
            )
            if batches is None:
                raise RuntimeError(f"rocAL epoch ended before expected optimizer step {step + 1}/{target_steps}")
            image_batches, label_batches = batches
        local_images_processed += sum(labels.numel() for labels in label_batches)
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        loss = _train_step(
            model,
            optimizer,
            loss_fn,
            image_batches,
            label_batches,
            args.precision,
        )
        scheduler.step()
        stop.record()
        stop.synchronize()
        local_times.append(
            (time.perf_counter() - wall_start) * 1000.0 if rocal_loader is not None else float(start.elapsed_time(stop))
        )
        loss_value = float(loss.detach())
        if args.run_mode != "perf":
            loss_value = _reduce_mean(loss_value, world_size, device)
        losses.append(loss_value)
        step += 1
        elapsed = time.perf_counter() - window_start
        if step % args.loss_curve_sample_every == 0:
            loss_time_series.append({"step": step, "loss": loss_value, "time_seconds": elapsed})
        if step in milestones:
            milestone_losses[str(step)] = loss_value
        all_finite = all_finite and math.isfinite(loss_value)
        epoch_boundary = steps_per_epoch and step % steps_per_epoch == 0
        if validation_loader is not None and epoch_boundary and (step // steps_per_epoch) % args.eval_every_epochs == 0:
            eval_start = time.perf_counter()
            evaluation = _evaluate(model, validation_loader, eval_loss_fn, args, device)
            eval_seconds += time.perf_counter() - eval_start
            evaluation.update({"step": step, "time_seconds": elapsed})
            evaluations.append(evaluation)
            if args.stop_on_convergence and _evaluation_meets_stop_target(evaluation, args):
                break
    if validation_loader is not None and (not evaluations or evaluations[-1]["step"] != step):
        eval_start = time.perf_counter()
        evaluation = _evaluate(model, validation_loader, eval_loss_fn, args, device)
        eval_seconds += time.perf_counter() - eval_start
        evaluation.update({"step": step, "time_seconds": time.perf_counter() - window_start})
        evaluations.append(evaluation)
    torch.cuda.synchronize(device)
    # Evaluation passes are excluded so images/s measures training throughput only.
    measured_window_s = _reduce_max(time.perf_counter() - window_start - eval_seconds, device)
    codecarbon = _stop_codecarbon(tracker, codecarbon, rank)
    critical_times = _gather_step_times(local_times, rank, world_size, device)
    loss_initial = _reduce_mean(losses[0], world_size, device)
    loss_final = _reduce_mean(losses[-1], world_size, device)

    finite_tensor = torch.tensor(1 if all_finite else 0, device=device)
    dist.all_reduce(finite_tensor, op=dist.ReduceOp.MIN)
    if not finite_tensor.item():
        raise RuntimeError("non-finite loss observed during the measured window")

    allocated = torch.tensor(torch.cuda.max_memory_allocated(device) / 1e6, device=device)
    reserved = torch.tensor(torch.cuda.max_memory_reserved(device) / 1e6, device=device)
    processed_images = torch.tensor(local_images_processed, dtype=torch.int64, device=device)
    dist.all_reduce(allocated, op=dist.ReduceOp.MAX)
    dist.all_reduce(reserved, op=dist.ReduceOp.MAX)
    dist.all_reduce(processed_images, op=dist.ReduceOp.SUM)

    args.completed_steps = step
    learning_rate_final = optimizer.param_groups[0]["lr"]
    checkpoint_metrics = (
        _checkpoint_roundtrip(
            model,
            optimizer,
            scheduler,
            loss_fn,
            image_batches,
            label_batches,
            args,
            device,
            rank,
        )
        if args.checkpoint_path
        else {}
    )
    if args.codecarbon_required and not codecarbon.get("active"):
        raise RuntimeError(f"required CodeCarbon tracking failed: {codecarbon.get('reason')}")

    artifact_error = ""
    if rank == 0:
        try:
            ordered = sorted(critical_times)
            mean_ms = statistics.fmean(critical_times)
            global_batch = args.batch_size * args.gradient_accumulation_steps * world_size
            total_images = int(processed_images.item())
            images_per_sec = total_images / measured_window_s
            images_per_sec_per_gpu = images_per_sec / world_size
            tflops_per_sec_per_gpu = images_per_sec_per_gpu * args.training_flops_per_image / 1e12
            loss_values = [point["loss"] for point in loss_time_series]
            x_mean = (len(loss_values) - 1) / 2.0 if loss_values else 0.0
            y_mean = statistics.fmean(loss_values) if loss_values else 0.0
            slope_denominator = sum((index - x_mean) ** 2 for index in range(len(loss_values)))
            slope = (
                sum((index - x_mean) * (value - y_mean) for index, value in enumerate(loss_values)) / slope_denominator
                if slope_denominator
                else 0.0
            )
            # Convergence follows the jaxmaxtext definition: every configured
            # target must hold, and when both an eval-cadence target and a
            # per-step training-loss target are set, the later point wins.
            convergence = None
            eval_point = None
            if args.convergence_top1 is not None or args.convergence_eval_loss is not None:
                eval_point = next(
                    (item for item in evaluations if _evaluation_meets_stop_target(item, args)),
                    None,
                )
            train_point = None
            if args.convergence_train_loss is not None:
                train_point = next(
                    (p for p in loss_time_series if p["loss"] <= args.convergence_train_loss),
                    None,
                )
            wanted_eval = args.convergence_top1 is not None or args.convergence_eval_loss is not None
            wanted_train = args.convergence_train_loss is not None
            if (not wanted_eval or eval_point is not None) and (not wanted_train or train_point is not None):
                reached = [p for p in (eval_point, train_point) if p is not None]
                if reached:
                    convergence = max(reached, key=lambda p: p["step"])
            final_evaluation = evaluations[-1] if evaluations else {}
            milestone_metrics = _scorecard_milestone_metrics(milestone_losses)
            codecarbon_metrics = {}
            if codecarbon.get("active") and "emissions_kg_co2eq" in codecarbon:
                codecarbon_metrics = {"codecarbon_tracking_active": 1.0}
            artifact = {
                "schema_version": 1,
                "workload": args.workload,
                "model": args.model,
                "backend": args.backend,
                "precision": args.precision,
                "image_size": args.image_size,
                "batch_size_per_gpu": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "effective_global_batch_size": global_batch,
                "world_size": world_size,
                "phase": args.phase,
                "run_mode": args.run_mode,
                "synthetic_data": args.data_mode == "synthetic",
                "data_mode": args.data_mode,
                "rocal_device": args.rocal_device,
                "augmentation": args.augmentation,
                "dataset_path": args.dataset_path,
                "torch_version": torch.__version__,
                "hip_version": torch.version.hip,
                "device": torch.cuda.get_device_name(device),
                "measurement_window_s": measured_window_s,
                "completed_steps": step,
                "processed_images": total_images,
                "completed_epochs": step // steps_per_epoch if steps_per_epoch else None,
                "steps_per_epoch": steps_per_epoch,
                "batches_per_epoch": batches_per_epoch,
                "samples_per_rank_per_epoch": (
                    _rocal_samples_per_rank(rocal_loader, world_size) if rocal_loader is not None else None
                ),
                "loss_time_series": loss_time_series,
                "milestone_losses": milestone_losses,
                "evaluations": evaluations,
                "codecarbon": codecarbon,
                "training_flops_per_image": args.training_flops_per_image,
                "peak_tflops_per_gpu": args.peak_tflops_per_gpu,
                "lr_schedule": {
                    "name": args.lr_schedule,
                    "warmup_epochs": args.lr_warmup_epochs,
                    "milestones_epochs": [
                        int(epoch) for epoch in args.lr_milestones_epochs.split(",") if epoch.strip()
                    ],
                    "gamma": args.lr_gamma,
                },
                "step_times_ms": critical_times,
                "metrics": {
                    "images_per_sec": images_per_sec,
                    "images_per_sec_per_gpu": images_per_sec_per_gpu,
                    "tflops_per_sec_per_gpu": tflops_per_sec_per_gpu,
                    "mfu_pct": 100.0 * tflops_per_sec_per_gpu / args.peak_tflops_per_gpu,
                    "step_time_ms_mean": mean_ms,
                    "step_time_ms_p50": _percentile(ordered, 50),
                    "step_time_ms_p95": _percentile(ordered, 95),
                    "peak_memory_allocated_mb": allocated.item(),
                    "peak_memory_reserved_mb": reserved.item(),
                    **data_loader_metrics,
                    "learning_rate_initial": learning_rate_initial,
                    "learning_rate_final": learning_rate_final,
                    **checkpoint_metrics,
                    "loss_initial": loss_initial,
                    "loss_final": loss_final,
                    **milestone_metrics,
                    "loss_curve_points": len(loss_time_series),
                    "loss_curve_decreased": float(
                        len(loss_time_series) >= args.loss_curve_minimum_points and slope < 0
                    ),
                    **final_evaluation,
                    **(
                        {
                            "convergence_step": convergence["step"],
                            "convergence_time_seconds": convergence["time_seconds"],
                        }
                        if convergence is not None
                        else {}
                    ),
                    **codecarbon_metrics,
                },
            }
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(artifact, indent=2))
            print("PYTORCH_VISION_OK " + json.dumps(artifact["metrics"]), flush=True)
        except Exception as exc:
            artifact_error = f"{type(exc).__name__}: {exc}"

    _finish(rank, world_size, artifact_error, args.collective_timeout_seconds)


def _finish(rank, world_size, artifact_error, timeout_s):
    """Wait until every rank is done with the GPU, then leave.

    The exit is coordinated through the rendezvous store rather than a
    collective: NCCL sends are asynchronous, so a rank that exits straight
    after its last collective can cut off data its peers are still receiving.
    A final broadcast ended that way on four nodes, with the peers unpickling
    garbage after a 120 s timeout. Each rank synchronizes its device before
    checking in, so the count reaching world_size means no transfer is left.
    """
    store = dist.distributed_c10d._get_default_store()
    if rank == 0:
        store.set("cvs_artifact_error", artifact_error)
    torch.cuda.synchronize()
    store.add("cvs_ranks_done", 1)
    deadline = time.monotonic() + timeout_s
    while store.add("cvs_ranks_done", 0) < world_size:
        if time.monotonic() > deadline:
            raise RuntimeError(f"ranks did not all finish within {timeout_s}s of rank {rank}")
        time.sleep(0.2)
    error = store.get("cvs_artifact_error").decode()
    if error:
        raise RuntimeError(f"result artifact failed on rank 0: {error}")
    _exit_cleanly()


STARTUP_OK = "PYTORCH_VISION_STARTUP_OK"
STARTUP_FAILED = "PYTORCH_VISION_STARTUP_FAILED"


def _start_distributed(device, local_rank, collective_timeout_s, startup_timeout_s):
    """Bring up NCCL and prove it with one all-reduce, or leave at once.

    On MI325X with ROCm 7.2.4 about one 32-rank start in ten segfaults a rank
    inside libdrm_amdgpu during the NCCL connect. The surviving ranks then wait
    in NCCL teardown until the launch timeout. Any failure here, or no progress
    within startup_timeout_s, prints STARTUP_FAILED and exits without
    destructors, so the launcher can tell a failed start from a failed run.
    """

    def fail(reason):
        print(f"{STARTUP_FAILED} local_rank={local_rank} {reason}", flush=True)
        sys.stderr.flush()
        os._exit(1)

    deadline = threading.Timer(startup_timeout_s, fail, args=(f"no progress in {startup_timeout_s}s",))
    deadline.daemon = True
    deadline.start()
    try:
        dist.init_process_group(
            backend="nccl",
            device_id=device,
            timeout=datetime.timedelta(seconds=collective_timeout_s),
        )
        probe = torch.ones(1, device=device)
        dist.all_reduce(probe)
        torch.cuda.synchronize()
    except Exception as exc:
        fail(f"{type(exc).__name__}: {exc}".splitlines()[0])
    deadline.cancel()
    if local_rank == 0:
        print(STARTUP_OK, flush=True)


def _wait_for_all_ranks(name, timeout_s):
    """Meet every rank through the rendezvous store rather than through NCCL.

    Per-rank setup can take minutes longer on one node than another: listing
    ImageNet's 1.28M files on a node that had never read them took 5m40s while
    the warm node finished in seconds and timed out in its first collective.
    Ranks that arrive early wait here, outside NCCL's collective timeout.
    """
    store = dist.distributed_c10d._get_default_store()
    key = f"pytorch_vision_ready/{name}"
    store.add(key, 1)
    deadline = time.monotonic() + timeout_s
    while store.add(key, 0) < dist.get_world_size():
        if time.monotonic() > deadline:
            raise RuntimeError(f"not every rank finished {name} within {timeout_s}s")
        time.sleep(0.5)


def _exit_cleanly():
    """Leave without running rocAL and HIP static destructors.

    They segfault a rank at interpreter exit after a clean run, and destroying
    the process group first does the same on more than one node. The result
    file is already written.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
