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
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=["torchvision"], default="torchvision")
    parser.add_argument("--precision", choices=["BF16", "FP16", "FP32"], default="BF16")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--measure-steps", type=int, default=50)
    parser.add_argument(
        "--run-mode",
        choices=["smoke", "perf", "train_5k", "train_to_accuracy"],
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
    parser.add_argument("--learning-rate", type=float, default=0.1)
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
    parser.add_argument("--data-mode", choices=["synthetic", "rocal"], default="synthetic")
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


def _build_model(args, device):
    from torchvision.models import get_model

    model = get_model(args.model, weights=None, num_classes=args.num_classes)
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
    pipe = Pipeline(
        batch_size=batch_size,
        num_threads=args.rocal_num_threads,
        device_id=int(os.environ["LOCAL_RANK"]),
        seed=2026 + rank,
        rocal_cpu=cpu,
        tensor_dtype=types.FLOAT,
        tensor_layout=types.NCHW,
        prefetch_queue_depth=6,
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        output_memory_type=types.HOST_MEMORY if cpu else types.DEVICE_MEMORY,
    )
    with pipe:
        jpegs, _labels = fn.readers.file(
            file_root=data_path,
            pad_last_batch=split == "train",
        )
        if split == "train":
            decoded = fn.decoders.image_slice(
                jpegs,
                output_type=types.RGB,
                file_root=data_path,
                shard_id=rank,
                num_shards=world_size,
                random_shuffle=True,
            )
            resized = fn.resize(
                decoded,
                resize_width=args.image_size,
                resize_height=args.image_size,
                output_layout=types.NHWC,
                output_dtype=types.UINT8,
                interpolation_type=types.TRIANGULAR_INTERPOLATION,
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
            resized = fn.resize(
                decoded,
                resize_shorter=256,
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
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
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


def _rocal_batches_per_epoch(loader):
    sample_count = int(loader.iterator_length)
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


def _evaluate(model, loader, loss_fn, args, device):
    model.eval()
    loader.reset()
    totals = torch.zeros(4, dtype=torch.float64, device=device)
    # Per-class label histogram. A correct label mapping over a full ImageNet
    # validation set covers every class; a scrambled or truncated label space
    # shows up here as missing classes or a lopsided distribution, neither of
    # which accuracy alone would reveal.
    label_hist = torch.zeros(args.num_classes, dtype=torch.int64, device=device)
    steps = 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_target = (
        args.eval_sample_count // world_size + (1 if rank < args.eval_sample_count % world_size else 0)
        if args.eval_steps is None
        else None
    )
    local_samples = 0
    try:
        while steps < args.eval_steps if args.eval_steps is not None else local_samples < local_target:
            try:
                batches = _distributed_next_rocal_batches(
                    loader,
                    1,
                    args.channels_last,
                    device,
                    heavy=False,
                    reset_on_end=False,
                )
                if batches is None:
                    break
                images, labels = batches
            except StopIteration:
                break
            if local_target is not None:
                remaining = local_target - local_samples
                images[0] = images[0][:remaining]
                labels[0] = labels[0][:remaining]
            with torch.no_grad(), _autocast(args.precision):
                output = model(images[0])
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
        loader.reset()
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


def main():
    args = _parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        backend="nccl",
        device_id=device,
        timeout=datetime.timedelta(seconds=args.collective_timeout_seconds),
    )

    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)

    if args.backend != "torchvision":
        raise ValueError(f"unsupported model backend: {args.backend}")
    model = _build_model(args, device)
    model = DistributedDataParallel(model, device_ids=[local_rank], gradient_as_bucket_view=True)
    model.train()

    optimizer = _build_optimizer(model, args)
    loss_fn = torch.nn.CrossEntropyLoss()

    rocal_loader = None
    validation_loader = None
    data_loader_metrics = {}
    if args.data_mode == "rocal":
        rocal_loader = _build_rocal_loader(args, rank, world_size, split="train")
        if args.eval_enabled:
            # Size the eval batch so the per-rank shard divides exactly; see
            # _exact_eval_batch_size. Without this the final partial batch is
            # padded with repeats that corrupt the per-class distribution.
            shard = args.eval_sample_count // world_size if args.eval_sample_count else 0
            eval_bs = _exact_eval_batch_size(shard, args.batch_size)
            if rank == 0 and eval_bs != args.batch_size:
                print(
                    f"PYTORCH_VISION_EVAL_BATCH shard={shard} train_bs={args.batch_size} eval_bs={eval_bs}",
                    flush=True,
                )
            validation_loader = _build_rocal_loader(args, rank, world_size, split="val", batch_size=eval_bs)
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

    batches_per_epoch = _rocal_batches_per_epoch(rocal_loader) if rocal_loader is not None else None
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
    epoch_local_samples = 0
    while step < target_steps:
        if args.max_duration_seconds is not None and time.perf_counter() - window_start >= args.max_duration_seconds:
            break
        if rocal_loader is not None and steps_per_epoch and step and step % steps_per_epoch == 0:
            rocal_loader.reset()
        wall_start = time.perf_counter()
        if rocal_loader is not None:
            batches = _distributed_next_rocal_batches(
                rocal_loader,
                args.gradient_accumulation_steps,
                args.channels_last,
                device,
                args.augmentation == "heavy",
                reset_on_end=args.max_epochs is None,
            )
            if batches is None:
                raise RuntimeError(f"rocAL epoch ended before expected optimizer step {step + 1}/{target_steps}")
            image_batches, label_batches = batches
            if args.max_epochs is not None:
                epoch_local_samples += sum(labels.numel() for labels in label_batches)
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
        if args.max_epochs is not None and epoch_boundary:
            expected_local_samples = int(rocal_loader.iterator_length)
            complete = torch.tensor(
                int(epoch_local_samples == expected_local_samples),
                dtype=torch.int32,
                device=device,
            )
            dist.all_reduce(complete, op=dist.ReduceOp.MIN)
            if not complete.item():
                raise RuntimeError(
                    "rocAL epoch sample mismatch on at least one rank; "
                    f"local_consumed={epoch_local_samples}, local_expected={expected_local_samples}"
                )
            epoch_local_samples = 0
        if validation_loader is not None and epoch_boundary and (step // steps_per_epoch) % args.eval_every_epochs == 0:
            evaluation = _evaluate(model, validation_loader, loss_fn, args, device)
            evaluation.update({"step": step, "time_seconds": elapsed})
            evaluations.append(evaluation)
            if args.stop_on_convergence and _evaluation_meets_stop_target(evaluation, args):
                break
    if validation_loader is not None and (not evaluations or evaluations[-1]["step"] != step):
        evaluation = _evaluate(model, validation_loader, loss_fn, args, device)
        evaluation.update({"step": step, "time_seconds": time.perf_counter() - window_start})
        evaluations.append(evaluation)
    torch.cuda.synchronize(device)
    measured_window_s = _reduce_max(time.perf_counter() - window_start, device)
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
                "samples_per_rank_per_epoch": (int(rocal_loader.iterator_length) if rocal_loader is not None else None),
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

    artifact_error_payload = [artifact_error]
    dist.broadcast_object_list(artifact_error_payload, src=0)
    if artifact_error_payload[0]:
        dist.destroy_process_group()
        raise RuntimeError(f"result artifact failed on rank 0: {artifact_error_payload[0]}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
