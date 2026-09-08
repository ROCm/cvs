"""Container-side ResNet training benchmark launched with torchrun."""

from __future__ import annotations

import argparse
import contextlib
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
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--training-flops-per-image", type=float, required=True)
    parser.add_argument("--peak-tflops-per-gpu", type=float, required=True)
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--checkpoint-loss-tolerance", type=float, default=1e-5)
    parser.add_argument("--keep-checkpoint", action="store_true")
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
    return torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )


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


def _device_used_mb(device):
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return (total_bytes - free_bytes) / 1e6


def _checkpoint_roundtrip(model, optimizer, loss_fn, image_batches, label_batches, args, device, rank):
    metrics = {
        "checkpoint_save_seconds": 0.0,
        "checkpoint_load_seconds": 0.0,
        "checkpoint_state_match": 1.0,
        "checkpoint_loss_delta": 0.0,
        "checkpoint_resume_model_max_abs_delta": 0.0,
        "checkpoint_resume_optimizer_max_abs_delta": 0.0,
    }
    if not args.checkpoint_path:
        raise ValueError("--checkpoint-path is required for W1 checkpoint validation")

    dist.barrier()
    error = ""
    if rank == 0:
        checkpoint_path = Path(args.checkpoint_path)
        try:
            raw_model = model.module
            optimizer_step = args.warmup_steps + args.measure_steps

            save_start = time.perf_counter()
            model_state = _to_cpu(raw_model.state_dict())
            optimizer_state = _to_cpu(optimizer.state_dict())
            checkpoint = {
                "model": model_state,
                "optimizer": optimizer_state,
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
            load_start = time.perf_counter()
            loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            restored_model.load_state_dict(loaded["model"], strict=True)
            restored_optimizer.load_state_dict(loaded["optimizer"])
            torch.cuda.synchronize(device)
            metrics["checkpoint_load_seconds"] = time.perf_counter() - load_start

            initial_model_match = _nested_equal(model_state, restored_model.state_dict())
            initial_optimizer_match = _nested_equal(optimizer_state, restored_optimizer.state_dict())
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
            reference_model_state = _to_cpu(raw_model.state_dict())
            reference_optimizer_state = _to_cpu(optimizer.state_dict())

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
            resumed_model_delta = _nested_max_abs_delta(
                reference_model_state,
                restored_model.state_dict(),
            )
            resumed_optimizer_delta = _nested_max_abs_delta(
                reference_optimizer_state,
                restored_optimizer.state_dict(),
            )
            metrics["checkpoint_state_match"] = float(initial_model_match and initial_optimizer_match and step_match)
            metrics["checkpoint_resume_model_max_abs_delta"] = resumed_model_delta
            metrics["checkpoint_resume_optimizer_max_abs_delta"] = resumed_optimizer_delta
            print(
                "CHECKPOINT_PARITY "
                f"initial_model={initial_model_match} "
                f"initial_optimizer={initial_optimizer_match} "
                f"step={step_match} "
                f"resumed_model={resumed_model_match} "
                f"resumed_optimizer={resumed_optimizer_match} "
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
    dist.init_process_group(backend="nccl", device_id=device)

    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)

    if args.backend != "torchvision":
        raise ValueError(f"unsupported model backend: {args.backend}")
    model = _build_model(args, device)
    model = DistributedDataParallel(model, device_ids=[local_rank], gradient_as_bucket_view=True)
    model.train()

    optimizer = _build_optimizer(model, args)
    loss_fn = torch.nn.CrossEntropyLoss()

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

    dist.barrier()
    for _ in range(args.warmup_steps):
        _train_step(
            model,
            optimizer,
            loss_fn,
            image_batches,
            label_batches,
            args.precision,
        )
    torch.cuda.synchronize(device)
    dist.barrier()
    torch.cuda.reset_peak_memory_stats(device)

    local_times = []
    losses = []
    all_finite = True
    used_before_mb = _device_used_mb(device)
    window_start = time.perf_counter()
    for _ in range(args.measure_steps):
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
        stop.record()
        stop.synchronize()
        local_times.append(float(start.elapsed_time(stop)))
        loss_value = float(loss.detach())
        losses.append(loss_value)
        all_finite = all_finite and math.isfinite(loss_value)
    torch.cuda.synchronize(device)
    measured_window_s = _reduce_max(time.perf_counter() - window_start, device)
    used_after_mb = _device_used_mb(device)

    critical_times = _gather_step_times(local_times, rank, world_size, device)
    loss_initial = _reduce_mean(losses[0], world_size, device)
    loss_final = _reduce_mean(losses[-1], world_size, device)

    finite_tensor = torch.tensor(1 if all_finite else 0, device=device)
    dist.all_reduce(finite_tensor, op=dist.ReduceOp.MIN)
    if not finite_tensor.item():
        raise RuntimeError("non-finite loss observed during the measured window")

    allocated = torch.tensor(torch.cuda.max_memory_allocated(device) / 1e6, device=device)
    reserved = torch.tensor(torch.cuda.max_memory_reserved(device) / 1e6, device=device)
    used = torch.tensor(max(used_before_mb, used_after_mb), device=device)
    dist.all_reduce(allocated, op=dist.ReduceOp.MAX)
    dist.all_reduce(reserved, op=dist.ReduceOp.MAX)
    dist.all_reduce(used, op=dist.ReduceOp.MAX)

    checkpoint_metrics = _checkpoint_roundtrip(
        model,
        optimizer,
        loss_fn,
        image_batches,
        label_batches,
        args,
        device,
        rank,
    )

    artifact_error = ""
    if rank == 0:
        try:
            ordered = sorted(critical_times)
            mean_ms = statistics.fmean(critical_times)
            global_batch = args.batch_size * args.gradient_accumulation_steps * world_size
            images_per_sec = global_batch * args.measure_steps / measured_window_s
            images_per_sec_per_gpu = images_per_sec / world_size
            tflops_per_sec_per_gpu = images_per_sec_per_gpu * args.training_flops_per_image / 1e12
            artifact = {
                "schema_version": 1,
                "workload": "W1",
                "model": args.model,
                "backend": args.backend,
                "precision": args.precision,
                "image_size": args.image_size,
                "batch_size_per_gpu": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "effective_global_batch_size": global_batch,
                "world_size": world_size,
                "synthetic_data": True,
                "torch_version": torch.__version__,
                "hip_version": torch.version.hip,
                "device": torch.cuda.get_device_name(device),
                "measurement_window_s": measured_window_s,
                "training_flops_per_image": args.training_flops_per_image,
                "peak_tflops_per_gpu": args.peak_tflops_per_gpu,
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
                    "device_memory_used_mb_observed": used.item(),
                    **checkpoint_metrics,
                    "loss_initial": loss_initial,
                    "loss_final": loss_final,
                },
            }
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(artifact, indent=2))
            print("PYTORCH_VISION_W1_OK " + json.dumps(artifact["metrics"]), flush=True)
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
