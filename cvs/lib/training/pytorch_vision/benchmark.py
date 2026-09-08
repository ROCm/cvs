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


def _train_step(model, optimizer, loss_fn, images, labels, precision):
    optimizer.zero_grad(set_to_none=True)
    with _autocast(precision):
        output = model(images)
        loss = loss_fn(output, labels)
    loss.backward()
    optimizer.step()
    return loss


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
    from torchvision.models import get_model

    model = get_model(args.model, weights=None, num_classes=args.num_classes)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    model = model.to(device)
    model = DistributedDataParallel(model, device_ids=[local_rank], gradient_as_bucket_view=True)
    model.train()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    generator = torch.Generator(device=device)
    generator.manual_seed(2026 + rank)
    images = torch.randn(
        args.batch_size,
        3,
        args.image_size,
        args.image_size,
        device=device,
        generator=generator,
    )
    labels = torch.randint(
        0,
        args.num_classes,
        (args.batch_size,),
        device=device,
        generator=generator,
    )
    if args.channels_last:
        images = images.contiguous(memory_format=torch.channels_last)

    dist.barrier()
    for _ in range(args.warmup_steps):
        _train_step(model, optimizer, loss_fn, images, labels, args.precision)
    torch.cuda.synchronize(device)
    dist.barrier()
    torch.cuda.reset_peak_memory_stats(device)

    local_times = []
    losses = []
    all_finite = True
    window_start = time.perf_counter()
    for _ in range(args.measure_steps):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        loss = _train_step(model, optimizer, loss_fn, images, labels, args.precision)
        stop.record()
        stop.synchronize()
        local_times.append(float(start.elapsed_time(stop)))
        loss_value = float(loss.detach())
        losses.append(loss_value)
        all_finite = all_finite and math.isfinite(loss_value)
    torch.cuda.synchronize(device)
    measured_window_s = _reduce_max(time.perf_counter() - window_start, device)

    critical_times = _gather_step_times(local_times, rank, world_size, device)
    loss_initial = _reduce_mean(losses[0], world_size, device)
    loss_final = _reduce_mean(losses[-1], world_size, device)

    finite_tensor = torch.tensor(1 if all_finite else 0, device=device)
    dist.all_reduce(finite_tensor, op=dist.ReduceOp.MIN)
    if not finite_tensor.item():
        raise RuntimeError("non-finite loss observed during the measured window")

    allocated = torch.tensor(torch.cuda.max_memory_allocated(device) / 1e6, device=device)
    reserved = torch.tensor(torch.cuda.max_memory_reserved(device) / 1e6, device=device)
    dist.all_reduce(allocated, op=dist.ReduceOp.MAX)
    dist.all_reduce(reserved, op=dist.ReduceOp.MAX)

    if rank == 0:
        ordered = sorted(critical_times)
        mean_ms = statistics.fmean(critical_times)
        global_batch = args.batch_size * world_size
        images_per_sec = global_batch * args.measure_steps / measured_window_s
        artifact = {
            "schema_version": 1,
            "workload": "W1",
            "model": args.model,
            "backend": args.backend,
            "precision": args.precision,
            "image_size": args.image_size,
            "batch_size_per_gpu": args.batch_size,
            "world_size": world_size,
            "synthetic_data": True,
            "torch_version": torch.__version__,
            "hip_version": torch.version.hip,
            "device": torch.cuda.get_device_name(device),
            "measurement_window_s": measured_window_s,
            "step_times_ms": critical_times,
            "metrics": {
                "images_per_sec": images_per_sec,
                "images_per_sec_per_gpu": images_per_sec / world_size,
                "step_time_ms_mean": mean_ms,
                "step_time_ms_p50": _percentile(ordered, 50),
                "step_time_ms_p95": _percentile(ordered, 95),
                "peak_memory_allocated_mb": allocated.item(),
                "peak_memory_reserved_mb": reserved.item(),
                "loss_initial": loss_initial,
                "loss_final": loss_final,
            },
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(artifact, indent=2))
        print("PYTORCH_VISION_W1_OK " + json.dumps(artifact["metrics"]), flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
