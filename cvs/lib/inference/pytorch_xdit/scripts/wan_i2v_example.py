"""
Wan2.2-I2V-A14B-Diffusers xFuser image-to-video benchmark.

Usage::

    torchrun --nproc_per_node=8 wan_i2v_example.py \\
        --model /model \\
        --input_image /path/to/image.jpg \\
        --output_directory outputs

Optional extra Python packages (cluster-specific) may be prepended via the
``CVS_WAN_XFUSER_PYPACKAGES`` environment variable.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time
from typing import Any

import torch
import torch.distributed as dist

_EXTRA_PYPACKAGES = os.environ.get("CVS_WAN_XFUSER_PYPACKAGES", "").strip()
if _EXTRA_PYPACKAGES and _EXTRA_PYPACKAGES not in sys.path:
    sys.path.insert(0, _EXTRA_PYPACKAGES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input_image", required=True)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--ulysses_degree", type=int, default=8)
    parser.add_argument("--ring_degree", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=1)
    parser.add_argument("--num_repetitions", type=int, default=1)
    parser.add_argument(
        "--output_type",
        default="np",
        help="xFuser/diffusers output type. Do not use latent if you need video.mp4.",
    )
    parser.add_argument("--prompt", default="A person walking in a park, cinematic style.")
    parser.add_argument(
        "--output_directory",
        default="/outputs/outputs",
        help="Directory for xFuser save_output, rank0_step JSON, and video.mp4.",
    )
    parser.add_argument(
        "--save_video_path",
        default="/outputs/outputs/video.mp4",
        help="Canonical MP4 path for CVS artifact checks.",
    )
    parser.add_argument("--video_fps", type=int, default=16)
    parser.add_argument(
        "--benchmark_output_directory",
        default="/outputs/outputs",
        help="Directory for rank0_step*.json timing files",
    )
    return parser.parse_args()


def _unwrap_run_output(run_output: Any) -> Any:
    if isinstance(run_output, tuple) and run_output:
        return run_output[0]
    return run_output


def _extract_video_frames(run_output: Any) -> Any | None:
    output = _unwrap_run_output(run_output)
    if output is None:
        return None
    if hasattr(output, "videos") and output.videos:
        videos = output.videos
        if isinstance(videos, list) and videos:
            return videos[0]
        return videos
    if hasattr(output, "frames") and output.frames:
        frames = output.frames
        if isinstance(frames, list) and frames:
            return frames[0]
        return frames
    return None


def _export_video(run_output: Any, save_path: str, fps: int) -> bool:
    frames = _extract_video_frames(run_output)
    if frames is None:
        return False

    from diffusers.utils import export_to_video

    parent = os.path.dirname(save_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    export_to_video(frames, save_path, fps=fps)
    print(f"Saved video to {save_path}")
    return True


def _copy_first_mp4(output_directory: str, save_path: str) -> bool:
    mp4_files = sorted(glob.glob(os.path.join(output_directory, "*.mp4")))
    if not mp4_files:
        return False
    parent = os.path.dirname(save_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    shutil.copy(mp4_files[0], save_path)
    print(f"Copied video {mp4_files[0]} to {save_path}")
    return True


def _write_step_timing(output_dir: str, step: int, elapsed: float) -> None:
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"rank0_step{step}.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump({"total_time": elapsed}, handle)


def main() -> None:
    args = parse_args()
    if not args.benchmark_output_directory:
        args.benchmark_output_directory = args.output_directory
    if not args.save_video_path:
        args.save_video_path = os.path.join(args.output_directory, "video.mp4")

    os.makedirs(args.output_directory, exist_ok=True)
    os.makedirs(args.benchmark_output_directory, exist_ok=True)

    from xfuser import xFuserArgs
    from xfuser.runner import xFuserModelRunner

    ulysses = args.ulysses_degree
    ring = args.ring_degree
    dit_parallel_size = ulysses * ring

    config = xFuserArgs(
        model="Wan2.2-I2V",
        dit_parallel_size=dit_parallel_size,
        ulysses_degree=ulysses,
        ring_degree=ring,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        prompt=args.prompt,
        output_type=args.output_type,
        warmup_steps=args.warmup_steps,
        input_images=[args.input_image],
        output_directory=args.output_directory,
    )

    runner = xFuserModelRunner(vars(config))
    runner.model.settings.model_name = args.model
    raw_args = vars(config)
    if raw_args.get("input_images") is None:
        raw_args["input_images"] = [args.input_image]
    input_args = runner.preprocess_args(raw_args)
    runner.initialize(input_args)

    for _ in range(max(int(args.warmup_steps), 0)):
        runner.run(input_args)

    rank = dist.get_rank() if dist.is_initialized() else 0
    local_rank = int(torch.cuda.current_device())
    last_step = max(int(args.num_repetitions), 1) - 1

    for step in range(max(int(args.num_repetitions), 1)):
        torch.cuda.synchronize()
        start_time = time.time()
        output, timings = runner.run(input_args)
        torch.cuda.synchronize()
        elapsed = time.time() - start_time

        if rank == 0:
            _write_step_timing(args.benchmark_output_directory, step, elapsed)
            peak_mem = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}") / 1e9
            print(f"epoch time: {elapsed:.2f} sec, memory: {peak_mem:.2f} GB")
            if step == last_step:
                try:
                    runner.save(output=output, timings=timings)
                except Exception as exc:
                    print(f"xFuser save() failed: {exc}", file=sys.stderr)
                if not _export_video(output, args.save_video_path, args.video_fps):
                    if not _copy_first_mp4(args.output_directory, args.save_video_path):
                        print(
                            "Video export failed: runner output had no decodable frames. "
                            "Use --output_type np (not latent).",
                            file=sys.stderr,
                        )

    runner.cleanup()


if __name__ == "__main__":
    main()
