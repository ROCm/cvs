"""
Wan2.1-T2V-14B-Diffusers xDiT text-to-video benchmark.
Usage: torchrun ... wan_t2v_example.py --model /path/to/Wan2.1-T2V-14B-Diffusers --prompt "..."
"""

import argparse
import dataclasses
import glob
import json
import os
import shutil
import sys
import time

import torch
import torch.distributed as dist

_RANK_STRING_FIELDS = ("determinism_check_report_ranks",)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--xfuser_model", default="Wan2.1-T2V")
    parser.add_argument("--prompt", default="a cute rabbit hopping through a sunlit meadow")
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--ulysses_degree", type=int, default=1)
    parser.add_argument("--ring_degree", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--num_repetitions", type=int, default=1)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--flow_shift", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attention_backend", default="AITER")
    parser.add_argument("--cross_attention_backend", default="AITER")
    parser.add_argument("--output_type", default="pil")
    parser.add_argument("--output_directory", default="/outputs")
    parser.add_argument("--save_video_path", default="results/video_i2v.mp4")
    parser.add_argument("--video_fps", type=int, default=16)
    parser.add_argument("--timing_json_path", default="results/timing.json")
    return parser.parse_args()


def _nonempty(value):
    if value is None:
        return False
    try:
        return len(value) > 0
    except TypeError:
        return True


def _output_has_media(output):
    if output is None:
        return False
    return _nonempty(getattr(output, "videos", None))


def _video_frames(output):
    videos = getattr(output, "videos", None)
    if isinstance(videos, list) and len(videos) == 1:
        return videos[0]
    return videos


def _media_owner_rank(output):
    """Parallel VAE decode may leave frames on the last rank rather than rank 0."""
    if not dist.is_initialized():
        return 0 if _output_has_media(output) else None
    rank = dist.get_rank()
    holder = torch.tensor([rank if _output_has_media(output) else -1], dtype=torch.int64, device="cuda")
    dist.all_reduce(holder, op=dist.ReduceOp.MAX)
    chosen = int(holder.item())
    if chosen < 0:
        return None
    return chosen


def _copy_first_mp4(output_directory, save_path):
    mp4_files = sorted(glob.glob(os.path.join(output_directory, "**", "*.mp4"), recursive=True))
    save_abs = os.path.abspath(save_path)
    mp4_files = [path for path in mp4_files if os.path.abspath(path) != save_abs]
    if not mp4_files:
        return False
    parent = os.path.dirname(save_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    shutil.copy(mp4_files[0], save_path)
    print(f"Copied video {mp4_files[0]} to {save_path}")
    return True


def _export_video(output, save_path, fps):
    frames = _video_frames(output)
    if not _nonempty(frames):
        print("Video export failed: no frames in runner output", file=sys.stderr)
        return False
    parent = os.path.dirname(save_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    try:
        from diffusers.utils import export_to_video

        export_to_video(frames, save_path, fps=fps)
        print(f"Saved video to {save_path}")
        return True
    except Exception as exc:
        print(f"Video export failed: {exc}", file=sys.stderr)
        return False


def _string_only_defaults(args_cls, names):
    """Some xFuser builds default these fields to None but reject non-strings in __post_init__."""
    if not dataclasses.is_dataclass(args_cls):
        return {}
    supported = {field.name for field in dataclasses.fields(args_cls)}
    return {name: "" for name in names if name in supported}


def _as_rank_string(value):
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set, frozenset)):
        return ",".join(str(item) for item in value)
    return str(value)


def _stringify_rank_fields(raw_args, names):
    """__post_init__ rewrites these to lists, but re-parsing the same dict demands strings again."""
    for name in names:
        if name not in raw_args:
            continue
        coerced = _as_rank_string(raw_args[name])
        if coerced != raw_args[name]:
            print(f"wan_t2v_example: coerced {name}={raw_args[name]!r} to {coerced!r}")
        raw_args[name] = coerced


def _write_timing_json(timing_json_path, pipe_times):
    parent = os.path.dirname(timing_json_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = [{"pipe_time": float(elapsed)} for elapsed in pipe_times]
    with open(timing_json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Wrote timing JSON to {timing_json_path}")


def main():
    args = parse_args()
    if not os.path.isabs(args.save_video_path):
        args.save_video_path = os.path.join(args.output_directory, args.save_video_path)
    if not os.path.isabs(args.timing_json_path):
        args.timing_json_path = os.path.join(args.output_directory, args.timing_json_path)
    os.makedirs(args.output_directory, exist_ok=True)

    from xfuser import xFuserArgs
    from xfuser.runner import xFuserModelRunner

    ulysses = args.ulysses_degree
    ring = args.ring_degree
    config_kwargs = {
        "model": args.xfuser_model,
        "attention_backend": args.attention_backend,
        "cross_attention_backend": args.cross_attention_backend,
        "dit_parallel_size": ulysses * ring,
        "ulysses_degree": ulysses,
        "ring_degree": ring,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "guidance_scale": args.guidance_scale,
        "flow_shift": args.flow_shift,
        "output_type": args.output_type,
        "output_directory": args.output_directory,
        "warmup_steps": 0,
        "num_iterations": 1,
        "input_images": [],
    }
    config_kwargs.update(_string_only_defaults(xFuserArgs, _RANK_STRING_FIELDS))
    config = xFuserArgs(**config_kwargs)
    raw_args = vars(config)
    _stringify_rank_fields(raw_args, _RANK_STRING_FIELDS)

    runner = xFuserModelRunner(raw_args)
    runner.model.settings.model_name = args.model
    if hasattr(runner.model.settings, "fps"):
        runner.model.settings.fps = args.video_fps
    _stringify_rank_fields(raw_args, _RANK_STRING_FIELDS)
    input_args = runner.preprocess_args(raw_args)
    runner.initialize(input_args)

    pipe_times = []
    last_output = None
    last_timings = None
    rank = dist.get_rank() if dist.is_initialized() else 0
    try:
        for _ in range(max(int(args.warmup_steps), 0)):
            runner.run(input_args)

        local_rank = int(torch.cuda.current_device())
        for step in range(max(int(args.num_repetitions), 1)):
            torch.cuda.synchronize()
            start_time = time.time()
            output, timings = runner.run(input_args)
            torch.cuda.synchronize()
            elapsed = time.time() - start_time
            if rank == 0:
                pipe_times.append(elapsed)
                peak_mem = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}") / 1e9
                print(f"step {step}: epoch time: {elapsed:.2f} sec, memory: {peak_mem:.2f} GB")
            last_output = output
            last_timings = timings

        if rank == 0:
            _write_timing_json(args.timing_json_path, pipe_times)

        # save() may all_reduce to pick the rank that owns the frames.
        try:
            runner.save(output=last_output, timings=last_timings)
        except Exception as exc:
            print(f"rank {rank}: xFuser save() failed: {exc}", file=sys.stderr)

        owner = _media_owner_rank(last_output)
        if owner is not None and rank == owner and not os.path.isfile(args.save_video_path):
            if not _export_video(last_output, args.save_video_path, args.video_fps):
                _copy_first_mp4(args.output_directory, args.save_video_path)

        if dist.is_initialized():
            dist.barrier()

        if rank == 0 and not os.path.isfile(args.save_video_path):
            if not _copy_first_mp4(args.output_directory, args.save_video_path):
                print(f"video was not written to {args.save_video_path}", file=sys.stderr)
                raise SystemExit(1)
    finally:
        runner.cleanup()


if __name__ == "__main__":
    main()
