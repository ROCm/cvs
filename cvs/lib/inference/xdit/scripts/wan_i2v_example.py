"""
Wan2.2-I2V-A14B-Diffusers xDiT image-to-video benchmark.
Usage: torchrun ... wan_i2v_example.py --model /path/to/model --input_image /path/to/image.jpg
"""

import argparse
import dataclasses
import glob
import json
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

_RANK_STRING_FIELDS = ("determinism_check_report_ranks",)
# rocm/pytorch-xdit ships ffmpeg without libx264, and imageio does not raise when that encoder is missing.
_VIDEO_CODECS = ("libx264", "libopenh264", "mpeg4")

_EXTRA_PYPACKAGES = os.environ.get("CVS_WAN_XFUSER_PYPACKAGES", "").strip()
if _EXTRA_PYPACKAGES and _EXTRA_PYPACKAGES not in sys.path:
    sys.path.insert(0, _EXTRA_PYPACKAGES)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input_image", required=True)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--ulysses_degree", type=int, default=8)
    parser.add_argument("--ring_degree", type=int, default=2)
    parser.add_argument("--warmup_steps", type=int, default=1)
    parser.add_argument("--num_repetitions", type=int, default=1)
    parser.add_argument(
        "--output_type",
        default="pil",
        help="Must be 'pil' or 'latent' in this xFuser build.",
    )
    parser.add_argument("--prompt", default="A person walking in a park, cinematic style.")
    parser.add_argument("--output_directory", default="/outputs")
    parser.add_argument("--save_video_path", default="results/video_i2v.mp4")
    parser.add_argument("--video_fps", type=int, default=16)
    parser.add_argument(
        "--timing_json_path",
        default="results/timing.json",
        help="Flux-style timing JSON: list of {\"pipe_time\": seconds} entries.",
    )
    parser.add_argument("--fid_steps", default="")
    parser.add_argument("--fid_samples", type=int, default=2)
    parser.add_argument("--fid_budget_s", type=float, default=0)
    return parser.parse_args()


def _unwrap_run_output(run_output):
    if isinstance(run_output, tuple) and run_output:
        return run_output[0]
    return run_output


def _extract_video_frames(run_output):
    output = _unwrap_run_output(run_output)
    if output is None:
        return None
    for name in ("videos", "frames", "images"):
        value = getattr(output, name, None)
        if not _nonempty(value):
            continue
        if isinstance(value, (list, tuple)):
            return value[0]
        return value
    if isinstance(output, (list, tuple)) and output:
        return output
    return None


def _to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _split_video_tensor(arr):
    if arr.ndim != 4:
        return [arr]
    if arr.shape[1] in (1, 3, 4):
        arr = np.transpose(arr, (0, 2, 3, 1))
    if arr.shape[-1] in (1, 2, 3, 4):
        return [arr[i] for i in range(arr.shape[0])]
    raise ValueError(f"Unsupported video tensor shape: {arr.shape}")


def _normalize_single_frame(arr):
    if arr.ndim == 4:
        frames = _split_video_tensor(arr)
        arr = frames[0] if len(frames) == 1 else None
        if arr is None:
            raise ValueError("Expected one frame, got video tensor")

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] < min(arr.shape[1], arr.shape[2]):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    return arr[..., :3]


def _iter_frames(raw_frames):
    if raw_frames is None:
        return []

    try:
        from PIL import Image

        if isinstance(raw_frames, Image.Image):
            return [_normalize_single_frame(np.asarray(raw_frames.convert("RGB")))]
    except ImportError:
        pass

    if isinstance(raw_frames, (list, tuple)):
        if len(raw_frames) == 1:
            return _iter_frames(raw_frames[0])
        out = []
        for item in raw_frames:
            out.extend(_iter_frames(item))
        return out

    arr = _to_numpy(raw_frames)
    if arr.ndim == 4:
        return [_normalize_single_frame(f) for f in _split_video_tensor(arr)]
    return [_normalize_single_frame(arr)]


def _write_sample_frames(arrays, save_path, limit=8):
    """The FID scorer needs two images per step. One video's frames cover that if the mp4 will not decode."""
    if not arrays or not save_path.lower().endswith(".mp4"):
        return
    parent = os.path.dirname(save_path)
    stem = os.path.splitext(os.path.basename(save_path))[0]
    chosen = np.linspace(0, len(arrays) - 1, num=min(limit, len(arrays)), dtype=int)
    try:
        from PIL import Image
    except ImportError:
        return
    for slot, index in enumerate(chosen):
        Image.fromarray(arrays[int(index)]).save(os.path.join(parent, f"{stem}_frame_{slot}.png"))


def _write_mp4(arrays, save_path, fps):
    import imageio

    errors = []
    for codec in _VIDEO_CODECS:
        if os.path.isfile(save_path):
            os.remove(save_path)
        try:
            writer = imageio.get_writer(save_path, fps=fps, codec=codec, macro_block_size=1)
            for arr in arrays:
                writer.append_data(arr)
            writer.close()
        except Exception as exc:
            errors.append(f"{codec}: {exc}")
            continue
        if os.path.isfile(save_path) and os.path.getsize(save_path) > 0:
            return codec
        errors.append(f"{codec}: encoder produced no file")
    print(f"Video export failed: {'; '.join(errors)}", file=sys.stderr)
    return None


def _export_video(run_output, save_path, fps):
    raw = _extract_video_frames(run_output)
    arrays = _iter_frames(raw)
    if not arrays:
        print("Video export failed: no frames in runner output", file=sys.stderr)
        return False

    parent = os.path.dirname(save_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    _write_sample_frames(arrays, save_path)
    codec = _write_mp4(arrays, save_path, fps)
    if codec is None:
        return False
    print(f"Saved video to {save_path} ({len(arrays)} frames, {codec})")
    return True


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
    for name in ("videos", "frames", "images"):
        if _nonempty(getattr(output, name, None)):
            return True
    return isinstance(output, (list, tuple)) and len(output) > 0


def _media_owner_rank(output):
    """Pick one rank that holds decoded frames. Parallel VAE decode is not always rank 0."""
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
    """``__post_init__`` rewrites these to lists, but re-parsing the same dict demands strings again."""
    for name in names:
        if name not in raw_args:
            continue
        coerced = _as_rank_string(raw_args[name])
        if coerced != raw_args[name]:
            print(f"wan_i2v_example: coerced {name}={raw_args[name]!r} to {coerced!r}")
        raw_args[name] = coerced


def _write_timing_json(timing_json_path, pipe_times):
    parent = os.path.dirname(timing_json_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = [{"pipe_time": float(t)} for t in pipe_times]
    with open(timing_json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Wrote timing JSON to {timing_json_path}")


def _run_wan_fid_sweep(args, runner, input_args, rank, last_wall_s, fid_started):
    raw_steps = str(getattr(args, "fid_steps", "") or "").strip()
    if not raw_steps:
        return
    from fid_sweep import broadcast_value, count_step_samples, rewrite_manifest_from_disk, run_fid_sweep

    steps = [int(part) for part in raw_steps.split(",") if part.strip()]
    samples = int(args.fid_samples)
    budget_s = float(args.fid_budget_s)
    base_seed = int(getattr(args, "seed", 0) or 0)
    manifest_path = os.path.join(args.output_directory, "results", "fid", "manifest.json")

    def generate_one(nfe, sample_index):
        run_args = dict(input_args)
        run_args["num_inference_steps"] = int(nfe)
        run_args["seed"] = base_seed + sample_index + 1
        torch.cuda.synchronize()
        wall_start = time.time()
        output, _timings = runner.run(run_args)
        torch.cuda.synchronize()
        elapsed = time.time() - wall_start
        dest = os.path.join(args.output_directory, "results", "fid", f"nfe_{int(nfe)}", f"seed_{sample_index}.mp4")
        owner = _media_owner_rank(output)
        if owner is not None and rank == owner:
            _export_video(output, dest, args.video_fps)
        if dist.is_initialized():
            dist.barrier()
        return elapsed

    try:
        run_fid_sweep(
            rank,
            steps,
            samples,
            budget_s,
            fid_started,
            last_wall_s if rank == 0 else 0.0,
            int(args.num_inference_steps),
            generate_one,
            manifest_path,
        )
        pipe_s = float(last_wall_s) if last_wall_s and last_wall_s > 0 else 0.0
        for step in steps:
            for _attempt in range(samples):
                decision = None
                if rank == 0:
                    dest = os.path.join(args.output_directory, "results", "fid", f"nfe_{int(step)}")
                    have = count_step_samples(dest)
                    elapsed = time.monotonic() - fid_started
                    allowance = budget_s if budget_s > 0 else elapsed + pipe_s * samples * max(len(steps), 1) * 1.25
                    if have < samples and pipe_s > 0 and elapsed + pipe_s * 1.25 <= allowance:
                        decision = (int(step), have)
                decision = broadcast_value(decision, rank)
                if not decision:
                    break
                generate_one(*decision)
    except Exception as exc:
        print(f"FID sweep failed after the timed benchmark: {exc}")
    finally:
        if rank == 0:
            rewrite_manifest_from_disk(manifest_path, steps, samples, budget_s)


def main():
    fid_started = time.monotonic()
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
    dit_parallel_size = ulysses * ring

    config_kwargs = {
        "model": "Wan2.2-I2V",
        "dit_parallel_size": dit_parallel_size,
        "ulysses_degree": ulysses,
        "ring_degree": ring,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "prompt": args.prompt,
        "output_type": args.output_type,
        "warmup_steps": args.warmup_steps,
        "input_images": [args.input_image],
        "output_directory": args.output_directory,
    }
    config_kwargs.update(_string_only_defaults(xFuserArgs, _RANK_STRING_FIELDS))

    config = xFuserArgs(**config_kwargs)

    raw_args = vars(config)
    _stringify_rank_fields(raw_args, _RANK_STRING_FIELDS)

    runner = xFuserModelRunner(raw_args)
    runner.model.settings.model_name = args.model
    if raw_args.get("input_images") is None:
        raw_args["input_images"] = [args.input_image]
    _stringify_rank_fields(raw_args, _RANK_STRING_FIELDS)
    input_args = runner.preprocess_args(raw_args)
    runner.initialize(input_args)

    pipe_times = []
    last_output = None
    last_timings = None

    try:
        for _ in range(max(int(args.warmup_steps), 0)):
            runner.run(input_args)

        rank = dist.get_rank() if dist.is_initialized() else 0
        local_rank = int(torch.cuda.current_device())
        elapsed = 0.0

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

        # save() selects the rank that owns the frames and may all_reduce, so every rank must call it.
        try:
            runner.save(output=last_output, timings=last_timings)
        except Exception as exc:
            print(f"rank {rank}: xFuser save() failed: {exc}", file=sys.stderr)

        owner = _media_owner_rank(last_output)
        if (
            args.output_type == "pil"
            and owner is not None
            and rank == owner
            and not os.path.isfile(args.save_video_path)
        ):
            if not _export_video(last_output, args.save_video_path, args.video_fps):
                _copy_first_mp4(args.output_directory, args.save_video_path)

        if dist.is_initialized():
            dist.barrier()

        if rank == 0 and args.output_type == "pil" and not os.path.isfile(args.save_video_path):
            if not _copy_first_mp4(args.output_directory, args.save_video_path):
                print(f"video was not written to {args.save_video_path}", file=sys.stderr)
                raise SystemExit(1)

        _run_wan_fid_sweep(args, runner, input_args, rank, elapsed, fid_started)
    finally:
        runner.cleanup()


if __name__ == "__main__":
    main()
