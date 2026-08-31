"""
Wan2.2-I2V-A14B-Diffusers xDiT image-to-video benchmark.
Usage: torchrun ... wan_i2v_example.py --model /path/to/model --input_image /path/to/image.jpg
"""

import argparse
import glob
import json
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

SHARED_HOME = "/shared/amdgpu/home/dl_dcgpu_aac_service_request_qle"
_EXTRA_PYPACKAGES = os.environ.get("CVS_WAN_XFUSER_PYPACKAGES", "").strip() or f"{SHARED_HOME}/pypackages"
if _EXTRA_PYPACKAGES not in sys.path:
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
    return parser.parse_args()


def _unwrap_run_output(run_output):
    if isinstance(run_output, tuple) and run_output:
        return run_output[0]
    return run_output


def _extract_video_frames(run_output):
    output = _unwrap_run_output(run_output)
    if output is None:
        return None
    if hasattr(output, "videos") and output.videos:
        videos = output.videos
        return videos[0] if isinstance(videos, list) and videos else videos
    if hasattr(output, "frames") and output.frames:
        frames = output.frames
        return frames[0] if isinstance(frames, list) and frames else frames
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


def _export_video(run_output, save_path, fps):
    raw = _extract_video_frames(run_output)
    arrays = _iter_frames(raw)
    if not arrays:
        print("Video export failed: no frames in runner output", file=sys.stderr)
        return False

    parent = os.path.dirname(save_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    try:
        import imageio

        writer = imageio.get_writer(save_path, fps=fps, codec="libx264")
        for arr in arrays:
            writer.append_data(arr)
        writer.close()
        print(f"Saved video to {save_path} ({len(arrays)} frames)")
        return True
    except Exception as exc:
        print(f"Video export failed: {exc}", file=sys.stderr)
        return False


def _copy_first_mp4(output_directory, save_path):
    mp4_files = sorted(glob.glob(os.path.join(output_directory, "*.mp4")))
    if not mp4_files:
        return False
    parent = os.path.dirname(save_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    shutil.copy(mp4_files[0], save_path)
    print(f"Copied video {mp4_files[0]} to {save_path}")
    return True


def _write_timing_json(timing_json_path, pipe_times):
    parent = os.path.dirname(timing_json_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = [{"pipe_time": float(t)} for t in pipe_times]
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

    pipe_times = []
    last_output = None
    last_timings = None

    try:
        for _ in range(max(int(args.warmup_steps), 0)):
            runner.run(input_args)

        rank = dist.get_rank() if dist.is_initialized() else 0
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
            try:
                runner.save(output=last_output, timings=last_timings)
            except Exception as exc:
                print(f"xFuser save() failed: {exc}", file=sys.stderr)
            if args.output_type == "pil":
                if not _export_video(last_output, args.save_video_path, args.video_fps):
                    _copy_first_mp4(args.output_directory, args.save_video_path)
    finally:
        runner.cleanup()


if __name__ == "__main__":
    main()
