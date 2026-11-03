"""
Score saved FLUX PNGs, or sampled WAN video frames, with CLIPScore.

CLIPScore is cosine similarity between CLIP image and text embeddings,
multiplied by 100. The one-line ``CVS_CLIP_SCORE`` record is what the
controller parses; ``clipscore.json`` is the copy left next to the media.

WAN output is an MP4, so the video path samples frames and reports
``mean_frame_clip_score`` instead of an image ``clip_score``.

Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
"""

import argparse
import json
import sys
from pathlib import Path

CLIP_SCORE_MARKER = "CVS_CLIP_SCORE "
FLUX_IMAGE_PATTERNS = ("flux_*.png", "flux2_*.png")
WAN_VIDEO_NAMES = ("video.mp4", "video_i2v.mp4")
DEFAULT_SAMPLE_FRAMES = 8
CLIP_CONTEXT_LENGTH = 77


def clip_score_from_cosine(cosine):
    """Map a cosine similarity in [-1, 1] onto the CLIPScore scale."""
    return max(float(cosine), 0.0) * 100.0


def mean_clip_score(values):
    if not values:
        raise ValueError("no CLIP scores")
    return sum(values) / len(values)


def find_flux_images(image_dir):
    root = Path(image_dir)
    found = []
    for pattern in FLUX_IMAGE_PATTERNS:
        found.extend(path for path in root.rglob(pattern) if path.is_file() and path.stat().st_size > 0)
    return sorted(set(found))


def find_wan_video(video_dir, video_name=None):
    """Return the shallowest non-empty WAN MP4 under ``video_dir``."""
    root = Path(video_dir)
    names = (video_name,) if video_name else WAN_VIDEO_NAMES
    found = []
    for name in names:
        found.extend(path for path in root.rglob(name) if path.is_file() and path.stat().st_size > 0)
    if not found:
        return None
    return sorted(found, key=lambda path: (len(path.relative_to(root).parts), str(path)))[0]


def sample_frame_indices(frame_count, sample_count):
    """Evenly spaced frame indexes, including the first and last frame."""
    count = int(frame_count)
    if count < 1:
        raise ValueError("video has no frames")
    samples = int(sample_count)
    if samples < 1:
        raise ValueError("sample_frames must be at least 1")
    if samples >= count:
        return list(range(count))
    # One sample is the I2V conditioning frame.
    if samples == 1:
        return [0]
    raw = [int(round(i * (count - 1) / (samples - 1))) for i in range(samples)]
    chosen = []
    used = set()
    for index in raw:
        if index not in used:
            chosen.append(index)
            used.add(index)
            continue
        replacement = next((candidate for candidate in range(count) if candidate not in used), None)
        if replacement is None:
            break
        chosen.append(replacement)
        used.add(replacement)
    chosen.sort()
    return chosen


def build_clipscore_payload(prompt, model, image_scores, prompt_truncated=False):
    scores = [{"image": name, "clip_score": round(float(value), 4)} for name, value in image_scores]
    return {
        "prompt": prompt,
        "model": model,
        "image_count": len(scores),
        "scores": scores,
        "clip_score": round(mean_clip_score([item["clip_score"] for item in scores]), 4),
        "prompt_truncated": bool(prompt_truncated),
    }


def build_frame_clipscore_payload(prompt, model, video_name, frame_count, frame_scores, prompt_truncated=False):
    scores = [{"frame": int(index), "clip_score": round(float(value), 4)} for index, value in frame_scores]
    values = [item["clip_score"] for item in scores]
    return {
        "prompt": prompt,
        "model": model,
        "video": video_name,
        "frame_count": int(frame_count),
        "sample_count": len(scores),
        "frame_indices": [item["frame"] for item in scores],
        "scores": scores,
        "mean_frame_clip_score": round(mean_clip_score(values), 4),
        "min_frame_clip_score": round(min(values), 4),
        "prompt_truncated": bool(prompt_truncated),
    }


def clip_context_length(model):
    """CLIP text context, including the start and end tokens."""
    config = getattr(model, "config", None)
    for source in (getattr(config, "text_config", None), config):
        length = getattr(source, "max_position_embeddings", None)
        if length:
            return int(length)
    return CLIP_CONTEXT_LENGTH


def clip_token_count(tokenizer, prompt):
    encoded = tokenizer(prompt, add_special_tokens=True)
    ids = encoded["input_ids"]
    if hasattr(ids, "shape"):
        return int(ids.shape[-1])
    if ids and isinstance(ids[0], (list, tuple)):
        return len(ids[0])
    return len(ids)


def _cosine(image_embeds, text_embeds):
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
    return float((image_embeds * text_embeds).sum().item())


def _frame_to_image(frame):
    from PIL import Image

    if hasattr(frame, "detach"):
        frame = frame.detach().cpu().numpy()
    if getattr(frame, "ndim", None) == 2:
        frame = frame[:, :, None].repeat(3, axis=2)
    elif getattr(frame, "shape", (None,))[-1] == 4:
        frame = frame[..., :3]
    return Image.fromarray(frame).convert("RGB")


def _load_clip_processor(model_name):
    from transformers import CLIPProcessor

    try:
        return CLIPProcessor.from_pretrained(model_name, use_fast=False, local_files_only=True)
    except TypeError:
        return CLIPProcessor.from_pretrained(model_name, local_files_only=True)


def _offline_clip_error(model_name, exc):
    return RuntimeError(
        f"CLIP model {model_name!r} is not available offline in this container ({exc}). "
        "Stage it under HF_HOME or set clip_model to a local directory. "
        "This test does not download weights."
    )


def score_labeled_images(labeled_images, prompt, model_name):
    # CPU keeps CLIP weights off the GPUs that just held the diffusion model.
    import torch
    from transformers import CLIPModel

    device = "cpu"
    try:
        model = CLIPModel.from_pretrained(model_name, local_files_only=True).to(device)
        processor = _load_clip_processor(model_name)
    except Exception as exc:
        raise _offline_clip_error(model_name, exc) from exc
    model.eval()
    context = clip_context_length(model)
    token_count = clip_token_count(processor.tokenizer, prompt)
    truncated = token_count > context
    if truncated:
        print(f"CLIP prompt truncated from {token_count} to {context} tokens", file=sys.stderr)
    scored = []
    for label, image in labeled_images:
        inputs = processor(
            text=[prompt],
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=context,
        )
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        cosine = _cosine(outputs.image_embeds, outputs.text_embeds)
        scored.append((label, clip_score_from_cosine(cosine)))
    return scored, truncated


def score_images(image_paths, prompt, model_name):
    from PIL import Image

    labeled = []
    for path in image_paths:
        with Image.open(path) as handle:
            labeled.append((path.name, handle.convert("RGB")))
    return score_labeled_images(labeled, prompt, model_name)


def score_video_frames(frames, indices, prompt, model_name):
    labeled = [(index, _frame_to_image(frames[index])) for index in indices]
    return score_labeled_images(labeled, prompt, model_name)


def _read_with_imageio(path):
    import imageio.v2 as imageio

    reader = imageio.get_reader(str(path), "ffmpeg")
    try:
        return [frame for frame in reader]
    finally:
        reader.close()


def _read_with_cv2(path):
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open {path}")
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    return frames


def _read_with_torchvision(path):
    from torchvision.io import read_video

    video, _, _ = read_video(str(path), pts_unit="sec")
    if video is None or int(video.shape[0]) == 0:
        return []
    return [video[index] for index in range(int(video.shape[0]))]


def read_video_frames(path):
    """Decode ``path`` with the first video reader the container provides."""
    errors = []
    saw_empty = False
    for reader in (_read_with_imageio, _read_with_cv2, _read_with_torchvision):
        try:
            frames = reader(path)
        except ImportError as exc:
            errors.append(str(exc))
            continue
        except Exception as exc:
            errors.append(str(exc))
            continue
        if frames:
            return frames
        saw_empty = True
    if saw_empty:
        raise ValueError(f"{path} has no frames")
    detail = "; ".join(errors) if errors else "no video decoder available"
    raise ValueError(f"cannot decode {path}: {detail}")


def _write_payload(path, payload):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(CLIP_SCORE_MARKER + json.dumps(payload, separators=(",", ":")), flush=True)


def _score_flux_images(args):
    images = find_flux_images(args.image_dir)
    if not images:
        print(f"No FLUX PNG images found under {args.image_dir}", file=sys.stderr)
        return 1
    try:
        scored, truncated = score_images(images, args.prompt, args.model)
    except Exception as exc:
        print(f"CLIP score failed: {exc}", file=sys.stderr)
        return 1
    payload = build_clipscore_payload(args.prompt, args.model, scored, prompt_truncated=truncated)
    _write_payload(args.output, payload)
    return 0


def _score_video(args):
    sample_count = DEFAULT_SAMPLE_FRAMES if args.sample_frames is None else args.sample_frames
    if sample_count < 1:
        print("sample_frames must be at least 1", file=sys.stderr)
        return 1
    video = find_wan_video(args.image_dir, args.video_name)
    if video is None:
        expected = args.video_name or "video.mp4 or video_i2v.mp4"
        print(f"No WAN video {expected} found under {args.image_dir}", file=sys.stderr)
        return 1
    try:
        frames = read_video_frames(video)
        indices = sample_frame_indices(len(frames), sample_count)
        scored, truncated = score_video_frames(frames, indices, args.prompt, args.model)
        payload = build_frame_clipscore_payload(
            args.prompt,
            args.model,
            video.name,
            len(frames),
            scored,
            prompt_truncated=truncated,
        )
    except Exception as exc:
        print(f"CLIP score failed: {exc}", file=sys.stderr)
        return 1
    _write_payload(args.output, payload)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score FLUX images or WAN video frames with CLIP")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sample-frames", type=int, default=None)
    parser.add_argument("--video-name", default=None)
    args = parser.parse_args(argv)
    if args.sample_frames is not None or args.video_name:
        return _score_video(args)
    return _score_flux_images(args)


if __name__ == "__main__":
    sys.exit(main())
