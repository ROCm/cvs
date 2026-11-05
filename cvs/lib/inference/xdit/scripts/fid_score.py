"""
Frechet Inception Distance at each sampling-step directory.

``--generated-root`` holds one subdirectory per NFE (``10`` or ``nfe_10``).
Each directory needs at least two images, or frames taken from videos.
``--reference-dir`` is the real-image set. ``--inception`` is a local
Inception-v3 checkpoint. This script does not download weights.

Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
"""

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

MARKER = "CVS_FID_CURVE "
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
VIDEO_SUFFIXES = (".mp4", ".webm", ".mov")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _nfe_from_name(name):
    stem = name[4:] if name.startswith("nfe_") else name
    try:
        step = int(stem)
    except ValueError:
        return None
    if step < 1:
        return None
    return step


def _list_media(directory):
    images = []
    videos = []
    for dirpath, _dirnames, names in os.walk(directory):
        for entry in sorted(names):
            path = os.path.join(dirpath, entry)
            lower = entry.lower()
            if lower.endswith(IMAGE_SUFFIXES):
                images.append(path)
            elif lower.endswith(VIDEO_SUFFIXES):
                videos.append(path)
    return images, videos


def _load_image(path):
    with Image.open(path) as handle:
        return handle.convert("RGB")


def _video_frames(path, sample_frames):
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise SystemExit(f"video FID needs imageio ({exc})") from exc
    reader = imageio.get_reader(str(path), "ffmpeg")
    try:
        count = reader.count_frames()
        if count < 1:
            return []
        indexes = np.linspace(0, count - 1, num=min(sample_frames, count), dtype=int)
        frames = []
        for index in indexes:
            frame = reader.get_data(int(index))
            frames.append(Image.fromarray(frame).convert("RGB"))
        return frames
    finally:
        reader.close()


def _collect_images(directory, sample_frames):
    images, videos = _list_media(directory)
    frames = [_load_image(path) for path in images]
    for path in videos:
        frames.extend(_video_frames(path, sample_frames))
    return frames


def _preprocess(image):
    image = image.resize((299, 299), Image.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
    return torch.from_numpy(array).permute(2, 0, 1)


def _load_inception(path, device):
    from torchvision.models import inception_v3

    # init_weights=True pulls SciPy, which this image does not ship. The checkpoint replaces the weights.
    try:
        model = inception_v3(weights=None, transform_input=False, aux_logits=True, init_weights=False)
    except TypeError:
        model = inception_v3(weights=None, transform_input=False, aux_logits=True)
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    cleaned = {key.replace("module.", ""): value for key, value in state.items()}
    model.load_state_dict(cleaned, strict=False)
    model.fc = torch.nn.Identity()
    model.eval()
    return model.to(device)


def _features(model, images, device):
    batch = torch.stack([_preprocess(image) for image in images]).to(device)
    with torch.no_grad():
        output = model(batch)
    if isinstance(output, (tuple, list)):
        output = output[0]
    array = output.detach().float().cpu().numpy()
    return array.reshape(array.shape[0], -1)


def _stats(features):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, np.atleast_2d(sigma)


def _matrix_sqrt(matrix):
    # rocm/pytorch-xdit does not ship scipy. Eigenvalues of a covariance product
    # are real; tiny negatives are rounding error.
    values, vectors = np.linalg.eig(np.asarray(matrix, dtype=np.float64))
    values = np.clip(np.real(values), 0.0, None)
    root = vectors @ np.diag(np.sqrt(values)) @ np.linalg.inv(vectors)
    return np.real(root)


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    sigma1 = sigma1 + np.eye(sigma1.shape[0]) * eps
    sigma2 = sigma2 + np.eye(sigma2.shape[0]) * eps
    diff = mu1 - mu2
    covmean = _matrix_sqrt(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = _matrix_sqrt((sigma1 + offset).dot(sigma2 + offset))
    value = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    return float(np.real(value))


def _generated_reference(generated_root, sample_frames):
    """Highest step-count directory this run wrote, when no external reference is present."""
    best_dir = None
    best_nfe = -1
    best_images = []
    for name in sorted(os.listdir(generated_root)):
        directory = os.path.join(generated_root, name)
        if not os.path.isdir(directory):
            continue
        nfe = _nfe_from_name(name)
        if nfe is None or nfe <= best_nfe:
            continue
        found = _collect_images(directory, sample_frames)
        if len(found) < 2:
            continue
        best_dir, best_nfe, best_images = directory, nfe, found
    if best_dir is None:
        raise SystemExit(
            f"{generated_root} has no step directory with 2 images to use as the FID reference"
        )
    print(f"FID reference is the generated samples in {best_dir} ({len(best_images)} images)", flush=True)
    return best_images, best_dir


def _score_directory(model, device, directory, reference_stats, sample_frames):
    images = _collect_images(directory, sample_frames)
    if len(images) < 2:
        raise SystemExit(f"{directory} has {len(images)} sample(s); FID needs at least 2")
    mu, sigma = _stats(_features(model, images, device))
    ref_mu, ref_sigma = reference_stats
    return frechet_distance(mu, sigma, ref_mu, ref_sigma), len(images)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--generated-root", required=True)
    parser.add_argument("--inception", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sampler", default="")
    parser.add_argument("--sample-frames", type=int, default=8)
    args = parser.parse_args()

    if not os.path.isdir(args.generated_root):
        raise SystemExit(f"generated root not found: {args.generated_root}")
    if not os.path.isfile(args.inception):
        raise SystemExit(f"inception checkpoint not found: {args.inception}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_inception(args.inception, device)
    if os.path.isdir(args.reference_dir):
        reference_images = _collect_images(args.reference_dir, args.sample_frames)
        reference_label = args.reference_dir
    else:
        reference_images = []
        reference_label = args.reference_dir
    if len(reference_images) < 2:
        reference_images, reference_label = _generated_reference(args.generated_root, args.sample_frames)
    reference_stats = _stats(_features(model, reference_images, device))

    points = []
    for name in sorted(os.listdir(args.generated_root)):
        directory = os.path.join(args.generated_root, name)
        if not os.path.isdir(directory):
            continue
        nfe = _nfe_from_name(name)
        if nfe is None:
            continue
        try:
            fid, count = _score_directory(model, device, directory, reference_stats, args.sample_frames)
        except SystemExit as exc:
            print(f"Skipping {directory}: {exc}")
            continue
        point = {"nfe": nfe, "fid": fid, "sample_count": count}
        if args.sampler:
            point["sampler"] = args.sampler
        points.append(point)
    points.sort(key=lambda item: item["nfe"])
    if len(points) < 2:
        raise SystemExit("FID curve needs at least two NFE directories under " + args.generated_root)

    payload = {
        "sampler": args.sampler,
        "reference_count": len(reference_images),
        "reference_dir": reference_label,
        "inception": args.inception,
        "points": points,
    }
    parent = os.path.dirname(args.output)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(MARKER + json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
