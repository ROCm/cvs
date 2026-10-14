"""
FLUX and WAN CLIPScore command builder and threshold check.

The scorer runs inside the benchmark container, where the PNG or MP4 bodies
exist. Remote artifact staging copies media names only, so the controller
reads the one-line ``CVS_CLIP_SCORE`` record from that command. WAN scores
sampled video frames and gates on ``mean_frame_clip_score``.

Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
"""

import base64
import json
import shlex
from pathlib import Path

DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_SAMPLE_FRAMES = 8
WAN_NATIVE_CLIP_VIDEO = "video.mp4"
WAN_DIFFUSERS_CLIP_VIDEO = "video_i2v.mp4"
CLIP_SCORE_MARKER = "CVS_CLIP_SCORE "
CLIP_SCORE_TIMEOUT_S = 900
CLIP_SCORE_SCRIPT = Path(__file__).resolve().parent / "scripts" / "clipscore.py"


def resolve_clip_model(params):
    configured = str((params or {}).get("clip_model") or "").strip()
    return configured or DEFAULT_CLIP_MODEL


def resolve_clip_sample_frames(params):
    configured = (params or {}).get("clip_sample_frames")
    if configured is None or str(configured).strip() == "":
        return DEFAULT_SAMPLE_FRAMES
    count = int(configured)
    if count < 1:
        raise ValueError("clip_sample_frames must be at least 1")
    return count


def wan_clip_video_name(diffusers):
    return WAN_DIFFUSERS_CLIP_VIDEO if diffusers else WAN_NATIVE_CLIP_VIDEO


def build_clip_model_probe_cmd(model, hf_home="/hf_home"):
    """Return a container command that prints CLIP_OK only when weights are already local."""
    script = (
        "import sys\n"
        "from pathlib import Path\n"
        f"model = {str(model)!r}\n"
        "path = Path(model)\n"
        "if path.is_dir() and any(path.rglob('config.json')):\n"
        "    print('CLIP_OK')\n"
        "    raise SystemExit(0)\n"
        "try:\n"
        "    from huggingface_hub import snapshot_download\n"
        "    snapshot_download(repo_id=model, local_files_only=True)\n"
        "except Exception as exc:\n"
        "    print('CLIP_MISSING ' + str(exc))\n"
        "    raise SystemExit(1)\n"
        "print('CLIP_OK')\n"
    )
    home = shlex.quote(str(hf_home or "/hf_home"))
    return f"HF_HOME={home} HF_HUB_OFFLINE=1 python3 -c " + shlex.quote(script)


def clip_model_missing_message(model, hosts):
    where = f" on: {', '.join(hosts)}" if hosts else ""
    return (
        f"CLIP model {model!r} is not available offline{where}. "
        "Stage it under the container HF_HOME or set clip_model to a local directory. "
        "This test does not download CLIP weights. "
        "enforce_thresholds does not skip a missing CLIP model."
    )


def clipscore_output_path(output_dir):
    return str(output_dir).rstrip("/") + "/results/clipscore.json"


def build_clipscore_cmd(prompt, image_dir, output_json, model, sample_frames=None, video_name=None):
    """Return a container shell command that scores media under ``image_dir``."""
    script = CLIP_SCORE_SCRIPT.read_text(encoding="utf-8")
    encoded = base64.b64encode(script.encode("utf-8")).decode("ascii")
    launcher = (
        "import base64, pathlib, subprocess, sys\n"
        "path = pathlib.Path('/tmp/cvs_clipscore.py')\n"
        f"path.write_bytes(base64.b64decode('{encoded}'))\n"
        "raise SystemExit(subprocess.call([sys.executable, str(path), *sys.argv[1:]]))\n"
    )
    argv = [
        "HF_HUB_OFFLINE=1",
        "python3",
        "-c",
        shlex.quote(launcher),
        "--prompt",
        shlex.quote(str(prompt)),
        "--image-dir",
        shlex.quote(str(image_dir)),
        "--output",
        shlex.quote(str(output_json)),
        "--model",
        shlex.quote(str(model)),
    ]
    if sample_frames is not None:
        argv.extend(["--sample-frames", shlex.quote(str(int(sample_frames)))])
    if video_name:
        argv.extend(["--video-name", shlex.quote(str(video_name))])
    return " ".join(argv)


def parse_clipscore_text(text):
    """Return the last CLIPScore payload printed by the container scorer."""
    payload = None
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped.startswith(CLIP_SCORE_MARKER):
            continue
        payload = json.loads(stripped[len(CLIP_SCORE_MARKER) :])
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError("CLIP score payload must be an object")
    if "clip_score" not in payload and "mean_frame_clip_score" not in payload:
        raise ValueError("CLIP score payload missing clip_score or mean_frame_clip_score")
    return payload


def _threshold_entry(expected_results, gpu_type):
    if gpu_type in expected_results:
        return expected_results[gpu_type], gpu_type
    if "auto" in expected_results:
        return expected_results["auto"], "auto"
    return None, None


def _validate_min_score(result, expected_results, gpu_type, score_key, threshold_key, label, unit):
    selected, used = _threshold_entry(expected_results or {}, gpu_type)
    if selected is None:
        return False, f"No threshold found for GPU type '{gpu_type}' and no 'auto' fallback"

    minimum = selected.get(threshold_key)
    if minimum is None:
        return False, f"Threshold missing '{threshold_key}' for GPU type '{used}'"

    score = float(result[score_key])
    model = result.get("model") or DEFAULT_CLIP_MODEL
    count = int(result.get("sample_count" if unit == "frame(s)" else "image_count") or 0)
    if score >= float(minimum):
        message = (
            f"PASS: {label} {score:.2f} >= min {float(minimum):.2f} ({count} {unit}, model {model}, GPU: {gpu_type})"
        )
        return True, message

    message = f"FAIL: {label} {score:.2f} < min {float(minimum):.2f} ({count} {unit}, model {model}, GPU: {gpu_type})"
    return False, message


def validate_clip_score(result, expected_results, gpu_type="auto"):
    """
    Compare a parsed image CLIPScore payload with the selected GPU threshold.

    Returns ``(passed, message)``.
    """
    return _validate_min_score(
        result,
        expected_results,
        gpu_type,
        "clip_score",
        "min_clip_score",
        "CLIPScore",
        "image(s)",
    )


def validate_mean_frame_clip_score(result, expected_results, gpu_type="auto"):
    """
    Compare a sampled-frame WAN CLIPScore payload with the selected GPU threshold.

    Returns ``(passed, message)``.
    """
    return _validate_min_score(
        result,
        expected_results,
        gpu_type,
        "mean_frame_clip_score",
        "min_mean_frame_clip_score",
        "mean frame CLIPScore",
        "frame(s)",
    )
