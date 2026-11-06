"""
FID versus sampling-step curve for FLUX and WAN Run Deck charts.

A point is one Frechet Inception Distance at one step count (NFE). One image
is not a distribution, so a curve needs at least two step counts and more than
one sample at each count. The latency run does not produce that curve.

Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
"""

import base64
import shlex
from pathlib import Path

FID_CURVE_MARKER = "CVS_FID_CURVE "
FID_SCORE_TIMEOUT_S = 1800
FID_EXIT_RESERVE_S = 180
FID_DEFAULT_SAMPLES = 2
FID_SWEEP_MOUNT_PATH = "/benchmark/fid_sweep.py"
FID_SCORE_SCRIPT = Path(__file__).resolve().parent / "scripts" / "fid_score.py"
FID_CURVE_REL = "results/fid_curve.json"
FID_MANIFEST_REL = "results/fid/manifest.json"


def fid_curve_container_path(output_dir):
    return str(output_dir).rstrip("/") + "/" + FID_CURVE_REL


def fid_samples_container_root(output_dir):
    return str(output_dir).rstrip("/") + "/results/fid"


def fid_manifest_container_path(output_dir):
    return str(output_dir).rstrip("/") + "/" + FID_MANIFEST_REL


def fid_sweep_host_path():
    return str(Path(__file__).resolve().parent / "scripts" / "fid_sweep.py")


def fid_budget_seconds(timeout_s):
    """Seconds of the command timeout left for the run after the exit reserve."""
    return max(0, int(timeout_s) - FID_EXIT_RESERVE_S)


def fid_deadline_shell(budget_s):
    """Epoch seconds. The node shell evaluates this when the timed command starts."""
    return f"$(($(date +%s) + {int(budget_s)}))"


def fid_cli_flags(params, _budget_s):
    """Extra launcher flags, or "" when the curve is off.

    ``$fid_deadline`` is the epoch from ``fid_deadline_shell``. The launch command
    sets it before container start, pip, and torchrun, so those are inside the budget.
    """
    steps = parse_fid_nfe_steps(params)
    if not steps:
        return ""
    samples = FID_DEFAULT_SAMPLES
    if (params or {}).get("fid_samples") not in (None, ""):
        samples = int(params.get("fid_samples"))
    if samples < 2:
        raise ValueError("fid_samples must be at least 2")
    steps_arg = ",".join(str(step) for step in steps)
    return f"--fid_steps {steps_arg} --fid_samples {samples} --fid_deadline $fid_deadline"


def fid_sweep_mount_when_enabled(params):
    """``/benchmark/fid_sweep.py`` when the curve is on, else empty."""
    if not parse_fid_nfe_steps(params):
        return ""
    return FID_SWEEP_MOUNT_PATH


def attach_fid_sweep_mount(inference_dict):
    """Bind-mount fid_sweep.py beside the other /benchmark scripts."""
    host_path = fid_sweep_host_path()
    container_config = inference_dict.setdefault("container_config", {})
    volume_dict = dict(container_config.get("volume_dict") or {})
    mounted = {str(target) for target in volume_dict.values()}
    if FID_SWEEP_MOUNT_PATH not in mounted:
        volume_dict[host_path] = FID_SWEEP_MOUNT_PATH
        container_config["volume_dict"] = volume_dict
    return host_path


def parse_fid_nfe_steps(params):
    """Return configured step counts, or an empty list when the curve is off."""
    raw = (params or {}).get("fid_nfe_steps")
    if raw is None or raw == "" or raw == []:
        return []
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        values = [raw]
    elif isinstance(raw, str):
        values = [part.strip() for part in raw.split(",") if part.strip()]
    elif isinstance(raw, (list, tuple)):
        values = list(raw)
    else:
        raise ValueError("fid_nfe_steps must be a list of sampling step counts")
    steps = []
    for value in values:
        step = int(value)
        if step < 1:
            raise ValueError("fid_nfe_steps values must be at least 1")
        if step not in steps:
            steps.append(step)
    if len(steps) < 2:
        raise ValueError("fid_nfe_steps needs at least two step counts to draw a curve")
    return steps


def _point_from_entry(entry, default_sampler):
    if isinstance(entry, dict):
        nfe = entry.get("nfe", entry.get("steps"))
        fid = entry.get("fid")
        sampler = str(entry.get("sampler") or default_sampler or "")
    elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
        nfe, fid = entry[0], entry[1]
        sampler = default_sampler or ""
    else:
        return None
    try:
        return str(sampler), float(nfe), float(fid)
    except (TypeError, ValueError):
        return None


def iter_fid_points(raw):
    """Yield ``(sampler, nfe, fid)`` from a curve payload."""
    default_sampler = ""
    entries = raw
    if isinstance(raw, dict):
        default_sampler = str(raw.get("sampler") or "")
        entries = raw.get("points") or []
    for entry in entries or []:
        point = _point_from_entry(entry, default_sampler)
        if point is not None:
            yield point


def fid_viewer_curves(raw):
    """``extra_curves`` entries the interactive viewer plots, keyed by sampler.

    Each value carries its points, ``default_on``, and ``x_title``. A plain list
    stays off, with a Step axis, which is how the training decks ship their extras.
    """
    grouped = {}
    for sampler, nfe, fid in iter_fid_points(raw):
        label = f"FID · {sampler}" if sampler else "FID"
        grouped.setdefault(label, {})[nfe] = fid
    curves = {}
    for label, points in grouped.items():
        series = [[nfe, points[nfe]] for nfe in sorted(points)]
        if len(series) >= 2:
            curves[label] = {"points": series, "default_on": True, "x_title": "NFE"}
    return curves


def build_fid_score_cmd(reference_dir, generated_root, inception_path, output_json, sampler=""):
    """Return a container command that writes ``results/fid_curve.json``."""
    script = FID_SCORE_SCRIPT.read_text(encoding="utf-8")
    encoded = base64.b64encode(script.encode("utf-8")).decode("ascii")
    launcher = (
        "import base64, pathlib, subprocess, sys\n"
        "path = pathlib.Path('/tmp/cvs_fid_score.py')\n"
        f"path.write_bytes(base64.b64decode('{encoded}'))\n"
        "raise SystemExit(subprocess.call([sys.executable, str(path), *sys.argv[1:]]))\n"
    )
    argv = [
        "HF_HUB_OFFLINE=1",
        "python3",
        "-c",
        shlex.quote(launcher),
        "--reference-dir",
        shlex.quote(str(reference_dir)),
        "--generated-root",
        shlex.quote(str(generated_root)),
        "--inception",
        shlex.quote(str(inception_path)),
        "--output",
        shlex.quote(str(output_json)),
    ]
    if sampler:
        argv.extend(["--sampler", shlex.quote(str(sampler))])
    return " ".join(argv)


def fid_absent_skip_reason(inception_path, error=""):
    """Skip message when the Inception checkpoint is not set or not in the container."""
    if not str(inception_path or "").strip():
        return "Skipping FID vs steps. fid_inception_path is empty. CVS does not download the Inception checkpoint."
    text = str(error or "")
    if "inception checkpoint not found" in text:
        return (
            "Skipping FID vs steps. The Inception checkpoint is not in the container. CVS does not download it. " + text
        )
    return ""


def parse_fid_curve_text(text):
    """Return the last FID curve payload printed by the container scorer."""
    import json

    payload = None
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped.startswith(FID_CURVE_MARKER):
            continue
        payload = json.loads(stripped[len(FID_CURVE_MARKER) :])
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError("FID curve payload must be an object")
    points = list(iter_fid_points(payload))
    if len(points) < 2:
        raise ValueError("FID curve needs at least two NFE points")
    return payload
