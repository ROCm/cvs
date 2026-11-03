"""
Offline model-tree verification helpers for PyTorch XDit inference tests.

Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
"""

from __future__ import annotations

import shlex
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from cvs.lib.inference.xdit.xdit_wan_job import (
    WAN_MODEL_FORMAT_DIFFUSERS,
    resolve_wan_model_format,
)
from cvs.lib.utils_lib import wan_hf_snapshot_offline_check_commands

HF_SNAPSHOT_MARKER = "HF_SNAPSHOT="
HF_DOWNLOAD_ERROR_MARKER = "HF_DOWNLOAD_ERROR="
HF_SNAPSHOT_DOWNLOAD_TIMEOUT_S = 14400
CONTAINER_HF_TOKEN_PATH = "/run/secrets/hf_token"
_HF_DOWNLOAD_SCRIPT = """
import os
import sys

try:
    from huggingface_hub import snapshot_download
except ImportError as exc:
    print("HF_DOWNLOAD_ERROR=huggingface_hub is required in the container: %s" % exc)
    raise SystemExit(1)

token = None
token_file = os.environ.get("XDIT_HF_TOKEN_FILE") or ""
if token_file:
    try:
        with open(token_file, encoding="utf-8") as handle:
            token = handle.read().strip() or None
    except OSError as exc:
        print("HF_DOWNLOAD_ERROR=unable to read token file: %s" % exc)
        raise SystemExit(1)

kwargs = {"repo_id": os.environ["XDIT_HF_REPO"], "token": token}
revision = os.environ.get("XDIT_HF_REVISION") or ""
if revision:
    kwargs["revision"] = revision
try:
    path = snapshot_download(**kwargs)
except Exception as exc:
    print("HF_DOWNLOAD_ERROR=%s" % exc)
    raise SystemExit(1)
print("HF_SNAPSHOT=%s" % path)
"""


def is_local_model_path(model):
    return isinstance(model, str) and model.startswith("/")


def _exec_text(value):
    if isinstance(value, dict):
        return str(value.get("output") or value.get("stdout") or "")
    return str(value or "")


def _secret_str(value):
    if value is None:
        return ""
    return str(value)


def build_hf_snapshot_download_cmd(repo, revision="", hf_home="/hf_home", token="", token_file=""):
    parts = [
        f"HF_HOME={shlex.quote(hf_home)}",
        f"XDIT_HF_REPO={shlex.quote(repo)}",
    ]
    if revision:
        parts.append(f"XDIT_HF_REVISION={shlex.quote(revision)}")
    token_file = str(token_file or "").strip()
    if token_file:
        parts.append(f"XDIT_HF_TOKEN_FILE={shlex.quote(token_file)}")
    parts.append("python -c")
    parts.append(shlex.quote(_HF_DOWNLOAD_SCRIPT.strip()))
    return " ".join(parts)


def parse_hf_snapshot_output(output):
    text = _exec_text(output)
    snapshot = None
    error = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(HF_SNAPSHOT_MARKER):
            snapshot = stripped[len(HF_SNAPSHOT_MARKER) :].strip()
        elif stripped.startswith(HF_DOWNLOAD_ERROR_MARKER):
            error = stripped[len(HF_DOWNLOAD_ERROR_MARKER) :].strip()
    return snapshot, error


def container_snapshot_to_host(container_path, inference):
    container_path = str(container_path or "").rstrip("/")
    hf_home = str(inference.get("hf_home") or "").rstrip("/")
    hf_home_container = str(inference.get("hf_home_container") or "/hf_home").rstrip("/")
    if not container_path or not hf_home:
        return None
    if container_path == hf_home_container or container_path.startswith(hf_home_container + "/"):
        return hf_home + container_path[len(hf_home_container) :]
    return None


def download_hf_snapshot(orch, inference, token="", timeout=HF_SNAPSHOT_DOWNLOAD_TIMEOUT_S):
    repo = str(inference.get("model_repo") or "")
    revision = str(inference.get("model_rev") or "")
    hf_home = str(inference.get("hf_home_container") or "/hf_home")
    token_file = str(inference.get("hf_token_file_container") or "").strip()
    if not token_file and str(inference.get("hf_token_file") or "").strip():
        token_file = CONTAINER_HF_TOKEN_PATH
    cmd = build_hf_snapshot_download_cmd(repo, revision=revision, hf_home=hf_home, token_file=token_file)
    results = orch.exec(cmd, timeout=timeout)
    snapshots = {}
    errors = []
    for host, output in (results or {}).items():
        snapshot, error = parse_hf_snapshot_output(output)
        if snapshot:
            snapshots[host] = snapshot
            continue
        detail = error or _exec_text(output).strip() or "no snapshot path printed"
        errors.append(f"{host}: Hugging Face download of {repo!r} failed: {detail}")
    return snapshots, errors


def build_diffusers_local_model_required_checks(host_model_path: str) -> Dict[str, str]:
    """Shell checks for a complete diffusers model tree (FLUX, WAN Diffusers)."""
    base = host_model_path.rstrip("/")
    q = shlex.quote

    def ok_file(rel: str) -> str:
        return f"test -f {q(base + '/' + rel)} && echo OK || echo MISSING"

    def ok_weights(subdir: str) -> str:
        prefix = q(f"{base}/{subdir}")
        return (
            f"test -f {prefix}/diffusion_pytorch_model.safetensors "
            f"-o -f {prefix}/diffusion_pytorch_model.safetensors.index.json "
            f"-o -f {prefix}/pytorch_model.bin "
            f"-o -f {prefix}/pytorch_model.bin.index.json "
            f"&& echo OK || echo MISSING"
        )

    return {
        "model_index.json": ok_file("model_index.json"),
        "transformer/config.json": ok_file("transformer/config.json"),
        "transformer weights": ok_weights("transformer"),
        "vae/config.json": ok_file("vae/config.json"),
        "vae weights": ok_weights("vae"),
    }


def first_required_check_failure(
    s_phdl,
    required_checks: Mapping[str, str],
) -> Optional[Tuple[str, List[str]]]:
    """Run labeled shell checks on all nodes; return the first failing label and nodes."""
    for label, cmd in required_checks.items():
        res = s_phdl.exec(cmd, print_console=False)
        bad = [node for node, out in (res or {}).items() if "OK" not in (out or "")]
        if bad:
            return label, bad
    return None


def incomplete_local_model_error(
    label: str,
    bad_nodes: Sequence[str],
    host_model_path: str,
    *,
    layout_description: str,
) -> str:
    return (
        f"Local {layout_description} model directory appears incomplete. "
        f"Missing/invalid '{label}' on {len(bad_nodes)} node(s): {', '.join(bad_nodes)}. "
        f"Model path: {host_model_path}. "
        "Ensure the tree contains full weights (not just configs or LFS pointer stubs)."
    )


def verify_required_checks_on_nodes(
    s_phdl,
    host_model_path: str,
    required_checks: Mapping[str, str],
    *,
    layout_description: str,
) -> Optional[str]:
    """Return an error message when any required check fails, else None."""
    failure = first_required_check_failure(s_phdl, required_checks)
    if failure is None:
        return None
    label, bad_nodes = failure
    return incomplete_local_model_error(label, bad_nodes, host_model_path, layout_description=layout_description)


def resolve_wan_local_model_required_checks(
    host_model_path: str,
    *,
    model_format: Optional[str] = None,
    model_repo: str = "",
) -> Dict[str, str]:
    """Return file-level checks for a WAN local model path (native or Diffusers)."""
    fmt = model_format or resolve_wan_model_format(None, model_repo, host_model_path)
    if fmt == WAN_MODEL_FORMAT_DIFFUSERS:
        return build_diffusers_local_model_required_checks(host_model_path)
    return wan_hf_snapshot_offline_check_commands(host_model_path)


def wan_native_hf_snapshot_required_checks(
    snapshot_dir_host: str,
    model_repo: str,
) -> Optional[Dict[str, str]]:
    """Return native Wan2.2 HF snapshot checks, or None for Diffusers repo ids."""
    if "diffusers" in str(model_repo).lower():
        return None
    return wan_hf_snapshot_offline_check_commands(snapshot_dir_host)
