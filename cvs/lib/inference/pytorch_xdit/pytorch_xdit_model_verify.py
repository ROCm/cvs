"""
Offline model-tree verification helpers for PyTorch XDit inference tests.

Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
"""

from __future__ import annotations

import shlex
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from cvs.lib.inference.pytorch_xdit.pytorch_xdit_wan_job import (
    WAN_MODEL_FORMAT_DIFFUSERS,
    resolve_wan_model_format,
)
from cvs.lib.utils_lib import wan_hf_snapshot_offline_check_commands


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
