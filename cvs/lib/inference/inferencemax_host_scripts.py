'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

# Fallback when no on-disk file exists under ``cvs.lib.dtni.vllm_benchmark_scripts`` (see
# :func:`bundled_script_body`). Prefer adding scripts there rather than growing this dict.

from typing import Optional

from cvs.lib.dtni.vllm_benchmark_scripts import bundled_scripts_dir

BUNDLED_SERVER_SCRIPTS: dict[str, str] = {}


def bundled_script_body(filename: str) -> Optional[str]:
    """Return shell source for ``filename`` (basename): package ``vllm_benchmark_scripts`` first, then dict."""
    base = filename.replace("\\", "/").rsplit("/", 1)[-1]
    p = bundled_scripts_dir() / base
    if p.is_file():
        return p.read_text(encoding="utf-8")
    return BUNDLED_SERVER_SCRIPTS.get(base)
