"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ExecScope(Enum):
    """Which set of cluster hosts a command targets."""

    ALL = "all"
    HEAD = "head"
    SUBSET = "subset"  # caller supplies an explicit hosts list


class ExecTarget(Enum):
    """Whether a command goes through the runtime wrap_cmd or runs on the raw host shell."""

    HOST = "host"  # bypass runtime.wrap_cmd; run on the bare host shell
    RUNTIME = "runtime"  # default; goes through runtime.wrap_cmd


@dataclass(frozen=True)
class ExecResult:
    """One host's result from an exec call. Returned per-host inside dict[str, ExecResult]."""

    host: str
    output: str  # combined stdout+stderr
    exit_code: int  # -1 when the command could not execute (timeout, unreachable, etc.)
