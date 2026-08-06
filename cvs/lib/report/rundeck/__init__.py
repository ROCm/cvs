'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

CVS Run Deck — generic session report engine.
'''

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "generate_rundeck":
        from cvs.lib.report.rundeck.generate_rundeck import generate_rundeck

        return generate_rundeck
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["generate_rundeck"]
