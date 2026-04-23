"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

from __future__ import annotations


class OrchestratorConfigError(ValueError):
    """Raised by the loader and the per-axis factories when a cluster.json or
    runtime.config block is malformed, missing required fields, or asks for an
    unimplemented backend.

    Lives in its own module so the runtime / transport / launcher factories
    can raise it without depending on cvs/core/config.py (which doesn't exist
    until Phase C).
    """

    def __init__(self, message: str, problems: list[str] | None = None):
        if problems:
            joined = "\n  - ".join(problems)
            super().__init__(
                f"{message} ({len(problems)} problems):\n  - {joined}"
            )
        else:
            super().__init__(message)
        self.problems: list[str] = list(problems or [])
