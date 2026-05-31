"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

from __future__ import annotations

from typing import ClassVar

from cvs.lib.config.base import BaseTestConfig


class TrainingTestConfig(BaseTestConfig):
    """Common base for training-kind configs (megatron, jax, ...).

    Provided in the spine so the discriminated-union machinery and markers are
    exercised by the contract even though no real training adapter ships in v1.
    Sweep axes such as ``parallelism_combos`` belong here and are rejected for
    inference configs.
    """

    WORKLOAD_KIND: ClassVar[str] = "training"
