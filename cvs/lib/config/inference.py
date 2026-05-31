"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

from __future__ import annotations

from typing import ClassVar

from cvs.lib.config.base import BaseTestConfig


class InferenceTestConfig(BaseTestConfig):
    """Common base for inference-kind configs (vLLM, sglang, inferencemax, ...).

    Sweep axes such as ``concurrency`` are valid here and rejected for training
    configs (enforced by each framework's typed ``SweepParams``).
    """

    WORKLOAD_KIND: ClassVar[str] = "inference"
