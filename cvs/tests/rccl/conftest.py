'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

RCCL suite fixtures for Run Deck session publishing.
'''

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def cvs_results_dict():
    """Populated by RCCL tests after graph conversion at module teardown."""
    return {}


@pytest.fixture(scope="module")
def variant_config(request):
    """Minimal variant placeholder for RCCL Run Deck run card."""
    config_file = request.config.getoption("--config_file")
    return type(
        "RcclVariant",
        (),
        {
            "enforce_thresholds": False,
            "model": type("Model", (), {"id": "RCCL"})(),
            "gpu_arch": "—",
            "framework": "RCCL",
            "params": type("Params", (), {"tensor_parallelism": "—"})(),
            "config_file": config_file,
        },
    )()


@pytest.fixture(scope="module")
def golden_results():
    """Optional golden reference for RCCL regression matrix compare."""
    return {}
