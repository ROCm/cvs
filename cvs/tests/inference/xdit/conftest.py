"""
Shared pytest fixtures for xDiT inference suites.

Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
"""

import pytest


@pytest.fixture(scope="module")
def xdit_results():
    """Collect normalized diffusion results for the session Run Deck."""
    return []
