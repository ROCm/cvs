'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Shared pytest wiring for Megatron training suites (suite reports).
'''

import pytest

from cvs.lib.report.presets.megatron_8b_single import MEGATRON_LLAMA3_8B_SINGLE_REPORT_CONFIG
from cvs.lib.report.training_wiring import (
    bind_training_suite_report_session,
    configure_training_suite_report,
)


@pytest.fixture(scope="module")
def training_res_dict():
    return {}


def pytest_configure(config):
    configure_training_suite_report(config, MEGATRON_LLAMA3_8B_SINGLE_REPORT_CONFIG)


@pytest.fixture(scope="module", autouse=True)
def _suite_report_session(training_res_dict):
    bind_training_suite_report_session(training_res_dict=training_res_dict)
    yield
