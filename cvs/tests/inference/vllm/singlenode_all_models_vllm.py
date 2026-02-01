'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.

'''

import pytest

from cvs.lib import globals
from cvs.lib.test_utils import CVSTestSuiteRunner

log = globals.log


def test_run_all_model_suites(pytestconfig):
    """
    Meta-test that discovers and runs all individual model test suites.

    This test:
    1. Finds all singlenode_*_vllm.py files in the same directory
    2. Runs pytest.main() on each discovered test module
    3. Collects and reports results

    Args:
        pytestconfig: Built-in pytest fixture with config/options
    """
    # Initialize test suite runner (auto-discovers test modules matching pattern)
    runner = CVSTestSuiteRunner(log, pytestconfig, test_pattern='singlenode_*_vllm.py')

    if not runner.test_modules:
        pytest.skip("No model test modules found matching pattern 'singlenode_*_vllm.py'")

    log.info(f"Discovered {len(runner.test_modules)} model test suites:")
    for suite_name in runner.test_modules:
        log.info(f"  - {suite_name}")

    # Verify required arguments are present
    cluster_file = pytestconfig.getoption("cluster_file")
    config_file = pytestconfig.getoption("config_file")

    if not cluster_file or not config_file:
        pytest.fail("Missing required options: --cluster_file and --config_file")

    # Run all test suites
    runner.run()

    # Generate consolidated report
    runner.generate_summary_report()
