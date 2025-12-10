'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import pytest

import re
import sys
import os
import sys
import time
import json
import logging

from cvs.lib.parallel_ssh_lib import *
from cvs.lib.utils_lib import *
from cvs.lib.verify_lib import *
from cvs.lib import globals

log = globals.log


# NOTE: This module assumes the following symbols are available in scope:
# - log: a configured logger
# - Pssh: parallel SSH helper class
# - fail_test: helper that records/logs a failure (and may raise)
# - update_test_result: helper to finalize a test's pass/fail status
# - print_test_output: helper to pretty-print per-node command output
# - convert_hms_to_secs: helper to convert "HH:MM:SS" to seconds
# - globals.error_list: global list used to accumulate test errors across steps


# Importing additional cmd line args to script ..
@pytest.fixture(scope="module")
def cluster_file(pytestconfig):
    """
    Retrieve the --cluster_file CLI option value provided to pytest.

    Returns:
      str: Path to the cluster JSON file.
    """
    return pytestconfig.getoption("cluster_file")


@pytest.fixture(scope="module")
def config_file(pytestconfig):
    """
    Retrieve the --config_file CLI option value provided to pytest.

    Returns:
      str: Path to the test configuration JSON file.
    """
    return pytestconfig.getoption("config_file")


# Importing the cluster and cofig files to script to access node, switch, test config params
@pytest.fixture(scope="module")
def cluster_dict(cluster_file):
    """
    Load the full cluster configuration from JSON for use by tests.

    Returns:
    dict: Parsed cluster configuration (nodes, credentials, etc).
    """
    with open(cluster_file) as json_file:
        cluster_dict = json.load(json_file)

    # Resolve path placeholders like {user-id} in cluster config
    cluster_dict = resolve_cluster_config_placeholders(cluster_dict)

    log.info(cluster_dict)
    return cluster_dict


@pytest.fixture(scope="module")
def config_dict(config_file, cluster_dict):
    """
    Load the AGFHC test configuration subsection from the provided JSON.

    Returns:
      dict: The 'agfhc' configuration map with keys like 'path', 'package_path', durations, etc.
    """
    with open(config_file) as json_file:
        config_dict_t = json.load(json_file)
    config_dict = config_dict_t['agfhc']

    # Resolve path placeholders like {user-id}, {home-mount-dir}, etc.
    config_dict = resolve_test_config_placeholders(config_dict, cluster_dict)

    log.info(config_dict)
    return config_dict


def scan_agfc_results(out_dict):
    """
    Parse AGFHC run outputs from all nodes and fail on unexpected patterns.

    Args:
      out_dict (dict): Mapping node -> command stdout/stderr combined string.

    Behavior:
      - Requires 'return code AGFHC_SUCCESS' to appear in each node's output.
      - Fails if any of the patterns FAIL|ERROR|ABORT are present (case-insensitive).
    """

    for host in out_dict.keys():
        if not re.search('code AGFHC_SUCCESS', out_dict[host], re.I):
            fail_test(f'Test failed on node {host} - AGFHC_SUCCESS code NOT seen in test result')

        if re.search('FAIL|ERROR|ABORT', out_dict[host], re.I):
            fail_test(f'Test failed on node {host} - FAIL or ERROR or ABORT patterns seen')


# Create connection to DUTs and export for later use ..
@pytest.fixture(scope="module")
def phdl(cluster_dict):
    """
    Build a parallel SSH handle to all nodes in the cluster.

    Returns:
    Pssh: A handle to execute commands across all nodes.
    """
    nhdl_dict = {}
    print(cluster_dict)
    node_list = list(cluster_dict['node_dict'].keys())
    phdl = Pssh(log, node_list, user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'], stop_on_errors=False)
    return phdl


def test_agfhc_hbm(
    phdl,
    config_dict,
):
    """
    Run AGFHC HBM test for a duration specified in config (HH:MM:SS).

    Steps:
      - Convert configured HBM test duration to seconds for timeout.
      - Extract H/M/S to form the -t argument for agfhc.
      - Execute agfhc HBM test across nodes with a small buffer added to timeout.
      - Scan outputs for success and absence of error patterns.
      - Print node outputs and update test result aggregation.

    Assumes:
      - convert_hms_to_secs, scan_agfc_results, print_test_output, update_test_result exist in scope.
      - config_dict['path'] and config_dict['hbm_test_duration'] are valid.
    """
    globals.error_list = []
    log.info('Testcase Run HBM Test')
    path = config_dict['path']
    duration = convert_hms_to_secs(config_dict['hbm_test_duration'])
    (hours, mins, secs) = config_dict['hbm_test_duration'].split(":")
    out_dict = phdl.exec(f'sudo {path}/agfhc -t hbm:d={hours}h{mins}m{secs}s --simple-output', timeout=duration + 120)
    scan_agfc_results(out_dict)
    print_test_output(log, out_dict)
    update_test_result()


# 2 hrs
def test_agfhc_hbm1_lvl5(
    phdl,
    config_dict,
):
    """
    Run AGFHC HBM1 level 5 recipe.

    Steps:
      - Execute 'agfhc -s hbm1 -r hbm_lvl5'.
      - Validate output and update aggregated test status.

    """
    globals.error_list = []
    log.info('Testcase Run HBM1 Test - hbm_lvl5')
    path = config_dict['path']
    out_dict = phdl.exec(f'sudo {path}/agfhc -s hbm1 -r hbm_lvl5', timeout=(60 * 300))
    scan_agfc_results(out_dict)
    print_test_output(log, out_dict)
    update_test_result()


# 2 hrs
def test_agfhc_hbm2_lvl5(
    phdl,
    config_dict,
):
    """
    Run AGFHC HBM2 level 5 recipe:
      - 'agfhc -s hbm2 -r hbm_lvl5'
    Validates output and updates test status.
    """
    globals.error_list = []
    log.info('Testcase Run HBM2 Test - hbm_lvl5')
    path = config_dict['path']
    out_dict = phdl.exec(f'sudo {path}/agfhc -s hbm2 -r hbm_lvl5', timeout=(60 * 300))
    scan_agfc_results(out_dict)
    print_test_output(log, out_dict)
    update_test_result()


# 30 min
def test_agfhc_hbm3_lvl3(
    phdl,
    config_dict,
):
    """
    Run AGFHC HBM3 level 3 recipe:
      - 'agfhc -s hbm3 -r hbm_lvl3'
    Shorter test; validates output and records results.
    """
    globals.error_list = []
    log.info('Testcase Run HBM3 Test - hbm_lvl3')
    path = config_dict['path']
    out_dict = phdl.exec(f'sudo {path}/agfhc -s hbm3 -r hbm_lvl3', timeout=(60 * 100))
    scan_agfc_results(out_dict)
    print_test_output(log, out_dict)
    update_test_result()


def test_agfhc_dma_all_lvl1(
    phdl,
    config_dict,
):
    """
    Run AGFHC aggregate level 1 recipe:
      - 'agfhc -r all_lvl1'
    Validates output and updates test result.
    """
    globals.error_list = []
    log.info('Testcase Run all_lvl1')
    path = config_dict['path']
    out_dict = phdl.exec(f'sudo {path}/agfhc -r all_lvl1', timeout=(60 * 30))
    scan_agfc_results(out_dict)
    print_test_output(log, out_dict)
    update_test_result()


def test_agfhc_dma_lvl1(
    phdl,
    config_dict,
):
    """
    Run AGFHC DMA level 1 recipe:
      - 'agfhc -r dma_lvl1'
    Validates output and records results.
    """
    globals.error_list = []
    log.info('Testcase Run DMA lvl1')
    path = config_dict['path']
    out_dict = phdl.exec(f'sudo {path}/agfhc -r dma_lvl1', timeout=(60 * 30))
    scan_agfc_results(out_dict)
    print_test_output(log, out_dict)
    update_test_result()


def test_agfhc_gfx_lvl1(
    phdl,
    config_dict,
):
    """
    Run AGFHC GFX level 1 recipe:
      - 'agfhc -r gfx_lvl1'
    Validates output and updates aggregated status.
    """
    globals.error_list = []
    log.info('Testcase Run GFX lvl1')
    path = config_dict['path']
    out_dict = phdl.exec(f'sudo {path}/agfhc -r gfx_lvl1', timeout=(60 * 60))
    scan_agfc_results(out_dict)
    print_test_output(log, out_dict)
    update_test_result()


def test_agfhc_pcie_lvl1(
    phdl,
    config_dict,
):
    """
    Pytest: Run the AGFHC PCIe level-1 recipe on all nodes and validate results.

    Args:
      phdl: Parallel SSH/process handle able to execute commands across cluster nodes.
      config_dict (dict): Test configuration that must include:
        - 'path': directory where the agfhc binary is installed.

    Behavior:
      - Resets the global error accumulator (globals.error_list) for a clean test run.
      - Executes: sudo <path>/agfhc -r pcie_lvl1 with a 60-minute timeout.
      - Scans output for success (and absence of error patterns) via scan_agfc_results.
      - Prints per-node output and updates the aggregated test result.

    Assumptions:
      - scan_agfc_results, print_test_output, update_test_result are available in scope.
      - config_dict['path'] is valid on each node.
      - phdl.exec(cmd, timeout) returns a dict mapping node -> stdout/stderr string.
    """
    globals.error_list = []
    log.info('Testcase Run PCIe lvl1')
    path = config_dict['path']
    out_dict = phdl.exec(f'sudo {path}/agfhc -r pcie_lvl1', timeout=(60 * 60))
    scan_agfc_results(out_dict)
    print_test_output(log, out_dict)
    update_test_result()


def test_agfhc_pcie_lvl3(
    phdl,
    config_dict,
):
    """
    Pytest: Run the AGFHC PCIe level-3 recipe on all nodes and validate results.

    Args:
      phdl: Parallel SSH handle for remote command execution.
      config_dict (dict): Includes 'path' to the agfhc binary.

    Behavior:
      - Clears the error list for this test.
      - Runs: sudo <path>/agfhc -r pcie_lvl3 (60-minute timeout).
      - Checks for AGFHC_SUCCESS and error signatures.
      - Prints outputs per node and updates overall test status.
    """
    globals.error_list = []
    log.info('Testcase Run PCIe lvl3')
    path = config_dict['path']
    out_dict = phdl.exec(f'sudo {path}/agfhc -r pcie_lvl3', timeout=(60 * 60))
    scan_agfc_results(out_dict)
    print_test_output(log, out_dict)
    update_test_result()


def test_agfhc_xgmi_lvl1(
    phdl,
    config_dict,
):
    """
    Pytest: Run the AGFHC XGMI level-1 recipe and validate outputs.

    Args:
      phdl: Parallel SSH handle.
      config_dict (dict): Includes 'path' to agfhc.

    Behavior:
      - Clears global error list.
      - Runs: sudo <path>/agfhc -r xgmi_lvl1 (90-minute timeout).
      - Validates success markers and absence of failure patterns.
      - Prints outputs, then updates aggregated test result.
    """
    globals.error_list = []
    log.info('Testcase Run XGMI lvl1')
    path = config_dict['path']
    out_dict = phdl.exec(f'sudo {path}/agfhc -r xgmi_lvl1', timeout=(60 * 90))
    scan_agfc_results(out_dict)
    print_test_output(log, out_dict)
    update_test_result()


def test_agfhc_all_perf(
    phdl,
    config_dict,
):
    """
    Pytest: Run the AGFHC 'all_perf' performance recipe across nodes.

    Args:
      phdl: Parallel SSH handle.
      config_dict (dict): Must include 'path'.

    Behavior:
      - Resets error accumulator.
      - Runs: sudo <path>/agfhc -r all_perf (90-minute timeout).
      - Scans outputs to ensure success and no fatal patterns.
      - Prints outputs and updates the aggregated test result.
    """
    globals.error_list = []
    log.info('Testcase Run all_perf')
    path = config_dict['path']
    out_dict = phdl.exec(f'sudo {path}/agfhc -r all_perf', timeout=(60 * 90))
    scan_agfc_results(out_dict)
    print_test_output(log, out_dict)
    update_test_result()


def test_agfhc_all_lvl5(
    phdl,
    config_dict,
):
    """
    Pytest: Query/validate the AGFHC 'all_lvl5' recipe information.

    Args:
      phdl: Parallel SSH handle.
      config_dict (dict): Must include 'path'.

    Behavior:
      - Runs: sudo <path>/agfhc --recipe-info all_lvl5 (260-minute timeout).
      - Scans outputs for success/absence of error markers.
      - Prints outputs and updates the test result.
    """
    log.info('Testcase all lvl5')
    path = config_dict['path']
    out_dict = phdl.exec(f'sudo {path}/agfhc --recipe-info all_lvl5', timeout=(60 * 260))
    scan_agfc_results(out_dict)
    print_test_output(log, out_dict)
    update_test_result()


# 180m
# @pytest.mark.dependency(depends=["init"])
# def test_agfhc_rochpl(phdl, config_dict, ):
#    log.info('Testcase rochpl for 180 min')
#    path = config_dict['path']
#    out_dict = phdl.exec(f'sudo {path}/agfhc rochpl:d=180m', timeout=(60*650))
#    scan_agfc_results(out_dict)
#    print_test_output(log, out_dict)
#    update_test_result()
