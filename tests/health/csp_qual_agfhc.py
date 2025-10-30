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

sys.path.insert( 0, './lib' )
from parallel_ssh_lib import *
from utils_lib import *
from verify_lib import *

import globals

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
    log.info(cluster_dict)
    return cluster_dict



@pytest.fixture(scope="module")
def config_dict(config_file):
    """
    Load the AGFHC test configuration subsection from the provided JSON.

    Returns:
      dict: The 'agfhc' configuration map with keys like 'path', 'package_path', durations, etc.
    """
    with open(config_file) as json_file:
        config_dict_t = json.load(json_file)
    config_dict = config_dict_t['agfhc']
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
        if not re.search( 'code AGFHC_SUCCESS', out_dict[host], re.I ):
            fail_test(f'Test failed on node {host} - AGFHC_SUCCESS code NOT seen in test result')

        if re.search( 'FAIL|ERROR|ABORT', out_dict[host], re.I ):
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
    phdl = Pssh( log, node_list, user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'] )
    return phdl



# Get the version of AGFHC
@pytest.mark.dependency()
def test_version_check( phdl, config_dict ):
    globals.error_list = []
    path = config_dict['path']
    out_dict = phdl.exec(f'sudo {path}/agfhc -v')
    for node in out_dict.keys():
        if not re.search( 'agfhc version:', out_dict[node], re.I ):
            fail_test(f'Failed to print the AGFHC version on node {node}, installation not proper')
    update_test_result()



# 2 hrs test
@pytest.mark.dependency(depends=["test_version_check"])
def test_all_lvl5(phdl, config_dict, ):
    globals.error_list = []
    log.info('Testcase Run all_lvl5 Test')
    path = config_dict['path']
    out_dict = phdl.exec(f'sudo {path}/agfhc -r all_lvl5', timeout=(2*60*60)+30)
    scan_agfc_results(out_dict)
    print_test_output(log, out_dict)
    update_test_result()


# 8 hrs
@pytest.mark.dependency(depends=["test_version_check"])
def test_agfhc_hbm_lvl5(phdl, config_dict, ):
    """
    Run AGFHC HBM1 level 5 recipe for 4 iterations

    Steps:
      - Validate output and update aggregated test status.

    """
    globals.error_list = []
    log.info('Testcase Run HBM Test - hbm_lvl5')
    path = config_dict['path']
    out_dict = phdl.exec(f'sudo {path}/agfhc -r hbm_lvl5:i=4', timeout=(60*60*8)+60)
    scan_agfc_results(out_dict)
    print_test_output(log, out_dict)
    update_test_result()


# 4 hrs
@pytest.mark.dependency(depends=["test_version_check"])
def test_agfhc_minihpl(phdl, config_dict, ):

    """
    Run AGFHC miniHPL:
    Validates output and updates test status.
    """
    globals.error_list = []
    log.info('Testcase Run AGFHC miniHPL')
    path = config_dict['path']
    out_dict = phdl.exec(f'sudo {path}/agfhc -t minihpl:d=4h', timeout=(60*60*4)+60)
    scan_agfc_results(out_dict)
    print_test_output(log, out_dict)
    update_test_result()



# 5 min
@pytest.mark.dependency(depends=["test_version_check"])
def test_agfhc_xgmi_lvl1(phdl, config_dict, ):
    """
    Run AGFHC XGMI lvl1 recipe:
    Shorter test; validates output and records results.
    """
    globals.error_list = []
    log.info('Testcase Run XGMI lvl1')
    path = config_dict['path']
    out_dict = phdl.exec(f'sudo {path}/agfhc -r xgmi_lvl1', timeout=(60*5)+30)
    scan_agfc_results(out_dict)
    print_test_output(log, out_dict)
    update_test_result()



# 10 min
# adding some additional time for buffer
@pytest.mark.dependency(depends=["test_version_check"])
def test_agfhc_pcie_lvl2(phdl, config_dict, ):
    """
    Run AGFHC pcie lvl2:
    Validates output and updates test result.
    """
    globals.error_list = []
    log.info('Testcase Run PCIe lvl2')
    path = config_dict['path']
    out_dict = phdl.exec(f'sudo {path}/agfhc -r pcie_lvl2', timeout=(60*30)+30)
    scan_agfc_results(out_dict)
    print_test_output(log, out_dict)
    update_test_result()



@pytest.mark.dependency(depends=["test_version_check"])
def test_agfhc_all_perf(phdl, config_dict, ):
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
    out_dict = phdl.exec(f'sudo {path}/agfhc -r all_perf', timeout=(60*90))
    scan_agfc_results(out_dict)
    print_test_output(log, out_dict)
    update_test_result()


