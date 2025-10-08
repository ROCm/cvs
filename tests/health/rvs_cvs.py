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

import globals

log = globals.log


# Importing additional cmd line args to script ..
@pytest.fixture(scope="module")
def cluster_file(pytestconfig):
    return pytestconfig.getoption("cluster_file")


@pytest.fixture(scope="module")
def config_file(pytestconfig):
    return pytestconfig.getoption("config_file")


# Importing the cluster and cofig files to script to access node, switch, test config params
@pytest.fixture(scope="module")
def cluster_dict(cluster_file):
    with open(cluster_file) as json_file:
        cluster_dict = json.load(json_file)
    log.info(cluster_dict)
    return cluster_dict


@pytest.fixture(scope="module")
def config_dict(config_file):
    with open(config_file) as json_file:
        config_dict_t = json.load(json_file)
    config_dict = config_dict_t['rvs']
    log.info(config_dict)
    return config_dict


def determine_rvs_config_path(phdl, config_dict, config_file):
    """
    Determine the correct configuration file path for RVS tests.
    First checks for MI300X-specific config, falls back to default if not found.

    Args:
      phdl: Parallel SSH handle
      config_dict: RVS configuration dictionary
      config_file: Name of the configuration file to look for

    Returns:
      str: Full path to the configuration file to use
    """
    mi300x_path = f"{config_dict['config_path_mi300x']}/{config_file}"
    default_path = f"{config_dict['config_path_default']}/{config_file}"

    # Check for MI300X specific config first
    out_dict = phdl.exec(f'ls -l {mi300x_path}', timeout=30)
    for node in out_dict.keys():
        if not re.search('No such file', out_dict[node], re.I):
            log.info(f'Using MI300X-specific config: {mi300x_path}')
            return mi300x_path

    # Fall back to default config
    out_dict = phdl.exec(f'ls -l {default_path}', timeout=30)
    for node in out_dict.keys():
        if not re.search('No such file', out_dict[node], re.I):
            log.info(f'Using default config: {default_path}')
            return default_path

    # If neither exists, still return the default path (test will fail appropriately)
    log.warning(f'Configuration file {config_file} not found in either location, using default path')
    return default_path


def parse_rvs_gst_results(out_dict, exp_dict):
    """
    Parse RVS GST (GPU Stress Test) results and validate against expected values.

    Args:
      out_dict: Dictionary of node -> command output
      exp_dict: Dictionary of expected results
    """
    for node in out_dict.keys():
        # Check for "met: FALSE" which indicates target GFLOPS not achieved
        if re.search(r'met:\s*FALSE', out_dict[node], re.I):
            fail_test(f'RVS GST target GFLOPS not met on node {node}')

        # Look for GFLOPS performance results
        gflops_matches = re.findall(r'GFLOPS\s*([0-9\.]+)', out_dict[node], re.I)
        if gflops_matches:
            for gflops_value in gflops_matches:
                if float(gflops_value) < float(exp_dict.get('min_gflops', '0')):
                    fail_test(f"RVS GST GFLOPS {gflops_value} is below expected minimum {exp_dict['min_gflops']} on node {node}")

        # Check for overall test result
        if re.search(r'PASS', out_dict[node], re.I):
            log.info(f'RVS GST test passed on node {node}')
        elif re.search(r'FAIL', out_dict[node], re.I):
            fail_test(f'RVS GST test failed on node {node}')


def parse_rvs_iet_results(out_dict, exp_dict):
    """
    Parse RVS IET (Input EDPp Test) results and validate against expected values.

    Args:
      out_dict: Dictionary of node -> command output
      exp_dict: Dictionary of expected results
    """
    for node in out_dict.keys():
        # Look for power violations
        power_violation_matches = re.findall(r'power.*violation.*:?\s*([0-9]+)', out_dict[node], re.I)
        if power_violation_matches:
            for violation_count in power_violation_matches:
                if int(violation_count) > int(exp_dict.get('max_power_violation', '0')):
                    fail_test(f"RVS IET power violations {violation_count} exceeds maximum {exp_dict['max_power_violation']} on node {node}")

        # Check for overall test result
        if re.search(r'PASS', out_dict[node], re.I):
            log.info(f'RVS IET test passed on node {node}')
        elif re.search(r'FAIL', out_dict[node], re.I):
            fail_test(f'RVS IET test failed on node {node}')


def parse_rvs_pebb_results(out_dict, exp_dict):
    """
    Parse RVS PEBB (PCI Express Bandwidth Benchmark) results and validate against expected values.

    Args:
      out_dict: Dictionary of node -> command output  
      exp_dict: Dictionary of expected results
    """
    for node in out_dict.keys():
        # Look for bandwidth measurements (in GB/s)
        bw_matches = re.findall(r'bandwidth.*:?\s*([0-9\.]+)\s*GB/s', out_dict[node], re.I)
        if bw_matches:
            for bw_value in bw_matches:
                if float(bw_value) < float(exp_dict.get('min_bandwidth_gbps', '0')):
                    fail_test(f"RVS PEBB bandwidth {bw_value} GB/s is below expected minimum {exp_dict['min_bandwidth_gbps']} GB/s on node {node}")

        # Check for overall test result
        if re.search(r'PASS', out_dict[node], re.I):
            log.info(f'RVS PEBB test passed on node {node}')
        elif re.search(r'FAIL', out_dict[node], re.I):
            fail_test(f'RVS PEBB test failed on node {node}')


@pytest.fixture(scope="module")
def phdl(cluster_dict):
    print(cluster_dict)
    node_list = list(cluster_dict['node_dict'].keys())
    phdl = Pssh( log, node_list, user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'] )
    return phdl


def test_rvs_gpu_enumeration(phdl, config_dict):
    """
    Run RVS GPU enumeration test to detect and validate GPU presence.

    This is a basic connectivity and detection test.
    """
    globals.error_list = []
    log.info('Testcase Run RVS GPU Enumeration Test')

    rvs_path = config_dict['path']

    # Run GPU enumeration (using gpup module)
    out_dict = phdl.exec(f'{rvs_path}/rvs -g', timeout=60)
    print_test_output(log, out_dict)
    scan_test_results(out_dict)

    # Validate that GPUs are detected
    for node in out_dict.keys():
        if not re.search(r'GPU|device', out_dict[node], re.I):
            fail_test(f'No GPUs detected in RVS enumeration on node {node}')

    update_test_result()


def test_rvs_memory_test(phdl, config_dict):
    """
    Run RVS memory test using available configuration.

    This test validates GPU memory functionality.
    """
    globals.error_list = []
    log.info('Testcase Run RVS Memory Test')

    rvs_path = config_dict['path']

    # Try to find memory test config, fall back to simple memory test
    mem_config_path = determine_rvs_config_path(phdl, config_dict, 'mem.conf')

    # Check if memory config exists
    out_dict = phdl.exec(f'ls -l {mem_config_path}', timeout=30)
    config_exists = False
    for node in out_dict.keys():
        if not re.search('No such file', out_dict[node], re.I):
            config_exists = True
            break

    if config_exists:
        # Run with configuration file
        out_dict = phdl.exec(f'sudo {rvs_path}/rvs -c {mem_config_path}', timeout=1800)
        print_test_output(log, out_dict)
        scan_test_results(out_dict)
    else:
        #fail the test if no config file found
        fail_test(f'No memory test configuration file found at {mem_config_path}')

    # Basic validation - look for PASS/FAIL results
    for node in out_dict.keys():
        if re.search(r'FAIL|ERROR:', out_dict[node], re.I):
            fail_test(f'RVS memory test failed on node {node}')

    update_test_result()


def test_rvs_gst_single(phdl, config_dict):
    """
    Run RVS GST (GPU Stress Test) - Single GPU validation test.

    This test runs the GPU stress test configuration to validate GPU functionality
    and performance under load.
    """
    globals.error_list = []
    log.info('Testcase Run RVS GST Single GPU Test')

    rvs_path = config_dict['path']
    config_file = 'gst_single.conf'
    config_path = determine_rvs_config_path(phdl, config_dict, config_file)

    # Get test configuration
    test_config = next((test for test in config_dict['tests'] if test['name'] == 'gst_single'), {})
    timeout = test_config.get('timeout', 1800)

    # Run RVS GST test
    out_dict = phdl.exec(f'sudo {rvs_path}/rvs -c {config_path}', timeout=timeout)
    print_test_output(log, out_dict)
    scan_test_results(out_dict)

    # Parse and validate results
    parse_rvs_gst_results(out_dict, config_dict['results'].get('gst_single', {}))
    update_test_result()

@pytest.mark.skip(reason="Test implementation and output parsing pending")
def test_rvs_iet_single(phdl, config_dict):
    """
    Run RVS IET (Input EDPp Test) - Single GPU validation test.
    
    This test validates power consumption and thermal behavior under load.
    """
    globals.error_list = []
    log.info('Testcase Run RVS IET Single GPU Test')
    
    rvs_path = config_dict['path']
    config_file = 'iet_single.conf'
    config_path = determine_rvs_config_path(phdl, config_dict, config_file)
    
    # Get test configuration
    test_config = next((test for test in config_dict['tests'] if test['name'] == 'iet_single'), {})
    timeout = test_config.get('timeout', 180)
    
    # Run RVS IET test
    out_dict = phdl.exec(f'sudo {rvs_path}/rvs.py -c {config_path}', timeout=timeout)
    print_test_output(log, out_dict)
    scan_test_results(out_dict)
    
    # Parse and validate results
    parse_rvs_iet_results(out_dict, config_dict['results'].get('iet_single', {}))
    update_test_result()


@pytest.mark.skip(reason="Test implementation and output parsing pending")
def test_rvs_pebb_single(phdl, config_dict):
    """
    Run RVS PEBB (PCI Express Bandwidth Benchmark) - Single GPU test.
    
    This test measures and validates PCI Express bandwidth performance.
    """
    globals.error_list = []
    log.info('Testcase Run RVS PEBB Single GPU Test')
    
    rvs_path = config_dict['path']  
    config_file = 'pebb_single.conf'
    config_path = determine_rvs_config_path(phdl, config_dict, config_file)
    
    # Get test configuration
    test_config = next((test for test in config_dict['tests'] if test['name'] == 'pebb_single'), {})
    timeout = test_config.get('timeout', 240)
    
    # Run RVS PEBB test
    out_dict = phdl.exec(f'sudo {rvs_path}/rvs.py -c {config_path}', timeout=timeout)
    print_test_output(log, out_dict)
    scan_test_results(out_dict)
    
    # Parse and validate results
    parse_rvs_pebb_results(out_dict, config_dict['results'].get('pebb_single', {}))
    update_test_result()

