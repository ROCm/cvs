'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent
publication and does not imply publication or any waiver of confidentiality.
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


@pytest.fixture(scope="module")
def phdl(cluster_dict):
    print(cluster_dict)
    node_list = list(cluster_dict['node_dict'].keys())
    phdl = Pssh( log, node_list, user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'] )
    return phdl

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


def parse_rvs_test_results(test_config, out_dict):
    """
    Generic parser for RVS test results that validates against expected patterns.

    Args:
      test_config: Test configuration dictionary containing name and fail_regex_pattern
      out_dict: Dictionary of node -> command output
    """
    test_name = test_config.get('name', 'unknown')
    fail_pattern = test_config.get('fail_regex_pattern', r'\[ERROR\s*\]')

    for node in out_dict.keys():
        # Check for failure pattern
        if re.search(fail_pattern, out_dict[node], re.I):
            fail_test(f'RVS {test_name} test failed on node {node}')
        else:
            log.info(f'RVS {test_name} test passed on node {node}')

def execute_rvs_test(phdl, config_dict, test_name):
    """
    Generic function to execute any RVS test.

    Args:
      phdl: Parallel SSH handle
      config_dict: RVS configuration dictionary
      test_name: Name of the test to execute
    """
    globals.error_list = []

    # Get test configuration
    test_config = next((test for test in config_dict['tests'] if test['name'] == test_name), None)

    if not test_config:
        fail_test(f'Test configuration for {test_name} not found')
        update_test_result()
        return

    log.info(f'Testcase Run RVS {test_config.get("description", test_name)}')

    rvs_path = config_dict['path']
    config_file = test_config.get('config_file')
    timeout = test_config.get('timeout', 1800)

    # Determine config path
    config_path = determine_rvs_config_path(phdl, config_dict, config_file)

    # Run RVS test
    out_dict = phdl.exec(f'{rvs_path}/rvs -c {config_path}', timeout=timeout)
    print_test_output(log, out_dict)
    scan_test_results(out_dict)

    # Parse and validate results
    parse_rvs_test_results(test_config, out_dict)
    update_test_result()


def test_rvs_gpu_enumeration(phdl, config_dict):
    """
    Run RVS GPU enumeration test to detect and validate GPU presence.
    This is a basic connectivity and detection test.

    Args:
      phdl: Parallel SSH handle
      config_dict: RVS configuration dictionary
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
        if re.search(r'No supported GPUs available', out_dict[node], re.I):
            fail_test(f'No GPUs detected in RVS enumeration on node {node}')

    update_test_result()

def test_rvs_gpup_single(phdl, config_dict):
    """
    Run RVS GPUP (GPU Properties) test.
    This test validates GPU properties and capabilities.

    Args:
      phdl: Parallel SSH handle
      config_dict: RVS configuration dictionary
    """
    test_name = 'gpup_single'
    execute_rvs_test(phdl, config_dict, test_name)


def test_rvs_mem_test(phdl, config_dict):
    """
    Run RVS Memory Test.
    This test validates GPU memory functionality and integrity.

    Args:
      phdl: Parallel SSH handle
      config_dict: RVS configuration dictionary
    """
    test_name = 'mem_test'
    execute_rvs_test(phdl, config_dict, test_name)


def test_rvs_gst_single(phdl, config_dict):
    """
    Run RVS GST (GPU Stress Test) - Single GPU validation test.
    This test runs the GPU stress test configuration to validate GPU functionality
    and performance under load.

    Args:
      phdl: Parallel SSH handle
      config_dict: RVS configuration dictionary
    """
    test_name = 'gst_single'
    execute_rvs_test(phdl, config_dict, test_name)


def test_rvs_iet_single(phdl, config_dict):
    """
    Run RVS IET (Input EDPp Test) - Single GPU validation test.
    This test validates power consumption and thermal behavior under load.

    Args:
      phdl: Parallel SSH handle
      config_dict: RVS configuration dictionary
    """
    test_name = 'iet_single'
    execute_rvs_test(phdl, config_dict, test_name)


def test_rvs_pebb_single(phdl, config_dict):
    """
    Run RVS PEBB (PCI Express Bandwidth Benchmark).
    This test measures and validates PCI Express bandwidth performance.

    Args:
      phdl: Parallel SSH handle
      config_dict: RVS configuration dictionary
    """
    test_name = 'pebb_single'
    execute_rvs_test(phdl, config_dict, test_name)


def test_rvs_pbqt_single(phdl, config_dict):
    """
    Run RVS PBQT (P2P Benchmark and Qualification Tool).
    This test validates peer-to-peer communication between GPUs.

    Args:
      phdl: Parallel SSH handle
      config_dict: RVS configuration dictionary
    """
    test_name = 'pbqt_single'
    execute_rvs_test(phdl, config_dict, test_name)


def test_rvs_peqt_single(phdl, config_dict):
    """
    Run RVS PEQT (PCI Express Qualification Tool).
    This test validates PCI Express link quality and stability.

    Args:
      phdl: Parallel SSH handle
      config_dict: RVS configuration dictionary
    """
    test_name = 'peqt_single'
    execute_rvs_test(phdl, config_dict, test_name)


def test_rvs_rcqt_single(phdl, config_dict):
    """
    Run RVS RCQT (ROCm Configuration Qualification Tool).
    This test validates ROCm configuration and system setup.

    Args:
      phdl: Parallel SSH handle
      config_dict: RVS configuration dictionary
    """
    test_name = 'rcqt_single'
    execute_rvs_test(phdl, config_dict, test_name)


def test_rvs_tst_single(phdl, config_dict):
    """
    Run RVS TST (Thermal Stress Test).
    This test validates GPU thermal management under stress conditions.

    Args:
      phdl: Parallel SSH handle
      config_dict: RVS configuration dictionary
    """
    test_name = 'tst_single'
    execute_rvs_test(phdl, config_dict, test_name)


def test_rvs_babel_stream(phdl, config_dict):
    """
    Run RVS BABEL Benchmark test.
    This test runs the BABEL streaming benchmark for GPU memory bandwidth validation.

    Args:
      phdl: Parallel SSH handle
      config_dict: RVS configuration dictionary
    """
    test_name = 'babel_stream'
    execute_rvs_test(phdl, config_dict, test_name)

