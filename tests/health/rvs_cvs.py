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
from packaging import version

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

    # Resolve path placeholders like {user-id} in cluster config
    cluster_dict = resolve_cluster_config_placeholders(cluster_dict)

    log.info(cluster_dict)
    return cluster_dict


@pytest.fixture(scope="module")
def config_dict(config_file, cluster_dict):
    with open(config_file) as json_file:
        config_dict_t = json.load(json_file)
    config_dict = config_dict_t['rvs']

    # Resolve path placeholders like {user-id}, {home-mount-dir}, etc.
    config_dict = resolve_test_config_placeholders(config_dict, cluster_dict)

    log.info(config_dict)
    return config_dict


@pytest.fixture(scope="module")
def phdl(cluster_dict):
    print(cluster_dict)
    node_list = list(cluster_dict['node_dict'].keys())
    phdl = Pssh( log, node_list, user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'] )
    return phdl


@pytest.fixture(scope="module")
def rvs_test_level(config_dict):
    """Get RVS test level from config file, default to 4 if not provided or invalid"""
    level = config_dict.get('rvs_test_level', 4)

    # Validate level is between 0 and 5
    # Level 0 is special: run all individual tests regardless of RVS version
    try:
        level_int = int(level)
        if 0 <= level_int <= 5:
            log.info(f'Using RVS test level from config file: {level_int}')
            return level_int
        else:
            log.warning(f'Invalid RVS test level in config: {level}. Using default level 4')
            return 4
    except (ValueError, TypeError):
        log.warning(f'Invalid RVS test level format in config: {level}. Using default level 4')
        return 4


@pytest.fixture(scope="module")
def rvs_version(phdl, config_dict):
    """
    Detect RVS version from all nodes.
    Returns the minimum version across all nodes.
    """
    return get_rvs_version(phdl, config_dict['path'])


def get_rvs_version(phdl, rvs_path):
    """
    Get RVS version from all nodes.

    Args:
      phdl: Parallel SSH handle
      rvs_path: Path to RVS binary

    Returns:
      str: Minimum version across all nodes (e.g., "1.2.0", "1.3.0")
    """
    version_dict = {}

    out_dict = phdl.exec(f'{rvs_path}/rvs --version', timeout=30)

    for node in out_dict.keys():
        output = out_dict[node].strip()

        # Extract version - output is just the version number like "1.2.0" or "1.3.0"
        # First try to match version pattern (digits.digits.digits)
        version_match = re.search(r'(\d+\.\d+\.\d+)', output)

        if version_match:
            node_version = version_match.group(1)
            log.info(f'Node {node}: RVS version {node_version}')
            version_dict[node] = node_version
        else:
            log.warning(f'Node {node}: Could not detect RVS version from output: {output}')
            version_dict[node] = "0.0.0"

    # Return minimum version across all nodes
    if version_dict:
        min_version = min(version_dict.values(), key=lambda v: version.parse(v))
        log.info(f'Using minimum RVS version across all nodes: {min_version}')
        return min_version
    else:
        log.error('Could not determine RVS version from any node')
        return "0.0.0"


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection based on RVS version and test level.

    Logic:
    - If rvs_test_level == 0: Run all individual tests, skip level_config
    - If RVS version < 1.3.0: Run all individual tests, skip level_config
    - If RVS version >= 1.3.0 and rvs_test_level != 0: Run only level_config, skip individual tests
    """
    # We need to get RVS version, but fixtures aren't available here yet
    # So we'll mark tests and handle skipping in the test functions themselves
    pass


def should_skip_level_test(rvs_version_str, rvs_test_level):
    """
    Determine if level_config test should be skipped.

    Args:
      rvs_version_str: RVS version string (e.g., "1.2.0")
      rvs_test_level: Test level (0-5)

    Returns:
      tuple: (should_skip, reason)
    """
    # Skip if level is 0 (user wants individual tests)
    if rvs_test_level == 0:
        return (True, "rvs_test_level=0: Running individual tests instead")

    # Skip if RVS version < 1.3.0
    if version.parse(rvs_version_str) < version.parse("1.3.0"):
        return (True, f"RVS version {rvs_version_str} < 1.3.0: LEVEL configs not supported")

    return (False, None)


def should_skip_individual_test(rvs_version_str, rvs_test_level):
    """
    Determine if individual tests should be skipped.

    Args:
      rvs_version_str: RVS version string (e.g., "1.2.0")
      rvs_test_level: Test level (0-5)

    Returns:
      tuple: (should_skip, reason)
    """
    # Don't skip if level is 0 (user explicitly wants individual tests)
    if rvs_test_level == 0:
        return (False, None)

    # Don't skip if RVS version < 1.3.0 (level test not available)
    if version.parse(rvs_version_str) < version.parse("1.3.0"):
        return (False, None)

    # Skip individual tests if RVS >= 1.3.0 and level != 0
    return (True, f"RVS version {rvs_version_str} >= 1.3.0: Running LEVEL-{rvs_test_level} test instead")



def get_gpu_device_name(phdl):
    """
    Detect GPU device name from amd-smi JSON output to match with RVS config folders.

    Args:
      phdl: Parallel SSH handle

    Returns:
      dict: Dictionary of node -> device name (e.g., 'MI300X', 'MI308X', 'MI300XHF', etc.)
    """
    device_map = {}

    # Execute amd-smi command to get GPU information in JSON format
    out_dict = phdl.exec('sudo amd-smi static -a -g 0 --json', timeout=30)

    for node in out_dict.keys():
        output = out_dict[node]

        try:
            # Parse JSON output
            gpu_info = json.loads(output)

            # Extract market name from the first GPU
            if 'gpu_data' in gpu_info and len(gpu_info['gpu_data']) > 0:
                market_name = gpu_info['gpu_data'][0].get('asic', {}).get('market_name', '')

                if market_name:
                    # Remove "AMD Instinct " prefix and any spaces
                    device_name = market_name.replace('AMD Instinct ', '').replace(' ', '')

                    if device_name:
                        log.info(f'Node {node}: Detected GPU device from market_name: {market_name} -> {device_name}')
                        device_map[node] = device_name
                    else:
                        log.warning(f'Node {node}: Market name found but device name is empty after processing')
                        device_map[node] = None
                else:
                    log.warning(f'Node {node}: Market name not found in JSON output')
                    device_map[node] = None
            else:
                log.warning(f'Node {node}: No GPU data found in JSON output')
                device_map[node] = None

        except json.JSONDecodeError as e:
            log.error(f'Node {node}: Failed to parse JSON output from amd-smi: {e}')
            device_map[node] = None
        except Exception as e:
            log.error(f'Node {node}: Error processing amd-smi output: {e}')
            device_map[node] = None

    return device_map


def get_available_device_folders(phdl, base_config_path):
    """
    Get list of available MI300 variant device-specific folders in RVS config directory.
    Only returns folders starting with 'MI3' (MI300 variants).

    Args:
      phdl: Parallel SSH handle
      base_config_path: Base path for RVS configurations

    Returns:
      dict: Dictionary of node -> list of available MI300 device folders
    """
    available_folders = {}

    out_dict = phdl.exec(f'ls -d {base_config_path}/*/ 2>/dev/null', timeout=30)

    for node in out_dict.keys():
        output = out_dict[node]

        # Extract folder names from paths
        folders = []
        for line in output.split('\n'):
            if line.strip():
                # Extract folder name from path like '/opt/rocm/.../MI300X/'
                folder_match = re.search(r'/([^/]+)/$', line.strip())
                if folder_match:
                    folder_name = folder_match.group(1)
                    # Only add folders that start with 'MI3' (MI300 variants)
                    if folder_name.startswith('MI3'):
                        folders.append(folder_name)

        available_folders[node] = folders
        log.info(f'Node {node}: Available MI300 variant device folders: {folders}')

    return available_folders


def determine_rvs_config_path(phdl, config_dict, config_file):
    """
    Determine the correct configuration file path for RVS tests.
    Dynamically detects GPU device and checks for device-specific config folders.

    Args:
      phdl: Parallel SSH handle
      config_dict: RVS configuration dictionary
      config_file: Name of the configuration file to look for

    Returns:
      str: Full path to the configuration file to use
      None: If config file is not found on any node
    """
    base_path = config_dict['config_path_default']
    default_path = f"{base_path}/{config_file}"

    # Step 1: Detect GPU device name on each node
    device_map = get_gpu_device_name(phdl)

    # Step 2: Get available device-specific folders
    available_folders = get_available_device_folders(phdl, base_path)

    # Step 3: Check for device-specific config on each node
    device_specific_exists = False
    default_exists = False
    chosen_path = None

    for node in device_map.keys():
        device_name = device_map.get(node)
        node_folders = available_folders.get(node, [])

        if device_name and device_name in node_folders:
            # Device-specific folder exists, check if config file is present
            device_specific_path = f"{base_path}/{device_name}/{config_file}"

            out_dict = phdl.exec(f'ls -l {device_specific_path}', timeout=30)
            if not re.search('No such file', out_dict[node], re.I):
                log.info(f'Node {node}: Device-specific config found: {device_specific_path}')
                device_specific_exists = True
                chosen_path = device_specific_path
            else:
                log.info(f'Node {node}: Device-specific folder exists but config file not found: {device_specific_path}')
        else:
            log.info(f'Node {node}: No device-specific folder match for device: {device_name}')

    # If device-specific config exists on any node, use it
    if device_specific_exists:
        log.info(f'Using device-specific config: {chosen_path}')
        return chosen_path

    # Step 4: Fall back to default config (no device subfolder)
    log.info(f'Falling back to default config path')
    out_dict = phdl.exec(f'ls -l {default_path}', timeout=30)

    for node in out_dict.keys():
        if not re.search('No such file', out_dict[node], re.I):
            log.info(f'Node {node}: Default config found: {default_path}')
            default_exists = True
        else:
            log.error(f'Node {node}: Default config not found: {default_path}')

    # Use default path if it exists on any node
    if default_exists:
        log.info(f'Using default config: {default_path}')
        return default_path

    # If neither exists on any node, return None
    log.error(f'Configuration file {config_file} not found in either device-specific or default location on any node')
    return None

def parse_rvs_test_results(test_config, out_dict):
    """
    Generic parser for RVS test results that validates against expected patterns.

    Args:
      test_config: Test configuration dictionary containing name and fail_regex_pattern
      out_dict: Dictionary of node -> command output
    """
    test_name = test_config.get('name', 'unknown')
    fail_pattern = test_config.get('fail_regex_pattern', r'\[ERROR\s*\]')

    # Standard parsing for all tests
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
    timeout = test_config.get('timeout', 9000)

    # Determine config path
    config_path = determine_rvs_config_path(phdl, config_dict, config_file)

    # TEMP-FIX start
    if test_name == 'gst_single':
        # Check if MI355X string is present in the config_path
        if config_path and ('MI355X' in config_path or 'MI350X' in config_path ):
            # Use specific config file for MI355X gst_single test with absolute path
            config_path = os.path.join(os.getcwd(), "input/config_file/health/rvs_mi355x_gst_single_edited.conf")
            log.info(f'Using custom MI355X config path for gst_single test: {config_path}')
    # TEMP-FIX  end

    if config_path is not None:
        # Run RVS test
        if test_name == 'peqt_single':
            # PEQT test requires elevated permissions
            rvs_cmd = f'sudo {rvs_path}/rvs -c {config_path}'
        else:
            rvs_cmd = f'{rvs_path}/rvs -c {config_path}'

        out_dict = phdl.exec(f'{rvs_cmd}', timeout=timeout)
        print_test_output(log, out_dict)
        scan_test_results(out_dict)

        # Parse and validate results
        parse_rvs_test_results(test_config, out_dict)
    else:
        fail_test(f'Configuration file [{config_file}] for {test_name} not found on any/some node.')

    update_test_result()




def parse_rvs_level_results(test_config, out_dict, level):
    """
    Parser for RVS LEVEL-based test results.
    Checks for multiple failure patterns from all RVS modules.

    Args:
      test_config: Test configuration dictionary
      out_dict: Dictionary of node -> command output
      level: RVS test level (1-5)
    """
    test_name = f'level_{level}_config'
    fail_patterns = test_config.get('fail_regex_patterns', [])

    if not fail_patterns:
        log.warning(f'No fail patterns defined for RVS LEVEL {level} test')
        return

    for node in out_dict.keys():
        output = out_dict[node]
        node_passed = True
        failures_found = []

        # Check each failure pattern
        for pattern in fail_patterns:
            if re.search(pattern, output, re.I):
                failures_found.append(pattern)
                node_passed = False

        # Report results
        if not node_passed:
            fail_msg = f'RVS LEVEL-{level} test failed on node {node}. Failure patterns found: {", ".join(failures_found)}'
            fail_test(fail_msg)
        else:
            log.info(f'RVS LEVEL-{level} test passed on node {node}: All module checks passed')


################################################################################
# Testcases for RVS modules
################################################################################

def test_rvs_level_config(phdl, config_dict, rvs_version, rvs_test_level):
    """
    Run RVS LEVEL-based configuration test.
    This test runs all RVS modules collectively using the -r (run level) option.

    Valid levels: 1-5 (default: 4)
    Level 1: Quick basic checks
    Level 2: Standard validation
    Level 3: Extended validation
    Level 4: Comprehensive testing (default)
    Level 5: Maximum stress testing

    Args:
      phdl: Parallel SSH handle
      config_dict: RVS configuration dictionary
      rvs_test_level: Test level (0-5)
      rvs_version: RVS version string
    """
    # Check if test should be skipped
    should_skip, skip_reason = should_skip_level_test(rvs_version, rvs_test_level)
    if should_skip:
        pytest.skip(f"test_rvs_level_config: {skip_reason}")

    globals.error_list = []

    log.info(f'Testcase Run RVS LEVEL-{rvs_test_level} Configuration Test')
    log.info(f'RVS Version: {rvs_version}')

    # Get test configuration
    test_config = next((test for test in config_dict['tests'] if test['name'] == 'level_config'), None)

    if not test_config:
        log.warning('LEVEL test configuration not found in config file. Using default settings.')
        test_config = {
            'name': 'level_config',
            'description': f'RVS LEVEL-{rvs_test_level} Comprehensive Test',
            'timeout': 7200,
            'fail_regex_patterns': [],
            'expected_pass_patterns': []
        }

    rvs_path = config_dict['path']
    timeout = test_config.get('timeout', 7200)

    # Run RVS with level configuration
    # The -r option runs all modules with predefined configuration for that level
    rvs_cmd = f'sudo {rvs_path}/rvs -r {rvs_test_level}'

    log.info(f'Executing: {rvs_cmd}')
    out_dict = phdl.exec(rvs_cmd, timeout=timeout)
    print_test_output(log, out_dict)
    scan_test_results(out_dict)

    # Parse and validate results
    parse_rvs_level_results(test_config, out_dict, rvs_test_level)

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

def test_rvs_gpup_single(phdl, config_dict, rvs_version, rvs_test_level):
    """
    Run RVS GPUP (GPU Properties) test.
    This test validates GPU properties and capabilities.

    Args:
      phdl: Parallel SSH handle
      config_dict: RVS configuration dictionary
      rvs_version: RVS version string
      rvs_test_level: Test level (0-5)
    """
    # Check if test should be skipped
    should_skip, skip_reason = should_skip_individual_test(rvs_version, rvs_test_level)
    if should_skip:
        pytest.skip(f"test_rvs_gpup_single: {skip_reason}")

    test_name = 'gpup_single'
    execute_rvs_test(phdl, config_dict, test_name)


def test_rvs_mem_test(phdl, config_dict, rvs_version, rvs_test_level):
    """
    Run RVS Memory Test.
    This test validates GPU memory functionality and integrity.

    Args:
      phdl: Parallel SSH handle
      config_dict: RVS configuration dictionary
      rvs_version: RVS version string
      rvs_test_level: Test level (0-5)
    """
    # Check if test should be skipped
    should_skip, skip_reason = should_skip_individual_test(rvs_version, rvs_test_level)
    if should_skip:
        pytest.skip(f"test_rvs_mem_test: {skip_reason}")

    test_name = 'mem_test'
    execute_rvs_test(phdl, config_dict, test_name)


def test_rvs_gst_single(phdl, config_dict, rvs_version, rvs_test_level):
    """
    Run RVS GST (GPU Stress Test) - Single GPU validation test.
    This test runs the GPU stress test configuration to validate GPU functionality
    and performance under load.

    Args:
      phdl: Parallel SSH handle
      config_dict: RVS configuration dictionary
      rvs_version: RVS version string
      rvs_test_level: Test level (0-5)
    """
    # Check if test should be skipped
    should_skip, skip_reason = should_skip_individual_test(rvs_version, rvs_test_level)
    if should_skip:
        pytest.skip(f"test_rvs_gst_single: {skip_reason}")

    test_name = 'gst_single'
    execute_rvs_test(phdl, config_dict, test_name)


def test_rvs_iet_single(phdl, config_dict, rvs_version, rvs_test_level):
    """
    Run RVS IET (Input EDPp Test) - Single GPU validation test.
    This test validates power consumption and thermal behavior under load.

    Args:
      phdl: Parallel SSH handle
      config_dict: RVS configuration dictionary
      rvs_version: RVS version string
      rvs_test_level: Test level (0-5)
    """
    # Check if test should be skipped
    should_skip, skip_reason = should_skip_individual_test(rvs_version, rvs_test_level)
    if should_skip:
        pytest.skip(f"test_rvs_iet_single: {skip_reason}")

    test_name = 'iet_single'
    execute_rvs_test(phdl, config_dict, test_name)


def test_rvs_pebb_single(phdl, config_dict, rvs_version, rvs_test_level):
    """
    Run RVS PEBB (PCI Express Bandwidth Benchmark).
    This test measures and validates PCI Express bandwidth performance.

    Args:
      phdl: Parallel SSH handle
      config_dict: RVS configuration dictionary
      rvs_version: RVS version string
      rvs_test_level: Test level (0-5)
    """
    # Check if test should be skipped
    should_skip, skip_reason = should_skip_individual_test(rvs_version, rvs_test_level)
    if should_skip:
        pytest.skip(f"test_rvs_pebb_single: {skip_reason}")

    test_name = 'pebb_single'
    execute_rvs_test(phdl, config_dict, test_name)


def test_rvs_pbqt_single(phdl, config_dict, rvs_version, rvs_test_level):
    """
    Run RVS PBQT (P2P Benchmark and Qualification Tool).
    This test validates peer-to-peer communication between GPUs.

    Args:
      phdl: Parallel SSH handle
      config_dict: RVS configuration dictionary
      rvs_version: RVS version string
      rvs_test_level: Test level (0-5)
    """
    # Check if test should be skipped
    should_skip, skip_reason = should_skip_individual_test(rvs_version, rvs_test_level)
    if should_skip:
        pytest.skip(f"test_rvs_pbqt_single: {skip_reason}")

    test_name = 'pbqt_single'
    execute_rvs_test(phdl, config_dict, test_name)


def test_rvs_peqt_single(phdl, config_dict, rvs_version, rvs_test_level):
    """
    Run RVS PEQT (PCI Express Qualification Tool).
    This test validates PCI Express link quality and stability.

    Args:
      phdl: Parallel SSH handle
      config_dict: RVS configuration dictionary
      rvs_version: RVS version string
      rvs_test_level: Test level (0-5)
    """
    # Check if test should be skipped
    should_skip, skip_reason = should_skip_individual_test(rvs_version, rvs_test_level)
    if should_skip:
        pytest.skip(f"test_rvs_peqt_single: {skip_reason}")

    test_name = 'peqt_single'
    execute_rvs_test(phdl, config_dict, test_name)


def test_rvs_rcqt_single(phdl, config_dict, rvs_version, rvs_test_level):
    """
    Run RVS RCQT (ROCm Configuration Qualification Tool).
    This test validates ROCm configuration and system setup.

    Args:
      phdl: Parallel SSH handle
      config_dict: RVS configuration dictionary
      rvs_version: RVS version string
      rvs_test_level: Test level (0-5)
    """
    # Check if test should be skipped
    should_skip, skip_reason = should_skip_individual_test(rvs_version, rvs_test_level)
    if should_skip:
        pytest.skip(f"test_rvs_rcqt_single: {skip_reason}")

    test_name = 'rcqt_single'
    execute_rvs_test(phdl, config_dict, test_name)


def test_rvs_tst_single(phdl, config_dict, rvs_version, rvs_test_level):
    """
    Run RVS TST (Thermal Stress Test).
    This test validates GPU thermal management under stress conditions.

    Args:
      phdl: Parallel SSH handle
      config_dict: RVS configuration dictionary
      rvs_version: RVS version string
      rvs_test_level: Test level (0-5)
    """
    # Check if test should be skipped
    should_skip, skip_reason = should_skip_individual_test(rvs_version, rvs_test_level)
    if should_skip:
        pytest.skip(f"test_rvs_tst_single: {skip_reason}")

    test_name = 'tst_single'
    execute_rvs_test(phdl, config_dict, test_name)


def test_rvs_babel_stream(phdl, config_dict, rvs_version, rvs_test_level):
    """
    Run RVS BABEL Benchmark test.
    This test runs the BABEL streaming benchmark for GPU memory bandwidth validation.

    Args:
      phdl: Parallel SSH handle
      config_dict: RVS configuration dictionary
      rvs_version: RVS version string
      rvs_test_level: Test level (0-5)
    """
    # Check if test should be skipped
    should_skip, skip_reason = should_skip_individual_test(rvs_version, rvs_test_level)
    if should_skip:
        pytest.skip(f"test_rvs_babel_stream: {skip_reason}")

    test_name = 'babel_stream'
    execute_rvs_test(phdl, config_dict, test_name)

