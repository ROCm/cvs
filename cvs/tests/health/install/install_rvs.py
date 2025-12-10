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


# Importing additional cmd line args to script ..
@pytest.fixture(scope="module")
def cluster_file(pytestconfig):
    """
    Retrieve the --cluster_file CLI option provided to pytest.

    Args:
      pytestconfig: Built-in pytest fixture exposing command-line options.

    Returns:
      str: Path to the cluster configuration JSON file.
    """
    return pytestconfig.getoption("cluster_file")


@pytest.fixture(scope="module")
def config_file(pytestconfig):
    """
    Retrieve the --config_file CLI option provided to pytest.

    Args:
      pytestconfig: Built-in pytest fixture exposing command-line options.

    Returns:
      str: Path to the test configuration JSON file.

    Notes:
      - Ensure your pytest invocation includes: --config_file=/path/to/config.json
      - Module scope ensures this is resolved once per module.
    """
    return pytestconfig.getoption("config_file")


# Importing the cluster and cofig files to script to access node, switch, test config params
@pytest.fixture(scope="module")
def cluster_dict(cluster_file):
    """
    Load and return the entire cluster configuration.

    Args:
      cluster_file (str): Path to the cluster configuration JSON file.

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
    Load and return the RVS test configuration subsection.

    Args:
      config_file (str): Path to the test configuration JSON file.

    Returns:
      dict: The 'rvs' configuration block containing expected results, paths, etc.
    """
    with open(config_file) as json_file:
        config_dict_t = json.load(json_file)
    config_dict = config_dict_t['rvs']

    # Resolve path placeholders like {user-id}, {home-mount-dir}, etc.
    config_dict = resolve_test_config_placeholders(config_dict, cluster_dict)

    log.info(config_dict)
    return config_dict


@pytest.fixture(scope="module")
def shdl(cluster_dict):
    """
    Build and return a parallel SSH handle (Pssh) for the head node only.

    Args:
      cluster_dict (dict): Cluster metadata fixture (see phdl docstring).

    Returns:
      Pssh: Handle configured for the first node (head node) in node_dict.

    Notes:
      - Useful when commands should be executed only from a designated head node.
      - Module scope ensures a single connection context for the duration of the module.
    """
    nhdl_dict = {}
    node_list = list(cluster_dict['node_dict'].keys())
    head_node = node_list[0]
    shdl = Pssh(log, [head_node], user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'])
    return shdl


# Create connection to DUTs and export for later use ..
@pytest.fixture(scope="module")
def phdl(cluster_dict):
    """
    Create a parallel SSH handle for all cluster nodes.

    Args:
      cluster_dict (dict): Loaded cluster configuration with at least:
        - node_dict: mapping of node -> details
        - username: SSH username
        - priv_key_file: path to SSH key

    Returns:
      Pssh: Handle that executes commands on all nodes and returns dict[node] -> output.
    """
    print(cluster_dict)
    node_list = list(cluster_dict['node_dict'].keys())
    phdl = Pssh(log, node_list, user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'])
    return phdl


@pytest.mark.dependency(name="init")
def test_install_rvs(phdl, shdl, config_dict):
    """
    Install/Build ROCmValidationSuite (RVS) and verify installation on all nodes.

    Steps:
      - Check if RVS is already installed via package manager
      - If not installed, clone RVS repo under git_install_path
      - Build and install RVS on all nodes
      - Verify RVS executable exists and configuration files are accessible

    Depending on the flag nfs_install in the config_file, decide if you want to use shdl (single node)
    or all nodes (phdl)
    If the nfs_install flag is set to True, then it assumes the install_dir is on a common file system
    that is accessible from all the nodes and installs from just a single node, otherwise install on
    all the nodes.

    Args:
      phdl: Parallel SSH handle for all nodes.
      shdl: Head-node SSH handle
      config_dict (dict): Includes:
        - git_install_path: directory to clone/build
        - git_url: repository URL
        - path: expected installation path for RVS binary
        - nfs_install: whether to use NFS-shared installation
    """
    globals.error_list = []

    # For install case, if the systems are using NFS, use single connection to head node
    if config_dict['nfs_install'] is True:
        hdl = shdl
    else:
        hdl = phdl

    log.info('Testcase install RVS (ROCmValidationSuite)')
    git_install_path = config_dict['git_install_path']
    git_url = config_dict['git_url']
    rvs_path = config_dict['path']

    # Check if RVS is already installed via system packages
    out_dict = phdl.exec('which rvs', timeout=30)
    rvs_found = False
    for node in out_dict.keys():
        if out_dict[node].strip() and re.search('rvs', out_dict[node], re.I):
            log.info(f'RVS appears to be already installed on node {node} at: {out_dict[node].strip()}')
            rvs_found = True

    # Check if RVS config files exist
    out_dict = phdl.exec(
        f'ls -l {config_dict["config_path_default"]}/gst_single.conf || ls -l {config_dict["config_path_mi300x"]}/gst_single.conf',
        timeout=30,
    )
    config_found = False
    for node in out_dict.keys():
        if not re.search('No such file', out_dict[node], re.I):
            log.info(f'RVS configuration files found on node {node}')
            config_found = True

    # If RVS is not found or configs are missing, install it
    if not rvs_found or not config_found:
        log.info('RVS not found, attempting to install from artifactory repo first')

        # First try to install from artifactory repo
        package_installed = False
        out_dict = hdl.exec('sudo apt-get update -y', timeout=600)
        out_dict = hdl.exec(
            'sudo apt-get install -y libpci3 libpci-dev doxygen unzip cmake git libyaml-cpp-dev', timeout=600
        )
        out_dict = hdl.exec('sudo apt-get install -y rocblas rocm-smi-lib', timeout=600)
        out_dict = hdl.exec('sudo apt-get install -y rocm-validation-suite', timeout=600)

        for node in out_dict.keys():
            if re.search(
                'Unable to locate package|Package.*not found|E: Could not get lock|dpkg: error', out_dict[node], re.I
            ):
                log.info(f'RVS package installation failed on node {node}, will try building from source')
            else:
                log.info(f'RVS package installation successful on node {node}')
                package_installed = True

        # If package installation failed, build from source
        if not package_installed:
            log.info('Installing RVS from source')

            # Check if install directory exists, otherwise create
            out_dict = hdl.exec(f'ls -ld {git_install_path}')
            for node in out_dict.keys():
                if re.search('No such file', out_dict[node]):
                    hdl.exec(f'mkdir -p {git_install_path}')

            # Remove any existing RVS directory and clone fresh
            out_dict = hdl.exec(f'rm -rf {git_install_path}/ROCmValidationSuite')
            out_dict = hdl.exec(f'cd {git_install_path};git clone {git_url}', timeout=300)

            # Build and install RVS
            try:
                out_dict = hdl.exec(
                    f'cd {git_install_path}/ROCmValidationSuite; cmake -B ./build -DROCM_PATH=/opt/rocm -DCMAKE_INSTALL_PREFIX=/opt/rocm -DCPACK_PACKAGING_INSTALL_PREFIX=/opt/rocm',
                    timeout=600,
                )
                out_dict = hdl.exec(
                    f'cd {git_install_path}/ROCmValidationSuite/build; make -j$(nproc) package', timeout=600
                )
                out_dict = hdl.exec(
                    f'cd {git_install_path}/ROCmValidationSuite/build; sudo dpkg -i rocm-validation-suite_*.deb',
                    timeout=600,
                )

                for node in out_dict.keys():
                    if re.search('Error|FAILED|No such file', out_dict[node], re.I):
                        fail_test(f'RVS build/installation failed on node {node}')

            except Exception as e:
                fail_test(f'RVS installation failed with exception: {e}')

    # Verify RVS installation
    out_dict = phdl.exec('which rvs || ls -l /opt/rocm/bin/rvs*', timeout=60)
    for node in out_dict.keys():
        if re.search('not found|No such file', out_dict[node], re.I) and not re.search('rvs', out_dict[node]):
            fail_test(f'RVS installation verification failed on node {node}')

    # Verify config files are accessible
    out_dict = phdl.exec(
        f'ls -l {config_dict["config_path_mi300x"]}/gst_single.conf || ls -l {config_dict["config_path_default"]}/gst_single.conf',
        timeout=60,
    )
    for node in out_dict.keys():
        if re.search('No such file', out_dict[node], re.I):
            fail_test(f'RVS configuration files not found on node {node}')

    log.info('RVS installation and verification completed successfully')
    update_test_result()
