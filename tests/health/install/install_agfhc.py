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
      - nhdl_dict is currently unused; it can be removed unless used elsewhere.
    """
    nhdl_dict = {}
    node_list = list(cluster_dict['node_dict'].keys())
    head_node = node_list[0]
    shdl = Pssh( log, [head_node], user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'] )
    return shdl


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



@pytest.mark.dependency(name="init")
def test_install_agfhc(phdl, shdl, config_dict, ):
    """
    Install AGFHC from the tarball package provided in the input config_file.json - package_tar_ball
    under the directory specified in the config_file - install_dir

    Depending on the flag nfs_install in the config_file, decide if you want to use shdl (single node)
    or all nodes (phdl)
    If the nfs_install flag is set to True, then it assumes the install_dir is on a common file system
    that is accessible from all the nodes and installs from just a single node, otherwise install on
    all the nodes.
    """
    globals.error_list = []
    log.info('Testcase install agfhc')
    install_dir = config_dict['install_dir']
    package_tar_ball = config_dict['package_tar_ball']

    if config_dict['nfs_install'] is True:
        hdl = shdl
    else:
        hdl = phdl


    # Check if install directory exists, otherwise create.
    out_dict = phdl.exec( f'ls -ld {install_dir}' )
    for node in out_dict.keys():
        print(f'node ip {node}')
        print(out_dict[node])
        if re.search( 'No such file or directory', out_dict[node], re.I ):
            print(f'Install directory {install_dir} does not exist, creating')
            hdl.exec(f'mkdir -p {install_dir}')


    # Copy the package to the install directory and untar
    hdl.exec( f'cd {install_dir};cp {package_tar_ball} . && tar -xvf {package_tar_ball}' )

    # install the untarred file
    try:
        out_dict = hdl.exec(f'cd {install_dir};sudo ./install', timeout=90)
        for node in out_dict.keys():
            print(out_dict[node])
            if re.search( 'Error|No such file', out_dict[node], re.I ):
                fail_test(f'Installation of AGFHC failed on node {node}')
    except Exception as e:
        print(f'Install of AGFHC failed, hit exception {e}')


    # verify agfhc path exists after installation ..
    out_dict = phdl.exec(f'ls -l {config_dict['path']}/agfhc')
    for node in out_dict.keys():
        print(out_dict[node])
        if re.search( 'No such file', out_dict[node], re.I ):
            fail_test(f'Installation of AGFHC failed on node {node}')
    update_test_result()
