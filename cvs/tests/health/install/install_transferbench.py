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
    Load and return the TransferBench test configuration subsection.

    Args:
      config_file (str): Path to the test configuration JSON file.

    Returns:
      dict: The 'transferbench' configuration block containing expected bandwidths, paths, etc.

    """
    with open(config_file) as json_file:
        config_dict_t = json.load(json_file)
    config_dict = config_dict_t['transferbench']

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
      - nhdl_dict is currently unused; it can be removed unless used elsewhere.
    """
    nhdl_dict = {}
    node_list = list(cluster_dict['node_dict'].keys())
    head_node = node_list[0]
    shdl = Pssh( log, [head_node], user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'] )
    return shdl





# Create connection to DUT, MTPs, Switches and export for later use ..
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
    phdl = Pssh( log, node_list, user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'] )
    return phdl







def test_install_transferbench(phdl, shdl, config_dict ):
    """
    Install/Build TransferBench and verify installation on all nodes.

    Steps:
      - Remove any stale working directory on the head node (if using shared/NFS paths).
      - Clone TransferBench repo under git_install_path (via head-node hdl).
      - Build on all nodes using CC=hipcc (via phdl).
      - Verify '/opt/amd/transferbench' contains 'TransferBench' on each node.


    Depending on the flag nfs_install in the config_file, decide if you want to use shdl (single node)
    or all nodes (phdl)
    If the nfs_install flag is set to True, then it assumes the install_dir is on a common file system
    that is accessible from all the nodes and installs from just a single node, otherwise install on
    all the nodes.

    Args:
      hdl: Head-node SSH handle
      phdl: Parallel SSH handle for all nodes.
      config_dict (dict): Includes:
        - git_install_path: directory to clone/build
        - git_url: repository URL
    """


    globals.error_list = []

    # For install case, if the systems are using NFS, use single connection to 
    if config_dict['nfs_install'] is True:
        hdl = shdl
    else:
        hdl = phdl


    log.info('Testcase install transferbench')
    git_install_path = config_dict['git_install_path']
    git_url = config_dict['git_url']

    out_dict = shdl.exec(f'ls -ld {git_install_path}')
    for node in out_dict.keys():
        if re.search( 'No such file', out_dict[node] ):
            hdl.exec(f'mkdir -p {git_install_path}')

    out_dict = hdl.exec(f'rm -rf {git_install_path}/TransferBench')
    out_dict = hdl.exec(f'cd {git_install_path};git clone {git_url}', timeout=120)

    # Build
    out_dict = hdl.exec(f'cd {git_install_path}/TransferBench;CC=hipcc make', timeout=500 )

    # Verify installation happened fine on all nodes
    out_dict = phdl.exec(f'ls -l {git_install_path}/TransferBench')
    for node in out_dict.keys():
        if not re.search( 'TransferBench', out_dict[node] ):
            fail_test(f'Transfer bench installation failed on node {node}')
    update_test_result()   
 

