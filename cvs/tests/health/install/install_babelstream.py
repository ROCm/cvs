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
import time


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
      - Module scope ensures this runs once per module to avoid repeated lookups.
    """
    return pytestconfig.getoption("config_file")


# Importing the cluster and cofig files to script to access node, switch, test config params
@pytest.fixture(scope="module")
def cluster_dict(cluster_file):
    """
    Load and return the entire cluster configuration from JSON.

    Args:
    cluster_file (str): Path to the cluster JSON file.

    Returns:
    dict: Parsed cluster configuration (nodes, credentials, etc.).
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
    Load and return the 'babelstream' subsection from the test configuration JSON.

    Args:
      config_file (str): Path to the test configuration JSON.

    Returns:
      dict: The 'babelstream' configuration block, expected to include:
        - path: location where hip-stream (BabelStream HIP binary) will live
        - git_install_path: directory to clone and build BabelStream
        - git_url: BabelStream repository URL
        - results: expected performance thresholds for kernels (copy/add/mul/triad/dot)
    """
    with open(config_file) as json_file:
        config_dict_t = json.load(json_file)
    config_dict = config_dict_t['babelstream']

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
    shdl = Pssh(log, [head_node], user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'])
    return shdl


@pytest.fixture(scope="module")
def phdl(cluster_dict):
    """
    Create a parallel SSH handle (Pssh) for executing commands across all cluster nodes.

    Args:
      cluster_dict (dict): Cluster metadata containing at least:
        - node_dict: mapping of node name/IP -> details
        - username: SSH username for nodes
        - priv_key_file: path to SSH private key

    Returns:
      Pssh: A handle that runs commands in parallel and returns a dict of node -> output.

    """
    print(cluster_dict)
    node_list = list(cluster_dict['node_dict'].keys())
    phdl = Pssh(log, node_list, user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'])
    return phdl


@pytest.mark.dependency(name="init")
def test_install_babelstream(phdl, shdl, config_dict):
    """
    Install BabelStream (HIP model) across nodes if not already present, and verify the hip-stream binary.

    Steps:
      1) Check if the designated path exists on the head node.
      2) If absent, on all nodes:
         - git clone the repository,
         - configure with cmake (MODEL=hip, compiler=hipcc),
         - build the project.
      3) Verify 'hip-stream' is present on all nodes.
      4) Export the build path into PATH for the current shell context.
      5) Update the test result status.

    Args:
      hdl: Single-node SSH handler for quick checks on head node (NFS/shared path).
      phdl: Parallel SSH handler for running commands on all nodes.
      config_dict (dict): Contains:
        - path: where hip-stream will reside
        - git_install_path: directory to clone/build BabelStream
        - git_url: repo URL to clone from

    """
    globals.error_list = []
    log.info('Testcase install babelstream')
    path = config_dict['path']
    git_install_path = config_dict['git_install_path']
    git_url = config_dict['git_url']
    print(git_install_path)
    if config_dict['nfs_install'] is True:
        hdl = shdl
    else:
        hdl = phdl

    out_dict = shdl.exec(f'ls -l {path}')
    for node in out_dict.keys():
        output = out_dict[node]
    if re.search('No such file', output, re.I):
        out_dict = hdl.exec(f'cd {git_install_path};git clone {git_url};cd')
        out_dict = hdl.exec(
            f'cd {git_install_path}/BabelStream;cmake -Bbuild -H. -DMODEL=hip -DCMAKE_CXX_COMPILER=hipcc'
        )
        out_dict = hdl.exec(f'cd {git_install_path}/BabelStream;cmake --build build')
        out_dict = phdl.exec(f'ls -l {path}')
        for node in out_dict.keys():
            if not re.search('hip-stream', out_dict[node], re.I):
                fail_test(f'Installation of BabelStream failed, hip-stream file not found on node {node}')
        phdl.exec(f'export PATH={git_install_path}/BabelStream/build:$PATH')
    update_test_result()


def test_install_open_mpi(
    phdl,
    config_dict,
):
    """
    Install Open MPI across all nodes and verify that mpiexec is available.

    Args:
      phdl: Parallel SSH handle capable of executing commands on all nodes. Expected to
            return a dict mapping node -> command output for each exec call.
      config_dict (dict): Test configuration. Includes:
        - 'path': Base path used elsewhere; not directly used here but kept for consistency.

    Behavior:
      - Resets the global error list for a clean test run.
      - Updates package indexes (apt) and installs Open MPI components.
      - Verifies installation by checking for 'mpiexec' on each node.
      - Records failures via fail_test and finalizes the test status via update_test_result.

    Assumptions:
      - Target systems are Debian/Ubuntu-based (uses apt/apt-get). Adapt for RHEL/CentOS if needed.
      - phdl.exec supports a 'timeout' parameter and returns a dict of node outputs.
      - fail_test and update_test_result are available in the test environment.
    """
    globals.error_list = []
    log.info('Testcase install openmpi')
    if config_dict['nfs_install'] is True:
        hdl = shdl
    else:
        hdl = phdl
    path = config_dict['path']
    out_dict = phdl.exec(f'sudo apt update -y', timeout=200)
    out_dict = phdl.exec(f'sudo apt-get install -y openmpi-bin openmpi-common libopenmpi-dev', timeout=200)
    out_dict = phdl.exec('which mpiexec')
    for node in out_dict.keys():
        if not re.search('mpiexec', out_dict[node]):
            fail_test(f'Open MPI installation failed on node {node}')
    update_test_result()
