'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import pytest

import re
import time
import json


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
    """
    return pytestconfig.getoption("config_file")


@pytest.fixture(scope="module")
def cluster_dict(cluster_file):
    """
    Load and return the entire cluster configuration.

    Args:
      cluster_file (str): Path to the cluster configuration JSON (from cluster_file fixture).

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
    Load and return the rocBLAS test configuration subsection.

    Args:
      config_file (str): Path to the test configuration JSON.

    Returns:
      dict: The 'rocblas' configuration block, expected to include expected GFLOP thresholds
            and other test settings specific to rocBLAS runs.

    """
    with open(config_file) as json_file:
        config_dict_t = json.load(json_file)
    config_dict = config_dict_t['rocblas']

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
    Build a parallel SSH handle (Pssh) for the entire cluster.

    Args:
      cluster_dict (dict): Cluster config that must include:
        - 'node_dict': mapping of node_identifier -> details
        - 'username': SSH username
        - 'priv_key_file': path to SSH private key

    Returns:
      Pssh: A handle that executes commands across all nodes and returns dict[node] -> output.

    """
    nhdl_dict = {}
    print(cluster_dict)
    node_list = list(cluster_dict['node_dict'].keys())
    phdl = Pssh(log, node_list, user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'])
    return phdl


@pytest.mark.dependency(name="init")
def test_rocblas_install(
    phdl,
    shdl,
    config_dict,
):
    """
    Install rocBLAS (clients only) from source and verify rocblas-bench presence.

    Args:
      hdl: Single-node SSH handler used for quick checks on the head node.
      phdl: Parallel SSH handler to run commands across all nodes.
      config_dict (dict): Must include:
        - 'path': final location expected to contain rocblas-bench
        - 'git_install_path': directory to clone/build rocBLAS
        - 'rocm_version': repo tag/branch to checkout (e.g., '6.2')
        - 'git_url': repository URL for rocBLAS

    Steps:
      - Install build prerequisites via apt.
      - Init git environment and clone rocBLAS into git_install_path.
      - Checkout rocm-{rocm_version}.
      - Run install.sh --clients-only --library-path /opt/rocm.
      - Verify rocblas-bench exists under config_dict['path'] on all nodes.
      - update_test_result() to finalize test status.
    """

    globals.error_list = []
    print('Testcase install rocblas')

    if config_dict['nfs_install'] is True:
        hdl = shdl
    else:
        hdl = phdl

    path = config_dict['path']
    git_install_path = config_dict['git_install_path']
    rocm_version = config_dict['rocm_version']
    hdl.exec(f'sudo rm -rf {git_install_path}/rocBLAS')

    git_url = config_dict['git_url']
    out_dict = hdl.exec('sudo apt update -y', timeout=200)
    out_dict = hdl.exec('sudo apt install -y libgtest-dev', timeout=200)
    out_dict = hdl.exec('sudo apt install -y cmake', timeout=200)
    out_dict = hdl.exec('sudo apt install -y gfortran', timeout=200)
    out_dict = hdl.exec('sudo apt install -y hipblaslt-dev', timeout=200)

    time.sleep(2)
    # out_dict = phdl.exec('git init')
    out_dict = hdl.exec(f'cd {git_install_path};git clone {git_url}', timeout=100)
    time.sleep(2)

    out_dict = hdl.exec(f'cd {git_install_path}/rocBLAS;git checkout rocm-{rocm_version}', timeout=60)

    time.sleep(2)
    # out_dict = hdl.exec(f'cd {git_install_path}/rocBLAS;./install.sh --clients-only --library-path /opt/rocm-{rocm_version}', timeout=700 )
    out_dict = phdl.exec(
        f'cd {git_install_path}/rocBLAS;sudo ./install.sh -dc --clients-only --library-path /opt/rocm-{rocm_version}',
        timeout=700,
    )
    out_dict = phdl.exec(f'ls -l {path}')
    for node in out_dict.keys():
        if not re.search('rocblas-bench', out_dict[node], re.I):
            fail_test(f'rocblas installation failed, rocblas-bench not found on node {node}')
    update_test_result()
