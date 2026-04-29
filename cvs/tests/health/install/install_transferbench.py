'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import pytest

import re
import json


from cvs.lib.parallel_ssh_lib import *
from cvs.lib.transferbench_ssh_helpers import (
    assert_container_running,
    docker_opts_from_config,
    docker_spec_for_ssh_hosts,
    log_transferbench_docker_resolution,
)
from cvs.lib.utils_lib import *

from cvs.lib import globals

log = globals.log


# Importing additional cmd line args to script ..


def detect_rocm_path(phdl, config_rocm_path):
    """
    Detect the ROCm installation path, supporting both old (/opt/rocm) and new (/opt/rocm/core-X.Y) layouts.

    Args:
        phdl: Parallel SSH handle
        config_rocm_path (str): Configured ROCm path from config file (empty string for auto-detect)

    Returns:
        str: Detected ROCm path
    """
    # If rocm_path is explicitly configured, use it
    if config_rocm_path and config_rocm_path != '<changeme>':
        log.info(f'Using configured ROCm path: {config_rocm_path}')
        return config_rocm_path

    # Auto-detect ROCm path
    log.info('Auto-detecting ROCm path...')

    # Try new ROCm 7.x structure first (/opt/rocm/core-X.Y)
    out_dict = phdl.exec('ls -d /opt/rocm/core-* 2>/dev/null | sort -V | tail -1')
    for node, output in out_dict.items():
        if output and '/opt/rocm/core-' in output:
            rocm_path = output.strip()
            log.info(f'Detected ROCm path (new layout): {rocm_path}')
            return rocm_path

    # Fall back to legacy /opt/rocm
    out_dict = phdl.exec('test -d /opt/rocm && echo "/opt/rocm"')
    for node, output in out_dict.items():
        if '/opt/rocm' in output:
            log.info('Detected ROCm path (legacy layout): /opt/rocm')
            return '/opt/rocm'

    # If nothing found, default to /opt/rocm (will fail gracefully later)
    log.warning('Could not detect ROCm path, defaulting to /opt/rocm')
    return '/opt/rocm'


def detect_hip_compiler(phdl, rocm_path):
    """
    Detect the HIP compiler (hipcc or amdclang++) for the given ROCm installation.

    Args:
        phdl: Parallel SSH handle
        rocm_path (str): ROCm installation path

    Returns:
        str: Full path to the HIP compiler
    """
    # Try hipcc first (ROCm 7.x)
    out_dict = phdl.exec(f'test -f {rocm_path}/bin/hipcc && echo "{rocm_path}/bin/hipcc"')
    for node, output in out_dict.items():
        if output and 'hipcc' in output:
            log.info(f'Detected HIP compiler: {rocm_path}/bin/hipcc')
            return f'{rocm_path}/bin/hipcc'

    # Fall back to amdclang++ (older ROCm versions)
    out_dict = phdl.exec(f'test -f {rocm_path}/bin/amdclang++ && echo "{rocm_path}/bin/amdclang++"')
    for node, output in out_dict.items():
        if output and 'amdclang++' in output:
            log.info(f'Detected HIP compiler: {rocm_path}/bin/amdclang++')
            return f'{rocm_path}/bin/amdclang++'

    # Default to hipcc if nothing found
    log.warning(f'Could not detect HIP compiler, defaulting to {rocm_path}/bin/hipcc')
    return f'{rocm_path}/bin/hipcc'


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

    log.info("%s", cluster_dict)
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

    log.info("%s", config_dict)
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
    node_list = list(cluster_dict['node_dict'].keys())
    env_vars = cluster_dict.get("env_vars")
    head_node = node_list[0]
    shdl = Pssh(log, [head_node], user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'], env_vars=env_vars)
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
    log.info("%s", cluster_dict)
    env_vars = cluster_dict.get("env_vars")
    node_list = list(cluster_dict['node_dict'].keys())
    phdl = Pssh(log, node_list, user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'], env_vars=env_vars)
    return phdl


def test_install_transferbench(phdl, shdl, config_dict, cluster_dict):
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
      cluster_dict (dict): Cluster config (optional Docker keys: transferbench_use_docker,
        transferbench_container_id).
      config_dict (dict): Includes:
        - git_install_path: directory to clone/build
        - git_url: repository URL
        - use_docker / container_id (optional): when enabled, build runs inside that container on each node.

    """

    globals.error_list = []

    node_list = list(cluster_dict['node_dict'].keys())

    use_docker, container_spec = docker_opts_from_config(config_dict, cluster_dict, host_list=node_list)
    log_transferbench_docker_resolution(
        log,
        'install_transferbench',
        use_docker,
        container_spec,
        config_dict,
        cluster_dict,
        info_always=True,
    )

    if use_docker:
        assert_container_running(phdl, container_spec)
        env_vars = cluster_dict.get('env_vars')
        head_node = node_list[0]
        phdl = Pssh(
            log,
            node_list,
            user=cluster_dict['username'],
            pkey=cluster_dict['priv_key_file'],
            env_vars=env_vars,
            docker_container_id=docker_spec_for_ssh_hosts(container_spec, node_list),
        )
        shdl = Pssh(
            log,
            [head_node],
            user=cluster_dict['username'],
            pkey=cluster_dict['priv_key_file'],
            env_vars=env_vars,
            docker_container_id=docker_spec_for_ssh_hosts(container_spec, [head_node]),
        )

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
        if re.search('No such file', out_dict[node]):
            hdl.exec(f'mkdir -p {git_install_path}')

    out_dict = hdl.exec(f'rm -rf {git_install_path}/TransferBench')
    out_dict = hdl.exec(f'cd {git_install_path};git clone {git_url}', timeout=120)

    # Detect ROCm path and compiler
    rocm_path = detect_rocm_path(phdl, config_dict.get('rocm_path', ''))
    hip_compiler = detect_hip_compiler(phdl, rocm_path)

    # Build with explicit ROCM_PATH and HIPCC
    out_dict = hdl.exec(
        f'cd {git_install_path}/TransferBench;ROCM_PATH={rocm_path} HIPCC={hip_compiler} make', timeout=500
    )

    # Verify installation happened fine on all nodes
    out_dict = phdl.exec(f'ls -l {git_install_path}/TransferBench')
    for node in out_dict.keys():
        if not re.search('TransferBench', out_dict[node]):
            fail_test(f'Transfer bench installation failed on node {node}')
    update_test_result()
