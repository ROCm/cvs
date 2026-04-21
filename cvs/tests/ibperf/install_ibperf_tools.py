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
from cvs.lib.utils_lib import *
from cvs.lib.verify_lib import *

from cvs.lib import globals

log = globals.log

ib_bw_dict = {}
ib_lat_dict = {}

rccl_res_dict = {}


# Importing additional cmd line args to script ..
@pytest.fixture(scope="module")
def cluster_file(pytestconfig):
    """
    Return the path to the cluster configuration JSON file passed via pytest CLI.

    Expects:
      - pytest to be invoked with: --cluster_file <path>

    Args:
      pytestconfig: Built-in pytest config object used to access CLI options.

    Returns:
      str: Filesystem path to the cluster configuration file.
    """
    return pytestconfig.getoption("cluster_file")


@pytest.fixture(scope="module")
def config_file(pytestconfig):
    """
    Return the path to the test configuration JSON file passed via pytest CLI.

    Expects:
      - pytest to be invoked with: --config_file <path>

    Args:
      pytestconfig: Built-in pytest config object used to access CLI options.

    Returns:
      str: Filesystem path to the test configuration file.
    """
    return pytestconfig.getoption("config_file")


@pytest.fixture(scope="module")
def cluster_dict(cluster_file):
    """
    Load and expose full cluster configuration for the test module.

    Behavior:
      - Opens the JSON at cluster_file and parses it into a Python dict.
      - Logs the parsed dictionary for visibility and debugging.
      - Returns the entire cluster configuration (node list, credentials, etc.).

    Args:
      cluster_file (str): Path to the cluster configuration JSON.

    Returns:
      dict: Parsed cluster configuration. Expected keys include:
            - 'node_dict': Map of node name -> node metadata
            - 'username': SSH username
            - 'priv_key_file': Path to SSH private key
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
    Load and return the RCCL-specific configuration dictionary for the test module.

    Args:
      config_file (str): Path to a JSON config file provided by another fixture.

    Notes:
      - Expects the JSON file to contain a top-level key "ibperf".
      - Uses module scope so the config is parsed once per test module.
    """
    with open(config_file) as json_file:
        config_dict_t = json.load(json_file)
    config_dict = config_dict_t['ibperf']

    # Resolve path placeholders like {user-id}, {home-mount-dir}, etc.
    config_dict = resolve_test_config_placeholders(config_dict, cluster_dict)
    log.info("%s", config_dict)
    return config_dict


@pytest.fixture(scope="module")
def phdl(cluster_dict):
    """
    Build and return a parallel SSH handle (Pssh) for all cluster nodes.

    Args:
      cluster_dict (dict): Cluster metadata fixture containing:
        - node_dict: dict of node_name -> node_details
        - username: SSH username
        - priv_key_file: path to SSH private key

    Returns:
      Pssh: Handle configured for all nodes (for broadcast/parallel operations).

    Notes:
      - Prints the cluster_dict for quick debugging; consider replacing with log.debug.
      - Module-scoped so a single shared handle is used across all tests in the module.
      - nhdl_dict is currently unused; it can be removed unless used elsewhere.
      - Assumes Pssh(log, node_list, user=..., pkey=...) is available in scope.
    """
    log.info("%s", cluster_dict)
    env_vars = cluster_dict.get("env_vars")
    node_list = list(cluster_dict['node_dict'].keys())
    if len(node_list) < 2:
        raise ValueError('At least 2 nodes are required to run this test')
    if len(node_list) % 2 != 0:
        log.info(
            f'Odd number of nodes ({len(node_list)}) detected; popping last node from the cluster to make the count even'
        )
        node_list.pop()
    phdl = Pssh(log, node_list, user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'], env_vars=env_vars, wrapper=wrapper_for_cluster(cluster_dict))
    return phdl


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
    shdl = Pssh(log, [head_node], user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'], env_vars=env_vars, wrapper=wrapper_for_cluster(cluster_dict))
    return shdl


@pytest.fixture(scope="module")
def vpc_node_list(cluster_dict):
    """
    Collect and return a list of VPC IPs for all nodes in the cluster.

    Args:
      cluster_dict (dict): Cluster metadata fixture containing node_dict with vpc_ip per node.

    Returns:
      list[str]: List of VPC IP addresses in the cluster, ordered by node_dict iteration.

    Notes:
      - Iteration order depends on the insertion order of node_dict.
      - Consider validating that each node entry contains a 'vpc_ip' key.
    """
    vpc_node_list = []
    node_list = list(cluster_dict['node_dict'].keys())

    if len(node_list) < 2:
        raise ValueError('At least 2 nodes are required to run this test')

    if len(node_list) % 2 != 0:
        log.info(
            f'Odd number of nodes ({len(node_list)}) detected; popping last node from the cluster to make the count even'
        )
        node_list.pop()
    for node in node_list:
        vpc_node_list.append(cluster_dict['node_dict'][node]['vpc_ip'])
    return vpc_node_list


def detect_rocm_path(phdl, config_rocm_path):
    """
    Detect the ROCm installation path, supporting both old (/opt/rocm) and new (/opt/rocm/core-X.Y) layouts.
    Args:
        phdl: Parallel SSH handle
        config_rocm_path (str): Configured ROCm path from config file ('<changeme>' for auto-detect)
    Returns:
        str: Detected ROCm path
    """
    # If rocm_path is explicitly configured, validate and use it
    if config_rocm_path and config_rocm_path != '<changeme>':
        out_dict = phdl.exec(
            f'test -d {config_rocm_path}/lib && ls {config_rocm_path}/lib/libamdhip64.so* 2>/dev/null | head -1'
        )
        for node, output in out_dict.items():
            if output.strip() and 'libamdhip64.so' in output:
                log.info(f'Using configured ROCm path: {config_rocm_path} (validated)')
                return config_rocm_path
            else:
                log.warning(
                    f'Configured ROCm path {config_rocm_path} does not contain required libraries, will auto-detect'
                )

    # Auto-detect ROCm path
    log.info('Auto-detecting ROCm path...')

    # Try new ROCm 7.x structure first (/opt/rocm/core-X.Y)
    out_dict = phdl.exec('ls -d /opt/rocm/core-* 2>/dev/null | sort -V | tail -1')
    for node, output in out_dict.items():
        if output and '/opt/rocm/core-' in output:
            rocm_path = output.strip()
            validate_dict = phdl.exec(
                f'test -d {rocm_path}/lib && ls {rocm_path}/lib/libamdhip64.so* 2>/dev/null | head -1'
            )
            for _, lib_output in validate_dict.items():
                if lib_output.strip() and 'libamdhip64.so' in lib_output:
                    log.info(f'Detected ROCm path (new layout): {rocm_path}')
                    return rocm_path

    # Fall back to legacy /opt/rocm
    out_dict = phdl.exec('test -d /opt/rocm/lib && ls /opt/rocm/lib/libamdhip64.so* 2>/dev/null | head -1')
    for node, output in out_dict.items():
        if output.strip() and 'libamdhip64.so' in output:
            log.info('Detected ROCm path (legacy layout): /opt/rocm')
            return '/opt/rocm'

    log.warning('Could not detect ROCm path with required libraries, defaulting to /opt/rocm')
    return '/opt/rocm'


def test_install_ib_perf(phdl, shdl, config_dict):
    # We install on the first node using shdl handle
    # install standard rdma packages
    globals.error_list = []

    if re.search('true', config_dict['install_perf_package'], re.I):
        shdl.exec(f'mkdir -p {config_dict["install_dir"]}')
        phdl.exec('sudo apt update -y', timeout=200)
        phdl.exec('sudo apt install -y git build-essential autoconf automake libtool pkg-config', timeout=200)
        phdl.exec('sudo apt install -y libibverbs-dev librdmacm-dev ibverbs-providers rdma-core', timeout=200)
        phdl.exec('sudo apt install -y libibumad-dev')
        phdl.exec('sudo apt install -y libpci-dev')
        phdl.exec('sudo apt install -y numactl')
        shdl.exec(f'cd {config_dict["install_dir"]}; git clone https://github.com/linux-rdma/perftest')
        shdl.exec(f'cd {config_dict["install_dir"]}/perftest; ./autogen.sh', timeout=100)
        rocm_path = detect_rocm_path(shdl, config_dict.get('rocm_dir', '<changeme>'))
        log.info(f'Using ROCm path for perftest configure: {rocm_path}')
        shdl.exec(
            f'cd {config_dict["install_dir"]}/perftest; ./configure --prefix={config_dict["install_dir"]}/perftest --with-rocm={rocm_path} --enable-rocm',
            timeout=200,
        )
        shdl.exec(f'cd {config_dict["install_dir"]}/perftest; make', timeout=100)
        shdl.exec(f'cd {config_dict["install_dir"]}/perftest; make install', timeout=100)

        # Verify if the installation went fine ..
        out_dict = phdl.exec(f'{config_dict["install_dir"]}/perftest/ib_write_bw -h | grep -i rocm --color=never')
        for node in out_dict.keys():
            if not re.search('GPUDirect RDMA', out_dict[node], re.I):
                fail_test(
                    f'IB Perf package installation on node {node} failed, ib_write_bw not showing expected use_rocm output'
                )
    update_test_result()
