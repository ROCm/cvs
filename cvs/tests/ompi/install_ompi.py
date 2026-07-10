'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import pytest

import json


from cvs.lib.parallel_ssh_lib import *
from cvs.lib.utils_lib import *
from cvs.lib.verify_lib import *

from cvs.lib import globals

log = globals.log


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
    Load and return the OMPI-specific configuration dictionary for the test module.
    """
    with open(config_file) as json_file:
        config_dict_t = json.load(json_file)

    ompi_cfg = config_dict_t['ompi']

    # Resolve path placeholders like {user-id}, {home-mount-dir}, etc.
    ompi_cfg = resolve_test_config_placeholders(ompi_cfg, cluster_dict)
    log.info("%s", ompi_cfg)
    return ompi_cfg


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
    phdl = Pssh(log, node_list, user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'], env_vars=env_vars)
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
    shdl = Pssh(log, [head_node], user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'], env_vars=env_vars)
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


def _install_ucx(hdl, config_dict):
    ucx_url = config_dict["ucx_url"]
    install_dir = config_dict["install_dir"].rstrip('/')
    tarball_name = ucx_url.split('/')[-1]
    ucx_src = tarball_name.rstrip('.tar.gz')
    hdl.exec(
        f"mkdir -p {install_dir}/{ucx_src} && cd {install_dir}/{ucx_src} && wget -q {ucx_url} && tar -zxf {tarball_name} --strip-components=1",
        timeout=600,
    )
    hdl.exec(f"mkdir -p {install_dir}/{ucx_src}/build")
    hdl.exec(f"mkdir -p {install_dir}/{ucx_src}/install")
    rocm_path = detect_rocm_path(hdl, config_dict.get('rocm_dir', '<changeme>'))
    log.info(f'Using ROCm path for ucx configure: {rocm_path}')
    hdl.exec(
        f'cd {install_dir}/{ucx_src}/build; ../configure --prefix={install_dir}/{ucx_src}/install --with-rocm={rocm_path}',
        timeout=500,
    )
    hdl.exec(f'cd {install_dir}/{ucx_src}/build; make -j $(nproc)', timeout=500)
    hdl.exec(f'cd {install_dir}/{ucx_src}/build; make install', timeout=500)
    return ucx_src


def test_install_ompi(phdl, shdl, config_dict):
    """
    Build and install OpenMPI (OMPI) according to ompi_config.json,
    """
    globals.error_list = []
    nfs_install = config_dict["nfs_install"]
    install_dir = config_dict["install_dir"].rstrip('/')
    ompi_url = config_dict["ompi_url"]
    ucx_install = config_dict["ucx_install"]

    tarball_name = ompi_url.split('/')[-1]
    ompi_src_dir = f"{install_dir}/ompi-5.0.10"
    ompi_build_dir = f"{ompi_src_dir}/build"
    ompi_install_prefix = f"{ompi_src_dir}/install"

    # if NFS install is not true we have to install ompi on every
    # node in the cluster. Therefore, we decide on the handle first
    # if NFS then we will use the shdl otherwise phdl
    hdl = ''
    if nfs_install == "True":
        hdl = shdl
    else:
        hdl = phdl

    # if ucx_install is true then install ucx first
    if ucx_install == "True":
        ucx_src = _install_ucx(hdl, config_dict)

    # Build ompi configure options from config_dict
    cfg_opts = [
        f"--prefix={ompi_install_prefix}",
        "--enable-orterun-prefix-by-default",
        "--enable-mca-no-build=btl-uct",
    ]

    if config_dict.get("disable_oshmem", False):
        cfg_opts.append("--disable-oshmem")

    if config_dict.get("disable_mpi_fortran", False):
        cfg_opts.append("--disable-mpi-fortran")

    val = config_dict.get("hwloc")
    if val.strip().lower() == "internal":
        cfg_opts.append("--with-hwloc=internal")

    val = config_dict.get("libevent")
    if val.strip().lower() == "internal":
        cfg_opts.append("--with-libevent=internal")

    val = config_dict.get("pmix")
    if val.strip().lower() == "internal":
        cfg_opts.append("--with-pmix=internal")

    val = config_dict.get("prrte")
    if val.strip().lower() == "internal":
        cfg_opts.append("--with-prrte=internal")

    # Configure with UCX is user requested
    if ucx_install == "True":
        cfg_opts.append(f"--with-ucx={install_dir}/{ucx_src}/install")

    configure_cmd = f"../configure {' '.join(cfg_opts)}"

    log.info("OMPI install_dir: %s", install_dir)
    log.info("OMPI source dir: %s", ompi_src_dir)
    log.info("OMPI build dir: %s", ompi_build_dir)
    log.info("OMPI install prefix: %s", ompi_install_prefix)
    log.info("OMPI configure cmd: %s", configure_cmd)

    hdl.exec(f'cd {install_dir} && wget -q -O {tarball_name} {ompi_url}', timeout=600)
    hdl.exec(
        f"cd {install_dir} && "
        f"rm -rf {ompi_src_dir} && "
        f"mkdir -p {ompi_src_dir} && "
        f"tar -zxf {tarball_name} -C {ompi_src_dir} --strip-components=1 && "
        f"mkdir -p {ompi_build_dir}",
        timeout=600,
    )
    hdl.exec(f"cd {ompi_build_dir} && {configure_cmd}", timeout=900)
    hdl.exec(f"cd {ompi_build_dir} && make -j $(nproc)", timeout=3600)
    hdl.exec(f"cd {ompi_build_dir} && make install", timeout=1800)
    update_test_result()
