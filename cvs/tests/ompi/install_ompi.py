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


def _run_or_fail(shdl, cmd, timeout=None, desc=None):
    """
    Helper to run a command on the head node via shdl and record a failure with
    fail_test() if it fails on any host.

    shdl.exec is expected to be called with detailed=True so that the return
    value is {host: {"output": <str>, "exit_code": <int>}}.
    """
    if desc:
        log.info("Running: %s", desc)
    else:
        log.info("Running cmd: %s", cmd)

    out = shdl.exec(cmd, timeout=timeout, detailed=True, print_console=True)

    # shdl is Pssh over a single host list [head_node], but we still iterate
    for host, result in out.items():
        # result should be {'output': str, 'exit_code': int}
        exit_code = result.get("exit_code", -1)
        output = result.get("output", "")

        log.info("Host %s exit_code=%s", host, exit_code)
        if exit_code != 0:
            msg = (
                f"Command failed on host {host}: {cmd}\n"
                f"Description: {desc or 'N/A'}\n"
                f"Exit code: {exit_code}\n"
                f"Output:\n{output}"
            )
            log.error("%s", msg)
            fail_test(msg)  # record the failure but do not raise here

    return out


def test_install_ompi(phdl, shdl, config_dict):
    """
    Build and install OpenMPI (OMPI) on the head node according to ompi_config.json,
    then verify on all nodes via phdl.
    """
    globals.error_list = []

    # Extract config
    install_dir = config_dict["install_dir"].rstrip('/')  # e.g. /home/ahskabir
    ompi_url = config_dict["ompi_url"]
    tarball_name = ompi_url.split('/')[-1]  # openmpi-5.0.10.tar.gz
    ompi_src_dir = f"{install_dir}/ompi-5.0.10"
    ompi_build_dir = f"{ompi_src_dir}/build"
    ompi_install_prefix = f"{ompi_src_dir}/install"

    # Build configure options from config_dict
    cfg_opts = [
        f"--prefix={ompi_install_prefix}",
        "--enable-orterun-prefix-by-default",
        "--enable-mca-no-build=btl-uct",
    ]

    def is_true(val):
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.strip().lower() == "true"
        return False

    if is_true(config_dict.get("disable_oshmem", False)):
        cfg_opts.append("--disable-oshmem")

    if is_true(config_dict.get("disable_mpi_fortran", False)):
        cfg_opts.append("--disable-mpi-fortran")

    def add_internal_opt(key, name):
        val = config_dict.get(key)
        if isinstance(val, str) and val.strip().lower() == "internal":
            cfg_opts.append(f"--with-{name}=internal")

    add_internal_opt("hwloc", "hwloc")
    add_internal_opt("libevent", "libevent")
    add_internal_opt("pmix", "pmix")
    add_internal_opt("prrte", "prrte")

    configure_cmd = f"../configure {' '.join(cfg_opts)}"

    log.info("OMPI install_dir: %s", install_dir)
    log.info("OMPI source dir: %s", ompi_src_dir)
    log.info("OMPI build dir: %s", ompi_build_dir)
    log.info("OMPI install prefix: %s", ompi_install_prefix)
    log.info("OMPI configure cmd: %s", configure_cmd)

    # 1. Prepare directory and fetch tarball (head node)
    _run_or_fail(
        shdl,
        f"mkdir -p {install_dir}",
        desc=f"Create OMPI base install_dir {install_dir}",
    )

    _run_or_fail(
        shdl,
        f"cd {install_dir} && wget -O {tarball_name} {ompi_url}",
        timeout=600,
        desc="Download OpenMPI tarball",
    )

    # 2. Extract sources and create build dir (head node)
    _run_or_fail(
        shdl,
        f"cd {install_dir} && "
        f"rm -rf {ompi_src_dir} && "
        f"mkdir -p {ompi_src_dir} && "
        f"tar -zxf {tarball_name} -C {ompi_src_dir} --strip-components=1 && "
        f"mkdir -p {ompi_build_dir}",
        timeout=600,
        desc="Extract OMPI sources and create build directory",
    )

    # 3. Configure (head node)
    _run_or_fail(
        shdl,
        f"cd {ompi_build_dir} && {configure_cmd}",
        timeout=900,
        desc="Configure OpenMPI",
    )

    # 4. Build (head node)
    _run_or_fail(
        shdl,
        f"cd {ompi_build_dir} && make -j $(nproc)",
        timeout=3600,
        desc="Build OpenMPI",
    )

    # 5. Install (head node)
    _run_or_fail(
        shdl,
        f"cd {ompi_build_dir} && make install",
        timeout=1800,
        desc="Install OpenMPI",
    )

    # 6. Verification on ALL nodes via phdl
    #
    # Strategy:
    # - Ensure mpirun binary exists at the install prefix on the head node (already
    #   implied by successful install, but we can still check on each node).
    # - Typically, the install lives on a shared FS (e.g. NFS) mounted at the same
    #   path on all nodes. So each node should see:
    #     {ompi_install_prefix}/bin/mpirun
    #
    # Here we just test presence and executability on every node.
    verify_cmd = f"test -x {ompi_install_prefix}/bin/mpirun && {ompi_install_prefix}/bin/mpirun --version | head -n 1"

    log.info("Verifying OMPI installation on all nodes using phdl")
    out_dict = phdl.exec(verify_cmd, timeout=300, detailed=True, print_console=True)

    for node, result in out_dict.items():
        exit_code = result.get("exit_code", -1)
        output = result.get("output", "")

        log.info("Node %s OMPI verify exit_code=%s", node, exit_code)
        if exit_code != 0:
            msg = (
                f"OMPI verification failed on node {node}\n"
                f"Command: {verify_cmd}\n"
                f"Exit code: {exit_code}\n"
                f"Output:\n{output}"
            )
            log.error("%s", msg)
            fail_test(msg)

    # Final test result bookkeeping
    update_test_result()
