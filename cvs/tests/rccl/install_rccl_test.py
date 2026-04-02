'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import pytest
import json

from cvs.lib import globals
from cvs.lib.utils_lib import *
from cvs.lib import rccl_lib
from cvs.lib.parallel_ssh_lib import Pssh

log = globals.log


# Pytest fixtures - only the required ones
@pytest.fixture(scope="session")
def cluster_dict(pytestconfig):
    """Load cluster configuration from CLI argument."""
    cluster_file = pytestconfig.getoption("cluster_file")
    with open(cluster_file) as json_file:
        cluster_dict = json.load(json_file)

    # Resolve path placeholders like {user-id} in cluster config
    cluster_dict = resolve_cluster_config_placeholders(cluster_dict)
    return cluster_dict


@pytest.fixture(scope="session")
def config_dict(pytestconfig):
    """Load RCCL configuration from CLI argument."""
    config_file = pytestconfig.getoption("config_file")
    with open(config_file) as json_file:
        config_dict = json.load(json_file)
    return config_dict


@pytest.fixture(scope="session")
def shdl(cluster_dict):
    """Create single SSH handle for head node operations."""
    all_nodes = list(cluster_dict['node_dict'].keys())
    head_node = all_nodes[0]
    return Pssh(log, [head_node], user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'])


def test_build_rccl_tests(shdl, cluster_dict, config_dict):
    """Build rccl-tests on all nodes from the ROCm/rocm-systems monorepo.

    Sparse-clones ``projects/rccl-tests`` and builds with ``make MPI=0``.
    Skips the build if the binary already exists on all nodes.
    """
    globals.error_list = []

    log.info("=" * 60)
    log.info("RCCL-TESTS: Build")
    log.info("=" * 60)

    rccl_cfg = config_dict.get("rccl", {})
    install_path = rccl_cfg.get("rccl_tests_dir", "/tmp/rccl-tests")
    repo_url = rccl_cfg.get("rccl_tests_repo", "https://github.com/ROCm/rocm-systems.git")
    repo_branch = rccl_cfg.get("rccl_tests_branch", "develop")

    # MPI configuration - derive from cluster size and existing config fields
    all_nodes = list(cluster_dict['node_dict'].keys())
    num_nodes = len(all_nodes)
    with_mpi = num_nodes > 1  # Automatically enable MPI for multi-node setups
    mpi_home = rccl_cfg.get("mpi_path_var")  # Use existing mpi_path_var
    # install.sh --rccl_home / --rocm_home must point at the ROCm/RCCL *install* (e.g. /opt/rocm), not rccl_dir (work tree).
    rocm_path = (rccl_cfg.get("rocm_path_var") or "/opt/rocm").rstrip("/")
    rccl_path = (rccl_cfg.get("rccl_path_var") or rocm_path).rstrip("/")

    log.info(f"[rccl-tests] Detected {num_nodes} node(s), MPI support: {'enabled' if with_mpi else 'disabled'}")

    try:
        # Check if we need to install MPI
        mpi_exists = False
        if with_mpi and mpi_home:
            # Check if MPI installation exists
            check_mpi = shdl.exec(f"test -f {mpi_home}/bin/mpirun && echo EXISTS", timeout=10, print_console=False)
            mpi_exists = any("EXISTS" in v for v in check_mpi.values())

        # Install MPI if needed for multi-node setup
        if with_mpi and not mpi_exists:
            log.info("[rccl-tests] Installing MPI dependencies (not found at configured path)...")
            # Install MPI directly at the location specified in rccl.json
            mpi_install_path = mpi_home  # Use /mnt/scratch1/amd/ichristo/mpi directly
            mpi_paths = rccl_lib.install_mpi(
                shdl=shdl,
                install_path=mpi_install_path,
                rocm_path=rocm_path,
                ucx_version=rccl_cfg.get("ucx_version", "1.18.0"),
                ompi_version=rccl_cfg.get("ompi_version", "5.0.7"),
            )
            mpi_home = mpi_paths["ompi_path"]
            log.info(f"[rccl-tests] MPI installed at: {mpi_home}")
        elif with_mpi and mpi_exists:
            log.info(f"[rccl-tests] Using existing MPI installation at: {mpi_home}")

        install_dir = rccl_lib.install_rccl_tests(
            shdl=shdl,
            install_path=install_path,
            repo_url=repo_url,
            repo_branch=repo_branch,
            with_mpi=with_mpi,
            mpi_home=mpi_home,
            rccl_home=rccl_path,
            rocm_home=rocm_path,
        )

        mpi_status = "with MPI" if with_mpi else "without MPI"
        log.info(f"[rccl-tests] Installation successful on all {num_nodes} node(s) {mpi_status}: {install_dir}")
        update_test_result()
    except Exception as e:
        fail_test(f"RCCL-tests installation failed: {e}")
