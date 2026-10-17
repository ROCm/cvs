'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import json

import pytest

from cvs.lib.utils_lib import *
from cvs.lib.verify_lib import *
from cvs.lib import globals

log = globals.log


def _exec_or_fail(hdl, cmd, timeout=None):
    out = hdl.exec(cmd, timeout=timeout, detailed=True)
    for node, res in out.items():
        if res.get('exit_code', -1) != 0:
            fail_test(f"Command failed (exit {res['exit_code']}) on {node}: {cmd!r}\n{res['output']}")
    return out


@pytest.fixture(scope="module")
def config_dict(pytestconfig):
    """Load and return the OMPI-specific configuration dictionary for the test module."""
    cluster_file = pytestconfig.getoption("cluster_file")
    config_file = pytestconfig.getoption("config_file")

    with open(cluster_file) as f:
        cluster_dict = json.load(f)
    cluster_dict = resolve_cluster_config_placeholders(cluster_dict)

    with open(config_file) as f:
        config_dict_t = json.load(f)

    ompi_cfg = config_dict_t['ompi']
    ompi_cfg = resolve_test_config_placeholders(ompi_cfg, cluster_dict)
    log.info("%s", ompi_cfg)
    return ompi_cfg


def detect_rocm_path(orch, config_rocm_path):
    """
    Detect the ROCm installation path, supporting both old (/opt/rocm) and new (/opt/rocm/core-X.Y) layouts.

    Args:
        orch: Orchestrator instance
        config_rocm_path (str): Configured ROCm path from config file (empty string for auto-detect)

    Returns:
        str: Detected ROCm path
    """
    if config_rocm_path and config_rocm_path != '<changeme>':
        log.info(f'Using configured ROCm path: {config_rocm_path}')
        return config_rocm_path

    log.info('Auto-detecting ROCm path...')

    out_dict = orch.exec('ls -d /opt/rocm/core-* 2>/dev/null | sort -V | tail -1')
    for output in out_dict.values():
        if output and '/opt/rocm/core-' in output:
            rocm_path = output.strip()
            log.info(f'Detected ROCm path (new layout): {rocm_path}')
            return rocm_path

    out_dict = orch.exec('test -d /opt/rocm', detailed=True)
    for info in out_dict.values():
        if info.get('exit_code') == 0:
            log.info('Detected ROCm path (legacy layout): /opt/rocm')
            return '/opt/rocm'

    log.warning('Could not detect ROCm path, defaulting to /opt/rocm')
    return '/opt/rocm'


def _install_ucx(hdl, config_dict):
    ucx_url = config_dict["ucx_url"]
    install_dir = config_dict["install_dir"].rstrip('/')
    tarball_name = ucx_url.split('/')[-1]
    ucx_src = tarball_name.rstrip('.tar.gz')
    _exec_or_fail(
        hdl,
        f"mkdir -p {install_dir}/{ucx_src} && cd {install_dir}/{ucx_src} && wget -q {ucx_url} && tar -zxf {tarball_name} --strip-components=1",
        timeout=600,
    )
    _exec_or_fail(hdl, f"mkdir -p {install_dir}/{ucx_src}/build")
    _exec_or_fail(hdl, f"mkdir -p {install_dir}/{ucx_src}/install")
    rocm_path = detect_rocm_path(hdl, config_dict.get('rocm_dir', '<changeme>'))
    log.info(f'Using ROCm path for ucx configure: {rocm_path}')
    _exec_or_fail(
        hdl,
        f'cd {install_dir}/{ucx_src}/build; ../configure --prefix={install_dir}/{ucx_src}/install --with-rocm={rocm_path}',
        timeout=500,
    )
    _exec_or_fail(hdl, f'cd {install_dir}/{ucx_src}/build; make -j $(nproc)', timeout=500)
    _exec_or_fail(hdl, f'cd {install_dir}/{ucx_src}/build; make install', timeout=500)
    return ucx_src


def test_install_ompi(orch, config_dict):
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

    # NFS install: build once on the head node (shared mount covers all nodes).
    # Local install: build in parallel across every node via orch.all.
    if nfs_install == "True":
        hdl = orch.head
    else:
        hdl = orch.all

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

    _exec_or_fail(hdl, f'cd {install_dir} && wget -q -O {tarball_name} {ompi_url}', timeout=600)
    _exec_or_fail(
        hdl,
        f"cd {install_dir} && "
        f"rm -rf {ompi_src_dir} && "
        f"mkdir -p {ompi_src_dir} && "
        f"tar -zxf {tarball_name} -C {ompi_src_dir} --strip-components=1 && "
        f"mkdir -p {ompi_build_dir}",
        timeout=600,
    )
    _exec_or_fail(hdl, f"cd {ompi_build_dir} && {configure_cmd}", timeout=900)
    _exec_or_fail(hdl, f"cd {ompi_build_dir} && make -j $(nproc)", timeout=3600)
    _exec_or_fail(hdl, f"cd {ompi_build_dir} && make install", timeout=1800)
    update_test_result()
