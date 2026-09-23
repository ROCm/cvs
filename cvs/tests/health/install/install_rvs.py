'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import json
import re
import shlex

import pytest

from cvs.core.scheduler import is_managed_compute
from cvs.lib import globals
from cvs.lib.utils_lib import *
from cvs.lib.verify_lib import *

log = globals.log

_RVS_PATH_KEYS = ('path', 'config_path_mi300x', 'config_path_default')
_TARBALL_INDEX_URL = 'https://repo.amd.com/rocm/rvs/tarball/'
_DEFAULT_EXTRAS_DIR = '/opt/rocm/extras-7'


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

    # Glob + pipeline need a shell; docker-exec does not spawn one.
    out_dict = orch.exec(
        "bash -c 'ls -d /opt/rocm/core-* 2>/dev/null | sort -V | tail -1'",
    )
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


def detect_hip_compiler(orch, rocm_path):
    """
    Detect the HIP compiler (hipcc or amdclang++) for the given ROCm installation.

    Args:
        orch: Orchestrator instance
        rocm_path (str): ROCm installation path

    Returns:
        str: Full path to the HIP compiler
    """
    out_dict = orch.exec(f'test -f {rocm_path}/bin/hipcc', detailed=True)
    for info in out_dict.values():
        if info.get('exit_code') == 0:
            log.info(f'Detected HIP compiler: {rocm_path}/bin/hipcc')
            return f'{rocm_path}/bin/hipcc'

    out_dict = orch.exec(f'test -f {rocm_path}/bin/amdclang++', detailed=True)
    for info in out_dict.values():
        if info.get('exit_code') == 0:
            log.info(f'Detected HIP compiler: {rocm_path}/bin/amdclang++')
            return f'{rocm_path}/bin/amdclang++'

    log.warning(f'Could not detect HIP compiler, defaulting to {rocm_path}/bin/hipcc')
    return f'{rocm_path}/bin/hipcc'


def _realign_install_paths(config_dict, old_root, new_root):
    for key in _RVS_PATH_KEYS:
        if key in config_dict and isinstance(config_dict[key], str):
            config_dict[key] = config_dict[key].replace(old_root, new_root)


def _gst_conf_ls_cmd(config_dict):
    mi300x = config_dict['config_path_mi300x']
    default = config_dict['config_path_default']
    inner = f'ls -l {mi300x}/gst_single.conf 2>/dev/null || ls -l {default}/gst_single.conf 2>/dev/null'
    return f'bash -c {shlex.quote(inner)}'


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

    cluster_dict = resolve_cluster_config_placeholders(cluster_dict)

    log.info("%s", cluster_dict)
    return cluster_dict


@pytest.fixture(scope="module")
def config_dict(config_file, cluster_dict):
    """
    Load and return the RVS test configuration subsection.

    Args:
      config_file (str): Path to the test configuration JSON file.

    Returns:
      dict: The 'rvs' configuration block containing expected results, paths, etc.
    """
    with open(config_file) as json_file:
        config_dict_t = json.load(json_file)
    config_dict = config_dict_t['rvs']

    config_dict = resolve_test_config_placeholders(config_dict, cluster_dict)

    log.info("%s", config_dict)
    return config_dict


def _try_apt_install_rvs(orch, config_dict, rocm_path, sudo_prefix):
    """Install rocm-validation-suite via apt. Returns (package_installed, rocm_path)."""
    package_installed = False
    orch.exec(f'{sudo_prefix}apt-get update -y', timeout=600)
    orch.exec(
        f'{sudo_prefix}apt-get install -y libpci3 libpci-dev doxygen unzip cmake git libyaml-cpp-dev',
        timeout=600,
    )
    orch.exec(f'{sudo_prefix}apt-get install -y rocblas rocm-smi-lib', timeout=600)
    out_dict = orch.exec(f'{sudo_prefix}apt-get install -y rocm-validation-suite', timeout=600)

    for node, output in out_dict.items():
        if re.search(
            'Unable to locate package|Package.*not found|E: Could not get lock|dpkg: error'
            '|has no installation candidate|unmet dependencies|not available',
            output,
            re.IGNORECASE,
        ):
            log.warning(f'RVS package installation failed on node {node}, will try tarball install')
        else:
            log.info(f'RVS package installation successful on node {node}')
            package_installed = True

    if not package_installed:
        return False, rocm_path

    verify_inner = f'which rvs 2>/dev/null || ls {config_dict["path"]} 2>/dev/null'
    verify_bin = orch.exec(f'bash -c {shlex.quote(verify_inner)}', timeout=60)
    rvs_bin_found = False
    for node, output in verify_bin.items():
        stripped = output.strip()
        if stripped and 'rvs' in stripped and not re.search('No such file', stripped, re.IGNORECASE):
            rvs_bin_found = True
            if '/opt/rocm/bin/rvs' in stripped and not stripped.startswith(rocm_path):
                actual_rocm = '/opt/rocm'
                log.info(
                    f'RVS installed to {actual_rocm}/bin/rvs; updating rocm_path from {rocm_path} to {actual_rocm}'
                )
                _realign_install_paths(config_dict, rocm_path, actual_rocm)
                rocm_path = actual_rocm
            break
    if not rvs_bin_found:
        log.warning('RVS binary not found after package install, falling back to tarball install')
        return False, rocm_path
    return True, rocm_path


def _install_rvs_tarball(orch, config_dict, rocm_path, git_install_path, sudo_prefix):
    """Extract the pre-built RVS tarball. Returns the extract root used as rocm_path."""
    log.info('Installing RVS from pre-built tarball at repo.amd.com')

    extras_dir = _DEFAULT_EXTRAS_DIR if sudo_prefix else git_install_path.rstrip('/')
    if not sudo_prefix:
        log.info(
            'No passwordless sudo; extracting RVS tarball under git_install_path=%s '
            '(set path/config_path_* in the health config to this tree for rvs_cvs)',
            extras_dir,
        )

    out_dict = orch.exec(f'test -d {git_install_path}', detailed=True)
    if any(info.get('exit_code') != 0 for info in out_dict.values()):
        orch.exec(f'mkdir -p {git_install_path}')

    list_inner = f"curl -sSL {_TARBALL_INDEX_URL} | grep -oE 'amdrocm7-rvs-[^\"]+\\.tar\\.gz' | sort -V -u | tail -1"
    out_dict = orch.exec(f'bash -c {shlex.quote(list_inner)}', timeout=60)
    latest_tarball = ''
    for node, output in out_dict.items():
        stripped = output.strip()
        if stripped.endswith('.tar.gz'):
            latest_tarball = stripped
            log.info(f'Latest RVS tarball detected on node {node}: {latest_tarball}')
            break

    if not latest_tarball:
        fail_test(f'Could not determine latest RVS tarball from {_TARBALL_INDEX_URL}')
        return rocm_path

    runtime_lib_path = config_dict.get('rocm_runtime_lib_path') or ''
    tarball_default_libs = '/install/lib:/install/lib/rocm_sysdeps:/install/lib/llvm/lib'
    ld_prefix_parts = [f'{extras_dir}/lib']
    if runtime_lib_path:
        ld_prefix_parts.append(runtime_lib_path)
    ld_prefix_parts.append(tarball_default_libs)
    ld_prefix = ':'.join(ld_prefix_parts)

    install_cmd = (
        f'cd {git_install_path} && '
        f'rm -f amdrocm7-rvs-*.tar.gz && '
        f'wget -q {_TARBALL_INDEX_URL}{latest_tarball} && '
        f'{sudo_prefix}mkdir -p {extras_dir} && '
        f'{sudo_prefix}tar -xzf {latest_tarball} -C {extras_dir} && '
        f'export LD_LIBRARY_PATH={ld_prefix}:$LD_LIBRARY_PATH && '
        f'ldd {extras_dir}/bin/rvs; echo "RVS_INSTALL_STATUS:$?"'
    )
    out_dict = orch.exec(f'bash -c {shlex.quote(install_cmd)}', timeout=1200)
    for node, output in out_dict.items():
        if not re.search(r'RVS_INSTALL_STATUS:0', output):
            fail_test(f'RVS tarball install failed on node {node}')

    log.info(f'RVS installed via tarball to {extras_dir}; updating rocm_path from {rocm_path} to {extras_dir}')
    _realign_install_paths(config_dict, rocm_path, extras_dir)
    return extras_dir


@pytest.mark.dependency(name="init")
def test_install_rvs(orch, config_dict):
    """
    Install/Build ROCmValidationSuite (RVS) and verify installation on all nodes.

    Steps:
      - Check if RVS is already installed via package manager
      - If not installed, clone RVS repo under git_install_path
      - Build and install RVS on all nodes
      - Verify RVS executable exists and configuration files are accessible

    Install commands and verification run across all nodes via ``orch.exec(...)``.

    Args:
      orch: Orchestrator instance.
      config_dict (dict): Includes:
        - git_install_path: directory to clone/build
        - git_url: repository URL
        - path: expected installation path for RVS binary
    """
    globals.error_list = []

    rocm_path = detect_rocm_path(orch, config_dict.get('rocm_path', ''))
    log.info(f"Using ROCm path: {rocm_path}")

    # Skip the prefix rewrite when the value already starts with rocm_path; otherwise a
    # config like path="/opt/rocm/extras-7/bin" with rocm_path="/opt/rocm/extras-7" would
    # double up to "/opt/rocm/extras-7/extras-7/bin". Replace only the first occurrence
    # so we never touch a `/opt/rocm` segment deeper in the path.
    for key in _RVS_PATH_KEYS:
        if key in config_dict and isinstance(config_dict[key], str):
            value = config_dict[key]
            if not value.startswith(rocm_path):
                value = value.replace('/opt/rocm', rocm_path, 1)
            value = value.replace('<changeme>', rocm_path)
            config_dict[key] = value

    log.info(
        f"Using config paths: MI300X={config_dict.get('config_path_mi300x')}, default={config_dict.get('config_path_default')}"
    )

    log.info('Testcase install RVS (ROCmValidationSuite)')
    git_install_path = config_dict['git_install_path']
    sudo_prefix = orch.sudo_prefix()

    out_dict = orch.exec('which rvs', timeout=30)
    rvs_found = False
    for node, output in out_dict.items():
        if output.strip() and re.search('rvs', output, re.IGNORECASE):
            log.info(f'RVS appears to be already installed on node {node} at: {output.strip()}')
            rvs_found = True

    out_dict = orch.exec(_gst_conf_ls_cmd(config_dict), timeout=30)
    config_found = False
    for node, output in out_dict.items():
        if re.search(r'gst_single\.conf', output, re.IGNORECASE):
            log.info(f'RVS configuration files found on node {node}')
            config_found = True

    if not rvs_found or not config_found:
        package_installed = False
        if is_managed_compute():
            log.warning('Managed Spur/Slurm step: skipping apt install, using RVS tarball')
        else:
            log.warning('RVS not found, attempting to install from artifactory repo first')
            package_installed, rocm_path = _try_apt_install_rvs(orch, config_dict, rocm_path, sudo_prefix)

        if not package_installed:
            rocm_path = _install_rvs_tarball(orch, config_dict, rocm_path, git_install_path, sudo_prefix)

    verify_inner = f'which rvs || ls -l {rocm_path}/bin/rvs*'
    out_dict = orch.exec(f'bash -c {shlex.quote(verify_inner)}', timeout=60)
    for node, output in out_dict.items():
        if re.search('not found|No such file', output, re.IGNORECASE) and not re.search('rvs', output):
            fail_test(f'RVS installation verification failed on node {node}')

    out_dict = orch.exec(_gst_conf_ls_cmd(config_dict), timeout=60)
    for node, output in out_dict.items():
        if re.search('No such file', output, re.IGNORECASE):
            fail_test(f'RVS configuration files not found on node {node}')

    log.info('RVS installation and verification completed successfully')
    update_test_result()
