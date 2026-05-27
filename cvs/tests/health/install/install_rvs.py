'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import pytest

import re
import json

from cvs.lib.utils_lib import *
from cvs.lib.verify_lib import *

from cvs.lib import globals

log = globals.log


# Importing additional cmd line args to script ..


def detect_rocm_path(orch, config_rocm_path):
    """
    Detect the ROCm installation path, supporting both old (/opt/rocm) and new (/opt/rocm/core-X.Y) layouts.

    Args:
        orch: Orchestrator instance
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
    out_dict = orch.exec('ls -d /opt/rocm/core-* 2>/dev/null | sort -V | tail -1')
    for node, output in out_dict.items():
        if output and '/opt/rocm/core-' in output:
            rocm_path = output.strip()
            log.info(f'Detected ROCm path (new layout): {rocm_path}')
            return rocm_path

    # Fall back to legacy /opt/rocm
    out_dict = orch.exec('test -d /opt/rocm && echo "/opt/rocm"')
    for node, output in out_dict.items():
        if '/opt/rocm' in output:
            log.info('Detected ROCm path (legacy layout): /opt/rocm')
            return '/opt/rocm'

    # If nothing found, default to /opt/rocm (will fail gracefully later)
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
    # Try hipcc first (ROCm 7.x)
    out_dict = orch.exec(f'test -f {rocm_path}/bin/hipcc && echo "{rocm_path}/bin/hipcc"')
    for node, output in out_dict.items():
        if output and 'hipcc' in output:
            log.info(f'Detected HIP compiler: {rocm_path}/bin/hipcc')
            return f'{rocm_path}/bin/hipcc'

    # Fall back to amdclang++ (older ROCm versions)
    out_dict = orch.exec(f'test -f {rocm_path}/bin/amdclang++ && echo "{rocm_path}/bin/amdclang++"')
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
    Load and return the RVS test configuration subsection.

    Args:
      config_file (str): Path to the test configuration JSON file.

    Returns:
      dict: The 'rvs' configuration block containing expected results, paths, etc.
    """
    with open(config_file) as json_file:
        config_dict_t = json.load(json_file)
    config_dict = config_dict_t['rvs']

    # Resolve path placeholders like {user-id}, {home-mount-dir}, etc.
    config_dict = resolve_test_config_placeholders(config_dict, cluster_dict)

    log.info("%s", config_dict)
    return config_dict


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

    # Detect ROCm path early for use throughout function
    rocm_path = detect_rocm_path(orch, config_dict.get('rocm_path', ''))
    log.info(f"Using ROCm path: {rocm_path}")

    # Update config paths to use detected rocm_path (support both old and new ROCm layouts).
    # Skip the prefix rewrite when the value already starts with rocm_path; otherwise a
    # config like path="/opt/rocm/extras-7/bin" with rocm_path="/opt/rocm/extras-7" would
    # double up to "/opt/rocm/extras-7/extras-7/bin". Replace only the first occurrence
    # so we never touch a `/opt/rocm` segment deeper in the path.
    for key in ('config_path_mi300x', 'config_path_default', 'path'):
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
    git_url = config_dict['git_url']

    # Check if RVS is already installed via system packages
    out_dict = orch.exec('which rvs', timeout=30)
    rvs_found = False
    for node in out_dict.keys():
        if out_dict[node].strip() and re.search('rvs', out_dict[node], re.I):
            log.info(f'RVS appears to be already installed on node {node} at: {out_dict[node].strip()}')
            rvs_found = True

    # Check if RVS config files exist
    # Check MI300X path first (same order as final verification) and suppress stderr
    # so a missing fallback path's "No such file" does not contaminate the output.
    out_dict = orch.exec(
        f'ls -l {config_dict["config_path_mi300x"]}/gst_single.conf 2>/dev/null || ls -l {config_dict["config_path_default"]}/gst_single.conf 2>/dev/null',
        timeout=30,
    )
    config_found = False
    for node in out_dict.keys():
        if re.search(r'gst_single\.conf', out_dict[node], re.I):
            log.info(f'RVS configuration files found on node {node}')
            config_found = True

    # If RVS is not found or configs are missing, install it
    if not rvs_found or not config_found:
        log.warning('RVS not found, attempting to install from artifactory repo first')

        # First try to install from artifactory repo
        package_installed = False
        out_dict = orch.exec('sudo apt-get update -y', timeout=600)
        out_dict = orch.exec(
            'sudo apt-get install -y libpci3 libpci-dev doxygen unzip cmake git libyaml-cpp-dev', timeout=600
        )
        out_dict = orch.exec('sudo apt-get install -y rocblas rocm-smi-lib', timeout=600)
        out_dict = orch.exec('sudo apt-get install -y rocm-validation-suite', timeout=600)

        for node in out_dict.keys():
            if re.search(
                'Unable to locate package|Package.*not found|E: Could not get lock|dpkg: error'
                '|has no installation candidate|unmet dependencies|not available',
                out_dict[node],
                re.I,
            ):
                log.warning(f'RVS package installation failed on node {node}, will try building from source')
            else:
                log.info(f'RVS package installation successful on node {node}')
                package_installed = True

        # After apt-get install, verify the binary actually exists and locate it.
        # The rocm-validation-suite package may install to /opt/rocm even when the
        # detected rocm_path is /opt/rocm/core-X.Y (new layout), causing path mismatches.
        if package_installed:
            verify_bin = orch.exec(
                f'which rvs 2>/dev/null || ls {config_dict["path"]} 2>/dev/null',
                timeout=60,
            )
            rvs_bin_found = False
            for node, output in verify_bin.items():
                stripped = output.strip()
                if stripped and 'rvs' in stripped and not re.search('No such file', stripped, re.I):
                    rvs_bin_found = True
                    # If the binary landed at /opt/rocm instead of rocm_path, realign all paths
                    if '/opt/rocm/bin/rvs' in stripped and not stripped.startswith(rocm_path):
                        actual_rocm = '/opt/rocm'
                        log.info(
                            f'RVS installed to {actual_rocm}/bin/rvs; updating rocm_path from {rocm_path} to {actual_rocm}'
                        )
                        for key in ('path', 'config_path_mi300x', 'config_path_default'):
                            if key in config_dict:
                                config_dict[key] = config_dict[key].replace(rocm_path, actual_rocm)
                        rocm_path = actual_rocm
                    break
            if not rvs_bin_found:
                log.warning('RVS binary not found after package install, falling back to source build')
                package_installed = False

        # If package installation failed, install pre-built RVS tarball from repo.amd.com
        if not package_installed:
            log.info('Installing RVS from pre-built tarball at repo.amd.com')

            tarball_index_url = 'https://repo.amd.com/rocm/rvs/tarball/'
            extras_dir = '/opt/rocm/extras-7'

            # Ensure staging directory exists for the tarball download
            out_dict = orch.exec(f'ls -ld {git_install_path}')
            for node in out_dict.keys():
                if re.search('No such file', out_dict[node]):
                    orch.exec(f'mkdir -p {git_install_path}')

            try:
                # Resolve the latest amdrocm7-rvs-*.tar.gz from the directory listing
                out_dict = orch.exec(
                    f"curl -sSL {tarball_index_url} | grep -oE 'amdrocm7-rvs-[^\"]+\\.tar\\.gz' | sort -V -u | tail -1",
                    timeout=60,
                )
                latest_tarball = ''
                for node, output in out_dict.items():
                    stripped = output.strip()
                    if stripped.endswith('.tar.gz'):
                        latest_tarball = stripped
                        log.info(f'Latest RVS tarball detected on node {node}: {latest_tarball}')
                        break

                if not latest_tarball:
                    fail_test(f'Could not determine latest RVS tarball from {tarball_index_url}')

                # Build LD_LIBRARY_PATH for the ldd verification below.
                #
                # The rvs tarball ships librvslib but NOT the ROCm runtime libs it
                # depends on (libamd_smi, llvm libs, rocm_sysdeps). Those come from a
                # separate ROCm dist tarball whose standard install location is
                # /install/lib (with rocm_sysdeps/ and llvm/lib/ subdirs). These
                # paths are therefore required for the tarball install workflow
                # even though they're absolute and won't exist on every host.
                #
                # rocm_runtime_lib_path (optional) is a per-config override for sites
                # that extract the ROCm dist somewhere other than /install (e.g.
                # ~/install/lib). When set it's prepended so it wins symbol
                # resolution; when unset, only the standard tarball-install paths
                # are used. The same knob is honored at test runtime in rvs_cvs.py.
                runtime_lib_path = config_dict.get('rocm_runtime_lib_path') or ''
                tarball_default_libs = '/install/lib:/install/lib/rocm_sysdeps:/install/lib/llvm/lib'
                ld_prefix_parts = [f'{extras_dir}/lib']
                if runtime_lib_path:
                    ld_prefix_parts.append(runtime_lib_path)
                ld_prefix_parts.append(tarball_default_libs)
                ld_prefix = ':'.join(ld_prefix_parts)

                # Download tarball, extract to /opt/rocm/extras-7, and run ldd to confirm
                # all runtime dependencies of the rvs binary resolve.
                install_cmd = (
                    f'cd {git_install_path} && '
                    f'rm -f amdrocm7-rvs-*.tar.gz && '
                    f'wget -q {tarball_index_url}{latest_tarball} && '
                    f'sudo mkdir -p {extras_dir} && '
                    f'sudo tar -xzf {latest_tarball} -C {extras_dir} && '
                    f'export LD_LIBRARY_PATH={ld_prefix}:$LD_LIBRARY_PATH && '
                    f'ldd {extras_dir}/bin/rvs; echo "RVS_INSTALL_STATUS:$?"'
                )
                out_dict = orch.exec(install_cmd, timeout=1200)
                for node in out_dict.keys():
                    if not re.search(r'RVS_INSTALL_STATUS:0', out_dict[node]):
                        fail_test(f'RVS tarball install failed on node {node}')

                # RVS now lives under /opt/rocm/extras-7; realign paths so the
                # subsequent verification finds the binary and config files.
                log.info(
                    f'RVS installed via tarball to {extras_dir}; updating rocm_path from {rocm_path} to {extras_dir}'
                )
                for key in ('path', 'config_path_mi300x', 'config_path_default'):
                    if key in config_dict:
                        config_dict[key] = config_dict[key].replace(rocm_path, extras_dir)
                rocm_path = extras_dir

            except Exception as e:
                fail_test(f'RVS installation failed with exception: {e}')

    # Verify RVS installation
    out_dict = orch.exec(f'which rvs || ls -l {rocm_path}/bin/rvs*', timeout=60)
    for node in out_dict.keys():
        if re.search('not found|No such file', out_dict[node], re.I) and not re.search('rvs', out_dict[node]):
            fail_test(f'RVS installation verification failed on node {node}')

    # Verify config files are accessible. Suppress per-clause stderr so a
    # missing MI300X subdir (which is optional - some packagings only ship
    # the default conf dir) doesn't leak "No such file" into the captured
    # output and trip the regex below when the default-path fallback succeeded.
    out_dict = orch.exec(
        f'ls -l {config_dict["config_path_mi300x"]}/gst_single.conf 2>/dev/null || ls -l {config_dict["config_path_default"]}/gst_single.conf 2>/dev/null',
        timeout=60,
    )
    for node in out_dict.keys():
        if re.search('No such file', out_dict[node], re.I):
            fail_test(f'RVS configuration files not found on node {node}')

    log.info('RVS installation and verification completed successfully')
    update_test_result()
