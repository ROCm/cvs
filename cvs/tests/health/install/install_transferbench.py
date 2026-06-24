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

    # Try new ROCm 7.x structure first (/opt/rocm/core-X.Y). Wrapped in
    # `bash -c` because we need glob expansion, stderr redirect and a pipeline
    # to pick the highest-versioned core dir; ContainerOrchestrator's
    # docker-exec transport does not spawn a shell on its own.
    out_dict = orch.exec(
        "bash -c 'ls -d /opt/rocm/core-* 2>/dev/null | sort -V | tail -1'",
    )
    for node, output in out_dict.items():
        if output and '/opt/rocm/core-' in output:
            rocm_path = output.strip()
            log.info(f'Detected ROCm path (new layout): {rocm_path}')
            return rocm_path

    # Fall back to legacy /opt/rocm. One logical operation per orch.exec:
    # use exit code rather than a `test ... && echo` shell short-circuit.
    out_dict = orch.exec('test -d /opt/rocm', detailed=True)
    for node, info in out_dict.items():
        if info.get('exit_code') == 0:
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
    # Try hipcc first (ROCm 7.x). One logical operation per orch.exec:
    # use the test(1) exit code rather than a `test ... && echo` chain.
    out_dict = orch.exec(f'test -f {rocm_path}/bin/hipcc', detailed=True)
    for node, info in out_dict.items():
        if info.get('exit_code') == 0:
            log.info(f'Detected HIP compiler: {rocm_path}/bin/hipcc')
            return f'{rocm_path}/bin/hipcc'

    # Fall back to amdclang++ (older ROCm versions).
    out_dict = orch.exec(f'test -f {rocm_path}/bin/amdclang++', detailed=True)
    for node, info in out_dict.items():
        if info.get('exit_code') == 0:
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


def test_install_transferbench(orch, config_dict):
    """
    Install/Build TransferBench and verify installation on all nodes.

    Steps:
      - Ensure git_install_path exists on all nodes.
      - Clone TransferBench repo under git_install_path on every node.
      - Checkout the configured git tag.
      - Build on all nodes using detected ROCM_PATH and HIPCC.
      - Verify the build artifact is present on each node.

    Clone, build, and verify all run on every node via ``orch.exec``.

    Args:
      orch: Orchestrator instance.
      config_dict (dict): Includes:
        - git_install_path: directory to clone/build
        - git_url: repository URL
        - git_tag: git tag to checkout after clone
    """

    globals.error_list = []

    log.info('Testcase install transferbench')
    git_install_path = config_dict['git_install_path']
    git_url = config_dict['git_url']

    out_dict = orch.exec(f'ls -ld {git_install_path}')
    for node in out_dict.keys():
        if re.search('No such file', out_dict[node]):
            orch.exec(f'mkdir -p {git_install_path}')

    out_dict = orch.exec(f'rm -rf {git_install_path}/TransferBench')
    # Clone with explicit destination, no cwd dependency.
    out_dict = orch.exec(
        f'git clone {git_url} {git_install_path}/TransferBench',
        timeout=120,
    )

    tb_src = f'{git_install_path}/TransferBench'
    git_tag = config_dict['git_tag']
    out_dict = orch.exec(
        f"bash -c 'cd {tb_src} && git checkout {git_tag}'",
        timeout=120,
    )

    # Detect ROCm path and compiler
    rocm_path = detect_rocm_path(orch, config_dict.get('rocm_path', ''))
    hip_compiler = detect_hip_compiler(orch, rocm_path)

    # Build with explicit ROCM_PATH and HIPCC. The TransferBench Makefile
    # uses cwd-relative paths so cwd MUST be the source tree; we wrap the
    # build in `bash -c` explicitly to make the cwd dependency visible
    # at the call site rather than via an implicit `cd X; cmd` chain.
    out_dict = orch.exec(
        f"bash -c 'cd {tb_src} && ROCM_PATH={rocm_path} HIPCC={hip_compiler} make'",
        timeout=500,
    )

    # Verify installation happened fine on all nodes
    out_dict = orch.exec(f'ls -l {git_install_path}/TransferBench')
    for node in out_dict.keys():
        if not re.search('TransferBench', out_dict[node]):
            fail_test(f'Transfer bench installation failed on node {node}')
    update_test_result()
