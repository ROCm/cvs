'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import pytest

import os
import re
import time
import json

from cvs.lib.utils_lib import *
from cvs.lib.verify_lib import *

from cvs.lib import globals

log = globals.log


# NOTE: This module assumes the following symbols are available in scope:
# - log: a configured logger
# - fail_test: helper that records/logs a failure (and may raise)
# - update_test_result: helper to finalize a test's pass/fail status
# - print_test_output: helper to pretty-print per-node command output
# - convert_hms_to_secs: helper to convert "HH:MM:SS" to seconds
# - globals.error_list: global list used to accumulate test errors across steps


# Importing additional cmd line args to script ..
@pytest.fixture(scope="module")
def cluster_file(pytestconfig):
    """
    Retrieve the --cluster_file CLI option value provided to pytest.

    Returns:
      str: Path to the cluster JSON file.
    """
    return pytestconfig.getoption("cluster_file")


@pytest.fixture(scope="module")
def config_file(pytestconfig):
    """
    Retrieve the --config_file CLI option value provided to pytest.

    Returns:
      str: Path to the test configuration JSON file.
    """
    return pytestconfig.getoption("config_file")


# Importing the cluster and cofig files to script to access node, switch, test config params
@pytest.fixture(scope="module")
def cluster_dict(cluster_file):
    """
    Load the full cluster configuration from JSON for use by tests.

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
    Load the AGFHC test configuration subsection from the provided JSON.

    Returns:
      dict: The 'agfhc' configuration map with keys like 'path', 'package_path', durations, etc.
    """
    with open(config_file) as json_file:
        config_dict_t = json.load(json_file)
    config_dict = config_dict_t['agfhc']

    # Resolve path placeholders like {user-id}, {home-mount-dir}, etc.
    config_dict = resolve_test_config_placeholders(config_dict, cluster_dict)

    log.info("%s", config_dict)
    return config_dict


@pytest.mark.dependency(name="init")
def test_install_agfhc(
    orch,
    config_dict,
):
    """
    Install AGFHC from the tarball package provided in the input config_file.json - package_tar_ball
    under the directory specified in the config_file - install_dir

    Install runs on every node via ``orch.exec``.
    """
    globals.error_list = []
    log.info('Testcase install agfhc')
    install_dir = config_dict['install_dir']
    package_tar_ball = config_dict['package_tar_ball']

    # Check if install directory exists, otherwise create.
    out_dict = orch.exec(f'ls -ld {install_dir}')
    for node in out_dict.keys():
        log.info(f'node ip {node}')
        log.info("%s", out_dict[node])
        if re.search('No such file or directory', out_dict[node], re.I):
            log.info(f'Install directory {install_dir} does not exist, creating')
            orch.exec(f'mkdir -p {install_dir}')

    # Copy the package to the install directory and untar. One logical
    # operation per orch.exec so that ContainerOrchestrator's docker-exec
    # transport (which doesn't spawn a shell) handles them correctly. Absolute
    # paths replace what used to be `cd <dir>; cp ...` chains.
    orch.exec(f'cp {package_tar_ball} {install_dir}')
    tarball_basename = os.path.basename(package_tar_ball)
    orch.exec(f'tar -xvf {install_dir}/{tarball_basename} -C {install_dir}')

    time.sleep(10)

    # install the untarred file. AGFHC's `./install` is a relative-cwd
    # script (reads sibling files via relative paths) so cwd MUST be
    # install_dir; we wrap that single call in `bash -c` explicitly to make
    # the cwd dependency visible at the call site rather than smuggling it
    # in via a `cd X; cmd` shell chain.
    #
    # --rocm-tar uses dpkg-deb direct extraction instead of apt-based dep
    # resolution. Required when /opt/rocm came from a TheRock tarball
    # (libs not tracked by dpkg, apt fails with "rocm-device-libs / hipcc /
    # lib32gcc-s1 not installable"). Safe on apt-rocm systems too: dpkg-deb
    # just extracts the bundled debs into /, which is correct either way.
    try:
        out_dict = orch.exec(
            f"sudo bash -c 'cd {install_dir} && ./install --rocm-tar'",
            timeout=90,
        )
        for node in out_dict.keys():
            log.info("%s", out_dict[node])
            if re.search('Error|No such file', out_dict[node], re.I):
                fail_test(f'Installation of AGFHC failed on node {node}')
    except Exception as e:
        log.error(f'Install of AGFHC failed, hit exception {e}')

    # verify agfhc path exists after installation ..
    out_dict = orch.exec(f'ls -l {config_dict["path"]}/agfhc')
    for node in out_dict.keys():
        log.info("%s", out_dict[node])
        if re.search('No such file', out_dict[node], re.I):
            fail_test(f'Installation of AGFHC failed on node {node}')
    update_test_result()
