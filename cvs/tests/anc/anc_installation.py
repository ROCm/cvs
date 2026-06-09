'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent
publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import os
import pytest
import json

from cvs.lib.parallel_ssh_lib import Pssh
from cvs.lib.utils_lib import (
    fail_test,
    update_test_result,
    print_test_output,
    resolve_cluster_config_placeholders,
    resolve_test_config_placeholders,
)
from cvs.lib import globals

log = globals.log

ANC_TOOLS_PREFIX = "/opt/amdtools"
RESOURCES_DIR = os.path.join(os.path.dirname(__file__), "resources")


@pytest.fixture(scope="module")
def cluster_file(pytestconfig):
    """
    Retrieve the --cluster_file CLI option value provided to pytest.

    Returns:
      str: Path to the ANC cluster JSON file.
    """
    return pytestconfig.getoption("cluster_file")


@pytest.fixture(scope="module")
def config_file(pytestconfig):
    """
    Retrieve the --config_file CLI option value provided to pytest.

    Returns:
      str: Path to the ANC test configuration JSON file.
    """
    return pytestconfig.getoption("config_file")


@pytest.fixture(scope="module")
def cluster_dict(cluster_file):
    """
    Load and resolve the ANC cluster configuration from JSON.

    Returns:
      dict: Parsed and resolved cluster configuration.
    """
    with open(cluster_file) as json_file:
        cluster_dict = json.load(json_file)

    cluster_dict = resolve_cluster_config_placeholders(cluster_dict)
    log.info("ANC cluster config: %s", cluster_dict)
    return cluster_dict


@pytest.fixture(scope="module")
def config_dict(config_file, cluster_dict):
    """
    Load and resolve the ANC test configuration from JSON.

    Placeholders such as {home} are resolved using cluster_dict
    (e.g. cvs_home "{home}/cvs" -> "/home/<user>/cvs").

    Returns:
      dict: Parsed and resolved test configuration.
    """
    with open(config_file) as json_file:
        config_dict = json.load(json_file)

    config_dict = resolve_test_config_placeholders(config_dict, cluster_dict)
    log.info("ANC test config: %s", config_dict)
    return config_dict


@pytest.fixture(scope="module")
def phdl(cluster_dict):
    """
    Parallel SSH handle for all ANC cluster nodes.

    Returns:
      Pssh: Handle targeting every node in cluster_dict["node_dict"].
    """
    node_list = list(cluster_dict["node_dict"].keys())

    return Pssh(
        log,
        node_list,
        user=cluster_dict["username"],
        pkey=cluster_dict["priv_key_file"],
    )


class TestAncInstallPreTasks:
    """Pre-tasks: validate connectivity and gather host information."""

    def test_get_hostname(self, phdl, cluster_dict):
        """
        Retrieve hostname from all remote nodes in the ANC cluster.

        Validates that each node returns a non-empty hostname response.
        """
        globals.error_list = []
        log.info("ANC Pre-Task: Getting hostname from remote nodes")

        out_dict = phdl.exec("hostname", timeout=30)
        print_test_output(log, out_dict)

        for host, output in out_dict.items():
            hostname = output.strip()
            if not hostname:
                fail_test(f"Empty hostname returned from node {host}")
            else:
                log.info("Node %s reports hostname: %s", host, hostname)

        update_test_result()

    def _backup_download_install_anc_in_node(self, phdl, config_dict):
        """
        Download and install ANC binary on remote nodes (fresh install each run).

        Steps:
          1. cd to cvs_home on the remote node
          2. Remove anc_root_dir if present (force fresh install)
          3. wget the anc_release_url tarball
          4. Extract the outer tarball
          5. Extract anc-tool*.tar.gz to cvs_home, anc-content*.tar.gz to /opt/amdtools/
          6. List directories two levels from pwd
          7. Verify anc_root_dir exists
          8. SCP validate_exe_paths.py to remote, execute, then remove
          9. Verify each exe_path directory exists on the node
          10. If any path missing -> error; if all present -> success
        """
        globals.error_list = []

        anc_cfg = config_dict["anc"]
        cvs_home = anc_cfg["cvs_home"]
        anc_root_dir = anc_cfg["anc_root_dir"]
        anc_release_url = anc_cfg["anc_release_url"]
        tarball = anc_release_url.rsplit("/", 1)[-1]

        log.info("ANC Pre-Task: Download/install ANC on remote nodes")
        log.info("  cvs_home=%s, anc_root_dir=%s", cvs_home, anc_root_dir)

        # Phase 1: Download and extract (steps 1-7)
        install_cmd = (
            f"set -e && "
            f"mkdir -p '{cvs_home}' && cd '{cvs_home}' && "
            # Step 2: Remove existing if present
            f"if [ -d '{anc_root_dir}' ]; then "
            f"echo 'Removing existing {anc_root_dir}...' && "
            f"sudo rm -rf '{anc_root_dir}'; "
            f"fi && "
            # Step 3: Download
            f"echo 'Downloading ANC release...' && "
            f"wget -q '{anc_release_url}' -O '{tarball}' && "
            # Step 4: Extract outer tarball
            f"echo 'Extracting outer archive...' && "
            f"tar -xzf '{tarball}' && "
            f"rm -f '{tarball}' && "
            # Step 5: Extract anc-tool to current dir, anc-content to /opt/amdtools
            f"echo 'Extracting anc-tool archive to {cvs_home}...' && "
            f"tar -xzf anc-tool*.tar.gz && "
            f"rm -f anc-tool*.tar.gz && "
            f"echo 'Extracting anc-content to {ANC_TOOLS_PREFIX}/...' && "
            f"(mkdir -p {ANC_TOOLS_PREFIX} 2>/dev/null || sudo mkdir -p {ANC_TOOLS_PREFIX}) && "
            f"(tar -xzf anc-content*.tar.gz -C {ANC_TOOLS_PREFIX} 2>/dev/null || "
            f"sudo tar -xzf anc-content*.tar.gz -C {ANC_TOOLS_PREFIX}) && "
            f"rm -f anc-content*.tar.gz && "
            # Step 6: List directories
            f"echo '--- Directory listing (2 levels) ---' && "
            f"find . -maxdepth 2 -type d | sort && "
            # Step 7: Verify anc_root_dir
            f"echo '--- Verification ---' && "
            f"test -d '{anc_root_dir}' && echo 'ANC_INSTALL_SUCCESS'"
        )

        out_dict = phdl.exec(install_cmd, timeout=300)
        print_test_output(log, out_dict)

        for host, output in out_dict.items():
            if "ANC_INSTALL_SUCCESS" not in output:
                fail_test(
                    f"ANC installation failed on {host}: "
                    f"'{anc_root_dir}' not found under {cvs_home}"
                )
            else:
                log.info("Node %s: ANC directory installed successfully", host)

        # Bail early if install phase failed
        if globals.error_list:
            update_test_result()
            return

        # Phase 2: SCP validation script, execute, cleanup (steps 8-10)
        content_dir = f"{ANC_TOOLS_PREFIX}/{anc_root_dir}/content"
        local_script = os.path.join(RESOURCES_DIR, "validate_exe_paths.py")
        remote_script = f"/tmp/validate_exe_paths.py"

        log.info("Uploading validation script to remote nodes...")
        phdl.upload_file(local_script, remote_script)

        log.info("Validating exe_path entries from %s", content_dir)
        validate_dict = phdl.exec(
            f"python3 {remote_script} '{content_dir}'", timeout=60
        )
        print_test_output(log, validate_dict)

        # Cleanup: remove the script from remote nodes
        log.info("Removing validation script from remote nodes...")
        phdl.exec(f"rm -f {remote_script}", timeout=10)

        for host, output in validate_dict.items():
            if "VALIDATION_FAILED" in output:
                fail_test(
                    f"exe_path validation failed on {host}: "
                    f"one or more tool paths are missing"
                )
            elif "VALIDATION_SUCCESS" in output:
                log.info("Node %s: All exe_path entries validated", host)
            elif "ERROR:" in output:
                fail_test(f"Validation error on {host}: {output.strip()}")
            else:
                fail_test(
                    f"Unexpected validation output on {host}: {output.strip()}"
                )

        update_test_result()


    def test_download_install_anc_in_node_cvs_home(self, phdl, config_dict):
        """
        Download and install ANC binary on remote nodes (fresh install each run).

        Same as test_download_install_anc_in_node, except the content archive is
        extracted into cvs_home instead of /opt/amdtools, and the
        anc_root_dir/content/base directory is removed after extraction.

        Steps:
          1. cd to cvs_home on the remote node
          2. Remove anc_root_dir if present (force fresh install)
          3. wget the anc_release_url tarball
          4. Extract the outer tarball
          5. Extract anc-tool*.tar.gz and anc-content*.tar.gz to cvs_home
          6. Remove anc_root_dir/content/base
          7. List directories two levels from pwd
          8. Verify anc_root_dir exists
          9. SCP validate_exe_paths.py to remote, execute, then remove
          10. Verify each exe_path directory exists on the node
          11. If any path missing -> error; if all present -> success
        """
        globals.error_list = []

        anc_cfg = config_dict["anc"]
        cvs_home = anc_cfg["cvs_home"]
        anc_root_dir = anc_cfg["anc_root_dir"]
        anc_release_url = anc_cfg["anc_release_url"]
        tarball = anc_release_url.rsplit("/", 1)[-1]

        log.info("ANC Pre-Task: Download/install ANC on remote nodes (cvs_home)")
        log.info("  cvs_home=%s, anc_root_dir=%s", cvs_home, anc_root_dir)

        # Phase 1: Download and extract (steps 1-8)
        install_cmd = (
            f"set -e && "
            f"mkdir -p '{cvs_home}' && cd '{cvs_home}' && "
            # Step 2: Remove existing if present
            f"if [ -d '{anc_root_dir}' ]; then "
            f"echo 'Removing existing {anc_root_dir}...' && "
            f"rm -rf '{anc_root_dir}'; "
            f"fi && "
            # Step 3: Download
            f"echo 'Downloading ANC release...' && "
            f"wget -q '{anc_release_url}' -O '{tarball}' && "
            # Step 4: Extract outer tarball
            f"echo 'Extracting outer archive...' && "
            f"tar -xzf '{tarball}' && "
            f"rm -f '{tarball}' && "
            # Step 5: Extract anc-tool and anc-content both to cvs_home
            f"echo 'Extracting anc-tool archive to {cvs_home}...' && "
            f"tar -xzf anc-tool*.tar.gz && "
            f"rm -f anc-tool*.tar.gz && "
            f"echo 'Extracting anc-content archive to {cvs_home}...' && "
            f"tar -xzf anc-content*.tar.gz && "
            f"rm -f anc-content*.tar.gz && "
            # Step 6: Remove content/base
            f"echo 'Removing {anc_root_dir}/content/base...' && "
            f"rm -rf '{anc_root_dir}/content/base' && "
            # Step 7: List directories
            f"echo '--- Directory listing (2 levels) ---' && "
            f"find . -maxdepth 2 -type d | sort && "
            # Step 8: Verify anc_root_dir
            f"echo '--- Verification ---' && "
            f"test -d '{anc_root_dir}' && echo 'ANC_INSTALL_SUCCESS'"
        )

        out_dict = phdl.exec(install_cmd, timeout=300)
        print_test_output(log, out_dict)

        for host, output in out_dict.items():
            if "ANC_INSTALL_SUCCESS" not in output:
                fail_test(
                    f"ANC installation failed on {host}: "
                    f"'{anc_root_dir}' not found under {cvs_home}"
                )
            else:
                log.info("Node %s: ANC directory installed successfully", host)

        # Bail early if install phase failed
        if globals.error_list:
            update_test_result()
            return

        # Phase 2: SCP validation script, execute, cleanup (steps 9-11)
        content_dir = f"{cvs_home}/{anc_root_dir}/content"
        local_script = os.path.join(RESOURCES_DIR, "validate_exe_paths.py")
        remote_script = f"/tmp/validate_exe_paths.py"

        log.info("Uploading validation script to remote nodes...")
        phdl.upload_file(local_script, remote_script)

        log.info("Validating exe_path entries from %s", content_dir)
        validate_dict = phdl.exec(
            f"python3 {remote_script} '{content_dir}'", timeout=60
        )
        print_test_output(log, validate_dict)

        # Cleanup: remove the script from remote nodes
        log.info("Removing validation script from remote nodes...")
        phdl.exec(f"rm -f {remote_script}", timeout=10)

        for host, output in validate_dict.items():
            if "VALIDATION_FAILED" in output:
                fail_test(
                    f"exe_path validation failed on {host}: "
                    f"one or more tool paths are missing"
                )
            elif "VALIDATION_SUCCESS" in output:
                log.info("Node %s: All exe_path entries validated", host)
            elif "ERROR:" in output:
                fail_test(f"Validation error on {host}: {output.strip()}")
            else:
                fail_test(
                    f"Unexpected validation output on {host}: {output.strip()}"
                )

        update_test_result()


class TestAncInstallCoreTasks:
    """Core ANC execution tasks (placeholder for ANC workload)."""
    pass


class TestAncInstallPostTasks:
    """Post-tasks: cleanup and result collection (placeholder)."""
    pass
