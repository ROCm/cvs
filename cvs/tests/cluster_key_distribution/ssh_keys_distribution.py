'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent
publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import json

import pytest

import cvs.lib.ssh_keys_lib as ssh_keys_lib
from cvs.lib import globals
from cvs.lib.utils_lib import (
    fail_test,
    resolve_cluster_config_placeholders,
    resolve_test_config_placeholders,
    update_test_result,
)

log = globals.log


@pytest.fixture(scope="module")
def cluster_file(pytestconfig):
    return pytestconfig.getoption("cluster_file")


@pytest.fixture(scope="module")
def config_file(pytestconfig):
    return pytestconfig.getoption("config_file")


@pytest.fixture(scope="module")
def cluster_dict(cluster_file):
    with open(cluster_file) as f:
        cluster_dict = json.load(f)
    cluster_dict = resolve_cluster_config_placeholders(cluster_dict)
    log.info("%s", cluster_dict)
    return cluster_dict


@pytest.fixture(scope="module")
def config_dict(config_file, cluster_dict):
    with open(config_file) as f:
        raw = json.load(f)
    subsection = raw["ssh_key_distribution"]
    subsection = resolve_test_config_placeholders(subsection, cluster_dict)
    log.info("%s", subsection)
    return subsection


@pytest.fixture(scope="module")
def norm_config(config_dict):
    try:
        return ssh_keys_lib.validate_key_distribution_config(config_dict)
    except ValueError as e:
        pytest.fail(f"ssh_key_distribution config invalid: {e}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_prepare_ssh_dir(orch, norm_config):
    globals.error_list = []
    log.info("Testcase: ensure ~/.ssh exists with mode 700 on all nodes")

    cmd = ssh_keys_lib.build_ensure_ssh_dir_cmd(norm_config["remote_ssh_dir"])
    out = orch.exec(cmd, timeout=30, detailed=True)

    for node, detail in out.items():
        exit_code = detail.get("exit_code", 0) if isinstance(detail, dict) else 0
        if exit_code != 0:
            fail_test(f"mkdir ~/.ssh failed on {node}: {detail}")

    update_test_result()


def test_distribute_cluster_keys(orch, norm_config):
    globals.error_list = []
    log.info("Testcase: distribute shared cluster keypair to all nodes")

    results = ssh_keys_lib.upload_cluster_keys(orch, norm_config)

    for node, ok in results.items():
        if not ok:
            fail_test(f"cluster key upload/chmod failed on {node}")

    # Verify remote presence
    remote_ssh_dir = norm_config["remote_ssh_dir"]
    key_name = norm_config["key_name"]
    check_cmd = f"test -f {remote_ssh_dir}/{key_name} && test -f {remote_ssh_dir}/{key_name}.pub"
    out = orch.exec(check_cmd, timeout=30, detailed=True)
    for node, detail in out.items():
        exit_code = detail.get("exit_code", 0) if isinstance(detail, dict) else 0
        if exit_code != 0:
            fail_test(f"cluster key files not found on {node}")

    update_test_result()


def test_authorize_cluster_key(orch, norm_config):
    globals.error_list = []
    log.info("Testcase: authorize cluster pubkey in authorized_keys on all nodes")

    results = ssh_keys_lib.authorize_cluster_pubkey(orch, norm_config)
    for node, ok in results.items():
        if not ok:
            fail_test(f"authorize cluster pubkey failed on {node}")

    update_test_result()


def test_authorize_controlling_station(orch, norm_config):
    globals.error_list = []

    controlling = norm_config.get("controlling_station_pubkey_path", "")
    if not controlling:
        pytest.skip("no controlling_station_pubkey_path configured")

    log.info("Testcase: authorize controlling station pubkey in authorized_keys on all nodes")

    results = ssh_keys_lib.authorize_controlling_station(orch, norm_config)
    for node, ok in results.items():
        if not ok:
            fail_test(f"authorize controlling station key failed on {node}")

    update_test_result()


def test_write_ssh_config(orch, cluster_dict, norm_config):
    globals.error_list = []
    log.info("Testcase: write ~/.ssh/config Host block on all nodes")

    results = ssh_keys_lib.install_ssh_config(orch, cluster_dict, norm_config)
    for node, ok in results.items():
        if not ok:
            fail_test(f"install_ssh_config failed on {node}")

    # Verify permissions
    remote_ssh_dir = norm_config["remote_ssh_dir"]
    perm_cmd = f"stat -c '%a' {remote_ssh_dir}/config"
    out = orch.exec(perm_cmd, timeout=30)
    for node, output in out.items():
        perm = output.strip()
        if perm != "600":
            fail_test(f"~/.ssh/config permissions are {perm!r} (expected 600) on {node}")

    update_test_result()


def test_verify_passwordless_ssh(orch, cluster_dict, norm_config):
    globals.error_list = []

    nodes = list(cluster_dict.get("node_dict", {}).keys())
    if len(nodes) < 2:
        pytest.skip("single-node cluster: no peer to verify")

    if not norm_config.get("verify_connectivity", True):
        pytest.skip("verify_connectivity disabled in config")

    log.info("Testcase: verify passwordless SSH between node pairs")

    pair_results = ssh_keys_lib.verify_passwordless_ssh(orch, cluster_dict, norm_config)
    for (src, dst), ok in pair_results.items():
        if not ok:
            fail_test(f"passwordless SSH {src} -> {dst} failed")

    update_test_result()
