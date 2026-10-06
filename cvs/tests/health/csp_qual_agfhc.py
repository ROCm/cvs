'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import pytest

import re
import time
import json

from cvs.lib.parallel_ssh_lib import *
from cvs.lib.utils_lib import *
from cvs.lib.verify_lib import *

from cvs.lib import agfhc_rundeck, globals
from cvs.lib.report.health_lifecycle import HealthLifecycle, timed_stage

log = globals.log


# NOTE: This module assumes the following symbols are available in scope:
# - log: a configured logger
# - Pssh: parallel SSH helper class
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


@pytest.fixture(scope="module")
def lifecycle():
    """Wall-clock of each CSP Qual AGFHC step, bound as the deck lifecycle source."""
    return HealthLifecycle()


@pytest.fixture(scope="module")
def agfhc_res_dict():
    """
    Module-scoped structured AGFHC results for the Run Deck status matrix.

    Each recipe merges its per-node verdict into this dict. The csp_qual_agfhc
    profile names this fixture in sources.results, so session binding captures
    it at module teardown.
    """
    return {}


def _capture_agfhc_rundeck(agfhc_res_dict, cluster_dict, group, out_dict, recorder=None, results_json=None):
    """Best-effort: a reporting problem must not change the AGFHC pass/fail."""
    if agfhc_res_dict is None:
        return
    try:
        meta = agfhc_rundeck.make_meta(cluster_dict, 'csp_qual_agfhc')
        if recorder is None:
            agfhc_rundeck.record_outputs(agfhc_res_dict, group, out_dict, meta=meta, results_json=results_json)
        else:
            recorder(agfhc_res_dict, out_dict, meta=meta)
    except Exception as exc:
        log.warning("AGFHC '%s': could not capture Run Deck results: %s", group, exc)


def _run_agfhc_recipe(phdl, config_dict, args, timeout, stage, out_name, agfhc_res_dict, cluster_dict, lifecycle):
    path = config_dict['path']
    log_dir = config_dict['log_dir']
    with timed_stage(lifecycle, stage):
        out_dict = phdl.exec(
            f'sudo {path}/agfhc {args} --simple-output -o {log_dir}/{out_name}',
            timeout=timeout,
        )
    scan_agfc_results(out_dict)
    results_json = get_log_results(phdl, out_dict)
    print_test_output(log, out_dict)
    _capture_agfhc_rundeck(agfhc_res_dict, cluster_dict, stage, out_dict, results_json=results_json)
    update_test_result()


def scan_agfc_results(out_dict):
    """
    Parse AGFHC run outputs from all nodes and fail on unexpected patterns.

    Args:
      out_dict (dict): Mapping node -> command stdout/stderr combined string.

    Behavior:
      - Requires 'return code AGFHC_SUCCESS' to appear in each node's output.
      - Fails if any of the patterns FAIL|ERROR|ABORT are present (case-insensitive).
    """

    for host in out_dict.keys():
        if not re.search('code AGFHC_SUCCESS', out_dict[host], re.I):
            fail_test(f'Test failed on node {host} - AGFHC_SUCCESS code NOT seen in test result')

        if re.search('FAIL|ERROR|ABORT', out_dict[host], re.I):
            fail_test(f'Test failed on node {host} - FAIL or ERROR or ABORT patterns seen')


def get_log_results(phdl, out_dict):
    res_cmd_list = []
    jrl_cmd_list = []
    err_cmd_list = []
    # Check results.json
    for node in out_dict.keys():
        match = re.search(r'Log directory:\s+([a-z0-9\/\-\_]+)', out_dict[node], re.I)
        log_dir = match.group(1)
        res_cmd_list.append(f'sudo cat {log_dir}/results.json')
        jrl_cmd_list.append(f'sudo cat {log_dir}/journal.log')
        err_cmd_list.append(f'sudo cat {log_dir}/error.json')
    res_dict = phdl.exec_cmd_list(res_cmd_list)
    for node in res_dict.keys():
        pattern = r'"total_failed":\s+0,'
        if not re.search(pattern, res_dict[node], re.I):
            fail_test(f'Total failed tests in results.json is not zero on node {node}')
            log.info('Dumping journal log from all nodes for reference')
            phdl.exec_cmd_list(jrl_cmd_list)
            phdl.exec_cmd_list(err_cmd_list)
    return res_dict


# Create connection to DUTs and export for later use ..
@pytest.fixture(scope="module")
def phdl(cluster_dict):
    """
    Build a parallel SSH handle to all nodes in the cluster.

    Returns:
    Pssh: A handle to execute commands across all nodes.
    """
    log.info("%s", cluster_dict)
    env_vars = cluster_dict.get("env_vars")
    node_list = list(cluster_dict['node_dict'].keys())
    phdl = Pssh(log, node_list, user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'], env_vars=env_vars)
    return phdl


# Get the version of AGFHC
@pytest.mark.dependency()
def test_version_check(phdl, config_dict, agfhc_res_dict, cluster_dict, lifecycle):
    globals.error_list = []
    path = config_dict['path']
    log_dir = config_dict['log_dir']
    with timed_stage(lifecycle, 'version_check'):
        out_dict = phdl.exec(f'sudo {path}/agfhc -v')
    for node in out_dict.keys():
        if not re.search('agfhc version:', out_dict[node], re.I):
            fail_test(f'Failed to print the AGFHC version on node {node}, installation not proper')
    # create the log directory to capture test logs
    try:
        phdl.exec(f'sudo rm -rf {log_dir}')
        time.sleep(2)
        phdl.exec(f'sudo mkdir {log_dir}')
    except Exception:
        log.error(f'Error creating log directory {log_dir}')
    ls_dict = phdl.exec(f'sudo ls -ld {log_dir}')
    for node in ls_dict.keys():
        if re.search('no such', ls_dict[node], re.I):
            fail_test(f'Error creating the log directory {log_dir} on node {node}')
    _capture_agfhc_rundeck(
        agfhc_res_dict,
        cluster_dict,
        'version_check',
        out_dict,
        recorder=agfhc_rundeck.record_version_check,
    )
    update_test_result()


# 2 hrs test
@pytest.mark.dependency(depends=["test_version_check"])
def test_all_lvl5(phdl, config_dict, agfhc_res_dict, cluster_dict, lifecycle):
    globals.error_list = []
    log.info('Testcase Run all_lvl5 Test')
    _run_agfhc_recipe(
        phdl,
        config_dict,
        '-r all_lvl5',
        (60 * 60 * 3) + 30,
        'all_lvl5',
        'test_all_lvl5',
        agfhc_res_dict,
        cluster_dict,
        lifecycle,
    )


# 1 iteration = 2 hrs with i=2
@pytest.mark.dependency(depends=["test_version_check"])
def test_agfhc_hbm_lvl5(phdl, config_dict, agfhc_res_dict, cluster_dict, lifecycle):
    """
    Run AGFHC HBM1 level 5 recipe for 4 iterations

    Steps:
      - Validate output and update aggregated test status.

    """
    globals.error_list = []
    log.info('Testcase Run HBM Test - hbm_lvl5')
    _run_agfhc_recipe(
        phdl,
        config_dict,
        '-r hbm_lvl5:i=2',
        (60 * 60 * 10) + 60,
        'hbm_lvl5',
        'test_agfhc_hbm_lvl5',
        agfhc_res_dict,
        cluster_dict,
        lifecycle,
    )


# 4 hrs
@pytest.mark.dependency(depends=["test_version_check"])
def test_agfhc_minihpl(phdl, config_dict, agfhc_res_dict, cluster_dict, lifecycle):
    """
    Run AGFHC miniHPL:
    Validates output and updates test status.
    """
    globals.error_list = []
    log.info('Testcase Run AGFHC miniHPL')
    _run_agfhc_recipe(
        phdl,
        config_dict,
        '-t minihpl:d=4h',
        (60 * 60 * 5) + 60,
        'minihpl',
        'test_agfhc_minihpl',
        agfhc_res_dict,
        cluster_dict,
        lifecycle,
    )


# 5 min
@pytest.mark.dependency(depends=["test_version_check"])
def test_agfhc_xgmi_lvl1(phdl, config_dict, agfhc_res_dict, cluster_dict, lifecycle):
    """
    Run AGFHC XGMI lvl1 recipe:
    Shorter test; validates output and records results.
    """
    globals.error_list = []
    log.info('Testcase Run XGMI lvl1')
    _run_agfhc_recipe(
        phdl,
        config_dict,
        '-r xgmi_lvl1',
        (60 * 20) + 30,
        'xgmi_lvl1',
        'test_agfhc_xgmi_lvl1',
        agfhc_res_dict,
        cluster_dict,
        lifecycle,
    )


# 10 min
# adding some additional time for buffer
@pytest.mark.dependency(depends=["test_version_check"])
def test_agfhc_pcie_lvl2(phdl, config_dict, agfhc_res_dict, cluster_dict, lifecycle):
    """
    Run AGFHC pcie lvl2:
    Validates output and updates test result.
    """
    globals.error_list = []
    log.info('Testcase Run PCIe lvl2')
    _run_agfhc_recipe(
        phdl,
        config_dict,
        '-r pcie_lvl2',
        (60 * 50) + 30,
        'pcie_lvl2',
        'test_agfhc_pcie_lvl2',
        agfhc_res_dict,
        cluster_dict,
        lifecycle,
    )


@pytest.mark.dependency(depends=["test_version_check"])
def test_agfhc_all_perf(phdl, config_dict, agfhc_res_dict, cluster_dict, lifecycle):
    """
    Pytest: Run the AGFHC 'all_perf' performance recipe across nodes.

    Args:
      phdl: Parallel SSH handle.
      config_dict (dict): Must include 'path'.

    Behavior:
      - Resets error accumulator.
      - Runs: sudo <path>/agfhc -r all_perf (90-minute timeout).
      - Scans outputs to ensure success and no fatal patterns.
      - Prints outputs and updates the aggregated test result.
    """
    globals.error_list = []
    log.info('Testcase Run all_perf')
    _run_agfhc_recipe(
        phdl,
        config_dict,
        '-r all_perf',
        60 * 120,
        'all_perf',
        'test_agfhc_all_perf',
        agfhc_res_dict,
        cluster_dict,
        lifecycle,
    )
