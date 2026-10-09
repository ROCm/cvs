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

from cvs.lib.utils_lib import *
from cvs.lib.verify_lib import *

from cvs.lib import agfhc_rundeck, globals
from cvs.lib.report.health_lifecycle import HealthLifecycle, timed_stage

log = globals.log


def _payload_sudo_prefix(orch):
    """Sudo for a command passed to orch.exec().

    sudo_prefix() is host passwordless sudo. Container exec already applies
    that to `docker exec`, and the payload runs inside the container.
    """
    if getattr(orch, 'orchestrator_type', None) == 'container':
        return ''
    return orch.sudo_prefix()


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


def _build_agfhc_cmd(orch, path, args):
    """Build an AGFHC CLI invocation with Spur/container-safe sudo."""
    return f'{_payload_sudo_prefix(orch)}{path}/agfhc {args}'


def _run_agfhc_recipe(orch, config_dict, args, timeout, stage, out_name, agfhc_res_dict, cluster_dict, lifecycle):
    path = config_dict['path']
    log_dir = config_dict['log_dir']
    cmd = _build_agfhc_cmd(orch, path, f'{args} --simple-output -o {log_dir}/{out_name}')
    with timed_stage(lifecycle, stage):
        out_dict = orch.exec(cmd, timeout=timeout)
    scan_agfc_results(out_dict)
    results_json = get_log_results(orch, out_dict)
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


def _exec_cmds_by_host(orch, commands_by_host):
    """Run one command per host via orch.all.exec_cmd_list when available."""
    handle = getattr(orch, 'all', None)
    hosts = list(
        getattr(handle, 'reachable_hosts', None) or getattr(handle, 'host_list', None) or commands_by_host.keys()
    )
    hosts = [host for host in hosts if host in commands_by_host]
    executor = getattr(handle, 'exec_cmd_list', None) if handle is not None else None
    if callable(executor):
        try:
            output = executor([commands_by_host[host] for host in hosts], timeout=90, print_console=False)
            if isinstance(output, dict):
                normalized = {}
                for host in hosts:
                    value = output.get(host, '')
                    if isinstance(value, dict):
                        value = value.get('output', '')
                    normalized[host] = str(value)
                return normalized
        except TypeError:
            pass
    result = {}
    for host in hosts:
        out = orch.exec(commands_by_host[host], hosts=[host])
        result[host] = str(out.get(host, ''))
    return result


def get_log_results(orch, out_dict):
    sudo_prefix = _payload_sudo_prefix(orch)
    res_cmds = {}
    jrl_cmds = {}
    err_cmds = {}
    for node in out_dict.keys():
        match = re.search(r'Log directory:\s+([a-z0-9\/\-\_]+)', out_dict[node], re.I)
        log_dir = match.group(1)
        res_cmds[node] = f'{sudo_prefix}cat {log_dir}/results.json'
        jrl_cmds[node] = f'{sudo_prefix}cat {log_dir}/journal.log'
        err_cmds[node] = f'{sudo_prefix}cat {log_dir}/error.json'
    res_dict = _exec_cmds_by_host(orch, res_cmds)
    for node in res_dict.keys():
        pattern = r'"total_failed":\s+0,'
        if not re.search(pattern, res_dict[node], re.I):
            fail_test(f'Total failed tests in results.json is not zero on node {node}')
            log.info('Dumping journal log from all nodes for reference')
            _exec_cmds_by_host(orch, jrl_cmds)
            _exec_cmds_by_host(orch, err_cmds)
    return res_dict


# Get the version of AGFHC
@pytest.mark.dependency()
def test_version_check(orch, config_dict, agfhc_res_dict, cluster_dict, lifecycle):
    globals.error_list = []
    path = config_dict['path']
    log_dir = config_dict['log_dir']
    sudo_prefix = _payload_sudo_prefix(orch)
    with timed_stage(lifecycle, 'version_check'):
        out_dict = orch.exec(_build_agfhc_cmd(orch, path, '-v'))
    for node in out_dict.keys():
        if not re.search('agfhc version:', out_dict[node], re.I):
            fail_test(f'Failed to print the AGFHC version on node {node}, installation not proper')
    # create the log directory to capture test logs
    try:
        orch.exec(f'{sudo_prefix}rm -rf {log_dir}')
        time.sleep(2)
        orch.exec(f'{sudo_prefix}mkdir {log_dir}')
    except Exception:
        log.error(f'Error creating log directory {log_dir}')
    ls_dict = orch.exec(f'{sudo_prefix}ls -ld {log_dir}')
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
def test_all_lvl5(orch, config_dict, agfhc_res_dict, cluster_dict, lifecycle):
    globals.error_list = []
    log.info('Testcase Run all_lvl5 Test')
    _run_agfhc_recipe(
        orch,
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
def test_agfhc_hbm_lvl5(orch, config_dict, agfhc_res_dict, cluster_dict, lifecycle):
    """
    Run AGFHC HBM1 level 5 recipe for 4 iterations

    Steps:
      - Validate output and update aggregated test status.

    """
    globals.error_list = []
    log.info('Testcase Run HBM Test - hbm_lvl5')
    _run_agfhc_recipe(
        orch,
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
def test_agfhc_minihpl(orch, config_dict, agfhc_res_dict, cluster_dict, lifecycle):
    """
    Run AGFHC miniHPL:
    Validates output and updates test status.
    """
    globals.error_list = []
    log.info('Testcase Run AGFHC miniHPL')
    _run_agfhc_recipe(
        orch,
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
def test_agfhc_xgmi_lvl1(orch, config_dict, agfhc_res_dict, cluster_dict, lifecycle):
    """
    Run AGFHC XGMI lvl1 recipe:
    Shorter test; validates output and records results.
    """
    globals.error_list = []
    log.info('Testcase Run XGMI lvl1')
    _run_agfhc_recipe(
        orch,
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
def test_agfhc_pcie_lvl2(orch, config_dict, agfhc_res_dict, cluster_dict, lifecycle):
    """
    Run AGFHC pcie lvl2:
    Validates output and updates test result.
    """
    globals.error_list = []
    log.info('Testcase Run PCIe lvl2')
    _run_agfhc_recipe(
        orch,
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
def test_agfhc_all_perf(orch, config_dict, agfhc_res_dict, cluster_dict, lifecycle):
    """
    Pytest: Run the AGFHC 'all_perf' performance recipe across nodes.

    Args:
      orch: Shared orchestrator fixture (baremetal SSH, Spur HTTP, or container).
      config_dict (dict): Must include 'path'.

    Behavior:
      - Resets error accumulator.
      - Runs: <path>/agfhc -r all_perf with Spur/container-safe sudo (90-minute timeout).
      - Scans outputs to ensure success and no fatal patterns.
      - Prints outputs and updates the aggregated test result.
    """
    globals.error_list = []
    log.info('Testcase Run all_perf')
    _run_agfhc_recipe(
        orch,
        config_dict,
        '-r all_perf',
        60 * 120,
        'all_perf',
        'test_agfhc_all_perf',
        agfhc_res_dict,
        cluster_dict,
        lifecycle,
    )
