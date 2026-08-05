'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import pytest

import re
import os
import time
import json
import itertools

from cvs.lib import rccl_lib
from cvs.lib import html_lib
from cvs.lib.parallel_ssh_lib import *
from cvs.lib.utils_lib import *
from cvs.lib.verify_lib import *
from cvs.lib import globals

log = globals.log


rccl_res_dict = {}


# Importing additional cmd line args to script ..
@pytest.fixture(scope="module")
def cluster_file(pytestconfig):
    """
    Return the path to the cluster configuration JSON file passed via pytest CLI.

    Expects:
      - pytest to be invoked with: --cluster_file <path>

    Args:
      pytestconfig: Built-in pytest config object used to access CLI options.

    Returns:
      str: Filesystem path to the cluster configuration file.
    """
    return pytestconfig.getoption("cluster_file")


@pytest.fixture(scope="module")
def config_file(pytestconfig):
    """
    Return the path to the test configuration JSON file passed via pytest CLI.

    Expects:
      - pytest to be invoked with: --config_file <path>

    Args:
      pytestconfig: Built-in pytest config object used to access CLI options.

    Returns:
      str: Filesystem path to the test configuration file.
    """
    return pytestconfig.getoption("config_file")


@pytest.fixture(scope="module")
def cluster_dict(cluster_file):
    """
    Load and expose full cluster configuration for the test module.

    Behavior:
      - Opens the JSON at cluster_file and parses it into a Python dict.
      - Logs the parsed dictionary for visibility and debugging.
      - Returns the entire cluster configuration (node list, credentials, etc.).

    Args:
      cluster_file (str): Path to the cluster configuration JSON.

    Returns:
      dict: Parsed cluster configuration. Expected keys include:
            - 'node_dict': Map of node name -> node metadata
            - 'username': SSH username
            - 'priv_key_file': Path to SSH private key
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
    Load and return the RCCL-specific configuration dictionary for the test module.

    Args:
      config_file (str): Path to a JSON config file provided by another fixture.

    Returns:
      dict: The value of the "rccl" key from the loaded JSON, logged for visibility.

    Notes:
      - Expects the JSON file to contain a top-level key "rccl".
      - Uses module scope so the config is parsed once per test module.
      - Consider adding validation (e.g., assert "rccl" in config) to fail fast on bad configs.
    """
    with open(config_file) as json_file:
        config_dict_t = json.load(json_file)
    config_dict = config_dict_t['rccl']

    # Resolve path placeholders like {user-id}, {home-mount-dir}, etc.
    config_dict = resolve_test_config_placeholders(config_dict, cluster_dict)
    log.info("%s", config_dict)
    return config_dict


@pytest.fixture(scope="module")
def phdl(cluster_dict):
    """
    Build and return a parallel SSH handle (Pssh) for all cluster nodes.

    Args:
      cluster_dict (dict): Cluster metadata fixture containing:
        - node_dict: dict of node_name -> node_details
        - username: SSH username
        - priv_key_file: path to SSH private key

    Returns:
      Pssh: Handle configured for all nodes (for broadcast/parallel operations).

    Notes:
      - Prints the cluster_dict for quick debugging; consider replacing with log.debug.
      - Module-scoped so a single shared handle is used across all tests in the module.
      - nhdl_dict is currently unused; it can be removed unless used elsewhere.
      - Assumes Pssh(log, node_list, user=..., pkey=...) is available in scope.
    """
    log.info("%s", cluster_dict)
    env_vars = cluster_dict.get("env_vars")
    node_list = list(cluster_dict['node_dict'].keys())
    phdl = Pssh(log, node_list, user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'], env_vars=env_vars)
    return phdl


@pytest.fixture(scope="module")
def shdl(cluster_dict):
    """
    Build and return a parallel SSH handle (Pssh) for the head node only.

    Args:
      cluster_dict (dict): Cluster metadata fixture (see phdl docstring).

    Returns:
      Pssh: Handle configured for the first node (head node) in node_dict.

    Notes:
      - Useful when commands should be executed only from a designated head node.
      - Module scope ensures a single connection context for the duration of the module.
      - nhdl_dict is currently unused; it can be removed unless used elsewhere.
    """
    node_list = list(cluster_dict['node_dict'].keys())
    env_vars = cluster_dict.get("env_vars")
    head_node = node_list[0]
    shdl = Pssh(log, [head_node], user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'], env_vars=env_vars)
    return shdl


@pytest.fixture(scope="module")
def vpc_node_list(cluster_dict):
    """
    Collect and return a list of VPC IPs for all nodes in the cluster.

    Args:
      cluster_dict (dict): Cluster metadata fixture containing node_dict with vpc_ip per node.

    Returns:
      list[str]: List of VPC IP addresses in the cluster, ordered by node_dict iteration.

    Notes:
      - Iteration order depends on the insertion order of node_dict.
      - Consider validating that each node entry contains a 'vpc_ip' key.
    """
    vpc_node_list = []
    for node in list(cluster_dict['node_dict'].keys()):
        vpc_node_list.append(cluster_dict['node_dict'][node]['vpc_ip'])
    return vpc_node_list


def pytest_generate_tests(metafunc):
    """
    Pytest parametrization using regression object format.

    Uses 'regression' object with NCCL_* env names -> lists (Cartesian product)
    Special handling: NCCL_MIN_NCHANNELS and NCCL_MAX_NCHANNELS are paired (not Cartesian)
    """
    config_file = metafunc.config.getoption("config_file")
    if not config_file or not os.path.exists(config_file):
        log.warning(f'Warning: Missing or invalid config file {config_file}')
        return

    with open(config_file) as fp:
        cfg = json.load(fp)
    rccl = cfg.get("rccl", {})

    # Get regression object (required)
    regression = rccl.get("regression", {})
    if not regression:
        log.error("No regression object found in config - required for parametrization")
        return

    # Validate and handle paired channel config
    has_min_channels = "NCCL_MIN_NCHANNELS" in regression
    has_max_channels = "NCCL_MAX_NCHANNELS" in regression
    paired_channels = None

    if has_min_channels != has_max_channels:
        raise ValueError("NCCL_MIN_NCHANNELS and NCCL_MAX_NCHANNELS must be both present or both absent")

    if has_min_channels and has_max_channels:
        min_vals = regression["NCCL_MIN_NCHANNELS"]
        max_vals = regression["NCCL_MAX_NCHANNELS"]

        if len(min_vals) != len(max_vals):
            raise ValueError(
                f"NCCL_MIN_NCHANNELS ({len(min_vals)} items) and NCCL_MAX_NCHANNELS ({len(max_vals)} items) "
                "must have equal length for paired channel configuration"
            )

        # Store paired channels in internal variable
        paired_channels = list(zip(min_vals, max_vals))

        # Remove both from regression dict to prevent Cartesian product
        del regression["NCCL_MIN_NCHANNELS"]
        del regression["NCCL_MAX_NCHANNELS"]

    # Build product from regression object
    env_axes = []
    for key in sorted(regression.keys()):
        value = regression[key]
        if isinstance(value, list) and value:
            env_axes.append((key, value))

    if env_axes and "rccl_collective" in metafunc.fixturenames:
        # Always parametrize collectives
        rccl_collective_list = rccl.get("rccl_collective", ["all_reduce_perf"])

        # Build environment variable combinations as dicts for **regression_params
        env_fixture_names = [name for name, _ in env_axes]
        env_domains = [dict(env_axes)[name] for name in env_fixture_names]
        env_params, env_ids = [], []

        # Determine if we need to add channel pairs to the product
        channel_fixture_names = []
        if paired_channels is not None:
            channel_fixture_names = ["NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS"]
            env_domains.append(paired_channels)

        for env_combo in itertools.product(*env_domains):
            env_dict = dict(zip(env_fixture_names + channel_fixture_names, env_combo))

            # Unpack paired channels if present
            if paired_channels is not None:
                min_ch, max_ch = env_dict.pop("NCCL_MIN_NCHANNELS")
                env_dict["NCCL_MIN_NCHANNELS"] = min_ch
                env_dict["NCCL_MAX_NCHANNELS"] = max_ch

            env_params.append(env_dict)
            env_ids.append("|".join(f"{k}={v}" for k, v in env_dict.items()))

        # Parametrize collectives and regression_params dict
        metafunc.parametrize("rccl_collective", rccl_collective_list)
        metafunc.parametrize("regression_params", env_params, ids=env_ids)


# Start of test cases.


def test_collect_hostinfo(phdl):
    """
    Collect basic ROCm/host info from all nodes.

    Behavior:
      - Executes common ROCm commands to capture version and agent info.
      - Does not parse output; relies on update_test_result to finalize status.

    Notes:
      - globals.error_list is reset before test (pattern used across tests).
    """

    globals.error_list = []
    phdl.exec('cat /opt/rocm/.info/version')
    phdl.exec('hipconfig')
    phdl.exec('rocm_agent_enumerator')
    update_test_result()


def test_collect_networkinfo(phdl):
    """
    Collect basic RDMA/verbs info from all nodes.

    Behavior:
      - Executes 'rdma link' and 'ibv_devinfo' to snapshot network capabilities.
      - Does not parse output; relies on update_test_result to finalize status.
    """

    globals.error_list = []
    phdl.exec('rdma link')
    phdl.exec('ibv_devinfo')
    update_test_result()


def test_disable_firewall(phdl):
    globals.error_list = []
    sudo_status = get_passwordless_sudo_status(phdl)
    no_sudo_nodes = [node for node, ok in sudo_status.items() if not ok]
    if no_sudo_nodes:
        log.warning(
            "Skipping firewall disable check because passwordless sudo is unavailable on nodes: %s", no_sudo_nodes
        )
        update_test_result()
        return
    phdl.exec('sudo service ufw stop')
    time.sleep(2)
    out_dict = phdl.exec('sudo service ufw status')
    for node in out_dict.keys():
        if not re.search('inactive|dead|stopped|disabled|not be found|unrecognized service', out_dict[node], re.I):
            fail_test(f'Service ufw not disabled properly on node {node}')
    update_test_result()


def test_print_env_once(phdl, shdl, config_dict):
    """Single test to print environment script - don't dump env in every test."""
    globals.error_list = []
    env_script = config_dict.get('env_source_script', '/dev/null')
    if env_script and str(env_script).lower() != 'none':
        # Cat the env script file to show its contents
        cmd = f'echo === Environment Script: {env_script} === && cat {env_script}'
        shdl.exec(cmd)
    update_test_result()


def test_rccl_perf(phdl, shdl, cluster_dict, config_dict, rccl_collective, regression_params):
    """
    Execute RCCL regression test across the cluster with parametrized environment overrides.

    Parameters (from fixtures and config):
      - phdl: parallel execution handle for nodes (expects exec/exec_cmd_list).
      - shdl: switch or auxiliary handle used by rccl_lib (implementation-specific).
      - cluster_dict: cluster topology and credentials (expects node_dict, username, etc.).
      - config_dict: test configuration with RCCL/MPI paths, env, and thresholds.
      - rccl_collective: which RCCL collective test to run (e.g., "all_reduce_perf").
      - regression_params: dict of all regression parametrized values (NCCL_ALGO, NCCL_PROTO, NCCL_*_NCHANNELS, etc.)

    Flow:
      1) Capture start time to bound dmesg checks later.
      2) Optionally snapshot cluster metrics before the test (for debugging/compare).
      3) Build env_overrides dict from all regression parameters.
      4) Invoke rccl_lib.rccl_regression with parameters built from config and fixtures.
      5) Capture end time and verify dmesg for errors between start/end.
      6) Optionally snapshot metrics again and compare before/after.
      7) Call update_test_result() to finalize test status.

    Notes:
      - cluster_snapshot_debug controls whether before/after snapshots are taken.
    """

    globals.error_list = []
    sudo_status = get_passwordless_sudo_status(phdl)
    can_use_sudo = all(sudo_status.values())
    if not can_use_sudo:
        no_sudo_nodes = [node for node, ok in sudo_status.items() if not ok]
        log.warning(
            "Skipping dmesg markers/verification and sudo-only snapshots because passwordless sudo is unavailable "
            "on nodes: %s",
            no_sudo_nodes,
        )

    params_str = ' '.join(f'{k}={v}' for k, v in regression_params.items())
    if can_use_sudo:
        phdl.exec(f'sudo echo "Starting Test {rccl_collective} {params_str}" | sudo tee /dev/kmsg')

    # start_time = phdl.exec('date')
    start_time = phdl.exec('date +"%a %b %e %H:%M"')
    node_list = list(cluster_dict['node_dict'].keys())

    # Build list of nodes and their VPC IPs (used by the RCCL test)
    # make sure the VPC IPs are reachable from all nodes for passwordless ssh
    # otherwise use the regular mgmt-ip if that is reachable.
    vpc_node_list = []
    for node in list(cluster_dict['node_dict'].keys()):
        vpc_node_list.append(cluster_dict['node_dict'][node]['vpc_ip'])

    # Get cluster snapshot ..
    if can_use_sudo and re.search(
        'True', config_dict.get('cvs_params', {}).get('cluster_snapshot_debug', 'False'), re.I
    ):
        cluster_dict_before = create_cluster_metrics_snapshot(phdl)

    # Build env_overrides from all regression parameters (convert values to strings)
    env_overrides = {k: str(v) for k, v in regression_params.items()}

    env_script = config_dict.get('env_source_script', '/dev/null')
    result_dict = rccl_lib.rccl_regression(
        phdl,
        shdl,
        rccl_collective,
        env_script,
        config_dict['mpi_params'],
        config_dict['rccl_test_params'],
        config_dict['cvs_params'],
        node_list,
        vpc_node_list,
        env_overrides,
    )

    log.info("%s", result_dict)
    key_name = f'{rccl_collective}-{params_str}'
    rccl_res_dict[key_name] = result_dict

    # Scan dmesg between start and end times cluster wide ..
    # end_time = phdl.exec('date')
    if can_use_sudo:
        phdl.exec(f'sudo echo "End of Test {rccl_collective} {params_str}" | sudo tee /dev/kmsg')

    end_time = phdl.exec('date +"%a %b %e %H:%M"')
    if can_use_sudo:
        verify_dmesg_for_errors(phdl, start_time, end_time, till_end_flag=True)

    # Get new cluster snapshot and compare ..
    if can_use_sudo and re.search(
        'True', config_dict.get('cvs_params', {}).get('cluster_snapshot_debug', 'False'), re.I
    ):
        cluster_dict_after = create_cluster_metrics_snapshot(phdl)
        compare_cluster_metrics_snapshots(cluster_dict_before, cluster_dict_after)

    # Update test results based on any failures ..
    update_test_result()


def test_gen_graph(request):
    log.info('Final Global result dict')
    log.info("%s", rccl_res_dict)
    rccl_graph_dict = rccl_lib.convert_to_graph_dict(rccl_res_dict)
    log.info("%s", rccl_graph_dict)

    proc_id = os.getpid()

    html_file = f'/tmp/rccl_perf_report_{proc_id}.html'

    html_lib.add_html_begin(html_file)
    html_lib.build_rccl_amcharts_graph(html_file, 'rccl', rccl_graph_dict)
    html_lib.insert_chart(html_file, 'rccl')
    html_lib.build_rccl_result_table(html_file, rccl_graph_dict)
    html_lib.add_json_data(html_file, json.dumps(rccl_graph_dict))
    html_lib.add_html_end(html_file)

    # Add the HTML file to the report bundle with clickable link
    copied_path = request.config._html_report_manager.add_html_to_report(
        html_file, link_name="RCCL Multi Node Performance Report", request=request
    )

    if copied_path:
        log.info(f'Perf report saved and added to report bundle: {copied_path}')
    else:
        log.info(
            f'Perf report is saved under {html_file}, pls copy it to your web server under /var/www/html folder to view'
        )
