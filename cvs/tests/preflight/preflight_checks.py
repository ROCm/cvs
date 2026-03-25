'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import pytest
import json

from cvs.lib import preflight_lib
from cvs.lib.parallel_ssh_lib import *
from cvs.lib.utils_lib import *
from cvs.lib.verify_lib import *
from cvs.parsers.schemas import validate_config_file

from cvs.lib import globals

log = globals.log

# Global results storage for HTML report generation
preflight_results = {}


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
    Load and validate cluster configuration from JSON file.

    Args:
      cluster_file (str): Path to cluster configuration file.

    Returns:
      dict: Validated cluster configuration dictionary.
    """
    cluster_config = validate_config_file(cluster_file, config_type="cluster")
    cluster_dict = cluster_config.model_dump()

    # Resolve path placeholders
    cluster_dict = resolve_cluster_config_placeholders(cluster_dict)
    log.info(f"Loaded cluster configuration with {len(cluster_dict['node_dict'])} nodes")

    return cluster_dict


@pytest.fixture(scope="module")
def config_dict(config_file, cluster_dict):
    """
    Load and validate test configuration from JSON file.

    Args:
      config_file (str): Path to test configuration file.
      cluster_dict (dict): Cluster configuration for placeholder resolution.

    Returns:
      dict: Validated test configuration dictionary.
    """
    with open(config_file) as json_file:
        config_dict_t = json.load(json_file)

    if 'preflight' not in config_dict_t:
        raise ValueError("Configuration file must contain 'preflight' section")

    config_dict = config_dict_t['preflight']

    # Resolve path placeholders
    config_dict = resolve_test_config_placeholders(config_dict, cluster_dict)
    log.info("Loaded preflight configuration")
    log.info(config_dict)

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
    """
    node_list = list(cluster_dict['node_dict'].keys())
    log.info(f"Creating parallel SSH handle for {len(node_list)} nodes")

    phdl = Pssh(log, node_list, user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'])
    return phdl


@pytest.fixture(scope="module")
def shdl(cluster_dict):
    """
    Build and return a parallel SSH handle (Pssh) for the head node only.

    Args:
      cluster_dict (dict): Cluster metadata fixture (see phdl docstring).

    Returns:
      Pssh: Handle configured for head node only (for single-node operations).
    """
    head_node = cluster_dict['head_node_dict']['mgmt_ip']
    log.info(f"Creating single SSH handle for head node: {head_node}")

    shdl = Pssh(log, [head_node], user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'])
    return shdl


def test_node_reachability(phdl):
    """
    Test basic SSH connectivity to all cluster nodes.

    This is a prerequisite test that ensures all nodes are reachable
    before running the actual preflight checks.
    """
    log.info("Testing node reachability via SSH")

    # Simple connectivity test
    cmd = "echo 'SSH_OK'"
    out_dict = phdl.exec(cmd)

    failed_nodes = []
    for node, output in out_dict.items():
        if 'SSH_OK' not in output:
            failed_nodes.append(node)
            fail_test(f"Node {node} is not reachable via SSH")

    if failed_nodes:
        fail_test(f"Failed to reach nodes: {', '.join(failed_nodes)}")

    log.info(f"All {len(out_dict)} nodes are reachable via SSH")
    update_test_result()


def test_gid_consistency(phdl, config_dict):
    """
    Test GID consistency across specified RDMA interfaces in the cluster.

    Verifies that the specified GID index exists and is valid on the
    specified RDMA interfaces across all cluster nodes.
    """
    global preflight_results

    gid_index = config_dict.get('gid_index', '3')
    expected_interfaces = config_dict.get('rdma_interfaces', ["rocep28s0", "rocep62s0", "rocep79s0", "rocep96s0"])
    log.info(f"Testing GID consistency for index {gid_index} on interfaces: {expected_interfaces}")

    results = preflight_lib.check_gid_consistency(phdl, gid_index, expected_interfaces)
    preflight_results['gid_consistency'] = results

    # Analyze results and report failures
    failed_nodes = []
    total_interfaces = 0
    ok_interfaces = 0

    for node, result in results.items():
        if result['status'] == 'FAIL':
            failed_nodes.append(node)
            for error in result['errors']:
                log.error(f"Node {node}: {error}")

        for interface, interface_result in result['interfaces'].items():
            total_interfaces += 1
            if interface_result.get('status') == 'OK':
                ok_interfaces += 1

    if failed_nodes:
        fail_test(f"GID consistency check failed on nodes: {', '.join(failed_nodes)}")

    log.info(
        f"GID consistency check passed: {ok_interfaces}/{total_interfaces} interfaces have valid GID index {gid_index}"
    )
    update_test_result()


def test_rocm_version_consistency(phdl, config_dict):
    """
    Test ROCm version consistency across all cluster nodes.

    Verifies that all nodes are running the expected ROCm version
    as specified in the configuration.
    """
    global preflight_results

    expected_version = config_dict.get('expected_rocm_version', '6.2.0')
    log.info(f"Testing ROCm version consistency (expected: {expected_version})")

    results = preflight_lib.check_rocm_versions(phdl, expected_version)
    preflight_results['rocm_versions'] = results

    # Analyze results and report failures
    failed_nodes = []
    version_summary = {}

    for node, result in results.items():
        detected_version = result['detected_version']
        if detected_version in version_summary:
            version_summary[detected_version].append(node)
        else:
            version_summary[detected_version] = [node]

        if result['status'] == 'FAIL':
            failed_nodes.append(node)
            for error in result['errors']:
                log.error(f"Node {node}: {error}")

    if failed_nodes:
        fail_test(f"ROCm version consistency check failed on nodes: {', '.join(failed_nodes)}")

    log.info(f"ROCm version consistency check passed: All {len(results)} nodes running version {expected_version}")
    update_test_result()


def test_interface_name_consistency(phdl, config_dict):
    """
    Test RDMA interface presence and consistency across all cluster nodes.

    Verifies that the expected RDMA interfaces are present on all nodes
    as specified in the configuration.
    """
    global preflight_results

    expected_interfaces = config_dict.get('rdma_interfaces', ["rocep28s0", "rocep62s0", "rocep79s0", "rocep96s0"])
    log.info(f"Testing interface presence (expected: {expected_interfaces})")

    results = preflight_lib.check_interface_names(phdl, expected_interfaces)
    preflight_results['interface_names'] = results

    # Analyze results and report failures
    failed_nodes = []
    total_interfaces = 0
    compliant_interfaces = 0

    for node, result in results.items():
        if result['status'] == 'FAIL':
            failed_nodes.append(node)
            for error in result['errors']:
                log.error(f"Node {node}: {error}")

        for interface in result['interfaces']:
            total_interfaces += 1
            if interface['expected'] and interface.get('functional', True):
                compliant_interfaces += 1

    if failed_nodes:
        fail_test(f"Interface naming consistency check failed on nodes: {', '.join(failed_nodes)}")

    log.info(
        f"Interface presence check passed: {compliant_interfaces}/{total_interfaces} interfaces are expected interfaces"
    )
    update_test_result()


def test_rdma_connectivity(phdl, cluster_dict, config_dict):
    """
    Test RDMA connectivity between cluster nodes using ibv_rc_pingpong.

    Uses direct IB verbs (same as RCCL) for more accurate connectivity testing
    that can detect issues that rping might miss.

    Tests connectivity based on the specified mode (basic, full_mesh, or sample)
    and reports any connection failures.
    """
    global preflight_results

    node_list = list(cluster_dict['node_dict'].keys())
    mode = config_dict.get('rdma_connectivity_check', 'basic')
    port_range = config_dict.get('rping_port_range', '9000-9999')
    timeout = int(config_dict.get('rping_timeout', '10'))
    expected_interfaces = config_dict.get('rdma_interfaces', ["rocep28s0", "rocep62s0", "rocep79s0", "rocep96s0"])
    gid_index = config_dict.get('gid_index', '3')

    log.info(
        f"Testing RDMA connectivity using ibv_rc_pingpong (mode: {mode}, timeout: {timeout}s, interfaces: {expected_interfaces}, GID: {gid_index})"
    )

    results = preflight_lib.check_rdma_connectivity(
        phdl, node_list, mode, port_range, timeout, expected_interfaces, gid_index
    )
    preflight_results['rdma_connectivity'] = results

    # Handle skipped test
    if results.get('skipped', False):
        log.info("RDMA connectivity test skipped by configuration")
        update_test_result()
        return

    # Analyze results and report failures
    if results['failed_pairs'] > 0:
        failed_pairs = []
        for pair_key, pair_result in results['pair_results'].items():
            if pair_result['status'] == 'FAIL':
                failed_pairs.append(pair_key)
                for error in pair_result['error_details']:
                    log.error(f"Pair {pair_key}: {error}")

        fail_test(f"RDMA connectivity check failed for {results['failed_pairs']} pairs: {', '.join(failed_pairs[:5])}")

    log.info(
        f"RDMA connectivity check passed: {results['successful_pairs']}/{results['total_pairs']} pairs connected successfully"
    )
    update_test_result()


def test_generate_preflight_report(config_dict, request):
    """
    Generate comprehensive preflight check report.

    Creates a summary of all preflight check results and generates
    an HTML report for easy review.
    """
    global preflight_results

    log.info("Generating preflight check report")

    # Ensure we have results from all checks
    required_checks = ['gid_consistency', 'rocm_versions', 'interface_names', 'rdma_connectivity']
    missing_checks = [check for check in required_checks if check not in preflight_results]

    if missing_checks:
        fail_test(f"Missing results for checks: {', '.join(missing_checks)}")

    # Generate comprehensive summary
    summary = preflight_lib.generate_preflight_summary(
        preflight_results['gid_consistency'],
        preflight_results['rdma_connectivity'],
        preflight_results['rocm_versions'],
        preflight_results['interface_names'],
    )

    preflight_results['summary'] = summary

    # Log summary to console
    log.info("=== PREFLIGHT CHECK SUMMARY ===")
    for check_name, check_summary in summary['checks'].items():
        status_icon = "✅" if check_summary['status'] == 'PASS' else "❌"
        log.info(
            f"{status_icon} {check_name.replace('_', ' ').title()}: {check_summary['status']} - {check_summary['summary']}"
        )

    log.info(f"\nOverall Status: {summary['overall_status']}")

    if summary['recommendations']:
        log.info("\nRecommendations:")
        for i, recommendation in enumerate(summary['recommendations'], 1):
            log.info(f"{i}. {recommendation}")

    # Generate HTML report
    html_report_path = None
    try:
        if config_dict.get('generate_html_report', 'true').lower() == 'true':
            html_report_path = preflight_lib.generate_html_report(preflight_results, config_dict)
            log.info(f"HTML report generated: {html_report_path}")
        else:
            log.info("HTML report generation disabled in configuration")
    except Exception as e:
        log.warning(f"Failed to generate HTML report: {e}")

    # Add HTML report to main test report bundle
    if html_report_path and hasattr(request.config, '_html_report_manager'):
        try:
            copied_path = request.config._html_report_manager.add_html_to_report(
                html_report_path, link_name="Preflight Checks Report", request=request
            )

            if copied_path:
                log.info(f'Preflight report saved and added to report bundle: {copied_path}')
            else:
                log.info(
                    f'Preflight report is saved under {html_report_path}, please copy it to your web server under /var/www/html folder to view'
                )
        except Exception as e:
            log.warning(f"Failed to add preflight report to bundle: {e}")
            log.info(f"Preflight report available at: {html_report_path}")

    # Fail the test if overall status is FAIL
    if summary['overall_status'] == 'FAIL':
        fail_test("One or more preflight checks failed - see detailed results above")

    log.info("All preflight checks passed - cluster is ready for performance testing")
    update_test_result()
