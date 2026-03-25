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


# Override update_test_result for preflight tests to be reporting-only
def preflight_update_test_result():
    """
    Preflight-specific test result handler that reports issues but doesn't fail tests.
    This allows preflight to be a comprehensive reporting tool rather than a pass/fail test.
    """
    if len(globals.error_list) > 0:
        log.info(f"Preflight detected {len(globals.error_list)} issues (see detailed logs above)")
        # Clear the error list to prevent test failure
        globals.error_list.clear()
    # Always pass - preflight is for reporting, not failing


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

    phdl = Pssh(log, node_list, user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'], stop_on_errors=False,
                timeout=15, num_retries=1, retry_delay=2)
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

    shdl = Pssh(
        log, [head_node], user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'], stop_on_errors=False
    )
    return shdl


def test_node_reachability(phdl):
    """
    Test basic SSH connectivity to all cluster nodes.

    Logs unreachable nodes but continues with reachable ones.
    This allows preflight tests to run on available nodes.
    """
    # Clear any previous errors for preflight reporting mode
    globals.error_list.clear()

    log.info("Testing node reachability via SSH")

    # Simple connectivity test
    cmd = "echo 'SSH_OK'"
    out_dict = phdl.exec(cmd)

    failed_nodes = []
    reachable_nodes = []

    for node, output in out_dict.items():
        if 'SSH_OK' not in output:
            failed_nodes.append(node)
            if 'ABORT: Host Unreachable Error' in output:
                log.warning(f"Node {node} is unreachable (will be excluded from further tests)")
            else:
                log.error(f"Node {node} failed connectivity test: {output.strip()}")
        else:
            reachable_nodes.append(node)

    if failed_nodes:
        log.warning(f"Unreachable nodes ({len(failed_nodes)}): {', '.join(failed_nodes)}")
        log.info(f"Continuing preflight tests with {len(reachable_nodes)} reachable nodes")

    log.info(f"Node reachability: {len(reachable_nodes)}/{len(out_dict)} nodes reachable")

    # Store reachability results for summary
    global preflight_results
    preflight_results['node_reachability'] = {
        'total_nodes': len(out_dict),
        'reachable_nodes': len(reachable_nodes),
        'unreachable_nodes': failed_nodes,
        'status': 'PASS' if len(failed_nodes) == 0 else 'WARNING',
    }

    # Prune unreachable nodes from phdl so subsequent tests only run on reachable nodes
    phdl.prune_unreachable_hosts()

    preflight_update_test_result()


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
        log.warning(f"GID consistency issues on {len(failed_nodes)} nodes: {', '.join(failed_nodes)}")
    else:
        log.info("GID consistency check: All nodes passed")

    log.info(f"GID consistency results: {ok_interfaces}/{total_interfaces} interfaces have valid GID index {gid_index}")
    preflight_update_test_result()


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
        log.warning(f"ROCm version inconsistencies on {len(failed_nodes)} nodes: {', '.join(failed_nodes)}")
    else:
        log.info("ROCm version consistency check: All reachable nodes passed")

    log.info(
        f"ROCm version results: {len(results) - len(failed_nodes)}/{len(results)} nodes have expected version {expected_version}"
    )
    preflight_update_test_result()


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
        log.warning(f"Interface naming inconsistencies on {len(failed_nodes)} nodes: {', '.join(failed_nodes)}")
    else:
        log.info("Interface naming consistency check: All reachable nodes passed")

    log.info(
        f"Interface presence results: {compliant_interfaces}/{total_interfaces} interfaces are expected interfaces"
    )
    preflight_update_test_result()


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
    parallel_group_size = config_dict.get('parallel_group_size', 128)

    log.info(
        f"Testing RDMA connectivity using parallel algorithm (mode: {mode}, group_size: {parallel_group_size}, timeout: {timeout}s, interfaces: {expected_interfaces}, GID: {gid_index})"
    )

    results = preflight_lib.check_rdma_connectivity(
        phdl, node_list, mode, port_range, timeout, expected_interfaces, gid_index, parallel_group_size, config_dict
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

        log.warning(
            f"RDMA connectivity issues: {results['failed_pairs']} failed pairs out of {results['total_pairs']} total"
        )
        for pair in failed_pairs[:5]:  # Log first 5 failed pairs
            log.warning(f"Failed pair: {pair}")
    else:
        log.info("RDMA connectivity check: All tested pairs connected successfully")

    log.info(
        f"RDMA connectivity results: {results['successful_pairs']}/{results['total_pairs']} pairs connected successfully"
    )
    preflight_update_test_result()


@pytest.mark.skip(reason="SSH connectivity test - work in progress")
def test_ssh_full_mesh_connectivity(phdl, cluster_dict, config_dict):
    """
    Test SSH connectivity between all cluster nodes (full mesh).

    This test validates that every node can SSH to every other node,
    which is critical for MPI job launches and distributed computing frameworks.
    Catches SSH key distribution issues, firewall problems, and hostname resolution
    failures that could cause distributed jobs to fail.
    """
    global preflight_results

    # Check if SSH full mesh testing is enabled
    if not config_dict.get('ssh_full_mesh_check', False):
        log.info("SSH full mesh connectivity testing is disabled - skipping")
        preflight_results['ssh_connectivity'] = {
            'status': 'SKIPPED',
            'message': 'SSH full mesh testing disabled in configuration',
        }
        pytest.skip("SSH full mesh testing disabled in configuration")
        return

    log.info("Testing SSH full mesh connectivity between all cluster nodes")

    # Get SSH timeout from config
    ssh_timeout = config_dict.get('ssh_connection_timeout', 5)

    # Get reachable nodes from cluster
    node_list = list(cluster_dict['node_dict'].keys())
    log.info(f"Testing SSH connectivity across {len(node_list)} nodes")

    # Run SSH full mesh connectivity test
    results = preflight_lib.test_ssh_full_mesh_connectivity(
        phdl, node_list, timeout=ssh_timeout, config_dict=config_dict
    )

    # Store results for report generation
    preflight_results['ssh_connectivity'] = results

    # Determine test outcome
    if results.get('skipped', False):
        log.info("SSH connectivity test was skipped")
        preflight_results['ssh_connectivity']['status'] = 'SKIPPED'
        preflight_results['ssh_connectivity']['message'] = 'Insufficient nodes for SSH mesh testing'
        pytest.skip("Need at least 2 nodes for SSH mesh testing")
        return

    total_pairs = results['total_pairs']
    successful_pairs = results['successful_pairs']
    failed_pairs = results['failed_pairs']

    # Calculate success rate
    success_rate = (successful_pairs / total_pairs * 100) if total_pairs > 0 else 0

    # Log detailed results
    if successful_pairs == 0 and total_pairs > 0:
        # No successful connections parsed - this indicates a test infrastructure issue
        preflight_results['ssh_connectivity']['status'] = 'ERROR'
        preflight_results['ssh_connectivity']['message'] = (
            f'SSH test infrastructure error: no results parsed from {total_pairs} expected tests'
        )

        log.error(f"SSH test infrastructure error: no results were parsed from any of the {total_pairs} expected tests")
        log.error("This suggests an issue with script execution or result collection, not SSH connectivity itself")

        log.error(
            f"SSH full mesh test infrastructure failed: no results parsed from {total_pairs} tests. "
            f"Check script execution and result collection logic."
        )

    elif failed_pairs > 0:
        preflight_results['ssh_connectivity']['status'] = 'FAIL'
        preflight_results['ssh_connectivity']['message'] = (
            f'{successful_pairs}/{total_pairs} SSH connections successful ({success_rate:.1f}%)'
        )

        log.error(f"SSH connectivity issues detected: {failed_pairs}/{total_pairs} connections failed")

        # Analyze failure patterns
        pair_results = results.get('pair_results', {})
        failed_connections = [(pair, error) for pair, error in pair_results.items() if 'SUCCESS' not in error]

        # Group failures by error type for better analysis
        error_patterns = {}
        problematic_nodes = set()

        for pair, error in failed_connections:
            source_node, target_node = pair.split(' → ')
            problematic_nodes.add(target_node)  # Target nodes are usually the problem

            # Categorize error types
            if 'Host key verification failed' in error:
                error_type = 'Host key verification failed'
            elif 'Connection timed out' in error or 'Connection refused' in error:
                error_type = 'Connection/Network issue'
            elif 'Permission denied' in error:
                error_type = 'SSH key/Authentication issue'
            elif 'Name resolution failed' in error:
                error_type = 'Hostname resolution issue'
            else:
                error_type = 'Other SSH issue'

            if error_type not in error_patterns:
                error_patterns[error_type] = []
            error_patterns[error_type].append(pair)

        # Log error analysis
        log.error("SSH connectivity failure analysis:")
        for error_type, affected_pairs in error_patterns.items():
            log.error(f"  {error_type}: {len(affected_pairs)} connections")
            # Show a few examples
            for pair in affected_pairs[:3]:
                log.error(f"    Example: {pair}")
            if len(affected_pairs) > 3:
                log.error(f"    ... and {len(affected_pairs) - 3} more")

        # Identify most problematic nodes
        if problematic_nodes:
            log.error(f"Nodes with SSH connectivity issues: {', '.join(sorted(problematic_nodes))}")

        # Log critical failure for distributed computing
        log.error(
            f"SSH full mesh connectivity failed: {failed_pairs}/{total_pairs} connections failed. "
            f"This will prevent MPI jobs and distributed frameworks from working properly. "
            f"Check SSH keys, firewalls, and hostname resolution on problematic nodes."
        )

    else:
        # successful_pairs > 0 and failed_pairs == 0
        preflight_results['ssh_connectivity']['status'] = 'PASS'
        preflight_results['ssh_connectivity']['message'] = f'All {total_pairs} SSH connections successful'

        log.info("SSH full mesh connectivity: All tested connections successful")
        log.info("✅ Cluster is ready for MPI and distributed computing workloads")

    log.info(f"SSH connectivity results: {successful_pairs}/{total_pairs} connections successful ({success_rate:.1f}%)")
    preflight_update_test_result()


def test_generate_preflight_report(config_dict, request):
    """
    Generate comprehensive preflight check report.

    Creates a summary of all preflight check results and generates
    an HTML report for easy review.
    """
    global preflight_results

    log.info("Generating preflight check report")

    # Ensure we have results from all checks
    required_checks = ['gid_consistency', 'rocm_versions', 'interface_names', 'rdma_connectivity', 'ssh_connectivity']
    missing_checks = [check for check in required_checks if check not in preflight_results]

    if missing_checks:
        log.error(f"Missing results for checks: {', '.join(missing_checks)}")
        # Create empty results for missing checks to allow report generation
        for check in missing_checks:
            preflight_results[check] = {'status': 'SKIPPED', 'message': 'Check was skipped due to earlier failures'}

    # Generate comprehensive summary
    summary = preflight_lib.generate_preflight_summary(
        preflight_results['gid_consistency'],
        preflight_results['rdma_connectivity'],
        preflight_results['rocm_versions'],
        preflight_results['interface_names'],
        preflight_results.get('node_reachability'),
        preflight_results.get('ssh_connectivity'),
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

    # Report overall status but don't fail the test
    if summary['overall_status'] == 'FAIL':
        log.warning("One or more preflight checks detected issues - see detailed results above")
        log.info("Preflight report generated successfully - review HTML report for detailed analysis")
    else:
        log.info("All preflight checks passed - cluster is ready for performance testing")
    preflight_update_test_result()
