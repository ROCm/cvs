'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import functools
import inspect

import pytest
import json

# Import new modular preflight classes
from cvs.lib.preflight.gid_consistency import GidConsistencyCheck
from cvs.lib.rdma_gid_lib import DEFAULT_GID_TYPE
from cvs.lib.preflight.version_check import RocmVersionCheck
from cvs.lib.preflight.interface_consistency import InterfaceConsistencyCheck
from cvs.lib.preflight.ifoe_l2_connectivity import IfoeL2ConnectivityCheck
from cvs.lib.preflight.scaleup_fabric import NodeHealthCheck
from cvs.lib.preflight.transferbench_smoke import TransferBenchSmokeCheck
from cvs.lib.preflight.node_smoke import (
    NODE_SMOKE_TIER1_LABEL,
    NODE_SMOKE_TIER2_LABEL,
    NODE_SMOKE_TIER3_LABEL,
    NodeSmokeCheck,
)
from cvs.lib.preflight.tier3_info import NodeSmokeTier3Check
from cvs.lib.preflight.node_smoke_rows import (
    build_tier1_metric_rows,
    build_tier2_metric_rows,
    build_tier3_metric_rows,
    row_is_failure,
    tier1_runner_outcome,
    tier2_runner_outcome,
    unexplained_tier1_nodes,
)

# RdmaConnectivityCheck not used - using legacy function temporarily
from cvs.lib.preflight.report import PreflightReportGenerator, preflight_check_display_name
from cvs.lib.preflight.rundeck import (
    GID_CONSISTENCY,
    IFOE_L2,
    INTERFACE_NAMES,
    NODE_HEALTH,
    NODE_REACHABILITY,
    NODE_SMOKE_TIER1,
    NODE_SMOKE_TIER2,
    NODE_SMOKE_TIER3,
    RDMA_CONNECTIVITY,
    ROCM_VERSIONS,
    TRANSFERBENCH,
    build_preflight_deck,
)
from cvs.lib.report.health_lifecycle import HealthLifecycle, timed_stage
from cvs.lib.utils_lib import *
from cvs.lib.verify_lib import *
from cvs.parsers.schemas import (
    normalize_legacy_preflight_node_smoke_sections,
    normalize_legacy_preflight_rdma_config,
    validate_config_file,
)

from cvs.lib import globals

log = globals.log

# Filled by the module fixture while a pytest session is running. Library unit
# tests call the check functions directly and leave this empty.
_rundeck_slot = {'target': None, 'cluster': None}


def _publish_preflight_rundeck():
    """Refresh the Run Deck fixture from preflight_results. Never changes the verdict."""
    target = _rundeck_slot.get('target')
    if not isinstance(target, dict):
        return
    try:
        built = build_preflight_deck(preflight_results, _rundeck_slot.get('cluster'))
        target.clear()
        target.update(built)
    except Exception as exc:
        log.warning("Preflight: could not capture Run Deck results: %s", exc)


def _timed_stage(label):
    """Record wall-clock for one preflight check. Missing lifecycle is a no-op."""

    def deco(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            lifecycle = inspect.signature(fn).bind_partial(*args, **kwargs).arguments.get('lifecycle')
            with timed_stage(lifecycle, label):
                return fn(*args, **kwargs)

        return wrapped

    return deco


def get_nested_config(config_dict, section, key, default):
    """
    Get configuration value from nested structure.

    Args:
        config_dict: Full configuration dictionary
        section: Section name (e.g., 'node_check', 'connectivity_check.rdma')
        key: Parameter key within the section
        default: Default value if not found

    Returns:
        Configuration value or default
    """
    if not config_dict:
        return default

    # Handle nested sections like 'connectivity_check.rdma'
    sections = section.split('.')
    current = config_dict

    for sec in sections:
        if isinstance(current, dict) and sec in current:
            current = current[sec]
        else:
            return default

    if isinstance(current, dict) and key in current:
        return current[key]
    return default


def _config_flag_enabled(value, default=True):
    """Normalize mixed bool/string config flags."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'on')
    return bool(value)


def _reject_flat_preflight_checks(config_dict):
    if not isinstance(config_dict, dict):
        return
    removed = sorted(set(config_dict) & {'node_health', 'l2ping', 'transferbench'})
    if removed:
        raise ValueError(
            "Unsupported flat preflight block(s): "
            + ', '.join(removed)
            + "; use preflight.node_check and preflight.connectivity_check.ifoe"
        )


def _node_check_config(config_dict):
    """Return the customer-facing generic node-check configuration."""
    _reject_flat_preflight_checks(config_dict)
    config = config_dict.get('node_check', {}) if isinstance(config_dict, dict) else {}
    if not isinstance(config, dict):
        raise ValueError("preflight.node_check must be an object")
    unknown = sorted(
        key for key in set(config) - {'enabled', 'gpus_per_node', 'expected_rocm_version'} if not key.startswith('_')
    )
    if unknown:
        raise ValueError("Unsupported preflight.node_check option(s): " + ', '.join(unknown))
    return config


def _ifoe_config(config_dict):
    """Return the customer-facing IFoE configuration."""
    _reject_flat_preflight_checks(config_dict)
    config = get_nested_config(config_dict, 'connectivity_check', 'ifoe', {})
    if not isinstance(config, dict):
        raise ValueError("preflight.connectivity_check.ifoe must be an object")
    unknown = sorted(
        key for key in set(config) - {'fabric_checks', 'l2ping', 'transferbench'} if not key.startswith('_')
    )
    if unknown:
        raise ValueError("Unsupported preflight.connectivity_check.ifoe option(s): " + ', '.join(unknown))
    return config


def _rdma_config(config_dict):
    config = get_nested_config(config_dict, 'connectivity_check', 'rdma', {})
    if not isinstance(config, dict):
        raise ValueError("preflight.connectivity_check.rdma must be an object")
    return config


def _rdma_enabled(config_dict):
    return str(_rdma_config(config_dict).get('connectivity_mode', 'basic')).strip().lower() != 'skip'


def _node_health_enabled(config_dict):
    return _config_flag_enabled(_node_check_config(config_dict).get('enabled'), default=True)


def _node_health_fabric_checks_enabled(config_dict):
    enabled = _config_flag_enabled(_ifoe_config(config_dict).get('fabric_checks'), default=False)
    if enabled and not _node_health_enabled(config_dict):
        raise ValueError("preflight.connectivity_check.ifoe.fabric_checks requires node_check.enabled=true")
    return enabled


def _node_health_admission_failed(config_dict):
    if not _node_health_enabled(config_dict):
        return False
    return (preflight_results.get('node_health') or {}).get('status') == 'FAIL'


def _blocked_by_node_health(check_name):
    """Create a non-successful downstream result when the mandatory gate failed."""
    return {
        'status': 'BLOCKED',
        'blocked': True,
        'skipped': False,
        'message': f"{check_name} was not run because mandatory node-health admission failed",
    }


def _node_health_admitted_port_ids():
    """Return mask-admitted IFoE ports recorded by a passing MI4XX gate.

    AFM can report a physical link UP even when the corresponding station is
    intentionally masked.  IFoE ``ports: up`` consumes this map so it tests
    only ports that node health has admitted from the station policy.
    """
    admission = preflight_results.get('node_health') or {}
    if admission.get('status') != 'PASS' or not admission.get('fabric_checks'):
        return {}
    result = {}
    for node, node_result in (admission.get('node_results') or {}).items():
        ports = {}
        for bdf, inventory in (node_result.get('afm_port_inventory') or {}).items():
            if 'mask_enabled_port_ids' in inventory:
                ports[bdf] = list(inventory.get('mask_enabled_port_ids') or [])
        if ports:
            result[node] = ports
    return result


# Global results storage for HTML report generation
preflight_results = {}


def _prune_nodes_from_phdl(phdl, failed_nodes, reason):
    """
    Remove ``failed_nodes`` from ``phdl.reachable_hosts`` and recreate the parallel client.

    Later preflight steps then only target hosts that passed the previous check.
    """
    if not failed_nodes:
        return
    remove = {n for n in failed_nodes if n}
    on_host = [h for h in phdl.reachable_hosts if h in remove]
    if not on_host:
        return
    pruned = phdl.prune_nodes(on_host)
    if not pruned:
        return
    log.info(f"{reason} Pruned {len(pruned)} node(s) from further preflight tests: {', '.join(sorted(pruned))}")


# Recorded statuses meaning the check never executed.
_PREFLIGHT_NOT_RUN_STATUSES = ('SKIPPED', 'BLOCKED')


def _recorded_statuses(result):
    """Status strings held by a recorded result.

    Checks record either a single status dict or a per-node mapping of them.
    """
    if not isinstance(result, dict):
        return []
    if result.get('status'):
        return [str(result['status']).upper()]
    return [
        str(node_result['status']).upper()
        for node_result in result.values()
        if isinstance(node_result, dict) and node_result.get('status')
    ]


def _summary_result(result):
    """True when the dict is a check summary rather than a flat node-to-result map."""
    return any(
        key in result
        for key in (
            'status',
            'skipped',
            'message',
            'mode',
            'failed_nodes',
            'node_results',
            'nodes',
            'vpod_membership',
            'pod_membership',
            'setup_results',
        )
    )


def _nodes_with_status(node_map, status):
    if not isinstance(node_map, dict):
        return []
    return sorted(
        str(node)
        for node, node_result in node_map.items()
        if isinstance(node_result, dict) and str(node_result.get('status') or '').upper() == status
    )


def _recorded_failed_nodes(result):
    """Hostnames that failed. Summary sections such as vpod_membership are not hosts."""
    listed = result.get('failed_nodes')
    if isinstance(listed, (list, tuple)) and any(listed):
        return sorted(str(node) for node in listed if node)
    for key in ('node_results', 'nodes'):
        nested = result.get(key)
        if isinstance(nested, dict) and nested:
            return _nodes_with_status(nested, 'FAIL')
    if _summary_result(result):
        return []
    return _nodes_with_status(result, 'FAIL')


def _recorded_outcome_message(result, default):
    if result.get('message'):
        return str(result['message'])
    failed = _recorded_failed_nodes(result)
    if failed:
        return f"{default} on {len(failed)} node(s): {', '.join(failed)}"
    return default


def preflight_update_test_result(result=None):
    """Clear accumulated errors and mirror a recorded result onto this pytest row.

    Preflight stays diagnostic: a failure here never stops the checks that follow.
    But the row has to show what was recorded, so a check that was skipped by
    configuration or that failed must not report as passed. Pass the result only
    after storing it in ``preflight_results``, so the generated report keeps the
    detail even though this row ends early.
    """
    if len(globals.error_list) > 0:
        log.info(f"Preflight detected {len(globals.error_list)} issues (see detailed logs above)")
        # Clear the error list so the generic handler cannot fail the test for us.
        globals.error_list.clear()

    _publish_preflight_rundeck()

    if not isinstance(result, dict):
        return

    statuses = _recorded_statuses(result)
    if 'FAIL' in statuses:
        pytest.fail(_recorded_outcome_message(result, 'Preflight check failed'))
    if (
        result.get('skipped')
        or result.get('blocked')
        or (statuses and all(status in _PREFLIGHT_NOT_RUN_STATUSES for status in statuses))
    ):
        pytest.skip(_recorded_outcome_message(result, 'Preflight check did not run'))


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
    config_dict, compatibility_warning = normalize_legacy_preflight_rdma_config(config_dict)
    if compatibility_warning:
        log.warning(compatibility_warning)

    config_dict, smoke_warning = normalize_legacy_preflight_node_smoke_sections(config_dict)
    if smoke_warning:
        log.warning(smoke_warning)

    # Resolve path placeholders
    config_dict = resolve_test_config_placeholders(config_dict, cluster_dict)
    log.info("Loaded preflight configuration")
    log.info(config_dict)

    return config_dict


@pytest.fixture(scope="module")
def lifecycle():
    """Wall-clock of each preflight check, bound as the deck lifecycle source."""
    return HealthLifecycle()


@pytest.fixture(scope="module")
def preflight_res_dict():
    """Status-matrix results for the preflight Run Deck. The profile names this fixture."""
    return {}


@pytest.fixture(scope="module", autouse=True)
def _bind_preflight_rundeck(preflight_res_dict, cluster_dict):
    _rundeck_slot['target'] = preflight_res_dict
    _rundeck_slot['cluster'] = cluster_dict
    yield
    _publish_preflight_rundeck()
    _rundeck_slot['target'] = None
    _rundeck_slot['cluster'] = None


@_timed_stage(NODE_REACHABILITY)
def test_node_reachability(orch, lifecycle=None):
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
    # orch.exec() runs inside the container under ContainerOrchestrator; reachability must
    # probe host SSH on orch.all.
    out_dict = orch.all.exec(cmd, timeout=60)  # Generous timeout for multiprocessing coordination

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
        'nodes': {node: {'status': 'PASS' if node in reachable_nodes else 'FAIL'} for node in out_dict},
        'status': 'PASS' if len(failed_nodes) == 0 else 'WARNING',
    }

    # Drop all nodes that did not return SSH_OK (explicit prune; not only SSH client exceptions)
    _prune_nodes_from_phdl(orch.all, failed_nodes, "Reachability:")

    preflight_update_test_result(preflight_results['node_reachability'])


@_timed_stage(NODE_HEALTH)
def test_node_health(orch, config_dict, cluster_dict, lifecycle=None):
    """Perform mandatory GPU health and optional MI4XX fabric admission.

    The gate is intentionally read-only.  It records diagnostics and lets the
    remaining checks run, but downstream scale-up checks are marked BLOCKED when
    admission fails.  Results are stored before this row reports its verdict, so
    the report test still writes a complete artifact.
    """
    global preflight_results

    if not _node_health_enabled(config_dict):
        preflight_results['node_health'] = {
            'status': 'SKIPPED',
            'skipped': True,
            'fabric_checks': False,
            'message': 'Node-health admission is not enabled',
            'node_results': {},
            'vpod_membership': {'status': 'SKIPPED', 'errors': []},
        }
        preflight_update_test_result(preflight_results['node_health'])
        return

    health_config = _node_check_config(config_dict)
    fabric_checks = _node_health_fabric_checks_enabled(config_dict)
    gpus_per_node = int(health_config.get('gpus_per_node', 4))
    log.info(
        "Running mandatory node-health admission (gpus_per_node=%d, fabric_checks=%s) on %d reachable host(s)",
        gpus_per_node,
        fabric_checks,
        len(orch.all.reachable_hosts),
    )
    checker = NodeHealthCheck(
        orch,
        expected_gpus_per_node=gpus_per_node,
        fabric_checks=fabric_checks,
        config_dict=config_dict,
    )
    results = checker.run()
    declared_nodes = sorted(cluster_dict.get('node_dict', {}).keys())
    tested_nodes = sorted((results.get('node_results') or {}).keys())
    missing_nodes = sorted(set(declared_nodes) - set(tested_nodes))
    results['failure_mode'] = 'gate'
    results['coverage'] = {
        'expected_nodes': declared_nodes,
        'tested_nodes': tested_nodes,
        'missing_nodes': missing_nodes,
        'complete': not missing_nodes,
    }
    if missing_nodes:
        message = 'Mandatory node-health admission did not run on declared node(s): ' + ', '.join(missing_nodes)
        results['status'] = 'FAIL'
        results.setdefault('errors', []).append(message)
        if fabric_checks:
            membership = results.setdefault('vpod_membership', {})
            membership['status'] = 'FAIL'
            membership.setdefault('errors', []).append(message)

    preflight_results['node_health'] = results
    if results.get('status') == 'FAIL':
        log.error("Mandatory node-health admission FAILED")
        for node, result in (results.get('node_results') or {}).items():
            for error in result.get('errors') or []:
                log.error("Node %s health admission: %s", node, error)
        for error in results.get('errors') or []:
            log.error("Node-health admission: %s", error)
    elif fabric_checks:
        vpod = (results.get('vpod_membership') or {}).get('vpod_accelerators') or []
        log.info("Mandatory node-health and MI4XX fabric admission PASS; AFM vPOD accelerators=%s", vpod)
    else:
        log.info("Mandatory generic GPU node-health admission PASS")
    preflight_update_test_result(results)


@_timed_stage(ROCM_VERSIONS)
def test_rocm_version_consistency(orch, config_dict, lifecycle=None):
    """
    Test ROCm version consistency across all cluster nodes.

    Verifies that all nodes are running the expected ROCm version
    as specified in the configuration.

    Nodes that fail this check are **not** removed from ``phdl`` so the next test
    (RDMA interface consistency) still runs on the full reachability-passed set.
    """
    global preflight_results

    if not _node_health_enabled(config_dict):
        preflight_results['rocm_versions'] = {
            'status': 'SKIPPED',
            'skipped': True,
            'message': 'ROCm validation skipped because preflight.node_check is disabled',
        }
        preflight_update_test_result(preflight_results['rocm_versions'])
        return

    expected_version = get_nested_config(config_dict, 'node_check', 'expected_rocm_version', '6.2.0')
    log.info(f"Testing ROCm version consistency (expected: {expected_version})")

    version_checker = RocmVersionCheck(orch, expected_version, config_dict)
    results = version_checker.run()
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
    # Intentionally do not prune ROCm failures from phdl (see docstring).
    preflight_update_test_result(results)


@_timed_stage(IFOE_L2)
def test_ifoe_l2_connectivity(orch, config_dict, cluster_dict, lifecycle=None):
    """Run IFoE L2 before RDMA-specific interface and GID pruning.

    IFoE uses AFM/vPOD topology rather than conventional RDMA interfaces.
    Keeping this test ahead of the legacy RDMA eligibility filters ensures an
    absent or separately configured RDMA NIC cannot suppress IFoE validation.
    """
    _run_ifoe_l2_connectivity(orch, config_dict, cluster_dict)


@_timed_stage(TRANSFERBENCH)
def test_ifoe_transferbench_smoke(orch, config_dict, lifecycle=None):
    """Run TransferBench after IFoE L2 and before RDMA eligibility pruning.

    TransferBench validates the IFoE data path and must see the same
    node-health-admitted host set as L2 ping. Conventional RDMA interface or
    GID failures must not suppress this independent scale-up validation.
    """
    _run_ifoe_transferbench_smoke(orch, config_dict)


@_timed_stage(INTERFACE_NAMES)
def test_interface_name_consistency(orch, config_dict, lifecycle=None):
    """
    Test RDMA interface presence and consistency across all cluster nodes.

    Verifies that the expected RDMA interfaces are present on all nodes
    as specified in the configuration.

    Nodes that fail are removed from ``phdl`` before the GID consistency check.
    """
    global preflight_results

    if not _rdma_enabled(config_dict):
        preflight_results['interface_names'] = {
            'status': 'SKIPPED',
            'skipped': True,
            'message': 'RDMA interface validation skipped because RDMA connectivity mode is skip',
        }
        preflight_update_test_result(preflight_results['interface_names'])
        return

    expected_interfaces = get_nested_config(
        config_dict, 'connectivity_check.rdma', 'interfaces', ["rocep28s0", "rocep62s0", "rocep79s0", "rocep96s0"]
    )
    log.info(f"Testing interface presence (expected: {expected_interfaces})")

    interface_checker = InterfaceConsistencyCheck(orch, expected_interfaces, config_dict)
    results = interface_checker.run()
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

    _prune_nodes_from_phdl(orch.all, failed_nodes, "Interface consistency:")
    preflight_update_test_result(results)


@_timed_stage(GID_CONSISTENCY)
def test_gid_consistency(orch, config_dict, lifecycle=None):
    """
    Test GID consistency across specified RDMA interfaces in the cluster.

    Verifies that the specified GID index exists and has the expected type on the
    specified RDMA interfaces across all cluster nodes.

    Nodes that fail are removed from ``phdl`` before RDMA connectivity testing.
    """
    global preflight_results

    if not _rdma_enabled(config_dict):
        preflight_results['gid_consistency'] = {
            'status': 'SKIPPED',
            'skipped': True,
            'message': 'RDMA GID validation skipped because RDMA connectivity mode is skip',
        }
        preflight_update_test_result(preflight_results['gid_consistency'])
        return

    gid_index = get_nested_config(config_dict, 'connectivity_check.rdma', 'gid_index', '3')
    gid_type = get_nested_config(config_dict, 'connectivity_check.rdma', 'gid_type', DEFAULT_GID_TYPE)
    expected_interfaces = get_nested_config(
        config_dict, 'connectivity_check.rdma', 'interfaces', ["rocep28s0", "rocep62s0", "rocep79s0", "rocep96s0"]
    )
    log.info(f"Testing GID consistency for index {gid_index} (type {gid_type}) on interfaces: {expected_interfaces}")

    gid_checker = GidConsistencyCheck(orch, gid_index, expected_interfaces, config_dict, expected_gid_type=gid_type)
    results = gid_checker.run()
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

    log.info(
        f"GID consistency results: {ok_interfaces}/{total_interfaces} interfaces have a valid {gid_type} GID at index {gid_index}"
    )

    _prune_nodes_from_phdl(orch.all, failed_nodes, "GID consistency:")
    preflight_update_test_result(results)


# Each tier publishes one pytest row per catalog check. The runner test above the checks
# executes Primus once; the check rows only read the payload it stored.
_NODE_SMOKE_TIERS = {
    'tier1': (NODE_SMOKE_TIER1_LABEL, build_tier1_metric_rows, ('node_smoke_tier1', 'node_smoke')),
    'tier2': (NODE_SMOKE_TIER2_LABEL, build_tier2_metric_rows, ('node_smoke_tier1', 'node_smoke')),
    'tier3': (NODE_SMOKE_TIER3_LABEL, build_tier3_metric_rows, ('node_smoke_tier3', 'tier3_info')),
}
_node_smoke_rows_by_metric = {}


def _node_smoke_rows(tier):
    """Resolve a tier's per-check rows once and index them by catalog metric key."""
    if tier not in _node_smoke_rows_by_metric:
        _, build_rows, result_keys = _NODE_SMOKE_TIERS[tier]
        results = next((preflight_results.get(key) for key in result_keys if preflight_results.get(key)), {})
        _node_smoke_rows_by_metric[tier] = {row['metric']: row for row in build_rows(results)}
    return _node_smoke_rows_by_metric[tier]


def _format_measurement(row):
    actual = row.get('actual')
    if actual is None:
        return ''
    unit = row.get('unit') or ''
    text = f"{actual} {unit}".strip()
    spec = row.get('spec') or {}
    if spec.get('value') is not None:
        text += f" (gate {spec.get('kind', '')} {spec['value']})".replace('  ', ' ')
    return text


def _report_node_smoke_check(tier, check):
    """Turn one catalog check into this pytest row's verdict."""
    label = _NODE_SMOKE_TIERS[tier][0]
    metric = check.get('metric')
    if not metric:
        pytest.skip(f"{label}: no nodes resolved from the cluster file at collection time")

    row = _node_smoke_rows(tier).get(metric)
    if row is None:
        pytest.skip(f"{label} did not run, so '{check['label']}' has no result")

    measurement = _format_measurement(row)
    reason = row.get('reason') or ''
    log.info(
        "%s | %s | %s%s%s",
        label,
        check['label'],
        row['status'].upper(),
        f" | {measurement}" if measurement else '',
        f" | {reason}" if reason else '',
    )
    if row['status'] == 'skip':
        pytest.skip(reason or f"{label}: '{check['label']}' was not reported by Primus")
    if row['status'] not in ('pass', 'record'):
        pytest.fail(f"{label}: '{check['label']}' failed. {reason or measurement or 'No detail reported'}")


def _report_tier1_from_rows(results):
    """Fail Tier 1 only for Tier 1 checks, or a node failure Tier 2 did not record."""
    rows = build_tier1_metric_rows(results)
    tier2_rows = build_tier2_metric_rows(results) if results.get('tier2_perf') else []
    outcome = tier1_runner_outcome(results, rows, tier2_rows)
    if outcome == 'skip':
        preflight_update_test_result(
            {'skipped': True, 'message': f'{NODE_SMOKE_TIER1_LABEL}: Primus reported no Tier 1 checks'}
        )
        return

    failed = [row for row in rows if row_is_failure(row)]
    unexplained = unexplained_tier1_nodes(results, tier2_rows)
    total = results.get('total_nodes', 0)
    if failed:
        log.warning("%s FAIL on %d/%d check(s)", NODE_SMOKE_TIER1_LABEL, len(failed), len(rows))
    elif unexplained:
        log.warning(
            "%s FAIL on %d/%d node(s): %s",
            NODE_SMOKE_TIER1_LABEL,
            len(unexplained),
            total,
            ", ".join(unexplained),
        )
    else:
        log.info("%s PASS on %d/%d nodes", NODE_SMOKE_TIER1_LABEL, len(results.get('passing_nodes') or []), total)

    preflight_update_test_result()

    if failed:
        pytest.fail(f"{NODE_SMOKE_TIER1_LABEL} failed on {len(failed)}/{len(rows)} check(s)")
    if unexplained:
        pytest.fail(f"{NODE_SMOKE_TIER1_LABEL} failed on {len(unexplained)}/{total} node(s): {', '.join(unexplained)}")


def _report_tier_node_verdict(label, results):
    """Log the tier's node roll-up and fail the row when any node did not pass.

    Called only after the results are stored, so later checks and the generated report
    still see them even though this row fails.
    """
    failed_nodes = results.get('failed_nodes') or []
    unknown_nodes = results.get('unknown_nodes') or []
    total = results.get('total_nodes', 0)
    not_passing = failed_nodes + unknown_nodes

    if not_passing:
        log.warning("%s FAIL on %d/%d node(s): %s", label, len(not_passing), total, ", ".join(not_passing))
    else:
        log.info("%s PASS on %d/%d nodes", label, len(results.get('passing_nodes') or []), total)

    preflight_update_test_result()

    if not_passing:
        pytest.fail(f"{label} failed on {len(not_passing)}/{total} node(s): {', '.join(not_passing)}")


@_timed_stage(NODE_SMOKE_TIER1)
def test_node_smoke_tier1(orch, config_dict, lifecycle=None):
    """
    Run Node Smoke Tier 1 (Primus ``node_smoke``) on each reachable node via primus-cli.

    Runs by default; disable via ``node_smoke_tier1.connectivity_mode`` (legacy:
    ``node_smoke``) in the preflight config.  Uses parallel SSH — no Slurm required.

    Node Smoke Tier 2 perf sanity (``node_smoke_tier1.tier2_perf``, default on) enables
    ``--tier2-perf``: large GEMM TFLOPS floor, HBM D2D bandwidth, and local
    multi-GPU RCCL all-reduce thresholds (``gemm_tflops_min``, ``hbm_gbs_min``,
    ``rccl_gbs_min``, etc.).

    Nodes that fail are reported but are **not** pruned from ``phdl``.
    """
    global preflight_results

    if not orch.all.reachable_hosts:
        log.warning("%s skipped: no reachable hosts remain after earlier preflight pruning", NODE_SMOKE_TIER1_LABEL)
        skipped = {
            'mode': 'skip',
            'skipped': True,
            'message': f'No reachable nodes available for {NODE_SMOKE_TIER1_LABEL}',
            'node_results': {},
        }
        preflight_results['node_smoke_tier1'] = skipped
        preflight_results['node_smoke'] = skipped
        preflight_update_test_result(skipped)
        return

    node_list = list(orch.all.reachable_hosts)
    log.info("Running %s on %d reachable host(s)", NODE_SMOKE_TIER1_LABEL, len(node_list))

    checker = NodeSmokeCheck(orch, node_list, config_dict)
    results = checker.run()
    preflight_results['node_smoke_tier1'] = results
    preflight_results['node_smoke'] = results

    if results.get('skipped'):
        log.info("%s: %s", NODE_SMOKE_TIER1_LABEL, results.get('message', 'skipped'))
        preflight_update_test_result(results)
        return

    _report_tier1_from_rows(results)


def test_node_smoke_tier1_check(tier1_check):
    """One row per configured GPU and per node collector Primus reports."""
    _report_node_smoke_check('tier1', tier1_check)


@_timed_stage(NODE_SMOKE_TIER2)
def test_node_smoke_tier2(orch, config_dict, lifecycle=None):
    """Summarize Node Smoke Tier 2 perf sanity, which rides along with the Tier 1 run.

    Does not re-run Primus. Disable with ``node_smoke_tier1.tier2_perf=false``.
    """
    global preflight_results

    results = preflight_results.get('node_smoke_tier1') or preflight_results.get('node_smoke') or {}
    if not results:
        preflight_update_test_result({'skipped': True, 'message': f'{NODE_SMOKE_TIER1_LABEL} has not run'})
        return
    if results.get('skipped'):
        preflight_update_test_result({'skipped': True, 'message': f'{NODE_SMOKE_TIER1_LABEL} was skipped'})
        return
    if not results.get('tier2_perf'):
        preflight_update_test_result(
            {'skipped': True, 'message': 'Tier 2 perf sanity is off (node_smoke_tier1.tier2_perf=false)'}
        )
        return

    rows = build_tier2_metric_rows(results)
    if tier2_runner_outcome(rows) == 'skip':
        preflight_update_test_result(
            {'skipped': True, 'message': f'{NODE_SMOKE_TIER2_LABEL}: Primus reported no Tier 2 metrics'}
        )
        return

    failed = [row for row in rows if row_is_failure(row)]
    if failed:
        log.warning("%s FAIL on %d/%d check(s)", NODE_SMOKE_TIER2_LABEL, len(failed), len(rows))
    else:
        log.info("%s PASS on %d check(s)", NODE_SMOKE_TIER2_LABEL, len(rows))

    preflight_update_test_result()

    if failed:
        pytest.fail(f"{NODE_SMOKE_TIER2_LABEL} failed on {len(failed)}/{len(rows)} check(s)")


def test_node_smoke_tier2_check(tier2_check):
    """One row per Node Smoke Tier 2 catalog check (2 per GPU + 1 local RCCL)."""
    _report_node_smoke_check('tier2', tier2_check)


@_timed_stage(NODE_SMOKE_TIER3)
def test_node_smoke_tier3(orch, config_dict, lifecycle=None):
    """
    Run Node Smoke Tier 3 (Primus ``preflight --host --gpu --network``) across the cluster.

    Runs by default; disable via ``node_smoke_tier3.connectivity_mode`` (legacy:
    ``tier3_info``).  Uses parallel SSH with torchrun — no Slurm required.

    Nodes that fail are reported but are **not** pruned from ``phdl``.
    """
    global preflight_results

    if not orch.all.reachable_hosts:
        log.warning("%s skipped: no reachable hosts remain after earlier preflight pruning", NODE_SMOKE_TIER3_LABEL)
        skipped = {
            'mode': 'skip',
            'skipped': True,
            'message': f'No reachable nodes available for {NODE_SMOKE_TIER3_LABEL}',
            'node_results': {},
        }
        preflight_results['node_smoke_tier3'] = skipped
        preflight_results['tier3_info'] = skipped
        preflight_update_test_result(skipped)
        return

    node_list = list(orch.all.reachable_hosts)
    log.info("Running %s on %d reachable host(s)", NODE_SMOKE_TIER3_LABEL, len(node_list))

    checker = NodeSmokeTier3Check(orch, node_list, config_dict)
    results = checker.run()
    preflight_results['node_smoke_tier3'] = results
    preflight_results['tier3_info'] = results

    if results.get('skipped'):
        log.info("%s: %s", NODE_SMOKE_TIER3_LABEL, results.get('message', 'skipped'))
        preflight_update_test_result(results)
        return

    _report_tier_node_verdict(NODE_SMOKE_TIER3_LABEL, results)


def test_node_smoke_tier3_check(tier3_check):
    """One row per Tier 3 group Primus reports (host, GPU, network)."""
    _report_node_smoke_check('tier3', tier3_check)


def _l2ping_config(config_dict):
    """Return the customer-facing l2ping configuration."""
    config = _ifoe_config(config_dict).get('l2ping', {})
    if not isinstance(config, dict):
        raise ValueError("preflight.connectivity_check.ifoe.l2ping must be an object")
    unknown = sorted(
        key
        for key in set(config) - {'enabled', 'pings_per_port', 'loss_threshold_pct', 'ping_timeout'}
        if not key.startswith('_')
    )
    if unknown:
        raise ValueError("Unsupported preflight.connectivity_check.ifoe.l2ping option(s): " + ', '.join(unknown))
    return config


def _l2ping_enabled(config_dict):
    return _config_flag_enabled(_l2ping_config(config_dict).get('enabled'), default=False)


def _run_ifoe_l2_connectivity(orch, config_dict, cluster_dict):
    """
    Test IFoE L2 connectivity using ``afmctl test ping``.

    Runs ``afmctl test ping`` on each reachable node for every configured
    (BDF, dst-accelerator) pairing and validates the per-port pass/fail
    counts and Summary loss percentages against the configured threshold.

    Configuration lives under ``connectivity_check.ifoe.l2ping`` in the preflight config file. The
    check is opt-in: when ``enabled`` is false or omitted it records a SKIPPED
    result without contacting nodes. When enabled, l2ping is a strict
    admission gate: any IFoE failure, incomplete coverage, or missing required
    cluster node fails pytest after the structured result has been saved.
    Nodes that fail L2 ping are not pruned from ``orch.all`` so the report and
    subsequent diagnostics can still run.
    """
    global preflight_results

    if _node_health_admission_failed(config_dict):
        blocked = _blocked_by_node_health('IFoE L2 connectivity')
        blocked.update({'mode': 'blocked', 'node_results': {}, 'failure_mode': 'gate'})
        preflight_results['ifoe_l2_connectivity'] = blocked
        log.warning(blocked['message'])
        preflight_update_test_result(blocked)
        return

    l2ping_config = _l2ping_config(config_dict)
    if not _l2ping_enabled(config_dict):
        log.info("IFoE L2 connectivity test skipped because connectivity_check.ifoe.l2ping is disabled")
        preflight_results['ifoe_l2_connectivity'] = {
            'mode': 'skip',
            'skipped': True,
            'message': 'IFoE L2 connectivity test skipped by configuration',
            'node_results': {},
        }
        preflight_update_test_result(preflight_results['ifoe_l2_connectivity'])
        return

    declared_nodes = sorted(cluster_dict.get('node_dict', {}).keys())
    if not orch.all.reachable_hosts:
        message = "No reachable nodes available for IFoE L2 connectivity testing"
        log.warning("IFoE L2 connectivity skipped: no reachable hosts remain after earlier preflight pruning")
        preflight_results['ifoe_l2_connectivity'] = {
            'mode': 'run',
            'skipped': False,
            'status': 'FAIL',
            'message': message,
            'node_results': {},
            'failure_mode': 'gate',
            'coverage': {
                'expected_nodes': declared_nodes,
                'tested_nodes': [],
                'missing_nodes': declared_nodes,
                'complete': False,
            },
        }
        preflight_update_test_result(preflight_results['ifoe_l2_connectivity'])
        return

    pings_per_port = int(l2ping_config.get('pings_per_port', 3))
    if pings_per_port < 1:
        raise ValueError("preflight.connectivity_check.ifoe.l2ping.pings_per_port must be at least 1")

    loss_threshold_pct = float(l2ping_config.get('loss_threshold_pct', 0.0))
    if loss_threshold_pct < 0.0 or loss_threshold_pct > 100.0:
        raise ValueError("preflight.connectivity_check.ifoe.l2ping.loss_threshold_pct must be between 0 and 100")

    ping_timeout = int(l2ping_config.get('ping_timeout', 600))
    if ping_timeout < 30:
        raise ValueError("preflight.connectivity_check.ifoe.l2ping.ping_timeout must be at least 30 seconds")

    log.info(
        "Running strict IFoE L2 full-mesh connectivity (pings_per_port=%d, ping_timeout=%ds, loss_threshold_pct=%.2f) on %d host(s)",
        pings_per_port,
        ping_timeout,
        loss_threshold_pct,
        len(orch.all.reachable_hosts),
    )

    checker = IfoeL2ConnectivityCheck(
        orch,
        afmctl_path='afmctl',
        bdfs=[],
        dst_accelerators=[0],
        mesh_mode='full_mesh',
        ports='up',
        port_discovery='auto',
        pings_per_port=pings_per_port,
        per_ping_timeout=None,
        traffic_types=['ifoe_req', 'ifoe_resp', 'non_ifoe'],
        loss_threshold_pct=loss_threshold_pct,
        ssh_timeout=ping_timeout,
        use_sudo=True,
        json_args=['--json'],
        allow_text_fallback=False,
        skip_pass=True,
        bdf_discovery='auto',
        require_complete_coverage=True,
        strict_discovery=True,
        admitted_port_ids_by_node=_node_health_admitted_port_ids(),
        config_dict=config_dict,
    )

    node_results = checker.run()

    failed_nodes = [n for n, r in node_results.items() if r.get('status') == 'FAIL']
    tested_nodes = sorted(node_results.keys())
    missing_nodes = sorted(set(declared_nodes) - set(tested_nodes))
    incomplete_nodes = sorted(n for n, r in node_results.items() if not (r.get('coverage') or {}).get('complete', True))
    total_invocations = 0
    failed_invocations = 0
    for r in node_results.values():
        for accel_block in (r.get('accelerators') or {}).values():
            for invocation in accel_block.values():
                if invocation.get('status') == 'SKIPPED':
                    continue
                total_invocations += 1
                if invocation.get('status') == 'FAIL':
                    failed_invocations += 1

    summary_status = 'FAIL' if failed_nodes or missing_nodes or incomplete_nodes else 'PASS'
    preflight_results['ifoe_l2_connectivity'] = {
        'mode': 'run',
        'skipped': False,
        'status': summary_status,
        'node_results': node_results,
        'total_nodes': len(node_results),
        'failed_nodes': failed_nodes,
        'total_invocations': total_invocations,
        'failed_invocations': failed_invocations,
        'pings_per_port': pings_per_port,
        'loss_threshold_pct': loss_threshold_pct,
        'ping_timeout': ping_timeout,
        'traffic_types': ['ifoe_req', 'ifoe_resp', 'non_ifoe'],
        'mesh_mode': 'full_mesh',
        'ports': 'up',
        'port_discovery': 'auto',
        'failure_mode': 'gate',
        'require_complete_coverage': True,
        'strict_discovery': True,
        'coverage': {
            'expected_nodes': declared_nodes,
            'tested_nodes': tested_nodes,
            'missing_nodes': missing_nodes,
            'incomplete_nodes': incomplete_nodes,
            'complete': not missing_nodes and not incomplete_nodes,
        },
    }

    if summary_status == 'FAIL':
        preflight_results['ifoe_l2_connectivity']['message'] = (
            "IFoE L2 preflight gate failed "
            f"({len(failed_nodes)} failed node(s), {len(missing_nodes)} missing node(s), "
            f"{len(incomplete_nodes)} incomplete node(s)); see preflight report"
        )
        log.warning(
            "IFoE L2 connectivity FAIL on %d/%d tested node(s): %s",
            len(failed_nodes),
            len(node_results),
            ", ".join(failed_nodes) or 'coverage failure only',
        )
        if missing_nodes:
            log.error("IFoE L2 required nodes not tested: %s", ", ".join(missing_nodes))
        if incomplete_nodes:
            log.error("IFoE L2 incomplete coverage on node(s): %s", ", ".join(incomplete_nodes))
        for node in failed_nodes:
            errors = node_results[node].get('errors', [])
            for err in errors[:20]:
                log.error("Node %s IFoE L2: %s", node, err)
            if len(errors) > 20:
                log.error(
                    "Node %s IFoE L2: %d additional errors suppressed; see report artifacts", node, len(errors) - 20
                )
    else:
        log.info(
            "IFoE L2 connectivity PASS on %d/%d nodes (%d/%d invocations succeeded)",
            len(node_results) - len(failed_nodes),
            len(node_results),
            total_invocations - failed_invocations,
            total_invocations,
        )

    preflight_update_test_result(preflight_results['ifoe_l2_connectivity'])


def _transferbench_config(config_dict):
    """Return the customer-facing TransferBench configuration."""
    config = _ifoe_config(config_dict).get('transferbench', {})
    if not isinstance(config, dict):
        raise ValueError("preflight.connectivity_check.ifoe.transferbench must be an object")
    supported = {'enabled', 'scope', 'profile', 'message_sizes', 'iterations', 'warmup_iterations'}
    unknown = sorted(key for key in set(config) - supported if not key.startswith('_'))
    if unknown:
        raise ValueError("Unsupported preflight.connectivity_check.ifoe.transferbench option(s): " + ', '.join(unknown))
    return config


def _transferbench_enabled(config_dict):
    return _config_flag_enabled(_transferbench_config(config_dict).get('enabled'), default=False)


def _transferbench_timeout(message_sizes, iterations, warmup_iterations):
    """Derive a conservative per-invocation timeout from workload intensity."""
    workload_units = max(1, len(message_sizes)) * max(1, iterations + warmup_iterations)
    return max(600, 150 * workload_units)


def _run_ifoe_transferbench_smoke(orch, config_dict):
    """Test IFoE scale-up via TransferBench candidate-branch smoketest (AIMVT-181).

    Builds on L2 reachability via ``afmctl test ping`` by exercising the IFoE
    data path one layer above L2: it asks every reachable node to run
    the TransferBench candidate-branch ``smoketest`` preset and validates that
    the binary completes with exit code zero and no ``FAIL`` cells.

    Two precondition gates run before the binary is invoked:

      1. **vPod membership** – MI4XX consumes the mandatory AFM admission;
         generic profiles query ``amd-smi fabric --json``. The
         selected nodes must share a single vPOD (the smoketest preset itself
         exits with ``ERR_FATAL`` when ranks span multiple virtual pods).
      2. **Reachable host count** – ``multi_rank`` mode requires at least
         two reachable nodes; otherwise we degrade to ``per_node`` mode and
         log a warning.

    Configuration lives under ``connectivity_check.ifoe.transferbench`` in the preflight config file.
    The check is opt-in through ``enabled``. Once enabled, a failed run is a
    mandatory preflight gate; skip-budget warnings remain non-fatal. Failed
    nodes are not pruned from ``orch.all`` so downstream diagnostics and reporting
    can still run.
    """
    global preflight_results

    if _node_health_admission_failed(config_dict):
        blocked = _blocked_by_node_health('IFoE TransferBench smoketest')
        blocked.update({'mode': 'blocked', 'nodes': {}, 'totals': {}, 'pod_membership': {}})
        preflight_results['transferbench_smoke'] = blocked
        log.warning(blocked['message'])
        preflight_update_test_result(blocked)
        return

    transferbench_config = _transferbench_config(config_dict)
    if not _transferbench_enabled(config_dict):
        log.info("IFoE TransferBench smoketest skipped because connectivity_check.ifoe.transferbench is disabled")
        preflight_results['transferbench_smoke'] = {
            'mode': 'skip',
            'skipped': True,
            'message': 'IFoE TransferBench smoketest skipped by configuration',
            'nodes': {},
        }
        preflight_update_test_result(preflight_results['transferbench_smoke'])
        return

    if not orch.all.reachable_hosts:
        message = 'No reachable nodes available for TransferBench smoketest'
        log.warning(message)
        preflight_results['transferbench_smoke'] = {
            'mode': 'run',
            'skipped': False,
            'status': 'FAIL',
            'message': message,
            'nodes': {},
        }
        preflight_update_test_result(preflight_results['transferbench_smoke'])
        return

    scope = str(transferbench_config.get('scope', 'node')).strip().lower()
    if scope not in ('node', 'cluster'):
        raise ValueError("preflight.connectivity_check.ifoe.transferbench.scope must be 'node' or 'cluster'")
    profile = str(transferbench_config.get('profile', 'smoketest')).strip().lower()
    if profile != 'smoketest':
        raise ValueError(
            "preflight.connectivity_check.ifoe.transferbench.profile must be a CVS-supported profile: smoketest"
        )
    message_sizes = transferbench_config.get('message_sizes', ['1K', '16M'])
    if not isinstance(message_sizes, (list, tuple)) or not message_sizes:
        raise ValueError("preflight.connectivity_check.ifoe.transferbench.message_sizes must be a non-empty list")
    message_sizes = [str(size).strip() for size in message_sizes]
    if any(not size for size in message_sizes):
        raise ValueError("preflight.connectivity_check.ifoe.transferbench.message_sizes entries must not be empty")
    iterations = int(transferbench_config.get('iterations', 2))
    warmup_iterations = int(transferbench_config.get('warmup_iterations', 0))
    if iterations < 1:
        raise ValueError("preflight.connectivity_check.ifoe.transferbench.iterations must be at least 1")
    if warmup_iterations < 0:
        raise ValueError("preflight.connectivity_check.ifoe.transferbench.warmup_iterations must be at least 0")
    rank_mode = 'per_node' if scope == 'node' else 'multi_rank'
    ssh_timeout = _transferbench_timeout(message_sizes, iterations, warmup_iterations)
    afm_vpod_admission = None
    if _node_health_fabric_checks_enabled(config_dict):
        # The MI4XX gate is authoritative.  Never fall back to the unreliable
        # amd-smi fabric topology path.
        afm_vpod_admission = (preflight_results.get('node_health') or {}).get('vpod_membership')

    log.info(
        "Running TransferBench profile=%s scope=%s message_sizes=%s iterations=%d warmups=%d timeout=%ds on %d host(s)",
        profile,
        scope,
        message_sizes,
        iterations,
        warmup_iterations,
        ssh_timeout,
        len(orch.all.reachable_hosts),
    )

    checker = TransferBenchSmokeCheck(
        orch,
        tb_binary='TransferBench',
        amd_smi_binary='amd-smi',
        use_sudo=True,
        preset=profile,
        size_list=message_sizes,
        num_iterations=iterations,
        num_warmups=warmup_iterations,
        always_validate=True,
        run_parallel=True,
        use_bdma=False,
        force_single_pod=True,
        rank_mode=rank_mode,
        socket_master_port=31337,
        master_node=None,
        max_skip_pct=25.0,
        ssh_timeout=ssh_timeout,
        extra_env={},
        skip_pod_check=False,
        afm_vpod_admission=afm_vpod_admission,
        config_dict=config_dict,
    )

    results = checker.run()

    preflight_results['transferbench_smoke'] = {
        'mode': 'run',
        'skipped': False,
        'status': results.get('status'),
        'scope': scope,
        'profile': profile,
        'message_sizes': message_sizes,
        'iterations': iterations,
        'warmup_iterations': warmup_iterations,
        'ssh_timeout': ssh_timeout,
        'rank_mode': results.get('rank_mode'),
        'pod_membership': results.get('pod_membership') or {},
        'nodes': results.get('nodes') or {},
        'totals': results.get('totals') or {},
        'errors': results.get('errors') or [],
        'max_skip_pct': 25.0,
    }

    totals = results.get('totals') or {}
    if results.get('status') == 'FAIL':
        preflight_results['transferbench_smoke']['message'] = (
            'TransferBench preflight gate failed; see preflight report'
        )
        log.warning(
            "IFoE TransferBench smoketest FAIL: %d/%d node(s) failed, %d warning(s); cluster errors: %s",
            totals.get('nodes_fail', 0),
            totals.get('nodes_total', 0),
            totals.get('nodes_warning', 0),
            "; ".join(results.get('errors') or []) or 'none',
        )
        for node, node_result in (results.get('nodes') or {}).items():
            if node_result.get('status') == 'FAIL':
                for err in node_result.get('errors') or []:
                    log.error("Node %s TransferBench smoketest: %s", node, err)
    elif results.get('status') == 'WARNING':
        log.warning(
            "IFoE TransferBench smoketest WARNING: %d node(s) exceeded skip budget (max %s%%)",
            totals.get('nodes_warning', 0),
            25.0,
        )
    else:
        log.info(
            "IFoE TransferBench smoketest PASS on %d/%d node(s) (tests pass/fail/skip = %d/%d/%d)",
            totals.get('nodes_pass', 0),
            totals.get('nodes_total', 0),
            totals.get('tests_pass', 0),
            totals.get('tests_fail', 0),
            totals.get('tests_skip', 0),
        )

    preflight_update_test_result(preflight_results['transferbench_smoke'])


@_timed_stage(RDMA_CONNECTIVITY)
def test_rdma_connectivity(orch, cluster_dict, config_dict, lifecycle=None):
    """
    Test RDMA connectivity between cluster nodes using ibv_rc_pingpong.

    Uses direct IB verbs (same as RCCL) for more accurate connectivity testing
    that can detect issues that rping might miss.

    Tests connectivity based on the specified mode (basic, full_mesh, or skip)
    and reports any connection failures.

    ``phdl`` excludes nodes that failed reachability, interface consistency, or GID
    consistency; those steps prune before the next. ROCm version mismatches are reported
    but **not** pruned. Results may include ``excluded_nodes_interface_check`` and
    ``excluded_nodes_gid`` for the report (hosts already removed from ``phdl``).
    """
    global preflight_results

    mode = get_nested_config(config_dict, 'connectivity_check.rdma', 'connectivity_mode', 'basic')
    if mode == 'skip':
        preflight_results['rdma_connectivity'] = {
            'status': 'SKIPPED',
            'mode': 'skip',
            'skipped': True,
            'message': 'RDMA interface, GID, and connectivity validation skipped by configuration',
            'total_pairs': 0,
            'successful_pairs': 0,
            'failed_pairs': 0,
            'pair_results': {},
            'node_status': {},
        }
        log.info("RDMA interface, GID, and connectivity validation skipped by configuration")
        preflight_update_test_result(preflight_results['rdma_connectivity'])
        return

    if _node_health_admission_failed(config_dict):
        blocked = _blocked_by_node_health('RDMA connectivity')
        blocked.update(
            {
                'mode': 'blocked',
                'total_pairs': 0,
                'successful_pairs': 0,
                'failed_pairs': 0,
                'pair_results': {},
                'node_status': {},
            }
        )
        preflight_results['rdma_connectivity'] = blocked
        log.warning(blocked['message'])
        preflight_update_test_result(blocked)
        return

    # Host list matches prior-step pruning (reachability, interface, GID); not full cluster_dict.
    node_list = list(orch.all.reachable_hosts)

    iface_results = preflight_results.get('interface_names') or {}
    excluded_nodes_interface_check = sorted(
        n for n, r in iface_results.items() if isinstance(r, dict) and r.get('status') == 'FAIL'
    )

    gid_results = preflight_results.get('gid_consistency') or {}
    excluded_nodes_gid = sorted(n for n, r in gid_results.items() if isinstance(r, dict) and r.get('status') == 'FAIL')

    log.info(
        f"RDMA connectivity: {len(node_list)} host(s) after reachability / interface / GID pruning "
        f"(ROCm mismatches are not pruned)."
    )

    port_range = get_nested_config(config_dict, 'connectivity_check.rdma', 'ibv_test_port_range', '10000-50000')
    timeout = int(get_nested_config(config_dict, 'connectivity_check.rdma', 'ibv_test_timeout', 90))
    expected_interfaces = get_nested_config(
        config_dict,
        'connectivity_check.rdma',
        'interfaces',
        ["rocep28s0", "rocep62s0", "rocep79s0", "rocep96s0"],
    )
    gid_index = get_nested_config(config_dict, 'connectivity_check.rdma', 'gid_index', '3')
    parallel_group_size = get_nested_config(
        config_dict,
        'connectivity_check.rdma',
        'nodes_per_full_mesh_group',
        get_nested_config(
            config_dict,
            'connectivity_check.rdma',
            'parallel_group_size',
            get_nested_config(config_dict, 'parallelism', 'parallel_group_size', 128),
        ),
    )

    log.info(
        f"Testing RDMA connectivity using parallel algorithm (mode: {mode}, group_size: {parallel_group_size}, timeout: {timeout}s, interfaces: {expected_interfaces}, GID: {gid_index})"
    )

    if len(orch.all.reachable_hosts) < 2:
        log.warning(
            'RDMA connectivity skipped: fewer than 2 hosts remain after reachability / interface / GID pruning.'
        )
        skip_results = {
            'mode': mode,
            'skipped': True,
            'message': 'Too few nodes for RDMA after reachability, interface, and GID pruning',
            'total_pairs': 0,
            'successful_pairs': 0,
            'failed_pairs': 0,
            'pair_results': {},
            'node_status': {},
            'excluded_nodes_interface_check': excluded_nodes_interface_check,
            'excluded_nodes_gid': excluded_nodes_gid,
        }
        preflight_results['rdma_connectivity'] = skip_results
        preflight_update_test_result(skip_results)
        return

    # Use the new modular RDMA connectivity check (supports all modes)
    from cvs.lib.preflight.rdma_connectivity import RdmaConnectivityCheck

    rdma_checker = RdmaConnectivityCheck(
        orch, node_list, mode, port_range, timeout, expected_interfaces, gid_index, parallel_group_size, config_dict
    )
    results = rdma_checker.run()
    if excluded_nodes_interface_check:
        results['excluded_nodes_interface_check'] = excluded_nodes_interface_check
    if excluded_nodes_gid:
        results['excluded_nodes_gid'] = excluded_nodes_gid
    preflight_results['rdma_connectivity'] = results

    # Handle skipped test
    if results.get('skipped', False):
        log.info("RDMA connectivity test skipped by configuration")
        preflight_update_test_result(results)
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
    if results['failed_pairs']:
        results['status'] = 'FAIL'
        results.setdefault(
            'message', f"RDMA connectivity failed on {results['failed_pairs']}/{results['total_pairs']} pair(s)"
        )
    preflight_update_test_result(results)


def test_generate_preflight_report(orch, config_dict, request):
    """
    Generate comprehensive preflight check report.

    Creates a summary of all preflight check results and generates
    an HTML report for easy review.
    """
    global preflight_results

    log.info("Generating preflight check report")

    # Ensure we have results from all checks
    required_checks = [
        'node_health',
        'gid_consistency',
        'rocm_versions',
        'interface_names',
        'node_smoke_tier1',
        'node_smoke_tier3',
        'ifoe_l2_connectivity',
        'transferbench_smoke',
        'rdma_connectivity',
    ]
    missing_checks = [check for check in required_checks if check not in preflight_results]

    if missing_checks:
        log.error(f"Missing results for checks: {', '.join(missing_checks)}")
        # Create empty results for missing checks to allow report generation
        for check in missing_checks:
            preflight_results[check] = {'status': 'SKIPPED', 'message': 'Check was skipped due to earlier failures'}

    # Generate comprehensive summary using new report generator
    report_generator = PreflightReportGenerator(orch, preflight_results, config_dict)
    report_results = report_generator.run()
    summary = report_results['summary']

    preflight_results['summary'] = summary

    # Log summary to console
    log.info("=== PREFLIGHT CHECK SUMMARY ===")
    for check_name, check_summary in summary['checks'].items():
        status_icon = "✅" if check_summary['status'] == 'PASS' else "❌"
        log.info(
            f"{status_icon} {preflight_check_display_name(check_name)}: {check_summary['status']} - {check_summary['summary']}"
        )

    log.info(f"\nOverall Status: {summary['overall_status']}")

    if summary['recommendations']:
        log.info("\nRecommendations:")
        for i, recommendation in enumerate(summary['recommendations'], 1):
            log.info(f"{i}. {recommendation}")

    # Generate HTML report
    html_report_path = None
    try:
        if _config_flag_enabled(get_nested_config(config_dict, 'reporting', 'generate_html_report', 'true')):
            # HTML report is generated as part of the report generator run() above
            html_report_path = report_results.get('html_report')
            log.info(f"HTML report generated: {html_report_path}")
            rdma_csv = report_results.get('rdma_pairs_csv')
            if rdma_csv:
                log.info(f"RDMA pairs CSV generated: {rdma_csv}")
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

            rdma_csv = report_results.get('rdma_pairs_csv')
            if rdma_csv:
                try:
                    csv_copied = request.config._html_report_manager.add_html_to_report(
                        rdma_csv, link_name="RDMA failed pairs (CSV)", request=request
                    )
                    if csv_copied:
                        log.info(f'RDMA failed pairs CSV added to report bundle: {csv_copied}')
                except Exception as e:
                    log.warning(f"Failed to add RDMA CSV to report bundle: {e}")

            if copied_path:
                log.info(f'Preflight report saved and added to report bundle: {copied_path}')
            else:
                log.info(
                    f'Preflight report is saved under {html_report_path}, please copy it to your web server under /var/www/html folder to view'
                )
        except Exception as e:
            log.warning(f"Failed to add preflight report to bundle: {e}")
            log.info(f"Preflight report available at: {html_report_path}")

    # A mandatory node-health admission failure is deliberately reported only after
    # report artifacts are written, so downstream checks can be marked
    # BLOCKED with actionable diagnostics rather than disappearing from the
    # preflight output.
    node_health_results = preflight_results.get('node_health') or {}
    if _node_health_enabled(config_dict) and node_health_results.get('status') != 'PASS':
        pytest.fail("Mandatory node-health admission gate failed; see preflight report")

    # Report overall status but don't fail generic/report-only preflight profiles.
    if summary['overall_status'] == 'FAIL':
        log.warning("One or more preflight checks detected issues - see detailed results above")
        log.info("Preflight report generated successfully - review HTML report for detailed analysis")
    else:
        log.info("All preflight checks passed - cluster is ready for performance testing")
    preflight_update_test_result()
