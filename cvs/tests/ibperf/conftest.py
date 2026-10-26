'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import json
import re

import pytest

from cvs.core.orchestrators.factory import OrchestratorConfig, OrchestratorFactory
from cvs.lib import globals, ibperf_lib, ibperf_reporting
from cvs.lib.utils_lib import resolve_cluster_config_placeholders, resolve_test_config_placeholders

log = globals.log
_RESULTS_KEY = pytest.StashKey[dict]()


@pytest.fixture(scope="module")
def cluster_dict(pytestconfig):
    cluster_file = pytestconfig.getoption("cluster_file")
    if not cluster_file:
        pytest.fail("--cluster_file is required")
    with open(cluster_file) as fp:
        d = resolve_cluster_config_placeholders(json.load(fp))
    log.info('Loaded cluster config: %d nodes, user=%s', len(d.get('node_dict', {})), d.get('username'))
    log.debug('Cluster config: %s', d)
    return d


@pytest.fixture(scope="module")
def config_dict(pytestconfig, cluster_dict):
    config_file = pytestconfig.getoption("config_file")
    if not config_file:
        pytest.fail("--config_file is required")
    with open(config_file) as fp:
        d = resolve_test_config_placeholders(json.load(fp)['ibperf'], cluster_dict)
    log.info(
        'Loaded ibperf config: install_dir=%s, msg_sizes=%s, qp_counts=%s, bw_tests=%s, lat_tests=%s',
        d.get('install_dir'),
        d.get('msg_size_list'),
        d.get('qp_count_list'),
        d.get('ib_bw_test_list'),
        d.get('ib_lat_test_list'),
    )
    if re.search('True', d.get('verify_bw', ''), re.I):
        for path in ibperf_lib.find_unmatched_thresholds(
            d.get('expected_results', {}), d.get('msg_size_list', []), d.get('qp_count_list', [])
        ):
            log.warning(
                'expected_results.%s is not covered by msg_size_list/qp_count_list and will not be checked', path
            )
    log.debug('Ibperf config: %s', d)
    return d


class IbperfOrch:
    """The orchestrator plus the exec_cmd_list/prune_nodes that ibperf_lib calls on the handle it is given."""

    def __init__(self, orch):
        self._orch = orch

    def __getattr__(self, name):
        return getattr(self._orch, name)

    def exec_cmd_list(self, cmd_list, timeout=None, print_console=True):
        # ContainerOrchestrator runs these inside the containers; BaremetalOrchestrator has no exec_cmd_list.
        run = getattr(self._orch, 'exec_cmd_list', None) or self._orch.all.exec_cmd_list
        return run(cmd_list, timeout=timeout, print_console=print_console)

    def prune_nodes(self, nodes_to_remove):
        return self._orch.all.prune_nodes(nodes_to_remove)


@pytest.fixture(scope="module")
def orch(cluster_dict):
    """Suite-local orchestrator over every cluster node for installation and verification."""
    node_dict = cluster_dict['node_dict']
    if len(node_dict) < 2:
        raise ValueError('At least 2 nodes are required to run this test')
    cfg = OrchestratorConfig.from_configs({**cluster_dict, 'node_dict': node_dict}, {})
    o = OrchestratorFactory.create_orchestrator(log, cfg)
    # No setup_sshd(): perftest pairs connect over TCP without MPI.
    if o.orchestrator_type == 'container' and not o.setup_containers():
        o.close()
        pytest.fail('Failed to launch the ibperf container on every node')
    yield IbperfOrch(o)
    if o.orchestrator_type == 'container':
        o.teardown_containers()
    o.close()


@pytest.fixture(scope="module")
def cvs_results_dict(request):
    """The Run Deck store, shared with pytest_runtest_makereport whichever of them needs it first."""
    return request.config.stash.setdefault(_RESULTS_KEY, {})


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Record every failed or errored test in the Run Deck, so a test the deck does not chart still fails it."""
    outcome = yield
    report = outcome.get_result()
    if report.failed:
        reason = call.excinfo.exconly() if call.excinfo else report.outcome
        results = item.config.stash.setdefault(_RESULTS_KEY, {})
        ibperf_reporting.record_test_failure(results, item.name, report.when, reason)


@pytest.fixture(scope="module")
def ibperf_variant_config(config_dict, orch):
    """Run-card fields for the ibperf Run Deck; the tests add dmabuf and nic_count once probed."""
    return {
        'hosts': list(orch.hosts),
        'node_count': len(orch.hosts),
        'ib_bw_test_list': ibperf_lib.configured_bw_tests(config_dict),
        'ib_lat_test_list': ibperf_lib.configured_lat_tests(config_dict),
        'msg_size_list': config_dict.get('msg_size_list'),
        'qp_count_list': config_dict.get('qp_count_list'),
        'duration': config_dict.get('duration'),
        'expected_results': config_dict.get('expected_results'),
        'verify_bw': config_dict.get('verify_bw'),
    }
