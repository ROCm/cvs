'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import json
from types import SimpleNamespace

import pytest

from cvs.core.orchestrators.factory import OrchestratorConfig, OrchestratorFactory
from cvs.lib import globals
from cvs.lib.topology import nic_runner
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
    return d


@pytest.fixture(scope="module")
def config_dict(pytestconfig, cluster_dict):
    config_file = pytestconfig.getoption("config_file")
    if not config_file:
        pytest.fail("--config_file is required")
    with open(config_file) as fp:
        d = resolve_test_config_placeholders(json.load(fp)['topology'], cluster_dict)
    log.info(
        'Loaded topology config: coverage=%s, flag_below_pct_of_median=%s, traffic_class=%s',
        d.get('coverage'),
        d.get('flag_below_pct_of_median'),
        d.get('traffic_class'),
    )
    return d


@pytest.fixture(scope="module")
def settings(config_dict, cluster_dict):
    """Validated run settings; a bad config fails here rather than mid-run."""
    return nic_runner.build_settings(config_dict, cluster_dict)


@pytest.fixture(scope="module")
def lifecycle():
    """Per-module container state: ``failed`` skips the tests after a failed launch,
    ``torn_down`` stops the orch fixture's leak guard once test_teardown_containers ran."""
    return SimpleNamespace(failed=False, torn_down=False)


@pytest.fixture(autouse=True)
def skip_after_failed_launch(request, lifecycle):
    if lifecycle.failed and request.node.name != 'test_teardown_containers':
        pytest.skip('topology container launch failed')


def launch_containers(orch, lifecycle):
    if orch.orchestrator_type != 'container':
        pytest.skip('baremetal orchestrator: no container to launch')
    lifecycle.failed = True
    if not orch.setup_containers():
        pytest.fail('setup_containers() returned False')
    lifecycle.failed = False


def teardown_containers(orch, lifecycle):
    lifecycle.torn_down = True
    if orch.orchestrator_type != 'container':
        pytest.skip('baremetal orchestrator: no container to tear down')
    if not orch.teardown_containers():
        pytest.fail('teardown_containers() returned False')


def close_orch(o, lifecycle):
    try:
        if o.orchestrator_type == 'container' and not lifecycle.torn_down:
            log.info('orch fixture leak-guard: tearing down containers (test_teardown_containers did not run)')
            o.teardown_containers()
    finally:
        o.close()


@pytest.fixture(scope="module")
def orch(cluster_dict, lifecycle):
    cfg = OrchestratorConfig.from_configs(cluster_dict, {})
    o = OrchestratorFactory.create_orchestrator(log, cfg)
    yield o
    close_orch(o, lifecycle)


@pytest.fixture(scope="module")
def cvs_results_dict(request):
    """The Run Deck store, shared with whichever consumer needs it first."""
    return request.config.stash.setdefault(_RESULTS_KEY, {})


@pytest.fixture(scope="module")
def nic_bandwidth_variant_config(settings, orch):
    """Run-card fields for the NIC bandwidth Run Deck."""
    return {
        'hosts': list(orch.hosts),
        'node_count': len(orch.hosts),
        'coverage': settings['coverage'],
        'traffic_class': settings['traffic_class'],
        'flag_below_pct_of_median': settings['flag_below_pct_of_median'],
        'message_size': settings['message_size'],
        'queue_pairs': settings['queue_pairs'],
        'iterations': settings['iterations'],
        'warmup_iterations': settings['warmup_iterations'],
    }
