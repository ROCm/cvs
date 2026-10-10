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
from cvs.lib import globals, ibperf_lib
from cvs.lib.utils_lib import resolve_cluster_config_placeholders, resolve_test_config_placeholders

log = globals.log


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


@pytest.fixture(scope="module")
def orch(cluster_dict):
    """Suite-local orchestrator over an even number of nodes.

    perftest runs as server/client pairs, so an odd last node is left out of
    ``orch.all``; the shared ``orch`` fixture cannot trim nodes.
    """
    node_dict = cluster_dict['node_dict']
    if len(node_dict) < 2:
        raise ValueError('At least 2 nodes are required to run this test')
    if len(node_dict) % 2 != 0:
        dropped = list(node_dict)[-1]
        log.info(
            'Odd number of nodes (%d); excluding last node %s to form server/client pairs', len(node_dict), dropped
        )
        node_dict = {node: info for node, info in node_dict.items() if node != dropped}
    cfg = OrchestratorConfig.from_configs({**cluster_dict, 'node_dict': node_dict}, {})
    o = OrchestratorFactory.create_orchestrator(log, cfg)
    yield o
    o.close()
