'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import json
import os

from cvs.lib.preflight.node_smoke_counts import DEFAULT_GPUS_PER_NODE
from cvs.lib.preflight.node_smoke_rows import (
    tier1_check_catalog,
    tier2_check_catalog,
    tier3_check_catalog_entries,
)

# Argument name -> catalog builder for the per-check Node Smoke test rows.
NODE_SMOKE_CHECK_ARGS = ('tier1_check', 'tier2_check', 'tier3_check')

# Tier runner -> the per-check test that reports the same outcome in finer detail.
# The runner still executes Primus; its report row is redundant once the check rows
# are present, so it is dropped from the HTML rather than listed alongside them.
NODE_SMOKE_RUNNER_ROWS = {
    'test_node_smoke_tier1': 'test_node_smoke_tier1_check',
    'test_node_smoke_tier2': 'test_node_smoke_tier2_check',
    'test_node_smoke_tier3': 'test_node_smoke_tier3_check',
}

_collected_test_names = set()


def _load_json(path):
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding='utf-8') as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def _collection_topology(config):
    """Read node list and GPU count straight from the CLI JSON files.

    Runs during collection, before the fixtures that validate and normalize these
    files exist, so it stays tolerant of anything it cannot parse.
    """
    cluster = _load_json(config.getoption('cluster_file', default=None))
    hosts = [host for host in (cluster.get('node_dict') or {}) if host]

    preflight = _load_json(config.getoption('config_file', default=None)).get('preflight') or {}
    gpus_per_node = DEFAULT_GPUS_PER_NODE
    for section in ('node_smoke_tier1', 'node_smoke', 'node_check'):
        value = (preflight.get(section) or {}).get('gpus_per_node')
        if value:
            gpus_per_node = int(value)
            break
    return hosts, gpus_per_node


def _check_catalog(argname, hosts, gpus_per_node):
    if argname == 'tier3_check':
        return tier3_check_catalog_entries()
    if argname == 'tier2_check':
        return tier2_check_catalog(hosts, gpus_per_node)
    return tier1_check_catalog(hosts, gpus_per_node)


def pytest_generate_tests(metafunc):
    """Expand the Node Smoke tiers into one pytest row per catalog check."""
    argname = next((arg for arg in NODE_SMOKE_CHECK_ARGS if arg in metafunc.fixturenames), None)
    if argname is None:
        return

    hosts, gpus_per_node = _collection_topology(metafunc.config)
    catalog = _check_catalog(argname, hosts, gpus_per_node)
    if not catalog:
        # No resolvable nodes: keep one row so the tier is still visible in the report.
        catalog = [{'node': '', 'metric': '', 'label': argname, 'id': 'no-nodes-resolved'}]

    metafunc.parametrize(argname, catalog, ids=[entry['id'] for entry in catalog])


def pytest_collection_modifyitems(items):
    _collected_test_names.clear()
    _collected_test_names.update(item.originalname or item.name for item in items)


def pytest_html_results_table_row(report, cells):
    """Drop a tier runner's row when its per-check rows are in the same report."""
    name = report.nodeid.rsplit('::', 1)[-1].split('[', 1)[0]
    if NODE_SMOKE_RUNNER_ROWS.get(name) in _collected_test_names:
        cells.clear()
