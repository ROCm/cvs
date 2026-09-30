'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import json
import os
import re

from cvs.lib.preflight.node_smoke_counts import DEFAULT_GPUS_PER_NODE
from cvs.lib.preflight.node_smoke_rows import (
    tier1_check_catalog,
    tier2_check_catalog,
    tier3_check_catalog_entries,
    tier_runner_row_hidden,
)

# Argument name -> catalog builder for the per-check Node Smoke test rows.
NODE_SMOKE_CHECK_ARGS = ('tier1_check', 'tier2_check', 'tier3_check')

# Tier runner -> the per-check test that reports the same outcome in finer detail.
# The runner still executes Primus. Its HTML row is dropped only when a check row
# already shows the same verdict, so a tier-level failure is not deleted with it.
NODE_SMOKE_RUNNER_ROWS = {
    'test_node_smoke_tier1': 'test_node_smoke_tier1_check',
    'test_node_smoke_tier2': 'test_node_smoke_tier2_check',
    'test_node_smoke_tier3': 'test_node_smoke_tier3_check',
}

_collected_test_names = set()
_check_outcomes = {}
_runner_outcomes = {}
_RESULT_CELL = re.compile(r'class="col-result">([^<]+)')


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
    _check_outcomes.clear()
    _runner_outcomes.clear()
    _collected_test_names.update(item.originalname or item.name for item in items)


def _test_name(nodeid):
    return nodeid.rsplit('::', 1)[-1].split('[', 1)[0]


def _remember_outcome(report):
    """Record call/setup verdicts. Check rows finish after the runner row is stored."""
    name = _test_name(report.nodeid)
    if name in NODE_SMOKE_RUNNER_ROWS.values():
        bucket = _check_outcomes.setdefault(name, {'failed': False, 'passed': False})
        if report.failed:
            bucket['failed'] = True
        elif report.when == 'call' and report.passed:
            bucket['passed'] = True
        return
    if name not in NODE_SMOKE_RUNNER_ROWS:
        return
    if report.failed:
        _runner_outcomes[name] = 'failed'
    elif report.when == 'call' and _runner_outcomes.get(name) != 'failed':
        if report.passed:
            _runner_outcomes[name] = 'passed'
        elif report.skipped:
            _runner_outcomes[name] = 'skipped'


def pytest_runtest_logreport(report):
    _remember_outcome(report)


def _stored_outcome(entry):
    for cell in entry.get('resultsTableRow') or []:
        match = _RESULT_CELL.search(str(cell))
        if match:
            return match.group(1).strip().lower()
    return ''


def drop_hidden_node_smoke_runners(tests, outcome_counts, runner_outcomes, check_outcomes, collected_checks):
    """Remove runner rows whose verdict is already on a check row.

    pytest-html stores a row when that test finishes. The tier runner finishes
    before its check rows, so the row hook cannot know whether they failed.
    """
    for nodeid in list(tests):
        name = _test_name(nodeid)
        companion = NODE_SMOKE_RUNNER_ROWS.get(name)
        if companion not in collected_checks:
            continue
        bucket = check_outcomes.get(companion) or {'failed': False, 'passed': False}
        runner_outcome = runner_outcomes.get(name, 'skipped')
        if not tier_runner_row_hidden(runner_outcome, True, bucket['failed'], bucket['passed']):
            continue
        for entry in tests.pop(nodeid):
            label = _stored_outcome(entry)
            if outcome_counts.get(label, 0) > 0:
                outcome_counts[label] -= 1


def _html_report_data(config):
    try:
        from pytest_html.basereport import BaseReport
    except ImportError:
        return None
    for plugin in config.pluginmanager.get_plugins():
        if isinstance(plugin, BaseReport):
            return plugin._report
    return None


def pytest_sessionfinish(session):
    report_data = _html_report_data(session.config)
    if report_data is None:
        return
    counts = {name: bucket['value'] for name, bucket in report_data.outcomes.items()}
    drop_hidden_node_smoke_runners(
        report_data.data['tests'],
        counts,
        _runner_outcomes,
        _check_outcomes,
        _collected_test_names,
    )
    for name, bucket in report_data.outcomes.items():
        bucket['value'] = counts.get(name, bucket['value'])
