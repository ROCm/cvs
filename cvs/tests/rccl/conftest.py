'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import json
from pathlib import Path

import pytest

from cvs.lib import globals
from cvs.lib.report.benchmark_metric_registry import (
    benchmark_metric_rows_from_item,
    benchmark_metric_rows_from_report,
    mark_collapsible_result_cell,
    patch_benchmark_metrics_into_html,
    stamp_benchmark_metric_rows_on_report,
)
from cvs.lib.report.profiles.hooks.rccl_session import variant_from_config
from cvs.lib.report.render.perf_metric_table import is_benchmark_metrics_extra, render_benchmark_metrics_html
from cvs.lib.report.subtest_reports import called_from_subtest_context, is_subtest_report
from cvs.lib.utils_lib import resolve_cluster_config_placeholders, resolve_test_config_placeholders
from cvs.tests.rccl._case_report import RCCL_CASE_TESTS

log = globals.log


def _is_case_report(report):
    test_name = report.nodeid.rsplit('::', 1)[-1].split('[', 1)[0]
    return report.when == 'call' and test_name in RCCL_CASE_TESTS and not is_subtest_report(report)


def _is_full_log_extra(extra):
    return isinstance(extra, dict) and extra.get('format_type') == 'url' and extra.get('name') == 'Full Log'


def _attach_case_panel(report, rows):
    """Keep the parent's Full Log link and attach the complete case panel."""
    if not rows:
        return
    try:
        from pytest_html import extras as pytest_html_extras
    except ImportError:
        return
    extras = []
    has_full_log = False
    for extra in getattr(report, 'extras', []) or []:
        if is_benchmark_metrics_extra(extra) or (has_full_log and _is_full_log_extra(extra)):
            continue
        has_full_log = has_full_log or _is_full_log_extra(extra)
        extras.append(extra)
    extras.append(pytest_html_extras.html(render_benchmark_metrics_html(rows)))
    report.extras = extras
    stamp_benchmark_metric_rows_on_report(report, rows)


@pytest.hookimpl(hookwrapper=True, trylast=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if called_from_subtest_context() or not _is_case_report(report):
        return
    _attach_case_panel(report, benchmark_metric_rows_from_item(item))


@pytest.hookimpl(hookwrapper=True, trylast=True)
def pytest_runtest_logreport(report):
    yield
    if _is_case_report(report):
        _attach_case_panel(report, benchmark_metric_rows_from_report(report))


@pytest.hookimpl(trylast=True)
def pytest_html_results_table_html(report, data):
    if _is_case_report(report) and benchmark_metric_rows_from_report(report):
        del data[:]


@pytest.hookimpl(trylast=True)
def pytest_html_results_table_row(report, cells):
    if _is_case_report(report) and benchmark_metric_rows_from_report(report):
        cells[0] = mark_collapsible_result_cell(str(cells[0]))


@pytest.hookimpl(hookwrapper=True, trylast=True)
def pytest_sessionfinish(session, exitstatus):
    """Expand RCCL parent rows into per-case rows in the final HTML report."""
    yield
    htmlpath = getattr(session.config.option, 'htmlpath', None)
    if htmlpath:
        for test_name in RCCL_CASE_TESTS:
            patch_benchmark_metrics_into_html(Path(htmlpath), benchmark_test_name=test_name)


@pytest.fixture(scope="module")
def cluster_file(pytestconfig):
    return pytestconfig.getoption("cluster_file")


@pytest.fixture(scope="module")
def config_file(pytestconfig):
    return pytestconfig.getoption("config_file")


@pytest.fixture(scope="module")
def cluster_dict(cluster_file):
    with open(cluster_file) as json_file:
        cluster_dict = json.load(json_file)
    cluster_dict = resolve_cluster_config_placeholders(cluster_dict)
    log.info("%s", cluster_dict)
    return cluster_dict


@pytest.fixture(scope="module")
def config_dict(config_file, cluster_dict):
    with open(config_file) as json_file:
        config_dict_t = json.load(json_file)
    config_dict = resolve_test_config_placeholders(config_dict_t['rccl'], cluster_dict)
    log.info("%s", config_dict)
    return config_dict


@pytest.fixture(scope="module")
def node_list(cluster_dict):
    return list(cluster_dict['node_dict'])


@pytest.fixture(scope="module")
def cvs_results_dict():
    return {}


@pytest.fixture(scope="module")
def variant_config(request):
    try:
        return variant_from_config(
            request.getfixturevalue("config_dict"),
            request.getfixturevalue("cluster_dict"),
            suite_name=request.module.__name__.rsplit(".", 1)[-1],
            raw_results=getattr(request.module, "rccl_res_dict", None),
            run_nodes=getattr(request.module, "rccl_run_nodes", None),
        )
    except Exception:
        log.warning("RCCL Run Deck variant metadata unavailable", exc_info=True)
        return None


@pytest.fixture(scope="module")
def vpc_node_list(cluster_dict):
    return [cluster_dict['node_dict'][node]['vpc_ip'] for node in cluster_dict['node_dict']]
