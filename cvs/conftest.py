"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

import importlib.metadata
import logging
import sys
from pathlib import Path

import pytest

from cvs.lib.report_plugins import HtmlReportManager

log = logging.getLogger(__name__)


def _sync_suite_name_from_args(config):
    """Derive suite stem from the first ``*.py`` target in ``config.args``."""
    suite_name = "test"
    for arg in config.args:
        bare = arg.split("::")[0]
        if not bare.startswith("-") and bare.endswith(".py"):
            suite_name = Path(bare).stem
            break
    config._suite_name = suite_name
    config._test_html_dir = f"{suite_name}_html"


def _ensure_html_report_manager(config):
    """Create ``HtmlReportManager`` once; safe if ``pytest_configure`` did not run."""
    _sync_suite_name_from_args(config)
    mgr = getattr(config, "_html_report_manager", None)
    if mgr is not None:
        return mgr

    config._html_report_manager = HtmlReportManager(config)
    return config._html_report_manager


def _auto_register_inference_suite_report(config):
    from cvs.lib.report.auto_register import try_auto_register_inference_suite_report

    _sync_suite_name_from_args(config)
    return try_auto_register_inference_suite_report(config)


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    """Single hook: manager first, then auto-register after suite ``pytest_configure`` hooks."""
    _ensure_html_report_manager(config)
    _auto_register_inference_suite_report(config)


@pytest.fixture(scope="module", autouse=True)
def _cvs_inference_suite_report_module(request):
    """Bind module-scoped suite fixtures into the session store at module teardown.

    ``inf_res_dict``, ``variant_config``, and ``lifecycle`` are module fixtures;
    ``bind_session_results()`` writes into a session-level store consumed at
    ``pytest_sessionfinish`` when generating the run deck.
    """
    from cvs.lib.report.registry import bind_session_results, get_suite_report_config
    from cvs.lib.report.types import InferenceReportConfig

    if not isinstance(get_suite_report_config(request.config), InferenceReportConfig):
        yield
        return

    inf_res_dict = None
    variant_config = None
    lifecycle = None
    try:
        inf_res_dict = request.getfixturevalue("inf_res_dict")
    except pytest.FixtureLookupError:
        log.warning(
            "Inference suite report preset registered but inf_res_dict fixture is missing; "
            "session-end report will be skipped"
        )
        yield
        return
    try:
        variant_config = request.getfixturevalue("variant_config")
    except pytest.FixtureLookupError:
        log.warning(
            "Inference suite report preset registered but variant_config fixture is missing; "
            "session-end report will be skipped"
        )
        yield
        return
    try:
        lifecycle = request.getfixturevalue("lifecycle")
    except pytest.FixtureLookupError:
        log.warning(
            "Inference suite report preset registered but lifecycle fixture is missing; "
            "session-end report will be skipped"
        )
        yield
        return

    def _bind_at_module_end():
        bind_session_results(
            inf_res_dict=inf_res_dict,
            variant_config=variant_config,
            lifecycle=lifecycle,
        )

    request.addfinalizer(_bind_at_module_end)
    yield


# Add all additional cmd line arguments for the script
def pytest_addoption(parser):
    # Check if options already exist (they might be added by cvs core package)
    try:
        parser.addoption(
            "--cluster_file",
            action="store",
            required=True,
            help="Input file with all the details of the cluster, nodes, switches in JSON format",
        )
    except ValueError:
        # Option already exists, skip
        pass

    try:
        parser.addoption(
            "--config_file",
            action="store",
            required=True,
            help="Input file with all configurations and parameters for tests in JSON format",
        )
    except ValueError:
        # Option already exists, skip
        pass


def pytest_metadata(metadata):
    """Add CVS version metadata for both console output and HTML report."""

    # Get CVS version - try package metadata first, fallback to version.txt
    try:
        cvs_version = importlib.metadata.version("cvs")
    except importlib.metadata.PackageNotFoundError:
        # Fallback for development mode (running from cloned repo)
        try:
            version_file = Path(__file__).parent.parent / "version.txt"
            cvs_version = version_file.read_text().strip()
        except Exception as e:
            cvs_version = f"Unknown (Error: {e})"

    # Parse command line arguments to get our custom options (just for display)
    cluster_file = "Not specified"
    config_file = "Not specified"

    for i, arg in enumerate(sys.argv):
        if arg == "--cluster_file" and i + 1 < len(sys.argv):
            cluster_file = Path(sys.argv[i + 1]).name  # Just filename for display
        elif arg == "--config_file" and i + 1 < len(sys.argv):
            config_file = Path(sys.argv[i + 1]).name  # Just filename for display

    # Add custom metadata
    metadata["CVS version"] = cvs_version
    metadata["Cluster File"] = cluster_file
    metadata["Config File"] = config_file


# Order of execution of hooks: (function names are standard names recognized by plugin manager)
# pytest_sessionstart
# pytest_runtest_makereport (for each test phase)
# pytest_html_results_table_html (when pytest-html renders each row)
# pytest_html_results_summary (when pytest-html builds summary section)
# pytest_sessionfinish (end of session)


# Prepare a clean per-run log directory before tests start.
def pytest_sessionstart(session):
    _auto_register_inference_suite_report(session.config)
    _ensure_html_report_manager(session.config).setup_log_dir()


# Capture each test report and attach a per-test external log link.
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):  # noqa: ARG001
    outcome = yield
    report = outcome.get_result()
    report.extras = _ensure_html_report_manager(item.config).write_test_log(report, item.originalname)

    from cvs.lib.report.registry import get_suite_report_config
    from cvs.lib.report.types import InferenceReportConfig

    if isinstance(get_suite_report_config(item.config), InferenceReportConfig):
        from cvs.lib.report.inference_wiring import (
            attach_inference_suite_lifecycle_table,
            attach_inference_suite_report_row_extra,
        )

        attach_inference_suite_lifecycle_table(item, report)
        attach_inference_suite_report_row_extra(item, report)


# Replace inline pytest-html log content with a short externalized-log message.
def pytest_html_results_table_html(report, data):
    HtmlReportManager.replace_table_html(report, data)


# Inject CSS overrides in Summary section.
def pytest_html_results_summary(prefix, summary, postfix):
    HtmlReportManager.inject_style_overrides(prefix)


# Bundle the final HTML report and per-test log files into a zip at session end.
@pytest.hookimpl(hookwrapper=True)
def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    yield  # wait for pytest-html and all other plugins to finish writing the report
    mgr = _ensure_html_report_manager(session.config)
    mgr.generate_suite_reports(session)
    mgr.create_zip_bundle(session)
