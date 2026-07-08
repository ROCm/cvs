"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

import importlib.metadata
import json
import logging
from pathlib import Path

import pytest

from cvs.lib.report_plugins import HtmlReportManager, cli_option_value

log = logging.getLogger(__name__)

def _maybe_autocollect_html(config, suite_name):
    '''
    Enable pytest-html for ANC suites without an explicit --html.

    When the ANC config has COLLECT_HTML_REPORTS truthy (default) and the user
    did not pass --html on the command line, derive the report path from
    anc.log_folder_path and set config.option.htmlpath so pytest-html generates
    the report anyway. Runs at
    tryfirst pytest_configure time (before pytest-html's own configure and before
    HtmlReportManager reads config.option.htmlpath). An explicit --html always
    wins. Best-effort: any failure leaves HTML reporting exactly as the command
    line specified.
    '''
    # Explicit --html wins; do nothing.
    if getattr(config.option, "htmlpath", None):
        return

    cluster_file = config.getoption("cluster_file", default=None)
    config_file = config.getoption("config_file", default=None)
    if not cluster_file or not config_file:
        return

    try:
        from cvs.lib.utils_lib import (
            resolve_cluster_config_placeholders,
            resolve_test_config_placeholders,
        )
        from cvs.lib.anc_lib import (
            COLLECT_HTML_REPORTS_KEY,
            _as_bool,
            new_run_timestamp,
            resolve_anc_html_report_path,
        )

        with open(config_file) as fh:
            config_dict = json.load(fh)
        # Only ANC suites participate in this auto-collection.
        if "anc" not in config_dict:
            return

        with open(cluster_file) as fh:
            cluster_dict = json.load(fh)
        cluster_dict = resolve_cluster_config_placeholders(cluster_dict)
        config_dict = resolve_test_config_placeholders(config_dict, cluster_dict)

        if not _as_bool(config_dict["anc"].get(COLLECT_HTML_REPORTS_KEY), default=True):
            return

        timestamp = new_run_timestamp()
        # Match the per-node log folder naming (test_<group>), whose suite files
        # are named anc_test_<group>; drop the leading "anc_".
        report_name = suite_name
        if report_name.startswith("anc_"):
            report_name = report_name[len("anc_") :]
        html_path = resolve_anc_html_report_path(config_dict, cluster_dict, report_name, timestamp)
        Path(html_path).parent.mkdir(parents=True, exist_ok=True)
        config.option.htmlpath = html_path
        # Portable single-file report (matches the usual manual invocation).
        if hasattr(config.option, "self_contained_html"):
            config.option.self_contained_html = True
    except Exception:
        # Never let report auto-setup break test collection/run.
        return


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

    _maybe_autocollect_html(config, config._suite_name)
    config._html_report_manager = HtmlReportManager(config)
    return config._html_report_manager


def _auto_register_inference_suite_report(config):
    from cvs.lib.report.auto_register import try_auto_register_inference_suite_report

    _sync_suite_name_from_args(config)
    return try_auto_register_inference_suite_report(config)


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    _ensure_html_report_manager(config)
    _auto_register_inference_suite_report(config)


@pytest.fixture(scope="session", autouse=True)
def _cvs_inference_suite_report_session(request):
    """Initialize the session report store when a suite preset is registered."""
    from cvs.lib.report.registry import clear_session_results, get_suite_report_config
    from cvs.lib.report.types import InferenceReportConfig

    if not isinstance(get_suite_report_config(request.config), InferenceReportConfig):
        yield
        return

    clear_session_results()
    yield


@pytest.fixture(scope="module", autouse=True)
def _cvs_inference_suite_report_bind_module(request, _cvs_inference_suite_report_session):
    """Bind module-scoped suite fixtures into the session store at module teardown."""
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

    # Parse command line arguments to get our custom options (just for display).
    # Handles both `--opt value` and `--opt=value` forms.
    cluster_arg = cli_option_value("--cluster_file")
    config_arg = cli_option_value("--config_file")
    cluster_file = Path(cluster_arg).name if cluster_arg else "Not specified"
    config_file = Path(config_arg).name if config_arg else "Not specified"

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
