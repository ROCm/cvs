"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

import importlib.metadata
import sys
from pathlib import Path

import pytest

from cvs.lib.report_plugins import HtmlReportManager


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    suite_name = "test"
    for arg in config.args:
        bare = arg.split("::")[0]
        if not bare.startswith("-") and bare.endswith(".py"):
            suite_name = Path(bare).stem
            break
    config._suite_name = suite_name
    config._test_html_dir = f"{suite_name}_html"
    config._html_report_manager = HtmlReportManager(config)


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
    session.config._html_report_manager.setup_log_dir()


# Capture each test report and attach a per-test external log link.
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):  # noqa: ARG001
    outcome = yield
    report = outcome.get_result()
    report.extras = item.config._html_report_manager.write_test_log(report, item.originalname)


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
    session.config._html_report_manager.create_zip_bundle(session)
