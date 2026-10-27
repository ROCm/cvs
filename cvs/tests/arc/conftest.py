'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent
publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.

Shared pytest fixtures for arc_installation and the ARC flow and item suites.
Every suite file sits flat in this directory. Each suite loads
the same cluster/config JSON; the ``orch`` execution handle itself comes from
the repo-root tests/conftest.py.
ARC runs on baremetal only -- the autouse ``_skip_arc_on_container`` fixture
below skips every ARC test cleanly when the orchestrator is a container.

This conftest lives at tests/arc/ so its fixtures apply to every generated
flow/item suite beside it.
'''

import json

import pytest

from cvs.core.orchestrators.factory import OrchestratorConfig
from cvs.lib.utils_lib import (
    resolve_cluster_config_placeholders,
    resolve_test_config_placeholders,
)
from cvs.lib import globals
from cvs.lib import arc_lib, arc_rack_lib

log = globals.log


@pytest.fixture(scope="module", autouse=True)
def _skip_arc_on_container(pytestconfig):
    '''
    Skip every ARC test on a container orchestrator, BEFORE the orch fixture runs.

    ARC installs packages and runs ``arc.py`` on controllers
    directly on the host OS -- none of which is modelled for the container
    backend, so ARC is baremetal-only. This autouse fixture resolves the
    orchestrator type from config WITHOUT building ``orch`` (reading it off the
    live handle would be too late: the repo-root ``orch`` fixture launches the
    container during its own setup). Being autouse and module-scoped, it is
    instantiated before the non-autouse ``orch`` fixture of the same scope, so a
    container run is skipped cleanly instead of paying -- or failing -- container
    setup for tests that would never run.
    '''
    cluster_file = pytestconfig.getoption("cluster_file")
    config_file = pytestconfig.getoption("config_file")
    if not cluster_file or not config_file:
        return
    cfg = OrchestratorConfig.from_configs(cluster_file, config_file)
    # Match OrchestratorFactory's case-insensitive normalization so a "Container"
    # / "CONTAINER" config is skipped here rather than slipping through to build a
    # container backend ARC cannot use.
    if (cfg.orchestrator or "").lower() == "container":
        pytest.skip("ARC is not supported under container orchestration (baremetal only)")


# Merge any extra report links stashed on the test item during the run (e.g. ARC
# log archives attached by arc_lib._attach_arc_logs_to_html). pytest-html 4.x
# renders links from report.extras; the core makereport hook sets report.extras
# first, so this wrapper appends afterwards on the "call" phase. Applies to the
# per-group and per-item ARC suites in this directory.
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):  # noqa: ARG001
    outcome = yield
    report = outcome.get_result()
    if report.when == "call":
        pending = getattr(item, "_arc_html_extras", None)
        if pending:
            report.extras = list(getattr(report, "extras", [])) + list(pending)


@pytest.fixture(scope="module")
def cluster_file(pytestconfig):
    '''Path to the ARC cluster JSON file, provided via --cluster_file.'''
    return pytestconfig.getoption("cluster_file")


@pytest.fixture(scope="module")
def config_file(pytestconfig):
    '''Path to the ARC test configuration JSON file, provided via --config_file.'''
    return pytestconfig.getoption("config_file")


@pytest.fixture(scope="module")
def cluster_dict(cluster_file):
    '''Load and resolve the ARC cluster configuration from JSON.'''
    with open(cluster_file) as json_file:
        cluster_dict = json.load(json_file)

    cluster_dict = resolve_cluster_config_placeholders(cluster_dict)
    log.info("ARC cluster config: %s", arc_rack_lib.redacted(cluster_dict))
    return cluster_dict


@pytest.fixture(scope="module")
def config_dict(config_file, cluster_dict, pytestconfig):
    '''
    Load and resolve the ARC test configuration from JSON.

    Placeholders such as {home} are resolved using cluster_dict
    (e.g. a "{home}/logs" value -> "/home/<user>/logs").

    Any field still left at the ``<changeme>`` placeholder is caught up front by
    resolve_test_config_placeholders (which hard-exits with a clear message), so
    it is not re-checked here. This fixture additionally fails the run before any
    ARC command runs on:

      - a configured arc_version that is invalid or GREATER than the version
        parsed from arc_release_url -- the archive must be able to satisfy the
        request, so the rule is arc_version <= url_version (a lower requested
        version is fine; aborts before any node is contacted).

    Collected ARC logs and the pytest HTML/log reports all land under this run's
    run_dir (resolved by RunLayout), so there is no artifact-path config to check.

    Failing here (fixture setup) means a bad config costs seconds, not a full
    suite run on the nodes.
    '''
    with open(config_file) as json_file:
        config_dict = json.load(json_file)

    config_dict = resolve_test_config_placeholders(config_dict, cluster_dict)
    log.info("ARC test config: %s", config_dict)

    suite_name = getattr(pytestconfig, "_suite_name", "") or ""
    if suite_name.startswith("arc") and "arc" in config_dict:
        problems = arc_lib.validate_arc_config(config_dict, cluster_dict)
        if problems:
            pytest.fail(
                "ARC config error (fix before running): " + "; ".join(problems),
                pytrace=False,
            )

    return config_dict


@pytest.fixture(scope="module")
def arc_res_dict():
    '''
    Module-scoped structured ARC results for the Run Deck ``status_matrix`` deck.

    Each flow/item run has arc_lib.run_arc_flows / run_arc_items merge its
    per-controller records into this dict (keyed by selection, then controller
    label). Each suite's Run Deck profile (arc_test_<key>.json) names this
    fixture in ``sources.results``, so the session binding captures it at module teardown
    and the deck is generated at session finish. Starts empty; the install-only
    suite never touches it (no deck profile registered for that stem).
    '''
    return {}
