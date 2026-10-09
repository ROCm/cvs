'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent
publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.

Shared pytest fixtures for the ANC CVS suites (anc_installation, the per-group
suites under cpu/ and gpu/, and the per-family individual-item suites under
computerocker/, memrocker/, oblex/, gemm/, xgmi/, ualink/, pcie/, babel/,
basic/). Each suite loads the same cluster/config JSON; the ``orch`` execution
handle itself comes from the repo-root tests/conftest.py.
ANC runs on baremetal only -- the autouse ``_skip_anc_on_container`` fixture
below skips every ANC test cleanly when the orchestrator is a container.

This conftest lives at tests/anc/ so its fixtures also apply to the generated
per-group/per-item suites in all those subfolders.
'''

import json

import pytest

from cvs.core.orchestrators.factory import OrchestratorConfig
from cvs.lib.utils_lib import (
    resolve_cluster_config_placeholders,
    resolve_test_config_placeholders,
)
from cvs.lib import globals
from cvs.lib import anc_lib

log = globals.log


@pytest.fixture(scope="module", autouse=True)
def _skip_anc_on_container(pytestconfig):
    '''
    Skip every ANC test on a container orchestrator, BEFORE the orch fixture runs.

    ANC installs packages, runs ``sudo ./anc.py`` and tars root-owned log trees
    directly on the host OS -- none of which is modelled for the container
    backend, so ANC is baremetal-only. This autouse fixture resolves the
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
    # container backend ANC cannot use.
    if (cfg.orchestrator or "").lower() == "container":
        pytest.skip("ANC is not supported under container orchestration (baremetal only)")


# Merge any extra report links stashed on the test item during the run (e.g. ANC
# log archives attached by anc_lib._attach_anc_logs_to_html). pytest-html 4.x
# renders links from report.extras; the core makereport hook sets report.extras
# first, so this wrapper appends afterwards on the "call" phase. Applies to the
# per-group cpu/ and gpu/ suites and the per-family item suites under this directory.
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):  # noqa: ARG001
    outcome = yield
    report = outcome.get_result()
    if report.when == "call":
        pending = getattr(item, "_anc_html_extras", None)
        if pending:
            report.extras = list(getattr(report, "extras", [])) + list(pending)


@pytest.fixture(scope="module")
def cluster_file(pytestconfig):
    '''Path to the ANC cluster JSON file, provided via --cluster_file.'''
    return pytestconfig.getoption("cluster_file")


@pytest.fixture(scope="module")
def config_file(pytestconfig):
    '''Path to the ANC test configuration JSON file, provided via --config_file.'''
    return pytestconfig.getoption("config_file")


@pytest.fixture(scope="module")
def cluster_dict(cluster_file):
    '''Load and resolve the ANC cluster configuration from JSON.'''
    with open(cluster_file) as json_file:
        cluster_dict = json.load(json_file)

    cluster_dict = resolve_cluster_config_placeholders(cluster_dict)
    log.info("ANC cluster config: %s", cluster_dict)
    return cluster_dict


@pytest.fixture(scope="module")
def config_dict(config_file, cluster_dict, pytestconfig):
    '''
    Load and resolve the ANC test configuration from JSON.

    Placeholders such as {home} are resolved using cluster_dict
    (e.g. a "{home}/logs" value -> "/home/<user>/logs").

    Any field still left at the ``<changeme>`` placeholder is caught up front by
    resolve_test_config_placeholders (which hard-exits with a clear message), so
    it is not re-checked here. This fixture additionally fails the run before any
    ANC command runs on:

      - a configured anc_version that is invalid or GREATER than the version
        parsed from anc_release_url -- the archive must be able to satisfy the
        request, so the rule is anc_version <= url_version (a lower requested
        version is fine; aborts before any node is contacted).

    Collected ANC logs and the pytest HTML/log reports all land under this run's
    run_dir (resolved by RunLayout), so there is no artifact-path config to check.

    Failing here (fixture setup) means a bad config costs seconds, not a full
    suite run on the nodes.
    '''
    with open(config_file) as json_file:
        config_dict = json.load(json_file)

    config_dict = resolve_test_config_placeholders(config_dict, cluster_dict)
    log.info("ANC test config: %s", config_dict)

    suite_name = getattr(pytestconfig, "_suite_name", "") or ""
    if suite_name.startswith("anc") and "anc" in config_dict:
        problems = anc_lib.validate_anc_config(config_dict, cluster_dict)
        if problems:
            pytest.fail(
                "ANC config error (fix before running): " + "; ".join(problems),
                pytrace=False,
            )

    return config_dict


@pytest.fixture(scope="module")
def anc_res_dict():
    '''
    Module-scoped structured ANC results for the Run Deck ``status_matrix`` deck.

    Each ``test_<group>``/``test_<item>`` run has anc_lib.run_anc_groups /
    run_anc_items merge its per-node records into this dict (keyed by group/item,
    then node label). Each suite's Run Deck profile (anc_test_cpu.json /
    anc_test_gpu.json / anc_test_<family>.json) names this fixture in
    ``sources.results``, so the session binding captures it at module teardown
    and the deck is generated at session finish. Starts empty; the install-only
    suite never touches it (no deck profile registered for that stem).
    '''
    return {}
