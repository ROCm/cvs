'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent
publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.

Shared pytest fixtures for the ANC CVS suites (anc_installation, the per-group
suites under cpu/ and gpu/, the per-item suite under individual_items/, and the
exec-all suites). Each suite loads the same cluster/config JSON and opens one
parallel-SSH handle across all nodes.

This conftest lives at tests/anc/ so its fixtures also apply to the generated
per-group/per-item suites in the cpu/, gpu/ and individual_items/ subfolders.
'''

import json

import pytest

from cvs.lib.parallel_ssh_lib import Pssh
from cvs.lib.utils_lib import (
    resolve_cluster_config_placeholders,
    resolve_test_config_placeholders,
)
from cvs.lib import globals
from cvs.lib import anc_lib

log = globals.log


# Merge any extra report links stashed on the test item during the run (e.g. ANC
# log archives attached by anc_lib._attach_anc_logs_to_html). pytest-html 4.x
# renders links from report.extras; the core makereport hook sets report.extras
# first, so this wrapper appends afterwards on the "call" phase. Applies to the
# per-group cpu/ and gpu/ suites and the exec-all suites under this directory.
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
def phdl(cluster_dict):
    '''Parallel SSH handle targeting every node in cluster_dict["node_dict"].'''
    node_list = list(cluster_dict["node_dict"].keys())

    return Pssh(
        log,
        node_list,
        user=cluster_dict["username"],
        pkey=cluster_dict["priv_key_file"],
    )


@pytest.fixture(scope="module")
def anc_res_dict():
    '''
    Module-scoped structured ANC results for the Run Deck ``status_matrix`` deck.

    Each ``test_<group>``/``test_<item>`` run has anc_lib.run_anc_groups /
    run_anc_items merge its per-node records into this dict (keyed by group/item,
    then node label). The Run Deck profiles (anc_test_cpu.json /
    anc_test_gpu.json / anc_test_individual_items.json) name this fixture in
    ``sources.results``, so the session binding captures it at module teardown
    and the deck is generated at session finish. Starts empty; the install-only
    suite never touches it (no deck profile registered for that stem).
    '''
    return {}
