'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent
publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.

Shared pytest fixtures for the ANC CVS suites (anc_installation, the per-group
suites under cpu/ and gpu/, and the exec-all suites). Each suite loads the same
cluster/config JSON and opens one parallel-SSH handle across all nodes.

This conftest lives at tests/anc/ so its fixtures also apply to the generated
per-group suites in the cpu/ and gpu/ subfolders.
'''

import json

import pytest

from cvs.lib.parallel_ssh_lib import Pssh
from cvs.lib.utils_lib import (
    resolve_cluster_config_placeholders,
    resolve_test_config_placeholders,
)
from cvs.lib import globals

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
def config_dict(config_file, cluster_dict):
    '''
    Load and resolve the ANC test configuration from JSON.

    Placeholders such as {home} are resolved using cluster_dict
    (e.g. cvs_home "{home}/cvs" -> "/home/<user>/cvs").
    '''
    with open(config_file) as json_file:
        config_dict = json.load(json_file)

    config_dict = resolve_test_config_placeholders(config_dict, cluster_dict)
    log.info("ANC test config: %s", config_dict)
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
