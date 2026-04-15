"""
Single RCCL pytest entrypoint for simplified CVS runs.
"""

import json

import pytest

from cvs.lib import globals
from cvs.lib.rccl_cvs import load_rccl_config, run_rccl
from cvs.lib.utils_lib import resolve_cluster_config_placeholders, update_test_result


@pytest.fixture(scope="module")
def cluster_file(pytestconfig):
    return pytestconfig.getoption("cluster_file")


@pytest.fixture(scope="module")
def config_file(pytestconfig):
    return pytestconfig.getoption("config_file")


@pytest.fixture(scope="module")
def cluster_dict(cluster_file):
    with open(cluster_file) as handle:
        cluster = json.load(handle)
    return resolve_cluster_config_placeholders(cluster)


@pytest.fixture(scope="module")
def rccl_config(config_file, cluster_dict):
    return load_rccl_config(config_file, cluster_dict)


def test_rccl_cvs(cluster_dict, rccl_config):
    globals.error_list = []
    artifact = run_rccl(cluster_dict, rccl_config)
    globals.log.info(
        "RCCL result: run_id=%s cases=%s passed=%s failed=%s",
        artifact["run_id"],
        artifact["summary"]["cases_total"],
        artifact["summary"]["cases_passed"],
        artifact["summary"]["cases_failed"],
    )
    update_test_result()
