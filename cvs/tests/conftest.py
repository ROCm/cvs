"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

import pytest

from cvs.core.orchestrators.factory import OrchestratorConfig, OrchestratorFactory
from cvs.lib import globals

log = globals.log


@pytest.fixture(scope="module")
def config_schema():
    """The suite's ``SuiteConfig`` subclass; a suite opts in to schema loading by overriding this fixture.

    ``cvs.config_validation`` is imported only on the schema path, so a suite
    without a schema never loads pydantic or the schema code.
    """
    return None


@pytest.fixture(scope="module")
def suite_mode(request, config_schema):
    if config_schema is None:
        return None
    from cvs.config_validation.loaders.suite_config import mode_from_module

    try:
        return mode_from_module(config_schema, request.module.__name__)
    except ValueError as exc:
        pytest.fail(str(exc))


@pytest.fixture(scope="module")
def suite_config(pytestconfig, config_schema):
    if config_schema is None:
        return None
    from cvs.config_validation.loaders.suite_config import load_suite_config

    return load_suite_config(
        config_schema, pytestconfig.getoption("config_file"), pytestconfig.getoption("cluster_file"), log
    )


@pytest.fixture(scope="module")
def orch(request, pytestconfig, suite_config, suite_mode):
    if suite_config is None:
        yield from _orch_from_files(pytestconfig)
        return
    from cvs.config_validation.loaders.orch import build_orch, cluster_with_hosts

    schema, config = suite_config.schema, suite_config.config
    # Only a suite that launches as tests is sure to have a lifecycle its hooks can use; health's is timing-only.
    lifecycle = None if schema.launch_in_fixture else request.getfixturevalue("lifecycle")
    hosts = schema.select_hosts(config, suite_config.cluster, suite_mode)
    o = build_orch(log, suite_config, cluster_with_hosts(suite_config.cluster, hosts))
    try:
        schema.after_orch(o, config, lifecycle)
        if schema.launch_in_fixture and o.orchestrator_type == "container":
            _launch_and_setup_sshd(o)
        yield o
        schema.teardown(o, config, lifecycle)
    finally:
        o.close()


def _orch_from_files(pytestconfig):
    cluster_file = pytestconfig.getoption("cluster_file")
    config_file = pytestconfig.getoption("config_file")
    if not cluster_file or not config_file:
        pytest.fail("orch fixture requires --cluster_file and --config_file")

    cfg = OrchestratorConfig.from_configs(cluster_file, config_file)
    orch = OrchestratorFactory.create_orchestrator(log, cfg)

    if cfg.orchestrator == "container":
        _launch_and_setup_sshd(orch)

    yield orch

    try:
        if cfg.orchestrator == "container":
            orch.teardown_containers()
    finally:
        orch.close()


def _launch_and_setup_sshd(orch):
    if not orch.setup_containers():
        pytest.fail(
            f"Failed to launch container : "
            f"{orch.get_container_name(orch.container_config, orch.container_config['image'])}"
        )
    if not orch.setup_sshd():
        pytest.fail("Failed to setup sshd in container")
