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
def orch(pytestconfig):
    cluster_file = pytestconfig.getoption("cluster_file")
    config_file = pytestconfig.getoption("config_file")
    if not cluster_file or not config_file:
        pytest.fail("orch fixture requires --cluster_file and --config_file")

    cfg = OrchestratorConfig.from_configs(cluster_file, config_file)
    orch = OrchestratorFactory.create_orchestrator(log, cfg)

    if cfg.orchestrator == "container":
        if not orch.setup_containers():
            pytest.fail(
                f"Failed to launch container : "
                f"{orch.get_container_name(orch.container_config, orch.container_config['image'])}"
            )
        if not orch.setup_sshd():
            pytest.fail("Failed to setup sshd in container")

    yield orch

    if cfg.orchestrator == "container":
        orch.teardown_containers()
