'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

# Tier-1 dispatch tests for cvs/tests/health/install/install_rvs.py after the
# orch-fixture port. Pins:
#   1. install_rvs dispatches via orch.exec (not orch.exec_on_head).
#   2. The call signature install_rvs uses (`orch.exec(cmd, timeout=...)`) is
#      accepted by both BaremetalOrchestrator and ContainerOrchestrator orch
#      instances.
#   3. The legacy nfs_install head-only dispatch was dropped (regression canary
#      if someone re-introduces it).
#
# orch.exec / orch.exec_on_head are stubbed directly on the constructed
# orchestrator instance (not at the Pssh layer) to sidestep the signature drift
# between BaremetalOrchestrator.exec(cmd, hosts=None, timeout=None) and
# ContainerOrchestrator.exec(cmd, hosts=None, timeout=None, detailed=False).

import unittest
from unittest.mock import MagicMock, patch

from cvs.core.orchestrators.factory import OrchestratorConfig
from cvs.core.orchestrators.baremetal import BaremetalOrchestrator
from cvs.core.orchestrators.container import ContainerOrchestrator
from cvs.lib import globals
from cvs.tests.health.install import install_rvs as install_rvs_mod


def _bm_config():
    """Minimal OrchestratorConfig that satisfies BaremetalOrchestrator.__init__
    without touching disk or SSH."""
    return OrchestratorConfig(
        orchestrator="baremetal",
        node_dict={"10.0.0.1": {}, "10.0.0.2": {}},
        username="testuser",
        priv_key_file="/dev/null",
        password=None,
        head_node_dict={"mgmt_ip": "10.0.0.1"},
        container={},
    )


def _ct_config():
    """Minimal OrchestratorConfig for ContainerOrchestrator.__init__. launch=False
    keeps setup_containers from being triggered as a side effect."""
    return OrchestratorConfig(
        orchestrator="container",
        node_dict={"10.0.0.1": {}, "10.0.0.2": {}},
        username="testuser",
        priv_key_file="/dev/null",
        password=None,
        head_node_dict={"mgmt_ip": "10.0.0.1"},
        container={
            "enabled": True,
            "launch": False,
            "image": "rocm/cvs:test",
            "name": "cvs_iter_test",
            "runtime": {"name": "docker", "args": {}},
        },
    )


# Single happy-path output that satisfies every regex check in
# install_rvs.test_install_rvs:
#   line 220: re.search('rvs', out, re.I)               -> matches "rvs"
#   line 233: re.search(r'gst_single\.conf', out, re.I) -> matches "gst_single.conf"
#   line 330: re.search('not found|No such file', out)  -> does NOT match
#   line 339: re.search('No such file', out)            -> does NOT match
_HAPPY_OUT = {
    "10.0.0.1": "/opt/rocm/bin/rvs and gst_single.conf",
    "10.0.0.2": "/opt/rocm/bin/rvs and gst_single.conf",
}


def _config_dict():
    """Minimal config_dict for install_rvs.test_install_rvs. `rocm_path` is set
    explicitly so detect_rocm_path returns early without consuming any exec calls."""
    return {
        "git_install_path": "/tmp/install",
        "git_url": "https://example.invalid/rvs",
        "path": "/opt/rocm/bin/rvs",
        "config_path_mi300x": "/opt/rocm/share/rocm-validation-suite/conf",
        "config_path_default": "/opt/rocm/share/rocm-validation-suite/conf",
        "rocm_path": "/opt/rocm",
    }


class TestInstallRvsBaremetalDispatch(unittest.TestCase):
    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_dispatches_via_orch_exec_only(self, _mock_pssh):
        orch = BaremetalOrchestrator(MagicMock(), _bm_config())
        orch.exec = MagicMock(return_value=_HAPPY_OUT)
        orch.exec_on_head = MagicMock()
        globals.error_list = []

        install_rvs_mod.test_install_rvs(orch, _config_dict())

        orch.exec.assert_any_call("which rvs", timeout=30)
        orch.exec_on_head.assert_not_called()
        self.assertEqual(globals.error_list, [])


class TestInstallRvsContainerDispatch(unittest.TestCase):
    @patch("cvs.core.orchestrators.container.RuntimeFactory")
    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_dispatches_via_orch_exec_only(self, _mock_pssh, mock_rf):
        runtime = MagicMock(name="docker_runtime")
        mock_rf.create.return_value = runtime

        orch = ContainerOrchestrator(MagicMock(), _ct_config())
        orch.exec = MagicMock(return_value=_HAPPY_OUT)
        orch.exec_on_head = MagicMock()
        globals.error_list = []

        install_rvs_mod.test_install_rvs(orch, _config_dict())

        orch.exec.assert_any_call("which rvs", timeout=30)
        orch.exec_on_head.assert_not_called()
        self.assertEqual(globals.error_list, [])


if __name__ == "__main__":
    unittest.main()
