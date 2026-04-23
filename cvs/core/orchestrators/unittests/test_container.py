'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

# Unit tests for cvs/core/orchestrators/container.py: ContainerOrchestrator construction
# (incl. SSH port override and runtime wiring) and the setup_containers / teardown_containers
# short-circuit semantics inherited by the rvs_cvs.py orch fixture. Mocks Pssh and
# RuntimeFactory so tests run with no SSH or container runtime.
#
# The teardown_containers short-circuit at cvs/core/orchestrators/container.py:439 is
# pinned here so any future change has a loud canary.

import unittest
from unittest.mock import MagicMock, patch

from cvs.core.orchestrators.factory import OrchestratorConfig
from cvs.core.orchestrators.container import ContainerOrchestrator


def _make_orch_config(launch=False, enabled=True):
    """Minimal OrchestratorConfig that satisfies ContainerOrchestrator.__init__
    without touching disk or SSH."""
    return OrchestratorConfig(
        orchestrator="container",
        node_dict={"10.0.0.1": {}, "10.0.0.2": {}},
        username="testuser",
        priv_key_file="/dev/null",
        password=None,
        head_node_dict={"mgmt_ip": "10.0.0.1"},
        container={
            "enabled": enabled,
            "launch": launch,
            "image": "rocm/cvs:test",
            "name": "cvs_iter_test",
            "runtime": {"name": "docker", "args": {}},
        },
    )


class TestContainerOrchestrator(unittest.TestCase):
    def _make(self, _mock_pssh, _mock_rf, launch=False, enabled=True):
        cfg = _make_orch_config(launch=launch, enabled=enabled)
        runtime = MagicMock(name="docker_runtime")
        _mock_rf.create.return_value = runtime
        orch = ContainerOrchestrator(MagicMock(), cfg)
        return orch, runtime

    @patch("cvs.core.orchestrators.container.RuntimeFactory")
    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_init_creates_runtime_via_factory(self, _mock_pssh, mock_rf):
        orch, runtime = self._make(_mock_pssh, mock_rf)
        self.assertIs(orch.runtime, runtime)
        # docker is the default runtime when container.runtime.name == "docker".
        mock_rf.create.assert_called_once()
        self.assertEqual(mock_rf.create.call_args[0][0], "docker")

    @patch("cvs.core.orchestrators.container.RuntimeFactory")
    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_init_sets_orchestrator_type(self, _mock_pssh, mock_rf):
        orch, _ = self._make(_mock_pssh, mock_rf)
        self.assertEqual(orch.orchestrator_type, "container")

    @patch("cvs.core.orchestrators.container.RuntimeFactory")
    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_init_overrides_ssh_port_to_container_sshd(self, _mock_pssh, mock_rf):
        orch, _ = self._make(_mock_pssh, mock_rf)
        self.assertEqual(orch.ssh_port, 2224)

    @patch("cvs.core.orchestrators.container.RuntimeFactory")
    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_setup_containers_launch_true_delegates_to_runtime(self, _mock_pssh, mock_rf):
        orch, runtime = self._make(_mock_pssh, mock_rf, launch=True)
        runtime.setup_containers.return_value = True
        result = orch.setup_containers()
        self.assertTrue(result)
        runtime.setup_containers.assert_called_once()

    @patch("cvs.core.orchestrators.container.RuntimeFactory")
    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_setup_containers_short_circuits_when_disabled(self, _mock_pssh, mock_rf):
        orch, runtime = self._make(_mock_pssh, mock_rf, enabled=False)
        result = orch.setup_containers()
        self.assertTrue(result)  # Disabled is treated as success / no-op.
        runtime.setup_containers.assert_not_called()

    @patch("cvs.core.orchestrators.container.RuntimeFactory")
    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_teardown_containers_short_circuits_when_launch_false(self, _mock_pssh, mock_rf):
        # launch:false means containers are externally managed; teardown is a no-op.
        orch, runtime = self._make(_mock_pssh, mock_rf, launch=False)
        orch.container_id = "test_container"
        self.assertTrue(orch.teardown_containers())
        runtime.teardown_containers.assert_not_called()

    @patch("cvs.core.orchestrators.container.RuntimeFactory")
    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_teardown_containers_calls_runtime_when_launch_true(self, _mock_pssh, mock_rf):
        # launch:true means CVS owns the container lifecycle; teardown delegates to runtime.
        orch, runtime = self._make(_mock_pssh, mock_rf, launch=True)
        orch.container_id = "test_container"
        runtime.teardown_containers.return_value = True
        self.assertTrue(orch.teardown_containers())
        runtime.teardown_containers.assert_called_once_with("test_container")
        # container_id cleared on successful teardown.
        self.assertIsNone(orch.container_id)

    @patch("cvs.core.orchestrators.container.RuntimeFactory")
    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_teardown_containers_short_circuits_when_disabled(self, _mock_pssh, mock_rf):
        orch, runtime = self._make(_mock_pssh, mock_rf, enabled=False)
        orch.container_id = "test_container"
        self.assertTrue(orch.teardown_containers())
        runtime.teardown_containers.assert_not_called()

    @patch("cvs.core.orchestrators.container.RuntimeFactory")
    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_teardown_containers_short_circuits_when_no_container_id(self, _mock_pssh, mock_rf):
        # container_id stays None when prepare/setup never ran; teardown is a no-op
        # even with launch=True (CVS-owned), since there is nothing to tear down.
        orch, runtime = self._make(_mock_pssh, mock_rf, launch=True)
        self.assertIsNone(orch.container_id)
        self.assertTrue(orch.teardown_containers())
        runtime.teardown_containers.assert_not_called()

    @patch("cvs.core.orchestrators.container.RuntimeFactory")
    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_init_requires_container_config(self, _mock_pssh, _mock_rf):
        # ContainerOrchestrator raises if 'container' config is empty. Construct
        # the config directly to bypass the helper's auto-population.
        cfg = OrchestratorConfig(
            orchestrator="container",
            node_dict={"10.0.0.1": {}},
            username="testuser",
            priv_key_file="/dev/null",
            container={},
        )
        with self.assertRaises(ValueError):
            ContainerOrchestrator(MagicMock(), cfg)


if __name__ == "__main__":
    unittest.main()
