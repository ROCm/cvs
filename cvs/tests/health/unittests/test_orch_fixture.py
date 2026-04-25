'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

# Unit tests for the new `orch` fixture in cvs/tests/health/rvs_cvs.py. The fixture
# body is invoked directly via `orch_fixture.__wrapped__(...)` (the underlying function
# before @pytest.fixture wraps it) with mocked OrchestratorFactory, OrchestratorConfig,
# and pytest's `request` / `pytestconfig`. This exercises the conditional container
# lifecycle gate (cfg.orchestrator == "container") without launching a pytest session
# or touching disk / SSH.

import unittest
from unittest.mock import MagicMock, patch

import pytest


class TestOrchFixture(unittest.TestCase):

    def _invoke_orch_fixture(
        self,
        MockCfg,
        MockFactory,
        orchestrator,
        setup_containers_ret=True,
        setup_sshd_ret=True,
    ):
        """Run the `orch` fixture body with the given orchestrator string and
        setup_* return values. Returns (result, orch_obj, request)."""
        from cvs.tests.health.rvs_cvs import orch as orch_fixture

        cfg = MagicMock(orchestrator=orchestrator)
        MockCfg.from_configs.return_value = cfg

        orch_obj = MagicMock(name="orchestrator_instance")
        orch_obj.setup_containers.return_value = setup_containers_ret
        orch_obj.setup_sshd.return_value = setup_sshd_ret
        MockFactory.create_orchestrator.return_value = orch_obj

        request = MagicMock(name="pytest_request")
        pytestconfig = MagicMock(name="pytestconfig")
        pytestconfig.getoption.side_effect = ["/dev/null", "/dev/null"]

        # __wrapped__ is the underlying function before @pytest.fixture wraps it;
        # this is the standard pattern for testing fixture bodies in isolation.
        result = orch_fixture.__wrapped__(pytestconfig, request)
        return result, orch_obj, request

    # ------------------------------------------------------------------
    # Baremetal path
    # ------------------------------------------------------------------

    @patch("cvs.tests.health.rvs_cvs.OrchestratorFactory")
    @patch("cvs.tests.health.rvs_cvs.OrchestratorConfig")
    def test_baremetal_path_returns_orch_no_container_setup(self, MockCfg, MockFactory):
        result, orch_obj, request = self._invoke_orch_fixture(MockCfg, MockFactory, "baremetal")
        # No container lifecycle calls on the baremetal path.
        orch_obj.setup_containers.assert_not_called()
        orch_obj.setup_sshd.assert_not_called()
        # Fixture returns the orch object the factory produced.
        self.assertIs(result, orch_obj)

    @patch("cvs.tests.health.rvs_cvs.OrchestratorFactory")
    @patch("cvs.tests.health.rvs_cvs.OrchestratorConfig")
    def test_baremetal_path_does_not_register_finalizer(self, MockCfg, MockFactory):
        # Explicit gate: baremetal MUST NOT register a teardown_containers finalizer
        # (which would attribute-error on a BaremetalOrchestrator at runtime).
        _result, _orch_obj, request = self._invoke_orch_fixture(MockCfg, MockFactory, "baremetal")
        request.addfinalizer.assert_not_called()

    # ------------------------------------------------------------------
    # Container path - happy
    # ------------------------------------------------------------------

    @patch("cvs.tests.health.rvs_cvs.OrchestratorFactory")
    @patch("cvs.tests.health.rvs_cvs.OrchestratorConfig")
    def test_container_path_calls_setup_and_registers_finalizer(self, MockCfg, MockFactory):
        result, orch_obj, request = self._invoke_orch_fixture(MockCfg, MockFactory, "container")
        # Setup methods called in order.
        orch_obj.setup_containers.assert_called_once()
        orch_obj.setup_sshd.assert_called_once()
        # Finalizer registered with teardown_containers (NOT cleanup, NOT a lambda).
        request.addfinalizer.assert_called_once_with(orch_obj.teardown_containers)
        self.assertIs(result, orch_obj)

    # ------------------------------------------------------------------
    # Container path - failure modes
    # ------------------------------------------------------------------

    @patch("cvs.tests.health.rvs_cvs.OrchestratorFactory")
    @patch("cvs.tests.health.rvs_cvs.OrchestratorConfig")
    def test_container_setup_containers_failure_calls_pytest_fail(self, MockCfg, MockFactory):
        # setup_containers returning False -> pytest.fail; setup_sshd never reached;
        # finalizer NOT registered (pytest.fail interrupts fixture body).
        with self.assertRaises(pytest.fail.Exception) as ctx:
            self._invoke_orch_fixture(
                MockCfg, MockFactory, "container", setup_containers_ret=False
            )
        self.assertIn("setup_containers", str(ctx.exception))

    @patch("cvs.tests.health.rvs_cvs.OrchestratorFactory")
    @patch("cvs.tests.health.rvs_cvs.OrchestratorConfig")
    def test_container_setup_sshd_failure_calls_pytest_fail(self, MockCfg, MockFactory):
        with self.assertRaises(pytest.fail.Exception) as ctx:
            self._invoke_orch_fixture(
                MockCfg, MockFactory, "container", setup_sshd_ret=False
            )
        self.assertIn("setup_sshd", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
