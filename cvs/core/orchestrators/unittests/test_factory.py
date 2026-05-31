'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

# Unit tests for cvs/core/orchestrators/factory.py: OrchestratorFactory.create_orchestrator
# dispatch and OrchestratorConfig construction / from_configs validation. Mocks Pssh and
# RuntimeFactory so tests run without SSH or container runtime.

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from cvs.core.orchestrators.factory import (
    DEFAULT_CONTAINER_SETUP_SCRIPT,
    OrchestratorConfig,
    OrchestratorFactory,
    _resolve_container_setup_script,
)
from cvs.core.orchestrators.baremetal import BaremetalOrchestrator
from cvs.core.orchestrators.container import ContainerOrchestrator


def _make_orch_config(orchestrator="baremetal", with_container=False, lifetime="per_run"):
    """Minimal OrchestratorConfig that satisfies BaremetalOrchestrator and (optionally)
    ContainerOrchestrator __init__ without touching disk or SSH."""
    kwargs = dict(
        orchestrator=orchestrator,
        node_dict={"10.0.0.1": {}, "10.0.0.2": {}},
        username="testuser",
        priv_key_file="/dev/null",
        password=None,
        head_node_dict={"mgmt_ip": "10.0.0.1"},
        container={},
    )
    if with_container or orchestrator == "container":
        kwargs["container"] = {
            "lifetime": lifetime,
            "image": "rocm/cvs:test",
            "name": "cvs_iter_test",
            "runtime": {"name": "docker", "args": {}},
        }
    return OrchestratorConfig(**kwargs)


class TestOrchestratorFactory(unittest.TestCase):
    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_create_returns_baremetal_for_baremetal_string(self, _mock_pssh):
        cfg = _make_orch_config(orchestrator="baremetal")
        orch = OrchestratorFactory.create_orchestrator(MagicMock(), cfg)
        self.assertIsInstance(orch, BaremetalOrchestrator)

    @patch("cvs.core.orchestrators.container.RuntimeFactory")
    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_create_returns_container_for_container_string(self, _mock_pssh, _mock_rf):
        cfg = _make_orch_config(orchestrator="container")
        orch = OrchestratorFactory.create_orchestrator(MagicMock(), cfg)
        self.assertIsInstance(orch, ContainerOrchestrator)

    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_create_raises_for_unsupported_string(self, _mock_pssh):
        # Bypass OrchestratorConfig.from_configs validation by constructing
        # the config object directly with an unsupported orchestrator value.
        cfg = _make_orch_config(orchestrator="slurm")
        with self.assertRaises(ValueError):
            OrchestratorFactory.create_orchestrator(MagicMock(), cfg)

    def test_create_raises_typeerror_for_non_config(self):
        with self.assertRaises(TypeError):
            OrchestratorFactory.create_orchestrator(MagicMock(), {"orchestrator": "baremetal"})

    def test_get_supported_backends_lists_both(self):
        backends = OrchestratorFactory.get_supported_backends()
        self.assertIn("baremetal", backends)
        self.assertIn("container", backends)


class TestOrchestratorConfig(unittest.TestCase):
    def test_get_returns_default_for_missing_key(self):
        cfg = _make_orch_config()
        self.assertEqual(cfg.get("nonexistent_key", "fallback"), "fallback")

    def test_get_returns_value_for_existing_attr(self):
        cfg = _make_orch_config()
        self.assertEqual(cfg.get("username"), "testuser")

    def test_from_configs_defaults_orchestrator_to_baremetal_when_omitted(self):
        cluster = {"node_dict": {"1.1.1.1": {}}, "username": "u", "priv_key_file": "/dev/null"}
        cfg = OrchestratorConfig.from_configs(cluster)
        self.assertEqual(cfg.orchestrator, "baremetal")

    def test_from_configs_merges_testsuite_over_cluster(self):
        cluster = {
            "orchestrator": "baremetal",
            "node_dict": {"1.1.1.1": {}},
            "username": "cluster_user",
            "priv_key_file": "/dev/null",
        }
        testsuite = {"username": "testsuite_user"}
        cfg = OrchestratorConfig.from_configs(cluster, testsuite)
        self.assertEqual(cfg.username, "testsuite_user")

    def test_from_configs_raises_when_node_dict_missing(self):
        cluster = {"username": "u", "priv_key_file": "/dev/null"}
        with self.assertRaises(ValueError):
            OrchestratorConfig.from_configs(cluster)

    def test_from_configs_raises_when_username_missing(self):
        cluster = {"node_dict": {"1.1.1.1": {}}, "priv_key_file": "/dev/null"}
        with self.assertRaises(ValueError):
            OrchestratorConfig.from_configs(cluster)

    def test_from_configs_raises_when_priv_key_file_missing(self):
        cluster = {"node_dict": {"1.1.1.1": {}}, "username": "u"}
        with self.assertRaises(ValueError):
            OrchestratorConfig.from_configs(cluster)

    def test_init_raises_valueerror_for_missing_required_key(self):
        # Direct construction with a missing required key should surface a clear
        # ValueError listing what's missing, not a bare KeyError.
        with self.assertRaises(ValueError) as ctx:
            OrchestratorConfig(
                orchestrator="baremetal",
                node_dict={"1.1.1.1": {}},
                username="u",
                # priv_key_file deliberately omitted
            )
        self.assertIn("priv_key_file", str(ctx.exception))

    @patch.dict(os.environ, {"USER": "alice", "LOGNAME": "", "USERNAME": ""}, clear=False)
    def test_from_configs_reads_from_file_path(self):
        # Round-trip a real JSON file through from_configs to exercise the
        # path-handling branch (the rvs_cvs.py orch fixture goes through this).
        # Also pins that {user-id} placeholders are resolved when input came from disk.
        cluster = {
            "orchestrator": "baremetal",
            "node_dict": {"1.1.1.1": {}, "1.1.1.2": {}},
            "username": "{user-id}",
            "priv_key_file": "/home/{user-id}/.ssh/id_rsa",
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(cluster, f)
            path = f.name
        try:
            cfg = OrchestratorConfig.from_configs(path)
            self.assertEqual(cfg.orchestrator, "baremetal")
            self.assertEqual(len(cfg.node_dict), 2)
            self.assertEqual(cfg.username, "alice")
            self.assertEqual(cfg.priv_key_file, "/home/alice/.ssh/id_rsa")
        finally:
            os.unlink(path)

    # ------------------------------------------------------------------
    # Placeholder resolution at the from_configs boundary.
    #
    # Pins that {user-id} in the cluster file (and any merged testsuite
    # overrides) is substituted before the values reach BaremetalOrchestrator /
    # Pssh / docker run. Env is patched so the substituted value is
    # deterministic regardless of host $USER (CI / sudo / containers).
    # ------------------------------------------------------------------

    @patch.dict(os.environ, {"USER": "alice", "LOGNAME": "", "USERNAME": ""}, clear=False)
    def test_from_configs_resolves_user_id_in_username_and_priv_key(self):
        cluster = {
            "node_dict": {"1.1.1.1": {}},
            "username": "{user-id}",
            "priv_key_file": "/home/{user-id}/.ssh/id_rsa",
        }
        cfg = OrchestratorConfig.from_configs(cluster)
        self.assertEqual(cfg.username, "alice")
        self.assertEqual(cfg.priv_key_file, "/home/alice/.ssh/id_rsa")

    @patch.dict(os.environ, {"USER": "alice", "LOGNAME": "", "USERNAME": ""}, clear=False)
    def test_from_configs_resolves_user_id_throughout_container_block(self):
        # The recursive walk must descend into container at every nesting shape:
        # top-level string scalar (image, name), and list item under a nested dict
        # (runtime.args.volumes).
        cluster = {
            "orchestrator": "container",
            "node_dict": {"1.1.1.1": {}},
            "username": "{user-id}",
            "priv_key_file": "/home/{user-id}/.ssh/id_rsa",
            "container": {
                "lifetime": "per_run",
                "image": "rocm/{user-id}:test",
                "name": "{user-id}_cvs",
                "runtime": {
                    "name": "docker",
                    "args": {"volumes": ["/home/{user-id}/data:/data"]},
                },
            },
        }
        cfg = OrchestratorConfig.from_configs(cluster)
        self.assertEqual(cfg.container["image"], "rocm/alice:test")
        self.assertEqual(cfg.container["name"], "alice_cvs")
        self.assertEqual(
            cfg.container["runtime"]["args"]["volumes"],
            ["/home/alice/data:/data"],
        )

    @patch.dict(os.environ, {"USER": "alice", "LOGNAME": "", "USERNAME": ""}, clear=False)
    def test_from_configs_does_not_resolve_user_id_in_testsuite(self):
        # Pins the new contract: from_configs runs placeholder resolution on
        # cluster_config BEFORE the merge, so testsuite-side values pass through
        # verbatim. Placeholder resolution for testsuite subsections is the per-
        # test config_dict fixture's responsibility (resolve_test_config_placeholders),
        # which is scoped to the subsection the test consumes. Resolving the merged
        # dict at from_configs would walk testsuite subsections and trip the
        # <changeme> guard on legit auto-detect sentinels (transferbench.rocm_path,
        # rvs.path) belonging to tests that don't even use that subsection.
        cluster = {
            "node_dict": {"1.1.1.1": {}},
            "username": "literal_user",
            "priv_key_file": "/dev/null",
        }
        testsuite = {"username": "{user-id}"}
        cfg = OrchestratorConfig.from_configs(cluster, testsuite)
        self.assertEqual(cfg.username, "{user-id}")

    def test_from_configs_exits_on_unresolved_changeme(self):
        # Pins that the resolver's <changeme> guard propagates through from_configs
        # for the cluster portion. Defends against a future change that disables
        # the guard and silently leaks <changeme> into Pssh / docker run.
        cluster = {
            "node_dict": {"1.1.1.1": {}},
            "username": "<changeme>",
            "priv_key_file": "/dev/null",
        }
        with self.assertRaises(SystemExit) as ctx:
            OrchestratorConfig.from_configs(cluster)
        self.assertEqual(ctx.exception.code, 1)

    def test_from_configs_does_not_trip_changeme_in_testsuite(self):
        # Regression test for the orch-fixture <changeme> regression: testsuite
        # subsections (transferbench, rvs, ...) legitimately use <changeme> as a
        # soft auto-detect sentinel for rocm_path / path. The per-test fixture
        # (and detect_rocm_path) handles it. from_configs must NOT walk the
        # testsuite subsections through the <changeme>-strict resolver -- the old
        # behaviour caused cross-subsection guard trips (e.g. running rvs_cvs
        # tripped on transferbench.rocm_path: <changeme>).
        cluster = {
            "node_dict": {"1.1.1.1": {}},
            "username": "literal_user",
            "priv_key_file": "/dev/null",
        }
        testsuite = {
            "transferbench": {"rocm_path": "<changeme>"},
            "rvs": {"path": "<changeme>/bin", "rocm_path": "<changeme>"},
        }
        cfg = OrchestratorConfig.from_configs(cluster, testsuite)
        self.assertEqual(cfg.username, "literal_user")


class TestResolveContainerSetupScript(unittest.TestCase):
    """container.setup_script default injection + path validation."""

    def test_empty_block_untouched(self):
        # Baremetal path: empty container block stays empty (no script injected).
        self.assertEqual(_resolve_container_setup_script({}), {})

    def test_falsy_setup_script_defaults(self):
        # absent / None / "" are all falsy -> packaged default (no disable value).
        for name, container in [
            ("absent", {"image": "x"}),
            ("none", {"image": "x", "setup_script": None}),
            ("empty", {"image": "x", "setup_script": ""}),
        ]:
            with self.subTest(name):
                out = _resolve_container_setup_script(dict(container))
                self.assertEqual(out["setup_script"], DEFAULT_CONTAINER_SETUP_SCRIPT)

    def test_packaged_default_script_exists_on_disk(self):
        # The default must actually be present (packaged + __file__-relative),
        # otherwise every container run with no setup_script would fail to read it.
        self.assertTrue(os.path.isfile(DEFAULT_CONTAINER_SETUP_SCRIPT))

    def test_existing_user_path_kept_as_absolute(self):
        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
            f.write("#!/bin/bash\ntrue\n")
            path = f.name
        try:
            out = _resolve_container_setup_script({"image": "x", "setup_script": path})
            self.assertEqual(out["setup_script"], os.path.abspath(path))
        finally:
            os.unlink(path)

    def test_missing_user_path_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _resolve_container_setup_script({"image": "x", "setup_script": "/no/such/setup_script.sh"})
        self.assertIn("setup_script", str(ctx.exception))

    def test_missing_packaged_default_raises(self):
        # The F2 fix branch: a broken install whose packaged default is missing
        # must fail fast at resolution (ValueError), not as an OSError mid-run.
        with patch(
            "cvs.core.orchestrators.factory.DEFAULT_CONTAINER_SETUP_SCRIPT",
            "/nonexistent/default_container_setup.sh",
        ):
            with self.assertRaises(ValueError) as ctx:
                _resolve_container_setup_script({"image": "x"})
        self.assertIn("default", str(ctx.exception).lower())

    def test_relative_user_path_resolved_to_abspath(self):
        # A relative setup_script is resolved against the cwd and stored absolute.
        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
            f.write("#!/bin/bash\ntrue\n")
            path = f.name
        try:
            rel = os.path.relpath(path, os.getcwd())
            out = _resolve_container_setup_script({"image": "x", "setup_script": rel})
            self.assertTrue(os.path.isabs(out["setup_script"]))
            self.assertEqual(out["setup_script"], os.path.abspath(path))
        finally:
            os.unlink(path)

    def test_tilde_user_path_is_expanded(self):
        # A ~ path is expanded before the existence check (proven via the resolved
        # path in the error message since the file does not exist).
        with self.assertRaises(ValueError) as ctx:
            _resolve_container_setup_script({"image": "x", "setup_script": "~/__cvs_missing_setup__.sh"})
        msg = str(ctx.exception)
        self.assertIn(os.path.expanduser("~"), msg)
        self.assertNotIn("~", msg.split("resolved to", 1)[-1])

    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_orchestratorconfig_init_injects_default_setup_script(self, _mock_pssh):
        # Direct construction routes through __init__ -> the same resolver, so a
        # container config with no setup_script gets the packaged default.
        cfg = OrchestratorConfig(
            orchestrator="container",
            node_dict={"1.1.1.1": {}},
            username="u",
            priv_key_file="/dev/null",
            container={"lifetime": "per_run", "image": "x"},
        )
        self.assertEqual(cfg.container["setup_script"], DEFAULT_CONTAINER_SETUP_SCRIPT)

    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_from_configs_null_setup_script_defaults(self, _mock_pssh):
        # JSON null (Python None) for setup_script is treated like "absent" and
        # resolves to the packaged default -- not "disable provisioning".
        cluster = {
            "orchestrator": "container",
            "node_dict": {"1.1.1.1": {}},
            "username": "u",
            "priv_key_file": "/dev/null",
            "container": {"lifetime": "per_run", "image": "x", "setup_script": None},
        }
        cfg = OrchestratorConfig.from_configs(cluster)
        self.assertEqual(cfg.container["setup_script"], DEFAULT_CONTAINER_SETUP_SCRIPT)

    @patch("cvs.core.orchestrators.baremetal.Pssh")
    def test_baremetal_init_does_not_inject_setup_script(self, _mock_pssh):
        # Empty container block (baremetal) must not gain a setup_script key.
        cfg = OrchestratorConfig(
            orchestrator="baremetal",
            node_dict={"1.1.1.1": {}},
            username="u",
            priv_key_file="/dev/null",
        )
        self.assertEqual(cfg.container, {})


if __name__ == "__main__":
    unittest.main()
