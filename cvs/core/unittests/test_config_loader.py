"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

import json
import logging
import tempfile
import unittest
from pathlib import Path

from cvs.core.config import OrchestratorConfig, OrchestratorConfigError, load_config
from cvs.core.transports.pssh import PsshTransport


def _write_cluster(td: Path, payload: dict, name: str = "cluster.json") -> Path:
    p = td / name
    p.write_text(json.dumps(payload))
    return p


def _write_testsuite(td: Path, payload: dict, name: str = "config.json") -> Path:
    p = td / name
    p.write_text(json.dumps(payload))
    return p


class TestLoadConfigLegacyShape(unittest.TestCase):
    """Case 1: production main cluster.json, flat keys only."""

    def test_legacy_flat_keys_synthesize_pssh_transport_and_hostshell_runtime(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            cf = _write_cluster(
                td,
                {
                    "node_dict": {
                        "n1": {"mgmt_ip": "10.0.0.1"},
                        "n2": {"mgmt_ip": "10.0.0.2"},
                    },
                    "head_node_dict": {"name": "n1"},
                    "username": "ci",
                    "priv_key_file": "~/.ssh/id_rsa",
                    "env_vars": {"NCCL_DEBUG": "INFO"},
                },
            )

            cfg = load_config(cf)

            self.assertIsInstance(cfg, OrchestratorConfig)
            # Transport synthesized from flat keys
            self.assertEqual(cfg.transport["name"], "pssh")
            self.assertEqual(cfg.transport["username"], "ci")
            self.assertEqual(cfg.transport["priv_key_file"], "~/.ssh/id_rsa")
            self.assertEqual(cfg.transport["node_dict"]["n1"]["mgmt_ip"], "10.0.0.1")
            self.assertEqual(cfg.transport["env_vars"], {"NCCL_DEBUG": "INFO"})
            # Default runtime / launchers
            self.assertEqual(cfg.runtime, {"name": "hostshell"})
            self.assertEqual(cfg.launchers, {"mpi": {"install_dir": "/opt/openmpi"}})


class TestLoadConfigDockerOverlay(unittest.TestCase):
    """Case 2: legacy flat keys + new runtime overlay."""

    def test_runtime_overlay_replaces_default_runtime(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            cf = _write_cluster(
                td,
                {
                    "node_dict": {"n1": {"mgmt_ip": "10.0.0.1"}},
                    "username": "ci",
                    "priv_key_file": "~/.ssh/id_rsa",
                    "runtime": {
                        "name": "docker",
                        "config": {
                            "image": "rocm/cvs:latest",
                            "container_name": "cvs-runner",
                        },
                    },
                },
            )

            cfg = load_config(cf)

            # Synthesized transport from Case 1 is preserved
            self.assertEqual(cfg.transport["name"], "pssh")
            self.assertEqual(cfg.transport["username"], "ci")
            # Runtime block is surfaced (DockerRuntime.parse_config validates image)
            self.assertEqual(cfg.runtime["name"], "docker")
            self.assertEqual(cfg.runtime["config"]["image"], "rocm/cvs:latest")
            self.assertEqual(cfg.runtime["config"]["container_name"], "cvs-runner")


class TestLoadConfigRejectsFriendIntermediate(unittest.TestCase):
    """Case 3: friend's orchestrator + container shape is explicitly rejected."""

    def test_top_level_orchestrator_key_rejected_with_migration_hint(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            cf = _write_cluster(
                td,
                {
                    "orchestrator": "container",
                    "node_dict": {"n1": {}},
                    "username": "ci",
                    "priv_key_file": "~/.ssh/id_rsa",
                    "container": {"image": "rocm/cvs:latest"},
                },
            )
            with self.assertRaises(OrchestratorConfigError) as ctx:
                load_config(cf)
            msg = str(ctx.exception)
            self.assertIn("orchestrator", msg)
            self.assertIn("container", msg)
            # Migration hint points users at the new runtime block
            self.assertIn("runtime", msg)

    def test_top_level_container_only_also_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            cf = _write_cluster(
                td,
                {
                    "node_dict": {"n1": {}},
                    "username": "ci",
                    "priv_key_file": "~/.ssh/id_rsa",
                    "container": {"image": "rocm/cvs:latest"},
                },
            )
            with self.assertRaises(OrchestratorConfigError):
                load_config(cf)


class TestLoadConfigAggregatedValidation(unittest.TestCase):
    """Case 4: validation errors are aggregated, not raised one-at-a-time."""

    def test_multiple_problems_reported_in_one_exception(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            cf = _write_cluster(
                td,
                {
                    "node_dict": {"n1": {}},
                    # missing username, missing priv_key_file
                    "runtime": {"name": "docker", "config": {}},  # missing image
                },
            )
            with self.assertRaises(OrchestratorConfigError) as ctx:
                load_config(cf)
            problems = ctx.exception.problems
            self.assertGreaterEqual(len(problems), 3, problems)
            joined = "\n".join(problems)
            self.assertIn("username", joined)
            self.assertIn("priv_key_file", joined)
            self.assertIn("image", joined)


class TestLoadConfigEnvVarsRegression(unittest.TestCase):
    """Bug 5 regression: env_vars from cluster.json must thread through to PsshTransport.env_prefix.

    On the friend's branch BaremetalOrchestrator silently dropped env_vars. The
    new loader puts it into the synthesized transport block; PsshTransport
    threads it into both Pssh handles.
    """

    def test_env_vars_propagates_to_transport_env_prefix(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            keyf = td / "fake_key"
            keyf.write_text("not a real key")
            cf = _write_cluster(
                td,
                {
                    "node_dict": {"n1": {"mgmt_ip": "10.0.0.1"}},
                    "username": "ci",
                    "priv_key_file": str(keyf),
                    "env_vars": {
                        "NCCL_DEBUG": "INFO",
                        "PATH": "/opt/rocm/bin:$PATH",
                    },
                },
            )
            cfg = load_config(cf)
            self.assertEqual(
                cfg.transport["env_vars"],
                {"NCCL_DEBUG": "INFO", "PATH": "/opt/rocm/bin:$PATH"},
            )

            # And verify it actually reaches the transport, not just the dict.
            transport = PsshTransport(
                hosts=list(cfg.transport["node_dict"].keys()),
                head_node="n1",
                username=cfg.transport["username"],
                priv_key_file=cfg.transport["priv_key_file"],
                env_vars=cfg.transport.get("env_vars"),
                log=logging.getLogger("test"),
            )
            self.assertIn("NCCL_DEBUG=INFO", transport.env_prefix)
            self.assertIn("/opt/rocm/bin", transport.env_prefix)


class TestLoadConfigTestsuitePassthrough(unittest.TestCase):
    """testsuite_config.json is parsed and exposed at cfg.testsuite verbatim."""

    def test_testsuite_file_is_loaded_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            cf = _write_cluster(
                td,
                {
                    "node_dict": {"n1": {}},
                    "username": "ci",
                    "priv_key_file": "~/.ssh/id_rsa",
                },
            )
            ts = _write_testsuite(
                td, {"rccl": {"no_of_global_ranks": "16", "rccl_dir": "/opt/rccl-tests/"}}
            )
            cfg = load_config(cf, ts)
            self.assertEqual(cfg.testsuite["rccl"]["no_of_global_ranks"], "16")
            self.assertEqual(cfg.testsuite["rccl"]["rccl_dir"], "/opt/rccl-tests/")


class TestLoadConfigRawIsRaw(unittest.TestCase):
    """cfg.raw is byte-for-byte the user input. Loader does NOT silently rewrite."""

    def test_raw_preserves_unrecognized_top_level_keys(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            cf = _write_cluster(
                td,
                {
                    "node_dict": {"n1": {"mgmt_ip": "10.0.0.1", "vpc_ip": "192.168.1.1"}},
                    "username": "ci",
                    "priv_key_file": "~/.ssh/id_rsa",
                    "some_legacy_field": "some_value",
                    "another": {"nested": "thing"},
                },
            )
            cfg = load_config(cf)
            self.assertEqual(cfg.raw["some_legacy_field"], "some_value")
            self.assertEqual(cfg.raw["another"], {"nested": "thing"})
            # node metadata also untouched
            self.assertEqual(
                cfg.raw["node_dict"]["n1"], {"mgmt_ip": "10.0.0.1", "vpc_ip": "192.168.1.1"}
            )


if __name__ == "__main__":
    unittest.main()
