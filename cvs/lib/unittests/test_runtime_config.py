"""Unit tests for cvs/lib/runtime_config.py (CVS docker-mode P3)."""

import unittest

from cvs.lib.runtime_config import (
    DEFAULT_INSTALLS_NO_AGFHC,
    DEFAULT_INSTALLS_WITH_AGFHC,
    RuntimeConfig,
    RuntimeConfigError,
    parse_runtime,
)


class TestParseRuntimeAbsentBlock(unittest.TestCase):
    """Cluster.json files without a `runtime` block must default to host mode."""

    def test_no_runtime_key(self):
        cfg = parse_runtime({"username": "u", "node_dict": {"h": {}}})
        self.assertEqual(cfg.mode, "host")
        self.assertFalse(cfg.is_docker())
        self.assertIsNone(cfg.image)

    def test_explicit_none_runtime(self):
        cfg = parse_runtime({"runtime": None})
        self.assertEqual(cfg.mode, "host")

    def test_empty_dict_input(self):
        # Defensive: parser should not crash on edge inputs that prior code
        # never produced but a user might construct manually.
        cfg = parse_runtime({})
        self.assertEqual(cfg.mode, "host")

    def test_host_mode_explicit(self):
        cfg = parse_runtime({"runtime": {"mode": "host"}})
        self.assertEqual(cfg.mode, "host")
        self.assertFalse(cfg.is_docker())

    def test_host_mode_ignores_docker_fields(self):
        # A user can stub `mode: host` and leave docker fields lying around;
        # parser must not error -- it just returns a host-mode config.
        cfg = parse_runtime(
            {"runtime": {"mode": "host", "image": "leftover", "container_envs": {}}}
        )
        self.assertEqual(cfg.mode, "host")
        self.assertIsNone(cfg.image)


class TestParseRuntimeDockerValid(unittest.TestCase):
    def test_minimal_docker(self):
        cfg = parse_runtime(
            {"runtime": {"mode": "docker", "image": "ghcr.io/x/cvs-runner:1"}}
        )
        self.assertTrue(cfg.is_docker())
        self.assertEqual(cfg.image, "ghcr.io/x/cvs-runner:1")
        # All optional fields default sensibly
        self.assertEqual(cfg.container_name, "cvs-runner")
        self.assertEqual(cfg.container_envs, {})
        self.assertEqual(cfg.ensure_image, "pull")
        self.assertIsNone(cfg.agfhc_tarball)
        self.assertIsNone(cfg.expected_gfx_arch)
        self.assertIsNone(cfg.installs)

    def test_full_docker(self):
        cfg = parse_runtime(
            {
                "runtime": {
                    "mode": "docker",
                    "image": "ghcr.io/x/cvs-runner:1",
                    "container_name": "cvs-runner-test",
                    "container_envs": {"NCCL_DEBUG": "INFO"},
                    "ensure_image": "load:/var/lib/cvs/img.tar",
                    "agfhc_tarball": "/var/lib/cvs/agfhc.tar.bz2",
                    "expected_gfx_arch": "gfx942",
                    "installs": ["install_rvs", "install_transferbench"],
                }
            }
        )
        self.assertEqual(cfg.container_name, "cvs-runner-test")
        self.assertEqual(cfg.container_envs, {"NCCL_DEBUG": "INFO"})
        self.assertEqual(cfg.ensure_image, "load:/var/lib/cvs/img.tar")
        self.assertEqual(cfg.agfhc_tarball, "/var/lib/cvs/agfhc.tar.bz2")
        self.assertEqual(cfg.expected_gfx_arch, "gfx942")
        self.assertEqual(cfg.installs, ["install_rvs", "install_transferbench"])

    def test_empty_installs_list_preserved(self):
        # Explicit empty list means "do not run any installs at prepare_runtime"
        # (user has pre-baked their own image). Must NOT be silently turned
        # into the default list.
        cfg = parse_runtime(
            {
                "runtime": {
                    "mode": "docker",
                    "image": "x",
                    "installs": [],
                }
            }
        )
        self.assertEqual(cfg.installs, [])
        self.assertEqual(cfg.resolved_installs(), [])


class TestResolvedInstalls(unittest.TestCase):
    def test_default_no_agfhc(self):
        cfg = RuntimeConfig(mode="docker", image="x")
        self.assertEqual(cfg.resolved_installs(), list(DEFAULT_INSTALLS_NO_AGFHC))

    def test_default_with_agfhc(self):
        cfg = RuntimeConfig(
            mode="docker", image="x", agfhc_tarball="/path/to/agfhc.tar.bz2"
        )
        self.assertEqual(cfg.resolved_installs(), list(DEFAULT_INSTALLS_WITH_AGFHC))

    def test_explicit_overrides_default(self):
        cfg = RuntimeConfig(
            mode="docker",
            image="x",
            agfhc_tarball="/p",  # would normally add install_agfhc
            installs=["install_rvs"],  # explicit list wins
        )
        self.assertEqual(cfg.resolved_installs(), ["install_rvs"])


class TestParseRuntimeErrors(unittest.TestCase):
    def _err(self, runtime):
        with self.assertRaises(RuntimeConfigError):
            parse_runtime({"runtime": runtime})

    def test_runtime_not_object(self):
        self._err("not an object")
        self._err(["a", "list"])

    def test_invalid_mode(self):
        self._err({"mode": "kubernetes"})

    def test_docker_missing_image(self):
        self._err({"mode": "docker"})

    def test_docker_empty_image(self):
        self._err({"mode": "docker", "image": ""})
        self._err({"mode": "docker", "image": "   "})
        self._err({"mode": "docker", "image": 42})

    def test_invalid_container_name(self):
        self._err({"mode": "docker", "image": "x", "container_name": ""})
        self._err({"mode": "docker", "image": "x", "container_name": 0})

    def test_invalid_container_envs(self):
        self._err({"mode": "docker", "image": "x", "container_envs": "not a dict"})
        self._err({"mode": "docker", "image": "x", "container_envs": {1: "v"}})
        self._err({"mode": "docker", "image": "x", "container_envs": {"k": 1}})

    def test_invalid_ensure_image(self):
        self._err({"mode": "docker", "image": "x", "ensure_image": "fetch"})
        self._err({"mode": "docker", "image": "x", "ensure_image": 1})

    def test_invalid_agfhc_tarball(self):
        self._err({"mode": "docker", "image": "x", "agfhc_tarball": ""})
        self._err({"mode": "docker", "image": "x", "agfhc_tarball": 0})

    def test_invalid_expected_gfx_arch(self):
        self._err({"mode": "docker", "image": "x", "expected_gfx_arch": ""})
        self._err({"mode": "docker", "image": "x", "expected_gfx_arch": 0})

    def test_invalid_installs(self):
        self._err({"mode": "docker", "image": "x", "installs": "not a list"})
        self._err({"mode": "docker", "image": "x", "installs": [42]})
        self._err({"mode": "docker", "image": "x", "installs": ["bad_name"]})

    def test_ensure_image_load_form(self):
        # load:<path> shape is permitted (validation here is lightweight)
        cfg = parse_runtime(
            {
                "runtime": {
                    "mode": "docker",
                    "image": "x",
                    "ensure_image": "load:/tmp/img.tar",
                }
            }
        )
        self.assertEqual(cfg.ensure_image, "load:/tmp/img.tar")


class TestExistingClusterJsonRoundTrip(unittest.TestCase):
    """Every existing cluster.json file shipped with CVS must parse cleanly
    with mode == 'host' (no behavior change). This is the most important
    backward-compat invariant for P3."""

    def test_shipped_cluster_json(self):
        # Resolve the path relative to the cvs package directory rather than
        # via importlib.resources, because `cvs.input.cluster_file` is not a
        # proper Python package (no __init__.py) on this codebase. This test
        # is the most important backward-compat invariant for P3, so it must
        # not be skipped due to packaging quirks.
        import json
        import os
        import cvs

        cvs_dir = os.path.dirname(cvs.__file__)
        cluster_json_path = os.path.join(
            cvs_dir, "input", "cluster_file", "cluster.json"
        )
        self.assertTrue(
            os.path.exists(cluster_json_path),
            f"shipped cluster.json missing at {cluster_json_path}",
        )
        with open(cluster_json_path) as f:
            cluster_dict = json.load(f)
        cfg = parse_runtime(cluster_dict)
        self.assertEqual(cfg.mode, "host")
        self.assertFalse(cfg.is_docker())


if __name__ == "__main__":
    unittest.main()
