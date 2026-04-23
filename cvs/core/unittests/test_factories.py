"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

import logging
import unittest

from cvs.core.errors import OrchestratorConfigError
from cvs.core.launchers.factory import build_launchers
from cvs.core.launchers.mpi import MpiLauncher
from cvs.core.runtimes.docker import DockerRuntime
from cvs.core.runtimes.factory import build_runtime
from cvs.core.runtimes.hostshell import HostShellRuntime


class TestBuildRuntime(unittest.TestCase):
    """Catches the silent factory rot when classes get renamed or a runtime
    branch gets accidentally deleted."""

    def test_hostshell_default_when_no_block_provided(self):
        rt = build_runtime({})
        self.assertIsInstance(rt, HostShellRuntime)

    def test_explicit_hostshell(self):
        rt = build_runtime({"name": "hostshell"})
        self.assertIsInstance(rt, HostShellRuntime)

    def test_docker_with_required_image(self):
        rt = build_runtime({"name": "docker", "config": {"image": "rocm/cvs:latest"}})
        self.assertIsInstance(rt, DockerRuntime)
        self.assertEqual(rt.image, "rocm/cvs:latest")

    def test_docker_missing_image_raises(self):
        with self.assertRaises(OrchestratorConfigError) as ctx:
            build_runtime({"name": "docker", "config": {}})
        self.assertIn("image", str(ctx.exception))

    def test_unknown_runtime_name_raises_with_message(self):
        with self.assertRaises(OrchestratorConfigError) as ctx:
            build_runtime({"name": "enroot"})
        msg = str(ctx.exception)
        self.assertIn("enroot", msg)
        self.assertIn("not implemented", msg)

    def test_apptainer_also_rejected(self):
        with self.assertRaises(OrchestratorConfigError):
            build_runtime({"name": "apptainer"})

    def test_name_is_case_insensitive(self):
        rt = build_runtime({"name": "DOCKER", "config": {"image": "rocm/cvs:latest"}})
        self.assertIsInstance(rt, DockerRuntime)


class TestBuildLaunchers(unittest.TestCase):
    def test_empty_dict_yields_no_launchers(self):
        self.assertEqual(build_launchers({}), {})
        self.assertEqual(build_launchers(None), {})

    def test_mpi_with_install_dir(self):
        launchers = build_launchers({"mpi": {"install_dir": "/usr"}})
        self.assertIn("mpi", launchers)
        self.assertIsInstance(launchers["mpi"], MpiLauncher)
        self.assertEqual(launchers["mpi"].install_dir, "/usr")

    def test_unknown_launcher_name_raises(self):
        with self.assertRaises(OrchestratorConfigError) as ctx:
            build_launchers({"torchrun": {}})
        self.assertIn("torchrun", str(ctx.exception))


class TestBuildTransport(unittest.TestCase):
    """build_transport lives in cvs.core.factory; covered indirectly via load_config
    elsewhere, but assert the unknown-name case here."""

    def test_unknown_transport_name_raises(self):
        from cvs.core.factory import build_transport

        with self.assertRaises(OrchestratorConfigError) as ctx:
            build_transport({"name": "slurm"}, log=logging.getLogger("test"))
        self.assertIn("slurm", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
