'''Unit tests for rccl-tests install GPU architecture detection and early exit.'''

import unittest
from unittest.mock import MagicMock, patch

from _pytest.outcomes import Failed

from cvs.tests.rccl import install_rccl_tests as install_mod
from cvs.tests.rccl.install_rccl_tests import _detect_gpu_arch, _gpu_archs_from_rocminfo


_MI300_ROCMINFO = """\
  Name:                    gfx942
      Name:                    amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-
      Name:                    amdgcn-amd-amdhsa--gfx9-4-generic:sramecc+:xnack-
  Name:                    gfx942
      Name:                    amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-
      Name:                    amdgcn-amd-amdhsa--gfx9-4-generic:sramecc+:xnack-
"""

_GFX12_ROCMINFO = """\
  Name:                    gfx1250
      Name:                    amdgcn-amd-amdhsa--gfx1250:sramecc+:xnack-
      Name:                    amdgcn-amd-amdhsa--gfx12-generic:sramecc+:xnack-
"""


class TestGpuArchFromRocminfo(unittest.TestCase):
    def test_agent_name_ignores_generic_isa(self):
        self.assertEqual(_gpu_archs_from_rocminfo(_MI300_ROCMINFO), ['gfx942'])
        self.assertEqual(_gpu_archs_from_rocminfo(_GFX12_ROCMINFO), ['gfx1250'])

    def test_letter_suffix_arch_is_kept(self):
        output = "  Name:                    gfx90a\n"
        self.assertEqual(_gpu_archs_from_rocminfo(output), ['gfx90a'])

    def test_empty_output(self):
        self.assertEqual(_gpu_archs_from_rocminfo(""), [])


class TestDetectGpuArch(unittest.TestCase):
    def test_configured_arch_skips_rocminfo(self):
        hdl = MagicMock()
        detected = _detect_gpu_arch(hdl, {"gpu_arch": "gfx942"})
        self.assertEqual(detected, "gfx942")
        hdl.exec.assert_not_called()

    def test_rocminfo_agent_name_is_used(self):
        hdl = MagicMock()
        hdl.exec.return_value = {"node0": _MI300_ROCMINFO}
        detected = _detect_gpu_arch(hdl, {"rocm_path": "/opt/rocm"})
        self.assertEqual(detected, "gfx942")
        hdl.exec.assert_called_once()
        cmd = hdl.exec.call_args.args[0]
        self.assertIn("/opt/rocm/bin/rocminfo", cmd)
        self.assertNotIn("grep", cmd)

    def test_rocminfo_failure_returns_empty(self):
        hdl = MagicMock()
        hdl.exec.side_effect = OSError("rocminfo missing")
        self.assertEqual(_detect_gpu_arch(hdl, {}), "")


class TestInstallRcclTestsEarlyExit(unittest.TestCase):
    def test_rccl_lib_failure_skips_rccl_tests_build(self):
        orch = MagicMock()
        config = {
            "nfs_install": "True",
            "rccl_lib_install": "True",
            "rccl_lib_install_dir": "/tmp/rccl-build",
            "rccl_tests_install_dir": "/tmp/rccl-tests",
            "rccl_repository": "https://example.com/rccl.git",
            "ompi_install_dir": "/opt/ompi",
        }

        def _lib_failed(*_args, **_kwargs):
            install_mod.fail_test("RCCL library build failed")
            return None, ""

        with (
            patch.object(install_mod, "_check_install_build_deps"),
            patch.object(install_mod, "_check_ompi_installed", return_value=True),
            patch.object(install_mod, "_install_rccl_lib", side_effect=_lib_failed),
            patch.object(install_mod, "_install_rccl_tests") as build_tests,
        ):
            with self.assertRaises(Failed):
                install_mod.test_install_rccl_tests(orch, config)
        build_tests.assert_not_called()
