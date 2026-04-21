"""Unit tests for cvs/lib/arch_detect.py (CVS docker-mode P6)."""

import unittest
from unittest.mock import MagicMock

from cvs.lib.arch_detect import _first_gfx_in_rocminfo, detect_cluster_gfx_arch
from cvs.lib.runtime_config import RuntimeConfig


def _rocminfo_with_gfx(gfx):
    """Build a minimal rocminfo-style output containing the given gfx arch."""
    return f"""\
=====================
HSA System Attributes
=====================
HSA Agents
==========
*******
Agent 1
*******
  Name:                    AMD EPYC 7573X 32-Core Processor
  Vendor Name:             CPU
*******
Agent 3
*******
  Name:                    {gfx}
  Marketing Name:
  Vendor Name:             AMD
"""


class TestFirstGfxInRocminfo(unittest.TestCase):
    def test_finds_gfx942(self):
        self.assertEqual(_first_gfx_in_rocminfo(_rocminfo_with_gfx("gfx942")), "gfx942")

    def test_finds_gfx90a(self):
        self.assertEqual(_first_gfx_in_rocminfo(_rocminfo_with_gfx("gfx90a")), "gfx90a")

    def test_returns_none_when_no_gfx(self):
        self.assertIsNone(_first_gfx_in_rocminfo("no gfx here"))

    def test_returns_none_on_empty(self):
        self.assertIsNone(_first_gfx_in_rocminfo(""))
        self.assertIsNone(_first_gfx_in_rocminfo(None))

    def test_skips_cpu_name_line(self):
        # CPU agent's "Name: AMD EPYC ..." must not match.
        out = "  Name:                    AMD EPYC 7573X\n  Name:                    gfx942"
        self.assertEqual(_first_gfx_in_rocminfo(out), "gfx942")

    def test_robust_to_ssh_banner(self):
        # Banner text never contains a top-level "Name: gfx<X>" line.
        prefix = "Some banner text\nWelcome to Ubuntu\n"
        self.assertEqual(
            _first_gfx_in_rocminfo(prefix + _rocminfo_with_gfx("gfx942")), "gfx942"
        )


class TestDetectClusterGfxArch(unittest.TestCase):
    def _phdl(self, per_node_output):
        m = MagicMock()
        m.exec.return_value = per_node_output
        return m

    def _cfg(self, expected=None, container="cvs-runner"):
        return RuntimeConfig(mode="docker", image="x", container_name=container, expected_gfx_arch=expected)

    def test_single_node_single_arch(self):
        phdl = self._phdl({"node-01": _rocminfo_with_gfx("gfx942")})
        self.assertEqual(detect_cluster_gfx_arch(phdl, self._cfg()), "gfx942")

    def test_multi_node_homogeneous(self):
        phdl = self._phdl(
            {
                "node-01": _rocminfo_with_gfx("gfx942"),
                "node-02": _rocminfo_with_gfx("gfx942"),
            }
        )
        self.assertEqual(detect_cluster_gfx_arch(phdl, self._cfg()), "gfx942")

    def test_multi_node_heterogeneous_raises(self):
        phdl = self._phdl(
            {
                "node-01": _rocminfo_with_gfx("gfx942"),
                "node-02": _rocminfo_with_gfx("gfx90a"),
            }
        )
        with self.assertRaises(RuntimeError) as ctx:
            detect_cluster_gfx_arch(phdl, self._cfg())
        msg = str(ctx.exception)
        self.assertIn("single-arch cluster invariant violated", msg)
        self.assertIn("node-01=gfx942", msg)
        self.assertIn("node-02=gfx90a", msg)

    def test_expected_arch_match(self):
        phdl = self._phdl({"node-01": _rocminfo_with_gfx("gfx942")})
        self.assertEqual(detect_cluster_gfx_arch(phdl, self._cfg(expected="gfx942")), "gfx942")

    def test_expected_arch_mismatch_raises(self):
        phdl = self._phdl({"node-01": _rocminfo_with_gfx("gfx90a")})
        with self.assertRaises(RuntimeError) as ctx:
            detect_cluster_gfx_arch(phdl, self._cfg(expected="gfx942"))
        msg = str(ctx.exception)
        self.assertIn("expected_gfx_arch='gfx942'", msg)
        self.assertIn("'gfx90a'", msg)

    def test_unparseable_output_raises(self):
        phdl = self._phdl({"node-01": "rocminfo: command not found"})
        with self.assertRaises(RuntimeError) as ctx:
            detect_cluster_gfx_arch(phdl, self._cfg())
        self.assertIn("could not detect gfx arch on node-01", str(ctx.exception))

    def test_container_name_threaded_into_command(self):
        phdl = self._phdl({"node-01": _rocminfo_with_gfx("gfx942")})
        detect_cluster_gfx_arch(phdl, self._cfg(container="my-runner"))
        sent_cmd = phdl.exec.call_args[0][0]
        self.assertIn("docker exec my-runner", sent_cmd)
        self.assertIn("rocminfo", sent_cmd)


if __name__ == "__main__":
    unittest.main()
