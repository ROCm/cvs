'''Unit tests for atom dmesg helpers.'''

import unittest
from unittest.mock import MagicMock

from cvs.lib.inference.atom.atom_dmesg import capture_dmesg_timestamp, verify_dmesg_window


class TestAtomDmesg(unittest.TestCase):
    def test_capture_dmesg_timestamp(self):
        orch = MagicMock()
        orch.exec.return_value = {"n1": "Mon Jan  2 03:04:05 2026"}
        out = capture_dmesg_timestamp(orch)
        self.assertEqual(out["n1"], "Mon Jan  2 03:04:05 2026")

    def test_verify_dmesg_window_skips_when_missing_times(self):
        orch = MagicMock()
        out = verify_dmesg_window(orch, {}, {"n1": "t"})
        self.assertEqual(out, {})
        orch.exec.assert_not_called()


if __name__ == "__main__":
    unittest.main()
