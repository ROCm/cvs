import unittest
from types import SimpleNamespace

from cvs.lib.report.pytest_extras import sweep_cell_result_key


class TestSweepCellResultKey(unittest.TestCase):
    def test_uses_result_tuple_not_threshold_key(self):
        variant = SimpleNamespace(model=SimpleNamespace(id="org/model"), gpu_arch="mi300x")
        self.assertEqual(
            sweep_cell_result_key(variant, {"name": "balanced", "policy": "ignored"}, "1024", "1024", 16),
            ("org/model", "mi300x", "1024", "1024", "balanced", 16),
        )

    def test_normalizes_inputs_to_result_tuple_types(self):
        variant = SimpleNamespace(model=SimpleNamespace(id="org/model"))
        self.assertEqual(
            sweep_cell_result_key(variant, {"name": 7}, 1024, 1024, "16"),
            ("org/model", "", "1024", "1024", "7", 16),
        )
