import unittest

from pydantic import ValidationError

from cvs.schema.rccl import RCCL_WRONG_NONZERO_ERROR_TYPE, RcclTests
from cvs.schema.rccl_config import parse_rccl_thresholds_payload


class TestRcclSchemaValidation(unittest.TestCase):
    def _base_payload(self):
        return {
            "numCycle": 1,
            "name": "allreduce",  # exercise normalization
            "size": 1024,
            "type": "float",
            "redop": "sum",
            "inPlace": 0,
            "time": 1.0,
            "algBw": 100.0,
            "busBw": 90.0,
        }

    def test_wrong_na_normalizes_to_zero_and_passes(self):
        payload = {**self._base_payload(), "wrong": " N/A "}
        parsed = RcclTests.model_validate(payload)
        self.assertEqual(parsed.wrong, 0)

        payload2 = {**self._base_payload(), "wrong": "na"}
        parsed2 = RcclTests.model_validate(payload2)
        self.assertEqual(parsed2.wrong, 0)

    def test_wrong_positive_fails_after_normalization(self):
        payload = {**self._base_payload(), "wrong": 1}
        with self.assertRaises(ValidationError) as ctx:
            RcclTests.model_validate(payload)
        self.assertIn("SEVERE DATA CORRUPTION", str(ctx.exception))
        self.assertTrue(any(e.get("type") == RCCL_WRONG_NONZERO_ERROR_TYPE for e in ctx.exception.errors()))

        payload2 = {**self._base_payload(), "wrong": "1"}
        with self.assertRaises(ValidationError) as ctx2:
            RcclTests.model_validate(payload2)
        self.assertIn("SEVERE DATA CORRUPTION", str(ctx2.exception))
        self.assertTrue(any(e.get("type") == RCCL_WRONG_NONZERO_ERROR_TYPE for e in ctx2.exception.errors()))

    def test_thresholds_payload_rejects_extra_fields_under_collective(self):
        """per-collective object allows only ``bus_bw`` (strict shape)."""
        with self.assertRaises(ValidationError) as ctx:
            parse_rccl_thresholds_payload({"all_reduce_perf": {"bus_bw": {"1024": 1.0}, "legacy_results": {}}})
        self.assertIn("extra", str(ctx.exception).lower())

    def test_thresholds_payload_rejects_non_numeric_bus_bw_entries(self):
        with self.assertRaises(ValidationError) as ctx:
            parse_rccl_thresholds_payload({"all_reduce_perf": {"bus_bw": {"1024": "not-a-number"}}})
        self.assertIn("number", str(ctx.exception).lower())

    def test_thresholds_payload_rejects_empty_message_size_key(self):
        with self.assertRaises(ValidationError) as ctx:
            parse_rccl_thresholds_payload({"all_reduce_perf": {"bus_bw": {"": 1.0}}})
        self.assertIn("bus_bw", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
