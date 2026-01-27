import unittest

from pydantic import ValidationError

from cvs.schema.rccl import RcclTests


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

        payload2 = {**self._base_payload(), "wrong": "1"}
        with self.assertRaises(ValidationError) as ctx2:
            RcclTests.model_validate(payload2)
        self.assertIn("SEVERE DATA CORRUPTION", str(ctx2.exception))


if __name__ == "__main__":
    unittest.main()
