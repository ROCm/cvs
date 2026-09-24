import unittest

from pydantic import ValidationError

from cvs.schema.rccl import RcclTests, RcclTestsMultinodeRaw


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

    def test_multinode_topology_round_trips(self):
        topology = {'nodes': 2, 'ranks': 16, 'ranksPerNode': 8, 'gpusPerRank': 1}
        payload = {**self._base_payload(), **topology, 'wrong': '0'}
        parsed = RcclTestsMultinodeRaw.model_validate(payload)
        for key, value in topology.items():
            self.assertEqual(parsed.model_dump()[key], value)

    def test_inconsistent_rank_identity_fails(self):
        payload = {**self._base_payload(), 'nodes': 2, 'ranks': 2, 'ranksPerNode': 2, 'gpusPerRank': 8, 'wrong': '0'}
        with self.assertRaisesRegex(ValidationError, 'must equal nodes'):
            RcclTestsMultinodeRaw.model_validate(payload)

    def test_schema_accepts_captured_global_local_rank_label_swap(self):
        """Values captured from a real multi-node run exhibiting the AIMVT-334 label swap."""
        payload = {
            'numCycle': 0,
            'name': 'AllReduce',
            'nodes': 1,
            'ranks': 2,
            'ranksPerNode': 2,
            'gpusPerRank': 8,
            'size': 8,
            'type': 'float',
            'redop': 'sum',
            'inPlace': 0,
            'time': 65.3309,
            'algBw': 0.000122,
            'busBw': 0.00023,
            'wrong': '0',
        }
        parsed = RcclTestsMultinodeRaw.model_validate(payload)
        self.assertEqual(parsed.model_dump(), {**payload, 'wrong': 0})


if __name__ == "__main__":
    unittest.main()
