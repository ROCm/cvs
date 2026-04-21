import unittest

from pydantic import ValidationError

from cvs.schema.rccl import RCCL_WRONG_NONZERO_ERROR_TYPE, RcclTests
from cvs.schema.rccl_config import (
    RcclConfigFileRoot,
    RcclMatrixCartesian,
    RcclMatrixVariants,
    RcclRunInput,
    parse_rccl_thresholds_payload,
    threshold_collective_names,
)


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

    def test_threshold_collective_names_matches_expansion_union(self):
        """Keys allowed in ``validation.thresholds`` must match post-matrix collective names (spec §4.2)."""
        run = RcclRunInput(
            env_script="/e.sh",
            num_ranks=8,
            ranks_per_node=8,
            collectives=["all_reduce_perf", "broadcast_perf"],
            datatype="float",
            start_size="1",
            end_size="2",
            step_factor="2",
            warmups="1",
            iterations="1",
            cycles="1",
        )
        self.assertEqual(threshold_collective_names(run, None), {"all_reduce_perf", "broadcast_perf"})
        variants = RcclMatrixVariants(kind="variants", cases=[{"name": "v1"}])
        self.assertEqual(threshold_collective_names(run, variants), {"all_reduce_perf", "broadcast_perf"})
        cart_no_c = RcclMatrixCartesian(
            kind="cartesian",
            dimensions={"datatype": ["float"]},
        )
        self.assertEqual(threshold_collective_names(run, cart_no_c), {"all_reduce_perf", "broadcast_perf"})
        cart_c = RcclMatrixCartesian(
            kind="cartesian",
            dimensions={"collective": ["all_gather_perf"]},
        )
        self.assertEqual(threshold_collective_names(run, cart_c), {"all_gather_perf"})

    def test_nested_config_rejects_threshold_key_for_base_collective_when_matrix_replaces_them(self):
        root = {
            "rccl": {
                "run": {
                    "env_script": "/e.sh",
                    "num_ranks": 8,
                    "ranks_per_node": 8,
                    "collectives": ["all_reduce_perf"],
                    "datatype": "float",
                    "start_size": "1",
                    "end_size": "2",
                    "step_factor": "2",
                    "warmups": "1",
                    "iterations": "1",
                    "cycles": "1",
                },
                "validation": {
                    "profile": "thresholds",
                    "thresholds": {
                        "all_reduce_perf": {"bus_bw": {"1024": 1.0}},
                        "broadcast_perf": {"bus_bw": {"1024": 1.0}},
                    },
                },
                "artifacts": {
                    "output_dir": "/out",
                    "remote_work_dir": "/rw",
                    "export_raw": False,
                },
                "matrix": {"kind": "cartesian", "dimensions": {"collective": ["broadcast_perf"]}},
            }
        }
        with self.assertRaises(ValidationError) as ctx:
            RcclConfigFileRoot.model_validate(root)
        self.assertIn("all_reduce_perf", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
