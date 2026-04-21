"""Golden / ordering tests for ``cvs.lib.rccl.matrix_expand``."""

from __future__ import annotations

import types
import unittest

from cvs.lib.rccl.matrix_expand import (
    RcclMatrixExpansionInput,
    expand_rccl_matrix_cases,
    expand_rccl_no_matrix_cases,
    expansion_input_from_rccl_config,
)


def _base_run_slice(**kwargs):
    defaults = dict(
        collectives=("all_reduce_perf",),
        datatype="float",
        start_size="1M",
        end_size="1G",
        step_factor="2",
        warmups="2",
        iterations="5",
        cycles="1",
        matrix=None,
    )
    defaults.update(kwargs)
    if "collectives" in kwargs and isinstance(kwargs["collectives"], list):
        defaults["collectives"] = tuple(kwargs["collectives"])
    return RcclMatrixExpansionInput(**defaults)


class TestRcclMatrixExpand(unittest.TestCase):
    def test_no_matrix_one_case_per_collective_order(self):
        inp = _base_run_slice(collectives=["all_reduce_perf", "broadcast_perf", "all_gather_perf"])
        cases = expand_rccl_matrix_cases(inp)
        self.assertEqual([c.case_id for c in cases], ["c0_all_reduce_perf", "c1_broadcast_perf", "c2_all_gather_perf"])
        self.assertEqual([c.resolved["collective"] for c in cases], list(inp.collectives))
        self.assertTrue(all(c.resolved["env"] == {} for c in cases))

    def test_no_matrix_helper_ignores_matrix_field(self):
        inp = _base_run_slice(
            collectives=["a", "b"],
            matrix={"kind": "variants", "cases": [{"name": "n", "env": {"X": "1"}}]},
        )
        cases = expand_rccl_no_matrix_cases(inp)
        self.assertEqual([c.case_id for c in cases], ["c0_a", "c1_b"])
        self.assertTrue(all(c.resolved["env"] == {} for c in cases))

    def test_no_matrix_display_name_matches_runner_pattern(self):
        inp = _base_run_slice()
        cases = expand_rccl_matrix_cases(inp)
        self.assertEqual(cases[0].name, "all_reduce_perf (float)")

    def test_variants_outer_collectives_inner(self):
        inp = _base_run_slice(
            collectives=["a", "b"],
            matrix={
                "kind": "variants",
                "cases": [
                    {"name": "default", "env": {}},
                    {"name": "p2p", "env": {"NCCL_P2P_NET_CHUNKSIZE": "8388608"}},
                ],
            },
        )
        cases = expand_rccl_matrix_cases(inp)
        self.assertEqual(
            [c.case_id for c in cases],
            [
                "v_default__c0_a",
                "v_default__c1_b",
                "v_p2p__c0_a",
                "v_p2p__c1_b",
            ],
        )
        self.assertEqual(cases[2].resolved["env"], {"NCCL_P2P_NET_CHUNKSIZE": "8388608"})

    def test_cartesian_without_collective_dimension_outer_collectives(self):
        inp = _base_run_slice(
            collectives=["all_reduce_perf", "broadcast_perf"],
            matrix={
                "kind": "cartesian",
                "dimensions": {
                    "datatype": ["float", "bfloat16"],
                },
            },
        )
        cases = expand_rccl_matrix_cases(inp)
        self.assertEqual(len(cases), 4)
        self.assertEqual([c.resolved["collective"] for c in cases], ["all_reduce_perf"] * 2 + ["broadcast_perf"] * 2)
        self.assertEqual([c.resolved["datatype"] for c in cases], ["float", "bfloat16", "float", "bfloat16"])

    def test_cartesian_dimension_key_order_last_key_fastest(self):
        """Sorted keys: datatype, env.A, env.Z — env.Z varies fastest."""
        inp = _base_run_slice(
            matrix={
                "kind": "cartesian",
                "dimensions": {
                    "env.Z": ["z0", "z1"],
                    "datatype": ["f"],
                    "env.A": ["a0", "a1"],
                },
            },
        )
        cases = expand_rccl_matrix_cases(inp)
        env_paths = [tuple(sorted(c.resolved["env"].items())) for c in cases]
        self.assertEqual(
            env_paths,
            [
                (("A", "a0"), ("Z", "z0")),
                (("A", "a0"), ("Z", "z1")),
                (("A", "a1"), ("Z", "z0")),
                (("A", "a1"), ("Z", "z1")),
            ],
        )

    def test_cartesian_collective_dimension_collective_outer_in_product(self):
        """Sorted keys put ``collective`` first → it varies slowest vs datatype."""
        inp = _base_run_slice(
            matrix={
                "kind": "cartesian",
                "dimensions": {
                    "collective": ["all_reduce_perf", "broadcast_perf"],
                    "datatype": ["float", "bfloat16"],
                },
            },
        )
        cases = expand_rccl_matrix_cases(inp)
        self.assertEqual(
            [(c.resolved["collective"], c.resolved["datatype"]) for c in cases],
            [
                ("all_reduce_perf", "float"),
                ("all_reduce_perf", "bfloat16"),
                ("broadcast_perf", "float"),
                ("broadcast_perf", "bfloat16"),
            ],
        )

    def test_cartesian_with_collective_dimension_golden_case_id(self):
        inp = _base_run_slice(
            matrix={
                "kind": "cartesian",
                "dimensions": {
                    "collective": ["all_reduce_perf"],
                    "datatype": ["float"],
                    "env.NCCL_ALGO": ["Ring"],
                    "env.NCCL_PROTO": ["Simple"],
                },
            },
        )
        cases = expand_rccl_matrix_cases(inp)
        self.assertEqual(len(cases), 1)
        self.assertEqual(
            cases[0].case_id,
            "x_collective=all_reduce_perf__datatype=float__env.NCCL_ALGO=Ring__env.NCCL_PROTO=Simple__all_reduce_perf",
        )

    def test_cartesian_subset_matches_spec_readme_example_order(self):
        """Same dimension names as spec §10.3; first tuple is float / Ring / Simple."""
        inp = _base_run_slice(
            matrix={
                "kind": "cartesian",
                "dimensions": {
                    "datatype": ["float", "bfloat16"],
                    "env.NCCL_ALGO": ["Ring", "Tree"],
                    "env.NCCL_PROTO": ["Simple", "LL"],
                },
            },
        )
        cases = expand_rccl_matrix_cases(inp)
        self.assertEqual(len(cases), 8)
        first = cases[0]
        self.assertEqual(first.resolved["datatype"], "float")
        self.assertEqual(
            first.resolved["env"],
            {"NCCL_ALGO": "Ring", "NCCL_PROTO": "Simple"},
        )
        self.assertEqual(
            first.case_id,
            "x_datatype=float__env.NCCL_ALGO=Ring__env.NCCL_PROTO=Simple__all_reduce_perf",
        )

    def test_duplicate_case_id_gets_stable_suffix(self):
        """Normative collision rule: ``__dup`` + digest from canonical resolved JSON."""
        inp = _base_run_slice(
            collectives=["x"],
            matrix={
                "kind": "variants",
                "cases": [
                    {"name": "same", "env": {}},
                    {"name": "same", "env": {"K": "1"}},
                ],
            },
        )
        cases = expand_rccl_matrix_cases(inp)
        self.assertEqual(cases[0].case_id, "v_same__c0_x")
        self.assertTrue(cases[1].case_id.startswith("v_same__c0_x__dup"))
        self.assertEqual(cases[1].resolved["env"], {"K": "1"})

    def test_expansion_input_from_rccl_config_reads_matrix_echo(self):
        cfg = types.SimpleNamespace(
            collectives=["all_reduce_perf"],
            datatype="float",
            start_size="1",
            end_size="2",
            step_factor="2",
            warmups="1",
            iterations="1",
            cycles="1",
            config_echo={
                "matrix": {"kind": "variants", "cases": [{"name": "n", "env": {}}]},
            },
        )
        inp = expansion_input_from_rccl_config(cfg)
        self.assertEqual(inp.matrix, {"kind": "variants", "cases": [{"name": "n", "env": {}}]})

    def test_rejects_unknown_matrix_kind(self):
        with self.assertRaises(ValueError):
            expand_rccl_matrix_cases(_base_run_slice(matrix={"kind": "sparse", "dimensions": {}}))


if __name__ == "__main__":
    unittest.main()
