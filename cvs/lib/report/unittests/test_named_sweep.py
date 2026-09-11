'''Tests for named-cell sweep records used by training decks.'''

import unittest
from dataclasses import replace
from types import SimpleNamespace

from cvs.lib.report.cell_build import build_cell_record
from cvs.lib.report.inference_payload import build_chart_series
from cvs.lib.report.render.gate_matrix import build_gate_matrix_rows
from cvs.lib.report.sweep_shape import group_cells_by_shape
from cvs.lib.report.testing.fixtures import generic_inference_report_config
from cvs.lib.report.types import ReportChartSeries


class TestNamedSweep(unittest.TestCase):
    def test_named_cell_uses_embedded_id_and_label_axis(self):
        cell_id = "MBS=2,GBS=32,PRECISION=BF16"
        config = replace(
            generic_inference_report_config(),
            metric_prefix="training.",
            cell_highlights=(("tokens_per_sec", "Tokens/s"),),
            chart_series=(ReportChartSeries("tokens_per_sec", "Training throughput vs cells", "tok/s"),),
            sweep_throughput_metric="training.tokens_per_sec",
            headline_metric="training.tokens_per_sec",
            tier_metric_specs=lambda cell, tier: cell if tier == "throughput" else {},
        )
        variant = SimpleNamespace(
            enforce_thresholds=True,
            thresholds={
                cell_id: {
                    "training.tokens_per_sec": {"kind": "min", "value": 1000},
                }
            },
            cell_key=lambda *_args: self.fail("named cells must not use inference cell_key"),
        )
        cell = build_cell_record(
            config,
            key=("org/model", "MI355X", "2", "32", cell_id, "BF16"),
            host="all nodes",
            actuals={
                "_rundeck_cell_id": cell_id,
                "training.tokens_per_sec": 1200,
            },
            variant_config=variant,
            lifecycle_report={},
            multi_host=False,
        )

        self.assertTrue(cell["named_cell"])
        self.assertEqual(cell["cell_id"], cell_id)
        self.assertEqual(cell["sweep_label"], cell_id)
        self.assertEqual(cell["tiers"]["throughput"], "pass")
        self.assertEqual(list(group_cells_by_shape([cell])), [("", "")])
        self.assertEqual(
            build_chart_series(config, [cell])["tokens_per_sec"][0]["points"],
            [(cell_id, 1200.0)],
        )
        self.assertEqual(build_gate_matrix_rows([cell])[0]["label"], cell_id)


if __name__ == "__main__":
    unittest.main()
