'''Unit tests for the framework-agnostic training-cell builder.'''

import unittest
from types import SimpleNamespace

from cvs.lib.report.rundeck.config_builder import make_inference_report_config
from cvs.lib.report.training_cells import (
    _dimension_fields,
    build_training_cells,
    build_training_chart_series,
    build_training_results_table,
    build_training_summaries,
    chart_x_labels,
)
from cvs.lib.report.types import ReportChartSeries


def _parse_jax_sweep(sweep_name):
    dims = {}
    for token in str(sweep_name).split(","):
        if "=" not in token:
            continue
        key, _sep, value = token.partition("=")
        dims[key.strip().lower()] = value.strip()
    return dims


def _jax_dimensions(variant_config, sweep_name):
    dims = _parse_jax_sweep(sweep_name)
    dims["nodes"] = str(getattr(variant_config, "num_nodes", "") or "")
    return dims


def _jax_config():
    return make_inference_report_config(
        suite_id="jaxmaxtext",
        results_columns=(
            ("Model", None),
            ("GPU", None),
            ("BS", None),
            ("SL", None),
            ("Precision", None),
            ("Nodes", None),
            ("Throughput", "training.throughput_per_gpu"),
        ),
        metric_units={"throughput_per_gpu": "tok/s/GPU"},
        tier_metric_specs=lambda _cell, _tier: {},
        metric_tier_order=("throughput", "record"),
        metric_prefix="training.",
        cell_highlights=(("throughput_per_gpu", "Throughput"),),
        chart_series=(ReportChartSeries("throughput_per_gpu", "Throughput", "tok/s/GPU"),),
        headline_metric="training.throughput_per_gpu",
        inference_test_substring="test_training",
        cell_dimensions=_jax_dimensions,
        dimension_fields=(
            ("bs", "BS", "BS="),
            ("sl", "SL", "SL="),
            ("precision", "Precision", ""),
            ("nodes", "Nodes", "N="),
        ),
        sweep_series_label="JAX MaxText sweep",
    )


def _jax_variant():
    return SimpleNamespace(
        gpu_arch="MI300X",
        enforce_thresholds=False,
        thresholds={},
        num_nodes=2,
        train_params={"tokenizer_model": "meta-llama/Llama-3.1-8B"},
        cell_key=lambda name: name,
    )


def _jax_res():
    return {
        "BS=4,SL=8192,PRECISION=BF16": {
            "throughput_per_gpu": ["100", "150"],
            "_loss_curve": [[0, 2.5], [10, 2.0]],
            "_learning_rate_curve": [[0, 1e-4]],
        },
        "BS=8,SL=8192,PRECISION=BF16": {
            "throughput_per_gpu": ["200"],
        },
    }


class TestGenericTrainingCells(unittest.TestCase):
    def test_generic_dimensions_drive_cell_fields_and_subtitle(self):
        cells = build_training_cells(_jax_config(), _jax_variant(), _jax_res(), {})
        self.assertEqual(len(cells), 2)
        first = next(c for c in cells if c["bs"] == "4")
        self.assertEqual(first["sl"], "8192")
        self.assertEqual(first["precision"], "BF16")
        self.assertEqual(first["nodes"], "2")
        self.assertEqual(first["subtitle"], "BS=4 SL=8192 N=2 \u00b7 BF16")
        self.assertEqual(first["concurrency"], "BF16")
        self.assertEqual(first["actuals"]["training.throughput_per_gpu"], 150.0)
        self.assertEqual(first["loss_curve"], [[0, 2.5], [10, 2.0]])
        self.assertEqual(first["learning_rate_curve"], [[0, 1e-4]])
        self.assertNotIn("mbs", first)
        self.assertNotIn("tp", first)

    def test_results_table_headers_map_generic_fields(self):
        config = _jax_config()
        cells = build_training_cells(config, _jax_variant(), _jax_res(), {})
        table = build_training_results_table(config, cells)
        self.assertIn("BS", table["headers"])
        self.assertIn("SL", table["headers"])
        self.assertNotIn("MBS", table["headers"])
        bs_index = table["headers"].index("BS")
        self.assertIn("4", {row[bs_index] for row in table["rows"]})

    def test_chart_labels_and_series_use_generic_dimensions(self):
        config = _jax_config()
        cells = build_training_cells(config, _jax_variant(), _jax_res(), {})
        self.assertEqual(chart_x_labels(cells, [("bs", "BS="), ("sl", "SL=")]), ["BS=4", "BS=8"])
        series = build_training_chart_series(config, cells)
        entry = series["throughput_per_gpu"][0]
        self.assertEqual(entry["label"], "JAX MaxText sweep")
        self.assertEqual(entry["x_labels"], ["BS=4", "BS=8"])

    def test_summary_uses_configured_series_label(self):
        config = _jax_config()
        cells = build_training_cells(config, _jax_variant(), _jax_res(), {})
        summary = build_training_summaries(config, cells)[0]
        self.assertEqual(summary["label"], "JAX MaxText sweep")
        self.assertEqual(summary["meta"], "Peak at BS=8,SL=8192,PRECISION=BF16")


class TestDimensionFieldsFallback(unittest.TestCase):
    def test_empty_dimension_fields_fall_back_to_megatron(self):
        config = SimpleNamespace(dimension_fields=())
        fields = _dimension_fields(config)
        self.assertEqual([f[0] for f in fields], ["mbs", "gbs", "precision", "tp", "pp"])


if __name__ == "__main__":
    unittest.main()
