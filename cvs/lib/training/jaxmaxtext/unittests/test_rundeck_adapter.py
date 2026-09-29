'''Unit tests for the JAX MaxText -> shared Run Deck adapter.'''

import unittest

from cvs.lib.report.training_cells import build_training_cells, flatten_training_combo_actuals
from cvs.lib.training.jaxmaxtext.utils.rundeck_adapter import (
    flat_train_res_from_nested,
    jax_cell_dimensions,
)


def _nested():
    return {
        "mode": "distributed",
        "sweeps": {
            "PRECISION=BF16,SEQLEN=4096,BATCH=3": {
                "results": {
                    "training.tflops_per_sec_per_gpu": 185.4,
                    "training.final_loss": 2.01,
                    "training.eval_loss": None,
                },
                "step_metrics": [
                    {"step": 0, "loss": 2.5, "perplexity": 12.1, "TFLOP/s/device": 180.0, "Tokens/s/device": 3000.0},
                    {"step": 10, "loss": 2.0, "perplexity": float("nan"), "TFLOP/s/device": 185.4},
                ],
                "tb_scalars": {
                    "learning/current_learning_rate": [(0, 1e-4), (10, 9e-5)],
                    "learning/grad_norm": [(0, 3.0), (10, 2.0)],
                    "perf/step_time_seconds": [(0, 1.5), (10, 1.4)],
                },
                "planned_steps": 20,
            }
        },
    }


class TestJaxCellDimensions(unittest.TestCase):
    def test_parses_bs_sl_precision_config_key_format(self):
        # The real config keys use BS/SL/PRECISION, e.g. "BS=4,PRECISION=BF16,SL=8192".
        dims = jax_cell_dimensions(None, "BS=4,PRECISION=BF16,SL=8192")
        self.assertEqual(dims, {"bs": "4", "sl": "8192", "precision": "BF16"})

    def test_parses_batch_seqlen_aliases(self):
        dims = jax_cell_dimensions(None, "PRECISION=BF16,SEQLEN=4096,BATCH=3")
        self.assertEqual(dims, {"bs": "3", "sl": "4096", "precision": "BF16"})

    def test_missing_tokens_yield_empty_strings(self):
        self.assertEqual(jax_cell_dimensions(None, "default"), {"bs": "", "sl": "", "precision": ""})


class TestFlatTrainResFromNested(unittest.TestCase):
    def test_metrics_flatten_to_last_value_lists(self):
        flat = flat_train_res_from_nested(_nested())
        combo = flat["PRECISION=BF16,SEQLEN=4096,BATCH=3"]
        self.assertEqual(combo["tflops_per_sec_per_gpu"], ["185.4"])
        self.assertEqual(combo["final_loss"], ["2.01"])
        self.assertNotIn("eval_loss", combo)

    def test_stdout_and_tb_curves_attached_and_nan_dropped(self):
        combo = flat_train_res_from_nested(_nested())["PRECISION=BF16,SEQLEN=4096,BATCH=3"]
        self.assertEqual(combo["_loss_curve"], [[0, 2.5], [10, 2.0]])
        self.assertEqual(combo["_perplexity_curve"], [[0, 12.1]])
        self.assertEqual(combo["_throughput_curve"], [[0, 180.0], [10, 185.4]])
        self.assertEqual(combo["_tokens_curve"], [[0, 3000.0]])
        self.assertEqual(combo["_learning_rate_curve"], [[0, 1e-4], [10, 9e-5]])
        self.assertEqual(combo["_grad_norm_curve"], [[0, 3.0], [10, 2.0]])
        self.assertEqual(combo["_planned_steps"], 20)
        # Unmapped TB tags become selectable extra curves; mapped ones do not.
        self.assertEqual(combo["_extra_curves"], {"perf/step_time_seconds": [[0, 1.5], [10, 1.4]]})
        self.assertNotIn("learning/grad_norm", combo["_extra_curves"])

    def test_flat_output_feeds_shared_flatten(self):
        combo = flat_train_res_from_nested(_nested())["PRECISION=BF16,SEQLEN=4096,BATCH=3"]
        actuals = flatten_training_combo_actuals(combo)
        self.assertEqual(actuals["training.tflops_per_sec_per_gpu"], 185.4)
        self.assertNotIn("training._loss_curve", actuals)


class TestSharedBuilderConsumesJaxData(unittest.TestCase):
    def test_build_cells_via_jax_dimensions(self):
        from types import SimpleNamespace

        from cvs.lib.report.rundeck.config_builder import make_inference_report_config
        from cvs.lib.report.types import ReportChartSeries

        config = make_inference_report_config(
            suite_id="jaxmaxtext",
            results_columns=(
                ("BS", None),
                ("SL", None),
                ("Precision", None),
                ("TFLOPs", "training.tflops_per_sec_per_gpu"),
            ),
            metric_units={"tflops_per_sec_per_gpu": "TFLOP/s/GPU"},
            tier_metric_specs=lambda _cell, _tier: {},
            metric_tier_order=("throughput", "record"),
            metric_prefix="training.",
            cell_highlights=(("tflops_per_sec_per_gpu", "TFLOPs"),),
            chart_series=(ReportChartSeries("tflops_per_sec_per_gpu", "TFLOPs", "TFLOP/s/GPU"),),
            headline_metric="training.tflops_per_sec_per_gpu",
            cell_dimensions=jax_cell_dimensions,
            dimension_fields=(("bs", "BS", "BS="), ("sl", "SL", "SL="), ("precision", "Precision", "")),
            sweep_series_label="JAX MaxText sweep",
        )
        variant = SimpleNamespace(
            gpu_arch="MI300X", enforce_thresholds=False, thresholds={}, train_params={}, cell_key=lambda n: n
        )
        flat = flat_train_res_from_nested(_nested())
        cells = build_training_cells(config, variant, flat, {})
        self.assertEqual(len(cells), 1)
        cell = cells[0]
        self.assertEqual(cell["bs"], "3")
        self.assertEqual(cell["sl"], "4096")
        self.assertEqual(cell["precision"], "BF16")
        self.assertEqual(cell["subtitle"], "BS=3 SL=4096 \u00b7 BF16")
        self.assertEqual(cell["actuals"]["training.tflops_per_sec_per_gpu"], 185.4)
        self.assertEqual(cell["loss_curve"], [[0, 2.5], [10, 2.0]])
        self.assertEqual(cell["learning_rate_curve"], [[0, 1e-4], [10, 9e-5]])
        self.assertEqual(cell["extra_curves"], {"perf/step_time_seconds": [[0, 1.5], [10, 1.4]]})


if __name__ == "__main__":
    unittest.main()
