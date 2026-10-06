'''Unit tests for the per-sweep, per-metric verdict panel wiring in _common.metric.

Each sweep's test_metric evaluates every TRAINING_METRIC and records a pass/fail/
skip/record verdict row (consumed by the collapsible benchmark-metrics panel),
failing the test as a whole only when a gated metric is violated.
'''

import unittest
from types import SimpleNamespace

from _pytest.stash import Stash

from cvs.lib.report.benchmark_metric_registry import benchmark_metric_rows_for_nodeid
from cvs.tests.training.jaxmaxtext import _common

try:
    from _pytest.outcomes import Failed, Skipped
except ImportError:  # pragma: no cover - pytest always provides these
    Failed = Skipped = BaseException

_SWEEP = "BS=4,PRECISION=BF16,SL=8192"


def _request(nodeid):
    return SimpleNamespace(node=SimpleNamespace(nodeid=nodeid, stash=Stash()))


def _variant(thresholds, enforce=True):
    return SimpleNamespace(thresholds={_SWEEP: thresholds}, enforce_thresholds=enforce)


def _res(**metrics):
    return {"sweeps": {_SWEEP: {"results": {f"training.{k}": v for k, v in metrics.items()}}}}


def _rows_by_metric(nodeid):
    return {r["metric"]: r for r in benchmark_metric_rows_for_nodeid(nodeid)}


class BenchmarkMetricRowTests(unittest.TestCase):
    def test_row_shape_and_enforced_flag(self):
        row = _common._benchmark_metric_row("final_loss", {"kind": "max", "value": 3.0}, 2.1, "pass")
        self.assertEqual(row["metric"], "final_loss")
        self.assertEqual(row["status"], "pass")
        self.assertEqual(row["actual"], 2.1)
        self.assertTrue(row["enforced"])  # gated pass by default
        self.assertEqual(row["node"], "")
        # non-gating verdicts (record-only / skip) pass enforced=False explicitly
        self.assertFalse(_common._benchmark_metric_row("x", None, 1.0, "pass", enforced=False)["enforced"])
        self.assertFalse(_common._benchmark_metric_row("x", None, None, "skip", enforced=False)["enforced"])

    def test_no_record_status_is_used(self):
        # Verdicts mirror sglang: pass / fail / skip only (no "record").
        statuses = {
            _common._benchmark_metric_row("a", None, 1.0, "pass", enforced=False)["status"],
            _common._benchmark_metric_row("b", None, None, "skip", enforced=False)["status"],
        }
        self.assertEqual(statuses, {"pass", "skip"})


class MetricVerdictTests(unittest.TestCase):
    def test_failure_is_aggregated_and_every_metric_recorded(self):
        nodeid = "jaxmaxtext_distributed.py::test_metric[fail-case]"
        variant = _variant(
            {
                "training.tflops_per_sec_per_gpu": {"kind": "min", "value": 100},
                "training.final_loss": {"kind": "max", "value": 3.0},
            }
        )
        res = _res(tflops_per_sec_per_gpu=200.0, final_loss=9.0)
        with self.assertRaises(Failed):
            _common.metric(_SWEEP, res, variant, SimpleNamespace(failed=False), _request(nodeid))
        rows = _rows_by_metric(nodeid)
        # one verdict per declared metric (missing ones recorded as skip)
        self.assertEqual(len(rows), len(_common.TRAINING_METRICS))
        self.assertEqual(rows["tflops_per_sec_per_gpu"]["status"], "pass")
        self.assertEqual(rows["final_loss"]["status"], "fail")
        self.assertEqual(rows["tokens_per_sec_total"]["status"], "skip")

    def test_all_pass_does_not_raise(self):
        nodeid = "jaxmaxtext_distributed.py::test_metric[pass-case]"
        variant = _variant({"training.tflops_per_sec_per_gpu": {"kind": "min", "value": 100}})
        res = _res(tflops_per_sec_per_gpu=200.0)
        _common.metric(_SWEEP, res, variant, SimpleNamespace(failed=False), _request(nodeid))
        self.assertEqual(_rows_by_metric(nodeid)["tflops_per_sec_per_gpu"]["status"], "pass")

    def test_not_enforced_present_value_passes_non_gating(self):
        nodeid = "jaxmaxtext_distributed.py::test_metric[record-case]"
        variant = _variant({"training.tflops_per_sec_per_gpu": {"kind": "min", "value": 100}}, enforce=False)
        res = _res(tflops_per_sec_per_gpu=50.0)  # below gate, but not enforced -> pass (non-gating)
        _common.metric(_SWEEP, res, variant, SimpleNamespace(failed=False), _request(nodeid))
        row = _rows_by_metric(nodeid)["tflops_per_sec_per_gpu"]
        self.assertEqual(row["status"], "pass")
        self.assertFalse(row["enforced"])  # gate shown as reference, did not gate

    def test_info_spec_passes_non_gating(self):
        nodeid = "jaxmaxtext_distributed.py::test_metric[info-case]"
        variant = _variant({"training.final_loss": {"kind": "info", "value": 2.0}})
        res = _res(final_loss=2.5)
        _common.metric(_SWEEP, res, variant, SimpleNamespace(failed=False), _request(nodeid))
        row = _rows_by_metric(nodeid)["final_loss"]
        self.assertEqual(row["status"], "pass")
        self.assertFalse(row["enforced"])

    def test_skips_when_no_results(self):
        variant = _variant({})
        with self.assertRaises(Skipped):
            _common.metric(
                _SWEEP, {"sweeps": {}}, variant, SimpleNamespace(failed=False), _request("x::test_metric[s]")
            )


if __name__ == "__main__":
    unittest.main()
