'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs.lib.inference.utils.vllm_server_metrics.

Black-box tests authored from the behavioral spec only (impl-blind). The
module contains pure parsers for vLLM's engine-side Prometheus `/metrics`
exposition-format text: no I/O, no hardware, pure text/dict transformations.

Contract under test (from VLLM_PROMETHEUS_METRICS_SPEC.md Sec 3.2/7):
  parse_prometheus_text(raw) -> {metric_name: {"buckets": {le: count},
      "sum": float, "count": float}} for histograms, {metric_name: float}
      for bare gauges. Degrades to {} on empty/unparseable input; never raises.
  diff_histogram(before, after) -> {le: after_count - before_count}, clamped
      to >= 0. None if `after` has no buckets.
  histogram_quantile(buckets, q) -> linear interpolation between bucket
      boundaries; None on empty/zero-count buckets.
  to_prom_metrics(before_text, after_text) -> the composed prom.* dict;
      all-None (never partial, never a raise) if either scrape is
      missing/unparseable.

Framework: unittest.TestCase + self.subTest + unittest.mock (no pytest),
matching test_gpu.py's conventions.
'''

import unittest
from unittest.mock import MagicMock

from cvs.lib.inference.utils.vllm_server_metrics import (
    PROM_METRICS,
    PROM_METRIC_UNITS,
    diff_histogram,
    histogram_quantile,
    parse_prometheus_text,
    to_prom_metrics,
)
from cvs.lib.inference.vllm_job import scrape_vllm_metrics

# Shared bucket list (queue/prefill/decode/inference/e2e), seconds -- per
# VLLM_PROMETHEUS_METRICS_SPEC.md Sec 1.
_BUCKETS = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0]


def _histogram_text(name: str, cumulative_counts: dict, total_sum: float) -> str:
    """Build real Prometheus exposition-format text for one histogram metric.

    cumulative_counts: {le_str: cumulative_count}, must include "+Inf".
    """
    lines = [f"# HELP {name} test histogram", f"# TYPE {name} histogram"]
    for le, count in cumulative_counts.items():
        lines.append(f'{name}_bucket{{le="{le}"}} {count}')
    total = cumulative_counts["+Inf"]
    lines.append(f"{name}_sum {total_sum}")
    lines.append(f"{name}_count {total}")
    return "\n".join(lines)


def _full_scrape_text(queue_counts, queue_sum, prefill_counts, prefill_sum) -> str:
    parts = [
        _histogram_text("vllm:request_queue_time_seconds", queue_counts, queue_sum),
        _histogram_text("vllm:request_prefill_time_seconds", prefill_counts, prefill_sum),
        "# HELP vllm:num_requests_waiting test gauge",
        "# TYPE vllm:num_requests_waiting gauge",
        "vllm:num_requests_waiting 0",
    ]
    return "\n".join(parts)


def _cumulative(observations: list) -> dict:
    """Bucket a list of raw second-values into cumulative le-counts using
    the real shared bucket list, plus '+Inf'."""
    counts = {}
    running = 0
    for b in _BUCKETS:
        running += sum(1 for o in observations if o <= b)
        counts[str(b)] = float(running)
    counts["+Inf"] = float(len(observations))
    return counts


class TestParsePrometheusText(unittest.TestCase):
    def test_empty_and_none_degrade_to_empty_dict(self):
        for raw in (None, "", "   ", "\n\n"):
            with self.subTest(raw=repr(raw)):
                self.assertEqual(parse_prometheus_text(raw), {})

    def test_parses_histogram_buckets_sum_count(self):
        text = _histogram_text(
            "vllm:request_queue_time_seconds",
            {"0.3": 2.0, "0.5": 3.0, "+Inf": 5.0},
            total_sum=1.75,
        )
        out = parse_prometheus_text(text)
        self.assertIn("vllm:request_queue_time_seconds", out)
        hist = out["vllm:request_queue_time_seconds"]
        self.assertEqual(hist["buckets"], {"0.3": 2.0, "0.5": 3.0, "+Inf": 5.0})
        self.assertEqual(hist["sum"], 1.75)
        self.assertEqual(hist["count"], 5.0)

    def test_ignores_help_and_type_comment_lines(self):
        text = "\n".join(
            [
                "# HELP vllm:request_queue_time_seconds queue wait time",
                "# TYPE vllm:request_queue_time_seconds histogram",
                'vllm:request_queue_time_seconds_bucket{le="0.3"} 1',
                "vllm:request_queue_time_seconds_sum 0.2",
                "vllm:request_queue_time_seconds_count 1",
            ]
        )
        out = parse_prometheus_text(text)
        self.assertEqual(set(out.keys()), {"vllm:request_queue_time_seconds"})

    def test_parses_bare_gauge_line(self):
        text = "\n".join(
            [
                "# TYPE vllm:num_requests_waiting gauge",
                "vllm:num_requests_waiting 3",
            ]
        )
        out = parse_prometheus_text(text)
        self.assertEqual(out["vllm:num_requests_waiting"], 3.0)

    def test_multiple_metrics_coexist(self):
        text = _full_scrape_text(
            _cumulative([0.1, 0.2]), 0.3, _cumulative([0.4]), 0.4
        )
        out = parse_prometheus_text(text)
        self.assertIn("vllm:request_queue_time_seconds", out)
        self.assertIn("vllm:request_prefill_time_seconds", out)
        self.assertIn("vllm:num_requests_waiting", out)

    def test_never_raises_on_malformed_lines(self):
        garbage_texts = [
            "not a valid prometheus line at all",
            "vllm:request_queue_time_seconds_bucket{le=\"not_a_number_or_inf\"} abc",
            "\x00\x01\x02 binary garbage",
            "vllm:foo_sum not_a_float",
            "vllm:foo_count",
        ]
        for raw in garbage_texts:
            with self.subTest(raw=repr(raw)):
                try:
                    parse_prometheus_text(raw)
                except Exception as exc:  # noqa: BLE001
                    self.fail(f"parse_prometheus_text raised unexpectedly on {raw!r}: {exc!r}")

    def test_truncated_text_degrades_gracefully(self):
        # A bucket line cut off mid-value; count line normal.
        text = "vllm:request_queue_time_seconds_bucket{le=\"0.3\"} 1.\nvllm:request_queue_time_seconds_count 5"
        try:
            out = parse_prometheus_text(text)
        except Exception as exc:  # noqa: BLE001
            self.fail(f"parse_prometheus_text raised unexpectedly: {exc!r}")
        # The malformed bucket line is simply skipped; count line still parses.
        self.assertEqual(out.get("vllm:request_queue_time_seconds", {}).get("count"), 5.0)


class TestDiffHistogram(unittest.TestCase):
    def test_simple_before_after_diff(self):
        before = {"buckets": {"0.3": 2.0, "+Inf": 5.0}}
        after = {"buckets": {"0.3": 4.0, "+Inf": 9.0}}
        self.assertEqual(diff_histogram(before, after), {"0.3": 2.0, "+Inf": 4.0})

    def test_missing_before_bucket_treated_as_zero(self):
        before = {"buckets": {"+Inf": 5.0}}
        after = {"buckets": {"0.3": 1.0, "+Inf": 6.0}}
        self.assertEqual(diff_histogram(before, after), {"0.3": 1.0, "+Inf": 1.0})

    def test_none_before_treated_as_all_zero(self):
        after = {"buckets": {"0.3": 1.0, "+Inf": 3.0}}
        self.assertEqual(diff_histogram(None, after), {"0.3": 1.0, "+Inf": 3.0})

    def test_negative_diff_clamped_to_zero(self):
        # Simulates a scrape taken across a server restart: after < before.
        before = {"buckets": {"0.3": 10.0, "+Inf": 20.0}}
        after = {"buckets": {"0.3": 1.0, "+Inf": 2.0}}
        self.assertEqual(diff_histogram(before, after), {"0.3": 0.0, "+Inf": 0.0})

    def test_none_after_returns_none(self):
        before = {"buckets": {"0.3": 1.0, "+Inf": 1.0}}
        self.assertIsNone(diff_histogram(before, None))

    def test_empty_after_buckets_returns_none(self):
        self.assertIsNone(diff_histogram({"buckets": {}}, {"buckets": {}}))


class TestHistogramQuantile(unittest.TestCase):
    def test_zero_count_returns_none(self):
        self.assertIsNone(histogram_quantile({"0.3": 0.0, "+Inf": 0.0}, 0.5))

    def test_empty_or_none_returns_none(self):
        self.assertIsNone(histogram_quantile({}, 0.5))
        self.assertIsNone(histogram_quantile(None, 0.5))

    def test_all_mass_in_one_bucket(self):
        # Every observation lands at or below 0.3s (the first bucket).
        # Linear interpolation assumes uniform distribution between the
        # implicit lower bound (0) and this bucket's boundary (0.3):
        # target rank = 0.5*10 = 5; frac = (5-0)/(10-0) = 0.5;
        # interpolated = 0 + 0.5*(0.3-0) = 0.15.
        buckets = {"0.3": 10.0, "0.5": 10.0, "+Inf": 10.0}
        self.assertAlmostEqual(histogram_quantile(buckets, 0.5), 0.15)

    def test_hand_computed_interpolation_p50(self):
        # 0 <= x <= 0.3: 2 obs (cumulative 2); 0.3 < x <= 0.5: 8 obs (cumulative
        # 10); target rank for p50 of 10 total = 5. Falls in the (0.3, 0.5]
        # bucket: prev_bound=0.3 prev_count=2, bound=0.5 count=10.
        # frac = (5-2)/(10-2) = 0.375; interpolated = 0.3 + 0.375*(0.5-0.3) = 0.375
        buckets = {"0.3": 2.0, "0.5": 10.0, "+Inf": 10.0}
        self.assertAlmostEqual(histogram_quantile(buckets, 0.5), 0.375)

    def test_hand_computed_interpolation_p95(self):
        # 20 total obs: cumulative 0.3->5, 0.5->18, 1.0->20. p95 target rank=19.
        # Falls in (0.5, 1.0]: prev_bound=0.5 prev_count=18, bound=1.0 count=20.
        # frac = (19-18)/(20-18) = 0.5; interpolated = 0.5 + 0.5*(1.0-0.5) = 0.75
        buckets = {"0.3": 5.0, "0.5": 18.0, "1.0": 20.0, "+Inf": 20.0}
        self.assertAlmostEqual(histogram_quantile(buckets, 0.95), 0.75)

    def test_le_inf_only_bucket(self):
        # Degenerate case: only the +Inf bucket present.
        buckets = {"+Inf": 4.0}
        self.assertEqual(histogram_quantile(buckets, 0.5), float("inf"))

    def test_target_in_inf_bucket_clamps_to_highest_finite_bound(self):
        # 10 total obs, all but 1 land at or below 1.0s; the last only shows
        # up in "+Inf" (an overloaded request exceeding every finite bucket).
        # p95 target rank = 9.5, which only the "+Inf" bucket satisfies.
        # PromQL cannot interpolate past the last finite boundary, so it
        # clamps to it (1.0) instead of returning +Inf.
        buckets = {"0.3": 5.0, "0.5": 8.0, "1.0": 9.0, "+Inf": 10.0}
        self.assertEqual(histogram_quantile(buckets, 0.95), 1.0)

    def test_never_raises_on_malformed_le_values(self):
        try:
            out = histogram_quantile({"not_a_number": 1.0, "+Inf": 2.0}, 0.5)
        except Exception as exc:  # noqa: BLE001
            self.fail(f"histogram_quantile raised unexpectedly: {exc!r}")
        else:
            self.assertIsNone(out)


class TestToPromMetrics(unittest.TestCase):
    def test_all_prom_metrics_keys_present_shape(self):
        expected_keys = {f"prom.{short}" for short, _unit in PROM_METRICS}
        out = to_prom_metrics(None, None)
        self.assertEqual(set(out.keys()), expected_keys)

    def test_none_before_or_after_yields_all_none(self):
        after_text = _full_scrape_text(_cumulative([0.1]), 0.1, _cumulative([0.2]), 0.2)
        for before, after in ((None, after_text), (after_text, None), (None, None)):
            with self.subTest(before=before, after=after):
                out = to_prom_metrics(before, after)
                for k in out:
                    self.assertIsNone(out[k])

    def test_unparseable_text_yields_all_none_not_raise(self):
        try:
            out = to_prom_metrics("garbage before", "garbage after")
        except Exception as exc:  # noqa: BLE001
            self.fail(f"to_prom_metrics raised unexpectedly: {exc!r}")
        for k in out:
            self.assertIsNone(out[k])

    def test_end_to_end_realistic_before_after_pair(self):
        # "before" scrape: server already served 3 queue-wait observations
        # from a prior cell (0.1, 0.2, 0.4s) -- the reused-server baseline.
        before_text = _full_scrape_text(
            _cumulative([0.1, 0.2, 0.4]), 0.7, _cumulative([0.2, 0.3]), 0.5
        )
        # "after" scrape: this cell added 2 more queue-wait obs (0.6, 0.9s)
        # and 1 more prefill obs (1.2s) on top of the same server's counters.
        after_text = _full_scrape_text(
            _cumulative([0.1, 0.2, 0.4, 0.6, 0.9]),
            2.2,
            _cumulative([0.2, 0.3, 1.2]),
            1.7,
        )
        out = to_prom_metrics(before_text, after_text)
        # This cell's isolated queue-wait observations are exactly [0.6, 0.9]
        # (0.6 falls in bucket 0.8, 0.9 falls in bucket 1.0); p50 of 2 obs
        # falls in/around the first of the two remaining buckets.
        self.assertIsNotNone(out["prom.queue_time_p50_ms"])
        self.assertIsNotNone(out["prom.queue_time_p95_ms"])
        self.assertIsNotNone(out["prom.prefill_time_p50_ms"])
        self.assertIsNotNone(out["prom.prefill_time_p95_ms"])
        # Values are in ms (seconds * 1000), and in the right ballpark given
        # only [0.6, 0.9] contributed post-diff (600-1000ms range).
        self.assertGreater(out["prom.queue_time_p50_ms"], 500)
        self.assertLess(out["prom.queue_time_p50_ms"], 1100)

    def test_prom_metric_units_cover_every_metric(self):
        for short, unit in PROM_METRICS:
            with self.subTest(short=short):
                self.assertEqual(PROM_METRIC_UNITS[short], unit)


class TestScrapeVllmMetrics(unittest.TestCase):
    """I/O-boundary test for scrape_vllm_metrics (lives in vllm_job.py).

    Mirrors TestCaptureGpuMetrics's assert_called_once_with style: mock orch,
    pin the exact command string, verify degrade-on-failure never raises.
    """

    def test_happy_path_returns_raw_text(self):
        orch = MagicMock()
        orch.exec_on_head.return_value = {"node0": "vllm:num_requests_waiting 0\n"}
        out = scrape_vllm_metrics(orch, "http://0.0.0.0", "8888")
        self.assertEqual(out, "vllm:num_requests_waiting 0\n")
        orch.exec_on_head.assert_called_once_with("curl -sf http://0.0.0.0:8888/metrics")

    def test_timeout_kwarg_passed_through_when_given(self):
        orch = MagicMock()
        orch.exec_on_head.return_value = {"node0": "vllm:num_requests_waiting 0\n"}
        scrape_vllm_metrics(orch, "http://0.0.0.0", "8888", timeout_s=30)
        orch.exec_on_head.assert_called_once_with("curl -sf http://0.0.0.0:8888/metrics", timeout=30)

    def test_curl_failure_exception_degrades_to_none(self):
        orch = MagicMock()
        orch.exec_on_head.side_effect = RuntimeError("connection refused")
        try:
            out = scrape_vllm_metrics(orch, "http://0.0.0.0", "8888")
        except Exception as exc:  # noqa: BLE001
            self.fail(f"scrape_vllm_metrics raised unexpectedly: {exc!r}")
        self.assertIsNone(out)

    def test_empty_output_degrades_to_none(self):
        orch = MagicMock()
        orch.exec_on_head.return_value = {"node0": ""}
        self.assertIsNone(scrape_vllm_metrics(orch, "http://0.0.0.0", "8888"))

    def test_no_hosts_in_output_degrades_to_none(self):
        orch = MagicMock()
        orch.exec_on_head.return_value = {}
        self.assertIsNone(scrape_vllm_metrics(orch, "http://0.0.0.0", "8888"))


if __name__ == "__main__":
    unittest.main()
