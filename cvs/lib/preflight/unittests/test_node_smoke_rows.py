"""Unit tests for Node Smoke pytest-html metric rows."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from cvs.lib.preflight.node_smoke_counts import (
    TIER1_NODE_OPERATIONAL_COLLECTORS,
    TIER2_CHECKS_PER_GPU,
    TIER2_RCCL_CHECK,
    TIER3_TOP_LEVEL_GROUPS,
)
from cvs.lib.preflight.node_smoke_rows import (
    build_tier1_metric_rows,
    build_tier2_metric_rows,
    build_tier3_metric_rows,
    tier1_check_catalog,
    tier2_check_catalog,
    tier3_check_catalog_entries,
    tier_runner_row_hidden,
)


def _sample_payload(n_gpus=8, tier2=True, fail_reasons=None):
    per_gpu = [{"gpu": idx, "status": "PASS"} for idx in range(n_gpus)]
    tier1 = {"per_gpu": per_gpu}
    for key in TIER1_NODE_OPERATIONAL_COLLECTORS:
        tier1[key] = {"ok": True}
    payload = {"status": "FAIL" if fail_reasons else "PASS", "tier1": tier1}
    if fail_reasons:
        payload["fail_reasons"] = list(fail_reasons)
    if tier2:
        payload["tier2"] = {
            "per_gpu": [{"gpu": idx, "status": "PASS", "gemm_tflops": 800, "hbm_gbs": 3000} for idx in range(n_gpus)],
            "rccl": {"status": "PASS", "gbs": 120},
        }
    return payload


class TestNodeSmokeRows(unittest.TestCase):
    def test_tier1_catalog_per_node(self):
        payload = _sample_payload()
        results = {
            "gpus_per_node": 8,
            "node_results": {
                "node0": {"status": "PASS", "node_payload": payload},
                "node1": {"status": "PASS", "node_payload": payload},
            },
        }
        rows = build_tier1_metric_rows(results)
        expected = (8 + len(TIER1_NODE_OPERATIONAL_COLLECTORS)) * 2
        self.assertEqual(len(rows), expected)
        self.assertTrue(all(row["status"] == "pass" for row in rows))
        self.assertFalse(any("subprocess" in row["metric"] for row in rows))

    def test_tier1_fail_reason_marks_collector(self):
        payload = _sample_payload(fail_reasons=["gpu_processes: pid=99"])
        results = {
            "gpus_per_node": 8,
            "node_results": {
                "node0": {
                    "status": "FAIL",
                    "fail_reasons": ["gpu_processes: pid=99"],
                    "node_payload": payload,
                }
            },
        }
        rows = build_tier1_metric_rows(results)
        failed = [row for row in rows if row["status"] == "fail"]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["metric"], "node0/gpu_processes")
        self.assertIn("pid=99", failed[0]["reason"])

    def test_tier2_catalog_and_thresholds(self):
        payload = _sample_payload()
        results = {
            "tier2_perf": True,
            "gpus_per_node": 8,
            "tier2_thresholds": {"gemm_tflops_min": 600, "hbm_gbs_min": 2000, "rccl_gbs_min": 100},
            "node_results": {"node0": {"status": "PASS", "node_payload": payload}},
        }
        rows = build_tier2_metric_rows(results)
        self.assertEqual(len(rows), (TIER2_CHECKS_PER_GPU * 8) + TIER2_RCCL_CHECK)
        gemm = next(row for row in rows if row["metric"].endswith("/large_gemm"))
        self.assertEqual(gemm["actual"], 800)
        self.assertEqual(gemm["unit"], "TFLOPS")

    def test_tier1_node_pass_does_not_pass_unreported_checks(self):
        results = {
            "gpus_per_node": 8,
            "node_results": {"node0": {"status": "PASS", "node_payload": {"tier1": {}}}},
        }
        rows = build_tier1_metric_rows(results)
        self.assertTrue(rows)
        self.assertTrue(all(row["status"] == "skip" for row in rows))

    def test_tier1_unattributed_node_failure_does_not_invent_check_failures(self):
        reason = "could not determine node_smoke status from output"
        results = {
            "gpus_per_node": 8,
            "node_results": {"node0": {"status": "FAIL", "fail_reasons": [reason], "node_payload": {}}},
        }
        rows = build_tier1_metric_rows(results)
        self.assertTrue(rows)
        self.assertTrue(all(row["status"] == "skip" for row in rows))

    def test_tier2_skipped_without_flag(self):
        results = {"tier2_perf": False, "node_results": {"node0": {"status": "PASS", "node_payload": {}}}}
        self.assertEqual(build_tier2_metric_rows(results), [])

    def test_tier3_pass_requires_reported_groups(self):
        bare = {
            "skipped": False,
            "node_results": {"node0": {"status": "PASS"}, "node1": {"status": "PASS"}},
        }
        bare_rows = build_tier3_metric_rows(bare)
        self.assertEqual(len(bare_rows), len(TIER3_TOP_LEVEL_GROUPS))
        self.assertTrue(all(row["status"] == "skip" for row in bare_rows))

        reported = {
            "skipped": False,
            "node_results": {
                "node0": {"status": "PASS", "checks": ["host,gpu,network"]},
                "node1": {"status": "PASS", "checks": ["host,gpu,network"]},
            },
        }
        rows = build_tier3_metric_rows(reported)
        self.assertEqual(len(rows), len(TIER3_TOP_LEVEL_GROUPS))
        self.assertTrue(all(row["status"] == "pass" for row in rows))

    def test_tier3_fail_reason_marks_matching_group(self):
        results = {
            "skipped": False,
            "failed_nodes": ["node0"],
            "node_results": {
                "node0": {"status": "FAIL", "fail_reasons": ["CPU mismatch"], "checks": ["host,gpu,network"]},
            },
        }
        rows = build_tier3_metric_rows(results)
        failed = [row for row in rows if row["status"] == "fail"]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["label"], "Host")
        self.assertIn("CPU", failed[0]["reason"])
        self.assertTrue(all(row["status"] != "pass" for row in rows))

    def test_tier3_unattributed_failure_does_not_pass_or_fail_every_group(self):
        reason = "could not determine Tier 3 preflight status from output"
        results = {
            "skipped": False,
            "failed_nodes": ["node0"],
            "node_results": {"node0": {"status": "FAIL", "fail_reasons": [reason]}},
        }
        rows = build_tier3_metric_rows(results)
        self.assertEqual(len(rows), len(TIER3_TOP_LEVEL_GROUPS))
        self.assertTrue(all(row["status"] == "skip" for row in rows))


class TestNodeSmokeCheckCatalogs(unittest.TestCase):
    """The catalogs drive pytest parametrization, so they must match the result rows."""

    def test_tier1_catalog_size_and_unique_ids(self):
        catalog = tier1_check_catalog(["node1", "node0"], 8)
        self.assertEqual(len(catalog), (8 + len(TIER1_NODE_OPERATIONAL_COLLECTORS)) * 2)
        self.assertEqual(len({entry["id"] for entry in catalog}), len(catalog))
        self.assertEqual([entry["node"] for entry in catalog][0], "node0")

    def test_tier2_catalog_drops_rccl_on_single_gpu(self):
        multi = tier2_check_catalog(["node0"], 8)
        self.assertEqual(len(multi), TIER2_CHECKS_PER_GPU * 8 + TIER2_RCCL_CHECK)
        single = tier2_check_catalog(["node0"], 1)
        self.assertEqual(len(single), TIER2_CHECKS_PER_GPU)

    def test_tier3_catalog_size_and_unique_ids(self):
        catalog = tier3_check_catalog_entries()
        self.assertEqual(len(catalog), len(TIER3_TOP_LEVEL_GROUPS))
        self.assertEqual(len({entry["id"] for entry in catalog}), len(catalog))

    def test_empty_host_list_yields_empty_catalog(self):
        self.assertEqual(tier1_check_catalog([], 8), [])
        self.assertEqual(tier2_check_catalog(None, 8), [])

    def test_catalog_metrics_match_tier1_rows(self):
        payload = _sample_payload()
        results = {"gpus_per_node": 8, "node_results": {"node0": {"status": "PASS", "node_payload": payload}}}
        row_metrics = {row["metric"] for row in build_tier1_metric_rows(results)}
        catalog_metrics = {entry["metric"] for entry in tier1_check_catalog(["node0"], 8)}
        self.assertEqual(catalog_metrics, row_metrics)

    def test_catalog_metrics_match_tier2_rows(self):
        payload = _sample_payload()
        results = {
            "tier2_perf": True,
            "gpus_per_node": 8,
            "node_results": {"node0": {"status": "PASS", "node_payload": payload}},
        }
        row_metrics = {row["metric"] for row in build_tier2_metric_rows(results)}
        catalog_metrics = {entry["metric"] for entry in tier2_check_catalog(["node0"], 8)}
        self.assertEqual(catalog_metrics, row_metrics)

    def test_catalog_metrics_match_tier3_rows(self):
        results = {"skipped": False, "node_results": {"node0": {"status": "PASS"}}}
        row_metrics = {row["metric"] for row in build_tier3_metric_rows(results)}
        catalog_metrics = {entry["metric"] for entry in tier3_check_catalog_entries()}
        self.assertEqual(catalog_metrics, row_metrics)


class TestTierRunnerRowVisibility(unittest.TestCase):
    def test_failed_runner_stays_when_no_check_failed(self):
        self.assertFalse(tier_runner_row_hidden("failed", True, False, False))
        self.assertFalse(tier_runner_row_hidden("failed", True, False, True))

    def test_failed_runner_hides_when_a_check_failed(self):
        self.assertTrue(tier_runner_row_hidden("failed", True, True, True))

    def test_passed_runner_stays_when_no_check_passed(self):
        self.assertFalse(tier_runner_row_hidden("passed", True, False, False))

    def test_passed_runner_hides_when_a_check_passed(self):
        self.assertTrue(tier_runner_row_hidden("passed", True, False, True))

    def test_runner_stays_when_check_rows_were_not_collected(self):
        self.assertFalse(tier_runner_row_hidden("failed", False, False, False))
        self.assertFalse(tier_runner_row_hidden("passed", False, False, False))

    def test_drop_keeps_a_failed_runner_until_a_check_fails(self):
        from cvs.tests.preflight.conftest import drop_hidden_node_smoke_runners

        def _row(outcome):
            return {"resultsTableRow": [f'<td class="col-result">{outcome}</td>']}

        kept = {
            "preflight_checks.py::test_node_smoke_tier1": [_row("Failed")],
        }
        counts = {"failed": 1}
        drop_hidden_node_smoke_runners(
            kept,
            counts,
            {"test_node_smoke_tier1": "failed"},
            {"test_node_smoke_tier1_check": {"failed": False, "passed": False}},
            {"test_node_smoke_tier1_check"},
        )
        self.assertIn("preflight_checks.py::test_node_smoke_tier1", kept)
        self.assertEqual(counts["failed"], 1)

        hidden = {
            "preflight_checks.py::test_node_smoke_tier1": [_row("Failed")],
            "preflight_checks.py::test_node_smoke_tier1_check[gpu0]": [_row("Failed")],
        }
        counts = {"failed": 2}
        drop_hidden_node_smoke_runners(
            hidden,
            counts,
            {"test_node_smoke_tier1": "failed"},
            {"test_node_smoke_tier1_check": {"failed": True, "passed": False}},
            {"test_node_smoke_tier1_check"},
        )
        self.assertNotIn("preflight_checks.py::test_node_smoke_tier1", hidden)
        self.assertIn("preflight_checks.py::test_node_smoke_tier1_check[gpu0]", hidden)
        self.assertEqual(counts["failed"], 1)


if __name__ == "__main__":
    unittest.main()
