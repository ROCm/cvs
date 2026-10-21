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
    aggregate_node_smoke_test_counts,
)
from cvs.lib.preflight.node_smoke_rows import (
    build_tier1_metric_rows,
    build_tier2_metric_rows,
    build_tier3_metric_rows,
    tier1_check_catalog,
    tier1_runner_outcome,
    tier2_check_catalog,
    tier2_runner_outcome,
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

    def test_collected_tier1_sections_pass_without_an_ok_field(self):
        payload = {
            "status": "PASS",
            "tier1": {
                "per_gpu": [{"gpu": 0, "status": "PASS"}],
                "nics": {"expected_count": 8, "issues": [], "included_ports": ["rdma0"]},
                "host_limits": {"fail_reasons": [], "memlock_soft_bytes": 1024},
                "gpu_visibility": {"fail_reasons": [], "torch_visible": 8},
                "gpu_processes": {"error": "skipped"},
            },
        }
        results = {"gpus_per_node": 1, "node_results": {"node0": {"status": "PASS", "node_payload": payload}}}
        rows = {row["metric"]: row for row in build_tier1_metric_rows(results)}
        self.assertEqual(rows["node0/nics"]["status"], "pass")
        self.assertEqual(rows["node0/host_limits"]["status"], "pass")
        self.assertEqual(rows["node0/gpu_visibility"]["status"], "pass")
        self.assertEqual(rows["node0/gpu_processes"]["status"], "skip")
        self.assertEqual(rows["node0/xgmi"]["status"], "skip")

    def test_tier1_collector_error_fails(self):
        payload = {"tier1": {"nics": {"error": "collector failed", "expected_count": 8}}}
        results = {"gpus_per_node": 1, "node_results": {"node0": {"status": "FAIL", "node_payload": payload}}}
        rows = {row["metric"]: row for row in build_tier1_metric_rows(results)}
        self.assertEqual(rows["node0/nics"]["status"], "fail")
        self.assertIn("collector failed", rows["node0/nics"]["reason"])

    def test_tier1_collector_issues_fail(self):
        payload = {"tier1": {"nics": {"issues": ["rdma0 down"]}}}
        results = {"gpus_per_node": 1, "node_results": {"node0": {"status": "FAIL", "node_payload": payload}}}
        rows = {row["metric"]: row for row in build_tier1_metric_rows(results)}
        self.assertEqual(rows["node0/nics"]["status"], "fail")
        self.assertIn("rdma0 down", rows["node0/nics"]["reason"])

    def test_tier2_gemm_and_hbm_pass_from_gpu_details(self):
        payload = {
            "tier1": {
                "per_gpu": [
                    {"gpu": 0, "status": "PASS", "details": {"gemm_tflops": 761.35, "hbm_gbs": 4531.2}},
                    {"gpu": 1, "status": "PASS", "details": {}},
                ]
            },
            "tier2": {"rccl": {"status": "PASS", "gbs": 277.6}},
        }
        results = {
            "tier2_perf": True,
            "gpus_per_node": 2,
            "node_results": {"node0": {"status": "PASS", "node_payload": payload}},
        }
        rows = {row["metric"]: row for row in build_tier2_metric_rows(results)}
        self.assertEqual(rows["node0/gpu0/large_gemm"]["status"], "pass")
        self.assertEqual(rows["node0/gpu0/large_gemm"]["actual"], 761.35)
        self.assertEqual(rows["node0/gpu0/hbm_d2d"]["status"], "pass")
        self.assertEqual(rows["node0/gpu0/hbm_d2d"]["actual"], 4531.2)
        self.assertEqual(rows["node0/gpu1/large_gemm"]["status"], "skip")
        self.assertEqual(rows["node0/gpu1/hbm_d2d"]["status"], "skip")
        self.assertEqual(rows["node0/local_rccl"]["status"], "pass")

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

    def test_tier2_details_stay_hidden_when_tier2_is_disabled(self):
        results = {
            "tier2_perf": False,
            "gpus_per_node": 2,
            "node_results": {
                "node0": {
                    "status": "PASS",
                    "node_payload": {
                        "tier1": {
                            "per_gpu": [
                                {"gpu": 0, "status": "PASS", "details": {"gemm_tflops": 761.35, "hbm_gbs": 4531.2}},
                                {"gpu": 1, "status": "PASS", "details": {"gemm_tflops": 750.0, "hbm_gbs": 4400.0}},
                            ]
                        },
                        "tier2": {"rccl": {"status": "PASS", "gbs": 277.6}},
                    },
                }
            },
        }
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


class TestTierRunnerOutcomes(unittest.TestCase):
    def test_tier2_runner_skips_when_primus_reports_no_metrics(self):
        results = {
            "tier2_perf": True,
            "gpus_per_node": 8,
            "node_results": {
                "n1": {
                    "status": "PASS",
                    "node_payload": {"tier2": {"per_gpu": [{"gpu": idx} for idx in range(8)]}},
                }
            },
        }
        rows = build_tier2_metric_rows(results)
        self.assertEqual(len(rows), 17)
        self.assertTrue(all(row["status"] == "skip" for row in rows))
        self.assertEqual(tier2_runner_outcome(rows), "skip")

    def test_tier1_runner_does_not_inherit_a_tier2_miss(self):
        collectors = {key: {"ok": True} for key in TIER1_NODE_OPERATIONAL_COLLECTORS}
        results = {
            "tier2_perf": True,
            "gpus_per_node": 1,
            "total_nodes": 1,
            "failed_nodes": ["n1"],
            "unknown_nodes": [],
            "node_results": {
                "n1": {
                    "status": "FAIL",
                    "fail_reasons": ["large_gemm: 100 < 600"],
                    "node_payload": {
                        "tier1": {"per_gpu": [{"gpu": 0, "status": "PASS"}], **collectors},
                        "tier2": {"per_gpu": [{"gpu": 0, "gemm_tflops": 100, "hbm_gbs": 3000}]},
                    },
                }
            },
        }
        tier1_rows = build_tier1_metric_rows(results)
        tier2_rows = build_tier2_metric_rows(results)
        self.assertEqual(tier2_runner_outcome(tier2_rows), "fail")
        self.assertEqual(tier1_runner_outcome(results, tier1_rows, tier2_rows), "pass")

    def test_tier1_runner_still_fails_an_unattributed_node(self):
        results = {
            "tier2_perf": True,
            "gpus_per_node": 1,
            "total_nodes": 1,
            "failed_nodes": ["n1"],
            "node_results": {
                "n1": {
                    "status": "FAIL",
                    "fail_reasons": ["could not determine node_smoke status from output"],
                    "node_payload": {},
                }
            },
        }
        tier1_rows = build_tier1_metric_rows(results)
        tier2_rows = build_tier2_metric_rows(results)
        self.assertEqual(tier2_runner_outcome(tier2_rows), "skip")
        self.assertEqual(tier1_runner_outcome(results, tier1_rows, tier2_rows), "fail")

    def test_tier1_runner_fails_when_tier2_miss_and_setup_reason(self):
        collectors = {key: {"ok": True} for key in TIER1_NODE_OPERATIONAL_COLLECTORS}
        setup = "could not determine node_smoke status from output"
        results = {
            "tier2_perf": True,
            "gpus_per_node": 1,
            "total_nodes": 1,
            "failed_nodes": ["n1"],
            "unknown_nodes": [],
            "node_results": {
                "n1": {
                    "status": "FAIL",
                    "fail_reasons": ["large_gemm: 100 < 600", setup],
                    "node_payload": {
                        "tier1": {"per_gpu": [{"gpu": 0, "status": "PASS"}], **collectors},
                        "tier2": {"per_gpu": [{"gpu": 0, "gemm_tflops": 100, "hbm_gbs": 3000}]},
                    },
                }
            },
        }
        tier1_rows = build_tier1_metric_rows(results)
        tier2_rows = build_tier2_metric_rows(results)
        self.assertEqual(tier2_runner_outcome(tier2_rows), "fail")
        self.assertEqual(tier1_runner_outcome(results, tier1_rows, tier2_rows), "fail")


def _primus_payload(n_gpus=8, gpu_overrides=None, fail_reasons=None, host_limit_reasons=None, nic_issues=None):
    """Payload shaped like Primus dev/preflight-direct-test node_smoke --tier2-perf JSON.

    GEMM/HBM numbers live in ``tier1.per_gpu[i].details``; ``tier2`` only has ``rccl``;
    ``nics`` / ``host_limits`` / ``gpu_visibility`` carry problem lists, not ``ok``.
    """
    per_gpu = []
    for idx in range(n_gpus):
        entry = {
            "gpu": idx,
            "status": "PASS",
            "reason": "",
            "duration_sec": 1.5,
            "details": {"hbm_pre_touch_used_gib": 0.6, "gemm_tflops": 767.18, "hbm_gbs": 4476.1},
        }
        entry.update((gpu_overrides or {}).get(idx, {}))
        per_gpu.append(entry)
    tier1 = {
        "per_gpu": per_gpu,
        "gpu_processes": {"ok": True, "foreign_count": 0},
        "nics": {"expected_count": 8, "ports": [], "issues": list(nic_issues or [])},
        "host_limits": {"memlock_soft_bytes": -1, "fail_reasons": list(host_limit_reasons or [])},
        "gpu_low_level": {"ok": True, "per_gpu": []},
        "xgmi": {"ok": True, "non_xgmi_pairs": []},
        "tooling": {"ok": True, "latency_sec": 0.06},
        "gpu_visibility": {"expected_gpus": n_gpus, "torch_visible": n_gpus, "fail_reasons": []},
    }
    reasons = list(fail_reasons or [])
    return {
        "status": "FAIL" if reasons else "PASS",
        "fail_reasons": reasons,
        "tier1": tier1,
        "tier2": {"rccl": {"status": "PASS", "gbs": 278.1}},
    }


def _primus_results(payload, status="PASS"):
    return {
        "tier2_perf": True,
        "gpus_per_node": 8,
        "total_nodes": 1,
        "failed_nodes": [] if status == "PASS" else ["node0"],
        "tier2_thresholds": {"gemm_tflops_min": 600, "hbm_gbs_min": 2000, "rccl_gbs_min": 100},
        "node_results": {"node0": {"status": status, "fail_reasons": payload["fail_reasons"], "node_payload": payload}},
    }


class TestNodeSmokeRowsPrimusSchema(unittest.TestCase):
    """Rows built from the payload Primus actually writes, not the CVS-side ideal shape."""

    def test_every_tier1_and_tier2_row_is_measured(self):
        results = _primus_results(_primus_payload())
        tier1 = build_tier1_metric_rows(results)
        tier2 = build_tier2_metric_rows(results)
        self.assertEqual([r["metric"] for r in tier1 if r["status"] != "pass"], [])
        self.assertEqual([r["metric"] for r in tier2 if r["status"] != "pass"], [])
        self.assertEqual(len(tier2), TIER2_CHECKS_PER_GPU * 8 + TIER2_RCCL_CHECK)
        gemm = next(r for r in tier2 if r["metric"] == "node0/gpu3/large_gemm")
        hbm = next(r for r in tier2 if r["metric"] == "node0/gpu3/hbm_d2d")
        self.assertEqual((gemm["actual"], gemm["unit"]), (767.18, "TFLOPS"))
        self.assertEqual((hbm["actual"], hbm["unit"]), (4476.1, "GB/s"))
        counts = aggregate_node_smoke_test_counts(results)
        self.assertEqual(counts["tier1_tests_run"], 8 + len(TIER1_NODE_OPERATIONAL_COLLECTORS))
        self.assertEqual(counts["tier2_tests_run"], TIER2_CHECKS_PER_GPU * 8 + TIER2_RCCL_CHECK)

    def test_gemm_miss_fails_tier2_row_not_tier1(self):
        reason = "GEMM TFLOPS 500 < threshold 600"
        payload = _primus_payload(
            gpu_overrides={2: {"status": "FAIL", "reason": reason, "details": {"gemm_tflops": 500.0}}},
            fail_reasons=[f"gpu2: FAIL: {reason}"],
        )
        results = _primus_results(payload, status="FAIL")
        tier1 = build_tier1_metric_rows(results)
        tier2 = build_tier2_metric_rows(results)
        self.assertEqual(next(r for r in tier1 if r["metric"] == "node0/gpu2")["status"], "pass")
        gemm = next(r for r in tier2 if r["metric"] == "node0/gpu2/large_gemm")
        self.assertEqual((gemm["status"], gemm["actual"]), ("fail", 500.0))
        self.assertIn("500 < threshold 600", gemm["reason"])
        # Primus stops after the GEMM miss, so HBM on that GPU was never measured.
        self.assertEqual(next(r for r in tier2 if r["metric"] == "node0/gpu2/hbm_d2d")["status"], "skip")
        self.assertEqual(tier2_runner_outcome(tier2), "fail")
        self.assertEqual(tier1_runner_outcome(results, tier1, tier2), "pass")

    def test_tier1_gpu_failure_stays_tier1(self):
        reason = "pre-touch HBM busy: 43.0 GiB already in use"
        payload = _primus_payload(
            gpu_overrides={5: {"status": "FAIL", "reason": reason, "details": {}}},
            fail_reasons=[f"gpu5: FAIL: {reason}"],
        )
        results = _primus_results(payload, status="FAIL")
        tier1 = build_tier1_metric_rows(results)
        tier2 = build_tier2_metric_rows(results)
        self.assertEqual(next(r for r in tier1 if r["metric"] == "node0/gpu5")["status"], "fail")
        self.assertEqual(next(r for r in tier2 if r["metric"] == "node0/gpu5/large_gemm")["status"], "skip")
        self.assertEqual(tier1_runner_outcome(results, tier1, tier2), "fail")

    def test_collector_problem_lists_fail_their_row(self):
        payload = _primus_payload(
            host_limit_reasons=["ulimit -l = 8012 MiB < 32 GiB"],
            nic_issues=["rdma3 port 1 state DOWN"],
            fail_reasons=["host_limits: ulimit -l = 8012 MiB < 32 GiB", "nic: rdma3 port 1 state DOWN"],
        )
        rows = {r["metric"]: r for r in build_tier1_metric_rows(_primus_results(payload, status="FAIL"))}
        self.assertEqual(rows["node0/host_limits"]["status"], "fail")
        self.assertEqual(rows["node0/nics"]["status"], "fail")
        self.assertIn("rdma3", rows["node0/nics"]["reason"])
        self.assertEqual(rows["node0/gpu_visibility"]["status"], "pass")


if __name__ == "__main__":
    unittest.main()
