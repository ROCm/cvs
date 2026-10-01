"""Unit tests for Node Smoke tier test counting."""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from cvs.lib.preflight.node_smoke_counts import (
    TIER1_NODE_OPERATIONAL_COLLECTORS,
    TIER2_CHECKS_PER_GPU,
    TIER2_RCCL_CHECK,
    TIER3_CATALOG_COUNT,
    TIER3_CHECK_CATALOG,
    aggregate_node_smoke_test_counts,
    aggregate_tier3_test_counts,
    count_tier1_tests_from_payload,
    count_tier2_tests_from_payload,
    count_tier3_tests_from_results,
    format_tests_run_suffix,
    tier3_check_catalog,
)


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
_FIXTURE_JSON = os.path.join(_REPO_ROOT, '..', '..', 'preflight', 'node_smoke', 'smoke', 'tus1-p3-g57.json')


def _sample_tier1_payload(n_gpus=8, tier2=False):
    tier1 = {"per_gpu": [{"gpu": idx, "status": "PASS"} for idx in range(n_gpus)]}
    for key in TIER1_NODE_OPERATIONAL_COLLECTORS:
        tier1[key] = {"ok": True}
    tier1["gpu_info"] = [{"level": "info", "message": "GPU enumeration"}]
    tier1["host_info"] = [{"level": "info", "message": "Host identity"}]
    tier1["network_info"] = [{"level": "info", "message": "Network summary"}]
    tier1["fingerprint"] = {"kernel": "6.8.0"}
    tier1["clock"] = {"ok": True}
    payload = {"tier1": tier1}
    if tier2:
        payload["tier2"] = {
            "per_gpu": [{"gpu": idx, "gemm_tflops": 800, "hbm_gbs": 3000} for idx in range(n_gpus)],
            "rccl": {"status": "PASS", "gbs": 120},
        }
    return payload


class TestNodeSmokeCounts(unittest.TestCase):
    def test_format_tests_run_suffix(self):
        self.assertEqual(format_tests_run_suffix(0), "")
        self.assertEqual(format_tests_run_suffix(39, per_node=True, total_nodes=2), "; 39 tests run per node")
        self.assertEqual(format_tests_run_suffix(17, per_node=True, total_nodes=2), "; 17 tests run per node")
        self.assertEqual(
            format_tests_run_suffix(27, cluster_wide=True, total_nodes=2),
            "; 27 tests run cluster-wide",
        )

    def test_count_tier1_counts_reported_gpus_and_collectors_only(self):
        payload = _sample_tier1_payload(n_gpus=8)
        expected = 8 + len(TIER1_NODE_OPERATIONAL_COLLECTORS)
        self.assertEqual(count_tier1_tests_from_payload(payload), expected)
        self.assertEqual(count_tier1_tests_from_payload(payload, gpus_per_node=8), expected)
        self.assertEqual(count_tier1_tests_from_payload(None, gpus_per_node=8), 0)

    def test_count_tier1_ignores_inventory_and_drift_collectors(self):
        payload = _sample_tier1_payload(n_gpus=8)
        self.assertNotIn("gpu_info", TIER1_NODE_OPERATIONAL_COLLECTORS)
        self.assertEqual(count_tier1_tests_from_payload(payload), 8 + len(TIER1_NODE_OPERATIONAL_COLLECTORS))
        if not os.path.isfile(_FIXTURE_JSON):
            return
        with open(_FIXTURE_JSON, encoding="utf-8") as handle:
            fixture = json.load(handle)
        tier1 = fixture.get("tier1") or {}
        measured = sum(1 for entry in tier1.get("per_gpu") or [] if isinstance(entry, dict) and entry.get("status"))
        measured += sum(1 for key in TIER1_NODE_OPERATIONAL_COLLECTORS if key in tier1)
        self.assertEqual(count_tier1_tests_from_payload(fixture), measured)

    def test_count_tier2_counts_reported_metrics_only(self):
        self.assertEqual(count_tier2_tests_from_payload(None, gpus_per_node=8, tier2_enabled=True), 0)
        self.assertEqual(
            count_tier2_tests_from_payload(_sample_tier1_payload(), gpus_per_node=8, tier2_enabled=True),
            0,
        )
        measured = _sample_tier1_payload(tier2=True)
        self.assertEqual(
            count_tier2_tests_from_payload(measured, gpus_per_node=8, tier2_enabled=True),
            (TIER2_CHECKS_PER_GPU * 8) + TIER2_RCCL_CHECK,
        )
        single = _sample_tier1_payload(n_gpus=1, tier2=True)
        self.assertEqual(count_tier2_tests_from_payload(single, tier2_enabled=True), TIER2_CHECKS_PER_GPU)

    def test_aggregate_tier1_tier2_reports_per_node_not_cluster_sum(self):
        payload = _sample_tier1_payload(n_gpus=8, tier2=True)
        results = {
            "tier2_perf": True,
            "node_results": {
                "node0": {"node_payload": payload},
                "node1": {"node_payload": payload},
            },
        }
        counts = aggregate_node_smoke_test_counts(results, gpus_per_node=8)
        self.assertEqual(counts["tier1_tests_run"], 8 + len(TIER1_NODE_OPERATIONAL_COLLECTORS))
        self.assertEqual(counts["tier1_tests_run_total"], 2 * (8 + len(TIER1_NODE_OPERATIONAL_COLLECTORS)))
        self.assertEqual(counts["tier2_tests_run"], (TIER2_CHECKS_PER_GPU * 8) + TIER2_RCCL_CHECK)
        self.assertEqual(counts["tier2_tests_run_total"], 2 * ((TIER2_CHECKS_PER_GPU * 8) + TIER2_RCCL_CHECK))

    def test_tier3_finding_map_is_not_a_run_count(self):
        self.assertEqual(TIER3_CATALOG_COUNT, 27)
        self.assertEqual(len(TIER3_CHECK_CATALOG), 27)
        self.assertEqual(len(tier3_check_catalog()), 27)
        groups = [entry["group"] for entry in tier3_check_catalog()]
        self.assertEqual(groups.count("host"), 10)
        self.assertEqual(groups.count("gpu"), 8)
        self.assertEqual(groups.count("network"), 9)

    def test_count_tier3_uses_reported_groups_not_the_finding_map(self):
        self.assertEqual(
            count_tier3_tests_from_results(
                {
                    "skipped": False,
                    "report_markdown": "# Host Info\n\n## Host System\n",
                    "node_results": {"node0": {"status": "PASS"}, "node1": {"status": "PASS"}},
                }
            ),
            0,
        )
        self.assertEqual(
            count_tier3_tests_from_results(
                {
                    "skipped": False,
                    "node_results": {"node0": {"status": "PASS", "checks": ["host,gpu,network"]}},
                }
            ),
            3,
        )

    def test_aggregate_tier3_is_cluster_wide(self):
        counts = aggregate_tier3_test_counts(
            {
                "skipped": False,
                "node_results": {
                    "node0": {"status": "PASS", "checks": ["host,gpu,network"]},
                    "node1": {"status": "PASS", "checks": ["host,gpu,network"]},
                },
            }
        )
        self.assertEqual(counts["tier3_tests_run"], 3)
        self.assertEqual(counts["tier3_tests_run_total"], 3)
        self.assertEqual([entry["group"] for entry in counts["tier3_check_catalog"]], ["host", "gpu", "network"])

    def test_count_tier3_fallback_without_report(self):
        self.assertEqual(
            count_tier3_tests_from_results({"skipped": False, "checks": ["host,gpu,network"]}),
            3,
        )
        self.assertEqual(count_tier3_tests_from_results({"skipped": True}), 0)


if __name__ == "__main__":
    unittest.main()
