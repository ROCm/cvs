'''Tests for rerun order, verdict, recommendations, and the cluster manifest.'''

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from cvs.lib.report.consolidate.rollup import (
    ALL_PASSED,
    EVALUATION_INCOMPLETE,
    INCOMPLETE_DATA,
    NEEDS_ATTENTION,
    build_manifest,
    collection_verdict,
    deck_time,
    display_name,
    generated_range,
    host_key,
    order_runs,
    recommendations,
)
from cvs.lib.report.consolidate.unittests.deck_fixtures import deck, found
from cvs.lib.report.consolidate.warning_log import TIED_TIMESTAMP, WarningLog


def suite(suite_id, status="pass", **fields):
    '''A suite summary as the report builds it, with the private keys roll-up reads.'''
    data = {
        "id": suite_id,
        "name": fields.pop("name", suite_id),
        "status": status,
        "available": True,
        "partial": False,
        "counts": {"pass": 0, "fail": 0, "record": 0, "na": 0, "unknown": 0},
        "total": 0,
        "unit": "node/check groups",
        "nodes": [],
        "failedNodes": [],
        "cluster": None,
        "generated": "2026-11-10 19:32 UTC",
        "reportUrl": "runs/%s/r/deck.html" % suite_id,
        "_host_rows": True,
        "_provenance": {},
        "_source": "/runs/%s/deck.json" % suite_id,
    }
    data.update(fields)
    return data


class TestDeckTime(unittest.TestCase):
    def test_generated_at_utc_is_preferred_and_kept_in_utc(self):
        data = {"generated_at_utc": "2026-11-10T21:32:05+02:00", "generated_at": "2026-11-01 00:00 UTC"}
        self.assertEqual(deck_time(data), datetime(2026, 11, 10, 19, 32, 5, tzinfo=timezone.utc))
        self.assertEqual(
            deck_time({"generated_at_utc": "2026-11-10T19:32:05Z"}),
            datetime(2026, 11, 10, 19, 32, 5, tzinfo=timezone.utc),
        )

    def test_older_decks_use_generated_at(self):
        self.assertEqual(
            deck_time({"generated_at": "2026-11-10 19:32 UTC"}), datetime(2026, 11, 10, 19, 32, tzinfo=timezone.utc)
        )
        self.assertEqual(
            deck_time({"generated_at_utc": "not a time", "generated_at": "2026-11-10 19:32 UTC"}).minute, 32
        )

    def test_no_usable_timestamp(self):
        self.assertIsNone(deck_time({"generated_at": "yesterday"}))
        self.assertIsNone(deck_time({}))
        self.assertIsNone(deck_time(None))


class TestOrderRuns(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.warnings = WarningLog()

    def run_deck(self, run_id, **fields):
        source = self.root / run_id / "rvs_cvs_html" / "rvs_cvs_run_deck.json"
        return found(self.root, deck("rvs_cvs", **fields), "rvs_cvs_html/rvs_cvs_run_deck.json", source=source)

    def test_latest_first_with_seconds_beating_minutes(self):
        older = self.run_deck("a", generated_at="2026-11-09 08:00 UTC")
        minute = self.run_deck("b", generated_at="2026-11-10 19:32 UTC")
        seconds = self.run_deck("c", generated_at="2026-11-10 19:32 UTC", generated_at_utc="2026-11-10T19:32:05Z")
        self.assertEqual(order_runs("rvs_cvs", [older, minute, seconds], self.warnings), [seconds, minute, older])
        self.assertEqual(self.warnings.entries, [])

    def test_deck_without_a_timestamp_sorts_oldest(self):
        untimed = self.run_deck("z", generated_at="unknown")
        timed = self.run_deck("a", generated_at="2026-11-01 00:00 UTC")
        self.assertEqual(order_runs("rvs_cvs", [untimed, timed], self.warnings), [timed, untimed])

    def test_tie_is_broken_by_path_with_a_warning(self):
        first = self.run_deck("local-1")
        second = self.run_deck("local-2")
        with self.assertLogs(level="WARNING"):
            ordered = order_runs("rvs_cvs", [first, second], self.warnings)
        self.assertEqual(ordered, [second, first])
        self.assertEqual([entry["code"] for entry in self.warnings.entries], [TIED_TIMESTAMP])


class TestNames(unittest.TestCase):
    def test_display_name(self):
        self.assertEqual(display_name({"suite_id": "rvs_cvs", "report": {"title": "RVS Run Deck"}}, "rvs_cvs"), "RVS")
        self.assertEqual(
            display_name({"suite_id": "rccl", "report": {"title": "RCCL Run Deck"}}, "rccl_perf"), "RCCL \u00b7 perf"
        )
        self.assertEqual(
            display_name(
                {"suite_id": "anc_test_babel", "report": {"title": "ANC Run Deck \u00b7 Babel"}}, "anc_test_babel"
            ),
            "ANC \u00b7 Babel",
        )
        self.assertEqual(display_name({"suite_id": "ib_perf_bw_test", "report": {"title": "x"}}, None), "IB perf")
        self.assertEqual(display_name({"suite_id": "atom", "report": {}}, "atom"), "atom")
        self.assertEqual(display_name(None, "rvs_cvs"), "rvs_cvs")

    def test_host_key_matches_vpc_labels_to_plain_hosts(self):
        self.assertEqual(host_key("10.0.0.1_node-01"), "node-01")
        self.assertEqual(host_key("fe80::1_node-02"), "node-02")
        self.assertEqual(host_key("node_01"), "node_01")
        self.assertEqual(host_key("node-03"), "node-03")

    def test_generated_range(self):
        early = datetime(2026, 11, 3, 14, 5, tzinfo=timezone.utc)
        late = datetime(2026, 11, 10, 19, 32, 59, tzinfo=timezone.utc)
        self.assertEqual(generated_range([late, None, early]), "2026-11-03 14:05 \u2013 2026-11-10 19:32 UTC")
        self.assertEqual(generated_range([early, early]), "2026-11-03 14:05 UTC")
        self.assertEqual(generated_range([None]), "Unavailable")


class TestCollectionVerdict(unittest.TestCase):
    def test_needs_attention_when_any_suite_failed(self):
        self.assertEqual(
            collection_verdict([suite("a", "fail"), suite("b", available=False, status="unknown")]), NEEDS_ATTENTION
        )

    def test_incomplete_data(self):
        self.assertEqual(collection_verdict([suite("a"), suite("b", "unknown", available=False)]), INCOMPLETE_DATA)
        self.assertEqual(collection_verdict([suite("a"), suite("b", partial=True)]), INCOMPLETE_DATA)
        self.assertEqual(collection_verdict([suite("a"), suite("b", "unknown")]), INCOMPLETE_DATA)
        self.assertEqual(collection_verdict([]), INCOMPLETE_DATA)

    def test_evaluation_incomplete(self):
        self.assertEqual(collection_verdict([suite("a"), suite("b", "record")]), EVALUATION_INCOMPLETE)
        self.assertEqual(collection_verdict([suite("a"), suite("b", "na")]), EVALUATION_INCOMPLETE)

    def test_all_passed(self):
        self.assertEqual(collection_verdict([suite("a"), suite("b")]), ALL_PASSED)


class TestRecommendations(unittest.TestCase):
    def test_repeated_host_failures_across_host_row_suites(self):
        suites = [
            suite("rvs_cvs", "fail", name="RVS", nodes=["10.0.0.1_node-01"], failedNodes=["10.0.0.1_node-01"]),
            suite("agfhc_cvs", "fail", name="AGFHC", nodes=["node-01"], failedNodes=["node-01"]),
            suite("ib", "fail", name="IB perf", nodes=["node-01"], failedNodes=["node-01"], _host_rows=False),
        ]
        recs = recommendations(suites)
        self.assertEqual(
            recs[0],
            {"kind": "repeat_host", "suite": None, "message": "node-01 failed in 2 suites: RVS, AGFHC.", "link": None},
        )

    def test_failed_suite_message_counts_in_its_unit_and_names_nodes(self):
        nodes = ["n%d" % i for i in range(6)]
        failed = suite("rvs_cvs", "fail", name="RVS", total=24, nodes=nodes, failedNodes=nodes)
        failed["counts"]["fail"] = 6
        message = recommendations([failed])[0]["message"]
        self.assertEqual(message, "RVS: 6 of 24 node/check groups failed on n0, n1, n2, n3 and 2 more.")

    def test_failed_suite_without_failed_rows_points_to_its_deck(self):
        failed = suite("rccl_perf", "fail", name="RCCL \u00b7 perf", unit="recorded result rows", _host_rows=False)
        rec = recommendations([failed])[0]
        self.assertEqual(rec["message"], "RCCL \u00b7 perf reported a failure; open its Run Deck for details.")
        self.assertEqual(rec["link"], "runs/rccl_perf/r/deck.html")

    def test_unavailable_record_only_and_not_evaluated(self):
        suites = [
            suite("b", "unknown", name="B", available=False, _source="/runs/b/deck.json"),
            suite("c", "record", name="C"),
            suite("d", "na", name="D"),
        ]
        kinds = [(rec["kind"], rec["message"]) for rec in recommendations(suites)]
        self.assertEqual(
            kinds,
            [
                ("unavailable", "B: /runs/b/deck.json could not be read; rerun the suite or regenerate its Run Deck."),
                ("record_only", "C has no pass/fail gate; review its measurements."),
                ("not_evaluated", "D: nothing was evaluated."),
            ],
        )

    def test_all_passed(self):
        self.assertEqual(recommendations([suite("a")])[0]["kind"], "all_passed")


class TestManifest(unittest.TestCase):
    def test_hosts_clusters_and_per_suite_provenance(self):
        provenance = {
            "cvs_version": "0.4.0",
            "git_ref": "main@abc123",
            "image_display": "rocm/vllm:1.0",
            "cluster_file": "/c/cluster.json",
            "config_file": "",
        }
        suites = [
            suite(
                "rvs_cvs",
                "fail",
                name="RVS",
                cluster="helios",
                nodes=["10.0.0.1_node-01", "10.0.0.2_node-02"],
                failedNodes=["10.0.0.2_node-02"],
                _provenance=provenance,
            ),
            suite("agfhc_cvs", name="AGFHC", cluster="helios", nodes=["node-02", "node-09"]),
            suite("vllm_single", name="vLLM", cluster="other", nodes=["10.0.0.1"], _host_rows=False),
        ]
        manifest = build_manifest(suites)
        self.assertEqual(manifest["clusters"], {"helios": ["RVS", "AGFHC"], "other": ["vLLM"]})
        self.assertEqual(
            manifest["hosts"],
            [
                {"host": "node-01", "suites": ["RVS"], "failed_in": []},
                {"host": "node-02", "suites": ["RVS", "AGFHC"], "failed_in": ["RVS"]},
                {"host": "node-09", "suites": ["AGFHC"], "failed_in": []},
            ],
        )
        self.assertEqual(
            manifest["suites"][0],
            {
                "id": "rvs_cvs",
                "name": "RVS",
                "cluster": "helios",
                "cvs_version": "0.4.0",
                "git_ref": "main@abc123",
                "image": "rocm/vllm:1.0",
                "cluster_file": "/c/cluster.json",
                "config_file": None,
                "generated": "2026-11-10 19:32 UTC",
            },
        )


if __name__ == "__main__":
    unittest.main()
