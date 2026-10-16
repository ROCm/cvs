'''Unit tests for cvs/lib/ibperf_reporting.py: ibperf results in the status_matrix Run Deck shape.'''

import unittest

from cvs.lib.ibperf_reporting import UNAVAILABLE, record_bw_results, record_lat_results
from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.rundeck.payload import build_rundeck_payload
from cvs.lib.report.rundeck.render import render_rundeck_html

HOSTS = ["host-a", "host-b", "host-c", "host-d"]
VARIANT = {
    "hosts": HOSTS,
    "expected_results": {"ib_write_bw": {"8192": {"8": "180.0"}}},
    "verify_bw": "True",
    "ib_bw_test_list": ["ib_write_bw"],
}
WRITE = "ib_write_bw · client → server, 2 node pairs"
LAT = "ib_write_lat · client ↔ server, 2 node pairs"


def _bw(value_of):
    return {host: {nic: {"bw": str(value_of(host, nic))} for nic in (0, 1)} for host in HOSTS}


def _fleet():
    """host-a NIC 0 misses the 8 KiB QP 8 threshold; the host-c/host-d NIC 1 link runs at half rate at 64 KiB."""
    store = {}
    record_bw_results(
        store, "ib_write_bw", 8192, "8", _bw(lambda h, n: 170.0 if (h, n) == ("host-a", 0) else 185.0), VARIANT
    )
    record_bw_results(store, "ib_write_bw", 8192, "16", _bw(lambda h, n: 200.0), VARIANT)
    for qp in ("16", "8"):
        record_bw_results(
            store, "ib_write_bw", 65536, qp, _bw(lambda h, n: 190.0 if n == 1 and h in ("host-c", "host-d") else 380.0)
        )
    for size in (2, 8192, 65536, 1 << 20):
        lat = {h: {n: {"t_avg": "2.0", "t_typical": "1.9", "t_99_pct": "2.0"} for n in (0, 1)} for h in HOSTS}
        if size == 2:
            lat["host-b"][0]["t_99_pct"] = "9.0"
        record_lat_results(store, "ib_write_lat", size, lat)
    return store


def _payload(store):
    return build_rundeck_payload(
        profile=load_json_profile("ib_perf_bw_test"),
        store={"cvs_results_dict": store, "variant_config": {"node_count": 4, **VARIANT}},
        cvs_version="1.0.0",
    )


class TestRecord(unittest.TestCase):
    def test_keeps_failed_instances_and_runs_as_unavailable(self):
        store = {}
        record_bw_results(store, "ib_write_bw", 65536, "8", {"host-a": {0: {"bw": "390.5"}, 1: {}}})
        record_bw_results(store, "ib_write_bw", 2, "8", {"host-a": {0: {"bw": "N/A"}}})
        self.assertEqual(
            store["_ibperf"]["bw"],
            {
                "ib_write_bw|8|65536": {"host-a": {"0": {"bw": 390.5}, "1": None}},
                "ib_write_bw|8|2": {"host-a": {"0": None}},
            },
        )


class TestDeck(unittest.TestCase):
    def test_groups_are_titled_by_direction_with_qp_lines_and_split_latency(self):
        groups = _fleet()["groups"]
        self.assertEqual(list(groups), [WRITE, LAT])
        self.assertEqual(list(groups[WRITE]["nodes"]), ["QP 8", "QP 16"])
        qp8 = groups[WRITE]["nodes"]["QP 8"]
        self.assertEqual(qp8["status"], "fail")
        self.assertEqual(groups[WRITE]["nodes"]["QP 16"]["status"], "na")
        self.assertEqual(qp8["series"][1]["points"], [["8K", 170.0], ["64K", 190.0]])
        worst = groups[LAT]["nodes"]["worst NIC p99"]
        self.assertEqual(worst["status"], "fail")
        self.assertEqual([s["name"] for s in worst["series"]], ["Latency, up to 32 KiB", "Latency, from 32 KiB"])

    def test_heatmap_flags_nics_below_95_percent_of_median(self):
        heat = _fleet()["_meta"]["nic_grid"]["heatmaps"][0]
        self.assertEqual(heat["rows"], ["node00 ↔ node01", "node01 ↔ node00", "node02 ↔ node03", "node03 ↔ node02"])
        self.assertEqual(heat["values"][2], [100.0, 50.0])
        self.assertEqual((heat["threshold"], heat["direction"]), (95, "higher"))

    def test_results_table_has_a_row_per_run_with_threshold_status(self):
        table = _fleet()["_meta"]["results_table"]
        rows = [dict(zip(table["headers"], row)) for row in table["rows"]]
        self.assertEqual(len(rows), 8)
        self.assertEqual(
            [rows[0][k] for k in ("Median BW", "Min BW", "Slowest NIC", "BW threshold", "Status")],
            [185.0, 170.0, "node00 NIC 0", "180", "fail"],
        )
        self.assertEqual(rows[2]["Status"], "—")
        self.assertEqual(
            [rows[4][k] for k in ("Test", "Message size", "Median typical", "Median p99", "Worst p99")],
            ["ib_write_lat", "2B", 1.9, 2.0, 9.0],
        )

    def test_threshold_misses_fail_only_when_verify_bw_is_on(self):
        store = {}
        record_bw_results(store, "ib_write_bw", 8192, "8", _bw(lambda h, n: 170.0), {**VARIANT, "verify_bw": "False"})
        self.assertEqual(store["groups"][WRITE]["nodes"]["QP 8"]["status"], "na")
        self.assertEqual(store["_meta"]["results_table"]["rows"][0][9:11], ["180", "—"])
        self.assertEqual(_payload(store)["overall_status"], "na")

    def test_collection_failures_show_as_unavailable_and_fail_the_deck(self):
        variant = {**VARIANT, "verify_bw": "False"}
        store = {}
        record_bw_results(store, "ib_write_bw", 8192, "8", {h: {0: {}, 1: {}} for h in HOSTS}, variant)
        partial = _bw(lambda h, n: 380.0)
        partial["host-b"][1] = {}
        record_bw_results(store, "ib_write_bw", 65536, "8", partial, variant)
        self.assertEqual(store["_meta"]["nic_grid"]["heatmaps"][0]["values"][1], [100.0, UNAVAILABLE])
        rows = store["_meta"]["results_table"]["rows"]
        self.assertEqual([rows[0][k] for k in (3, 4, 10, 11)], ["—", "—", UNAVAILABLE, 0])
        self.assertEqual([rows[1][k] for k in (3, 10, 11)], [380.0, UNAVAILABLE, 7])
        self.assertEqual(store["groups"][WRITE]["nodes"]["QP 8"]["status"], "fail")
        self.assertEqual(_payload(store)["overall_status"], "fail")

    def test_deck_renders_run_card_groups_and_heatmap(self):
        payload = _payload(_fleet())
        self.assertEqual(payload["overall_status"], "fail")
        document = render_rundeck_html(payload)
        for text in (
            "node00 = host-a, node01 = host-b",
            f"<h3 class='chart-group-title'>{WRITE}</h3>",
            f"<h3 class='chart-group-title'>{LAT}</h3>",
            "NIC bandwidth at 64 KiB, worst % of fleet median over every test and QP",
            "does not affect validation gates",
        ):
            self.assertIn(text, document)


if __name__ == "__main__":
    unittest.main()
