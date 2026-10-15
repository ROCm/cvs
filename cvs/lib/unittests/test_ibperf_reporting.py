'''Unit tests for cvs/lib/ibperf_reporting.py: ibperf results in the status_matrix Run Deck shape.'''

import unittest

from cvs.lib.ibperf_reporting import record_bw_results, record_lat_results
from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.rundeck.payload import build_rundeck_payload
from cvs.lib.report.rundeck.render import render_rundeck_html

HOSTS = ["host-a", "host-b", "host-c", "host-d"]
VARIANT = {
    "hosts": HOSTS,
    "expected_results": {"ib_write_bw": {"8192": {"8": "180.0"}}},
    "ib_bw_test_list": ["ib_write_bw"],
}
WRITE = "ib_write_bw · client → server, 2 node pairs"
LAT = "ib_write_lat · client ↔ server, 2 node pairs"


def _bw(value_of):
    return {host: {nic: {"bw": str(value_of(host, nic)), "pps": "1.0"} for nic in (0, 1)} for host in HOSTS}


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


class TestRecord(unittest.TestCase):
    def test_keeps_per_instance_values_and_skips_failed_instances(self):
        store = {}
        per_node = {"host-a": {0: {"bw": "390.5", "pps": "0.745"}, 1: {}}, "host-b": {0: {"bw": "380.0"}}}
        record_bw_results(store, "ib_write_bw", 65536, "8", per_node)
        self.assertEqual(
            store["_ibperf"]["bw"]["ib_write_bw|8|65536"],
            {"host-a": {"0": {"bw": 390.5, "pps": 0.745}}, "host-b": {"0": {"bw": 380.0}}},
        )
        untouched = {}
        self.assertIs(record_bw_results(untouched, "ib_write_bw", 2, "8", {"host-a": {0: {}}}), untouched)
        self.assertEqual(untouched, {})


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

    def test_weak_nics_name_nodes_peers_and_breaches(self):
        meta = _fleet()["_meta"]
        rows = [dict(zip(meta["weak_nics"]["headers"], row)) for row in meta["weak_nics"]["rows"]]
        self.assertEqual(
            [(r["Status"], r["Node · NIC"], r["Pair peer"], r["QPs"], r["Relative"]) for r in rows],
            [
                ("outlier", "node02 · NIC 1", "node03", "8, 16", "50% of median"),
                ("outlier", "node03 · NIC 1", "node02", "8, 16", "50% of median"),
                ("below threshold", "node00 · NIC 0", "node01", "8", "92% of median"),
                ("outlier", "node01 · NIC 0", "node00", "—", "4.5× median"),
            ],
        )

    def test_results_table_has_a_row_per_run_with_threshold_status(self):
        table = _fleet()["_meta"]["results_table"]
        rows = [dict(zip(table["headers"], row)) for row in table["rows"]]
        self.assertEqual(
            [(r["Test"], r["QP"], r["Message size"]) for r in rows[:4]],
            [
                ("ib_write_bw", "8", "8K"),
                ("ib_write_bw", "8", "64K"),
                ("ib_write_bw", "16", "8K"),
                ("ib_write_bw", "16", "64K"),
            ],
        )
        self.assertEqual(
            [rows[0][k] for k in ("Median BW", "Min BW", "Slowest NIC", "BW threshold", "Status")],
            [185.0, 170.0, "node00 NIC 0", "180", "fail"],
        )
        self.assertEqual(rows[2]["Status"], "—")
        self.assertEqual(
            [rows[4][k] for k in ("Test", "Message size", "Median typical", "Median p99", "Worst p99")],
            ["ib_write_lat", "2B", 1.9, 2.0, 9.0],
        )

    def test_deck_renders_weak_nics_groups_and_heatmap(self):
        payload = build_rundeck_payload(
            profile=load_json_profile("ib_perf_bw_test"),
            store={"cvs_results_dict": _fleet(), "variant_config": {"node_count": 4, **VARIANT}},
            cvs_version="1.0.0",
        )
        self.assertEqual(payload["overall_status"], "fail")
        document = render_rundeck_html(payload)
        for text in (
            "Weak NICs and links",
            "node00 = host-a, node01 = host-b",
            f"<h3 class='chart-group-title'>{WRITE}</h3>",
            f"<h3 class='chart-group-title'>{LAT}</h3>",
            "Bandwidth, slowest NIC by message size",
            "Per-NIC bandwidth",
            "NIC bandwidth at 64 KiB, worst % of fleet median over every test and QP",
            "does not affect validation gates",
        ):
            self.assertIn(text, document)
        self.assertNotIn("<table class='status-matrix'>", document)
        self.assertIn("results-scroll' style='max-height:480px", document)
        self.assertIn("Median p99 <span class='th-unit'>(µs)</span>", document)
        self.assertIn("<td class='al-left'>ib_write_bw</td><td class='al-right'>8</td>", document)
        self.assertNotIn("Heatmaps", document.split("id='nics'>")[1].split("id='charts'>")[0])
        order = [document.index(f"id='{sid}'") for sid in ("run-card", "nics", "charts", "weak-nics", "results")]
        self.assertEqual(order, sorted(order))


if __name__ == "__main__":
    unittest.main()
