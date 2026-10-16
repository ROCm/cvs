'''Unit tests for cvs/lib/mori_reporting.py: Mori results in the status_matrix Run Deck shape.'''

import unittest

from cvs.lib.mori_reporting import record_ibgda_results, record_io_results, record_test_failure
from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.rundeck.payload import build_rundeck_payload
from cvs.lib.report.rundeck.render import render_rundeck_html

MIB = 1 << 20
VARIANT = {"hosts": ["host-a", "host-b"], "initiator": "host-a", "mori_device_list": "rdma0,rdma1,rdma2"}
READ = "MORI-IO READ · node01 → node00"
WRITE = "MORI-IO WRITE · node00 → node01"
IBGDA = "IBGDA WRITE · rdma0 → rdma1 inside each node"


def _io_row(size, avg_bw, avg_lat):
    return {"MsgSize_B": size, "BatchSize": 128, "TotalSize_MB": 1.0, "Avg_BW_GBps": avg_bw, "Avg_Lat_us": avg_lat}


def _io_parsed(op, ranks, buffer_size=4194304, batch=128, qp=1):
    return {
        "operation": op,
        "buffer_size": buffer_size,
        "transfer_batch_size": batch,
        "qp_count": qp,
        "ranks": {rank: {"rows": rows} for rank, rows in ranks.items()},
    }


def _case(op, rank_bw, sizes=(8, 32768, MIB), **shape):
    return _io_parsed(op, {r: [_io_row(s, bw, 40.0 + s / 1000) for s in sizes] for r, bw in rank_bw.items()}, **shape)


def _ibgda(nodes):
    return {
        "operation": "ibgda_write",
        "nodes": {
            node: {"rows": [{"size_bytes": s, "bandwidth_gb": bw, "time_ms": 0.5, "rate_mpps": 10.0} for s in (2, MIB)]}
            for node, bw in nodes.items()
        },
    }


class TestRecordIo(unittest.TestCase):
    def test_ranks_are_aggregated_per_size_and_kept_per_rank(self):
        store = {}
        parsed = _io_parsed(
            "io_read",
            {
                0: [_io_row(MIB, 47.0, 2900.0)],
                1: [_io_row(MIB, 45.0, 2950.0)],
                2: [_io_row(MIB, 46.3, 2910.0)],
            },
        )
        record_io_results(store, parsed, case_qp_count=1, status="pass", variant=VARIANT)
        entry = store["_mori"]["io"]["io_read|4194304|128|1"]["sizes"][str(MIB)]
        self.assertEqual(entry["ranks"], 3)
        self.assertEqual(entry["Avg_BW_GBps"], 46.1)
        self.assertEqual((entry["Slowest_rank"], entry["Slowest_rank_BW_GBps"]), (1, 45.0))
        self.assertEqual(entry["Avg_Lat_us"], 2920.0)
        self.assertEqual(entry["Worst_rank_Lat_us"], 2950.0)
        self.assertEqual(entry["Rank_BW_GBps"], {"0": 47.0, "1": 45.0, "2": 46.3})

    def test_slowest_rank_tie_goes_to_the_lowest_rank_and_bad_values_are_skipped(self):
        bad = _io_row(64, 0.0, 41.0)
        bad["Avg_BW_GBps"] = "n/a"
        parsed = _io_parsed("io_read", {7: [_io_row(64, 0.5, 40.0)], 3: [_io_row(64, 0.5, 42.0)], 5: [bad]})
        store = record_io_results({}, parsed)
        entry = store["_mori"]["io"]["io_read|4194304|128|1"]["sizes"]["64"]
        self.assertEqual(entry["Slowest_rank"], 3)
        self.assertEqual(entry["ranks"], 3)
        self.assertEqual(entry["Avg_BW_GBps"], 0.5)
        self.assertEqual(entry["Worst_rank_Lat_us"], 42.0)

    def test_runs_without_rows_leave_the_store_untouched(self):
        store = {"existing": 1}
        for parsed in (None, {}, _io_parsed("io_read", {0: [{"Avg_BW_GBps": 1.0}]}), _io_parsed("io_read", {})):
            with self.subTest(parsed=parsed):
                self.assertIs(record_io_results(store, parsed, variant=VARIANT), store)
                self.assertEqual(store, {"existing": 1})


class TestGroups(unittest.TestCase):
    def test_titles_name_the_direction_from_the_initiator(self):
        for initiator, read, write in (("host-a", READ, WRITE), ("host-b", WRITE.replace("WRITE", "READ"), None)):
            with self.subTest(initiator=initiator):
                store = {}
                record_io_results(store, _case("io_read", {0: 40.0}), variant={**VARIANT, "initiator": initiator})
                record_io_results(store, _case("io_write", {0: 40.0}), variant={**VARIANT, "initiator": initiator})
                titles = list(store["groups"])
                self.assertEqual(titles[0], read)
                if write:
                    self.assertEqual(titles[1], write)
                else:
                    self.assertEqual(titles[1], "MORI-IO WRITE · node01 → node00")

    def test_titles_fall_back_without_run_facts(self):
        for variant in (None, {"hosts": ["host-a"], "initiator": "host-a"}, {**VARIANT, "initiator": "other"}):
            with self.subTest(variant=variant):
                store = record_io_results({}, _case("io_write", {0: 40.0}), variant=variant)
                self.assertEqual(list(store["groups"]), ["MORI-IO WRITE"])
        for devices in ("rdma0", "^rdma0,rdma1", None):
            with self.subTest(devices=devices):
                store = record_ibgda_results(
                    {}, _ibgda({"host-a": 49.0}), variant={**VARIANT, "mori_device_list": devices}
                )
                self.assertEqual(list(store["groups"]), ["IBGDA WRITE inside each node"])

    def test_groups_keep_read_write_ibgda_order_and_cases_sort_by_shape(self):
        store = {}
        record_ibgda_results(store, _ibgda({"host-b": 49.0, "host-a": 48.0}), variant=VARIANT)
        record_io_results(store, _case("io_write", {0: 40.0}, buffer_size=131072, batch=256), variant=VARIANT)
        record_io_results(store, _case("io_read", {0: 40.0}), case_qp_count=8, variant=VARIANT)
        record_io_results(store, _case("io_read", {0: 40.0}), case_qp_count=1, variant=VARIANT)
        self.assertEqual(list(store["groups"]), [READ, WRITE, IBGDA])
        self.assertEqual(
            list(store["groups"][READ]["nodes"]),
            ["buffer 4 MiB · batch 128 · QP 1", "buffer 4 MiB · batch 128 · QP 8", "all cases"],
        )
        self.assertEqual(list(store["groups"][IBGDA]["nodes"]), ["node00", "node01", "both nodes"])

    def test_cell_series_split_latency_at_32_kib(self):
        store = record_io_results({}, _case("io_read", {0: 46.0, 1: 44.0}), variant=VARIANT)
        cell = store["groups"][READ]["nodes"]["buffer 4 MiB · batch 128 · QP 1"]
        bw, small, large = cell["series"]
        self.assertEqual((bw["name"], bw["unit"], bw["x_label"]), ("Bandwidth", "GB/s", "message size"))
        self.assertEqual(bw["points"], [["8B", 45.0], ["32K", 45.0], ["1M", 45.0]])
        self.assertEqual(small["name"], "Latency per batch, up to 32 KiB")
        self.assertEqual([x for x, _y in small["points"]], ["8B", "32K"])
        self.assertEqual([x for x, _y in large["points"]], ["32K", "1M"])
        self.assertNotIn("items", cell)

    def test_size_heatmaps_show_every_case_at_key_sizes(self):
        sizes = tuple(1 << n for n in range(3, 21))
        store = {}
        record_io_results(store, _case("io_read", {0: 46.0}, sizes=sizes), case_qp_count=1, variant=VARIANT)
        record_io_results(store, _case("io_read", {0: 30.0}, sizes=sizes[:15], buffer_size=131072), variant=VARIANT)
        bw, lat = store["groups"][READ]["nodes"]["all cases"]["heatmaps"]
        self.assertEqual(
            (bw["name"], bw["unit"], lat["name"], lat["unit"]),
            ("MORI-IO READ bandwidth", "GB/s", "MORI-IO READ latency per batch", "µs"),
        )
        self.assertEqual(bw["rows"], ["buffer 128 KiB · batch 128 · QP 1", "buffer 4 MiB · batch 128 · QP 1"])
        self.assertEqual(bw["cols"], ["32B", "256B", "2K", "16K", "128K", "1M"])
        self.assertEqual(bw["values"][0], [30.0, 30.0, 30.0, 30.0, 30.0, None])
        self.assertEqual(bw["values"][1], [46.0] * 6)
        self.assertEqual(lat["values"][1][-1], round(40.0 + MIB / 1000, 2))
        self.assertNotIn("threshold", bw)

    def test_rank_heatmap_covers_the_largest_case_both_operations_ran(self):
        store = {}
        record_io_results(store, _case("io_write", {0: 48.0, 1: 46.0, 2: 41.0}), case_qp_count=8, variant=VARIANT)
        record_io_results(store, _case("io_read", {0: 47.5, 1: 48.0, 2: 47.0}), case_qp_count=8, variant=VARIANT)
        record_io_results(store, _case("io_write", {0: 1.0}, sizes=(8, 32768)), case_qp_count=1, variant=VARIANT)
        record_io_results(store, _case("io_write", {0: 1.0}, batch=256), case_qp_count=8, variant=VARIANT)
        (heat,) = store["_meta"]["rank_grid"]["heatmaps"]
        self.assertEqual(heat["name"], "MORI-IO per rank at 1 MiB, both directions · buffer 4 MiB · batch 128 · QP 8")
        self.assertEqual(heat["rows"], ["node00 → node01", "node01 → node00"])
        self.assertEqual(heat["cols"], ["rank 0 · GPU 0", "rank 1 · GPU 1", "rank 2 · GPU 2"])
        self.assertEqual(heat["values"], [[48.0, 46.0, 41.0], [47.5, 48.0, 47.0]])
        self.assertEqual((heat["threshold"], heat["direction"]), (round(0.97 * 47.5, 2), "higher"))
        # The size heatmaps stay in the groups; the rank grid is only in its own section.
        cells = [cell for group in store["groups"].values() for cell in group["nodes"].values()]
        self.assertNotIn(heat["name"], [h["name"] for cell in cells for h in cell.get("heatmaps", [])])

    def test_rank_heatmap_needs_both_operations_and_marks_missing_ranks(self):
        store = record_io_results({}, _case("io_write", {0: 48.0, 1: 48.0}), variant=VARIANT)
        self.assertNotIn("rank_grid", store["_meta"])
        record_io_results(store, _case("io_read", {0: 48.0}), variant=VARIANT)
        heat = store["_meta"]["rank_grid"]["heatmaps"][0]
        self.assertEqual(heat["values"], [[48.0, 48.0], [48.0, None]])


class TestRecordIbgda(unittest.TestCase):
    def test_nodes_use_cluster_names_with_bandwidth_and_rate_series(self):
        store = record_ibgda_results({}, _ibgda({"host-b": 49.0, "host-a": 48.0}), status="pass", variant=VARIANT)
        nodes = store["groups"][IBGDA]["nodes"]
        self.assertEqual(list(nodes), ["node00", "node01", "both nodes"])
        bw, rate = nodes["node00"]["series"]
        self.assertEqual((bw["name"], bw["points"]), ("Bandwidth", [["2B", 48.0], ["1M", 48.0]]))
        self.assertEqual((rate["name"], rate["unit"]), ("Message rate", "Mpps"))
        heat = nodes["both nodes"]["heatmaps"][0]
        self.assertEqual(
            (heat["name"], heat["rows"], heat["cols"]), ("IBGDA WRITE bandwidth", ["node00", "node01"], ["2B", "1M"])
        )
        self.assertEqual(heat["values"], [[48.0, 48.0], [49.0, 49.0]])

    def test_nodes_without_rows_leave_the_store_untouched(self):
        store = {}
        self.assertIs(record_ibgda_results(store, {"nodes": {"host-a": {"rows": []}}}), store)
        self.assertEqual(store, {})


class TestMoriDeckProfile(unittest.TestCase):
    def _payload(self):
        store = {}
        record_ibgda_results(store, _ibgda({"host-a": 49.0, "host-b": 49.0}), variant=VARIANT)
        record_io_results(store, _case("io_read", {0: 47.0, 1: 46.0}), variant=VARIANT)
        record_io_results(store, _case("io_write", {0: 47.5, 1: 46.5}), variant=VARIANT)
        variant = {"gpu_name": "MI355X", "node_count": 2, "container_image": "example/mori:latest", **VARIANT}
        return build_rundeck_payload(
            profile=load_json_profile("mori_benchmark_test"),
            store={"cvs_results_dict": store, "variant_config": variant},
            cvs_version="1.0.0",
        )

    def test_deck_renders_per_op_groups_rank_section_and_results_without_script(self):
        document = render_rundeck_html(self._payload())
        for text in (
            "Mori Benchmark Run Deck",
            f"<h3 class='chart-group-title'>{READ}</h3>",
            f"<h3 class='chart-group-title'>{WRITE}</h3>",
            f"<h3 class='chart-group-title'>{IBGDA}</h3>",
            "MORI-IO per rank at 1 MiB, both directions · buffer 4 MiB · batch 128 · QP 1",
            "MORI-IO READ bandwidth · all cases",
            "MORI-IO WRITE latency per batch · all cases",
            "IBGDA WRITE bandwidth · both nodes",
            "node00 = host-a, node01 = host-b",
            "MI355X",
        ):
            self.assertIn(text, document)
        self.assertNotIn("<script", document)
        self.assertNotIn("id='failed-tests'", document)
        self.assertNotIn("<table class='status-matrix'>", document)
        self.assertNotIn("Viewer</a>", document)
        ranks, charts = document.split("<section class='panel' id='ranks'>")[1].split(
            "<section class='panel' id='charts'>"
        )
        self.assertIn("<h2>MORI-IO per rank</h2>", ranks)
        self.assertNotIn("Heatmaps", ranks)
        self.assertIn("hm-pass", ranks)
        self.assertNotIn("MORI-IO per rank", charts)
        self.assertIn("<section class='panel' id='results'><h2>Full results</h2>", charts)

    def test_an_uncharted_failed_test_fails_the_deck_and_is_listed(self):
        store = {}
        record_test_failure(store, "test_shmem_api", "call", "Failed: Following FAILURES seen -\n['shmem hang']")
        record_io_results(store, _case("io_read", {0: 47.0, 1: 46.0}), variant=VARIANT)
        payload = build_rundeck_payload(
            profile=load_json_profile("mori_benchmark_test"),
            store={"cvs_results_dict": store, "variant_config": VARIANT},
            cvs_version="1.0.0",
        )
        self.assertEqual(payload["overall_status"], "fail")
        self.assertEqual(
            payload["datasets"]["status_matrix"]["meta"]["failed_tests"]["rows"],
            [["test_shmem_api", "call", "Failed: Following FAILURES seen - ['shmem hang']"]],
        )
        self.assertIn("<section class='panel' id='failed-tests'><h2>Failed tests</h2>", render_rundeck_html(payload))

    def test_results_table_rounds_rank_and_ibgda_values_like_the_means(self):
        store = record_io_results({}, _case("io_read", {0: 47.123456, 1: 46.987654}, sizes=(MIB,)), variant=VARIANT)
        ibgda = _ibgda({"host-a": 48.123456})
        ibgda["nodes"]["host-a"]["rows"][0]["rate_mpps"] = 10.98765
        record_ibgda_results(store, ibgda, variant=VARIANT)
        io_row, ibgda_row = store["_meta"]["results_table"]["rows"][:2]
        self.assertEqual((io_row[5], io_row[7]), (46.99, 1088.58))
        self.assertEqual((ibgda_row[3], ibgda_row[8]), (48.12, 10.99))

    def test_results_table_has_a_row_per_case_and_ibgda_node_per_size(self):
        table = self._payload()["datasets"]["status_matrix"]["meta"]["results_table"]
        rows = [dict(zip(table["headers"], row)) for row in table["rows"]]
        self.assertEqual(len(rows), 3 + 3 + 2 * 2)
        self.assertEqual(
            [rows[0][k] for k in ("Operation", "Case or node", "Message size", "Bandwidth", "Status")],
            ["MORI-IO READ", "buffer 4 MiB · batch 128 · QP 1", "8B", 46.5, "pass"],
        )
        self.assertEqual((rows[0]["Slowest rank"], rows[0]["Slowest rank BW"]), (1, 46.0))
        self.assertEqual(table["units"][3:9], ["GB/s", None, "GB/s", "µs", "µs", "Mpps"])
        self.assertEqual((table["align"][1], table["align"][2], table["align"][-1]), ("left", "right", "left"))
        self.assertEqual(
            [rows[-1][k] for k in ("Operation", "Case or node", "Message size", "Message rate")],
            ["IBGDA WRITE", "node01", "1M", 10.0],
        )
        self.assertEqual([row["Message size"] for row in rows[:3]], ["8B", "32K", "1M"])


if __name__ == "__main__":
    unittest.main()
